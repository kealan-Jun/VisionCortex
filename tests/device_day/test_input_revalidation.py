"""A remount may change file identities, never the accepted media bytes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from visioncortex.device_day_contract import DeviceDayLayout
from visioncortex.device_day_inputs import binding, identity
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.input_availability import revalidate_input_seals


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    from visioncortex import nas_recordings
    source = tmp_path / 'capture' / 'slice'
    source.mkdir(parents=True)
    video, clock = source / 'rgb.mp4', source / 'frames.csv'
    video.write_bytes(b'owned video bytes')
    clock.write_bytes(b'owned clock bytes')
    record = {'recording_id': 'recording', 'source_signature': 'sealed-source', 'camera_key': 'camera',
              'configured_role': 'first_person', 'processable': True, 'video_path': str(video),
              'frames_path': str(clock), 'size_bytes': video.stat().st_size,
              'recording_start_us': 1_800_000_000_000_000, 'recording_end_us': 1_800_000_001_000_000}
    layout = DeviceDayLayout(tmp_path / 'archives', 'camera', record['recording_start_us'], tmp_path / 'receipts')
    runtime = tmp_path / 'runtime'
    runner = SimpleNamespace(config={'runtime': {'local_only': True},
                                    'collection_ingest': {'source_root': str(source.parent)}},
                             runtime_root=runtime, layout=lambda _: layout,
                             _receipt=lambda layout, record, stage: layout.receipts / record['recording_id'] / (stage + '.json'))
    paths = []
    for stage, originals in [('vision', [('video', video), ('clock', clock)]), ('stt', [('video', video)])]:
        sources = []
        for kind, original in originals:
            info = identity(original)
            checksum = hashlib.sha256(original.read_bytes()).hexdigest()
            retained = {'path': layout.relative(layout.source_path(record, kind, original)),
                        'size_bytes': info[2], 'sha256': checksum}
            sources.append({'kind': kind, 'original_path': str(original), 'resolved_path': str(original.resolve()),
                            'identity': [info[0] + 1, *info[1:]], 'size_bytes': info[2], 'mtime_ns': info[3],
                            'sha256': checksum, 'retained': retained})
        seal = {'stage': 'input-' + stage, 'status': 'ready', 'recording': record,
                'source_signature': record['source_signature'] if stage == 'vision' else None, 'sources': sources}
        path = runner._receipt(layout, record, 'input-' + stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(seal, indent=1))
        paths.append(path)
    monkeypatch.setattr(nas_recordings, '_inspect', lambda *args: deepcopy(record))
    return runner, record, paths


def test_revalidation_preserves_exact_old_seals_hashes_and_bindings_without_invoking_models(sealed):
    runner, record, paths = sealed
    before = {path: path.read_bytes() for path in paths}
    result = revalidate_input_seals(runner, record)
    assert result['changed_seals'] == ['input-vision', 'input-stt']
    assert result['model_invoked'] is result['capture_modified'] is False
    for path in paths:
        old, new = json.loads(before[path]), json.loads(path.read_bytes())
        assert binding(new) == binding(old)
        for prior, current in zip(old['sources'], new['sources'], strict=True):
            assert {k: v for k, v in prior.items() if k not in {'identity', 'resolved_path'}} == {
                k: v for k, v in current.items() if k not in {'identity', 'resolved_path'}}
            assert current['identity'] == identity(current['original_path'])
        history = list((path.parent / 'history').glob(path.stem + '-before-revalidation-*.json'))
        assert len(history) == 1 and history[0].read_bytes() == before[path]
    assert revalidate_input_seals(runner, record)['changed_seals'] == []
    assert len(list((runner.runtime_root / 'InputRevalidation').rglob('*.json'))) == 2


def test_changed_content_rejects_both_seals_before_any_publication(sealed):
    runner, record, paths = sealed
    before = {path: path.read_bytes() for path in paths}
    video = Path(record['video_path'])
    stat = video.stat()
    video.write_bytes(b'other video bytes')  # Equal size and restored mtime cannot confer trust.
    import os
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='content differs'):
        revalidate_input_seals(runner, record)
    assert all(path.read_bytes() == before[path] for path in paths)
    assert not list(paths[0].parent.glob('history/*'))


def test_second_independent_seal_failure_cannot_publish_the_first(sealed):
    runner, record, paths = sealed
    value = json.loads(paths[1].read_text())
    value['sources'][0]['sha256'] = value['sources'][0]['retained']['sha256'] = '0' * 64
    paths[1].write_text(json.dumps(value))
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match='content differs'):
        revalidate_input_seals(runner, record)
    assert all(path.read_bytes() == before[path] for path in paths)


def test_new_capture_version_is_never_aliased_to_the_old_seal(sealed, monkeypatch):
    from visioncortex import nas_recordings
    runner, record, paths = sealed
    monkeypatch.setattr(nas_recordings, '_inspect', lambda *args: record | {'source_signature': 'replacement'})
    with pytest.raises(ValueError, match='current input|Current input'):
        revalidate_input_seals(runner, record)


def test_live_stage_owner_blocks_revalidation(sealed):
    runner, record, paths = sealed
    queue = DeviceDayQueue(runner.runtime_root / 'queue-vision.sqlite3')
    queue.enqueue(record, 'revision')
    assert queue.claim('active-owner')
    with pytest.raises(ValueError, match='reconciled stage owners'):
        revalidate_input_seals(runner, record)


def test_timestamps_cannot_be_rewritten_even_when_content_matches(sealed):
    runner, record, paths = sealed
    value = json.loads(paths[0].read_text())
    value['sources'][0]['identity'][3] -= 1
    paths[0].write_text(json.dumps(value))
    with pytest.raises(ValueError, match='content differs'):
        revalidate_input_seals(runner, record)


def test_storage_identity_rejection_precedes_capture_access(sealed, monkeypatch):
    from visioncortex import device_day_write_guard, nas_recordings
    runner, record, paths = sealed
    runner.config['runtime']['local_only'] = False
    def forbidden(*args):
        raise AssertionError('Capture inspection ran before storage verification')
    monkeypatch.setattr(nas_recordings, '_inspect', forbidden)
    with pytest.raises(device_day_write_guard.NASUnavailable):
        revalidate_input_seals(runner, record)
