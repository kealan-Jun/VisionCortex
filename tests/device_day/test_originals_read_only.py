"""Production originals remain collector-owned, including legacy archives."""
from pathlib import Path
import os
import time

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_capture_files import retain_capture_files
from visioncortex.device_day_contract import file_hash, read_json
from visioncortex.device_day_inputs import InputChanged, copy_sealed


@pytest.mark.parametrize('inplace', [False, True])
def test_missing_historical_original_is_not_recreated(device_config, inplace):  # noqa: F811
    device_config['device_day'].update(raw_originals_read_only=True, inplace_preprocessing=inplace)
    capture(device_config)
    record, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    layout = runner.layout(record)
    layout.raw.mkdir(parents=True)
    original = Path(record['video_path'])
    before = original.read_bytes(), original.stat().st_mtime_ns
    result = runner.process(record, stage='retention')
    assert result['status'] == 'failed', result
    assert not list(layout.raw.rglob('*'))
    assert (original.read_bytes(), original.stat().st_mtime_ns) == before


def test_existing_originals_are_verified_and_new_catalog_is_hidden(device_config):  # noqa: F811
    capture(device_config)
    record, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    old_layout = runner.layout(record)
    before = {p.relative_to(old_layout.raw): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in old_layout.raw.rglob('*') if p.is_file()}
    device_config['device_day']['raw_originals_read_only'] = True
    readonly = DeviceDayRunner(device_config, FakeModels())
    assert readonly.process(record, stage='retention')['status'] == 'completed'
    reference = retain_capture_files(readonly, record)
    assert reference['path'].startswith('ProcessedClips/.Evidence/CaptureMetadata/')
    assert read_json(old_layout.root / reference['path'])['capture_sources_changed'] is False
    assert before == {p.relative_to(old_layout.raw): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in old_layout.raw.rglob('*') if p.is_file()}


def test_readonly_copy_guard_rejects_before_creating_any_path(tmp_path):
    source = tmp_path / 'capture.mp4'
    source.write_bytes(b'owned source')
    target = tmp_path / 'archive' / 'MetaVideo' / 'clip.mp4'
    with pytest.raises(InputChanged, match='creation is disabled'):
        copy_sealed({'device_day': {'raw_originals_read_only': True}},
                    {'original_path': str(source), 'retained': {}}, target)
    assert not target.parent.parent.exists()
    assert source.read_bytes() == b'owned source'


def test_catalog_updates_preserve_complete_old_bytes_and_reference(device_config):  # noqa: F811
    capture(device_config)
    record, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    device_config['device_day']['raw_originals_read_only'] = True
    readonly = DeviceDayRunner(device_config, FakeModels())
    first = retain_capture_files(readonly, record)
    layout = readonly.layout(record)
    first_path = layout.root / first['path']
    before = first_path.read_bytes()
    # A collector-owned additional source is missing a historical retained copy.
    # It remains pending, and must change only the generated catalog version.
    note = Path(record['video_path']).with_name('late-note.txt')
    note.write_bytes(b'owned late capture metadata')
    import os
    os.utime(note, (1, 1))
    second = retain_capture_files(readonly, record)
    assert second['path'] != first['path']
    assert first_path.read_bytes() == before and file_hash(first_path) == first['sha256']
    assert read_json(layout.root / second['path'])['pending'][0]['reason'] == 'original_destination_absent_read_only'
    history = readonly._receipt(layout, record, 'retention').parent / 'history'
    assert first in [read_json(p) for p in history.glob('capture-files-*.json')]
    assert note.read_bytes() == b'owned late capture metadata'


def test_retention_readonly_target_disappearing_after_guard_is_not_recreated(device_config, monkeypatch):  # noqa: F811
    from visioncortex import device_day
    capture(device_config)
    record, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    device_config['device_day']['raw_originals_read_only'] = True
    readonly = DeviceDayRunner(device_config, FakeModels())
    layout = readonly.layout(record)
    original = Path(record['video_path'])
    before = original.read_bytes(), original.stat().st_ino
    removed = []
    copy = device_day.copy_verified
    def disappearing(source, target, **kwargs):
        assert kwargs['allow_create'] is False
        assert target.is_file()  # The outer caller already passed its guard.
        target.unlink()
        removed.append(target)
        return copy(source, target, **kwargs)
    monkeypatch.setattr(device_day, 'copy_verified', disappearing)
    with pytest.raises(ValueError, match='creating a raw copy is disabled'):
        device_day.retained_recording(layout, record, device_config)
    assert removed and all(not path.exists() for path in removed)
    assert not list(layout.raw.rglob('*.partial'))
    assert (original.read_bytes(), original.stat().st_ino) == before


def test_capture_catalog_readonly_target_disappearing_after_guard_is_not_recreated(
        device_config, monkeypatch):  # noqa: F811
    from visioncortex import device_day
    folder = capture(device_config)
    extra = folder / 'note.txt'
    extra.write_bytes(b'owned recorder note')
    os.utime(extra, (time.time() - 1000, time.time() - 1000))
    record, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    first = retain_capture_files(runner, record)
    layout = runner.layout(record)
    notes = [entry for entry in read_json(layout.root / first['path'])['files'] if entry['original_name'] == 'note.txt']
    assert len(notes) == 1
    destination = layout.root / notes[0]['retained']['path']
    before = extra.read_bytes(), extra.stat().st_ino
    device_config['device_day']['raw_originals_read_only'] = True
    readonly = DeviceDayRunner(device_config, FakeModels())
    copy = device_day.copy_verified
    def disappearing(source, target, **kwargs):
        assert target == destination and kwargs['allow_create'] is False
        target.unlink()
        return copy(source, target, **kwargs)
    monkeypatch.setattr(device_day, 'copy_verified', disappearing)
    with pytest.raises(ValueError, match='creating a raw copy is disabled'):
        retain_capture_files(readonly, record)
    assert not destination.exists()
    assert not list(layout.raw.rglob('*.partial'))
    assert (extra.read_bytes(), extra.stat().st_ino) == before
