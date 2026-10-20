"""An explicit first role binding releases vision without repeating audio."""
from pathlib import Path
import json
import os

import pytest

from test_device_day import FakeModels, device_config  # noqa: F401
from test_recorder_metadata import config, incoming  # noqa: F401
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.nas_recordings import scan_recordings


class Models(FakeModels):
    def __init__(self):
        super().__init__()
        self.stt_calls = 0

    def transcribe(self, *args):
        self.stt_calls += 1
        return super().transcribe(*args)


def run_stage(runner, record, stage):
    worker = RetentionWorker(stage=stage)
    assert worker.admit(runner, record)
    queue = runner.queues[stage]
    claimed = queue.claim('fixture-' + stage, allowed={record['recording_id']})
    assert claimed is not None
    result = runner.process(claimed, stage=stage)
    queue.finish('fixture-' + stage, record['recording_id'], result, .001)
    assert result['status'] == 'completed', result.get('message', result)
    return result


def test_late_explicit_role_releases_pipeline_with_unchanged_stt_and_originals(config):  # noqa: F811
    folder = incoming(config, camera='new_cam01')
    config['collection_ingest'].update(discovery_mode='metadata_only', camera_role_map={})
    original, = scan_recordings(config)['recordings']
    models = Models()
    runner = DeviceDayRunner(config, models)
    for stage in ('retention', 'stt'):
        run_stage(runner, original, stage)
    assert runner.process(original, stage='vision')['status'] == 'needs_camera_role'
    assert models.stt_calls == 1 and models.vision_calls == 0
    layout = runner.layout(original)
    old_vision_seal = runner._receipt(layout, original, 'input-vision').read_bytes()
    old_stt_seal = runner._receipt(layout, original, 'input-stt').read_bytes()
    old_stt = runner._receipt(layout, original, 'stt').read_bytes()
    old_retention = runner._receipt(layout, original, 'retention').read_bytes()
    before = {path: (path.read_bytes(), path.stat().st_ino) for path in folder.rglob('*') if path.is_file()}
    config['collection_ingest']['camera_role_map']['new_cam01'] = 'third_person'
    fresh, = scan_recordings(config)['recordings']
    assert fresh['source_signature'] == original['source_signature']
    assert fresh['configured_role'] == 'third_person'
    runner = DeviceDayRunner(config, models)
    for stage in ('vision', 'understanding', 'report'):
        run_stage(runner, fresh, stage)
    assert runner.process(fresh, stage='stt')['status'] == 'completed'
    assert models.vision_calls == models.semantic_calls == models.stt_calls == 1
    assert runner._receipt(layout, fresh, 'input-stt').read_bytes() == old_stt_seal
    assert runner._receipt(layout, fresh, 'stt').read_bytes() == old_stt
    from visioncortex.input_availability import IntegrityMonitor, revalidated_input_seals
    with revalidated_input_seals(runner, fresh) as proof:
        assert proof['verified_seals'] == 2 and proof['changed_seals'] == []
    with runner.queues['retention'].connect() as db:
        retained_row = dict(db.execute('SELECT * FROM recordings').fetchone())
    assert IntegrityMonitor(config, runner)._check('retention', retained_row)['status'] == 'verified'
    assert runner._receipt(layout, fresh, 'input-stt').read_bytes() == old_stt_seal
    history = runner._receipt(layout, fresh, 'retention').parent / 'history'
    assert any(path.read_bytes() == old_vision_seal for path in history.glob('input-vision-*.json'))
    assert any(path.read_bytes() == old_retention for path in history.glob('retention-*.json'))
    assert read_json(runner._receipt(layout, fresh, 'publication'))['status'] == 'completed'
    assert all((path.read_bytes(), path.stat().st_ino) == expected for path, expected in before.items())
    assert Path(fresh['video_path']) == folder / 'rgb.mp4'


@pytest.mark.parametrize('change', ['known_role', 'forged_role', 'hardware_metadata'])
def test_role_supplement_rejects_known_view_change_or_unproven_source(config, change):  # noqa: F811
    folder = incoming(config, camera='new_cam01')
    config['collection_ingest'].update(discovery_mode='metadata_only', camera_role_map={})
    meta = folder / 'meta.json'
    data = read_json(meta)
    data['camera_announce_raw'] = {'device': {'vendor': 'fixture', 'serial_number': 'physical-1'}}
    meta.write_text(json.dumps(data))
    os.utime(meta, (1_789_000_000, 1_789_000_000))
    if change == 'known_role':
        config['collection_ingest']['camera_role_map']['new_cam01'] = 'first_person'
    original, = scan_recordings(config)['recordings']
    models = Models()
    runner = DeviceDayRunner(config, models)
    assert runner.process(original, stage='retention')['status'] == 'completed'
    layout = runner.layout(original)
    paths = [runner._receipt(layout, original, name) for name in ('input-vision', 'input-stt', 'retention')]
    before = {path: path.read_bytes() for path in paths}
    if change == 'hardware_metadata':
        old = meta.stat()
        raw = meta.read_bytes()
        meta.write_bytes(raw.replace(b'physical-1', b'physical-2'))
        os.utime(meta, ns=(old.st_atime_ns, old.st_mtime_ns))
    if change != 'forged_role':
        config['collection_ingest']['camera_role_map']['new_cam01'] = 'third_person'
    fresh, = scan_recordings(config)['recordings']
    if change == 'forged_role':
        fresh['configured_role'] = 'third_person'
    assert fresh['source_signature'] == original['source_signature']
    current = DeviceDayRunner(config, models)
    result = current.process(fresh, stage='vision')
    assert result['status'] == 'failed' and result['error_type'] == 'InputChanged'
    assert all(path.read_bytes() == prior for path, prior in before.items())
    assert models.vision_calls == models.semantic_calls == 0
