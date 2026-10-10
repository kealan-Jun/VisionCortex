from copy import deepcopy
from pathlib import Path

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from test_device_day_file_index import config
from test_device_day_time_lookup import ARCHIVE, BASE, index
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_file_index import FileIndexPublisher
from visioncortex.device_day_live_file_index import LiveFileIndex
from visioncortex.observed_inventory import observe


def test_explicit_archives_filter_precedes_archive_reads_and_default_retains_history(tmp_path):
    cfg = config(tmp_path)
    other = '2026-09-17_other_cam02'
    runtime = Path(cfg['storage']['local_runtime_root'])/'device-day'
    root = Path(cfg['storage']['archive_root'])
    observe(runtime, {'recordings': [
        {'recording_id': 'a', 'camera_key': ARCHIVE[11:], 'recording_start_us': BASE},
        {'recording_id': 'b', 'camera_key': other[11:], 'recording_start_us': BASE}]})
    atomic_json(root/ARCHIVE/'ProcessedClips/Index.json', index())
    # A forbidden/unavailable other archive would fail if targeted. Filtering
    # must prevent even that source-index read, not just skip its later output.
    bad = root/other/'ProcessedClips/Index.json'
    bad.parent.mkdir(parents=True)
    bad.write_text('malformed fixture')
    publisher = FileIndexPublisher(cfg)
    scoped = publisher.tick(archives=[ARCHIVE])
    assert scoped['published'] == 1 and scoped['errors'] == []
    assert not (root/other/'Comment/TimeIndex.json').exists()
    full = publisher.tick()
    assert any(e['archive'] == other for e in full['errors'])


def test_partial_windows_restore_after_external_report_and_understanding_overwrite(device_config):  # noqa: F811
    from visioncortex.device_day_content import time_folder
    from visioncortex.device_day_reports import render_day
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    runner.process(record, stage='retention')
    runner.process(record, stage='vision')
    canonical = read_json(layout.index)
    original_bytes = layout.index.read_bytes()
    segment = canonical['segments'][0]
    stage_path = layout.receipts/record['recording_id']/'understanding.json'
    key = 'a'*64
    atomic_json(stage_path, {'key': key, 'status': 'running'})
    window = layout.understanding/'ClipUnderstanding'/segment['segment_id']/key/'00-00_00-30'
    frame = segment['scene_frames'][0]
    atomic_json(window/'Input.json', {'input_key': 'owned-input', 'metadata': {
        'segment_id': segment['segment_id'], 'source_ref': segment['source_ref'],
        'start_ms': 0, 'end_ms': 30000, 'frames': [frame], 'comments': []}})
    atomic_json(window/'Result.json', {'input_key': 'owned-input', 'model_result': {
        'status': 'completed', 'summary': 'Owned partial window', 'activity_observed': 'inactive',
        'steps': [], 'frame_observations': [{'frame_id': frame['frame_id'], 'text': 'fixture'}], 'uncertainties': []}})
    observe(runner.runtime_root, {'recordings': [record]})
    device_config['device_day']['readable_content_enabled'] = True
    publisher = FileIndexPublisher(device_config)
    assert publisher.tick(archives=[layout.name])['published'] == 1
    report = layout.reports/'LaboratoryDailyReport.json'
    readable = layout.understanding/time_folder(segment['start_us'], segment['end_us'])/'Understanding.json'
    assert read_json(report)['entries'][0]['understanding_status'] == 'partial'
    assert read_json(readable)['understanding']['status'] == 'partial'
    assert publisher.tick(archives=[layout.name])['unchanged'] == 1
    # Emulate the older companion's renderer; execution and inputs are unchanged.
    render_day(layout, deepcopy(canonical))
    atomic_json(readable, {'status': 'pending', 'understanding': None})
    assert read_json(report)['entries'][0]['understanding_status'] != 'partial'
    assert publisher.tick(archives=[layout.name])['published'] == 1
    assert read_json(report)['entries'][0]['understanding_status'] == 'partial'
    assert read_json(readable)['understanding']['status'] == 'partial'
    assert publisher.tick(archives=[layout.name])['unchanged'] == 1
    assert read_json(stage_path)['status'] == 'running'
    assert layout.index.read_bytes() == original_bytes
    assert backend.semantic_calls == 0 and backend.vision_calls == 1


def test_live_selection_is_local_recent_bounded_and_rotates_cameras(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    runtime = Path(cfg['storage']['local_runtime_root'])/'device-day'
    current = BASE/1e6 + 100
    rows = [{'recording_id': str(i), 'camera_key': f'camera{i}_cam01', 'configured_role': 'first_person',
             'recording_start_us': BASE-i*1000000, 'recording_end_us': BASE+1} for i in range(3)]
    rows.append({'recording_id': 'old', 'camera_key': 'old_cam01', 'configured_role': 'first_person',
                 'recording_start_us': BASE-86400000000, 'recording_end_us': BASE-86300000000})
    cfg['device_day'] = {}
    observe(runtime, {'recordings': rows})
    worker = LiveFileIndex(cfg)
    seen = []
    monkeypatch.setattr(worker.publisher, 'tick', lambda *, archives: seen.append(archives) or {'published': 0})
    assert len(worker.tick(now=current)['archives']) == 2
    assert seen[0] == ['2026-09-17_camera0_cam01', '2026-09-17_camera1_cam01']
    assert worker.tick(now=current+5)['archives'][0] == '2026-09-17_camera2_cam01'
    assert not Path(cfg['storage']['archive_root']).exists()
