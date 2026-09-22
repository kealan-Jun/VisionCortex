"""Date/view input naming uses the existing device/day queue and archive."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import pytest

from visioncortex.config import load_config
from visioncortex.device_day_contract import DIRECTORIES, DeviceDayLayout
from visioncortex.input_availability import configured_record
from visioncortex.nas_recordings import scan_recordings


@pytest.fixture
def config(default_config, tmp_path):
    c = default_config
    c['device_day'].update(enabled=True, start_date=None, camera_lanes=True)
    c['collection_ingest'].update(enabled=True, mode='directory_metadata', settle_seconds=0, poll_seconds=1,
        source_root=str(tmp_path/'nas'), camera_directory_glob='*_cam*',
        additional_camera_directories=['lab_video'],
        directory_camera_bindings=[{'directory_glob': f'lab_video/20??-??-??-{n}/*',
                                    'camera_key': f'lab-video-{n}'} for n in (1, 3)],
        camera_group_map={'lab-video-1': 'lab-video', 'lab-video-3': 'lab-video'},
        camera_role_map={'same_cam01': 'third_person', 'lab-video-1': 'first_person',
                         'lab-video-3': 'third_person'})
    c['storage'].update(archive_root=str(tmp_path/'archive'), local_cache_root=str(tmp_path/'cache'),
                        local_runtime_root=str(tmp_path/'runtime'))
    return c


def capture(c, folder, *, closed=True):
    root = Path(c['collection_ingest']['source_root'])/folder
    root.mkdir(parents=True)
    (root/'rgb.mp4').write_bytes(b'fixture-not-decoded')
    (root/'frames.csv').write_text('global_timestamp_us,rgb_video_frame_index\n1789459935236097,0\n')
    m = {'camera_key': 'same_cam01', 'closed': closed, 'frames_publish_state': 'finalized',
         'recording_session_id': 42, 'rgb_file': 'rgb.mp4', 'frames_file': 'frames.csv',
         'segment_start_us': 1789459935236097, 'segment_end_us': 1789459945236097}
    (root/'meta.json').write_text(json.dumps(m))
    (root/'recording_ready.json').write_text(json.dumps(m | {'ready': True,
         'recording_complete': False, 'recording_quality_status': 'partial'}))
    return root


def test_lab_date_view_folders_join_existing_inventory_and_archive(config):
    capture(config, 'same_cam01/2026-09-15/161156')
    for view in (1, 3):
        capture(config, f'lab_video/2026-09-15-{view}/161156')
    result = scan_recordings(config)
    assert not result['errors'] and len(result['recordings']) == 3
    assert set(result['camera_directories']) == {'same_cam01', 'lab_video'}
    rows = {r['camera_key']: r for r in result['recordings']}
    assert set(rows) == {'same_cam01', 'lab-video-1', 'lab-video-3'}
    assert rows['same_cam01']['configured_role'] == 'third_person'
    for name in ('lab-video-1', 'lab-video-3'):
        row = rows[name]
        assert row['recorder_camera_key'] == 'same_cam01'
        assert row['processable'] and not row['capture_complete']
        assert row['audio']['files'] == []
        layout = DeviceDayLayout(Path(config['storage']['archive_root']), name, row['recording_start_us'])
        layout.create()
        assert layout.root.name == '2026-09-15_'+name
        assert set(p.name for p in layout.root.iterdir()) == set(DIRECTORIES)
    assert rows['lab-video-1']['configured_role'] == 'first_person'
    assert len(result['batches']) == 2
    batch = next(b for b in result['batches'] if b['source_group'] == 'lab-video')
    assert {r['camera_key'] for r in batch['recordings']} == {'lab-video-1', 'lab-video-3'}
    assert batch['device_day_processable']


def test_reinspection_preserves_alias_and_unknown_view_does_not_inherit_old_role(config):
    from visioncortex.nas_recordings import _inspect
    folder = capture(config, 'lab_video/2026-09-15-1/161156')
    raw = _inspect(Path(config['collection_ingest']['source_root']), folder/'rgb.mp4', time.time(), 0)
    mapped = configured_record(config, raw)
    assert configured_record(config, mapped) == mapped
    assert mapped['camera_key'] == 'lab-video-1'
    assert mapped['source_signature'] == raw['source_signature']
    capture(config, 'lab_video/2026-09-15-2/161156')
    unknown = next(r for r in scan_recordings(config)['recordings'] if '-2/' in r['relative_path'])
    assert unknown['configured_role'] is None
    assert unknown['camera_binding_status'] == 'needs_directory_binding'
    assert raw['camera_key'] == 'same_cam01'


def test_monitor_discovers_new_lab_date_in_the_existing_service(config):
    from visioncortex.device_day_monitor import CameraMonitor
    from visioncortex.device_day_service import DeviceDayService
    from visioncortex.observed_inventory import read_inventory
    capture(config, 'lab_video/2026-09-15-1/161156', closed=False)
    service = DeviceDayService(lambda: config, threading.Lock())
    stop = threading.Event()
    monitor = CameraMonitor(config, stop, service.observe)
    try:
        monitor.poll()
        capture(config, 'lab_video/2026-09-16-3/162103')
        deadline = time.monotonic()+4
        while len(monitor.records) < 2 or monitor.dirty or sum(r['processable'] for r in monitor.records.values()) != 1:
            assert time.monotonic() < deadline
            monitor.poll()
            time.sleep(.01)
        rows = read_inventory(Path(config['storage']['local_runtime_root'])/'device-day')['recordings']
        assert len(rows) == 2 and sum(r['processable'] for r in rows) == 1
        assert {r['camera_key'] for r in rows} == {'lab-video-1', 'lab-video-3'}
    finally:
        stop.set()
        for thread in monitor.threads.values():
            thread.join(timeout=3)


def test_cross_view_pairing_uses_the_same_algorithms_within_each_lab(config, monkeypatch):
    from visioncortex import device_day_multiview as module
    from visioncortex.device_day_cross_view import associate
    from visioncortex.device_day_timeline import build_timeline, day_bounds
    day = '2026-09-15'
    start = day_bounds(day)[0]
    names = ['same_cam01', 'lab-video-1', 'lab-video-3']
    indexes = [(day+'_'+name, {'segments': [{'segment_id': name, 'recording_id': name,
        'start_us': start, 'end_us': start+1000000, 'start_ms': 0, 'end_ms': 1000,
        'activity': 'active', 'key_frames': [], 'scene_frames': [],
        'source_ref': {'path': 'MetaVideo/a.mp4', 'sha256': name}}]}) for name in names]
    entries = build_timeline(day, indexes)['entries']
    edges = associate(entries, config['collection_ingest']['camera_role_map'], config, audit=False)
    assert len(edges) == 1
    assert 'lab-video-1' in edges[0]['first_person'] and 'lab-video-3' in edges[0]['third_person']
    seen = []
    def shared(c, date, rows):
        cameras = {name[11:] for name, _ in rows}
        seen.append(cameras)
        assert c is config and date == day
        return {'status': 'completed', 'key': str(len(seen)), 'experiments': [{'cameras': sorted(cameras)}],
                'algorithms': ['existing_shared_analysis'], 'aligned_recordings': []}
    monkeypatch.setattr(module, '_build_source_multiview', shared)
    result = module.build_device_day_multiview(config, day, indexes)
    assert seen == [{'same_cam01'}, {'lab-video-1', 'lab-video-3'}]
    assert result['status'] == 'completed' and len(result['experiments']) == 2
    assert result['algorithms'] == ['existing_shared_analysis']


def test_ambiguous_binding_fails_closed_and_other_profiles_do_not_scan_lab(config):
    c = deepcopy(config)
    c['collection_ingest']['directory_camera_bindings'] *= 2
    capture(c, 'lab_video/2026-09-15-1/161156')
    result = scan_recordings(c)
    assert not result['recordings'] and 'ambiguous' in result['errors'][0]['message']
    for name in ('rtx3090ti-ubuntu-local.yaml', 'rtx3050-6gb-ubuntu20-production.yaml'):
        settings = load_config(Path('configs')/name)['collection_ingest']
        assert not settings['additional_camera_directories'] and not settings['directory_camera_bindings']


@pytest.mark.parametrize('path', ['../elsewhere', '/outside', 'lab_video/../outside/*'])
def test_binding_cannot_escape_the_configured_capture_directory(config, path):
    from visioncortex.capture_layout import validate
    config['collection_ingest']['directory_camera_bindings'][0]['directory_glob'] = path
    with pytest.raises(ValueError):
        validate(config['collection_ingest'])
