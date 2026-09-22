"""Laboratory isolation with local recorder fixtures; no NAS/models/cloud."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import pytest

from visioncortex.lab_sources import additional_laboratories, laboratory_configs, select_laboratory
from visioncortex.nas_recordings import scan_recordings
from visioncortex.runtime_control import resource_database, resource_slot


@pytest.fixture
def lab_config(default_config, tmp_path):
    config = default_config
    config['device_day'].update(enabled=True, start_date=None, camera_lanes=True,
        additional_labs=[{'name': 'LabVideo', 'source_root': str(tmp_path/'nas/lab_video'),
            'archive_root': str(tmp_path/'nas/lab_video/VisionCortexExperimentArchive'),
            'cache_root': str(tmp_path/'nas/lab_video/VisionCortexExperimentCache'),
            'runtime_root': str(tmp_path/'runtime/LabVideo'),
            'camera_directory_glob': '20??-??-??-*',
            'camera_role_map': {'shared_cam01': 'first_person'}}])
    config['collection_ingest'].update(enabled=True, mode='directory_metadata',
        source_root=str(tmp_path/'nas'), camera_directory_glob='*_cam*',
        camera_role_map={'shared_cam01': 'third_person'}, settle_seconds=0)
    config['storage'].update(archive_root=str(tmp_path/'nas/Archive'),
        local_cache_root=str(tmp_path/'nas/Cache'), local_runtime_root=str(tmp_path/'runtime'))
    config['runtime'] = {'resource_limits': {'vision': 1}, 'admission_timeout_seconds': .1}
    return config


def capture(root, folder, *, closed=True):
    path = root/folder
    path.mkdir(parents=True)
    (path/'rgb.mp4').write_bytes(b'fixture-not-decoded')
    (path/'frames.csv').write_text('global_timestamp_us,rgb_video_frame_index\n1789459935236097,0\n')
    meta = {'camera_key': 'shared_cam01', 'closed': closed, 'frames_publish_state': 'finalized',
            'recording_session_id': 42, 'rgb_file': 'rgb.mp4', 'frames_file': 'frames.csv',
            'segment_start_us': 1789459935236097, 'segment_end_us': 1789459945236097}
    (path/'meta.json').write_text(json.dumps(meta))
    (path/'recording_ready.json').write_text(json.dumps(meta | {
        'ready': True, 'recording_complete': False, 'recording_quality_status': 'partial'}))
    return path


def test_date_view_layout_is_monitored_without_cross_lab_identity_or_audio(lab_config):
    from visioncortex.device_day_service import DeviceDayService
    from visioncortex.device_day_monitor import CameraMonitor
    from visioncortex.observed_inventory import read_inventory
    child = select_laboratory(lab_config, 'LabVideo')
    capture(Path(lab_config['collection_ingest']['source_root']), 'shared_cam01/2026-09-15/161156')
    capture(Path(child['collection_ingest']['source_root']), '2026-09-15-1/161156')
    capture(Path(child['collection_ingest']['source_root']), '2026-09-15-1/162103', closed=False)
    parent_records = scan_recordings(lab_config)['recordings']
    assert len(parent_records) == 1
    assert parent_records[0]['configured_role'] == 'third_person'
    service = DeviceDayService(lambda: child, threading.Lock())
    stop = threading.Event()
    monitor = CameraMonitor(child, stop, service.observe)
    try:
        monitor.poll()
        deadline = time.monotonic()+3
        while len(monitor.records) != 2 or monitor.dirty:
            assert time.monotonic() < deadline
            monitor.poll()
            time.sleep(.01)
        rows = read_inventory(Path(child['storage']['local_runtime_root'])/'device-day')['recordings']
        assert len(rows) == 2
        assert {r['camera_key'] for r in rows} == {'shared_cam01'}
        assert {r['configured_role'] for r in rows} == {'first_person'}
        assert sum(r['processable'] for r in rows) == 1
        assert all(not r['capture_complete'] for r in rows)
        assert all(r['audio']['files'] == [] for r in rows)
        assert not (Path(lab_config['storage']['local_runtime_root'])/'device-day').exists()
    finally:
        stop.set()
        for thread in monitor.threads.values():
            thread.join(timeout=3)


def test_laboratory_settings_isolate_recovery_and_keep_same_model_pool_identity(lab_config):
    lab_config['device_day']['completed_vision_receipts'] = {'path': '/parent/receipt'}
    lab_config['device_day']['completed_stage_receipts'] = {'path': '/parent/stages'}
    before = deepcopy(lab_config)
    child = select_laboratory(lab_config, 'LabVideo')
    assert lab_config == before
    assert child['models'] == lab_config['models']
    assert child['performance'] == lab_config['performance']
    assert child['device_day']['additional_labs'] == []
    assert 'completed_vision_receipts' not in child['device_day']
    assert 'completed_stage_receipts' not in child['device_day']
    for key in ('archive_root', 'local_runtime_root', 'local_cache_root', 'local_input_root', 'index_csv'):
        assert child['storage'][key] != lab_config['storage'][key]
    assert resource_database(child) == resource_database(lab_config)


def test_independent_lab_queues_share_actual_resource_admission(lab_config):
    child = select_laboratory(lab_config, 'LabVideo')
    with resource_slot(lab_config, 'vision'):
        with pytest.raises(TimeoutError, match='admission'):
            with resource_slot(child, 'vision'):
                pytest.fail('Second lab bypassed the shared capacity')
    with resource_slot(child, 'vision'):
        assert resource_database(child).is_file()


@pytest.mark.parametrize('key,value', [
    ('archive_root', 'parent_archive'), ('cache_root', 'parent_cache'),
    ('runtime_root', 'parent_runtime'), ('source_root', '../outside'),
    ('runtime_root', '/mnt/runtime'),
])
def test_reject_lab_storage_collisions(lab_config, key, value):
    aliases = {'parent_archive': 'archive_root', 'parent_cache': 'local_cache_root',
               'parent_runtime': 'local_runtime_root'}
    lab_config['device_day']['additional_labs'][0][key] = lab_config['storage'][aliases[value]] if value in aliases else value
    with pytest.raises(ValueError):
        laboratory_configs(lab_config)


def test_local_profile_and_other_machine_do_not_inherit_laboratory(monkeypatch):
    from visioncortex.config import load_config
    for name in ('rtx3090ti-ubuntu-local.yaml', 'rtx3050-6gb-ubuntu20-production.yaml'):
        assert laboratory_configs(load_config(Path('configs')/name)) == {}
    monkeypatch.setenv('VISIONCORTEX_LOCAL_RUNTIME_ROOT', '/temporary/deployed-runtime')
    config = load_config(Path('configs/rtx3090ti-ubuntu-production.yaml'))
    child = select_laboratory(config, 'LabVideo')
    assert child['storage']['local_runtime_root'].endswith('/Runtime/LabVideo')
    assert child['runtime']['resource_root'] == '/temporary/deployed-runtime'


def test_lab_selection_fails_closed_in_web_settings(lab_config, monkeypatch):
    from visioncortex import api
    monkeypatch.setattr(api, 'load_config', lambda *args: deepcopy(lab_config))
    monkeypatch.setattr(api.ai_settings, 'apply_active', lambda value: value)
    monkeypatch.setenv('VISIONCORTEX_LAB', 'LabVideo')
    assert api._settings()['collection_ingest']['source_root'].endswith('/lab_video')
    monkeypatch.setenv('VISIONCORTEX_LAB', 'MissingLab')
    with pytest.raises(ValueError, match='Unknown laboratory'):
        api._settings()


def test_lab_worker_owns_isolated_monitor_and_stops_before_releasing_owner(lab_config, monkeypatch):
    from visioncortex import device_day_service, runtime_process
    active, events = [], []
    @contextmanager
    def owner(config):
        events.append('owned')
        yield
        assert all(s.stop_event.is_set() for s in active)
        events.append('released')
    monkeypatch.setattr(runtime_process, 'worker_owner', owner)
    monkeypatch.setattr(device_day_service.DeviceDayService, 'start', lambda self: active.append(self))
    monkeypatch.setattr(device_day_service.DeviceDayService, 'stop', lambda self: events.append('stopped'))
    observed = threading.Event()
    def monitor(config, *, service, stop, publish):
        publish(config, {'monitor': {'status': 'watching'}})
        observed.set()
        stop.wait(3)
    primary = device_day_service.DeviceDayService(lambda: lab_config, threading.Lock())
    with additional_laboratories(lambda: lab_config, primary, monitor):
        assert observed.wait(2)
        assert active[0].gpu_lock is primary.gpu_lock
        assert not active[0].stop_event.is_set()
        assert not primary.stop_event.is_set()
    assert events == ['owned', 'stopped', 'released']
    root = Path(select_laboratory(lab_config, 'LabVideo')['storage']['local_runtime_root'])
    assert json.loads((root/'state/nas-recording-monitor.json').read_text())['monitor']['status'] == 'watching'
