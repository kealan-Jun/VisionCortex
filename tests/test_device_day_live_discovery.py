from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import threading
from zoneinfo import ZoneInfo

from test_device_day import capture, device_config  # noqa: F401
from visioncortex import device_day_monitor
from visioncortex.device_day_live_discovery import discovery_config
from visioncortex.observed_inventory import read_inventory


def test_config_covers_configured_roles_and_physical_bindings_only():
    source = {'storage': {}, 'device_day': {'enabled': False}, 'collection_ingest': {
        'camera_role_map': {'a_cam01': 'first_person', 'b_cam02': 'third_person',
                            'lab-video-1': 'first_person', 'unknown': 'unset'},
        'directory_camera_bindings': [{'camera_key': 'lab-video-1', 'directory_glob': 'lab_video/20??-??-??-1/*'}]}}
    before = deepcopy(source)
    configured = discovery_config(source)
    assert configured['collection_ingest']['camera_directories'] == ['a_cam01', 'b_cam02', 'lab_video']
    assert configured['collection_ingest']['poll_seconds'] == 5
    assert configured['device_day']['enabled']
    assert source == before


def test_previous_day_late_closure_is_discovered_but_older_days_are_not_walked(device_config, monkeypatch):  # noqa: F811
    from visioncortex.device_day_service import DeviceDayService
    today = datetime.now(ZoneInfo('Asia/Shanghai')).date()
    root = Path(device_config['collection_ingest']['source_root'])
    camera = root / 'a_cam01'
    for offset in (0, 1, 2):
        day = today - timedelta(days=offset)
        start = round(datetime.combine(day, datetime.min.time(), ZoneInfo('Asia/Shanghai')).timestamp()*1e6)
        made = capture(device_config, start=start, label=f'fixture{offset}')
        target = camera / day.isoformat() / made.name
        target.parent.mkdir(parents=True, exist_ok=True)
        made.rename(target)
    config = discovery_config(device_config)
    config['collection_ingest']['camera_directories'] = ['a_cam01']
    stop = threading.Event()
    observed = []
    service = DeviceDayService(lambda: config, threading.Lock())
    real_scan = device_day_monitor.scan_recordings
    def scan(current, **kwargs):
        skipped = kwargs['skip_folders']
        assert str(camera / (today-timedelta(days=2)).isoformat()) in skipped
        assert current['collection_ingest']['capture_since_date'] == (today-timedelta(days=1)).isoformat()
        result = real_scan(current, **kwargs)
        observed.extend(result['recordings'])
        stop.set()
        return result
    monkeypatch.setattr(device_day_monitor, 'scan_recordings', scan)
    monitor = device_day_monitor.CameraMonitor(config, stop, service.observe, live_only=True)
    monitor.poll(include_recordings=False)
    monitor.threads['a_cam01', 'live'].join(5)
    assert len(observed) == 2
    stored = read_inventory(Path(config['storage']['local_runtime_root']) / 'device-day')
    assert len(stored['recordings']) == 2
    assert set(monitor.threads) == {('a_cam01', 'live')}


def test_summary_does_not_build_day_batches_and_dead_lane_restarts(tmp_path, monkeypatch):
    config = {'storage': {'local_runtime_root': str(tmp_path)}, 'device_day': {},
              'collection_ingest': {}}
    monkeypatch.setattr(device_day_monitor, '_root', lambda _: tmp_path)
    monkeypatch.setattr(device_day_monitor, '_camera_directories', lambda *a: [tmp_path/'camera'])
    monkeypatch.setattr(device_day_monitor, '_recording_batches', lambda *a: (_ for _ in ()).throw(AssertionError('batch build')))
    stop = threading.Event()
    monitor = device_day_monitor.CameraMonitor(config, stop, lambda *a: None, live_only=True)
    first, second = threading.Event(), threading.Event()
    def lane(camera, mode):
        if not first.is_set():
            first.set()
            raise RuntimeError('owned unexpected lane failure')
        second.set()
        stop.wait(5)
    monkeypatch.setattr(monitor, '_lane', lane)
    monitor.poll(include_recordings=False)
    assert first.wait(2)
    monitor.threads['camera', 'live'].join(2)
    monitor.poll(include_recordings=False)
    assert second.wait(2)
    assert monitor.restarts['camera', 'live'] == 1
    stop.set()
    monitor.threads['camera', 'live'].join(2)


def test_unavailable_source_keeps_existing_catalog(device_config, monkeypatch):  # noqa: F811
    from visioncortex.device_day_service import DeviceDayService
    from visioncortex.observed_inventory import observe
    config = discovery_config(device_config)
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    previous = {'recording_id': 'existing', 'camera_key': 'a_cam01', 'recording_start_us': 1789005600000000}
    observe(root, {'recordings': [previous]})
    stop = threading.Event()
    service = DeviceDayService(lambda: config, threading.Lock())
    monitor = device_day_monitor.CameraMonitor(config, stop, service.observe, live_only=True)
    original_wait = stop.wait
    def wait(timeout):
        stop.set()
        return original_wait(0)
    monkeypatch.setattr(stop, 'wait', wait)
    monitor._lane('missing_camera', 'live')
    assert monitor.states['missing_camera', 'live']['status'] == 'retrying'
    assert read_inventory(root)['recordings'] == [previous]
