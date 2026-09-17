"""Real local locks and deterministic scans; no NAS or model invocation."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from visioncortex import device_day_monitor as module
from visioncortex.device_day import exclusive
from visioncortex.observed_inventory import read_inventory
from visioncortex.device_day_service import DeviceDayService


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.01)


def test_live_only_monitor_keeps_old_closed_folders_out_of_scans(tmp_path, monkeypatch):
    from visioncortex.observed_inventory import observe
    old = {'recording_id': 'old', 'recording_start_us': 10, 'recording_end_us': 20,
           'processable': True, 'video_path': str(tmp_path / 'camera/old/rgb.mp4')}
    observe(tmp_path / 'device-day', {'recordings': [old]})
    settings = {'storage': {'local_runtime_root': str(tmp_path)},
                'device_day': {'process_since_us': 30}, 'collection_ingest': {}}
    stop = threading.Event()
    monkeypatch.setattr(module, '_root', lambda _: tmp_path)
    monkeypatch.setattr(module, '_camera_directories', lambda *a: [tmp_path / 'camera'])
    monkeypatch.setattr(module, '_recording_batches', lambda *a: [])
    seen = []
    def scan(config, on_record, skip_folders):
        seen.append(skip_folders)
        stop.set()
        return {'errors': []}
    monkeypatch.setattr(module, 'scan_recordings', scan)
    monitor = module.CameraMonitor(settings, stop, lambda *a: None)
    monitor.poll()
    monitor.threads['camera', 'live'].join(2)
    assert set(monitor.threads) == {('camera', 'live')}
    assert seen == [{str(tmp_path / 'camera/old')}]


def test_concurrent_inventory_writers_preserve_every_camera(tmp_path, monkeypatch):
    from visioncortex import device_day_service, device_day_latency
    config = {'device_day': {'enabled': True}, 'storage': {'local_runtime_root': str(tmp_path)}}
    service = DeviceDayService(lambda: config, None)
    monkeypatch.setattr(device_day_latency, 'observe', lambda *args: None)
    real_write = device_day_service.atomic_json
    barrier = threading.Barrier(12)

    def slow_write(*args):
        # Widen the real OS-lock contention window deterministically enough
        # for all simultaneous callers to enter, without changing lock code.
        time.sleep(.02)
        return real_write(*args)

    monkeypatch.setattr(device_day_service, 'atomic_json', slow_write)

    def observe(i):
        barrier.wait(timeout=3)
        record = {'recording_id': str(i), 'camera_key': f'camera{i}_cam01',
                  'recording_start_us': 1789005600000000}
        service.observe(config, {'recordings': [record], 'errors': [{'path': f'error{i}'}]})

    with ThreadPoolExecutor(max_workers=12) as workers:
        list(workers.map(observe, range(12)))
    data = read_inventory(tmp_path / 'device-day')
    assert {r['recording_id'] for r in data['recordings']} == {str(i) for i in range(12)}
    assert {r['path'] for r in data['errors']} == {f'error{i}' for i in range(12)}
    assert service.wakeup.is_set()


def test_legacy_file_lock_does_not_block_incremental_discoveries(tmp_path, monkeypatch):
    from visioncortex import device_day_latency
    monkeypatch.setattr(device_day_latency, 'observe', lambda *args: None)
    monkeypatch.setattr(module, '_root', lambda config: tmp_path)
    monkeypatch.setattr(module, '_camera_directories', lambda *args: [])
    monkeypatch.setattr(module, '_recording_batches', lambda *args: [])
    config = {'device_day': {'enabled': True}, 'storage': {'local_runtime_root': str(tmp_path)},
              'collection_ingest': {}}
    service = DeviceDayService(lambda: config, None)
    monitor = module.CameraMonitor(config, threading.Event(), service.observe)
    record = {'recording_id': 'slice', 'camera_key': 'camera_cam01', 'recording_start_us': 1789005600000000}
    monitor.dirty['slice'] = record
    monitor.discovery['slice'] = {'observed_at': 42}
    with exclusive(tmp_path / 'device-day/observed-inventory.lock'):
        snapshot = monitor.poll()
        assert snapshot['pending_publication_count'] == 0
    assert read_inventory(tmp_path / 'device-day')['recordings'] == [record]


def test_slow_publication_keeps_new_revision_and_does_not_lock_scanners():
    entered, release = threading.Event(), threading.Event()
    written = []

    def publish(config, inventory):
        written.append(inventory)
        entered.set()
        assert release.wait(3)

    monitor = module.CameraMonitor({}, threading.Event(), publish)
    old = {'recording_id': 'slice', 'updated_at': 'old'}
    newer = {'recording_id': 'slice', 'updated_at': 'new'}
    monitor.dirty['slice'] = old
    monitor.discovery['slice'] = {'observed_at': 1}
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(monitor.flush)
        try:
            assert entered.wait(2)
            assert monitor.lock.acquire(timeout=.2), 'NAS scanners were blocked by publication'
            try:
                monitor.dirty['slice'] = newer
                monitor.discovery['slice'] = {'observed_at': 2}
                monitor.dirty['second'] = {'recording_id': 'second'}
            finally:
                monitor.lock.release()
            # Another producer need not wait for the current disk write.
            workers.submit(monitor.flush).result(timeout=.5)
        finally:
            release.set()
        first.result(timeout=2)
    assert monitor.dirty['slice'] is newer
    assert monitor.discovery['slice']['observed_at'] == 2
    monitor.flush()
    assert len(written) == 2
    assert written[0]['recordings'] == [old]
    assert newer in written[1]['recordings']
    assert monitor.dirty == monitor.discovery == {}


def test_repeated_publication_failure_keeps_lane_alive_and_retries(monkeypatch):
    stop = threading.Event()
    broken = [True]
    written = []
    record = {'recording_id': 'slice', 'camera_key': 'camera'}

    def scan(config, on_record):
        on_record(record)
        return {'errors': []}

    def publish(config, inventory):
        if broken[0]:
            raise BlockingIOError('temporary contention')
        written.extend(inventory['recordings'])

    monkeypatch.setattr(module, 'scan_recordings', scan)
    monitor = module.CameraMonitor({'collection_ingest': {}, 'device_day': {}}, stop, publish)
    thread = threading.Thread(target=monitor._run_lane, args=('camera', 'live'))
    thread.start()
    try:
        wait_until(lambda: monitor.states.get(('camera', 'live'), {}).get('status') == 'retrying')
        assert thread.is_alive()
        assert monitor.dirty['slice'] == record
        with pytest.raises(BlockingIOError):
            monitor.flush()
        assert thread.is_alive()
        broken[0] = False
        monitor.flush()
        assert written == [record]
        assert not monitor.dirty
    finally:
        stop.set()
        thread.join(2)


def test_dead_lane_restarts_without_restarting_healthy_lane(tmp_path, monkeypatch):
    stop = threading.Event()
    cameras = [tmp_path / 'camera']
    monkeypatch.setattr(module, '_root', lambda config: tmp_path)
    monkeypatch.setattr(module, '_camera_directories', lambda *args: list(cameras))
    monkeypatch.setattr(module, '_recording_batches', lambda *args: [])
    attempts = {}

    def scan(config, on_record):
        key = threading.current_thread().name
        attempts[key] = attempts.get(key, 0) + 1
        if key == 'nas-live-camera' and attempts[key] == 1:
            raise RuntimeError('unexpected scanner failure')
        on_record({'recording_id': key, 'camera_key': key})
        return {'errors': []}

    monkeypatch.setattr(module, 'scan_recordings', scan)
    observed = []
    monitor = module.CameraMonitor({'collection_ingest': {}, 'device_day': {}}, stop,
                                   lambda config, inventory: observed.extend(inventory['recordings']))
    try:
        monitor.poll()
        dead = monitor.threads['camera', 'live']
        dead.join(2)
        assert not dead.is_alive()
        healthy = monitor.threads['camera', 'history']
        cameras.append(tmp_path / 'another')
        monitor.poll()
        recovered = monitor.threads['camera', 'live']
        wait_until(lambda: monitor.states['camera', 'live']['status'] == 'watching')
        assert recovered is not dead and recovered.is_alive()
        assert monitor.threads['camera', 'history'] is healthy
        assert len(monitor.threads) == 4
        snapshot = monitor.poll()
        live = next(s for s in snapshot['discovery_lanes'] if s['camera_key'] == 'camera' and s['mode'] == 'live')
        assert live['thread_alive'] and live['restart_count'] == 1
        assert any(r['recording_id'] == 'nas-live-camera' for r in observed)
        stop.set()
        for thread in monitor.threads.values():
            thread.join(2)
        monitor.poll()
        assert all(not thread.is_alive() for thread in monitor.threads.values())
    finally:
        stop.set()
        for thread in monitor.threads.values():
            thread.join(2)
