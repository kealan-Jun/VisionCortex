"""Independent today/yesterday metadata discovery; no model or dispatcher startup."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

from .device_day_contract import atomic_json, digest
from .capture_layout import discovery_mode, metadata_directories


def publish_status(root, status):
    """The independent monitor owns the scheduling snapshot it actually saw."""
    atomic_json(root / 'LiveDiscovery' / 'Service.json', status)
    observed = datetime.fromtimestamp(status['updated_at'], timezone.utc).isoformat()
    atomic_json(root.parent / 'state' / 'nas-recording-monitor.json', status | {
        'monitor_source': 'independent_live_discovery',
        'monitor': {'status': status['status'], 'poll_seconds': status.get('poll_seconds', 5),
                    'observed_at': observed}})


def discovery_config(config, *, metadata_cameras=()):
    result = deepcopy(config)
    ingest = result['collection_ingest']
    roles = {name for name, role in ingest.get('camera_role_map', {}).items()
             if role in {'first_person', 'third_person'}}
    roots = set(roles)
    only = discovery_mode(ingest) == 'metadata_only'
    if only:
        roots.update(metadata_cameras)
    else:
        for rule in ingest.get('directory_camera_bindings') or []:
            if rule['camera_key'] in roles:
                roots.discard(rule['camera_key'])
                roots.add(rule['directory_glob'].split('/')[0])
    if (not roots and not only) or any(Path(name).name != name or name in {'.', '..'} for name in roots):
        raise ValueError('Live discovery requires configured camera directories')
    ingest.update(camera_directories=sorted(roots), additional_camera_directories=[], poll_seconds=5)
    result.setdefault('device_day', {})['enabled'] = True
    return result


def live_config(config):
    """Discover producer-created device/days independently of view registration.

    The bounded metadata probe does not infer roles or visit legacy inputs.
    Existing configured lanes continue waiting for their producer directories.
    """
    cameras = set()
    if discovery_mode(config['collection_ingest']) == 'metadata_only':
        probe = deepcopy(config)
        today = datetime.now(ZoneInfo('Asia/Shanghai')).date()
        probe['collection_ingest'].update(
            camera_directories=[], capture_date=None,
            capture_since_date=(today - timedelta(days=1)).isoformat(),
            metadata_capture_dates=[today.isoformat(), (today - timedelta(days=1)).isoformat()])
        cameras.update(raw.parent.name[11:] for raw in metadata_directories(probe))
    return discovery_config(config, metadata_cameras=cameras)


def discovery_status(settings, snapshot, *, recordings=()):
    roles = settings['collection_ingest'].get('camera_role_map') or {}
    resolved = {record['camera_key'] for record in recordings
                if record.get('configured_role') in {'first_person', 'third_person'}}
    unresolved = {record['camera_key'] for record in recordings
                  if record.get('configured_role') not in {'first_person', 'third_person'}}
    unconfigured = sorted(name for name in snapshot['camera_directories']
                          if name in unresolved or (
                              roles.get(name) not in {'first_person', 'third_person'} and name not in resolved)) if (
                              discovery_mode(settings['collection_ingest']) == 'metadata_only') else []
    degraded = bool(unconfigured) or any(
        s['status'] in {'failed', 'retrying', 'slow_or_unavailable'} or not s['thread_alive']
        for s in snapshot['discovery_lanes'])
    return {**snapshot, 'status': 'degraded' if degraded else 'watching',
            'updated_at': time.time(), 'poll_seconds': 5,
            'scope': 'current_and_previous_beijing_date_metadata_only',
            'unconfigured_camera_directories': unconfigured,
            'alerts': [{'type': 'camera_role_unconfigured', 'camera_key': name,
                        'message': '已发现采集目录，等待配置第一或第三人称视角'}
                       for name in unconfigured]}


def serve(config_path, stop):
    from .config import load_config
    from .device_day import exclusive
    from .device_day_monitor import CameraMonitor
    from .device_day_service import DeviceDayService
    from .device_day_recovery_worker import verify_storage
    def load_settings(*, expected_storage=None, expected_source=None):
        config = load_config(Path(config_path))
        if expected_storage is not None and (
                {k: config['storage'].get(k) for k in expected_storage} != expected_storage
                or config['collection_ingest']['source_root'] != expected_source):
            raise ValueError('Live discovery storage roots changed')
        verify_storage(config)
        return live_config(config)
    settings = load_settings()
    root = Path(settings['storage']['local_runtime_root']) / 'device-day'
    fixed_roots = {k: settings['storage'].get(k) for k in ('local_runtime_root', 'local_cache_root', 'archive_root')}
    fixed_source = settings['collection_ingest']['source_root']
    service = DeviceDayService(lambda: settings, threading.Lock())
    # Only its standard observation writer is used; start/dispatch is never called.
    lane_stop = threading.Event()
    monitor = CameraMonitor(settings, lane_stop, service.observe, live_only=True)
    generation = digest(settings)
    def drain():
        lane_stop.set()
        for thread in monitor.threads.values():
            thread.join()
        monitor.flush()
    with exclusive(root / 'locks' / 'live-discovery.lock'):
        try:
            while not stop.is_set():
                try:
                    current = load_settings(expected_storage=fixed_roots, expected_source=fixed_source)
                    if digest(current) != generation:
                        drain()
                        settings = current
                        generation = digest(settings)
                        lane_stop = threading.Event()
                        monitor = CameraMonitor(settings, lane_stop, service.observe, live_only=True)
                    snapshot = monitor.poll(include_recordings=False)
                    with monitor.lock:
                        recordings = list(monitor.records.values())
                    status = discovery_status(settings, snapshot, recordings=recordings)
                except Exception as exc:
                    status = {'status': 'retrying', 'updated_at': time.time(), 'error_type': type(exc).__name__,
                              'scope': 'current_and_previous_beijing_date_metadata_only'}
                publish_status(root, status)
                stop.wait(5)
        finally:
            drain()
            publish_status(root, {'status': 'stopped', 'updated_at': time.time(),
                                 'scope': 'current_and_previous_beijing_date_metadata_only'})


def main(argv=None):
    import argparse
    import signal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        serve(args.config, stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
