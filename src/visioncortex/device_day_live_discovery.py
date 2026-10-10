"""Independent today/yesterday metadata discovery; no model or dispatcher startup."""
from copy import deepcopy
from pathlib import Path
import threading
import time

from .device_day_contract import atomic_json, digest


def discovery_config(config):
    result = deepcopy(config)
    ingest = result['collection_ingest']
    roles = {name for name, role in ingest.get('camera_role_map', {}).items()
             if role in {'first_person', 'third_person'}}
    roots = set(roles)
    for rule in ingest.get('directory_camera_bindings') or []:
        if rule['camera_key'] in roles:
            roots.discard(rule['camera_key'])
            roots.add(rule['directory_glob'].split('/')[0])
    if not roots or any(Path(name).name != name or name in {'.', '..'} for name in roots):
        raise ValueError('Live discovery requires configured camera directories')
    ingest.update(camera_directories=sorted(roots), additional_camera_directories=[], poll_seconds=5)
    result.setdefault('device_day', {})['enabled'] = True
    return result


def serve(config_path, stop):
    from .config import load_config
    from .device_day import exclusive
    from .device_day_monitor import CameraMonitor
    from .device_day_service import DeviceDayService
    settings = discovery_config(load_config(Path(config_path)))
    root = Path(settings['storage']['local_runtime_root']) / 'device-day'
    fixed_roots = {k: settings['storage'].get(k) for k in ('local_runtime_root', 'local_cache_root', 'archive_root')}
    fixed_source = settings['collection_ingest']['source_root']
    state_path = root / 'LiveDiscovery' / 'Service.json'
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
                    current = discovery_config(load_config(Path(config_path)))
                    if ({k: current['storage'].get(k) for k in fixed_roots} != fixed_roots
                            or current['collection_ingest']['source_root'] != fixed_source):
                        raise ValueError('Live discovery storage roots changed')
                    if digest(current) != generation:
                        drain()
                        settings = current
                        generation = digest(settings)
                        lane_stop = threading.Event()
                        monitor = CameraMonitor(settings, lane_stop, service.observe, live_only=True)
                    snapshot = monitor.poll(include_recordings=False)
                    degraded = any(s['status'] in {'failed', 'retrying', 'slow_or_unavailable'}
                                   or not s['thread_alive'] for s in snapshot['discovery_lanes'])
                    status = {'status': 'degraded' if degraded else 'watching', 'updated_at': time.time(), 'poll_seconds': 5,
                              'scope': 'current_and_previous_beijing_date_metadata_only', **snapshot}
                except Exception as exc:
                    status = {'status': 'retrying', 'updated_at': time.time(), 'error_type': type(exc).__name__,
                              'scope': 'current_and_previous_beijing_date_metadata_only'}
                atomic_json(state_path, status)
                stop.wait(5)
        finally:
            drain()
            atomic_json(state_path, {'status': 'stopped', 'updated_at': time.time(),
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
