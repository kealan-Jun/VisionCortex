"""Publish current validated understanding windows and partial day reports."""
from contextlib import ExitStack
import json
from pathlib import Path
import time

from .device_day_contract import archive_name, atomic_json, digest
from .device_day_file_index import FileIndexPublisher
from .device_day_schedule import in_processing_scope
from .sqlite_store import connection


class LiveFileIndex:
    def __init__(self, config):
        self.config = config
        self.publisher = FileIndexPublisher(config)
        self.attempted = {}

    def tick(self, *, limit=2, now=None):
        from .device_day import exclusive
        current = time.time() if now is None else now
        runtime = self.publisher.runtime
        with ExitStack() as locks:
            try:
                locks.enter_context(exclusive(runtime/'locks'/'live-file-index.lock'))
            except BlockingIOError:
                return {'status': 'running_elsewhere'}
            path = runtime/'observed-inventory.sqlite3'
            if not path.is_file():
                return {'status': 'waiting_for_inventory', 'archives': []}
            with connection(path, readonly=True) as db:
                rows = db.execute("SELECT payload FROM observations WHERE "
                    "MAX(COALESCE(json_extract(payload,'$.recording_start_us'),0),"
                    "COALESCE(json_extract(payload,'$.recording_end_us'),0))>=? "
                    "ORDER BY json_extract(payload,'$.recording_start_us') DESC", ((current-14400)*1e6,)).fetchall()
            latest = {}
            for row in rows:
                record = json.loads(row['payload'])
                if (record.get('configured_role') not in {'first_person', 'third_person'}
                        or not in_processing_scope(self.config.get('device_day', {}), record)):
                    continue
                name = archive_name(record['camera_key'], record['recording_start_us'])
                latest[name] = max(latest.get(name, 0), record['recording_start_us'])
            # Newest unseen generations lead; bounded rounds then rotate other
            # active cameras, so one busy archive cannot starve all the others.
            names = sorted(latest, key=lambda n: (self.attempted.get((n, latest[n]), 0), -latest[n]))[:limit]
            result = self.publisher.tick(archives=names)
            for name in names:
                self.attempted[name, latest[name]] = current
            self.attempted = {k: v for k, v in self.attempted.items() if latest.get(k[0]) == k[1]}
            return result | {'archives': names, 'scope': 'recent_readable_projections', 'model_invoked': False}


def serve(config_path, stop):
    from .config import load_config
    worker, generation, roots, state_path = None, None, None, None
    while not stop.is_set():
        started = time.time()
        try:
            config = load_config(Path(config_path))
            current_roots = {k: config['storage'].get(k) for k in ('archive_root', 'local_cache_root', 'local_runtime_root')}
            if roots is not None and roots != current_roots:
                raise ValueError('Live file publisher storage roots changed')
            roots = current_roots
            key = digest(config)
            if worker is None or generation != key:
                worker, generation = LiveFileIndex(config), key
            state_path = worker.publisher.runtime/'LiveFileIndex'/'Service.json'
            atomic_json(state_path, {'status': 'running', 'started_at': started, 'model_invoked': False})
            status = {'status': 'watching', 'started_at': started, 'updated_at': time.time(),
                      'last_result': worker.tick(), 'model_invoked': False}
            status['updated_at'] = time.time()
        except Exception as exc:
            status = {'status': 'failed', 'started_at': started, 'updated_at': time.time(),
                      'error_type': type(exc).__name__, 'model_invoked': False}
        if state_path is not None:
            atomic_json(state_path, status)
        stop.wait(5)
    if state_path is not None:
        atomic_json(state_path, {'status': 'stopped', 'updated_at': time.time(), 'model_invoked': False})


def main(argv=None):
    import argparse
    import signal
    import threading
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
