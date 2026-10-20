"""Replay bounded standard publication journals without starting model work."""
from contextlib import ExitStack
from pathlib import Path
import time

from .device_day_contract import atomic_json, digest, read_json


class PublicationReplay:
    def __init__(self, *, field_fetch=None, stop=None):
        self.field_fetch = field_fetch
        self.stop = stop

    def tick(self, runner):
        from .device_day import exclusive
        from .publication_journal import PublicationJournal, reconcile
        from .device_day_schedule import processing_cutoff
        with ExitStack() as locks:
            try:
                locks.enter_context(exclusive(runner.runtime_root / 'locks' / 'publication-replay.lock'))
            except BlockingIOError:
                return {'status': 'running_elsewhere', 'published': 0}
            # The standard replay owns per-record stage locks, validates every
            # receipt/artifact, prefers recent records and bounds each round.
            published = reconcile(runner, stop=self.stop) if self.stop is not None else reconcile(runner)
            pending = PublicationJournal(runner.runtime_root).pending(1, since_us=processing_cutoff(runner.settings))
            if self.stop is not None and self.stop.is_set():
                return {'status': 'stopping', 'published': published, 'pending_publications': bool(pending)}
            field_settings = runner.settings.get('field_recognition') or {}
            field_result = {'status': 'no_input', 'consumer_wired': False, 'model_invoked': False}
            if field_settings.get('enabled', False):
                from .application.field_event_consumer import FieldEventConsumer
                options = {'fetch_json': self.field_fetch} if self.field_fetch is not None else {}
                try:
                    consumer = FieldEventConsumer(runner.runtime_root, field_settings,
                                                  timezone=runner.settings.get('timezone', 'Asia/Shanghai'), **options)
                    field_result = consumer.tick()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    field_result = {'status': 'blocked', 'error_type': type(exc).__name__,
                                    'reason': getattr(exc, 'reason', 'field_event_consumer_unavailable'),
                                    'consumer_wired': True, 'model_invoked': False, 'business_accepted': False}
            return {'status': 'completed', 'published': published, 'pending_publications': bool(pending),
                    'field_events': field_result}


class ReadableReplay:
    """Publish through the owning file-index service without private glue scripts."""
    def __init__(self, stop):
        self.stop, self.checked, self.publisher, self.generation = stop, {}, None, None

    def tick(self, runner):
        if self.stop.is_set():
            return {'status': 'stopping'}
        from .device_day import exclusive
        from .device_day_contract import archive_name, read_json
        from .device_day_file_index import FileIndexPublisher
        from .device_day_recovery_worker import verify_storage
        from .device_day_write_guard import storage_guard
        from .device_day_io import slot
        from .observed_inventory import read_inventory
        from .runtime_control import execution_context, resource_slot
        key = digest(runner.config)
        if self.publisher is None or self.generation != key:
            self.publisher, self.generation = FileIndexPublisher(runner.config), key
            state = runner.runtime_root / 'ReadablePublication' / 'Service.json'
            if state.is_file():
                self.checked = read_json(state).get('checked', {})
        with exclusive(runner.runtime_root / 'locks' / 'readable-file-publication-consumer.lock'):
            verify_storage(runner.config)
            with storage_guard(runner.config), execution_context(job_id='readable-publication',
                    source='retention_recovery', priority=20, background_fair=True,
                    stop=self.stop, yield_signal=self.stop):
                records = read_inventory(runner.runtime_root)['recordings']
                names = {archive_name(r['camera_key'], r['recording_start_us']) for r in records}
                from .application.field_archive_context import archive_days
                field_discovery = None
                try:
                    names.update(archive_days(runner.config))
                except (OSError, ValueError) as exc:
                    field_discovery = {'status': 'unavailable', 'error_type': type(exc).__name__,
                        'automatic_action': 'retry_field_archive_discovery', 'other_publication_continues': True}
                self.checked = {name: stamp for name, stamp in self.checked.items() if name in names}
                if not names or self.stop.is_set():
                    return {'status': 'waiting_for_inventory' if not names else 'stopping'} | (
                        {'field_archive_discovery': field_discovery} if field_discovery else {})
                name = min(names, key=lambda n: (self.checked.get(n, 0),
                           -int(n[:10].replace('-', '')), n))
                self.checked[name] = time.time()
                with resource_slot(runner.config, 'cpu'), slot(runner.config):
                    result = self.publisher.tick(archives=[name])
                return {'status': 'published', 'archive': name, 'result': result} | (
                    {'field_archive_discovery': field_discovery} if field_discovery else {})


def serve(config_path, stop, *, readable=False):
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day import DeviceDayRunner
    replay = ReadableReplay(stop) if readable else PublicationReplay(stop=stop)
    directory = 'ReadablePublication' if readable else 'PublicationReplay'
    scope = 'readable_file_projection' if readable else 'standard_publication_journal'
    from .build_identity import identity
    build = identity()
    runner, generation, roots, state_path = None, None, None, None
    while not stop.is_set():
        started = time.time()
        try:
            config = apply_active(load_config(Path(config_path)))
            current_roots = {key: config['storage'].get(key) for key in
                             ('local_runtime_root', 'local_cache_root', 'archive_root')}
            if roots is not None and roots != current_roots:
                raise ValueError('Publication replay storage roots changed')
            roots = current_roots
            key = digest(config)
            if runner is None or key != generation:
                runner, generation = DeviceDayRunner(config), key
            state_path = runner.runtime_root / directory / 'Service.json'
            initial = {'status': 'running', 'started_at': started,
                       'scope': scope, 'model_invoked': False, 'build': build}
            if readable:
                initial['checked'] = replay.checked or (read_json(state_path).get('checked', {}) if state_path.is_file() else {})
            atomic_json(state_path, initial)
            result = replay.tick(runner)
            status = {'status': 'watching', 'started_at': started, 'updated_at': time.time(),
                      'last_result': result, 'scope': scope, 'model_invoked': False, 'build': build}
        except Exception as exc:
            status = {'status': 'failed', 'started_at': started, 'updated_at': time.time(),
                      'error_type': type(exc).__name__, 'scope': scope, 'model_invoked': False, 'build': build}
        if state_path is not None:
            if readable:
                status['checked'] = replay.checked
            atomic_json(state_path, status)
        stop.wait(5)
    if state_path is not None:
        final = {'status': 'stopped', 'updated_at': time.time(),
                 'scope': scope, 'model_invoked': False, 'build': build}
        if readable:
            final['checked'] = replay.checked
        atomic_json(state_path, final)


def main(argv=None):
    import argparse
    import signal
    import threading
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--readable', action='store_true', help='Publish existing readable file views without models')
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        serve(args.config, stop, readable=args.readable)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
