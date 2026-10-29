"""Replay bounded standard publication journals without starting model work."""
from contextlib import ExitStack
from pathlib import Path
from threading import Event
import time

from .device_day_contract import atomic_json, digest, read_json

# Publication-only recipe: completed captures must be revisited after a
# naming migration without invalidating model receipts or issuing paid work.
READABLE_LAYOUT_POLICY = 'source-recording-full-extent-names/8'


class FrozenReceiptKeyOwner:
    """Ask the actual immutable producer for its keys; never alias a new hash.

    Rendering changes may use another build. Its receipt keys still belong to
    the frozen model producer, whose source/configuration is explicitly pinned.
    No execution endpoint is exposed by this transport.
    """
    def __init__(self, setting):
        import hashlib
        import os
        import subprocess
        import sys
        from threading import Lock
        from .local_storage import require_local_path

        if setting.get('schema_version') != 'visioncortex-frozen-receipt-key-owner/1':
            raise ValueError('Unknown frozen receipt key owner contract')
        source, config = Path(setting['source_root']), Path(setting['config_path'])
        require_local_path(source)
        require_local_path(config)
        if (not config.is_file() or config.is_symlink() or config.stat().st_size > 8*1024**2
                or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != setting['source_sha']
                or subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()
                or hashlib.sha256(config.read_bytes()).hexdigest() != setting['config_sha256']):
            raise ValueError('Frozen receipt owner source/configuration changed')
        helper = Path(__file__).parents[2]/'tools/migrations/read_frozen_receipt_keys.py'
        if not helper.is_file() or helper.is_symlink():
            raise ValueError('Frozen receipt key transport is unavailable')
        env = dict(os.environ, PYTHONPATH=str(source/'src'), PYTHONDONTWRITEBYTECODE='1')
        # The old producer is a reader here, not a result of the rendering build.
        env.pop('VISIONCORTEX_BUILD_MANIFEST', None)
        self._lock = Lock()
        self.process = subprocess.Popen([sys.executable, str(helper), str(config)], cwd=source,
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def key(self, stage, recording, inputs, *, binding=True, context=None):
        import json
        import re
        import select
        encoded = (json.dumps({'stage': stage, 'record': recording, 'inputs': inputs,
                               'binding': binding, 'context': context}, separators=(',', ':'))+'\n').encode()
        if len(encoded) > 64*1024**2:
            raise ValueError('Frozen receipt key request exceeds its bound')
        with self._lock:
            if self.process.poll() is not None:
                raise ValueError('Frozen receipt key owner has stopped')
            self.process.stdin.write(encoded)
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], 60)[0]:
                self.close()
                raise TimeoutError('Frozen receipt key owner exceeded its deadline')
            line = self.process.stdout.readline(4097)
            value = json.loads(line)
            if set(value) != {'key'} or not re.fullmatch('[a-f0-9]{64}', value['key']):
                raise ValueError('Frozen producer rejected receipt key verification')
            return value['key']

    def close(self):
        import subprocess
        try:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=5)
        finally:
            self.process.stdin.close()
            self.process.stdout.close()


def publication_runner(config):
    """Compose a read-only producer-key port with the current publication owner."""
    from .device_day import DeviceDayRunner
    setting = config.get('device_day', {}).get('publication_receipt_key_owner')
    if not setting:
        return DeviceDayRunner(config)

    class Reader(DeviceDayRunner):
        def __init__(self):
            super().__init__(config, backend=object())
            self.receipt_key_owner = FrozenReceiptKeyOwner(setting)
            vision = self.settings.get('publication_vision_key_owner')
            self.vision_key_owner = FrozenReceiptKeyOwner(vision) if vision else None

        def _key(self, stage, recording, inputs, *, _binding=True):
            if _binding and self._completed_execution_bindings.historical_candidate(stage, recording):
                # Historical read-only descriptors bind this rendering build
                # through its exact approved reader profile as before.
                return super()._key(stage, recording, inputs, _binding=_binding)
            context = self._context(self.layout(recording), recording) if stage == 'report' and self.settings.get(
                'derived_video_retention', False) else None
            if stage == 'vision' and self.vision_key_owner is not None:
                current = self.vision_key_owner.key(stage, recording, inputs, binding=_binding)
                path = self._receipt(self.layout(recording), recording, 'vision')
                # The stage still belongs to an actual pinned producer. An
                # observed key only chooses which producer to ask; _load must
                # independently authenticate the complete receipt and bytes.
                if path.is_file():
                    import ijson
                    with path.open('rb') as handle:
                        actual = next(ijson.items(handle, 'key'), None)
                    if actual == current:
                        return current
            return self.receipt_key_owner.key(stage, recording, inputs, binding=_binding, context=context)

        def _backend(self):
            raise RuntimeError('Publication receipt readers cannot invoke models')

        def process(self, *args, **kwargs):
            raise RuntimeError('Publication receipt readers cannot execute stages')

        def close(self):
            self.receipt_key_owner.close()
            if self.vision_key_owner is not None:
                self.vision_key_owner.close()

    return Reader()


def local_change_hints(runtime_root, records, names):
    """Durable scheduling hints; consumers must still verify source evidence."""
    from .device_day_contract import archive_name, digest
    from .sqlite_store import connection
    grouped = {name: [] for name in names}
    owners = {r['recording_id']: archive_name(r['camera_key'], r['recording_start_us']) for r in records}
    path = runtime_root / 'timeline-invalidations.sqlite3'
    if path.is_file():
        with connection(path, readonly=True) as db:
            for row in db.execute('SELECT archive,digest FROM inputs ORDER BY archive'):
                if row['archive'] in grouped:
                    grouped[row['archive']].append(['canonical', row['digest']])
    for stage in ('retention', 'vision', 'stt', 'understanding', 'report'):
        path = runtime_root / f'queue-{stage}.sqlite3'
        if not path.is_file():
            continue
        with connection(path, readonly=True) as db:
            for row in db.execute("SELECT recording_id,revision,updated_at,status FROM recordings "
                    "WHERE status='completed' OR json_extract(result,'$.status') IN "
                    "('partial','paused_for_live','scheduled_checkpoint') ORDER BY recording_id"):
                owner = owners.get(row['recording_id'])
                if owner in grouped:
                    grouped[owner].append([stage, row['recording_id'], row['revision'], row['updated_at'], row['status']])
    return {name: digest(values) for name, values in grouped.items() if values}


class PublicationReplay:
    def __init__(self, *, field_fetch=None, stop=None):
        self.field_fetch = field_fetch
        self.stop = stop if stop is not None else Event()

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
            from .runtime_control import execution_context
            # Source verification and projection compose nested NAS readers.
            # Give them one resumable job scope, as the readable publisher does.
            with execution_context(job_id='publication-replay', source='retention_recovery',
                    priority=20, background_fair=True, stop=self.stop, yield_signal=self.stop):
                published = reconcile(runner, stop=self.stop, limit=4)
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
        self.published_hints, self.retry_at, self.turn = {}, {}, 0

    @staticmethod
    def _local_hints(runner, records, names):
        """Scheduling hints only; the owning publisher still verifies facts."""
        from .media.retained_intervals import EXPORT_POLICY, POLICY
        from .application.clip_delivery import VERSION, enabled
        hints = local_change_hints(runner.runtime_root, records, names)
        # A policy upgrade must revisit completed days even when producers
        # have no new model result. Persist this selective publication version.
        source_groups = enabled(getattr(runner, 'config', {}))
        return {name: digest(['readable-retention-policy/2', POLICY, EXPORT_POLICY,
                             VERSION if source_groups else None,
                             READABLE_LAYOUT_POLICY if source_groups else None, value])
                for name, value in hints.items()}

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
            if self.publisher is not None:
                self.publisher.close()
            self.publisher, self.generation = FileIndexPublisher(runner.config), key
            state = runner.runtime_root / 'ReadablePublication' / 'Service.json'
            if state.is_file():
                saved = read_json(state)
                self.checked = saved.get('checked', {})
                self.published_hints = saved.get('published_hints', {})
                self.retry_at = saved.get('retry_at', {})
                self.turn = saved.get('fair_turn', 0)
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
                self.published_hints = {n: h for n, h in self.published_hints.items() if n in names}
                self.retry_at = {n: stamp for n, stamp in self.retry_at.items() if n in names}
                if not names or self.stop.is_set():
                    return {'status': 'waiting_for_inventory' if not names else 'stopping'} | (
                        {'field_archive_discovery': field_discovery} if field_discovery else {})
                import sqlite3
                hint_error = None
                try:
                    hints = self._local_hints(runner, records, names)
                except (OSError, ValueError, sqlite3.Error) as exc:
                    hints = {}
                    hint_error = {'status': 'unavailable', 'error_type': type(exc).__name__}
                now = time.time()
                eligible = {n for n in names if self.retry_at.get(n, 0) <= now}
                if not eligible:
                    return {'status': 'publication_backoff', 'retry_at': min(self.retry_at.values())}
                changed = {n for n in eligible if n in hints and hints[n] != self.published_hints.get(n)}
                cutoff = (now - runner.config.get('device_day', {}).get('live_priority_seconds', 14400)) * 1e6
                recent = {archive_name(r['camera_key'], r['recording_start_us']) for r in records
                          if max(r.get('recording_start_us', 0), r.get('recording_end_us', 0)) >= cutoff}
                self.turn += 1
                priority = (changed & recent) or changed or (eligible & recent)
                lane = 'changed_or_live' if priority and self.turn % 3 else 'historical_fair_share'
                pool = priority if lane == 'changed_or_live' else eligible
                if (runner.config.get('device_day', {}).get('latest_first', False) and changed
                        and lane == 'changed_or_live'):
                    latest = {n: 0 for n in names}
                    for record in records:
                        key = archive_name(record['camera_key'], record['recording_start_us'])
                        latest[key] = max(latest.get(key, 0), record['recording_start_us'])
                    pool, lane = changed, 'latest_changed_input'
                    name = min(pool, key=lambda n: (-latest.get(n, 0), self.checked.get(n, 0), n))
                else:
                    name = min(pool, key=lambda n: (self.checked.get(n, 0), -int(n[:10].replace('-', '')), n))
                self.checked[name] = time.time()
                try:
                    with resource_slot(runner.config, 'cpu'), slot(runner.config):
                        result = self.publisher.tick(archives=[name])
                except Exception:
                    self.retry_at[name] = time.time() + 30
                    raise
                if (result.get('status') == 'completed' and not result.get('busy') and not result.get('errors')
                        and (result.get('published', 0) or result.get('unchanged', 0))):
                    if name in hints:
                        self.published_hints[name] = hints[name]
                    self.retry_at.pop(name, None)
                else:
                    self.retry_at[name] = time.time() + 30
                return {'status': 'published', 'archive': name, 'result': result, 'scheduling_lane': lane} | (
                    {'field_archive_discovery': field_discovery} if field_discovery else {}) | (
                    {'local_change_hints': hint_error} if hint_error else {})


def serve(config_path, stop, *, readable=False):
    from .ai_settings import apply_active
    from .config import load_config
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
                if runner is not None and hasattr(runner, 'close'):
                    runner.close()
                runner, generation = publication_runner(config), key
            state_path = runner.runtime_root / directory / 'Service.json'
            initial = {'status': 'running', 'started_at': started,
                       'scope': scope, 'model_invoked': False, 'build': build}
            if readable:
                saved = read_json(state_path) if state_path.is_file() else {}
                initial.update(checked=replay.checked or saved.get('checked', {}),
                    published_hints=replay.published_hints or saved.get('published_hints', {}),
                    retry_at=replay.retry_at or saved.get('retry_at', {}),
                    fair_turn=replay.turn or saved.get('fair_turn', 0))
            atomic_json(state_path, initial)
            result = replay.tick(runner)
            status = {'status': 'watching', 'started_at': started, 'updated_at': time.time(),
                      'last_result': result, 'scope': scope, 'model_invoked': False, 'build': build}
        except Exception as exc:
            status = {'status': 'failed', 'started_at': started, 'updated_at': time.time(),
                      'error_type': type(exc).__name__, 'scope': scope, 'model_invoked': False, 'build': build}
        if state_path is not None:
            if readable:
                status.update(checked=replay.checked, published_hints=replay.published_hints,
                              retry_at=replay.retry_at, fair_turn=replay.turn)
            atomic_json(state_path, status)
        stop.wait(5)
    if state_path is not None:
        final = {'status': 'stopped', 'updated_at': time.time(),
                 'scope': scope, 'model_invoked': False, 'build': build}
        if readable:
            final.update(checked=replay.checked, published_hints=replay.published_hints,
                         retry_at=replay.retry_at, fair_turn=replay.turn)
        atomic_json(state_path, final)
    if runner is not None and hasattr(runner, 'close'):
        runner.close()
    if readable and replay.publisher is not None:
        replay.publisher.close()


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
