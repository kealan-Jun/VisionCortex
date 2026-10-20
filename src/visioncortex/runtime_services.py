"""Independent operational workers; model-free progress and input integrity."""

from contextlib import contextmanager
import logging
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time

from .device_day_contract import STAGES, atomic_json, digest
from .device_day_retry import CHECKS, provider_binding_proof
from .device_day_retry_worker import CACHE_CHECK, RECHECK_VERSION, SOURCE_CHECK, VERSION, _parents
from .sqlite_store import connection


def _publish_progress(config, control, interval):
    """Only local read models; never share the inference process's Python locks."""
    from .device_day_progress import snapshot
    path = Path(config['storage']['local_runtime_root']) / 'device-day' / 'ProgressSnapshot.json'
    try:
        while not control.poll():
            try:
                started = time.monotonic()
                value = snapshot(config)
                value.update(snapshot_producer_pid=os.getpid(),
                             snapshot_collection_seconds=time.monotonic()-started)
                atomic_json(path, value)
            except Exception as exc:
                logging.getLogger(__name__).warning('Progress publication unavailable: %s', type(exc).__name__)
            if control.poll(interval):
                break
    finally:
        control.close()


class ProgressPublisher:
    """Supervised spawn process: no fork of a loaded CUDA runtime."""
    def __init__(self, config, *, interval=5):
        self.config = config
        self.interval = interval
        self.context = multiprocessing.get_context('spawn')
        # A killed child can strand a multiprocessing.Event's semaphore. Give
        # each generation a fresh pipe, with no shared synchronization locks.
        self._lock = threading.Lock()
        self._closed = False
        self._control = None
        self.process = None

    def ensure(self):
        with self._lock:
            if self._closed or self.process is not None and self.process.is_alive():
                return
            if self.process is not None:
                self.process.join(timeout=0)
                self.process.close()
                self.process = None
            if self._control is not None:
                self._control.close()
            reader, self._control = self.context.Pipe(duplex=False)
            process = self.context.Process(target=_publish_progress,
                args=(self.config, reader, self.interval), name='progress-publisher', daemon=True)
            try:
                process.start()
                self.process = process
            except BaseException:
                self._control.close()
                self._control = None
                process.close()
                raise
            finally:
                reader.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._control is not None:
                try:
                    self._control.send_bytes(b'\x00')
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    self._control.close()
                    self._control = None
            if self.process is None:
                return
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if not self.process.is_alive():
                self.process.close()
                self.process = None


@contextmanager
def services(config):
    stop = threading.Event()
    root = Path(config["storage"]["local_runtime_root"])

    def loop(name, interval, action):
        while not stop.is_set():
            try:
                action()
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "%s unavailable: %s", name, type(exc).__name__
                )
            stop.wait(interval)

    from .task_events import EventStore
    from .knowledge import Knowledge
    from .input_availability import Reconciler

    events = EventStore(root)
    knowledge = Knowledge(config) if config["storage"].get("archive_root") else None
    availability = Reconciler(config)

    progress = ProgressPublisher(config)

    tasks = [
        ("progress-supervisor", 5, progress.ensure),
        ("task-outbox", 2, events.harvest),
        (
            "knowledge-index",
            5,
            lambda: knowledge.tick(stop=stop) if knowledge else None,
        ),
        ("input-availability", 5, availability.tick),
    ]
    if config.get('device_day', {}).get('enabled'):
        from .device_day_time_lookup import refresh_photo_index
        from .device_day_file_index import FileIndexPublisher
        file_indexes = FileIndexPublisher(config)
        tasks.append(('file-time-index', 5, file_indexes.tick))
        for camera in config.get('collection_ingest', {}).get('camera_role_map', {}):
            tasks.append(('capture-photos-'+camera, 5,
                          lambda camera=camera: refresh_photo_index(config, camera)))
    subscriptions = config.get("runtime", {}).get("result_callbacks", [])
    if subscriptions:
        tasks.append(("result-callbacks", 5, lambda: events.dispatch(subscriptions)))
    threads = [
        threading.Thread(target=loop, args=task, name=task[0], daemon=True)
        for task in tasks
    ]
    for thread in threads:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        deadline = time.monotonic() + 5
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        progress.close()


def serve_integrity(config_path, stop):
    """Independent guarded checker; it never instantiates a model backend."""
    from .build_identity import identity
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day import DeviceDayRunner, exclusive
    from .device_day_consumers import process_start_ticks
    from .input_availability import IntegrityMonitor
    from .local_storage import require_local_path
    from .runtime_control import ExecutionCancelled, ExecutionYielded, execution_context, resource_slot
    config = apply_active(load_config(Path(config_path)))
    root = require_local_path(Path(config['storage']['local_runtime_root'])) / 'device-day'
    runner = DeviceDayRunner(config, backend=object())
    monitor = IntegrityMonitor(config, runner)
    state = {'schema_version': 'visioncortex-input-integrity-service/1', 'pid': os.getpid(),
             'process_start_ticks': process_start_ticks(os.getpid()), 'build': identity(),
             'status': 'starting', 'started_at': time.time(), 'tick_count': 0, 'completed_ticks': 0,
             'model_invoked': False, 'capture_modified': False, 'settings': monitor.options}
    lock, heartbeat_stop = threading.Lock(), threading.Event()
    path = root / 'InputIntegrity/Service.json'

    def publish(**fields):
        with lock:
            state.update(fields)
            atomic_json(path, state | {'updated_at': time.time()})

    def heartbeat():
        while not heartbeat_stop.wait(2):
            try:
                publish()
            except OSError:
                pass

    with exclusive(root / 'locks/input-integrity-service.lock'):
        thread = threading.Thread(target=heartbeat, name='integrity-heartbeat', daemon=True)
        thread.start()
        last_error = None
        try:
            while not stop.is_set():
                try:
                    publish(status=('disabled' if not monitor.options['enabled'] else 'failed' if last_error else 'running'),
                            phase='checking', tick_started_at=time.time(),
                            tick_count=state['tick_count'] + 1)
                    with execution_context(job_id='input-integrity', source='retention_recovery',
                                           priority=1, stop=stop, yield_signal=stop):
                        with resource_slot(config, 'cpu'):
                            result = monitor.tick(stop=stop)
                    if last_error:
                        logging.getLogger(__name__).info('Input integrity service recovered')
                    last_error = None
                    publish(status='running' if monitor.options['enabled'] else 'disabled',
                            phase='waiting', last_completed_at=time.time(), last_result=result,
                            completed_ticks=state['completed_ticks'] + 1)
                except ExecutionYielded:
                    publish(phase='waiting', last_result={'status': 'yielded_to_foreground'})
                except ExecutionCancelled:
                    break
                except Exception as exc:
                    if last_error != type(exc).__name__:
                        logging.getLogger(__name__).error('Input integrity service check failed: %s', type(exc).__name__)
                    last_error = type(exc).__name__
                    publish(status='failed', phase='waiting', error_type=type(exc).__name__)
                stop.wait(monitor.options['interval_seconds'])
        finally:
            heartbeat_stop.set()
            thread.join(timeout=5)
            publish(status='stopped', phase='stopped')


def serve_availability(config_path, stop, *, interval=5):
    """Own the existing bounded input reconciliation loop without model startup."""
    from .build_identity import identity
    from .config import load_config
    from .device_day import exclusive
    from .device_day_consumers import process_start_ticks
    from .device_day_io import slot
    from .device_day_queue import DeviceDayQueue
    from .device_day_recovery_worker import verify_storage
    from .device_day_write_guard import storage_guard
    from .input_availability import Availability, Reconciler
    from .local_storage import require_local_path
    from .runtime_control import ExecutionCancelled, ExecutionYielded, execution_context, resource_slot
    config = load_config(Path(config_path))
    root = require_local_path(Path(config['storage']['local_runtime_root'])) / 'device-day'
    reconciler, availability = Reconciler(config), Availability(root)
    queues = [DeviceDayQueue(root / f'queue-{stage}.sqlite3') for stage in STAGES]
    build = identity()
    state = {'schema_version': 'visioncortex-input-reconciliation-service/1', 'source_sha': build.get('commit'),
             'build': build, 'pid': os.getpid(), 'process_start_ticks': process_start_ticks(os.getpid()),
             'started_at': time.time(), 'status': 'starting', 'ticks': 0, 'tick_count': 0,
             'model_invoked': False, 'capture_modified': False}
    path = root / 'InputReconciliation/Service.json'
    lock, heartbeat_stop = threading.Lock(), threading.Event()
    def publish(**fields):
        with lock:
            state.update(fields)
            atomic_json(path, state | {'updated_at': time.time()})
    def heartbeat():
        while not heartbeat_stop.wait(2):
            try:
                publish()
            except OSError:
                pass
    with exclusive(root / 'locks/input-availability-consumer.lock'):
        thread = threading.Thread(target=heartbeat, name='input-availability-heartbeat', daemon=True)
        thread.start()
        try:
            while not stop.is_set():
                try:
                    publish(status='running', phase='checking', tick_count=state['tick_count'] + 1)
                    verify_storage(config)
                    with storage_guard(config), execution_context(job_id='input-reconciliation',
                            source='retention_recovery', priority=20, background_fair=True,
                            stop=stop, yield_signal=stop):
                        with resource_slot(config, 'cpu'), slot(config):
                            reconciler.tick()
                            states = availability.states()
                            for stage, queue in zip(STAGES, queues, strict=True):
                                queue.sync_availability(availability.states_for_stage(stage, states=states))
                    with lock:
                        state.pop('error_type', None)
                    publish(status='reconciled', phase='waiting', ticks=state['ticks'] + 1,
                            last_completed_at=time.time(),
                            ready=sum(item['state'] == 'ready' for item in states.values()),
                            missing=sum(item['state'] == 'missing' for item in states.values()))
                except ExecutionCancelled:
                    break
                except ExecutionYielded:
                    publish(status='waiting', phase='waiting', last_result={'status': 'yielded_to_foreground'})
                except Exception as exc:
                    logging.getLogger(__name__).error('Input reconciliation failed: %s', type(exc).__name__)
                    publish(status='failed', phase='waiting', error_type=type(exc).__name__)
                stop.wait(interval)
        finally:
            heartbeat_stop.set()
            thread.join(timeout=5)
            publish(status='stopped', phase='stopped')


def serve_progress(config_path, stop, *, interval=5):
    """Publish local queue observations when inference owners run separately."""
    from .config import load_config
    from .device_day import exclusive
    from .local_storage import require_local_path
    config = load_config(Path(config_path))
    root = require_local_path(Path(config['storage']['local_runtime_root'])) / 'device-day'

    class Control:
        def poll(self, timeout=None):
            return stop.is_set() if timeout is None else stop.wait(timeout)

        def close(self):
            pass

    with exclusive(root / 'locks' / 'progress-publisher.lock'):
        _publish_progress(config, Control(), interval)


class NativeInfrastructureVerifier:
    """Current native inputs and receipts, without a historical site runtime.

    Only independently probeable infrastructure faults are supported. Model,
    quality, decode and unknown-submission failures retain their own gates.
    """
    supported = frozenset({'database_lock', 'storage_access', 'storage_capacity', 'provider_binding'})

    def __init__(self, config):
        self.pending = None

    def verify_source_view(self, config, stop):
        from .device_day_recovery_worker import verify_storage
        from .device_day_write_guard import storage_guard
        from .runtime_control import check_cancelled
        check_cancelled(stop)
        verify_storage(config)
        with storage_guard(config):
            return {'status': 'PROVEN'}

    @staticmethod
    def rejection_reason(exc):
        return {'Target queue has no current source admission': 'current_source_not_admitted',
                'Current queue recipe changed; normal admission is required': 'current_recipe_requires_normal_admission',
                'Current parent execution is not verified': 'current_parent_execution_unverified',
                'Current input does not match the sealed recording': 'current_source_metadata_changed',
                'Current inventory source differs from the queue': 'current_inventory_source_changed',
                'Original content differs from its input seal': 'current_source_content_conflict',
                'Current parent artifact is not verified': 'current_parent_artifact_unverified',
                'Readmission repair class is not authorized': 'readmission_repair_class_not_authorized',
                'Readmission paid request evidence is unavailable': 'readmission_request_evidence_unavailable',
                'Readmission requires an unowned known failed result': 'readmission_result_not_safe',
                'Readmission current worker scope is unavailable': 'readmission_worker_scope_unavailable',
                'Readmission historical scope is not authorized': 'readmission_historical_scope_not_authorized',
                'Readmission current scheduling observation is unavailable': 'readmission_scheduling_unavailable',
                'Readmission original input seals are unavailable': 'input_seal_unavailable',
                'Readmission requires an actually changed execution recipe': 'execution_recipe_unchanged',
                'Readmission normal worker admission is waiting': 'readmission_normal_admission_waiting'}.get(str(exc))

    @staticmethod
    def _identity(path):
        from .device_day_inputs import identity
        return tuple(identity(path))

    @staticmethod
    def _row(runner, stage, rid):
        with connection(runner.runtime_root / f'queue-{stage}.sqlite3', readonly=True) as db:
            value = db.execute('SELECT * FROM recordings WHERE recording_id=?', (rid,)).fetchone()
            return dict(value) if value else None

    @staticmethod
    def _inventory(runner, rid):
        with connection(runner.runtime_root / 'observed-inventory.sqlite3', readonly=True) as db:
            value = db.execute('SELECT payload FROM observations WHERE id=?', (rid,)).fetchone()
            if not value:
                raise ValueError('Current inventory is not verified')
            return json.loads(value[0])

    def _repair(self, config, runner, stage, row, kind):
        from .device_day_admission import admission_status
        from .device_day_contract import file_hash
        if kind not in self.supported:
            raise ValueError('Repair class requires its own explicit execution proof')
        path = runner.runtime_root / f'queue-{stage}.sqlite3'
        # Probe an actual writer transaction and roll it back; no business
        # value or retry budget is changed by this health check.
        with connection(path) as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT * FROM recordings WHERE recording_id=?',
                                 (row['recording_id'],)).fetchone()
            if not current or dict(current) != row:
                raise ValueError('Current verification changed before grant')
            db.execute('UPDATE recordings SET retry_limit=retry_limit WHERE recording_id=?',
                       (row['recording_id'],))
            db.rollback()
        if kind == 'provider_binding':
            provider_binding_proof(config)
        if kind == 'storage_capacity' and (not runner.settings.get('local_storage_reserves')
                                           or admission_status(config, stage) is not None):
            raise ValueError('Current storage reserve is not verified')
        if kind == 'storage_access':
            import os
            import uuid
            from .device_day_contract import atomic_bytes
            from .device_day_write_guard import storage_guard
            folder = runner._receipt(runner.layout(json.loads(row['payload'])),
                                     json.loads(row['payload']), stage).parent
            probe = folder / ('.repair-probe-' + uuid.uuid4().hex)
            with storage_guard(config) as guard:
                mount, resolved = guard.mount_for(probe)
                parent = guard.directory(mount, resolved.relative_to(mount.path).parts[:-1]) if mount else None
                try:
                    atomic_bytes(probe, b'owned infrastructure probe\n')
                finally:
                    if parent is not None:
                        try:
                            os.unlink(probe.name, dir_fd=parent)
                        except FileNotFoundError:
                            pass
                        finally:
                            os.close(parent)
                    else:
                        probe.unlink(missing_ok=True)
        return {'identity': {'repair_kind': kind, 'implementation_sha256': file_hash(Path(__file__)),
                             'policy_sha256': digest({'kind': kind, 'source_limit': 8 * 1024**3,
                                                      'reserves': runner.settings.get('local_storage_reserves')}),
                             'verification_recipe_sha256': digest('native-current-source-parent-infrastructure/1')},
                'checks': {name: 'PROVEN' for name in CHECKS[kind]}, 'validated_model_repair': None}

    def verify(self, config, stage, row, kind, stop):
        from .device_day_write_guard import storage_guard
        with storage_guard(config):
            return self._verify(config, stage, row, kind, stop)

    def _verify(self, config, stage, row, kind, stop, *, readmission=False, normal_handoff=False):
        from .device_day import DeviceDayRunner, visual_input
        from .device_day_contract import dependencies_for, file_hash, read_json, safe_child
        from .device_day_inputs import binding
        from .input_availability import _IntegrityBytes, _PacedInput, configured_record
        from .nas_recordings import _inspect
        from .runtime_control import check_cancelled
        self.pending = None
        if kind not in self.supported and not normal_handoff:
            raise ValueError('Repair class requires its own explicit execution proof')
        runner = DeviceDayRunner(config, backend=object())
        record = json.loads(row['payload'])
        rid, layout = record['recording_id'], runner.layout(record)
        # A repair budget never establishes permission for historical work.
        # Reject before NAS hashing when no exact-source admission exists.
        if not readmission and (row.get('processing_cohort') not in {'live', 'current_recovery', 'historical_backfill'}
                or row.get('admission_signature') != record.get('source_signature')
                or row.get('admitted_at') is None):
            raise ValueError('Target queue has no current source admission')
        if row.get('lease_owner') or row.get('lease_until'):
            raise ValueError('Target queue still has a stage owner')
        root = Path(config['collection_ingest']['source_root'])
        video = Path(record['video_path'])
        if video.is_symlink() or not video.resolve().is_relative_to(root.resolve()):
            raise ValueError('Current capture source is not a regular owned input')
        fresh = configured_record(config, _inspect(root, video, time.time(),
            config['collection_ingest'].get('settle_seconds', 120),
            bool(config['collection_ingest'].get('discover_plain_video_csv', False))))
        fields = ('recording_id', 'source_signature', 'camera_key', 'configured_role', 'recording_start_us',
                  'recording_end_us', 'video_path', 'frames_path', 'size_bytes', 'recording_session_id')
        if (not fresh.get('processable', fresh.get('available'))
                or any(fresh.get(k) != record.get(k) for k in fields)
                or fresh.get('audio', {}) != record.get('audio', {})):
            raise ValueError('Current input does not match the sealed recording')
        inventory = self._inventory(runner, rid)
        if any(inventory.get(k) != record.get(k) for k in fields):
            raise ValueError('Current inventory source differs from the queue')
        paths = {video, Path(record['frames_path']),
                 *(Path(x['path']) for x in fresh.get('audio', {}).get('files', []))}
        prefix = video.name[:-len('rgb.mp4')] if video.name.endswith('rgb.mp4') else ''
        paths.update(p for suffix in ('meta.json', 'recording_ready.json', 'calibration.json')
                     if (p := video.with_name(prefix + suffix)).is_file())
        snapshots, hashes, total = {}, {}, 0

        def read(path):
            if path.stat().st_size > 128 * 1024**2:
                raise ValueError('Current repair receipt exceeds its declared bound')
            before = self._identity(path)
            if path in snapshots and snapshots[path] != before:
                raise ValueError('Artifact receipt changed during verification')
            value = read_json(path)
            if self._identity(path) != before:
                raise ValueError('Artifact receipt changed during verification')
            snapshots[path] = before
            return value

        for path in sorted(paths):
            check_cancelled(stop)
            if path.parent != video.parent or path.is_symlink():
                raise ValueError('Current source set escapes its recording')
            before = self._identity(path)
            total += before[2]
            if total > 8 * 1024**3:
                raise ValueError('Current repair source exceeds its declared bound')
            hashes[str(path)] = file_hash(_PacedInput(config, path))
            if self._identity(path) != before:
                raise ValueError('Original source changed during verification')
            snapshots[path] = before
        seals = {}
        for primary in ('vision', 'stt'):
            path = runner._receipt(layout, record, 'input-' + primary)
            if path.is_file():
                seals[primary] = read(path)
                for source in seals[primary]['sources']:
                    if hashes.get(source['original_path']) != source['sha256']:
                        raise ValueError('Original content differs from its input seal')
        verifier = _IntegrityBytes(config, 8 * 1024**3 - total)
        parents, parent_keys, parent_hashes = {}, {}, {}
        context = runner._context(layout, record) if stage == 'understanding' else {'comments': [], 'protocol': None}
        for parent in STAGES:
            if parent not in _parents(stage):
                continue
            saved = read(runner._receipt(layout, record, parent))
            inputs = (record if parent == 'retention' else visual_input(parents['retention']) if parent == 'vision'
                      else parents['retention'] if parent == 'stt' else
                      {'vision': parents['vision'], 'stt': parents['stt'], 'context': context})
            key = runner._key(parent, record, inputs)
            if saved.get('status') != 'completed' or not runner._accepts_receipt(saved, key):
                raise ValueError('Current parent execution is not verified')
            for base, refs in ((layout.root, saved.get('artifacts', [])),
                               (runner.backend_root, saved.get('audit_artifacts', []))):
                for ref in refs:
                    if base == runner.backend_root and ref.get('storage_root') != 'local_cache_root':
                        raise ValueError('Current parent storage is not verified')
                    target = safe_child(base, ref['path'])
                    if not verifier(target, ref):
                        raise ValueError('Current parent artifact is not verified')
                    snapshots[target] = self._identity(target)
            parents[parent], parent_keys[parent], parent_hashes[parent] = saved, key, digest(saved)
        context_files = {}
        if stage == 'understanding':
            for name in ('Comment.jsonl', 'Protocol.json'):
                path = layout.comments / name
                context_files[name] = file_hash(_PacedInput(config, path)) if path.is_file() else None
                snapshots[path] = self._identity(path) if path.is_file() else None
        inputs = (record if stage == 'retention' else visual_input(parents['retention']) if stage == 'vision'
                  else parents['retention'] if stage == 'stt' else
                  {'vision': parents['vision'], 'stt': parents['stt'], 'context': context} if stage == 'understanding'
                  else parents['understanding'])
        model_key = runner._key(stage, record, inputs)
        if stage in {'retention', 'vision', 'stt'} and (layout.receipts / rid / 'input-state.json').is_file():
            signatures = ([record['source_signature'], record.get('audio', {}).get('source_signature')]
                          if stage == 'retention' else record.get('audio', {}).get('source_signature')
                          if stage == 'stt' else record['source_signature'])
            revision = digest(['inplace-queue/1', stage, rid, signatures, runner._key(stage, record, {})])
        else:
            dependencies = dependencies_for(runner.settings, record)[stage]
            selected = record if stage == 'retention' else inputs if stage == 'vision' else {p: parents[p] for p in dependencies}
            revision = runner._key(stage, record, [selected, context if stage == 'understanding' else None])
        if revision != row['revision'] and not readmission:
            raise ValueError('Current queue recipe changed; normal admission is required')
        repair = None if normal_handoff else self._repair(config, runner, stage, row, kind)
        proof = {'schema_version': VERSION, 'status': 'PROVEN', 'verified_at': time.time(), 'stage': stage,
                 'queue_row_sha256': digest(row), 'source_signature': record['source_signature'],
                 'inventory_sha256': digest(inventory), 'source_proof_sha256': digest(hashes),
                 'current_queue_revision': revision, 'current_model_key': model_key,
                 'parent_receipt_sha256': parent_hashes, 'parent_execution_sha256': parent_keys,
                 'context_file_sha256': context_files, 'verified_context_sha256': digest(context),
                 'source_verification': SOURCE_CHECK, 'cache_preservation_scope': CACHE_CHECK,
                 'model_invoked': False, 'capture_modified': False, 'cache_modified': False, 'repair': repair}
        # A seal cannot be silently replaced while its byte-bound proof is used.
        for primary, saved in seals.items():
            if binding(saved) != binding(read(runner._receipt(layout, record, 'input-' + primary))):
                raise ValueError('Input seal changed during verification')
        final = configured_record(config, _inspect(root, video, time.time(),
            config['collection_ingest'].get('settle_seconds', 120),
            bool(config['collection_ingest'].get('discover_plain_video_csv', False))))
        if (not final.get('processable', final.get('available'))
                or any(final.get(k) != record.get(k) for k in fields)
                or final.get('audio', {}) != record.get('audio', {})
                or stage == 'understanding' and digest(runner._context(layout, record)) != digest(context)):
            raise ValueError('Current input changed during verification')
        self.pending = (digest(proof), runner, snapshots, digest(inventory), repair, digest(config), kind)
        self.recheck(config, stage, row, proof, stop)
        return proof

    def _readmission_scope(self, config, stage, row, kind, *, normal_recipe=False):
        """Scheduling permission is separate from a verified repair budget."""
        from .device_day_admission import admission_status
        from .device_day_backfill_policy import live_demand
        from .device_day_cohort import recovery_entries
        from .device_day_night_schedule import stage_admitted
        from .device_day_retry import failure_kind
        from .device_day_schedule import in_processing_scope
        settings = config.get('device_day') or {}
        policy = settings.get('retry_repair') or {}
        failure = json.loads(row.get('result') or '{}')
        authorized = ((settings.get('input_integrity') or {}).get('enabled') is True
                      and failure.get('error_type') == 'InputChanged') if normal_recipe else (
            policy.get('enabled') is True and kind in policy.get('allowed_classes', [])
            and kind in self.supported and failure_kind(failure) == kind)
        if not authorized:
            raise ValueError('Readmission repair class is not authorized')
        # Neither a failed stage row nor an absent response proves a paid
        # request was never submitted. Those stages keep their ledger/cache
        # reconciliation gates; this handoff is strictly model-free.
        if stage not in {'retention', 'vision'}:
            raise ValueError('Readmission paid request evidence is unavailable')
        if (row.get('status') != 'failed' or row.get('input_status') != 'ready'
                or row.get('blocked_reason') or row.get('lease_owner') or row.get('lease_until')
                or row.get('attempts', 0) < max(settings.get('failure_retry_limit') or 3, row.get('retry_limit') or 0)
                or failure.get('status') in {'submitted_unknown', 'result_unknown'}):
            raise ValueError('Readmission requires an unowned known failed result')
        record = json.loads(row['payload'])
        if (not in_processing_scope(settings, record) or not stage_admitted(config, stage)
                or admission_status(config, stage) is not None):
            raise ValueError('Readmission current worker scope is unavailable')
        entry = recovery_entries(settings).get((stage, record['recording_id']))
        restored = entry and entry['source_signature'] == record.get('source_signature')
        recent = max(record.get('recording_start_us') or 0, record.get('recording_end_us') or 0) >= (
            time.time() - settings.get('live_priority_seconds', 14400)) * 1_000_000
        backfill = settings.get('backfill') or {}
        if restored:
            cohort = 'current_recovery'
        elif recent and row.get('processing_cohort') != 'historical_backfill':
            cohort = 'live'
        elif backfill.get('enabled') is True and stage in backfill.get('stages', []):
            cohort = 'historical_backfill'
        else:
            raise ValueError('Readmission historical scope is not authorized')
        demand = live_demand(config, stage)
        if demand['blocked']:
            raise ValueError('Readmission current scheduling observation is unavailable')
        return cohort

    def readmit(self, config, stage, row, kind, stop):
        """Re-enter normal admission after exact current byte/parent proof.

        Same-recipe failures retain attempts and result until the ordinary
        verifier grants a finite retry. A genuinely new recipe goes through
        the existing worker's enqueue path and archives the former queue row.
        No completion, execution alias or paid submission is created here.
        """
        from .device_day_contract import read_json
        from .device_day_retention_worker import RetentionWorker
        from .device_day_write_guard import storage_guard
        from .runtime_control import check_cancelled
        cohort = self._readmission_scope(config, stage, row, kind)
        with storage_guard(config):
            proof = self._verify(config, stage, row, kind, stop, readmission=True)
            _, runner, *_ = self.pending
            record = json.loads(row['payload'])
            receipt = runner._receipt(runner.layout(record), record, stage)
            if receipt.is_file() and read_json(receipt).get('status') in {'submitted_unknown', 'result_unknown'}:
                raise ValueError('Readmission requires an unowned known failed result')
            self._recheck(config, stage, row, proof, stop)
            if self._readmission_scope(config, stage, row, kind) != cohort:
                raise ValueError('Readmission current worker scope is unavailable')
            check_cancelled(stop)
            proof_id = digest(proof)
            evidence_path = runner.runtime_root / 'RetryRepair' / 'Readmission' / stage / (proof_id + '.json')
            atomic_json(evidence_path, proof | {'previous_queue_revision': row['revision'],
                                              'processing_cohort': cohort, 'paid_request_scope': 'not_applicable'})
            if proof['current_queue_revision'] == row['revision']:
                admitted = runner.queues[stage].grant_admission(record, cohort=cohort,
                    reason='verified_infrastructure_readmission:' + proof_id, expected=row)
                if not admitted:
                    raise ValueError('Current verification changed before grant')
                status = 'source_readmitted'
            else:
                if self._row(runner, stage, row['recording_id']) != row:
                    raise ValueError('Current verification changed before grant')
                admitted = RetentionWorker(stage=stage, admission_cohort=cohort).admit(runner, record)
                current = self._row(runner, stage, row['recording_id'])
                if not admitted or not current or current['revision'] != proof['current_queue_revision']:
                    raise ValueError('Readmission normal worker admission is waiting')
                status = 'recipe_readmitted'
            return {'status': status, 'stage': stage, 'recording_id': row['recording_id'],
                    'previous_revision': row['revision'], 'current_revision': proof['current_queue_revision'],
                    'proof_sha256': proof_id, 'model_invoked': False, 'completed': False}

    def normal_readmit(self, config, stage, row, stop):
        """Hand off an actually changed primary recipe, never grant a repair budget."""
        from .device_day import DeviceDayRunner
        from .device_day_contract import read_json
        from .device_day_recovery_worker import verify_storage
        from .device_day_retention_worker import RetentionWorker
        from .device_day_write_guard import storage_guard
        from .input_availability import revalidated_input_seals
        cohort = self._readmission_scope(config, stage, row, None, normal_recipe=True)
        verify_storage(config)
        with storage_guard(config):
            runner = DeviceDayRunner(config, backend=object())
            record = json.loads(row['payload'])
            with revalidated_input_seals(runner, record) as seals:
                if not seals['verified_seals']:
                    raise ValueError('Readmission original input seals are unavailable')
                proof = self._verify(config, stage, row, None, stop, readmission=True, normal_handoff=True)
                if proof['current_queue_revision'] == row['revision']:
                    raise ValueError('Readmission requires an actually changed execution recipe')
                receipt = runner._receipt(runner.layout(record), record, stage)
                if receipt.is_file() and read_json(receipt).get('status') in {'submitted_unknown', 'result_unknown'}:
                    raise ValueError('Readmission requires an unowned known failed result')
                self._recheck(config, stage, row, proof, stop)
                if self._readmission_scope(config, stage, row, None, normal_recipe=True) != cohort:
                    raise ValueError('Readmission current worker scope is unavailable')
                proof_id = digest(proof)
                atomic_json(runner.runtime_root / 'InputIntegrity/Readmission' / stage / (proof_id + '.json'),
                            proof | {'previous_revision': row['revision'], 'input_seal_proof': seals,
                                     'retry_budget_granted': False})
                admitted = RetentionWorker(stage=stage, admission_cohort=cohort).admit(runner, record)
                current = self._row(runner, stage, row['recording_id'])
                if not admitted or not current or current['revision'] != proof['current_queue_revision']:
                    raise ValueError('Readmission normal worker admission is waiting')
                return {'status': 'normal_recipe_readmitted', 'stage': stage, 'recording_id': row['recording_id'],
                        'previous_revision': row['revision'], 'current_revision': current['revision'],
                        'proof_sha256': proof_id, 'retry_budget_granted': False,
                        'model_invoked': False, 'capture_modified': False, 'completed': False}

    def recheck(self, config, stage, row, proof, stop):
        from .device_day_write_guard import storage_guard
        with storage_guard(config):
            return self._recheck(config, stage, row, proof, stop)

    def _recheck(self, config, stage, row, proof, stop):
        from .runtime_control import check_cancelled
        check_cancelled(stop)
        if not self.pending or self.pending[0] != digest(proof):
            raise ValueError('Verification did not originate in this process')
        _, runner, snapshots, inventory, repair, config_hash, kind = self.pending
        if (digest(config) != config_hash or self._row(runner, stage, row['recording_id']) != row
                or digest(self._inventory(runner, row['recording_id'])) != inventory
                or (self._repair(config, runner, stage, row, kind) if kind is not None else None) != repair):
            raise ValueError('Current verification changed before grant')
        for path, before in snapshots.items():
            after = self._identity(path) if path.is_file() else None
            if before != after:
                raise ValueError('Current verification changed before grant')
        return {'schema_version': RECHECK_VERSION, 'status': 'PROVEN', 'checked_at': time.time(),
                'verification_sha256': digest(proof), 'queue_row_sha256': digest(row)}


def native_verifier(config):
    return NativeInfrastructureVerifier(config)


def main(argv=None):
    import argparse
    import signal
    parser = argparse.ArgumentParser(description='Independent model-free operational services')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--input-integrity', action='store_true')
    modes.add_argument('--input-availability', action='store_true')
    modes.add_argument('--progress', action='store_true')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    logging.basicConfig(level=logging.INFO)
    if args.progress:
        serve_progress(args.config, stop)
    elif args.input_availability:
        serve_availability(args.config, stop)
    else:
        serve_integrity(args.config, stop)


if __name__ == '__main__':
    main()
