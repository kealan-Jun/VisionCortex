"""Continuously process ready slices through the shared production queues."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import time
import uuid

from .device_day_contract import DEPENDENCIES, STAGES, atomic_json, digest, read_json
from .device_day_inplace import enqueue, retention_policy
from .device_day_schedule import in_processing_scope
from .input_availability import configured_record
from .sqlite_store import connection


class RetentionWorker:
    """One claim per tick; local discovery only, existing execution validates NAS."""

    def __init__(self, *, stage='retention', failure_cooldown=60, recent_seconds=None):
        if stage not in STAGES:
            raise ValueError('Unknown device/day stage')
        self.stage = stage
        self.recent_seconds = recent_seconds
        self.failure_cooldown = max(5, float(failure_cooldown))
        self.deferred = {}

    def candidates(self, runner, *, limit=16):
        path = runner.runtime_root / 'observed-inventory.sqlite3'
        if not path.is_file():
            return []
        current = time.time()
        with connection(path, readonly=True) as db:
            if self.recent_seconds is None:
                rows = db.execute('SELECT payload FROM observations')
            else:
                # The live lane must not decode the historical inventory on
                # every poll. Captures overlapping the window remain eligible.
                rows = db.execute("SELECT payload FROM observations WHERE "
                    "MAX(COALESCE(json_extract(payload,'$.recording_start_us'),0),"
                    "COALESCE(json_extract(payload,'$.recording_end_us'),0))>=?",
                    ((current - self.recent_seconds) * 1_000_000,))
            records = [json.loads(row[0]) for row in rows]
        if not records:
            return []
        ids = json.dumps([record['recording_id'] for record in records])
        selected = 'recording_id IN (SELECT value FROM json_each(?))'
        with connection(runner.queues[self.stage].path, readonly=True) as db:
            states = {r['recording_id']: dict(r) for r in db.execute(
                'SELECT recording_id,status,lease_until,attempts,updated_at FROM recordings WHERE '
                + selected, (ids,))}
        visual = set()
        if self.stage == 'retention':
            with connection(runner.queues['vision'].path, readonly=True) as db:
                visual = {r[0] for r in db.execute("SELECT recording_id FROM recordings "
                    "WHERE status='completed' AND " + selected, (ids,))}
        ready = None
        if self.stage in {'understanding', 'report'}:
            for parent in self.parents():
                with connection(runner.queues[parent].path, readonly=True) as db:
                    completed = {r[0] for r in db.execute(
                        "SELECT recording_id FROM recordings WHERE status='completed' AND "
                        + selected, (ids,))}
                ready = completed if ready is None else ready & completed
        maximum = runner.settings.get('failure_retry_limit') or 3
        candidates = []
        for record in records:
            rid = record['recording_id']
            if ready is not None and rid not in ready:
                continue
            state = states.get(rid, {})
            if (state.get('status') == 'completed'
                    or state.get('status') == 'running' and (state.get('lease_until') or 0) >= current
                    or state.get('status') == 'failed' and (
                        state.get('attempts', 0) >= maximum
                        or current - state.get('updated_at', current) < self.failure_cooldown)
                    or self.deferred.get(rid, 0) > current):
                continue
            if not in_processing_scope(runner.settings, record):
                continue
            if self.recent_seconds is not None and max(record.get('recording_start_us', 0),
                    record.get('recording_end_us', 0)) < (current - self.recent_seconds) * 1_000_000:
                continue
            record = configured_record(runner.config, record)
            if (not record.get('processable', record.get('available'))
                    or record.get('configured_role') not in {'first_person', 'third_person'}):
                continue
            if self.stage in {'vision', 'understanding', 'report'}:
                order = (-record['recording_start_us'], record['camera_key'])
            elif self.stage == 'stt':
                audio = record.get('audio') or {}
                if audio.get('status') != 'provided':
                    continue
                order = (-record['recording_start_us'], record['camera_key'])
            else:
                policy = retention_policy(runner, record)
                deadline = policy['deadline']
                urgent = policy['cleanup_risk'] or (deadline is not None and
                            current >= deadline - policy['safety_margin_seconds'])
                order = (not urgent, deadline if deadline is not None else float('inf'),
                         rid not in visual, -record['recording_start_us'], record['camera_key'])
            candidates.append((order, record))
        # A camera contributes its newest ready slice; queue.claim retains
        # camera fairness and all running leases, including the main service's.
        chosen, cameras = [], set()
        for _, record in sorted(candidates, key=lambda item: item[0]):
            if record['camera_key'] in cameras:
                continue
            cameras.add(record['camera_key'])
            chosen.append(record)
            if len(chosen) >= limit:
                break
        return chosen

    def parents(self):
        needed = set(DEPENDENCIES[self.stage])
        for _ in STAGES:
            needed.update(parent for stage in tuple(needed) for parent in DEPENDENCIES[stage])
        return [stage for stage in STAGES if stage in needed]

    def admit(self, runner, record):
        from .device_day import load_context, visual_input
        from .device_day_inplace import active, receipt
        native = active(runner, record)
        if self.stage not in {'understanding', 'report'} and native:
            return enqueue(runner, record, self.stage)
        if runner._completion_hold(record, self.stage):
            return False
        # The publication receipt belongs to the native inplace executor.
        # Historical executions retain their original verified parent receipts.
        if native and receipt(runner, record, 'publication').get('status') != 'completed':
            return False
        layout = runner.layout(record)
        record = record | {'archive_date': layout.name[:10]}
        context = (load_context(layout, record) if self.stage in {'understanding', 'report'}
                   else {'comments': [], 'protocol': None})
        prerequisites = {}
        for parent in self.parents():
            retained = prerequisites.get('retention')
            inputs = (record if parent == 'retention' else visual_input(retained) if parent == 'vision'
                      else retained if parent == 'stt' else
                      {'vision': prerequisites.get('vision'), 'stt': prerequisites.get('stt'), 'context': context})
            key = runner._key(parent, record, inputs)
            path = runner._receipt(layout, record, parent)
            saved = read_json(path) if path.is_file() else {}
            if saved.get('status') != 'completed' or not runner._accepts_receipt(saved, key):
                return False
            loaded = runner._load(path, key, layout)
            if loaded is None:
                return False
            prerequisites[parent] = loaded
        # This is the ordinary single-record scheduling recipe. process()
        # rechecks publication, identities and all artifact bytes after claim;
        # it never runs a missing parent model for a single-stage request.
        inputs = (record if self.stage == 'retention' else
                  {parent: prerequisites[parent] for parent in DEPENDENCIES[self.stage]})
        if self.stage == 'vision':
            inputs = visual_input(inputs['retention'])
        revision = runner._key(self.stage, record, [inputs, context if self.stage == 'understanding' else None])
        runner.queues[self.stage].enqueue(record, revision)
        runner.queues[self.stage].resume_prerequisite(record['recording_id'], revision)
        return True

    def tick(self, runner, stop):
        from .device_day import exclusive
        from .device_day_night_schedule import paused_stages, stage_admitted
        from .runtime_control import ExecutionCancelled
        if self.stage in paused_stages(runner.config):
            return {'status': 'paused_by_user'}
        from .device_day_admission import admission_status
        hold = admission_status(runner.config, self.stage)
        if hold:
            return hold
        if self.stage in {'understanding', 'report'}:
            if not stage_admitted(runner.config, self.stage):
                return {'status': 'waiting_for_night_window'}
            from .device_day_provider_gate import ProviderGate
            if ProviderGate(runner.config).blocks(self.stage):
                return {'status': 'waiting_for_provider'}
        with ExitStack() as locks:
            try:
                locks.enter_context(exclusive(runner.runtime_root / 'locks' / f'{self.stage}-worker.lock'))
            except BlockingIOError:
                return {'status': 'running_elsewhere'}
            admitted = set()
            for record in self.candidates(runner):
                if stop.is_set():
                    return {'status': 'stopping'}
                try:
                    if self.admit(runner, record):
                        admitted.add(record['recording_id'])
                    elif self.stage in {'understanding', 'report'}:
                        self.deferred[record['recording_id']] = time.time() + self.failure_cooldown
                except (OSError, ValueError, KeyError, TypeError):
                    self.deferred[record['recording_id']] = time.time() + self.failure_cooldown
            if not admitted or stop.is_set():
                return {'status': 'waiting', 'admitted': len(admitted)}
            queue = runner.queues[self.stage]
            # The main service may have failed a candidate during admission.
            # Honor that fresh failure's cooldown before this worker retries it.
            with connection(queue.path, readonly=True) as db:
                for row in db.execute("SELECT recording_id,updated_at FROM recordings WHERE status='failed'"):
                    if time.time() - row['updated_at'] < self.failure_cooldown:
                        admitted.discard(row['recording_id'])
            owner = self.stage + '-worker-' + uuid.uuid4().hex
            # One older slice may still wait in the original worker's I/O
            # path. Speech can use its second configured global slot for the
            # newest slice; neither its lease nor its stage lock is disturbed.
            speech_capacity = ((runner.config.get('runtime') or {}).get('resource_limits') or {}).get('stt', 2)
            camera_limit = min(2, speech_capacity) if self.stage == 'stt' else 1
            if self.stage == 'understanding':
                # A long, all-frame slice already owned by the main service
                # must leave one lane for a newer slice. Provider requests
                # still acquire the same cross-process cloud resource slots.
                cloud_capacity = ((runner.config.get('runtime') or {}).get('resource_limits') or {}).get('cloud', 4)
                camera_limit = min(2, cloud_capacity)
            if self.stage == 'vision':
                vision_capacity = ((runner.config.get('runtime') or {}).get('resource_limits') or {}).get('vision', 12)
                camera_limit = min(vision_capacity, runner.settings.get('vision_jobs_per_camera', 1) + 1)
            with queue.heartbeat(owner):
                record = queue.claim(owner, retry=True, allowed=admitted, camera_serial=True,
                                     camera_limit=camera_limit,
                                     max_attempts=runner.settings.get('failure_retry_limit') or 3)
                if record is None:
                    return {'status': 'waiting', 'admitted': len(admitted)}
                started = time.perf_counter()
                try:
                    # Stop only prevents new claims. An already leased archive
                    # copy and publication finish normally during handover.
                    if self.stage == 'vision':
                        from .device_day_io import live_vision_lane
                        with live_vision_lane():
                            result = runner.process(record, stage=self.stage, retry=True)
                    else:
                        result = runner.process(record, stage=self.stage, retry=True)
                except ExecutionCancelled as exc:
                    result = {'status': 'cancelled', 'recording_id': record['recording_id'],
                              'error_type': type(exc).__name__, 'message': str(exc)[:1000]}
                except Exception as exc:
                    result = {'status': 'failed', 'recording_id': record['recording_id'],
                              'error_type': type(exc).__name__, 'message': str(exc)[:1000]}
                seconds = time.perf_counter() - started
                queue.finish(owner, record['recording_id'], result, seconds)
                if result.get('status') != 'completed':
                    self.deferred[record['recording_id']] = time.time() + self.failure_cooldown
                return {'status': 'processed', 'recording_id': record['recording_id'],
                        'wall_seconds': seconds, 'result': result}


def serve(config_path, stop, *, stage='retention'):
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day import DeviceDayRunner
    from .device_day_backfill import BackfillSupervisor, SUPPORTED_STAGES, options
    from .device_day_consumers import WorkerStatus
    # Preserve the live-only admission lane. History uses a separate idle-only
    # fallback and shares this runner's resident models, never its stage lock.
    worker = RetentionWorker(stage=stage, recent_seconds=14400 if stage in {'vision', 'understanding', 'report'} else None)
    runner, generation, roots, status_path, consumer = None, None, None, None, None
    backfill = BackfillSupervisor(stage, stop) if stage in SUPPORTED_STAGES else None
    try:
        while not stop.is_set():
            delay = 5
            try:
                config = apply_active(load_config(Path(config_path)))
                settings = config.get('device_day') or {}
                if not settings.get('enabled') or not settings.get('inplace_preprocessing') or not settings.get('latest_first'):
                    raise ValueError('Continuous stages require enabled inplace latest-first processing')
                current_roots = {key: config['storage'].get(key) for key in
                                 ('local_runtime_root', 'local_cache_root', 'archive_root')}
                if roots is not None and roots != current_roots:
                    raise ValueError('Retention worker storage roots changed')
                roots = current_roots
                policy = options(settings)
                enabled = backfill is not None and policy['enabled'] and stage in policy['stages']
                key = digest(config)
                if backfill is not None and (not enabled or generation != key):
                    backfill.request_yield()
                if runner is None or generation != key:
                    runner = DeviceDayRunner(config)
                    generation = key
                    if consumer is not None:
                        consumer.__exit__(None, None, None)
                    worker.recent_seconds = settings.get('live_priority_seconds', 14400) if stage in {'vision', 'understanding', 'report'} else None
                    consumer = WorkerStatus(config, stage, recent_seconds=worker.recent_seconds,
                                            backfill_enabled=enabled).__enter__()
                directory = {'retention': 'RetentionWorker', 'stt': 'SpeechWorker', 'vision': 'VisionWorker',
                             'understanding': 'UnderstandingWorker', 'report': 'ReportWorker'}[stage]
                status_path = runner.runtime_root / directory / 'Service.json'
                atomic_json(status_path, {'status': 'running', 'updated_at': time.time(), 'stage': stage})
                consumer.update(scope='recent' if worker.recent_seconds is not None else 'all',
                                recording_id=None, backfill_status='checking_live')
                result = worker.tick(runner, stop)
                history = None
                if backfill is not None:
                    allow_start = enabled and result.get('status') == 'waiting' and result.get('admitted', 0) == 0
                    history = backfill.poll(runner, allow_start=allow_start)
                if enabled:
                    delay = 1
                    if history and history.get('status') == 'processed':
                        delay = 0
                    consumer.update(backfill_status=(history.get('reason') or history['status']) if history else 'waiting_for_live')
                status = {'status': 'running', 'updated_at': time.time(), 'stage': stage,
                          'last_result': result, 'backfill_result': history}
                consumer.update(last_result=history or result)
                if any(value and (value.get('result') or {}).get('status') == 'completed'
                       for value in (result, history)):
                    consumer.update(last_success_at=time.time())
            except Exception as exc:
                if backfill is not None:
                    backfill.request_yield()
                status = {'status': 'failed', 'updated_at': time.time(), 'stage': stage,
                          'error_type': type(exc).__name__}
                if consumer is not None:
                    consumer.update(status='failed', error_type=type(exc).__name__)
            if status_path is not None:
                atomic_json(status_path, status)
            stop.wait(delay)
    finally:
        if backfill is not None:
            backfill.close()
        if consumer is not None:
            consumer.__exit__(None, None, None)
        if status_path is not None:
            atomic_json(status_path, {'status': 'stopped', 'updated_at': time.time(), 'stage': stage})


def main(argv=None):
    import argparse
    import signal
    import threading
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=STAGES, default='retention')
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        serve(args.config, stop, stage=args.stage)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
