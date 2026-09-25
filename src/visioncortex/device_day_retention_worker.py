"""Continuously process ready slices through the shared production queues."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import time
import uuid

from .device_day_contract import atomic_json, digest
from .device_day_inplace import enqueue, retention_policy
from .device_day_schedule import in_processing_scope
from .input_availability import configured_record
from .sqlite_store import connection


class RetentionWorker:
    """One claim per tick; local discovery only, existing execution validates NAS."""

    def __init__(self, *, stage='retention', failure_cooldown=60):
        if stage not in {'retention', 'stt', 'vision'}:
            raise ValueError('Only independent retention, speech and vision stages are supported')
        self.stage = stage
        self.failure_cooldown = max(5, float(failure_cooldown))
        self.deferred = {}

    def candidates(self, runner, *, limit=16):
        path = runner.runtime_root / 'observed-inventory.sqlite3'
        if not path.is_file():
            return []
        current = time.time()
        with connection(runner.queues[self.stage].path, readonly=True) as db:
            states = {r['recording_id']: dict(r) for r in db.execute(
                'SELECT recording_id,status,lease_until,attempts,updated_at FROM recordings')}
        visual = set()
        if self.stage == 'retention':
            with connection(runner.queues['vision'].path, readonly=True) as db:
                visual = {r[0] for r in db.execute("SELECT recording_id FROM recordings WHERE status='completed'")}
        with connection(path, readonly=True) as db:
            records = [json.loads(r[0]) for r in db.execute('SELECT payload FROM observations')]
        maximum = runner.settings.get('failure_retry_limit') or 3
        candidates = []
        for record in records:
            rid = record['recording_id']
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
            record = configured_record(runner.config, record)
            if (not record.get('processable', record.get('available'))
                    or record.get('configured_role') not in {'first_person', 'third_person'}):
                continue
            if self.stage == 'vision':
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

    def tick(self, runner, stop):
        from .device_day import exclusive
        from .device_day_night_schedule import paused_stages
        from .runtime_control import ExecutionCancelled
        if self.stage in paused_stages(runner.config):
            return {'status': 'paused_by_user'}
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
                    if enqueue(runner, record, self.stage):
                        admitted.add(record['recording_id'])
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
    worker = RetentionWorker(stage=stage)
    runner, generation, roots, status_path = None, None, None, None
    while not stop.is_set():
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
            key = digest(config)
            if runner is None or generation != key:
                runner = DeviceDayRunner(config)
                generation = key
            directory = {'retention': 'RetentionWorker', 'stt': 'SpeechWorker', 'vision': 'VisionWorker'}[stage]
            status_path = runner.runtime_root / directory / 'Service.json'
            atomic_json(status_path, {'status': 'running', 'updated_at': time.time(), 'stage': stage})
            result = worker.tick(runner, stop)
            status = {'status': 'running', 'updated_at': time.time(), 'stage': stage, 'last_result': result}
        except Exception as exc:
            status = {'status': 'failed', 'updated_at': time.time(), 'stage': stage,
                      'error_type': type(exc).__name__}
        if status_path is not None:
            atomic_json(status_path, status)
        stop.wait(5)
    if status_path is not None:
        atomic_json(status_path, {'status': 'stopped', 'updated_at': time.time(), 'stage': stage})


def main(argv=None):
    import argparse
    import signal
    import threading
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['retention', 'stt', 'vision'], default='retention')
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
