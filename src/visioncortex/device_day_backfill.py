"""Bounded historical work using the existing live worker's model pool."""
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import time
import uuid

from .device_day_backfill_policy import BACKFILL_SOURCE, historical_candidates, live_demand
from .device_day_retention_worker import RetentionWorker
from .runtime_control import ExecutionCancelled, ExecutionYielded, execution_context


SUPPORTED_STAGES = ('retention', 'vision', 'stt', 'understanding', 'report')


def options(settings):
    policy = settings.get('backfill', {})
    if not isinstance(policy, dict):
        raise ValueError('device_day.backfill must be a mapping')
    enabled = policy.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('device_day.backfill.enabled must be a boolean')
    if enabled and settings.get('camera_lanes') is not True:
        raise ValueError('device_day.backfill requires camera_lanes to preserve live admission')
    stages = policy.get('stages', list(SUPPORTED_STAGES))
    if (not isinstance(stages, list) or not stages
            or any(not isinstance(stage, str) or stage not in SUPPORTED_STAGES for stage in stages)
            or len(stages) != len(set(stages))):
        raise ValueError('device_day.backfill.stages must name supported live workers')
    result = {'enabled': enabled, 'stages': stages}
    mode = policy.get('mode', 'idle')
    if not isinstance(mode, str) or mode not in {'idle', 'fair'}:
        raise ValueError('device_day.backfill.mode must be idle or fair')
    result['mode'] = mode
    for name, default, minimum in (('idle_seconds', 15, 0), ('quantum_seconds', 30, 1),
                                   ('failure_cooldown_seconds', 60, 5),
                                   ('monitor_max_age_seconds', 30, 1),
                                   ('fair_interval_seconds', 60, 1),
                                   ('fair_turn_timeout_seconds', 15, 1)):
        value = policy.get(name, default)
        if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum):
            raise ValueError(f'device_day.backfill.{name} must be finite and >= {minimum}')
        result[name] = value
    return result


class YieldSignal:
    """Observe arrivals at reusable boundaries; no cancellation of paid calls."""
    def __init__(self, config, stage, stop, quantum):
        self.config, self.stage, self.stop = config, stage, stop
        self.deadline = time.monotonic() + quantum
        self.reason = None
        self._idle_checked_at = None

    def is_set(self, *, ignore_quantum=False):
        stopping = self.stop.is_set()
        current = time.monotonic()
        quantum_elapsed = not ignore_quantum and current >= self.deadline
        if self.reason is not None and self.reason != 'quantum_elapsed':
            return True
        self.reason = None
        if stopping:
            self.reason = 'stopping'
        elif quantum_elapsed:
            self.reason = 'quantum_elapsed'
        elif (self._idle_checked_at is None
              or not self._idle_checked_at <= current < self._idle_checked_at + .1):
            # Hash/decode boundaries can arrive thousands of times per second.
            # Only this signal's successful idle probe is reusable, for at most
            # 100ms; stop/quantum checks above remain immediate at every call.
            self._idle_checked_at = None
            demand = live_demand(self.config, self.stage)
            if demand['blocked']:
                self.reason = demand['status']
            else:
                self._idle_checked_at = current
        return self.reason is not None


class BackfillWorker:
    """One durable unit at a time, with a live check before every admission."""
    def __init__(self, stage):
        if stage not in SUPPORTED_STAGES:
            raise ValueError('Unsupported background stage')
        self.stage = stage
        self.idle_since = None
        self.deferred = {}
        self.admitter = RetentionWorker(stage=stage)
        self.next_fair_at = 0.0

    def tick(self, runner, stop):
        from .device_day import exclusive
        from .device_day_admission import admission_status
        from .device_day_night_schedule import paused_stages, stage_admitted
        from .device_day_provider_gate import ProviderGate
        policy = options(runner.settings)
        fair = policy['mode'] == 'fair'
        context = {'source': BACKFILL_SOURCE, 'priority': 2 if fair else 100,
                   'background_fair': fair}
        if not policy['enabled'] or self.stage not in policy['stages']:
            return {'status': 'disabled'}
        if stop.is_set():
            return {'status': 'stopping'}
        if self.stage in paused_stages(runner.config):
            self.idle_since = None
            return {'status': 'paused_by_user'}
        hold = admission_status(runner.config, self.stage)
        if hold:
            self.idle_since = None
            return hold
        if self.stage in {'understanding', 'report'}:
            if not stage_admitted(runner.config, self.stage):
                return {'status': 'waiting_for_night_window'}
            if ProviderGate(runner.config).blocks(self.stage):
                return {'status': 'waiting_for_provider'}
        demand = live_demand(runner.config, self.stage)
        if demand['blocked']:
            self.idle_since = None
            return {'status': demand['status'], 'reasons': demand['reasons']}
        now = time.monotonic()
        if fair and now < self.next_fair_at:
            return {'status': 'waiting_for_fair_interval'}
        if self.idle_since is None:
            self.idle_since = now
        if not fair and now - self.idle_since < policy['idle_seconds']:
            return {'status': 'waiting_for_idle'}
        with ExitStack() as locks:
            try:
                # Separate from every live stage lock. At most one historical
                # recording runs across processes/stages on this runtime root.
                locks.enter_context(exclusive(runner.runtime_root / 'locks' / 'backfill.lock'))
            except BlockingIOError:
                if fair:
                    self.next_fair_at = now + 1
                return {'status': 'history_running_elsewhere'}
            self.deferred = {rid: until for rid, until in self.deferred.items() if until > time.time()}
            rows = historical_candidates(runner.config, self.stage, limit=16, exclude=set(self.deferred))
            if not rows:
                if fair:
                    self.next_fair_at = now + policy['fair_interval_seconds']
                result = {'status': 'waiting', 'admitted': 0}
                if self.deferred:
                    result['reason'] = 'waiting_for_prerequisite_validation'
                return result
            if fair:
                from .device_day_contract import atomic_json, read_json
                path = runner.runtime_root / 'backfill-fair.json'
                previous = read_json(path) if path.is_file() else {}
                stages = policy['stages']
                last = previous.get('last_stage')
                start = (stages.index(last) + 1) % len(stages) if last in stages else 0
                rotation = stages[start:] + stages[:start]
                paused = paused_stages(runner.config)
                turn = next((stage for stage in rotation if stage not in paused and
                    (stage == self.stage or historical_candidates(runner.config, stage, limit=1))), None)
                if turn != self.stage:
                    waiting = previous.get('waiting_since')
                    stamp = time.time()
                    if (previous.get('waiting_for') != turn or type(waiting) not in (int, float)
                            or not math.isfinite(waiting) or waiting > stamp):
                        waiting = stamp
                        atomic_json(path, previous | {'waiting_for': turn, 'waiting_since': stamp})
                    if stamp - waiting < policy['fair_turn_timeout_seconds']:
                        self.next_fair_at = now + 1
                        return {'status': 'waiting_for_history_turn', 'next_stage': turn}
                    # A missing or busy stage must not strand every other
                    # consumer. The global lock makes the bounded handoff atomic.
                atomic_json(path, {'schema_version': 'device-day-fair-history/1',
                                   'last_stage': self.stage, 'admitted_at': time.time()})
                self.next_fair_at = now + policy['fair_interval_seconds']
            if self.stage == 'vision' and not fair and hasattr(runner, '_backend'):
                backend = runner._backend()
                ready = getattr(backend, 'background_vision_ready', None)
                if ready is not None and not ready():
                    prepare = getattr(backend, 'prepare_background_vision', None)
                    if prepare is None:
                        return {'status': 'waiting_for_warm_model', 'reason': 'waiting_for_warm_model'}
                    signal = YieldSignal(runner.config, self.stage, stop, policy['quantum_seconds'])
                    try:
                        with execution_context(**context, yield_signal=signal):
                            prepared = prepare(signal)
                    except ExecutionYielded as exc:
                        self.idle_since = None
                        return {'status': 'paused_for_live', 'reason': signal.reason or exc.reason}
                    if not prepared:
                        return {'status': 'waiting_for_warm_model', 'reason': 'waiting_for_warm_model'}
            admitted = set()
            for record in rows:
                demand = live_demand(runner.config, self.stage)
                if stop.is_set() or demand['blocked']:
                    self.idle_since = None
                    return {'status': 'waiting_for_live'}
                try:
                    # Receipt/media verification itself uses idle resources.
                    signal = YieldSignal(runner.config, self.stage, stop, policy['quantum_seconds'])
                    with execution_context(**context, yield_signal=signal):
                        if self.admitter.admit(runner, record):
                            admitted.add(record['recording_id'])
                        else:
                            self.deferred[record['recording_id']] = time.time() + policy['failure_cooldown_seconds']
                except ExecutionYielded as exc:
                    self.idle_since = None
                    return {'status': 'paused_for_live', 'reason': signal.reason or exc.reason}
                except (OSError, ValueError, KeyError, TypeError):
                    self.deferred[record['recording_id']] = time.time() + policy['failure_cooldown_seconds']
                if admitted:
                    break
            if not admitted:
                result = {'status': 'waiting', 'admitted': 0}
                if rows or self.deferred:
                    result['reason'] = 'waiting_for_prerequisite_validation'
                return result
            signal = YieldSignal(runner.config, self.stage, stop, policy['quantum_seconds'])
            if signal.is_set():
                self.idle_since = None
                return {'status': 'paused_for_live', 'reason': signal.reason}
            queue = runner.queues[self.stage]
            owner = 'backfill-' + self.stage + '-' + uuid.uuid4().hex
            with queue.heartbeat(owner):
                record = queue.claim(owner, retry=True, allowed=admitted, camera_serial=True,
                                     max_attempts=runner.settings.get('failure_retry_limit') or 3)
                if record is None:
                    return {'status': 'waiting', 'admitted': len(admitted)}
                started = time.perf_counter()
                try:
                    with execution_context(job_id=record['recording_id'], **context, yield_signal=signal):
                        result = runner.process(record, stage=self.stage, retry=True)
                except ExecutionYielded as exc:
                    result = {'status': 'paused_for_live', 'error_type': type(exc).__name__, 'reason': exc.reason}
                except ExecutionCancelled as exc:
                    result = {'status': 'cancelled', 'error_type': type(exc).__name__}
                except Exception as exc:
                    result = {'status': 'failed', 'error_type': type(exc).__name__}
                seconds = time.perf_counter() - started
                queue.finish(owner, record['recording_id'], result, seconds)
                state = result.get('status')
                reason = result.get('reason') or signal.reason
                if state in {'paused_for_live', 'cancelled'}:
                    # A time quantum re-enters immediately; a live arrival
                    # requires a fresh idle interval after it has drained.
                    if reason != 'quantum_elapsed':
                        self.idle_since = None
                    if reason in {'waiting_for_warm_model', 'model_preparation_busy', 'shared_inference_required'}:
                        self.deferred[record['recording_id']] = time.time() + policy['failure_cooldown_seconds']
                elif state != 'completed':
                    self.deferred[record['recording_id']] = time.time() + policy['failure_cooldown_seconds']
                return {'status': 'processed', 'scope': 'history',
                        'recording_id': record['recording_id'], 'wall_seconds': seconds,
                        'reason': reason, 'result': result}


class BackfillSupervisor:
    """A separate thread keeps long historical units out of the live loop."""
    def __init__(self, stage, stop):
        self.stage, self.stop = stage, stop
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='device-day-backfill')
        self.worker = BackfillWorker(stage)
        self.future = None
        self.generation = None
        self.yield_stop = threading.Event()

    def request_yield(self):
        self.yield_stop.set()

    def poll(self, runner, *, allow_start):
        completed = None
        if self.future is not None and self.future.done():
            try:
                completed = self.future.result()
            except Exception as exc:
                completed = {'status': 'failed', 'error_type': type(exc).__name__}
            self.future = None
        generation = id(runner)
        if self.generation != generation or not allow_start:
            if self.future is not None:
                self.yield_stop.set()
        if self.future is not None:
            return {'status': 'running', 'scope': 'history', 'last_result': completed}
        if not allow_start or self.stop.is_set():
            return completed or {'status': 'waiting_for_live'}
        if generation != self.generation:
            self.worker = BackfillWorker(self.stage)
            self.generation = generation
        policy = options(runner.settings)
        if policy['mode'] == 'fair' and time.monotonic() < self.worker.next_fair_at:
            return completed or {'status': 'waiting_for_fair_interval'}
        # Waiting admission is polled by the ordinary live loop. A completed
        # unit can submit the next one immediately, after that loop checks live.
        self.yield_stop.clear()
        parent, local = self.stop, self.yield_stop
        class CombinedStop:
            def is_set(self):
                return parent.is_set() or local.is_set()
        self.future = self.executor.submit(self.worker.tick, runner, CombinedStop())
        return completed or {'status': 'running', 'scope': 'history'}

    def close(self):
        self.yield_stop.set()
        # Paid admitted calls persist before yielding. Do not tear down their
        # pool or runner while a response is being saved.
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.stage == 'vision':
            from .shared_inference import stop_background_preparation
            stop_background_preparation()
