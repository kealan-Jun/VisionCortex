"""Bounded process-local batching; only one thread owns each role model.

Frame packets and returned boxes retain caller order. Tracking, temporal action
logic and publication stay in the caller. This changes execution, not labels.
"""
from concurrent.futures import Future, TimeoutError as FutureTimeout
from collections import Counter
import hashlib
import json
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
import time
import uuid

_POOLS = {}
_LOCK = Lock()


class _InferencePool(list):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.initialized = False
        self.growth_failures = 0
        self.last_growth_error_type = None
        self.last_growth_failure_phase = None
        self.last_growth_broker_id = None
        self.retry_after = 0.0

    def stats(self):
        brokers = list(self)
        return {'context_limit': self.capacity, 'allocated_contexts': len(brokers),
                'healthy_contexts': sum(b.healthy() for b in brokers),
                'initializing_or_stopping_contexts': sum(
                    b.thread.is_alive() and not b.healthy() for b in brokers),
                'growth_failures': self.growth_failures,
                'last_growth_error_type': self.last_growth_error_type,
                'last_growth_failure_phase': self.last_growth_failure_phase,
                'last_growth_broker_id': self.last_growth_broker_id,
                'growth_retry_after_seconds': max(0.0, self.retry_after - time.monotonic())}


class InferenceBroker:
    def __init__(self, factory, wait_seconds=.005, timeout_seconds=120, *, defer_ready=False):
        self.queue = Queue(maxsize=32)
        self.wait_seconds = wait_seconds
        self.timeout_seconds = timeout_seconds
        self.ready = Future()
        self.stop = Event()
        self.lock = Lock()
        self.leases = 0
        self.calls = self.frames = self.mixed_calls = 0
        self.engine_batches = []
        self.engine_batch_size_counts = Counter()
        self.scanner = None
        self.pool = None
        self.broker_id = uuid.uuid4().hex
        self.initialization_phase = 'factory'
        self.initialization_failure_phase = None
        self.thread = Thread(target=self._run, args=(factory,), daemon=True, name='shared-role-inference')
        self.thread.start()
        if not defer_ready:
            self.wait_ready()

    def wait_ready(self):
        try:
            self.scanner = self.ready.result(timeout=self.timeout_seconds)
        except BaseException:
            self.stop.set()
            raise

    def healthy(self):
        return self.scanner is not None and self.thread.is_alive() and not self.stop.is_set()

    def _run(self, factory):
        scanner, carry = None, None
        try:
            scanner = factory()
            if self.stop.is_set():
                raise RuntimeError('Inference initialization cancelled before preparation')
            self.initialization_phase = 'prepare'
            if hasattr(scanner, 'prepare'):
                scanner.prepare()
            if self.stop.is_set():
                raise RuntimeError('Inference initialization cancelled before ready')
            self.initialization_phase = 'ready'
            self.ready.set_result(scanner)
            while not self.stop.is_set():
                try:
                    first = carry or self.queue.get(timeout=.1)
                except Empty:
                    continue
                carry = None
                group = [first]
                size = len(first[0])
                deadline = time.monotonic()+self.wait_seconds
                while size < scanner.batch_size:
                    try:
                        item = self.queue.get(timeout=max(0, deadline-time.monotonic()))
                    except Empty:
                        break
                    if size+len(item[0]) > scanner.batch_size:
                        carry = item
                        break
                    group.append(item)
                    size += len(item[0])
                # A cancelled waiter must not poison another camera's request.
                group = [(packets, future) for packets, future in group
                         if future.set_running_or_notify_cancel()]
                if not group:
                    continue
                try:
                    packets = [packet for request, _ in group for packet in request]
                    output = scanner.infer(packets)
                    if len(output) != len(packets):
                        raise RuntimeError('Shared inference output count differs from submitted frames')
                    with self.lock:
                        self.calls += 1
                        self.frames += len(packets)
                        self.mixed_calls += len(group) > 1
                        self.engine_batches = list(getattr(scanner, 'last_engine_batch_sizes', []))
                        self.engine_batch_size_counts.update(self.engine_batches)
                    offset = 0
                    for request, future in group:
                        future.set_result(output[offset:offset+len(request)])
                        offset += len(request)
                except BaseException as exc:
                    for _, future in group:
                        future.set_exception(exc)
                    if not isinstance(exc, Exception):
                        raise
        except BaseException as exc:
            if not self.ready.done():
                self.initialization_failure_phase = (
                    getattr(scanner, 'initialization_failure_phase', None)
                    or self.initialization_phase)
                self.initialization_phase = 'failed'
                exc.add_note(f'Inference broker {self.broker_id}: '
                             f'initialization failed during {self.initialization_failure_phase}')
                self.ready.set_exception(exc)
        finally:
            self.stop.set()
            pending = [carry] if carry else []
            while True:
                try:
                    pending.append(self.queue.get_nowait())
                except Empty:
                    break
            for _, future in pending:
                if not future.done() and future.set_running_or_notify_cancel():
                    future.set_exception(RuntimeError('Inference worker stopped'))
            if scanner is not None and hasattr(scanner, 'close'):
                scanner.close()  # CUDA resources are destroyed by their owner.

    def submit(self, packets):
        from .runtime_control import check_cancelled
        if not packets:
            return []
        future, deadline = Future(), time.monotonic()+self.timeout_seconds
        try:
            while True:
                check_cancelled()
                if self.stop.is_set():
                    raise RuntimeError('Inference worker unavailable')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Inference admission timed out')
                try:
                    self.queue.put((list(packets), future), timeout=.1)
                    break
                except Full:
                    continue
            while True:
                check_cancelled()
                if time.monotonic() >= deadline:
                    # Native CUDA calls cannot safely be killed in a thread.
                    # Quarantine this broker; never start unlimited replacements.
                    self.stop.set()
                    raise TimeoutError('Inference worker exceeded its execution deadline')
                try:
                    return future.result(timeout=.1)
                except FutureTimeout:
                    if future.done():
                        raise
        finally:
            if not future.done():
                future.cancel()

    def close(self, timeout=5):
        self.stop.set()
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def stats(self):
        with self.lock:
            return {'scope': 'process_role_pool_cumulative_not_per_video', 'model_calls': self.calls,
                    'broker_id': self.broker_id,
                    'initialization_phase': self.initialization_phase,
                    'initialization_failure_phase': self.initialization_failure_phase,
                    'frames': self.frames, 'mixed_request_calls': self.mixed_calls,
                    'last_engine_batch_sizes': list(self.engine_batches), 'waiting_requests': self.queue.qsize(),
                    'engine_batch_size_counts': {str(size): count for size, count in
                                                 sorted(self.engine_batch_size_counts.items())},
                    'engine_batch_size_max': max(self.engine_batch_size_counts, default=0),
                    'effective_batch_size': getattr(self.scanner, 'batch_size', None),
                    'oom_batch_contractions': [dict(item) for item in
                                               getattr(self.scanner, 'batch_contractions', [])],
                    'worker_alive': self.thread.is_alive(), 'quarantined': self.stop.is_set(),
                    'context_pool': self.pool.stats() if self.pool is not None else None,
                    'model_component_seconds': getattr(self.scanner, 'component_timings', None).snapshot() if hasattr(self.scanner, 'component_timings') else {}}


class ScannerLease:
    def __init__(self, broker, appearance_enabled=False):
        from .performance_stages import StageTimings
        self.broker = broker
        self.closed = False
        self.appearance_enabled = appearance_enabled
        self.component_timings = StageTimings()
        self.last_inference_batch_sizes = []
        self.exact_batch_padding_frames = 0

    def __getattr__(self, name):
        return getattr(self.broker.scanner,name)

    def infer(self, packets):
        if self.closed:
            raise RuntimeError('Inference lease is already closed')
        begun = time.perf_counter()
        result = self.broker.submit(packets)
        self.component_timings.add("shared_queue_and_service_seconds",time.perf_counter()-begun)
        if self.appearance_enabled:
            from .detection import _roi_appearance_signature
            result = [[box.model_copy(update={"appearance_signature": _roi_appearance_signature(packet.frame,box.xyxy_norm)})
                       for box in boxes] for packet,boxes in zip(packets,result,strict=True)]
        # These are caller submissions, NOT counts of physical GPU batches.
        self.last_inference_batch_sizes = [len(packets)] if packets else []
        return result

    def close(self):
        # Bounded pools remain warm for subsequent recorder slices.
        with _LOCK:
            if not self.closed:
                self.closed = True
                self.broker.leases -= 1

    def shared_statistics(self):
        return self.broker.stats()


def acquire_scanner(factory, role, config, image_size, batch_size, appearance_enabled=False):
    if not config['performance'].get('shared_inference_enabled',False):
        scanner = factory(role,config,image_size,batch_size,appearance_enabled=appearance_enabled)
        try:
            if hasattr(scanner, 'prepare'):
                scanner.prepare()
        except BaseException:
            if hasattr(scanner, 'close'):
                scanner.close()
            raise
        return scanner
    capacity = config['performance'].get('shared_inference_contexts_per_pool', 3)
    if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 8:
        raise ValueError('shared_inference_contexts_per_pool must be an integer in [1,8]')
    from pathlib import Path
    model_stats = {}
    for key,value in config['models'].items():
        if isinstance(value,str) and Path(value).is_file():
            stat = Path(value).stat()
            model_stats[key] = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    key = hashlib.sha256(json.dumps([str(role),config['models'],config['performance'],model_stats,
                                    image_size,batch_size],sort_keys=True,default=str).encode()).hexdigest()
    with _LOCK:
        brokers = _POOLS.get(key)
        if brokers is None:
            if len(_POOLS) >= 4:
                idle = next((old for old, items in _POOLS.items() if all(b.leases == 0 for b in items)), None)
                if idle is None or not all(b.close() for b in _POOLS[idle]):
                    raise RuntimeError('Inference pool capacity exhausted; active models remain bounded')
                del _POOLS[idle]
            brokers = _InferencePool(capacity)
            _POOLS[key] = brokers
        healthy = [b for b in brokers if b.healthy()]
        if brokers.initialized and not healthy:
            raise RuntimeError('Inference pool quarantined; worker restart required')
        # Failed initialization is removable only once its owning thread has
        # actually exited. Timed-out native calls retain a physical pool slot.
        brokers[:] = [b for b in brokers if b.leases or b.thread.is_alive()]
        stopping = any(b.stop.is_set() and b.thread.is_alive() for b in brokers)
        # Allocate a context only for an active caller. Creating three eager
        # models for every role/phase consumes memory and cold-start time even
        # when just one camera has a ready slice. Released contexts stay warm.
        if (len(brokers) < capacity and not any(item.leases == 0 for item in healthy)
                and not stopping and time.monotonic() >= brokers.retry_after):
            candidate = InferenceBroker(lambda:factory(role,config,image_size,batch_size,appearance_enabled=False),
                                        timeout_seconds=config['performance'].get('shared_inference_timeout_seconds', 120),
                                        defer_ready=True)
            candidate.pool = brokers
            brokers.append(candidate)
            try:
                candidate.wait_ready()
                brokers.initialized = True
            except Exception as exc:
                brokers.growth_failures += 1
                brokers.last_growth_error_type = type(exc).__name__
                brokers.last_growth_failure_phase = candidate.initialization_failure_phase or candidate.initialization_phase
                brokers.last_growth_broker_id = candidate.broker_id
                brokers.retry_after = time.monotonic() + 30
                # Optional capacity growth must not fail a video that can use
                # an identical live model. First/all-bad contexts still fail.
                if not any(b.healthy() for b in brokers):
                    raise
        healthy = [b for b in brokers if b.healthy()]
        if not healthy:
            raise RuntimeError('Inference pool quarantined; worker restart required')
        broker = min(healthy, key=lambda item:(item.leases,item.queue.qsize()))
        broker.leases += 1
    return ScannerLease(broker,appearance_enabled)


def close_pools(timeout=5):
    """Drain owned workers during process shutdown; retain stuck brokers."""
    with _LOCK:
        for key in list(_POOLS):
            if all(b.close(timeout) for b in _POOLS[key]):
                del _POOLS[key]
        return not _POOLS
