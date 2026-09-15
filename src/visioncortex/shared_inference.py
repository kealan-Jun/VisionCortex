"""Bounded process-local batching; only one thread owns each role model.

Frame packets and returned boxes retain caller order. Tracking, temporal action
logic and publication stay in the caller. This changes execution, not labels.
"""
from concurrent.futures import Future, TimeoutError as FutureTimeout
import hashlib
import json
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
import time

_POOLS = {}
_LOCK = Lock()


class InferenceBroker:
    def __init__(self, factory, wait_seconds=.005, timeout_seconds=120):
        self.queue = Queue(maxsize=32)
        self.wait_seconds = wait_seconds
        self.timeout_seconds = timeout_seconds
        self.ready = Future()
        self.stop = Event()
        self.lock = Lock()
        self.leases = 0
        self.calls = self.frames = self.mixed_calls = 0
        self.engine_batches = []
        self.thread = Thread(target=self._run, args=(factory,), daemon=True, name='shared-role-inference')
        self.thread.start()
        try:
            self.scanner = self.ready.result(timeout=timeout_seconds)
        except BaseException:
            self.stop.set()
            raise

    def _run(self, factory):
        scanner, carry = None, None
        try:
            scanner = factory()
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
                    'frames': self.frames, 'mixed_request_calls': self.mixed_calls,
                    'last_engine_batch_sizes': list(self.engine_batches), 'waiting_requests': self.queue.qsize(),
                    'worker_alive': self.thread.is_alive(), 'quarantined': self.stop.is_set(),
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
        self.batch_contractions = []

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
        return factory(role,config,image_size,batch_size,appearance_enabled=appearance_enabled)
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
            brokers = []
            _POOLS[key] = brokers
        # Allocate a context only for an active caller. Creating three eager
        # models for every role/phase consumes memory and cold-start time even
        # when just one camera has a ready slice. Released contexts stay warm.
        if len(brokers) < capacity and not any(item.leases == 0 for item in brokers):
            brokers.append(InferenceBroker(lambda:factory(role,config,image_size,batch_size,appearance_enabled=False),
                                           timeout_seconds=config['performance'].get('shared_inference_timeout_seconds', 120)))
        healthy = [b for b in brokers if not b.stop.is_set()]
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
