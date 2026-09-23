from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, get_ident
import pytest
from visioncortex.shared_inference import InferenceBroker, ScannerLease


class Scanner:
    batch_size = 4
    def __init__(self):
        self.owner = get_ident()
        self.last_engine_batch_sizes = []
    def infer(self, packets):
        assert get_ident() == self.owner
        self.last_engine_batch_sizes = [len(packets)]
        return [('box',p) for p in packets]


def test_cross_camera_batch_keeps_source_order_and_model_owner():
    broker = InferenceBroker(Scanner,wait_seconds=.1)
    barrier = Barrier(4)
    def submit(camera):
        barrier.wait()
        return broker.submit([(camera,1)])
    with ThreadPoolExecutor(max_workers=4) as pool:
        out=list(pool.map(submit,range(4)))
    assert out == [[('box',(i,1))] for i in range(4)]
    assert broker.stats()['model_calls']==1
    assert broker.stats()['mixed_request_calls']==1
    assert broker.submit([])==[]


def test_error_propagation_does_not_drop_next_request():
    class FailsOnce(Scanner):
        def infer(self, packets):
            if packets==['bad']:
                raise ValueError('invalid input')
            return super().infer(packets)
    broker=InferenceBroker(FailsOnce)
    with pytest.raises(ValueError,match='invalid input'):
        broker.submit(['bad'])
    assert broker.submit(['good']) == [('box','good')]


def test_oversized_request_and_late_tail_are_not_lost():
    broker=InferenceBroker(Scanner)
    assert broker.submit(list(range(7))) == [('box',n) for n in range(7)]
    assert broker.submit([8]) == [('box',8)]


def test_engine_batch_statistics_count_successful_physical_batches():
    class SplitScanner(Scanner):
        def infer(self, packets):
            if packets == ['bad']:
                raise ValueError('invalid input')
            result = super().infer(packets)
            self.last_engine_batch_sizes = [
                len(packets[start:start + self.batch_size])
                for start in range(0, len(packets), self.batch_size)
            ]
            return result

    broker = InferenceBroker(SplitScanner)
    try:
        lease = ScannerLease(broker)
        assert lease.infer(list(range(16))) == [('box', n) for n in range(16)]
        assert lease.last_inference_batch_sizes == [16]
        first_stats = lease.shared_statistics()
        assert first_stats['scope'] == 'process_role_pool_cumulative_not_per_video'
        assert first_stats['engine_batch_size_counts'] == {'4': 4}
        assert first_stats['engine_batch_size_max'] == 4
        assert first_stats['effective_batch_size'] == 4
        assert first_stats['oom_batch_contractions'] == []

        with pytest.raises(ValueError, match='invalid input'):
            lease.infer(['bad'])
        assert lease.shared_statistics()['engine_batch_size_counts'] == {'4': 4}
        lease.infer(list(range(5)))
        stats = lease.shared_statistics()
        assert stats['engine_batch_size_counts'] == {'1': 1, '4': 5}
        assert stats['engine_batch_size_max'] == 4
        assert stats['last_engine_batch_sizes'] == [4, 1]
        assert stats['model_calls'] == 2
        assert stats['frames'] == 21
        assert first_stats['engine_batch_size_counts'] == {'4': 4}
    finally:
        broker.close()


def test_shared_statistics_report_backend_batch_contractions():
    class ContractingScanner(Scanner):
        batch_size = 16

        def __init__(self):
            super().__init__()
            self.batch_contractions = []

        def infer(self, packets):
            result = super().infer(packets)
            self.batch_contractions.append({'from_batch_size': 16, 'to_batch_size': 4})
            self.batch_size = 4
            self.last_engine_batch_sizes = [4, 4, 4, 4]
            return result

    broker = InferenceBroker(ContractingScanner)
    try:
        lease = ScannerLease(broker)
        assert lease.shared_statistics()['effective_batch_size'] == 16
        lease.infer(list(range(16)))
        stats = lease.shared_statistics()
        assert stats['engine_batch_size_counts'] == {'4': 4}
        assert stats['engine_batch_size_max'] == 4
        assert stats['effective_batch_size'] == 4
        expected = [{'from_batch_size': 16, 'to_batch_size': 4}]
        assert stats['oom_batch_contractions'] == expected
        assert lease.batch_contractions == expected
        stats['oom_batch_contractions'][0]['to_batch_size'] = 1
        assert lease.shared_statistics()['oom_batch_contractions'] == expected
    finally:
        broker.close()


def test_contexts_grow_with_active_cameras_and_reuse_released_models(monkeypatch):
    import visioncortex.shared_inference as shared
    monkeypatch.setattr(shared, '_POOLS', {})
    created = []
    def factory(*args, **kwargs):
        scanner = Scanner()
        created.append(scanner)
        return scanner
    config = {'models': {}, 'performance': {'shared_inference_enabled': True,
                                            'shared_inference_contexts_per_pool': 2}}
    first = shared.acquire_scanner(factory, 'first_person', config, 640, 16)
    assert len(created) == 1
    second = shared.acquire_scanner(factory, 'first_person', config, 640, 16)
    assert len(created) == 2
    third = shared.acquire_scanner(factory, 'first_person', config, 640, 16)
    assert len(created) == 2
    assert first.infer([1]) == [('box', 1)]
    second.close()
    fourth = shared.acquire_scanner(factory, 'first_person', config, 640, 16)
    assert fourth.broker is second.broker
    assert len(created) == 2
    for lease in (first, third, fourth):
        lease.close()


@pytest.fixture
def shared_pool(monkeypatch):
    import visioncortex.shared_inference as shared
    monkeypatch.setattr(shared, '_POOLS', {})
    yield shared
    assert shared.close_pools(timeout=1)


def shared_config():
    return {'models': {}, 'performance': {'shared_inference_enabled': True,
                                         'shared_inference_contexts_per_pool': 2}}


def test_failed_growth_reuses_same_model_and_cools_down_without_changing_batch(shared_pool):
    shared, config, calls = shared_pool, shared_config(), []
    def factory(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            # Real TensorRT initialization OOM may surface through YOLO.names.
            raise AttributeError("'str' object has no attribute 'names'")
        return Scanner()
    first = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    assert first.infer([1]) == [('box', 1)]
    fallback = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    assert fallback.broker is first.broker
    assert fallback.batch_size == 4
    assert fallback.infer([2, 3]) == [('box', 2), ('box', 3)]
    for _ in range(5):
        lease = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
        assert lease.broker is first.broker
        lease.close()
    assert len(calls) == 2
    stats = first.shared_statistics()['context_pool']
    assert stats['growth_failures'] == 1
    assert stats['last_growth_error_type'] == 'AttributeError'
    assert 0 < stats['growth_retry_after_seconds'] <= 30
    pool = next(iter(shared._POOLS.values()))
    for broker in pool:
        if broker.stop.is_set():
            broker.thread.join(1)
    pool.retry_after = 0  # Elapse the cooldown without wall-clock sleeping.
    expanded = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    assert len(calls) == 3
    assert expanded.broker is not first.broker
    assert len(pool) == 2
    for lease in (first, fallback, expanded):
        lease.close()


def test_first_context_failure_never_uses_a_different_pool(shared_pool):
    shared, config = shared_pool, shared_config()
    first = shared.acquire_scanner(lambda *a, **kw: Scanner(), 'first_person', config, 640, 4)
    def failure(*args, **kwargs):
        raise AttributeError('context initialization failed')
    with pytest.raises(AttributeError, match='context initialization failed'):
        shared.acquire_scanner(failure, 'third_person', config, 640, 4)
    assert first.infer([1]) == [('box', 1)]
    first.close()


def test_all_quarantined_contexts_do_not_get_replaced_or_reused(shared_pool):
    shared, config, calls = shared_pool, shared_config(), []
    def factory(*args, **kwargs):
        calls.append(1)
        return Scanner()
    first = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    assert first.broker.close()
    first.close()
    with pytest.raises(RuntimeError, match='quarantined'):
        shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    assert len(calls) == 1


def test_timed_out_growth_keeps_its_slot_until_thread_exits(shared_pool):
    shared, config, calls = shared_pool, shared_config(), []
    config['performance']['shared_inference_timeout_seconds'] = .05
    release = Event()
    def factory(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            assert release.wait(3)
        return Scanner()
    first = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    leases = [first]
    try:
        leases.append(shared.acquire_scanner(factory, 'first_person', config, 640, 4))
        assert leases[-1].broker is first.broker
        pool = next(iter(shared._POOLS.values()))
        pending = pool[1]
        assert pending.stop.is_set() and pending.thread.is_alive()
        assert len(pool) == 2
        pool.retry_after = 0
        for _ in range(5):
            leases.append(shared.acquire_scanner(factory, 'first_person', config, 640, 4))
        assert len(calls) == 2
        assert first.shared_statistics()['context_pool']['last_growth_error_type'] == 'TimeoutError'
        release.set()
        pending.thread.join(1)
        assert not pending.thread.is_alive()
        leases.append(shared.acquire_scanner(factory, 'first_person', config, 640, 4))
        assert len(calls) == 3
        assert pending not in pool
        assert len(pool) == 2
    finally:
        release.set()
        for lease in leases:
            lease.close()


def test_initialization_timeout_cannot_be_evicted_while_thread_is_alive(shared_pool, monkeypatch):
    shared, config, calls = shared_pool, shared_config(), []
    config['performance']['shared_inference_timeout_seconds'] = .05
    release = Event()
    def factory(role, *args, **kwargs):
        calls.append(role)
        if role == 'blocked':
            assert release.wait(3)
        return Scanner()
    leases = []
    try:
        with pytest.raises(TimeoutError):
            shared.acquire_scanner(factory, 'blocked', config, 640, 4)
        pending = next(iter(shared._POOLS.values()))[0]
        close = pending.close
        monkeypatch.setattr(pending, 'close', lambda timeout=5: close(timeout=0))
        for role in ('a', 'b', 'c'):
            leases.append(shared.acquire_scanner(factory, role, config, 640, 4))
        with pytest.raises(RuntimeError, match='capacity exhausted'):
            shared.acquire_scanner(factory, 'd', config, 640, 4)
        assert calls == ['blocked', 'a', 'b', 'c']
        assert pending.thread.is_alive()
        release.set()
        pending.thread.join(1)
        leases.append(shared.acquire_scanner(factory, 'd', config, 640, 4))
        assert len(shared._POOLS) == 4
    finally:
        release.set()
        for lease in leases:
            lease.close()


def test_expansion_does_not_swallow_base_exceptions(shared_pool):
    shared, config, calls = shared_pool, shared_config(), []
    def factory(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise KeyboardInterrupt()
        return Scanner()
    first = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    try:
        with pytest.raises(KeyboardInterrupt):
            shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    finally:
        first.close()


def test_ready_waits_for_owned_preparation_and_does_not_count_warmup():
    entered, release = Event(), Event()
    closed = []

    class PreparedScanner(Scanner):
        def prepare(self):
            assert self.owner == get_ident()
            entered.set()
            assert release.wait(3)

        def close(self):
            closed.append(get_ident())

    broker = InferenceBroker(PreparedScanner, defer_ready=True)
    try:
        assert entered.wait(1)
        assert not broker.ready.done()
        assert not broker.healthy()
        assert broker.initialization_phase == 'prepare'
        release.set()
        broker.wait_ready()
        assert broker.healthy()
        assert broker.stats()['model_calls'] == 0
        assert broker.stats()['frames'] == 0
        assert broker.stats()['engine_batch_size_counts'] == {}
        assert broker.submit([1]) == [('box', 1)]
    finally:
        release.set()
        assert broker.close()
    assert closed == [broker.thread.ident]


@pytest.mark.parametrize('fails', [False, True])
def test_nonshared_scanner_prepares_before_return_and_closes_on_failure(fails):
    from visioncortex.shared_inference import acquire_scanner

    events = []

    class DirectScanner(Scanner):
        def prepare(self):
            assert self.owner == get_ident()
            events.append('prepare')
            if fails:
                raise ValueError('cannot initialize')

        def close(self):
            events.append('close')

    def factory(*args, **kwargs):
        return DirectScanner()

    if fails:
        with pytest.raises(ValueError, match='cannot initialize'):
            acquire_scanner(factory, 'first_person', {'performance': {}}, 640, 4)
        assert events == ['prepare', 'close']
    else:
        scanner = acquire_scanner(factory, 'first_person', {'performance': {}}, 640, 4)
        assert events == ['prepare']
        scanner.close()


def test_preparation_failure_uses_healthy_context_and_preserves_failure_phase(shared_pool):
    shared, config, created, closed = shared_pool, shared_config(), [], []

    class PreparedScanner(Scanner):
        def __init__(self):
            super().__init__()
            created.append(self)

        def prepare(self):
            assert self.owner == get_ident()
            if len(created) == 2:
                self.initialization_failure_phase = 'warmup'
                raise AttributeError('context unavailable')

        def close(self):
            assert self.owner == get_ident()
            closed.append(self)

    def factory(*args, **kwargs):
        return PreparedScanner()
    first = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    fallback = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    try:
        assert fallback.broker is first.broker
        pool = next(iter(shared._POOLS.values()))
        failed = pool[1]
        failed.thread.join(1)
        assert not failed.thread.is_alive()
        assert closed == [created[1]]
        stats = first.shared_statistics()['context_pool']
        assert stats['last_growth_failure_phase'] == 'warmup'
        assert stats['last_growth_broker_id'] == failed.broker_id
        assert stats['last_growth_error_type'] == 'AttributeError'
        assert failed.broker_id != first.broker.broker_id
        assert fallback.infer([1]) == [('box', 1)]
        for _ in range(3):
            lease = shared.acquire_scanner(factory, 'first_person', config, 640, 4)
            lease.close()
        assert len(created) == 2
    finally:
        fallback.close()
        first.close()


def test_first_preparation_failure_is_cleaned_up_and_never_ready(shared_pool):
    shared, config, created, closed = shared_pool, shared_config(), [], []

    class FailedPreparation(Scanner):
        def __init__(self):
            super().__init__()
            created.append(self)

        def prepare(self):
            self.initialization_failure_phase = 'predictor_setup'
            raise RuntimeError('CUDA out of memory')

        def close(self):
            assert self.owner == get_ident()
            closed.append(self)

    def factory(*args, **kwargs):
        return FailedPreparation()
    with pytest.raises(RuntimeError, match='CUDA out of memory') as error:
        shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    failed = next(iter(shared._POOLS.values()))[0]
    failed.thread.join(1)
    assert closed == created
    assert not failed.healthy()
    assert failed.stats()['initialization_phase'] == 'failed'
    assert failed.stats()['model_calls'] == 0
    assert 'predictor_setup' in error.value.__notes__[0]
    with pytest.raises(RuntimeError, match='quarantined'):
        shared.acquire_scanner(factory, 'first_person', config, 640, 4)
    assert len(created) == 1


def test_timed_out_preparation_keeps_slot_and_closes_in_owner_thread(shared_pool):
    shared, config, created, closed = shared_pool, shared_config(), [], []
    config['performance']['shared_inference_timeout_seconds'] = .05
    release = Event()

    class SlowPreparation(Scanner):
        def __init__(self):
            super().__init__()
            created.append(self)

        def prepare(self):
            if len(created) == 2:
                assert release.wait(3)

        def close(self):
            assert self.owner == get_ident()
            closed.append(self)

    def factory(*args, **kwargs):
        return SlowPreparation()
    leases = [shared.acquire_scanner(factory, 'first_person', config, 640, 4)]
    try:
        leases.append(shared.acquire_scanner(factory, 'first_person', config, 640, 4))
        assert leases[1].broker is leases[0].broker
        pool = next(iter(shared._POOLS.values()))
        pending = pool[1]
        assert pending.thread.is_alive() and pending.stop.is_set()
        assert not pending.healthy()
        pool.retry_after = 0
        leases.append(shared.acquire_scanner(factory, 'first_person', config, 640, 4))
        assert len(created) == 2
        release.set()
        pending.thread.join(1)
        assert closed == [created[1]]
        assert pending.initialization_phase == 'failed'
    finally:
        release.set()
        for lease in leases:
            lease.close()
