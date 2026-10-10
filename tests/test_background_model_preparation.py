"""Synthetic factories only: idle preparation must never queue live on a warm future."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time

import pytest

from visioncortex.runtime_control import ExecutionYielded, execution_context


class Scanner:
    batch_size = 4
    def infer(self, packets):
        return packets


def config():
    return {'models': {}, 'performance': {'shared_inference_enabled': True,
             'shared_inference_contexts_per_pool': 2}, 'device_day': {'backfill': {'enabled': True}}}


@pytest.fixture
def shared(monkeypatch):
    import visioncortex.shared_inference as module
    monkeypatch.setattr(module, '_POOLS', {})
    monkeypatch.setattr(module, '_BACKGROUND_WARM', None)
    monkeypatch.setattr(module, '_BACKGROUND_RETRY', {})
    yield module
    assert module.close_pools(timeout=1)


def background(module, factory, settings, role='first_person', signal=None):
    with execution_context(source='device_day_backfill', yield_signal=signal):
        return module.acquire_scanner(factory, role, settings, 640, 4)


def test_cold_idle_pool_prepares_one_model_and_live_reuses_it(shared):
    calls = []
    def factory(*a, **kw):
        calls.append(1)
        return Scanner()
    old = background(shared, factory, config())
    try:
        live = shared.acquire_scanner(factory, 'first_person', config(), 640, 4)
        try:
            assert live.broker is old.broker and calls == [1]
            assert live.infer([1]) == [1]
            assert live.broker.foreground_used
        finally:
            live.close()
    finally:
        old.close()


def test_live_never_waits_for_background_cold_prepare(shared):
    begun, release, arrival = Event(), Event(), Event()
    calls = []
    class Slow(Scanner):
        def prepare(self):
            begun.set()
            assert release.wait(5)
    def factory(*a, **kw):
        calls.append(1)
        return Slow() if len(calls) == 1 else Scanner()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(background, shared, factory, config(), 'first_person', arrival)
        try:
            assert begun.wait(2)
            before = time.monotonic()
            live = shared.acquire_scanner(factory, 'first_person', config(), 640, 4)
            try:
                assert time.monotonic() - before < 1
                assert live.infer([2]) == [2] and len(calls) == 2
                assert not pending.done()
                arrival.set()
                with pytest.raises(ExecutionYielded):
                    pending.result(timeout=2)
            finally:
                live.close()
        finally:
            arrival.set()
            release.set()
    for brokers in shared._POOLS.values():
        for broker in brokers:
            if broker.background_initializing:
                broker.thread.join(2)
                assert not broker.healthy()


def test_arrival_before_prepare_invalidates_private_factory_result(shared):
    begun, release, arrival, prepared = Event(), Event(), Event(), Event()
    class Pending(Scanner):
        def prepare(self):
            prepared.set()
    def factory(*a, **kw):
        begun.set()
        assert release.wait(5)
        return Pending()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(background, shared, factory, config(), 'first_person', arrival)
        try:
            assert begun.wait(2)
            arrival.set()
            with pytest.raises(ExecutionYielded):
                pending.result(timeout=2)
        finally:
            release.set()
    broker = next(iter(shared._POOLS.values()))[0]
    broker.thread.join(2)
    assert not prepared.is_set() and not broker.healthy()
    live = shared.acquire_scanner(lambda *a, **kw: Scanner(), 'first_person', config(), 640, 4)
    live.close()


@pytest.mark.parametrize('change', ['disabled', 'capacity_one', 'nonshared'])
def test_cold_preparation_requires_enabled_reserved_shared_capacity(shared, change):
    settings = config()
    if change == 'disabled':
        settings['device_day']['backfill']['enabled'] = False
    elif change == 'capacity_one':
        settings['performance']['shared_inference_contexts_per_pool'] = 1
    else:
        settings['performance']['shared_inference_enabled'] = False
    with pytest.raises(ExecutionYielded):
        background(shared, lambda *a, **kw: pytest.fail('Unexpected cold creation'), settings)
    assert shared._POOLS == {}


def test_disabled_background_cannot_lease_even_a_ready_model(shared):
    settings = config()
    live = shared.acquire_scanner(lambda *a, **kw: Scanner(), 'first_person', settings, 640, 4)
    live.close()
    settings['device_day']['backfill']['enabled'] = False
    with pytest.raises(ExecutionYielded, match='disabled'):
        background(shared, lambda *a, **kw: pytest.fail('Unexpected load'), settings)


def test_four_historical_keys_rotate_only_unused_background_residents(shared):
    for role in ('fp-coarse', 'fp-fine', 'tp-coarse', 'tp-fine'):
        lease = background(shared, lambda *a, **kw: Scanner(), config(), role)
        lease.close()
        assert len(shared._POOLS) <= 3
    assert len(shared._POOLS) == 3


def test_retirement_closes_only_snapshot_while_live_adds_to_the_old_key(shared):
    closing, release = Event(), Event()
    calls = []
    class RetiringScanner(Scanner):
        def close(self):
            closing.set()
            assert release.wait(5)
    def factory(*a, **kw):
        calls.append(1)
        return RetiringScanner() if len(calls) == 1 else Scanner()
    for role in ('fp-coarse', 'fp-fine', 'tp-coarse'):
        lease = background(shared, factory, config(), role)
        lease.close()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(background, shared, factory, config(), 'tp-fine')
        live = None
        try:
            assert closing.wait(2)
            live = shared.acquire_scanner(factory, 'fp-coarse', config(), 640, 4)
            assert live.broker.healthy() and live.infer([1]) == [1]
            release.set()
            new_history = pending.result(timeout=2)
            new_history.close()
            assert live.broker.healthy() and live.infer([2]) == [2]
        finally:
            release.set()
            if live is not None:
                live.close()


def test_live_reuse_of_background_warm_preserves_quarantine_semantics(shared):
    old = background(shared, lambda *a, **kw: Scanner(), config())
    old.close()
    broker = old.broker
    assert not broker.pool.initialized
    live = shared.acquire_scanner(lambda *a, **kw: pytest.fail('Unexpected expansion'),
                                  'first_person', config(), 640, 4)
    assert live.broker is broker and broker.pool.initialized
    live.close()
    assert broker.close(timeout=1)
    with pytest.raises(RuntimeError, match='quarantined'):
        shared.acquire_scanner(lambda *a, **kw: pytest.fail('Replaced a quarantined live pool'),
                               'first_person', config(), 640, 4)


def test_live_new_key_cleanup_never_holds_lock_or_removes_later_live_owner(shared):
    closing, release = Event(), Event()
    calls = []
    class RetiringScanner(Scanner):
        def close(self):
            closing.set()
            assert release.wait(5)
    def factory(*a, **kw):
        calls.append(1)
        return RetiringScanner() if len(calls) == 1 else Scanner()
    for role in ('history-a', 'history-b', 'history-c'):
        lease = background(shared, factory, config(), role)
        lease.close()
    resident = shared.acquire_scanner(factory, 'live-resident', config(), 640, 4)
    resident_broker = resident.broker
    resident.close()
    assert len(shared._POOLS) == 4
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(shared.acquire_scanner, factory, 'live-new', config(), 640, 4)
        leases = []
        try:
            assert closing.wait(2)
            before = time.monotonic()
            resident = shared.acquire_scanner(factory, 'live-resident', config(), 640, 4)
            leases.append(resident)
            assert resident.broker is resident_broker
            assert time.monotonic() - before < 1 and resident.infer([1]) == [1]
            # Reusing the retiring old key can append a foreground context to
            # its shared list while the native background owner is closing.
            replacement = shared.acquire_scanner(factory, 'history-a', config(), 640, 4)
            leases.append(replacement)
            assert replacement.infer([2]) == [2]
            assert not pending.done()
            release.set()
            newest = pending.result(timeout=2)
            leases.append(newest)
            assert newest.infer([3]) == [3]
            assert replacement.broker.healthy() and replacement.infer([4]) == [4]
            assert resident.broker.healthy() and resident.infer([5]) == [5]
            assert len(shared._POOLS) == 4
        finally:
            release.set()
            for lease in leases:
                lease.close()


def test_foreground_used_pools_are_never_retired_to_prepare_history(shared):
    brokers = []
    for role in ('fp-coarse', 'fp-fine', 'tp-coarse'):
        lease = background(shared, lambda *a, **kw: Scanner(), config(), role)
        lease.close()
        live = shared.acquire_scanner(lambda *a, **kw: Scanner(), role, config(), 640, 4)
        brokers.append(live.broker)
        live.close()
    with pytest.raises(ExecutionYielded) as caught:
        background(shared, lambda *a, **kw: pytest.fail('Consumed the reserved live key'), config(), 'tp-fine')
    assert caught.value.reason == 'cold_warm_disabled_capacity'
    assert all(b.healthy() and not b.stop.is_set() for b in brokers)


def test_private_validation_does_not_hold_foreground_validation_mutex(default_config, monkeypatch):
    from copy import deepcopy
    from visioncortex.device_day_models import DeviceDayModels
    settings = deepcopy(default_config)
    settings['device_day']['backfill'] = {'enabled': True}
    settings['performance'].update(shared_inference_enabled=True, shared_inference_contexts_per_pool=2)
    backend = DeviceDayModels(settings)
    begun, release, arrival = Event(), Event(), Event()
    def validate(*a, check_boundary=None):
        check_boundary()
        begun.set()
        assert release.wait(5)
        check_boundary()
        return {'complete-private-report': True}
    monkeypatch.setattr('visioncortex.detection.validate_models', validate)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(backend.prepare_background_vision, arrival)
        try:
            assert begun.wait(2)
            assert backend._validation_lock.acquire(blocking=False)
            backend._validation_lock.release()
            arrival.set()
        finally:
            release.set()
        with pytest.raises(ExecutionYielded):
            pending.result(timeout=2)
    assert not backend.background_vision_ready()
    arrival.clear()
    assert backend.prepare_background_vision(arrival)
    assert backend._validated == {'complete-private-report': True}


def test_failed_cold_preparation_cools_down_without_quarantining_live(shared):
    calls = []
    def failure(*a, **kw):
        calls.append(1)
        raise ValueError('Synthetic initialization failure')
    with pytest.raises(ValueError):
        background(shared, failure, config())
    for pool in shared._POOLS.values():
        for broker in pool:
            broker.thread.join(2)
    with pytest.raises(ExecutionYielded) as caught:
        background(shared, failure, config())
    assert caught.value.reason == 'model_preparation_cooldown' and calls == [1]
    live = shared.acquire_scanner(lambda *a, **kw: Scanner(), 'first_person', config(), 640, 4)
    live.close()


def test_stop_drains_pending_initializer_without_holding_pool_lock(shared):
    begun, release = Event(), Event()
    def factory(*a, **kw):
        begun.set()
        assert release.wait(5)
        return Scanner()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(background, shared, factory, config())
        try:
            assert begun.wait(2)
            # Factory startup can precede the caller's short registration
            # section. Stop after that section, rather than racing its lock.
            with shared._LOCK:
                assert shared._BACKGROUND_WARM is not None
            assert not shared.stop_background_preparation(timeout=0)
            assert shared._LOCK.acquire(blocking=False)
            shared._LOCK.release()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='cancelled'):
            pending.result(timeout=2)
    assert shared.stop_background_preparation(timeout=2)


@pytest.mark.parametrize('change', ['disabled', 'capacity_one', 'nonshared'])
def test_private_validation_respects_cold_preparation_gates(default_config, monkeypatch, change):
    from copy import deepcopy
    from visioncortex.device_day_models import DeviceDayModels
    settings = deepcopy(default_config)
    settings['device_day']['backfill'] = {'enabled': change != 'disabled'}
    settings['performance'].update(shared_inference_enabled=change != 'nonshared',
                                   shared_inference_contexts_per_pool=1 if change == 'capacity_one' else 2)
    backend = DeviceDayModels(settings)
    monkeypatch.setattr('visioncortex.detection.validate_models', lambda *a, **kw: pytest.fail('Unexpected validation'))
    with pytest.raises(ExecutionYielded):
        backend.prepare_background_vision(Event())
    assert not backend.background_vision_ready()


def test_blocked_validation_boundary_never_owns_publication_mutex(default_config, monkeypatch):
    from copy import deepcopy
    from visioncortex.device_day_models import DeviceDayModels
    settings = deepcopy(default_config)
    settings['device_day']['backfill'] = {'enabled': True}
    settings['performance'].update(shared_inference_enabled=True, shared_inference_contexts_per_pool=2)
    backend = DeviceDayModels(settings)
    blocked, release = Event(), Event()
    class Signal:
        checks = 0
        def is_set(self, **kwargs):
            assert backend._validation_lock.acquire(blocking=False)
            backend._validation_lock.release()
            self.checks += 1
            if self.checks == 2:
                blocked.set()
                assert release.wait(5)
            return False
    monkeypatch.setattr('visioncortex.detection.validate_models', lambda *a, **kw: {'complete': True})
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(backend.prepare_background_vision, Signal())
        try:
            assert blocked.wait(2)
            assert backend._validation_lock.acquire(blocking=False)
            backend._validation_lock.release()
        finally:
            release.set()
        assert pending.result(timeout=2)
