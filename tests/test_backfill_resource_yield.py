"""Local lease and synthetic boundary evidence; no NAS, media or model access."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event
from threading import Timer
import hashlib
import sys
import time

import pytest

from visioncortex.runtime_control import (
    CURRENT, ExecutionContext, ExecutionYielded, ResourceCoordinator,
    check_cancelled, check_yield, defer_yield, execution_context, resource_slot,
)


def config(tmp_path, capacity=2):
    return {'storage': {'local_runtime_root': str(tmp_path)},
            'runtime': {'resource_limits': {'cloud': capacity}}}


def test_idle_backfill_never_enqueues_and_keeps_a_live_unit(tmp_path):
    coordinator = ResourceCoordinator(tmp_path/'resources.sqlite3')
    background = ExecutionContext(job_id='old', source='device_day_backfill', priority=100)
    live = ExecutionContext(job_id='new', source='nas', priority=-1)
    assert coordinator.try_claim('history', 'cloud', 1, 2, background)
    assert not coordinator.try_claim('history-too', 'cloud', 1, 2, background)
    assert {row['job'] for row in coordinator.snapshot()} == {'old'}
    assert coordinator.try_claim('live', 'cloud', 1, 2, live)
    assert not coordinator.try_claim('waiter', 'cloud', 1, 2, live)
    coordinator.release('history')
    before = coordinator.snapshot()
    assert not coordinator.try_claim('history-retry', 'cloud', 1, 2, background)
    assert coordinator.snapshot() == before
    assert coordinator.try_claim('waiter', 'cloud', 1, 2, live)
    coordinator.release('waiter')
    coordinator.release('live')
    assert coordinator.snapshot() == []


@pytest.mark.parametrize('capacity', [1, 2])
def test_busy_history_acquire_yields_immediately_and_releases_no_other_lease(tmp_path, capacity):
    coordinator = ResourceCoordinator(tmp_path/'resources.sqlite3')
    ctx = ExecutionContext(job_id='live', source='nas')
    assert coordinator.try_claim('holder', 'cloud', 1, capacity, ctx)
    before = coordinator.snapshot()
    with pytest.raises(ExecutionYielded), execution_context(source='device_day_backfill'):
        with coordinator.acquire('cloud', capacity=capacity, timeout=3600):
            pytest.fail('Busy background work must not start')
    assert coordinator.snapshot() == before
    coordinator.release('holder')


def test_yield_is_inherited_and_never_turns_an_inflight_unit_into_cancellation():
    arrival = Event()
    with execution_context(source='device_day_backfill', yield_signal=arrival):
        with execution_context(job_id='nested'):
            assert CURRENT.get().yield_signal is arrival
            arrival.set()
            check_cancelled()
            with defer_yield():
                check_yield()
                assert CURRENT.get().yield_signal is None
            persisted = []
            with pytest.raises(ExecutionYielded):
                check_yield(persist=lambda: persisted.append('sealed'))
            assert persisted == ['sealed']
    assert CURRENT.get().yield_signal is None
    with execution_context(yield_signal=lambda: True), pytest.raises(ExecutionYielded):
        check_yield()


def test_nested_paid_resource_reuses_only_its_own_thread_lease(tmp_path):
    settings = config(tmp_path)
    coordinator = ResourceCoordinator(tmp_path/'state/resources.sqlite3')
    with execution_context(source='device_day_backfill'), resource_slot(settings, 'cloud'):
        with resource_slot(settings, 'cloud'):
            assert len(coordinator.snapshot()) == 1
        context = copy_context()
        def other_thread():
            with pytest.raises(ExecutionYielded):
                with resource_slot(settings, 'cloud'):
                    pytest.fail('Copied context cannot reuse another thread lease')
        with ThreadPoolExecutor(1) as pool:
            pool.submit(context.run, other_thread).result(3)
        assert len(coordinator.snapshot()) == 1
    assert coordinator.snapshot() == []


def test_nonpaid_hash_yields_without_accepting_partial_content(tmp_path):
    from visioncortex.device_day_contract import file_hash
    from visioncortex.device_day_inputs import hash_content
    path = tmp_path/'owned-source.bin'
    content = b'x' * (10 * 1024 * 1024)
    path.write_bytes(content)
    checks = []
    def arrival():
        checks.append(True)
        return len(checks) > 1
    with execution_context(source='device_day_backfill', yield_signal=arrival), pytest.raises(ExecutionYielded):
        hash_content(path)
    assert path.read_bytes() == content
    assert hash_content(path) == file_hash(path) == hashlib.sha256(content).hexdigest()


def test_owned_nonpaid_probe_yields_and_reaps_its_child():
    from visioncortex.source_frames import _run
    arrival = Event()
    timer = Timer(.05, arrival.set)
    begun = time.monotonic()
    timer.start()
    try:
        with execution_context(source='device_day_backfill', yield_signal=arrival), pytest.raises(ExecutionYielded):
            _run([sys.executable, '-c', 'import time; time.sleep(20)'],
                 capture_output=True, check=True, timeout=30)
        assert time.monotonic() - begun < 3
    finally:
        timer.cancel()


def test_quantum_only_yields_at_new_durable_units(tmp_path):
    from visioncortex.device_day_contract import file_hash
    from visioncortex.device_day_inputs import hash_content
    from visioncortex.source_frames import _run
    class ExpiredQuantum:
        reason = 'quantum_elapsed'
        def is_set(self, *, ignore_quantum=False):
            return not ignore_quantum
    path = tmp_path/'owned-source.bin'
    content = b'x' * (10 * 1024 * 1024)
    path.write_bytes(content)
    with execution_context(source='device_day_backfill', yield_signal=ExpiredQuantum()):
        check_yield(checkpoint=False)
        assert hash_content(path) == file_hash(path) == hashlib.sha256(content).hexdigest()
        result = _run([sys.executable, '-c', 'print("owned synthetic probe")'],
                      capture_output=True, check=True, timeout=3)
        assert result.stdout.strip() == b'owned synthetic probe'
        with pytest.raises(ExecutionYielded) as caught:
            check_yield()
        assert caught.value.reason == 'quantum_elapsed'
