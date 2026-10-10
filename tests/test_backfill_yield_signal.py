"""Bounded boundary probing; no NAS, model or service calls are performed."""
from threading import Event
from types import SimpleNamespace

import pytest

from visioncortex import device_day_backfill as module


@pytest.fixture
def probe(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    state = {'blocked': False, 'status': 'idle', 'reads': 0}
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)

    def demand(*args):
        state['reads'] += 1
        return {'blocked': state['blocked'], 'status': state['status']}

    monkeypatch.setattr(module, 'live_demand', demand)
    stop = Event()
    signal = module.YieldSignal({}, 'vision', stop, 30)
    return SimpleNamespace(clock=clock, state=state, stop=stop, signal=signal)


def test_high_frequency_boundaries_read_metadata_once_per_signal(probe):
    for index in range(1000):
        probe.clock.now = 100.0 + index / 100000
        assert not probe.signal.is_set()
    assert probe.state['reads'] == 1 and probe.signal.reason is None
    second = module.YieldSignal({}, 'vision', probe.stop, 30)
    assert not second.is_set()
    assert probe.state['reads'] == 2


def test_live_arrival_is_seen_at_the_100ms_bound(probe):
    assert not probe.signal.is_set()
    probe.state.update(blocked=True, status='waiting_for_live')
    probe.clock.now = 100.099
    assert not probe.signal.is_set()
    assert probe.state['reads'] == 1
    probe.clock.now = 100.1
    assert probe.signal.is_set()
    assert probe.signal.reason == 'waiting_for_live' and probe.state['reads'] == 2


def test_slow_probe_does_not_extend_the_100ms_observation_window(probe, monkeypatch):
    def slow_demand(*args):
        probe.state['reads'] += 1
        probe.clock.now += .101
        return {'blocked': False, 'status': 'idle'}
    monkeypatch.setattr(module, 'live_demand', slow_demand)
    assert not probe.signal.is_set()
    assert not probe.signal.is_set()
    assert probe.state['reads'] == 2


@pytest.mark.parametrize('ignore_quantum', [False, True])
def test_stop_does_not_wait_for_idle_probe_interval(probe, ignore_quantum):
    assert not probe.signal.is_set()
    probe.stop.set()
    assert probe.signal.is_set(ignore_quantum=ignore_quantum)
    assert probe.signal.reason == 'stopping' and probe.state['reads'] == 1


def test_quantum_is_immediate_and_ignoring_it_still_checks_live(probe):
    signal = module.YieldSignal({}, 'vision', probe.stop, .05)
    assert not signal.is_set()
    probe.clock.now = 100.05
    assert signal.is_set()
    assert signal.reason == 'quantum_elapsed' and probe.state['reads'] == 1
    assert not signal.is_set(ignore_quantum=True)
    assert signal.reason is None and probe.state['reads'] == 1
    probe.state.update(blocked=True, status='waiting_for_monitor')
    probe.clock.now = 100.1
    assert signal.is_set(ignore_quantum=True)
    assert signal.reason == 'waiting_for_monitor' and probe.state['reads'] == 2


def test_clock_rollback_cannot_reuse_the_idle_probe(probe):
    assert not probe.signal.is_set()
    probe.clock.now = 99.999
    probe.state.update(blocked=True, status='waiting_for_live')
    assert probe.signal.is_set()
    assert probe.signal.reason == 'waiting_for_live' and probe.state['reads'] == 2


def test_real_blocked_reason_stays_sticky_with_immediate_stop_checks(probe):
    reads = []
    probe.signal.stop = SimpleNamespace(is_set=lambda: reads.append(True) or False)
    probe.state.update(blocked=True, status='status_unavailable')
    assert probe.signal.is_set()
    probe.state.update(blocked=False, status='idle')
    probe.clock.now = 200.0
    for _ in range(10):
        assert probe.signal.is_set(ignore_quantum=True)
        assert probe.signal.reason == 'status_unavailable'
    assert len(reads) == 11 and probe.state['reads'] == 1


def test_failed_probe_does_not_authorize_reuse_of_an_old_idle_result(probe, monkeypatch):
    assert not probe.signal.is_set()
    probe.clock.now = 100.1
    monkeypatch.setattr(module, 'live_demand', lambda *a: (_ for _ in ()).throw(ValueError('Unreadable monitor')))
    with pytest.raises(ValueError, match='Unreadable monitor'):
        probe.signal.is_set()
    probe.clock.now = 100.05
    monkeypatch.setattr(module, 'live_demand', lambda *a: {'blocked': True, 'status': 'waiting_for_monitor'})
    assert probe.signal.is_set() and probe.signal.reason == 'waiting_for_monitor'
