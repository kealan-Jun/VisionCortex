"""Continuous live demand cannot consume every historical scheduling turn."""
from threading import Event
from types import SimpleNamespace

import pytest

from test_device_day_backfill_policy import LocalState, NOW, recording
from test_device_day_backfill import runner as runner_fixture
from visioncortex import device_day_backfill as backfill
from visioncortex.device_day_backfill_policy import historical_candidate_ids, live_demand
from visioncortex.runtime_control import ExecutionContext, ResourceCoordinator, CURRENT, execution_context

runner = runner_fixture


def test_fair_history_keeps_live_demand_visible_without_an_idle_veto(tmp_path):
    state = LocalState(tmp_path / 'runtime')
    state.config['device_day']['backfill'] = {'mode': 'fair'}
    state.observe(recording('live'), recording('old', age=1000), recording('older', age=2000))
    state.resource('vision', state='running')
    result = live_demand(state.config, 'vision', NOW)
    assert result['status'] == 'fair_capacity' and not result['blocked']
    assert result['counts']['eligible'] and result['counts']['resource_running']
    assert historical_candidate_ids(state.config, 'vision', now=NOW) == ['older', 'old']
    state.monitor(age=31)
    assert live_demand(state.config, 'vision', NOW)['blocked']


def test_fair_history_respects_audited_extra_budget_and_old_schema(tmp_path):
    state = LocalState(tmp_path / 'runtime')
    state.config['device_day']['backfill'] = {'mode': 'fair'}
    row = recording('old', age=1000)
    state.observe(row)
    state.queue('vision', row, status='failed', attempts=3)
    assert historical_candidate_ids(state.config, 'vision', now=NOW) == []
    state.write(state.base / 'queue-vision.sqlite3', 'ALTER TABLE recordings ADD COLUMN retry_limit INTEGER DEFAULT 0')
    state.write(state.base / 'queue-vision.sqlite3', 'UPDATE recordings SET retry_limit=6')
    assert historical_candidate_ids(state.config, 'vision', now=NOW) == ['old']
    state.write(state.base / 'queue-vision.sqlite3', 'UPDATE recordings SET attempts=6')
    assert historical_candidate_ids(state.config, 'vision', now=NOW) == []


def test_fair_resource_waiter_ages_and_never_exceeds_capacity(tmp_path):
    coordinator = ResourceCoordinator(tmp_path / 'resources.sqlite3')
    live = ExecutionContext(source='nas', priority=1)
    history = ExecutionContext(source='device_day_backfill', priority=2, background_fair=True)
    assert coordinator.try_claim('live', 'vision', 1, 1, live, now=100)
    assert not coordinator.try_claim('history', 'vision', 1, 1, history, now=100)
    assert not coordinator.try_claim('next-live', 'vision', 1, 1, live, now=140)
    coordinator.release('live')
    assert not coordinator.try_claim('next-live', 'vision', 1, 1, live, now=140)
    assert coordinator.try_claim('history', 'vision', 1, 1, history, now=140)
    assert not coordinator.try_claim('next-live', 'vision', 1, 1, live, now=140)
    with execution_context(source='device_day_backfill', background_fair=True):
        with execution_context(job_id='nested'):
            assert CURRENT.get().background_fair
    assert not CURRENT.get().background_fair


def test_fair_turn_rotates_stages_and_throttles_before_another_turn(runner, monkeypatch):
    runner.settings['backfill'].update(mode='fair', stages=['vision', 'report'], fair_interval_seconds=60)
    clock = [100.0]
    monkeypatch.setattr(backfill.time, 'monotonic', lambda: clock[0])
    old = backfill.historical_candidates
    monkeypatch.setattr(backfill, 'historical_candidates', lambda *a, **kw: runner.rows)
    seen = []
    runner.process = lambda record, **kw: seen.append((kw['stage'], CURRENT.get().background_fair)) or {'status': 'completed'}
    vision, report = backfill.BackfillWorker('vision'), backfill.BackfillWorker('report')
    assert vision.tick(runner, Event())['status'] == 'processed'
    assert vision.tick(runner, Event())['status'] == 'waiting_for_fair_interval'
    # Skip unrelated prerequisite-byte validation only at this mocked port;
    # downstream receipt/hash acceptance has its own integration tests.
    report.admitter.admit = lambda owner, row: owner.queues['report'].enqueue(row, 'report-key') or True
    assert report.tick(runner, Event())['status'] == 'processed'
    assert seen == [('vision', True), ('report', True)]
    clock[0] = 161
    assert report.tick(runner, Event())['status'] == 'waiting_for_history_turn'
    monkeypatch.setattr(backfill, 'historical_candidates', old)


def test_fair_supervisor_does_not_occupy_live_loop_while_interval_waits(runner, monkeypatch):
    runner.settings['backfill'].update(mode='fair')
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(backfill.time, 'monotonic', lambda: clock.now)
    supervisor = backfill.BackfillSupervisor('vision', Event())
    supervisor.generation = id(runner)
    supervisor.worker.next_fair_at = 160
    try:
        assert supervisor.poll(runner, allow_start=True)['status'] == 'waiting_for_fair_interval'
        assert supervisor.future is None
    finally:
        supervisor.close()


def test_missing_stage_cannot_hold_every_other_history_turn(runner, monkeypatch):
    runner.settings['backfill'].update(mode='fair', stages=['vision', 'report'], fair_turn_timeout_seconds=15)
    clock = [100.0]
    monkeypatch.setattr(backfill.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(backfill.time, 'time', lambda: clock[0])
    monkeypatch.setattr(backfill, 'historical_candidates', lambda *a, **kw: runner.rows)
    report = backfill.BackfillWorker('report')
    report.admitter.admit = lambda owner, row: owner.queues['report'].enqueue(row, 'report-key') or True
    assert report.tick(runner, Event())['status'] == 'waiting_for_history_turn'
    clock[0] = 114
    assert report.tick(runner, Event())['status'] == 'waiting_for_history_turn'
    clock[0] = 115
    assert report.tick(runner, Event())['status'] == 'processed'


@pytest.mark.parametrize('value', ['unbounded', None, 1, []])
def test_unknown_fair_mode_is_rejected(value):
    with pytest.raises(ValueError, match='mode'):
        backfill.options({'backfill': {'mode': value}})
