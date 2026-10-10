"""Live/background scheduling characterization without NAS or real models."""
import threading
import time
from types import SimpleNamespace

import pytest

from visioncortex import device_day_backfill as module
from visioncortex.device_day import exclusive
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.observed_inventory import observe
from visioncortex.runtime_control import CURRENT, check_yield


def record(rid, *, recent=False):
    start = int((time.time() - (60 if recent else 86400)) * 1_000_000)
    return {'recording_id': rid, 'camera_key': 'camera', 'configured_role': 'first_person',
            'recording_start_us': start, 'recording_end_us': start + 1_000_000,
            'processable': True, 'source_signature': rid}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    root = tmp_path / 'device-day'
    settings = {'enabled': True, 'inplace_preprocessing': True, 'latest_first': True, 'camera_lanes': True,
                'backfill': {'enabled': True, 'idle_seconds': 0}, 'failure_retry_limit': 3}
    config = {'storage': {'local_runtime_root': str(tmp_path)}, 'device_day': settings,
              'collection_ingest': {}}
    queues = {stage: DeviceDayQueue(root / f'queue-{stage}.sqlite3', latest_first=True)
              for stage in ('retention', 'vision', 'stt', 'understanding', 'report')}
    runner = SimpleNamespace(config=config, settings=settings, runtime_root=root, queues=queues)
    runner.process = lambda item, **kw: {'status': 'completed'}
    rows = [record('history')]
    observe(root, {'recordings': rows})
    monkeypatch.setattr(module, 'live_demand', lambda *a: {'blocked': False, 'status': 'idle', 'reasons': []})
    def candidates(*args, exclude=(), **kwargs):
        with queues['vision'].connect() as db:
            completed = {r[0] for r in db.execute("SELECT recording_id FROM recordings WHERE status='completed'")}
        return [r for r in rows if r['recording_id'] not in completed | set(exclude)]
    monkeypatch.setattr(module, 'historical_candidates', candidates)
    def admit(runner, item, stage):
        queues[stage].enqueue(item, item['source_signature'])
        return True
    monkeypatch.setattr('visioncortex.device_day_retention_worker.enqueue', admit)
    monkeypatch.setattr('visioncortex.device_day_inplace.active', lambda *a, **kw: True)
    runner.rows = rows
    return runner


def test_history_is_opt_in_and_never_uses_live_stage_lock(runner):
    worker = module.BackfillWorker('vision')
    runner.settings['backfill']['enabled'] = False
    assert worker.tick(runner, threading.Event())['status'] == 'disabled'
    runner.settings['backfill']['enabled'] = True
    with exclusive(runner.runtime_root / 'locks' / 'vision-worker.lock'):
        assert worker.tick(runner, threading.Event())['result']['status'] == 'completed'


def test_global_history_lock_does_not_change_queue(runner):
    with exclusive(runner.runtime_root / 'locks' / 'backfill.lock'):
        assert module.BackfillWorker('vision').tick(runner, threading.Event())['status'] == 'history_running_elsewhere'
    with runner.queues['vision'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0] == 0


def test_idle_drain_continues_without_per_round_total_limit(runner):
    worker = module.BackfillWorker('vision')
    runner.rows.extend(record(str(i)) for i in range(4))
    seen = [worker.tick(runner, threading.Event())['recording_id'] for _ in range(5)]
    assert len(set(seen)) == 5
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'


def test_arrival_yields_and_refunds_only_current_attempt(runner, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    queue = runner.queues['vision']
    item = runner.rows[0]
    queue.enqueue(item, item['source_signature'])
    with queue.connect() as db:
        db.execute('UPDATE recordings SET attempts=2')
    demand = {'blocked': False, 'status': 'idle', 'reasons': []}
    monkeypatch.setattr(module, 'live_demand', lambda *a: demand)
    saved = []
    def process(item, **kw):
        assert CURRENT.get().source == 'device_day_backfill'
        saved.append('durable-unit')
        demand.update(blocked=True, status='waiting_for_live')
        clock[0] += .1
        check_yield()
    runner.process = process
    result = module.BackfillWorker('vision').tick(runner, threading.Event())
    assert result['result']['status'] == 'paused_for_live'
    assert saved == ['durable-unit']
    with queue.connect() as db:
        row = db.execute('SELECT status,attempts,lease_owner FROM recordings').fetchone()
    assert tuple(row) == ('queued', 2, None)
    assert CURRENT.get().source != 'device_day_backfill'


def test_invalid_front_candidates_are_deferred_without_starving_other_work(runner):
    runner.rows.insert(0, record('invalid'))
    worker = module.BackfillWorker('vision')
    original = worker.admitter.admit
    worker.admitter.admit = lambda r, item: False if item['recording_id'] == 'invalid' else original(r, item)
    assert worker.tick(runner, threading.Event())['recording_id'] == 'history'
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'


def test_live_tick_completes_while_history_is_in_a_long_unit(runner):
    entered, release = threading.Event(), threading.Event()
    def process(item, **kw):
        if item['recording_id'] == 'history':
            entered.set()
            assert release.wait(5)
            check_yield()
        return {'status': 'completed'}
    runner.process = process
    supervisor = module.BackfillSupervisor('vision', threading.Event())
    try:
        supervisor.poll(runner, allow_start=True)
        assert entered.wait(2)
        latest = record('latest', recent=True)
        observe(runner.runtime_root, {'recordings': [latest]})
        result = RetentionWorker(stage='vision', recent_seconds=14400).tick(runner, threading.Event())
        assert result['recording_id'] == 'latest'
        assert result['result']['status'] == 'completed'
        assert not supervisor.future.done()
        supervisor.poll(runner, allow_start=False)
        release.set()
        assert supervisor.future.result(timeout=2)['result']['status'] == 'paused_for_live'
    finally:
        release.set()
        supervisor.close()


def test_history_reuses_runner_and_never_submits_a_second_concurrent_unit(runner):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def process(item, **kw):
        calls.append(item['recording_id'])
        entered.set()
        assert release.wait(5)
        return {'status': 'completed'}
    runner.process = process
    supervisor = module.BackfillSupervisor('vision', threading.Event())
    try:
        for _ in range(5):
            supervisor.poll(runner, allow_start=True)
        assert entered.wait(2)
        assert calls == ['history']
        release.set()
        assert supervisor.future.result(timeout=2)['result']['status'] == 'completed'
    finally:
        release.set()
        supervisor.close()


def test_quantum_does_not_restart_an_uncommitted_large_hash(runner, monkeypatch):
    clock = [0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    signal = module.YieldSignal(runner.config, 'vision', threading.Event(), 30)
    clock[0] = 31
    assert signal.is_set()
    assert signal.reason == 'quantum_elapsed'
    assert not signal.is_set(ignore_quantum=True)
    demand = {'blocked': True, 'status': 'waiting_for_live'}
    monkeypatch.setattr(module, 'live_demand', lambda *a: demand)
    clock[0] += .1
    assert signal.is_set(ignore_quantum=True)
    demand['blocked'] = False
    assert signal.is_set(ignore_quantum=True)


def test_cold_pool_wait_is_visible_without_failure_or_hot_retry(runner):
    calls = []
    def process(item, **kw):
        calls.append(item['recording_id'])
        return {'status': 'paused_for_live', 'reason': 'waiting_for_warm_model'}
    runner.process = process
    worker = module.BackfillWorker('vision')
    result = worker.tick(runner, threading.Event())
    assert result['reason'] == 'waiting_for_warm_model'
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'
    assert calls == ['history']
    with runner.queues['vision'].connect() as db:
        assert tuple(db.execute('SELECT status,attempts FROM recordings').fetchone()) == ('queued', 0)


def test_cold_validation_is_rejected_before_media_admission(runner):
    runner._backend = lambda: SimpleNamespace(background_vision_ready=lambda: False)
    worker = module.BackfillWorker('vision')
    worker.admitter.admit = lambda *a: (_ for _ in ()).throw(AssertionError('must not read media'))
    assert worker.tick(runner, threading.Event())['status'] == 'waiting_for_warm_model'
    with runner.queues['vision'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0] == 0


def test_idle_cold_validation_prepares_without_waiting_for_a_new_capture(runner):
    prepared = []
    backend = SimpleNamespace(background_vision_ready=lambda: bool(prepared))
    def prepare(signal):
        assert CURRENT.get().source == 'device_day_backfill'
        assert CURRENT.get().yield_signal is signal
        prepared.append(True)
        return True
    backend.prepare_background_vision = prepare
    runner._backend = lambda: backend
    result = module.BackfillWorker('vision').tick(runner, threading.Event())
    assert prepared == [True]
    assert result['result']['status'] == 'completed'


def test_empty_history_does_not_prepare_models_or_hold_understanding_for_preparation(runner):
    runner.rows.clear()
    runner._backend = lambda: (_ for _ in ()).throw(AssertionError('No model preparation without work'))
    assert module.BackfillWorker('vision').tick(runner, threading.Event()) == {'status': 'waiting', 'admitted': 0}


def test_live_arrival_during_cold_validation_prevents_media_admission(runner, monkeypatch):
    from visioncortex.runtime_control import ExecutionYielded
    backend = SimpleNamespace(background_vision_ready=lambda: False)
    def prepare(signal):
        monkeypatch.setattr(module, 'live_demand', lambda *a: {'blocked': True, 'status': 'waiting_for_live'})
        assert signal.is_set(ignore_quantum=True)
        raise ExecutionYielded('Latest arrival', reason=signal.reason)
    backend.prepare_background_vision = prepare
    runner._backend = lambda: backend
    worker = module.BackfillWorker('vision')
    worker.admitter.admit = lambda *a: (_ for _ in ()).throw(AssertionError('must not read media'))
    assert worker.tick(runner, threading.Event())['reason'] == 'waiting_for_live'
    with runner.queues['vision'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0] == 0


def test_real_adapter_cold_background_never_deserializes_models(default_config, monkeypatch):
    from visioncortex.device_day_models import DeviceDayModels
    from visioncortex.runtime_control import ExecutionYielded, execution_context
    backend = DeviceDayModels(default_config)
    monkeypatch.setattr('visioncortex.detection.validate_models', lambda *a: (_ for _ in ()).throw(AssertionError('cold load')))
    with execution_context(source='device_day_backfill'):
        with pytest.raises(ExecutionYielded) as error:
            backend._vision(None, None, None, None)
    assert error.value.reason == 'waiting_for_warm_model'
    assert not backend.background_vision_ready()


@pytest.mark.parametrize('policy', [[], {'enabled': 'true'}, {'stages': ['report']},
                                  {'stages': ['vision', 'vision']}, {'quantum_seconds': float('nan')}])
def test_invalid_backfill_policy_fails_closed(policy):
    with pytest.raises(ValueError):
        module.options({'backfill': policy})
