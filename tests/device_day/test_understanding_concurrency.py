import threading
from types import SimpleNamespace

import pytest

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker, understanding_capacity
from visioncortex.runtime_control import CURRENT


def test_understanding_capacity_honors_worker_stage_and_cloud_limits():
    assert understanding_capacity({'runtime': {'resource_limits': {'cloud': 4}}},
        {'understanding_workers': 8, 'stage_worker_limits': {'understanding': 3}}) == 3
    assert understanding_capacity({'runtime': {'resource_limits': {'cloud': 2}}},
        {'understanding_workers': 4}) == 2
    for value in (0, True, 1.5, 65):
        with pytest.raises(ValueError):
            understanding_capacity({}, {'understanding_workers': value})


def test_actual_concurrent_claims_keep_context_and_save_each_result_before_shutdown(tmp_path, monkeypatch):
    root = tmp_path/'runtime'
    settings = {'understanding_workers': 2, 'stage_worker_limits': {'understanding': 2}}
    config = {'runtime': {'local_only': True, 'resource_limits': {'cloud': 2}},
              'storage': {'local_runtime_root': str(root)}, 'device_day': settings}
    queue = DeviceDayQueue(root/'queue-understanding.sqlite3')
    runner = SimpleNamespace(runtime_root=root, queues={'understanding': queue}, settings=settings, config=config)
    records = [{'recording_id': name, 'source_signature': name, 'camera_key': name, 'configured_role': 'first_person',
                'recording_start_us': 1, 'recording_end_us': 2} for name in ('a', 'b', 'c')]
    worker = RetentionWorker(stage='understanding')
    monkeypatch.setattr(worker, 'candidates', lambda runner: records)
    def admit(runner, record):
        queue.enqueue(record, record['source_signature'])
        queue.grant_admission(record, cohort='live', reason='synthetic_concurrency_fixture')
        return True
    monkeypatch.setattr(worker, 'admit', admit)
    barrier = threading.Barrier(2, timeout=3)
    stop, lock = threading.Event(), threading.Lock()
    active, peak, saved = 0, 0, []
    def process(record, **kwargs):
        nonlocal active, peak
        assert CURRENT.get().job_id == record['recording_id']
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        # Already admitted responses finish saving even if shutdown arrives.
        stop.set()
        with lock:
            saved.append(record['recording_id'])
            active -= 1
        return {'status': 'completed', 'recording_id': record['recording_id']}
    runner.process = process
    result = worker.tick(runner, stop)
    assert result['status'] == 'processed' and peak == 2
    assert len(saved) == 2
    with queue.connect() as db:
        rows = list(db.execute('SELECT status,COUNT(*) AS n FROM recordings GROUP BY status'))
    assert {r['status']: r['n'] for r in rows} == {'completed': 2, 'queued': 1}
    assert worker.tick(runner, stop)['status'] == 'stopping'
