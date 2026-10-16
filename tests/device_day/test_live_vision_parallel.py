from copy import deepcopy
from pathlib import Path
from threading import Barrier, Event, Lock
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.runtime_control import CURRENT, ExecutionYielded, ResourceCoordinator, ExecutionContext, execution_context


def runner_for(tmp_path, default_config, process, *, capacity=2, jobs=2):
    config = deepcopy(default_config)
    config['storage']['local_runtime_root'] = str(tmp_path / 'Runtime')
    config['runtime']['resource_limits']['vision'] = capacity
    config['device_day']['live_vision_jobs'] = jobs
    config['device_day']['stage_worker_limits'] = {'vision': 2}
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    queue = DeviceDayQueue(root / 'queue-vision.sqlite3', latest_first=True)
    records = [dict(recording_id=rid, camera_key=camera, recording_start_us=stamp,
                    source_signature=rid, configured_role='first_person', processable=True)
               for rid, camera, stamp in [('older', 'a', 100), ('newest', 'a', 400),
                                          ('second', 'b', 300), ('third', 'c', 200), ('fourth', 'd', 150)]]
    worker = RetentionWorker(stage='vision')
    worker.candidates = lambda runner: records

    def admit(runner, record):
        queue.enqueue(record, record['source_signature'])
        return True

    worker.admit = admit
    runner = SimpleNamespace(config=config, settings=config['device_day'], runtime_root=root,
                             queues={'vision': queue}, process=process)
    return worker, runner, queue


def statuses(queue):
    with queue.connect() as db:
        return {row['recording_id']: dict(row) for row in db.execute('SELECT * FROM recordings')}


def test_two_cameras_overlap_with_independent_contexts_and_latest_claims(tmp_path, default_config):
    barrier = Barrier(2)
    lock = Lock()
    active = set()
    observed = []

    def process(record, **kwargs):
        assert CURRENT.get().job_id == record['recording_id']
        assert CURRENT.get().yield_signal is stop and CURRENT.get().stop is None
        with lock:
            active.add(record['camera_key'])
            observed.append(set(active))
        barrier.wait(timeout=5)
        with lock:
            active.remove(record['camera_key'])
        return {'status': 'completed'}

    stop = Event()
    worker, runner, queue = runner_for(tmp_path, default_config, process)
    result = worker.tick(runner, stop)
    assert result['status'] == 'processed'
    assert {r['recording_id'] for r in result['results']} == {'newest', 'second'}
    assert max(map(len, observed)) == 2
    assert result['result']['status'] == 'completed'
    saved = statuses(queue)
    assert saved['newest']['status'] == saved['second']['status'] == 'completed'
    assert saved['older']['status'] == saved['third']['status'] == 'queued'
    assert all(saved[r]['lease_owner'] is None for r in saved)


@pytest.mark.parametrize('capacity,jobs', [(1, 2), (2, 1)])
def test_capacity_and_default_single_job_path_remain_bounded(tmp_path, default_config, capacity, jobs):
    calls = []

    def process(record, **kwargs):
        calls.append(record['recording_id'])
        return {'status': 'completed'}

    worker, runner, queue = runner_for(tmp_path, default_config, process, capacity=capacity, jobs=jobs)
    result = worker.tick(runner, Event())
    assert calls == ['newest']
    assert result['recording_id'] == 'newest'
    assert 'results' not in result
    assert statuses(queue)['second']['status'] == 'queued'


def test_shutdown_preserves_both_claims_and_attempt_budgets(tmp_path, default_config):
    barrier = Barrier(2)
    stop = Event()

    def process(record, **kwargs):
        barrier.wait(timeout=5)
        stop.set()
        raise ExecutionYielded('graceful stop', reason='stopping')

    worker, runner, queue = runner_for(tmp_path, default_config, process)
    result = worker.tick(runner, stop)
    assert result['result']['status'] == 'partial'
    saved = statuses(queue)
    for rid in ('newest', 'second'):
        assert saved[rid]['status'] == 'queued'
        assert saved[rid]['attempts'] == 0
        assert saved[rid]['lease_owner'] is None
    assert worker.deferred == {}


def test_one_camera_failure_does_not_discard_other_completion(tmp_path, default_config):
    barrier = Barrier(2)

    def process(record, **kwargs):
        barrier.wait(timeout=5)
        if record['camera_key'] == 'a':
            raise ValueError('input verification failed')
        return {'status': 'completed'}

    worker, runner, queue = runner_for(tmp_path, default_config, process)
    result = worker.tick(runner, Event())
    assert result['result']['status'] == 'partial'
    saved = statuses(queue)
    assert saved['newest']['status'] == 'failed'
    assert saved['second']['status'] == 'completed'
    assert saved['older']['status'] == 'queued'


def test_four_cameras_overlap_and_retain_individual_leases(tmp_path, default_config):
    barrier = Barrier(4)
    lock = Lock()
    active, observed = set(), []

    def process(record, **kwargs):
        assert CURRENT.get().job_id == record['recording_id']
        with lock:
            active.add(record['camera_key'])
            observed.append(set(active))
        barrier.wait(timeout=5)
        with lock:
            active.remove(record['camera_key'])
        return {'status': 'completed'}

    worker, runner, queue = runner_for(tmp_path, default_config, process, capacity=4, jobs=4)
    runner.settings['stage_worker_limits']['vision'] = 4
    result = worker.tick(runner, Event())
    assert {r['recording_id'] for r in result['results']} == {'newest', 'second', 'third', 'fourth'}
    assert max(map(len, observed)) == 4
    saved = statuses(queue)
    assert saved['older']['status'] == 'queued'
    assert sum(r['status'] == 'completed' for r in saved.values()) == 4
    assert all(r['lease_owner'] is None for r in saved.values())


@pytest.mark.parametrize('lanes', [2, 4])
def test_live_io_lanes_run_without_using_archive_slot(tmp_path, default_config, lanes):
    from visioncortex.device_day_activity import job
    from visioncortex.device_day_io import live_vision_lane, slot
    config = deepcopy(default_config)
    config['storage']['local_runtime_root'] = str(tmp_path)
    config['device_day'].update(live_vision_jobs=lanes, nas_io_slots=3, stage_worker_limits={'vision': lanes})
    config['runtime']['resource_limits']['vision'] = lanes
    coordinator = ResourceCoordinator(tmp_path / 'state/resources.sqlite3')
    ready, finish = Barrier(lanes + 1), Event()

    def live(rid):
        with execution_context(job_id=rid), job('vision', rid), live_vision_lane(), slot(config):
            ready.wait(timeout=5)
            assert finish.wait(5)

    with ThreadPoolExecutor(max_workers=lanes) as pool:
        futures = [pool.submit(live, str(i)) for i in range(lanes)]
        try:
            ready.wait(timeout=5)
            leases = coordinator.snapshot()
            assert sum(row['state'] == 'running' and row['resource'] == 'nas-live-vision' for row in leases) == lanes
            assert not coordinator.try_claim('third', 'nas-live-vision', 1, lanes, ExecutionContext())
            coordinator.release('third')
            with slot(config, copy=True):
                assert any(row['state'] == 'running' and row['resource'] == 'nas-io' for row in coordinator.snapshot())
        finally:
            finish.set()
        for future in futures:
            future.result(timeout=5)
    assert coordinator.snapshot() == []
