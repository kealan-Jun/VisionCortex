"""Local paid-port mocks prove recent handover keeps reusable window receipts."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import pytest

from test_scene_requests import Analyzer, response, run, scene, today  # noqa: F401
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.multimodal_usage import UsageLedger
from visioncortex.mllm import ArkAnalyzer
from visioncortex.runtime_control import CURRENT, ResourceCoordinator, execution_context, resource_slot


def worker_for(config, stage, process):
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    queue = DeviceDayQueue(root / f'queue-{stage}.sqlite3')
    record = {'recording_id': 'owned', 'camera_key': 'cam', 'configured_role': 'first_person',
              'source_signature': 'unchanged', 'recording_start_us': 1789005600000000, 'processable': True}
    worker = RetentionWorker(stage=stage)
    worker.candidates = lambda runner: [record]
    def admit(runner, item):
        queue.enqueue(item, 'same-recipe')
        return True
    worker.admit = admit
    runner = SimpleNamespace(runtime_root=root, queues={stage: queue}, config=config,
                             settings={'failure_retry_limit': 3}, process=process)
    return worker, runner, queue


def state(queue):
    with queue.connect() as db:
        return dict(db.execute('SELECT * FROM recordings').fetchone())


def test_recent_paid_window_finishes_cache_and_usage_before_handover_then_reuses(scene):  # noqa: F811
    config, metadata, paths, folder = scene
    config['runtime']['resource_limits'] = {'cloud': 1}
    stop = Event()
    coordinator = ResourceCoordinator(Path(config['storage']['local_runtime_root']) / 'state/resources.sqlite3')
    class StoppingAnalyzer(Analyzer):
        def _call(self, *args, **kwargs):
            assert CURRENT.get().stop is None and CURRENT.get().yield_signal is None
            with resource_slot(config, 'cloud'):
                assert len(coordinator.snapshot()) == 1
                if self.calls == 0:
                    stop.set()
                return super()._call(*args, **kwargs)
    analyzer = StoppingAnalyzer()
    second = (config, deepcopy(metadata) | {'segment_id': 'second'}, paths, folder / 'Second')
    starts = []
    def process(record, **kwargs):
        assert CURRENT.get().job_id == record['recording_id']
        assert CURRENT.get().source == 'nas' and CURRENT.get().priority == 3
        for index, window in enumerate((scene, second)):
            starts.append(index)
            run(window, analyzer)
        return {'status': 'completed'}
    worker, runner, queue = worker_for(config, 'understanding', process)
    result = worker.tick(runner, stop)
    assert result['result']['status'] == 'paused_for_live' and result['result']['reason'] == 'stopping'
    assert starts == [0] and analyzer.calls == 1
    assert read_json(folder / 'Result.json')['model_result']['status'] == 'completed'
    stats = UsageLedger(config).summary(today())
    assert stats['pending_calls'] == 0 and stats['known_tokens']['total_tokens'] == 30
    assert state(queue)['status'] == 'queued' and state(queue)['attempts'] == 0
    assert state(queue)['lease_owner'] is None and worker.deferred == {}
    assert coordinator.snapshot() == []
    stop.clear()
    result = worker.tick(runner, stop)
    assert result['result']['status'] == 'completed'
    assert starts == [0, 0, 1] and analyzer.calls == 2, 'the saved first window cannot be billed again'
    assert state(queue)['status'] == 'completed'


def test_recent_failed_paid_attempt_internal_retry_is_sealed_before_yield(scene, monkeypatch):  # noqa: F811
    config, _, _, folder = scene
    config['runtime']['resource_limits'] = {'cloud': 1}
    stop = Event()
    coordinator = ResourceCoordinator(Path(config['storage']['local_runtime_root']) / 'state/resources.sqlite3')
    config['mllm'].update(enabled=True, compact_scene_metadata=False, max_retries=2)
    monkeypatch.setattr('visioncortex.mllm.model_api_key', lambda *a, **kw: 'synthetic-only')
    monkeypatch.setattr('visioncortex.mllm.time.sleep', lambda seconds: None)
    requests = []
    def handler(request):
        import json
        requests.append(request)
        assert len(coordinator.snapshot()) == 1, 'internal retry reuses only this thread cloud lease'
        assert CURRENT.get().stop is None and CURRENT.get().yield_signal is None
        if len(requests) == 1:
            stop.set()
            raise httpx.ReadError('synthetic transport failure', request=request)
        body = json.loads(request.content)
        supplied = json.loads(body['messages'][1]['content'][0]['text'])
        payload = {'id': 'offline-request', 'choices': [{'message': {'content': json.dumps(response(supplied))},
                   'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}}
        return httpx.Response(200, json=payload)
    analyzer = ArkAnalyzer(config)
    analyzer.client.close()
    analyzer.client = httpx.Client(transport=httpx.MockTransport(handler))
    worker, runner, queue = worker_for(config, 'understanding', lambda *a, **kw: run(scene, analyzer))
    result = worker.tick(runner, stop)
    assert result['result']['status'] == 'paused_for_live'
    raw = read_json(folder / 'Result.json')['model_result']
    assert raw['attempts'] == 2 and raw['attempt_receipts'][0]['status'] == 'failed'
    assert raw['usage']['unknown_attempt_count'] == 1, 'unknown billed attempt usage cannot become zero'
    assert UsageLedger(config).summary(today())['pending_calls'] == 0
    assert state(queue)['status'] == 'queued' and state(queue)['attempts'] == 0
    assert coordinator.snapshot() == []
    stop.clear()
    # Exercise the same owner again, returning normal stage completion after
    # validating/reusing the failed+successful attempt receipt from its cache.
    runner.process = lambda *a, **kw: (run(scene, analyzer), {'status': 'completed'})[1]
    assert worker.tick(runner, stop)['result']['status'] == 'completed'
    assert len(requests) == 2, 'restart must reuse the entire saved attempt chain without a new request'
    analyzer.close()


@pytest.mark.parametrize('stage', ['vision', 'understanding'])
def test_runner_returned_pause_refunds_only_claim_and_keeps_failure_budget(tmp_path, stage):
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    stop = Event()
    def process(*args, **kwargs):
        stop.set()
        return {'status': 'paused_for_live', 'reason': 'live_demand'}
    worker, runner, queue = worker_for(config, stage, process)
    worker.admit(runner, worker.candidates(runner)[0])
    with queue.connect() as db:
        db.execute('UPDATE recordings SET attempts=2')
    assert worker.tick(runner, stop)['result']['reason'] == 'stopping'
    row = state(queue)
    assert row['status'] == 'queued' and row['attempts'] == 2 and row['revision'] == 'same-recipe'
    assert row['lease_owner'] is None and worker.deferred == {}


def test_retention_handover_finishes_copy_without_new_yield_context(tmp_path):
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    stop = Event()
    def process(*args, **kwargs):
        assert CURRENT.get().yield_signal is None
        stop.set()
        return {'status': 'completed'}
    worker, runner, queue = worker_for(config, 'retention', process)
    assert worker.tick(runner, stop)['result']['status'] == 'completed'
    assert state(queue)['status'] == 'completed'


def test_recent_resource_borrowing_is_same_thread_path_and_resource_only(tmp_path):
    config = {'storage': {'local_runtime_root': str(tmp_path)},
              'runtime': {'resource_limits': {'cloud': 1, 'cpu': 1}, 'admission_timeout_seconds': .01}}
    other = deepcopy(config)
    other['storage']['local_runtime_root'] = str(tmp_path / 'Other')
    coordinator = ResourceCoordinator(tmp_path / 'state/resources.sqlite3')
    other_coordinator = ResourceCoordinator(tmp_path / 'Other/state/resources.sqlite3')
    with execution_context(source='nas', yield_signal=Event()), resource_slot(config, 'cloud'):
        with resource_slot(config, 'cloud'):
            assert len(coordinator.snapshot()) == 1
        with resource_slot(config, 'cpu'):
            assert {row['resource'] for row in coordinator.snapshot()} == {'cloud', 'cpu'}
        with resource_slot(other, 'cloud'):
            assert len(other_coordinator.snapshot()) == 1 and len(coordinator.snapshot()) == 1
        copied = copy_context()
        def in_other_thread():
            with pytest.raises(TimeoutError), resource_slot(config, 'cloud'):
                pytest.fail('A copied context cannot borrow another thread lease')
        with ThreadPoolExecutor(1) as pool:
            pool.submit(copied.run, in_other_thread).result(3)
        assert len(coordinator.snapshot()) == 1
    assert coordinator.snapshot() == other_coordinator.snapshot() == []
