"""Persistent queue tests; no NAS or real models are involved."""
from types import SimpleNamespace
import json
import threading
import time

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.observed_inventory import observe
from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401


def setup(tmp_path, monkeypatch, records):
    root = tmp_path / 'runtime'
    queues = {stage: DeviceDayQueue(root / f'queue-{stage}.sqlite3', latest_first=True)
              for stage in ('retention', 'vision')}
    runner = SimpleNamespace(runtime_root=root, queues=queues,
        settings={'failure_retry_limit': 3}, config={'collection_ingest': {}})
    runner.process = lambda record, **kw: {'status': 'completed', 'recording_id': record['recording_id']}
    observe(root, {'recordings': records})
    def admit(runner, record, stage):
        runner.queues[stage].enqueue(record, record['source_signature'])
        return True
    monkeypatch.setattr('visioncortex.device_day_retention_worker.enqueue', admit)
    return runner


def record(rid, start, camera='a', **extra):
    return {'recording_id': rid, 'camera_key': camera, 'configured_role': 'first_person',
            'recording_start_us': start, 'recording_end_us': start + 1_000_000,
            'source_signature': rid, 'processable': True, **extra}


def test_new_arrivals_continue_without_restarting_and_completed_never_replays(tmp_path, monkeypatch):
    runner = setup(tmp_path, monkeypatch, [record('old', 1_000_000), record('new', 2_000_000)])
    worker = RetentionWorker()
    assert worker.tick(runner, threading.Event())['recording_id'] == 'new'
    observe(runner.runtime_root, {'recordings': [record('arrived', 3_000_000)]})
    assert worker.tick(runner, threading.Event())['recording_id'] == 'arrived'
    assert worker.tick(runner, threading.Event())['recording_id'] == 'old'
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'


def test_existing_camera_lease_is_never_stolen_and_camera_fairness_remains(tmp_path, monkeypatch):
    rows = [record('a-old', 1_000_000), record('a-new', 3_000_000), record('b', 2_000_000, 'b')]
    runner = setup(tmp_path, monkeypatch, rows)
    queue = runner.queues['retention']
    queue.enqueue(rows[0], rows[0]['source_signature'])
    assert queue.claim('main-service', camera_serial=True)['recording_id'] == 'a-old'
    worker = RetentionWorker()
    assert worker.tick(runner, threading.Event())['recording_id'] == 'b'
    with queue.connect() as db:
        row = db.execute("SELECT status,lease_owner,attempts FROM recordings WHERE recording_id='a-old'").fetchone()
    assert tuple(row) == ('running', 'main-service', 1)


def test_failure_has_persisted_cooldown_and_bounded_attempts_across_restart(tmp_path, monkeypatch):
    row = record('bad', 1_000_000)
    runner = setup(tmp_path, monkeypatch, [row])
    runner.process = lambda *a, **kw: {'status': 'failed', 'error_type': 'InputChanged'}
    worker = RetentionWorker()
    assert worker.tick(runner, threading.Event())['result']['status'] == 'failed'
    assert RetentionWorker().tick(runner, threading.Event())['status'] == 'waiting'
    queue = runner.queues['retention']
    with queue.connect() as db:
        db.execute("UPDATE recordings SET attempts=3,updated_at=? WHERE recording_id='bad'", (time.time() - 100,))
    assert RetentionWorker().tick(runner, threading.Event())['status'] == 'waiting'
    with queue.connect() as db:
        assert db.execute("SELECT attempts FROM recordings WHERE recording_id='bad'").fetchone()[0] == 3


def test_ready_vision_preferred_but_known_source_deadline_protected(tmp_path, monkeypatch):
    ready, new = record('ready', 1_000_000), record('new', 3_000_000)
    runner = setup(tmp_path, monkeypatch, [ready, new])
    queue = runner.queues['vision']
    queue.enqueue(ready, 'vision')
    queue.claim('vision')
    queue.finish('vision', 'ready', {'status': 'completed'}, 1)
    worker = RetentionWorker()
    assert worker.candidates(runner)[0]['recording_id'] == 'ready'
    urgent = record('urgent', 500_000, source_expires_at=time.time() + 1)
    observe(runner.runtime_root, {'recordings': [urgent]})
    assert worker.candidates(runner)[0]['recording_id'] == 'urgent'


def test_standard_admission_rejection_does_not_claim(tmp_path, monkeypatch):
    runner = setup(tmp_path, monkeypatch, [record('new', 1_000_000)])
    monkeypatch.setattr('visioncortex.device_day_retention_worker.enqueue', lambda *a: False)
    runner.process = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('must not execute'))
    assert RetentionWorker().tick(runner, threading.Event()) == {'status': 'waiting', 'admitted': 0}


def test_stop_prevents_claims_and_failure_does_not_block_another_camera(tmp_path, monkeypatch):
    rows = [record('bad', 3_000_000), record('good', 2_000_000, 'b')]
    runner = setup(tmp_path, monkeypatch, rows)
    worker = RetentionWorker()
    stop = threading.Event()
    stop.set()
    assert worker.tick(runner, stop)['status'] == 'stopping'
    stop.clear()
    runner.process = lambda r, **kw: {'status': 'failed' if r['recording_id'] == 'bad' else 'completed'}
    assert worker.tick(runner, stop)['result']['status'] == 'failed'
    assert worker.tick(runner, stop)['recording_id'] == 'good'


def test_existing_standalone_instance_holds_global_lock(tmp_path, monkeypatch):
    from visioncortex.device_day import exclusive
    runner = setup(tmp_path, monkeypatch, [record('new', 1_000_000)])
    with exclusive(runner.runtime_root / 'locks' / 'retention-worker.lock'):
        assert RetentionWorker().tick(runner, threading.Event())['status'] == 'running_elsewhere'


def test_explicit_pause_never_claims_or_changes_queue(tmp_path, monkeypatch):
    runner = setup(tmp_path, monkeypatch, [record('new', 1_000_000)])
    runner.config['device_day'] = {'paused_stages': ['retention']}
    assert RetentionWorker().tick(runner, threading.Event())['status'] == 'paused_by_user'
    with runner.queues['retention'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0] == 0


def test_real_inplace_receipt_pipeline_is_reused_without_visual_rerun(device_config):  # noqa: F811
    from visioncortex.device_day import DeviceDayRunner
    device_config['device_day'].update(inplace_preprocessing=True, latest_first=True)
    capture(device_config)
    item, _ = item_and_layout(device_config)
    models = FakeModels()
    runner = DeviceDayRunner(device_config, models)
    assert runner.run_once({'recordings': [item]}, stage='vision')['results'][0]['status'] == 'completed'
    observe(runner.runtime_root, {'recordings': [item]})
    original = json.loads(runner._receipt(runner.layout(item), item, 'vision').read_bytes())
    result = RetentionWorker().tick(runner, threading.Event())
    assert result['result']['status'] == 'completed', result
    current = json.loads(runner._receipt(runner.layout(item), item, 'vision').read_bytes())
    assert models.vision_calls == 1
    for field in ('key', 'segments', 'artifacts', 'input_binding'):
        assert current[field] == original[field]
    assert RetentionWorker().tick(runner, threading.Event())['status'] == 'waiting'
