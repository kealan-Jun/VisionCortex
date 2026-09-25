"""Persistent queue tests; no NAS or real models are involved."""
from types import SimpleNamespace
from contextlib import contextmanager
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
              for stage in ('retention', 'vision', 'stt')}
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


def test_speech_is_latest_first_independent_of_vision_and_repeats_for_new_arrivals(tmp_path, monkeypatch):
    audio = {'status': 'provided', 'capture_complete': True}
    rows = [record('old', 1_000_000, audio=audio),
            record('new', 2_000_000, audio={'status': 'provided', 'capture_complete': False}),
            record('open', 3_000_000, audio={'status': 'pending_publication', 'capture_complete': False})]
    runner = setup(tmp_path, monkeypatch, rows)
    calls = []
    def process(record, *, stage, retry):
        calls.append((record['recording_id'], stage, retry))
        return {'status': 'completed'}
    runner.process = process
    # A missing vision queue proves speech candidate selection cannot read it.
    del runner.queues['vision']
    worker = RetentionWorker(stage='stt')
    assert worker.tick(runner, threading.Event())['recording_id'] == 'new'
    observe(runner.runtime_root, {'recordings': [record('arrived', 4_000_000, audio=audio)]})
    assert worker.tick(runner, threading.Event())['recording_id'] == 'arrived'
    assert worker.tick(runner, threading.Event())['recording_id'] == 'old'
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'
    assert calls == [('new', 'stt', True), ('arrived', 'stt', True), ('old', 'stt', True)]


def test_speech_main_lease_and_completed_receipt_are_not_reclaimed(tmp_path, monkeypatch):
    audio = {'status': 'provided', 'capture_complete': True}
    done, busy = record('done', 1_000_000, audio=audio), record('busy', 2_000_000, audio=audio)
    runner = setup(tmp_path, monkeypatch, [done, busy])
    queue = runner.queues['stt']
    queue.enqueue(done, 'done')
    queue.claim('old-owner')
    queue.finish('old-owner', 'done', {'status': 'completed', 'original_provider': 'aliyun'}, 1)
    queue.enqueue(busy, 'busy')
    queue.claim('main-service')
    with queue.connect() as db:
        before = [dict(row) for row in db.execute('SELECT * FROM recordings ORDER BY recording_id')]
    assert RetentionWorker(stage='stt').tick(runner, threading.Event())['status'] == 'waiting'
    with queue.connect() as db:
        assert before == [dict(row) for row in db.execute('SELECT * FROM recordings ORDER BY recording_id')]


def test_speech_pause_and_cli_stage_are_respected(tmp_path, monkeypatch):
    from visioncortex import device_day_retention_worker as module
    runner = setup(tmp_path, monkeypatch, [])
    runner.config['device_day'] = {'paused_stages': ['stt']}
    assert RetentionWorker(stage='stt').tick(runner, threading.Event())['status'] == 'paused_by_user'
    observed = []
    monkeypatch.setattr(module, 'serve', lambda config, stop, stage: observed.append((config, stage)))
    module.main(['--config', 'owned-fixture.yaml', '--stage', 'stt'])
    assert observed == [('owned-fixture.yaml', 'stt')]


def test_speech_can_pass_one_old_waiter_but_never_exceeds_two_camera_slots(tmp_path, monkeypatch):
    audio = {'status': 'provided', 'capture_complete': True}
    old, new = record('old', 1_000_000, audio=audio), record('new', 2_000_000, audio=audio)
    runner = setup(tmp_path, monkeypatch, [old, new])
    runner.config['runtime'] = {'resource_limits': {'stt': 2}}
    queue = runner.queues['stt']
    queue.enqueue(old, 'old')
    queue.claim('original-main', camera_serial=True)
    with queue.connect() as db:
        original = dict(db.execute("SELECT * FROM recordings WHERE recording_id='old'").fetchone())
    worker = RetentionWorker(stage='stt')
    assert worker.tick(runner, threading.Event())['recording_id'] == 'new'
    with queue.connect() as db:
        assert original == dict(db.execute("SELECT * FROM recordings WHERE recording_id='old'").fetchone())
    second = record('second', 3_000_000, audio=audio)
    queue.enqueue(second, 'second')
    assert queue.claim('second-owner', camera_serial=True, camera_limit=2)['recording_id'] == 'second'
    observe(runner.runtime_root, {'recordings': [record('third', 4_000_000, audio=audio)]})
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'
    with queue.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM recordings WHERE status='running'").fetchone()[0] == 2


def test_speech_respects_configured_capacity_of_one(tmp_path, monkeypatch):
    audio = {'status': 'provided', 'capture_complete': True}
    old, new = record('old', 1_000_000, audio=audio), record('new', 2_000_000, audio=audio)
    runner = setup(tmp_path, monkeypatch, [old, new])
    runner.config['runtime'] = {'resource_limits': {'stt': 1}}
    queue = runner.queues['stt']
    queue.enqueue(old, 'old')
    queue.claim('original-main')
    assert RetentionWorker(stage='stt').tick(runner, threading.Event())['status'] == 'waiting'


def test_latest_vision_uses_explicit_lane_and_one_extra_camera_slot(tmp_path, monkeypatch):
    from visioncortex import device_day_io
    rows = [record('older-one', 1_000_000, capture_complete=True),
            record('older-two', 2_000_000, capture_complete=True),
            record('newest', 3_000_000, capture_complete=False),
            record('open', 4_000_000, capture_complete=False, processable=False)]
    runner = setup(tmp_path, monkeypatch, rows)
    runner.settings['vision_jobs_per_camera'] = 2
    runner.config['runtime'] = {'resource_limits': {'vision': 12}}
    queue = runner.queues['vision']
    for row in rows[:2]:
        queue.enqueue(row, row['source_signature'])
        queue.claim(row['recording_id'], camera_serial=True, camera_limit=2)
    active = []
    @contextmanager
    def lane():
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    monkeypatch.setattr(device_day_io, 'live_vision_lane', lane, raising=False)
    calls = []
    def process(row, *, stage, retry):
        assert active == [True]
        calls.append((row['recording_id'], stage))
        return {'status': 'completed'}
    runner.process = process
    with queue.connect() as db:
        previous = [dict(row) for row in db.execute("SELECT * FROM recordings WHERE status='running' ORDER BY recording_id")]
    worker = RetentionWorker(stage='vision')
    assert worker.tick(runner, threading.Event())['recording_id'] == 'newest'
    assert calls == [('newest', 'vision')]
    assert active == []
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'
    with queue.connect() as db:
        assert previous == [dict(row) for row in db.execute("SELECT * FROM recordings WHERE status='running' ORDER BY recording_id")]


def test_vision_cannot_claim_fourth_slice_and_new_arrival_is_continuously_seen(tmp_path, monkeypatch):
    from visioncortex import device_day_io
    from contextlib import nullcontext
    old = [record(str(i), i * 1_000_000, capture_complete=True) for i in (1, 2, 3)]
    runner = setup(tmp_path, monkeypatch, old)
    runner.settings['vision_jobs_per_camera'] = 2
    runner.config['runtime'] = {'resource_limits': {'vision': 12}}
    queue = runner.queues['vision']
    for row in old:
        queue.enqueue(row, row['source_signature'])
        queue.claim(row['recording_id'], camera_serial=True, camera_limit=3)
    fresh = record('fresh', 5_000_000, capture_complete=True)
    observe(runner.runtime_root, {'recordings': [fresh]})
    worker = RetentionWorker(stage='vision')
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'
    monkeypatch.setattr(device_day_io, 'live_vision_lane', nullcontext, raising=False)
    queue.finish('3', '3', {'status': 'completed'}, 1)
    assert worker.tick(runner, threading.Event())['recording_id'] == 'fresh'


def test_vision_configured_total_capacity_and_pause_still_limit_claims(tmp_path, monkeypatch):
    rows = [record('old', 1_000_000, capture_complete=True), record('new', 2_000_000, capture_complete=True)]
    runner = setup(tmp_path, monkeypatch, rows)
    runner.settings['vision_jobs_per_camera'] = 2
    runner.config['runtime'] = {'resource_limits': {'vision': 1}}
    queue = runner.queues['vision']
    queue.enqueue(rows[0], 'old')
    queue.claim('old-owner')
    worker = RetentionWorker(stage='vision')
    assert worker.tick(runner, threading.Event())['status'] == 'waiting'
    runner.config['device_day'] = {'paused_stages': ['vision']}
    assert worker.tick(runner, threading.Event())['status'] == 'paused_by_user'


def test_dedicated_live_lane_does_not_fill_with_another_cameras_history(tmp_path, monkeypatch):
    now = int(time.time() * 1_000_000)
    rows = [record('history', now - 86400_000_000, camera='old'),
            record('live', now - 60_000_000, camera='current', capture_complete=False)]
    runner = setup(tmp_path, monkeypatch, rows)
    worker = RetentionWorker(stage='vision', recent_seconds=14400)
    assert [r['recording_id'] for r in worker.candidates(runner)] == ['live']
