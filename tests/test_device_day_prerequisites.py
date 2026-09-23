import json
from pathlib import Path
import threading
import time

import pytest

from test_device_day import device_config as _device_config, capture, FakeModels, item_and_layout
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_prerequisites import verified_prerequisite
from visioncortex.device_day_recovery import repair_completed_retention, RetentionRecovery


@pytest.fixture
def device_config(default_config, tmp_path):
    config = _device_config.__wrapped__(default_config, tmp_path)
    config['device_day'].update(camera_lanes=True, failure_retry_limit=3)
    return config


def prepared(config):
    capture(config)
    record, layout = item_and_layout(config)
    backend = FakeModels()
    runner = DeviceDayRunner(config, backend)
    inventory = {'recordings': [record]}
    assert runner.run_once(inventory, stage='retention')['results'][0]['status'] == 'completed'
    assert runner._prepare_stage(inventory, 'vision', None) == {record['recording_id']}
    receipt = read_json(runner._receipt(layout, record, 'retention'))
    video = layout.root / next(x['retained']['path'] for x in receipt['sources'] if x['kind'] == 'video')
    return runner, record, layout, inventory, video, backend


def queue_row(queue):
    with queue.connect() as db:
        return dict(db.execute('SELECT * FROM recordings').fetchone())


def expire_readiness(runner):
    runner._prepared.clear()
    runner._record_readiness['vision'].clear()


def test_missing_archive_is_waiting_without_spending_model_attempt(device_config):
    runner, record, _, inventory, video, backend = prepared(device_config)
    video.unlink()
    # Cached admission can race a missing artifact. Execution still fails closed.
    result = runner.run_once(inventory, stage='vision', retry=True)
    assert result['results'][0]['status'] == 'waiting_for_prerequisite'
    assert result['results'][0]['prerequisite_stage'] == 'retention'
    queue = runner.queues['vision']
    row = queue_row(queue)
    assert (row['status'], row['attempts'], row['lease_owner'], row['lease_until']) == (
        'waiting_for_prerequisite', 0, None, None)
    assert backend.vision_calls == 0
    assert not queue.has_processing_work(3)
    assert queue.snapshot()['prerequisite_waits_by_stage'] == {'retention': 1}
    for _ in range(3):
        expire_readiness(runner)
        assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    assert queue_row(queue)['attempts'] == 0
    assert runner.process(record, stage='vision')['status'] == 'waiting_for_prerequisite'


def test_initial_admission_leaves_artifact_checks_to_the_claimed_worker(device_config):
    runner, _, _, inventory, video, backend = prepared(device_config)
    video.unlink()
    expire_readiness(runner)
    assert runner._prepare_stage(inventory, 'vision', None)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'][0]['status'] == 'waiting_for_prerequisite'
    assert queue_row(runner.queues['vision'])['status'] == 'waiting_for_prerequisite'
    assert queue_row(runner.queues['vision'])['attempts'] == 0
    assert backend.vision_calls == 0


def test_only_exact_last_legacy_prerequisite_failure_is_refunded_then_verified(device_config):
    runner, record, _, inventory, video, backend = prepared(device_config)
    queue = runner.queues['vision']
    for _ in range(3):
        assert queue.claim('worker', retry=True, max_attempts=3)
        queue.finish('worker', record['recording_id'], {'status': 'failed', 'error_type': 'ValueError',
                     'message': 'Prerequisite stage retention is not ready'}, 1)
    original = video.read_bytes()
    video.unlink()
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    assert (queue_row(queue)['status'], queue_row(queue)['attempts']) == ('waiting_for_prerequisite', 2)
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    assert queue_row(queue)['attempts'] == 2
    # A same-size corrupt restoration cannot release the wait.
    video.write_bytes(b'x' * len(original))
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    assert not runner._prerequisite_checks.verify_pending(runner)['prerequisite_artifacts_verified']
    assert backend.vision_calls == 0
    video.write_bytes(original)
    # A service restart preserves the durable wait/budget and obtains a new
    # exact background verification before it may resume this revision.
    runner = DeviceDayRunner(device_config, backend)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    assert runner._prerequisite_checks.verify_pending(runner)['prerequisite_artifacts_verified']
    expire_readiness(runner)
    result = runner.run_once(inventory, stage='vision', retry=True)
    assert result['results'][0]['status'] == 'completed'
    assert backend.vision_calls == 1
    assert queue_row(queue)['attempts'] == 3
    with queue.connect() as db:
        events = [json.loads(row[0]) for row in db.execute(
            "SELECT data FROM task_events WHERE state='waiting_for_prerequisite'")]
    assert len(events) == 1
    assert events[0]['legacy_failure_refunded'] is True
    assert events[0]['refunded_attempts'] == 1


def test_real_model_failure_budget_and_new_owner_are_preserved(device_config):
    runner, record, _, inventory, _, backend = prepared(device_config)
    queue = runner.queues['vision']
    for _ in range(3):
        queue.claim('worker', retry=True, max_attempts=3)
        queue.finish('worker', record['recording_id'], {'status': 'failed', 'error_type': 'ValueError',
                     'message': 'Coarse sampling incomplete; cannot label inactivity'}, 1)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    assert (queue_row(queue)['status'], queue_row(queue)['attempts']) == ('failed', 3)
    assert backend.vision_calls == 0
    queue.enqueue(record, 'new-revision')
    assert queue.claim('new-owner')
    assert not queue.wait_for_prerequisite(record['recording_id'], 'new-revision', 'retention')
    assert queue_row(queue)['lease_owner'] == 'new-owner'
    queue.finish('old-owner', record['recording_id'], {'status': 'waiting_for_prerequisite'}, 1)
    assert queue_row(queue)['lease_owner'] == 'new-owner'


def test_unchanged_corrupt_prerequisite_is_not_rehashed_each_poll(device_config, monkeypatch):
    runner, record, layout, _, video, _ = prepared(device_config)
    path = runner._receipt(layout, record, 'retention')
    receipt = read_json(path)
    key = runner._key('retention', record, record)
    calls = []
    original = runner._load
    monkeypatch.setattr(runner, '_load', lambda *a: calls.append(1) or original(*a))
    good = video.read_bytes()
    video.write_bytes(b'x' * len(good))
    for _ in range(4):
        assert not verified_prerequisite(runner, path, key, layout, receipt)
    assert calls == [1]
    video.write_bytes(good)
    assert verified_prerequisite(runner, path, key, layout, receipt)
    assert calls == [1, 1]


def test_retention_repair_requeues_only_verified_unchanged_source_once(device_config):
    runner, record, _, inventory, video, backend = prepared(device_config)
    video.unlink()
    expire_readiness(runner)
    runner.run_once(inventory, stage='vision', retry=True)
    queue = runner.queues['retention']
    before = queue_row(queue)
    from visioncortex.input_availability import Availability
    availability = Availability(runner.runtime_root)
    availability.mark(record, 'missing', observed=1)
    result = repair_completed_retention(runner, before)
    assert result['status'] == 'retention_repair_queued'
    assert result['source_signature_unchanged'] is True
    assert result['model_invoked'] is False
    assert queue_row(queue)['status'] == 'queued'
    assert availability.states()[record['recording_id']]['state'] == 'ready'
    assert repair_completed_retention(runner, before)['status'] == 'unchanged_retention_repair_already_attempted'
    assert backend.vision_calls == 0
    runner._prepared.clear()
    assert runner.run_once(inventory, stage='retention', retry=True)['results'][0]['status'] == 'completed'
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    assert runner._prerequisite_checks.verify_pending(runner)['prerequisite_artifacts_verified']
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'][0]['status'] == 'completed'
    assert backend.vision_calls == 1


def test_missing_capture_preserves_historical_retention_and_bounded_recovery(device_config):
    runner, record, _, inventory, video, _ = prepared(device_config)
    video.unlink()
    Path(record['video_path']).unlink()
    expire_readiness(runner)
    runner.run_once(inventory, stage='vision', retry=True)
    queue = runner.queues['retention']
    before = queue_row(queue)
    worker = RetentionRecovery()
    assert worker.tick(runner)['status'] == 'waiting_for_retention_source'
    assert worker.tick(runner)['status'] == 'idle'
    assert queue_row(queue) == before
    assert queue_row(runner.queues['vision'])['status'] == 'waiting_for_prerequisite'


def test_transient_archive_failure_does_not_revoke_completed_retention(device_config, monkeypatch):
    runner, _, _, _, _, _ = prepared(device_config)
    queue = runner.queues['retention']
    before = queue_row(queue)
    real = runner._load
    calls = []
    def transient(*args):
        calls.append(1)
        return None if len(calls) == 1 else real(*args)
    monkeypatch.setattr(runner, '_load', transient)
    assert repair_completed_retention(runner, before)['status'] == 'prerequisite_restored'
    assert queue_row(queue) == before


def test_retention_repair_rejects_changed_source_and_preserves_new_owner(device_config, monkeypatch):
    runner, record, _, _, video, _ = prepared(device_config)
    video.unlink()
    queue = runner.queues['retention']
    before = queue_row(queue)
    monkeypatch.setattr('visioncortex.nas_recordings._inspect',
                        lambda *a: record | {'source_signature': 'changed'})
    assert repair_completed_retention(runner, before)['status'] == 'changed_input_waiting_for_monitor'
    assert queue_row(queue) == before
    monkeypatch.setattr('visioncortex.nas_recordings._inspect', lambda *a: record)
    queue.enqueue(record, 'new-revision')
    assert queue.claim('new-owner')
    assert repair_completed_retention(runner, before)['status'] == 'queue_changed'
    assert queue_row(queue)['lease_owner'] == 'new-owner'


def test_expired_prerequisite_claim_can_wait_without_refunding_old_attempt(device_config):
    runner, record, _, inventory, video, _ = prepared(device_config)
    queue = runner.queues['vision']
    queue.claim('old-owner')
    row = queue_row(queue)
    assert not queue.wait_for_prerequisite(record['recording_id'], row['revision'], 'retention')
    with queue.connect() as db:
        db.execute('UPDATE recordings SET lease_until=0')
    video.unlink()
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'][0]['status'] == 'waiting_for_prerequisite'
    after = queue_row(queue)
    assert (after['status'], after['attempts'], after['lease_owner']) == ('waiting_for_prerequisite', 1, None)
    with queue.connect() as db:
        event = json.loads(db.execute("SELECT data FROM task_events WHERE state='waiting_for_prerequisite'").fetchone()[0])
    # Only the newly claimed prerequisite wait is refunded; the unknown
    # expired attempt remains charged.
    assert event['prerequisite_stage'] == 'retention'


def test_missing_retention_receipt_can_be_rebuilt_from_unchanged_capture(device_config):
    runner, record, layout, inventory, _, backend = prepared(device_config)
    path = runner._receipt(layout, record, 'retention')
    path.unlink()
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    queue = runner.queues['retention']
    result = repair_completed_retention(runner, queue_row(queue))
    assert result['status'] == 'retention_repair_queued'
    assert result['receipt_missing'] is True
    assert backend.vision_calls == 0
    runner._prepared.clear()
    assert runner.run_once(inventory, stage='retention', retry=True)['results'][0]['status'] == 'completed'
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'] == []
    assert runner._prerequisite_checks.verify_pending(runner)['prerequisite_artifacts_verified']
    expire_readiness(runner)
    assert runner.run_once(inventory, stage='vision', retry=True)['results'][0]['status'] == 'completed'


def test_service_recovery_callback_clears_transient_rejection_and_resumes(device_config, monkeypatch):
    from visioncortex import device_day_service as module
    device_config['device_day']['paused_stages'] = ['stt', 'understanding', 'report']
    runner, record, layout, inventory, _, backend = prepared(device_config)
    queue = runner.queues['vision']
    row = queue_row(queue)
    assert queue.wait_for_prerequisite(record['recording_id'], row['revision'], 'retention')
    path = runner._receipt(layout, record, 'retention')
    receipt = read_json(path)
    key = runner._key('retention', record, record)
    with monkeypatch.context() as transient:
        transient.setattr(runner, '_load', lambda *a: None)
        assert not verified_prerequisite(runner, path, key, layout, receipt)
    assert runner._rejected_prerequisites
    assert not verified_prerequisite(runner, path, key, layout, receipt)
    generation = runner._prerequisite_generation['vision']
    monkeypatch.setattr(module, 'DeviceDayRunner', lambda settings: runner)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 10**12})())
    service = module.DeviceDayService(lambda: device_config, threading.Lock())
    service.observe(device_config, inventory)
    service.start()
    try:
        deadline = time.monotonic() + 8
        while queue_row(queue)['status'] != 'completed':
            assert time.monotonic() < deadline, service.last_result
            time.sleep(.05)
        assert runner._prerequisite_generation['vision'] > generation
        assert not runner._rejected_prerequisites
        assert backend.vision_calls == 1
    finally:
        service.stop()
