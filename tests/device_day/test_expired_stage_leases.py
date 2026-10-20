"""Process-death recovery uses verified local receipts and never invokes models."""
import json
import time
import pytest

from test_device_day import FakeModels, capture, item_and_layout, device_config as device_config_fixture
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker

device_config = device_config_fixture


def record():
    return {'recording_id': 'owned', 'source_signature': 'source-1', 'camera_key': 'camera-1',
            'configured_role': 'first_person', 'recording_start_us': 1, 'recording_end_us': 2}


def expired(queue):
    queue.enqueue(record(), 'execution-1')
    queue.claim('old-process')
    with queue.connect() as db:
        db.execute('UPDATE recordings SET lease_until=?,attempts=2', (time.time()-10,))
    return queue.expired()[0]


def test_paid_expired_lease_cannot_be_claimed_or_erased_by_recipe_change(tmp_path):
    queue = DeviceDayQueue(tmp_path/'queue-understanding.sqlite3')
    old = expired(queue)
    assert queue.claim('new-process', retry=True) is None
    queue.enqueue(record(), 'execution-2')
    assert queue.expired() == [old]
    proof = {'queue_revision': old['revision'], 'source_signature': 'source-1', 'model_invoked': False,
             'reason': 'no_local_completion_receipt'}
    result = {'status': 'result_unknown', 'recording_id': 'owned', 'reason': 'result_unknown'}
    assert queue.recover_expired(old, result, proof)
    assert not queue.recover_expired(old, result, proof)
    queue.enqueue(record(), 'execution-3')
    with queue.connect() as db:
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
        saved = dict(db.execute('SELECT * FROM lease_recoveries').fetchone())
    assert row['status'] == 'blocked' and row['blocked_reason'] == 'result_unknown'
    assert row['revision'] == old['revision'] and row['attempts'] == 2
    assert row['lease_owner'] is None and row['lease_until'] is None
    assert json.loads(saved['previous_row']) == old and json.loads(saved['proof']) == proof


def test_renewed_or_changed_lease_is_not_overwritten_by_stale_recovery(tmp_path):
    queue = DeviceDayQueue(tmp_path/'queue-understanding.sqlite3')
    old = expired(queue)
    queue.renew('old-process')
    proof = {'queue_revision': old['revision'], 'source_signature': 'source-1', 'model_invoked': False}
    assert queue.recover_expired(old, {'status': 'result_unknown', 'recording_id': 'owned'}, proof) is False
    with queue.connect() as db:
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
    assert row['status'] == 'running' and row['lease_owner'] == 'old-process' and row['attempts'] == 2


def completed_stage(device_config):
    capture(device_config)
    item, layout = item_and_layout(device_config)
    device_config['device_day']['understanding_coverage_since_us'] = item['recording_start_us']-1
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    assert runner.process(item, stage='retention')['status'] == 'completed'
    assert runner.process(item, stage='vision')['status'] == 'completed'
    worker = RetentionWorker(stage='understanding')
    assert worker.admit(runner, item)
    queue = runner.queues['understanding']
    assert queue.claim('old-process')
    assert runner.process(item, stage='understanding')['status'] == 'completed'
    with queue.connect() as db:
        db.execute('UPDATE recordings SET lease_until=?,attempts=2', (time.time()-10,))
    return runner, worker, queue, layout, item, backend


def test_expired_completion_requires_current_keys_and_verified_artifacts_without_rebilling(device_config):
    runner, worker, queue, _, _, backend = completed_stage(device_config)
    calls = backend.semantic_calls
    recovered = worker.recover_expired(runner)
    assert recovered == {'completed': 1, 'result_unknown': 0, 'busy': 0}
    with queue.connect() as db:
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
        proof = json.loads(db.execute('SELECT proof FROM lease_recoveries').fetchone()[0])
    assert row['status'] == 'completed' and row['attempts'] == 2
    assert proof['status'] == 'verified' and proof['verified_artifacts'] > 0
    assert proof['receipt_sha256'] and proof['expected_execution_key']
    assert backend.semantic_calls == calls


def test_damaged_completed_artifact_becomes_local_unknown_with_original_attempts(device_config):
    runner, worker, queue, layout, item, backend = completed_stage(device_config)
    from visioncortex.device_day_contract import read_json
    saved = read_json(runner._receipt(layout, item, 'understanding'))
    path = layout.root/saved['artifacts'][0]['path']
    path.write_bytes(b'corrupt completed output')
    calls = backend.semantic_calls
    assert worker.recover_expired(runner) == {'completed': 0, 'result_unknown': 1, 'busy': 0}
    with queue.connect() as db:
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
    assert row['status'] == 'blocked' and row['blocked_reason'] == 'result_unknown'
    assert row['attempts'] == 2 and backend.semantic_calls == calls


@pytest.mark.parametrize('pinned', [False, True])
def test_changed_recipe_recovers_only_with_exact_receipt_and_queue_continuity_pin(device_config, tmp_path, monkeypatch, pinned):
    from visioncortex.device_day_contract import atomic_json, digest, file_hash, read_json
    from visioncortex.device_day_engine_rollover import CompletedStageReceipts
    runner, worker, queue, layout, item, backend = completed_stage(device_config)
    expected = queue.expired()[0]
    path = runner._receipt(layout, item, 'understanding')
    original_bytes, saved = path.read_bytes(), read_json(path)
    original = runner._key
    def migrated(stage, *args, **kwargs):
        key = original(stage, *args, **kwargs)
        return digest(['reviewed_new_executor', key]) if stage == 'understanding' else key
    monkeypatch.setattr(runner, '_key', migrated)
    vision = read_json(runner._receipt(layout, item, 'vision'))
    key = runner._key('understanding', item, {'vision': vision, 'stt': None, 'context': runner._context(layout, item)})
    if pinned:
        manifest = tmp_path/'Continuity.json'
        atomic_json(manifest, {'schema_version': 'visioncortex-resume-checkpoints/1', 'entries': [{
            'stage': 'understanding', 'recording_id': item['recording_id'], 'expected_key': key,
            'receipt_key': saved['key'], 'receipt_digest': digest(saved), 'queue_revision': expected['revision']}]})
        runner._completed_stage_receipts = CompletedStageReceipts({'completed_stage_receipts': {
            'path': str(manifest), 'sha256': file_hash(manifest)}})
    calls = backend.semantic_calls
    recovery = worker.recover_expired(runner)
    assert recovery['completed'] == int(pinned) and recovery['result_unknown'] == int(not pinned)
    assert backend.semantic_calls == calls and path.read_bytes() == original_bytes
    with queue.connect() as db:
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
        proof = json.loads(db.execute('SELECT proof FROM lease_recoveries').fetchone()[0])
    assert row['attempts'] == expected['attempts']
    if pinned:
        assert proof['queue_continuity'] == 'exact_manifest_pin'
        assert proof['receipt_key'] != proof['expected_execution_key']
