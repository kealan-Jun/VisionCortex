"""Only an explicit bounded operator grant can reopen an uncertain paid stage."""
import json

import pytest

from visioncortex.device_day_queue import DeviceDayQueue


def blocked(tmp_path):
    queue = DeviceDayQueue(tmp_path/'queue-understanding.sqlite3')
    record = {'recording_id': 'record', 'source_signature': 'sealed-source',
              'configured_role': 'first_person', 'camera_key': 'camera', 'recording_start_us': 1800000000000000}
    queue.enqueue(record, 'a'*64)
    queue.claim('old-worker')
    queue.finish('old-worker', record['recording_id'], {'status': 'submitted_unknown', 'reason': 'submitted_unknown'}, 1)
    with queue.connect() as db:
        row = dict(db.execute('select * from recordings').fetchone())
    proof = {'authorization_id': 'b'*64, 'recording_id': record['recording_id'],
             'source_signature': record['source_signature'], 'potential_duplicate_billing_accepted': True,
             'current_source_verified': 'PROVEN', 'prior_results_checked': 'PROVEN', 'prerequisites_verified': 'PROVEN'}
    return queue, record, row, proof


def test_reanalysis_preserves_old_attempts_and_limits_new_recipe_to_one_attempt(tmp_path):
    queue, record, old, proof = blocked(tmp_path)
    assert queue.authorize_reanalysis(old, 'c'*64, proof)
    with queue.connect() as db:
        current = dict(db.execute('select * from recordings').fetchone())
        saved = dict(db.execute('select * from reanalysis_authorizations').fetchone())
    assert json.loads(saved['previous_row']) == old
    assert current['attempts'] == old['attempts'] and current['revision'] == 'c'*64
    assert saved['maximum_attempts'] == old['attempts']+1
    assert queue.claim('new-worker')['recording_id'] == record['recording_id']
    assert queue.has_processing_work(max_attempts=999)
    queue.finish('new-worker', record['recording_id'], {'status': 'failed', 'error_type': 'RuntimeError'}, 1)
    assert queue.claim('third-worker', retry=True, max_attempts=999) is None
    assert not queue.has_processing_work(max_attempts=999)
    assert not queue.authorize_reanalysis(old, 'c'*64, proof)


def test_resume_after_live_yield_does_not_spend_another_attempt(tmp_path):
    queue, record, old, proof = blocked(tmp_path)
    assert queue.authorize_reanalysis(old, 'c'*64, proof)
    queue.claim('worker')
    queue.finish('worker', record['recording_id'], {'status': 'paused_for_live'}, 1)
    assert queue.claim('resumed')['recording_id'] == record['recording_id']
    queue.finish('resumed', record['recording_id'], {'status': 'completed'}, 1)
    with queue.connect() as db:
        row = db.execute('select attempts,status from recordings').fetchone()
    assert row['status'] == 'completed' and row['attempts'] == old['attempts']+1


def test_recipe_upgrade_cannot_erase_the_authorized_source_attempt_ceiling(tmp_path):
    queue, record, old, proof = blocked(tmp_path)
    assert queue.authorize_reanalysis(old, 'c'*64, proof)
    assert not queue.enqueue(record, 'd'*64)
    queue.claim('worker')
    queue.finish('worker', record['recording_id'], {'status': 'failed'}, 1)
    assert not queue.enqueue(record, 'e'*64)
    assert queue.claim('extra-worker', retry=True, max_attempts=999) is None
    with queue.connect() as db:
        row = dict(db.execute('select * from recordings').fetchone())
    assert row['revision'] == 'c'*64 and row['attempts'] == old['attempts']+1


def test_a_new_source_is_not_bound_to_an_old_source_reanalysis_grant(tmp_path):
    queue, record, old, proof = blocked(tmp_path)
    assert queue.authorize_reanalysis(old, 'c'*64, proof)
    queue.enqueue(record | {'source_signature': 'new-sealed-source'}, 'd'*64)
    with queue.connect() as db:
        row = dict(db.execute('select * from recordings').fetchone())
    assert row['revision'] == 'd'*64 and row['attempts'] == 0


@pytest.mark.parametrize('change', [{'potential_duplicate_billing_accepted': False},
    {'source_signature': 'other-source'}, {'recording_id': 'other'},
    {'authorization_id': ''}, {'current_source_verified': 'NOT_PROVEN'}, {'prior_results_checked': None}, {'prerequisites_verified': None}])
def test_unapproved_or_different_source_never_reopens_queue(tmp_path, change):
    queue, record, old, proof = blocked(tmp_path)
    with pytest.raises(ValueError, match='operator authorization'):
        queue.authorize_reanalysis(old, 'c'*64, proof | change)
    with queue.connect() as db:
        assert dict(db.execute('select * from recordings').fetchone()) == old


def test_changed_queue_generation_is_not_overwritten(tmp_path):
    queue, record, old, proof = blocked(tmp_path)
    with queue.connect() as db:
        db.execute("update recordings set updated_at=updated_at+1")
    assert queue.authorize_reanalysis(old, 'c'*64, proof) is False


def test_verified_blocked_completion_restores_without_new_attempt_or_reanalysis_grant(tmp_path):
    queue, record, old, _ = blocked(tmp_path)
    result = {'status': 'completed', 'recording_id': record['recording_id'], 'stage': 'understanding'}
    proof = {'queue_revision': old['revision'], 'source_signature': record['source_signature'],
             'model_invoked': False, 'status': 'verified', 'receipt_sha256': 'd'*64,
             'expected_execution_key': 'e'*64}
    assert queue.reconcile_completed_unknown(old, result, proof)
    with queue.connect() as db:
        current = dict(db.execute('select * from recordings').fetchone())
        saved = db.execute('select previous_row from lease_recoveries').fetchone()
        assert db.execute('select count(*) from reanalysis_authorizations').fetchone()[0] == 0
    assert current['status'] == 'completed' and current['attempts'] == old['attempts']
    assert json.loads(saved[0]) == old
    assert queue.reconcile_completed_unknown(old, result, proof) is False


def test_blocked_completion_never_accepts_missing_receipt_or_source_proof(tmp_path):
    queue, record, old, _ = blocked(tmp_path)
    result = {'status': 'completed', 'recording_id': record['recording_id']}
    with pytest.raises(ValueError, match='exact-source'):
        queue.reconcile_completed_unknown(old, result, {})
    with pytest.raises(ValueError, match='verified completed'):
        queue.reconcile_completed_unknown(old, {'status': 'result_unknown'}, {})
