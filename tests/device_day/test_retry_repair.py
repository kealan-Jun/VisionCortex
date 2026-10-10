"""Repair grants cross CLI, local queue CAS, scheduling and audit boundaries."""
import json
import sqlite3
import time

import pytest

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retry import (
    EVIDENCE_SCHEMA, PROVIDER_BINDING_ERROR, apply_retry_plan, main, plan_retry,
    provider_binding_proof, retry_limit,
)


def exhausted(tmp_path, result=None):
    queue = DeviceDayQueue(tmp_path / 'queue-understanding.sqlite3')
    record = {'recording_id': 'record', 'configured_role': 'first_person',
              'camera_key': 'camera', 'processable': True, 'source_signature': 'source-identity'}
    queue.enqueue(record, 'execution-identity')
    result = result or {'status': 'failed', 'error_type': 'PermissionError', 'message': 'input denied'}
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?,updated_at=?",
                   (json.dumps(result), time.time() - 100))
    return queue


def evidence(kind='storage_access'):
    checks = {'current_source_verified': 'PROVEN', 'successful_requests_preserved': 'PROVEN'}
    checks.update({key: 'PROVEN' for key in {
        'storage_access': ('storage_mount_verified', 'source_access_verified'),
        'provider_binding': ('provider_binding_verified',),
    }[kind]})
    return {'schema_version': EVIDENCE_SCHEMA, 'repair_kind': kind,
            'recording_id': 'record', 'revision': 'execution-identity',
            'source_signature': 'source-identity', 'repair_revision': 'a' * 64, 'checks': checks}


def row(queue):
    with queue.connect() as db:
        return dict(db.execute('SELECT * FROM recordings').fetchone())


def test_repair_budget_is_consumed_without_erasing_failure_or_repeating_successful_request(tmp_path):
    queue = exhausted(tmp_path)
    cached = tmp_path / 'paid-request-result.json'
    cached.write_text('{"status":"completed","request_identity":"same-request"}')
    before = row(queue)
    plan = plan_retry(queue.path, 'record', evidence())
    assert row(queue) == before  # Planning is read-only, not an admission.
    assert apply_retry_plan(queue, plan)
    assert row(queue)['attempts'] == 3
    assert row(queue)['status'] == 'failed'
    assert row(queue)['result'] == before['result']
    assert queue.has_processing_work(max_attempts=3)
    for attempt in range(4, 7):
        assert queue.claim('worker', retry=True, max_attempts=3)['recording_id'] == 'record'
        assert row(queue)['attempts'] == attempt
        queue.finish('worker', 'record', {'status': 'failed', 'error_type': 'PermissionError'}, 1)
    assert queue.claim('worker', retry=True, max_attempts=3) is None
    assert not queue.has_processing_work(max_attempts=3)
    with queue.connect() as db:
        saved = dict(db.execute('SELECT * FROM retry_authorizations').fetchone())
        event = dict(db.execute("SELECT * FROM task_events WHERE state='retry_authorized'").fetchone())
    assert saved['previous_result'] == before['result']
    assert saved['previous_attempts'] == 3
    assert event['attempt'] == 3
    assert json.loads(event['data'])['additional_attempts'] == 3
    assert cached.read_text() == '{"status":"completed","request_identity":"same-request"}'
    assert not apply_retry_plan(queue, plan)
    with pytest.raises(ValueError, match='already authorized'):
        plan_retry(queue.path, 'record', evidence())


@pytest.mark.parametrize('update', [
    "status='completed'", "status='running',lease_owner='active',lease_until=9999999999",
    "input_status='missing'", "revision='new-input'", "attempts=4",
    "result='{\"status\":\"failed\",\"error_type\":\"PermissionError\",\"message\":\"new failure\"}'",
])
def test_concurrent_completion_claim_input_or_failure_change_rejects_plan(tmp_path, update):
    queue = exhausted(tmp_path)
    plan = plan_retry(queue.path, 'record', evidence())
    with queue.connect() as db:
        db.execute('UPDATE recordings SET ' + update)
    before = row(queue)
    assert not apply_retry_plan(queue, plan)
    assert row(queue) == before
    with queue.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM retry_authorizations').fetchone()[0] == 0


def test_changed_source_evidence_unknown_causes_or_incomplete_checks_fail_closed(tmp_path):
    queue = exhausted(tmp_path)
    wrong = evidence() | {'source_signature': 'different-source'}
    with pytest.raises(ValueError, match='does not match'):
        plan_retry(queue.path, 'record', wrong)
    missing = evidence() | {'checks': {'source_access_verified': 'PROVEN'}}
    with pytest.raises(ValueError, match='missing a required'):
        plan_retry(queue.path, 'record', missing)
    with pytest.raises(ValueError, match='does not match'):
        plan_retry(queue.path, 'record', evidence() | {'api_key': 'must-not-be-saved'})
    with queue.connect() as db:
        db.execute('UPDATE recordings SET result=?', (json.dumps({'error_type': 'UnknownError'}),))
    with pytest.raises(ValueError, match='does not match'):
        plan_retry(queue.path, 'record', evidence())


def test_source_revision_resets_only_current_budget_but_keeps_prior_authorization(tmp_path):
    queue = exhausted(tmp_path)
    assert apply_retry_plan(queue, plan_retry(queue.path, 'record', evidence()))
    queue.enqueue(json.loads(row(queue)['payload']) | {'source_signature': 'new-source'}, 'new-execution')
    assert row(queue)['retry_limit'] == 0
    assert row(queue)['attempts'] == 0
    with queue.connect() as db:
        assert db.execute('SELECT previous_attempts FROM retry_authorizations').fetchone()[0] == 3


def test_local_provider_proof_is_current_and_does_not_expose_or_call_key(tmp_path, monkeypatch):
    from visioncortex.mllm_provider import connection_identity
    from visioncortex.provider_connection import adapter_identity
    settings = {'provider': 'aliyun', 'base_url': 'https://provider.example.test/v1', 'model': 'vision-model',
                'api_protocol': 'chat_completions', 'credential_ref': 'a' * 32}
    saved = {'connection': settings, 'api_key': 'fixture-private-key', 'verification': {
        'status': 'verified', 'model_invocation': 'PROVEN', 'connection_sha256': connection_identity(settings),
        'adapter_sha256': adapter_identity(), 'checked_at': '2026-10-10T00:00:00Z'}}
    monkeypatch.setattr('visioncortex.provider_credentials.read_revision', lambda _ref: saved)
    config = {'mllm': settings}
    proof = provider_binding_proof(config)
    assert 'fixture-private-key' not in json.dumps(proof)
    queue = exhausted(tmp_path, {'status': 'failed', 'error_type': 'RuntimeError', 'message': PROVIDER_BINDING_ERROR})
    repair = evidence('provider_binding') | {'provider_binding': proof}
    plan = plan_retry(queue.path, 'record', repair, config=config)
    with pytest.raises((ValueError, RuntimeError)):
        apply_retry_plan(queue, plan, config={'mllm': settings | {'model': 'changed-model'}})
    assert row(queue)['retry_limit'] == 0
    assert apply_retry_plan(queue, plan, config=config)
    assert 'fixture-private-key' not in json.dumps(plan)


def test_operator_cli_plan_then_apply_and_no_cross_queue_replay(tmp_path, capsys):
    queue = exhausted(tmp_path)
    proof, plan = tmp_path / 'repair.json', tmp_path / 'plan.json'
    proof.write_text(json.dumps(evidence()))
    assert main(['plan', '--queue', str(queue.path), '--recording-id', 'record',
                 '--evidence', str(proof), '--plan', str(plan)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'planned'
    assert main(['apply', '--queue', str(queue.path), '--plan', str(plan)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'applied'
    assert main(['apply', '--queue', str(queue.path), '--plan', str(plan)]) == 2
    other = exhausted(tmp_path / 'different')
    with pytest.raises(ValueError, match='different local queue'):
        apply_retry_plan(other, json.loads(plan.read_text()))


def test_legacy_rows_keep_default_limit_and_schema_upgrade_preserves_attempts(tmp_path):
    path = tmp_path / 'queue-vision.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE recordings(recording_id TEXT PRIMARY KEY,revision TEXT NOT NULL,payload TEXT NOT NULL,'
                   'status TEXT NOT NULL,queued_at REAL NOT NULL,updated_at REAL NOT NULL,lease_owner TEXT,lease_until REAL,'
                   'attempts INTEGER NOT NULL DEFAULT 0,result TEXT,completed_at REAL,wall_seconds REAL)')
        db.execute("INSERT INTO recordings VALUES('legacy','v1','{}','failed',0,0,NULL,NULL,3,'{}',NULL,NULL)")
    assert retry_limit({'attempts': 3}, 3) == 3
    queue = DeviceDayQueue(path)
    assert row(queue)['attempts'] == 3
    assert row(queue)['retry_limit'] == 0
    assert queue.claim('worker', retry=True, max_attempts=3) is None


def test_hold_refunds_one_attempt_without_replenishing_repair_budget(tmp_path):
    queue = exhausted(tmp_path)
    assert apply_retry_plan(queue, plan_retry(queue.path, 'record', evidence()))
    queue.claim('worker', retry=True, max_attempts=3)
    queue.finish('worker', 'record', {'status': 'waiting_for_provider'}, 1)
    assert row(queue)['attempts'] == 3
    assert row(queue)['retry_limit'] == 6
