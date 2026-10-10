"""Bounded, evidence-bound retries without erasing failures or model receipts.

Planning reads local metadata only. Applying a plan requires an independently
verified repair and compares the failed row atomically. Ordinary workers consume
the additional budget through DeviceDayQueue.claim; attempts remain cumulative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from .sqlite_store import connection


SCHEMA = 'visioncortex-device-day-retry/1'
EVIDENCE_SCHEMA = 'visioncortex-device-day-repair/1'
PROVIDER_BINDING_ERROR = '任务的厂商配置与已验证凭据不一致，请在 AI 服务设置中重新验证。'
CHECKS = {
    'provider_binding': {'provider_binding_verified'},
    'storage_access': {'storage_mount_verified', 'source_access_verified'},
    'database_lock': {'queue_write_verified', 'lock_owner_reconciled'},
    'vision_nms': {'vision_model_verified', 'nms_repair_verified'},
    'media_decode': {'source_decode_verified'},
    'receipt_projection': {'bounded_projection_verified', 'queue_write_verified'},
    'resource_coordination': {'resource_policy_verified', 'lease_ownership_verified'},
    'storage_capacity': {'local_reserves_verified', 'queue_write_verified'},
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def retry_limit(row, baseline=3):
    """Pure metadata helper; legacy rows have no additional authorization."""
    value = dict(row).get('retry_limit', 0)
    return max(baseline, value if type(value) is int and value >= 0 else 0)


def failure_kind(result):
    """Recognize narrow repair classes without publishing arbitrary messages."""
    message = str(result.get('message') or '')
    error = result.get('error_type')
    if error == 'RuntimeError' and message == PROVIDER_BINDING_ERROR:
        return 'provider_binding'
    if error == 'OverflowError' and message == 'string longer than INT_MAX bytes':
        return 'receipt_projection'
    if ((error == 'RuntimeError' and message == 'Resource lease was lost; output cannot be marked complete')
            or (error == 'TimeoutError' and message in {
                'Resource admission timed out: vision', 'Resource admission timed out: nas-io-read'})):
        return 'resource_coordination'
    if ((error == 'OperationalError' and message == 'database or disk is full')
            or (error == 'OSError' and message in {'No space left on device', '[Errno 28] No space left on device'})):
        return 'storage_capacity'
    if error in {'FileNotFoundError', 'PermissionError', 'OSError'}:
        return 'storage_access'
    if error == 'OperationalError' and 'database is locked' in message.lower():
        return 'database_lock'
    if 'nms' in message.lower() and any(word in message.lower() for word in ('timeout', 'time limit', 'timed out')):
        return 'vision_nms'
    if any(word in message.lower() for word in ('decode source frame', 'source frame decoding', 'source-frame decoding',
                                                'required source frame could not be decoded')):
        return 'media_decode'
    return None


def provider_binding_proof(config):
    """Verify the current saved binding locally; never invoke or reveal a key.

    This proves an existing verification receipt still matches the current
    adapter/connection and its private credential. It makes no fresh quota or
    network-health claim. Recovered execution keeps request-cache reuse intact.
    """
    from .mllm_provider import connection_identity, vision_request_identity
    from .provider_connection import adapter_identity, verification_matches
    from .provider_credentials import model_api_key, read_revision
    settings = config.get('mllm') or {}
    reference = settings.get('credential_ref')
    if not reference:
        raise ValueError('A saved verified provider binding is required')
    saved = read_revision(reference)
    verification = saved.get('verification') or {}
    if not verification_matches(settings, verification) or not model_api_key(settings):
        raise ValueError('Current provider binding is not verified')
    return {'connection_sha256': connection_identity(settings),
            'request_policy_sha256': vision_request_identity(settings),
            'adapter_sha256': adapter_identity(),
            'credential_revision_sha256': _digest(reference),
            'verification_sha256': _digest({key: verification.get(key) for key in
                ('status', 'model_invocation', 'connection_sha256', 'adapter_sha256', 'checked_at')})}


def _validate_evidence(evidence, row, result, *, config=None):
    record = json.loads(row['payload'])
    kind = evidence.get('repair_kind')
    allowed = {'schema_version', 'repair_kind', 'recording_id', 'revision', 'source_signature',
               'repair_revision', 'checks', 'provider_binding'}
    if (set(evidence) - allowed or evidence.get('schema_version') != EVIDENCE_SCHEMA or kind not in CHECKS
            or failure_kind(result) != kind
            or evidence.get('recording_id') != row['recording_id']
            or evidence.get('revision') != row['revision']
            or not record.get('source_signature')
            or evidence.get('source_signature') != record['source_signature']
            or not isinstance(evidence.get('repair_revision'), str)
            or re.fullmatch(r'[0-9a-f]{64}', evidence['repair_revision']) is None):
        raise ValueError('Repair evidence does not match this failed input and cause')
    required = CHECKS[kind] | {'current_source_verified', 'successful_requests_preserved'}
    checks = evidence.get('checks') or {}
    if set(checks) != required or any(value != 'PROVEN' for value in checks.values()):
        raise ValueError('Repair evidence is missing a required verification')
    if kind == 'provider_binding':
        if config is None or evidence.get('provider_binding') != provider_binding_proof(config):
            raise ValueError('Repair evidence does not match the current verified provider binding')
    elif 'provider_binding' in evidence:
        raise ValueError('Provider binding is not evidence for this repair class')
    # A stable repair identity prevents the same fix from extending the budget
    # again after another failure. It excludes wall-clock observation fields.
    return _digest({'kind': kind, 'repair_revision': evidence['repair_revision'],
                    'source_signature': evidence['source_signature'],
                    'provider_binding': evidence.get('provider_binding')})


def plan_retry(path, recording_id, evidence, *, baseline=3, budget=3, config=None):
    """Read-only plan for one exhausted failure and a verified specific repair."""
    if type(baseline) is not int or baseline < 1 or type(budget) is not int or not 1 <= budget <= 3:
        raise ValueError('Retry budgets must be positive and at most three additional attempts')
    with connection(Path(path), readonly=True) as db:
        row = db.execute('SELECT * FROM recordings WHERE recording_id=?', (recording_id,)).fetchone()
        if row is None:
            raise ValueError('Recording is not in this queue')
        row = dict(row)
        if (row['status'] != 'failed' or row['input_status'] != 'ready'
                or row['attempts'] < retry_limit(row, baseline)):
            raise ValueError('Only an input-ready exhausted failure may receive a repair budget')
        result = json.loads(row['result'] or '{}')
        repair_id = _validate_evidence(evidence, row, result, config=config)
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='retry_authorizations'").fetchone():
            if db.execute('SELECT 1 FROM retry_authorizations WHERE recording_id=? AND revision=? AND repair_id=?',
                          (recording_id, row['revision'], repair_id)).fetchone():
                raise ValueError('This verified repair has already authorized its retry budget')
    return {'schema_version': SCHEMA, 'queue_path': str(Path(path).resolve()), 'recording_id': recording_id,
            'revision': row['revision'], 'source_signature': evidence['source_signature'],
            'expected_attempts': row['attempts'], 'expected_retry_limit': row.get('retry_limit', 0),
            'failure_sha256': _digest(result), 'repair_id': repair_id,
            'baseline': baseline, 'budget': budget, 'retry_limit': row['attempts'] + budget,
            'evidence': evidence}


def apply_retry_plan(queue, plan, *, config=None):
    """CAS grant; retain original failure in the same local queue transaction.

    No task is marked successful, attempt count reset, execution identity
    changed, or request/result file removed. A newly claimed execution still
    validates source and parent receipts and reuses matching paid results.
    """
    if plan.get('schema_version') != SCHEMA:
        raise ValueError('Unknown retry plan schema')
    if plan.get('queue_path') != str(queue.path.resolve()):
        raise ValueError('Retry plan belongs to a different local queue')
    budget, baseline = plan.get('budget'), plan.get('baseline')
    if type(budget) is not int or not 1 <= budget <= 3 or type(baseline) is not int or baseline < 1:
        raise ValueError('Invalid retry budget')
    with queue.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM recordings WHERE recording_id=?', (plan.get('recording_id'),)).fetchone()
        if row is None:
            return False
        row = dict(row)
        result = json.loads(row['result'] or '{}')
        if (row['status'] != 'failed' or row['input_status'] != 'ready'
                or row['revision'] != plan.get('revision')
                or row['attempts'] != plan.get('expected_attempts')
                or row.get('retry_limit', 0) != plan.get('expected_retry_limit')
                or row['attempts'] < retry_limit(row, baseline)
                or _digest(result) != plan.get('failure_sha256')
                or plan.get('retry_limit') != row['attempts'] + budget):
            return False
        repair_id = _validate_evidence(plan.get('evidence') or {}, row, result, config=config)
        if repair_id != plan.get('repair_id'):
            raise ValueError('Retry plan repair identity changed')
        if plan.get('source_signature') != json.loads(row['payload']).get('source_signature'):
            return False
        changed = db.execute('INSERT OR IGNORE INTO retry_authorizations VALUES(?,?,?,?,?,?,?,?,?)',
            (row['recording_id'], row['revision'], repair_id, time.time(), row['attempts'],
             row.get('retry_limit', 0), plan['retry_limit'], row['result'] or '{}',
             json.dumps(plan['evidence'], sort_keys=True, ensure_ascii=False))).rowcount
        if not changed:
            return False
        db.execute('UPDATE recordings SET retry_limit=? WHERE recording_id=?',
                   (plan['retry_limit'], row['recording_id']))
        from .task_events import append
        append(db, row['recording_id'], 'retry_authorized', revision=row['revision'], attempt=row['attempts'],
               data={'repair_kind': plan['evidence']['repair_kind'], 'repair_id': repair_id,
                     'failure_sha256': plan['failure_sha256'], 'retry_limit': plan['retry_limit'],
                     'additional_attempts': budget})
        return True


def main(argv=None):
    """Local operator CLI; provider repairs revalidate saved credentials only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('plan', 'apply'))
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--recording-id')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--budget', type=int, default=3)
    parser.add_argument('--baseline', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        config = None
        if args.config:
            from .config import load_config
            from .ai_settings import apply_active
            config = apply_active(load_config(args.config))
        if args.operation == 'plan':
            if not args.evidence or not args.recording_id:
                parser.error('plan requires --evidence and --recording-id')
            value = plan_retry(args.queue, args.recording_id, json.loads(args.evidence.read_text()),
                               baseline=args.baseline, budget=args.budget, config=config)
            from .device_day_contract import atomic_json
            atomic_json(args.plan, value)
            print(json.dumps({'status': 'planned', 'recording_id': value['recording_id'],
                              'retry_limit': value['retry_limit'], 'repair_id': value['repair_id']}))
        else:
            from .device_day_queue import DeviceDayQueue
            if not args.queue.is_file():
                raise ValueError('Queue does not exist')
            value = json.loads(args.plan.read_text())
            applied = apply_retry_plan(DeviceDayQueue(args.queue), value, config=config)
            print(json.dumps({'status': 'applied' if applied else 'stale_or_already_applied',
                              'recording_id': value.get('recording_id')}))
            return 0 if applied else 2
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        # Never echo exception text from private config/credential readers.
        print(json.dumps({'status': 'rejected', 'reason': 'repair_evidence_or_current_binding_unverified'}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
