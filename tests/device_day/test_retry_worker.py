"""Automatic grants cross fresh verification, bounded scanning and queue CAS."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from visioncortex.device_day_contract import digest
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retry import CHECKS, failure_kind
from visioncortex.device_day_retry_worker import (
    CACHE_CHECK, RECHECK_VERSION, SOURCE_CHECK, VERSION, RetryRepair, _parents, settings, verify_proof,
    main, rejection_reason, retry_service_snapshot, serve,
)


@contextmanager
def free_slot(*args, **kwargs):
    yield


@pytest.fixture(autouse=True)
def no_resource_wait(monkeypatch):
    monkeypatch.setattr('visioncortex.device_day_io.slot', free_slot)
    monkeypatch.setattr('visioncortex.runtime_control.resource_slot', free_slot)


def configuration(tmp_path, allowed=None):
    return {'storage': {'local_runtime_root': str(tmp_path / 'runtime')}, 'runtime': {'local_only': True},
            'device_day': {'enabled': True, 'failure_retry_limit': 3, 'retry_repair': {
                'enabled': True, 'allowed_classes': allowed or ['storage_access'], 'budget': 1,
                'batch_size': 5, 'cooldown_seconds': 5}}}


def exhausted(config, stage='vision', rid='record', result=None, start=20):
    queue = DeviceDayQueue(Path(config['storage']['local_runtime_root']) / 'device-day' / f'queue-{stage}.sqlite3')
    queue.enqueue({'recording_id': rid, 'configured_role': 'first_person', 'camera_key': 'camera',
                   'source_signature': 'source-' + rid, 'recording_start_us': start, 'processable': True}, 'revision-' + rid)
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=? WHERE recording_id=?",
                   (json.dumps(result or {'error_type': 'PermissionError', 'message': 'input denied'}), rid))
    return queue


def row(queue, rid='record'):
    with queue.connect() as db:
        return dict(db.execute('SELECT * FROM recordings WHERE recording_id=?', (rid,)).fetchone())


class Verifier:
    """Fixture port hashes real owned bytes and rechecks their complete binding."""
    def __init__(self, tmp_path):
        self.source = tmp_path / 'source'
        self.source.write_bytes(b'owned fixture source')
        self.calls, self.rechecks, self.guard_calls = [], [], 0
        self.after_verify = None
        self.pending = False
        self.mutate_proof = lambda value: value

    def verify_source_view(self, config, stop):
        self.guard_calls += 1
        return {'status': 'PROVEN'}

    def proof(self, stage, value, kind):
        checksum = hashlib.sha256(self.source.read_bytes()).hexdigest()
        identity = {'repair_kind': kind, 'implementation_sha256': 'a' * 64,
                    'policy_sha256': 'b' * 64, 'verification_recipe_sha256': 'c' * 64}
        actual = None
        if kind == 'vision_nms':
            identity['weights_sha256'] = 'd' * 64
            actual = {'status': 'PROVEN', 'receipt_sha256': 'e' * 64, 'model_owner_sha256': 'a' * 64,
                      'weights_sha256': 'd' * 64, 'runtime_policy_sha256': 'b' * 64}
        return {'schema_version': VERSION, 'status': 'PROVEN', 'verified_at': time.time(), 'stage': stage,
                'queue_row_sha256': digest(value), 'source_signature': json.loads(value['payload'])['source_signature'],
                'inventory_sha256': checksum, 'source_proof_sha256': checksum,
                'current_queue_revision': value['revision'], 'current_model_key': digest({'input': checksum, 'stage': stage}),
                'parent_receipt_sha256': {parent: checksum for parent in _parents(stage)},
                'parent_execution_sha256': {parent: checksum for parent in _parents(stage)},
                'context_file_sha256': {'Comment.jsonl': None, 'Protocol.json': checksum} if stage == 'understanding' else {},
                'verified_context_sha256': checksum, 'source_verification': SOURCE_CHECK,
                'cache_preservation_scope': CACHE_CHECK, 'model_invoked': False, 'capture_modified': False,
                'cache_modified': False, 'repair': {'identity': identity,
                    'checks': {check: 'PROVEN' for check in CHECKS[kind]}, 'validated_model_repair': actual}}

    def verify(self, config, stage, value, kind, stop):
        self.calls.append((stage, value['recording_id']))
        if self.pending:
            raise ValueError('parent not yet completed')
        result = self.mutate_proof(self.proof(stage, value, kind))
        if self.after_verify:
            self.after_verify()
        return result

    def recheck(self, config, stage, value, proof, stop):
        self.rechecks.append((stage, value['recording_id']))
        current = self.proof(stage, value, proof['repair']['identity']['repair_kind'])
        current['verified_at'] = proof['verified_at']
        if current != proof:
            raise ValueError('source parents or context changed')
        return {'schema_version': RECHECK_VERSION, 'status': 'PROVEN', 'checked_at': time.time(),
                'verification_sha256': digest(proof), 'queue_row_sha256': digest(value)}


def run_until(worker, wanted, limit=40):
    for _ in range(limit):
        result = worker.tick(threading.Event())
        if result['status'] == wanted:
            return result
    pytest.fail('Expected bounded repair result: ' + wanted)


def test_one_grant_preserves_failure_attempts_and_paid_cache_across_restart(tmp_path):
    config = configuration(tmp_path)
    queue = exhausted(config)
    paid = tmp_path / 'paid-result.json'
    paid.write_text('{"status":"completed","request_key":"same"}')
    before = row(queue)
    verifier = Verifier(tmp_path)
    result = RetryRepair(config, verifier).tick(threading.Event())
    assert result['status'] == 'granted' and len(verifier.calls) == len(verifier.rechecks) == 1
    after = row(queue)
    assert after == before | {'retry_limit': 4}
    assert paid.read_text() == '{"status":"completed","request_key":"same"}'
    assert queue.claim('ordinary-worker', retry=True, max_attempts=3)['recording_id'] == 'record'
    queue.finish('ordinary-worker', 'record', json.loads(before['result']), 1)
    run_until(RetryRepair(config, verifier), 'verification_rejected')
    assert row(queue)['retry_limit'] == 4 and row(queue)['attempts'] == 4
    with queue.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM retry_authorizations').fetchone()[0] == 1


@pytest.mark.parametrize('change', ['source', 'failed_row', 'stop'])
def test_immediate_recheck_or_cas_rejects_changed_input_without_grant(tmp_path, change):
    config = configuration(tmp_path)
    queue = exhausted(config)
    verifier, stop = Verifier(tmp_path), threading.Event()
    def alter():
        if change == 'source':
            verifier.source.write_bytes(b'changed after complete source proof')
        elif change == 'failed_row':
            with queue.connect() as db:
                db.execute("UPDATE recordings SET result='{\"error_type\":\"QualityError\"}'")
        else:
            stop.set()
    verifier.after_verify = alter
    if change == 'stop':
        from visioncortex.runtime_control import ExecutionCancelled
        with pytest.raises(ExecutionCancelled):
            RetryRepair(config, verifier).tick(stop)
    else:
        assert RetryRepair(config, verifier).tick(stop)['status'] in {'verification_rejected', 'stale_or_already_applied'}
    assert row(queue)['retry_limit'] == 0


@pytest.mark.parametrize('field,value', [('status', 'PARTIAL_EVIDENCE'), ('verified_at', 0),
    ('queue_row_sha256', '0' * 64), ('current_queue_revision', 'unknown-alias'), ('parent_execution_sha256', {}),
    ('source_verification', 'size_and_mtime_only'), ('model_invoked', True), ('cache_modified', True)])
def test_incomplete_or_stale_proof_never_grants(tmp_path, field, value):
    config = configuration(tmp_path)
    queue = exhausted(config)
    verifier = Verifier(tmp_path)
    verifier.mutate_proof = lambda proof: proof | {field: value}
    assert RetryRepair(config, verifier).tick(threading.Event())['status'] == 'verification_rejected'
    assert not verifier.rechecks and row(queue)['retry_limit'] == 0


def test_bounded_checkpoint_advances_past_old_unknowns_and_respects_cutoff(tmp_path):
    config = configuration(tmp_path)
    config['device_day']['process_since_us'] = 10
    queue = exhausted(config, rid='outside', start=1)
    for index in range(8):
        exhausted(config, rid=f'quality-{index}', result={'error_type': 'RuntimeError', 'message': 'coverage too small'})
    exhausted(config, rid='eligible')
    verifier = Verifier(tmp_path)
    worker = RetryRepair(config, verifier)
    results = []
    for _ in range(30):
        result = worker.tick(threading.Event())
        results.append(result)
        assert result['scanned'] <= 5
        if result['status'] == 'granted':
            break
    assert results[-1]['recording_id'] == 'eligible'
    assert sum(value['outside_scope'] for value in results) == 1
    assert verifier.calls == [('vision', 'eligible')]
    assert row(queue, 'outside')['retry_limit'] == 0


def test_pending_parent_is_rechecked_after_bounded_durable_cooldown(tmp_path):
    config = configuration(tmp_path)
    queue = exhausted(config)
    verifier = Verifier(tmp_path)
    verifier.pending = True
    worker = RetryRepair(config, verifier)
    assert worker.tick(threading.Event())['status'] == 'verification_rejected'
    verifier.pending = False
    restarted = RetryRepair(config, verifier)
    for _ in range(4):
        assert restarted.tick(threading.Event())['status'] == 'waiting'
    assert len(verifier.calls) == 1
    # Expire only the fixture checkpoint; no wall-clock scheduler assumptions.
    from visioncortex.sqlite_store import connection
    with connection(restarted.path) as db:
        db.execute('UPDATE checks SET next_check=0')
    assert run_until(restarted, 'granted')['recording_id'] == 'record'
    assert row(queue)['retry_limit'] == 4


def test_stages_rotate_with_at_most_one_grant_per_tick(tmp_path):
    config = configuration(tmp_path)
    for stage in ('retention', 'vision', 'stt', 'understanding', 'report'):
        exhausted(config, stage)
    verifier = Verifier(tmp_path)
    worker = RetryRepair(config, verifier)
    results = [worker.tick(threading.Event()) for _ in range(5)]
    assert [result['stage'] for result in results] == ['retention', 'vision', 'stt', 'understanding', 'report']
    assert all(result['status'] == 'granted' for result in results)


def test_systemic_nms_model_proof_allows_new_input_budget_without_claiming_it_was_inferred(tmp_path):
    config = configuration(tmp_path, ['vision_nms'])
    queue = exhausted(config, result={'error_type': 'RuntimeError',
        'message': 'NMS postprocessing repeatedly timed out; refusing incomplete frame evidence'})
    verifier = Verifier(tmp_path)
    result = RetryRepair(config, verifier).tick(threading.Event())
    assert result['status'] == 'granted' and row(queue)['status'] == 'failed'
    proof = verifier.proof('vision', row(queue), 'vision_nms')
    proof['repair']['validated_model_repair']['weights_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='actual model proof'):
        verify_proof(proof, row(queue), 'vision', 'vision_nms', config)


@pytest.mark.parametrize('error,message,kind', [
    ('OverflowError', 'string longer than INT_MAX bytes', 'receipt_projection'),
    ('RuntimeError', 'Resource lease was lost; output cannot be marked complete', 'resource_coordination'),
    ('TimeoutError', 'Resource admission timed out: vision', 'resource_coordination'),
    ('TimeoutError', 'Resource admission timed out: nas-io-read', 'resource_coordination'),
    ('OperationalError', 'database or disk is full', 'storage_capacity'),
    ('OSError', '[Errno 28] No space left on device', 'storage_capacity'),
])
def test_exact_framework_failure_requires_its_specific_checks(tmp_path, error, message, kind):
    config = configuration(tmp_path, [kind])
    queue = exhausted(config, result={'error_type': error, 'message': message})
    verifier = Verifier(tmp_path)
    assert failure_kind(json.loads(row(queue)['result'])) == kind
    broken = deepcopy(verifier.proof('vision', row(queue), kind))
    broken['repair']['checks'] = {'source_access_verified': 'PROVEN'}
    with pytest.raises(ValueError, match='Specific repair'):
        verify_proof(broken, row(queue), 'vision', kind, config)
    assert RetryRepair(config, verifier).tick(threading.Event())['status'] == 'granted'


@pytest.mark.parametrize('error,message', [('OverflowError', 'another overflow'),
    ('TimeoutError', 'Resource admission timed out: unknown'), ('RuntimeError', 'pixel equality check failed'),
    ('RuntimeError', 'Resource lease was lost'), ('OperationalError', 'malformed disk image')])
def test_unknown_or_quality_errors_stay_exhausted(tmp_path, error, message):
    config = configuration(tmp_path, list(CHECKS.keys() - {'media_decode'}))
    queue = exhausted(config, result={'error_type': error, 'message': message})
    verifier = Verifier(tmp_path)
    assert RetryRepair(config, verifier).tick(threading.Event())['status'] == 'waiting'
    assert not verifier.calls and row(queue)['retry_limit'] == 0


def test_explicit_classes_and_bounds_are_required(tmp_path):
    config = configuration(tmp_path)
    for change in ({'allowed_classes': ['quality_failure']}, {'batch_size': 129}, {'budget': 4}):
        with pytest.raises(ValueError):
            settings(config | {'device_day': {'retry_repair': config['device_day']['retry_repair'] | change}})


def test_manual_nms_same_canonical_fix_cannot_grant_again_after_auto_upgrade(tmp_path):
    from visioncortex.device_day_retry import apply_retry_plan, plan_retry
    config = configuration(tmp_path, ['vision_nms'])
    queue = exhausted(config, result={'error_type': 'RuntimeError',
        'message': 'NMS postprocessing repeatedly timed out; refusing incomplete frame evidence'})
    verifier = Verifier(tmp_path)
    before = row(queue)
    proof = verifier.proof('vision', before, 'vision_nms')
    evidence = verify_proof(proof, before, 'vision', 'vision_nms', config)
    assert evidence['repair_revision'] == proof['repair']['identity']['policy_sha256']
    assert apply_retry_plan(queue, plan_retry(queue.path, 'record', evidence, budget=1))
    queue.claim('ordinary-worker', retry=True, max_attempts=3)
    queue.finish('ordinary-worker', 'record', json.loads(before['result']), 1)
    assert RetryRepair(config, verifier).tick(threading.Event())['status'] == 'verification_rejected'
    assert row(queue)['retry_limit'] == 4


def test_storage_or_source_namespace_rejection_precedes_source_verification(tmp_path):
    config = configuration(tmp_path)
    queue = exhausted(config)
    verifier = Verifier(tmp_path)
    verifier.verify_source_view = lambda *_: {'status': 'NOT_PROVEN'}
    assert RetryRepair(config, verifier).tick(threading.Event())['status'] == 'verification_rejected'
    assert not verifier.calls and row(queue)['retry_limit'] == 0


def test_serve_actually_runs_grant_tick_and_publishes_completed_count(tmp_path, monkeypatch):
    from visioncortex.device_day_consumers import process_start_ticks
    config = configuration(tmp_path)
    queue = exhausted(config)
    verifier, stop = Verifier(tmp_path), threading.Event()
    monkeypatch.setattr('visioncortex.config.load_config', lambda _path: config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda value: value)
    monkeypatch.setattr('visioncortex.device_day_retry_worker.load_verifier', lambda _config: verifier)
    monkeypatch.setattr('visioncortex.build_identity.identity', lambda: {'commit': 'a' * 40})
    original = RetryRepair.tick
    def one_tick(worker, event):
        result = original(worker, event)
        event.set()
        return result
    monkeypatch.setattr(RetryRepair, 'tick', one_tick)
    serve(tmp_path / 'config.yaml', stop)
    snapshot = retry_service_snapshot(config)
    assert row(queue)['retry_limit'] == 4 and verifier.calls == [('vision', 'record')]
    assert snapshot['completed_tick_count'] == snapshot['granted_count'] == 1
    assert snapshot['owner_state'] == 'stopped'
    assert snapshot['configured_enabled'] is True and snapshot['allowed_classes'] == ['storage_access']
    assert snapshot['pid'] > 0 and snapshot['build']['commit'] == 'a' * 40
    assert snapshot['proof_scope'] == 'verified_retry_budget_grants_not_stage_completion'
    path = Path(config['storage']['local_runtime_root']) / 'device-day/RetryRepair/Service.json'
    value = json.loads(path.read_text())
    value.update(status='running', process_start_ticks=process_start_ticks(value['pid']) + 1)
    path.write_text(json.dumps(value))
    assert retry_service_snapshot(config)['owner_state'] == 'unverified'


def test_configured_interval_is_used_after_completed_tick(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config['device_day']['retry_repair']['interval_seconds'] = 17
    exhausted(config)
    monkeypatch.setattr('visioncortex.config.load_config', lambda _path: config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda value: value)
    monkeypatch.setattr('visioncortex.device_day_retry_worker.load_verifier', lambda _config: Verifier(tmp_path))
    monkeypatch.setattr('visioncortex.build_identity.identity', lambda: {'commit': 'a' * 40})
    class ImmediateStop(threading.Event):
        def __init__(self):
            super().__init__()
            self.waits = []
        def wait(self, timeout=None):
            self.waits.append(timeout)
            self.set()
            return True
    stop = ImmediateStop()
    serve(tmp_path / 'config.yaml', stop)
    assert stop.waits == [17]


def test_once_cli_runs_one_real_grant_with_redacted_output(tmp_path, monkeypatch, capsys):
    config = configuration(tmp_path)
    queue = exhausted(config)
    monkeypatch.setattr('visioncortex.config.load_config', lambda _path: config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda value: value)
    monkeypatch.setattr('visioncortex.device_day_retry_worker.load_verifier', lambda _config: Verifier(tmp_path))
    assert main(['--config', str(tmp_path / 'config.yaml'), '--once']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'granted' and result['model_invoked'] is False and result['cache_modified'] is False
    assert row(queue)['retry_limit'] == 4


def test_reason_codes_never_echo_generic_exception_text(tmp_path):
    verifier = Verifier(tmp_path)
    assert rejection_reason(verifier, ValueError('secret-provider-key-value')) == 'current_verification_rejected'
    verifier.rejection_reason = lambda _exc: 'secret-provider-key-value'
    assert rejection_reason(verifier, ValueError('another private value')) == 'current_verification_rejected'
    verifier.rejection_reason = lambda _exc: 'parent_queue_not_completed_retention'
    assert rejection_reason(verifier, ValueError('private value')) == 'parent_queue_not_completed_retention'


def test_observer_missing_config_is_unknown_and_file_symlink_is_unavailable(tmp_path):
    config = configuration(tmp_path)
    path = Path(config['storage']['local_runtime_root']) / 'device-day/RetryRepair/Service.json'
    path.parent.mkdir(parents=True)
    value = {'schema_version': 'visioncortex-retry-repair-worker/1', 'updated_at': time.time(), 'status': 'running',
             'tick_count': 1, 'completed_tick_count': 0, 'granted_count': 0, 'rejected_count': 0}
    path.write_text(json.dumps(value))
    snapshot = retry_service_snapshot(config)
    assert snapshot['configured_enabled'] is None and snapshot['allowed_classes'] is None
    real = tmp_path / 'real-state.json'
    real.write_text(path.read_text())
    path.unlink()
    path.symlink_to(real)
    assert retry_service_snapshot(config)['available'] is False
