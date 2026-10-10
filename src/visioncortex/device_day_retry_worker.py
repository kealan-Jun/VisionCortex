"""Bounded automatic repair grants; model execution remains with stage workers.

An explicitly configured verification port owns site-specific current source,
transitive parent, context and repair verification. It must verify afresh and
recheck immediately before the queue CAS. Stored claims are never sufficient.
This worker neither invokes models nor changes receipts or paid request caches.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import re
import time

from .device_day_contract import DEPENDENCIES, STAGES, digest
from .device_day_retry import CHECKS, EVIDENCE_SCHEMA, apply_retry_plan, failure_kind, plan_retry, provider_binding_proof
from .local_storage import require_local_path
from .sqlite_store import connection

VERSION = 'visioncortex-device-day-retry-verification/1'
RECHECK_VERSION = 'visioncortex-device-day-retry-recheck/1'
SOURCE_CHECK = 'complete_stable_bytes_and_original_source_snapshot'
CACHE_CHECK = 'no_mutation_and_unchanged_reviewed_request_cache_recipe'
# Decoder and content/quality failures need their own explicit execution proof;
# they are deliberately absent from automatic infrastructure repair.
KINDS = frozenset({'provider_binding', 'database_lock', 'storage_access', 'vision_nms',
                   'receipt_projection', 'resource_coordination', 'storage_capacity'})


def rejection_reason(verifier, exc):
    """The selected owner may expose fixed reason slugs, never exception text."""
    method = getattr(verifier, 'rejection_reason', None)
    try:
        reason = method(exc) if callable(method) else None
    except Exception:
        reason = None
    if isinstance(reason, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,95}', reason):
        return reason
    if type(exc) is ValueError:
        known = {'This verified repair has already authorized its retry budget': 'repair_already_authorized',
                 'Only an input-ready exhausted failure may receive a repair budget': 'failed_budget_state_changed',
                 'Current source verification is incomplete or stale': 'current_source_proof_incomplete_or_stale',
                 'Current parent or context verification is incomplete': 'current_parent_context_proof_incomplete',
                 'Specific repair verification is incomplete': 'specific_repair_proof_incomplete',
                 'Current verification changed before grant': 'current_proof_changed_before_grant'}
        return known.get(str(exc), 'current_verification_rejected')
    return 'current_verification_rejected'


def settings(config):
    value = (config.get('device_day') or {}).get('retry_repair') or {}
    allowed = value.get('allowed_classes', [])
    if not isinstance(allowed, list) or any(kind not in KINDS for kind in allowed) or len(set(allowed)) != len(allowed):
        raise ValueError('Automatic repair classes must be explicitly configured')
    result = value | {'allowed_classes': allowed}
    for name, default, lower, upper in (('batch_size', 128, 5, 128), ('budget', 1, 1, 3),
                                       ('cooldown_seconds', 900, 5, 86400), ('interval_seconds', 5, 1, 300)):
        item = value.get(name, default)
        if type(item) is not int or not lower <= item <= upper:
            raise ValueError('Automatic repair bounds are invalid')
        result[name] = item
    return result


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _fresh(value, now):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= now - value <= 30


def _parents(stage):
    needed = set(DEPENDENCIES[stage])
    for _ in STAGES:
        needed |= {parent for current in needed for parent in DEPENDENCIES[current]}
    return needed


def verify_proof(proof, row, stage, kind, config, *, now=None):
    """Validate a fresh exact-input proof from the selected verification owner.

    Hash-only fields keep arbitrary private config/messages out of grant logs.
    The repair identity consists of reviewed implementation, policy and verifier
    recipe digests, never observation timestamps or regenerated trial receipts.
    """
    now = time.time() if now is None else now
    fields = {'schema_version', 'status', 'verified_at', 'stage', 'queue_row_sha256', 'source_signature',
              'inventory_sha256', 'source_proof_sha256', 'current_queue_revision', 'current_model_key',
              'parent_receipt_sha256', 'parent_execution_sha256', 'context_file_sha256',
              'verified_context_sha256', 'source_verification', 'cache_preservation_scope',
              'model_invoked', 'capture_modified', 'cache_modified', 'repair'}
    record = json.loads(row['payload'])
    if (not isinstance(proof, dict) or set(proof) != fields or proof['schema_version'] != VERSION
            or proof['status'] != 'PROVEN' or not _fresh(proof['verified_at'], now)
            or proof['stage'] != stage or proof['queue_row_sha256'] != digest(row)
            or proof['source_signature'] != record.get('source_signature')
            or proof['current_queue_revision'] != row['revision']
            or proof['source_verification'] != SOURCE_CHECK or proof['cache_preservation_scope'] != CACHE_CHECK
            or any(proof[name] is not False for name in ('model_invoked', 'capture_modified', 'cache_modified'))
            or any(not _hash(proof[name]) for name in ('inventory_sha256', 'source_proof_sha256',
                                                     'current_model_key', 'verified_context_sha256'))):
        raise ValueError('Current source verification is incomplete or stale')
    parents, receipts, context = proof['parent_execution_sha256'], proof['parent_receipt_sha256'], proof['context_file_sha256']
    if (not isinstance(parents, dict) or set(parents) != _parents(stage)
            or not isinstance(receipts, dict) or not _parents(stage) <= set(receipts)
            or set(receipts) - _parents(stage) - {'publication'}
            or any(not _hash(value) for value in [*parents.values(), *receipts.values()])
            or not isinstance(context, dict) or set(context) != ({'Comment.jsonl', 'Protocol.json'} if stage == 'understanding' else set())
            or any(value is not None and not _hash(value) for value in context.values())):
        raise ValueError('Current parent or context verification is incomplete')
    repair = proof['repair']
    if (not isinstance(repair, dict) or set(repair) != {'identity', 'checks', 'validated_model_repair'}
            or repair['checks'] != {check: 'PROVEN' for check in CHECKS[kind]}):
        raise ValueError('Specific repair verification is incomplete')
    identity = repair['identity']
    identity_fields = {'repair_kind', 'implementation_sha256', 'policy_sha256', 'verification_recipe_sha256'}
    if kind == 'vision_nms':
        identity_fields.add('weights_sha256')
    if (not isinstance(identity, dict) or set(identity) != identity_fields
            or identity['repair_kind'] != kind
            or any(not _hash(identity[name]) for name in identity_fields - {'repair_kind'})):
        raise ValueError('Stable repair revision is missing')
    actual = repair['validated_model_repair']
    if kind == 'vision_nms':
        if (stage != 'vision' or not isinstance(actual, dict)
                or set(actual) != {'status', 'receipt_sha256', 'model_owner_sha256', 'weights_sha256', 'runtime_policy_sha256'}
                or actual['status'] != 'PROVEN' or not _hash(actual['receipt_sha256'])
                or actual['model_owner_sha256'] != identity['implementation_sha256']
                or actual['weights_sha256'] != identity['weights_sha256']
                or actual['runtime_policy_sha256'] != identity['policy_sha256']):
            raise ValueError('NMS repair requires actual model proof for the current systemic fix')
    elif actual is not None:
        raise ValueError('Model evidence is not an infrastructure repair')
    evidence = {'schema_version': EVIDENCE_SCHEMA, 'repair_kind': kind, 'recording_id': row['recording_id'],
                'revision': row['revision'], 'source_signature': record['source_signature'],
                'repair_revision': digest(identity), 'checks': repair['checks'] |
                {'current_source_verified': 'PROVEN', 'successful_requests_preserved': 'PROVEN'}}
    if kind == 'vision_nms':
        # The policy digest already binds the validated model owner/weights.
        # Reuse the same reviewed identity as a prior manual NMS grant so moving
        # it to automatic repair cannot authorize the same fix a second time.
        evidence['repair_revision'] = identity['policy_sha256']
    if kind == 'provider_binding':
        if stage != 'understanding':
            raise ValueError('Provider binding repair is understanding only')
        evidence['provider_binding'] = provider_binding_proof(config)
        # Unrelated control or verifier upgrades must not replenish the same
        # repaired credential binding's budget.
        evidence['repair_revision'] = digest({'provider_binding': evidence['provider_binding']})
    return evidence


class RetryRepair:
    def __init__(self, config, verifier):
        self.config, self.verifier, self.options = config, verifier, settings(config)
        self.root = require_local_path(Path(config['storage']['local_runtime_root'])) / 'device-day'
        self.path = require_local_path(self.root / 'RetryRepair' / 'Checks.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript('PRAGMA journal_mode=WAL; '
                            'CREATE TABLE IF NOT EXISTS checks(identity TEXT PRIMARY KEY,next_check REAL,status TEXT); '
                            'CREATE TABLE IF NOT EXISTS cursors(stage TEXT PRIMARY KEY,position INTEGER);')

    def _checkpoint(self, identity, status):
        with connection(self.path) as db:
            db.execute('DELETE FROM checks WHERE next_check<?', (time.time(),))
            db.execute('INSERT OR REPLACE INTO checks VALUES(?,?,?)',
                       (identity, time.time() + self.options['cooldown_seconds'], status))

    def _select(self, stop):
        from .device_day_schedule import in_processing_scope
        from .runtime_control import check_cancelled
        counts = {'scanned': 0, 'outside_scope': 0, 'unknown_or_disabled': 0, 'cooling': 0}
        with connection(self.path) as db:
            cursors = dict(db.execute('SELECT stage,position FROM cursors'))
        start = cursors.get('_next_stage', 0) % len(STAGES)
        limit = self.options['batch_size'] // len(STAGES)
        baseline = (self.config.get('device_day') or {}).get('failure_retry_limit', 3)
        for offset in range(len(STAGES)):
            stage = STAGES[(start + offset) % len(STAGES)]
            path = require_local_path(self.root / f'queue-{stage}.sqlite3')
            if not path.is_file():
                continue
            with connection(path, readonly=True, timeout=2) as db:
                columns = {row[1] for row in db.execute('PRAGMA table_info(recordings)')}
                retry = 'retry_limit' if 'retry_limit' in columns else '0'
                rows = [dict(row) for row in db.execute(
                    f"SELECT rowid AS _scan_rowid,* FROM recordings WHERE rowid>? AND status='failed' "
                    f"AND input_status='ready' AND attempts>=MAX(?,{retry}) ORDER BY rowid LIMIT ?",
                    (cursors.get(stage, 0), baseline, limit))]
            if not rows:
                with connection(self.path) as db:
                    db.execute('INSERT OR REPLACE INTO cursors VALUES(?,0)', (stage,))
                continue
            for row in rows:
                check_cancelled(stop)
                position = row.pop('_scan_rowid')
                with connection(self.path) as db:
                    db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)', (stage, position))
                counts['scanned'] += 1
                if not in_processing_scope(self.config.get('device_day') or {}, json.loads(row['payload'])):
                    counts['outside_scope'] += 1
                    continue
                failure = json.loads(row['result'] or '{}')
                kind = failure_kind(failure)
                if kind == 'vision_nms' and (failure.get('error_type') != 'RuntimeError'
                        or failure.get('message') != 'NMS postprocessing repeatedly timed out; refusing incomplete frame evidence'):
                    kind = None
                if kind not in self.options['allowed_classes']:
                    counts['unknown_or_disabled'] += 1
                    continue
                identity = digest({'stage': stage, 'row': row})
                with connection(self.path, readonly=True) as db:
                    prior = db.execute('SELECT next_check FROM checks WHERE identity=?', (identity,)).fetchone()
                if prior and prior['next_check'] > time.time():
                    counts['cooling'] += 1
                    continue
                with connection(self.path) as db:
                    db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)', ('_next_stage', (start + offset + 1) % len(STAGES)))
                return (stage, path, row, kind, identity), counts
        with connection(self.path) as db:
            db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)', ('_next_stage', (start + 1) % len(STAGES)))
        return None, counts

    def tick(self, stop):
        from .device_day import exclusive
        from .device_day_io import slot
        from .device_day_queue import DeviceDayQueue
        from .device_day_recovery_worker import verify_storage
        from .runtime_control import check_cancelled, execution_context, resource_slot
        try:
            with exclusive(self.root / 'locks' / 'retry-repair-worker.lock'):
                candidate, counts = self._select(stop)
                if candidate is None:
                    return {'status': 'waiting', **counts}
                stage, path, row, kind, identity = candidate
                try:
                    if re.fullmatch(r'[A-Za-z0-9_-]{1,128}', row['recording_id']) is None:
                        raise ValueError('Recording lock identity is invalid')
                    # The source namespace and exact mount guard precede every
                    # archive read, including site verifier initialization.
                    verify_storage(self.config)
                    guarded = self.verifier.verify_source_view(self.config, stop)
                    if not isinstance(guarded, dict) or guarded.get('status') != 'PROVEN':
                        raise ValueError('Source namespace is not verified')
                    with exclusive(self.root / 'locks' / (row['recording_id'] + '.' + stage + '.lock')), execution_context(
                            job_id='retry-repair:' + row['recording_id'], source='repair_verification',
                            priority=2, stop=stop, yield_signal=stop):
                        with resource_slot(self.config, 'cpu'), slot(self.config):
                            proof = self.verifier.verify(self.config, stage, row, kind, stop)
                            evidence = verify_proof(proof, row, stage, kind, self.config)
                            plan = plan_retry(path, row['recording_id'], evidence, budget=self.options['budget'],
                                              baseline=self.config.get('device_day', {}).get('failure_retry_limit', 3),
                                              config=self.config)
                            queue = DeviceDayQueue(path)
                            verify_storage(self.config)
                            guarded = self.verifier.verify_source_view(self.config, stop)
                            if not isinstance(guarded, dict) or guarded.get('status') != 'PROVEN':
                                raise ValueError('Source namespace changed before grant')
                            rechecked = self.verifier.recheck(self.config, stage, row, proof, stop)
                            if (not isinstance(rechecked, dict) or set(rechecked) != {'schema_version', 'status',
                                    'checked_at', 'verification_sha256', 'queue_row_sha256'}
                                    or rechecked['schema_version'] != RECHECK_VERSION or rechecked['status'] != 'PROVEN'
                                    or not _fresh(rechecked['checked_at'], time.time())
                                    or rechecked['verification_sha256'] != digest(proof)
                                    or rechecked['queue_row_sha256'] != digest(row)):
                                raise ValueError('Current verification changed before grant')
                            check_cancelled(stop)
                            applied = apply_retry_plan(queue, plan, config=self.config)
                    status = 'granted' if applied else 'stale_or_already_applied'
                    result = {'status': status, 'stage': stage, 'recording_id': row['recording_id'],
                              'repair_kind': kind, 'repair_id': plan['repair_id'], 'retry_limit': plan['retry_limit']}
                except Exception as exc:
                    from .runtime_control import ExecutionCancelled
                    if isinstance(exc, ExecutionCancelled):
                        raise
                    status = 'verification_rejected'
                    result = {'status': status, 'stage': stage, 'recording_id': row['recording_id'],
                              'repair_kind': kind, 'error_type': type(exc).__name__,
                              'reason': rejection_reason(self.verifier, exc)}
                self._checkpoint(identity, status)
                return result | counts
        except BlockingIOError:
            return {'status': 'running_elsewhere'}


def load_verifier(config):
    """Explicit site composition; no discovery of private helpers or secrets."""
    entry = settings(config).get('verification_factory', '')
    if not isinstance(entry, str) or re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', entry) is None:
        raise ValueError('An explicit repair verification factory is required')
    module, name = entry.split(':')
    value = getattr(importlib.import_module(module), name)(config)
    if any(not callable(getattr(value, method, None)) for method in ('verify_source_view', 'verify', 'recheck')):
        raise ValueError('Repair verification owner is incomplete')
    return value


def retry_service_snapshot(config, *, now=None, proc_root=Path('/proc')):
    """Bounded local observer; owner freshness and budget grants are distinct."""
    from .device_day_consumers import process_start_ticks
    now = time.time() if now is None else now
    try:
        root = require_local_path(Path(config['storage']['local_runtime_root']))
        path = root / 'device-day/RetryRepair/Service.json'
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Repair service observation is a symlink')
        path = require_local_path(path)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return {'available': False, 'reason': 'repair_service_observation_unavailable', 'local_metadata_only': True}
    if not path.is_file():
        return {'available': False, 'reason': 'repair_service_observation_absent', 'local_metadata_only': True}
    try:
        if path.stat().st_size > 65536:
            raise ValueError('Repair service observation is oversized')
        value = json.loads(path.read_text())
        if value.get('schema_version') != 'visioncortex-retry-repair-worker/1':
            raise ValueError('Unknown repair service observation')
        stamp = value.get('updated_at')
        age = now - stamp if type(stamp) in (int, float) else float('inf')
        commit = (value.get('build') or {}).get('commit')
        current = process_start_ticks(value.get('pid'), proc_root)
        owner = 'unverified'
        if (math.isfinite(age) and 0 <= age <= 15 and current is not None
                and current == value.get('process_start_ticks') and isinstance(commit, str)
                and re.fullmatch('[0-9a-f]{40}', commit)):
            owner = 'verified' if value.get('status') not in {'stopped', 'failed'} else value['status']
        elif math.isfinite(age) and age > 15:
            owner = 'stale'
        counts = {name: value.get(name) for name in ('tick_count', 'completed_tick_count', 'granted_count', 'rejected_count')}
        if any(type(item) is not int or item < 0 for item in counts.values()):
            raise ValueError('Repair service counters unavailable')
        last = value.get('last_result') or {}
        actual_classes = value.get('allowed_classes')
        if not isinstance(actual_classes, list) or any(kind not in KINDS for kind in actual_classes):
            actual_classes = None
        build = value.get('build') or {}
        last_fields = {name: last[name] for name in ('scanned', 'outside_scope', 'unknown_or_disabled', 'cooling')
                       if type(last.get(name)) is int and last[name] >= 0}
        for name, valid in [('status', last.get('status') in {'granted', 'waiting', 'verification_rejected',
                             'stale_or_already_applied', 'running_elsewhere', 'rejected'}),
                            ('stage', last.get('stage') in STAGES), ('repair_kind', last.get('repair_kind') in KINDS)]:
            if valid:
                last_fields[name] = last[name]
        for name, pattern in [('recording_id', r'[A-Za-z0-9_-]{1,128}'),
                              ('error_type', r'[A-Z][A-Za-z0-9_]{0,63}'), ('reason', r'[a-z][a-z0-9_]{0,95}')]:
            if isinstance(last.get(name), str) and re.fullmatch(pattern, last[name]):
                last_fields[name] = last[name]
        return {'available': True, 'owner_state': owner, 'status': value.get('status'),
                'observed_at': stamp if type(stamp) in (int, float) and math.isfinite(stamp) else None,
                'phase': value.get('phase') if value.get('phase') in {'checking', 'waiting', 'cancelled'} else None,
                'pid': value.get('pid') if type(value.get('pid')) is int else None,
                'process_start_ticks': value.get('process_start_ticks') if type(value.get('process_start_ticks')) is int else None,
                'build': {name: build[name] for name in ('commit', 'source_digest') if
                          (name == 'commit' and isinstance(build.get(name), str) and re.fullmatch('[0-9a-f]{40}', build[name]))
                          or (name == 'source_digest' and _hash(build.get(name)))},
                'configured_enabled': value.get('configured_enabled') if type(value.get('configured_enabled')) is bool else None,
                'allowed_classes': actual_classes,
                'error_type': value.get('error_type') if isinstance(value.get('error_type'), str)
                              and re.fullmatch(r'[A-Z][A-Za-z0-9_]{0,63}', value['error_type']) else None,
                'heartbeat_age_seconds': max(0, age) if math.isfinite(age) else None, **counts,
                'last_result': last_fields,
                'local_metadata_only': True, 'proof_scope': 'verified_retry_budget_grants_not_stage_completion'}
    except (OSError, ValueError, TypeError, AttributeError):
        return {'available': False, 'reason': 'repair_service_observation_unavailable', 'local_metadata_only': True}


def serve(config_path, stop):
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day_recovery_worker import RecoveryStatus
    from .runtime_control import ExecutionCancelled
    worker, generation, root, status = None, None, None, None
    try:
        while not stop.is_set():
            interval = 5
            try:
                config = apply_active(load_config(require_local_path(Path(config_path))))
                current = require_local_path(Path(config['storage']['local_runtime_root'])) / 'device-day'
                if root is not None and current != root:
                    raise ValueError('Automatic repair runtime root changed')
                root = current
                if status is None:
                    status = RecoveryStatus(root)
                    status.path = root / 'RetryRepair' / 'Service.json'
                    status.state.update(schema_version='visioncortex-retry-repair-worker/1', granted_count=0,
                                        rejected_count=0, model_invoked=False, cache_modified=False, capture_modified=False)
                    status.__enter__()
                options = settings(config)
                interval = options['interval_seconds']
                status.publish(configured_enabled=options.get('enabled') is True,
                               allowed_classes=options['allowed_classes'])
                if options.get('enabled') is not True or not options['allowed_classes']:
                    status.publish(status='disabled')
                    stop.wait(options['interval_seconds'])
                    continue
                if generation != digest(config):
                    worker = RetryRepair(config, load_verifier(config))
                    generation = digest(config)
                status.publish(status='running', phase='checking', tick_started_at=time.time(),
                               tick_count=status.state['tick_count'] + 1)
                result = worker.tick(stop)
                status.publish(phase='waiting', last_tick_completed_at=time.time(), last_result=result,
                               completed_tick_count=status.state['completed_tick_count'] + 1,
                               granted_count=status.state['granted_count'] + int(result['status'] == 'granted'),
                               rejected_count=status.state['rejected_count'] + int(result['status'] == 'verification_rejected'))
            except ExecutionCancelled:
                break
            except Exception as exc:
                if status is None:
                    raise RuntimeError('Automatic repair startup failed: ' + type(exc).__name__) from None
                status.publish(status='failed', phase='waiting', error_type=type(exc).__name__)
            stop.wait(interval)
    finally:
        if status is not None:
            status.__exit__(None, None, None)


def run_once(config_path, stop):
    from .ai_settings import apply_active
    from .config import load_config
    config = apply_active(load_config(require_local_path(Path(config_path))))
    options = settings(config)
    if options.get('enabled') is not True or not options['allowed_classes']:
        return {'status': 'disabled'}
    return RetryRepair(config, load_verifier(config)).tick(stop)


def main(argv=None):
    import argparse
    import signal
    import threading
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--once', action='store_true', help='Verify at most one candidate and exit')
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if args.once:
            try:
                result = run_once(args.config, stop)
            except Exception as exc:
                result = {'status': 'rejected', 'error_type': type(exc).__name__}
            print(json.dumps(result | {'model_invoked': False, 'cache_modified': False, 'capture_modified': False}))
            return 2 if result['status'] in {'verification_rejected', 'rejected'} else 0
        serve(args.config, stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
