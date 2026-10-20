"""Continuously process ready slices through the shared production queues."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .device_day_contract import DEPENDENCIES, STAGES, atomic_json, digest, read_json
from .device_day_inplace import enqueue, retention_policy
from .device_day_retry import retry_limit
from .device_day_schedule import in_processing_scope
from .input_availability import configured_record, stage_input_state
from .sqlite_store import connection
from .device_day import DeviceDayRunner


def understanding_capacity(config, settings):
    """Honor both the stage setting and the shared provider admission ceiling."""
    values = [settings.get('understanding_workers', 1),
              (settings.get('stage_worker_limits') or {}).get('understanding', 64),
              ((config.get('runtime') or {}).get('resource_limits') or {}).get('cloud', 4)]
    if any(type(value) is not int or not 1 <= value <= 64 for value in values):
        raise ValueError('Understanding worker limits must be integers between 1 and 64')
    return min(values)


def concurrent_paid_history(settings, stage):
    policy = settings.get('backfill') or {}
    authorized = policy.get('paid_history_authorized', False)
    if type(authorized) is not bool:
        raise ValueError('backfill.paid_history_authorized must be a boolean')
    return (stage in {'understanding', 'report'} and authorized and policy.get('enabled') is True
            and stage in policy.get('stages', []))


def admitted_history_in_pool(settings, stage):
    """Keep verified older admissions in latest-first primary processing too."""
    if stage in {'understanding', 'report'}:
        return concurrent_paid_history(settings, stage)
    policy = settings.get('backfill') or {}
    return policy.get('enabled') is True and stage in policy.get('stages', [])


def foreground_history_config(config):
    """Use the catalog's newest-first order only for foreground selection.

    The existing idle selector orders newest first before its bounded LIMIT.
    This copied query configuration never reaches execution or the independent
    fair backfill lane; permission, input, lease and provider checks still run.
    """
    settings = config.get('device_day') or {}
    if not settings.get('latest_first', False):
        return config
    policy = settings.get('backfill') or {}
    return config | {'device_day': settings | {'backfill': policy | {'mode': 'idle'}}}


class DeferredPrimaryPublicationRunner(DeviceDayRunner):
    """Keep native primary queue completion independent of whole-day rendering.

    Native execute() has already saved the stage receipt and dirty journal.
    Returning False leaves formal publication pending; the standard replay
    still verifies source/artifacts and owns the completed publication receipt.
    CLI, legacy stages and multimodal/report publication retain the base path.
    """
    def refresh_index(self, layout, *, recording_id=None):
        from .device_day_activity import current_stage
        if (recording_id is not None and current_stage() in {'retention', 'vision', 'stt'}
                and (layout.receipts / recording_id / 'input-state.json').is_file()):
            return False
        return super().refresh_index(layout, recording_id=recording_id)


class RetentionWorker:
    """Bounded concurrent claims; existing execution still validates every input."""

    def __init__(self, *, stage='retention', failure_cooldown=60, recent_seconds=None, admission_cohort='live'):
        if stage not in STAGES:
            raise ValueError('Unknown device/day stage')
        self.stage = stage
        self.admission_cohort = admission_cohort
        self.recent_seconds = recent_seconds
        self.failure_cooldown = max(5, float(failure_cooldown))
        self.deferred = {}

    def _input_available(self, runner, record, *, states=None):
        if states is None:
            path = runner.runtime_root / 'InputAvailability.sqlite3'
            if not path.is_file():
                return True
            with connection(path, readonly=True) as db:
                row = db.execute('SELECT * FROM inputs WHERE id=?', (record['recording_id'],)).fetchone()
                state = dict(row) if row else {}
        else:
            state = states.get(record['recording_id'], {})
        return (state.get('signature') != record.get('source_signature')
                or stage_input_state(record, self.stage, state).get('state', 'ready') == 'ready')

    def candidates(self, runner, *, limit=16):
        path = runner.runtime_root / 'observed-inventory.sqlite3'
        if not path.is_file():
            return []
        current = time.time()
        queue = runner.queues.get(self.stage)
        authorized_history = concurrent_paid_history(runner.settings, self.stage)
        include_history = admitted_history_in_pool(runner.settings, self.stage)
        admissions = queue.admissions(include_history=True) if queue is not None and include_history else (
            queue.admissions() if queue is not None else {})
        # Ready paid history joins the same bounded pool as live work. Its
        # admission must not wait for the global preprocessing backfill lock.
        # The read-only selector checks source versions, parent completions,
        # monitor freshness, cooldowns and the existing retry budget; admit()
        # still verifies the actual receipts and artifact bytes below.
        eligible_history = {}
        if authorized_history and self.recent_seconds is not None:
            from .device_day_backfill_policy import historical_candidates
            try:
                eligible_history = {r['recording_id']: r['source_signature'] for r in historical_candidates(
                    foreground_history_config(runner.config), self.stage, limit=min(128, limit * 4),
                    exclude={rid for rid, until in self.deferred.items() if until > current})}
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Unavailable history observations cannot admit new work.
        from .device_day_cohort import recovery_entries
        restored = {rid: entry['source_signature'] for (stage, rid), entry in
                    recovery_entries(runner.settings).items() if stage == self.stage}
        preserved = admissions | restored | eligible_history
        with connection(path, readonly=True) as db:
            if self.recent_seconds is None:
                rows = db.execute('SELECT payload FROM observations')
            else:
                # The live lane must not decode the historical inventory on
                # every poll. Captures overlapping the window remain eligible.
                rows = db.execute("SELECT payload FROM observations WHERE "
                    "MAX(COALESCE(json_extract(payload,'$.recording_start_us'),0),"
                    "COALESCE(json_extract(payload,'$.recording_end_us'),0))>=? OR "
                    "json_extract(payload,'$.recording_id') IN (SELECT key FROM json_each(?))",
                    ((current - self.recent_seconds) * 1_000_000, json.dumps(preserved)))
            records = [json.loads(row[0]) for row in rows]
        if not records:
            return []
        ids = json.dumps([record['recording_id'] for record in records])
        selected = 'recording_id IN (SELECT value FROM json_each(?))'
        availability = {}
        availability_path = runner.runtime_root / 'InputAvailability.sqlite3'
        if availability_path.is_file():
            with connection(availability_path, readonly=True) as db:
                availability = {row['id']: dict(row) for row in db.execute(
                    'SELECT * FROM inputs WHERE id IN (SELECT value FROM json_each(?))', (ids,))}
        with connection(runner.queues[self.stage].path, readonly=True) as db:
            states = {r['recording_id']: dict(r) for r in db.execute(
                'SELECT recording_id,status,lease_until,attempts,retry_limit,updated_at,blocked_reason,'
                'first_ready_at,last_dispatch_at,queued_at,processing_cohort,'
                "json_extract(result,'$.retry_at') AS retry_at,"
                "json_extract(payload,'$.activity_kind') AS activity_kind,"
                "json_extract(payload,'$.source_signature') AS source_signature,"
                "json_extract(payload,'$.audio.source_signature') AS audio_signature,payload FROM recordings WHERE "
                + selected, (ids,))}
        visual = set()
        if self.stage == 'retention':
            with connection(runner.queues['vision'].path, readonly=True) as db:
                visual = {r[0] for r in db.execute("SELECT recording_id FROM recordings "
                    "WHERE status='completed' AND " + selected, (ids,))}
        ready = None
        if self.stage in {'understanding', 'report'}:
            requirements = {r['recording_id']: self.parents(runner, r) for r in records}
            completed_parents = {}
            for parent in set(p for needed in requirements.values() for p in needed):
                with connection(runner.queues[parent].path, readonly=True) as db:
                    completed = {r[0] for r in db.execute(
                        "SELECT recording_id FROM recordings WHERE status='completed' AND "
                        + selected, (ids,))}
                completed_parents[parent] = completed
            ready = {rid for rid, needed in requirements.items()
                     if all(rid in completed_parents[p] for p in needed)}
        maximum = runner.settings.get('failure_retry_limit') or 3
        candidates = []
        for record in records:
            rid = record['recording_id']
            bindings = getattr(runner, '_completed_execution_bindings', None)
            if bindings and bindings.historical_candidate('understanding', record):
                continue
            if ready is not None and rid not in ready:
                continue
            state = states.get(rid, {})
            if (state.get('retry_at') or 0) > current:
                continue
            if state.get('blocked_reason'):
                continue
            if (state.get('processing_cohort') == 'historical_backfill' and self.admission_cohort == 'live'
                    and restored.get(rid) != record.get('source_signature')
                    and not (include_history and (admissions | eligible_history).get(rid) == record.get('source_signature'))):
                continue
            same_video = state.get('source_signature') == record.get('source_signature')
            same_audio = state.get('audio_signature') == (record.get('audio') or {}).get('source_signature')
            from .device_day_inputs import same_stage_source, stage_input_ready
            same_source = (same_stage_source(json.loads(state['payload']), record, self.stage)
                           if self.stage == 'stt' and state.get('payload') else
                           same_video and (self.stage == 'vision' or same_audio))
            # Independent recorder audio may arrive after a no-audio receipt.
            # Only changed source metadata reopens a completion; admission and
            # execution still verify each source and preserve old receipts.
            completed_current = state.get('status') == 'completed' and same_source
            if (completed_current
                    or state.get('status') == 'running' and (state.get('lease_until') or 0) >= current
                    or state.get('status') == 'failed' and same_source and (
                        state.get('attempts', 0) >= retry_limit(state, maximum)
                        or current - state.get('updated_at', current) < self.failure_cooldown)
                    or self.deferred.get(rid, 0) > current):
                continue
            if not in_processing_scope(runner.settings, record):
                continue
            if self.recent_seconds is not None and max(record.get('recording_start_us', 0),
                    record.get('recording_end_us', 0)) < (current - self.recent_seconds) * 1_000_000 and (
                        preserved.get(rid) != record.get('source_signature')):
                continue
            record = configured_record(runner.config, record)
            if (not stage_input_ready(record, self.stage)
                    or not self._input_available(runner, record, states=availability)
                    or self.stage in {'vision', 'understanding', 'report'}
                    and record.get('configured_role') not in {'first_person', 'third_person'}):
                continue
            if self.stage in {'vision', 'understanding', 'report'}:
                order = (-record['recording_start_us'], record['camera_key'])
            elif self.stage == 'stt':
                audio = record.get('audio') or {}
                # No independent recorder audio is a runnable input, not a
                # missing prerequisite. transcribe() probes the sealed video
                # for an embedded track before it can publish no_audio.
                if audio.get('status', 'not_provided') not in {'provided', 'no_input', 'not_provided'}:
                    continue
                order = (-record['recording_start_us'], record['camera_key'])
            else:
                policy = retention_policy(runner, record)
                deadline = policy['deadline']
                urgent = policy['cleanup_risk'] or (deadline is not None and
                            current >= deadline - policy['safety_margin_seconds'])
                age_order = (-record['recording_start_us'], rid not in visual) if runner.settings.get('latest_first') else (
                    rid not in visual, -record['recording_start_us'])
                order = (not urgent, deadline if deadline is not None else float('inf'),
                         *age_order, record['camera_key'])
            candidates.append((order, record))
        # A camera contributes its newest ready slice; queue.claim retains
        # camera fairness and all running leases, including the main service's.
        scheduling = runner.queues[self.stage].scheduling
        if scheduling['enabled']:
            with connection(runner.queues[self.stage].path, readonly=True) as db:
                costs = {(r['kind'], r['identity']): r['seconds'] for r in db.execute('SELECT * FROM dispatch_costs')}
            def fair_order(item):
                order, record = item
                state = states.get(record['recording_id'], {})
                ready = state.get('last_dispatch_at') or state.get('first_ready_at') or state.get('queued_at') or current
                aged = current-ready >= scheduling['max_ready_wait_seconds']
                category = state.get('activity_kind') or state.get('processing_cohort') or 'unresolved'
                cost = costs.get(('class', category), 0) / scheduling['resource_shares'].get(category, 1)
                return (not aged, ready if aged else float('inf'), cost,
                        costs.get(('camera', record['camera_key']), 0), order)
            # Do not collapse one camera to its newest item before the durable
            # scheduler can see older admitted checkpoints.
            return [record for _, record in sorted(candidates, key=fair_order)[:limit]]
        chosen, cameras = [], set()
        for _, record in sorted(candidates, key=lambda item: item[0]):
            if record['camera_key'] in cameras:
                continue
            cameras.add(record['camera_key'])
            chosen.append(record)
            if len(chosen) >= limit:
                break
        return chosen

    def parents(self, runner=None, recording=None):
        from .device_day_contract import dependencies_for
        dependencies = dependencies_for(runner.settings, recording) if runner is not None else DEPENDENCIES
        needed = set(dependencies[self.stage])
        for _ in STAGES:
            needed.update(parent for stage in tuple(needed) for parent in dependencies[stage])
        return [stage for stage in STAGES if stage in needed]

    def _verified_prerequisites(self, runner, record, layout, context):
        from .device_day import visual_input
        prerequisites = {}
        for parent in self.parents(runner, record):
            retained = prerequisites.get('retention')
            inputs = (record if parent == 'retention' else visual_input(retained) if parent == 'vision'
                      else retained if parent == 'stt' else
                      {'vision': prerequisites.get('vision'), 'stt': prerequisites.get('stt'), 'context': context})
            key = runner._key(parent, record, inputs)
            loaded = runner._load(runner._receipt(layout, record, parent), key, layout)
            if loaded is None:
                return None
            prerequisites[parent] = loaded
        return prerequisites

    def _expired_completion(self, runner, record, expected):
        """Use the same prerequisite/key/artifact gates as ordinary admission."""
        from .device_day import load_context, visual_input
        from .device_day_contract import dependencies_for, file_hash
        from .device_day_inplace import active
        inventory = runner.runtime_root/'observed-inventory.sqlite3'
        if inventory.is_file():
            with connection(inventory, readonly=True) as db:
                observed = db.execute('SELECT payload FROM observations WHERE id=?', (record['recording_id'],)).fetchone()
            if observed is None or json.loads(observed[0]).get('source_signature') != record.get('source_signature'):
                return None, {'reason': 'current_source_not_admitted'}
        if self.stage not in {'understanding', 'report'} and active(runner, record):
            return None, {'reason': 'native_input_revalidation_required'}
        layout = runner.layout(record)
        context = (getattr(runner, '_context', load_context)(layout, record) if self.stage in {'understanding', 'report'}
                   else {'comments': [], 'protocol': None})
        prerequisites = self._verified_prerequisites(runner, record, layout, context)
        if prerequisites is None:
            return None, {'reason': 'current_prerequisite_not_verified'}
        dependencies = dependencies_for(runner.settings, record)
        inputs = record if self.stage == 'retention' else {p: prerequisites[p] for p in dependencies[self.stage]}
        if self.stage == 'vision':
            inputs = visual_input(inputs['retention'])
        queue_key = runner._key(self.stage, record, [inputs, context if self.stage == 'understanding' else None])
        execution_inputs = (record if self.stage == 'retention' else inputs if self.stage == 'vision'
            else prerequisites.get('retention') if self.stage == 'stt'
            else {'vision': prerequisites.get('vision'), 'stt': prerequisites.get('stt'), 'context': context}
                if self.stage == 'understanding' else prerequisites['understanding'])
        key = runner._key(self.stage, record, execution_inputs)
        path = runner._receipt(layout, record, self.stage)
        saved = runner._load(path, key, layout)
        if saved is None:
            return None, {'reason': 'current_stage_receipt_not_verified', 'expected_execution_key': key}
        if queue_key != expected['revision'] and not (
                runner._completed_stage_receipts.accepts(saved, key, queue_revision=expected['revision'])
                or runner._completed_execution_bindings.accepts(saved, key, queue_revision=expected['revision'])):
            return None, {'reason': 'current_queue_recipe_changed', 'current_queue_revision': queue_key}
        proof = {'status': 'verified', 'expected_execution_key': key,
                 'receipt_path': str(path), 'receipt_sha256': file_hash(path),
                 'receipt_key': saved['key'], 'verified_artifacts': len(saved['artifacts']),
                 'current_queue_revision': queue_key,
                 'queue_continuity': 'same_recipe' if queue_key == expected['revision'] else 'exact_manifest_pin'}
        return {'status': 'completed', 'recording_id': record['recording_id'], 'stage': self.stage,
                'archive': layout.name, 'formal_publication': 'pending_replay',
                'message': '已校验本地完成回执与产物，恢复过期阶段租约；未新增模型调用'}, proof

    def recover_expired(self, runner, *, limit=16):
        """Reconcile paid-stage process death before any candidate can be claimed."""
        from .device_day import exclusive
        from .device_day_contract import file_hash, safe_child
        queue = runner.queues[self.stage]
        if not queue.requires_expired_reconciliation:
            return {'completed': 0, 'result_unknown': 0, 'busy': 0}
        result = {'completed': 0, 'result_unknown': 0, 'busy': 0}
        for expected in queue.expired(limit=limit):
            record = json.loads(expected['payload'])
            bindings = getattr(runner, '_completed_execution_bindings', None)
            if bindings and bindings.historical_candidate('understanding', record):
                continue
            proof = {'model_invoked': False, 'queue_revision': expected['revision'],
                     'source_signature': record.get('source_signature')}
            try:
                with ExitStack() as locks:
                    for stage in ('all', self.stage):
                        locks.enter_context(exclusive(runner.runtime_root/'locks'/f'{record["recording_id"]}.{stage}.lock'))
                    completed, evidence = self._expired_completion(runner, record, expected)
                    proof.update(evidence)
                    if completed is None:
                        layout = runner.layout(record)
                        path = runner._receipt(layout, record, self.stage)
                        saved = read_json(path) if path.is_file() else {}
                        proof.update(receipt_status=saved.get('status', 'missing'),
                                     receipt_sha256=file_hash(path) if path.is_file() else None)
                        manifest = saved.get('sampling_coverage_manifest')
                        if not manifest and self.stage == 'understanding' and isinstance(saved.get('key'), str):
                            from .device_day_content_paths import private_folder
                            candidate = private_folder(layout, 'MultimodalUnderstanding', 'RequestPlans',
                                record['recording_id'], saved['key'])/'SamplingCoverage.json'
                            if candidate.is_file():
                                manifest = layout.relative(candidate)
                        proof['local_request_ledger'] = 'request_identity_not_available'
                        if manifest:
                            audit = read_json(safe_child(layout.root, manifest))
                            if audit.get('recording_id') != record['recording_id']:
                                raise ValueError('Expired lease request manifest crossed its recording')
                            from .multimodal_usage import UsageLedger
                            ledger = UsageLedger(runner.config, readonly=True)
                            keys = {r['request_key'] for r in audit.get('requests', []) if r.get('request_key')}
                            proof['local_request_ledger'] = {'identified_requests': len(keys),
                                'unresolved_requests': sum(ledger.unresolved(key) for key in keys),
                                'sampling_manifest_sha256': file_hash(safe_child(layout.root, manifest))}
                        completed = {'status': 'result_unknown', 'recording_id': record['recording_id'],
                            'stage': self.stage, 'reason': 'result_unknown',
                            'message': '原进程租约已过期，本地证据不足以确认完成或未提交；隔离等待本地回执恢复，禁止自动重发'}
                    if queue.recover_expired(expected, completed, proof):
                        result[completed['status']] += 1
            except BlockingIOError:
                result['busy'] += 1
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                # A missing/corrupt receipt remains unknown, while the exact
                # previous lease and attempt history remain recoverable.
                proof.update(reason='local_reconciliation_unavailable', error_type=type(exc).__name__)
                unknown = {'status': 'result_unknown', 'recording_id': record['recording_id'],
                           'stage': self.stage, 'reason': 'result_unknown', 'error_type': type(exc).__name__}
                if queue.recover_expired(expected, unknown, proof):
                    result['result_unknown'] += 1
        return result

    def admit(self, runner, record):
        bindings = getattr(runner, '_completed_execution_bindings', None)
        if bindings and bindings.historical_candidate('understanding', record):
            return False
        from .device_day import load_context, visual_input
        from .device_day_inplace import active, receipt
        from .device_day_inputs import same_stage_source, stage_input_ready
        if not stage_input_ready(record, self.stage) or not self._input_available(runner, record):
            return False
        # Durable waits apply to both foreground and historical admitters,
        # before rereading or hashing any parent artifacts on the NAS.
        with connection(runner.queues[self.stage].path, readonly=True) as db:
            waiting = db.execute("SELECT json_extract(result,'$.retry_at'),payload FROM recordings "
                                 "WHERE recording_id=?", (record['recording_id'],)).fetchone()
        if (waiting and same_stage_source(json.loads(waiting[1]), record, self.stage)
                and (waiting[0] or 0) > time.time()):
            return False
        native = active(runner, record)
        if self.stage not in {'understanding', 'report'} and native:
            admitted = enqueue(runner, record, self.stage)
            if admitted:
                self.save_admission(runner, record)
            return admitted
        if runner._completion_hold(record, self.stage):
            return False
        # The publication receipt belongs to the native inplace executor.
        # Historical executions retain their original verified parent receipts.
        if native and receipt(runner, record, 'publication').get('status') != 'completed':
            return False
        layout = runner.layout(record)
        record = record | {'archive_date': layout.name[:10]}
        context = (getattr(runner, '_context', load_context)(layout, record) if self.stage in {'understanding', 'report'}
                   else {'comments': [], 'protocol': None})
        prerequisites = self._verified_prerequisites(runner, record, layout, context)
        if prerequisites is None:
            return False
        # This is the ordinary single-record scheduling recipe. process()
        # rechecks publication, identities and all artifact bytes after claim;
        # it never runs a missing parent model for a single-stage request.
        from .device_day_contract import dependencies_for
        dependencies = dependencies_for(runner.settings, record)
        inputs = (record if self.stage == 'retention' else
                  {parent: prerequisites[parent] for parent in dependencies[self.stage]})
        if self.stage == 'vision':
            inputs = visual_input(inputs['retention'])
        if self.stage == 'understanding':
            visual = prerequisites.get('vision') or {}
            record = record | {'activity_kind': 'experiment_candidate' if any(
                s.get('activity') == 'active' for s in visual.get('segments', [])) else 'unresolved'}
        revision = runner._key(self.stage, record, [inputs, context if self.stage == 'understanding' else None])
        runner.queues[self.stage].enqueue(record, revision)
        runner.queues[self.stage].resume_prerequisite(record['recording_id'], revision)
        self.save_admission(runner, record)
        return True

    def save_admission(self, runner, record):
        from .device_day_cohort import recovery_entries
        entry = recovery_entries(runner.settings).get((self.stage, record['recording_id']))
        restored = entry and entry['source_signature'] == record['source_signature']
        cohort = self.admission_cohort
        if (cohort == 'live' and admitted_history_in_pool(runner.settings, self.stage)
                and max(record.get('recording_start_us', 0), record.get('recording_end_us', 0))
                < (time.time() - runner.settings.get('live_priority_seconds', 14400)) * 1_000_000):
            cohort = 'historical_backfill'
        runner.queues[self.stage].grant_admission(record,
            cohort='current_recovery' if restored else cohort,
            reason=entry['admission_reason'] if restored else 'ready_in_current_worker_scope')

    def _execute_claimed(self, runner, record, owner, stop):
        from .runtime_control import ExecutionCancelled, ExecutionYielded, execution_context
        from .multimodal_usage import SubmittedUnknown
        from .device_day_write_guard import NASUnavailable
        queue = runner.queues[self.stage]
        started = time.perf_counter()
        try:
            # Archive copies retain their finish-before-handover
            # semantics. Incremental work observes shutdown only at a
            # reusable unit boundary, without cancelling an issued call.
            incremental = self.stage in {'vision', 'understanding'}
            historical = False
            if admitted_history_in_pool(runner.settings, self.stage):
                with queue.connect() as db:
                    row = db.execute('SELECT processing_cohort FROM recordings WHERE recording_id=?',
                                     (record['recording_id'],)).fetchone()
                    historical = bool(row and row['processing_cohort'] == 'historical_backfill')
            context = (execution_context(job_id=record['recording_id'],
                                         source='device_day_backfill' if historical else 'nas',
                                         priority=20 if historical else 3, background_fair=historical,
                                         stop=None, yield_signal=stop)
                       if incremental else nullcontext())
            with context:
                if self.stage == 'vision':
                    from .device_day_io import live_vision_lane
                    with live_vision_lane():
                        result = runner.process(record, stage=self.stage, retry=True)
                else:
                    result = runner.process(record, stage=self.stage, retry=True)
            if incremental and result.get('status') == 'paused_for_live' and stop.is_set():
                result = result | {'reason': 'stopping'}
        except ExecutionYielded as exc:
            result = {'status': 'paused_for_live', 'recording_id': record['recording_id'],
                      'error_type': type(exc).__name__,
                      'reason': 'stopping' if stop.is_set() else exc.reason}
        except ExecutionCancelled as exc:
            result = {'status': 'cancelled', 'recording_id': record['recording_id'],
                      'error_type': type(exc).__name__, 'message': str(exc)[:1000]}
        except SubmittedUnknown as exc:
            result = {'status': 'submitted_unknown', 'recording_id': record['recording_id'],
                      'reason': exc.reason, 'error_type': type(exc).__name__}
        except NASUnavailable as exc:
            result = {'status': 'waiting_for_storage', 'recording_id': record['recording_id'],
                      'reason': exc.reason, 'error_type': type(exc).__name__}
        except Exception as exc:
            result = {'status': 'failed', 'recording_id': record['recording_id'],
                      'error_type': type(exc).__name__, 'message': str(exc)[:1000]}
        seconds = time.perf_counter() - started
        queue.finish(owner, record['recording_id'], result, seconds)
        if result.get('status') != 'completed' and not (
                self.stage in {'vision', 'understanding'} and result.get('status') in {'paused_for_live', 'scheduled_checkpoint'}):
            self.deferred[record['recording_id']] = time.time() + self.failure_cooldown
        return {'status': 'processed', 'recording_id': record['recording_id'],
                'wall_seconds': seconds, 'result': result}

    def tick(self, runner, stop):
        from .device_day import exclusive
        from .device_day_night_schedule import paused_stages, stage_admitted
        if self.stage in paused_stages(runner.config):
            return {'status': 'paused_by_user'}
        from .device_day_admission import admission_status
        hold = admission_status(runner.config, self.stage)
        if hold:
            return hold
        with ExitStack() as locks:
            try:
                locks.enter_context(exclusive(runner.runtime_root / 'locks' / f'{self.stage}-worker.lock'))
            except BlockingIOError:
                return {'status': 'running_elsewhere'}
            recovery = self.recover_expired(runner)
            recovery_fields = {'expired_lease_recovery': recovery} if any(recovery.values()) else {}
            if self.stage in {'understanding', 'report'}:
                if not stage_admitted(runner.config, self.stage):
                    return {'status': 'waiting_for_night_window'} | recovery_fields
                from .device_day_provider_gate import ProviderGate
                if ProviderGate(runner.config).blocks(self.stage):
                    return {'status': 'waiting_for_provider'} | recovery_fields
            admitted = set()
            for record in self.candidates(runner):
                if stop.is_set():
                    return {'status': 'stopping'}
                try:
                    if self.admit(runner, record):
                        admitted.add(record['recording_id'])
                    elif self.stage in {'understanding', 'report'}:
                        self.deferred[record['recording_id']] = time.time() + self.failure_cooldown
                except (OSError, ValueError, KeyError, TypeError):
                    self.deferred[record['recording_id']] = time.time() + self.failure_cooldown
            if not admitted or stop.is_set():
                return {'status': 'waiting', 'admitted': len(admitted)} | recovery_fields
            queue = runner.queues[self.stage]
            # The main service may have failed a candidate during admission.
            # Honor that fresh failure's cooldown before this worker retries it.
            with connection(queue.path, readonly=True) as db:
                for row in db.execute("SELECT recording_id,updated_at FROM recordings WHERE status='failed'"):
                    if time.time() - row['updated_at'] < self.failure_cooldown:
                        admitted.discard(row['recording_id'])
            owner = self.stage + '-worker-' + uuid.uuid4().hex
            # One older slice may still wait in the original worker's I/O
            # path. Speech can use its second configured global slot for the
            # newest slice; neither its lease nor its stage lock is disturbed.
            speech_capacity = ((runner.config.get('runtime') or {}).get('resource_limits') or {}).get('stt', 2)
            camera_limit = min(2, speech_capacity) if self.stage == 'stt' else 1
            if self.stage == 'understanding':
                # A long, all-frame slice already owned by the main service
                # must leave one lane for a newer slice. Provider requests
                # still acquire the same cross-process cloud resource slots.
                cloud_capacity = ((runner.config.get('runtime') or {}).get('resource_limits') or {}).get('cloud', 4)
                camera_limit = min(2, cloud_capacity)
            if self.stage == 'vision':
                vision_capacity = ((runner.config.get('runtime') or {}).get('resource_limits') or {}).get('vision', 12)
                camera_limit = min(vision_capacity, runner.settings.get('vision_jobs_per_camera', 1) + 1)
            jobs = understanding_capacity(runner.config, runner.settings) if self.stage == 'understanding' else 1
            if self.stage == 'vision':
                from .device_day_io import live_vision_capacity
                jobs = live_vision_capacity(runner.config)
                if jobs > 1:
                    camera_limit = 1
            with queue.heartbeat(owner):
                records = []
                for _ in range(jobs):
                    if stop.is_set():
                        break
                    record = queue.claim(owner, retry=True, allowed=admitted, camera_serial=True,
                                         camera_limit=camera_limit,
                                         max_attempts=runner.settings.get('failure_retry_limit') or 3)
                    if record is None:
                        break
                    records.append(record)
                if not records:
                    return {'status': 'waiting', 'admitted': len(admitted)}
                if len(records) == 1:
                    return self._execute_claimed(runner, records[0], owner, stop)
                with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix='device-' + self.stage) as pool:
                    futures = [pool.submit(copy_context().run, self._execute_claimed, runner, record, owner, stop)
                               for record in records]
                    results = [future.result() for future in futures]
                return {'status': 'processed', 'concurrency': jobs, 'results': results,
                        'result': {'status': 'completed' if all(r['result'].get('status') == 'completed'
                                                               for r in results) else 'partial'}}



def serve(config_path, stop, *, stage='retention'):
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day import DeviceDayRunner
    from .device_day_backfill import BackfillSupervisor, SUPPORTED_STAGES, options
    from .device_day_consumers import WorkerStatus
    # Preserve the recent admission lane. History uses its own bounded
    # scheduling policy and shares this runner's models, never its stage lock.
    worker = RetentionWorker(stage=stage, recent_seconds=14400 if stage in {'vision', 'understanding', 'report'} else None)
    runner, generation, roots, status_path, consumer = None, None, None, None, None
    backfill = BackfillSupervisor(stage, stop) if stage in SUPPORTED_STAGES else None
    from .device_day_synthesis_worker import DaySynthesisSupervisor
    synthesis = DaySynthesisSupervisor(stop) if stage == 'report' else None
    from .device_day_transcript_publication import TranscriptPublicationSupervisor
    transcripts = TranscriptPublicationSupervisor(stop) if stage == 'stt' else None
    synthesis_not_before = 0
    try:
        while not stop.is_set():
            delay = 5
            try:
                config = apply_active(load_config(Path(config_path)))
                settings = config.get('device_day') or {}
                if not settings.get('enabled') or not settings.get('inplace_preprocessing') or not settings.get('latest_first'):
                    raise ValueError('Continuous stages require enabled inplace latest-first processing')
                defer_publication = settings.get('defer_primary_publication', False)
                if type(defer_publication) is not bool:
                    raise ValueError('device_day.defer_primary_publication must be a boolean')
                current_roots = {key: config['storage'].get(key) for key in
                                 ('local_runtime_root', 'local_cache_root', 'archive_root')}
                if roots is not None and roots != current_roots:
                    raise ValueError('Retention worker storage roots changed')
                roots = current_roots
                policy = options(settings)
                enabled = backfill is not None and policy['enabled'] and stage in policy['stages']
                fair = enabled and policy.get('mode', 'idle') == 'fair'
                key = digest(config)
                if backfill is not None and (not enabled or generation != key):
                    backfill.request_yield()
                if runner is None or generation != key:
                    runner_type = (DeferredPrimaryPublicationRunner if defer_publication and stage in {
                        'retention', 'vision', 'stt'} else DeviceDayRunner)
                    if fair and stage == 'vision':
                        from .device_day_fair_backend import create_backend
                        runner = runner_type(config, backend=create_backend(config))
                    else:
                        runner = runner_type(config)
                    generation = key
                    if consumer is not None:
                        consumer.__exit__(None, None, None)
                    recent_lane = stage in {'vision', 'understanding', 'report'} or fair
                    worker.recent_seconds = settings.get('live_priority_seconds', 14400) if recent_lane else None
                    consumer = WorkerStatus(config, stage, recent_seconds=worker.recent_seconds,
                                            backfill_enabled=enabled).__enter__()
                directory = {'retention': 'RetentionWorker', 'stt': 'SpeechWorker', 'vision': 'VisionWorker',
                             'understanding': 'UnderstandingWorker', 'report': 'ReportWorker'}[stage]
                status_path = runner.runtime_root / directory / 'Service.json'
                atomic_json(status_path, {'status': 'running', 'updated_at': time.time(), 'stage': stage})
                transcript_status = transcripts.poll(runner) if transcripts is not None else None
                consumer.update(scope='recent' if worker.recent_seconds is not None else 'all',
                                recording_id=None, backfill_status='checking_live')
                # Poll the fair lane before a live tick can occupy this
                # process for a full recording. History may spend a long time
                # hashing or waiting for resources, so live admission continues
                # through the same queue leases and bounded resource pools.
                history = backfill.poll(runner, allow_start=True) if fair else None
                history_running = fair and history and history.get('status') == 'running'
                result = worker.tick(runner, stop)
                day_result = None
                if synthesis is not None:
                    day_result = (synthesis.poll(config) if time.time() >= synthesis_not_before else
                                  {'status': 'retry_backoff', 'retry_at': synthesis_not_before})
                    if day_result.get('status') in {'failed', 'resource_wait', 'waiting_for_provider_receipt', 'waiting'}:
                        # The supervisor may already have one bounded unit in
                        # flight. Avoid polling/resubmitting empty or failing
                        # whole-day scans continuously while stage work runs.
                        synthesis_not_before = time.time() + 60
                if backfill is not None and not fair:
                    allow_start = enabled and result.get('status') == 'waiting' and result.get('admitted', 0) == 0
                    history = backfill.poll(runner, allow_start=allow_start)
                if enabled:
                    delay = 5 if result.get('status') == 'waiting' and not result.get('admitted') else 1
                    if history and history.get('status') == 'processed':
                        delay = 0
                    consumer.update(backfill_status=(history.get('reason') or history['status']) if history else 'waiting_for_live')
                status = {'status': 'running', 'updated_at': time.time(), 'stage': stage,
                          'last_result': result, 'backfill_result': history, 'day_synthesis': day_result,
                          'transcript_publication': transcript_status}
                consumer.update(scope='history' if history_running else (
                                    'recent' if worker.recent_seconds is not None else 'all'),
                                last_result=history or result)
                if any(value and ((value.get('result') or {}).get('status') == 'completed'
                       or any((item.get('result') or {}).get('status') == 'completed'
                              for item in value.get('results', []))) for value in (result, history)):
                    consumer.update(last_success_at=time.time())
            except Exception as exc:
                if backfill is not None:
                    backfill.request_yield()
                status = {'status': 'failed', 'updated_at': time.time(), 'stage': stage,
                          'error_type': type(exc).__name__}
                if consumer is not None:
                    consumer.update(status='failed', error_type=type(exc).__name__)
            if status_path is not None:
                atomic_json(status_path, status)
            stop.wait(delay)
    finally:
        if transcripts is not None:
            transcripts.close()
        if synthesis is not None:
            synthesis.close()
        if backfill is not None:
            backfill.close()
        if consumer is not None:
            consumer.__exit__(None, None, None)
        if status_path is not None:
            atomic_json(status_path, {'status': 'stopped', 'updated_at': time.time(), 'stage': stage})


def main(argv=None):
    import argparse
    import signal
    import threading
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=STAGES, default='retention')
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        serve(args.config, stop, stage=args.stage)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
