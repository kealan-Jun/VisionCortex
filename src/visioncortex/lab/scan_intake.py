"""Consume saved QR evidence into the existing explicit operator scan command.

The decoder's completed job is reused. Background visibility, multiple codes and
an unknown source clock never acquire an operator/equipment association.
"""
from __future__ import annotations

from datetime import timedelta

from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, fingerprint, ident
from .media import Media
from .operator_scan import OperatorScans, _text, declaration_activity
from .timeline import Timeline

SCHEMA = 'vc-lab-scan-intake/1'
_FIELDS = {'intent', 'scan_kind', 'actor', 'reason', 'person_id', 'run_id', 'batch_id'}
_RETRY_SECONDS = (30, 120)
_CAUSAL_SCOPE_SQL = '''AND c.occurred_at<%s AND i.payload->>'camera_id'=%s
    AND regexp_replace(lower(i.payload->'scan_intent'->>'person_id'),
        'urn:uuid:|[-{}]', '', 'g') IS NOT DISTINCT FROM %s
    AND c.time_basis->>'uncertainty_us'='0'
    AND (c.time_basis->'clock_sync_verified'='true'::jsonb
        OR c.time_basis->'clock_sync_valid'='true'::jsonb)
    AND COALESCE(c.time_basis->'clock_sync_verified','null'::jsonb)<>'false'::jsonb
    AND COALESCE(c.time_basis->'clock_sync_valid','null'::jsonb)<>'false'::jsonb'''


def verified_clock(basis):
    return (isinstance(basis, dict) and bool(basis.get('basis'))
            and type(basis.get('uncertainty_us')) is int and basis['uncertainty_us'] == 0
            and (basis.get('clock_sync_verified') is True or basis.get('clock_sync_valid') is True)
            and basis.get('clock_sync_verified') is not False
            and basis.get('clock_sync_valid') is not False)


def _declaration(raw):
    if (type(raw) is not dict or set(raw) - _FIELDS
            or not {'intent', 'scan_kind', 'actor', 'reason'} <= set(raw)
            or raw.get('intent') != 'operator_scan'
            or type(raw.get('scan_kind')) is not str or raw['scan_kind'] not in {'equipment', 'material'}):
        raise ValueError('Unsupported explicit producer scan declaration')
    result = dict(raw)
    _text(result['actor'], 'scan source actor')
    _text(result['reason'], 'scan source reason')
    for key in ('person_id', 'run_id', 'batch_id'):
        if result.get(key) is not None:
            if type(result[key]) is not str:
                raise ValueError('Explicit scan identity must be a UUID string')
            result[key] = str(ident(result[key]))
    return result


class ScanIntake:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings

    def status(self):
        with self.core.db.transaction(readonly=True) as conn:
            equipment = conn.execute('SELECT state,count(*) AS count FROM equipment_identity WHERE customer_id=%s GROUP BY state',
                                     (self.core.customer,)).fetchall()
            scans = conn.execute('SELECT count(*) AS count FROM hj_operator_scan_event WHERE customer_id=%s',
                                 (self.core.customer,)).fetchone()['count']
            cameras = conn.execute("SELECT count(*) AS count FROM entity WHERE customer_id=%s AND kind='capture_device'",
                                   (self.core.customer,)).fetchone()['count']
        return dict(schema_version=SCHEMA, explicit_saved_qr_consumer=True,
                    background_qr_declares_operation=False, equipment_by_state=equipment,
                    registered_camera_count=cameras, persisted_operator_scan_count=scans,
                    live_source_manifest_configured=self.settings.live_sources_manifest is not None,
                    publication_enabled=self.settings.timeline_publish,
                    hardware_source_routing_verified=False, field_scan_acceptance='NOT_PROVEN')

    def consume(self, key, *, job_id, intent, scan_kind, actor, reason,
                person_id=None, run_id=None, batch_id=None):
        if (type(intent) is not str or intent not in {'operator_scan', 'background_qr'}
                or type(scan_kind) is not str or scan_kind not in {'equipment', 'material'}):
            raise ValueError('Explicit scan intent and equipment/material kind required')
        _text(actor, 'scan source actor')
        _text(reason, 'scan source reason')
        request = clean(dict(schema_version=SCHEMA, job_id=job_id, intent=intent, scan_kind=scan_kind,
                             actor=actor, reason=reason, person_id=person_id, run_id=run_id, batch_id=batch_id))

        def action(conn):
            Timeline(self.core).lock(conn)
            job = conn.execute('''SELECT j.*,a.id AS media_id,a.verified_sha256,a.version,
                c.time_basis,c.occurred_at,i.payload AS receipt
                FROM processing_job j JOIN capture c ON c.customer_id=j.customer_id AND c.id=j.capture_id
                JOIN media_asset a ON a.customer_id=c.customer_id AND a.capture_id=c.id
                JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                  AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                WHERE j.customer_id=%s AND j.id=%s AND j.worker_kind='qr' AND c.kind='photo' ''',
                (self.core.customer, ident(job_id))).fetchone()
            if job is None:
                raise Missing('QR photo job outside customer scope')
            if job['status'] != 'completed':
                raise Conflict('Only completed QR evidence can declare a scan')
            result = job['result']
            rules = result.get('processing_identity', {}).get('processing_rules', {})
            if rules != job['rule_identity'] or rules.get('sha256') != job['rule_fingerprint']:
                raise Conflict('QR result and its original processing rules differ')
            location = Media(self.core, self.settings).locate(job['media_id'])
            if location['status'] != 'available' or location['version'] != job['version']:
                raise Conflict('Original QR evidence is unavailable or changed')
            camera_code = job['receipt'].get('camera_id')
            camera = conn.execute("SELECT id FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s",
                                  (self.core.customer, camera_code)).fetchone()
            if camera is None:
                raise Conflict('QR capture requires its explicitly registered source camera')
            observations = conn.execute('''SELECT id,evidence_ids FROM observation WHERE customer_id=%s
                AND job_id=%s AND kind='qr' ''', (self.core.customer, job['id'])).fetchall()
            if len(observations) != 1 or not observations[0]['evidence_ids']:
                raise Conflict('QR job lacks one persisted original evidence observation')
            evidence = observations[0]['evidence_ids']
            values = result.get('decoded_values')
            if (not isinstance(values, list) or len(values) > 64
                    or any(type(v) is not str or not v or len(v.encode()) > 4096 for v in values)):
                raise Conflict('Saved decoder result has invalid bounded QR values')
            unique = sorted(set(values))
            at = job['occurred_at'] if verified_clock(job['time_basis']) else None
            pins = clean(dict(schema_version=SCHEMA, qr_job_id=job['id'], qr_observation_id=observations[0]['id'],
                media_id=job['media_id'], source_sha256=job['verified_sha256'], media_version=job['version'],
                result_sha256=fingerprint(result), processing_rules=rules, camera_id=camera['id'],
                source_camera_key=camera_code, original_time_basis=job['time_basis'],
                source_reported_at=job['occurred_at'], event_time_verified=at is not None,
                decoded_values=unique, declared_intent=intent, model_invoked=False,
                equipment_activation_performed=False, panel_attribution='not_inferred', quantity_written=False))
            if intent == 'operator_scan' and len(unique) == 1:
                scan = OperatorScans(self.core).record('scan-media-operator:' + str(job['id']), intent=intent, scan_kind=scan_kind,
                    qr_value=unique[0], occurred_at=clean(at), evidence_ids=list(map(str, evidence)),
                    actor=actor, reason=reason, person_id=person_id, camera_id=str(camera['id']),
                    run_id=run_id, batch_id=batch_id)
                return dict(status=scan['status'], scan=scan, evidence=pins)
            status = ('passive_observation' if intent == 'background_qr' else
                      'no_qr_decoded' if not unique else 'ambiguous_multiple_qr')
            activity = declaration_activity(conn, self.core, text='扫码证据：' + status, at=at,
                evidence_ids=evidence, payload=dict(scan_intake=pins, status=status,
                    operator_context_changed=False, actor=actor, reason=reason,
                    origin='saved_qr_machine_evidence', human_review_performed=False,
                    source_type_presentation='machine_qr_observation'))
            return dict(status=status, activity_id=activity, evidence=pins, operator_context_changed=False)

        return self.core.command(key, request, lambda conn:
            self.core.command('scan-media-source:' + str(ident(job_id)), request, action))

    def _ready(self, limit, *, before=None, camera=None, person=None):
        dependency = before is not None
        where = ''
        params = [self.core.customer]
        if dependency:
            # A dependency is earlier physical event time in the exact scope,
            # never an adjacent camera/person or an unverified wall clock.
            where = _CAUSAL_SCOPE_SQL
            params += [before, camera, ident(person).hex if person else None]
        else:
            where = "AND (retry.result IS NULL OR (retry.result->>'next_retry_at')::timestamptz<=now())"
        order = 'c.occurred_at,j.created_at,j.id' if dependency else 'c.occurred_at DESC NULLS LAST,j.created_at DESC,j.id'
        params.append(limit)
        with self.core.db.transaction(readonly=True) as conn:
            return conn.execute('''SELECT j.id,c.occurred_at,c.time_basis,
                    i.payload->>'camera_id' AS camera,i.payload->'scan_intent' AS declaration,
                    retry.result AS retry,(retry.result->>'next_retry_at')::timestamptz>now() AS waiting
                FROM processing_job j JOIN capture c ON c.customer_id=j.customer_id AND c.id=j.capture_id
                JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                    AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                LEFT JOIN LATERAL (SELECT r.result FROM command_receipt r
                    WHERE r.customer_id=j.customer_id AND r.key IN
                        ('scan-intake-failure:'||j.id::text||':1','scan-intake-failure:'||j.id::text||':2')
                    ORDER BY r.key DESC LIMIT 1) retry ON true
                WHERE j.customer_id=%s AND j.worker_kind='qr' AND j.status='completed' AND c.kind='photo'
                    AND i.payload->'scan_intent'->>'intent'='operator_scan'
                    AND NOT EXISTS(SELECT 1 FROM command_receipt r WHERE r.customer_id=j.customer_id
                        AND r.key IN ('scan-intake:'||j.id::text,'scan-media-source:'||j.id::text))
                ''' + where + ' ORDER BY ' + order + ' LIMIT %s', params).fetchall()

    def _unready_target(self, job, person):
        with self.core.db.transaction(readonly=True) as conn:
            return conn.execute('''SELECT j.id,j.status FROM processing_job j
                JOIN capture c ON c.customer_id=j.customer_id AND c.id=j.capture_id
                JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                    AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                WHERE j.customer_id=%s AND j.worker_kind='qr' AND j.status<>'completed' AND c.kind='photo'
                    AND i.payload->'scan_intent'->>'intent'='operator_scan'
                    AND (i.payload->'scan_intent'->>'run_id' IS NOT NULL
                        OR i.payload->'scan_intent'->>'batch_id' IS NOT NULL)
                ''' + _CAUSAL_SCOPE_SQL + ' ORDER BY c.occurred_at DESC,j.created_at DESC,j.id LIMIT 1',
                (self.core.customer, job['occurred_at'], job['camera'], ident(person).hex if person else None)).fetchone()

    def _failure(self, job, exc, *, permanent, dependency_failed=False):
        """Append bounded error receipts; never edit the source or a scan snapshot."""
        job_id = str(job['id'])
        attempt = 1 + ((job.get('retry') or {}).get('attempt') or 0)
        request = dict(schema_version=SCHEMA, job_id=job_id, attempt=attempt,
                       declaration_fingerprint=fingerprint(job['declaration']))

        def action(conn):
            # Match automatic consume lock order before recording its terminal
            # marker, so a simultaneous successful consume wins atomically.
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                         (str(self.core.customer) + ':scan-intake:' + job_id,))
            Timeline(self.core).lock(conn)
            old = conn.execute('SELECT result FROM command_receipt WHERE customer_id=%s AND key IN (%s,%s) LIMIT 1',
                (self.core.customer, 'scan-intake:' + job_id, 'scan-media-source:' + job_id)).fetchone()
            if old:
                return old['result']
            exhausted = attempt > len(_RETRY_SECONDS)
            terminal = permanent or dependency_failed or exhausted
            now = conn.execute('SELECT now() AS at').fetchone()['at']
            result = clean(dict(status='needs_attention', reason=str(exc)[:512], error_code=type(exc).__name__,
                attempt=attempt, permanent_declaration_error=permanent, automatic_retry_exhausted=exhausted,
                causal_dependency_failed=dependency_failed,
                next_retry_at=None if terminal else now + timedelta(seconds=_RETRY_SECONDS[attempt - 1]),
                explicit_resume_required=terminal, source_or_scan_snapshot_changed=False))
            if terminal:
                conn.execute('INSERT INTO command_receipt(customer_id,key,fingerprint,result) VALUES(%s,%s,%s,%s)',
                    (self.core.customer, 'scan-intake:' + job_id, fingerprint(request), Jsonb(result)))
            return result
        return self.core.command('scan-intake-failure:' + job_id + ':' + str(attempt), request, action)

    def tick(self, limit=20):
        if self.settings.processing_profile != 'timeline-v1':
            return dict(enabled=False)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Bounded scan intake limit required')
        jobs = self._ready(limit)
        outcomes, seen = [], set()
        spent = 0

        def consume(job):
            nonlocal spent
            seen.add(job['id'])
            spent += 1
            try:
                declaration = _declaration(job['declaration'])
                result = self.consume('scan-intake:' + str(job['id']), job_id=str(job['id']), **declaration)
            except (Conflict, Missing, OSError) as exc:
                result = self._failure(job, exc, permanent=False)
            except ValueError as exc:
                result = self._failure(job, exc, permanent=True)
            outcomes.append(dict(job_id=str(job['id']), **result))
            return result['status'] != 'needs_attention'

        for job in jobs:
            if spent >= limit:
                break
            if job['id'] in seen:
                continue
            try:
                declaration = _declaration(job['declaration'])
            except ValueError:
                consume(job)
                continue
            if (declaration['scan_kind'] == 'material' and job['occurred_at'] is not None
                    and verified_clock(job['time_basis'])):
                if not (declaration.get('run_id') or declaration.get('batch_id')):
                    pending = self._unready_target(job, declaration.get('person_id'))
                    if pending:
                        if pending['status'] == 'failed':
                            spent += 1
                            seen.add(job['id'])
                            result = self._failure(job, Conflict('Earlier exact-scope explicit target QR job failed'),
                                permanent=False, dependency_failed=True)
                            outcomes.append(dict(job_id=str(job['id']), **result))
                        else:
                            outcomes.append(dict(job_id=str(job['id']), status='deferred',
                                reason='earlier_explicit_target_qr_unfinished', dependency_job_id=str(pending['id']),
                                source_or_scan_snapshot_changed=False))
                        continue
                dependencies = self._ready(limit - spent + 1, before=job['occurred_at'], camera=job['camera'],
                                           person=declaration.get('person_id'))
                blocked = False
                for previous in dependencies:
                    if previous['id'] in seen:
                        continue
                    if previous['waiting'] or spent >= limit:
                        blocked = True
                        break
                    if not consume(previous):
                        blocked = True
                        break
                if blocked or spent >= limit:
                    outcomes.append(dict(job_id=str(job['id']), status='deferred',
                        reason='earlier_exact_scope_declarations_pending', source_or_scan_snapshot_changed=False))
                    continue
            consume(job)
        return dict(enabled=True, consumed=spent, scheduling='latest_first_with_exact_scope_causal_dependencies', outcomes=outcomes)
