"""Consume saved QR evidence into the existing explicit operator scan command.

The decoder's completed job is reused. Background visibility, multiple codes and
an unknown source clock never acquire an operator/equipment association.
"""
from __future__ import annotations

from .domain import Conflict, Missing, clean, fingerprint, ident
from .media import Media
from .operator_scan import OperatorScans, _text, declaration_activity
from .timeline import Timeline

SCHEMA = 'vc-lab-scan-intake/1'


def verified_clock(basis):
    return (isinstance(basis, dict) and bool(basis.get('basis'))
            and type(basis.get('uncertainty_us')) is int and basis['uncertainty_us'] == 0
            and (basis.get('clock_sync_verified') is True or basis.get('clock_sync_valid') is True)
            and basis.get('clock_sync_verified') is not False
            and basis.get('clock_sync_valid') is not False)


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
        if intent not in {'operator_scan', 'background_qr'} or scan_kind not in {'equipment', 'material'}:
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
                    operator_context_changed=False, actor=actor, reason=reason))
            return dict(status=status, activity_id=activity, evidence=pins, operator_context_changed=False)

        return self.core.command(key, request, lambda conn:
            self.core.command('scan-media-source:' + str(ident(job_id)), request, action))

    def tick(self, limit=20):
        if self.settings.processing_profile != 'timeline-v1':
            return dict(enabled=False)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Bounded scan intake limit required')
        with self.core.db.transaction(readonly=True) as conn:
            jobs = conn.execute('''SELECT j.id,i.payload->'scan_intent' AS declaration
                FROM processing_job j JOIN capture c ON c.customer_id=j.customer_id AND c.id=j.capture_id
                JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                    AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                WHERE j.customer_id=%s AND j.worker_kind='qr' AND j.status='completed'
                    AND i.payload->'scan_intent'->>'intent'='operator_scan'
                    AND NOT EXISTS(SELECT 1 FROM command_receipt r WHERE r.customer_id=j.customer_id
                        AND r.key='scan-intake:'||j.id::text)
                ORDER BY c.occurred_at DESC NULLS LAST,j.created_at DESC,j.id LIMIT %s''',
                (self.core.customer, limit)).fetchall()
        outcomes = []
        allowed = {'intent', 'scan_kind', 'actor', 'reason', 'person_id', 'run_id', 'batch_id'}
        for job in jobs:
            declaration = job['declaration']
            try:
                if (type(declaration) is not dict or set(declaration) - allowed
                        or not {'intent', 'scan_kind', 'actor', 'reason'} <= set(declaration)):
                    raise ValueError('Unsupported explicit producer scan declaration')
                result = self.consume('scan-intake:' + str(job['id']), job_id=str(job['id']), **declaration)
                outcomes.append(dict(job_id=str(job['id']), status=result['status']))
            except (Conflict, Missing, ValueError) as exc:
                outcomes.append(dict(job_id=str(job['id']), status='needs_attention', reason=str(exc)))
        return dict(enabled=True, consumed=len(outcomes), outcomes=outcomes)
