"""OCR weighing episodes over the existing Core ledger and command journal.

An episode has an evidenced baseline, one instrument/material/target and one
capture scope. Photos update its total; they are not independent consumptions.
All writes, including Core corrections, commit in the original transaction.
"""
from __future__ import annotations

from decimal import Decimal
from uuid import uuid5

from .domain import Conflict, Missing, amount, clean, fingerprint, ident, instant
from .operator_scan import OperatorScans
from .rules import Rules
from .timeline import Timeline
from .timeline_clock import source_clock_supported

OPEN = 'weighing-open:'
EVENT = 'weighing-event:'
SCHEMA = 'vc-lab-weighing/1'


def same_context(left, right):
    keys = ('instrument_id', 'camera_id', 'batch_id', 'run_id', 'material_id', 'source_lot_id')
    if any(left.get(k) != right.get(k) for k in keys):
        return False
    # Seeing the same label again adds evidence, not a new taking operation.
    for key in ('assignment_revision', 'content_binding'):
        if left['label_snapshot'].get(key) != right['label_snapshot'].get(key):
            return False
    return True


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError('Bounded actor and reason required')


class Weighing:
    def __init__(self, core):
        self.core, self.customer = core, core.customer

    def _reading(self, conn, activity_id, field_key):
        activity = conn.execute("SELECT * FROM hj_activity WHERE customer_id=%s AND id=%s AND source_type='ocr'",
                                (self.customer, ident(activity_id))).fetchone()
        if not activity:
            raise Missing('Original OCR activity outside this customer')
        payload = activity['payload']
        task = conn.execute("SELECT * FROM hj_timeline_task WHERE customer_id=%s AND id=%s AND kind='ocr'",
                            (self.customer, ident(payload.get('task_id')))).fetchone() if payload.get('task_id') else None
        if (not task or task['status'] != 'completed' or str(activity['id']) not in
                (task['result'] or {}).get('activity_ids', []) or task['model_revision'] != activity['model_revision']):
            raise Conflict('Weighing requires a completed original OCR task')
        fields = [f for f in payload.get('fields', []) if f.get('field_key') == field_key]
        if len(fields) != 1:
            raise Conflict('One exact panel field required; whole-photo tokens cannot debit stock')
        field = fields[0]
        if (field.get('value') is None or field.get('status') not in {'candidate', 'confirmed'}
                or field.get('issues') or field.get('quality_issues')
                or field.get('role') not in {'displayed_value', 'process_value'}):
            raise Conflict('Unreadable, conflicting or non-measurement field')
        value, unit = amount(field['value'], field.get('unit'))
        if unit != 'g' or field.get('unit_basis') not in {
                'verified_on_panel', 'verified_instrument_specification'}:
            raise Conflict('Inventory calculations require an evidenced mass unit')
        sources = conn.execute('''SELECT e.*,m.kind,m.start_at,m.stream_key,m.session_key,m.time_basis,
            a.version,a.verified_sha256,a.evidence_status,i.payload AS receipt
            FROM evidence_ref e JOIN hj_timeline_media m USING(customer_id,media_id)
            JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
            JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
            JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
              AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
            WHERE e.customer_id=%s AND e.id=ANY(%s)''', (self.customer, activity['evidence_ids'])).fetchall()
        if len(sources) != 1 or sources[0]['kind'] != 'photo':
            raise Conflict('One original photograph required for a weighing observation')
        source = sources[0]
        if (source['evidence_status'] != 'available' or source['media_version'] != source['version']
                or not source['verified_sha256'] or source['receipt'].get('sha256') not in {None, source['verified_sha256']}):
            raise Conflict('Weighing source version or SHA differs')
        if (not source_clock_supported(source['time_basis'], source['receipt'].get('source_kind'))
                or activity['start_at'] is None or activity['start_at'] != source['start_at']):
            raise Conflict('Source-relative readings are retained but cannot resolve historical stock labels')
        if (source['receipt'].get('source_kind') != 'fixture'
                and not payload.get('execution_verification', '').startswith('local_')):
            raise Conflict('Automatic stock use requires persisted local OCR execution evidence')
        camera = conn.execute("SELECT id,code FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s",
                              (self.customer, source['receipt'].get('camera_id'))).fetchone()
        if not camera:
            raise Conflict('Exact capture camera required, without person attribution')
        iid = None
        registration = task['request'].get('frame_registration')
        if registration:
            document = registration['registration']
            if (document.get('status') == 'verified' and document.get('source_sha256') == source['verified_sha256']
                    and any(r.get('field_key') == field_key for r in document.get('regions', []))):
                iid = document['instrument_id']
        elif field.get('association_status') == 'spatial_identity_supported':
            context = field.get('dynamic_context') or {}
            if (context.get('identity_basis') == 'registered_panel_qr_region'
                    and context.get('source_sha256') == source['verified_sha256']
                    and context.get('layout_fingerprint') and context.get('qr_region')):
                iid = context.get('instrument_id')
        if not iid or str(activity['instrument_id']) != str(iid):
            raise Conflict('This panel must have its own equipment identity proof')
        self.core._entity(conn, iid, 'instrument')
        return clean(dict(activity_id=activity['id'], field_key=field_key, value=value, unit=unit,
            displayed_value=field.get('value_text', field['value']), instrument_id=iid,
            camera_id=camera['id'], camera_code=camera['code'], occurred_at=activity['start_at'],
            media_id=source['media_id'], source_sha256=source['verified_sha256'], media_version=source['version'],
            source_path=source['receipt']['relative_path'], evidence_ids=activity['evidence_ids'],
            task_id=task['id'], model_revision=activity['model_revision'],
            source_key=fingerprint(dict(sha256=source['verified_sha256'], field_key=field_key))))

    def _context(self, conn, reading):
        context = OperatorScans(self.core)._context(conn, instant(reading['occurred_at']), None, reading['camera_id'])
        if context['equipment_id'] != reading['instrument_id'] or not context['batch_id']:
            raise Conflict('Equipment and target batch must be declared in this exact capture scope')
        materials = context['materials']
        if not materials:
            raise Conflict('No declared material in the equipment/target context')
        latest = max(m['occurred_at'] for m in materials)
        selected = [m for m in materials if m['occurred_at'] == latest]
        identities = {m['resolution']['target_id'] for m in selected}
        if len(identities) != 1:
            raise Conflict('Simultaneous different materials cannot be assigned to one scale reading')
        selected = sorted(selected, key=lambda m: m['event_id'])[-1]
        resolved = selected['resolution']
        instrument = self.core._entity(conn, context['equipment_id'], 'instrument')
        batch = self.core._entity(conn, context['batch_id'], 'process_batch')
        run = Timeline(self.core)._run(conn, context['run_id']) if context['run_id'] else None
        samples = [self.core._entity(conn, sid, 'material_unit') for sid in run['sample_ids']] if run else []
        return clean(dict(instrument_id=context['equipment_id'], camera_id=reading['camera_id'],
            batch_id=context['batch_id'], run_id=context['run_id'], material_id=resolved['target_id'],
            source_lot_id=resolved.get('source_lot_id'), material_scan_event_id=selected['event_id'],
            label_snapshot=resolved, instrument_snapshot=instrument, target_batch_snapshot=batch,
            target_samples=samples, target_scan_event_ids=context['target_declaration_event_ids']))

    def _head(self, conn, session_id):
        row = conn.execute('SELECT result FROM command_receipt WHERE customer_id=%s AND key=%s',
                           (self.customer, OPEN + str(ident(session_id)))).fetchone()
        if not row:
            raise Missing('Weighing episode outside this customer')
        return row['result']

    def _latest(self, conn, session_id):
        head = self._head(conn, session_id)
        row = conn.execute("SELECT result FROM command_receipt WHERE customer_id=%s AND starts_with(key,%s) AND result->>'schema_version'='vc-lab-weighing/1' ORDER BY (result->>'sequence')::integer DESC LIMIT 1",
                           (self.customer, EVENT + str(session_id) + ':')).fetchone()
        return row['result'] if row else head

    def _lock(self, conn, session_id):
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                     ('weighing:' + str(self.customer) + ':' + str(session_id),))

    def active(self, conn, camera_id, instrument_id):
        rows = conn.execute("""SELECT coalesce(e.result,h.result) AS result FROM command_receipt h
            LEFT JOIN LATERAL (SELECT result FROM command_receipt e WHERE e.customer_id=h.customer_id
                AND starts_with(e.key,%s||(h.result->>'session_id')||':')
                AND e.result->>'schema_version'='vc-lab-weighing/1'
                ORDER BY (result->>'sequence')::integer DESC LIMIT 1) e ON true
            WHERE h.customer_id=%s AND starts_with(h.key,%s)
                AND h.result#>>'{context,instrument_id}'=%s
                AND coalesce(e.result,h.result)->>'state'<>'closed' LIMIT 3""",
            (EVENT,self.customer,OPEN,str(instrument_id))).fetchall()
        active = [r['result'] for r in rows]
        if len(active) > 1:
            raise Conflict('Conflicting active weighing episodes')
        if active and active[0]['context']['camera_id'] != str(camera_id):
            raise Conflict('Another capture scope owns this instrument episode; no cross-camera debit inference')
        return active[0] if active else None

    def start(self, key, baseline_activity_id, field_key, *, mode='net_after_tare', actor, reason):
        _text(actor); _text(reason)
        if mode not in {'net_after_tare', 'difference'}:
            raise ValueError('Explicit weighing mode required')
        request = dict(baseline_activity_id=baseline_activity_id, field_key=field_key, mode=mode, actor=actor, reason=reason)
        def action(conn):
            reading = self._reading(conn, baseline_activity_id, field_key)
            context = self._context(conn, reading)
            if mode == 'net_after_tare' and Decimal(reading['value']) != 0:
                raise Conflict('Net weighing needs an original zero/tare reading')
            sid = str(uuid5(self.customer, fingerprint(dict(source=reading['source_key'], context=context, mode=mode))))
            self._lock(conn, 'instrument:' + reading['instrument_id'])
            old = conn.execute('SELECT result FROM command_receipt WHERE customer_id=%s AND key=%s', (self.customer, OPEN + sid)).fetchone()
            if old:
                return old['result']
            if self.active(conn, reading['camera_id'], reading['instrument_id']):
                raise Conflict('Close the preceding weighing episode before a new baseline')
            material = self.core._states(conn, [context['material_id']])[ident(context['material_id'])]
            if material['unit'] != reading['unit']:
                raise Conflict('Material stock dimension differs from the scale field')
            result = dict(schema_version=SCHEMA, session_id=sid, sequence=0, state='open', status='baseline_recorded',
                mode=mode, context=context, baseline=reading, reading=reading, quantity='0', unit=reading['unit'],
                delta_used='0', operation_id=None, operation_revision=None, quantity_ledger_written=False,
                actor=actor, reason=reason, processing_rules=Rules().identity(), human_verified=False)
            self.core.command(OPEN + sid, request, lambda _: result)
            Timeline(self.core)._dirty(conn, instant(reading['occurred_at']))
            return result
        return self.core.command(key, request, action)

    def observe(self, key, session_id, activity_id, *, actor, reason, expected_sequence=None, correction=False, existing_operation_id=None):
        _text(actor); _text(reason)
        if type(correction) is not bool or (correction or existing_operation_id) and type(expected_sequence) is not int:
            raise ValueError('A correction requires an explicit expected sequence')
        sid = str(ident(session_id))
        request = dict(session_id=sid, activity_id=activity_id, actor=actor, reason=reason,
                       expected_sequence=expected_sequence, correction=correction)
        if existing_operation_id:request['existing_operation_id']=str(ident(existing_operation_id))
        def action(conn):
            self._lock(conn, sid)
            head, previous = self._head(conn, sid), self._latest(conn, sid)
            if previous['state'] == 'closed':
                raise Conflict('Weighing episode is closed')
            if expected_sequence is not None and previous['sequence'] != expected_sequence:
                raise Conflict('Weighing episode changed')
            reading = self._reading(conn, activity_id, head['reading']['field_key'])
            context = self._context(conn, reading)
            if not same_context(context, head['context']) or reading['camera_id'] != head['reading']['camera_id']:
                raise Conflict('Equipment/material/target history changed; start a separate episode')
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                ('material-use:'+str(self.customer)+':'+context['camera_id']+':'+context['material_scan_event_id'],))
            if reading['unit'] != head['unit']:
                raise Conflict('Scale unit changed within this weighing episode')
            event_key = EVENT + sid + ':' + fingerprint(request)[:40]
            old_source = conn.execute("SELECT result FROM command_receipt WHERE customer_id=%s AND starts_with(key,%s) AND result#>>'{reading,source_key}'=%s ORDER BY (result->>'sequence')::integer LIMIT 1",
                (self.customer, EVENT + sid + ':', reading['source_key'])).fetchone()
            desired = Decimal(reading['value']) - Decimal(head['baseline']['value'])
            previous_quantity = Decimal(previous['quantity'])
            status = 'same_total_no_new_debit'
            apply = False
            if not correction and old_source and old_source['result']['reading']['value'] != reading['value']:
                status = 'same_source_conflicting_OCR_requires_correction'
            elif not correction and instant(reading['occurred_at']) < instant(previous['reading']['occurred_at']):
                status = 'late_reading_retained_without_reversing_stock'
            elif not correction and previous['state'] == 'needs_attention':
                status = 'episode_requires_correction'
            elif not correction and previous.get('linked_existing') and desired != previous_quantity:
                status = 'cross_modal_quantity_conflict_requires_correction'
            elif desired < 0 or not correction and desired < previous_quantity:
                status = 'reading_decreased_reset_or_correction_required'
            elif desired != previous_quantity:
                apply = True
            result = dict(previous, sequence=previous['sequence']+1, reading=reading,
                status=status, actor=actor, reason=reason, correction=correction,
                quantity_ledger_written=False, delta_used='0')
            if existing_operation_id:
                if previous['operation_id'] or correction:
                    raise Conflict('Explicit existing-use link requires an unposted weighing episode')
                operation=conn.execute('SELECT * FROM operation WHERE customer_id=%s AND id=%s',
                                       (self.customer,ident(existing_operation_id))).fetchone()
                lines=conn.execute('SELECT * FROM operation_line WHERE customer_id=%s AND operation_id=%s',
                                   (self.customer,ident(existing_operation_id))).fetchall()
                if (not operation or operation['status']!='confirmed'
                        or operation['metadata'].get('effective_kind',operation['kind'])!='consume'
                        or str(operation['batch_id'])!=context['batch_id'] or len(lines)!=1
                        or str(lines[0]['material_id'])!=context['material_id']
                        or lines[0]['direction']!='input' or lines[0]['quantity']!=desired or lines[0]['unit']!=head['unit']):
                    raise Conflict('Explicit cross-modal link differs in material, target, quantity or current operation')
                result.update(quantity=str(desired),operation_id=str(operation['id']),operation_revision=operation['revision'],
                              status='existing_usage_linked_no_new_debit',linked_existing=True,state='open')
                apply=False
            if apply:
                line = dict(material_id=context['material_id'], quantity=str(desired), unit=head['unit'],
                            quantity_basis='source_pinned_OCR_weighing_episode')
                evidence = sorted(set(head['baseline']['evidence_ids'] + reading['evidence_ids']))
                if desired == 0 and previous['operation_id']:
                    operation = self.core.cancel_consumption(event_key + ':ledger', previous['operation_id'],
                        expected_revision=previous['operation_revision'], actor=actor,reason=reason,evidence_ids=evidence)
                elif previous['operation_id']:
                    operation = self.core.correct_operation(event_key + ':ledger', previous['operation_id'], [line],
                        expected_revision=previous['operation_revision'], actor=actor,
                        reason=reason + '; weighing evidence ' + fingerprint(reading), evidence_ids=evidence)
                else:
                    duplicate=conn.execute("SELECT 1 FROM operation WHERE customer_id=%s AND status='confirmed' AND metadata#>>'{usage_scope,camera_id}'=%s AND metadata#>>'{usage_scope,material_scan_event_id}'=%s LIMIT 1",
                                           (self.customer,context['camera_id'],context['material_scan_event_id'])).fetchone()
                    if duplicate:
                        raise Conflict('Potential speech/photo duplicate: link the explicit existing use occurrence first')
                    operation = self.core.operate(event_key + ':ledger', 'consume', [line],
                        occurred_at=reading['occurred_at'], actor=actor,
                        operation_id=str(uuid5(ident(sid), 'stock-use')), batch_id=context['batch_id'],
                        run_id=context['run_id'], evidence_ids=evidence,
                        metadata=dict(weighing_session_id=sid, context_snapshot=context, baseline=head['baseline'], reading=reading,
                                      automatic_OCR_policy=True, human_verified=False))
                result.update(quantity=str(desired), delta_used=str(desired-previous_quantity),
                    operation_id=operation['operation_id'], operation_revision=operation['revision'],
                    status='stock_corrected' if correction else 'stock_updated',
                    state='closed' if desired == 0 else 'open', quantity_ledger_written=True, ledger_result=operation)
            elif status in {'same_source_conflicting_OCR_requires_correction', 'reading_decreased_reset_or_correction_required', 'cross_modal_quantity_conflict_requires_correction'}:
                result['state'] = 'needs_attention'
            self.core.command(event_key, request, lambda _: result)
            Timeline(self.core)._dirty(conn, instant(reading['occurred_at']))
            return result
        return self.core.command(key, request, action)

    def close(self, key, session_id, *, expected_sequence, actor, reason):
        _text(actor); _text(reason)
        sid = str(ident(session_id))
        request = dict(session_id=sid, expected_sequence=expected_sequence, actor=actor, reason=reason)
        def action(conn):
            self._lock(conn, sid)
            previous = self._latest(conn, sid)
            if previous['sequence'] != expected_sequence or previous['state'] == 'closed':
                raise Conflict('Weighing episode changed or already closed')
            result = dict(previous, sequence=previous['sequence']+1, state='closed', status='episode_closed',
                          quantity_ledger_written=False, delta_used='0', actor=actor, reason=reason)
            self.core.command(EVENT + sid + ':close:' + fingerprint(request)[:24], request, lambda _: result)
            Timeline(self.core)._dirty(conn, instant(previous['reading']['occurred_at']))
            return result
        return self.core.command(key, request, action)

    def get(self, session_id):
        with self.core.db.transaction(readonly=True) as conn:
            return clean(self._latest(conn, str(ident(session_id))))
