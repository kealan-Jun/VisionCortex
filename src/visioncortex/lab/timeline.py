"""Versioned recording/traceability logic for customer VC Lab, independent of detection.

Business QR binding is ours. Hardware pairing and Agent/Feishu writeback are not.
Observations remain evidence-backed candidates, never material ledger operations.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation

from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, fingerprint, ident, instant
from .timeline_policy import RecordPolicy, TimelinePolicies, ZONE

VERSION = 'vc-lab-timeline/1'
KINDS = {'loading', 'start', 'settings', 'transfer', 'reading', 'unloading', 'end',
         'correction', 'anomaly', 'other_work', 'rest', 'chat', 'waiting', 'unknown', 'speech'}


def seconds(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid media offset') from exc
    if not result.is_finite() or result < 0 or result.as_tuple().exponent < -6:
        raise ValueError('Media offsets must be finite nonnegative microseconds')
    return result


def offset_time(start, offset):
    return start + timedelta(microseconds=int(offset * 1000000)) if start else None


def day_bounds(day):
    day = datetime.strptime(str(day), '%Y-%m-%d').date()
    return (datetime.combine(day, time.min, ZONE), datetime.combine(day + timedelta(days=1), time.min, ZONE))


class Timeline:
    def __init__(self, core, policies=None):
        self.core = core
        self.customer = core.customer
        self.policies = policies or TimelinePolicies()

    def _command(self, key, op, request, action):
        def locked(conn):
            self.lock(conn)
            return action(conn)
        return self.core.command(key, dict(version=VERSION, op=op, request=request), locked)

    def lock(self, conn):
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', ('timeline:' + str(self.customer),))

    def _run(self, conn, run_id):
        row = conn.execute('SELECT r.*,e.code,e.metadata AS registration FROM hj_experiment_record r JOIN entity e ON e.customer_id=r.customer_id AND e.id=r.id WHERE r.customer_id=%s AND r.id=%s', (self.customer, ident(run_id))).fetchone()
        if not row:
            raise Missing('Experiment record outside customer scope')
        return row

    def _dirty(self, conn, *times):
        days = {at.astimezone(ZONE).date() for at in times if at is not None}
        for day in days:
            conn.execute('INSERT INTO hj_timeline_dirty_day(customer_id,day) VALUES(%s,%s) ON CONFLICT DO NOTHING', (self.customer, day))

    def _touch(self, conn, run_id, start, end):
        row = conn.execute('UPDATE hj_experiment_record SET first_observed_at=LEAST(first_observed_at,%s),last_observed_at=GREATEST(last_observed_at,%s),revision=revision+1 WHERE customer_id=%s AND id=%s RETURNING *', (start, end, self.customer, ident(run_id))).fetchone()
        self._dirty(conn, start, end, row['first_observed_at'], row['last_observed_at'])
        self._dirty_run_days(conn, run_id)
        self.core._emit(conn, 'experiment_record', row['id'], row['revision'], dict(row, occurred_at=row['last_observed_at']))

    def _archive(self, conn, run_id, at, reason, evidence_ids=()):
        row = conn.execute("UPDATE hj_experiment_record SET record_status='archived',closed_at=%s,closure_reason=%s,closure_evidence_ids=%s,revision=revision+1 WHERE customer_id=%s AND id=%s AND record_status='open' RETURNING *", (at, reason, list(evidence_ids), self.customer, ident(run_id))).fetchone()
        if row:
            self._dirty(conn, row['first_observed_at'], row['last_observed_at'])
            self._dirty_run_days(conn, run_id)
            self.core._emit(conn, 'experiment_record', row['id'], row['revision'], dict(row, occurred_at=row['last_observed_at']))
        return row

    def _dirty_run_days(self, conn, run_id):
        conn.execute("""INSERT INTO hj_timeline_dirty_day(customer_id,day)
          SELECT %s,day FROM (
            SELECT (a.start_at AT TIME ZONE 'Asia/Shanghai')::date AS day
              FROM hj_activity a JOIN hj_activity_run r ON r.customer_id=a.customer_id AND r.activity_id=a.id WHERE r.customer_id=%s AND r.run_id=%s
            UNION SELECT (a.end_at AT TIME ZONE 'Asia/Shanghai')::date
              FROM hj_activity a JOIN hj_activity_run r ON r.customer_id=a.customer_id AND r.activity_id=a.id WHERE r.customer_id=%s AND r.run_id=%s
            UNION SELECT (occurred_at AT TIME ZONE 'Asia/Shanghai')::date FROM hj_experiment_binding WHERE customer_id=%s AND run_id=%s
          ) days WHERE day IS NOT NULL ON CONFLICT DO NOTHING""", (self.customer, self.customer, ident(run_id), self.customer, ident(run_id), self.customer, ident(run_id)))

    def label_event(self, key, tag_id, at, *, target_id=None, actor, reason):
        """Assign/release a reusable sample label; supports delayed historical events."""
        when = instant(at)
        if not actor or not reason:
            raise ValueError('Label change requires actor and reason')

        def action(conn):
            tag = self.core._entity(conn, tag_id, 'tag')
            if tag['metadata'].get('mode') == 'permanent':
                raise Conflict('Permanent equipment codes cannot be recycled as sample labels')
            if target_id:
                self.core._entity(conn, target_id, 'material_unit')
            old = conn.execute('SELECT * FROM hj_label_event WHERE customer_id=%s AND tag_id=%s AND occurred_at=%s', (self.customer, ident(tag_id), when)).fetchone()
            if old:
                if old['target_id'] != (ident(target_id) if target_id else None):
                    raise Conflict('Conflicting label assignment at the same event time')
                return old
            row = conn.execute('INSERT INTO hj_label_event(customer_id,id,tag_id,target_id,occurred_at,actor,reason) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *', (self.customer, ident(), ident(tag_id), ident(target_id) if target_id else None, when, actor, reason)).fetchone()
            self.core._emit(conn, 'sample_label_event', row['id'], 1, row)
            self._dirty(conn, when)
            self._queue_associations(conn, tag_id=tag_id)
            return row
        return self._command(key, 'label_event', dict(tag_id=tag_id, target_id=target_id, at=at, actor=actor, reason=reason), action)

    def _resolve_label(self, conn, tag_id, at, until=None):
        self.core._entity(conn, tag_id, 'tag')
        if at is None:
            return dict(status='unknown_time', target_id=None)
        end = until or at
        if end < at:
            raise ValueError('Invalid label query interval')
        any_events = conn.execute('SELECT 1 FROM hj_label_event WHERE customer_id=%s AND tag_id=%s LIMIT 1', (self.customer, ident(tag_id))).fetchone()
        if any_events:
            row = conn.execute('SELECT * FROM hj_label_event WHERE customer_id=%s AND tag_id=%s AND occurred_at<=%s ORDER BY occurred_at DESC LIMIT 1', (self.customer, ident(tag_id), at)).fetchone()
            changes = conn.execute('SELECT id FROM hj_label_event WHERE customer_id=%s AND tag_id=%s AND occurred_at>%s AND occurred_at<=%s LIMIT 1', (self.customer, ident(tag_id), at, end)).fetchone()
            if changes:
                return dict(status='crosses_rebinding', target_id=None)
            if row and not row['target_id']:
                return dict(status='unassigned', target_id=None, assignment_id=row['id'] if row else None)
            if row:
                return dict(status='resolved', target_id=row['target_id'], assignment_id=row['id'], valid_from=row['occurred_at'])
        # Existing permanent codes and reusable-label history remain usable.
        rows = conn.execute('SELECT * FROM tag_assignment WHERE customer_id=%s AND tag_id=%s AND valid_from<=%s AND (valid_to IS NULL OR valid_to>%s)', (self.customer, ident(tag_id), end, at)).fetchall()
        if len(rows) == 1 and rows[0]['valid_from'] <= at and (rows[0]['valid_to'] is None or rows[0]['valid_to'] > end):
            return dict(status='resolved', target_id=rows[0]['target_id'], assignment_id=rows[0]['id'], valid_from=rows[0]['valid_from'])
        return dict(status='unassigned' if not rows else 'crosses_rebinding', target_id=None)

    def resolve_label(self, tag_id, at, until=None):
        with self.core.db.transaction(readonly=True) as conn:
            return clean(self._resolve_label(conn, tag_id, instant(at) if at else None, instant(until) if until else None))

    def resolve_qr(self, value, at):
        with self.core.db.transaction(readonly=True) as conn:
            tag = conn.execute("SELECT id FROM entity WHERE customer_id=%s AND kind='tag' AND code=%s", (self.customer, value)).fetchone()
            if not tag:
                return dict(decoded_value=value, status='unregistered', target_id=None, identity_only=True)
            result = self._resolve_label(conn, tag['id'], instant(at) if at else None)
            target = self.core._entity(conn, result['target_id']) if result.get('target_id') else None
            return clean(dict(result, tag_id=tag['id'], decoded_value=value, target=target, identity_only=True))

    def create_run(self, key, code, experiment_type, sample_ids, *, person_id=None, batch_id=None, previous_run_ids=None, metadata=None, policy_instrument_id=None):
        if not experiment_type or not sample_ids or len(sample_ids) > 1000:
            raise ValueError('Experiment type and bounded explicit samples are required')
        samples = sorted({ident(x) for x in sample_ids}, key=str)

        def action(conn):
            for sample in samples:
                self.core._entity(conn, sample, 'material_unit')
            if person_id:
                self.core._entity(conn, person_id, 'person')
            if batch_id:
                self.core._entity(conn, batch_id, 'process_batch')
            if policy_instrument_id:
                self.core._entity(conn, policy_instrument_id, 'instrument')
            entity = self.core._register(conn, 'experiment_run', code, dict(metadata or {}, timeline_version=VERSION))
            if conn.execute('SELECT 1 FROM hj_experiment_record WHERE customer_id=%s AND id=%s', (self.customer, entity['id'])).fetchone():
                raise Conflict('Experiment code already exists; use its existing record ID')
            policy = self.policies.record(experiment_type=experiment_type, instrument_id=policy_instrument_id)
            row = conn.execute('INSERT INTO hj_experiment_record(customer_id,id,experiment_type,sample_ids,person_id,batch_id,policy,policy_fingerprint) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *', (self.customer, entity['id'], experiment_type, samples, ident(person_id) if person_id else None, ident(batch_id) if batch_id else None, Jsonb(policy), fingerprint(policy))).fetchone()
            for parent in previous_run_ids or []:
                self._link(conn, parent, row['id'], 'next_step')
            self.core._emit(conn, 'experiment_record', row['id'], 1, row)
            return row
        return self._command(key, 'create_run', dict(code=code, experiment_type=experiment_type, sample_ids=sample_ids, person_id=person_id, batch_id=batch_id, previous_run_ids=previous_run_ids, metadata=metadata, policy_instrument_id=policy_instrument_id), action)

    def _link(self, conn, parent_id, child_id, relation):
        self._run(conn, parent_id)
        self._run(conn, child_id)
        if relation not in {'next_step', 'sample_derivation'} or ident(parent_id) == ident(child_id):
            raise ValueError('Invalid experiment relationship')
        cycle = conn.execute('WITH RECURSIVE descendants(id) AS (SELECT %s::uuid UNION SELECT r.child_id FROM hj_run_relation r JOIN descendants d ON r.parent_id=d.id WHERE r.customer_id=%s) SELECT 1 FROM descendants WHERE id=%s', (ident(child_id), self.customer, ident(parent_id))).fetchone()
        if cycle:
            raise Conflict('Experiment lineage cannot contain cycles')
        old = conn.execute('SELECT relation FROM hj_run_relation WHERE customer_id=%s AND parent_id=%s AND child_id=%s', (self.customer, ident(parent_id), ident(child_id))).fetchone()
        if old and old['relation'] != relation:
            raise Conflict('Existing sample relationship has different semantics')
        conn.execute('INSERT INTO hj_run_relation(customer_id,parent_id,child_id,relation) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING', (self.customer, ident(parent_id), ident(child_id), relation))

    def link_runs(self, key, parent_id, child_id, *, relation='next_step'):
        def action(conn):
            self._link(conn, parent_id, child_id, relation)
            for rid in (parent_id, child_id):
                run = self._run(conn, rid)
                self._dirty(conn, run['first_observed_at'], run['last_observed_at'])
            return dict(parent_id=parent_id, child_id=child_id, relation=relation)
        return self._command(key, 'link_runs', dict(parent_id=parent_id, child_id=child_id, relation=relation), action)

    def bind(self, key, run_id, instrument_id, sample_id, at, *, actor, person_id=None,
             position='', semantics='association', evidence_ids=None, label_assignment_id=None, source='hj_business_binding'):
        when = instant(at)
        if semantics not in {'association', 'preparation', 'handover'} or not actor:
            raise ValueError('Binding semantics and actor are required')
        if len(position) > 100:
            raise ValueError('Instrument position too long')

        def action(conn):
            run = self._run(conn, run_id)
            instrument = self.core._entity(conn, instrument_id, 'instrument')
            sample = self.core._entity(conn, sample_id, 'material_unit')
            if ident(sample_id) not in run['sample_ids']:
                raise Conflict('Bound sample does not participate in this experiment')
            wearer = ident(person_id) if person_id else run['person_id']
            if wearer:
                self.core._entity(conn, wearer, 'person')
            if run['person_id'] and wearer != run['person_id']:
                raise Conflict('Binding person differs from this experiment record')
            evidence = self.core._evidence(conn, evidence_ids or [])
            policy = self.policies.record(experiment_type=run['experiment_type'], instrument_id=instrument_id)
            handover = semantics == 'handover' and policy['handover'] == 'exclusive_position'
            snapshot = dict(instrument=instrument, sample=sample, experiment_code=run['code'], experiment_type=run['experiment_type'], experiment_information=run['registration'], actor=actor,
                            person=self.core._entity(conn, wearer, 'person') if wearer else None,
                            batch=self.core._entity(conn, run['batch_id'], 'process_batch') if run['batch_id'] else None,
                            applied_policy=policy)
            row = conn.execute('INSERT INTO hj_experiment_binding(customer_id,id,run_id,instrument_id,sample_id,person_id,position,occurred_at,semantics,handover_effective,evidence_ids,label_assignment_id,source,business_snapshot) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *', (self.customer, ident(), ident(run_id), ident(instrument_id), ident(sample_id), wearer, position, when, semantics, handover, evidence, ident(label_assignment_id) if label_assignment_id else None, source, Jsonb(clean(snapshot)))).fetchone()
            self._touch(conn, run_id, when, when)
            if handover:
                prior = conn.execute("SELECT * FROM hj_experiment_binding WHERE customer_id=%s AND instrument_id=%s AND position=%s AND occurred_at<%s AND semantics<>'preparation' ORDER BY occurred_at DESC,id LIMIT 1", (self.customer, ident(instrument_id), position, when)).fetchone()
                if prior and prior['run_id'] != ident(run_id):
                    # A different active position/instrument can keep this run going.
                    others = conn.execute("SELECT 1 FROM hj_experiment_binding b WHERE b.customer_id=%s AND b.run_id=%s AND b.semantics<>'preparation' AND b.occurred_at<=%s AND (b.instrument_id<>%s OR b.position<>%s) AND NOT EXISTS(SELECT 1 FROM hj_experiment_binding n WHERE n.customer_id=b.customer_id AND n.instrument_id=b.instrument_id AND n.position=b.position AND n.handover_effective AND n.occurred_at>b.occurred_at AND n.occurred_at<=%s) LIMIT 1", (self.customer, prior['run_id'], when, ident(instrument_id), position, when)).fetchone()
                    if not others:
                        self._archive(conn, prior['run_id'], when, 'verified_position_handover', evidence)
            self.core._emit(conn, 'experiment_binding', row['id'], 1, dict(row, actor=actor))
            self._queue_associations(conn, sample_id=sample_id, instrument_id=instrument_id)
            return row
        return self._command(key, 'bind', dict(run_id=run_id, instrument_id=instrument_id, sample_id=sample_id, at=at, actor=actor, person_id=person_id, position=position, semantics=semantics, evidence_ids=evidence_ids, label_assignment_id=label_assignment_id, source=source), action)

    def bind_qr(self, key, run_id, equipment_qr, sample_qr, at, *, actor, person_id=None, position='', semantics='association', scan_evidence_ids=None):
        """Software business binding after decoded QR identities, not hardware pairing."""
        when = instant(at)

        def action(conn):
            resolved = []
            for code, expected in ((equipment_qr, 'instrument'), (sample_qr, 'material_unit')):
                tag = conn.execute("SELECT id FROM entity WHERE customer_id=%s AND kind='tag' AND code=%s", (self.customer, code)).fetchone()
                if not tag:
                    raise Missing('QR identity is not registered for this customer')
                item = self._resolve_label(conn, tag['id'], when)
                if item['status'] != 'resolved':
                    raise Conflict('QR has no unambiguous assignment at scan time')
                self.core._entity(conn, item['target_id'], expected)
                resolved.append(item)
            return self.bind('qr-binding:' + fingerprint(key), run_id, resolved[0]['target_id'], resolved[1]['target_id'], at,
                             actor=actor, person_id=person_id, position=position, semantics=semantics,
                             evidence_ids=scan_evidence_ids, label_assignment_id=resolved[1]['assignment_id'], source='hj_qr_binding')
        return self._command(key, 'bind_qr', dict(run_id=run_id, equipment_qr=equipment_qr, sample_qr=sample_qr, at=at, actor=actor, person_id=person_id, position=position, semantics=semantics, scan_evidence_ids=scan_evidence_ids), action)

    def register_temporal_ocr(self, key, media_id, registration, *, actor, reason):
        """Explicit post-intake temporal readout registration; native intake is unchanged."""
        from .timeline_tasks import TimelineTasks
        return TimelineTasks(self).register_temporal_ocr(key, media_id, registration, actor=actor, reason=reason)

    def admit(self, conn, capture_id, media_id, receipt):
        """Called in the intake transaction, after native completion/hash verification."""
        self.lock(conn)
        kind = receipt['kind']
        duration = seconds(receipt.get('end_seconds') or 0)
        # Native receipt times refer to media zero; evidence ranges are not durations.
        start = instant(receipt['captured_at']) if receipt.get('captured_at') else None
        end = offset_time(start, duration)
        if kind != 'photo' and not duration:
            start = end = None  # Unknown duration cannot establish time coverage.
        row = conn.execute('INSERT INTO hj_timeline_media(customer_id,media_id,capture_id,kind,stream_key,session_key,start_at,end_at,duration_seconds,time_basis) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *', (self.customer, media_id, capture_id, kind, receipt['source_instance'], str(receipt['session_id']) if receipt.get('session_id') else None, start, end, duration, Jsonb(receipt['time_basis']))).fetchone()
        kinds = {'video': ['video_understanding'], 'audio': ['transcription'], 'photo': ['ocr']}[kind]
        if kind == 'video' and receipt.get('native_record', {}).get('embedded_audio'):
            kinds.append('transcription')
        for worker in kinds:
            conn.execute('INSERT INTO hj_timeline_task(customer_id,id,media_id,kind) VALUES(%s,%s,%s,%s)', (self.customer, ident(), media_id, worker))
        self._dirty(conn, start, end)
        return row

    def _spans(self, conn, spans):
        if not spans or len(spans) > 100:
            raise ValueError('Bounded original evidence spans are required')
        refs, starts, ends = [], [], []
        uncertainty = 0
        unknown = False
        for span in spans:
            media = conn.execute('SELECT m.*,a.version AS media_version FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id WHERE m.customer_id=%s AND m.media_id=%s', (self.customer, ident(span['media_id']))).fetchone()
            if not media:
                raise Missing('Evidence media outside this customer timeline')
            raw_uncertainty = media['time_basis'].get('uncertainty_us', 0)
            if type(raw_uncertainty) is not int or raw_uncertainty < 0:
                raise ValueError('Clock uncertainty must be nonnegative integer microseconds')
            uncertainty = max(uncertainty, raw_uncertainty)
            start, end = seconds(span.get('start_seconds', 0)), seconds(span.get('end_seconds', 0))
            if end < start or end > media['duration_seconds'] or (media['kind'] != 'photo' and end == start):
                raise ValueError('Evidence escaped the original media interval')
            eid = ident()
            conn.execute('INSERT INTO evidence_ref(customer_id,id,media_id,media_version,start_seconds,end_seconds) VALUES(%s,%s,%s,%s,%s,%s)', (self.customer, eid, media['media_id'], media['media_version'], start, end))
            refs.append(eid)
            if media['start_at'] is None:
                unknown = True
            else:
                starts.append(offset_time(media['start_at'], start))
                ends.append(offset_time(media['start_at'], end))
        return refs, (None if unknown else min(starts)), (None if unknown else max(ends)), uncertainty

    def _association(self, conn, *, start, end, sample_id, instrument_id, person_id, tag_id, run_ids):
        resolution = self._resolve_label(conn, tag_id, start, end) if tag_id else {}
        if tag_id:
            if resolution.get('status') != 'resolved':
                return [], 'unresolved_label', resolution, sample_id
            if sample_id and ident(sample_id) != resolution['target_id']:
                return [], 'identity_conflict', resolution, sample_id
            sample_id = resolution['target_id']
            self.core._entity(conn, sample_id, 'material_unit')
        if run_ids:
            runs = [self._run(conn, rid) for rid in sorted(set(run_ids), key=str)]
            if any(sample_id and ident(sample_id) not in r['sample_ids'] for r in runs):
                raise Conflict('Explicit experiment does not contain the observed sample')
            if any(person_id and r['person_id'] and ident(person_id) != r['person_id'] for r in runs):
                raise Conflict('Explicit person conflicts with the experiment')
            return [r['id'] for r in runs], 'explicit', resolution, sample_id
        # Time alone, or the wearer's latest binding, is never sufficient.
        if start is None or not sample_id or not instrument_id:
            return [], 'unassigned', resolution, sample_id
        rows = conn.execute("SELECT DISTINCT b.run_id FROM hj_experiment_binding b WHERE b.customer_id=%s AND b.sample_id=%s AND b.instrument_id=%s AND b.occurred_at<=%s AND b.semantics<>'preparation' AND (%s::uuid IS NULL OR b.person_id=%s) AND NOT EXISTS(SELECT 1 FROM hj_experiment_binding n WHERE n.customer_id=b.customer_id AND n.instrument_id=b.instrument_id AND n.position=b.position AND n.handover_effective AND n.occurred_at>b.occurred_at AND n.occurred_at<=%s)", (self.customer, ident(sample_id), ident(instrument_id), start, ident(person_id) if person_id else None, ident(person_id) if person_id else None, start)).fetchall()
        return ([rows[0]['run_id']] if len(rows) == 1 else []), ('bound_identity' if len(rows) == 1 else 'ambiguous' if rows else 'unassigned'), resolution, sample_id

    def add_activity(self, key, kind, source_type, text, spans, *, run_ids=None, sample_id=None,
                     instrument_id=None, person_id=None, tag_id=None, model_revision=None, payload=None):
        if kind not in KINDS or source_type not in {'video', 'speech', 'ocr', 'reviewed_note'}:
            raise ValueError('Unsupported activity kind/source')
        if not isinstance(text, str) or len(text) > 20000:
            raise ValueError('Bounded activity text required')
        if source_type != 'reviewed_note' and not model_revision:
            raise ValueError('Model-generated observations require a model revision')

        def action(conn):
            for value, entity_kind in ((sample_id, 'material_unit'), (instrument_id, 'instrument'), (person_id, 'person')):
                if value:
                    self.core._entity(conn, value, entity_kind)
            refs, start, end, uncertainty = self._spans(conn, spans)
            if (payload or {}).get('field_mapping') == 'temporal_registered_video' and (payload or {}).get('temporal_world_clock_status') != 'verified':
                # The original PTS remains usable when an archive's epoch-looking
                # source clock has not been calibrated. Never resolve a wearer
                # or experiment from that unverified wall clock.
                start = end = None
            delta = timedelta(microseconds=uncertainty)
            runs, status, resolution, resolved_sample = self._association(conn, start=start-delta if start else None, end=end+delta if end else None, sample_id=sample_id, instrument_id=instrument_id, person_id=person_id, tag_id=tag_id, run_ids=run_ids)
            priority = 'low' if kind in {'rest', 'chat', 'waiting'} else 'unknown' if kind == 'unknown' else 'high'
            aid = ident()
            stored_payload = dict(payload or {}, _association_input=clean(dict(sample_id=sample_id, run_ids=run_ids)), clock_uncertainty_us=uncertainty)
            row = conn.execute('INSERT INTO hj_activity(customer_id,id,kind,source_type,text_content,start_at,end_at,sample_id,instrument_id,person_id,tag_id,label_resolution,evidence_ids,payload,model_revision,association_status,priority) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *', (self.customer, aid, kind, source_type, text, start, end, ident(resolved_sample) if resolved_sample else None, ident(instrument_id) if instrument_id else None, ident(person_id) if person_id else None, ident(tag_id) if tag_id else None, Jsonb(clean(resolution)), refs, Jsonb(clean(stored_payload)), model_revision, status, priority)).fetchone()
            for rid in runs:
                conn.execute('INSERT INTO hj_activity_run(customer_id,activity_id,run_id,basis) VALUES(%s,%s,%s,%s)', (self.customer, aid, rid, status))
                self._touch(conn, rid, start, end)
                if kind in {'end', 'unloading'} and end:
                    self._archive(conn, rid, end, 'spoken_end' if source_type == 'speech' else 'visible_end_activity', refs)
            self._dirty(conn, start, end)
            self.core._emit(conn, 'experiment_activity', aid, 1, dict(row, run_ids=runs, occurred_at=start))
            return dict(row, run_ids=runs)
        return self._command(key, 'add_activity', dict(kind=kind, source_type=source_type, text=text, spans=spans, run_ids=run_ids, sample_id=sample_id, instrument_id=instrument_id, person_id=person_id, tag_id=tag_id, model_revision=model_revision, payload=payload), action)

    def _queue_associations(self, conn, *, tag_id=None, sample_id=None, instrument_id=None):
        conn.execute('INSERT INTO hj_timeline_pending_association(customer_id,activity_id) SELECT customer_id,id FROM hj_activity WHERE customer_id=%s AND ((%s::uuid IS NOT NULL AND tag_id=%s) OR (%s::uuid IS NOT NULL AND (sample_id=%s OR sample_id IS NULL) AND instrument_id=%s)) ON CONFLICT DO NOTHING', (self.customer, ident(tag_id) if tag_id else None, ident(tag_id) if tag_id else None, ident(sample_id) if sample_id else None, ident(sample_id) if sample_id else None, ident(instrument_id) if instrument_id else None))

    def _recalculate(self, conn, run_id):
        old = self._run(conn, run_id)
        bounds = conn.execute('SELECT min(lo) AS lo,max(hi) AS hi FROM (SELECT b.occurred_at AS lo,b.occurred_at AS hi FROM hj_experiment_binding b WHERE b.customer_id=%s AND b.run_id=%s UNION ALL SELECT a.start_at,a.end_at FROM hj_activity a JOIN hj_activity_run r ON r.customer_id=a.customer_id AND r.activity_id=a.id WHERE r.customer_id=%s AND r.run_id=%s) x', (self.customer, ident(run_id), self.customer, ident(run_id))).fetchone()
        row = conn.execute('UPDATE hj_experiment_record SET first_observed_at=%s,last_observed_at=%s,revision=revision+1 WHERE customer_id=%s AND id=%s RETURNING *', (bounds['lo'], bounds['hi'], self.customer, ident(run_id))).fetchone()
        self._dirty(conn, old['first_observed_at'], old['last_observed_at'], row['first_observed_at'], row['last_observed_at'])
        self._dirty_run_days(conn, run_id)
        self.core._emit(conn, 'experiment_record', row['id'], row['revision'], dict(row, occurred_at=row['last_observed_at']))

    def _revise_association(self, conn, activity, run_ids, *, actor, reason, resolution, status):
        revision = conn.execute('SELECT COALESCE(MAX(revision),0)+1 AS n FROM hj_activity_association_revision WHERE customer_id=%s AND activity_id=%s', (self.customer, activity['id'])).fetchone()['n']
        row = conn.execute('INSERT INTO hj_activity_association_revision(customer_id,id,activity_id,run_ids,reason,actor,label_resolution,association_status,revision) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *', (self.customer, ident(), activity['id'], [ident(x) for x in run_ids], reason, actor, Jsonb(clean(resolution)), status, revision)).fetchone()
        previous = conn.execute('DELETE FROM hj_activity_run WHERE customer_id=%s AND activity_id=%s RETURNING run_id', (self.customer, activity['id'])).fetchall()
        for rid in set(map(ident, run_ids)):
            conn.execute('INSERT INTO hj_activity_run(customer_id,activity_id,run_id,basis) VALUES(%s,%s,%s,%s)', (self.customer, activity['id'], rid, status))
        for rid in set([ident(x) for x in run_ids] + [x['run_id'] for x in previous]):
            self._recalculate(conn, rid)
        self._dirty(conn, activity['start_at'], activity['end_at'])
        self.core._emit(conn, 'activity_association_revision', row['id'], revision, dict(row, occurred_at=activity['start_at']))
        return row

    def associate_activity(self, key, activity_id, run_ids, *, actor, reason):
        """Append a reviewed association revision; original interpretation is immutable."""
        if not actor or not reason:
            raise ValueError('Association revision needs actor and reason')

        def action(conn):
            activity = conn.execute('SELECT * FROM hj_activity WHERE customer_id=%s AND id=%s', (self.customer, ident(activity_id))).fetchone()
            if not activity:
                raise Missing('Activity outside customer scope')
            for rid in run_ids:
                self._run(conn, rid)
            resolution = self._resolve_label(conn, activity['tag_id'], activity['start_at'], activity['end_at']) if activity['tag_id'] else {}
            return self._revise_association(conn, activity, run_ids, actor=actor, reason=reason, resolution=resolution, status='reviewed_association')
        return self._command(key, 'associate_activity', dict(activity_id=activity_id, run_ids=run_ids, actor=actor, reason=reason), action)

    def reconcile_pending(self, limit=100):
        revised = []
        with self.core.db.transaction() as conn:
            self.lock(conn)
            rows = conn.execute('SELECT a.* FROM hj_timeline_pending_association p JOIN hj_activity a ON a.customer_id=p.customer_id AND a.id=p.activity_id WHERE p.customer_id=%s ORDER BY a.created_at,a.id LIMIT %s', (self.customer, limit)).fetchall()
            for activity in rows:
                latest = conn.execute('SELECT * FROM hj_activity_association_revision WHERE customer_id=%s AND activity_id=%s ORDER BY revision DESC LIMIT 1', (self.customer, activity['id'])).fetchone()
                if not latest or latest['actor'] == 'timeline-system':
                    inputs = activity['payload'].get('_association_input', {})
                    delta = timedelta(microseconds=activity['payload'].get('clock_uncertainty_us', 0))
                    start, end = activity['start_at'], activity['end_at']
                    try:
                        runs, status, resolution, _ = self._association(conn, start=start-delta if start else None, end=end+delta if end else None, sample_id=inputs.get('sample_id'), instrument_id=activity['instrument_id'], person_id=activity['person_id'], tag_id=activity['tag_id'], run_ids=inputs.get('run_ids'))
                    except Conflict:
                        runs, status = [], 'identity_conflict'
                        resolution = self._resolve_label(conn, activity['tag_id'], start, end) if activity['tag_id'] else {}
                    previous = conn.execute('SELECT run_id FROM hj_activity_run WHERE customer_id=%s AND activity_id=%s ORDER BY run_id', (self.customer, activity['id'])).fetchall()
                    old_resolution = latest['label_resolution'] if latest else activity['label_resolution']
                    old_status = latest['association_status'] if latest else activity['association_status']
                    if set(runs) != {r['run_id'] for r in previous} or fingerprint(clean(resolution)) != fingerprint(old_resolution) or status != old_status:
                        self._revise_association(conn, activity, runs, actor='timeline-system', reason='historical_binding_evidence_arrived', resolution=resolution, status=status)
                        revised.append(str(activity['id']))
                conn.execute('DELETE FROM hj_timeline_pending_association WHERE customer_id=%s AND activity_id=%s', (self.customer, activity['id']))
        return dict(revised_activities=revised)

    def close_due(self, now=None, limit=100):
        now = instant(now) if now else datetime.now(timezone.utc)
        closed = []
        with self.core.db.transaction() as conn:
            self.lock(conn)
            rows = conn.execute("""SELECT * FROM hj_experiment_record WHERE customer_id=%s AND record_status='open' AND last_observed_at IS NOT NULL
              AND LEAST(CASE WHEN (policy->>'daily_archive')::boolean THEN
                ((date_trunc('day',last_observed_at AT TIME ZONE 'Asia/Shanghai') + interval '1 day') AT TIME ZONE 'Asia/Shanghai') END,
                last_observed_at + (policy->>'inactivity_seconds')::integer * interval '1 second')
                + (policy->>'arrival_grace_seconds')::integer * interval '1 second' <= %s
              ORDER BY last_observed_at,id LIMIT %s""", (self.customer, now, limit)).fetchall()
            for row in rows:
                policy = RecordPolicy(**row['policy'])
                candidates = []
                if policy.daily_archive:
                    next_day = row['last_observed_at'].astimezone(ZONE).date() + timedelta(days=1)
                    candidates.append((datetime.combine(next_day, time.min, ZONE), 'daily_record_archive'))
                if policy.inactivity_seconds is not None:
                    candidates.append((row['last_observed_at'] + timedelta(seconds=policy.inactivity_seconds), 'record_inactivity_archive'))
                deadline, reason = min(candidates)
                if now >= deadline + timedelta(seconds=policy.arrival_grace_seconds):
                    if self._archive(conn, row['id'], now, reason):
                        closed.append(str(row['id']))
        return dict(archived=closed, physical_completion_inferred=False)
