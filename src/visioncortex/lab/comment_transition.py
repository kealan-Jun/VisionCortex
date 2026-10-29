"""Thin, opt-in application of persisted reviewed oral Comment interpretations.

This adapter does not transcribe, parse prose or independently verify semantics.
It pins existing speech and original audio evidence plus a separate reviewed_note.
All prose remains untrusted evidence data; stage completion never ends a run.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import re

from .domain import Conflict, Missing, clean, fingerprint, ident, instant
from .operator_scan import OperatorScans, _text, declaration_activity
from .rules import Rules
from .timeline import Timeline

SCHEMA = 'vc-lab-comment-transition/1'
INTERPRETATIONS = {'stage_start', 'stage_complete', 'next_stage_plan', 'experiment_complete'}


class CommentTransitions:
    def __init__(self, core):
        self.core, self.customer = core, core.customer
        self.timeline = Timeline(core)
        self.scopes = OperatorScans(core)

    def _activity(self, conn, activity_id, source_type):
        row = conn.execute('SELECT * FROM hj_activity WHERE customer_id=%s AND id=%s',
                           (self.customer, ident(activity_id))).fetchone()
        if row is None:
            raise Missing('Comment activity outside customer scope')
        if row['source_type'] != source_type:
            raise Conflict('Comment source or review has the wrong activity type')
        return row

    def _audio(self, conn, source, evidence_ids, camera_id):
        """Pin stored source identity, extent and clock without claiming byte re-read."""
        if not evidence_ids or len(evidence_ids) > 16 or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError('One to sixteen unique original audio evidence references required')
        ids = self.core._evidence(conn, evidence_ids)
        if set(ids) != set(source['evidence_ids']):
            raise Conflict('Comment evidence differs from the selected speech activity')
        rows = conn.execute('''SELECT e.*,m.kind,m.stream_key,m.session_key,
            m.start_at AS media_start_at,m.duration_seconds,m.time_basis,
            a.version AS current_version,a.verified_sha256,a.evidence_status,
            c.source_kind,c.source_capture_id,c.source_revision,i.payload AS receipt
            FROM evidence_ref e JOIN hj_timeline_media m USING(customer_id,media_id)
            JOIN media_asset a ON a.customer_id=e.customer_id AND a.id=e.media_id
            JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
            JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
            WHERE e.customer_id=%s AND e.id=ANY(%s) ORDER BY e.media_id,e.start_seconds,e.id''',
                            (self.customer, ids)).fetchall()
        if len(rows) != len(ids):
            raise Conflict('Original audio provenance is incomplete')
        camera = self.core._entity(conn, camera_id, 'capture_device') if camera_id else None
        pins, starts, ends, attention = [], [], [], []
        for row in rows:
            receipt, basis = row['receipt'], row['time_basis']
            if (row['kind'] != 'audio' and not (row['kind'] == 'video' and
                    receipt.get('native_record', {}).get('embedded_audio') is True)):
                raise Conflict('Comment requires original audio or explicitly registered embedded audio')
            if (row['media_version'] != row['current_version'] or row['evidence_status'] != 'available'
                    or not re.fullmatch('[0-9a-f]{64}', row['verified_sha256'] or '')
                    or receipt.get('sha256') != row['verified_sha256']):
                raise Conflict('Comment source version/hash is unavailable or conflicts with the receipt')
            lo, hi = row['start_seconds'], row['end_seconds']
            if lo is None or hi is None or not 0 <= lo < hi <= row['duration_seconds']:
                raise Conflict('Comment source range is missing or outside the original media')
            if camera and row['stream_key'] != camera['code'] and receipt.get('camera_id') not in {str(camera['id']), camera['code']}:
                raise Conflict('Comment source camera conflicts with the exact declared scope')
            if (type(basis) is not dict or not isinstance(basis.get('basis'), str) or not basis['basis']
                    or type(basis.get('uncertainty_us')) is not int or basis['uncertainty_us'] != 0
                    or basis.get('clock_sync_valid') is False or basis.get('clock_sync_verified') is False
                    or row['media_start_at'] is None):
                attention.append('unknown_or_uncertain_original_audio_clock')
            if (fingerprint(basis) != fingerprint(receipt.get('time_basis')) or
                    (instant(receipt['captured_at']) if receipt.get('captured_at') else None) != row['media_start_at']):
                raise Conflict('Comment source clock metadata differs from the original receipt')
            start = row['media_start_at'] + timedelta(microseconds=int(lo * 1000000)) if row['media_start_at'] else None
            end = row['media_start_at'] + timedelta(microseconds=int(hi * 1000000)) if row['media_start_at'] else None
            if start is not None:
                starts.append(start); ends.append(end)
            pins.append(clean(dict(evidence_id=row['id'], media_id=row['media_id'], media_version=row['media_version'],
                source_sha256=row['verified_sha256'], start_seconds=lo, end_seconds=hi,
                occurred_at=start, ended_at=end, time_basis=basis, source_kind=row['source_kind'],
                stream_key=row['stream_key'], session_key=row['session_key'],
                source_capture_id=row['source_capture_id'], source_revision=row['source_revision'],
                original_audio_reference='/internal/media/' + str(row['media_id']),
                receipt_sha256=fingerprint(receipt))))
        if (source['start_at'] != (min(starts) if len(starts) == len(rows) else None) or
                source['end_at'] != (max(ends) if len(ends) == len(rows) else None)):
            raise Conflict('Speech activity time differs from its original audio extent')
        return ids, pins, sorted(set(attention))

    def apply(self, key, *, interpretation, run_id, sample_id, batch_id,
              source_activity_id, source_text_sha256, source_evidence_ids, text,
              occurred_at, actor, reason, person_id=None, camera_id=None, review_activity_id=None):
        if interpretation not in INTERPRETATIONS:
            raise ValueError('Explicit structured Comment interpretation required')
        _text(actor, 'review application actor'); _text(reason, 'review application reason')
        _text(text, 'original Comment text', 80000)
        if type(source_text_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', source_text_sha256):
            raise ValueError('Original speech text SHA256 required')
        if not isinstance(source_evidence_ids, list):
            raise ValueError('Explicit original audio evidence list required')
        at = instant(occurred_at) if occurred_at is not None else None
        request = dict(interpretation=interpretation, run_id=run_id, sample_id=sample_id, batch_id=batch_id,
            source_activity_id=source_activity_id, source_text_sha256=source_text_sha256,
            source_evidence_ids=source_evidence_ids, text=text, occurred_at=occurred_at, actor=actor,
            reason=reason, person_id=person_id, camera_id=camera_id, review_activity_id=review_activity_id)

        def action(conn):
            self.timeline.lock(conn)
            self.scopes._scope(conn, person_id, camera_id)
            run = self.timeline._run(conn, run_id)
            self.core._entity(conn, sample_id, 'material_unit')
            if ident(sample_id) not in run['sample_ids']:
                raise Conflict('Comment sample is not an explicit member of the selected run')
            if batch_id:
                self.core._entity(conn, batch_id, 'process_batch')
            if run['batch_id'] != (ident(batch_id) if batch_id else None):
                raise Conflict('Comment target process batch differs from the selected run')
            source = self._activity(conn, source_activity_id, 'speech')
            if text != source['text_content'] or hashlib.sha256(text.encode()).hexdigest() != source_text_sha256:
                raise Conflict('Comment text or SHA differs from the original speech activity')
            if source['sample_id'] and source['sample_id'] != ident(sample_id):
                raise Conflict('Comment source sample conflicts with the declared sample')
            if source['person_id'] and source['person_id'] != (ident(person_id) if person_id else None):
                raise Conflict('Comment person scope conflicts with the explicit speech person')
            linked = conn.execute('SELECT run_id FROM hj_activity_run WHERE customer_id=%s AND activity_id=%s',
                                  (self.customer, source['id'])).fetchall()
            if linked and {r['run_id'] for r in linked} != {ident(run_id)}:
                raise Conflict('Comment source has a conflicting experiment association')
            evidence, pins, attention = self._audio(conn, source, source_evidence_ids, camera_id)
            if at is None:
                attention.append('unknown_comment_event_time')
            elif not attention and not any(p['occurred_at'] is not None and instant(p['occurred_at']) <= at <= instant(p['ended_at']) for p in pins):
                raise Conflict('Comment event time falls outside the original audio spans')
            review = None
            expected = clean(dict(source_activity_id=source['id'], source_text_sha256=source_text_sha256,
                source_evidence_ids=sorted(map(str, evidence)), interpretation=interpretation,
                run_id=run_id, sample_id=sample_id, batch_id=batch_id,
                person_id=person_id, camera_id=camera_id))
            if review_activity_id is None:
                attention.append('persisted_structured_interpretation_review_required')
            else:
                review = self._activity(conn, review_activity_id, 'reviewed_note')
                declared = review['payload'].get('comment_interpretation_review')
                if not isinstance(declared, dict) or any(declared.get(k) != v for k, v in expected.items()):
                    raise Conflict('Persisted Comment review does not pin the exact interpretation/source/scope')
                if declared.get('status') != 'reviewed' or not declared.get('actor') or not declared.get('reason'):
                    attention.append('interpretation_review_is_pending_or_ambiguous')
                if not review['evidence_ids']:
                    raise Conflict('Comment review lacks original audio evidence')
                reviewed = conn.execute('SELECT media_id,media_version,start_seconds,end_seconds FROM evidence_ref WHERE customer_id=%s AND id=ANY(%s)',
                                        (self.customer, review['evidence_ids'])).fetchall()
                expected_ranges = {(p['media_id'], p['media_version'], p['start_seconds'], p['end_seconds']) for p in pins}
                if {(str(r['media_id']), r['media_version'], str(r['start_seconds']), str(r['end_seconds'])) for r in reviewed} != expected_ranges:
                    raise Conflict('Comment review audio ranges differ from the source speech')
            if interpretation == 'experiment_complete':
                if run['record_status'] != 'open':
                    attention.append('run_is_already_archived')
                if at is not None and run['last_observed_at'] and run['last_observed_at'] > at:
                    attention.append('late_completion_conflicts_with_later_run_activity')
            status = 'needs_attention' if attention else 'applied_reviewed_interpretation'
            # Unknown/uncertain clock must not acquire a day or inherited current identity.
            effective_at = at if not any('clock' in r or r == 'unknown_comment_event_time' for r in attention) else None
            payload = clean(dict(schema_version=SCHEMA, processing_rules=Rules().identity(),
                comment_transition=interpretation, status=status, attention_reasons=sorted(set(attention)),
                source_activity_id=source['id'], source_activity_sha256=fingerprint(source),
                source_text_sha256=source_text_sha256, original_text=text, source_audio=pins,
                source_evidence_ids=evidence, declared_occurred_at=at, person_id=person_id, camera_id=camera_id,
                run_id=run_id, sample_id=sample_id, batch_id=batch_id, actor=actor, reason=reason,
                review_activity_id=review['id'] if review else None,
                review_activity_sha256=fingerprint(review) if review else None,
                semantic_authority='persisted_upstream_review_not_independent_asr_semantic_verification',
                speaker_attribution='not_inferred_from_wearer_or_operator_context',
                planned_stage_started=False, declared_stage_started=status == 'applied_reviewed_interpretation' and interpretation == 'stage_start',
                equipment_released=False, quantity_ledger_written=False,
                whole_experiment_completed=status == 'applied_reviewed_interpretation' and interpretation == 'experiment_complete',
                text_is_untrusted_evidence_data=True, original_media_bytes_reverified=False))
            activity_id = declaration_activity(conn, self.core, text=text, at=effective_at, evidence_ids=evidence,
                payload=payload, run_id=run_id, sample_id=sample_id, person_id=person_id,
                kind='end' if payload['whole_experiment_completed'] else 'settings')
            from .operation_segments import mark_operation_reports_dirty
            mark_operation_reports_dirty(conn, self.customer, effective_at)
            if payload['whole_experiment_completed']:
                self.timeline._archive(conn, run_id, at, 'explicit_reviewed_whole_experiment_comment', evidence)
            return dict(activity_id=activity_id, status=status, snapshot=payload)
        return self.core.command(key, dict(schema_version=SCHEMA, **request), action)
