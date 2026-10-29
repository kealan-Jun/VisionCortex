"""Automatic, source-pinned Comment intake; prose is never a completion rule.

ASR candidates stay pending without an independently persisted structured
interpretation. Existing reviewed interpretations use CommentTransitions; this
module does not manufacture reviews, infer a speaker, or call a semantic model.
"""
from __future__ import annotations

import hashlib

from .comment_transition import CommentTransitions
from .domain import Conflict, Missing, clean, fingerprint, ident

SCHEMA = 'vc-lab-comment-intake/1'
PREFIX = 'comment-intake:1:'


def transcript_candidate(ref, spans, execution, *, aligned, original_asr_text=None):
    """Checkpoint the original interval separately from the encoded ASR input."""
    return clean(dict(schema_version=SCHEMA, status='pending_interpretation',
        original_source=dict(media_id=ref['media_id'], media_version=ref['media_version'],
                             sha256=ref['sha256'], source_kind=ref['source_kind']),
        original_source_spans=spans, alignment_available=aligned,
        original_asr_text=original_asr_text,
        original_asr_text_scope='whole_prepared_input_may_include_context',
        asr_execution=execution, semantic_interpretation=None, speaker_identity=None,
        attention_reasons=['persisted_structured_interpretation_required', 'speaker_identity_unknown'],
        physical_action_confirmed=False, whole_experiment_completed=False))


class CommentIntake:
    def __init__(self, core):
        self.core, self.customer = core, core.customer
        self.transitions = CommentTransitions(core)

    def consume(self, source_activity_id, *, review_activity_id=None):
        """Durable idempotent receipt, separate from the immutable ASR activity."""
        source_id = ident(source_activity_id)
        review_id = ident(review_activity_id) if review_activity_id else None
        key = PREFIX + ('review:' + str(review_id) if review_id else 'source:' + str(source_id))

        def action(conn):
            self.transitions.timeline.lock(conn)
            source = self.transitions._activity(conn, source_id, 'speech')
            text = source['text_content']
            digest = hashlib.sha256(text.encode()).hexdigest()
            reasons, pins = [], []
            try:
                _, pins, attention = self.transitions._audio(conn, source, list(source['evidence_ids']), None)
                reasons.extend(attention)
            except (Conflict, Missing, ValueError) as exc:
                reasons.append('original_audio_provenance_unavailable_or_conflicting')
                provenance_error = type(exc).__name__
            else:
                provenance_error = None
            if source['person_id'] is None:
                reasons.append('speaker_identity_unknown')
            if source['start_at'] is None or source['end_at'] is None:
                reasons.append('unknown_comment_event_time')
            if (source['payload'].get('contains_context') or
                    source['payload'].get('comment_candidate', {}).get('alignment_available') is False):
                reasons.append('precise_transcript_alignment_unavailable')
            result = clean(dict(schema_version=SCHEMA, status='pending_interpretation',
                source_activity_id=source_id, source_text_sha256=digest, original_text=text,
                source_model_revision=source['model_revision'], source_evidence_ids=source['evidence_ids'],
                source_audio=pins, provenance_error=provenance_error,
                review_activity_id=review_id, semantic_interpretation=None,
                speaker_attribution='explicit_persisted_speech_person_only_never_wearer_inference',
                whole_experiment_completed=False, physical_action_confirmed=False,
                quantity_ledger_written=False))
            if review_id is None:
                reasons.append('persisted_structured_interpretation_required')
            else:
                review = self.transitions._activity(conn, review_id, 'reviewed_note')
                declared = review['payload'].get('comment_interpretation_review')
                if not isinstance(declared, dict):
                    reasons.append('persisted_structured_interpretation_required')
                elif (declared.get('status') != 'reviewed' or not declared.get('actor')
                      or not declared.get('reason')):
                    reasons.append('interpretation_is_pending_or_ambiguous')
                else:
                    # These are explicit upstream identities, never current
                    # scanner/wearer context or tokens guessed from ASR prose.
                    if not declared.get('run_id') or not declared.get('sample_id'):
                        reasons.append('explicit_run_and_sample_required')
                    if declared.get('person_id') != str(source['person_id']):
                        reasons.append('speaker_scope_missing_or_conflicting')
                    if not reasons:
                        try:
                            # A rejected upstream interpretation cannot leave a
                            # partial transition in the enclosing intake receipt.
                            with conn.transaction():
                                applied = self.transitions.apply(PREFIX + 'apply:' + str(review_id),
                                    interpretation=declared.get('interpretation'),
                                    run_id=declared.get('run_id'), sample_id=declared.get('sample_id'),
                                    batch_id=declared.get('batch_id'), source_activity_id=str(source_id),
                                    source_text_sha256=digest, source_evidence_ids=list(source['evidence_ids']),
                                    text=text, occurred_at=source['end_at'], actor='automatic-comment-intake',
                                    reason='consume persisted source-pinned upstream interpretation',
                                    person_id=declared.get('person_id'), camera_id=declared.get('camera_id'),
                                    review_activity_id=str(review_id))
                        except (Conflict, Missing, ValueError):
                            reasons.append('structured_interpretation_source_or_target_conflict')
                        else:
                            result.update(status=applied['status'], transition=applied,
                                          semantic_interpretation=declared['interpretation'],
                                          whole_experiment_completed=applied['snapshot']['whole_experiment_completed'])
            result['attention_reasons'] = sorted(set(reasons))
            return result

        return self.core.command(key, dict(schema_version=SCHEMA, source_activity_id=str(source_id),
                                          review_activity_id=str(review_id) if review_id else None), action)

    def consume_ready(self, *, limit=20, task_id=None):
        """Newest first, bounded; late reviews remain independently discoverable.

        Completion of ASR and Comment intake are separate durable transactions.
        A crash after ASR submit resumes here without claiming or inferring again.
        """
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Bounded Comment intake limit required')
        task = str(ident(task_id)) if task_id else None
        with self.core.db.transaction(readonly=True) as conn:
            sources = conn.execute("""SELECT a.id FROM hj_activity a
                WHERE a.customer_id=%s AND a.source_type='speech'
                AND (%s::text IS NULL OR a.payload->>'task_id'=%s)
                AND NOT EXISTS(SELECT 1 FROM command_receipt c WHERE c.customer_id=a.customer_id
                    AND c.key=%s||a.id::text)
                ORDER BY a.start_at DESC NULLS LAST,a.created_at DESC,a.id LIMIT %s""",
                                   (self.customer, task, task, PREFIX+'source:', limit)).fetchall()
            reviews = conn.execute("""SELECT r.id,a.id AS source_id FROM hj_activity r
                JOIN hj_activity a ON a.customer_id=r.customer_id AND a.source_type='speech'
                    AND a.id::text=r.payload#>>'{comment_interpretation_review,source_activity_id}'
                WHERE r.customer_id=%s AND r.source_type='reviewed_note'
                AND (%s::text IS NULL OR a.payload->>'task_id'=%s)
                AND NOT EXISTS(SELECT 1 FROM command_receipt c WHERE c.customer_id=r.customer_id
                    AND c.key=%s||r.id::text)
                ORDER BY a.start_at DESC NULLS LAST,r.created_at DESC,r.id LIMIT %s""",
                                   (self.customer, task, task, PREFIX+'review:', limit)).fetchall()
        results = [self.consume(row['id']) for row in sources]
        results.extend(self.consume(row['source_id'], review_activity_id=row['id']) for row in reviews)
        return dict(schema_version=SCHEMA, processed=len(results), results=results,
                    result_fingerprint=fingerprint(results), models_invoked=0)
