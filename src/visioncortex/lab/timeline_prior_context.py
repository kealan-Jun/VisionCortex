"""Bounded, read-only historical evidence for video understanding.

Physical time and knowledge time are separate: a late correction cannot change
what an already claimed attempt knew. Entries remain candidates or declarations;
neither a report nor a previous model answer becomes a new physical observation.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
import hashlib
import json

from .domain import Conflict, clean, fingerprint, ident, instant
from .timeline import offset_time, seconds

SCHEMA = 'vc-lab-multimodal-prior-context/1'
_PAYLOAD_KEYS = ('fields', 'value_candidate', 'comment_candidate',
                 'comment_interpretation_review', 'operator_scan', 'operator_context',
                 'scan_intake', 'status', 'conflicts', 'temporal_world_clock_status',
                 'text', 'tokens', 'decoded_values', 'raw_text', 'readings', 'video_qr',
                 'normalized_fields', 'source_activity_id', 'original_audio_sha256',
                 'source_start_seconds', 'source_end_seconds', 'native_transcript_reference',
                 'native_asr_models', 'native_alignment_status', 'unaligned_units')


def _bytes(value):
    return len(json.dumps(clean(value), ensure_ascii=False, separators=(',', ':')).encode())


def _clock(row):
    basis = row['time_basis']
    if (row['start_at'] is None or not isinstance(basis, dict) or not basis.get('basis')
            or type(basis.get('uncertainty_us')) is not int or basis['uncertainty_us'] != 0
            or basis.get('clock_sync_verified') is False or basis.get('clock_sync_valid') is False):
        return False
    if (fingerprint(row['receipt']) != row['receipt_fingerprint']
            or fingerprint(basis) != fingerprint(row['receipt'].get('time_basis'))
            or not row['receipt'].get('captured_at')
            or instant(row['receipt']['captured_at']) != row['start_at']):
        return False
    return (basis.get('clock_sync_verified') is True or basis.get('clock_sync_valid') is True
            or (row['receipt'].get('source_kind') == 'fixture'
                and basis['basis'] in {'synthetic_fixture_clock', 'fixture_receipt'}))


def _media(conn, customer, media_id):
    return conn.execute('''SELECT m.*,a.version,a.verified_sha256,a.evidence_status,
        c.source_kind,c.source_capture_id,c.source_revision,i.payload AS receipt,
        i.fingerprint AS receipt_fingerprint,e.id AS camera_id
        FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
        JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
        JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
          AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
        LEFT JOIN entity e ON e.customer_id=c.customer_id AND e.kind='capture_device'
          AND e.code=i.payload->>'camera_id'
        WHERE m.customer_id=%s AND m.media_id=%s''', (customer, ident(media_id))).fetchone()


def _pins(conn, customer, ids, scope, cutoff, cache, *, window_start=None):
    pins = []
    for eid in ids:
        evidence = conn.execute('SELECT * FROM evidence_ref WHERE customer_id=%s AND id=%s',
                                (customer, eid)).fetchone()
        if evidence is None:
            return None, 'missing_evidence'
        mid = str(evidence['media_id'])
        if mid not in cache:
            cache[mid] = _media(conn, customer, mid)
        media = cache[mid]
        if (media is None or media['evidence_status'] != 'available'
                or media['version'] != evidence['media_version']
                or media['receipt'].get('sha256') not in {None, media['verified_sha256']}):
            return None, 'unavailable_or_superseded_evidence'
        if str(media['camera_id']) != scope['camera_id']:
            return None, 'different_or_unknown_camera'
        if not scope['run_id'] and window_start is None and media['session_key'] != scope['session_key']:
            return None, 'different_or_unknown_session_background'
        if not _clock(media):
            return None, 'unknown_source_clock'
        if evidence['start_seconds'] is None or evidence['end_seconds'] is None:
            return None, 'unknown_evidence_extent'
        lo, hi = evidence['start_seconds'], evidence['end_seconds']
        if not 0 <= lo <= hi <= media['duration_seconds']:
            return None, 'invalid_evidence_extent'
        # Original extents are half-open: a nonempty previous packet may end
        # exactly where this packet begins. A point event at the boundary is
        # not history and cannot borrow the first current frame.
        if (offset_time(media['start_at'], hi) > cutoff
                or offset_time(media['start_at'], lo) >= cutoff):
            return None, 'evidence_reaches_current_or_future_input'
        if window_start is not None and offset_time(media['start_at'], lo) < window_start:
            return None, 'contemporaneous_evidence_crosses_core_boundary'
        pins.append(clean(dict(evidence_id=eid, media_id=mid, media_version=media['version'],
            sha256=media['verified_sha256'], start_seconds=lo, end_seconds=hi,
            start_at=offset_time(media['start_at'], lo), end_at=offset_time(media['start_at'], hi),
            time_basis=media['time_basis'], camera_id=media['camera_id'],
            source_kind=media['source_kind'], capture_id=media['capture_id'],
            source_capture_id=media['source_capture_id'], source_revision=media['source_revision'],
            receipt_sha256=fingerprint(media['receipt']), session_key=media['session_key'],
            cross_channel_session=media['session_key'] != scope['session_key'])))
    return pins, None


def _association(conn, customer, row, knowledge):
    revision = conn.execute('''SELECT * FROM hj_activity_association_revision
        WHERE customer_id=%s AND activity_id=%s AND created_at<%s
        ORDER BY revision DESC LIMIT 1''', (customer, row['id'], knowledge)).fetchone()
    if revision:
        return list(map(str, revision['run_ids'])), clean(dict(
            revision_id=revision['id'], revision=revision['revision'], created_at=revision['created_at'],
            status=revision['association_status'], label_resolution=revision['label_resolution']))
    # Current hj_activity_run is mutable and may contain a later reassociation.
    original = row['payload'].get('_association_input', {}).get('run_ids') or []
    return list(map(str, original)), clean(dict(status=row['association_status'],
        label_resolution=row['label_resolution'], basis='original_explicit_input'))


def _observations(conn, customer, ids, knowledge):
    rows = conn.execute('''SELECT * FROM observation WHERE customer_id=%s
        AND evidence_ids && %s AND created_at<%s ORDER BY created_at,id LIMIT 9''',
        (customer, ids, knowledge)).fetchall() if ids else []
    result = []
    for row in rows[:8]:
        # Include only the observation whose entire original evidence was pinned.
        if not set(row['evidence_ids']) <= set(ids):
            continue
        facts = conn.execute('''SELECT DISTINCT ON(f.candidate_id) f.id,f.candidate_id,
            f.revision,f.scope,f.value,f.unit,f.confirmed_at FROM fact_revision f
            JOIN field_candidate c ON c.customer_id=f.customer_id AND c.id=f.candidate_id
            WHERE f.customer_id=%s AND c.observation_id=%s AND f.confirmed_at<%s
            ORDER BY f.candidate_id,f.revision DESC''', (customer, row['id'], knowledge)).fetchall()
        result.append(clean(dict(observation_id=row['id'], kind=row['kind'],
            source_event_id=row['source_event_id'], source_revision=row['source_revision'],
            source_kind=row['source_kind'], occurred_at=row['occurred_at'], created_at=row['created_at'],
            payload_sha256=fingerprint(row['payload']), evidence_ids=row['evidence_ids'],
            original_candidate_payload={k: row['payload'][k] for k in _PAYLOAD_KEYS if k in row['payload']},
            candidate_status='original_candidate_not_current_mutable_field_status',
            confirmed_fact_revisions_asof=facts)))
    return result, len(rows) > 8


def _source_qr(conn, customer, row, knowledge):
    """The existing QR decoder uses source_type=video, but is not a VLM answer."""
    task_id = row['payload'].get('task_id')
    if (not task_id or not row['payload'].get('video_qr')
            or row['model_revision'] != 'model-free-video-qr/1'):
        return False
    try:
        if type(task_id) is not str:
            return False
        task_id = ident(task_id)
    except (ValueError, TypeError):
        return False
    task = conn.execute('''SELECT result FROM hj_timeline_task WHERE customer_id=%s AND id=%s
        AND kind='qr' AND status='completed' AND model_revision='model-free-video-qr/1'
        AND completed_at<%s''', (customer, task_id, knowledge)).fetchone()
    return bool(task and isinstance(task['result'], dict)
        and str(row['id']) in task['result'].get('activity_ids', [])
        and task['result'].get('execution_verification') == 'local_model_free_qr_pixels_verified'
        and any(fingerprint(e.get('payload', {}).get('video_qr')) == fingerprint(row['payload']['video_qr'])
                for e in task['result'].get('events', [])))


def build_prior_context(core, request, window, policy):
    """Return an immutable prompt snapshot; this function performs no writes."""
    document = dict(schema_version=SCHEMA, enabled=True, historical_prior=[], contemporaneous_evidence=[], omitted={},
                    status='unavailable', accuracy='NOT_PROVEN',
                    semantics='source_pinned_prior_and_contemporaneous_candidates_never_new_physical_actions')

    def done(reason=None):
        if reason:
            document['reason'] = reason
        result = clean(document)
        if _bytes(result) + 80 > policy['max_utf8_bytes']:
            result = {key: result[key] for key in ('schema_version', 'enabled', 'task_id',
                      'physical_cutoff', 'core_end', 'knowledge_cutoff') if key in result} | dict(
                status='context_capacity_rejected', reason='provenance_metadata_exceeds_byte_budget',
                rejected_metadata_sha256=fingerprint(document), historical_prior=[], contemporaneous_evidence=[])
        result['sha256'] = fingerprint(result)
        if _bytes(result) > policy['max_utf8_bytes']:
            raise ValueError('Prior context metadata cannot fit the configured byte budget')
        return result

    if not request.get('task_id') or not request.get('input_fingerprint'):
        return done('persisted_attempt_required')
    with core.db.transaction(readonly=True) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        task = conn.execute('''SELECT t.*,a.started_at AS knowledge_cutoff,a.request AS attempt_request
            FROM hj_timeline_task t JOIN hj_timeline_task_attempt a
              ON a.customer_id=t.customer_id AND a.task_id=t.id AND a.token=t.lease_token
            WHERE t.customer_id=%s AND t.id=%s AND t.status='leased' AND a.outcome='leased' ''',
            (core.customer, ident(request['task_id']))).fetchone()
        if task is None or fingerprint(task['attempt_request']) != fingerprint(request):
            return done('persisted_attempt_request_mismatch')
        knowledge = task['knowledge_cutoff']
        current = _media(conn, core.customer, window['core']['media_id'])
        if current is None:
            return done('current_source_clock_unknown')
        if not _clock(current):
            from .timeline_source_context import source_relative_context
            return source_relative_context(conn, core, request, window, current, task, policy)
        if current['camera_id'] is None:
            return done('current_source_camera_unregistered')
        primary = [r for r in window['input_spans'] if r['media_id'] == str(current['media_id'])]
        if (len(primary) != 1 or current['evidence_status'] != 'available'
                or primary[0]['media_version'] != current['version']
                or primary[0]['sha256'] != current['verified_sha256']
                or current['receipt'].get('sha256') not in {None, current['verified_sha256']}):
            raise Conflict('Prior context current source identity differs from the claimed input')
        explicit = request.get('context_scope') or {}
        runs = (request.get('retained_clip') or {}).get('run_ids') or []
        if len(runs) > 1 and not explicit.get('run_id'):
            return done('multiple_experiments_require_explicit_context_scope')
        rid = str(ident(explicit['run_id'])) if explicit.get('run_id') else str(ident(runs[0])) if runs else None
        camera = str(current['camera_id'])
        if explicit.get('camera_id') and str(ident(explicit['camera_id'])) != camera:
            return done('explicit_camera_conflicts_with_original_source')
        if rid and runs and rid not in list(map(str, runs)):
            return done('explicit_run_conflicts_with_retained_clip')
        person = str(ident(explicit['person_id'])) if explicit.get('person_id') else None
        if person and conn.execute("SELECT id FROM entity WHERE customer_id=%s AND id=%s AND kind='person'",
                                  (core.customer, ident(person))).fetchone() is None:
            return done('explicit_person_unregistered')
        if rid and conn.execute('SELECT id FROM hj_experiment_record WHERE customer_id=%s AND id=%s AND created_at<%s',
                                (core.customer, ident(rid), knowledge)).fetchone() is None:
            return done('explicit_experiment_unavailable_at_cutoff')
        if not rid and not current['session_key']:
            return done('background_requires_same_explicit_recording_session')
        scope = dict(run_id=rid, camera_id=camera, person_id=person, session_key=current['session_key'],
                     mode='same_explicit_experiment' if rid else 'same_camera_session_background',
                     person_attribution='explicit_person_scope' if person else 'unattributed_correlation_only',
                     panel_attribution='never_inferred_from_current_qr_equipment')
        cutoff = offset_time(current['start_at'], seconds(window['core']['start_seconds']))
        window_end = offset_time(current['start_at'], seconds(window['core']['end_seconds']))
        lower = cutoff - timedelta(seconds=policy['lookback_seconds'])
        document.update(status='bounded_prior_asof', scope=scope, physical_cutoff=cutoff,
                        core_end=window_end, knowledge_cutoff=knowledge, task_id=task['id'],
                        input_fingerprint=request['input_fingerprint'],
                        lower_bound=lower, cross_package_stage_end_inferred=False)
        omitted, cache = Counter(), {}
        rows = conn.execute('''SELECT * FROM hj_activity WHERE customer_id=%s AND created_at<%s
            AND (end_at>=%s OR end_at IS NULL)
            AND (start_at<%s OR start_at IS NULL)
            ORDER BY end_at DESC NULLS LAST,id LIMIT 129''',
            (core.customer, knowledge, lower, window_end)).fetchall()
        omitted['query_budget'] += int(len(rows) > 128)
        for row in rows[:128]:
            if row['payload'].get('task_id') == str(task['id']):
                omitted['current_task_output'] += 1
                continue
            if row['start_at'] is None or row['end_at'] is None:
                omitted['unknown_activity_clock'] += 1
                continue
            if row['end_at'] > window_end:
                omitted['activity_extends_beyond_core_end'] += 1
                continue
            contemporaneous = row['start_at'] >= cutoff
            if not contemporaneous and row['end_at'] > cutoff:
                omitted['activity_crosses_core_boundary'] += 1
                continue
            source_qr = contemporaneous and row['source_type'] == 'video' and _source_qr(conn, core.customer, row, knowledge)
            if contemporaneous and not (row['source_type'] in {'speech', 'ocr'} or source_qr
                    or (row['source_type'] == 'reviewed_note'
                        and (row['payload'].get('operator_scan_event') or row['payload'].get('scan_intake')))):
                omitted['current_window_non_channel_or_video_output'] += 1
                continue
            if person and row['person_id'] is not None and str(row['person_id']) != person:
                omitted['different_known_person'] += 1
                continue
            run_ids, association = _association(conn, core.customer, row, knowledge)
            if rid and run_ids != [rid] and (not contemporaneous or run_ids):
                omitted['different_or_ambiguous_experiment'] += 1
                continue
            pins, reason = _pins(conn, core.customer, row['evidence_ids'], scope,
                window_end if contemporaneous else cutoff, cache,
                window_start=cutoff if contemporaneous else None)
            scan = None
            if row['payload'].get('operator_scan_event'):
                try:
                    scan_id = row['payload']['operator_scan_event']
                    if type(scan_id) is not str:
                        raise ValueError('Invalid declaration UUID')
                    scan_id = ident(scan_id)
                except (TypeError, ValueError):
                    omitted['malformed_operator_declaration_reference'] += 1
                    continue
                scan = conn.execute('''SELECT * FROM hj_operator_scan_event WHERE customer_id=%s
                    AND id=%s AND processed_at<%s AND occurred_at<%s''',
                    (core.customer, scan_id, knowledge,
                     window_end if contemporaneous else cutoff)).fetchone()
                if (scan is None or str(scan['camera_id']) != camera
                        or (person and scan['person_id'] is not None and str(scan['person_id']) != person)
                        or fingerprint(scan['payload']) != scan['payload_sha256']
                        or fingerprint(row['payload'].get('operator_scan')) != scan['payload_sha256']
                        or scan['occurred_at'] != row['start_at'] or scan['occurred_at'] != row['end_at']
                        or set(scan['evidence_ids']) != set(row['evidence_ids'])):
                    reason = 'operator_declaration_source_scope_or_hash_mismatch'
            if not row['evidence_ids'] and not scan:
                reason = 'no_original_evidence_or_immutable_declaration'
            if (not rid or contemporaneous) and not row['evidence_ids']:
                # A declaration without media has no recording-session provenance.
                reason = 'background_declaration_recording_session_unknown'
            if reason:
                omitted[reason] += 1
                continue
            observations, truncated = _observations(conn, core.customer, row['evidence_ids'], knowledge)
            omitted['observation_budget'] += int(truncated)
            payload = row['payload']
            entry = clean(dict(type='contemporaneous_channel_candidate' if contemporaneous else 'activity_prior',
                activity_id=row['id'], kind=row['kind'],
                source_type=row['source_type'], channel_type='source_qr' if source_qr else row['source_type'],
                text=row['text_content'],
                text_sha256=hashlib.sha256(row['text_content'].encode()).hexdigest(),
                payload_sha256=fingerprint(payload), created_at=row['created_at'], model_revision=row['model_revision'],
                start_at=row['start_at'], end_at=row['end_at'], person_id=row['person_id'],
                instrument_id=row['instrument_id'], sample_id=row['sample_id'], tag_id=row['tag_id'],
                person_attribution='explicit_activity_person' if row['person_id'] else 'unknown_not_inferred',
                context_relation='same_camera_time_correlation' if contemporaneous else 'historical_source_context',
                cross_channel_session=any(p['cross_channel_session'] for p in pins),
                actor_attribution='unknown_not_inferred_from_correlated_camera_or_channel',
                run_ids=run_ids, association_asof=association,
                run_attribution='persisted_association' if run_ids else 'unassigned_not_inferred',
                status='contemporaneous_candidate_not_video_action' if contemporaneous else 'prior_candidate_not_new_observation', evidence=pins,
                interpretation={k: payload[k] for k in _PAYLOAD_KEYS if k in payload},
                observations=observations,
                declaration=({k: scan[k] for k in ('id', 'payload_sha256', 'processed_at', 'occurred_at')}
                             if scan else None)))
            section = 'contemporaneous_evidence' if contemporaneous else 'historical_prior'
            proposed = dict(document, **{section: [*document[section], entry]}, omitted=dict(omitted))
            if (len(document['historical_prior']) + len(document['contemporaneous_evidence']) >= policy['max_items']
                    or _bytes(proposed) + 256 > policy['max_utf8_bytes']):
                omitted['context_item_or_byte_budget'] += 1
                continue
            document[section].append(entry)
        for section in ('historical_prior', 'contemporaneous_evidence'):
            document[section].sort(key=lambda e: (e['start_at'], e['activity_id']))
        selected = {e['activity_id'] for e in document['historical_prior']}
        if rid and selected:
            reports = conn.execute('''SELECT * FROM hj_daily_report WHERE customer_id=%s AND created_at<%s
                AND day>=%s AND day<=%s ORDER BY created_at DESC,revision DESC LIMIT 9''',
                (core.customer, knowledge, lower.date(), cutoff.date())).fetchall()
            for report in reports[:8]:
                if fingerprint(report['document']) != report['fingerprint']:
                    omitted['report_integrity'] += 1
                    continue
                experiment = next((e for e in report['document'].get('experiments', [])
                                   if str(e.get('record', {}).get('id')) == rid), None)
                ids = sorted({str(a['id']) for a in (experiment or {}).get('activities', [])} & selected)
                if not ids:
                    continue
                entry = clean(dict(type='historical_report_summary_reference', run_id=rid,
                    day=str(report['day']), revision=report['revision'], created_at=report['created_at'],
                    document_sha256=report['fingerprint'], earlier_source_activity_ids=ids,
                    status='summary_of_already_listed_sources_not_additional_actions'))
                if (len(document['historical_prior']) + len(document['contemporaneous_evidence']) < policy['max_items']
                        and _bytes(dict(document, historical_prior=[*document['historical_prior'], entry], omitted=dict(omitted))) + 256
                        <= policy['max_utf8_bytes']):
                    document['historical_prior'].append(entry)
                else:
                    omitted['report_context_budget'] += 1
                break  # One newest eligible persisted summary, never a report generated here.
        document['omitted'] = {key: count for key, count in sorted(omitted.items()) if count}
        if _bytes(document) + 80 > policy['max_utf8_bytes']:
            document['historical_prior'] = []
            document['contemporaneous_evidence'] = []
            document['status'] = 'context_capacity_rejected'
    return done()
