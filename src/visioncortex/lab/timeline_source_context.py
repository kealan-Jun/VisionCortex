"""Same-source auxiliary evidence when wall clocks are unknown.

No cross-camera/session/time matching is attempted. A verified source interval
can still supply its own OCR/embedded speech/QR candidates to video understanding
without claiming that its wall clock or speaker identity has been calibrated.
"""
from __future__ import annotations

from collections import Counter

from .domain import Conflict, clean, fingerprint, ident
from .retained_clips import RetainedClips, original_spans
from .timeline import Timeline, seconds


def source_relative_context(conn, core, request, window, current, task, policy):
    # Reuse the canonical evidence reader; import only when the main builder has
    # already loaded, avoiding a second set of provenance rules.
    from .timeline_prior_context import SCHEMA, _bytes, _media, _source_qr, _observations, _association
    result = dict(schema_version=SCHEMA, enabled=True, status='source_relative_only',
        reason='current_source_clock_unknown', world_clock_status='uncalibrated',
        historical_prior=[], contemporaneous_evidence=[], omitted={},
        task_id=str(task['id']), input_fingerprint=request['input_fingerprint'],
        knowledge_cutoff=task['knowledge_cutoff'], physical_cutoff=None,
        scope=dict(camera_id=str(current['camera_id']) if current['camera_id'] else None,
            person_id=None, mode='same_source_intervals_only',
            person_attribution='not_inferred', panel_attribution='not_inferred_from_current_equipment'),
        source_core=window['core'], cross_package_stage_end_inferred=False,
        semantics='source_relative_candidate_evidence_no_world_time_or_physical_action_confirmation')
    omitted = Counter()
    primary = [r for r in window['input_spans'] if r['media_id'] == str(current['media_id'])]
    if (len(primary) != 1 or current['evidence_status'] != 'available'
            or primary[0]['sha256'] != current['verified_sha256']
            or primary[0]['media_version'] != current['version']
            or current['receipt'].get('sha256') not in {None, current['verified_sha256']}
            or fingerprint(current['receipt']) != current['receipt_fingerprint']):
        raise Conflict('Source-relative prior input identity conflicts with persisted source')
    if current['camera_id'] is None:
        result = dict(schema_version=SCHEMA, enabled=True, status='unavailable',
                      reason='current_source_camera_unregistered', historical_prior=[], contemporaneous_evidence=[])
        result['sha256'] = fingerprint(result)
        return result
    ranges = [dict(media_id=str(current['media_id']), start_seconds=window['core']['start_seconds'],
                   end_seconds=window['core']['end_seconds'])]
    mapping = RetainedClips(Timeline(core), None).get(conn, current['media_id'])
    if mapping is not None:
        if request.get('retained_clip') != mapping:
            raise Conflict('Source-relative retained mapping differs from claimed request')
        ranges.extend(original_spans(mapping, window['core']['start_seconds'], window['core']['end_seconds']))
        result['retained_mapping_sha256'] = fingerprint(mapping)
        result['mapping_verification'] = mapping['derivation_verification']
    allowed = {}
    for part in ranges:
        allowed.setdefault(str(part['media_id']), []).append((seconds(part['start_seconds']), seconds(part['end_seconds'])))
    rows = conn.execute('''SELECT DISTINCT a.* FROM hj_activity a
        JOIN evidence_ref e ON e.customer_id=a.customer_id AND e.id=ANY(a.evidence_ids)
        WHERE a.customer_id=%s AND a.created_at<%s AND e.media_id=ANY(%s)
        ORDER BY a.created_at DESC,a.id LIMIT 129''',
        (core.customer, task['knowledge_cutoff'], [ident(mid) for mid in allowed])).fetchall()
    omitted['query_budget'] += int(len(rows) > 128)
    cache = {str(current['media_id']): current}
    person = (request.get('context_scope') or {}).get('person_id')
    run_id = (request.get('context_scope') or {}).get('run_id')
    requested_runs = [str(run_id)] if run_id else list(map(str, (request.get('retained_clip') or {}).get('run_ids') or []))
    for row in rows[:128]:
        if row['payload'].get('task_id') == str(task['id']):
            omitted['current_task_output'] += 1
            continue
        if person and row['person_id'] is not None and str(row['person_id']) != str(person):
            omitted['different_explicit_person'] += 1
            continue
        associated, association = _association(conn, core.customer, row, task['knowledge_cutoff'])
        if requested_runs and associated and not set(associated) <= set(requested_runs):
            omitted['different_explicit_experiment'] += 1
            continue
        source_qr = row['source_type'] == 'video' and _source_qr(conn, core.customer, row, task['knowledge_cutoff'])
        if not (row['source_type'] in {'speech', 'ocr'} or source_qr
                or row['source_type'] == 'reviewed_note' and row['payload'].get('scan_intake')):
            omitted['non_auxiliary_channel'] += 1
            continue
        pins, valid = [], True
        for eid in row['evidence_ids']:
            e = conn.execute('SELECT * FROM evidence_ref WHERE customer_id=%s AND id=%s',
                             (core.customer, eid)).fetchone()
            mid = str(e['media_id']) if e else None
            if e is None or mid not in allowed or e['start_seconds'] is None or e['end_seconds'] is None:
                valid = False
                break
            lo, hi = e['start_seconds'], e['end_seconds']
            if not any(a <= lo <= hi <= b and lo < b for a, b in allowed[mid]):
                valid = False
                break
            if mid not in cache:
                cache[mid] = _media(conn, core.customer, mid)
            source = cache[mid]
            if (source is None or source['evidence_status'] != 'available'
                    or e['media_version'] != source['version']
                    or source['receipt'].get('sha256') not in {None, source['verified_sha256']}
                    or str(source['camera_id']) != str(current['camera_id'])
                    or fingerprint(source['receipt']) != source['receipt_fingerprint']):
                valid = False
                break
            pins.append(clean(dict(evidence_id=eid, media_id=mid, media_version=source['version'],
                source_sha256=source['verified_sha256'], start_seconds=lo, end_seconds=hi,
                receipt_sha256=source['receipt_fingerprint'], original_time_basis=source['time_basis'],
                time_semantics='source_relative_only_no_wall_clock_claim',
                declared_sha256=source['receipt'].get('sha256'),
                digest_basis='ingress_verified_original_bytes')))
        if not valid or not pins:
            omitted['not_wholly_within_same_source_intervals'] += 1
            continue
        observations, truncated = _observations(conn, core.customer, row['evidence_ids'], task['knowledge_cutoff'])
        omitted['observation_budget'] += int(truncated)
        entry = clean(dict(type='source_relative_channel_candidate', activity_id=row['id'],
            source_type=row['source_type'], channel_type='source_qr' if source_qr else row['source_type'],
            text=row['text_content'], model_revision=row['model_revision'],
            payload_sha256=fingerprint(row['payload']), evidence=pins, observations=observations,
            instrument_id=row['instrument_id'], sample_id=row['sample_id'],
            stored_run_ids=associated, association_revision=association,
            actor_identity='not_inferred_from_source_clock_or_wearer',
            physical_action_confirmed=False, original_candidate_payload=row['payload']))
        proposed = dict(result, contemporaneous_evidence=[*result['contemporaneous_evidence'], entry], omitted=dict(omitted))
        if (len(result['contemporaneous_evidence']) >= policy['max_items']
                or _bytes(proposed)+256 > policy['max_utf8_bytes']):
            omitted['context_budget'] += 1
            continue
        result['contemporaneous_evidence'].append(entry)
    result['omitted'] = {k: n for k, n in omitted.items() if n}
    if _bytes(result)+80 > policy['max_utf8_bytes']:
        result = dict(schema_version=SCHEMA, enabled=True, status='context_capacity_rejected',
            reason='source_relative_metadata_exceeds_budget', historical_prior=[], contemporaneous_evidence=[])
    result = clean(result)
    result['sha256'] = fingerprint(result)
    return result
