"""Derived operation envelopes; packet boundaries never close a physical phase.

These projections preserve immutable QR and reviewed speech anchors. Source spans
are replay selections, not materialized clips or proof of continuous activity.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from decimal import Decimal

from .domain import clean, fingerprint, instant
from .timeline_policy import ZONE

SCHEMA = 'vc-lab-operation-segments/1'


def mark_operation_reports_dirty(conn, customer, at):
    """Late anchors revise existing carry-over snapshots, never overwrite them."""
    if at is None:
        return
    days = conn.execute('''SELECT DISTINCT day FROM hj_daily_report WHERE customer_id=%s
        AND day>=%s AND document#>'{operation_segments,enabled}'='true'::jsonb
        ORDER BY day LIMIT 4097''', (customer, at.astimezone(ZONE).date())).fetchall()
    if len(days) > 4096:
        raise ValueError('Bounded operation report history exceeded')
    for row in days:
        conn.execute('INSERT INTO hj_timeline_dirty_day(customer_id,day) VALUES(%s,%s) ON CONFLICT DO NOTHING',
                     (customer, row['day']))


def project_anchors(scans, comments, start, end, pairing_seconds):
    """Pure chronological projection, separate scope and simultaneous-code ambiguity."""
    groups = defaultdict(list)
    for scan in scans:
        if scan.get('occurred_at') and scan.get('scan_kind') == 'equipment':
            groups[(str(scan.get('person_id')), str(scan.get('camera_id')))].append(scan)
    phases, ignored = [], []
    for rows in groups.values():
        by_time = defaultdict(list)
        for row in rows:
            by_time[instant(row['occurred_at'])].append(row)
        current = None
        run_id, batch_id, target_anchors = None, None, []
        for at, simultaneous in sorted(by_time.items()):
            declared_runs = {s['payload'].get('run_id') for s in simultaneous if s['payload'].get('run_explicit')}
            declared_batches = {s['payload'].get('batch_id') for s in simultaneous
                                if s['payload'].get('batch_explicit') or s['payload'].get('run_explicit')}
            if len(declared_runs) > 1 or len(declared_batches) > 1:
                run_id, batch_id, target_anchors = None, None, []
            elif declared_runs or declared_batches:
                run_id = next(iter(declared_runs)) if declared_runs else None
                batch_id = next(iter(declared_batches)) if declared_batches else batch_id
                target_anchors = [str(s['id']) for s in simultaneous
                                  if s['payload'].get('run_explicit') or s['payload'].get('batch_explicit')]
            targets = {s['payload'].get('resolution', {}).get('target_id') for s in simultaneous}
            good = all(s['payload'].get('status') == 'declared_equipment' for s in simultaneous)
            if not good or len(targets) != 1 or None in targets:
                ignored.append(dict(event_ids=[str(s['id']) for s in simultaneous],
                                    reason='ambiguous_or_unverified_equipment', occurred_at=at))
                # An ambiguous observation cannot continue the old attribution.
                if current:
                    current.update(end_at=at, status='ambiguous_equipment_boundary')
                    current = None
                continue
            target = next(iter(targets))
            anchors = [dict(event_id=str(s['id']), occurred_at=at,
                            payload_sha256=s['payload_sha256'], evidence_ids=s['evidence_ids'])
                       for s in simultaneous]
            if (current and current['equipment_id'] == target and current['run_id'] == run_id
                    and current['target_batch_id'] == batch_id):
                current['qr_anchors'].extend(anchors)
                continue
            if current:
                current.update(end_at=at, status='switched_without_paired_end_comment')
            first = simultaneous[0]
            current = dict(segment_id='operation:' + str(first['id']),
                person_id=first.get('person_id'), camera_id=first.get('camera_id'),
                equipment_id=target, run_id=run_id, target_batch_id=batch_id,
                target_declaration_event_ids=list(target_anchors), start_at=at, end_at=None,
                status='open_missing_start_comment', qr_anchors=anchors,
                comment_anchors=[], experiment_end_implied=False,
                value_class='experimental_candidate', value_basis='configured_equipment_scan',
                material_usage_confirmed=False, quantity_ledger_written=False)
            phases.append(current)
    allowance = timedelta(seconds=pairing_seconds)
    for comment in sorted(comments, key=lambda c: (c['start_at'], str(c['id']))):
        payload = comment['payload']
        kind = payload.get('comment_transition')
        if (payload.get('status') != 'applied_reviewed_interpretation'
                or kind not in {'stage_start', 'stage_complete', 'experiment_complete'}):
            continue
        at = instant(comment['start_at'])
        if kind == 'stage_start':
            # A repeated code after an explicit end is not itself a new start.
            # A later reviewed start plus a new presence anchor can reopen the
            # same instrument as a distinct phase, without a toggle heuristic.
            for old in list(phases):
                if (old['status'] != 'paired_end' or at <= instant(old['end_at'])
                        or str(old['person_id']) != str(payload.get('person_id'))
                        or str(old['camera_id']) != str(payload.get('camera_id'))
                        or old['run_id'] and old['run_id'] != payload.get('run_id')):
                    continue
                future = [q for q in old['qr_anchors'] if instant(q['occurred_at']) > instant(old['end_at'])
                          and abs(instant(q['occurred_at'])-at) <= allowance]
                if future:
                    first = min(future, key=lambda q: instant(q['occurred_at']))
                    next_qr = [q for q in old['qr_anchors'] if instant(q['occurred_at']) >= instant(first['occurred_at'])]
                    phases.append(dict(old, segment_id='operation:' + first['event_id'],
                        start_at=first['occurred_at'], end_at=None, status='open_missing_start_comment',
                        qr_anchors=next_qr, comment_anchors=[], whole_experiment_end_declared=False))
                    old['qr_anchors'] = [q for q in old['qr_anchors'] if q not in next_qr]
        matching = []
        for phase in phases:
            if (str(phase['person_id']) != str(payload.get('person_id'))
                    or str(phase['camera_id']) != str(payload.get('camera_id'))):
                continue
            if phase['run_id'] and phase['run_id'] != payload.get('run_id'):
                continue
            qr = [a for a in phase['qr_anchors'] if abs(instant(a['occurred_at']) - at) <= allowance]
            if not qr:
                continue
            if phase['end_at'] and at > phase['end_at']:
                continue
            if kind != 'stage_start' and at < instant(phase['start_at']):
                continue
            if kind == 'stage_start' and abs(instant(phase['start_at']) - at) > allowance:
                continue
            matching.append((phase, qr))
        if len(matching) != 1:
            ignored.append(dict(activity_id=str(comment['id']), reason='unpaired_or_ambiguous_comment'))
            continue
        phase, qr = matching[0]
        phase['comment_anchors'].append(dict(activity_id=str(comment['id']),
            source_activity_id=payload.get('source_activity_id'), interpretation=kind,
            occurred_at=at, payload_sha256=fingerprint(payload),
            evidence_ids=comment['evidence_ids'], qr_event_ids=[q['event_id'] for q in qr]))
        if kind == 'stage_start':
            phase['start_at'] = min(instant(phase['start_at']), at)
            if phase['end_at'] is None:
                phase['status'] = 'open_paired_start'
        else:
            phase['end_at'] = max(at, min(instant(q['occurred_at']) for q in qr))
            phase['status'] = 'paired_end'
            phase['whole_experiment_end_declared'] = kind == 'experiment_complete'
    selected = []
    for phase in sorted(phases, key=lambda p: (instant(p['start_at']), p['segment_id'])):
        if instant(phase['start_at']) >= end or (phase['end_at'] and instant(phase['end_at']) <= start):
            continue
        phase['continues_from_previous_day'] = instant(phase['start_at']) < start
        phase['continues_after_day'] = phase['end_at'] is None or instant(phase['end_at']) > end
        phase['boundary_semantics'] = 'anchor_envelope_not_exact_physical_action_boundary'
        phase['continuous_activity_proven'] = False
        phase['missing_anchors'] = [k for k, ok in (
            ('start_comment', any(a['interpretation'] == 'stage_start' for a in phase['comment_anchors'])),
            ('end_comment_and_qr', phase['status'] == 'paired_end')) if not ok]
        selected.append(phase)
    return clean(dict(schema_version=SCHEMA, enabled=True, segments=selected,
        unpaired_anchors=ignored, packet_boundary_closes_segment=False,
        anchor_pairing_seconds=pairing_seconds, pairing_window_is_not_inactivity_timeout=True,
        unbound_value_classification='independent_existing_value_pipeline',
        static_or_speech_alone_is_not_value_truth=True))


def operation_segments(conn, customer, start, end, policy):
    config = policy.document.get('operation_segments', {})
    if not config.get('enabled'):
        return dict(schema_version=SCHEMA, enabled=False)
    scans = conn.execute('''SELECT * FROM hj_operator_scan_event WHERE customer_id=%s
        AND occurred_at<%s AND scan_kind='equipment' ORDER BY occurred_at,id LIMIT 4097''',
        (customer, end)).fetchall()
    comments = conn.execute('''SELECT * FROM hj_activity WHERE customer_id=%s
        AND start_at<%s AND payload ? 'comment_transition' ORDER BY start_at,id LIMIT 4097''',
        (customer, end)).fetchall()
    if len(scans) > 4096 or len(comments) > 4096:
        return dict(schema_version=SCHEMA, enabled=True, status='needs_attention',
                    reason='bounded_history_exceeded', segments=[])
    for row in scans:
        if fingerprint(row['payload']) != row['payload_sha256']:
            raise ValueError('Immutable operation anchor hash mismatch')
    result = project_anchors(scans, comments, start, end, config['anchor_pairing_seconds'])
    media = conn.execute('''SELECT m.*,a.version,a.verified_sha256,a.evidence_status,
        i.payload AS receipt,cam.id AS registered_camera_id FROM hj_timeline_media m
        JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
        JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
        JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
          AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
        LEFT JOIN entity cam ON cam.customer_id=m.customer_id AND cam.kind='capture_device'
          AND (cam.code=i.payload->>'camera_id' OR cam.id::text=i.payload->>'camera_id')
        WHERE m.customer_id=%s AND m.kind='video' AND m.start_at<%s AND m.end_at>%s
          AND NOT EXISTS(SELECT 1 FROM media_derivation d WHERE d.customer_id=m.customer_id
                         AND d.child_id=m.media_id)
        ORDER BY m.start_at,m.media_id LIMIT 513''', (customer, end, start)).fetchall()
    if len(media) > 512:
        result.update(status='needs_attention', reason='bounded_media_exceeded')
        media = []
    for phase in result['segments']:
        lo, hi = max(start, instant(phase['start_at'])), min(end, instant(phase['end_at']) if phase['end_at'] else end)
        spans = []
        for item in media:
            receipt, clock = item['receipt'], item['time_basis']
            if (str(item['registered_camera_id']) != str(phase['camera_id'])
                    or item['evidence_status'] != 'available'
                    or receipt.get('sha256') != item['verified_sha256']
                    or not isinstance(clock, dict) or clock.get('uncertainty_us') != 0
                    or not (clock.get('clock_sync_verified') is True or clock.get('clock_sync_valid') is True
                            or receipt.get('source_kind') == 'fixture')):
                continue
            left, right = max(lo, item['start_at']), min(hi, item['end_at'])
            if left < right:
                spans.append(clean(dict(media_id=item['media_id'], media_version=item['version'],
                    source_sha256=item['verified_sha256'], start_at=left, end_at=right,
                    start_seconds=Decimal(str((left-item['start_at']).total_seconds())),
                    end_seconds=Decimal(str((right-item['start_at']).total_seconds())),
                    camera_id=phase['camera_id'], stream_key=item['stream_key'],
                    media_url='/internal/media/' + str(item['media_id']),
                    materialized_clip=False)))
        phase['source_spans'] = spans
        phase['recording_coverage_complete'] = False
        phase['coverage_status'] = 'indexed_ranges_only_gaps_not_filled'
    result['projection_sha256'] = fingerprint(result)
    return result
