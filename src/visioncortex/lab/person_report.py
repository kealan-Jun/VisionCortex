"""Versioned person/day evidence projection, never a physical trajectory model.

Direct operator declarations, explicit evidence associations, run relationships
and historical recorded_by provenance retain separate roles. No speaker, GPS,
continuous work, model coverage or physical action count is inferred.
"""
from __future__ import annotations

import json
from uuid import UUID

from .domain import clean, fingerprint
from .timeline import offset_time

SCHEMA = 'vc-lab-person-timelines/1'
MAX_EVENTS = 2000
MAX_PENDING = 100
MAX_HISTORY = 10000


def _id(value):
    try:
        return str(UUID(str(value))) if value is not None else None
    except ValueError:
        return None


def empty_person(person_id, code=None):
    return dict(person_id=str(person_id),person_code=code,events=[],untimed_events=[],
                event_count=0,physical_action_count=None,continuous_trajectory_known=False)


def person_timelines(conn, core, activities, runs, pending):
    """Read one customer snapshot; event identity is the existing activity UUID."""
    selected = activities[:MAX_EVENTS]
    untimed = pending[:MAX_PENDING]
    unique = {str(a['id']):a for a in selected+untimed}
    all_events = list(unique.values())
    run_by_id = {str(r['id']):r for r in runs}
    missing_runs={r for a in all_events for r in a['run_ids'] if str(r) not in run_by_id}
    if missing_runs:
        extra=conn.execute('SELECT * FROM hj_experiment_record WHERE customer_id=%s AND id=ANY(%s) LIMIT %s',
                           (core.customer,list(missing_runs),MAX_HISTORY+1)).fetchall()
        if len(extra)<=MAX_HISTORY:run_by_id.update({str(r['id']):r for r in extra})
    media_ids = {e['media_id'] for a in all_events for e in a['evidence']}
    media = conn.execute('''SELECT m.*,a.version AS current_version,i.payload AS capture_receipt
      FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
      JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
      JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
        AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
      WHERE m.customer_id=%s AND m.media_id=ANY(%s)''',(core.customer,list(media_ids))).fetchall() if media_ids else []
    media_by_id = {str(m['media_id']):m for m in media}
    camera_keys = {str(m['capture_receipt'].get('camera_id') or m['stream_key']) for m in media}
    cameras = conn.execute("SELECT id,code,metadata FROM entity WHERE customer_id=%s AND kind='capture_device' AND (code=ANY(%s) OR id::text=ANY(%s)) ORDER BY id LIMIT %s",
        (core.customer,list(camera_keys),list(camera_keys),MAX_HISTORY+1)).fetchall() if camera_keys else []
    camera_by_key = {}
    if len(cameras)<=MAX_HISTORY:
        for key in camera_keys:
            matches=[c for c in cameras if key in {str(c['id']),c['code']}]
            if len(matches)==1:camera_by_key[key]=matches[0]
    camera_ids = [c['id'] for c in cameras] if len(cameras)<=MAX_HISTORY else []
    history = conn.execute('SELECT * FROM camera_person_binding WHERE customer_id=%s AND camera_id=ANY(%s) ORDER BY camera_id,valid_from,id LIMIT %s',
        (core.customer,camera_ids,MAX_HISTORY+1)).fetchall() if camera_ids else []
    history_complete = len(history)<=MAX_HISTORY
    scan_ids = [UUID(v) for a in all_events if (v:=_id(a['payload'].get('operator_scan_event'))) is not None]
    scans = conn.execute('SELECT * FROM hj_operator_scan_event WHERE customer_id=%s AND id=ANY(%s)',(core.customer,scan_ids)).fetchall() if scan_ids else []
    scan_by_id = {str(s['id']):s for s in scans}
    persons, unknown, unknown_time = {}, [], []

    def recipient(pid):
        pid=str(pid)
        return persons.setdefault(pid,empty_person(pid))

    def clock_valid(m):
        if not m or m['start_at'] is None:return False
        basis=m['time_basis']
        if (not isinstance(basis,dict) or type(basis.get('basis')) is not str or not basis['basis'] or
                type(basis.get('uncertainty_us')) is not int or basis['uncertainty_us']!=0 or
                basis.get('clock_sync_valid') is False or basis.get('clock_sync_verified') is False):return False
        synthetic=(m['capture_receipt'].get('source_kind')=='fixture' and
                   basis['basis'] in {'synthetic_fixture_clock','fixture_receipt'})
        calibrated=(basis.get('clock_status')=='calibrated' and
                    (basis.get('clock_sync_valid') is True or basis.get('clock_sync_verified') is True))
        return synthetic or calibrated

    for activity in all_events:
        payload=activity['payload'];aid=str(activity['id'])
        known_time=(activity['start_at'] is not None and payload.get('temporal_world_clock_status')!='uncalibrated' and
                    all(clock_valid(media_by_id.get(str(e['media_id']))) for e in activity['evidence']))
        roles={};recording=[]
        def add(pid,basis,**reference):
            if pid is not None:
                roles.setdefault(str(pid),[]).append(dict(basis=basis,**clean(reference)))
        scan=scan_by_id.get(str(payload.get('operator_scan_event')))
        scan_valid=(scan is not None and fingerprint(scan['payload'])==scan['payload_sha256'] and
            fingerprint(payload.get('operator_scan'))==scan['payload_sha256'] and
            scan['person_id']==activity['person_id'] and scan['payload'].get('intent')=='operator_scan')
        if scan_valid:
            add(scan['person_id'],'direct_operator_declaration',scan_event_id=scan['id'],
                scan_kind=scan['scan_kind'],scope_camera_id=scan['camera_id'])
        elif activity['person_id'] is not None:
            add(activity['person_id'],'explicit_activity_person_association',
                authority='stored_evidence_association_not_speaker_or_physical_action_confirmation')
        related_runs=[]
        for rid in activity['run_ids']:
            run=run_by_id.get(str(rid))
            if run:
                related_runs.append(clean(dict(run_id=rid,sample_ids=run['sample_ids'],target_batch_id=run['batch_id'],
                    registered_person_id=run['person_id'],relationship='experiment_context_not_event_operator',
                    clips_url='/internal/timeline/experiments/'+str(rid)+'/clips')))
                add(run['person_id'],'experiment_related_evidence',run_id=rid,authority='run_owner_is_related_only')
        references=[]
        for evidence in activity['evidence']:
            mid=str(evidence['media_id']);m=media_by_id.get(mid)
            ref=clean(dict(evidence_id=evidence['id'],media_id=mid,media_version=evidence['media_version'],
                source_sha256=evidence['verified_sha256'],start_seconds=evidence['start_seconds'],end_seconds=evidence['end_seconds'],
                media_kind=evidence['kind'],media_url='/internal/media/'+mid,
                evidence_url='/internal/evidence/'+str(evidence['id'])))
            references.append(ref)
            attribution=dict(evidence_id=str(evidence['id']),status='unknown_recorded_by',camera_id=None)
            if m and known_time:
                key=str(m['capture_receipt'].get('camera_id') or m['stream_key'])
                camera=camera_by_key.get(key)
                calibrated=clock_valid(m)
                lo,hi=evidence['start_seconds'],evidence['end_seconds']
                if camera:
                    attribution['camera_id']=str(camera['id'])
                if (camera and camera['metadata'].get('view_role')=='first_person' and calibrated and
                        m['start_at'] is not None and lo is not None and hi is not None and
                        evidence['media_version']==m['current_version'] and history_complete):
                    start,end=offset_time(m['start_at'],lo),offset_time(m['start_at'],hi)
                    overlaps=[h for h in history if h['camera_id']==camera['id'] and
                              (h['valid_from']<=start if start==end else h['valid_from']<end) and
                              (h['valid_to'] is None or h['valid_to']>start)]
                    matches=[h for h in overlaps if h['valid_from']<=start and
                             (h['valid_to'] is None or (h['valid_to']>start if start==end else h['valid_to']>=end))]
                    if len(overlaps)==len(matches)==1:
                        h=matches[0]
                        attribution.update(status='recorded_by',person_id=str(h['person_id']),binding_id=str(h['id']),
                            valid_from=clean(h['valid_from']),valid_to=clean(h['valid_to']))
                        add(h['person_id'],'recorded_by',evidence_id=evidence['id'],camera_id=camera['id'],
                            binding_id=h['id'],valid_from=h['valid_from'],valid_to=h['valid_to'],
                            authority='camera_wearer_history_not_performed_by_or_speaker')
                    else:attribution['status']='wearer_history_ambiguous_missing_or_cross_boundary'
                elif camera and camera['metadata'].get('view_role')=='third_person':
                    attribution['status']='fixed_camera_has_no_wearer'
            recording.append(attribution)
        material=scan['payload'].get('resolution',{}) if scan_valid else {}
        materials=(payload.get('operator_context') or {}).get('materials',[]) if scan_valid else []
        source_lots={str(v) for v in [material.get('source_lot_id')]+[m.get('resolution',{}).get('source_lot_id') for m in materials] if v}
        samples={str(activity['sample_id'])} if activity['sample_id'] else set()
        source_units={str(material['target_id'])} if scan_valid and scan['scan_kind']=='material' and material.get('target_id') else set()
        source_units.update(str(m['resolution']['target_id']) for m in materials if m.get('resolution',{}).get('target_id'))
        event=clean(dict(event_id='activity:'+aid,activity_id=aid,start_at=activity['start_at'] if known_time else None,end_at=activity['end_at'] if known_time else None,
            stored_activity_interval=dict(start_at=activity['start_at'],end_at=activity['end_at'],authority='stored_reference_not_new_clock_calibration'),
            time_status='known_stored_evidence_time' if known_time else 'unknown_not_assigned_to_report_day',
            interval_semantics='evidence_extent_not_continuous_physical_work',source_type=activity['source_type'],
            kind=activity['kind'],text=activity['text_content'],text_is_untrusted_evidence_data=True,
            model_revision=activity['model_revision'],evidence=references,recording_attribution=recording,
            run_links=related_runs,explicit_sample_ids=sorted(samples),source_material_unit_ids=sorted(source_units),source_lot_ids=sorted(source_lots),
            target_batch_ids=sorted({r['target_batch_id'] for r in related_runs if r['target_batch_id']} |
                ({scan['payload']['batch_id']} if scan_valid and scan['payload'].get('batch_id') else set())),
            scan_event_id=str(scan['id']) if scan_valid else None,
            operator_person_id=str(scan['person_id']) if scan_valid and scan['person_id'] else None,
            equipment_declaration=(payload.get('operator_context') or {}).get('switch') if scan_valid and scan['scan_kind']=='equipment' else None,
            comment_transition=payload.get('comment_transition'),comment_status=payload.get('status') if payload.get('comment_transition') else None,
            planned_stage_is_arrival=False,speaker_person_id=None,speaker_attribution='not_established_by_this_projection',
            ocr_fields=payload.get('fields',[]) if activity['source_type']=='ocr' else [],
            video_understanding=activity['text_content'] if activity['source_type']=='video' else None,
            retained_clip=payload.get('retained_clip'),original_source_spans=payload.get('original_source_spans',[]),
            task_id=payload.get('task_id'),association_revisions=activity.get('association_revisions',[])))
        if not roles:
            (unknown if known_time else unknown_time).append(event)
        for pid,bases in roles.items():
            item=dict(event,person_attribution=bases)
            recipient(pid)['events' if known_time else 'untimed_events'].append(item)
    ids=[UUID(pid) for pid in persons]
    labels=conn.execute("SELECT id,code FROM entity WHERE customer_id=%s AND kind='person' AND id=ANY(%s)",(core.customer,ids)).fetchall() if ids else []
    codes={str(p['id']):p['code'] for p in labels}
    for pid,person in persons.items():
        person['person_code']=codes.get(pid)
        person['events'].sort(key=lambda e:(e['start_at'],e['event_id']))
        person['untimed_events'].sort(key=lambda e:e['event_id'])
        person['event_count']=len(person['events'])
    result=clean(dict(schema_version=SCHEMA,persons=sorted(persons.values(),key=lambda p:p['person_id']),
        unknown_person_events=unknown,unknown_time_unknown_person_events=unknown_time,
        event_identity='one_activity_id_per_observation; run_links_do_not_duplicate_events; views_not_merged_without_same_event_id',
        ordering='known_start_time_then_activity_id; simultaneous_events_not_physical_sequence_proof',
        coverage=dict(status='partial',continuous_day_trajectory_known=False,physical_action_count=None,
            indoor_or_gps_location_known=False,model_analysis_coverage='unknown',unanalysed_gaps='unknown_not_filled',
            raw_recording_coverage_is_model_analysis_coverage=False,
            events_truncated=len(activities)>MAX_EVENTS,pending_truncated=len(pending)>MAX_PENDING,
            wearer_history_truncated=not history_complete,cameras_truncated=len(cameras)>MAX_HISTORY),
        pending_scope='untimed_events_are_customer_pending_records_not_assigned_to_this_day',
        quantity_ledger_written=False))
    if len(json.dumps(result,ensure_ascii=False).encode())>8*1024**2:
        return dict(schema_version=SCHEMA,persons=[],unknown_person_events=[],unknown_time_unknown_person_events=[],
                    status='needs_attention',reason='person_projection_exceeds_8MiB_bound',coverage={'status':'unknown'})
    return result
