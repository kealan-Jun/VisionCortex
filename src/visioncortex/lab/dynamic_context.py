"""Persistent dynamic visual identity/focus, separate from experiments/materials.

Only Core.command writes these observations/projections. Focus can be a software
candidate; local pixel motion never confirms a physical operation or operator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
import json
import re
from uuid import uuid4

from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, fingerprint, ident, instant
from .dynamic_vision import contained, interaction_candidate, matrix, polygon, project


SCHEMA = 'vc-lab-dynamic-context/1'


@dataclass(frozen=True)
class DynamicPolicy:
    enter_samples: int = 3
    enter_span_seconds: str = '.1'
    max_gap_seconds: str = '.25'
    qr_hold_seconds: str = '.75'
    focus_idle_seconds: str = '1'
    max_events: int = 2048
    max_tracks: int = 8
    max_track_ids: int = 64
    max_snapshot_bytes: int = 8 * 1024**2
    candidate_focus_enabled: bool = True

    def __post_init__(self):
        if type(self.enter_samples) is not int or not 2 <= self.enter_samples <= 30:
            raise ValueError('Bounded independent entry support required')
        for name in ['enter_span_seconds','max_gap_seconds','qr_hold_seconds','focus_idle_seconds']:
            value = Fraction(getattr(self, name))
            if not 0 < value <= 5:
                raise ValueError('Explicit bounded source-time hysteresis required')
        if any(type(v) is not int for v in [self.max_events,self.max_tracks,self.max_snapshot_bytes]) or not 1 <= self.max_events <= 4096 or not 1 <= self.max_tracks <= 16 or not 1024 <= self.max_snapshot_bytes <= 8*1024**2:
            raise ValueError('Finite research event/track/snapshot bounds required')
        if type(self.max_track_ids) is not int or not self.max_tracks <= self.max_track_ids <= 64:
            raise ValueError('Bounded total track IDs per source required')


def seconds(value):
    if type(value) not in {str, int, Decimal, Fraction}:
        raise ValueError('Exact source time required')
    result = Fraction(value)
    if not 0 <= result <= 1000000:
        raise ValueError('Source time outside bounded media')
    return result


def _text(value):
    if type(value) is not str or not 1 <= len(value) <= 120:
        raise ValueError('Explicit bounded identity text required')
    return value


def _source_time(event):
    if type(event.get('source_pts')) is not int or abs(event['source_pts']) > 2**63 or type(event.get('time_base')) is not str or type(event.get('stream_origin')) is not str:
        raise ValueError('Original integer PTS and exact time-base string required')
    base = Fraction(event['time_base'])
    if base <= 0:
        raise ValueError('Positive original time base required')
    return seconds(Fraction(event['source_pts']) * base - Fraction(event['stream_origin']))


def _clock_known(media):
    basis = media['time_basis']
    return (bool(basis.get('basis')) and type(basis.get('uncertainty_us')) is int and
            basis['uncertainty_us'] == 0 and basis.get('clock_sync_valid') is not False and
            media['start_at'] is not None)


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Explicit source/pixel SHA256 required')
    return value


def _geometry(value):
    if not isinstance(value, dict) or value.get('coordinates') != 'canonical_normalized_v1':
        raise ValueError('Versioned canonical normalized device geometry required')
    result = dict(coordinates='canonical_normalized_v1')
    for name in ['device_polygon','panel_polygon','qr_region']:
        result[name] = polygon(value.get(name))
        if any(not 0 <= v <= 1 for p in result[name] for v in p):
            raise ValueError('Canonical geometry outside unit frame')
    if not contained(result['panel_polygon'], result['device_polygon']) or not contained(result['qr_region'], result['device_polygon']):
        raise ValueError('Panel/attached QR must lie inside the registered device boundary')
    controls=value.get('control_polygons',[])
    if not isinstance(controls,list) or len(controls)>8:
        raise ValueError('Bounded registered control polygons required')
    result['control_polygons'] = [polygon(p) for p in controls]
    for points in result['control_polygons']:
        if not contained(points, result['device_polygon']):
            raise ValueError('Controls must belong to the same registered device')
        # Actual screen digits/QR are not interaction-control pixels.
        from .panel_vision import libraries
        cv2, np = libraries()
        for excluded in [result['panel_polygon'], result['qr_region']]:
            area, _ = cv2.intersectConvexConvex(np.asarray(points,np.float32), np.asarray(excluded,np.float32))
            if area > 1e-8:
                raise ValueError('Interaction controls must exclude displayed digits and QR')
    return result


class DynamicContext:
    def __init__(self, core, policy=None):
        self.core = core
        self.policy = policy or DynamicPolicy()

    def video_inputs(self, media_id, source_session_key, link_pins):
        """Read actual customer source/evidence and exact spatial-link revisions."""
        if not isinstance(link_pins,list) or len(link_pins)>self.policy.max_tracks:
            raise ValueError('Bounded explicit panel link pins required')
        with self.core.db.transaction(readonly=True) as conn:
            media=self._media(conn,media_id,source_session_key)
            links=[]
            for pin in link_pins:
                link=conn.execute('SELECT * FROM hj_dynamic_panel_link WHERE customer_id=%s AND id=%s',(self.core.customer,ident(pin['link_id']))).fetchone()
                if not link: raise Missing('Panel spatial link outside customer scope')
                if type(pin.get('link_revision')) is not int or link['revision']!=pin['link_revision'] or link['layout_revision']!=pin.get('layout_revision'):
                    raise Conflict('Pinned panel link/layout revision changed')
                links.append(link)
            refs=conn.execute('SELECT id,media_id,media_version,start_seconds,end_seconds FROM evidence_ref WHERE customer_id=%s AND media_id=%s AND media_version=%s ORDER BY id',(self.core.customer,ident(media_id),media['version'])).fetchall()
            return clean(dict(media_id=media_id,source_session_key=source_session_key,source_stream_key=media['stream_key'],
                              source_sha256=media['verified_sha256'],media_version=media['version'],duration_seconds=media['duration_seconds'],
                              evidence_refs=refs,panel_links=links))

    def register_panel_link(self, key, instrument_id, panel_id, layout_revision, geometry, *, actor, reason, evidence_ids):
        geometry = _geometry(geometry)
        _text(panel_id); _text(layout_revision)
        if not actor or not reason or not evidence_ids:
            raise ValueError('Explicit spatial verification actor/reason/original evidence required')
        request = dict(action='register_dynamic_panel_link', instrument_id=instrument_id, panel_id=panel_id,
                       layout_revision=layout_revision, geometry=geometry, actor=actor, reason=reason, evidence_ids=evidence_ids)
        def action(conn):
            self.core._entity(conn, instrument_id, 'instrument')
            refs = self.core._evidence(conn, evidence_ids)
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', ('dynamic-link:'+str(self.core.customer)+':'+str(instrument_id)+':'+panel_id,))
            revision = conn.execute('SELECT COALESCE(MAX(revision),0)+1 AS n FROM hj_dynamic_panel_link WHERE customer_id=%s AND instrument_id=%s AND panel_id=%s', (self.core.customer,ident(instrument_id),panel_id)).fetchone()['n']
            row = conn.execute('''INSERT INTO hj_dynamic_panel_link(customer_id,id,instrument_id,panel_id,layout_revision,revision,geometry,evidence_ids,actor,reason)
                 VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *''',
                 (self.core.customer,uuid4(),ident(instrument_id),panel_id,layout_revision,revision,Jsonb(geometry),refs,actor,reason)).fetchone()
            self.core._emit(conn,'dynamic_panel_link',row['id'],revision,row)
            return row
        return self.core.command(key,request,action)

    def observe(self, key, media_id, source_session_key, frame_event):
        return self.append_batch(key,media_id,source_session_key,[frame_event])

    def append_batch(self, key, media_id, source_session_key, frame_events):
        _text(source_session_key)
        if not isinstance(frame_events,list) or not 1 <= len(frame_events) <= 128:
            raise ValueError('Append one to128 explicit source frames per transaction')
        request = dict(action='dynamic_frame_batch',media_id=media_id,source_session_key=source_session_key,
                       frame_events=frame_events,policy=asdict(self.policy))
        if len(json.dumps(clean(request),allow_nan=False)) > 1024**2:
            raise ValueError('Bounded frame batch exceeded')
        def action(conn):
            media = self._media(conn,media_id,source_session_key)
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', (f'dynamic:{self.core.customer}:{media_id}:{source_session_key}',))
            count = conn.execute('SELECT count(*) AS n FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s', (self.core.customer,ident(media_id),source_session_key)).fetchone()['n']
            new_keys = {event.get('frame_key') for event in frame_events}
            old = conn.execute('SELECT frame_key FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s AND frame_key=ANY(%s)',(self.core.customer,ident(media_id),source_session_key,list(new_keys))).fetchall()
            if count+len(new_keys)-len(old) > self.policy.max_events:
                return dict(status='needs_attention',reason='bounded_research_event_limit',accepted=0,raw_originals_unchanged=True)
            accepted = 0
            for event in frame_events:
                self._validate_event(conn,event,media)
                digest = fingerprint(event)
                existing = conn.execute('SELECT payload_sha256 FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s AND frame_key=%s', (self.core.customer,ident(media_id),source_session_key,event['frame_key'])).fetchone()
                if existing:
                    if existing['payload_sha256'] != digest:
                        raise Conflict('Frame idempotency key changed immutable content')
                    continue
                supersedes = event.get('supersedes_frame_key')
                if supersedes:
                    if not event.get('revision_actor') or not event.get('revision_reason'):
                        raise ValueError('Correction requires actor/reason')
                    parent = conn.execute('SELECT id FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s AND frame_key=%s',(self.core.customer,ident(media_id),source_session_key,supersedes)).fetchone()
                    if not parent: raise Missing('Correction parent outside this source/session')
                    child = conn.execute('SELECT id FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s AND supersedes_frame_key=%s LIMIT 1',(self.core.customer,ident(media_id),source_session_key,supersedes)).fetchone()
                    if child: raise Conflict('Correction parent already superseded; revise its current leaf instead')
                at = _source_time(event)
                label = media['start_at'] + timedelta(microseconds=round(at * 1000000)) if _clock_known(media) else None
                conn.execute('''INSERT INTO hj_dynamic_frame_event(customer_id,id,media_id,source_session_key,frame_key,source_seconds,payload,payload_sha256,supersedes_frame_key,event_at,observed_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (self.core.customer,uuid4(),ident(media_id),source_session_key,event['frame_key'],Decimal(at.numerator)/Decimal(at.denominator),Jsonb(clean(event)),digest,supersedes,label,instant(event['observed_at']) if event.get('observed_at') else None))
                accepted += 1
            projection = self._replay(conn,media,source_session_key)
            return dict(status=projection['snapshot']['status'],accepted=accepted,revision=projection['revision'],
                        revision_id=projection['id'],snapshot=projection['snapshot'])
        return self.core.command(key,request,action)

    def _media(self,conn,media_id,session):
        row=conn.execute('''SELECT m.*,a.version,a.verified_sha256,a.evidence_status,c.source_kind,i.payload AS receipt
            FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
            JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
            JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
            WHERE m.customer_id=%s AND m.media_id=%s''',(self.core.customer,ident(media_id))).fetchone()
        if not row or row['kind'] not in {'video','photo'}: raise Missing('Dynamic source outside customer video/photo scope')
        if (row['session_key'] or '') != session: raise Conflict('Explicit source session differs from original media')
        if row['evidence_status'] != 'available': raise Conflict('Dynamic source evidence unavailable')
        return row

    def _validate_event(self,conn,event,media):
        if not isinstance(event,dict) or event.get('schema_version') != 'vc-lab-dynamic-frame/1':
            raise ValueError('Versioned source frame observation required')
        _text(event.get('frame_key')); _sha(event.get('frame_rgb_sha256'))
        at=_source_time(event)
        if at>Fraction(media['duration_seconds']): raise ValueError('Source frame escaped original media')
        if event.get('source_sha256') != media['verified_sha256'] or event.get('media_version') != media['version']:
            raise Conflict('Dynamic input source SHA/version changed')
        if not isinstance(event.get('evidence_ids'),list) or not event['evidence_ids']:
            raise ValueError('Every frame observation requires original evidence references')
        self.core._evidence(conn,event['evidence_ids'])
        refs=conn.execute('SELECT * FROM evidence_ref WHERE customer_id=%s AND id=ANY(%s)',(self.core.customer,[ident(v) for v in event['evidence_ids']])).fetchall()
        if not any(r['media_id']==media['media_id'] and r['media_version']==media['version'] and Fraction(r['start_seconds'])<=at<=Fraction(r['end_seconds']) for r in refs):
            raise Conflict('Frame PTS is not covered by its pinned original evidence')
        tracks=event.get('tracks',[]); codes=event.get('qr_occurrences',[])
        if not isinstance(tracks,list) or len(tracks)>self.policy.max_tracks or not isinstance(codes,list) or len(codes)>64:
            raise ValueError('Bounded track and QR occurrence lists required')
        if len({t.get('track_id') for t in tracks})!=len(tracks): raise ValueError('Repeated track ID in one frame')
        for code in codes:
            if type(code.get('value')) is not str or not 1 <= len(code['value'].encode('utf-8')) <= 4096:
                raise ValueError('Bounded QR payload text required')
            polygon(code.get('polygon'))
        for track in tracks:
            _text(track.get('track_id')); matrix(track['canonical_to_frame'])
            if track.get('status') not in {'reliable','occluded','lost','reacquired','conflict'}: raise ValueError('Explicit tracking status required')
            link=conn.execute('SELECT * FROM hj_dynamic_panel_link WHERE customer_id=%s AND id=%s',(self.core.customer,ident(track['link_id']))).fetchone()
            if not link: raise Missing('Panel spatial link outside customer scope')
            if track.get('layout_revision')!=link['layout_revision']: raise Conflict('Track registration layout revision differs')
            # Supplied metrics are candidates, not a `verified=True` operation fact.
            reviewed=track.get('reviewed_operation')
            if reviewed:
                if not reviewed.get('actor') or not reviewed.get('reason') or not reviewed.get('evidence_ids'):
                    raise ValueError('Reviewed reference operation requires actor/reason/evidence')
                self.core._evidence(conn,reviewed['evidence_ids'])
                if not set(reviewed['evidence_ids']).issubset(set(event['evidence_ids'])):
                    raise Conflict('Reviewed operation must cite this source frame evidence')
        if event.get('operator_id') is not None:
            self.core._entity(conn,event['operator_id'],'person')
            if event.get('operator_basis') != 'reviewed_visual_operator' or not event.get('operator_actor') or not event.get('operator_reason'):
                raise ValueError('Actual operator requires explicit reviewed visual attribution, not wearer inference')

    def _resolve(self,conn,value,event_at,clock_known):
        tag=conn.execute("SELECT id FROM entity WHERE customer_id=%s AND kind='tag' AND code=%s",(self.core.customer,value)).fetchone()
        if not tag: return dict(status='unknown_code',instrument_id=None)
        if not clock_known or event_at is None: return dict(status='unknown_binding_clock',instrument_id=None)
        rows=conn.execute('''SELECT b.*,e.kind FROM tag_assignment b JOIN entity e ON e.customer_id=b.customer_id AND e.id=b.target_id
             WHERE b.customer_id=%s AND b.tag_id=%s AND b.valid_from<=%s AND (b.valid_to IS NULL OR b.valid_to>%s)''',(self.core.customer,tag['id'],event_at,event_at)).fetchall()
        if len(rows)!=1 or rows[0]['kind']!='instrument': return dict(status='unknown_or_ambiguous_instrument',instrument_id=None)
        equipment=conn.execute('SELECT * FROM equipment_identity WHERE customer_id=%s AND device_id=%s',(self.core.customer,rows[0]['target_id'])).fetchone()
        if equipment and (equipment['verified_at'] is None or equipment['verified_at']>event_at or (equipment['retired_at'] and equipment['retired_at']<=event_at)):
            return dict(status='equipment_not_active_at_event',instrument_id=None)
        return dict(status='registered_identity',instrument_id=str(rows[0]['target_id']),binding_id=str(rows[0]['id']),binding_version=rows[0]['version'])

    def _replay(self,conn,media,session):
        rows=conn.execute('SELECT * FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s ORDER BY source_seconds,frame_key',(self.core.customer,media['media_id'],session)).fetchall()
        superseded={r['supersedes_frame_key'] for r in rows if r['supersedes_frame_key']}
        active=[r for r in rows if r['frame_key'] not in superseded]
        active.sort(key=lambda r:(_source_time(r['payload']),r['frame_key']))
        policy=self.policy; gap=Fraction(policy.max_gap_seconds); hold=Fraction(policy.qr_hold_seconds)
        all_track_ids={t['track_id'] for row in active for t in row['payload'].get('tracks',[])}
        too_many_tracks=len(all_track_ids)>policy.max_track_ids
        if too_many_tracks: active=[]
        link_ids={ident(t['link_id']) for row in active for t in row['payload'].get('tracks',[])}
        link_cache={str(link['id']):link for link in conn.execute('SELECT * FROM hj_dynamic_panel_link WHERE customer_id=%s AND id=ANY(%s)',(self.core.customer,list(link_ids))).fetchall()} if link_ids else {}
        pts_observations={};frame_maps={}
        for row in active:
            event=row['payload'];codes=event.get('qr_occurrences',[])
            links={t['track_id']:link_cache[t['link_id']] for t in event.get('tracks',[])}
            mapped={t['track_id']:[c for c in codes if t['status'] in {'reliable','reacquired'} and contained(c['polygon'],project(links[t['track_id']]['geometry']['qr_region'],t['canonical_to_frame'])) and contained(c['polygon'],project(links[t['track_id']]['geometry']['device_polygon'],t['canonical_to_frame']))] for t in event.get('tracks',[])}
            ambiguity={id(c):sum(any(c is o for o in members) for members in mapped.values())>1 for c in codes}
            frame_maps[row['frame_key']]=(links,mapped,ambiguity)
            for track in event.get('tracks',[]):
                local=mapped[track['track_id']]
                semantic=dict(track=track,local_qr=local,frame_rgb_sha256=event['frame_rgb_sha256'],
                              source_sha256=event['source_sha256'],media_version=event['media_version'],
                              source_pts=event['source_pts'],time_base=event['time_base'],stream_origin=event['stream_origin'],
                              duplicate_counts={c['value']:sum(o['value']==c['value'] for o in codes) for c in local},
                              cross_region_ambiguity=[ambiguity[id(c)] for c in local])
                pts_observations.setdefault(_source_time(event),{}).setdefault(track['track_id'],set()).add(fingerprint(semantic))
        states={}; snapshots=[]; notifications=[]; focus=None; focus_intervals=[]; focus_start=None
        known=_clock_known(media)
        dependencies=[]; last_owner=None; last_wearer=None
        def close_focus(at,reason):
            nonlocal focus,focus_start
            if focus is not None:
                focus_intervals.append(dict(start_seconds=str(focus_start),end_seconds=str(at),track_id=focus['track_id'],instrument_id=focus['instrument_id'],focus_status=focus['focus_status'],operation_status=focus['operation_status'],reason_ended=reason))
            focus=None;focus_start=None
        def reset(state,at,reason):
            state['epoch']+=1;state['instrument_id']=None;state['candidate_count']=0;state['candidate_start']=None;state['last_qr']=None;state['break_pending']=True
            state['pixel_hashes']=set();state['previous_rgb_sha256']=None;state['last_interaction']=None;state['operation_status']='unknown'
            notifications.append(dict(track_id=state['track_id'],panel_id=state['panel_id'],epoch=state['epoch'],source_seconds=str(at),reason=reason,ocr_reset=True))
        for row in active:
            event=row['payload'];at=_source_time(event)
            # A returned snapshot is event-time causal; no future frame backdates entry.
            at_label=row['event_at'];codes=event.get('qr_occurrences',[])
            code_counts={c['value']:sum(o['value']==c['value'] for o in codes) for c in codes}
            owner=event.get('operator_id')
            wearer=None
            camera_code=media['receipt'].get('camera_id')
            if known and camera_code:
                binding=conn.execute('''SELECT b.person_id,b.id FROM camera_person_binding b JOIN entity c ON c.customer_id=b.customer_id AND c.id=b.camera_id
                    WHERE b.customer_id=%s AND c.kind='capture_device' AND c.code=%s AND c.metadata->>'view_role'='first_person'
                    AND b.valid_from<=%s AND (b.valid_to IS NULL OR b.valid_to>%s)''',(self.core.customer,camera_code,at_label,at_label)).fetchone()
                wearer=str(binding['person_id']) if binding else None
            context_changed=owner!=last_owner or wearer!=last_wearer
            if context_changed:
                close_focus(at,'operator_or_wearer_context_changed')
                for state in states.values(): state['candidate_count']=0;state['candidate_start']=None
            last_owner=owner;last_wearer=wearer
            for state in states.values():
                if at-state['last_seen']>gap and not state['break_pending']:
                    reset(state,at,'tracking_time_gap')
            links,mapped,ambiguity=frame_maps[row['frame_key']];resolutions={}
            eligible=[]
            for track in event.get('tracks',[]):
                tid=track['track_id'];link=links[tid]
                state=states.get(tid)
                if state is None:
                    state=dict(track_id=tid,panel_id=link['panel_id'],link_id=str(link['id']),epoch=1,instrument_id=None,last_seen=at,last_qr=None,candidate_count=0,candidate_start=None,break_pending=False,reason='new_track_requires_qr',operation_status='unknown',pixel_hashes=set(),previous_rgb_sha256=None,last_interaction=None)
                    states[tid]=state
                    notifications.append(dict(track_id=tid,panel_id=state['panel_id'],epoch=1,source_seconds=str(at),reason='new_track',ocr_reset=True))
                elif state['link_id']!=str(link['id']):
                    reset(state,at,'registered_link_or_layout_changed');state['link_id']=str(link['id']);state['panel_id']=link['panel_id']
                if track['status'] in {'occluded','lost','conflict'}:
                    if not state['break_pending']: reset(state,at,'track_'+track['status'])
                    state['reason']='track_'+track['status'];continue
                if track['status']=='reacquired' and not state['break_pending']: reset(state,at,'track_reacquired')
                if len(pts_observations[at][tid])>1:
                    if not state['break_pending']: reset(state,at,'conflicting_observations_at_same_source_pts')
                    state['reason']='conflicting_observations_at_same_source_pts'
                    state['last_seen']=at
                    continue
                observations=mapped[tid]
                conflict=any(code_counts[c['value']]>1 or ambiguity[id(c)] for c in observations)
                ids=[]
                for code in observations:
                    resolved=self._resolve(conn,code['value'],at_label,known)
                    resolutions[code['value']]=resolved
                    if resolved.get('instrument_id'): ids.append(resolved['instrument_id'])
                target=str(link['instrument_id'])
                if conflict or len(observations)>1 or (ids and set(ids)!={target}):
                    if not state['break_pending']: reset(state,at,'qr_or_panel_identity_conflict')
                    state['reason']='qr_or_panel_identity_conflict'
                elif len(observations)==1 and ids==[target]:
                    if state['instrument_id'] not in {None,target}: reset(state,at,'instrument_identity_changed')
                    state.update(instrument_id=target,last_qr=at,break_pending=False,reason='attached_qr_and_registered_geometry')
                elif observations:
                    if state['instrument_id'] is not None: reset(state,at,'unknown_or_unresolved_visible_qr')
                    state['reason']='unknown_or_unresolved_visible_qr'
                elif state['instrument_id'] is not None and state['last_qr'] is not None and at-state['last_qr']<=hold and not state['break_pending']:
                    state['reason']='short_qr_loss_reliable_track_hold'
                else:
                    if state['instrument_id'] is not None: reset(state,at,'qr_identity_hold_expired')
                    state['reason']='qr_identity_unresolved'
                state['last_seen']=at
                measurement=track.get('interaction')
                independent=event['frame_rgb_sha256'] not in state['pixel_hashes']
                metric_ok=interaction_candidate(measurement,event['frame_rgb_sha256'])
                if metric_ok:
                    expected=[project(p,track['canonical_to_frame']) for p in link['geometry'].get('control_polygons',[])]
                    metric_ok=(fingerprint(expected)==fingerprint(measurement.get('control_polygons')) and
                               measurement['previous_rgb_sha256']==state['previous_rgb_sha256'])
                reviewed=bool(track.get('reviewed_operation'))
                operation=reviewed or metric_ok
                if state['instrument_id'] and operation and independent:
                    if state['candidate_start'] is None: state['candidate_start']=at
                    state['candidate_count']+=1
                    state['last_interaction']=at
                    state['operation_status']='reviewed_reference' if reviewed else 'candidate'
                    if reviewed or (policy.candidate_focus_enabled and state['candidate_count']>=policy.enter_samples and at-state['candidate_start']>=Fraction(policy.enter_span_seconds)):
                        eligible.append(state)
                elif not operation:
                    state['candidate_count']=0;state['candidate_start']=None;state['operation_status']='unknown'
                state['pixel_hashes'].add(event['frame_rgb_sha256'])
                state['previous_rgb_sha256']=event['frame_rgb_sha256']
            if focus is not None:
                current=states[focus['track_id']]
                if current['instrument_id']!=focus['instrument_id'] or current['break_pending'] or at-current['last_seen']>gap:
                    close_focus(at,'focused_track_no_longer_reliable')
                elif current['last_interaction'] is None or at-current['last_interaction']>Fraction(policy.focus_idle_seconds):
                    close_focus(at,'interaction_context_idle_timeout_not_physical_completion')
            if len(eligible)>1:
                close_focus(at,'multiple_supported_interaction_candidates')
            elif len(eligible)==1:
                selected=eligible[0]
                if focus is None or focus['track_id']!=selected['track_id']:
                    close_focus(at,'sustained_interaction_focus_change')
                    focus=dict(track_id=selected['track_id'],instrument_id=selected['instrument_id'],focus_status='reviewed_reference' if selected['operation_status']=='reviewed_reference' else 'candidate',operation_status=selected['operation_status'])
                    focus_start=at
            panels=[]
            for state in states.values():
                reliable=not state['break_pending'] and at-state['last_seen']<=gap
                panels.append(dict(track_id=state['track_id'],panel_id=state['panel_id'],epoch=state['epoch'],instrument_id=state['instrument_id'] if reliable else None,
                                   association_status='spatial_identity_supported' if reliable and state['instrument_id'] else 'uncertain',reason=state['reason'],
                                   operation_status=state['operation_status'],visible=state['track_id'] in links,
                                   valid_until_seconds=str(min(state['last_seen']+gap,state['last_qr']+hold)) if state['last_qr'] is not None else str(at)))
            pending=[s['track_id'] for s in states.values() if s['instrument_id'] and s['candidate_count'] and (focus is None or focus['track_id']!=s['track_id'])]
            focus_until=min(states[focus['track_id']]['last_seen']+gap,states[focus['track_id']]['last_interaction']+Fraction(policy.focus_idle_seconds)) if focus else at
            snapshots.append(dict(source_seconds=str(at),frame_key=row['frame_key'],frame_rgb_sha256=event['frame_rgb_sha256'],event_at=clean(at_label),observed_at=clean(row['observed_at']),processed_at=clean(row['processed_at']),panels=panels,current_focus=focus.copy() if focus else None,focus_valid_until_seconds=str(focus_until),pending_focus_candidates=pending,wearer_id=wearer,actual_operator_id=owner,operator_attribution='reviewed_visual_operator' if owner else 'unknown'))
            dependencies.append(dict(event=row['payload_sha256'],resolutions=resolutions))
        if focus is not None:
            focus_intervals.append(dict(start_seconds=str(focus_start),end_seconds=None,**focus,reason_ended=None))
        body=dict(schema_version=SCHEMA,status='candidate_projection',media_id=str(media['media_id']),source_stream_key=media['stream_key'],source_session_key=session,
                  source_sha256=media['verified_sha256'],media_version=media['version'],clock_basis=media['time_basis'],physical_clock_known=known,
                  replay_mode='event_order_causal_decisions; late_replay_is_new_knowledge_not_live_latency',policy=asdict(policy),
                  frame_states=snapshots,focus_intervals=focus_intervals,ocr_notifications=notifications,
                  semantic_operation_attestation='none;pixel_measurements_only_candidates;reviewed_reference_is_explicit_human_input',
                  experiment_or_material_state_changed=False)
        from .rules import Rules
        body['processing_rules']=Rules().identity()
        if too_many_tracks:
            body.update(status='needs_attention',reason='bounded_total_track_id_limit')
        if len(json.dumps(clean(body),allow_nan=False))>policy.max_snapshot_bytes:
            body=dict(schema_version=SCHEMA,status='needs_attention',reason='bounded_snapshot_limit',media_id=str(media['media_id']),frame_states=[],focus_intervals=[],ocr_notifications=[])
        digest=fingerprint(dict(snapshot=body,dependencies=dependencies))
        previous=conn.execute('SELECT * FROM hj_dynamic_context_revision WHERE customer_id=%s AND media_id=%s AND source_session_key=%s ORDER BY revision DESC LIMIT 1',(self.core.customer,media['media_id'],session)).fetchone()
        if previous and previous['input_fingerprint']==digest: return previous
        rev=previous['revision']+1 if previous else 1
        result=conn.execute('''INSERT INTO hj_dynamic_context_revision(customer_id,id,media_id,source_session_key,revision,input_fingerprint,policy_fingerprint,snapshot)
             VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *''',(self.core.customer,uuid4(),media['media_id'],session,rev,digest,fingerprint(asdict(policy)),Jsonb(clean(body)))).fetchone()
        from .context_views import mark_context_dirty
        mark_context_dirty(conn,self.core.customer,*(row['event_at'] for row in rows if row['event_at'] is not None and known))
        self.core._emit(conn,'dynamic_context_revision',media['media_id'],rev,dict(media_id=media['media_id'],revision_id=result['id'],revision=rev,status=body['status'],ocr_notifications=notifications))
        return result

    def query_window(self,media_id,source_session_key,start_seconds,end_seconds,*,revision=None,known_at=None):
        start,end=seconds(start_seconds),seconds(end_seconds)
        if end<start: raise ValueError('Reversed query interval')
        with self.core.db.transaction(readonly=True) as conn:
            media=self._media(conn,media_id,source_session_key)
            if end>Fraction(media['duration_seconds']): raise ValueError('Query escaped source media')
            row=conn.execute('''SELECT * FROM hj_dynamic_context_revision WHERE customer_id=%s AND media_id=%s AND source_session_key=%s
                 AND (%s::integer IS NULL OR revision=%s) AND (%s::timestamptz IS NULL OR processed_at<=%s)
                 ORDER BY revision DESC LIMIT 1''',(self.core.customer,ident(media_id),source_session_key,revision,revision,instant(known_at) if known_at else None,instant(known_at) if known_at else None)).fetchone()
        if row is None: return dict(schema_version=SCHEMA,status='unknown',per_panel=[],current_focus=None)
        snapshot=row['snapshot'];frames=snapshot.get('frame_states',[]);selected=[]
        for index,frame in enumerate(frames):
            at=seconds(frame['source_seconds']);next_at=seconds(frames[index+1]['source_seconds']) if index+1<len(frames) else end
            if index+1<len(frames) and next_at==at: continue
            if (at<=start and (start<next_at or index+1==len(frames)) if start==end else at<end and next_at>start):
                selected.append((max(start,at),min(end,next_at),frame))
        grouped={}
        for lo,hi,frame in selected:
            for panel in frame['panels']:
                valid=seconds(panel['valid_until_seconds'])
                supported=panel['association_status']=='spatial_identity_supported' and (lo<valid if lo!=hi else lo<=valid) and hi<=valid
                group=grouped.setdefault((panel['track_id'],panel['panel_id']),[])
                group.append(dict(start_seconds=str(lo),end_seconds=str(hi),epoch=panel['epoch'],instrument_id=panel['instrument_id'] if supported else None,
                                  association_status=panel['association_status'] if supported else 'uncertain',current_focus=frame['current_focus'] if hi<=seconds(frame['focus_valid_until_seconds']) else None,pending_focus_candidates=frame['pending_focus_candidates'],reason=panel['reason']))
        per_panel=[]
        for (track,panel),segments in grouped.items():
            ids={v['instrument_id'] for v in segments};epochs={v['epoch'] for v in segments}
            complete=bool(segments) and seconds(segments[0]['start_seconds'])==start and seconds(segments[-1]['end_seconds'])==end and all(a['end_seconds']==b['start_seconds'] for a,b in zip(segments,segments[1:]))
            identity_ok=complete and len(ids)==1 and None not in ids and len(epochs)==1 and all(v['association_status']=='spatial_identity_supported' for v in segments)
            contexts={fingerprint(v['current_focus']) for v in segments}
            per_panel.append(dict(track_id=track,panel_id=panel,instrument_id=next(iter(ids)) if identity_ok else None,
                                  epoch=next(iter(epochs)) if len(epochs)==1 else None,
                                  physical_association_status='spatial_identity_supported' if identity_ok else 'uncertain',
                                  business_context_status='uncertain_crosses_context_boundary' if len(contexts)>1 else 'uncertain_pending_focus_change' if any(v['pending_focus_candidates'] for v in segments) else 'candidate_context_only' if segments[0]['current_focus'] else 'unknown',
                                  ocr_identity_key=[media['stream_key'],source_session_key,str(media_id),next(iter(ids)) if identity_ok else track,panel,next(iter(epochs))] if identity_ok else None,
                                  segments=segments))
        return clean(dict(schema_version=SCHEMA,status=snapshot['status'],revision=row['revision'],revision_id=row['id'],knowledge_processed_at=row['processed_at'],
                          media_id=str(media_id),source_stream_key=media['stream_key'],source_session_key=source_session_key,clock_basis=media['time_basis'],
                          requested_range=[str(start),str(end)],per_panel=per_panel,
                          current_focus=selected[-1][2]['current_focus'] if selected and end<=seconds(selected[-1][2]['focus_valid_until_seconds']) else None,
                          focus_role='display_only_not_authority_for_other_panels',ocr_notifications=snapshot.get('ocr_notifications',[]),
                          experimental_or_material_completion_implied=False))
