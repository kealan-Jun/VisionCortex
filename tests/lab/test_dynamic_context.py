"""Persisted real schema/pixel metrics with explicitly synthetic identities/media."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from contextlib import contextmanager
import socket
import threading
import time
from uuid import UUID, uuid4

import numpy as np
import httpx
import pytest
import qrcode
import uvicorn
from psycopg.types.json import Jsonb

from visioncortex.lab.domain import Conflict, Core, Missing
from visioncortex.lab.api import create_app
from visioncortex.lab.dynamic_context import DynamicContext, DynamicPolicy
from visioncortex.lab.dynamic_vision import frame_hash, interaction_pixels, project
from visioncortex.lab.dynamic_vision import qr_occurrences
from visioncortex.lab.timeline_report import render_report
from ._fixtures import db as evidence_db, settings as evidence_settings
from .test_timeline import at, hj as timeline_fixture, media, run, bind, record, DAY


@pytest.fixture(scope='session', name='db')
def isolated_db():
    return evidence_db.__wrapped__()


@pytest.fixture
def settings(core, tmp_path):
    return evidence_settings.__wrapped__(core, tmp_path)


@pytest.fixture(name='hj')
def context(core, settings):
    return timeline_fixture.__wrapped__(core, settings)


def box(x0,y0,x1,y1):
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


GEOMETRY = dict(coordinates='canonical_normalized_v1',device_polygon=box(0,0,1,1),
                panel_polygon=box(.1,.1,.7,.35),qr_region=box(.75,.1,.95,.3),
                control_polygons=[box(.1,.6,.35,.85)])


@pytest.fixture
def scene(hj):
    m = media(hj, 'dynamic-'+str(uuid4()), duration=4)
    core = hj['core']; dynamic = DynamicContext(core)
    devices=[];links=[]
    for i in range(2):
        issued=core.issue_equipment('reserve-'+str(i),actor='synthetic fixture')
        active=core.activate_equipment('activate-'+str(i),issued['device_id'],verified_at=at('08:00:00'),actor='synthetic fixture',evidence={'fixture_only':True})
        devices.append(active['device_id'])
        links.append(dynamic.register_panel_link('link-'+str(i),active['device_id'],'panel-'+str(i),'fixture-layout/1',GEOMETRY,
                     actor='synthetic reviewed geometry',reason='generated device layout only',evidence_ids=[m['evidence_id']]))
    with core.db.transaction(readonly=True) as conn:
        sha=conn.execute('SELECT verified_sha256 FROM media_asset WHERE customer_id=%s AND id=%s',(core.customer,UUID(m['media_id']))).fetchone()['verified_sha256']
    return dict(hj=hj,core=core,dynamic=dynamic,m=m,sha=sha,devices=devices,links=links)


def transform(i):
    return [[120.,0.,8.+128*i],[0.,120.,5.],[0.,0.,1.]]


def image(a=0,b=0):
    result=np.zeros((128,256,3),np.uint8)
    result[80:100,23:43]=a
    result[80:100,151:171]=b
    return result


def event(scene,n,*,current=None,previous=None,operate=None,codes=None,statuses=None):
    current=current if current is not None else image()
    tracks=[]
    for i,link in enumerate(scene['links']):
        track=dict(track_id='track-'+str(i),link_id=link['id'],layout_revision='fixture-layout/1',canonical_to_frame=transform(i),status=(statuses or {}).get(i,'reliable'))
        if operate == i:
            track['interaction']=interaction_pixels(previous,current,control_polygons=[project(p,transform(i)) for p in GEOMETRY['control_polygons']],background_polygons=[box(0,0,255,3)])
        tracks.append(track)
    return dict(schema_version='vc-lab-dynamic-frame/1',frame_key='frame-'+str(n),source_pts=int(Fraction(str(n))*100000),time_base='1/1000000',stream_origin='0',frame_rgb_sha256=frame_hash(current),source_sha256=scene['sha'],media_version=1,evidence_ids=[scene['m']['evidence_id']],observed_at=at('10:00:00'),tracks=tracks,
                qr_occurrences=codes if codes is not None else [dict(value=scene['devices'][i],polygon=project(box(.78,.14,.92,.26),transform(i))) for i in range(2)])


def append(scene,events,key=None):
    return scene['dynamic'].append_batch(key or str(uuid4()),scene['m']['media_id'],'fixture-session',events)


def query(scene,start,end=None,**kwargs):
    return scene['dynamic'].query_window(scene['m']['media_id'],'fixture-session',str(start),str(start if end is None else end),**kwargs)


def series(scene):
    frames=[image()]
    events=[event(scene,0,current=frames[0])]
    for n in range(1,7):
        frame=image(n*40 if n<=3 else 120,(n-3)*40 if n>3 else 0)
        events.append(event(scene,n,current=frame,previous=frames[-1],operate=0 if n<=3 else 1))
        frames.append(frame)
    return events


def test_two_visible_qrs_identify_their_own_panels_but_never_confirm_operation(scene):
    value=append(scene,[event(scene,0)])
    assert value['status']=='candidate_projection'
    result=query(scene,0)
    assert result['current_focus'] is None
    assert {p['instrument_id'] for p in result['per_panel']} == set(scene['devices'])
    assert all(p['physical_association_status']=='spatial_identity_supported' for p in result['per_panel'])
    assert all(p['business_context_status']=='unknown' for p in result['per_panel'])
    assert not result['experimental_or_material_completion_implied']


def test_actual_local_changes_candidate_switch_without_ending_experiment_or_writing_ledger(scene):
    hj=scene['hj'];rid=run(hj);binding=bind(hj,rid)
    with scene['core'].db.transaction(readonly=True) as conn:
        ledger_before=conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',(scene['core'].customer,)).fetchone()['n']
    append(scene,series(scene))
    first=query(scene,'.3');last=query(scene,'.6')
    assert first['current_focus']['instrument_id']==scene['devices'][0]
    assert last['current_focus']['instrument_id']==scene['devices'][1]
    assert last['current_focus']['focus_status']==last['current_focus']['operation_status']=='candidate'
    assert query(scene,'.4')['per_panel'][0]['business_context_status']=='uncertain_pending_focus_change'
    crossing=query(scene,'.3','.6')
    # Half-open range ending at decision .6 does not include its new frame.
    assert all(p['physical_association_status']=='spatial_identity_supported' for p in crossing['per_panel'])
    assert record(hj,rid)['record_status']=='open'
    with scene['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',(scene['core'].customer,)).fetchone()['n']==ledger_before
        assert conn.execute('SELECT id FROM hj_experiment_binding WHERE customer_id=%s AND id=%s',(scene['core'].customer,UUID(binding['id']))).fetchone()


def test_range_crossing_focus_boundary_does_not_move_other_panel_readings(scene):
    append(scene,series(scene))
    result=query(scene,'.5','.65')
    assert all(p['business_context_status']=='uncertain_crosses_context_boundary' for p in result['per_panel'])
    assert {p['instrument_id'] for p in result['per_panel']}==set(scene['devices'])
    assert all(p['ocr_identity_key'][3]==p['instrument_id'] for p in result['per_panel'])
    assert len({tuple(p['ocr_identity_key']) for p in result['per_panel']})==2


@pytest.mark.parametrize('mode',['unknown','duplicate','outside','wrong_registered_device'])
def test_qr_ambiguity_cannot_use_largest_or_latest_identity(scene,mode):
    e=event(scene,0)
    if mode=='unknown': e['qr_occurrences'][0]['value']='unknown-asset-'+'x'*180
    elif mode=='duplicate': e['qr_occurrences'][1]['value']=scene['devices'][0]
    elif mode=='outside': e['qr_occurrences'][0]['polygon']=box(1,1,4,4)
    else: e['qr_occurrences'][0]['value']=scene['devices'][1]
    append(scene,[e])
    result=query(scene,0)
    assert result['per_panel'][0]['instrument_id'] is None
    assert result['current_focus'] is None
    if mode=='duplicate': assert all(p['instrument_id'] is None for p in result['per_panel'])


def test_short_qr_loss_holds_reliable_identity_then_occlusion_resets_only_its_epoch(scene):
    events=[event(scene,0),event(scene,1,codes=[]),event(scene,2,statuses={0:'occluded'}),event(scene,3,statuses={0:'reacquired'})]
    append(scene,events)
    assert query(scene,'.1')['per_panel'][0]['epoch']==1
    reset=query(scene,'.2');restored=query(scene,'.3')
    assert reset['per_panel'][0]['instrument_id'] is None
    assert restored['per_panel'][0]['epoch']==2 and restored['per_panel'][1]['epoch']==1
    assert restored['per_panel'][0]['instrument_id']==scene['devices'][0]
    assert query(scene,'.1','.3')['per_panel'][0]['physical_association_status']=='uncertain'
    assert query(scene,'.1','.3')['per_panel'][1]['physical_association_status']=='spatial_identity_supported'
    assert all(n['track_id']=='track-0' for n in restored['ocr_notifications'] if n['source_seconds']!='0')


def test_unknown_clock_keeps_source_pts_and_null_world_event_time(scene):
    with scene['core'].db.transaction() as conn:
        conn.execute('UPDATE hj_timeline_media SET time_basis=%s WHERE customer_id=%s AND media_id=%s',(Jsonb({'basis':'sender_label_only','uncertainty_us':None,'clock_sync_valid':False}),scene['core'].customer,UUID(scene['m']['media_id'])))
    e=event(scene,21);e.update(source_pts=2300000,stream_origin='1/5')
    value=append(scene,[e])
    frame=value['snapshot']['frame_states'][0]
    assert frame['source_seconds']=='21/10' and frame['event_at'] is None
    assert all(p['instrument_id'] is None for p in query(scene,'2.1')['per_panel'])
    assert query(scene,'2.1')['clock_basis']['uncertainty_us'] is None


def test_nonzero_original_pts_subtracts_origin_without_changing_evidence_range(scene):
    e=event(scene,1);e.update(source_pts=2200000,stream_origin='2')
    value=append(scene,[e])
    assert value['snapshot']['frame_states'][0]['source_seconds']=='1/5'
    assert query(scene,'.2')['per_panel'][0]['instrument_id']==scene['devices'][0]
    with pytest.raises(ValueError,match='bounded media'):
        append(scene,[dict(e,frame_key='negative-offset',stream_origin='3')])


def test_immutable_idempotency_restart_late_replay_and_historical_revision(scene):
    late=event(scene,2)
    first=append(scene,[late],key='same-command')
    assert append(scene,[late],key='same-command')==first
    duplicate=append(scene,[late])
    assert duplicate['accepted']==0 and duplicate['revision']==first['revision']
    old=query(scene,'.2')
    scene['dynamic']=DynamicContext(scene['core'])
    assert query(scene,'.2')==old
    newer=append(scene,[event(scene,0),event(scene,1)])
    assert newer['revision']==first['revision']+1
    assert query(scene,'.2',revision=first['revision'])==old
    assert query(scene,'.2',known_at=old['knowledge_processed_at'])==old
    with pytest.raises(Conflict,match='immutable'):
        append(scene,[dict(late,frame_rgb_sha256='0'*64)])


def test_explicit_correction_preserves_raw_frame_and_old_projection(scene):
    first=append(scene,[event(scene,0)])
    correction=event(scene,0,codes=[])
    correction.update(frame_key='corrected',supersedes_frame_key='frame-0',revision_actor='fixture reviewer',revision_reason='fixture QR localization correction')
    append(scene,[correction])
    assert all(p['instrument_id'] is None for p in query(scene,0)['per_panel'])
    assert query(scene,0,revision=first['revision'])['per_panel'][0]['instrument_id']==scene['devices'][0]
    with scene['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s',(scene['core'].customer,UUID(scene['m']['media_id']))).fetchone()['n']==2


def test_duplicate_pixels_untrusted_operator_and_unrelated_previous_frame_are_not_support(scene):
    events=series(scene)[:4]
    # Pretending the same image is new at later PTS cannot accumulate entry support.
    events[2]['frame_rgb_sha256']=events[1]['frame_rgb_sha256']
    events[3]['frame_rgb_sha256']=events[1]['frame_rgb_sha256']
    append(scene,events)
    assert query(scene,'.3')['current_focus'] is None
    bad=event(scene,4);bad['operator_id']=scene['hj']['person']
    with pytest.raises(ValueError,match='operator'):
        append(scene,[bad])


def test_same_pts_conflicting_pixels_are_unresolved_instead_of_tie_order_winner(scene):
    a=event(scene,0);b=event(scene,0,current=image(100,0));b['frame_key']='same-pts-different-pixels'
    append(scene,[a,b])
    result=query(scene,0)
    assert all(p['instrument_id'] is None for p in result['per_panel'])
    assert result['current_focus'] is None


def test_same_pts_same_pixels_conflicting_interpretations_cannot_choose_lexical_winner(scene):
    valid=event(scene,0)
    conflicting=event(scene,0)
    conflicting['frame_key']='a-unknown-before-valid'
    conflicting['qr_occurrences'][0]['value']='unknown-conflicting-interpretation'
    append(scene,[valid,conflicting])
    result=query(scene,0)
    assert result['per_panel'][0]['instrument_id'] is None
    assert result['per_panel'][0]['ocr_identity_key'] is None
    assert result['per_panel'][1]['instrument_id']==scene['devices'][1]
    assert result['per_panel'][1]['epoch']==1
    assert result['current_focus'] is None
    assert result['per_panel'][0]['segments'][0]['reason']=='conflicting_observations_at_same_source_pts'


def test_correction_fork_rejected_and_leaf_correction_preserves_previous_revisions(scene):
    append(scene,[event(scene,0)])
    first=event(scene,0)
    first.update(frame_key='correction-a',supersedes_frame_key='frame-0',revision_actor='fixture',revision_reason='first correction')
    revision=append(scene,[first])['revision']
    fork=dict(first,frame_key='correction-fork')
    with pytest.raises(Conflict,match='already superseded'):
        append(scene,[fork])
    leaf=dict(first,frame_key='correction-b',supersedes_frame_key='correction-a',qr_occurrences=[])
    assert append(scene,[leaf])['revision']==revision+1
    assert query(scene,0,revision=revision)['per_panel'][0]['instrument_id']==scene['devices'][0]
    assert query(scene,0)['per_panel'][0]['instrument_id'] is None
    with scene['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s',(scene['core'].customer,UUID(scene['m']['media_id']))).fetchone()['n']==3


def test_boundaries_scope_and_limits_fail_closed_without_backlog(scene):
    e=event(scene,0)
    other=Core.customer_instance(scene['core'].db,'dynamic-other-'+str(uuid4()),scope='fixture')
    with pytest.raises(Missing):
        DynamicContext(Core(scene['core'].db,other['id'])).observe('outside',scene['m']['media_id'],'fixture-session',e)
    with pytest.raises(Conflict,match='session'):
        scene['dynamic'].observe('wrong-session',scene['m']['media_id'],'other-session',e)
    with pytest.raises(Conflict,match='SHA'):
        append(scene,[dict(e,source_sha256='0'*64)])
    scene['dynamic']=DynamicContext(scene['core'],replace(DynamicPolicy(),max_events=1))
    append(scene,[e])
    assert append(scene,[event(scene,1)])==dict(status='needs_attention',reason='bounded_research_event_limit',accepted=0,raw_originals_unchanged=True)


def test_invalid_geometry_cannot_label_screen_pixels_as_interaction_controls(scene):
    bad=deepcopy(GEOMETRY);bad['control_polygons']=[bad['panel_polygon']]
    with pytest.raises(ValueError,match='digits'):
        scene['dynamic'].register_panel_link('bad-geometry',scene['devices'][0],'screen','rev',bad,actor='fixture',reason='fixture',evidence_ids=[scene['m']['evidence_id']])


@contextmanager
def local_api(core,settings):
    listener=socket.socket()
    listener.bind(('127.0.0.1',0))
    port=listener.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(create_app(core,settings),host='127.0.0.1',port=port,log_level='error',lifespan='off'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[listener]},daemon=True)
    thread.start()
    try:
        deadline=time.monotonic()+5
        while not server.started and time.monotonic()<deadline:
            time.sleep(.01)
        assert server.started
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',headers={'Authorization':'Bearer '+settings.token},timeout=10) as client:
            yield client
    finally:
        server.should_exit=True
        thread.join(5)
        listener.close()
        assert not thread.is_alive()


def test_localhost_commands_to_dynamic_query_trace_and_daily_report(scene):
    """One existing API/runtime; actual generated QR/pixels + isolated PG only."""
    hj=scene['hj'];rid=run(hj);bind(hj,rid)
    with local_api(scene['core'],hj['settings']) as client:
        initial=client.get('/internal/timeline/reports/'+DAY)
        assert initial.status_code==200
        old_revision=initial.json()['revision']
        api_links=[]
        for i,instrument in enumerate(scene['devices']):
            result=client.post('/internal/commands/timeline-register-panel-link',json=dict(key='api-link-'+str(i),instrument_id=instrument,panel_id='panel-'+str(i),layout_revision='fixture-layout/1',geometry=GEOMETRY,actor='synthetic reviewed geometry',reason='localhost fixture only',evidence_ids=[scene['m']['evidence_id']]))
            assert result.status_code==200,result.text
            api_links.append(result.json())
        transforms=[[[300.,0.,10.+320*i],[0.,300.,10.],[0.,0.,1.]] for i in range(2)]
        events=[];previous=None
        for n in range(7):
            native=np.full((320,640,3),255,np.uint8)
            for i,uid in enumerate(scene['devices']):
                qr=qrcode.QRCode(box_size=1,border=4)
                qr.add_data(uid);qr.make(fit=True)
                pixels=np.asarray(qr.make_image().convert('RGB'))
                x=245+320*i;y=50
                native[y:y+len(pixels),x:x+len(pixels)]=pixels
                native[200:250,50+320*i:100+320*i]=(n*40 if n<=3 else 120) if i==0 else (n-3)*40 if n>3 else 0
            decoded=qr_occurrences(native)
            assert {v['value'] for v in decoded['occurrences']}==set(scene['devices'])
            tracks=[]
            for i,link in enumerate(api_links):
                track=dict(track_id='api-track-'+str(i),link_id=link['id'],layout_revision='fixture-layout/1',canonical_to_frame=transforms[i],status='reliable')
                if n and i==(0 if n<=3 else 1):
                    track['interaction']=interaction_pixels(previous,native,control_polygons=[project(p,transforms[i]) for p in GEOMETRY['control_polygons']],background_polygons=[box(0,0,639,3)])
                tracks.append(track)
            events.append(dict(schema_version='vc-lab-dynamic-frame/1',frame_key='api-frame-'+str(n),source_pts=n*100000,time_base='1/1000000',stream_origin='0',frame_rgb_sha256=decoded['frame_rgb_sha256'],source_sha256=scene['sha'],media_version=1,evidence_ids=[scene['m']['evidence_id']],tracks=tracks,qr_occurrences=decoded['occurrences'],observed_at=at('10:00:00')))
            previous=native
        body=dict(key='api-persistent-batch',media_id=scene['m']['media_id'],source_session_key='fixture-session',frame_events=events)
        submitted=client.post('/internal/commands/timeline-observe-context-batch',json=body)
        assert submitted.status_code==200,submitted.text
        assert client.post('/internal/commands/timeline-observe-context-batch',json=body).json()==submitted.json()
        window=client.post('/internal/commands/timeline-context-window',json=dict(media_id=scene['m']['media_id'],source_session_key='fixture-session',start_seconds='.6',end_seconds='.65'))
        assert window.status_code==200,window.text
        assert window.json()['current_focus']['instrument_id']==scene['devices'][1]
        assert {p['instrument_id'] for p in window.json()['per_panel']}==set(scene['devices'])
        trace=client.get('/internal/timeline/trace',params={'start':at('09:00:00'),'end':at('09:00:04')})
        assert trace.status_code==200,trace.text
        contexts=trace.json()['context_evidence']['dynamic_contexts']
        assert len(contexts)==1 and len(contexts[0]['focus_intervals'])==2
        assert contexts[0]['focus_intervals'][0]['reason_ended']=='sustained_interaction_focus_change'
        report=client.get('/internal/timeline/reports/'+DAY)
        assert report.status_code==200,report.text
        assert report.json()['revision']>old_revision
        text=render_report(report.json()['document'])
        assert scene['devices'][0]+' → '+scene['devices'][1] in text
        assert 'sustained_interaction_focus_change' in text
        assert '切换不结束设备实验' in text and 'candidate' in text
        assert report.json()['document']['experiments'][0]['record']['record_status']=='open'
    with scene['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s',(scene['core'].customer,UUID(scene['m']['media_id']))).fetchone()['n']==7
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',(scene['core'].customer,)).fetchone()['n']==0
        assert conn.execute("SELECT count(*) AS n FROM command_receipt WHERE customer_id=%s AND key='api-persistent-batch'",(scene['core'].customer,)).fetchone()['n']==1
        assert conn.execute("SELECT count(*) AS n FROM outbox WHERE customer_id=%s AND kind='dynamic_context_revision'",(scene['core'].customer,)).fetchone()['n']==1
