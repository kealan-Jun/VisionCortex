"""Automatic actual pixels -> QR/fit/metrics -> original Core observations.

All identities/scenes are generated fixtures. No real operation ground truth.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest
import qrcode

from visioncortex.lab.domain import Conflict, Core, Missing
from visioncortex.lab.dynamic_context import DynamicContext
from visioncortex.lab.dynamic_video import DynamicVideo, DynamicVideoRejected
from visioncortex.lab.dynamic_vision import frame_hash, interaction_candidate
from visioncortex.lab.rules import Rules
from .test_dynamic_context import (
    isolated_db as evidence_db_fixture, settings as evidence_settings_fixture,
    context as timeline_fixture, scene as dynamic_fixture, GEOMETRY,
    event as fixture_event, append as append_frames, query as query_window,
    transform as fixture_transform,
)


@pytest.fixture(scope='session',name='db')
def isolated_db():
    return evidence_db_fixture.__wrapped__()


@pytest.fixture
def settings(core,tmp_path):
    return evidence_settings_fixture.__wrapped__(core,tmp_path)


@pytest.fixture(name='hj')
def timeline_context(core,settings):
    return timeline_fixture.__wrapped__(core,settings)


@pytest.fixture
def scene(hj):
    return dynamic_fixture.__wrapped__(hj)


def native_frame(devices,a=0,b=0):
    rng=np.random.default_rng(631)
    native=np.repeat(rng.integers(30,180,(320,640,1),np.uint8),3,axis=2)
    native[:5,:5]=[17,39,231]
    for i,uid in enumerate(devices):
        qr=qrcode.QRCode(box_size=1,border=4);qr.add_data(uid);qr.make(fit=True)
        pixels=np.asarray(qr.make_image().convert('RGB'))
        x=245+320*i;y=50
        native[y:y+len(pixels),x:x+len(pixels)]=pixels
        native[200:250,50+320*i:100+320*i]=a if i==0 else b
    return native


def source_ref(sha='a'*64):
    return dict(sha256=sha,media_version=1,start_seconds='0',end_seconds='4')


def frame_ref(image,n,sha='a'*64,rotate=False):
    raw=np.ascontiguousarray(image[::-1,::-1]) if rotate else image
    h,w=image.shape[:2]
    return dict(pts=200000+n*100000,time_base='1/1000000',stream_origin='1/5',media_offset=str(Fraction(n,10)),source_sha256=sha,
                original_bgr_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),upright_rgb_sha256=frame_hash(image),
                source_to_upright=[[-1,0,w-1],[0,-1,h-1],[0,0,1]] if rotate else [[1,0,0],[0,1,0],[0,0,1]])


def configuration(links,reference,*,batch=1):
    panels=[]
    for i,link in enumerate(links):
        x0=10+320*i;x1=309+320*i
        panels.append(dict(link_id=link['id'],link_revision=link['revision'],layout_revision=link['layout_revision'],track_id='pixel-track-'+str(i),
                    reference_frame=dict(pts=200000,time_base='1/1000000',stream_origin='1/5',upright_rgb_sha256=frame_hash(reference),
                        device_quad_normalized=[[x0/639,10/319],[x1/639,10/319],[x1/639,309/319],[x0/639,309/319]]),
                    stable_boxes=[[.02,.02,.98,.98]],background_boxes=[[.4,.55,.85,.9]],registration_mode='rigid'))
    return dict(schema_version='vc-lab-dynamic-video/1',enabled=True,revision='generated-pixel-fixture/1',max_frames=20,batch_frames=batch,
                max_buffer_bytes=16*1024**2,max_seconds=30,panels=panels)


class PixelFixtureContext:
    """Unit context only; real Core/schema cases below separately exercise DB."""
    def __init__(self):
        self.devices=[str(uuid4()),str(uuid4())]
        self.links=[dict(id=str(uuid4()),revision=1,layout_revision='fixture-layout/1',geometry=deepcopy(GEOMETRY)) for _ in range(2)]
        self.events=[];self.calls=0
    def video_inputs(self,media,session,pins):
        return dict(source_sha256='a'*64,media_version=1,duration_seconds='4',evidence_refs=[dict(id=str(uuid4()),start_seconds='0',end_seconds='4')],panel_links=self.links[:len(pins)])
    def append_batch(self,key,media,session,events):
        self.calls+=1;self.events.extend(deepcopy(events))
        return dict(status='candidate_projection',revision=self.calls,snapshot={})


def test_actual_qr_structure_registration_and_control_pixels_are_automatic():
    context=PixelFixtureContext();reference=native_frame(context.devices)
    observer=DynamicVideo(context,'unit-media','unit-session',source_ref(),configuration(context.links,reference,batch=3))
    for n in range(7):
        image=native_frame(context.devices,n*40 if n<=3 else 120,(n-3)*40 if n>3 else 0)
        marker=observer.observe(image,frame_ref(image,n,rotate=True))
        assert marker['status']==('committed' if n%3==2 else 'buffered')
    observer.flush()
    assert context.calls==3 and len(context.events)==7
    first=context.events[0]
    assert {c['value'] for c in first['qr_occurrences']}==set(context.devices)
    assert first['original_rgb_sha256']!=first['upright_rgb_sha256']
    assert first['original_bgr_sha256']!=first['original_rgb_sha256']
    assert first['tracks'][0]['registration']['accepted']
    assert first['tracks'][0]['registration']['matches']>=6
    for n,event in enumerate(context.events[1:],1):
        selected=event['tracks'][0 if n<=3 else 1]
        assert interaction_candidate(selected.get('interaction'),event['upright_rgb_sha256'])
    receipt=observer.receipt()
    assert receipt['qr_calls']==7 and receipt['db_commits']==3
    assert receipt['peak_buffer_bytes']<=16*1024**2
    assert receipt['confirmed_operations']==receipt['new_identities_created']==0


def test_actual_camera_shift_fits_transform_but_cannot_supply_interaction():
    context=PixelFixtureContext();reference=native_frame(context.devices)
    observer=DynamicVideo(context,'unit-media','unit-session',source_ref(),configuration(context.links,reference))
    observer.observe(reference,frame_ref(reference,0))
    moved=cv2.warpAffine(reference,np.float32([[1,0,3],[0,1,0]]),(640,320))
    observer.observe(moved,frame_ref(moved,1))
    a,b=context.events
    shift=np.asarray(b['tracks'][0]['canonical_to_frame'])-np.asarray(a['tracks'][0]['canonical_to_frame'])
    assert shift[0,2]==pytest.approx(3,abs=.5)
    assert b['tracks'][0]['interaction_camera_motion_pixels']>.75
    assert 'interaction' not in b['tracks'][0]
    assert b['tracks'][0]['interaction_reason']=='camera_motion_gate_no_operation_support'


def test_no_link_unknown_qr_is_recorded_without_physical_identity_or_manual_events():
    context=PixelFixtureContext()
    image=np.asarray(qrcode.make('{"TYPE":"CONOW_BIZ","ASSET_TAG":"C20060260"}').convert('RGB'))[:,:,::-1].copy()
    config=configuration([],image)
    observer=DynamicVideo(context,'unit-media','unit-session',source_ref(),config)
    assert observer.observe(image,frame_ref(image,0))['status']=='committed'
    assert context.events[0]['tracks']==[] and context.events[0]['qr_occurrences']
    assert observer.receipt()['new_identities_created']==0


@pytest.mark.parametrize('failure',['raw_hash','source_hash','reference_hash','time','bytes','frame_count'])
def test_pixel_source_and_limits_fail_closed_and_never_flush_pending_on_error(failure):
    context=PixelFixtureContext();image=native_frame(context.devices)
    config=configuration(context.links,image,batch=5)
    if failure=='reference_hash': config['panels'][0]['reference_frame']['upright_rgb_sha256']='0'*64
    if failure=='bytes': config['max_buffer_bytes']=1024
    if failure=='frame_count': config['max_frames']=1
    observer=DynamicVideo(context,'unit-media','unit-session',source_ref(),config)
    meta=frame_ref(image,0)
    if failure=='raw_hash': meta['original_bgr_sha256']='0'*64
    if failure=='source_hash': meta['source_sha256']='0'*64
    if failure=='time': meta['media_offset']='1'
    if failure=='frame_count': observer.observe(image,meta);meta=frame_ref(image,1)
    with pytest.raises((DynamicVideoRejected,Conflict)):
        observer.observe(image,meta)
    assert not context.events
    with pytest.raises(DynamicVideoRejected,match='closed'):
        observer.flush()


def test_reference_with_no_stable_pixels_never_supplies_a_homography():
    context=PixelFixtureContext();image=np.zeros((320,640,3),np.uint8)
    observer=DynamicVideo(context,'unit-media','unit-session',source_ref(),configuration(context.links,image))
    observer.observe(image,frame_ref(image,0))
    assert context.events[0]['tracks']==[]
    assert observer.receipt()['registration_failures']==2


def test_automatic_pixels_append_real_core_focus_and_persistent_raw_frames(scene):
    reference=native_frame(scene['devices']);config=configuration(scene['links'],reference)
    observer=DynamicVideo(scene['dynamic'],scene['m']['media_id'],'fixture-session',source_ref(scene['sha']),config)
    for n in range(7):
        image=native_frame(scene['devices'],n*40 if n<=3 else 120,(n-3)*40 if n>3 else 0)
        assert observer.observe(image,frame_ref(image,n,scene['sha']))['status']=='committed'
    query=scene['dynamic'].query_window
    assert query(scene['m']['media_id'],'fixture-session','.3','.3')['current_focus']['instrument_id']==scene['devices'][0]
    assert query(scene['m']['media_id'],'fixture-session','.6','.6')['current_focus']['instrument_id']==scene['devices'][1]
    with scene['core'].db.transaction(readonly=True) as conn:
        events=conn.execute('SELECT payload FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s ORDER BY source_seconds',(scene['core'].customer,UUID(scene['m']['media_id']))).fetchall()
        assert len(events)==7 and all(e['payload']['pixel_producer']['schema_version']=='vc-lab-dynamic-video/1' for e in events)
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',(scene['core'].customer,)).fetchone()['n']==0
    assert observer.receipt()['processing_rules']==Rules().identity()


def test_registration_failure_resets_only_affected_panel_real_core(scene):
    reference=native_frame(scene['devices'])
    observer=DynamicVideo(scene['dynamic'],scene['m']['media_id'],'fixture-session',source_ref(scene['sha']),configuration(scene['links'],reference))
    observer.observe(reference,frame_ref(reference,0,scene['sha']))
    obscured=reference.copy();obscured[5:315,5:315]=0
    observer.observe(obscured,frame_ref(obscured,1,scene['sha']))
    result=scene['dynamic'].query_window(scene['m']['media_id'],'fixture-session','.1','.1')
    a,b=result['per_panel']
    assert a['instrument_id'] is None and a['epoch']==2
    assert b['instrument_id']==scene['devices'][1] and b['epoch']==1
    observer.observe(reference,frame_ref(reference,2,scene['sha']))
    assert scene['dynamic'].query_window(scene['m']['media_id'],'fixture-session','.2','.2')['per_panel'][0]['instrument_id']==scene['devices'][0]


def test_actual_customer_scope_and_link_revision_must_match_before_pixels(scene):
    reference=native_frame(scene['devices']);config=configuration(scene['links'],reference)
    bad=deepcopy(config);bad['panels'][0]['link_revision']+=1
    with pytest.raises(Conflict,match='revision'):
        DynamicVideo(scene['dynamic'],scene['m']['media_id'],'fixture-session',source_ref(scene['sha']),bad)
    other=Core.customer_instance(scene['core'].db,'pixel-other-'+str(uuid4()),scope='fixture')
    with pytest.raises(Missing):
        DynamicVideo(DynamicContext(Core(scene['core'].db,other['id'])),scene['m']['media_id'],'fixture-session',source_ref(scene['sha']),config)


def test_lost_track_stale_geometry_cannot_conflict_with_valid_other_panel(scene):
    append_frames(scene,[fixture_event(scene,0)])
    lost=fixture_event(scene,1,statuses={0:'lost'})
    # Deliberately invalid for current A: stale failed geometry overlaps B.
    # It exists only to reset A; it must never participate in QR ownership.
    lost['tracks'][0]['canonical_to_frame']=fixture_transform(1)
    append_frames(scene,[lost])
    a,b=query_window(scene,'.1')['per_panel']
    assert a['instrument_id'] is None and a['epoch']==2
    assert b['instrument_id']==scene['devices'][1] and b['epoch']==1
