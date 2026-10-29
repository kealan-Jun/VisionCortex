"""New frame-map counterexamples plus actual codec/isolated PG; no model calls."""
from ._support import decoder_python
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import shutil
import subprocess
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from psycopg.types.json import Jsonb

from visioncortex.lab.domain import Conflict, fingerprint
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.retained_frame_map import (FrameMappedClips, FrameMapLimits, LocalPixelProof,
    SCHEMA, load_pixel_readback, query_frames, validate_frame_map, verify_local_pixels)
from visioncortex.lab.temporal_video import decode_frames
from visioncortex.lab.timeline import Timeline
from ._fixtures import db as evidence_db, settings as evidence_settings


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def saved(p,x):
    p.write_text(json.dumps(x,indent=2)+'\n');return sha(p)


def proposal():
    def point(pts,tb,offset):
        return dict(pts=pts,time_base=tb,stream_origin_seconds='0',media_offset_seconds=offset,
                    RGB_sha256='a'*64,width=96,height=64)
    frames=[dict(clip=point(0,'1/30','0'),source=point(24724,'1/60','6181/15')),
            dict(clip=point(209,'1/30','209/30'),source=point(25142,'1/60','12571/30'))]
    return dict(schema_version=SCHEMA,source_ref=dict(sha256='b'*64,size_bytes=20),
        clip_ref=dict(sha256='c'*64,size_bytes=10),
        receipt_pins=dict(receipt_sha256='d'*64,frame_readback_sha256='e'*64),
        nominal_source_interval=dict(start_seconds='8241/20',end_seconds='8381/20',basis='synthetic selection'),
        content_boundaries=dict(status='unknown',ranges=[]),event_boundaries=dict(status='unknown',ranges=[]),
        context_padding=dict(status='unknown',before_seconds=None,after_seconds=None),
        export=dict(policy='synthetic/remux'),playback=dict(start_seconds='0',end_seconds='7017/1000',duration_seconds='7017/1000'),
        frames=frames,decoded_frame_count=2,world_clock=dict(status='unknown',epoch_origin=None),
        verification='supplied_not_independently_verified',continuous_time_coverage_proven=False)


def test_one_sixtieth_offset_and_7_017_container_are_recorded_without_rounding():
    x=validate_frame_map(proposal())
    assert x['nominal_source_interval']['end_seconds']=='8381/20'
    first=x['frames'][0]
    assert Fraction(first['source']['media_offset_seconds'])-Fraction(x['nominal_source_interval']['start_seconds'])==Fraction(1,60)
    result=query_frames(x,'209/30','7017/1000')
    assert len(result['mapped_frames'])==1
    assert result['unmapped_playback_extents'][0]['status']=='unknown'
    assert not result['continuous_time_coverage_proven']
    assert result['world_clock']['epoch_origin']is None
    assert x['content_boundaries']['status']=='unknown'


@pytest.mark.parametrize('bad', ['duplicate_clip','reverse_source','pixels','clock','padding','event_claim','business','coverage'])
def test_new_frame_map_rejects_lost_order_forged_truth_and_implicit_context(bad):
    x=proposal()
    if bad=='duplicate_clip':x['frames'][1]['clip']=deepcopy(x['frames'][0]['clip'])
    elif bad=='reverse_source':x['frames'][1]['source']=deepcopy(x['frames'][0]['source']);x['frames'][0]['source']=deepcopy(proposal()['frames'][1]['source'])
    elif bad=='pixels':x['frames'][0]['source']['RGB_sha256']='f'*64
    elif bad=='clock':x['world_clock']['epoch_origin']='2026-10-21T00:00:00Z'
    elif bad=='padding':x['context_padding']=dict(status='observed',before_seconds='3',after_seconds='3')
    elif bad=='event_claim':x['event_boundaries']=dict(status='confirmed',ranges=[])
    elif bad=='business':x['business_fact_confirmed']=True
    else:x['continuous_time_coverage_proven']=True
    with pytest.raises(ValueError):validate_frame_map(x)


def test_content_event_candidates_and_processing_padding_remain_independent():
    x=proposal();boundary=dict(start_seconds='8241/20',end_seconds='8381/20',evidence_frame_indices=[0,1],
        basis='synthetic annotation fixture, not real event truth',evidence_ref='fixture-review/1',
        start_censored=True,end_censored=True,physical_completion_confirmed=False)
    x['content_boundaries']=dict(status='candidate',ranges=[boundary])
    x['event_boundaries']=dict(status='candidate',ranges=[dict(boundary,end_censored=False)])
    x['context_padding']=dict(status='declared_processing_context',before_seconds='3',after_seconds='5',basis='fixture request revision')
    result=validate_frame_map(x)
    assert result['content_boundaries']['ranges'][0]['end_censored']is True
    assert result['event_boundaries']['ranges'][0]['end_censored']is False
    assert result['context_padding']['after_seconds']=='5'
    assert not result['event_boundaries']['ranges'][0]['physical_completion_confirmed']


def test_bounded_frame_count_and_checkpoint_metadata():
    with pytest.raises(ValueError,match='complete decoded'):
        validate_frame_map(proposal(),limits=FrameMapLimits(max_frames=1))
    x=proposal();x['extra']='x'*3000
    with pytest.raises(ValueError,match='checkpoint'):
        validate_frame_map(x,limits=FrameMapLimits(max_document_bytes=1000))


def provider():
    return dict(schema_version='vc-lab-temporal-video-provider/1',enabled=True,revision='fixture-owned-decoder/1',
                decoder_python=decoder_python(),max_input_frames=600,decoder_wait_seconds=30)


@pytest.fixture(scope='module')
def codec(tmp_path_factory):
    import importlib.util
    import os
    if not os.environ.get('VC_LAB_DECODER_PYTHON') and importlib.util.find_spec('av') is None:
        pytest.skip('Native codec verification requires PyAV or explicit VC_LAB_DECODER_PYTHON')
    root=tmp_path_factory.mktemp('frame-map-codec');source=root/'original.mkv';clip=root/'clip.mkv'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-f','lavfi','-i','testsrc2=s=96x64:r=30:d=0.4',
                    '-c:v','ffv1','-threads','1',str(source)],check=True)
    shutil.copyfile(source,clip)  # Generated fixture only; no original media copied.
    frames=[]
    for bgr,meta in decode_frames(source,'0','1',provider()):
        frames.append(dict(pts=meta['pts'],time_base=meta['time_base'],stream_origin_seconds=meta['stream_origin'],
            media_offset_seconds=meta['media_offset'],RGB_sha256=hashlib.sha256(np.ascontiguousarray(bgr[:,:,::-1]).tobytes()).hexdigest(),
            width=bgr.shape[1],height=bgr.shape[0]))
    receipt=dict(exact_complete_clip_RGB_sequence_matches_genuine_original=True,
        source_ref=dict(path=source.name,sha256=sha(source),size_bytes=source.stat().st_size),
        video=dict(path=clip.name,sha256=sha(clip),size_bytes=clip.stat().st_size),
        source_mapping=dict(source_start_ms=0,source_end_ms=400,derived_start_ms=0,derived_end_ms=400),
        existing_clip_probe=dict(metadata=dict(format=dict(start_time='0',duration='0.4'))))
    rp=root/'Receipt.json';fp=root/'Frames.json';rsha=saved(rp,receipt);fsha=saved(fp,dict(clip=frames,original=frames))
    doc=load_pixel_readback(rp,rsha,fp,fsha)
    return SimpleNamespace(root=root,source=source,clip=clip,doc=doc,receipt=rp,frames=fp,rsha=rsha,fsha=fsha)


def test_actual_codec_receipt_pin_and_local_all_frame_RGB_verification(codec):
    assert codec.doc['verification']=='supplied_not_independently_verified'
    assert len(codec.doc['frames'])==12
    proof=verify_local_pixels(codec.doc,codec.source,codec.clip,provider(),SimpleNamespace())
    assert proof.receipt['verification']=='local_pixel_verified'
    assert proof.decoded_clip_frames==12
    assert proof.receipt['whole_clip_frames_verified']
    assert not proof.receipt['world_clock_verified']
    with pytest.raises(Conflict,match='SHA'):
        load_pixel_readback(codec.receipt,'0'*64,codec.frames,codec.fsha)


def test_supplied_flags_and_partial_clip_mapping_cannot_mint_local_proof(codec):
    x=deepcopy(codec.doc);x['frames']=x['frames'][:-1];x['decoded_frame_count']-=1
    with pytest.raises(Conflict,match='Unmapped decoded clip'):
        verify_local_pixels(x,codec.source,codec.clip,provider(),SimpleNamespace())
    y=deepcopy(codec.doc)
    for point in[y['frames'][0]['clip'],y['frames'][0]['source']]:point['RGB_sha256']='e'*64
    with pytest.raises(Conflict,match='Actual local RGB'):
        verify_local_pixels(y,codec.source,codec.clip,provider(),SimpleNamespace())


def test_forged_longer_playback_cannot_become_local_verified(codec):
    x=deepcopy(codec.doc)
    x['playback'].update(end_seconds='1/2',duration_seconds='1/2')
    with pytest.raises(Conflict,match='Actual clip playback'):
        verify_local_pixels(x,codec.source,codec.clip,provider(),SimpleNamespace())


def test_serialized_or_foreign_authority_proof_is_rejected_before_storage_access():
    core=SimpleNamespace(customer=uuid4());maps=FrameMappedClips(SimpleNamespace(core=core,customer=core.customer),None)
    doc=validate_frame_map(proposal())
    for proof in[dict(verification='local_pixel_verified'),LocalPixelProof(fingerprint(doc),'b'*64,'c'*64,2,{},object())]:
        with pytest.raises(ValueError,match='App-owned'):
            maps.register('fake',uuid4(),uuid4(),doc,'fixture-selection',actor='fixture',reason='forgery negative',proof=proof)


@pytest.fixture(scope='session',name='db')
def isolated_db():return evidence_db.__wrapped__()


@pytest.fixture
def settings(core,tmp_path):return replace(evidence_settings.__wrapped__(core,tmp_path),processing_profile='timeline-v1')


def ingest_codec(core,settings,path,name):
    target=settings.incoming/(name+'.mkv');shutil.copyfile(path,target)
    # Ingest accepts video media by kind; cache extension is deliberately mp4.
    # FFV1 Matroska bytes remain unchanged and are identified by their SHA.
    target=target.rename(target.with_suffix('.mp4'))
    return Ingest(core,settings).accept(dict(capture_id=name,source_instance='fixture-frame-camera',source_revision='fixture/1',
        kind='video',relative_path=target.name,source_kind='fixture',session_id='original-session',captured_at=None,
        time_basis=dict(basis='uncalibrated_fixture',uncertainty_us=None),start_seconds='0',end_seconds='0.4',sha256=sha(target)))


def test_actual_PG_registration_relative_unknown_clock_local_proof_and_idempotency(core,settings,codec):
    source=ingest_codec(core,settings,codec.source,'source');clip=ingest_codec(core,settings,codec.clip,'clip')
    maps=FrameMappedClips(Timeline(core),settings)
    document=deepcopy(codec.doc);document['export']['legacy_integral']=2000.0
    proof=verify_local_pixels(document,codec.source,codec.clip,provider(),settings)
    out=maps.register('register',clip['media_id'],source['media_id'],document,'fixture-selection',actor='fixture',reason='new exact frame-map contract',proof=proof)
    assert out['verification']=='local_pixel_verified'and out['world_clock']['epoch_origin']is None
    replay=deepcopy(document);replay['export']['legacy_integral']=2000
    new_proof=verify_local_pixels(replay,codec.source,codec.clip,provider(),settings)
    assert maps.register('register',clip['media_id'],source['media_id'],replay,'fixture-selection',actor='fixture',reason='new exact frame-map contract',proof=new_proof)==out
    assert maps.register('same-mapping-new-command',clip['media_id'],source['media_id'],replay,'fixture-selection',actor='fixture',reason='new exact frame-map contract',proof=new_proof)==out
    with core.db.transaction(readonly=True)as conn:
        assert maps.get(conn,clip['media_id'])==out
        assert maps.validate(conn,core.customer,out)==out
        assert conn.execute('SELECT count(*)AS n FROM media_derivation WHERE customer_id=%s',(core.customer,)).fetchone()['n']==1
    with pytest.raises(Conflict,match='immutable'):
        maps.register('different',clip['media_id'],source['media_id'],document,'changed',actor='fixture',reason='different selection')
    with core.db.transaction()as conn:
        conn.execute('UPDATE media_asset SET version=version+1 WHERE customer_id=%s AND id=%s',(core.customer,source['media_id']))
    with core.db.transaction(readonly=True)as conn:
        with pytest.raises(Conflict,match='superseded'):
            maps.validate(conn,core.customer,out)


def test_existing_v1_derivation_and_foreign_media_never_overwritten(core,settings,codec):
    source=ingest_codec(core,settings,codec.source,'old-source');clip=ingest_codec(core,settings,codec.clip,'old-clip')
    maps=FrameMappedClips(Timeline(core),settings)
    with core.db.transaction()as conn:
        conn.execute('INSERT INTO media_derivation(customer_id,parent_id,child_id,transform)VALUES(%s,%s,%s,%s)',
            (core.customer,source['media_id'],clip['media_id'],Jsonb(dict(schema_version='vc-lab-retained-clip/1',fixture='immutable old generation'))))
    with pytest.raises(Conflict,match='immutable'):
        maps.register('do-not-replace',clip['media_id'],source['media_id'],codec.doc,'fixture-selection',actor='fixture',reason='new map cannot overwrite v1')
    with pytest.raises(LookupError,match='customer'):
        maps.register('foreign',clip['media_id'],uuid4(),codec.doc,'fixture-selection',actor='fixture',reason='cross-customer reference negative')
