from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import cv2
import pytest

from visioncortex.lab.domain import Conflict
from visioncortex.lab.live import ReceiverService, ReceiverSupervisor, load_sources
from visioncortex.lab.processing import ModelRegistry, Worker
from visioncortex.lab.retention import Retention


def test_wearer_rotation_uses_capture_time_and_preserves_unknown_gaps(core,settings,photo):
    camera=core.register('cam','capture_device','camera-one',{'view_role':'first_person'})
    first=core.register('p1','person','first')
    second=core.register('p2','person','second')
    core.assign_camera_person('wear1',camera['id'],first['id'],'2026-10-23T10:00:00+08:00',actor='fixture',reason='start')
    core.assign_camera_person('wear2',camera['id'],second['id'],'2026-10-23T11:00:00+08:00',actor='fixture',reason='rotation')
    core.assign_camera_person('remove',camera['id'],None,'2026-10-23T12:00:00+08:00',actor='fixture',reason='charging')
    assert core.camera_context('camera-one','2026-10-23T10:58:00+08:00')['wearer_id']==first['id']
    assert core.camera_context('camera-one','2026-10-23T11:00:00+08:00')['wearer_id']==second['id']
    assert core.camera_context('camera-one','2026-10-23T12:00:00+08:00')['wearer_id'] is None
    assert core.camera_context('camera-one',None)['wearer_id'] is None
    with pytest.raises(Conflict):
        core.assign_camera_person('backdated',camera['id'],first['id'],'2026-10-23T11:30:00+08:00',actor='fixture',reason='invalid')
    fixed=core.register('fixed','capture_device','fixed-one',{'view_role':'third_person'})
    with pytest.raises(Conflict):
        core.assign_camera_person('fixed-wearer',fixed['id'],first['id'],'2026-10-23T10:00:00+08:00',actor='fixture',reason='invalid')
    device=core.register('device','instrument','balance',{'field_recognition':{'model':'fixture'}})
    tag=core.register('tag','tag','Q-M001',{'mode':'permanent'})
    core.assign_tag('device-tag',tag['id'],device['id'],'2026-10-23T09:00:00+08:00',actor='fixture',reason='fixture')
    snapshots=Worker(core,settings).photo_bindings(settings.incoming/photo['relative_path'],'2026-10-23T10:58:00+08:00','camera-one')
    assert snapshots[0]['wearer_id']==first['id']


def test_live_sources_require_explicit_customer_devices_and_model(core,settings):
    assert ReceiverSupervisor(core,settings).tick()==dict(status='unconfigured',sources=0)
    source=core.register('cam','capture_device','camera-one',{'view_role':'first_person'})
    assert source
    with core.db.transaction(readonly=True) as conn:
        code=conn.execute('SELECT code FROM customer_instance WHERE id=%s',(core.customer,)).fetchone()['code']
    path=settings.runtime_root/'live.json'
    value=dict(schema_version='vc-lab-live-sources/1',customer_code=code,sources=[dict(camera_id='camera-one',receiver_url='http://receiver.invalid')])
    path.write_text(json.dumps(value))
    live=replace(settings,live_sources_manifest=path)
    assert ReceiverSupervisor(core,live).tick()==dict(status='blocked_model',sources=1)
    assert load_sources(core,path)[0]['view_role']=='first_person'
    value['sources'][0]['camera_id']='unknown'
    path.write_text(json.dumps(value))
    with pytest.raises(Conflict):load_sources(core,path)


def test_original_live_event_selection_reuses_frame_ocr_and_retains_video_layout(core,settings,photo,monkeypatch):
    core.register('cam','capture_device','camera-one',{'view_role':'first_person'})
    instrument=core.register('instrument','instrument','balance',{'name':'fixture balance','field_recognition':{'model':'fixture','camera_ids':['camera-one']}})
    tag=core.register('tag','tag','Q-M001',{'mode':'permanent'})
    core.assign_tag('tag-device',tag['id'],instrument['id'],'2026-10-23T09:00:00+08:00',actor='fixture',reason='fixture')
    model=dict(revision='fixture-resident',model_root=str(settings.runtime_root/'models'))
    monkeypatch.setattr(ModelRegistry,'descriptor',lambda *_:model)
    monkeypatch.setattr(ModelRegistry,'verified',lambda *_:model)
    def no_inference(*a,**kw):raise AssertionError('The exact frame OCR must be reused')
    monkeypatch.setattr('visioncortex.lab.processing.run_bounded',no_inference)
    detector=SimpleNamespace(enabled=lambda:False,snapshot=lambda:{},close=lambda:None)
    def factory(*_):
        return dict(panel_detector=detector,ocr_state={'status':'ready'},now=lambda:datetime.now(timezone.utc).isoformat())
    camera=lambda endpoint,target:SimpleNamespace(target=target,close=lambda:None)
    service=ReceiverService(core,settings,[dict(camera_id='camera-one',receiver_url='http://fixture.invalid',view_role='first_person')],model,camera_factory=camera,runtime_factory=factory)
    reader=service.readers[0]
    clock=[100.0]
    reader.clock=lambda:clock[0]
    frame=cv2.imread(str(settings.incoming/photo['relative_path']))
    at='2026-10-23T10:58:00+08:00'
    metadata=dict(camera_id='camera-one',sender_id='fixture',sequence=1,timestamp_us=1,global_timestamp_us=int(datetime.fromisoformat(at).timestamp()*1e6),clock_sync_valid=True)
    local=dict(status='completed',lines=[dict(text='5.00 g',confidence=.99)])
    try:
        reader.accept(frame,metadata,at,deepcopy(local))
        assert reader.state['evidence_saved']==1
        clock[0]+=1
        local['lines'][0]['text']='6.00 g'
        reader.accept(frame,dict(metadata,sequence=2),at,deepcopy(local))
        assert reader.state['evidence_saved']==1  # original two-second minimum
        clock[0]+=2
        reader.accept(frame,dict(metadata,sequence=3),at,deepcopy(local))
        assert reader.state['evidence_saved']==2
        # Process restart or variable decode age cannot create another capture of
        # the same physical frame or rewrite the original binding-at-capture.
        clock[0]+=3
        reader.last_saved_signature=None
        reader.accept(frame,dict(metadata,sequence=3,frame_age_ms=900),at,deepcopy(local))
        worker=Worker(core,settings)
        results=[worker.run_once() for _ in range(4)]
        assert all(r['status']=='completed' for r in results),results
        with core.db.transaction(readonly=True) as conn:
            jobs=conn.execute("SELECT result FROM processing_job WHERE customer_id=%s AND worker_kind='ocr'",(core.customer,)).fetchall()
            assert len(jobs)==2 and all(r['result']['resident_frame_ocr_reused'] for r in jobs)
            for job in jobs:
                doc=job['result']['field_document']
                assert all(r['reading_id'].startswith(doc['job_id']+':') and r['capture_id']==doc['capture_id'] for r in doc['readings'])
            assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',(core.customer,)).fetchone()['n']==0
        retention=Retention(core,settings)
        for _ in range(10):retention.tick()
        evidence=list(settings.incoming.glob('*/Comment/OCR/VideoReadings/*/Evidence.json'))
        assert len(evidence)==2
        assert all(json.loads(p.read_text())['category']=='VideoReadings' for p in evidence)
        assert not list(settings.incoming.rglob('Result*.json')) # whole-frame digits are unlocalized
        assert len(list(settings.incoming.glob('*/MetaData/VideoEvidence/*/Frame.png')))==2
    finally:
        service.close()
