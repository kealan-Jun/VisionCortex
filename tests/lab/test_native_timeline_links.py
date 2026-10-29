"""Native pure-builder/sole-writer bridge; synthetic local evidence, no models."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from visioncortex.lab.components import ROOT
from visioncortex.lab.native_comment_layout import module as original
from visioncortex.lab.native_linked_layout import module, bind_source
from visioncortex.lab.native_timeline_links import (association_path, association_revision, attach_day,
    attach_segment, build_sidecar, MAX_BYTES)
from visioncortex.lab.rules import Rules
from visioncortex.lab.timeline_publication import TimelinePublication


@pytest.fixture
def fixture(tmp_path):
    archive='2026-10-29_fixture-camera';root=tmp_path/archive;root.mkdir()
    start=int(datetime.fromisoformat('2026-10-29T09:00:00+08:00').timestamp()*1e6)
    segment=dict(segment_id='fixture-segment',recording_id='fixture-recording',activity='inactive',
        activity_label='synthetic pending',start_ms=0,end_ms=4000,start_us=start,end_us=start+4000000,
        source_ref=dict(path='MetaData/fixture.mp4',sha256='a'*64,size_bytes=123),key_frames=[],scene_frames=[],
        json_path='ProcessedClips/Clips/fixture/NoExperimentActivity.json')
    record=dict(recording_id=segment['recording_id'],start_us=start,end_us=start+4000000,
        camera_key=archive[11:],stages={'understanding':{'status':'pending'}},sources=[])
    index=dict(archive=archive,recordings=[record],segments=[segment],understandings=[],context={})
    return dict(root=root,archive=archive,index=index,record=record,segment=segment,customer=str(uuid4()),
        activity=dict(id=str(uuid4()),kind='settings',source_type='reviewed_note',start_at=None,end_at=None,
            payload={'operator_scan_event':str(uuid4())},evidence=[dict(id=str(uuid4()),media_id=str(uuid4()),
                verified_sha256='a'*64,start_seconds='1',end_seconds='2')]),rules=Rules().identity())


def publish(f,revision=1,*,camera=None,activity=None,fingerprint=None):
    archive=f['archive'] if camera is None else '2026-10-29_'+camera
    value=build_sidecar(archive=archive,customer_id=f['customer'],revision=revision,
        snapshot_fingerprint=fingerprint or str(revision)*64,rules=f['rules'],
        canonical_report=f['archive']+'/LaboratoryDailyReport/Timeline/report.json',
        activities=[activity or f['activity']],person_projection={},media_sources={f['activity']['evidence'][0]['media_id']:{'source_relative_path':archive+'/MetaData/fixture.mp4','source_sha256':'a'*64}})
    path=f['root'].parent/association_path(archive,revision,value['snapshot_fingerprint'],f['rules']['sha256'])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False))
    return path,value


def day(f,loader=module):
    return loader('device_day_reports').day_document(SimpleNamespace(root=f['root'],name=f['archive']),f['index'],
        {'status':'pending','findings':[]},readable={'视频留存统计':{},'全天时间线':[]},entries=[],report_generated_at='fixture-clock')


def segment(f,loader=module):
    return loader('device_day_content').segment_documents(f['index'],f['root'],f['segment'],None,None,f['record'],[])


def test_no_sidecar_keeps_native_builder_fields_and_serialized_bytes(fixture):
    f=fixture
    assert json.dumps(day(f),ensure_ascii=False)==json.dumps(day(f,original),ensure_ascii=False)
    assert segment(f)==segment(f,original)
    config={'storage':{'archive_root':str(f['root'].parent)}}
    assert module('device_day_content').content_revision(config,f['index'])==original('device_day_content').content_revision(config,f['index'])


def test_linked_main_preserves_actual_pending_understanding_and_all_old_fields(fixture):
    f=fixture;old_day=day(f);old_segment,old_retention=segment(f)
    path,_=publish(f);new_day=day(f);new_segment,new_retention=segment(f)
    assert {k:v for k,v in new_day.items() if k!='关联时间线'}==old_day
    assert {k:v for k,v in new_segment.items() if k!='关联时间线'}==old_segment
    assert new_retention==old_retention and new_segment['status']=='pending'
    assert new_segment['experiment_steps'] is None and new_segment['understanding'] is None and new_segment['video'] is None
    assert new_day['全天综合理解']=={'status':'pending','findings':[]}
    ref=new_segment['关联时间线'];assert ref['segment_evidence_links'][0]['activity_id']==f['activity']['id']
    assert ref['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert not ref['model_steps_generated'] and not ref['findings_generated'] and not ref['clip_created']
    assert ref['cross_person_projection']['scope']=='customer_all_streams_not_camera_narrative'


def test_late_revision_changes_original_trigger_and_preserves_r1(fixture):
    f=fixture;config={'storage':{'archive_root':str(f['root'].parent)}}
    first,_=publish(f);raw=first.read_bytes();before=module('device_day_content').content_revision(config,f['index'])
    after_activity=dict(f['activity'],id=str(uuid4()));publish(f,2,activity=after_activity)
    after=module('device_day_content').content_revision(config,f['index'])
    assert before!=after and first.read_bytes()==raw
    assert day(f)['关联时间线']['snapshot_revision']==2
    assert segment(f)[0]['关联时间线']['segment_evidence_links'][0]['activity_id']==after_activity['id']


def test_camera_isolation_and_unknown_source_do_not_attach_nearby_observation(fixture):
    f=fixture;publish(f,camera='other-camera')
    assert '关联时间线' not in day(f) and association_revision(f['root'],f['archive'])==()
    unknown=deepcopy(f['activity']);unknown['evidence'][0].pop('verified_sha256')
    publish(f,activity=unknown);value=segment(f)[0]
    assert value['关联时间线']['segment_evidence_links']==[]
    assert value['关联时间线']['segment_association_status']=='pending_no_exact_source_evidence'
    assert value['status']=='pending'


def test_invalid_sidecar_pin_conflict_size_and_link_rejected_before_main_write(fixture):
    f=fixture;path,value=publish(f);value['camera_key']='wrong-camera';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='pin mismatch'):day(f)
    path.unlink();path,value=publish(f)
    other=path.with_name(path.name.replace('1'*16,'b'*16));other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match='Conflicting'):day(f)
    other.unlink();path.write_bytes(b' '*(MAX_BYTES+1))
    with pytest.raises(ValueError,match='bound'):day(f)
    path.unlink();target=path.parent/'target';target.write_text('{}');path.symlink_to(target)
    with pytest.raises(OSError):day(f)


def test_only_existing_native_writer_writes_main_and_hook_drift_is_rejected(fixture,monkeypatch):
    f=fixture;publish(f)
    reports=module('device_day_reports');content=module('device_day_content')
    layout=SimpleNamespace(root=f['root'],name=f['archive'],reports=f['root']/'LaboratoryDailyReport')
    layout.reports.mkdir(exist_ok=True)
    calls=[]
    original_write=reports.atomic_json
    def write(path,value):calls.append(Path(path));return original_write(path,value)
    monkeypatch.setattr(reports,'atomic_json',write)
    narrative=module('device_day_narrative');synthesis=module('device_day_synthesis')
    monkeypatch.setattr(narrative,'day_record',lambda *a:{'视频留存统计':{}})
    monkeypatch.setattr(narrative,'report_candidates',lambda *a:[])
    monkeypatch.setattr(synthesis,'current_synthesis',lambda *a:{'status':'pending'})
    monkeypatch.setattr(reports,'preserve_prior_synthesis',lambda *a:None)
    reports.render_day(layout,f['index'],html_enabled=False)
    assert calls==[layout.reports/'LaboratoryDailyReport.json']
    value,_=segment(f);path=f['root']/'MultimodalUnderstanding/fixture/Understanding.json'
    content._write(path,value)
    assert json.loads(path.read_text())['关联时间线']['snapshot_revision']==1
    before=path.read_bytes()
    attach_day({},f['root'],f['archive']);attach_segment({},f['root'],f['index'],f['segment'])
    assert path.read_bytes()==before  # helpers are read-only; original _write owns main bytes.
    with pytest.raises(ValueError,match='no longer matches'):bind_source('device_day_reports','upstream drift')
    source=ROOT/'components/lab/visioncortex_pipeline/src/visioncortex_lab_pipeline/device_day_reports.py'
    assert 'native_timeline_links' not in source.read_text()
    assert 'native_timeline_links' not in (ROOT/'src/visioncortex/lab/native_comment_layout.py').read_text()


def test_timeline_publication_adds_only_sidecar_not_native_main(fixture):
    f=fixture;path,_=publish(f)
    # Same immutable writer path used by TimelinePublication; an existing identical sidecar is idempotent.
    guard=SimpleNamespace(guarded_atomic=lambda *a:False)
    relative=path.relative_to(f['root'].parent).as_posix()
    receipt=TimelinePublication._write(f['root'].parent,relative,path.read_bytes(),guard)
    assert receipt['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (f['root']/'LaboratoryDailyReport/LaboratoryDailyReport.json').exists()
    assert not list(f['root'].glob('MultimodalUnderstanding/*/Understanding.json'))


def test_nas_reader_without_guard_or_changed_mount_never_falls_back_to_local(fixture,monkeypatch):
    f=fixture;publish(f)
    import visioncortex.lab.native_timeline_links as links
    monkeypatch.setattr(links,'is_nas',lambda _:True)
    guard_module=module('device_day_write_guard')
    token=guard_module.CURRENT.set(None)
    try:
        with pytest.raises(OSError,match='current pinned guard'):day(f)
    finally:guard_module.CURRENT.reset(token)
    calls=[]
    class ChangedMount:
        def validate(self,**kwargs):calls.append('validate');raise OSError('synthetic mount changed')
        def mount_for(self,*a):raise AssertionError('must fail before path traversal')
    token=guard_module.CURRENT.set(ChangedMount())
    try:
        with pytest.raises(OSError,match='mount changed'):day(f)
    finally:guard_module.CURRENT.reset(token)
    assert calls==['validate']


def test_nas_reader_uses_pinned_fd_and_revalidates_without_creating_missing_directories(fixture,monkeypatch):
    f=fixture;publish(f)
    import os
    import visioncortex.lab.native_timeline_links as links
    monkeypatch.setattr(links,'is_nas',lambda _:True)
    descriptor=os.open(f['root'].parent,os.O_RDONLY|os.O_DIRECTORY)
    calls=[]
    class PinnedFixture:
        def validate(self,**kwargs):calls.append(kwargs)
        def mount_for(self,path):return SimpleNamespace(path=f['root'].parent,descriptor=descriptor,snapshot={'device':os.fstat(descriptor).st_dev}),path
    guard_module=module('device_day_write_guard');token=guard_module.CURRENT.set(PinnedFixture())
    try:
        assert day(f)['关联时间线']['snapshot_revision']==1
        assert calls[0]=={} and calls[-1]=={} and len(calls)>=7
        missing=f['root'].parent/'2026-10-29_missing-camera'
        assert attach_day({},missing,missing.name)=={}
        assert not missing.exists()
    finally:guard_module.CURRENT.reset(token);os.close(descriptor)


def test_native_dot_camera_archive_without_sidecar_preserves_original_builder(fixture):
    f=fixture
    f['archive']='2026-10-29_fixture.camera-v1'
    f['root']=f['root'].parent/f['archive'];f['root'].mkdir()
    f['index']['archive']=f['archive'];f['record']['camera_key']=f['archive'][11:]
    assert json.dumps(day(f),ensure_ascii=False)==json.dumps(day(f,original),ensure_ascii=False)
    assert association_revision(f['root'],f['archive'])==()
