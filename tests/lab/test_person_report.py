"""Person evidence/day boundaries on isolated synthetic PostgreSQL fixtures."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from uuid import uuid4
import wave

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from visioncortex.lab.api import create_app
from visioncortex.lab.domain import Core
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.timeline_report import render_report
import visioncortex.lab.person_report as projection

from .test_operator_scan import scene as operator_scene, scan, context, at
from .test_comment_transition import source, reviewed, apply


@pytest.fixture
def scene(core,settings):
    settings=replace(settings,processing_profile='timeline-v1')
    result=operator_scene.__wrapped__(core,settings)
    result['settings']=settings
    return result


def wearer(s,camera,person,clock,key=None):
    return s['core'].assign_camera_person(key or 'wearer-'+str(uuid4()),s['cameras'][camera],s['people'][person],at(clock),actor='synthetic fixture',reason='explicit historical wearer')


def media(s,name,clock='09:00:00',camera=0,kind='audio',captured=True,basis=None):
    extension={'audio':'.wav','video':'.mp4','photo':'.png'}[kind]
    path=s['settings'].incoming/(name+extension)
    if kind=='audio':
        with wave.open(str(path),'wb') as stream:
            stream.setparams((1,2,8000,0,'NONE','not compressed'));stream.writeframes(b'\0\0'*32000)
    elif kind=='photo':Image.new('RGB',(4,4),'white').save(path)
    else:path.write_bytes(b'synthetic projection fixture; no codec/model invocation')
    return Ingest(s['core'],s['settings']).accept(dict(capture_id=name,source_instance='source-channel-'+str(camera),
        source_revision='synthetic/1',kind=kind,relative_path=path.name,camera_id='fixture-camera-'+str(camera),
        session_id='fixture-session-'+name,captured_at=at(clock) if captured else None,
        time_basis=basis if basis is not None else {'basis':'synthetic_fixture_clock','uncertainty_us':0},
        start_seconds='0',end_seconds='0' if kind=='photo' else '4',source_kind='fixture',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def event(s,key,m,*,run_ids=None,person_id=None,source_type='speech',start='0',end='1',payload=None):
    return s['runtime'].timeline.add_activity(key,'speech' if source_type=='speech' else 'reading' if source_type=='ocr' else 'settings',
        source_type,'synthetic candidate '+key,[dict(media_id=m['media_id'],start_seconds=start,end_seconds=end)],
        run_ids=run_ids,person_id=person_id,model_revision='synthetic-no-inference/1',payload=payload)


def report(s):return s['runtime'].views.snapshot('2026-10-29')
def people(r):return {p['person_id']:p for p in r['document']['person_timelines']['persons']}
def bases(e):return {r['basis'] for r in e['person_attribution']}
def find(p,aid):return next(e for e in p['events'] if e['activity_id']==aid)


def test_A_to_B_active_declarations_keep_batch_material_sources_and_evidence_order(scene,tmp_path):
    s=scene
    a=scan(s,'A',run_id=s['run'])
    material=scan(s,'M',kind='material',value='fixture-reusable-label',when='09:01:00')
    b=scan(s,'B',value=s['devices'][1],when='09:02:00')
    r=report(s);p=people(r)[s['people'][0]]
    assert [e['scan_event_id'] for e in p['events']]==[a['event_id'],material['event_id'],b['event_id']]
    assert all('direct_operator_declaration' in bases(e) for e in p['events'])
    assert p['events'][-1]['equipment_declaration']['from_equipment_id']==s['devices'][0]
    assert p['events'][-1]['equipment_declaration']['to_equipment_id']==s['devices'][1]
    assert p['events'][-1]['target_batch_ids']==[s['batches'][0]]
    assert p['events'][1]['source_lot_ids']==[s['lots'][0]]
    assert not p['events'][-1]['equipment_declaration']['ends_previous_experiment']
    assert '主动操作声明' in render_report(r['document'])
    (tmp_path/'person-day.txt').write_text(render_report(r['document']))


def test_same_person_changes_camera_and_late_media_uses_old_wearer_interval(scene):
    s=scene
    old=wearer(s,0,0,'08:00:00');wearer(s,0,1,'10:00:00');wearer(s,1,0,'08:00:00')
    late=event(s,'late-old',media(s,'late-old','09:00:00',camera=0))
    now=event(s,'new-person',media(s,'new-person','10:01:00',camera=0))
    other=event(s,'same-person-camera1',media(s,'camera1','10:02:00',camera=1))
    p=people(report(s))
    assert {e['activity_id'] for e in p[s['people'][0]]['events']}=={late['id'],other['id']}
    assert {e['activity_id'] for e in p[s['people'][1]]['events']}=={now['id']}
    assert find(p[s['people'][0]],late['id'])['recording_attribution'][0]['binding_id']==old['binding']['id']
    assert all(bases(e)=={'recorded_by'} for row in p.values() for e in row['events'])


def test_speech_explicit_person_run_owner_and_wearer_are_distinct_roles(scene):
    s=scene;wearer(s,0,0,'08:00:00')
    rid=s['runtime'].timeline.create_run('owner-run','fixture-owned-run','fixture',s['materials'],person_id=s['people'][1])['id']
    a=event(s,'speech',media(s,'speech'),run_ids=[rid],person_id=s['people'][1])
    p=people(report(s))
    assert bases(find(p[s['people'][0]],a['id']))=={'recorded_by'}
    spoken=find(p[s['people'][1]],a['id'])
    assert bases(spoken)=={'explicit_activity_person_association','experiment_related_evidence'}
    assert spoken['speaker_person_id'] is None and spoken['operator_person_id'] is None


def test_concurrent_experiments_do_not_duplicate_one_event_or_merge_two_views(scene):
    s=scene
    runs=[s['runtime'].timeline.create_run('run-owner-'+str(i),'fixture-run-'+str(i),'fixture',s['materials'],person_id=s['people'][0])['id'] for i in range(2)]
    first=event(s,'view1',media(s,'view1',kind='video'),run_ids=runs,source_type='video',payload={'retained_clip':{'id':'synthetic-clip'},'original_source_spans':[{'media_id':'synthetic-source-reference'}]})
    second=event(s,'view2',media(s,'view2',kind='video',camera=1),run_ids=runs,source_type='video')
    ocr=event(s,'ocr',media(s,'ocr',kind='photo'),run_ids=runs,source_type='ocr',start='0',end='0',payload={'fields':[{'field_key':'temperature','value':'26.2','status':'candidate'}]})
    r=report(s);p=people(r)[s['people'][0]]
    assert len(p['events'])==3 and {e['activity_id'] for e in p['events']}=={first['id'],second['id'],ocr['id']}
    assert find(p,ocr['id'])['ocr_fields'][0]['status']=='candidate'
    assert all(len(e['run_links'])==2 and bases(e)=={'experiment_related_evidence'} for e in p['events'])
    assert p['physical_action_count'] is None
    assert find(p,first['id'])['retained_clip']['id']=='synthetic-clip'
    assert len(r['document']['experiments'])==2 and all(len(e['activities'])==3 for e in r['document']['experiments'])


def test_unknown_clock_or_person_never_fills_trajectory_or_uses_latest_wearer(scene):
    s=scene;wearer(s,0,0,'08:00:00')
    unknown=event(s,'unknown-clock',media(s,'unknown-clock',captured=False),person_id=s['people'][1])
    uncertain=event(s,'unverified-clock',media(s,'unverified-clock',basis={'basis':'unverified_clock'}))
    unassigned=event(s,'unassigned',media(s,'unassigned',camera=1))
    r=report(s);data=r['document']['person_timelines'];p=people(r)
    assert p[s['people'][1]]['events']==[] and p[s['people'][1]]['untimed_events'][0]['activity_id']==unknown['id']
    assert s['people'][0] not in p
    assert any(e['activity_id']==uncertain['id'] and e['start_at'] is None for e in data['unknown_time_unknown_person_events'])
    assert [e['activity_id'] for e in data['unknown_person_events']]==[unassigned['id']]
    assert data['coverage']['unanalysed_gaps']=='unknown_not_filled' and not data['coverage']['continuous_day_trajectory_known']


def test_evidence_crossing_wearer_boundary_does_not_pick_latest_or_single_full_binding(scene):
    s=scene;wearer(s,0,0,'08:00:00');wearer(s,0,1,'09:00:02')
    a=event(s,'crossing',media(s,'crossing'),end='4')
    data=report(s)['document']['person_timelines']
    assert data['persons']==[]
    e=next(e for e in data['unknown_person_events'] if e['activity_id']==a['id'])
    assert e['recording_attribution'][0]['status']=='wearer_history_ambiguous_missing_or_cross_boundary'


def test_late_reused_material_label_adds_revision_without_rewriting_old_snapshot(scene):
    s=scene;scan(s,'A',run_id=s['run'])
    s['core'].assign_tag('reuse',s['tag'],s['materials'][1],at('10:00:00'),actor='fixture',reason='reuse')
    new=scan(s,'new-material',kind='material',value='fixture-reusable-label',when='10:05:00')
    before=report(s);saved=deepcopy(before)
    old=scan(s,'old-late-material',kind='material',value='fixture-reusable-label',when='09:05:00')
    after=report(s);p=people(after)[s['people'][0]]
    assert before==saved and after['revision']>before['revision'] and after['fingerprint']!=before['fingerprint']
    assert find(p,old['activity_id'])['source_lot_ids']==[s['lots'][0]]
    assert find(p,new['activity_id'])['source_lot_ids']==[s['lots'][1]]
    assert [e['start_at'] for e in p['events']]==sorted(e['start_at'] for e in p['events'])
    with s['core'].db.transaction(readonly=True) as conn:
        retained=conn.execute('SELECT fingerprint FROM hj_daily_report WHERE customer_id=%s AND day=%s AND revision=%s',(s['core'].customer,'2026-10-29',before['revision'])).fetchone()
    assert retained['fingerprint']==before['fingerprint']


def test_person_day_api_auth_isolation_same_snapshot_and_old_report_renderer(scene):
    s=scene;scan(s,'A',run_id=s['run']);snapshot=report(s)
    with TestClient(create_app(s['core'],s['settings'])) as client:
        url='/internal/timeline/person-days/'+s['people'][0]+'/2026-10-29'
        assert client.get(url).status_code==401
        headers={'Authorization':'Bearer '+s['settings'].token}
        result=client.get(url,headers=headers)
        assert result.status_code==200 and result.json()['snapshot_fingerprint']==snapshot['fingerprint']
        assert result.json()['timeline']==people(snapshot)[s['people'][0]]
        other=Core(s['core'].db,Core.customer_instance(s['core'].db,'other-'+str(uuid4()),scope='fixture')['id'])
        foreign=other.register('person','person','fixture-foreign-person')['id']
        assert client.get('/internal/timeline/person-days/'+foreign+'/2026-10-29',headers=headers).status_code==404
    old=deepcopy(snapshot['document']);old.pop('person_timelines')
    assert '实验室记录日报' in render_report(old)
    assert all(k in snapshot['document'] for k in ('experiments','readouts','unassigned_activities','analysis_jobs'))


def test_comment_plan_is_not_arrival_or_speaker_and_B_scan_preserves_run(scene):
    s=scene;scan(s,'A',run_id=s['run'])
    src=source(s,s['settings'])
    done=apply(s,'stage',reviewed(s,src))
    plan=apply(s,'plan',reviewed(s,src,'next_stage_plan'))
    assert context(s)['equipment_id']==s['devices'][0]
    scan(s,'B',value=s['devices'][1],when='09:02:00')
    p=people(report(s))[s['people'][0]]
    for aid in (done['activity_id'],plan['activity_id']):
        e=find(p,aid)
        assert e['equipment_declaration'] is None and not e['planned_stage_is_arrival']
        assert e['speaker_person_id'] is None and e['operator_person_id'] is None
    assert context(s)['run_id']==s['run'] and context(s)['batch_id']==s['batches'][0]


def test_person_projection_budget_is_explicit_partial_not_complete(scene,monkeypatch):
    scan(scene,'A');scan(scene,'B',value=scene['devices'][1],when='09:01:00')
    monkeypatch.setattr(projection,'MAX_EVENTS',1)
    p=report(scene)['document']['person_timelines']
    assert p['coverage']['events_truncated'] and p['coverage']['status']=='partial'
    assert not p['coverage']['raw_recording_coverage_is_model_analysis_coverage']
