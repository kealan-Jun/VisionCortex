"""Real codecs and isolated DB/HTTP fixtures; no real-model quality assertions."""
import base64
from dataclasses import replace
import hashlib
import json
import subprocess

import pytest

from visioncortex.lab.domain import Conflict, Missing
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.native_video import prepare_complete, VideoCapacityExceeded, probe, frame_hashes
from visioncortex.lab.timeline_runtime import TimelineRuntime
from visioncortex.lab.archive_layout import contract_bytes
from .test_timeline import hj as fixture, at, run, activity, DAY
from .test_timeline_runner import fake_service, configured


@pytest.fixture(name='hj')
def context(core, settings):
    return fixture.__wrapped__(core, settings)


def native_model(model=None, **overrides):
    return dict(model or {}, video_input_mode='complete_retained_clip', spatial_policy='preserve_source_pixels',
                processor_resize_verified_disabled=True, max_model_len=8192, max_tokens=128,
                vision_token_geometry=dict(schema_version='qwen-vl-patch-grid/1', patch_size=16,
                                           merge_size=2, temporal_patch_size=2), **overrides)


def policy(hj, settings=None):
    path=hj['settings'].runtime_root/'complete-policy.json'
    path.write_text(json.dumps(dict(schema_version='vc-lab-timeline-policy/2',video_input_unit='complete_retained_clip',
                                    preceding_context_seconds=600)))
    return replace(settings or hj['settings'],timeline_policy_path=path)


def video(hj, name, clock='09:00:00', duration=4, size='160x128'):
    path=hj['settings'].incoming/(name+'.mp4')
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-f','lavfi','-i',
                    f'testsrc2=s={size}:r=10:d={duration}','-c:v','libx264','-threads','1',str(path)],check=True)
    return Ingest(hj['core'],hj['settings']).accept(dict(capture_id=name,source_instance='fixture-camera',
        source_revision='fixture/1',kind='video',session_id='fixture-session',relative_path=path.name,
        captured_at=at(clock),time_basis=dict(basis='synthetic_clock',uncertainty_us=0),start_seconds='0',
        end_seconds=str(duration),source_kind='fixture',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def registered(hj, runtime, *, runs=None):
    parent=video(hj,'parent')
    child=video(hj,'child','09:00:01',1)
    doc=runtime.clips.register('register',child['media_id'],
        [dict(media_id=parent['media_id'],start_seconds='1',end_seconds='2')],
        'existing-fixture-selection',run_ids=runs,actor='fixture',reason='synthetic time mapping only')
    return parent,child,doc


def test_raw_packets_are_not_claimed_as_complete_clips(hj,monkeypatch):
    video(hj,'raw')
    with fake_service('vlm') as (state,model):
        runtime=TimelineRuntime(hj['core'],policy(hj,configured(hj,native_model(model),'video_understanding',monkeypatch)))
        assert runtime.runner.run_once() is None
        assert state['posts']==0


def test_complete_clip_one_request_maps_to_original_and_keeps_experiment(hj,monkeypatch):
    rid=run(hj)
    with fake_service('vlm') as (state,model):
        runtime=TimelineRuntime(hj['core'],policy(hj,configured(hj,native_model(model),'video_understanding',monkeypatch)))
        parent,child,doc=registered(hj,runtime,runs=[rid])
        result=runtime.runner.run_once()
        assert result['status']=='completed', result
        assert state['posts']==1
        with hj['core'].db.transaction(readonly=True) as conn:
            task=conn.execute('SELECT * FROM hj_timeline_task WHERE customer_id=%s AND id=%s',
                             (hj['core'].customer,result['task_id'])).fetchone()
            stages=conn.execute('SELECT * FROM hj_timeline_timing WHERE customer_id=%s', (hj['core'].customer,)).fetchall()
        request=task['request']
        assert len(request['windows'])==1
        assert request['windows'][0]['core']['end_seconds']=='1.000000'
        assert len(request['windows'][0]['input_spans'])==1
        assert request['retained_clip']==doc
        encoded=state['bodies'][0]['messages'][0]['content'][1]['video_url']['url'].split(',',1)[1]
        assert base64.b64decode(encoded)==(hj['settings'].incoming/'child.mp4').read_bytes()
        trace=runtime.views.trace(at('09:00:00'),at('09:00:04'))
        observation=trace['activities'][0]
        assert observation['run_ids']==[rid]
        assert observation['start_at'].startswith('2026-10-27T01:00:01.100000')
        spans={s['media_id']:s for s in observation['evidence']}
        assert spans[parent['media_id']]['start_seconds']=='1.100000'
        assert spans[child['media_id']]['start_seconds']=='0.100000'
        assert {s['stage'] for s in stages}>={'queue_wait','provider_preflight','window_execution','checkpoint_commit','result_commit','attempt_wall'}
        assert all(float(s['elapsed_seconds'])>=0 for s in stages)
        assert not trace['indexed_coverage'][0]['overlaps']


def test_clip_mapping_immutable_and_source_version_pinned(hj):
    runtime=TimelineRuntime(hj['core'],policy(hj))
    parent,child,doc=registered(hj,runtime)
    args=[dict(media_id=parent['media_id'],start_seconds='1',end_seconds='2')]
    with pytest.raises(Conflict,match='immutable'):
        runtime.clips.register('changed',child['media_id'],args,'different-selection',actor='fixture',reason='different')
    with hj['core'].db.transaction() as conn:
        conn.execute('UPDATE media_asset SET version=version+1 WHERE customer_id=%s AND id=%s',(hj['core'].customer,parent['media_id']))
    with pytest.raises(Conflict,match='superseded'):
        runtime.tasks.claim('wrong-version','video_understanding','fixture/1')


def test_mapping_rejects_gap_wrong_clock_and_foreign_media(hj):
    runtime=TimelineRuntime(hj['core'],policy(hj))
    parent=video(hj,'parent')
    child=video(hj,'child','09:00:02',1)
    with pytest.raises(ValueError,match='gaps'):
        runtime.clips.register('bad',child['media_id'],[dict(media_id=parent['media_id'],start_seconds='1',end_seconds='2')],
                               'selection',actor='fixture',reason='wrong clock')
    from uuid import uuid4
    with pytest.raises((Missing,Conflict)):
        runtime.clips.register('foreign',child['media_id'],[dict(media_id=str(uuid4()),start_seconds='0',end_seconds='1')],
                               'selection',actor='fixture',reason='outside scope')


def test_prior_candidates_require_same_explicit_experiment(hj):
    runtime=TimelineRuntime(hj['core'],policy(hj))
    rid,other=run(hj,'A'),run(hj,'B')
    parent,child,_=registered(hj,runtime,runs=[rid])
    one=activity(hj,'same',parent,start=0,end=.4,kind='transfer',run_ids=[rid])
    activity(hj,'unrelated',parent,start=.5,end=.8,kind='transfer',run_ids=[other])
    request=runtime.tasks.claim('claim','video_understanding','fixture/1')['request']
    assert [p['activity_id'] for p in request['preceding_context']]==[one['id']]
    assert request['preceding_context'][0]['evidence_ids']
    assert len(request['windows'][0]['input_spans'])==1


@pytest.mark.parametrize('size,padded', [('160x128',[160,128]),('160x120',[160,128])])
def test_native_input_retains_every_decoded_pixel_and_frame(hj,tmp_path,size,padded):
    video(hj,'native',duration=1,size=size)
    path=hj['settings'].incoming/'native.mp4'
    ref=dict(input_unit='complete_retained_clip',start_seconds='0',end_seconds='1')
    raw,transform=prepare_complete(path,ref,native_model(),tmp_path,16*1024**2)
    out=tmp_path/'result.mp4';out.write_bytes(raw)
    assert transform['padded_size']==padded
    assert frame_hashes(path)==frame_hashes(out,crop=(160,int(size.split('x')[1])))
    assert len(frame_hashes(out))==10
    assert not transform['temporal_sampling'] and not transform['spatial_resize_and_lossy_encoding']
    if size=='160x128':assert raw==path.read_bytes()


def test_capacity_exceeded_is_terminal_visible_and_never_posts(hj,monkeypatch):
    with fake_service('vlm') as (state,model):
        model=native_model(model);model['max_model_len']=4096
        runtime=TimelineRuntime(hj['core'],policy(hj,configured(hj,model,'video_understanding',monkeypatch)))
        parent=video(hj,'large-parent',duration=2,size='1280x800')
        child=video(hj,'large-child','09:00:00',1,size='1280x800')
        runtime.clips.register('large',child['media_id'],[dict(media_id=parent['media_id'],start_seconds='0',end_seconds='1')],
                               'selection',actor='fixture',reason='capacity fixture')
        result=runtime.runner.run_once()
        assert result['status']=='needs_attention' and result['error_code']=='VideoCapacityExceeded'
        assert state['posts']==0 and runtime.runner.run_once() is None
        with hj['core'].db.transaction(readonly=True) as conn:
            row=conn.execute('SELECT attempts FROM hj_timeline_task WHERE customer_id=%s AND id=%s',(hj['core'].customer,result['task_id'])).fetchone()
        assert row['attempts']==1


def test_native_duration_and_payload_overflow_never_truncate(hj,tmp_path):
    video(hj,'native',duration=1)
    path=hj['settings'].incoming/'native.mp4'
    with pytest.raises(ValueError,match='entire'):
        prepare_complete(path,dict(input_unit='complete_retained_clip',start_seconds='0',end_seconds='.5'),native_model(),tmp_path,16*1024**2)
    with pytest.raises(VideoCapacityExceeded):
        prepare_complete(path,dict(input_unit='complete_retained_clip',start_seconds='0',end_seconds='1'),native_model(),tmp_path,1024)
    assert probe(path)['width']==160


def test_metrics_empty_or_partial_batch_has_no_eta_or_zero_measurements(hj):
    runtime=TimelineRuntime(hj['core'],policy(hj))
    before=runtime.metrics.day(DAY)
    assert before['stage_times']['model_http']['total_seconds'] is None
    assert before['complete_batch_wall_seconds'] is None
    video(hj,'raw')
    report=runtime.metrics.day(DAY)
    assert report['awaiting_clip_selection']==1 and report['selected_video_seconds']=='0'
    assert report['remaining_eta_seconds'] is None
    assert all(n['measured_processing_within_budget'] is None and n['acceptance']=='NOT_DEMONSTRATED' for n in report['nightly'])


def test_metrics_cover_http_db_publication_and_do_not_promote_partial_raw_day(hj,monkeypatch):
    root=hj['settings'].incoming
    (root/'.System').mkdir();(root/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    with fake_service('vlm') as (_,model):
        settings=policy(hj,configured(hj,native_model(model),'video_understanding',monkeypatch))
        runtime=TimelineRuntime(hj['core'],replace(settings,source_root=root,unified_archive_required=True,timeline_publish=True))
        registered(hj,runtime)
        assert runtime.runner.run_once()['status']=='completed'
        runtime.views.snapshot(DAY)
        assert runtime.publication.tick()['files']
        metrics=runtime.metrics.day(DAY)
    assert metrics['selected_video_count']==1 and metrics['selected_video_seconds']=='1.000000'
    assert metrics['awaiting_clip_selection']==1
    assert metrics['latest_report_published'] and metrics['all_selected_analysis_tasks_completed']
    assert not metrics['all_known_tasks_completed']
    assert metrics['stage_times']['model_http']['samples']==1
    assert metrics['stage_times']['source_locate_and_verify']['samples']==1
    assert metrics['stage_times']['publication_wall']['samples']==1
    assert metrics['observed_batch_span_seconds'] is not None and metrics['complete_batch_wall_seconds'] is None
    assert set(metrics['publication_attempts'][0]['stages'])=={'write_and_full_readback','publication_commit'}
    assert metrics['quality']=='NOT_PROVEN'


def test_publication_failure_timing_is_not_success(hj,monkeypatch):
    root=hj['settings'].incoming
    (root/'.System').mkdir();(root/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    video(hj,'raw')
    runtime=TimelineRuntime(hj['core'],replace(policy(hj),source_root=root,unified_archive_required=True,timeline_publish=True))
    runtime.views.snapshot(DAY)
    def fail(*args):raise OSError('synthetic output failure')
    monkeypatch.setattr(runtime.publication,'_write',fail)
    with pytest.raises(OSError):runtime.publication.tick()
    metrics=runtime.metrics.day(DAY)
    assert not metrics['latest_report_published']
    assert metrics['publication_attempts'][0]['outcome']=='failed'
    assert metrics['publication_attempts'][0]['error_code']=='OSError'


def test_preceding_context_is_bounded_and_excludes_superseded_evidence(hj):
    runtime=TimelineRuntime(hj['core'],policy(hj))
    rid=run(hj)
    parent,_,_=registered(hj,runtime,runs=[rid])
    long='较长的旧观察'*1000
    hj['timeline'].add_activity('long','transfer','video',long,[dict(media_id=parent['media_id'],start_seconds='0',end_seconds='.2')],
                                run_ids=[rid],model_revision='fixture/1')
    prior=activity(hj,'short',parent,start=.3,end=.5,kind='transfer',run_ids=[rid])
    with hj['core'].db.transaction(readonly=True) as conn:
        m=conn.execute('SELECT * FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s',(hj['core'].customer,parent['media_id'])).fetchone()
        child=conn.execute("SELECT m.* FROM hj_timeline_media m JOIN media_derivation d ON d.customer_id=m.customer_id AND d.child_id=m.media_id WHERE m.customer_id=%s LIMIT 1",(hj['core'].customer,)).fetchone()
        planned=runtime.tasks.plan(conn,child,'video_understanding')
    assert [p['activity_id'] for p in planned['preceding_context']]==[prior['id']]
    assert len(json.dumps(planned['preceding_context'],ensure_ascii=False).encode())<=4096
    # The clip pins its source version, so independently test the context filter
    # against a stale prior from a separate earlier recording.
    earlier=video(hj,'earlier','08:59:58',1)
    activity(hj,'stale',earlier,start=0,end=.5,kind='transfer',run_ids=[rid])
    with hj['core'].db.transaction() as conn:
        conn.execute('UPDATE media_asset SET version=version+1 WHERE customer_id=%s AND id=%s',(hj['core'].customer,earlier['media_id']))
    with hj['core'].db.transaction(readonly=True) as conn:
        assert [p['activity_id'] for p in runtime.tasks.plan(conn,child,'video_understanding')['preceding_context']]==[prior['id']]
    assert m['duration_seconds']==4
