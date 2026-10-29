"""Model-free QR queue on synthetic original codecs and the owned Unix fixture DB."""
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import subprocess
import sys
import time
from uuid import UUID, uuid4

import numpy as np
import pytest
import qrcode

from visioncortex.lab.domain import Conflict, Core
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.qr_video import REVISION, VideoQR, VideoQRRejected, validate_registration
from visioncortex.lab.timeline_commands import command_methods
from visioncortex.lab.timeline_runtime import TimelineRuntime
from ._fixtures import db as evidence_db, settings as evidence_settings


@pytest.fixture(scope='session',name='db')
def fixture_db():
    return evidence_db.__wrapped__()


@pytest.fixture
def settings(core,tmp_path):
    return replace(evidence_settings.__wrapped__(core,tmp_path),processing_profile='timeline-v1')


def original(core,settings,*,rotation=0,clock=True,name='native',camera='fixture-camera',frames=12):
    value = core.issue_equipment('reserved:'+name,actor='synthetic fixture')['qr_payload']
    qr = np.asarray(qrcode.make(value).convert('RGB'))[:,:,::-1].copy()
    h,w = qr.shape[:2]
    image = np.full((max(384,h+20),max(512,w+20),3),210,np.uint8)
    image[10:h+10,10:w+10] = qr
    raws = []
    for index in range(frames):
        copy = image.copy();copy[-8:,-8:] = index
        raws.append(np.ascontiguousarray(copy[::-1,::-1] if rotation else copy).tobytes())
    path = settings.incoming/'MetaData'/name/'rgb.mp4'
    path.parent.mkdir(parents=True)
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','bgr24',
                    '-s',f'{image.shape[1]}x{image.shape[0]}','-r','20','-i','pipe:0',
                    '-vf',r'settb=1/1000,setpts=200+N*50+mod(N\,2)*10',
                    '-vsync','0','-enc_time_base','1/1000','-c:v','libx264rgb','-crf','0',
                    '-preset','ultrafast','-video_track_timescale','1000',str(path)],
                   input=b''.join(raws),check=True)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    duration = str(Decimal(frames) / 20)
    receipt = dict(capture_id=name,source_instance=camera,source_revision='fixture/1',
                   camera_id=camera,session_id='fixture-session',kind='video',
                   relative_path=str(path.relative_to(settings.incoming)),
                   captured_at='2026-10-29T09:00:00+08:00',
                   time_basis={'basis':'synthetic_fixture_clock','uncertainty_us':0 if clock else 1000,
                               'clock_sync_valid':clock},source_kind='fixture',sha256=sha,
                   start_seconds='0',end_seconds=duration,native_record={'fixture_only':True})
    admitted = Ingest(core,settings).accept(receipt,native_original=True)
    registration = dict(schema_version='vc-lab-video-qr/1',enabled=True,revision='fixture-qr/1',
        source_sha256=sha,media_version=1,source_stream_key=camera,source_session_key='fixture-session',
        rotation_degrees_clockwise=rotation,orientation_basis='synthetic native inversion fixture',
        time_basis='original_media_pts',world_clock_status='verified' if clock else 'uncalibrated',
        source_intervals=[dict(start_seconds='0',end_seconds=duration)])
    return dict(path=path,sha=sha,value=value,receipt=receipt,admitted=admitted,registration=registration)


def register(runtime,sample,key='register'):
    return command_methods(runtime)['timeline-register-video-qr'](
        key,sample['admitted']['media_id'],sample['registration'],actor='synthetic fixture',reason='bounded generated codec')


def rows(core,table):
    with core.db.transaction(readonly=True) as conn:
        return conn.execute(f'SELECT * FROM {table} WHERE customer_id=%s',(core.customer,)).fetchall()


def forbid_provider(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('QR-only task must never read a model provider or call OCR')
    monkeypatch.setattr('visioncortex.lab.local_provider.LocalProviders.load',forbidden)
    monkeypatch.setattr('visioncortex.lab.local_provider.LocalHTTP.__init__',forbidden)


@pytest.mark.parametrize('rotation',[0,180])
def test_actual_native_qr_command_queue_frames_checkpoint_trace_and_report(core,settings,monkeypatch,rotation):
    sample = original(core,settings,rotation=rotation)
    before = sample['path'].read_bytes()
    runtime = TimelineRuntime(core,settings)
    assert all(t['kind']!='qr' for t in rows(core,'hj_timeline_task'))  # no default scheduling
    task = register(runtime,sample)
    forbid_provider(monkeypatch)
    result = runtime.runner.run_once()
    assert result['status']=='completed' and result['task_id']==task['task_id']
    task_row = next(r for r in rows(core,'hj_timeline_task') if str(r['id'])==task['task_id'])
    assert task_row['kind']=='qr' and task_row['model_revision']==REVISION
    assert task_row['result']['execution_verification']=='local_model_free_qr_pixels_verified'
    checkpoint = rows(core,'hj_timeline_window_result')[0]
    receipt = checkpoint['execution_receipt']['invocations'][0]['video_qr']
    assert receipt['model_invoked'] is False and receipt['qr_decode_calls']==12
    assert receipt['original_decoded_frame_count']==12
    assert receipt['coverage']['whole_media_completed'] is False
    assert receipt['source_sha_verified_before_and_after'] is True
    assert receipt['peak_reserved_buffer_bytes']<=64*1024**2
    assert len({p['frame_event_id'] for p in receipt['original_frames']})==12
    assert [p['pts'] for p in receipt['original_frames']]==sorted(p['pts'] for p in receipt['original_frames'])
    assert Fraction(receipt['original_frames'][0]['stream_origin'])>0
    assert all(any(q['value']==sample['value'] for q in p['qr_occurrences']) for p in receipt['original_frames'])
    assert all(p['identity_status']=='unknown_no_verified_panel_link' for p in receipt['original_frames'])
    assert all(p['original_rgb_sha256']!=p['upright_rgb_sha256'] for p in receipt['original_frames']) if rotation else all(p['original_rgb_sha256']==p['upright_rgb_sha256'] for p in receipt['original_frames'])
    assert sample['path'].read_bytes()==before
    assert len(rows(core,'hj_dynamic_frame_event'))==12
    assert len(rows(core,'hj_activity'))==1
    activity = rows(core,'hj_activity')[0]
    assert activity['kind']=='unknown' and activity['source_type']=='video'
    assert activity['instrument_id'] is activity['sample_id'] is activity['person_id'] is None
    trace = runtime.views.trace('2026-10-29T08:59:00+08:00','2026-10-29T09:01:00+08:00')
    assert sample['value'] in str(trace)
    runtime.views.refresh_dirty()
    report = runtime.views.snapshot('2026-10-29')
    assert sample['value'] in str(report)
    for table in ['field_candidate','fact_revision','quantity_ledger_entry','operation']:
        assert rows(core,table)==[]
    assert rows(core,'equipment_identity')[0]['state']=='reserved'


def test_unknown_source_clock_never_gets_world_time_or_latest_identity(core,settings,monkeypatch):
    sample = original(core,settings,clock=False)
    runtime = TimelineRuntime(core,settings);register(runtime,sample)
    forbid_provider(monkeypatch)
    assert runtime.runner.run_once()['status']=='completed'
    assert all(r['event_at'] is None for r in rows(core,'hj_dynamic_frame_event'))
    activity = rows(core,'hj_activity')[0]
    assert activity['start_at'] is activity['end_at'] is None
    assert activity['payload']['video_qr']['world_clock_status']=='uncalibrated'
    assert all(p['event_at'] is None for p in activity['payload']['video_qr']['original_frames'])


@pytest.mark.parametrize('change',[
    {'source_stream_key':'other-camera'},{'source_session_key':'other-session'},
    {'source_sha256':'a'*64},{'media_version':2},{'world_clock_status':'uncalibrated'},
    {'rotation_degrees_clockwise':90},{'fields':[{'value':'1'}]},
    {'decoder_python':'/arbitrary/python'},
    {'source_intervals':[{'start_seconds':'0','end_seconds':'10.01'}]},
])
def test_registration_fail_closed_on_source_scope_clock_fields_and_executable(core,settings,change):
    sample = original(core,settings)
    sample['registration'].update(change)
    runtime = TimelineRuntime(core,settings)
    with pytest.raises((VideoQRRejected,Conflict)):
        register(runtime,sample)
    assert all(r['kind']!='qr' for r in rows(core,'hj_timeline_task'))
    assert rows(core,'hj_dynamic_frame_event')==[]


def test_public_submit_cannot_replace_local_qr_pixels_with_caller_verified_flags(core,settings):
    sample = original(core,settings);runtime = TimelineRuntime(core,settings);register(runtime,sample)
    job = runtime.tasks.claim('claim','qr',REVISION)
    result = dict(schema_version='vc-lab-timeline-model-result/1',input_fingerprint=job['request']['input_fingerprint'],
                  covered_window_indices=[0],events=[dict(kind='reading',text='fake verified',fields=[dict(value='123')])],verified=True)
    with pytest.raises(Conflict,match='stored local pixel'):
        runtime.tasks.submit('forged',job['task_id'],job['lease_token'],REVISION,result)
    assert rows(core,'hj_activity')==rows(core,'field_candidate')==rows(core,'quantity_ledger_entry')==[]


def test_retry_reuses_completed_window_without_decoding_or_repeating_raw_events(core,settings,monkeypatch):
    sample = original(core,settings);runtime = TimelineRuntime(core,settings);task=register(runtime,sample)
    submit = runtime.tasks._submit
    def fail_once(*args,**kwargs):
        raise RuntimeError('synthetic interruption after committed checkpoint')
    monkeypatch.setattr(runtime.tasks,'_submit',fail_once)
    forbid_provider(monkeypatch)
    assert runtime.runner.run_once()['status']=='awaiting_provider'
    assert len(rows(core,'hj_timeline_window_result'))==1
    before = rows(core,'hj_dynamic_frame_event')
    monkeypatch.setattr(runtime.tasks,'_submit',submit)
    def no_decode(*args,**kwargs):
        raise AssertionError('Completed native window must not be decoded again')
    monkeypatch.setattr(VideoQR,'run',no_decode)
    with core.db.transaction() as conn:
        conn.execute('UPDATE hj_timeline_task SET next_attempt_at=now() WHERE customer_id=%s AND id=%s',(core.customer,UUID(task['task_id'])))
    assert runtime.runner.run_once()['status']=='completed'
    assert rows(core,'hj_dynamic_frame_event')==before
    assert len(rows(core,'hj_timeline_window_result'))==len(rows(core,'hj_activity'))==1


def test_source_change_after_actual_decode_keeps_prefix_unconfirmed_and_task_attention(core,settings,monkeypatch):
    from visioncortex.lab import qr_video
    sample = original(core,settings);runtime = TimelineRuntime(core,settings);task=register(runtime,sample)
    decode = qr_video.decode_frames
    def mutate_after_decode(*args,**kwargs):
        yield from decode(*args,**kwargs)
        with sample['path'].open('ab') as stream:
            stream.write(b'synthetic fixture source mutation')
    monkeypatch.setattr(qr_video,'decode_frames',mutate_after_decode)
    forbid_provider(monkeypatch)
    result = runtime.runner.run_once()
    assert result['status']=='needs_attention'
    assert next(r for r in rows(core,'hj_timeline_task') if str(r['id'])==task['task_id'])['attempts']==1
    assert rows(core,'hj_activity')==rows(core,'hj_timeline_window_result')==[]
    assert all(not e['payload']['tracks'] for e in rows(core,'hj_dynamic_frame_event'))
    assert rows(core,'fact_revision')==rows(core,'quantity_ledger_entry')==[]


def test_expired_deadline_never_decodes_or_appends(core,settings,monkeypatch):
    from visioncortex.lab import qr_video
    sample = original(core,settings)
    def no_decode(*args,**kwargs):
        raise AssertionError('Expired task must not start decoder')
    monkeypatch.setattr(qr_video,'decode_frames',no_decode)
    ref = dict(media_id=sample['admitted']['media_id'],media_version=1,sha256=sample['sha'],start_seconds='0',end_seconds='0.6')
    with pytest.raises(VideoQRRejected,match='deadline'):
        VideoQR(core,settings).run(sample['path'],ref,sample['registration'],deadline=time.monotonic()-1)
    assert rows(core,'hj_dynamic_frame_event')==[]


def test_other_customer_cannot_register_existing_video(core,settings):
    sample=original(core,settings)
    other=Core(core.db,Core.customer_instance(core.db,'qr-other-'+str(uuid4()),scope='fixture')['id'])
    from visioncortex.lab.domain import Missing
    with pytest.raises(Missing):
        TimelineRuntime(other,settings).tasks.register_video_qr('foreign',sample['admitted']['media_id'],sample['registration'],actor='fixture',reason='cross customer negative')


def test_task_frame_and_total_duration_budgets_are_explicit():
    doc=dict(schema_version='vc-lab-video-qr/1',enabled=True,revision='limits',source_sha256='a'*64,media_version=1,
             source_stream_key='camera',source_session_key='session',rotation_degrees_clockwise=0,
             orientation_basis='fixture',time_basis='original_media_pts',world_clock_status='uncalibrated',
             source_intervals=[dict(start_seconds='0',end_seconds='6'),dict(start_seconds='8',end_seconds='14')])
    with pytest.raises(VideoQRRejected,match='ten explicit'):
        validate_registration(doc)
    doc['source_intervals']=[dict(start_seconds='0',end_seconds='1')]
    with pytest.raises(VideoQRRejected,match='budget exhausted'):
        VideoQR(None,None).run(None,{},doc,max_frames=0)


def test_owned_decoder_absolute_deadline_terminates_only_its_child(monkeypatch):
    from visioncortex.lab import qr_video
    monkeypatch.setenv('VC_LAB_DECODER_PYTHON', sys.executable)
    monkeypatch.setattr(qr_video,'_DECODER','import time; time.sleep(10)')
    start=time.monotonic()
    with pytest.raises(VideoQRRejected,match='deadline'):
        list(qr_video.decode_frames('unused-fixture-path','0','1',
             dict(max_input_frames=600,deadline_monotonic=start+.15)))
    assert time.monotonic()-start<2


def test_decoder_requires_explicit_local_interpreter(monkeypatch):
    from visioncortex.lab import qr_video
    monkeypatch.delenv('VC_LAB_DECODER_PYTHON', raising=False)
    with pytest.raises(VideoQRRejected, match='VC_LAB_DECODER_PYTHON'):
        qr_video.decoder_python()
    monkeypatch.setenv('VC_LAB_DECODER_PYTHON', '/mnt/fixture-unmounted/python')
    with pytest.raises(VideoQRRejected, match='local'):
        qr_video.decoder_python()
