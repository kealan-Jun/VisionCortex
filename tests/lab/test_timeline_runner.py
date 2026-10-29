"""Synthetic codecs/localhost test services only. These do not score real models."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import threading
import wave

from PIL import Image
import pytest

from visioncortex.lab.archive_layout import contract_bytes
from visioncortex.lab.domain import Conflict
from visioncortex.lab.gauge import read_gauge
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.local_provider import LocalHTTP, LocalProviders, PreparedMedia, ProviderUnavailable
from visioncortex.lab.ocr_fields import consensus, field_candidate
from visioncortex.lab.timeline_runtime import TimelineRuntime
from visioncortex.lab.timeline_report import render_report
from visioncortex.lab.timeline_tasks import TimelineTasks
from .test_timeline import DAY, at, hj as timeline_fixture, media, run, activity


@pytest.fixture(name='hj')
def hj_fixture(core, settings):
    return timeline_fixture.__wrapped__(core, settings)


def calibration(direction='clockwise'):
    return dict(schema_version='vc-lab-gauge-calibration/1', status='verified', revision='fixture-cal/1',
                instrument_id='fixture-instrument', unit='MPa', verified_by='synthetic fixture',
                center=['0', '0'], direction=direction, tip_radius_range=['0.5', '1.5'],
                max_angle_uncertainty_degrees='2', report_increment='0.001',
                anchors=[dict(point=['1', '0'], value='-0.1'),
                         dict(point=['0', '1' if direction == 'clockwise' else '-1'], value='-0.05'),
                         dict(point=['-1', '0'], value='0')])


def needle(tip=None):
    return dict(source_sha256='a'*64, tip=tip or ['0.70710678', '0.70710678'],
                coordinate_frame='rectified_dial_plane', tip_status='verified_needle_tip',
                geometry_revision='synthetic-geometry/1', angle_uncertainty_degrees='1', reading_stage='vacuum_start')


@pytest.mark.parametrize('direction,tip', [('clockwise', ['1', '1']), ('counterclockwise', ['1', '-1'])])
def test_gauge_preserves_negative_pressure_and_interpolates_both_directions(direction, tip):
    result = read_gauge(calibration(direction), needle(tip), source_sha256='a'*64)
    assert result['value'] == '-0.075' and result['unit'] == 'MPa'
    assert float(result['interval'][0]) < -.075 < float(result['interval'][1])
    assert result['pressure_reference'] == 'unspecified' and not result['stable_pressure_proven']


@pytest.mark.parametrize('change', [dict(tip_status='counterweight'), dict(coordinate_frame='raw_perspective'),
    dict(source_sha256='b'*64), dict(angle_uncertainty_degrees='3'), dict(tip=['0.1','0.1'])])
def test_gauge_rejects_unverified_or_wrong_geometry(change):
    with pytest.raises(ValueError):
        read_gauge(calibration(), dict(needle(), **change), source_sha256='a'*64)


def test_gauge_does_not_extrapolate_and_rejects_inconsistent_scale():
    assert read_gauge(calibration(), needle(['0','-1']), source_sha256='a'*64)['value'] is None
    invalid = calibration()
    invalid['anchors'][1]['value'] = '1'
    with pytest.raises(ValueError, match='monotonic'):
        read_gauge(invalid, needle(), source_sha256='a'*64)


def raw_ocr(*texts, score=.99):
    return dict(results=[dict(rec_texts=list(texts), rec_scores=[score]*len(texts))])


def field(**overrides):
    return dict(field_key='temperature', role='process_value', channel='jacket', box=['0','0','1','1'],
                unit='°C', unit_basis='verified_on_panel', **overrides)


@pytest.mark.parametrize('texts,expected', [(('-','0.0713','g'), '-0.0713'), (('001.20g',), '001.20'), (('088',), '088')])
def test_ocr_exact_values_never_guess_missing_decimal(texts, expected):
    region = dict(field(), unit='g')
    assert field_candidate(region, raw_ocr(*texts))['value'] == expected


@pytest.mark.parametrize('texts', [(), ('1','23'), ('1,23',), ('26.1Hz',), ('-','-1'), ('8.8','88')])
def test_ocr_ambiguous_blank_wrong_unit_or_sign_requires_review(texts):
    result = field_candidate(field(), raw_ocr(*texts))
    assert result['value'] is None and result['status'] == 'needs_review'


def test_ocr_low_score_does_not_become_confirmed_and_clock_is_not_capture_time():
    assert field_candidate(field(), raw_ocr('26.2', score=.3))['value'] is None
    clock = dict(field(), field_key='panel_clock', role='clock', unit=None, data_type='clock')
    result = field_candidate(clock, raw_ocr('14:36'))
    assert result['value'] == '14:36' and not result['use_as_capture_time']


def test_burst_comparison_keeps_roles_channels_units_and_partial_failure():
    a = dict(field_candidate(field(), raw_ocr('26.2')), value='26.20')
    b = field_candidate(field(), raw_ocr('26.2'))
    c = dict(b, value=None)
    output = consensus([a,b,c,dict(b,role='setpoint'),dict(b,channel='material'),dict(b,unit='K')])
    assert len(output) == 4 and any(x['status'] == 'partial_candidate' for x in output)
    assert not any(x['auto_confirmed'] for x in output)


@contextmanager
def fake_service(kind='ocr'):
    state = dict(posts=0, bodies=[], fail_at=None, bad_hash=False, bad_vlm=False, ocr_text='26.2', redirect=False)
    revision, runtime = 'synthetic-'+kind+'/1', 'f'*64
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def respond(self, result, status=200):
            raw = json.dumps(result).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            if self.path == '/v1/models':
                self.respond(dict(data=[dict(id='fixture-vlm')]))
            elif self.path == '/health':
                self.respond(dict(status='ready', kind=kind, model_revision=revision))
            else:
                self.respond({}, 404)
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['posts'] += 1
            state['bodies'].append(body)
            if state['redirect']:
                self.send_response(302);self.send_header('Location','http://example.com/never-follow');self.end_headers();return
            if state['fail_at'] == state['posts']:
                self.respond(dict(error='synthetic failure'), 503); return
            if kind == 'vlm':
                event = dict(kind='transfer', text='合成测试动作，不代表真实实验。', start_seconds='0.1', end_seconds='99' if state['bad_vlm'] else '0.3')
                self.respond(dict(model='fixture-vlm', choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(events=[event]))))])); return
            raw = base64.b64decode(body['media_base64'])
            result = raw_ocr(state['ocr_text']) if kind == 'ocr' else dict(text='合成录音', words=[dict(text='合成录音',start_seconds=2.5,end_seconds=2.75)], time_basis='synthetic_alignment')
            result.update(source_sha256='0'*64 if state['bad_hash'] else hashlib.sha256(raw).hexdigest(),
                          model_revision=revision, runtime_revision=runtime, actual_model_invocation=True, cache_hit=False)
            self.respond(result)
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = dict(protocol='local-vllm-chat/1' if kind == 'vlm' else 'vc-lab-media-infer/1',
                 base_url=f'http://127.0.0.1:{server.server_port}', model_revision=revision,
                 runtime_revision=runtime, token_env='VC_LAB_FIXTURE_LOCAL_TOKEN')
    if kind == 'vlm':
        model.update(served_model='fixture-vlm', video_policy='all_frames', decoder_configuration_verified=True)
    try:
        yield state, model
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def configured(hj, model, kind, monkeypatch):
    monkeypatch.setenv('VC_LAB_FIXTURE_LOCAL_TOKEN','synthetic-local-only-test-token')
    path = hj['settings'].runtime_root/'providers.json'
    path.write_text(json.dumps(dict(schema_version='vc-lab-local-providers/1', enabled=True, customer_code=hj['settings'].customer_code,
                                   data_scope='fixture', models={kind:model})))
    return replace(hj['settings'], timeline_providers_path=path)


def capture(hj, filename, duration, kind, *, session='burst'):
    path = hj['settings'].incoming/filename
    return Ingest(hj['core'],hj['settings']).accept(dict(capture_id=filename, source_instance='fixture-camera', source_revision='fixture/1',
        kind=kind, session_id=session, group_expected_photos=3 if kind=='photo' else None, relative_path=filename, captured_at=at('09:00:00'),
        time_basis={'basis':'synthetic_clock','uncertainty_us':0}, start_seconds='0',end_seconds=str(duration),source_kind='fixture',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def panel(hj, name='panel.png'):
    path = hj['settings'].incoming/name
    Image.new('RGB',(120,100),'white').save(path)
    return capture(hj,name,0,'photo')


def registration(hj, m, regions=None):
    path = hj['settings'].incoming/'panel.png'
    return dict(schema_version='vc-lab-ocr-frame-registration/1', status='verified', source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                layout_revision='synthetic-panel/1',instrument_id=hj['instruments'][0], regions=regions or [field()])


@pytest.mark.parametrize('url', ['http://example.com:80','http://192.0.2.55:18110','http://user:secret@127.0.0.1:123','http://127.0.0.1:123/path'])
def test_provider_manifest_rejects_nonlocal_or_credential_urls(hj,monkeypatch,url):
    with fake_service() as (_,model):
        model['base_url']=url
        settings=configured(hj,model,'ocr',monkeypatch)
        with pytest.raises(ProviderUnavailable):
            LocalProviders(settings).load()


def test_default_runner_opens_no_services(hj):
    assert not TimelineRuntime(hj['core'],hj['settings']).runner.run_once()['enabled']


def test_actual_local_http_rejects_forged_payload_receipt(monkeypatch):
    monkeypatch.setenv('VC_LAB_FIXTURE_LOCAL_TOKEN','synthetic-token')
    with fake_service() as (state,model):
        state['bad_hash']=True
        with pytest.raises(Conflict):
            LocalHTTP(model).infer(b'synthetic-image','fixture')


def test_runner_preserves_capture_source_kind_instead_of_global_default(hj,monkeypatch):
    panel(hj)
    with fake_service() as (state,model):
        settings=replace(configured(hj,model,'ocr',monkeypatch),source_kind='synthetic_silence')
        runtime=TimelineRuntime(hj['core'],settings)
        assert runtime.runner.run_once()['status']=='completed'
        assert state['bodies'][0]['source_kind']=='fixture'


def test_local_http_does_not_follow_redirects(monkeypatch):
    monkeypatch.setenv('VC_LAB_FIXTURE_LOCAL_TOKEN','synthetic-token')
    with fake_service() as (state,model):
        state['redirect']=True
        with pytest.raises(ProviderUnavailable):
            LocalHTTP(model).infer(b'synthetic-image','fixture')
        assert state['posts']==1


def test_vlm_requires_verified_decoder_configuration(hj,monkeypatch):
    with fake_service('vlm') as (_,model):
        model['decoder_configuration_verified']=False
        with pytest.raises(ProviderUnavailable):
            LocalProviders(configured(hj,model,'video_understanding',monkeypatch)).load()


def test_ocr_runner_processes_more_than_six_fields_and_keeps_registration_history(hj,monkeypatch):
    m=panel(hj)
    regions=[dict(field(),field_key='channel_'+str(i),channel=str(i),box=['0','0','1','1']) for i in range(8)]
    runtime=TimelineRuntime(hj['core'],hj['settings'])
    doc=registration(hj,m,regions)
    saved=runtime.readouts.register_frame('reg',m['media_id'],doc,actor='fixture',reason='synthetic source regions')
    assert runtime.readouts.register_frame('reg',m['media_id'],doc,actor='fixture',reason='synthetic source regions') == saved
    with fake_service() as (state,model):
        settings=configured(hj,model,'ocr',monkeypatch)
        result=TimelineRuntime(hj['core'],settings).runner.run_once()
        assert result['status']=='completed',result
        assert state['posts']==8
    with hj['core'].db.transaction(readonly=True) as conn:
        row=conn.execute('SELECT result FROM hj_timeline_task WHERE customer_id=%s AND id=%s',(hj['core'].customer,result['task_id'])).fetchone()['result']
    assert row['execution_verification']=='local_transport_and_payload_verified'
    with hj['core'].db.transaction(readonly=True) as conn:
        session=conn.execute('SELECT session_id FROM capture WHERE customer_id=%s AND id=%s',(hj['core'].customer,m['capture_id'])).fetchone()['session_id']
    readout=runtime.views.readout(str(session))
    trace=runtime.views.trace(at('09:00:00'),at('09:00:01'))
    fields=trace['activities'][0]['payload']['fields']
    assert len(fields)==8 and all(f['value']=='26.2' for f in fields)
    assert len(readout['fields'])==8 and readout['burst_status']=='partial_or_unknown'
    bad=deepcopy(doc);bad['source_sha256']='0'*64
    with pytest.raises(ValueError):
        runtime.readouts.register_frame('bad-reg',m['media_id'],bad,actor='fixture',reason='wrong frame')


def test_failed_window_resumes_from_checkpoint_without_repeating_first_window(hj,monkeypatch):
    path=hj['settings'].incoming/'audio.wav'
    with wave.open(str(path),'wb') as out:
        out.setparams((1,2,16000,0,'NONE','not compressed'));out.writeframes(b'\0\0'*16000*31)
    capture(hj,'audio.wav',31,'audio')
    with fake_service('asr') as (state,model):
        state['fail_at']=2
        runtime=TimelineRuntime(hj['core'],configured(hj,model,'transcription',monkeypatch))
        first=runtime.runner.run_once()
        assert first['status']=='awaiting_provider' and state['posts']==2,first
        with hj['core'].db.transaction() as conn:
            conn.execute("UPDATE hj_timeline_task SET next_attempt_at='1970-01-01Z' WHERE customer_id=%s",(hj['core'].customer,))
        second=runtime.runner.run_once()
        assert second['status']=='completed' and state['posts']==3,second
    trace=runtime.views.trace(at('09:00:00'),at('09:01:00'))
    assert len(trace['activities'])==2
    assert trace['activities'][1]['evidence'][0]['start_seconds']=='30.500000'


def test_three_failures_stop_automatic_retry_and_expired_lease_cannot_be_renewed(hj,monkeypatch):
    panel(hj)
    with fake_service() as (state,model):
        state['bad_hash']=True
        runtime=TimelineRuntime(hj['core'],configured(hj,model,'ocr',monkeypatch))
        for i in range(3):
            value=runtime.runner.run_once()
            with hj['core'].db.transaction() as conn:
                conn.execute("UPDATE hj_timeline_task SET next_attempt_at='1970-01-01Z' WHERE customer_id=%s",(hj['core'].customer,))
        assert value['status']=='needs_attention' and runtime.runner.run_once() is None
        tasks=TimelineTasks(hj['timeline'])
        m=panel(hj,'second.png')
        job=tasks.claim('expired','ocr','fixture/1',now=datetime.now(timezone.utc)-timedelta(minutes=10))
        with pytest.raises(Conflict):
            tasks.renew(job['task_id'],job['lease_token'])
        tasks.reanalyze('reanalyze',m['media_id'],'ocr','manual-retry/1')


def test_renewal_rejects_wrong_token_and_idle_worker_does_not_grow_command_log(hj,monkeypatch):
    panel(hj)
    with fake_service() as (_,model):
        runtime=TimelineRuntime(hj['core'],configured(hj,model,'ocr',monkeypatch))
        job=runtime.tasks.claim('lease','ocr',model['model_revision'],lease_seconds=10)
        runtime.tasks.renew(job['task_id'],job['lease_token'],30)
        from uuid import uuid4
        with pytest.raises(Conflict):
            runtime.tasks.renew(job['task_id'],str(uuid4()))
        with hj['core'].db.transaction(readonly=True) as conn:
            before=conn.execute('SELECT count(*) AS n FROM command_receipt WHERE customer_id=%s',(hj['core'].customer,)).fetchone()['n']
        assert runtime.runner.run_once() is None and runtime.runner.run_once() is None
        with hj['core'].db.transaction(readonly=True) as conn:
            after=conn.execute('SELECT count(*) AS n FROM command_receipt WHERE customer_id=%s',(hj['core'].customer,)).fetchone()['n']
        assert before==after


def test_refined_ocr_registration_keeps_history_without_comparing_old_revision_as_new_photo(hj,monkeypatch):
    m=panel(hj)
    with fake_service() as (state,model):
        runtime=TimelineRuntime(hj['core'],configured(hj,model,'ocr',monkeypatch))
        doc=registration(hj,m)
        runtime.readouts.register_frame('first-reg',m['media_id'],doc,actor='fixture',reason='synthetic initial registration')
        assert runtime.runner.run_once()['status']=='completed'
        assert runtime.runner.run_once()['status']=='completed'
        state['ocr_text']='27.0'
        doc['layout_revision']='synthetic-panel/2'
        runtime.readouts.register_frame('corrected-reg',m['media_id'],doc,actor='fixture',reason='synthetic refined registration')
        assert runtime.runner.run_once()['status']=='completed'
        with hj['core'].db.transaction(readonly=True) as conn:
            session=conn.execute('SELECT session_id FROM capture WHERE customer_id=%s AND id=%s',(hj['core'].customer,m['capture_id'])).fetchone()['session_id']
        readout=runtime.views.readout(session)
        assert len(readout['superseded_observation_ids'])==2
        assert readout['fields'][0]['candidates'][0]['value']=='27.0'
        assert readout['fields'][0]['status']=='partial_candidate' and readout['fields'][0]['source_photo_count']==1


def test_media_preparer_rejects_a_false_duration_instead_of_claiming_input_coverage(hj):
    path=hj['settings'].incoming/'short.wav'
    with wave.open(str(path),'wb') as out:
        out.setparams((1,2,16000,0,'NONE','not compressed'));out.writeframes(b'\0\0'*16000)
    capture(hj,'short.wav',30,'audio')
    job=TimelineTasks(hj['timeline']).claim('prepare','transcription','fixture/1')
    ref=job['request']['windows'][0]['input_spans'][0]
    preparer=PreparedMedia(hj['core'],hj['settings'])
    with preparer.workspace() as folder, pytest.raises(ValueError,match='cover'):
        preparer.prepare(ref,'transcription',{},folder)


def test_video_transport_uses_real_codec_preserves_all_test_frames_and_bounds_events(hj,monkeypatch):
    path=hj['settings'].incoming/'video.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=white:s=160x120:r=24:d=2','-an','-c:v','libx264','-y',str(path)],check=True)
    capture(hj,'video.mp4',2,'video')
    with fake_service('vlm') as (state,model):
        runtime=TimelineRuntime(hj['core'],configured(hj,model,'video_understanding',monkeypatch))
        result=runtime.runner.run_once()
        assert result['status']=='completed',result
        video=next(x['video_url']['url'] for x in state['bodies'][0]['messages'][0]['content'] if x['type']=='video_url')
        derived=hj['settings'].runtime_root/'test-derived.mp4';derived.write_bytes(base64.b64decode(video.split(',',1)[1]))
        try:
            probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','json',str(derived)],capture_output=True,text=True,check=True)
            assert json.loads(probe.stdout)['streams'][0]['nb_read_frames']=='48'
        finally:
            derived.unlink()
        original_media=runtime.views.trace(at('09:00:00'),at('09:00:02'))['media'][0]['media_id']
        runtime.tasks.reanalyze('bad-time-replay',original_media,'video_understanding','bad-time-fixture')
        state['bad_vlm']=True
        assert runtime.runner.run_once()['status']=='awaiting_provider'


def test_nonzero_video_seek_keeps_requested_content_and_frames(hj):
    import io
    path=hj['settings'].incoming/'seek.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=160x120:r=12:d=2',
        '-f','lavfi','-i','color=c=blue:s=160x120:r=12:d=2','-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0',
        '-c:v','libx264','-y',str(path)],check=True)
    capture(hj,path.name,4,'video')
    ref=TimelineTasks(hj['timeline']).claim('seek','video_understanding','fixture/1')['request']['windows'][0]['input_spans'][0]
    ref=dict(ref,start_seconds='2',end_seconds='4')
    preparer=PreparedMedia(hj['core'],hj['settings'])
    with preparer.workspace() as folder:
        raw,_,receipt=preparer.prepare(ref,'video_understanding',dict(video_policy='all_frames'),folder)
        output=folder/'verified-seek.mp4';output.write_bytes(raw)
        png=subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-frames:v','1','-f','image2pipe','-vcodec','png','-'])
        pixel=Image.open(io.BytesIO(png)).convert('RGB').getpixel((80,60))
        assert pixel[2]>200 and pixel[0]<30,pixel
        assert receipt['transform']['decoded_frame_count']=='24'


@pytest.mark.parametrize('blank',[False,True])
def test_runner_automatic_layout_checkpoint_or_raw_fallback(hj,monkeypatch,blank):
    from .test_automatic_readout import template
    import cv2
    import numpy as np
    image,manifest,doc=template(hj['settings'].runtime_root/'models/fixture-panel')
    if not blank:
        import qrcode
        qr=qrcode.QRCode(box_size=3,border=4);qr.add_data('fixture-equipment-qr');qr.make(fit=True)
        pixels=np.array(qr.make_image().convert('RGB'))
        height,width=pixels.shape[:2]
        image[230:230+height,100:100+width]=pixels
        doc['profiles'][0]['identity_qr_box']=['.10','.60','.45','1']
        manifest.write_text(json.dumps(doc))
    path=hj['settings'].incoming/'automatic.png'
    cv2.imwrite(str(path),np.full_like(image,255) if blank else image)
    capture(hj,path.name,0,'photo')
    with fake_service() as (state,model):
        model['panel_profiles']=dict(path=str(manifest),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        runtime=TimelineRuntime(hj['core'],configured(hj,model,'ocr',monkeypatch))
        assert runtime.runner.run_once()['status']=='completed'
        assert state['posts']==1
        with hj['core'].db.transaction(readonly=True) as conn:
            row=conn.execute('SELECT normalized_result FROM hj_timeline_window_result WHERE customer_id=%s', (hj['core'].customer,)).fetchone()
        event=row['normalized_result']['events'][0]
        assert event['payload']['field_mapping']==('unmapped_raw_ocr' if blank else 'automatic_registered_layout')
        assert event['instrument_id']==(None if blank else hj['instruments'][0])
        if not blank:assert event['fields'][0]['value']=='26.2'


def test_publication_keeps_immutable_versions_and_evidence_links(hj):
    m=media(hj,'report-fixture',duration=30)
    rid=run(hj)
    activity(hj,'first',m,run_ids=[rid],kind='loading')
    root=hj['settings'].incoming
    (root/'.System').mkdir();(root/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    settings=replace(hj['settings'],source_root=root,unified_archive_required=True,timeline_publish=True)
    runtime=TimelineRuntime(hj['core'],settings)
    first=runtime.views.snapshot(DAY)
    result=runtime.publication.tick()
    assert len(result['files'])==6 and runtime.publication.tick() is None
    original={f['relative_path']:(root/f['relative_path']).read_bytes() for f in result['files']}
    activity(hj,'late',m,start=3,end=5,run_ids=[rid],kind='settings')
    second=runtime.views.snapshot(DAY)
    assert second['revision']>first['revision']
    runtime.publication.tick()
    assert all((root/name).read_bytes()==raw for name,raw in original.items())
    text=render_report(second['document'])
    assert '设备实际结束时间：未知' in text and m['media_id'] in text
    assert not list(root.glob('*/VC')) and not list(root.glob('*/FR'))


def test_publication_rejects_changed_immutable_bytes_and_wrong_archive(hj):
    media(hj,'report-fixture',duration=30)
    root=hj['settings'].incoming
    (root/'.System').mkdir();(root/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    runtime=TimelineRuntime(hj['core'],replace(hj['settings'],source_root=root,unified_archive_required=True,timeline_publish=True))
    runtime.views.snapshot(DAY)
    # Partial publication remains replayable, but an unrelated replacement is rejected.
    from visioncortex.lab.components import pipeline_module
    guard=pipeline_module('device_day_write_guard')
    (root/'owned.json').write_text('conflicting bytes')
    with pytest.raises(Conflict):
        runtime.publication._write(root,'owned.json',b'new bytes',guard)
    (root/'.System/ArchiveContract.json').write_text('{}')
    with pytest.raises(Conflict):
        runtime.publication.tick()


def test_publication_unifies_recorder_photo_and_audio_channels_under_exact_camera_key(hj):
    for index,kind in enumerate(('video','audio','photo')):
        name=f'channel-{index}'+dict(video='.mp4',audio='.wav',photo='.png')[kind]
        path=hj['settings'].incoming/name
        path.write_bytes(('synthetic transport channel '+kind).encode())
        Ingest(hj['core'],hj['settings']).accept(dict(capture_id=name,source_instance=('recorder:' if kind=='video' else 'photo:' if kind=='photo' else 'voice:')+'camera01',
            source_revision='fixture/1',kind=kind,camera_id='camera01',relative_path=name,captured_at=at('09:00:00'),
            time_basis={'basis':'synthetic_clock'},source_kind='fixture',end_seconds='0' if kind=='photo' else '30',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    root=hj['settings'].incoming
    (root/'.System').mkdir();(root/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    runtime=TimelineRuntime(hj['core'],replace(hj['settings'],source_root=root,unified_archive_required=True,timeline_publish=True))
    runtime.views.snapshot(DAY)
    result=runtime.publication.tick()
    assert len(result['files'])==6 and all(x['relative_path'].startswith(DAY+'_camera01/') for x in result['files'])
    view=json.loads((root/next(f['relative_path'] for f in result['files'] if '/Comment/Audio/' in f['relative_path'])).read_text())
    assert set(view['stream_keys'])=={'recorder:camera01','photo:camera01','voice:camera01'}
