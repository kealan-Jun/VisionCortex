"""Retained-ASR import with synthetic audio/text; zero model calls."""
from dataclasses import replace
import hashlib
import json
import wave

import pytest

from visioncortex.lab.domain import Conflict, clean
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.native_comment_intake import NativeCommentIntake, prepare
from visioncortex.lab.timeline_native_artifacts import artifact_documents
from visioncortex.lab.timeline_runtime import TimelineRuntime
from visioncortex.lab.rules import Rules


@pytest.fixture
def native(core,settings):
    settings=replace(settings,processing_profile='timeline-v1')
    camera='fixture-native-comment-camera';archive='2026-10-29_'+camera
    core.register('camera','capture_device',camera)
    raw=settings.incoming/archive/'MetaData/090000';raw.mkdir(parents=True)
    audio=raw/'audio.wav'
    with wave.open(str(audio),'wb') as stream:
        stream.setparams((1,2,8000,0,'NONE','not compressed'));stream.writeframes(b'\0\0'*16000)
    digest=hashlib.sha256(audio.read_bytes()).hexdigest();recording='fixture-retained-recording'
    record=dict(recording_id=recording,camera_key=camera,relative_path=archive+'/MetaData/090000/rgb.mp4',
        processable=True,capture_complete=True,audio=dict(status='provided',capture_complete=True,source_signature='fixture-audio-signature',
            start_us=1793235600000000,end_us=1793235602000000))
    accepted=Ingest(core,settings).accept(dict(capture_id=recording+':audio',source_instance='recorder:'+camera,
        source_revision='fixture-audio-signature',camera_id=camera,kind='audio',relative_path=archive+'/MetaData/090000/audio.wav',
        captured_at='2026-10-29T09:00:00+08:00',source_kind='fixture',time_basis={'basis':'uncalibrated_native_clock'},
        start_seconds='0',end_seconds='2',native_record=record))
    event='Comment/Audio/09-00-00.000000_09-00-02.000000'
    original='Comment/.Evidence/Recognition/'+recording+'/fixture-content/Transcript.txt'
    text='这是保留转写，不是已核实的实验指令。'
    model=dict(provider='fixture',model='retained-synthetic-ASR-not-invoked',revision='fixture/1')
    document=dict(recording_id=recording,device=camera,status='completed',outcome='transcribed',
        start_us=record['audio']['start_us'],end_us=record['audio']['end_us'],
        audio=dict(status='provided',capture_complete=True,source_signature='fixture-audio-signature',
                   artifacts=[dict(path='MetaData/090000/audio.wav',sha256=digest)]),
        original_transcript=dict(path=original,sha256=hashlib.sha256(text.encode()).hexdigest()),
        sentences=[dict(text=text,audio_start_seconds='.25',audio_end_seconds='1.5',model=model,
                        audio_ref=dict(path='MetaData/090000/audio.wav',sha256=digest))])
    base=settings.incoming/archive
    (base/event).mkdir(parents=True);(base/original).parent.mkdir(parents=True)
    (base/original).write_text(text);(base/event/'Transcript.txt').write_text(text)
    (base/event/'Transcript.json').write_text(json.dumps(document))
    (base/'Comment/Audio/Index.json').write_text(json.dumps(dict(entries=[dict(recording_id=recording,content=event+'/Transcript.json')])))
    return dict(core=core,settings=settings,runtime=TimelineRuntime(core,settings),accepted=accepted,
                intake=NativeCommentIntake(core,settings),document=document,base=base,event=event,archive=archive)


@pytest.mark.parametrize('missing_index',[False,True])
def test_completed_transcript_precedes_index_but_requires_exact_source_envelope(native,missing_index):
    n=native
    index=n['base']/'Comment/Audio/Index.json'
    if missing_index:
        index.unlink()
    else:
        index.write_text('{"entries":[]}')
    transcript=n['base']/n['event']/'Transcript.json'
    wrong=dict(n['document'],end_us=n['document']['end_us']+1)
    transcript.write_text(json.dumps(wrong))
    with pytest.raises(Conflict,match='exact recorder audio envelope'):
        n['intake'].consume(n['accepted']['media_id'])
    transcript.write_text(json.dumps(n['document']))
    result=n['intake'].consume(n['accepted']['media_id'])
    assert result['status']=='retained_native_ASR_linked' and result['new_model_calls']==0
    assert result['results'][0]['reference']['transcript']==n['archive']+'/'+n['event']+'/Transcript.json'


def test_retained_native_ASR_joins_timeline_without_new_inference_or_Audio_folder(native):
    n=native;core=n['core'];mid=n['accepted']['media_id']
    before=sorted(str(p) for p in n['settings'].incoming.rglob('*'))
    result=n['intake'].consume(mid)
    assert result['status']=='retained_native_ASR_linked' and result['new_model_calls']==0
    assert n['intake'].consume(mid)==result
    with core.db.transaction(readonly=True) as conn:
        activities=conn.execute("SELECT * FROM hj_activity WHERE customer_id=%s AND source_type='speech'",(core.customer,)).fetchall()
        assert len(activities)==1 and activities[0]['start_at'] is None
        assert 'retained-synthetic-ASR-not-invoked' in activities[0]['model_revision']
        task=conn.execute("SELECT * FROM hj_timeline_task WHERE customer_id=%s AND media_id=%s",(core.customer,mid)).fetchone()
        assert task['status']=='completed' and task['attempts']==0
        assert task['result']['model_invoked'] is False
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',(core.customer,)).fetchone()['n']==0
    assert sorted(str(p) for p in n['settings'].incoming.rglob('*'))==before
    report=n['runtime'].views.snapshot('2026-10-29')
    # The native wall clock was explicitly uncalibrated; source-relative speech
    # remains queryable without pretending it happened on the receipt day.
    assert activities[0]['payload']['aligned_units'][0]['start_seconds']=='0.25'
    source=dict(source_sha256=n['intake'].source(mid)['verified_sha256'])
    activity=clean(dict(activities[0],id=str(activities[0]['id']),evidence=[dict(media_id=mid)]))
    files,refs=artifact_documents(core,archive=n['archive'],camera='fixture-native-comment-camera',
        activities=[activity],media_sources={mid:source},rules=Rules().identity())
    assert files=={} and refs[0]['new_transcript_files_generated'] is False
    assert refs[0]['transcript']==n['archive']+'/'+n['event']+'/Transcript.json'
    assert report


@pytest.mark.parametrize('case',['camera','hash','range','path'])
def test_conflicting_native_transcript_never_imports_or_relabels_source(native,case):
    n=native;source=n['intake'].source(n['accepted']['media_id'])
    document=json.loads(json.dumps(n['document']))
    path=n['archive']+'/'+n['event']+'/Transcript.json'
    if case=='camera':document['device']='another-camera'
    if case=='hash':document['sentences'][0]['audio_ref']['sha256']='0'*64
    if case=='range':document['sentences'][0]['audio_end_seconds']='20'
    if case=='path':path=n['archive']+'/Comment/../Transcript.json'
    with pytest.raises((Conflict,ValueError)):
        prepare(document,source,path)


def test_native_audio_provenance_is_retained_but_uncalibrated_clock_cannot_complete_an_experiment(native):
    from visioncortex.lab.comment_intake import CommentIntake
    n=native;linked=n['intake'].consume(n['accepted']['media_id'])
    result=CommentIntake(n['core']).consume(linked['results'][0]['activity_id'])
    assert 'unknown_comment_event_time' in result['attention_reasons']
    assert 'original_audio_provenance_unavailable_or_conflicting' not in result['attention_reasons']
    assert result['status']=='pending_interpretation' and result['quantity_ledger_written'] is False


def test_explicit_inventory_root_alias_preserves_original_transcript_identity(native):
    from visioncortex.lab.native_comment_intake import inventory_alias
    n=native;source=n['intake'].source(n['accepted']['media_id'])
    record=source['receipt']['native_record']
    record.update(recording_session_id='fixture-session',recording_start_us=1,recording_end_us=2)
    row=dict(record,relative_path='ProducerArchive/'+record['relative_path'],recording_id='original-root-recording',
             source_signature='original-root-video-signature',audio=dict(source_signature='original-root-audio-signature'))
    source['recording_aliases']=[inventory_alias(source,row,'ProducerArchive')]
    document=json.loads(json.dumps(n['document']));document['recording_id']=row['recording_id']
    document['audio']['source_signature']=row['audio']['source_signature']
    document['original_transcript']['path']=document['original_transcript']['path'].replace('fixture-retained-recording',row['recording_id'])
    plan=prepare(document,source,n['archive']+'/'+n['event']+'/Transcript.json')
    assert plan['recording_id']==row['recording_id'] and plan['native_recording_id']=='fixture-retained-recording'
    assert plan['audio_sha256']==source['verified_sha256']
    for name,value in [('camera_key','another-camera'),('recording_session_id','another-session'),('recording_start_us',3)]:
        with pytest.raises(Conflict):inventory_alias(source,dict(row,**{name:value}),'ProducerArchive')
    document['sentences'][0]['audio_ref']['sha256']='0'*64
    with pytest.raises(Conflict):prepare(document,source,n['archive']+'/'+n['event']+'/Transcript.json')


def test_one_out_of_bounds_sentence_is_retained_unaligned_without_widening_audio(native):
    n=native;source=n['intake'].source(n['accepted']['media_id'])
    document=json.loads(json.dumps(n['document']))
    document['sentences'].append(dict(document['sentences'][0],text='原文保留；此句时间待核对',audio_start_seconds='1.8',audio_end_seconds='2.013'))
    plan=prepare(document,source,n['archive']+'/'+n['event']+'/Transcript.json')
    assert len(plan['aligned_units'])==1 and len(plan['unaligned_units'])==1
    assert plan['unaligned_units'][0]['reported_end_seconds']=='2.013'
    assert plan['source_end_seconds']=='2.000000' and '时间待核对' in plan['text']
    assert plan['alignment_status']=='partial_original_sentence_bounds'
