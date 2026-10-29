"""New projection bug boundaries; existing person tests retain their old receipts."""
import hashlib
import wave

from visioncortex.lab.ingest import Ingest

from .test_person_report import scene,wearer,event,report,people,find
from .test_operator_scan import scan,at

__all__=['scene']


def test_derived_replay_null_sync_and_uncertainty_zero_never_assigns_historical_wearer(scene):
    s=scene;wearer(s,0,0,'08:00:00')
    path=s['settings'].incoming/'derived-null.wav'
    with wave.open(str(path),'wb') as stream:
        stream.setparams((1,2,8000,0,'NONE','not compressed'));stream.writeframes(b'\0\0'*32000)
    m=Ingest(s['core'],s['settings']).accept(dict(capture_id='derived-null',source_instance='derived-channel',
        source_revision='fixture/1',kind='audio',relative_path=path.name,camera_id='fixture-camera-0',
        session_id='derived-fixture',captured_at=at('09:00:00'),source_kind='derived_replay',
        start_seconds='0',end_seconds='4',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        time_basis={'basis':'original_recording_clock','uncertainty_us':0,'clock_sync_valid':None,'clock_sync_verified':None}))
    a=event(s,'derived-null',m)
    data=report(s)['document']['person_timelines']
    assert data['persons']==[]
    projected=next(e for e in data['unknown_time_unknown_person_events'] if e['activity_id']==a['id'])
    assert projected['start_at'] is None and projected['recording_attribution'][0]['status']=='unknown_recorded_by'


def test_source_material_unit_is_not_explicit_experiment_sample(scene):
    s=scene;scan(s,'A',run_id=s['run'])
    material=scan(s,'M',kind='material',value='fixture-reusable-label',when='09:01:00')
    e=find(people(report(s))[s['people'][0]],material['activity_id'])
    assert e['source_material_unit_ids']==[s['materials'][0]]
    assert e['explicit_sample_ids']==[]
    assert e['source_lot_ids']==[s['lots'][0]] and e['target_batch_ids']==[s['batches'][0]]
