from copy import deepcopy
from datetime import datetime
import json
import os
import time

import pytest
from psycopg.types.json import Jsonb

from visioncortex.lab.components import pipeline_module
from visioncortex.lab.delivery import NativeDelivery
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.processing import Worker
from visioncortex.lab.rules import Rules


def native_fixture(settings,camera="camera-one"):
    import cv2
    import numpy as np
    start=int(datetime.fromisoformat('2026-10-23T10:00:00+08:00').timestamp()*1e6)
    folder=settings.incoming/f'2026-10-23_{camera}/MetaData/100000';folder.mkdir(parents=True)
    path=folder/'rgb.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),4,(96,64))
    for n in range(4):writer.write(np.full((64,96,3),n*40,dtype=np.uint8))
    writer.release()
    (folder/'frames.csv').write_text('rgb_video_frame_index,global_timestamp_us\n'+'\n'.join(f'{n},{start+n*250000}' for n in range(4)))
    meta=dict(recording_session_id='fixture-session',camera_key=camera,closed=True,frames_publish_state='finalized',recording_window_start_global_us=start,recording_window_end_global_us=start+750000)
    (folder/'meta.json').write_text(json.dumps(meta))
    (folder/'recording_ready.json').write_text(json.dumps(dict(meta,ready=True,recording_complete=True,recording_quality_status='complete')))
    for p in folder.iterdir():os.utime(p,(time.time()-2,time.time()-2))
    return folder/'recording_ready.json'


def test_native_delivery_is_automatic_idempotent_and_model_gated(core,settings):
    receipt=native_fixture(settings)
    value=Ingest(core,settings).native_receipt(receipt)[0]
    with core.db.transaction() as conn:
        conn.execute("UPDATE processing_job SET status='completed',processed_at=now(),result=%s WHERE customer_id=%s AND capture_id=%s AND worker_kind='vision'",(Jsonb({'fixture_only':True,'segments':[]}),core.customer,value['capture_id']))
    assert NativeDelivery(core,settings).tick()=={'created':1,'dependencies_ready':1}
    assert NativeDelivery(core,settings).tick()=={'created':0,'dependencies_ready':0}
    result=Worker(core,settings).run_once()
    assert result['status']=='blocked_model'
    with core.db.transaction(readonly=True) as conn:
        job=conn.execute("SELECT * FROM processing_job WHERE customer_id=%s AND worker_kind='understanding'",(core.customer,)).fetchone()
        assert job['input_fingerprint']
        assert conn.execute('SELECT count(*) AS n FROM processing_dependency WHERE customer_id=%s',(core.customer,)).fetchone()['n']==1
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',(core.customer,)).fetchone()['n']==0
    assert not list(settings.incoming.rglob('LaboratoryDailyReport.json'))


def test_existing_native_runner_publishes_report_without_reinvoking_parents(core,settings,monkeypatch):
    from visioncortex.lab.native_pipeline import deliver,ParentBackend
    path=native_fixture(settings)
    video=path.with_name('rgb.mp4')
    record=pipeline_module('nas_recordings')._inspect(settings.incoming,video,time.time(),.5,allow_plain=False)
    contract=pipeline_module('device_day_contract')
    layout=contract.DeviceDayLayout(settings.incoming,'camera-one',record['recording_start_us'],settings.runtime_root/'native/cache',raw_directory='MetaData',read_only_originals=True)
    layout.create()
    frame=layout.processed/'.Evidence/fixture/frame.png';frame.parent.mkdir(parents=True)
    import cv2
    import numpy as np
    image=np.zeros((64,96,3),dtype='uint8')
    assert cv2.imwrite(str(frame),image)
    sample=dict(contract.artifact(layout.root,frame),frame_id='fixture-frame',frame_kind='scene_sample',local_ms=0,global_us=record['recording_start_us'])
    source=contract.artifact(layout.root,video)
    segment=dict(segment_id='fixture-segment',recording_id=record['recording_id'],activity='inactive',activity_label='无实验活动',start_ms=0,end_ms=750,start_us=record['recording_start_us'],end_us=record['recording_end_us'],source_ref=source,key_frames=[],scene_frames=[sample],evidence_status='PARTIAL_EVIDENCE',physical_action_confirmed=False)
    paths=pipeline_module('device_day_content_paths')
    name=paths.clip_name(segment)
    segment['json_path']=f"ProcessedClips/Clips/{name}/{paths.layout_contract()['activities']['inactive']}.json"
    contract.atomic_json(layout.root/segment['json_path'],segment)
    visual=dict(segments=[segment],artifacts=[contract.artifact(layout.root,frame)],source_duration_ms=750,fixture_only=True)
    cfg=deepcopy(Rules().profile('pipeline')['settings'])
    cfg.update(storage=dict(archive_root=str(settings.incoming),local_runtime_root=str(settings.runtime_root/'native'),local_cache_root=str(settings.runtime_root/'native/cache')),collection_ingest=dict(enabled=True,source_root=str(settings.incoming),settle_seconds=.5),runtime=dict(local_only=True))
    cfg['models']={};cfg['mllm'].update(enabled=False,base_url='https://fixture.invalid',api_key_env='VC_LAB_FIXTURE_ABSENT')
    cfg['device_day'].update(enabled=True,raw_originals_read_only=True,delete_capture_sources=False,archive_html_enabled=False,readable_content_enabled=True,derived_video_retention=False)
    monkeypatch.setattr('visioncortex.lab.native_pipeline.configuration',lambda *_:cfg)
    invoked=[]
    def fixture_understand(self,layout,recording,vision,context,key):
        invoked.append(key)
        return dict(vision_key=vision['key'],context_digest=contract.digest(context),understandings=[],artifacts=[],fixture_only=True)
    monkeypatch.setattr(ParentBackend,'understand',fixture_understand)
    request=dict(source_root=str(settings.incoming),runtime_root=str(settings.runtime_root),native_record=record,vision_model={'camera_roles':{'camera-one':'first_person'}},model={'revision':'fixture-only'},parents=[dict(id='fixture-parent',worker_kind='vision',result=visual)],external_calls_authorized=False)
    result=deliver(request)
    assert result['analysis_outcome']=='observations',result
    assert result['model_invoked'] is False and result['accuracy']=='NOT_PROVEN'
    report=layout.root/result['daily_report']['path']
    assert report.is_file() and json.loads(report.read_text())
    again=deliver(request)
    assert again['native_stage_receipts']==result['native_stage_receipts']
    assert report.is_file()  # the old report version stays immutable after refresh
    assert len(invoked)==1
    comments=layout.root/'Comment'
    assert sorted(p.name for p in comments.iterdir())==['Audio']
    assert (comments/'Audio/Index.json').is_file()
    assert (comments/'Audio/TimeIndex.json').is_file()
    assert list((comments/'Audio').glob('*/Transcript.json'))
    index=json.loads((layout.root/'ProcessedClips/Index.json').read_text())
    assert index['time_index']['path']=='Comment/Audio/TimeIndex.json'
    for entry in json.loads((comments/'Audio/Index.json').read_text())['entries']:
        assert entry['content'].startswith('Comment/Audio/')
        assert (layout.root/entry['content']).is_file()
    assert not list(settings.incoming.rglob('*.html'))
    assert len(list(settings.incoming.rglob('*.mp4')))==1


def test_native_delivery_backlog_does_not_starve_after_limit(core,settings):
    from visioncortex.lab.domain import ident
    source=native_fixture(settings)
    capture=Ingest(core,settings).native_receipt(source)[0]
    rules=Rules().identity()
    # Many immutable executions of one fixture recording, representing different
    # completed model revisions; every exact parent must eventually be scheduled.
    with core.db.transaction() as conn:
        conn.execute("UPDATE processing_job SET status='completed',processed_at=now() WHERE customer_id=%s AND capture_id=%s",(core.customer,capture['capture_id']))
        for n in range(25):
            conn.execute("INSERT INTO processing_job(customer_id,id,capture_id,worker_kind,model_revision,status,rule_identity,rule_fingerprint,processed_at) VALUES(%s,%s,%s,'vision',%s,'completed',%s,%s,now())",(core.customer,ident(),capture['capture_id'],'fixture-'+str(n),Jsonb(rules),rules['sha256']))
    delivery=NativeDelivery(core,settings)
    assert delivery.tick(limit=20)['created']==20
    assert delivery.tick(limit=20)['created']==6
    assert delivery.tick(limit=20)['created']==0


def test_photo_and_native_video_share_one_device_day_directory(core,settings):
    from visioncortex.lab.archive_layout import day_directory
    from visioncortex.lab.retention import Retention
    import hashlib
    import qrcode
    camera='lab.cam_'+('x'*105)  # FR used to hash this while VC kept it intact.
    ready=native_fixture(settings,camera)
    ingest=Ingest(core,settings)
    ingest.native_receipt(ready)
    directory=day_directory(camera,'2026-10-23T10:58:00+08:00')
    original=settings.incoming/directory/'MetaData/105800/1.png'
    original.parent.mkdir(parents=True)
    qrcode.make('unbound-fixture').save(original)
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    ingest.accept(dict(capture_id='photo-same-camera',source_instance='photo:'+camera,source_revision='1',camera_id=camera,kind='photo',relative_path=original.relative_to(settings.incoming).as_posix(),captured_at='2026-10-23T10:58:00+08:00',time_basis={'basis':'fixture'},source_kind='fixture',sha256=digest))
    retain=Retention(core,settings)
    for _ in range(6):
        assert retain.tick().get('status')!='needs_attention'
    day=settings.incoming/directory
    assert list(day.glob('ProcessedClips/.Evidence/Retention/*/Retention.json'))
    assert list(day.glob('Comment/OCR/VoicePhotoReadings/*/Evidence.json'))
    assert sorted(p.name for p in settings.incoming.iterdir() if p.is_dir() and not p.name.startswith('.'))==[directory]
    assert hashlib.sha256(original.read_bytes()).hexdigest()==digest
    assert not list(settings.incoming.rglob('Result*.json'))


@pytest.mark.parametrize('raw_directory,complete', [('MetaData', True), ('MetaVideo', False)])
def test_nested_native_package_waits_for_sealing_and_stays_in_place(core, settings, raw_directory, complete):
    from visioncortex.lab.media import Media
    from visioncortex.lab.retention import Retention
    import hashlib
    ready = native_fixture(settings)
    destination = ready.parents[2]/raw_directory/'Video/100000'
    destination.parent.mkdir(parents=True, exist_ok=True)
    ready.parent.rename(destination)  # owned fixture simulates a producer path
    ready = destination/ready.name
    metadata_path = destination/'meta.json'
    metadata = json.loads(metadata_path.read_text())
    metadata_path.write_text(json.dumps(dict(metadata, closed=False)))
    status = json.loads(ready.read_text())
    ready.write_text(json.dumps(dict(status, recording_complete=complete, recording_quality_status='complete' if complete else 'partial')))
    marker = destination.parent/'.uploading'
    marker.write_text('fixture upload in progress')
    ingest = Ingest(core, settings)
    assert ingest.discover_once()['errors']
    marker.unlink()
    assert ingest.discover_once()['errors']  # ready alone cannot override closed=False
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM processing_job WHERE customer_id=%s', (core.customer,)).fetchone()['n'] == 0
    metadata_path.write_text(json.dumps(metadata))
    for path in destination.iterdir():
        os.utime(path, (time.time()-2, time.time()-2))
    snapshots = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in destination.iterdir()}
    result = ingest.discover_once()
    assert result['processed'] == 1 and not result['errors'], result
    assert ingest.discover_once()['skipped'] == 1
    with core.db.transaction(readonly=True) as conn:
        media = conn.execute('SELECT id FROM media_asset WHERE customer_id=%s', (core.customer,)).fetchone()
        payload = conn.execute('SELECT payload FROM inbox WHERE customer_id=%s', (core.customer,)).fetchone()['payload']
        assert payload['time_basis']['capture_complete'] is complete
    assert Media(core, settings).locate(media['id'])['path'] == destination/'rgb.mp4'
    assert Worker(core, settings).run_once()['status'] == 'blocked_model'
    assert Retention(core, settings).tick()['published'] == 1
    assert not list(settings.cache.rglob('*.mp4'))
    assert len(list(settings.incoming.rglob('*.mp4'))) == 1
    for path, snapshot in snapshots.items():
        assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == snapshot
