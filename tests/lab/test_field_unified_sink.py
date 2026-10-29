"""Real local SQLite acknowledgement and original-shaped FR file projection."""
from contextlib import contextmanager
import hashlib
import json
import sqlite3

import pytest

from visioncortex.lab.archive_layout import contract_bytes
from visioncortex.lab.field_unified_sink import UnifiedFRSink,SCHEMA


def encoded(value):return (json.dumps(value,sort_keys=True)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def source(tmp_path):
    old=tmp_path/'FieldRecognitionArchive';old.mkdir()
    destination=tmp_path/'runtime/incoming';destination.mkdir(parents=True)
    (destination/'.System').mkdir();(destination/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    runtime=tmp_path/'runtime'
    folder='2026-10-29_fixture-camera/VoicePhotoReadings/original-event'
    event=old/folder;event.mkdir(parents=True)
    result=encoded(dict(device_model='original',device_no=None,photo_time=None,values=[dict(value='24.2',unit=None)],
        camera_id='fixture-camera',instrument_id='fixture-instrument'))
    (event/'Result.json').write_bytes(result)
    readings=encoded(dict(schema='field-numeric-observations/1',readings=[dict(value='24.2',unit=None)],evidence='Evidence.json'))
    (event/'Readings.json').write_bytes(readings)
    unbound=encoded(dict(schema='field-unbound-readings/1',readings=[dict(value=None,raw_text='550 990')],evidence=folder+'/Evidence.json'))
    (event/'UnboundReadings.json').write_bytes(unbound)
    document=dict(job_id='fixture-job',capture_id='fixture-capture',model='original-ocr-pinned')
    receipt=encoded(dict(sequence=1,entity='jobs',entity_id='fixture-job',document=document))
    receipt_path='.System/Receipts/000001-'+sha(receipt)+'.json'
    (old/'.System/Receipts').mkdir(parents=True);(old/receipt_path).write_bytes(receipt)
    evidence=dict(schema='field-recognition-event/2',event_id='capture:fixture-capture',category='VoicePhotoReadings',
        camera_id='fixture-camera',sources=[dict(capture_id='fixture-capture',original=dict(path='MetaData/original.jpg',sha256='a'*64))],
        photos={'a'*64:dict(path='MetaData/original.jpg',sha256='a'*64)},observations=[document],
        measurement_files=[dict(instrument_id='fixture-instrument',result=folder+'/Result.json',sha256=sha(result))],
        readings=dict(path=folder+'/Readings.json',sha256=sha(readings)),
        unbound_readings=dict(path=folder+'/UnboundReadings.json',sha256=sha(unbound)),
        receipt_versions=[dict(sequence=1,receipt=receipt_path)])
    (event/'Evidence.json').write_bytes(encoded(evidence))
    (old/'.System/MigrationMap.json').write_bytes(encoded(dict(schema='field-recognition-paths/1',
        aliases={'Records/Jobs/fixture-job/Record.json':folder+'/Evidence.json'})))
    db=sqlite3.connect(tmp_path/'original-fr.sqlite3');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE archive_outbox(seq INTEGER PRIMARY KEY,entity TEXT,entity_id TEXT,document TEXT,receipt_path TEXT,archived_at REAL,published_at REAL,readable_at REAL)')
    db.execute('INSERT INTO archive_outbox VALUES(1,?,?,?,?,1,1,1)',('jobs','fixture-job',json.dumps(document),receipt_path));db.commit()

    @contextmanager
    def connection():yield db

    class Store:
        core={'db':connection}
        calls=0
        def configured_root(self):return old
        def step(self,batch_size=20):self.calls+=1

    manifest=dict(schema_version=SCHEMA,enabled=True,revision='fixture/1',source_archive_root=str(old),
        destination_root=str(destination),runtime_root=str(runtime),fr_package_root=str(tmp_path),
        initial_sequence=0,max_events_per_tick=1)
    yield dict(store=Store(),manifest=manifest,old=old,event=event,destination=destination,runtime=runtime,db=db,result=result,evidence=evidence)
    db.close()


def test_new_existing_fr_sqlite_and_result_owner_project_full_schema_without_raw_copy(source):
    s=source;sink=UnifiedFRSink(s['store'],s['manifest'])
    original={p:p.read_bytes() for p in s['old'].rglob('*') if p.is_file()}
    assert sink.step() is None
    actual=sink.last_projection;assert s['store'].calls==1
    event=actual['events'][0];assert event['status']=='projected'
    folder=s['destination']/event['directory']
    assert '/Comment/OCR/VoicePhotoReadings/' in str(folder)
    assert (folder/'Result.json').read_bytes()==s['result']
    evidence=json.loads((folder/'Evidence.json').read_bytes())
    assert evidence['observations']==s['evidence']['observations']
    assert evidence['measurement_files'][0]['sha256']==sha(s['result'])
    assert evidence['measurement_files'][0]['result']==event['directory']+'/Result.json'
    assert evidence['sources'][0]['original']['source_uri'].startswith('file:')
    assert evidence['sources'][0]['original']['raw_bytes_copied'] is False
    assert json.loads((folder/'UnboundReadings.json').read_bytes())['evidence']==event['directory']+'/Evidence.json'
    before={p:p.read_bytes() for p in folder.iterdir()}
    sink.project_once(sequence=1)
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert all(p.read_bytes()==raw for p,raw in original.items())
    assert all(p.suffix=='.json' for p in folder.iterdir())
    assert sink.project_once()['processed']==0


def test_new_unacknowledged_or_hash_changed_fr_member_cannot_publish_or_advance_cursor(source):
    s=source;sink=UnifiedFRSink(s['store'],s['manifest'])
    s['db'].execute('UPDATE archive_outbox SET readable_at=NULL');s['db'].commit()
    with pytest.raises(ValueError,match='not committed'):sink.project_once(sequence=1)
    s['db'].execute('UPDATE archive_outbox SET readable_at=1');s['db'].commit()
    (s['event']/'Result.json').write_bytes(b'changed-source-result')
    with pytest.raises(ValueError,match='hash differs'):sink.project_once()
    assert not sink.cursor.exists()
    assert not list(s['destination'].glob('*/Comment/OCR/*/*/Evidence.json'))


def test_new_guard_precedes_nas_metadata_and_original_worker_run_calls_facade(tmp_path,monkeypatch):
    import visioncortex.lab.field_unified_sink as sink_module
    import field_recognition.runtime.archive_worker as worker_module
    from pathlib import Path
    from types import SimpleNamespace

    manifest=dict(schema_version=SCHEMA,enabled=True,revision='fixture/1',
        source_archive_root='/mnt/fixture-unmounted-source/FieldRecognitionArchive',
        destination_root='/mnt/fixture-unmounted-destination/VisionCortexExperimentArchive',
        runtime_root=str(tmp_path),fr_package_root=str(tmp_path),initial_sequence=0,max_events_per_tick=1)
    path=tmp_path/'private.json';path.write_bytes(encoded(manifest));path.chmod(0o600)
    resolve=Path.resolve
    def reject_nas_metadata(p,*a,**kwargs):
        if str(p).startswith('/mnt/fixture-unmounted'):raise AssertionError('NAS metadata read before guard')
        return resolve(p,*a,**kwargs)
    calls=[]
    def mount_guard(*a,**kwargs):calls.append('mount_guard');raise OSError('unmounted fixture')
    with monkeypatch.context() as patch:
        patch.setattr(Path,'resolve',reject_nas_metadata)
        patch.setattr(sink_module,'verify_root',mount_guard)
        with pytest.raises(OSError,match='unmounted'):sink_module.load_manifest(path)
    assert calls==['mount_guard']

    class Store:
        core={}
        pending=2
        def configured_root(self):return tmp_path/'FieldRecognitionArchive'
        def snapshot(self):return dict(pending_receipts=self.pending)
        def step(self,batch_size=20):calls.append('original_step');self.pending-=1;return 13
        def next_poll_seconds(self):return 0
        def close(self):calls.append('close')
        integrity=SimpleNamespace(start=lambda:calls.append('integrity_start'))
    store=Store();manifest['source_archive_root']=str(tmp_path/'FieldRecognitionArchive')
    sink=UnifiedFRSink(store,manifest)
    monkeypatch.setattr(sink,'_config',lambda:{})
    monkeypatch.setattr(sink,'project_once',lambda:(calls.append('projection') or dict(processed=1)))
    assert sink.step()==13  # Even a store variant's integer return is preserved.
    core=dict(database=object(),archive_store=store,DATA=tmp_path)
    worker=worker_module.ArchiveWorker('fixture',SimpleNamespace(),core=core)
    worker.archive_store=sink
    monkeypatch.setattr(worker_module,'heartbeat',lambda *a,**k:None)
    worker.run(persistent=False)
    assert calls.count('original_step')==calls.count('projection')==2
    assert store.pending==0 and sink.snapshot()['unified_projection']['processed']==1
    assert 'integrity_start' in calls and 'close' in calls
