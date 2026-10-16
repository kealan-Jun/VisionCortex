"""Source-bound admission survives age/restart without opening paid history."""
import json
import sqlite3
import time
from types import SimpleNamespace

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.observed_inventory import observe
from test_device_day_retention_worker import record
from test_device_day import device_config as device_config_fixture

device_config = device_config_fixture


def test_old_schema_migration_does_not_grant_historical_admission(tmp_path):
    path = tmp_path/'queue-vision.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE recordings (recording_id TEXT PRIMARY KEY, revision TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL, queued_at REAL NOT NULL, updated_at REAL NOT NULL, lease_owner TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0, result TEXT, completed_at REAL, wall_seconds REAL)')
        db.execute('INSERT INTO recordings VALUES(?,?,?,?,?,?,NULL,NULL,3,NULL,NULL,NULL)',
                   ('old','v',json.dumps(record('old',1)),'failed',1,1))
    queue = DeviceDayQueue(path)
    assert queue.admissions() == {}
    with queue.connect() as db:
        assert db.execute('SELECT attempts,admitted_at FROM recordings').fetchone()[:] == (3,None)


def test_admitted_source_remains_eligible_after_four_hours_and_restart(tmp_path):
    now = time.time()
    source = record('admitted',int((now-5*3600)*1e6))
    history = record('never-admitted',int((now-24*3600)*1e6),camera='b')
    root=tmp_path/'device-day'
    queue=DeviceDayQueue(root/'queue-vision.sqlite3')
    observe(root, {'recordings':[source,history]})
    queue.enqueue(source,'same-key')
    queue.enqueue(history,'old-key')
    assert queue.grant_admission(source,at=now-5*3600)
    restarted=DeviceDayQueue(queue.path)
    runner=SimpleNamespace(runtime_root=root,queues={'vision':restarted},settings={},config={'collection_ingest':{}})
    chosen=RetentionWorker(stage='vision',recent_seconds=14400).candidates(runner)
    assert [r['recording_id'] for r in chosen] == ['admitted']
    revised=source|{'source_signature':'changed-content'}
    observe(root,{'recordings':[revised]})
    assert RetentionWorker(stage='vision',recent_seconds=14400).candidates(runner)==[]


def test_unknown_external_result_is_held_across_lease_and_revision(tmp_path):
    queue=DeviceDayQueue(tmp_path/'queue-understanding.sqlite3')
    item=record('unit',1)
    queue.enqueue(item,'v1')
    queue.claim('owner')
    queue.finish('owner','unit',{'status':'submitted_unknown','request_key':'paid-key'},1)
    queue.enqueue(item,'v2')
    assert queue.claim('restart',retry=True,max_attempts=3) is None
    assert queue.snapshot()['blocked_reasons']=={'submitted_unknown':1}
    assert not queue.release_hold('unit',reason='other',evidence='review')
    assert queue.release_hold('unit',reason='submitted_unknown',evidence='provider-receipt-verified')
    assert queue.claim('reviewed')['recording_id']=='unit'


def test_explicit_paid_history_uses_only_source_bound_admitted_work(tmp_path):
    source = record('historical', int((time.time() - 24 * 3600) * 1e6))
    root = tmp_path / 'device-day'
    queues = {stage: DeviceDayQueue(root / f'queue-{stage}.sqlite3')
              for stage in ('retention', 'vision', 'stt', 'understanding')}
    observe(root, {'recordings': [source]})
    for stage, queue in queues.items():
        queue.enqueue(source, 'v')
        if stage != 'understanding':
            queue.claim('parent')
            queue.finish('parent', source['recording_id'], {'status': 'completed'}, 1)
    queue = queues['understanding']
    queue.grant_admission(source, cohort='historical_backfill')
    assert queue.admissions() == {}
    assert queue.admissions(include_history=True) == {source['recording_id']: source['source_signature']}
    settings = {'understanding_coverage_since_us': 1,
                'backfill': {'enabled': True, 'stages': ['understanding'], 'paid_history_authorized': False}}
    runner = SimpleNamespace(runtime_root=root, queues=queues, settings=settings, config={'collection_ingest': {}})
    worker = RetentionWorker(stage='understanding', recent_seconds=14400)
    assert worker.candidates(runner) == []
    settings['backfill']['paid_history_authorized'] = True
    assert [row['recording_id'] for row in worker.candidates(runner)] == [source['recording_id']]
    observe(root, {'recordings': [source | {'source_signature': 'changed'}]})
    assert worker.candidates(runner) == []


def test_execution_revision_keeps_the_previous_attempts_and_failure(tmp_path):
    queue = DeviceDayQueue(tmp_path / 'queue-understanding.sqlite3')
    source = record('revision', 1)
    queue.enqueue(source, 'old-recipe')
    queue.claim('owner')
    queue.finish('owner', source['recording_id'], {'status': 'failed', 'reason': 'saved-failure'}, 1)
    queue.enqueue(source, 'new-recipe')
    with queue.connect() as db:
        old = json.loads(db.execute('SELECT snapshot FROM source_versions').fetchone()[0])
    assert old['revision'] == 'old-recipe' and old['attempts'] == 1
    assert json.loads(old['result'])['reason'] == 'saved-failure'


def test_cost_shares_devices_and_age_rotate_checkpoint_work(tmp_path,monkeypatch):
    clock=[1000.]
    monkeypatch.setattr('visioncortex.device_day_queue.time.time',lambda:clock[0])
    options={'enabled':True,'max_ready_wait_seconds':10,'resource_shares':{'experiment_candidate':4,'support_activity':1}}
    queue=DeviceDayQueue(tmp_path/'queue-understanding.sqlite3',latest_first=True,scheduling=options)
    a=record('a',2,camera='a',activity_kind='experiment_candidate',estimated_processing_seconds=5000)
    b=record('b',1,camera='b',activity_kind='support_activity',estimated_processing_seconds=5000)
    for item in (a,b):
        queue.enqueue(item,'v')
        queue.grant_admission(item,at=100)
    assert queue.claim('one')['recording_id']=='a'
    queue.finish('one','a',{'status':'scheduled_checkpoint'},40)
    clock[0]=1001
    assert queue.claim('two')['recording_id']=='b'
    queue.finish('two','b',{'status':'scheduled_checkpoint'},1)
    clock[0]=1020
    assert queue.claim('three')['recording_id']=='a'
    assert queue.snapshot()['overload_status']=='OVERLOADED'


def test_historical_admission_remains_separate_from_live_recovery(tmp_path):
    queue=DeviceDayQueue(tmp_path/'queue-understanding.sqlite3')
    item=record('history',1)
    queue.enqueue(item,'v')
    queue.grant_admission(item,cohort='historical_backfill',reason='explicit_history_authorization')
    assert queue.admissions()=={}
    with queue.connect() as db:
        assert db.execute('SELECT processing_cohort FROM recordings').fetchone()[0]=='historical_backfill'


def test_visual_only_stage_identity_excludes_unused_late_context(device_config):
    from visioncortex.device_day import DeviceDayRunner
    from test_device_day import FakeModels, capture, item_and_layout
    capture(device_config)
    item,_=item_and_layout(device_config)
    device_config['device_day']['understanding_coverage_since_us']=item['recording_start_us']-1
    runner=DeviceDayRunner(device_config,FakeModels())
    inputs={'vision':{'key':'same-real-visual-result'},'stt':{'key':'before'},'context':{'comments':[]}}
    original=runner._key('understanding',item,inputs)
    newer=inputs|{'stt':{'key':'after'},'context':{'comments':[{'text':'late actual speech'}]}}
    assert runner._key('understanding',item,newer)==original
    assert runner._key('understanding',item,[inputs,{'protocol':'before'}])==runner._key('understanding',item,[newer,{'protocol':'after'}])


def test_visual_observation_dependencies_do_not_wait_for_stt(device_config):
    from visioncortex.device_day_contract import dependencies_for
    item=record('video',100)
    device_config['device_day']['understanding_coverage_since_us']=1
    assert dependencies_for(device_config['device_day'],item)['understanding']==('vision',)
    assert dependencies_for({},item)['understanding']==('vision','stt')
