"""Synthetic business/codec fixtures; no real OCR or stock quality claim."""
from dataclasses import replace
from decimal import Decimal
import hashlib
import json

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from visioncortex.lab.api import create_app
from visioncortex.lab.domain import Conflict
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.timeline_runtime import TimelineRuntime
from visioncortex.lab.weighing import Weighing


def at(second):
    return f'2026-10-29T09:{second//60:02d}:{second%60:02d}+08:00'


@pytest.fixture
def s(core, settings, tmp_path):
    policy = tmp_path/'policy.json'
    policy.write_text(json.dumps(dict(schema_version='vc-lab-timeline-policy/1',
        weighing=dict(enabled=True, revision='fixture-only/1', max_activities_per_tick=4))))
    settings = replace(settings, processing_profile='timeline-v1', timeline_policy_path=policy)
    runtime = TimelineRuntime(core, settings)
    camera = core.register('camera', 'capture_device', 'fixture-weighing-camera')['id']
    device = core.issue_equipment('device', actor='fixture')['device_id']
    core.activate_equipment('activate', device, verified_at=at(0), actor='fixture', evidence={'synthetic': True})
    lot = core.register('lot', 'source_lot', 'fixture-source-lot')['id']
    material = core.register_material('material', 'fixture-material', 'g', '100',
                                      metadata={'source_lot_id':lot}, occurred_at=at(0))['material_id']
    sample = core.register_material('sample', 'fixture-target-sample', 'g', None)['material_id']
    batch = core.register('batch', 'process_batch', 'fixture-target-batch')['id']
    tag = core.register('material-tag', 'tag', 'fixture-material-QR', {'mode':'reusable'})['id']
    core.assign_tag('assignment', tag, material, at(0), actor='fixture', reason='generated label history')
    run = runtime.timeline.create_run('run', 'fixture-weighing-run', 'synthetic', [sample], batch_id=batch)['id']
    for kind, value, second in [('equipment', device, 1), ('material', 'fixture-material-QR', 2)]:
        runtime.operator_scans.record('scan-'+kind, intent='operator_scan', scan_kind=kind,
            qr_value=value, occurred_at=at(second), evidence_ids=[], actor='fixture', reason='generated explicit declaration',
            camera_id=camera, run_id=run)
    return dict(core=core, settings=settings, runtime=runtime, camera=camera, device=device,
                material=material, sample=sample, lot=lot, batch=batch, tag=tag, run=run,
                weighing=Weighing(core), root=tmp_path, media={})


def photo(s, key, second, value, *, iid=None, clock=True, issues=None, registered=True):
    p=s['settings'].incoming/(key+'.png')
    seed=hashlib.sha256(key.encode()).digest()
    Image.new('RGB',(16,16),tuple(seed[:3])).save(p)
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    capture=Ingest(s['core'],s['settings']).accept(dict(capture_id=key,source_instance='fixture-camera-source',
        source_revision='fixture/1', camera_id='fixture-weighing-camera', session_id='fixture-source-session',
        kind='photo',relative_path=p.name,captured_at=at(second),sha256=sha,source_kind='fixture',
        time_basis={'basis':'synthetic_fixture_clock','uncertainty_us':0} if clock else {'basis':'uncalibrated'},
        start_seconds='0',end_seconds='0'))
    iid=iid or s['device']
    registration=dict(schema_version='vc-lab-ocr-frame-registration/1', status='verified',
        source_sha256=sha,layout_revision='fixture-layout/1',instrument_id=iid,
        regions=[dict(field_key='mass',role='displayed_value',box=['0','0','1','1'],
                      unit='g',unit_basis='verified_on_panel')])
    if registered:
        s['runtime'].readouts.register_frame('register-'+key,capture['media_id'],registration,actor='fixture',reason='generated panel identity')
    ids=[]
    for n in range(2 if registered else 1):
        task=s['runtime'].tasks.claim('claim-'+key+str(n),'ocr','synthetic-fixture-no-inference/1')
        assert task['request']['primary_media_id']==capture['media_id']
        result=dict(schema_version='vc-lab-timeline-model-result/1', input_fingerprint=task['request']['input_fingerprint'],covered_window_indices=[0],
            events=[dict(kind='reading',text='synthetic fixture value',instrument_id=iid if registered else None,
                spans=[dict(media_id=capture['media_id'],start_seconds='0',end_seconds='0')],
                fields=[dict(field_key='mass',role='displayed_value',value=value,unit='g',
                    unit_basis='verified_on_panel',status='candidate',issues=issues or [])])])
        submitted=s['runtime'].tasks.submit('submit-'+key+str(n),task['task_id'],task['lease_token'],
                                           'synthetic-fixture-no-inference/1',result)
        ids.extend(submitted['activity_ids'])
    s['media'][key]=capture
    return ids[0]


def start(s, baseline, **extra):
    return s['weighing'].start('start-'+baseline,baseline,'mass',actor='fixture',reason='generated baseline',**extra)


def observe(s, head, reading, **extra):
    return s['weighing'].observe('observe-'+reading,head['session_id'],reading,
        actor='fixture',reason='generated weighing observation',**extra)


def balance(s):
    return Decimal(s['core'].balance(s['material'])['balance'])


def test_photos_update_one_episode_and_duplicates_do_not_sum_totals(s):
    h=start(s,photo(s,'tare',3,'0.0000'))
    a=photo(s,'two',4,'2.0000');first=observe(s,h,a)
    assert balance(s)==98 and Decimal(first['quantity'])==2
    assert observe(s,h,a)==first and balance(s)==98
    repeated=observe(s,h,photo(s,'two-again',5,'2.0000'))
    assert repeated['status']=='same_total_no_new_debit' and balance(s)==98
    final=observe(s,h,photo(s,'three',6,'3.0000'))
    assert Decimal(final['delta_used'])==1 and balance(s)==97
    with s['core'].db.transaction(readonly=True) as c:
        rows=c.execute("SELECT * FROM operation WHERE customer_id=%s AND kind<>'opening' ORDER BY created_at",
                       (s['core'].customer,)).fetchall()
        assert len(rows)==2 and rows[0]['status']=='corrected'
        assert len(rows[-1]['evidence_ids'])==3
    assert Decimal(s['weighing'].get(h['session_id'])['quantity'])==3


def test_difference_uses_recorded_baseline_and_never_the_gross_total(s):
    h=start(s,photo(s,'gross-baseline',3,'10.0000'),mode='difference')
    result=observe(s,h,photo(s,'gross-final',4,'12.3500'))
    assert Decimal(result['quantity'])==Decimal('2.3500') and balance(s)==Decimal('97.6500')


def test_explicit_correction_preserves_old_operation_and_restores_only_difference(s):
    h=start(s,photo(s,'tare',3,'0'))
    first=observe(s,h,photo(s,'two',4,'2'))
    correction=photo(s,'corrected',5,'1.5')
    result=observe(s,h,correction,correction=True,expected_sequence=first['sequence'])
    assert result['status']=='stock_corrected' and balance(s)==Decimal('98.5')
    assert Decimal(result['delta_used'])==Decimal('-.5')
    with pytest.raises(Conflict,match='changed'):
        s['weighing'].observe('stale',h['session_id'],correction,actor='fixture',reason='stale correction',
                              correction=True,expected_sequence=first['sequence'])


def test_decrease_does_not_refund_and_blocks_automatic_following_increase(s):
    h=start(s,photo(s,'tare',3,'0'))
    observe(s,h,photo(s,'two',4,'2'))
    decrease=observe(s,h,photo(s,'one',5,'1'))
    assert decrease['state']=='needs_attention' and balance(s)==98
    pending=observe(s,h,photo(s,'three',6,'3'))
    assert pending['status']=='episode_requires_correction' and balance(s)==98


def test_explicit_zero_correction_reverses_the_ledger_and_closes_only_this_weighing(s):
    h=start(s,photo(s,'tare',3,'0'))
    previous=observe(s,h,photo(s,'two',4,'2'))
    zero=photo(s,'verified-zero-correction',5,'0')
    result=observe(s,h,zero,correction=True,expected_sequence=previous['sequence'])
    assert result['state']=='closed' and Decimal(result['quantity'])==0 and balance(s)==100
    assert Decimal(s['core'].balance(s['material'])['used'])==0
    with s['core'].db.transaction(readonly=True) as c:
        assert c.execute('SELECT record_status FROM hj_experiment_record WHERE customer_id=%s AND id=%s',
                         (s['core'].customer,s['run'])).fetchone()['record_status']=='open'


def test_insufficient_stock_rolls_back_the_whole_quantity_update(s):
    h=start(s,photo(s,'tare',3,'0'))
    with pytest.raises(Conflict,match='Insufficient'):
        observe(s,h,photo(s,'too-large',4,'101'))
    assert balance(s)==100 and s['weighing'].get(h['session_id'])['sequence']==0


def test_zero_requires_evidence_new_episode_same_value_is_a_distinct_use(s):
    baseline=photo(s,'tare',3,'0');h=start(s,baseline)
    final=observe(s,h,photo(s,'two',4,'2'))
    s['weighing'].close('close',h['session_id'],expected_sequence=final['sequence'],actor='fixture',reason='finished this weighing')
    next_head=start(s,photo(s,'tare-next',5,'0'))
    observe(s,next_head,photo(s,'two-next',6,'2'))
    assert balance(s)==96 and h['session_id']!=next_head['session_id']
    assert s['runtime'].timeline.core is s['core']


@pytest.mark.parametrize('bad', ['clock','mapping','conflict','nonzero_tare'])
def test_missing_clock_panel_proof_or_readability_never_debits(s,bad):
    with pytest.raises((Conflict,ValueError)):
        start(s,photo(s,'bad',3,'1' if bad=='nonzero_tare' else '0',clock=bad!='clock',
            registered=bad!='mapping',issues=['ocr_view_conflict'] if bad=='conflict' else None))
    assert balance(s)==100


def test_reused_material_label_cannot_move_an_existing_episode_to_new_stock(s):
    h=start(s,photo(s,'tare',3,'0'))
    other=s['core'].register_material('other','fixture-material-next','g','50',metadata={'source_lot_id':s['lot']},occurred_at=at(0))['material_id']
    s['core'].assign_tag('reuse',s['tag'],other,at(4),actor='fixture',reason='reuse physical label')
    s['runtime'].operator_scans.record('scan-new',intent='operator_scan',scan_kind='material',qr_value='fixture-material-QR',
        occurred_at=at(5),evidence_ids=[],actor='fixture',reason='new content',camera_id=s['camera'],run_id=s['run'])
    with pytest.raises(Conflict,match='history changed'):
        observe(s,h,photo(s,'new-material',6,'2'))
    assert balance(s)==100 and Decimal(s['core'].balance(other)['balance'])==50
    assert s['weighing'].get(h['session_id'])['context']['material_id']==s['material']


def test_authenticated_API_and_same_runtime_use_the_existing_ledger(s):
    baseline=photo(s,'tare',3,'0');value=photo(s,'two',4,'2')
    client=TestClient(create_app(s['core'],s['settings']))
    body=dict(key='api-start',baseline_activity_id=baseline,field_key='mass',actor='fixture',reason='generated API')
    path='/internal/commands/timeline-start-weighing'
    assert client.post(path,json=body).status_code==401
    headers={'Authorization':'Bearer '+s['settings'].token}
    response=client.post(path,json=body,headers=headers);assert response.status_code==200,response.text
    sid=response.json()['session_id']
    result=client.post('/internal/commands/timeline-observe-weighing',json=dict(key='api-observe',session_id=sid,
        activity_id=value,actor='fixture',reason='generated API'),headers=headers)
    assert result.status_code==200,result.text
    assert balance(s)==98


def test_newest_first_intake_can_reuse_earlier_zero_without_repeating_OCR(s):
    photo(s,'tare',3,'0.0000');value=photo(s,'two',4,'2.0000')
    config={'enabled':True,'revision':'fixture-only/1','max_activities_per_tick':4}
    first=s['runtime'].weighing_intake.consume(value,config)
    assert first['status']=='processed' and balance(s)==98
    assert s['runtime'].weighing_intake.consume(value,config)['already_recorded'] is True
    s['runtime'].weighing_intake.tick()
    assert balance(s)==98


def test_weighing_reaches_existing_report_and_shallow_Comment_bindings(s):
    from visioncortex.lab.archive_layout import contract_bytes
    from visioncortex.lab.comment_presentation import CommentPresentationPublication
    from visioncortex.lab.timeline_report import render_report
    h=start(s,photo(s,'tare',3,'0'))
    observe(s,h,photo(s,'two',4,'2.3500'))
    snapshot=s['runtime'].views.snapshot('2026-10-29')
    assert snapshot['document']['context_evidence']['weighings']['episodes'][0]['quantity']=='2.3500'
    assert '累计用量 2.3500 g' in render_report(snapshot['document'])
    root=s['settings'].incoming
    (root/'.System').mkdir();(root/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    publication_settings=replace(s['settings'],source_root=root,unified_archive_required=True,timeline_publish=True)
    s['runtime'].publication=CommentPresentationPublication(s['runtime'],publication_settings,history=s['root']/'presentation-history')
    result=s['runtime'].publication.tick()
    assert result
    path=root/'2026-10-29_fixture-weighing-camera/Comment/Bindings.json'
    value=json.loads(path.read_text())
    assert value['material_quantity_written'] is True
    assert value['用料记录'][0]['累计用量']=='2.3500'
    assert value['用料记录'][0]['来源批次']=='fixture-source-lot'
    assert value['用料记录'][0]['目标样品']==['fixture-target-sample']
    assert value['用料记录'][0]['原图']=='two.png'
    assert value['schema_version']=='vc-lab-camera-bindings/1'
    assert (root/value['immutable_history']['relative_path']).is_file()
