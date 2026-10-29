"""Explicit operator declarations on an opt-in isolated PostgreSQL fixture only."""
from copy import deepcopy
from uuid import UUID,uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from visioncortex.lab.domain import Conflict,Core,Missing
from visioncortex.lab.operator_scan import OperatorScans
from visioncortex.lab.timeline_commands import command_methods
from visioncortex.lab.timeline_runtime import TimelineRuntime


def at(value):return '2026-10-29T'+value+'+08:00'


@pytest.fixture
def scene(core,settings):
    runtime=TimelineRuntime(core,settings)
    persons=[core.register('p'+str(i),'person','fixture-person-'+str(i))['id'] for i in range(2)]
    cameras=[core.register('c'+str(i),'capture_device','fixture-camera-'+str(i),{'view_role':'first_person'})['id'] for i in range(2)]
    devices=[]
    for i in range(2):
        d=core.issue_equipment('reserve'+str(i),actor='synthetic fixture')
        core.activate_equipment('activate'+str(i),d['device_id'],verified_at=at('08:00:00'),actor='synthetic fixture',evidence={'fixture':True})
        devices.append(d['device_id'])
    lots=[core.register('lot'+str(i),'source_lot','fixture-source-'+str(i))['id'] for i in range(2)]
    materials=[core.register_material('m'+str(i),'fixture-material-'+str(i),'g',None,metadata={'source_lot_id':lots[i]})['material_id'] for i in range(2)]
    batches=[core.register('batch'+str(i),'process_batch','fixture-batch-'+str(i))['id'] for i in range(2)]
    tag=core.register('tag','tag','fixture-reusable-label',{'mode':'reusable'})['id']
    core.assign_tag('tag1',tag,materials[0],at('08:00:00'),actor='fixture',reason='initial')
    run=runtime.timeline.create_run('run','fixture-run','synthetic',materials,batch_id=batches[0])['id']
    return dict(core=core,runtime=runtime,ops=runtime.operator_scans,people=persons,cameras=cameras,devices=devices,
                lots=lots,materials=materials,batches=batches,tag=tag,run=run)


def scan(s,key='scan',*,when='09:00:00',kind='equipment',value=None,person=0,camera=0,**extra):
    body=dict(intent='operator_scan',scan_kind=kind,qr_value=value or s['devices'][0],
              occurred_at=at(when) if when else None,evidence_ids=[],actor='synthetic operator',
              reason='explicit fixture scanner action',person_id=s['people'][person],camera_id=s['cameras'][camera])
    body.update(extra)
    return command_methods(s['runtime'])['timeline-operator-scan'](key,**body)


def context(s,when='09:30:00',person=0,camera=0):
    return command_methods(s['runtime'])['timeline-operator-context'](as_of=at(when) if when else None,
              person_id=s['people'][person],camera_id=s['cameras'][camera])


def rows(s,table):
    with s['core'].db.transaction(readonly=True) as conn:
        return conn.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE customer_id=%s ORDER BY to_jsonb(t)::text',(s['core'].customer,)).fetchall()


def test_device_scan_immediately_declares_context_without_experiment_or_material(scene):
    result=scan(scene)
    assert result['status']=='declared_equipment'
    c=result['context'];assert c['equipment_id']==scene['devices'][0] and c['status']=='needs_material_context'
    assert c['run_id'] is c['batch_id'] is None and not c['materials']
    assert c['panel_ocr_attribution']=='not_supplied_by_operator_current_context'
    assert not c['wearer_or_speaker_inferred']


def test_A_to_B_inherits_explicit_run_batch_material_without_ending_A(scene):
    before=rows(scene,'operation');run_before=rows(scene,'hj_experiment_record')
    scan(scene,'A',run_id=scene['run'],when='09:00:00')
    scan(scene,'material',kind='material',value='fixture-reusable-label',run_id=scene['run'],when='09:01:00')
    result=scan(scene,'B',value=scene['devices'][1],when='09:02:00')
    c=result['context']
    assert (c['equipment_id'],c['run_id'],c['batch_id'])==(scene['devices'][1],scene['run'],scene['batches'][0])
    assert c['materials'][0]['resolution']['material_unit']['id']==scene['materials'][0]
    assert c['materials'][0]['resolution']['source_lot']['id']==scene['lots'][0]
    assert c['switches'][-1]['from_equipment_id']==scene['devices'][0]
    assert not c['switches'][-1]['ends_previous_experiment']
    after=rows(scene,'hj_experiment_record')[0]['row']
    assert after['record_status']==run_before[0]['row']['record_status']=='open'
    assert after['closed_at'] is None and after['closure_reason'] is None
    assert rows(scene,'operation')==before and rows(scene,'quantity_ledger_entry')==[]


def test_explicit_new_batch_changes_association_scope_and_preserves_history(scene):
    scan(scene,'A',run_id=scene['run'])
    scan(scene,'M',kind='material',value='fixture-reusable-label',batch_id=scene['batches'][0],when='09:01:00')
    scan(scene,'B',value=scene['devices'][1],batch_id=scene['batches'][1],when='09:02:00')
    current=context(scene)
    assert current['batch_id']==scene['batches'][1] and current['run_id'] is None and current['materials']==[]
    assert len(context(scene,'09:01:30')['materials'])==1


def test_material_scan_without_any_prior_target_stays_pending(scene):
    scan(scene,'A')
    result=scan(scene,'M',kind='material',value='fixture-reusable-label',when='09:01:00')
    assert result['status']=='needs_material_context'
    c=context(scene)
    assert c['materials']==[] and len(c['pending_materials'])==1
    assert c['pending_materials'][0]['run_id'] is None
    assert rows(scene,'quantity_ledger_entry')==[]


def test_late_reused_label_resolves_event_history_and_pins_source_lot_snapshot(scene):
    core=scene['core']
    core.assign_tag('reuse',scene['tag'],scene['materials'][1],at('10:00:00'),actor='fixture',reason='reused label')
    new=scan(scene,'new',kind='material',value='fixture-reusable-label',when='10:05:00',batch_id=scene['batches'][0])
    old=scan(scene,'late',kind='material',value='fixture-reusable-label',when='09:05:00',batch_id=scene['batches'][0])
    a=old['snapshot']['resolution'];b=new['snapshot']['resolution']
    assert a['material_unit']['id']==scene['materials'][0] and a['assignment_revision']['version']==1
    assert b['material_unit']['id']==scene['materials'][1] and b['assignment_revision']['version']==2
    with core.db.transaction() as conn:
        conn.execute('UPDATE entity SET metadata=%s WHERE customer_id=%s AND id=%s',
                     (Jsonb({'source_lot_id':scene['lots'][1]}),core.customer,UUID(scene['materials'][0])))
    c=context(scene,'09:06:00')
    assert c['materials'][0]['resolution']['source_lot']['id']==scene['lots'][0]


def test_late_insertion_does_not_overwrite_later_or_other_scope(scene):
    scan(scene,'other',value=scene['devices'][1],person=1,camera=1,run_id=scene['run'],when='09:05:00')
    other=deepcopy(context(scene,person=1,camera=1))
    scan(scene,'later',value=scene['devices'][1],run_id=scene['run'],when='09:05:00')
    scan(scene,'late',value=scene['devices'][0],batch_id=scene['batches'][1],when='09:02:00')
    assert context(scene)['equipment_id']==scene['devices'][1]
    assert context(scene)['batch_id']==scene['batches'][0]
    assert context(scene,person=1,camera=1)==other
    assert context(scene,person=0,camera=1)['equipment_id'] is None


def test_unknown_time_never_inherits_current_person_equipment_or_batch(scene):
    scan(scene,'known',run_id=scene['run'])
    result=scan(scene,'unknown',value=scene['devices'][1],when=None)
    assert result['status']=='unknown_time' and result['context']['equipment_id'] is None
    assert context(scene,None)['equipment_id'] is None and context(scene)['equipment_id']==scene['devices'][0]


@pytest.mark.parametrize('intent',['background_qr','pixel_observation',None])
def test_background_pixel_producer_cannot_claim_operator_intent(scene,intent):
    with pytest.raises(ValueError,match='explicit operator_scan'):
        scan(scene,intent=intent)
    assert rows(scene,'hj_operator_scan_event')==[]


def test_retired_equipment_accepts_historical_active_scan_but_rejects_after_retirement(scene):
    core=scene['core']
    core.retire_equipment('retire',scene['devices'][0],at=at('10:00:00'),actor='fixture',reason='fixture retirement')
    old=scan(scene,'old',when='09:05:00')
    assert old['context']['equipment_id']==scene['devices'][0]
    new=scan(scene,'new',when='10:05:00')
    assert new['context']['equipment_id'] is None
    assert context(scene,'09:06:00')['equipment_id']==scene['devices'][0]


def test_reserved_or_material_code_cannot_be_declared_active_equipment(scene):
    d=scene['core'].issue_equipment('unactivated',actor='fixture')
    assert scan(scene,'reserved',value=d['device_id'])['context']['equipment_id'] is None
    assert scan(scene,'material-as-device',value='fixture-reusable-label',when='09:02:00')['context']['equipment_id'] is None


def test_idempotency_emits_once_and_different_body_conflicts(scene):
    result=scan(scene)
    assert scan(scene)==result
    assert len(rows(scene,'hj_operator_scan_event'))==1
    emitted=[r for r in rows(scene,'outbox') if r['row']['kind']=='operator_scan']
    assert len(emitted)==1
    with pytest.raises(Conflict):scan(scene,value=scene['devices'][1])


@pytest.mark.parametrize('mutation',['UPDATE hj_operator_scan_event SET event_key=event_key','DELETE FROM hj_operator_scan_event'])
def test_database_events_are_append_only(scene,mutation):
    scan(scene)
    with pytest.raises(psycopg.errors.RaiseException,match='append-only'):
        with scene['core'].db.transaction() as conn:
            conn.execute(mutation+' WHERE customer_id=%s',(scene['core'].customer,))


def test_run_batch_conflict_does_not_write_scan_or_material_ledger(scene):
    with pytest.raises(Conflict):scan(scene,run_id=scene['run'],batch_id=scene['batches'][1])
    assert rows(scene,'hj_operator_scan_event')==rows(scene,'quantity_ledger_entry')==[]


def test_simultaneous_conflicting_device_declarations_do_not_choose_lexical_winner(scene):
    scan(scene,'A');scan(scene,'B',value=scene['devices'][1])
    c=context(scene)
    assert c['equipment_id'] is None and c['identity_status']=='ambiguous_or_unresolved_operator_scan'


def test_cross_customer_and_missing_scope_fail_closed(scene):
    other=Core(scene['core'].db,Core.customer_instance(scene['core'].db,'operator-other-'+str(uuid4()),scope='fixture')['id'])
    with pytest.raises(Missing):
        OperatorScans(other).context(as_of=at('09:05:00'),person_id=scene['people'][0])
    with pytest.raises(ValueError):scene['ops'].context(as_of=at('09:05:00'))
    with pytest.raises(TypeError):scene['ops'].context(person_id=scene['people'][0])


def test_explicit_different_run_resets_material_scope_even_when_batch_is_same(scene):
    scan(scene,'A',run_id=scene['run'])
    scan(scene,'M',kind='material',value='fixture-reusable-label',run_id=scene['run'],when='09:01:00')
    other=scene['runtime'].timeline.create_run('run2','fixture-run2','synthetic',scene['materials'],batch_id=scene['batches'][0])['id']
    scan(scene,'B',value=scene['devices'][1],run_id=other,when='09:02:00')
    assert context(scene)['run_id']==other and context(scene)['materials']==[]
    assert len(context(scene,'09:01:30')['materials'])==1


def test_scan_declarations_reach_existing_trace_and_report_once(scene):
    result=scan(scene,'A',run_id=scene['run'])
    scan(scene,'M',kind='material',value='fixture-reusable-label',run_id=scene['run'],when='09:01:00')
    scan(scene,'B',value=scene['devices'][1],when='09:02:00')
    assert scan(scene,'A',run_id=scene['run'])==result
    trace=scene['runtime'].views.trace(at('08:59:00'),at('09:03:00'))
    assert scene['devices'][0] in str(trace) and scene['devices'][1] in str(trace)
    assert scene['lots'][0] in str(trace) and result['event_id'] in str(trace)
    scene['runtime'].views.refresh_dirty()
    report=scene['runtime'].views.snapshot('2026-10-29')
    assert 'operator_scan_event' in str(report) and scene['lots'][0] in str(report)
    activities=[r for r in rows(scene,'hj_activity') if 'operator_scan_event' in r['row']['payload']]
    assert len(activities)==3


def test_material_scan_inherits_only_exact_scope_prior_target_and_reaches_report(scene):
    device=scan(scene,'target-A',run_id=scene['run'])
    result=scan(scene,'material-inherit',kind='material',value='fixture-reusable-label',when='09:01:00')
    snapshot=result['snapshot']
    assert result['status']=='declared_material'
    assert (snapshot['run_id'],snapshot['batch_id'])==(scene['run'],scene['batches'][0])
    assert snapshot['target_selection']['kind']=='inherited_exact_scope_event_time_target'
    assert snapshot['target_selection']['source_scan_event_ids']==[device['event_id']]
    assert not snapshot['run_explicit'] and not snapshot['batch_explicit']
    other=scan(scene,'other-scope-material',kind='material',value='fixture-reusable-label',when='09:01:00',person=1)
    assert other['status']=='needs_material_context'
    scene['runtime'].views.refresh_dirty()
    report=scene['runtime'].views.snapshot('2026-10-29')
    assert result['event_id'] in str(report) and device['event_id'] in str(report)
    assert rows(scene,'quantity_ledger_entry')==[]


def test_material_inherited_target_snapshot_is_not_rehung_by_late_or_explicit_new_target(scene):
    scan(scene,'target-A',run_id=scene['run'])
    first=scan(scene,'material-fixed',kind='material',value='fixture-reusable-label',when='09:02:00')
    frozen=deepcopy(first['snapshot'])
    # Received later but physically before the material: history changes, stored selection does not.
    other=scene['runtime'].timeline.create_run('late-run','fixture-late-run','synthetic',scene['materials'],batch_id=scene['batches'][0])['id']
    scan(scene,'late-new-target',value=scene['devices'][1],when='09:01:00',run_id=other)
    assert context(scene)['run_id']==other and context(scene)['materials']==[]
    second=scan(scene,'explicit-target',kind='material',value='fixture-reusable-label',when='09:03:00',run_id=scene['run'])
    assert second['snapshot']['target_selection']['kind']=='explicit_target'
    assert second['snapshot']['batch_id']==scene['batches'][0]
    assert scan(scene,'material-fixed',kind='material',value='fixture-reusable-label',when='09:02:00')['snapshot']==frozen
    stored=[r['row']['payload'] for r in rows(scene,'hj_operator_scan_event') if r['row']['id']==first['event_id']]
    assert stored==[frozen] and rows(scene,'quantity_ledger_entry')==[]
