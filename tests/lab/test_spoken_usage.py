"""Generated evidence validates accounting, never real ASR/QR recognition quality."""
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import wave

import pytest

from visioncortex.lab.domain import Conflict
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.speech_candidates import parse_usage_statements
from visioncortex.lab.spoken_usage import SpokenUsage
from .test_weighing import s as weighing_fixture, photo, start, observe, balance, at

POLICY=dict(enabled=True,revision='fixture-oral-use/1',max_activities_per_tick=4,post_actual_declarations=True)


@pytest.fixture
def oral(core,settings,tmp_path):
    return weighing_fixture.__wrapped__(core,settings,tmp_path)


def speech(s,text,key='spoken',clock=True,source=None,lo='0',hi='2'):
    if source is None:
        path=s['settings'].incoming/(key+'.wav')
        with wave.open(str(path),'wb') as stream:
            stream.setparams((1,2,8000,0,'NONE','not compressed'))
            stream.writeframes(hashlib.sha256(key.encode()).digest()*10000)
        source=Ingest(s['core'],s['settings']).accept(dict(capture_id=key,source_instance='fixture-camera-source',
            source_revision='fixture/1',camera_id='fixture-weighing-camera',kind='audio',relative_path=path.name,
            captured_at=at(4),source_kind='fixture',time_basis={'basis':'synthetic_fixture_clock','uncertainty_us':0} if clock else {'basis':'uncalibrated'},
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),start_seconds='0',end_seconds='20'))
    activity=s['runtime'].timeline.add_activity('activity-'+key,'speech','speech',text,
        [dict(media_id=source['media_id'],start_seconds=lo,end_seconds=hi)],model_revision='generated-no-model/1',
        payload=dict(aligned_units=[dict(text=text,start_seconds=lo,end_seconds=hi)]))
    return activity['id'],source


def consume(s,text,key='spoken',**kwargs):
    activity,source=speech(s,text,key,**kwargs)
    engine=SpokenUsage(s['core'],s['settings'])
    return engine.consume(activity,POLICY),activity,source


def test_actual_use_and_remaining_are_not_two_debits_and_replay_is_idempotent(oral):
    s=oral;result,activity,_=consume(s,'我取了两克样品，现在还剩九十八克。')
    rows=result['statements']
    assert [r['semantics'] for r in rows]==['actual_increment','balance']
    assert rows[0]['quantity_ledger_written'] is True and rows[1]['quantity_ledger_written'] is False
    assert Decimal(rows[0]['inventory']['balance'])==98 and balance(s)==98
    assert s['runtime'].spoken_usage.consume(activity,POLICY)==result and balance(s)==98


@pytest.mark.parametrize('text',[
    '准备取2g','累计用了2g','现在还剩98g','更正，刚才取了2g','没取2g','我取了大约2g',
    '如果操作成功，我取了2g','我取了二百五克','现在温度120度，转速300转，运行10分钟',
    '更正，取了2g','取了0g','取了-2g','取了百百克',
])
def test_plans_totals_remaining_corrections_uncertainty_and_settings_never_auto_debit(oral,text):
    result,_,_=consume(oral,text)
    assert balance(oral)==100
    assert not any(r['quantity_ledger_written'] for r in result['statements'])


def test_inconsistent_remaining_keeps_both_statements_pending(oral):
    result,_,_=consume(oral,'取了2g，还剩90g')
    assert balance(oral)==100
    assert all('spoken_remaining_conflicts_with_inventory_arithmetic' in r['reasons'] for r in result['statements'])


def test_unknown_time_or_unbound_material_still_records_original_declaration(oral):
    result,_,_=consume(oral,'取了2g，还剩98g',clock=False)
    assert len(result['statements'])==2 and all(r['occurred_at'] is None for r in result['statements'])
    assert balance(oral)==100 and result['statements'][0]['source_text']=='取了2g，还剩98g'


def test_explicit_two_actual_additions_are_distinct_but_overlapping_ASR_resegmentation_is_not_new_use(oral):
    result,_,source=consume(oral,'先取了2g，又取了3g')
    assert balance(oral)==95 and len(result['statements'])==2
    other,_,_=consume(oral,'取了5g',key='resegmented',source=source,lo='.2',hi='1.8')
    assert balance(oral)==95 and 'overlapping_audio_requires_existing_occurrence_link' in other['statements'][0]['reasons']


def test_photo_then_speech_links_one_use_without_a_second_debit(oral):
    s=oral;head=start(s,photo(s,'tare',3,'0'));posted=observe(s,head,photo(s,'mass',4,'2'))
    result,_,_=consume(s,'取了2g')
    candidate=result['statements'][0]['candidate_id'];assert balance(s)==98
    linked=s['runtime'].spoken_usage.post('link-speech',candidate,expected_revision=1,
        expected_material_revision=2,actor='fixture',reason='explicit same occurrence',existing_operation_id=posted['operation_id'])
    assert linked['existing_operation_reused'] and balance(s)==98
    s['core'].correct_operation('correct-linked',posted['operation_id'],
        [dict(material_id=s['material'],quantity='1.5',unit='g')],expected_revision=1,actor='fixture',reason='verified correction')
    assert balance(s)==Decimal('98.5')


def test_speech_then_photo_requires_occurrence_link_and_never_silently_sums(oral):
    s=oral;spoken,_,_=consume(s,'取了2g')
    operation=spoken['statements'][0]['operation']['operation_id']
    h=start(s,photo(s,'tare',3,'0'));mass=photo(s,'mass',6,'2')
    with pytest.raises(Conflict,match='duplicate'):
        observe(s,h,mass)
    linked=observe(s,h,mass,existing_operation_id=operation,expected_sequence=0)
    assert linked['status']=='existing_usage_linked_no_new_debit' and balance(s)==98


def test_other_equipment_still_accepts_oral_volume_use_and_NAS_reading_projection(oral):
    s=oral
    liquid=s['core'].register_material('liquid','fixture-liquid','mL','100')['material_id']
    tag=s['core'].register('liquid-tag','tag','fixture-liquid-QR',{'mode':'reusable'})['id']
    s['core'].assign_tag('liquid-binding',tag,liquid,at(2),actor='fixture',reason='generated liquid sample')
    s['runtime'].operator_scans.record('liquid-scan',intent='operator_scan',scan_kind='material',qr_value='fixture-liquid-QR',
        occurred_at=at(3),evidence_ids=[],actor='fixture',reason='generated liquid context',camera_id=s['camera'],run_id=s['run'])
    result,_,_=consume(s,'加入了2毫升样品，还剩98毫升')
    assert result['statements'][0]['quantity_ledger_written'] is True
    assert Decimal(s['core'].balance(liquid)['balance'])==98
    snap=s['runtime'].views.snapshot('2026-10-29')
    entry=snap['document']['context_evidence']['spoken_material_usage']['entries'][0]
    assert entry['阅读摘要'][0]['数量']=='2' and entry['阅读摘要'][0]['已扣库存'] is True
    from visioncortex.lab.archive_layout import contract_bytes
    from visioncortex.lab.comment_presentation import CommentPresentationPublication
    root=s['settings'].incoming;(root/'.System').mkdir();(root/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    settings=replace(s['settings'],source_root=root,unified_archive_required=True,timeline_publish=True)
    s['runtime'].publication=CommentPresentationPublication(s['runtime'],settings,history=s['root']/'history')
    s['runtime'].publication.tick()
    value=json.loads((root/'2026-10-29_fixture-weighing-camera/Comment/Bindings.json').read_text())
    assert value['material_quantity_written'] is True and value['用料记录'][0]['原话']=='加入了2毫升样品，还剩98毫升'


@pytest.mark.parametrize('text,value', [('取了十一克','11'),('取了一百零二克','102'),('取了两点五克','2.5'),('取了二百三十克','230')])
def test_explicit_spoken_numbers_preserve_decimal(text,value):
    assert parse_usage_statements(text)[0]['value']['quantity']==value
