"""Model-free readable projection and mass-decimal counterexamples."""
from copy import deepcopy
import json
from types import SimpleNamespace
from uuid import uuid4

from visioncortex.lab.automatic_readout import _decimal_region
from visioncortex.lab.display_pixels import guard_decimal
from visioncortex.lab.ocr_fields import field_candidate
from visioncortex.lab.readable_readings import project_readings
from visioncortex.lab.timeline_native_artifacts import artifact_documents
from visioncortex.lab.timeline_publication import TimelinePublication
from .test_timeline_native_artifacts import NoTaskDB


def test_new_unbound_raw_identifiers_and_clock_stay_visible_without_becoming_balance_values():
    rows=[dict(value=None,display_value=t,text=t,unit=None,instrument_id=None,
        quality_issues=['unmapped_raw_token_not_a_confirmed_numeric_field']) for t in ['EQ8','14:10:32','001234','24.2 g']]
    before=deepcopy(rows);projected,summary=project_readings(rows)
    assert rows==before and len(projected)==summary['候选条数']==4
    assert [r['识别原文'] for r in summary['识别结果']]==['EQ8','14:10:32','001234','24.2 g']
    assert all(r['候选值'] is None and r['单位'] is None for r in summary['识别结果'])
    assert all('未映射' in r['字段'] for r in summary['识别结果'])


def test_new_readable_view_preserves_original_fr_fixed_decimal_and_uncertain_candidates():
    rule=dict(instrument_type='balance',measurement_name='质量',unit='g',decimal_places=4,
        fixed_decimal_display=True,confirmed_by_user=True)
    region=dict(field_key='mass',role='displayed_value',box=[0,0,1,1],unit='g',unit_basis='verified_on_panel')
    def parse(text,*,rules=None,source=None):
        result=dict(results=[dict(rec_texts=[text],rec_scores=[.99])])
        return field_candidate(source or region,result,display_rules=[rule] if rules is None else rules)
    dotted,normalized,mismatch=parse('-0.0120'),parse('02287'),parse('1234.0')
    assert dotted['value']=='-0.0120' and dotted['raw_value']=='-0.0120' and 'normalization' not in dotted
    assert normalized['value']=='0.2287' and normalized['raw_value']=='02287'
    assert normalized['normalization']['method']=='registered_fixed_decimal_v2'
    assert normalized['display_rule']==dict(rule,basis='user_confirmed_display_type')
    assert normalized['normalization']['instrument_id'] is None
    assert mismatch['value'] is None and mismatch['tentative_value']==mismatch['raw_value']=='1234.0'
    assert 'display_precision_mismatch' in mismatch['issues'] and 'normalization' not in mismatch
    conflict=parse('02287',rules=[rule,dict(rule,decimal_places=3)])
    assert conflict['raw_value']=='02287' and 'display_rule' not in conflict and 'normalization' not in conflict
    assert 'display_rule' not in parse('02287',source=dict(region,field_key='temperature'))
    assert 'display_rule' not in parse('02287',source=dict(region,unit=None,unit_basis=None))
    pixels=dict(visible_decimal_candidates=[])
    assert guard_decimal(normalized,pixels,_decimal_region(region,normalized))['value']=='0.2287'
    assert guard_decimal(normalized,pixels,_decimal_region(dict(region,decimal_places=3),normalized))['value'] is None
    rows=[dict(value=c['value'],unit=c['unit'],instrument_id=None,measurement_name='质量',raw_field=c,quality_issues=[])
        for c in [normalized,dotted,mismatch]]
    before=deepcopy(rows)
    projected,summary=project_readings(rows,field_mapping='verified_source_registration')
    assert rows==before and projected==before
    assert projected[0]['value']=='0.2287' and projected[1]['value']=='-0.0120' and projected[2]['value'] is None
    assert summary['识别结果'][0]['原规范化记录']==normalized['normalization']
    assert summary['识别结果'][0]['原显示规则']==normalized['display_rule']
    assert summary['识别结果'][0]['原候选值']=='02287'
    assert summary['识别结果'][0]['识别原文']=='02287'
    assert all(r['设备归属']=='未绑定或未核验' for r in summary['识别结果'])
    assert '待核对' in summary['识别结果'][2]['小数点依据']


def test_new_artifacts_add_chinese_summary_preserve_original_schema_raw_and_unknown_identity():
    mid,aid=str(uuid4()),str(uuid4())
    payload=dict(field_mapping='unmapped_raw_ocr',fields=[],raw_tokens=[dict(text='550 990',score='.9')],qr_hash='a'*64)
    activity=dict(id=aid,kind='reading',source_type='ocr',start_at=None,end_at=None,model_revision='supplied-only',
        payload=payload,evidence=[dict(media_id=mid,kind='photo',start_seconds='0',end_seconds='0')])
    files,refs=artifact_documents(SimpleNamespace(db=NoTaskDB(),customer=uuid4()),archive='2026-10-29_fixture-camera',
        camera='fixture-camera',activities=[activity],media_sources={mid:dict(kind='photo',source_sha256='b'*64,
        time_basis={},source_kind='fixture')},rules=dict(revision='fixture',sha256='c'*64))
    readings=files[refs[0]['readings']];unbound=files[refs[0]['unbound_readings']]
    assert readings['schema']=='field-numeric-observations/1' and unbound['schema']=='field-unbound-readings/1'
    assert readings['readings'][0]['raw_token']==payload['raw_tokens'][0]
    assert readings['readings'][0]['value'] is None and readings['readings'][0]['display_value']=='550 990'
    assert readings['中文摘要']['识别结果'][0]['识别原文']=='550 990'
    assert unbound['中文摘要']['已确认业务读数']==0
    assert files[refs[0]['evidence']]['observations'][0]['activity']['payload']==payload
    encoded=TimelinePublication._json(readings)
    assert next(iter(json.loads(encoded)))=='中文摘要'
    assert json.loads(encoded)==readings
    legacy={'z':1,'a':{'z':2,'a':3}}
    assert TimelinePublication._json(legacy)==(json.dumps(legacy,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
