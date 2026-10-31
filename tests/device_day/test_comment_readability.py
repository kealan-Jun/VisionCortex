"""Readable Comment preserves words, clocks, machine candidates and paths."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from visioncortex.comment_readability import field_context_document, readings_document, readings_text, transcript_text
from visioncortex.lab.comment_presentation import CommentPresentation, supported


def test_saved_speech_is_readable_without_changing_json_or_inventing_a_speaker():
    value = dict(date='2026-10-29', device='fixture-camera', audio=dict(time_basis='recorder_audio_clock'),
                 sentences=[dict(start_us=1793250345760000, end_us=1793250346920000, text='原话保留。')])
    before = deepcopy(value)
    text = transcript_text(value).decode()
    assert '原话保留。' in text and '说话人' not in text and '13:05:45.760' in text
    assert value == before
    value['capture_time_status'] = 'unknown_or_uncalibrated'
    value['sentences'][0].update(source_start_seconds='3.16', source_end_seconds='4.32')
    assert '录音内 3.16—4.32 秒' in transcript_text(value).decode()
    assert '13:05:45.760' not in transcript_text(value).decode()


@pytest.mark.parametrize('status,outcome,audio,expected', [
    ('completed','no_audio','no_input','未取得音频'),
    ('failed','error','provided','转写失败'),
    ('running',None,'provided','尚未完成'),
    ('completed','transcribed','provided','转写结果为空'),
])
def test_missing_failed_pending_and_empty_speech_are_distinct(status,outcome,audio,expected):
    text = transcript_text(dict(status=status,outcome=outcome,audio=dict(status=audio))).decode()
    assert expected in text


def reading():
    return dict(schema='field-numeric-observations/1', camera_id='any-camera',
                event_at=None,capture_time_status='unknown_or_uncalibrated',
                source_refs={'one':dict(source_reported_at='2026-10-29T17:41:35+08:00',source_relative_path='MetaData/Photos/photo.jpg')},
                readings=[dict(value=None,unit=None,text='12E8',instrument_id=None)],
                中文摘要=dict(已确认业务读数=0,候选条数=1,识别结果=[dict(识别原文='12E8',设备归属='未绑定或未核验',待核对原因=['字段未确定'])]))


def test_raw_tokens_do_not_become_values_or_calibrated_times():
    original = reading(); before = deepcopy(original)
    presented = readings_document(original)
    assert original == before and presented['readings'] == original['readings']
    assert '尚未校准' in presented['中文摘要']['时间']
    assert presented['中文摘要']['数值'] == []
    assert presented['中文摘要']['识别结果'] == '未读清'
    assert set(presented['中文摘要']) == {'时间', '数值', '原图或视频', '识别结果'}
    assert readings_document(presented) == presented
    encoded=readings_text(original).decode()
    assert encoded.index('"时间"') < encoded.index('"readings"')
    assert encoded.index('"数值"') < encoded.index('"readings"')
    assert readings_document({'schema':'other','readings':[]}) == {'schema':'other','readings':[]}


def test_only_existing_comment_members_change_and_history_is_recoverable(tmp_path):
    root=tmp_path/'archive'; history=tmp_path/'history'; root.mkdir()
    relative='2026-10-29_camera/Comment/OCR/VoicePhotoReadings/event/Readings.json'
    path=root/relative; path.parent.mkdir(parents=True)
    original=(json.dumps(reading(),ensure_ascii=False)+'\n').encode(); path.write_bytes(original)
    directories={str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir()}
    class Guard:
        @staticmethod
        def guarded_atomic(path,blocks): return False
    presenter=CommentPresentation(history)
    first=presenter.write(root,relative,original,Guard)
    assert first['presentation_revision'] and json.loads(path.read_bytes())['readings']==reading()['readings']
    assert directories=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir()}
    assert any(p.read_bytes()==original for p in history.glob('*.bytes'))
    assert presenter.write(root,relative,original,Guard)['sha256']==first['sha256']
    changed=reading(); changed['readings'][0]['value']='999'
    with pytest.raises(ValueError,match='machine evidence'):
        presenter.write(root,relative,json.dumps(changed).encode(),Guard)
    assert not supported('day/ProcessedClips/Clips/event/Readings.json')
    assert not supported('day/Comment/Bindings.json')
    assert not supported('../day/Comment/OCR/event/Readings.json')
    assert Path(root/relative).is_file()


def test_legacy_fr_rows_preserve_values_identity_and_review_status():
    value=dict(schema='field-numeric-observations/1',human_verified=False,
               readings=[dict(value=14.7725,unit='g',measurement_name='质量',instrument_id='historical-id',
                              result_stage='reviewed',captured_at='2026-10-29T14:52:22+08:00')])
    before=deepcopy(value);after=readings_document(value)
    assert value==before and after['readings']==value['readings']
    assert after['中文摘要']['数值']==['14.7725']
    assert set(after['中文摘要']) == {'时间', '数值', '原图或视频'}
    assert readings_document(after)==after


def test_number_and_decimal_are_saved_without_unit_role_or_sign_gates():
    value=reading()
    value['readings']=[dict(value=None,raw_field=dict(
        raw_tokens=[dict(text='-',score=.1),dict(text='0.07130',score=.99),dict(text='g',score=.1)],
        issues=['ambiguous_sign','ambiguous_unit_neighbours','field_role_not_observed']))]
    before=deepcopy(value);after=readings_document(value)
    assert after['中文摘要']['数值']==['0.07130']
    assert {k:v for k,v in after.items() if k!='中文摘要'} == {k:v for k,v in before.items() if k!='中文摘要'}
    assert value==before


def test_no_digit_join_decimal_invention_or_low_confidence_number():
    value=reading()
    value['readings']=[dict(raw_field=dict(raw_tokens=tokens)) for tokens in [
        [dict(text='12',score=.99),dict(text='34',score=.99)],
        [dict(text='14:36',score=.99)], [dict(text='EQ-000001',score=.99)],
        [dict(text='1.23',score=.2)]]]
    value['readings'].append(dict(raw_field=dict(raw_tokens=[dict(text='00713',score=.99)],
                                                issues=['integer_or_missing_decimal_unresolved'])))
    assert readings_document(value)['中文摘要']['数值']==[]
    value['readings']=[dict(raw_token=dict(text='61.20%',score=.99)),
                       dict(raw_field=dict(raw_tokens=[dict(text='0',score=.99)])),
                       dict(raw_token=dict(text='1436',score=.99))]
    assert readings_document(value)['中文摘要']['数值']==['61.20','0']


def test_conflicting_digits_never_selected_and_sign_conflict_keeps_same_number():
    value=reading()
    def row(text,box):
        return dict(raw_field=dict(raw_tokens=[dict(text=text,score=.99)],source_pixel_box=box,
                                   issues=['ocr_view_conflict']))
    value['readings']=[row('1.23',[0,0,80,20]),row('12.3',[0,0,80,20]),
                       row('-4.560',[0,30,80,50]),row('4.560',[0,30,80,50]),
                       dict(raw_token=dict(text='1.23',score=.99))]
    assert readings_document(value)['中文摘要']['数值']==['4.560']


def test_historical_binding_summary_preserves_original_events_and_unknown_people():
    value=dict(schema_version='visioncortex-field-archive-context/1',status='partial_archived_evidence',
        events=[dict(type='ocr_observation',event_id='original',start_us=1793250345760000,instrument_id='old-device',
                     payload=dict(normalized_values=[dict(value='0.2287',unit='g')]),clock=dict(uncertainty_us=None))],
        errors=[dict(reason='source_missing')],physical_action_verified=False)
    before=deepcopy(value);after=field_context_document(value)
    assert value==before and {k:v for k,v in after.items() if k!='中文摘要'}==value
    row=after['中文摘要']['记录'][0]
    assert row['原设备标识']=='old-device' and row['原记录读数']==value['events'][0]['payload']['normalized_values']
    assert '原操作人标识' not in row and '佩戴人标识' not in row
    assert after['physical_action_verified'] is False and after['中文摘要']['缺失或冲突项']==1
