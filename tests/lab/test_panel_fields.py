from copy import deepcopy
import pytest

from visioncortex.lab.panel_fields import detected_fields


def page(name, texts, boxes=None):
    return dict(panel_id=name, panel_box=[0, 0, 400, 100],
        geometry_basis='model_detected_lcd_screen_not_instrument_identity',
        rec_texts=texts, rec_scores=[.99]*len(texts),
        rec_boxes=boxes or [[10, 5+16*i, 90, 18+16*i] for i in range(len(texts))])


def document(*pages):
    return dict(panel_localization={'source_size':[400, 100]}, results=list(pages))


def test_detected_displays_keep_devices_unknown_and_do_not_borrow_units():
    p2=page('p2',['-0.0120'],[[210,5,290,18]]);p2['panel_box']=[200,0,400,100]
    value=document(page('p1',['0.2287g']),p2)
    before=deepcopy(value);fields,receipt=detected_fields(value)
    assert fields[0]['field_key']=='质量' and fields[0]['value']=='0.2287' and fields[0]['unit']=='g'
    assert fields[1]['field_key']=='显示读数' and fields[1]['value']=='-0.0120' and fields[1]['unit'] is None
    assert fields[0]['channel']!=fields[1]['channel'] and receipt['panels']==2
    assert all(f['physical_instrument_identity'] is None and f['auto_confirmed'] is False for f in fields)
    assert value==before


def test_fixed_mass_rule_cannot_be_applied_to_an_unknown_display_type():
    rules=[dict(instrument_type='balance',measurement_name='质量',unit='g',decimal_places=4,
                fixed_decimal_display=True,confirmed_by_user=True)]
    fields,_=detected_fields(document(page('p1',['02287g','0.2287g','02287'])),display_rules=rules)
    assert fields[0]['value'] is None and fields[0]['tentative_value']=='02287'
    assert 'normalization' not in fields[0]
    assert fields[1]['value']=='0.2287' and fields[1]['raw_tokens'][0]['text']=='0.2287g'
    assert fields[2]['value'] is None and fields[2]['tentative_value']=='02287'


def test_pv_sv_and_negative_value_are_separate_source_local_fields():
    fields,_=detected_fields(document(page('p1',['PV26.2℃','SV30.0℃'])))
    assert [f['role'] for f in fields]==['process_value','setpoint']
    assert [f['value'] for f in fields]==['26.2','30.0']
    assert all(f['unit']=='°C' for f in fields)
    p=page('p1',['-','0.0120','g'],[[5,10,9,25],[12,10,80,25],[85,10,96,25]])
    fields,_=detected_fields(document(p))
    assert fields[0]['value']=='-0.0120' and fields[0]['unit']=='g'


def test_clocks_model_numbers_bottle_text_and_merged_numbers_are_not_readings():
    d=document(page('p1',['14:38','12E8','550 990','型号123','OFF']))
    d['results'].append(dict(rec_texts=['型号120g'],rec_scores=[.99],rec_boxes=[[0,0,80,20]]))
    fields,receipt=detected_fields(d)
    assert fields==[] and receipt['panels']==1


def test_low_score_and_competing_units_stay_pending():
    p=page('p1',['26.2','℃','℉'],[[10,10,65,25],[70,10,85,25],[86,10,98,25]])
    fields,_=detected_fields(document(p))
    assert fields[0]['value'] is None and 'ambiguous_unit_neighbours' in fields[0]['issues']
    p=page('p1',['0.2287g']);p['rec_scores']=[.2]
    fields,_=detected_fields(document(p));assert fields[0]['value'] is None


@pytest.mark.parametrize('box', [[-1,0,20,20],[0,0,401,20],[0,0,float('nan'),20],[0,0,0,20]])
def test_invalid_or_cross_source_geometry_is_rejected(box):
    with pytest.raises(ValueError):detected_fields(document(page('p1',['1.23g'],[box])))


def test_public_reading_view_explains_panel_candidate_without_asset_binding():
    from visioncortex.lab.readable_readings import project_readings
    field=detected_fields(document(page('p1',['0.2287g'])))[0][0]
    row=dict(value=field['value'],unit=field['unit'],instrument_id=None,raw_field=field,quality_issues=[])
    output,summary=project_readings([row],field_mapping='detected_panel_ocr')
    assert output==[row] and summary['识别结果'][0]['字段']=='质量'
    assert summary['识别结果'][0]['面板']=='p1' and summary['已确认业务读数']==0


def whole(texts, boxes):
    return dict(rec_texts=texts,rec_scores=[.99]*len(texts),rec_boxes=boxes)


def test_missed_screen_keeps_text_quantity_explicitly_unverified():
    from visioncortex.lab.readable_readings import project_readings
    d=document(whole(['0.0713','g','1436','HTC-1'],
        [[10,10,90,35],[95,10,105,35],[200,10,260,35],[200,50,260,75]]))
    fields,receipt=detected_fields(d)
    assert len(fields)==1 and fields[0]['value']=='0.0713' and fields[0]['unit']=='g'
    assert receipt['panels']==0 and receipt['text_regions']==1
    assert fields[0]['screen_localization_supported'] is False
    _,summary=project_readings([dict(value=fields[0]['value'],unit='g',raw_field=fields[0])],field_mapping='detected_panel_ocr')
    assert '尚未确认属于屏幕' in summary['识别结果'][0]['字段依据']


def test_full_frame_unit_context_deduplicates_same_crop_number():
    d=document(whole(['湿度','61.2%'],[[10,10,40,30],[50,10,110,30]]),
               page('p1',['61.2'],[[50,10,100,30]]))
    fields,receipt=detected_fields(d)
    assert receipt['panels']==1 and len(fields)==1
    assert fields[0]['field_key']=='湿度' and fields[0]['unit']=='%' and fields[0]['value']=='61.2'
    assert fields[0]['additional_views'][0]['displayed_glyphs']=='61.2'


def test_whole_crop_conflict_never_selects_one_number():
    d=document(whole(['26.2℃'],[[10,10,95,30]]),page('p1',['28.2'],[[10,10,80,30]]))
    fields,_=detected_fields(d)
    assert len(fields)==2 and all(f['value'] is None for f in fields)
    assert all('ocr_view_conflict' in f['issues'] for f in fields)


def test_shared_unit_between_full_frame_numbers_is_ambiguous():
    d=document(whole(['1.23','4.56','g'],[[10,10,45,30],[48,10,65,30],[70,10,80,30]]))
    fields,_=detected_fields(d)
    assert len(fields)==2 and all(f['value'] is None for f in fields)
    assert all('ambiguous_unit_neighbours' in f['issues'] for f in fields)
