from ._support import research_helpers
import importlib.util

import pytest

_spec = importlib.util.spec_from_file_location('check_model_report', research_helpers() / 'check_model_report.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def fixture():
    facts = dict(date='2026-09-24',coverage='partial',not_provided=['SOP'],observations=[
        dict(source_id='photo',raw_display_texts=['OFF','1390'],uncertain_display_candidate='1130')])
    report = dict(date=facts['date'],coverage='partial',missing_information=['SOP'],
        summary=dict(source_ids=['photo'],description='部分画面'),observed_events=[],
        display_readings=[dict(source_ids=['photo'],raw_text='1390',unit=None,confirmed_measurement=False)],
        completed_experiments=[],material_consumption=[])
    return facts,report


def test_uncertain_digits_are_not_confirmed_readings():
    facts,report = fixture()
    assert _module.check_report_facts(report,facts) == []
    report['display_readings'][0]['raw_text'] = '1130'
    assert 'unsupported_or_uncertain_reading' in _module.check_report_facts(report,facts)


@pytest.mark.parametrize(('field','value','reason'),[
    ('summary',dict(source_ids=['invented']),'invalid_source_ids'),
    ('completed_experiments',['done'],'unconfirmed_completion_or_consumption'),
    ('coverage','full_day','coverage_missing_or_changed'),
    ('missing_information',[],'missing_information_not_preserved'),
])
def test_report_rejects_unsupported_facts(field,value,reason):
    facts,report = fixture()
    report[field] = value
    assert reason in _module.check_report_facts(report,facts)


def test_report_rejects_invented_units():
    facts,report = fixture()
    report['display_readings'][0]['unit'] = 'g'
    assert 'unconfirmed_unit_or_measurement' in _module.check_report_facts(report,facts)


def test_non_object_and_trailing_prose_are_not_parsed_as_valid_reports():
    with pytest.raises(ValueError):
        _module.parse_json_reply('[]')
    with pytest.raises(ValueError):
        _module.parse_json_reply('{} extra explanation')
