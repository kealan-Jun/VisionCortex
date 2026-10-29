"""Temporal field presentation cannot turn an uncertain value into confirmation."""
from visioncortex.lab.timeline_report import activity_lines


def activity(status, value):
    return dict(source_type='ocr', start_at=None, end_at=None, kind='reading',
                text_content='时序面板候选', evidence=[], payload=dict(fields=[dict(
                    field_key='temperature', role='process_value', status=status,
                    unit=None, value='untrusted outer value', temporal_display=dict(
                        value_text=value, observation_time_start='1/3',
                        observation_time_end='2/3', method='single_frame',
                        rejection_reasons=['decimal_unobserved'] if status != 'confirmed' else []))]))


def test_uncertain_temporal_value_is_not_presented_as_confirmed():
    text = '\n'.join(activity_lines(activity('uncertain', '99.9')))
    assert '99.9' not in text and 'untrusted outer value' not in text
    assert '未知' in text and '未确认' in text and 'decimal_unobserved' in text


def test_visual_status_does_not_become_business_confirmation():
    text = '\n'.join(activity_lines(activity('confirmed', '24.2')))
    assert '24.2' in text and '视觉支持候选，业务未确认' in text
    assert '源媒体PTS 1/3—2/3' in text and '单位未明确' in text


def test_missing_temporal_value_remains_unknown():
    text = '\n'.join(activity_lines(activity('confirmed', None)))
    assert '视觉支持候选' not in text and '未确认' in text


def test_old_activity_output_is_unchanged():
    old = activity('uncertain', None)
    old['payload'] = {}
    assert activity_lines(old) == ['  时间未知—时间未知 [面板识别/reading] 时序面板候选']
