import json
from datetime import datetime

import pytest

from visioncortex.application.field_event_consumer import context_events, validate_settings
from visioncortex.device_day_narrative import experimenter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def us(value):
    return round(datetime.fromisoformat(value).timestamp() * 1e6)


def archive(tmp_path):
    config = {'device_day': {'timezone': 'Asia/Shanghai', 'field_recognition': {
        'enabled': False, 'archive_context_enabled': True, 'archive_root': str(tmp_path)}}}
    row = {'camera_key': 'camera-1', 'recording_start_us': us('2026-09-16T14:00:00+08:00'),
           'recording_end_us': us('2026-09-16T14:10:00+08:00')}
    daily = tmp_path / '2026-09-16_camera-1/DailyReport/DailyReport.json'
    write(daily, {'events': [
        {'event_id': 'binding-1', 'category': 'Bindings', 'result': 'Bindings/Binding.json'},
        {'event_id': 'capture-1', 'category': 'Continuous', 'result': 'Photos/Evidence.json',
         'photo': 'Photos/Photo.jpg', 'measurement_files': [{'result': 'Photos/Result.json',
         'instrument_id': 'instrument-1', 'instrument_name': 'balance'}]}]})
    binding = tmp_path / 'Bindings/Binding.json'
    write(binding, {'binding': {'camera_id': 'camera-1', 'wearer_id': 'person-1',
        'operator': '实验员甲', 'binding_id': 'binding-1', 'started_at': '2026-09-16T13:00:00+08:00',
        'ended_at': '2026-09-16T15:00:00+08:00'}})
    write(tmp_path / 'Photos/Evidence.json', {'record_scope': 'test_only'})
    write(tmp_path / 'Photos/Result.json', {'photo_time': {'timestamp_ms': row['recording_start_us'] / 1000},
        'values': [{'name': '质量', 'value': None, 'unit': None}]})
    return config, row, binding, daily


def test_actual_binding_reading_and_source_paths(tmp_path):
    config, row, _, _ = archive(tmp_path)
    context = context_events(config, row)
    assert context['status'] == 'available_archived_evidence'
    assert experimenter(context, 'camera-1', row['recording_start_us'], row['recording_end_us'])['姓名'] == '实验员甲'
    reading = next(e for e in context['events'] if e['type'] == 'ocr_observation')
    assert reading['payload']['normalized_values'][0]['value'] is None
    assert reading['payload']['record_scope'] == 'test_only'
    assert reading['media']['photo']['available'] is False
    assert all(__import__('pathlib').Path(ref['path']).is_file() for ref in reading['evidence_refs'])


def test_binding_end_update_invalidates_cached_context(tmp_path):
    config, row, binding, _ = archive(tmp_path)
    assert experimenter(context_events(config, row), 'camera-1', row['recording_start_us'], row['recording_end_us'])['姓名'] == '实验员甲'
    data = json.loads(binding.read_text())
    data['binding']['ended_at'] = '2026-09-16T13:59:00+08:00'
    write(binding, data)
    assert experimenter(context_events(config, row), 'camera-1', row['recording_start_us'], row['recording_end_us'])['姓名'] == '未关联实验员'


def test_reject_cross_camera_or_escape(tmp_path):
    config, row, binding, daily = archive(tmp_path)
    data = json.loads(binding.read_text())
    data['binding']['camera_id'] = 'camera-2'
    write(binding, data)
    with pytest.raises(ValueError, match='crossed'):
        context_events(config, row)
    write(daily, {'events': [{'result': '../Outside.json', 'category': 'Bindings'}]})
    with pytest.raises(ValueError):
        context_events(config, row)


def test_missing_archive_and_explicit_mode(tmp_path):
    config, row, _, _ = archive(tmp_path)
    assert context_events(config, row | {'camera_key': 'camera-2'})['status'] == 'no_archive_for_camera_day'
    with pytest.raises(ValueError):
        validate_settings(config['device_day']['field_recognition'] | {'enabled': True})


def test_open_binding_does_not_extend_past_midnight(tmp_path):
    config, row, binding, _ = archive(tmp_path)
    data = json.loads(binding.read_text())
    data['binding'].pop('ended_at')
    write(binding, data)
    context = context_events(config, row)
    assert context['events'][0]['payload']['valid_to'] == '2026-09-17T00:00:00+08:00'
    assert context_events(config, row | {'recording_start_us': us('2026-09-17T00:00:00+08:00'),
        'recording_end_us': us('2026-09-17T00:10:00+08:00')})['events'] == []
