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


@pytest.mark.parametrize('damage', ['camera_id', 'instrument_id', 'entry_camera_id', 'foreign_camera_path', 'foreign_day_path'])
def test_measurement_identity_conflicts_are_isolated_without_relabelling_siblings(tmp_path, damage):
    config, row, _, daily = archive(tmp_path)
    data = json.loads(daily.read_text())
    event = data['events'][1]
    bad_path = 'Photos/Bad.json'
    result = {'photo_time': {'timestamp_ms': row['recording_start_us']/1000},
              'values': [{'name': '质量', 'value': 99, 'unit': 'g'}]}
    bad_entry = {'result': bad_path, 'instrument_id': 'instrument-1'}
    if damage in {'camera_id', 'instrument_id'}:
        result[damage] = 'camera-2' if damage == 'camera_id' else 'instrument-2'
    elif damage == 'entry_camera_id':
        bad_entry['camera_id'] = 'camera-2'
    else:
        bad_path = ('2026-09-16_camera-2' if damage == 'foreign_camera_path' else '2026-09-17_camera-1')+'/Photos/Bad.json'
        bad_entry['result'] = bad_path
    write(tmp_path/bad_path, result)
    event['measurement_files'].insert(0, bad_entry)
    write(daily, data)
    context = context_events(config, row)
    readings = [e for e in context['events'] if e['type'] == 'ocr_observation']
    assert context['status'] == 'partial_archived_evidence' and len(readings) == 1
    assert readings[0]['event_id'].endswith('Photos/Result.json')
    assert readings[0]['instrument_id'] == 'instrument-1'
    assert readings[0]['payload']['normalized_values'][0]['value'] is None
    assert len(context['errors']) == 1 and context['errors'][0]['measurement_index'] == 0
    assert context['errors'][0]['reason'] == 'field_archive_measurement_invalid'
    assert 'crossed' in context['errors'][0]['validation_error']


@pytest.mark.parametrize('malformed', [[], {'photo_time': []}, {'photo_time': 'invalid'},
    {'photo_time': {'timestamp_ms': float('inf')}}, {'photo_time': {'timestamp_ms': float('nan')}},
    {'photo_time': {'timestamp_ms': 1e308}}, {'photo_time': {'timestamp_ms': 'unknown'}}])
def test_bad_measurement_schema_or_timestamp_keeps_later_measurement_in_same_event(tmp_path, malformed):
    config, row, _, daily = archive(tmp_path)
    write(tmp_path/'Photos/Bad.json', malformed)
    data = json.loads(daily.read_text())
    data['events'][1]['measurement_files'].insert(0, {'result': 'Photos/Bad.json', 'instrument_id': 'instrument-1'})
    write(daily, data)
    context = context_events(config, row)
    readings = [e for e in context['events'] if e['type'] == 'ocr_observation']
    assert context['status'] == 'partial_archived_evidence' and len(readings) == 1
    assert readings[0]['event_id'] == 'capture-1:Photos/Result.json'
    assert len(context['errors']) == 1 and context['errors'][0]['measurement_index'] == 0
    assert context['errors'][0]['reason'] == 'field_archive_measurement_invalid'


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
    result = context_events(config, row)
    assert result['status'] == 'partial_archived_evidence'
    assert not any(e['type'] in {'instrument_binding', 'scene_binding'} for e in result['events'])
    assert 'crossed' in result['errors'][0]['validation_error']
    original = json.loads(daily.read_text())
    write(daily, {'events': [{'result': '../Outside.json', 'category': 'Bindings'}, None,
                            {'event_id': 'bad-schema'}, original['events'][1]]})
    result = context_events(config, row)
    assert result['status'] == 'partial_archived_evidence'
    assert [e['type'] for e in result['events']] == ['ocr_observation']
    assert len(result['errors']) == 3
    assert all(e['reason'] == 'field_archive_event_invalid' for e in result['errors'])


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


def test_unbound_or_failed_ocr_remains_an_independent_event_with_original_status(tmp_path):
    config, row, _, daily = archive(tmp_path)
    data = json.loads(daily.read_text())
    item = data['events'][1]
    item.update(event_at='2026-09-16T14:00:00+08:00', measurement_files=[], status='draft')
    write(daily, data)
    source = {'record_scope': 'test_only', 'status': 'draft', 'processing_outcome': 'recognition_skipped',
              'skip_reasons': ['missing binding'], 'unbound_readings': [{'value': '12.3', 'unit': None}],
              'decision': {'status': 'draft', 'revision': 1}}
    write(tmp_path/'Photos/Evidence.json', source)
    event = next(e for e in context_events(config, row)['events'] if e['type'] == 'ocr_observation')
    assert event['payload']['measurement_result_available'] is False
    assert event['payload']['normalized_values'] == []
    assert event['payload']['unbound_readings'] == source['unbound_readings']
    assert event['payload']['record_scope'] == 'test_only'
    assert event['wearer_id'] is None and event['instrument_id'] is None
    assert event['media']['result_file'] is None and event['evidence_refs']


def test_archived_updates_duplicate_delivery_and_rejection_change_only_the_source_revision(tmp_path):
    from visioncortex.application.field_archive_context import archive_revision
    config, row, _, daily = archive(tmp_path)
    first = context_events(config, row)
    version = archive_revision(config, '2026-09-16_camera-1')
    data = json.loads(daily.read_text())
    data['events'].append(data['events'][1])
    write(daily, data)
    duplicate = context_events(config, row)
    assert len(duplicate['events']) == len(first['events'])
    # A revised reading preserves its exact value and absent unit, without
    # reinterpreting rejection/withdrawal as an accepted physical action.
    write(tmp_path/'Photos/Result.json', {'photo_time': {'timestamp_ms': row['recording_start_us']/1000},
        'values': [{'name': '质量', 'value': 12.3, 'unit': None}]})
    write(tmp_path/'Photos/Evidence.json', {'record_scope': 'test_only', 'withdrawn': True,
        'decision': {'status': 'rejected', 'revision': 2}})
    changed = context_events(config, row)
    event = next(e for e in changed['events'] if e['type'] == 'ocr_observation')
    assert event['payload']['normalized_values'][0] == {'name': '质量', 'value': 12.3, 'unit': None}
    assert event['payload']['decision']['status'] == 'rejected' and event['payload']['withdrawn'] is True
    assert event['revision'] != next(e for e in first['events'] if e['type'] == 'ocr_observation')['revision']
    assert archive_revision(config, '2026-09-16_camera-1') != version
    assert changed['physical_action_verified'] is False


def test_one_unavailable_source_preserves_other_events_and_retry_provenance(tmp_path):
    config, row, binding, _ = archive(tmp_path)
    binding.unlink()
    context = context_events(config, row)
    assert context['status'] == 'partial_archived_evidence'
    assert [e['type'] for e in context['events']] == ['ocr_observation']
    assert context['errors'][0]['automatic_action'] == 'retry_source_next_publication'
    assert context['source_references']


def test_field_only_day_is_discovered_and_published_without_creating_raw_directory(tmp_path):
    from visioncortex.application.field_archive_context import archive_days, day_context, publish_context
    config, _, _, _ = archive(tmp_path/'Field')
    output = tmp_path/'VisionCortex'
    config['storage'] = {'archive_root': str(output)}
    assert archive_days(config) == {'2026-09-16_camera-1': us('2026-09-16T00:00:00+08:00')}
    events = day_context(config, '2026-09-16_camera-1')
    index = {'archive': '2026-09-16_camera-1', 'recordings': [], 'context': {
        'events': events['events'], 'field_event_status': events['status'],
        'field_source_references': events['source_references'], 'field_event_errors': events['errors']}}
    first = publish_context(config, index)
    value = json.loads((output/index['archive']/first['path']).read_text())
    assert len(value['events']) == 2 and value['model_invoked'] is False
    index['context']['events'] = []
    second = publish_context(config, index)
    assert first['revision'] != second['revision']
    snapshots = list((output/index['archive']/'Comment/.Evidence/FieldRecognition').glob('*.json'))
    assert len(snapshots) == 2
    assert not (output/index['archive']/'MetaData').exists()
    assert not (output/index['archive']/'MetaVideo').exists()
    first_snapshot = output/index['archive']/'Comment/.Evidence/FieldRecognition'/(first['revision']+'.json')
    first_snapshot.write_bytes(b'{"corrupt": true}')
    index['context']['events'] = events['events']
    with pytest.raises(ValueError, match='identity changed'):
        publish_context(config, index)
    assert first_snapshot.read_bytes() == b'{"corrupt": true}'


def test_file_publisher_discovers_late_field_only_inputs_and_revisions_without_models(tmp_path):
    from visioncortex.device_day_file_index import FileIndexPublisher
    config, _, _, daily = archive(tmp_path/'Field')
    output = tmp_path/'VisionCortex'
    config['storage'] = {'archive_root': str(output), 'local_runtime_root': str(tmp_path/'Runtime'),
                         'local_cache_root': str(tmp_path/'Cache')}
    config['collection_ingest'] = {'source_root': str(tmp_path/'Capture')}
    config['device_day']['readable_content_enabled'] = True
    publisher = FileIndexPublisher(config)
    first = publisher.tick()
    assert first['published'] == 1 and first['errors'] == []
    root = output/'2026-09-16_camera-1'
    report = root/'LaboratoryDailyReport/LaboratoryDailyReport.json'
    value = json.loads(report.read_text())
    assert len(value['全天记录']['现场事件时间线']) == 2
    assert publisher.tick()['unchanged'] == 1
    data = json.loads(daily.read_text())
    data['events'].append(dict(data['events'][1], event_id='late-ocr'))
    write(daily, data)
    assert publisher.tick()['published'] == 1
    assert len(json.loads((root/'Comment/FieldEvents.json').read_text())['events']) == 3
    assert len(json.loads(report.read_text())['全天记录']['现场事件时间线']) == 3
    assert len(list((root/'Comment/.Evidence/FieldRecognition').glob('*.json'))) == 2
    assert not (root/'MetaData').exists() and not (root/'MetaVideo').exists()
