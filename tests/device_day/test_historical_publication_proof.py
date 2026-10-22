"""Owned synthetic bytes prove delivery gates, never real video/model accuracy."""
from copy import deepcopy

import pytest

from test_historical_completion_reader import device_config, enable_runtime_profiles, historical_case
from visioncortex.device_day_contract import read_json, safe_child
from visioncortex.device_day_file_index import FileIndexPublisher
from visioncortex.device_day_historical_publication import checked, folder
from visioncortex.device_day_progress import processing_health
from visioncortex.observed_inventory import observe

__all__ = ['device_config', 'historical_case']


@pytest.fixture
def delivery(historical_case, monkeypatch):
    case = historical_case
    if case.selected:
        from types import SimpleNamespace
        def extract(_source, target, _start, _duration, _encoder):
            target.write_bytes(b'Synthetic exact one second video contract bytes')
        monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
        monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=1000, fps=30))
    observe(case.runner.runtime_root, {'recordings': [case.record]})
    assert case.runner.refresh_index(case.layout, recording_id=case.record['recording_id'])
    return case


def queue_rows(case):
    result = {}
    for name, queue in case.runner.queues.items():
        with queue.connect() as db:
            result[name] = [dict(row) for row in db.execute('SELECT * FROM recordings')]
    return result


def test_one_publication_proof_is_readable_only_by_explicit_exact_runtime_profiles(delivery):
    from visioncortex.device_day_contract import atomic_json, digest
    from visioncortex.device_day_historical_reader import reader_identity
    case = delivery
    profiles = enable_runtime_profiles(case)
    assert case.runner.refresh_index(case.layout, recording_id=case.record['recording_id'])
    before = queue_rows(case)
    outcome = FileIndexPublisher(profiles['Publication']).tick(archives=[case.layout.name])
    assert not outcome['errors'], outcome
    assert outcome['historical_publications'][0]['status'] == 'recovered_published'
    pointer = folder(case.config, case.entry)/'Published.json'
    reference = read_json(pointer)['receipt']
    proof_path = safe_child(case.layout.root, reference['path'])
    proof_bytes = proof_path.read_bytes()
    proof = read_json(proof_path)
    assert proof['reader_identity'] == reader_identity(profiles['Publication'])
    assert proof['reader_key'] == case.entry['reader_key']
    for name, config in profiles.items():
        result = checked(config, case.entry)
        assert result['status'] == 'recovered_published', (name, result)
        assert result['receipt'] == reference
        assert proof_path.read_bytes() == proof_bytes
    altered = deepcopy(profiles['Web'])
    altered['device_day']['input_integrity']['max_records'] += 1
    assert checked(altered, case.entry)['status'] == 'publication_verification_pending'
    modified_entry = deepcopy(case.entry)
    modified_entry['reader']['runtime_profiles'][0]['identity'] = 'f'*64
    assert checked(profiles['Web'], modified_entry)['status'] == 'publication_verification_pending'
    # A copied proof cannot replace its stable baseline with the reading host's identity.
    proof['reader_identity'] = reader_identity(profiles['Web'])
    replacement = proof_path.with_name('Publication'+digest(proof)+'.json')
    atomic_json(replacement, proof)
    from visioncortex.device_day_contract import artifact
    atomic_json(pointer, {'schema_version': proof['schema_version'], 'receipt': artifact(case.layout.root, replacement)})
    assert checked(profiles['Web'], case.entry)['status'] == 'publication_verification_pending'
    assert queue_rows(case) == before


def test_only_real_readable_publication_changes_progress_without_changing_original_queue(delivery):
    case = delivery
    before = queue_rows(case)
    receipts = {name: (case.layout.receipts/case.record['recording_id']/(name+'.json')).read_bytes()
                for name in ('retention', 'vision', 'stt', 'understanding')}
    initial = processing_health(case.config)['stages']['understanding']
    assert initial['categories'] == {'local_completed_unpublished': 1}
    publisher = FileIndexPublisher(case.config)
    outcome = publisher.tick(archives=[case.layout.name])
    assert not outcome['errors'], outcome
    assert outcome['historical_publications'][0]['status'] == 'recovered_published', outcome
    verification = checked(case.config, case.entry)
    assert verification['status'] == 'recovered_published', verification
    health = processing_health(case.config)
    stage = health['stages']['understanding']
    assert stage['categories'] == {'historical_recovered_published': 1}
    assert stage['workload']['recovered_published'] == 1
    assert stage['workload']['publication_pending'] == stage['workload']['result_unknown'] == 0
    assert health['publication']['historical_recovered_published_recordings'] == 1
    assert health['publication']['local_completed_unpublished_recordings'] == 0
    assert queue_rows(case) == before
    for name, raw in receipts.items():
        assert (case.layout.receipts/case.record['recording_id']/(name+'.json')).read_bytes() == raw
    second = publisher.tick(archives=[case.layout.name])
    assert second['unchanged'] == 1 and not second['errors'], second
    assert second['historical_publications'][0]['status'] == 'recovered_published'


@pytest.mark.parametrize('damage', ['pointer', 'proof', 'mu', 'clips', 'time_index', 'daily_report',
                                    'native_receipt', 'original_source', 'current_context'])
def test_missing_or_same_size_changed_evidence_returns_to_pending_without_model_or_queue(delivery, damage):
    case = delivery
    publisher = FileIndexPublisher(case.config)
    outcome = publisher.tick(archives=[case.layout.name])
    assert not outcome['errors'], outcome
    assert checked(case.config, case.entry)['status'] == 'recovered_published'
    before = queue_rows(case)
    pointer = folder(case.config, case.entry)/'Published.json'
    proof = safe_child(case.layout.root, read_json(pointer)['receipt']['path'])
    value = read_json(proof)
    relative = {'mu': next(ref['path'] for ref in value['files'] if ref['path'].endswith('/Understanding.json')),
                'clips': next(ref['path'] for ref in value['files'] if ref['path'].endswith('/Retention.json')),
                'time_index': 'Comment/TimeIndex.json', 'daily_report': 'LaboratoryDailyReport/LaboratoryDailyReport.json'}
    if damage == 'pointer':
        pointer.unlink()
    elif damage == 'current_context':
        comment = case.layout.comments/'Protocol.json'
        from visioncortex.device_day_contract import atomic_json
        atomic_json(comment, {'text': 'new factual context must not be called old model input'})
    else:
        path = (proof if damage == 'proof' else
                case.layout.receipts/case.record['recording_id']/'understanding.json' if damage == 'native_receipt' else
                safe_child(case.layout.root, case.parents['retention']['sources'][0]['retained']['path'])
                    if damage == 'original_source' else safe_child(case.layout.root, relative[damage]))
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
    assert checked(case.config, case.entry)['status'] == 'publication_verification_pending'
    stage = processing_health(case.config)['stages']['understanding']
    assert stage['workload']['publication_pending'] == 1
    assert stage['workload']['recovered_published'] == stage['workload']['result_unknown'] == 0
    assert queue_rows(case) == before


def test_missing_proof_replays_through_same_publisher_even_when_output_version_is_unchanged(delivery):
    case = delivery
    publisher = FileIndexPublisher(case.config)
    assert not publisher.tick(archives=[case.layout.name])['errors']
    before = queue_rows(case)
    pointer = folder(case.config, case.entry)/'Published.json'
    pointer.unlink()
    result = publisher.tick(archives=[case.layout.name])
    assert result['published'] == 1 and result['unchanged'] == 0 and not result['errors'], result
    assert pointer.is_file() and checked(case.config, case.entry)['status'] == 'recovered_published'
    assert queue_rows(case) == before


def test_wrong_native_key_or_selection_in_canonical_index_cannot_be_sealed(delivery):
    from visioncortex.device_day_contract import atomic_json
    from visioncortex.device_day_historical_publication import publish
    case = delivery
    assert not FileIndexPublisher(case.config).tick(archives=[case.layout.name])['errors']
    source = read_json(case.layout.index)
    source['recordings'][0]['stages']['understanding']['key'] = 'f'*64
    atomic_json(case.layout.index, source)
    with pytest.raises(ValueError, match='native_keys'):
        publish(case.config, case.entry, deepcopy(source))


def test_ordinary_publisher_has_no_historical_byte_proof_work(device_config, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Ordinary records must not gain historical full-byte verification')
    monkeypatch.setattr('visioncortex.device_day_historical_publication.checked', forbidden)
    monkeypatch.setattr('visioncortex.device_day_historical_publication.publish', forbidden)
    assert FileIndexPublisher(device_config)._historical_publication('2026-10-10_camera') == []


def test_photo_heartbeat_alone_does_not_invalidate_verified_publication(delivery):
    from pathlib import Path
    from visioncortex.device_day_contract import atomic_json
    case = delivery
    photo = Path(case.config['storage']['local_runtime_root'])/'device-day/CapturePhotos'/(case.layout.name+'.json')
    atomic_json(photo, {'observed_at': 'owned heartbeat one', 'devices': []})
    publisher = FileIndexPublisher(case.config)
    assert not publisher.tick(archives=[case.layout.name])['errors']
    assert checked(case.config, case.entry)['status'] == 'recovered_published'
    before = (case.layout.root/'Comment/TimeIndex.json').read_bytes()
    atomic_json(photo, {'observed_at': 'owned heartbeat two', 'devices': []})
    assert checked(case.config, case.entry)['status'] == 'recovered_published'
    result = publisher.tick(archives=[case.layout.name])
    assert result['unchanged'] == 1 and not result['errors'], result
    assert (case.layout.root/'Comment/TimeIndex.json').read_bytes() == before


def test_first_seal_rejects_field_reference_switch_to_another_owned_historical_snapshot(delivery, monkeypatch):
    from visioncortex.application.field_archive_context import publish_context
    from visioncortex.device_day_content import with_partial_understandings
    from visioncortex.device_day_historical_publication import publish
    from visioncortex import device_day_file_index as file_index
    case = delivery
    assert not FileIndexPublisher(case.config).tick(archives=[case.layout.name])['errors']
    index = with_partial_understandings(case.config, read_json(case.layout.index))
    assert publish(case.config, case.entry, index)['status'] == 'recovered_published'
    path = case.layout.root/'Comment/FieldEvents.json'
    current = path.read_bytes()
    # Two real owner-produced temporary snapshots; only one belongs to the
    # current association. Valid historical bytes cannot replace that identity.
    publish_context(case.config, index | {'context': index['context'] | {'field_event_status': 'owned_prior_snapshot'}})
    previous = path.read_bytes()
    path.write_bytes(current)
    verified = file_index.verified_field_events_reference
    def replaced_after_reference(*args):
        reference = verified(*args)
        assert reference is not None
        path.write_bytes(previous)
        return reference
    monkeypatch.setattr(file_index, 'verified_field_events_reference', replaced_after_reference)
    pointer = folder(case.config, case.entry)/'Published.json'
    pointer.unlink()
    with pytest.raises(ValueError, match='native_field_context_reference_changed'):
        publish(case.config, case.entry, index)
    assert not pointer.exists()


def test_late_archived_ocr_keeps_published_native_understanding_and_adds_only_current_overlay(delivery):
    """The normal publisher can add facts without widening the frozen reader."""
    from pathlib import Path
    from visioncortex.device_day import current_recording_context
    from visioncortex.device_day_contract import atomic_json
    case = delivery
    publisher = FileIndexPublisher(case.config)
    assert not publisher.tick(archives=[case.layout.name])['errors']
    source = read_json(case.layout.index)
    native = deepcopy(source['understandings'])
    receipts = {path: path.read_bytes() for path in (case.layout.receipts/case.record['recording_id']).glob('*.json')}
    before_queue = queue_rows(case)
    old_proof = (folder(case.config, case.entry)/'Published.json').read_bytes()
    old_mu = read_json(case.layout.root/'MultimodalUnderstanding/Index.json')['entries'][0]['content']
    before_mu = read_json(safe_child(case.layout.root, old_mu))
    assert not (case.layout.root/'Comment/FieldEvents.json').is_file() or not read_json(
        case.layout.root/'Comment/FieldEvents.json')['events']
    fr = Path(case.config['device_day']['field_recognition']['archive_root'])
    atomic_json(fr/case.layout.name/'DailyReport/DailyReport.json', {'events': [
        {'event_id': 'late-archived-reading', 'category': 'Continuous', 'result': 'Photos/Evidence.json',
         'measurement_files': [{'result': 'Photos/Result.json', 'instrument_id': 'instrument-native'}]}]})
    atomic_json(fr/'Photos/Evidence.json', {'test_only': True, 'status': 'recognized'})
    atomic_json(fr/'Photos/Result.json', {'camera_id': case.record['camera_key'],
        'instrument_id': 'instrument-native',
        'photo_time': {'timestamp_ms': case.record['recording_start_us']/1000+1000},
        'values': [{'name': 'owned synthetic reading', 'value': '12.30', 'unit': None}]})
    case.runner._context = lambda layout, record: current_recording_context(case.config, layout, record)
    # The frozen capability refuses changed model inputs. A previous
    # publisher-owned scope is independently verified and retained instead.
    assert case.runner.refresh_index(case.layout, recording_id=case.record['recording_id'])
    assert read_json(case.layout.index)['understandings'] == native
    assert read_json(case.layout.index)['historical_read_quarantines'][0]['previous_published_scope'] == 'verified_and_preserved'
    result = publisher.tick(archives=[case.layout.name])
    assert result['published'] == 1 and result['status'] == 'partial', result
    assert result['historical_publications'][0]['status'] == 'publication_verification_pending'
    assert read_json(case.layout.index)['understandings'] == native
    current_mu = read_json(safe_child(case.layout.root, old_mu))
    assert current_mu['understanding'] == before_mu['understanding']
    assert current_mu['内容记录']['溯源']['历史执行来源'] == before_mu['内容记录']['溯源']['历史执行来源']
    assert current_mu['内容记录']['关联更新状态']['current_association'] == 'pending_verification'
    assert len(current_mu['内容记录']['仪器与现场记录']) == 1
    field = read_json(case.layout.root/'Comment/FieldEvents.json')
    event, = field['events']
    assert event['capture_device_id'] == case.record['camera_key'] and event['instrument_id'] == 'instrument-native'
    assert event['payload']['normalized_values'] == [{'name': 'owned synthetic reading', 'value': '12.30', 'unit': None}]
    assert event['payload']['test_only'] is True and field['model_invoked'] is False
    daily = read_json(case.layout.root/'LaboratoryDailyReport/LaboratoryDailyReport.json')
    assert len(daily['全天记录']['现场事件时间线']) == 1
    assert len(daily['全天记录']['全天时间线']) == len(source['segments'])
    assert read_json(case.layout.root/'Comment/TimeIndex.json')['field_events_file']['path'] == 'Comment/FieldEvents.json'
    assert (folder(case.config, case.entry)/'Published.json').read_bytes() == old_proof
    assert checked(case.config, case.entry)['status'] == 'publication_verification_pending'
    assert queue_rows(case) == before_queue
    assert all(path.read_bytes() == raw for path, raw in receipts.items())


@pytest.mark.parametrize('damage', [None, 'proof', 'source_bytes', 'native_receipt'])
def test_quarantined_historical_record_does_not_block_another_completed_record(delivery, damage):
    from test_device_day import FakeModels, capture
    from visioncortex.device_day import DeviceDayRunner, current_recording_context
    from visioncortex.device_day_contract import atomic_json
    from visioncortex.nas_recordings import scan_recordings
    case = delivery
    assert not FileIndexPublisher(case.config).tick(archives=[case.layout.name])['errors']
    original_meanings = deepcopy(read_json(case.layout.index)['understandings'])
    original_paths = [safe_child(case.layout.root, row['content']) for row in
        read_json(case.layout.root/'MultimodalUnderstanding/Index.json')['entries']]
    before_queue = queue_rows(case)
    receipts = {path: path.read_bytes() for path in (case.layout.receipts/case.record['recording_id']).glob('*.json')}
    # A new note is current association data, never an old model input.
    atomic_json(case.layout.comments/'Protocol.json', {'text': 'new association fixture'})
    if damage == 'proof':
        pointer = read_json(folder(case.config, case.entry)/'Published.json')
        path = safe_child(case.layout.root, pointer['receipt']['path'])
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
    elif damage == 'source_bytes':
        path = safe_child(case.layout.root, case.parents['retention']['sources'][0]['retained']['path'])
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
    elif damage == 'native_receipt':
        path = case.layout.receipts/case.record['recording_id']/'understanding.json'
        path.write_bytes(b'{malformed owned fixture receipt')
        receipts[path] = path.read_bytes()
    capture(case.config, start=case.record['recording_end_us'], label='101500')
    second = next(row for row in scan_recordings(case.config)['recordings']
                  if row['recording_id'] != case.record['recording_id'])
    class CurrentPathModels(FakeModels):
        def understand(self, layout, recording, vision, context, key):
            from visioncortex.device_day_content_paths import analysis_folder
            from visioncortex.device_day_contract import artifact, digest
            self.semantic_calls += 1
            item = deepcopy(original_meanings[0])
            item.pop('historical_execution_read', None)
            item['segment_id'] = vision['segments'][0]['segment_id']
            path = analysis_folder(layout, vision['segments'][0], key)/'Result.json'
            atomic_json(path, item)
            return {'vision_key': vision['key'], 'context_digest': digest(context),
                    'understandings': [item], 'artifacts': [artifact(layout.root, path)]}
    models = CurrentPathModels()
    runner = DeviceDayRunner(case.config, models)
    processed = runner.process(second)
    assert processed['status'] == 'completed', processed
    assert models.vision_calls == models.semantic_calls == 1  # Only the new record.
    assert runner.refresh_index(case.layout, recording_id=second['recording_id'])
    canonical = read_json(case.layout.index)
    assert second['recording_id'] in {row['recording_id'] for row in canonical['recordings']}
    assert any(row['segment_id'] not in {m['segment_id'] for m in original_meanings}
               for row in canonical['understandings'])
    marker, = canonical['historical_read_quarantines']
    assert marker['recording_id'] == case.record['recording_id'] and marker['current_association'] == 'pending_verification'
    old = [row for row in canonical['understandings'] if row['segment_id'] in {m['segment_id'] for m in original_meanings}]
    if damage is None:
        assert marker['previous_published_scope'] == 'verified_and_preserved' and old == original_meanings
        row = next(row for row in canonical['recordings'] if row['recording_id'] == case.record['recording_id'])
        assert row['historical_association']['producer_context_unchanged'] is True
    else:
        assert marker['previous_published_scope'] == 'unavailable_original_files_retained' and old == []
    assert checked(case.config, case.entry)['status'] == 'publication_verification_pending'
    assert all(path.is_file() for path in original_paths)
    assert all(path.read_bytes() == raw for path, raw in receipts.items())
    assert queue_rows(case) == before_queue
    # The dedicated strict read port remains quarantined for the new context.
    assert current_recording_context(case.config, case.layout, case.record) != case.current_context


@pytest.mark.parametrize('damage', ['mu_video', 'mu_status', 'mu_decision', 'mu_narrative', 'mu_steps', 'mu_state',
                                    'retention_value', 'retention_video', 'transcript_json', 'transcript_text',
                                    'daily_audio', 'daily_events', 'daily_synthesis', 'daily_entries',
                                    'mu_index_entry', 'mu_index_progress', 'time_index_entry', 'field_events'])
def test_first_publication_proof_refuses_inconsistent_direct_user_fields(delivery, damage):
    from visioncortex.device_day_content_paths import clip_name
    from visioncortex.device_day_contract import atomic_json
    from visioncortex.device_day_historical_publication import publish
    case = delivery
    assert not FileIndexPublisher(case.config).tick(archives=[case.layout.name])['errors']
    index = read_json(case.layout.index)
    from visioncortex.device_day_content import with_partial_understandings
    index = with_partial_understandings(case.config, index)
    assert publish(case.config, case.entry, index)['status'] == 'recovered_published'
    mu = read_json(case.layout.root/'MultimodalUnderstanding/Index.json')['entries'][0]['content']
    candidate = index['segments'][0]
    if damage == 'field_events':
        path = case.layout.root/'Comment/FieldEvents.json'
    elif damage.startswith('mu_index_'):
        path = case.layout.root/'MultimodalUnderstanding/Index.json'
    elif damage == 'time_index_entry':
        path = case.layout.root/'Comment/TimeIndex.json'
    elif damage.startswith('daily_'):
        path = case.layout.root/'LaboratoryDailyReport/LaboratoryDailyReport.json'
    elif damage.startswith('transcript_'):
        audio, = read_json(case.layout.root/'Comment/Index.json')['entries']
        path = safe_child(case.layout.root, audio['text' if damage == 'transcript_text' else 'content'])
    else:
        path = safe_child(case.layout.root, mu if damage.startswith('mu_') else
                          'ProcessedClips/Clips/'+clip_name(candidate)+'/Retention.json')
    if damage == 'transcript_text':
        path.write_text('words the native producer never returned\n')
        (folder(case.config, case.entry)/'Published.json').unlink()
        with pytest.raises(ValueError, match='transcript_text'):
            publish(case.config, case.entry, index)
        assert not (folder(case.config, case.entry)/'Published.json').exists()
        return
    value = read_json(path)
    if damage in {'mu_video', 'retention_video'}:
        value['video'] = candidate['source_ref']
    elif damage == 'mu_status':
        value['video_status'] = 'available'
    elif damage == 'mu_decision':
        value['处理视频留存']['video_value'] = 'confirmed_no_value'
    elif damage == 'mu_narrative':
        value['内容记录']['留存判断']['决定'] = 'omit_static_derivative'
    elif damage == 'mu_steps':
        value['experiment_steps'] = [{'action': 'an action absent from the native model'}]
    elif damage == 'mu_state':
        value['status'] = 'unproven replacement state'
    elif damage == 'transcript_json':
        value.update(sentences=[{'text': 'words the native producer never returned'}], outcome='speech_recognized')
    elif damage == 'daily_audio':
        value['全天记录']['录音时间线'] = [{'原话': 'words the native producer never returned'}]
    elif damage == 'daily_events':
        value['全天记录']['现场事件时间线'] = [{'内容': 'an event absent from the native archive'}]
    elif damage == 'daily_synthesis':
        value['全天综合理解'] = {'status': 'completed_for_available_inputs',
                                  'findings': [{'title': 'no model receipt', 'text': 'an invented conclusion'}]}
    elif damage == 'daily_entries':
        value['entries'][0]['understanding'] = {'windows': [{'summary': 'not returned by native model'}]}
    elif damage == 'mu_index_entry':
        value['entries'][0].update(video_status='available', 状态说明='unsupported completed video claim')
    elif damage == 'mu_index_progress':
        value['理解进度']['已返回结果'] += 1
    elif damage == 'time_index_entry':
        value['recordings'][0]['capture_complete'] = not value['recordings'][0]['capture_complete']
    elif damage == 'field_events':
        value['events'] = [{'payload': {'value': 'fabricated', 'unit': 'fabricated'}}]
        time_path = case.layout.root/'Comment/TimeIndex.json'
        time_index = read_json(time_path)
        time_index.pop('field_events_file', None)
        atomic_json(time_path, time_index)
    else:
        value['video_value'] = 'confirmed_no_value'
    atomic_json(path, value)
    (folder(case.config, case.entry)/'Published.json').unlink()  # No trusted first seal exists.
    with pytest.raises(ValueError, match='readable_|clips_retention|daily_report_business|time_index_|native_field_context'):
        publish(case.config, case.entry, index)
    assert not (folder(case.config, case.entry)/'Published.json').exists()


@pytest.mark.parametrize('historical_case', ['selected'], indirect=True)
@pytest.mark.parametrize('damage', [None, 'missing_video', 'same_size_video', 'mapping', 'wrong_child_mu'])
def test_actual_selected_bytes_mapping_and_child_must_match_publication_proof(delivery, damage):
    from visioncortex.device_day_contract import artifact, atomic_json
    case = delivery
    publisher = FileIndexPublisher(case.config)
    result = publisher.tick(archives=[case.layout.name])
    if result.get('canonical_video_pending'):
        from visioncortex.publication_journal import reconcile
        assert result['historical_publications'][0]['status'] == 'publication_verification_pending'
        assert checked(case.config, case.entry)['status'] == 'publication_verification_pending'
        assert reconcile(case.runner) == 1  # Actual existing owner, no manual index/queue edits.
        result = publisher.tick(archives=[case.layout.name])
    assert not result['errors'], result
    assert result['historical_publications'][0]['status'] == 'recovered_published'
    pointer = read_json(folder(case.config, case.entry)/'Published.json')
    proof = read_json(safe_child(case.layout.root, pointer['receipt']['path']))
    selected, = proof['selected_videos']
    assert proof['selected_duration_ms'] == selected['duration_ms'] == 1000
    assert proof['partition_counts'] == {'pending': 1, 'selected': 1}
    assert selected['source_mapping']['source_start_ms'] == 0
    assert selected['source_mapping']['source_end_ms'] == 1000
    if damage is None:
        assert checked(case.config, case.entry)['status'] == 'recovered_published'
        return
    video = safe_child(case.layout.root, selected['video']['path'])
    if damage == 'missing_video':
        video.unlink()
    elif damage == 'same_size_video':
        raw = video.read_bytes()
        video.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
    elif damage == 'wrong_child_mu':
        reference = next(ref for ref in proof['files'] if ref['path'].endswith('/Understanding.json')
                         and read_json(safe_child(case.layout.root, ref['path']))['segment_id'] == selected['segment_id'])
        path = safe_child(case.layout.root, reference['path'])
        value = read_json(path)
        value['source_start_ms'] += 1
        atomic_json(path, value)
    else:
        receipt = safe_child(case.layout.root, selected['receipt']['path'])
        value = read_json(receipt)
        value['source_mapping']['source_end_ms'] += 1
        atomic_json(receipt, value)
        # Even a mutable index re-hash cannot authenticate a wrong sealed map.
        source = read_json(case.layout.index)
        row = next(row for row in source['segments'][0]['retention_intervals'] if row['video_value'] == 'selected')
        row['receipt'] = artifact(case.layout.root, receipt)
        row['source_mapping'] = value['source_mapping']
        atomic_json(case.layout.index, source)
    assert checked(case.config, case.entry)['status'] == 'publication_verification_pending'
    assert processing_health(case.config)['stages']['understanding']['workload']['publication_pending'] == 1
