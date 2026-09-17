from copy import deepcopy
from pathlib import Path

from test_device_day import FakeModels, capture, device_config, item_and_layout
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_content import content_revision, publish_content, with_partial_understandings
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_file_index import FileIndexPublisher

__all__ = ['device_config']


def test_readable_time_folders_do_not_change_sealed_paths_or_run_models(device_config):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    runner.process(record)
    original = read_json(layout.index)
    result = publish_content(device_config, original)
    transcript = layout.root / result['audio'][0]['content']
    understanding = layout.root / result['multimodal'][0]['content']
    assert transcript.name == 'Transcript.json' and transcript.parent.name.startswith('10-00-00.')
    assert read_json(transcript)['outcome'] == 'no_audio'
    assert read_json(understanding)['understanding']['windows'][0]['summary']
    assert read_json(layout.index) == original
    assert backend.vision_calls == backend.semantic_calls == 1
    before = transcript.stat().st_mtime_ns
    publish_content(device_config, original)
    assert transcript.stat().st_mtime_ns == before


def test_valid_window_is_visible_before_the_recording_finishes(device_config):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(record, stage='retention')
    runner.process(record, stage='vision')
    index = read_json(layout.index)
    segment = index['segments'][0]
    key = 'a' * 64
    folder = layout.understanding / 'ClipUnderstanding' / segment['segment_id'] / key / '00-00_00-30'
    stage = layout.receipts / record['recording_id'] / 'understanding.json'
    atomic_json(stage, {'key': key, 'status': 'running'})
    frame = segment['scene_frames'][0]
    metadata = {'segment_id': segment['segment_id'], 'source_ref': segment['source_ref'],
                'start_ms': 0, 'end_ms': 30000, 'frames': [frame], 'comments': []}
    atomic_json(folder/'Input.json', {'input_key': 'input', 'metadata': metadata})
    result = {'status': 'completed', 'summary': '真实结果的确定性替身', 'activity_observed': 'inactive',
              'steps': [], 'frame_observations': [{'frame_id': frame['frame_id'], 'text': '测试画面'}], 'uncertainties': []}
    atomic_json(folder/'Result.json', {'input_key': 'input', 'model_result': result})
    before = content_revision(device_config, index)
    partial = with_partial_understandings(device_config, index)
    assert not index['understandings']
    assert partial['understandings'][0]['status'] == 'partial'
    assert partial['understandings'][0]['windows'][0]['summary'] == result['summary']
    atomic_json(stage, {'key': key, 'status': 'failed', 'message': 'second window failed'})
    assert content_revision(device_config, index) != before
    partial = with_partial_understandings(device_config, index)
    assert partial['understandings'][0]['stage_status'] == 'failed'
    assert partial['recordings'][0]['stages']['understanding']['message'] == 'second window failed'
    atomic_json(folder/'Result.json', {'input_key': 'other input', 'model_result': result})
    assert not with_partial_understandings(device_config, index)['understandings']


def test_new_coverage_policy_never_invalidates_completed_cv(device_config):
    capture(device_config)
    record, _ = item_and_layout(device_config)
    original = DeviceDayRunner(device_config, FakeModels())
    changed = deepcopy(device_config)
    changed['device_day'].update(understanding_coverage_since_us=record['recording_start_us'],
                                 understanding_frames_per_request=48)
    current = DeviceDayRunner(changed, FakeModels())
    for stage in ['retention', 'vision', 'stt']:
        assert original._key(stage, record, {}) == current._key(stage, record, {})
    assert original._key('understanding', record, {}) != current._key('understanding', record, {})
    old = dict(record, recording_start_us=record['recording_start_us'] - 1)
    assert original._key('understanding', old, {}) == current._key('understanding', old, {})


def test_background_publisher_preserves_cv_index_and_updates_report(device_config):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    runner.process(record)
    from visioncortex.observed_inventory import observe
    observe(Path(device_config['storage']['local_runtime_root'])/'device-day', {'recordings': [record]})
    original = read_json(layout.index)
    device_config['device_day']['readable_content_enabled'] = True
    FileIndexPublisher(device_config).tick()
    assert read_json(layout.index) == original
    assert read_json(layout.comments/'TimeIndex.json')['readable_content']['multimodal']
    assert read_json(layout.reports/'LaboratoryDailyReport.json')['entries'][0]['understanding_status'] == 'completed'
    assert backend.vision_calls == backend.semantic_calls == 1


def test_projection_does_not_duplicate_dense_cv_audit(device_config):
    original = {'archive': '2026-09-10_a_cam01', 'recordings': [{'recording_id': 'r', 'processing': {
        'batches': [{'large': 'dense audit'}], 'clock_mapping': {'origin_us': 1}}}],
        'segments': [{'recording_id': 'r', 'activity_audit': {'large': 'dense audit'}}], 'understandings': []}
    projected = with_partial_understandings(device_config, original)
    assert 'batches' not in projected['recordings'][0]['processing']
    assert 'activity_audit' not in projected['segments'][0]
    assert projected['recordings'][0]['processing']['clock_mapping'] == {'origin_us': 1}
    assert original['recordings'][0]['processing']['batches']
