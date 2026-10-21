"""Published-file lookup fixtures prove contracts, not model/video accuracy."""
from copy import deepcopy

from fastapi import FastAPI
from fastapi.testclient import TestClient

from visioncortex.device_day_contract import atomic_json
from visioncortex.device_day_file_index import build_file_index
from visioncortex.device_day_timeline import day_bounds, install_routes, load_timeline


DAY = '2026-10-21'
BASE = day_bounds(DAY)[0]
ARCHIVE = DAY + '_camera_cam01'


def published(tmp_path):
    root = tmp_path / 'archive' / ARCHIVE
    config = {'storage': {'archive_root': str(root.parent), 'local_runtime_root': str(tmp_path / 'runtime')},
              'device_day': {}, 'collection_ingest': {'camera_role_map': {}, 'source_root': str(tmp_path/'capture')}}
    record = {'recording_id': 'r', 'camera_key': 'camera_cam01', 'start_us': BASE,
              'end_us': BASE + 10_000_000, 'sources': [
                  {'kind': 'video', 'retained': {'path': 'MetaData/Capture/rgb.mp4', 'sha256': 'source-video'}},
                  {'kind': 'audio_audio', 'retained': {'path': 'MetaData/Capture/audio.opus', 'sha256': 'source-audio'}}],
              'processing': {'clock_mapping': {'origin_us': BASE}},
              'audio': {'start_us': BASE + 500_000, 'end_us': BASE + 11_000_000},
              'transcription': {'status': 'completed', 'outcome': 'transcribed',
                  'transcript_file': {'path': 'Comment/00-00-00_00-00-11/Transcript.txt'},
                  'comments': [{'start_us': BASE + 3_000_000, 'end_us': BASE + 5_000_000,
                                'text': 'fixture transcript', 'transcript_path': 'Comment/00-00-00_00-00-11/Transcript.json'}]}}
    segment = {'recording_id': 'r', 'segment_id': 'candidate', 'activity': 'active',
               'start_us': BASE, 'end_us': BASE + 10_000_000, 'start_ms': 0, 'end_ms': 10000,
               'source_ref': record['sources'][0]['retained'], 'key_frames': [], 'scene_frames': [],
               'json_path': 'ProcessedClips/Clips/00-00-00_00-00-10_ExperimentActivity_candidate/ExperimentActivity.json'}
    content = 'MultimodalUnderstanding/00-00-00_00-00-10_ExperimentActivity_candidate/Understanding.json'
    index = {'archive': ARCHIVE, 'recordings': [record], 'segments': [segment],
             'understandings': [{'segment_id': 'candidate', 'status': 'partial', 'windows': []}],
             'readable_content': {'multimodal': [{'segment_id': 'candidate', 'content': {'path': content}}]}}
    time_index = build_file_index(config, index, {})
    atomic_json(root / 'Comment/TimeIndex.json', time_index)
    canonical = root / 'ProcessedClips/Index.json'
    canonical.parent.mkdir(parents=True)
    with canonical.open('wb') as handle:
        handle.truncate(128 * 1024**2 + 1)
    return config, root, time_index, content


def test_timeline_uses_small_published_times_without_opening_oversized_canonical(tmp_path):
    config, root, _, content = published(tmp_path)
    before = (root / 'ProcessedClips/Index.json').stat()
    result = load_timeline(config, DAY)
    assert not result['errors']
    assert result['indexed_recordings'] == len(result['entries']) == 1
    entry = result['entries'][0]
    assert entry['understanding_url'].endswith(content)
    assert entry['steps'] == entry['summaries'] == []
    assert entry['understanding_ready'] and not entry['physical_action_confirmed']
    assert (root / 'ProcessedClips/Index.json').stat() == before


def test_http_time_query_keeps_independent_audio_clock_and_real_transcript_reference(tmp_path):
    config, _, _, _ = published(tmp_path)
    app = FastAPI()
    install_routes(app, lambda: config)
    response = TestClient(app).get(f'/api/day-timeline/{DAY}/at', params={
        'at_us': BASE + 3_500_000, 'duration_seconds': 1})
    assert response.status_code == 200
    row = response.json()['devices'][0]['recordings'][0]
    assert row['video_offset_seconds'] == 3.5 and row['audio_offset_seconds'] == 3
    assert row['comments'][0]['text'] == 'fixture transcript'
    assert row['comments'][0]['transcript_url'].endswith('/Transcript.json')
    assert row['audio_ref']['sha256'] == 'source-audio'


def test_selected_child_does_not_duplicate_parent_observation_period(tmp_path):
    config, root, times, _ = published(tmp_path)
    parent = next(r for r in times['streams']['video'] if r['kind'] == 'video_segment')
    times['streams']['video'].insert(0, deepcopy(parent) | {
        'segment_id': 'selected-child', 'record_scope': 'retention_interval', 'candidate_segment_id': 'candidate'})
    atomic_json(root / 'Comment/TimeIndex.json', times)
    result = load_timeline(config, DAY)
    assert [r['segment_id'] for r in result['entries']] == ['candidate']
    assert result['covered_seconds'] == 10


def test_wrong_day_time_index_is_isolated_without_falling_back_to_large_file(tmp_path):
    config, root, times, _ = published(tmp_path)
    times['archive'] = '2026-10-20_camera_cam01'
    atomic_json(root / 'Comment/TimeIndex.json', times)
    result = load_timeline(config, DAY)
    assert not result['entries']
    assert result['errors'][0]['error_type'] == 'ReadModelDeferred'


def test_published_file_times_remain_readable_while_canonical_index_is_unavailable(tmp_path):
    config, root, _, _ = published(tmp_path)
    (root/'ProcessedClips/Index.json').unlink()
    result = load_timeline(config, DAY)
    assert not result['errors'] and len(result['media_recordings']) == 1


def test_legacy_time_index_keeps_independent_bounded_canonical_reader(tmp_path):
    from visioncortex.application.device_day_readmodel import DeviceDayReadModels
    _, root, times, _ = published(tmp_path)
    times['schema_version'] = 'visioncortex-file-time-index/1'
    atomic_json(root/'Comment/TimeIndex.json', times)
    legacy = {'archive': ARCHIVE, 'recordings': [], 'segments': [], 'understandings': []}
    atomic_json(root/'ProcessedClips/Index.json', legacy)
    result = DeviceDayReadModels(lambda value: value).timeline_index(root)
    assert result['read_projection']['source'] == 'ProcessedClips/Index.json'


def test_native_pts_metadata_is_preserved_without_reading_dense_cv_index(tmp_path):
    config, root, times, _ = published(tmp_path)
    points = [[i / 10, BASE + i * 100] for i in range(100000)]
    original = next(r for r in times['streams']['video'] if r['kind'] == 'original_video')
    original['clock_mapping']['points'] = points
    atomic_json(root/'Comment/TimeIndex.json', times)
    result = load_timeline(config, DAY)
    assert not result['errors']
    assert result['media_recordings'][0]['clock_mapping']['points'] == points
