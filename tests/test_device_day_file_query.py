from copy import deepcopy

import pytest

from test_device_day_file_index import config
from test_device_day_time_lookup import BASE, index
from visioncortex.device_day_file_index import build_file_index
from visioncortex.device_day_file_query import query_file_indexes


def separate_index(tmp_path):
    source = index()
    source['recordings'][0]['processing'] = {'clock_mapping': {'basis': 'recorder_csv_interpolation',
        'points': [[0, BASE], [1000, BASE+2000000], [3000, BASE+5000000]]}}
    source['segments'] = [{'recording_id': 'slice', 'segment_id': 'clip', 'activity': 'active',
        'start_us': BASE+2000000, 'end_us': BASE+5000000, 'start_ms': 1000, 'end_ms': 3000,
        'source_ref': {'path': 'MetaVideo/Original.mp4'}, 'video': {'path': 'ProcessedClips/Clips/Event.mp4'},
        'key_frames': [], 'scene_frames': [{'path': 'ProcessedClips/Clips/Scene.jpg', 'capture_us': BASE+3500000,
                                           'frame_kind': 'scene_sample', 'understanding_text': '融合说明'}]}]
    source['understandings'] = [{'segment_id': 'clip', 'windows': [{'summary': '融合说明引用了说话内容',
        'model_receipt': 'MultimodalUnderstanding/ClipUnderstanding/Result.json',
        'input': 'MultimodalUnderstanding/ClipUnderstanding/Input.json'}]}]
    return build_file_index(config(tmp_path), source, {})


def test_video_audio_and_combined_interpretation_remain_separate(tmp_path):
    import json
    data = separate_index(tmp_path)
    assert '融合说明' not in json.dumps(data, ensure_ascii=False)
    assert 'understanding' not in data['streams']['video'][1]
    assert data['streams']['audio'][0]['transcription']['sentences'][0]['text'] == '样本转写'
    assert data['streams']['multimodal'][0]['content_source'] == 'multimodal_context'
    assert data['streams']['multimodal'][0]['files'][0]['path'].endswith('/Result.json')


def test_one_time_returns_separate_files_with_their_own_seek_offsets(tmp_path):
    data = separate_index(tmp_path)
    before = deepcopy(data)
    result = query_file_indexes([data], BASE+3500000, .5)
    streams = result['matches'][0]['streams']
    original, clip = streams['video']
    assert original['file_offset_seconds'] == 2
    assert clip['file_offset_seconds'] == 1  # Relative to the exported clip, not original.
    assert clip['source_video_offset_seconds'] == 2
    assert clip['playback_file']['path'] == 'ProcessedClips/Clips/Event.mp4'
    assert streams['audio'][0]['file_offset_seconds'] == 3  # Audio has its own clock origin.
    assert streams['audio'][0]['transcription']['sentences'][0]['text'] == '样本转写'
    assert len(streams['images']) == 1
    assert data == before


def test_partial_video_cannot_extrapolate_missing_capture_time(tmp_path):
    data = separate_index(tmp_path)
    result = query_file_indexes([data], BASE+7000000)
    original = result['matches'][0]['streams']['video'][0]
    assert original['file_offset_seconds'] is None
    assert original['seek_status'] == 'outside_verified_clock_samples'
    assert result['matches'][0]['streams']['audio'][0]['transcription']['sentences'] == []


def test_half_open_boundaries_and_unknown_audio_time(tmp_path):
    data = separate_index(tmp_path)
    data['streams']['audio'][0]['start_us'] = None
    result = query_file_indexes([data], BASE+5000000)
    streams = result['matches'][0]['streams']
    assert len(streams['video']) == 1 and streams['audio'] == [] and streams['images'] == []
    assert query_file_indexes([data], BASE+10000000)['matches'] == []


@pytest.mark.parametrize('at,duration', [(True, 1), (0, 1), (BASE, float('nan')), (BASE, 0), (BASE, 3601)])
def test_invalid_query_rejected(at, duration):
    with pytest.raises(ValueError):
        query_file_indexes([], at, duration)
