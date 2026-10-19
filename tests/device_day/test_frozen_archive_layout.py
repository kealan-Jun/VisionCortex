"""Frozen public paths, private evidence, safe relocation; synthetic metadata only."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from visioncortex.device_day_content import publish_content
from visioncortex.device_day_content_paths import (
    LAYOUT_VERSION, analysis_folder, clip_name, layout_contract, migrate_display_layout,
)
from visioncortex.device_day_contract import DeviceDayLayout, artifact, atomic_json, read_json, safe_child, verify_artifact


def fixture(tmp_path):
    start = 1792376825000000
    segment = {'segment_id': 'segment-0123456789abcdef', 'recording_id': 'recording',
        'camera_key': 'camera_cam01', 'activity': 'inactive', 'start_us': start,
        'end_us': start + 120000000, 'start_ms': 0, 'end_ms': 120000,
        'source_ref': {'path': 'MetaData/102705/rgb.mp4', 'size_bytes': 3, 'sha256': 'a' * 64},
        'key_frames': [], 'scene_frames': [], 'video': None}
    segment['json_path'] = 'ProcessedClips/Clips/' + clip_name(segment) + '/NoExperimentActivity.json'
    index = {'archive': '2026-10-19_camera_cam01', 'recordings': [{'recording_id': 'recording',
        'camera_key': 'camera_cam01', 'start_us': start, 'end_us': segment['end_us']}],
        'segments': [segment], 'understandings': []}
    config = {'storage': {'archive_root': str(tmp_path), 'local_cache_root': str(tmp_path/'Cache')},
              'device_day': {}}
    return config, index


def test_public_understanding_matches_clip_and_does_not_create_internal_display_trees(tmp_path):
    config, index = fixture(tmp_path)
    original = deepcopy(index)
    result = publish_content(config, index)
    root = tmp_path/index['archive']
    name = clip_name(index['segments'][0])
    assert {p.name for p in (root/'MultimodalUnderstanding').iterdir()} == {name, 'Index.json'}
    leaf = root/'MultimodalUnderstanding'/name
    assert {p.name for p in leaf.iterdir()} == {'Understanding.json'}
    assert read_json(leaf/'Understanding.json')['layout_version'] == LAYOUT_VERSION
    retention = read_json(root/'ProcessedClips/Clips'/name/'Retention.json')
    assert retention['folder_retained'] and not retention['no_value_proven']
    assert retention['selection_status'] == 'pending_value_review'
    assert not list(root.rglob('*.html'))
    assert result['multimodal'][0]['content'] == 'MultimodalUnderstanding/' + name + '/Understanding.json'
    assert index == original
    private = analysis_folder(SimpleNamespace(root=root), index['segments'][0], 'a' * 64)
    assert private.relative_to(root).parts[:3] == ('MultimodalUnderstanding', '.Evidence', 'Analysis')


def test_declared_but_missing_or_truncated_video_is_not_reported_as_retained(tmp_path):
    config, index = fixture(tmp_path)
    root = tmp_path/index['archive']
    segment = index['segments'][0]
    path = root/'ProcessedClips/Clips'/clip_name(segment)/'NoExperimentActivity.mp4'
    segment['video'] = {'path': path.relative_to(root).as_posix(), 'size_bytes': 3, 'sha256': 'a' * 64}
    result = publish_content(config, index)
    assert result['multimodal'][0]['video_status'] == 'missing'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x')
    assert publish_content(config, index)['multimodal'][0]['video_status'] == 'missing'
    path.write_bytes(b'xxx')
    assert publish_content(config, index)['multimodal'][0]['video_status'] == 'available'


def test_cross_camera_projection_and_arbitrary_clip_paths_fail_closed(tmp_path):
    config, index = fixture(tmp_path)
    index['segments'][0]['camera_key'] = 'other_camera'
    with pytest.raises(ValueError, match='another device'):
        publish_content(config, index)
    index['segments'][0]['json_path'] = 'ProcessedClips/request-id/hash/Result.json'
    with pytest.raises(ValueError, match='frozen public layout'):
        clip_name(index['segments'][0])


def test_frozen_contract_keeps_value_selection_and_visible_paths():
    contract = layout_contract()
    assert contract['changes_require'] == 'explicit_human_user_instruction'
    assert contract['directories'] == ['MetaData', 'ProcessedClips', 'MultimodalUnderstanding', 'LaboratoryDailyReport', 'Comment']
    assert contract['understanding_files'] == ['Understanding.json']
    assert contract['understanding_uses_same_clip_name'] and not contract['visible_hash_directory_chains']
    assert contract['retain_valuable_experiment_video'] and contract['retain_valuable_non_experiment_video']
    assert not contract['inactive_label_is_discard_decision']


@pytest.mark.parametrize('field', ['experiment_steps', 'work_activities'])
def test_understanding_file_directly_contains_real_step_fields(tmp_path, field):
    config, index = fixture(tmp_path)
    step = {'start_ms': 0, 'end_ms': 1000, 'action': '操作', 'objects': ['对象'],
            'description': '可见操作描述', 'before_state': '操作前', 'after_state': '操作后',
            'visible_evidence': '可见变化', 'outcome': 'visible_change', 'basis': 'observed', 'frame_ids': ['f1']}
    index['understandings'] = [{'segment_id': index['segments'][0]['segment_id'], 'status': 'completed',
        'windows': [{'summary': '窗口观察', 'experiment_steps': [], 'work_activities': [], field: [step]}]}]
    result = publish_content(config, index)
    value = read_json(tmp_path/index['archive']/result['multimodal'][0]['content'])
    readable = value['内容记录']['实验步骤' if field == 'experiment_steps' else '非实验工作'][0]
    assert {key: readable[key] for key in ('动作', '操作对象', '操作前', '操作后', '可见依据')} == {
        '动作': '操作', '操作对象': ['对象'], '操作前': '操作前', '操作后': '操作后', '可见依据': '可见变化'}
    assert not list((tmp_path/index['archive']).rglob('*.html'))


def test_bounded_relocation_keeps_original_evidence_and_old_receipts_valid(tmp_path):
    layout = DeviceDayLayout(tmp_path, 'camera_cam01', 1792376825000000)
    old = layout.understanding/'RequestPlans'/'recording'/('a' * 64)/'SamplingCoverage.json'
    atomic_json(old, {'scope': 'synthetic-execution-proof'})
    reference = artifact(layout.root, old)
    original = old.read_bytes()
    raw = layout.raw/'NativeCapture/rgb.mp4'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'original-capture')
    with pytest.raises(ValueError, match='bounded batch'):
        migrate_display_layout(None, layout, max_bytes=1)
    assert old.read_bytes() == original and not (layout.understanding/'.Evidence').exists()
    result = migrate_display_layout(None, layout)
    assert len(result['moved']) == 1 and not (layout.understanding/'RequestPlans').exists()
    assert verify_artifact(layout.root, reference)
    assert '.Evidence' in safe_child(layout.root, reference['path']).parts
    assert raw.read_bytes() == b'original-capture'
    backup = layout.root/result['moved'][0]['backup']/'recording'/('a' * 64)/'SamplingCoverage.json'
    assert backup.read_bytes() == original
    assert migrate_display_layout(None, layout)['moved'] == []
