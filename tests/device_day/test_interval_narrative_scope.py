"""Readable child intervals distinguish original observations from video pixels."""
from copy import deepcopy

import pytest

from visioncortex.device_day_narrative import segment_record
from visioncortex.device_day_retention_projection import project_meaning


def case():
    origin = 1792044420000000  # 2026-10-15 14:07:00 +08:00
    step = {'start_ms': 232000, 'end_ms': 239000, 'action': '操作笔记本电脑', 'objects': ['笔记本电脑'],
            'description': '原模型观察到键盘与触控板操作', 'before_state': '232000前画面无人',
            'after_state': '239000后手离开画面', 'visible_evidence': '手部姿态与屏幕变化',
            'outcome': 'visible_change', 'basis': 'observed', 'category': 'computer',
            'frame_ids': [f'f{stamp}' for stamp in range(232, 240)]}
    meaning = {'segment_id': 'parent', 'status': 'completed', 'windows': [{'start_ms': 224000, 'end_ms': 254000,
        'summary': '前8秒画面无人物；其后可见电脑操作，全程未见实验器具操作。',
        'activity_kind': 'non_experiment', 'observability': 'clear', 'input': 'Input.json',
        'model_receipt': 'Result.json', 'experiment_steps': [], 'work_activities': [step], 'steps': [step],
        'source_frames': [{'frame_id': f'f{stamp}', 'local_ms': stamp * 1000,
                           'capture_us': origin + stamp * 1000000, 'path': f'Frames/{stamp}.jpg'}
                          for stamp in range(224, 254)], 'frame_observations': [], 'uncertainties': []}]}
    interval = {'start_ms': 232000, 'end_ms': 239000, 'video_value': 'selected',
                'reason': 'observed_non_experiment_work',
                'evidence': [{'input': 'Input.json', 'model_receipt': 'Result.json'}],
                'core_action_ranges': [{'start_ms': 232000, 'end_ms': 239000}], 'context_basis': [],
                'context_subranges': []}
    child = {'segment_id': 'child', 'candidate_segment_id': 'parent', 'recording_id': 'r',
             'start_ms': 232000, 'end_ms': 239000, 'start_us': origin + 232000000,
             'end_us': origin + 239000000, 'activity': 'inactive', 'key_frames': [],
             'source_ref': {'path': 'MetaData/140700/rgb.mp4'}, 'retention_interval': interval}
    recording = {'recording_id': 'r', 'camera_key': 'camera_cam02'}
    return child, meaning, recording


def test_seven_second_child_labels_original_thirty_second_summary_and_absolute_scope():
    child, meaning, recording = case()
    before = deepcopy(meaning)
    projected = project_meaning(meaning, child)
    view = segment_record(child, projected, recording)
    observation = view['分窗口观察'][0]
    assert observation['内容'] == meaning['windows'][0]['summary']  # Preserve actual model wording.
    assert '不能作为当前子片段的完整结论' in observation['范围说明']
    parent = observation['原模型窗口范围']
    assert (parent['原片开始毫秒'], parent['原片结束毫秒']) == (224000, 254000)
    assert (parent['开始时间'], parent['结束时间']) == (
        '2026-10-15T14:10:44.000+08:00', '2026-10-15T14:11:14.000+08:00')
    current = observation['当前片段范围']
    assert (current['开始时间'], current['结束时间']) == (view['开始时间'], view['结束时间'])
    assert current['区间语义'] == '左闭右开'
    assert view['溯源']['执行记录'][0]['原模型窗口范围'] == parent
    assert meaning == before


def test_exclusive_end_frame_is_only_outside_source_evidence_and_states_remain_scoped():
    child, meaning, recording = case()
    projected = project_meaning(meaning, child)
    before = deepcopy(projected)
    row = segment_record(child, projected, recording)['非实验工作'][0]
    assert row['画面引用'] == [f'f{stamp}' for stamp in range(232, 239)]
    assert [f['画面编号'] for f in row['片外原片边界证据']] == ['f239']
    boundary = row['片外原片边界证据'][0]
    assert boundary['原片毫秒'] == 239000 and boundary['原片证据']['path'] == 'Frames/239.jpg'
    assert '不属于当前派生视频内的像素' in boundary['说明']
    original = meaning['windows'][0]['work_activities'][0]
    assert (row['操作前'], row['操作后']) == (original['before_state'], original['after_state'])
    assert '可能来自当前子片段外' in row['前后状态说明']
    assert projected == before


def test_cross_boundary_step_is_not_presented_as_child_work():
    child, meaning, recording = case()
    crossing = meaning['windows'][0]['work_activities'][0]
    crossing.update(start_ms=231000, end_ms=240000, frame_ids=['f231', 'f239', 'f240'])
    before = deepcopy(meaning)
    projected = project_meaning(meaning, child)
    view = segment_record(child, projected, recording)
    assert view['实验步骤'] == view['非实验工作'] == []
    assert projected['windows'][0]['overlapping_parent_observations'][0]['original_observation'] == crossing
    assert view['分窗口观察'][0]['原模型窗口范围']['原片开始毫秒'] == 224000
    assert meaning == before


def test_unprojected_legacy_summary_and_step_fields_remain_compatible():
    child, meaning, recording = case()
    legacy = child | {'segment_id': 'parent', 'start_ms': 224000, 'end_ms': 254000,
                      'start_us': child['start_us'] - 8000000, 'end_us': child['end_us'] + 15000000}
    legacy.pop('retention_interval')
    before = deepcopy(meaning)
    view = segment_record(legacy, meaning, recording)
    assert view['分窗口观察'][0]['内容'] == meaning['windows'][0]['summary']
    assert '原模型窗口范围' not in view['分窗口观察'][0]
    assert '前后状态说明' not in view['非实验工作'][0]
    assert view['非实验工作'][0]['画面引用'] == meaning['windows'][0]['work_activities'][0]['frame_ids']
    assert meaning == before


def test_selected_context_uses_exact_partition_reason_and_evidence_without_inventing_work():
    child, _, recording = case()
    interval = child['retention_interval']
    interval.update(reason='explicit_bounded_necessary_context', core_action_ranges=[],
                    context_basis=[{'start_ms': 232000, 'end_ms': 239000, 'input': 'Input.json',
                                    'model_receipt': 'Result.json', 'reason': '实际来源支持的观看上下文'}],
                    context_subranges=[{'start_ms': 232000, 'end_ms': 239000}])
    before = deepcopy(child)
    view = segment_record(child, None, recording)
    retention = view['留存判断']
    assert view['实验步骤'] == view['非实验工作'] == []
    assert retention['决定'] == 'retain_derivative' and retention['视频价值'] == interval['video_value']
    assert retention['原因'] == interval['reason']
    assert retention['价值依据'] == [interval['reason']]
    assert retention['区间记录'] == [interval]
    assert retention['区间价值依据'] == interval['evidence']
    assert retention['动作原始区间'] == interval['core_action_ranges']
    assert retention['必要上下文依据'] == interval['context_basis']
    assert retention['必要上下文区间'] == interval['context_subranges']
    assert not retention['原视频允许删除'] and child == before


@pytest.mark.parametrize(('value', 'decision'), [
    ('pending', 'pending_value_review'), ('confirmed_no_value', 'omit_static_derivative'),
])
def test_child_value_status_is_not_inferred_again_from_returned_work(value, decision):
    child, meaning, recording = case()
    child['retention_interval'].update(video_value=value, reason='exact_partition_reason')
    view = segment_record(child, project_meaning(meaning, child), recording)
    assert view['非实验工作']  # Returned observations do not override published interval value.
    assert view['留存判断']['决定'] == decision
    assert view['留存判断']['视频价值'] == value
    assert view['留存判断']['原因'] == 'exact_partition_reason'
    assert view['留存判断']['价值依据'] == []
