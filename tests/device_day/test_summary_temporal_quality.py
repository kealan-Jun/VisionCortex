"""Impossible prose extents are separate from immutable model evidence."""
from copy import deepcopy

import pytest

from visioncortex.device_day_narrative import effective_meaning


@pytest.mark.parametrize('summary', ['约前4分钟（f1–f39）可见实验员操作。', 'The first 4 minutes show handling.'])
def test_impossible_local_duration_is_excluded_without_removing_bounded_steps(summary):
    step = {'description': '可见拿起器具', 'start_ms': 10, 'end_ms': 20}
    value = {'windows': [{'start_ms': 0, 'end_ms': 4700, 'summary': summary,
                         'model_receipt': 'Result.json', 'experiment_steps': [step]}]}
    original = deepcopy(value)
    result = effective_meaning(value)
    window, = result['windows']
    assert window['summary'] is None and window['original_summary'] == summary
    assert window['summary_status'] == 'invalid_temporal_extent'
    assert window['experiment_steps'] == [step]
    assert window['invalid_narrative_content'][0]['window_duration_ms'] == 4700
    assert value == original
    assert effective_meaning(result) == result


@pytest.mark.parametrize('summary', ['此前4分钟发生了什么无法确认。', '提前4分钟的计划无法确认。',
                                     'SOP要求等待4分钟，未确认执行。', '原片第4分钟的画面。',
                                     '前4秒可见操作。'])
def test_source_history_plans_and_valid_local_extent_are_not_removed(summary):
    value = {'windows': [{'start_ms': 0, 'end_ms': 4700, 'summary': summary}]}
    assert effective_meaning(value) is value


def test_projected_child_uses_parent_model_window_for_prose_scope():
    value = {'windows': [{'start_ms': 0, 'end_ms': 2000, 'summary': '前8秒可见人物。',
                         'projection_from_parent': True,
                         'parent_model_window': {'start_ms': 0, 'end_ms': 30000}}]}
    assert effective_meaning(value) is value
