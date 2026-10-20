"""Export connected positive spans without turning evidence changes into clips."""
from copy import deepcopy

import pytest

from test_retained_interval_partition import action_window, assert_partition, fixture, selected
from visioncortex.media.retained_intervals import partition


def test_overlapping_model_windows_export_one_union_and_keep_each_atomic_scope():
    segment, record, meaning = fixture()
    first, second = action_window(2700, 5266.666666666667), action_window(5000, 7766.666666666667)
    first['experiment_steps'][0].update(action='旋紧瓶盖', after_state='瓶盖仍位于瓶口')
    second['experiment_steps'][0].update(action='旋拧瓶盖', after_state='无法确认开启还是拧紧')
    meaning['windows'] = [first, second]
    before = deepcopy((segment, record, meaning))
    result = partition(segment, meaning, record)
    assert_partition(result)
    row, = selected(result)
    assert (row['start_ms'], row['end_ms']) == (2700, 7766.666666666667)
    assert row['export_group_basis'] == 'contiguous_selected_union'
    assert row['physical_action_confirmed'] is False
    assert [(r['start_ms'], r['end_ms']) for r in row['selected_subranges']] == [
        (2700, 5000), (5000, 5266.666666666667), (5266.666666666667, 7766.666666666667)]
    assert [len(r['evidence']) for r in row['selected_subranges']] == [1, 2, 1]
    assert len(row['evidence']) == len(row['core_action_ranges']) == 2
    assert [r['model_claimed_interval']['start_ms'] for r in row['evidence']] == [2700, 5000]
    assert row['evidence'][1]['after_state'] == '无法确认开启还是拧紧'
    assert all(r['outcome'] == 'incomplete_or_uncertain' for r in row['evidence'])
    assert row['context_basis'] == row['context_subranges'] == []
    assert (segment, record, meaning) == before


@pytest.mark.parametrize('field', ['experiment_steps', 'work_activities'])
def test_adjacent_selected_actions_share_video_but_remain_distinct_observations(field):
    segment, record, meaning = fixture()
    first, second = action_window(1000, 2000, field=field), action_window(2000, 4000, field=field)
    first[field][0].update(action='第一次操作', objects=['对象甲'])
    second[field][0].update(action='第二次操作', objects=['对象乙'])
    meaning['windows'] = [first, second]
    row, = selected(partition(segment, meaning, record))
    assert (row['start_ms'], row['end_ms']) == (1000, 4000)
    assert [r['action'] for r in row['evidence']] == ['第一次操作', '第二次操作']
    assert [r['objects'] for r in row['core_action_ranges']] == [['对象甲'], ['对象乙']]
    assert len(row['selected_subranges']) == 2
    assert row['context_basis'] == []  # Joining positive coverage adds no unobserved context.


def test_category_change_is_a_real_export_boundary_even_when_both_sides_selected():
    segment, record, meaning = fixture()
    meaning['windows'] = [action_window(1000, 2000), action_window(2000, 4000, field='work_activities')]
    rows = selected(partition(segment, meaning, record))
    assert [(r['start_ms'], r['end_ms'], r['activity_category']) for r in rows] == [
        (1000, 2000, 'experiment'), (2000, 4000, 'non_experiment')]
    assert all('selected_subranges' not in row for row in rows)


@pytest.mark.parametrize('gap_ms', [0.001, 400, 15000])
def test_no_pending_gap_is_bridged_by_connected_export_grouping(gap_ms):
    segment, record, meaning = fixture()
    meaning['windows'] = [action_window(1000, 1200), action_window(1200+gap_ms, 1900+gap_ms)]
    result = partition(segment, meaning, record)
    assert_partition(result)
    rows = selected(result)
    assert [(r['start_ms'], r['end_ms']) for r in rows] == [(1000, 1200), (1200+gap_ms, 1900+gap_ms)]
    gap, = [r for r in result['intervals'] if r['start_ms'] == 1200]
    assert gap['end_ms'] == 1200+gap_ms and gap['video_value'] == 'pending'
    assert all(row['context_basis'] == [] for row in rows)
    assert all('selected_subranges' not in row for row in rows)
    assert [row['end_ms']-row['start_ms'] for row in rows] == [200, 700]
    assert all(row['evidence'][0]['boundary_uncertainty'] for row in rows)


def test_conflicting_category_overlap_keeps_the_mixed_region_separate():
    segment, record, meaning = fixture()
    meaning['windows'] = [action_window(1000, 4000), action_window(2000, 3000, field='work_activities')]
    rows = selected(partition(segment, meaning, record))
    assert [(r['start_ms'], r['end_ms'], r['activity_category']) for r in rows] == [
        (1000, 2000, 'experiment'), (2000, 3000, 'unknown'), (3000, 4000, 'experiment')]
    assert rows[1]['activity_categories'] == ['experiment', 'non_experiment']
    assert len(rows[1]['evidence']) == 2
