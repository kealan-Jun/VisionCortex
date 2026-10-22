"""Exact visible spans, not analysis candidates, own derived video decisions."""
from copy import deepcopy

import pytest

from visioncortex.media.retained_intervals import disposition, partition

ORIGIN = 1791936000000000


def fixture():
    segment = {'segment_id': 'candidate', 'recording_id': 'record', 'activity': 'active',
        'start_ms': 0, 'end_ms': 900000, 'start_us': ORIGIN, 'end_us': ORIGIN + 900000000,
        'source_ref': {'path': 'MetaData/Native/Original.avi', 'sha256': 'a' * 64, 'size_bytes': 123},
        'key_frames': []}
    return segment, {'recording_id': 'record', 'start_us': ORIGIN}, {'segment_id': 'candidate', 'status': 'completed', 'windows': []}


def action_window(start, end, *, field='experiment_steps', basis='observed', frame_times=None):
    times = [start, end] if frame_times is None else frame_times
    frames = [{'frame_id': 'f' + str(index), 'local_ms': stamp,
        'path': f'MultimodalUnderstanding/.Evidence/{start}/Frame{index}.jpg',
        'sha256': str(index % 10) * 64, 'size_bytes': index + 1} for index, stamp in enumerate(times)]
    step = {'start_ms': start, 'end_ms': end, 'start_us': ORIGIN + round(start * 1000),
        'end_us': ORIGIN + round(end * 1000), 'basis': basis, 'action': '操作', 'objects': ['工具'],
        'description': '可见具体动作', 'before_state': '动作前', 'after_state': '动作后',
        'visible_evidence': '所引画面可见变化', 'outcome': 'incomplete_or_uncertain',
        'frame_ids': [f['frame_id'] for f in frames], 'comment_ids': []}
    if field == 'work_activities':
        step['category'] = 'computer'
    return {'start_ms': start, 'end_ms': end, 'activity_kind': 'experiment' if field == 'experiment_steps' else 'non_experiment',
        'observability': 'clear', 'experiment_steps': [step] if field == 'experiment_steps' else [],
        'work_activities': [step] if field == 'work_activities' else [], 'source_frames': frames,
        'input': f'MultimodalUnderstanding/.Evidence/{start}/Input.json',
        'model_receipt': f'MultimodalUnderstanding/.Evidence/{start}/Result.json'}


def assert_partition(result, start=0, end=900000):
    rows = result['intervals']
    assert rows[0]['start_ms'] == start and rows[-1]['end_ms'] == end
    assert all(row['start_ms'] < row['end_ms'] and row['start_us'] < row['end_us'] for row in rows)
    assert all(a['end_ms'] == b['start_ms'] and a['end_us'] == b['start_us']
               for a, b in zip(rows, rows[1:], strict=False))
    assert sum(row['end_ms'] - row['start_ms'] for row in rows) == end - start
    assert all(row['video_value'] in {'selected', 'pending', 'confirmed_no_value'} for row in rows)
    assert all('reason' in row and 'evidence' in row and 'context_basis' in row for row in rows)


def selected(result):
    return [row for row in result['intervals'] if row['video_value'] == 'selected']


def test_fifteen_minute_candidate_exports_only_25_plus_20_seconds_and_keeps_rest_pending():
    segment, record, meaning = fixture()
    meaning['windows'] = [action_window(130000, 155000), action_window(540000, 560000, field='work_activities')]
    before = deepcopy((segment, record, meaning))
    result = partition(segment, meaning, record)
    assert_partition(result)
    assert [(r['start_ms'], r['end_ms'], r['activity_category']) for r in selected(result)] == [
        (130000, 155000, 'experiment'), (540000, 560000, 'non_experiment')]
    assert sum(r['end_ms'] - r['start_ms'] for r in selected(result)) == 45000
    assert all(r['video_value'] == 'pending' for r in result['intervals'] if r not in selected(result))
    decision = disposition(segment, meaning, record)
    assert decision['selected_duration_ms'] == 45000 and decision['pending_duration_ms'] == 855000
    assert not decision['candidate_is_export_unit'] and result['default_padding_ms'] == 0
    assert decision['selection_policy'] == 'visioncortex-value-only-derivatives/7'
    assert decision['schema_version'] == 'visioncortex-derived-video-disposition/1'
    assert (segment, record, meaning) == before


def test_model_integer_bounds_shrink_to_actual_cited_frame_pts_and_never_expand():
    segment, record, meaning = fixture()
    window = action_window(130000, 155000, frame_times=[130020.25, 139012.5, 154979.5])
    meaning['windows'] = [window]
    result = partition(segment, meaning, record)
    assert_partition(result)
    row, = selected(result)
    assert (row['start_ms'], row['end_ms']) == (130020.25, 154979.5)
    evidence, = row['evidence']
    assert evidence['model_claimed_interval']['start_ms'] == 130000
    assert evidence['boundary_basis'] == 'observed_source_frame_extent'
    assert evidence['boundary_uncertainty'] and evidence['physical_action_confirmed'] is False
    assert row['start_us'] == ORIGIN + 130020250


@pytest.mark.parametrize('case', ['uncertain', 'inferred', 'one_frame', 'missing_frame', 'outside_frame',
    'boundary_uncertain', 'clock_mismatch', 'missing_receipt', 'outside_window', 'nonfinite', 'zero_length', 'missing_frame_hash'])
def test_uncertain_boundaries_and_incomplete_evidence_never_select_whole_window(case):
    segment, record, meaning = fixture()
    window = action_window(130000, 155000)
    step = window['experiment_steps'][0]
    if case in {'uncertain', 'inferred'}:
        step['basis'] = case
    elif case == 'one_frame':
        step['frame_ids'] = step['frame_ids'][:1]
    elif case == 'missing_frame':
        step['frame_ids'].append('absent')
    elif case == 'outside_frame':
        window['source_frames'][0]['local_ms'] = 129999
    elif case == 'boundary_uncertain':
        step['boundary_uncertain'] = True
    elif case == 'clock_mismatch':
        step['start_us'] += 1000
    elif case == 'missing_receipt':
        window.pop('model_receipt')
    elif case == 'outside_window':
        step['end_ms'] = 160000
    elif case == 'nonfinite':
        step['end_ms'] = float('nan')
    elif case == 'missing_frame_hash':
        window['source_frames'][0].pop('sha256')
    else:
        step['end_ms'] = step['start_ms']
    meaning['windows'] = [window]
    result = partition(segment, meaning, record)
    assert_partition(result)
    assert not selected(result)
    assert any(r['reason'] == 'boundary_uncertain' for r in result['intervals'])
    assert all(r['video_value'] == 'pending' for r in result['intervals'])


@pytest.mark.parametrize('state', ['partial', 'submitted_unknown', 'failed'])
def test_completed_window_is_visible_even_when_later_windows_failed(state):
    segment, record, meaning = fixture()
    meaning.update(status=state, windows=[action_window(130000, 155000)])
    row, = selected(partition(segment, meaning, record))
    assert (row['start_ms'], row['end_ms']) == (130000, 155000)


def test_cv_activity_keyframes_audio_and_ocr_points_do_not_select_candidates():
    segment, record, meaning = fixture()
    segment.update(activity_semantics='experiment_waiting', key_frames=[{'local_ms': 130000}])
    record['transcription'] = {'status': 'completed', 'comments': [{'start_us': ORIGIN + 1000,
        'end_us': ORIGIN + 2000, 'text': '开始'}]}
    context = {'events': [{'start_us': ORIGIN + 2000, 'end_us': ORIGIN + 2000, 'type': 'ocr_observation'}]}
    result = partition(segment, meaning, record, context)
    assert_partition(result)
    assert not selected(result) and result['intervals'][0]['video_value'] == 'pending'


@pytest.mark.parametrize('kind', ['recording', 'meaning', 'window', 'step'])
def test_foreign_source_identity_cannot_select_a_video_interval(kind):
    segment, record, meaning = fixture()
    window = action_window(130000, 155000)
    meaning['windows'] = [window]
    if kind == 'recording':
        record['recording_id'] = 'different-recording'
    elif kind == 'meaning':
        meaning['segment_id'] = 'different-candidate'
    else:
        row = window if kind == 'window' else window['experiment_steps'][0]
        row['source_ref'] = segment['source_ref'] | {'sha256': 'b' * 64}
    result = partition(segment, meaning, record)
    assert_partition(result)
    assert not selected(result)


def test_overlap_partitions_without_duplicate_duration_and_preserves_both_categories():
    segment, record, meaning = fixture()
    meaning['windows'] = [action_window(130000, 155000), action_window(150000, 160000, field='work_activities')]
    result = partition(segment, meaning, record)
    assert_partition(result)
    assert sum(r['end_ms'] - r['start_ms'] for r in selected(result)) == 30000
    overlap = next(r for r in selected(result) if r['start_ms'] == 150000)
    assert overlap['activity_category'] == 'unknown'
    assert overlap['activity_categories'] == ['experiment', 'non_experiment']
    assert len(overlap['evidence']) == 2


def test_source_offset_and_recorder_clock_mapping_preserve_original_coordinates():
    segment, record, meaning = fixture()
    segment.update(start_ms=100000, end_ms=600000)
    points = [[100000, ORIGIN + 100000000], [600000, ORIGIN + 600050000]]
    record['processing'] = {'clock_mapping': {'origin_us': ORIGIN, 'points': points}}
    segment.update(start_us=points[0][1], end_us=points[-1][1])
    window = action_window(130000, 155000)
    from visioncortex.media_time import capture_us
    for step in window['experiment_steps']:
        step.update(start_us=capture_us(record['processing']['clock_mapping'], step['start_ms']),
                    end_us=capture_us(record['processing']['clock_mapping'], step['end_ms']))
    meaning['windows'] = [window]
    result = partition(segment, meaning, record)
    assert_partition(result, 100000, 600000)
    row, = selected(result)
    assert row['start_ms'] == 130000 and row['start_us'] == ORIGIN + 130003000


@pytest.mark.parametrize('malformed', [False, True])
def test_invalid_recorder_mapping_preserves_candidate_timestamps_and_isolates_video(malformed):
    segment, record, meaning = fixture()
    record['processing'] = {'clock_mapping': {'origin_us': ORIGIN + 1000}}
    if malformed:
        record['processing']['clock_mapping']['points'] = [[0, ORIGIN], [0, ORIGIN + 1]]
    meaning['windows'] = [action_window(130000, 155000)]
    result = partition(segment, meaning, record)
    assert_partition(result)
    row, = result['intervals']
    assert row['video_value'] == 'pending' and row['reason'] == 'clock_mapping_mismatch'
    assert (row['start_us'], row['end_us']) == (segment['start_us'], segment['end_us'])


def static_window(start, end):
    window = action_window(start, end)
    window.update(activity_kind='static', experiment_steps=[], work_activities=[])
    window['source_frames'][-1]['local_ms'] -= 1
    window['coverage'] = {'all_decoded_frames_submitted': True,
        'native_pts_to_pixels_verified': True, 'input_frames': len(window['source_frames'])}
    return window


def test_only_fully_covered_continuous_static_can_confirm_no_value_and_keeps_time_gap_pending():
    segment, record, meaning = fixture()
    segment['activity'] = 'inactive'
    meaning['windows'] = [static_window(0, 100000), static_window(110000, 900000)]
    result = partition(segment, meaning, record)
    assert_partition(result)
    assert [(r['start_ms'], r['end_ms'], r['video_value']) for r in result['intervals']] == [
        (0, 100000, 'confirmed_no_value'), (100000, 110000, 'pending'), (110000, 900000, 'confirmed_no_value')]
    meaning['windows'] = [static_window(0, 900000)]
    assert disposition(segment, meaning, record)['no_value_confirmed'] is True
    meaning['windows'][0]['coverage']['all_decoded_frames_submitted'] = False
    assert disposition(segment, meaning, record)['video_value'] == 'pending'


def test_bad_window_cannot_be_hidden_by_another_static_window_and_does_not_hide_selected_sibling():
    segment, record, meaning = fixture()
    segment['activity'] = 'inactive'
    meaning['windows'] = [static_window(0, 900000), {'start_ms': 0, 'end_ms': float('nan')},
                         action_window(130000, 155000)]
    result = partition(segment, meaning, record)
    assert_partition(result)
    row, = selected(result)
    assert (row['start_ms'], row['end_ms']) == (130000, 155000)
    assert all(r['video_value'] == 'pending' for r in result['intervals'] if r != row)


def test_explicit_necessary_context_is_bounded_and_source_matched_with_no_padding():
    segment, record, meaning = fixture()
    context = {'video_context_intervals': [{'start_ms': 129000, 'end_ms': 130000,
        'source_ref': segment['source_ref'], 'basis': 'observed', 'reason': '可见准备过程',
        'evidence': [{'input': 'ContextInput.json', 'model_receipt': 'ContextResult.json'}],
        'activity_category': 'experiment'}]}
    meaning['windows'] = [action_window(130000, 155000)]
    rows = selected(partition(segment, meaning, record, context))
    assert [(r['start_ms'], r['end_ms']) for r in rows] == [(129000, 130000), (130000, 155000)]
    assert rows[0]['context_basis'][0]['reason'] == '可见准备过程'
    context['video_context_intervals'][0]['source_ref'] = {'path': 'MetaVideo/Other.mp4'}
    assert [(r['start_ms'], r['end_ms']) for r in selected(partition(segment, meaning, record, context))] == [(130000, 155000)]


def coherent_work_window(*, second_start=6000):
    first = action_window(1000, 2000, field='work_activities')
    second = action_window(second_start, second_start+1000, field='work_activities')
    for number, frame in enumerate(first['source_frames']):
        frame['frame_id'] = 'first' + str(number)
    for number, frame in enumerate(second['source_frames']):
        frame['frame_id'] = 'second' + str(number)
    for value, prefix in ((first, 'first'), (second, 'second')):
        step = value['work_activities'][0]
        step.update(objects=['键盘'], frame_ids=[prefix+'0', prefix+'1'],
                    before_state='键盘旁可见双手', after_state='手离开后键盘仍在桌面')
    second['work_activities'][0]['before_state'] = '此前画面未见手，键盘保持可见'
    gap = first['source_frames'][0] | {'frame_id': 'gap', 'local_ms': 4000, 'path': 'Gap.jpg'}
    first.update(start_ms=0, end_ms=max(30000, second_start+1000),
        source_frames=[*first['source_frames'], gap, *second['source_frames']],
        work_activities=[first['work_activities'][0], second['work_activities'][0]],
        frame_observations=[{'frame_id': 'gap', 'text': '未见手，键盘仍在同一操作位置且可见'}])
    return first


@pytest.mark.parametrize('experimental', [False, True])
def test_same_object_work_with_actual_gap_context_exports_one_coherent_group_without_combining_actions(experimental):
    segment, record, meaning = fixture()
    window = coherent_work_window()
    if experimental:
        window['experiment_steps'] = window.pop('work_activities')
        for step in window['experiment_steps']:
            step.pop('category')
    meaning['windows'] = [window]
    before = deepcopy(meaning)
    result = partition(segment, meaning, record)
    assert_partition(result)
    group, = selected(result)
    assert (group['start_ms'], group['end_ms'], group['activity_category']) == (
        1000, 7000, 'experiment' if experimental else 'non_experiment')
    assert [(r['start_ms'], r['end_ms']) for r in group['core_action_ranges']] == [(1000, 2000), (6000, 7000)]
    gap, = group['context_subranges']
    assert gap['kind'] == 'continuity_context' and (gap['start_ms'], gap['end_ms']) == (2000, 6000)
    assert [r['frame_id'] for r in gap['source_frames']] == ['gap']
    assert gap['preceding_after_state'] != gap['following_before_state']
    assert gap['frame_observations'][0]['text'] == window['frame_observations'][0]['text']
    assert '不证明间隙持续动作' in gap['reason'] and not gap['physical_action_confirmed']
    assert len(group['evidence']) == 2 and all(e['input'] and e['model_receipt'] for e in group['evidence'])
    assert meaning == before


@pytest.mark.parametrize('case', ['no_gap_frame', 'no_observation', 'different_object', 'different_category',
    'long_gap', 'unknown_state', 'other_activity', 'experiment_point_other_window',
    'object_absent', 'object_removed', 'positive_and_negative_observations'])
def test_proximity_alone_conflicts_or_missing_observations_never_bridge_work(case):
    segment, record, meaning = fixture()
    window = coherent_work_window(second_start=21000 if case == 'long_gap' else 6000)
    if case == 'no_gap_frame':
        window['source_frames'] = [f for f in window['source_frames'] if f['frame_id'] != 'gap']
    elif case == 'no_observation':
        window['frame_observations'] = []
    elif case == 'different_object':
        window['work_activities'][1]['objects'] = ['鼠标']
    elif case == 'different_category':
        window['work_activities'][1]['category'] = 'phone'
    elif case == 'unknown_state':
        window['work_activities'][0]['after_state'] = '未知'
    elif case == 'other_activity':
        other = deepcopy(window['work_activities'][0])
        other.update(start_ms=3500, end_ms=4500, category='phone', objects=['手机'], basis='uncertain')
        window['work_activities'].append(other)
    elif case == 'experiment_point_other_window':
        other = action_window(4000, 5000)
        step = other['experiment_steps'][0]
        step.update(start_ms=4000, end_ms=4000, basis='uncertain')
        meaning['windows'].append(other)
    elif case in {'object_absent', 'object_removed', 'positive_and_negative_observations'}:
        text = '未见键盘，画面为空' if case != 'object_removed' else '键盘已移出画面'
        window['frame_observations'][0]['text'] = text
        if case == 'positive_and_negative_observations':
            extra = deepcopy(next(f for f in window['source_frames'] if f['frame_id'] == 'gap'))
            extra.update(frame_id='gap-positive', local_ms=4500)
            window['source_frames'].append(extra)
            window['frame_observations'].append({'frame_id': extra['frame_id'], 'text': '键盘可见'})
    meaning['windows'].append(window)
    result = partition(segment, meaning, record)
    assert_partition(result)
    assert not any(context['kind'] == 'continuity_context' for group in selected(result) for context in group['context_basis'])
    assert not any(len(group['core_action_ranges']) > 1 for group in selected(result))


def test_same_work_with_continuous_selected_context_is_not_split_at_a_model_window():
    segment, record, meaning = fixture()
    first = coherent_work_window()
    second = deepcopy(first)
    second.update(input='OtherInput.json', model_receipt='OtherResult.json')
    first['work_activities'] = first['work_activities'][:1]
    second['work_activities'] = second['work_activities'][1:]
    meaning['windows'] = [first, second]
    row, = selected(partition(segment, meaning, record))
    assert (row['start_ms'], row['end_ms']) == (1000, 7000)
    assert len(row['core_action_ranges']) == 2
    assert {e['input'] for e in row['evidence']} == {first['input'], second['input']}


def test_work_retains_observed_process_context_beyond_a_two_second_action_ceiling():
    segment, record, meaning = fixture()
    window = action_window(3000, 4000, field='work_activities')
    window.update(start_ms=0, end_ms=10000, frame_observations=[])
    window['work_activities'][0]['objects'] = ['键盘']
    for identifier, stamp in [('far-before', 500), ('before', 2500), ('after', 5000), ('far-after', 7500)]:
        window['source_frames'].append(window['source_frames'][0] | {'frame_id': identifier, 'local_ms': stamp})
        window['frame_observations'].append({'frame_id': identifier, 'text': '键盘与双手位置可见'})
    meaning['windows'] = [window]
    group, = selected(partition(segment, meaning, record))
    assert (group['start_ms'], group['end_ms']) == (500, 7500)
    assert [(r['start_ms'], r['end_ms'], r['side']) for r in group['context_subranges']] == [
        (500, 3000, 'before'), (4000, 7500, 'after')]
    assert [(r['start_ms'], r['end_ms']) for r in group['core_action_ranges']] == [(3000, 4000)]
    assert all(r['search_ceiling_ms'] == 10000 for r in group['context_subranges'])


@pytest.mark.parametrize('field', ['experiment_steps', 'work_activities'])
def test_dense_observations_retain_watchable_context_and_keep_exact_subsecond_action(field):
    segment, record, meaning = fixture()
    window = action_window(3000, 3700, field=field)
    window.update(start_ms=0, end_ms=10000, frame_observations=[])
    for stamp in (1000, 1500, 2000, 2500, 2900, 3800, 4000, 4500, 5000, 5500, 5700):
        identifier = f'context-{stamp}'
        window['source_frames'].append(window['source_frames'][0] | {'frame_id': identifier, 'local_ms': stamp})
        window['frame_observations'].append({'frame_id': identifier, 'text': '工具与双手位置可见'})
    meaning['windows'] = [window]
    original = deepcopy(meaning)
    group, = selected(partition(segment, meaning, record))
    assert (group['start_ms'], group['end_ms']) == (1000, 5700)
    assert [(r['start_ms'], r['end_ms']) for r in group['core_action_ranges']] == [(3000, 3700)]
    assert [len(r['source_frames']) for r in group['context_subranges']] == [5, 6]
    assert all(not r['physical_action_confirmed'] for r in group['context_subranges'])
    assert meaning == original


@pytest.mark.parametrize('field', ['experiment_steps', 'work_activities'])
@pytest.mark.parametrize('barrier', ['missing', 'uncertain', 'other_activity', 'absent_object'])
def test_viewing_context_stops_at_first_problem_without_discarding_nearer_clear_context(field, barrier):
    segment, record, meaning = fixture()
    window = action_window(3000, 3700, field=field)
    window.update(start_ms=0, end_ms=10000, frame_observations=[])
    for stamp in (1500, 2000, 2500):
        identifier = f'context-{stamp}'
        window['source_frames'].append(window['source_frames'][0] | {'frame_id': identifier, 'local_ms': stamp})
        text = '工具可见' if stamp != 2000 else {
            'missing': '', 'uncertain': '工具附近画面模糊',
            'other_activity': '工具在旁，人员使用手机', 'absent_object': '未见工具'}[barrier]
        if text:
            window['frame_observations'].append({'frame_id': identifier, 'text': text})
    meaning['windows'] = [window]
    group, = selected(partition(segment, meaning, record))
    assert (group['start_ms'], group['end_ms']) == (2500, 3700)
    assert [f['local_ms'] for f in group['context_basis'][0]['source_frames']] == [2500]


def test_experiment_context_can_show_same_apparatus_waiting_without_promoting_uncertain_work():
    segment, record, meaning = fixture()
    window = action_window(3000, 3700)
    window.update(start_ms=0, end_ms=10000, frame_observations=[])
    window['experiment_steps'][0]['objects'] = ['磁力搅拌器']
    for stamp in (1500, 2500, 4000, 5000):
        identifier = f'context-{stamp}'
        window['source_frames'].append(window['source_frames'][0] | {'frame_id': identifier, 'local_ms': stamp})
        window['frame_observations'].append({'frame_id': identifier, 'text': '搅拌器亮屏，双手在台面下缘'})
    window['work_activities'] = [
        {'start_ms': 1500, 'end_ms': 2500, 'category': 'waiting', 'objects': ['磁力搅拌器', '烧杯'],
         'basis': 'uncertain', 'frame_ids': ['context-1500', 'context-2500']},
        {'start_ms': 4000, 'end_ms': 5000, 'category': 'waiting', 'objects': ['磁力搅拌器', '烧杯'],
         'basis': 'uncertain', 'frame_ids': ['context-4000', 'context-5000']}]
    meaning['windows'] = [window]
    group, = selected(partition(segment, meaning, record))
    assert (group['start_ms'], group['end_ms']) == (1500, 5000)
    assert group['activity_category'] == 'experiment'
    assert len(group['core_action_ranges']) == 1
    assert group['core_action_ranges'][0]['activity_category'] == 'experiment'
    assert all(r['basis'] == 'observed_context_only' for r in group['context_basis'])


def test_context_never_overlaps_another_windows_selected_waiting_core_into_a_tiny_mixed_clip():
    segment, record, meaning = fixture()
    window = action_window(3000, 3700)
    window.update(start_ms=0, end_ms=6000, frame_observations=[])
    for stamp in (4000, 4500, 5000):
        identifier = f'context-{stamp}'
        window['source_frames'].append(window['source_frames'][0] | {'frame_id': identifier, 'local_ms': stamp})
        window['frame_observations'].append({'frame_id': identifier, 'text': '工具在台上，双手放低'})
    waiting = action_window(4500, 7000, field='work_activities')
    waiting['work_activities'][0]['category'] = 'waiting'
    meaning['windows'] = [window, waiting]
    rows = selected(partition(segment, meaning, record))
    assert [(r['start_ms'], r['end_ms'], r['activity_category']) for r in rows] == [
        (3000, 4500, 'experiment'), (4500, 7000, 'non_experiment')]
    assert all('activity_categories' not in r for r in rows)


def test_readability_context_does_not_expand_a_short_action_to_the_entire_candidate():
    segment, record, meaning = fixture()
    segment.update(end_ms=5000, end_us=ORIGIN+5000000)
    window = action_window(2000, 3000, field='work_activities')
    window.update(start_ms=0, end_ms=5000, frame_observations=[])
    window['work_activities'][0]['objects'] = ['键盘']
    for identifier, stamp in [('before', 0), ('after', 5000)]:
        window['source_frames'].append(window['source_frames'][0] | {'frame_id': identifier, 'local_ms': stamp})
        window['frame_observations'].append({'frame_id': identifier, 'text': '键盘仍在桌面'})
    meaning['windows'] = [window]
    result = partition(segment, meaning, record)
    assert_partition(result, 0, 5000)
    assert not any(group['start_ms'] == 0 and group['end_ms'] == 5000 for group in selected(result))


@pytest.mark.parametrize('text', ['键盘可见，人员正在使用手机', '键盘可见，人员正在擦拭桌面', '键盘可见，人员正在移液'])
def test_explicit_different_activity_in_actual_gap_observation_vetoes_only_context(text):
    segment, record, meaning = fixture()
    window = coherent_work_window()
    window['frame_observations'][0]['text'] = text
    meaning['windows'] = [window]
    result = partition(segment, meaning, record)
    assert [(row['start_ms'], row['end_ms']) for row in selected(result)] == [(1000, 2000), (6000, 7000)]
    assert all(not row['context_basis'] for row in selected(result))
    assert all(row['activity_category'] == 'non_experiment' for row in selected(result))


def test_same_activity_and_negated_other_activity_do_not_become_false_context_conflicts():
    segment, record, meaning = fixture()
    window = coherent_work_window()
    for step in window['work_activities']:
        step['category'] = 'cleaning'
    window['frame_observations'][0]['text'] = '键盘可见，人员正在清洁键盘，未见使用手机，也没有移液'
    meaning['windows'] = [window]
    group, = selected(partition(segment, meaning, record))
    assert (group['start_ms'], group['end_ms']) == (1000, 7000)
    assert group['context_basis'][0]['kind'] == 'continuity_context'
    assert not group['context_basis'][0]['physical_action_confirmed']


def static_gap_work_window(*, second_start=6000):
    window = coherent_work_window(second_start=second_start)
    window['frame_observations'][0]['text'] = '场景静止无人'
    for identifier, stamp, text in [('gap-start', 2500, '手离开，场景恢复静止'),
                                    ('gap-end', 5500, '场景静止无人')]:
        window['source_frames'].append(window['source_frames'][0] | {'frame_id': identifier, 'local_ms': stamp})
        window['frame_observations'].append({'frame_id': identifier, 'text': text})
    return window


def test_all_actual_gap_samples_can_supply_narrow_static_viewing_context_without_object_or_motion_claims():
    segment, record, meaning = fixture()
    window = static_gap_work_window()
    meaning['windows'] = [window]
    original = deepcopy(meaning)
    group, = selected(partition(segment, meaning, record))
    assert (group['start_ms'], group['end_ms']) == (1000, 7000)
    assert [(r['start_ms'], r['end_ms']) for r in group['core_action_ranges']] == [(1000, 2000), (6000, 7000)]
    basis, = group['context_basis']
    assert basis['kind'] == 'anchored_static_scene_context' and basis['sampled_static_context_only']
    assert basis['object_presence_during_gap'] == basis['continuous_motion'] == 'NOT_PROVEN'
    assert basis['actual_input_gap_frame_count'] == len(basis['source_frames']) == len(basis['frame_observations']) == 3
    assert len(basis['endpoint_action_refs']) == 2
    assert all(ref['input'] and ref['model_receipt'] for ref in basis['endpoint_action_refs'])
    assert basis['start_ms'] == 2000 and basis['end_ms'] == 6000
    assert not basis['physical_action_confirmed'] and meaning == original


@pytest.mark.parametrize('bad', ['missing_observation', 'unknown', 'obscured', 'other_activity',
    'object_missing', 'ambiguous', 'not_static', 'moving', 'positive_and_unknown'])
def test_static_viewing_bridge_requires_every_actual_gap_sample_to_be_clear_and_explicit(bad):
    segment, record, meaning = fixture()
    window = static_gap_work_window()
    if bad == 'missing_observation':
        window['frame_observations'] = [r for r in window['frame_observations'] if r['frame_id'] != 'gap']
    else:
        window['frame_observations'][0]['text'] = {
            'unknown': '场景静止，具体情况未知', 'obscured': '场景无人，画面被遮挡',
            'other_activity': '场景静止，但人员正在使用手机', 'object_missing': '场景静止，未见键盘',
            'ambiguous': '同前', 'not_static': '画面并非静止', 'moving': '场景无人，物体正在移动',
            'positive_and_unknown': '键盘可见，但画面部分情况未知'}[bad]
    meaning['windows'] = [window]
    result = partition(segment, meaning, record)
    assert_partition(result)
    assert not any(context['kind'] == 'anchored_static_scene_context' for group in selected(result) for context in group['context_basis'])
    assert not any(len(group['core_action_ranges']) > 1 for group in selected(result))


def test_static_gap_rule_never_expands_a_single_one_second_work_or_connects_over_fifteen_seconds():
    segment, record, meaning = fixture()
    window = static_gap_work_window()
    window['work_activities'] = window['work_activities'][:1]
    meaning['windows'] = [window]
    group, = selected(partition(segment, meaning, record))
    assert (group['start_ms'], group['end_ms']) == (1000, 2000)
    assert not group['context_basis']
    meaning['windows'] = [static_gap_work_window(second_start=21000)]
    groups = selected(partition(segment, meaning, record))
    assert [(g['start_ms'], g['end_ms']) for g in groups] == [(1000, 2000), (21000, 22000)]
    assert all(not group['context_basis'] for group in groups)
