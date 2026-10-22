"""A source viewing span can contain distinct work and unknown scene samples."""
from copy import deepcopy

import pytest

from test_retained_interval_partition import ORIGIN, action_window, assert_partition, fixture, selected
from visioncortex.media.retained_intervals import partition


def work():
    segment, record, meaning = fixture()
    segment['activity'] = 'inactive'
    window = action_window(0, 16000, field='work_activities', frame_times=list(range(0, 16000, 1000)))
    template = window['work_activities'][0]
    window['work_activities'] = []
    for start, end, category, objects, after in (
        (1000, 2000, 'walking', ['走廊'], '到达门口'),
        (4000, 5000, 'other', ['门把手'], '门后状态未知'),
        (8000, 10000, 'walking', ['办公桌'], '到达电脑桌旁'),
    ):
        window['work_activities'].append(deepcopy(template) | {
            'start_ms': start, 'end_ms': end,
            'start_us': ORIGIN+start*1000, 'end_us': ORIGIN+end*1000,
            'category': category, 'objects': objects, 'before_state': '处于工作场景', 'after_state': after,
            'frame_ids': ['f'+str(start//1000), 'f'+str(end//1000)]})
    window['frame_observations'] = [{'frame_id': frame['frame_id'],
        'text': '画面模糊，细节无法确认'} for frame in window['source_frames']]
    window['coverage'] = {'native_pts_to_pixels_verified': True, 'baseline_sample_fps': 1}
    meaning['windows'] = [window]
    return segment, record, meaning


def test_distinct_work_steps_share_one_viewing_clip_without_claiming_gap_actions():
    segment, record, meaning = work()
    before = deepcopy((segment, record, meaning))
    result = partition(segment, meaning, record)
    assert_partition(result)
    clip, = selected(result)
    assert (clip['start_ms'], clip['end_ms']) == (1000, 10000)
    assert len(clip['evidence']) == len(clip['core_action_ranges']) == 3
    assert clip['evidence'][1]['after_state'] == '门后状态未知'
    assert [(v['start_ms'], v['end_ms']) for v in clip['context_subranges']] == [(2000, 4000), (5000, 8000)]
    for context in clip['context_subranges']:
        assert context['kind'] == 'device_work_viewing_context'
        assert context['actor_identity'] == context['same_object_identity'] == 'NOT_PROVEN'
        assert context['continuous_motion'] == 'NOT_PROVEN' and not context['physical_action_confirmed']
        assert all(row['text'] == '画面模糊，细节无法确认' for row in context['frame_observations'])
    assert clip['physical_action_confirmed'] is False and segment['key_frames'] == []
    assert (segment, record, meaning) == before


@pytest.mark.parametrize('defect', ['native_pixels', 'missing_observation', 'missing_frame_hash', 'sparse_gap', 'experiment_gap'])
def test_unverified_or_experimental_gap_stays_outside_viewing_connections(defect):
    segment, record, meaning = work()
    window = meaning['windows'][0]
    if defect == 'native_pixels':
        window['coverage']['native_pts_to_pixels_verified'] = False
    elif defect == 'missing_observation':
        window['frame_observations'] = [r for r in window['frame_observations'] if r['frame_id'] != 'f3']
    elif defect == 'missing_frame_hash':
        window['source_frames'][3].pop('sha256')
    elif defect == 'sparse_gap':
        window['source_frames'] = [f for f in window['source_frames'] if f['frame_id'] != 'f3']
    else:
        step = deepcopy(window['work_activities'][0])
        step.update(start_ms=2000, end_ms=4000, start_us=ORIGIN+2000000, end_us=ORIGIN+4000000,
                    basis='inferred', frame_ids=['f2', 'f4'])
        window['experiment_steps'] = [step]
    rows = selected(partition(segment, meaning, record))
    assert not any(row['start_ms'] < 3000 < row['end_ms'] for row in rows)


def test_real_capture_outage_is_not_hidden_by_a_short_encoded_video_gap():
    segment, record, meaning = work()
    record['processing'] = {'clock_mapping': {'points': [
        [0, ORIGIN], [2000, ORIGIN+2000000], [4000, ORIGIN+30000000], [900000, segment['end_us']]]}}
    for step in meaning['windows'][0]['work_activities']:
        step.pop('start_us')
        step.pop('end_us')
    result = partition(segment, meaning, record)
    assert not any(row['start_ms'] < 3000 < row['end_ms'] for row in selected(result))


def test_inferred_only_work_never_promotes_an_entire_window_to_video():
    segment, record, meaning = work()
    for step in meaning['windows'][0]['work_activities']:
        step['basis'] = 'inferred'
    assert selected(partition(segment, meaning, record)) == []
