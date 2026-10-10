import copy

import pytest

from visioncortex.device_day_steps import readable_meaning, reviewed_window, validate


def result():
    return {'summary': '手将移液器移向烧杯', 'activity_observed': 'active',
        'experiment_steps': [{'start_ms': 100, 'end_ms': 200, 'action': '移动移液器', 'objects': ['移液器', '烧杯'],
            'before_state': '位于管架上方', 'after_state': '位于烧杯口上方', 'visible_evidence': '两帧器具位置不同',
            'outcome': 'incomplete_or_uncertain', 'description': '手持移液器从管架上方移向烧杯',
            'frame_ids': ['a', 'b'], 'comment_ids': [], 'basis': 'observed'}],
        'frame_observations': [{'frame_id': 'a', 'text': '管架上方'}, {'frame_id': 'b', 'text': '烧杯上方'}],
        'uncertainties': ['未见液体变化']}


def check(value):
    return validate(value, [{'frame_id': 'a', 'local_ms': 100}, {'frame_id': 'b', 'local_ms': 200}],
                    0, 300, 1000000, {'origin_us': 1000000})


def test_operation_output_keeps_objects_states_absolute_time_and_visual_uncertainty():
    value = check(result())
    assert value['experiment_steps'][0]['start_us'] == 1100000
    assert value['experiment_steps'][0]['outcome'] == 'incomplete_or_uncertain'
    assert value['steps'][0]['description'] == result()['experiment_steps'][0]['description']
    assert value['accuracy'] == 'NOT_PROVEN'


@pytest.mark.parametrize('kind', ['inactive', 'speech', 'frame', 'state'])
def test_step_contract_rejects_fabricated_evidence_or_missing_structure(kind):
    value = result()
    if kind == 'inactive':
        value['activity_observed'] = 'inactive'
    elif kind == 'speech':
        value['experiment_steps'][0]['comment_ids'] = ['utterance']
    elif kind == 'frame':
        value['experiment_steps'][0]['frame_ids'] = ['invented']
    else:
        del value['experiment_steps'][0]['before_state']
    with pytest.raises(ValueError):
        check(value)


def test_static_legacy_captions_are_not_published_as_experiment_steps():
    original = {'windows': [{'activity_observed': 'inactive', 'steps': [{'description': '灯管、墙面'}]}]}
    saved = copy.deepcopy(original)
    value, steps, status = readable_meaning(original)
    assert original == saved and steps == []
    assert 'steps' not in value['windows'][0]
    assert value['windows'][0]['scene_and_behavior_observations'][0]['description'] == '灯管、墙面'
    assert status == 'no_operation_observed_in_supplied_frames'
    original['windows'][0]['activity_observed'] = 'active'
    assert readable_meaning(original)[1:] == (None, 'legacy_observations_only')


def test_explicit_review_requires_unchanged_images_and_model_receipt(tmp_path):
    from visioncortex.device_day_content import time_folder
    from visioncortex.device_day_contract import artifact, atomic_json
    source = tmp_path/'MetaVideo/Original.mp4'
    source.parent.mkdir()
    source.write_bytes(b'original source identity')
    segment = {'segment_id': 's', 'start_ms': 0, 'end_ms': 300, 'start_us': 1000000,
               'end_us': 1300000, 'source_ref': artifact(tmp_path, source)}
    folder = tmp_path/'MultimodalUnderstanding'/time_folder(1000000, 1300000)/'Analysis'
    folder.mkdir(parents=True)
    frames = []
    for identifier, stamp in [('a', 100), ('b', 200)]:
        path = folder/(identifier+'.jpg')
        path.write_bytes(identifier.encode())
        frames.append({'frame_id': identifier, 'local_ms': stamp, **artifact(tmp_path, path)})
    atomic_json(folder/'Input.json', {'metadata': {'segment_id': 's', 'source_ref': segment['source_ref'],
        'start_ms': 0, 'end_ms': 300, 'frames': frames, 'comments': [], 'protocol': None,
        'coverage': {'mode': 'historical_sampled_frames', 'all_decoded_frames_submitted': False}}})
    atomic_json(folder/'Result.json', {'model_result': {'status': 'completed', **result()}})
    atomic_json(folder/'StepReview.json', {'segment_id': 's', 'source_video': segment['source_ref'],
        'input': artifact(tmp_path, folder/'Input.json'), 'model_receipt': artifact(tmp_path, folder/'Result.json')})
    value = reviewed_window(tmp_path, segment, {'origin_us': 1000000})
    assert value['explicit_step_review'] and len(value['experiment_steps']) == 1
    assert not value['coverage']['all_decoded_frames_submitted']
    (folder/'a.jpg').write_bytes(b'different image')
    with pytest.raises(ValueError):
        reviewed_window(tmp_path, segment, {'origin_us': 1000000})
