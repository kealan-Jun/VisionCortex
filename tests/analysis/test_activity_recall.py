"""Video recall is distinct from physical-action confirmation and no-value."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from visioncortex.analysis.activity_recall import observations, selected_runs
from visioncortex.analysis.activity_recall_runtime import review
from visioncortex.schemas import SourceFrameIdentity, ViewRole
from visioncortex.source_frames import SampledFrame


def box(name, x, y=.3, confidence=.8):
    return {'class_name': name, 'confidence': confidence, 'xyxy_norm': [x, y, x+.04, y+.06]}


def rows(moving=True, pan=False):
    result = []
    for n in range(10):
        shift = n*.02 if pan else 0
        actors, objects = observations([box('gloved_hand', .4+shift+(n*.005 if moving else 0)),
            box('beaker', .44+shift), box('container', .3+shift)])
        result.append({'local_ms': n*1000, 'source_verified': True, 'actors': actors,
                       'objects': objects, 'source_frame': {'source_pts': n*1000}})
    return result


def test_repeated_local_hand_motion_recalls_full_observed_process_without_formal_actions():
    value = rows()
    before = deepcopy(value)
    intervals, decisions = selected_runs(value, period_ms=1000)
    assert intervals == [(0, 9000)] and decisions[0]['sample_count'] == 10
    assert decisions[0]['physical_action_confirmed'] is False
    assert decisions[0]['event_acceptance_mutated'] is False and value == before


@pytest.mark.parametrize('kind', ['static', 'pan', 'outlier', 'no_lab', 'no_actor', 'bad_source'])
def test_static_gloves_camera_pan_outliers_and_unrelated_scenes_cannot_seed_video(kind):
    value = rows(moving=False, pan=kind == 'pan')
    if kind == 'outlier':
        value[4]['actors'][0]['xyxy_norm'] = [.45, .3, .49, .36]
    for r in value:
        if kind == 'no_lab':
            r['objects'] = []
        elif kind == 'no_actor':
            r['actors'] = []
        elif kind == 'bad_source':
            r['source_verified'] = False
    assert selected_runs(value, period_ms=1000)[0] == []


@pytest.mark.parametrize('damage', ['actor_gap', 'unreadable', 'sampling_hole', 'anchors_lost'])
def test_unknown_observations_split_the_watchable_process(damage):
    value = rows()
    if damage == 'actor_gap':
        value[4]['actors'] = []
    elif damage == 'unreadable':
        value[4]['source_verified'] = False
    elif damage == 'anchors_lost':
        value[4]['objects'] = []
    else:
        for r in value[4:]:
            r['local_ms'] += 2000
    intervals, _ = selected_runs(value, period_ms=1000)
    assert len(intervals) == 2 and intervals[0][1] < intervals[1][0]


def test_large_actor_false_positive_and_remote_instruments_do_not_supply_lab_context():
    actors, _ = observations([box('gloved_hand', .1), box('beaker', .8), box('container', .9)])
    assert actors == []
    bad = box('gloved_hand', 0) | {'xyxy_norm': [0, 0, .9, .9]}
    assert observations([bad, box('beaker', .44), box('container', .3)])[0] == []


def test_review_uses_native_pts_caches_success_and_never_repeats_a_paid_call(tmp_path):
    calls = []
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    def frames(*args, **kwargs):
        for n in range(10):
            native = SourceFrameIdentity(source_path=tmp_path/'Source.mp4', source_size_bytes=10,
                source_mtime_ns=1, status='resolved', packet_position=n, source_pts=n*1000+100,
                time_base='1/1000', decoded_pixels_sha256=sha256(image.tobytes()).hexdigest())
            yield SampledFrame(n, n*1000, image, native)
    def infer(*args):
        n = len(calls)
        calls.append(n)
        return [box('gloved_hand', .4+n*.005), box('beaker', .44), box('container', .3)], {'status': 'executed', 'model': 'fake-local-boundary'}
    config = {'device_day': {'local_activity_recall': {'enabled': True}},
              'models': {'open_vocabulary_key_frame': {'enabled': True}}}
    kwargs = {'source_ref': {'sha256': 'a'*64, 'size_bytes': 10}, 'cache_directory': tmp_path/'Cache',
              'infer': infer, 'frames': frames, 'source_origin_ms': 100}
    view = SimpleNamespace(view_id='camera', role=ViewRole.THIRD_PERSON, video=tmp_path/'Source.mp4')
    intervals, report = review(view, None, [], 0, 10000, config, **kwargs)
    assert intervals == [(0, 9000)] and len(calls) == 10 and report['paid_requests'] == 0
    assert review(view, None, [], 0, 10000, config, **kwargs)[0] == intervals and len(calls) == 10
    saved = next((tmp_path/'Cache').glob('*.json'))
    saved.write_text(saved.read_text().replace('fake-local-boundary', 'changed'))
    with pytest.raises(ValueError, match='cache changed'):
        review(view, None, [], 0, 10000, config, **kwargs)


@pytest.mark.parametrize('gap', [False, True])
def test_only_complete_recall_between_two_fine_processes_can_close_sample_edges(tmp_path, gap):
    calls = []
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    def frames(*args, **kwargs):
        for n in range(10):
            native = SourceFrameIdentity(source_path=tmp_path/'Source.mp4', source_size_bytes=10,
                source_mtime_ns=1, status='resolved', packet_position=n, source_pts=n*1000+1250,
                time_base='1/1000', decoded_pixels_sha256=sha256(image.tobytes()).hexdigest())
            yield SampledFrame(n, n*1000+1000, image, native)
    def infer(*args):
        n = len(calls)
        calls.append(n)
        boxes = [box('beaker', .44), box('container', .3)]
        if not (gap and n == 4):
            boxes.append(box('gloved_hand', .4+n*.005))
        return boxes, {'status': 'executed'}
    config = {'device_day': {'local_activity_recall': {'enabled': True}},
              'models': {'open_vocabulary_key_frame': {'enabled': True}}}
    view = SimpleNamespace(view_id='camera', role=ViewRole.THIRD_PERSON, video=tmp_path/'Source.mp4')
    selected, report = review(view, None, [(0, 1000), (11000, 12000)], 0, 12000, config,
        source_ref={'sha256': 'a'*64, 'size_bytes': 10}, cache_directory=tmp_path/'Cache',
        infer=infer, frames=frames, source_origin_ms=0)
    if gap:
        assert len(selected) == 2
        assert all('edge_context' not in d for d in report['windows'][0]['decisions'])
    else:
        assert selected == [(1000, 11000)]
        decision, = report['windows'][0]['decisions']
        assert decision['observed_start_ms'] == 1250 and decision['observed_end_ms'] == 10250
        assert decision['edge_context']['ranges'] == [[1000, 1250], [10250, 11000]]
        assert decision['physical_action_confirmed'] is False


def test_two_prompt_names_for_one_object_do_not_supply_two_laboratory_anchors():
    actors, objects = observations([box('gloved_hand', .4), box('beaker', .44), box('container', .44)])
    assert len(objects) == 1 and actors == []
