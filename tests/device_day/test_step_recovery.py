"""Conservative receipt recovery and bounded correction, with no external API."""
from copy import deepcopy

import pytest

from visioncortex import scene_requests
from visioncortex.device_day_call_templates import CLIP_UNDERSTANDING, CLIP_VALIDATION_REPAIR
from visioncortex.device_day_contract import artifact, read_json
from visioncortex.device_day_steps import InvalidStepUnderstanding, repair_window, validate
from visioncortex.multimodal_usage import UsageLedger
from visioncortex.scene_transport import compact


def observation():
    return {'status': 'completed', 'attempts': 1, 'summary': '可见键盘操作，实际用途未知',
        'activity_observed': 'inactive', 'activity_kind': 'non_experiment', 'observability': 'clear',
        'experiment_steps': [], 'work_activities': [{'start_ms': 100, 'end_ms': 200,
            'action': '操作键盘', 'objects': ['键盘'], 'before_state': '手在键盘上方',
            'after_state': '手指位于键盘上', 'visible_evidence': '手指位置变化', 'outcome': 'visible_change',
            'description': '可见手指移动，输入内容未知', 'frame_ids': ['a', 'b'],
            'comment_ids': [], 'basis': 'observed', 'category': 'computer'}],
        'frame_observations': [{'frame_id': 'a', 'text': '键盘上方'}, {'frame_id': 'b', 'text': '手指位置变化'}],
        'uncertainties': [], 'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15,
                                     'cached_input_tokens': 0}}


def check(raw, frames=None):
    return validate(raw, frames or [{'frame_id': 'a', 'local_ms': 100}, {'frame_id': 'b', 'local_ms': 200}],
        0, 500, 1000000, {'origin_us': 1000000}, require_all_day=True)


@pytest.mark.parametrize('field', ['basis', 'outcome'])
def test_missing_work_assessment_is_unknown_and_cannot_confirm_completion(field):
    raw = observation()
    del raw['work_activities'][0][field]
    before = deepcopy(raw)
    parsed = check(raw)
    activity = parsed['work_activities'][0]
    assert activity['outcome'] == 'incomplete_or_uncertain'
    if field == 'basis':
        assert activity['basis'] == 'uncertain'
    assert parsed['step_normalizations'][0]['missing_fields'] == [field]
    assert parsed['uncertainties'] and raw == before


@pytest.mark.parametrize('field', ['basis', 'outcome', 'objects', 'before_state'])
def test_experiment_fields_remain_required(field):
    raw = observation()
    step = raw['work_activities'].pop()
    step.pop('category')
    del step[field]
    raw.update(activity_observed='active', activity_kind='experiment', experiment_steps=[step])
    with pytest.raises(InvalidStepUnderstanding):
        check(raw)


@pytest.mark.parametrize('category', ['waiting', 'walking'])
def test_body_activity_can_have_no_manipulated_object(category):
    raw = observation()
    raw['work_activities'][0].update(objects=[], category=category, action='等待或走动')
    assert check(raw)['work_activities'][0]['objects'] == []
    raw['work_activities'][0]['category'] = 'computer'
    with pytest.raises(InvalidStepUnderstanding, match='visible object'):
        check(raw)


def test_extra_context_frame_is_discarded_without_expanding_action_time_or_confirming_it():
    raw = observation()
    raw['work_activities'][0]['frame_ids'].append('c')
    raw['frame_observations'].append({'frame_id': 'c', 'text': '动作结束后的场景'})
    before = deepcopy(raw)
    parsed = check(raw, [{'frame_id': 'a', 'local_ms': 100}, {'frame_id': 'b', 'local_ms': 200},
                         {'frame_id': 'c', 'local_ms': 300}])
    step = parsed['work_activities'][0]
    assert step['frame_ids'] == ['a', 'b'] and (step['start_ms'], step['end_ms']) == (100, 200)
    assert step['basis'] == 'uncertain' and step['outcome'] == 'incomplete_or_uncertain'
    assert parsed['step_normalizations'][0]['discarded_out_of_interval_frame_ids'] == ['c']
    assert parsed['frame_observations'][-1]['frame_id'] == 'c' and raw == before


@pytest.mark.parametrize('change', ['unknown', 'wholly_outside'])
def test_missing_evidence_remains_rejected(change):
    raw = observation()
    if change == 'unknown':
        raw['work_activities'][0]['frame_ids'].append('invented')
    else:
        raw['work_activities'][0].update(start_ms=300, end_ms=400)
    with pytest.raises(InvalidStepUnderstanding):
        check(raw)


@pytest.fixture
def correction(default_config, tmp_path):
    config = deepcopy(default_config)
    config['storage'].update(local_cache_root=str(tmp_path/'cache'), local_runtime_root=str(tmp_path/'runtime'))
    frames, paths = [], []
    for identifier, stamp in [('a', 100), ('b', 200)]:
        path = tmp_path/(identifier+'.jpg')
        path.write_bytes(('synthetic pixels '+identifier).encode())
        frames.append(artifact(tmp_path, path) | {'frame_id': identifier, 'local_ms': stamp,
            'source_path': 'MetaVideo/Original.mp4', 'frame_kind': 'scene_sample'})
        paths.append((identifier, path))
    metadata = {'schema_version': 'visioncortex-device-day/1', 'segment_id': 's', 'start_ms': 0,
        'end_ms': 500, 'source_ref': {'path': 'MetaVideo/Original.mp4', 'sha256': 'a'*64},
        'activity': 'inactive', 'comments': [], 'protocol': None, 'frames': frames, 'content_source': 'video_only'}
    folder = tmp_path/'MultimodalUnderstanding/Time/Analysis/s'/('a'*64)/'Window'
    class Analyzer:
        def __init__(self):
            self.calls = []
            self.repair_is_invalid = False
        def _call(self, prompt, metadata, images, **kwargs):
            self.calls.append(prompt)
            raw = observation()
            if prompt == CLIP_UNDERSTANDING.prompt or self.repair_is_invalid:
                raw['work_activities'][0].update(start_ms=300, end_ms=400)
            return raw
    analyzer = Analyzer()
    return config, tmp_path, metadata, paths, folder, analyzer


def run_original(correction):
    config, _, metadata, paths, folder, analyzer = correction
    return scene_requests.window(config, analyzer, CLIP_UNDERSTANDING.prompt, metadata, paths,
        folder/'Input.json', folder/'Result.json', lambda raw: check(raw, metadata['frames']))


def repair(correction):
    config, root, metadata, paths, folder, analyzer = correction
    return repair_window(config, analyzer, root, metadata, paths, folder/'Input.json', folder/'Result.json',
                         1000000, {'origin_us': 1000000})


def test_correction_has_a_separate_template_receipt_and_is_paid_only_once(correction):
    config, _, _, _, folder, analyzer = correction
    with pytest.raises(InvalidStepUnderstanding):
        run_original(correction)
    original = (folder/'Result.json').read_bytes()
    _, parsed, reused, inp, result = repair(correction)
    assert not reused and parsed['work_activities'][0]['frame_ids'] == ['a', 'b']
    assert (folder/'Result.json').read_bytes() == original
    assert read_json(inp)['call_template'] == CLIP_VALIDATION_REPAIR.receipt()
    assert read_json(result)['model_result']['status'] == 'completed'
    with pytest.raises(InvalidStepUnderstanding):
        run_original(correction)
    assert repair(correction)[2] is True
    assert analyzer.calls == [CLIP_UNDERSTANDING.prompt, CLIP_VALIDATION_REPAIR.prompt]
    from datetime import datetime
    from zoneinfo import ZoneInfo
    day = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    assert UsageLedger(config).summary(day)['calls'] == 2


def test_invalid_correction_does_not_loop_or_resubmit(correction):
    analyzer = correction[-1]
    analyzer.repair_is_invalid = True
    with pytest.raises(InvalidStepUnderstanding):
        run_original(correction)
    for _ in range(2):
        with pytest.raises(InvalidStepUnderstanding):
            repair(correction)
    assert len(analyzer.calls) == 2


def test_changed_evidence_cannot_enter_paid_correction(correction):
    with pytest.raises(InvalidStepUnderstanding):
        run_original(correction)
    correction[3][0][1].write_bytes(b'changed frame')
    with pytest.raises(ValueError, match='intact original'):
        repair(correction)
    assert len(correction[-1].calls) == 1


def test_correction_candidate_uses_same_wire_frame_aliases_as_images(correction):
    _, _, metadata, paths, _, _ = correction
    metadata = deepcopy(metadata) | {'validation_repair': {'previous_observations': observation()}}
    original = deepcopy(metadata)
    wire, supplied, _ = compact(metadata, paths)
    previous = wire['validation_repair']['previous_observations']
    assert previous['work_activities'][0]['frame_ids'] == [label for label, _ in supplied]
    assert previous['frame_observations'][0]['frame_id'] == 'f1'
    assert metadata == original
