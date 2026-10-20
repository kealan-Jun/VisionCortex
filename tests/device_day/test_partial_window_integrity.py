"""Readable partial observations retain byte gates without relabelling execution."""
from copy import deepcopy

import pytest

from test_device_day import FakeModels, capture, item_and_layout, device_config as device_config_fixture
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_content import content_revision, with_partial_understandings
from visioncortex.device_day_contract import artifact, atomic_json, digest, read_json

device_config = device_config_fixture


def windows(config, *, receipt_artifacts=False):
    capture(config)
    record, layout = item_and_layout(config)
    runner = DeviceDayRunner(config, FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    assert runner.process(record, stage='vision')['status'] == 'completed'
    index = read_json(layout.index)
    segment = index['segments'][0]
    key = 'a'*64
    root = layout.understanding/'.Evidence/Analysis'/segment['segment_id']/key
    template = segment['scene_frames'][0]
    created, references = [], []
    for number, left in enumerate((0, 15000)):
        folder = root/f'Window{number}'
        frame_path = folder/'SceneFrames/Frame.jpg'
        frame_path.parent.mkdir(parents=True)
        frame_path.write_bytes((layout.root/template['path']).read_bytes())
        frame = template | artifact(layout.root, frame_path) | {'frame_id': f'frame-{number}', 'local_ms': left+1000}
        metadata = {'segment_id': segment['segment_id'], 'source_ref': segment['source_ref'],
            'start_ms': left, 'end_ms': left+15000, 'frames': [frame], 'comments': []}
        prompt = 'Deterministic fixture; no real model call'
        input_key = digest([metadata, prompt])
        request, result = folder/'Input.json', folder/'Result.json'
        raw = {'status': 'completed', 'summary': f'Validated fixture window {number}', 'activity_observed': 'inactive',
               'steps': [], 'frame_observations': [{'frame_id': frame['frame_id'], 'text': 'fixture observation'}],
               'uncertainties': []}
        atomic_json(request, {'metadata': metadata, 'prompt': prompt, 'input_key': input_key})
        atomic_json(result, {'input_key': input_key, 'model_result': raw})
        references.extend(artifact(layout.root, path) for path in (request, result))
        created.append((frame_path, request, result))
    atomic_json(runner._receipt(layout, record, 'understanding'), {'key': key, 'status': 'submitted_unknown',
        'artifacts': references if receipt_artifacts else []})
    return index, layout, created


@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_bad_frame_isolates_its_window_and_keeps_other_completed_window(device_config, damage):
    index, _, created = windows(device_config)
    previous_revision = content_revision(device_config, index)
    frame = created[0][0]
    if damage == 'missing':
        frame.unlink()
    else:
        frame.write_bytes(b'x'*frame.stat().st_size)
    assert content_revision(device_config, index) != previous_revision
    original = deepcopy(index)
    value = with_partial_understandings(device_config, index)
    assert index == original
    meaning = value['understandings'][0]
    assert meaning['status'] == 'partial' and meaning['stage_status'] == 'submitted_unknown'
    assert [window['summary'] for window in meaning['windows']] == ['Validated fixture window 1']
    assert value['partial_window_integrity_errors'][0]['reason'] == 'partial_window_frame_artifact_integrity_failed'
    window = meaning['windows'][0]
    assert window['frame_artifact_integrity'] == 'verified'
    assert window['request_identity_integrity'] == 'verified_exact_metadata_and_prompt'
    assert set(window['receipt_integrity'].values()) == {'unavailable_stage_artifact_reference'}


def test_saved_receipt_artifacts_and_recomputed_request_identity_guard_each_window(device_config):
    index, _, created = windows(device_config, receipt_artifacts=True)
    value = with_partial_understandings(device_config, index)
    assert len(value['understandings'][0]['windows']) == 2
    assert all(set(window['receipt_integrity'].values()) == {'verified_against_stage_artifact'}
               for window in value['understandings'][0]['windows'])
    result_path = created[0][2]
    result = read_json(result_path)
    result['model_result']['summary'] = 'Changed after saved artifact receipt'
    atomic_json(result_path, result)
    value = with_partial_understandings(device_config, index)
    assert [window['summary'] for window in value['understandings'][0]['windows']] == ['Validated fixture window 1']
    assert value['partial_window_integrity_errors'][0]['reason'] == 'partial_window_result_receipt_identity_changed'


def test_matching_but_forged_input_keys_cannot_replace_actual_metadata_identity(device_config):
    index, _, created = windows(device_config)
    for path in created[0][1:]:
        value = read_json(path)
        value['input_key'] = 'f'*64
        atomic_json(path, value)
    value = with_partial_understandings(device_config, index)
    assert [window['summary'] for window in value['understandings'][0]['windows']] == ['Validated fixture window 1']
    assert value['partial_window_integrity_errors'][0]['reason'] == 'partial_window_request_identity_changed'
