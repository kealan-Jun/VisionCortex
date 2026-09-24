import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from visioncortex import temporal_mask_handoff as handoff


@pytest.fixture
def payload(tmp_path):
    folder = tmp_path / 'frames/0000'
    folder.mkdir(parents=True)
    frames = []
    for i in range(2):
        path = folder / f'{i:05d}.jpg'
        Image.new('RGB', (12, 8), (i * 100, 70, 50)).save(path, quality=95)
        with Image.open(path) as image:
            digest = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()
        frames.append({'frame_index': i * 6, 'timestamp_ms': i * 200, 'clip_pts': i * 3072,
            'time_base': '1/15360', 'rgb_sha256': 'a' * 64,
            'input_file': f'frames/0000/{i:05d}.jpg', 'input_sha256': handoff.temporal._sha256(path),
            'input_rgb_sha256': digest})
    return {'schema_version': handoff.REQUEST_SCHEMA, 'dimensions': [12, 8],
        'source': {'camera_role': 'first_person', 'camera_id': 'camera-a',
                   'split': 'train', 'source_sha256': 'b' * 64},
        'parent_result_sha256': 'c' * 64,
        'input_transform': {'encoding': 'JPEG', 'quality': 95, 'resize': False},
        'windows': [{'id': 0, 'prompts': [{'id': 'object1', 'label': 'beaker', 'box': [1, 1, 10, 7]}],
                     'frames': frames}]}


@pytest.mark.parametrize('kind', ['hash', 'pixels', 'order', 'extra', 'role', 'holdout', 'box'])
def test_reject_bad_input_before_gpu(payload, tmp_path, kind):
    if kind == 'hash':
        payload['windows'][0]['frames'][0]['input_sha256'] = 'f' * 64
    elif kind == 'pixels':
        payload['windows'][0]['frames'][0]['input_rgb_sha256'] = 'f' * 64
    elif kind == 'order':
        payload['windows'][0]['frames'][1]['frame_index'] = 0
    elif kind == 'extra':
        (tmp_path / 'frames/0000/99999.jpg').write_bytes(b'extra')
    elif kind == 'role':
        payload['source']['camera_role'] = 'unknown'
    elif kind == 'holdout':
        payload['source']['split'] = 'test'
    elif kind == 'box':
        payload['windows'][0]['prompts'][0]['box'] = [1, 1, 13, 7]
    with pytest.raises(ValueError):
        handoff.validate_request(payload, tmp_path)


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('observation', [False, True])
def test_real_raster_receipt_and_missing_frame_failure(payload, tmp_path, monkeypatch, skip, observation):
    import torch
    if observation:
        payload['schema_version'] = handoff.OBSERVATION_REQUEST_SCHEMA
        payload['data_use'] = {'purpose': 'production_observation'}
        payload['source'].update(schema_version='labprism-local-observation/1', split=None,
                                training_use_authorized=False, independent_ground_truth=False)
    class Predictor:
        reset = False

        def init_state(self, *args, **kwargs):
            return {}

        def add_new_points_or_box(self, *args, **kwargs):
            assert kwargs['frame_idx'] == 0

        def propagate_in_video(self, *args, **kwargs):
            mask = torch.full((1, 1, 8, 12), -1.0)
            mask[0, 0, 1:7, 1:10] = 1
            mask[0, 0, 3:5, 4:7] = -1  # A real hole must survive the PNG.
            yield 0, [1], mask
            if not skip:
                yield 1, [1], torch.full_like(mask, -1.0)

        def reset_state(self, state):
            self.reset = True

    predictor = Predictor()
    monkeypatch.setattr(handoff.temporal, '_load_predictor', lambda config: (predictor, {'device': 'cpu'}))
    path = tmp_path / 'payload.json'
    path.write_text(json.dumps(payload))
    config = {'models': {'temporal_participant_segmentation': {'enabled': True, 'device': 'cpu'}}}
    output = tmp_path / 'out'
    if skip:
        with pytest.raises(ValueError, match='skipped'):
            handoff.run(path, config, output)
        assert not (output / 'receipt.json').exists()
    else:
        result = handoff.run(path, config, output)
        assert result.get('data_use') == payload.get('data_use')
        first, second = [f['temporal_instances'][0] for f in result['frames']]
        assert first['visible_pixels'] == 48
        assert len(first['mask_contours']) == 2
        assert second['visible_pixels'] == 0 and second['mask_contours'] == []
        pixels = np.asarray(Image.open(output / first['mask']['file']))
        assert pixels[3, 5] == 0 and pixels[1, 1] == 1
        assert second['state'] == 'memory_propagated'
        receipt = json.loads((output / 'receipt.json').read_text())
        assert receipt['files'][first['mask']['file']] == first['mask']['sha256']
    assert predictor.reset


@pytest.mark.parametrize('bad', [None, 'legacy', 'test', 'evaluation', 'training', 'truth', 'schema', 'missing'])
def test_explicit_observations_preserve_non_training_scope(payload, tmp_path, bad):
    payload['schema_version'] = handoff.OBSERVATION_REQUEST_SCHEMA
    payload['data_use'] = {'purpose': 'production_observation'}
    payload['source'].update(schema_version='labprism-local-observation/1', split=None,
                             training_use_authorized=False, independent_ground_truth=False)
    if bad == 'legacy':
        payload['schema_version'] = handoff.REQUEST_SCHEMA
    elif bad == 'test':
        payload['source']['split'] = 'test'
    elif bad == 'evaluation':
        payload['data_use']['purpose'] = 'evaluation'
    elif bad == 'training':
        payload['source']['training_use_authorized'] = True
    elif bad == 'truth':
        payload['source']['independent_ground_truth'] = True
    elif bad == 'schema':
        payload['source']['schema_version'] = 'unknown'
    elif bad == 'missing':
        del payload['data_use']
    if bad:
        with pytest.raises(ValueError):
            handoff.validate_request(payload, tmp_path)
    else:
        handoff.validate_request(payload, tmp_path)
