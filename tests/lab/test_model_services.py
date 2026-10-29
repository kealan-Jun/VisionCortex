"""Source/configuration checks only: no model, listener or child service starts."""
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ._support import ROOT


SOURCE = ROOT / 'tools/inference'


def load(name):
    spec = importlib.util.spec_from_file_location('fixture_' + name, SOURCE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(tmp_path):
    root = tmp_path / 'owned-runtime'
    root.mkdir()
    # Existing weights are outside the service root and remain in place.
    weights = tmp_path / 'existing-model-weights'
    weights.mkdir()
    (weights / 'inference.bin').write_bytes(b'fixture-weight-content-not-a-model')
    (weights / 'inference.json').write_text('{"fixture":true}')
    files = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in weights.iterdir()}
    manifest = tmp_path / 'private-checkpoint.json'
    manifest.write_text(json.dumps(dict(model_root=str(weights), revision='fixture/1', files=files)))
    token = tmp_path / 'existing-private-token'
    token.write_text('fixture-only-no-service-access-' + 'x' * 32)
    token.chmod(0o600)
    checkpoints = {name: {'manifest': str(manifest)} for name in ['fixture-vlm', 'fixture-det', 'fixture-rec']}
    config = dict(schema_version='vc-lab-model-services/1', token_file=str(token), checkpoints=checkpoints,
                  services=dict(ocr=dict(python=sys.executable, detection_checkpoint='fixture-det',
                        recognition_checkpoint='fixture-rec', detection_model_name='PP-OCRv5_mobile_det',
                        recognition_model_name='PP-OCRv5_mobile_rec'),
                    vlm=dict(python=sys.executable, checkpoint='fixture-vlm')))
    path = tmp_path / 'private-services.json'
    path.write_text(json.dumps(config))
    return root, weights, manifest, token, config, path


def test_imports_and_help_have_no_model_or_process_side_effects(monkeypatch):
    imported = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'qwen_asr', 'paddleocr', 'vllm'}
        return imported(name, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Import/help must not create runtime state or processes')
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(os, 'killpg', forbidden)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    for name in ['lab_model_service', 'lab_model_services']:
        module = load(name)
        with pytest.raises(SystemExit) as result:
            module.main(['--help'])
        assert result.value.code == 0


def test_checkpoint_reuses_full_existing_inventory_and_rejects_drift(tmp_path, monkeypatch):
    source = load('lab_model_service')
    root, weights, manifest, _, config, _ = setup(tmp_path)
    before = {p.name: p.read_bytes() for p in weights.iterdir()}
    assert source.checkpoint(root, 'fixture-det', config) == (weights, 'fixture/1')
    assert not (root / 'models').exists()
    assert {p.name: p.read_bytes() for p in weights.iterdir()} == before
    weight = weights / 'inference.bin'
    weight.write_bytes(b'X' + before[weight.name][1:])
    with pytest.raises(ValueError, match='integrity'):
        source.checkpoint(root, 'fixture-det', config)
    weight.write_bytes(before[weight.name])
    unexpected = weights / 'unlisted.bin'
    unexpected.write_bytes(b'unpinned')
    with pytest.raises(ValueError, match='inventory'):
        source.checkpoint(root, 'fixture-det', config)
    unexpected.unlink()
    weight.unlink()
    weight.symlink_to(manifest)
    with pytest.raises(ValueError, match='symlink'):
        source.checkpoint(root, 'fixture-det', config)
    with pytest.raises(ValueError, match='canonical local'):
        source.runtime_root('/mnt/fixture-unmounted/runtime')
    monkeypatch.setattr(source.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout=json.dumps({'filesystems': [{'fstype': 'cifs'}]})))
    with pytest.raises(ValueError, match='local storage'):
        source.runtime_root(root)


def test_manager_commands_use_same_repo_source_and_existing_secret(tmp_path, monkeypatch, capsys):
    module = load('lab_model_services')
    root, weights, _, token, config, config_path = setup(tmp_path)
    manager = module.ModelServices(root, config_path)
    command, environment = manager.command('ocr', probe_port=False)
    assert Path(command[2]) == SOURCE / 'lab_model_service.py'
    assert command[command.index('--runtime-root') + 1] == str(root)
    assert command[command.index('--config') + 1] == str(config_path)
    assert environment['VLLM_API_KEY'] == token.read_text()
    vlm, _ = manager.command('vlm', probe_port=False)
    assert vlm[:4] == [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve']
    assert str(weights) in vlm and '127.0.0.1' in vlm
    monkeypatch.setattr(os, 'killpg', lambda *a: pytest.fail('No process is owned'))
    assert manager.stop('ocr')['status'] == 'not_running'
    module.main(['--runtime-root', str(root), '--config', str(config_path), 'status', 'ocr'])
    assert json.loads(capsys.readouterr().out)['status'] == 'not_running'
    token.unlink()
    with pytest.raises(ValueError, match='Existing local'):
        module.ModelServices(root, config_path)
    assert not token.exists() and not (root / 'runtime').exists()
    token.write_text('x' * 64)
    token.chmod(0o644)
    with pytest.raises(ValueError, match='permissions'):
        module.ModelServices(root, config_path)


def test_ocr_selects_existing_ppv5_names_without_real_backend(tmp_path, monkeypatch):
    source = load('lab_model_service')
    root, _, _, _, config, _ = setup(tmp_path)
    observed = []
    def backend(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'paddleocr', SimpleNamespace(PaddleOCR=backend))
    infer, revision = source.make_ocr(root, config)
    assert callable(infer) and revision == 'fixture/1:fixture/1'
    assert len(observed) == 1
    assert observed[0]['text_detection_model_name'] == 'PP-OCRv5_mobile_det'
    assert observed[0]['text_recognition_model_name'] == 'PP-OCRv5_mobile_rec'
    assert observed[0]['use_doc_orientation_classify'] is False


def test_cache_identity_changes_with_verified_weights_or_effective_parameters(tmp_path):
    source = load('lab_model_service')
    root, weights, manifest, _, config, _ = setup(tmp_path)
    def identity():
        verified = {}
        for name in ['fixture-det', 'fixture-rec']:
            source.checkpoint(root, name, config, verified)
        return source.inference_identity('ocr', config, verified)
    before = identity()
    weight = weights / 'inference.bin'
    weight.write_bytes(b'changed fixture weights, same revision string')
    document = json.loads(manifest.read_text())
    document['files'][weight.name] = dict(bytes=weight.stat().st_size,
        sha256=hashlib.sha256(weight.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(document))
    assert document['revision'] == 'fixture/1'
    after_weights = identity()
    assert after_weights != before
    config['services']['ocr']['device'] = 'cpu'
    assert identity() != after_weights
    # Authentication changes are deliberately excluded from model identity.
    stable = identity()
    config['token_file'] = '/not-read-here/private-token'
    assert identity() == stable


def test_ppv6_onnx_service_explicit_names_and_backend_identity(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image
    source = load('lab_model_service')
    backend = load('ppocr_onnx')
    monkeypatch.setitem(sys.modules, 'ppocr_onnx', backend)
    root, _, _, _, config, _ = setup(tmp_path)
    config['services']['ocr'].update(engine='ppocr-onnx', device='cpu', input_mode='field_roi', cpu_threads=1,
        detection_model_name='PP-OCRv6_medium_det', recognition_model_name='PP-OCRv6_medium_rec')
    observed = []
    class Fake:
        def __init__(self, det, rec, settings):
            observed.append((det, rec, dict(settings)))
        def predict(self, pixels):
            assert pixels.shape == (12, 20, 3)
            return dict(results=[dict(rec_texts=['12?'], rec_scores=[.8], rec_polys=[])])
    monkeypatch.setattr(backend, 'PPOCRONNX', Fake)
    verified = {}
    infer, _ = source.make_ocr(root, config, verified)
    assert observed[0][2]['recognition_model_name'] == 'PP-OCRv6_medium_rec'
    raw = BytesIO()
    Image.new('RGB', (20, 12)).save(raw, format='PNG')
    assert infer(raw.getvalue())['results'][0]['rec_texts'] == ['12?']
    identity = source.inference_identity('ocr', config, verified)
    helper = tmp_path / 'changed-helper.py'
    helper.write_text('changed backend source')
    monkeypatch.setattr(backend, '__file__', str(helper))
    assert source.inference_identity('ocr', config, verified) != identity


@pytest.mark.parametrize('mode,polygons,expected', [('field_roi', [], 1), ('full_image', [], 0),
    ('full_image', [[[0, 0], [4, 0], [4, 4], [0, 4]]], 1)])
def test_onnx_field_is_explicit_and_zero_detection_never_falls_back(mode, polygons, expected):
    import numpy as np
    source = load('ppocr_onnx')
    model = source.PPOCRONNX.__new__(source.PPOCRONNX)
    model.np = np
    model.configuration = source.settings(dict(input_mode=mode, device='cpu',
        detection_model_name='PP-OCRv6_medium_det', recognition_model_name='PP-OCRv6_medium_rec'))
    model.configs = dict(det=dict(PostProcess={'fixture': True}))
    model.detect = lambda pixels: polygons
    model.crop = lambda pixels, polygon: pixels
    seen = []
    def recognize(pixels):
        seen.append(pixels.copy())
        return '0?5', .3
    model.recognize = recognize
    output = model.predict(np.zeros((8, 12, 3), np.uint8))
    assert len(seen) == expected
    assert output['results'][0]['rec_texts'] == ['0?5'] * expected
    assert output['onnx_execution']['detection_invoked'] == (mode == 'full_image')
    assert not output['onnx_execution']['complete_paddleocr_sdk_pipeline']


@pytest.mark.parametrize('settings', [dict(device='cpu'), dict(input_mode='field_roi', device='gpu:0'),
    dict(input_mode='field_roi', device='cpu', cpu_threads=True)])
def test_onnx_engine_rejects_implicit_geometry_device_or_thread_budget(settings):
    source = load('ppocr_onnx')
    with pytest.raises(ValueError):
        source.settings(settings)


def test_onnx_ctc_keeps_unknown_characters_and_only_removes_blank_and_repeat():
    import numpy as np
    source = load('ppocr_onnx')
    model = source.PPOCRONNX.__new__(source.PPOCRONNX)
    model.np = np
    model.cv = SimpleNamespace(resize=lambda pixels, size: np.zeros((size[1], size[0], 3), np.uint8))
    model.characters = ['blank', '1', '?']
    model._run = lambda kind, tensor: np.asarray([[[.05, .9, .05], [.05, .9, .05], [.9, .05, .05], [.1, .1, .8]]])
    text, score = model.recognize(np.zeros((20, 40, 3), np.uint8))
    assert text == '1?' and score == pytest.approx(.85)


def test_onnx_recognition_capacity_fails_before_any_recognition():
    import numpy as np
    source = load('ppocr_onnx')
    model = source.PPOCRONNX.__new__(source.PPOCRONNX)
    model.np = np
    model.configuration = dict(input_mode='full_image', max_recognitions=2)
    model.detect = lambda pixels: [[[0, 0]] * 4] * 3
    model.recognize = lambda pixels: pytest.fail('Capacity rejection must precede recognition')
    with pytest.raises(ValueError, match='capacity exceeded'):
        model.predict(np.zeros((8, 12, 3), np.uint8))


@pytest.mark.parametrize('failure', ['missing_names', 'wrong_name', 'unsupported_processor'])
def test_onnx_model_identity_and_known_processor_are_required(failure):
    source = load('ppocr_onnx')
    if failure == 'missing_names':
        with pytest.raises(ValueError, match='model names'):
            source.settings(dict(input_mode='field_roi', device='cpu'))
        return
    config = dict(Global=dict(model_name='PP-OCRv6_medium_rec'),
        PreProcess=dict(transform_ops=[dict(DecodeImage=dict(channel_first=False, img_mode='BGR')),
            dict(MultiLabelEncode={}), dict(RecResizeImg=dict(image_shape=[3, 48, 320])), dict(KeepKeys={})]),
        PostProcess=dict(name='CTCLabelDecode', character_dict=['1', '?']))
    if failure == 'wrong_name':
        config['Global']['model_name'] = 'PP-OCRv5_mobile_rec'
    else:
        config['PreProcess']['transform_ops'][2]['RecResizeImg']['image_shape'] = [3, 32, 320]
    with pytest.raises(ValueError):
        source.validate_model_configuration(config, 'rec', 'PP-OCRv6_medium_rec')


def test_onnx_contour_limit_reports_unexamined_regions(monkeypatch):
    import numpy as np
    source = load('ppocr_onnx')
    monkeypatch.setitem(sys.modules, 'pyclipper', SimpleNamespace())
    model = source.PPOCRONNX.__new__(source.PPOCRONNX)
    model.np = np
    model.cv = SimpleNamespace(resize=lambda pixels, size: np.zeros((size[1], size[0], 3), np.uint8),
        findContours=lambda *args: ([None] * 5, None), RETR_LIST=1, CHAIN_APPROX_SIMPLE=1)
    model.configs = dict(det=dict(PostProcess=dict(max_candidates=3, thresh=.2)))
    model._run = lambda kind, tensor: np.zeros((1, 1, 32, 32), np.float32)
    model._mini = lambda contour: (None, 0)
    assert model.detect(np.zeros((8, 12, 3), np.uint8)) == []
    assert model.detection_info == dict(total_contours=5, examined_contours=3,
        unexamined_contours=2, contour_limit=3, coverage='partial_unexamined_contours')
