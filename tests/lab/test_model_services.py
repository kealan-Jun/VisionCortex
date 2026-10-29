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
