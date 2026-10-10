from __future__ import annotations

import json
import builtins
from pathlib import Path

import pytest
import yaml

from repo_paths import ROOT
from visioncortex.config import _deep_merge, _load_profile, load_config
from visioncortex.device_registry import load_device_registry, resolve_view_role
from visioncortex.collection_catalog import _snapshot_path
from visioncortex.runtime_options import validate as validate_runtime


def _refuse_optional_model_imports(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'ultralytics', 'tensorrt', 'transformers', 'sam2'}:
            raise AssertionError('Unconfigured templates must stop before model imports')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def test_unconfigured_template_rejects_model_validation_before_model_imports(tmp_path, monkeypatch):
    from visioncortex.detection import validate_models
    config = load_config(ROOT / 'configs/rtx4060-laptop-production.yaml')
    output = tmp_path / 'uncreated-output'
    config['project']['output_root'] = str(output)
    _refuse_optional_model_imports(monkeypatch)
    with pytest.raises(ValueError, match='Configure a private site'):
        validate_models(config)
    assert not output.exists()


def test_unconfigured_template_rejects_pipeline_before_media_or_output(tmp_path, monkeypatch):
    from visioncortex import pipeline
    from visioncortex.schemas import RunManifest, ViewInput, ViewRole
    config = load_config(ROOT / 'configs/rtx3090ti-ubuntu-production.yaml')
    output, runtime = tmp_path / 'uncreated-output', tmp_path / 'uncreated-runtime'
    config['project']['output_root'] = str(output)
    config['storage']['local_runtime_root'] = str(runtime)
    manifest = RunManifest(experiment_id='EXAMPLE-SESSION', views=[
        ViewInput(view_id='wearable-01', role=ViewRole.FIRST_PERSON, video=tmp_path / 'missing-first.mp4'),
        ViewInput(view_id='fixed-01', role=ViewRole.THIRD_PERSON, video=tmp_path / 'missing-third.mp4'),
    ])
    def refuse_media(*_args, **_kwargs):
        raise AssertionError('Unconfigured templates must stop before media preflight')
    monkeypatch.setattr(pipeline, 'run_media_pipeline_preflight', refuse_media)
    _refuse_optional_model_imports(monkeypatch)
    with pytest.raises(ValueError, match='Configure a private site'):
        pipeline.EvidencePipeline(config).run(manifest)
    assert not output.exists()
    assert not runtime.exists()


@pytest.mark.parametrize('profile', [None, ROOT / 'configs/development-local.yaml'])
def test_first_clone_defaults_are_local_cpu_without_providers_or_collectors(profile):
    value = load_config(profile)
    assert value['project']['run_purpose'] == 'demo'
    assert value['runtime']['local_only'] is True
    assert value['performance']['device'] == 'cpu'
    assert value['performance']['tensor_rt'] == 'off'
    assert value['performance']['coarse_decode_lanes'] == ['cpu']
    assert value['performance']['fine_decode_lanes'] == ['cpu']
    for section in ('mllm', 'speech_recognition', 'collection_ingest', 'device_day'):
        assert value[section]['enabled'] is False
    assert value['storage']['sync_to_nas'] is False
    assert value['fixed_benchmark']['enabled'] is False
    assert value['fixed_benchmark']['experiment_id'] is None
    assert value['fixed_benchmark']['archive_name'] is None
    runtime = Path(value['storage']['local_runtime_root']).absolute()
    for key in ('archive_root', 'local_cache_root', 'local_staging_root', 'local_input_root'):
        assert Path(value['storage'][key]).absolute().is_relative_to(runtime)
    assert value['mllm']['base_url'] == ''
    assert value['mllm']['model'] == ''


@pytest.mark.parametrize('profile', [
    'development-local.yaml', 'rtx3090ti-ubuntu-local.yaml',
    'rtx3050-6gb-ubuntu20-local.yaml', 'rtx4050-6gb-windows-local.yaml',
])
def test_local_catalog_snapshot_stays_with_its_selected_runtime(profile):
    value = load_config(ROOT / 'configs' / profile)
    root = Path(value['storage']['local_runtime_root']).absolute()
    assert _snapshot_path(value).absolute().is_relative_to(root)


@pytest.mark.parametrize(('name', 'value'), [
    ('VISIONCORTEX_TENSORRT', 'required'),
    ('VISIONCORTEX_COARSE_DECODE_LANES', 'cuda'),
    ('VISIONCORTEX_FINE_DECODE_LANES', 'cuda'),
    ('VISIONCORTEX_NAS_ARCHIVE_ROOT', '/external/archive'),
    ('VISIONCORTEX_NAS_INDEX_CSV', '/external/index.csv'),
    ('VISIONCORTEX_OUTPUT_ROOT', '/external/output'),
    ('VISIONCORTEX_SELECTIVE_KEY_MATERIAL_VERIFICATION', 'true'),
    ('VISIONCORTEX_FIRST_PERSON_MODEL', '/site/model/first.pt'),
    ('VISIONCORTEX_THIRD_PERSON_MODEL', '/site/model/third.pt'),
    ('VISIONCORTEX_FIRST_PERSON_ENGINE', '/site/model/first.engine'),
    ('VISIONCORTEX_THIRD_PERSON_ENGINE', '/site/model/third.engine'),
])
def test_inherited_environment_cannot_turn_demo_into_another_runtime(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match='Demo|Local-only'):
        load_config()


def test_windows_unc_local_root_is_rejected_before_any_share_access():
    config = {'runtime': {'local_only': True}, 'storage': {
        'local_runtime_root': r'\\example-host\example-share\runtime',
    }}
    with pytest.raises(ValueError, match='must not point to a mounted share'):
        validate_runtime(config)


def test_explicit_site_overlay_can_configure_production_without_builtin_identity(tmp_path, monkeypatch):
    site = tmp_path / 'site.yaml'
    site.write_text(yaml.safe_dump({
        'project': {'run_purpose': 'production', 'site_configuration_required': False},
        'runtime': {'local_only': False},
        'storage': {'archive_root': '/site/archive', 'local_runtime_root': '/site/runtime'},
        'collection_ingest': {'camera_role_map': {'site-wearable': 'first_person'}},
    }))
    monkeypatch.setenv('VISIONCORTEX_SITE_CONFIG', str(site))
    value = load_config(ROOT / 'configs/rtx3090ti-ubuntu-production.yaml')
    assert value['project']['run_purpose'] == 'production'
    assert value['runtime']['local_only'] is False
    assert value['storage']['archive_root'] == '/site/archive'
    assert value['collection_ingest']['camera_role_map'] == {'site-wearable': 'first_person'}
    # A provided site identity is configuration, not permission to auto-run.
    assert value['collection_ingest']['enabled'] is False


def test_production_template_cannot_claim_configuration_without_explicit_site(tmp_path):
    profile = tmp_path / 'unconfigured.yaml'
    profile.write_text('project:\n  run_purpose: production\n  site_configuration_required: true\n')
    with pytest.raises(ValueError, match='explicitly configured private site'):
        load_config(profile)


def test_explicit_frozen_private_baseline_does_not_inherit_new_public_defaults(tmp_path, monkeypatch):
    baseline = _load_profile(ROOT / 'configs/default.yaml')
    baseline['project'].pop('run_purpose')
    baseline['project']['output_root'] = '/site/output'
    baseline['runtime']['local_only'] = False
    baseline['storage']['local_runtime_root'] = '/site/runtime'
    baseline['storage']['archive_root'] = '/site/archive'
    baseline['models']['first_person'] = '/site/models/first.pt'
    baseline['private_baseline_marker'] = 'fixed-private-contract'
    frozen = tmp_path / 'default.yaml'
    frozen.write_text(yaml.safe_dump(baseline))
    profile_data = {'models': {'third_person': '/site/models/third.pt'}}
    profile = tmp_path / 'profile.yaml'
    profile.write_text(yaml.safe_dump(profile_data))
    monkeypatch.setenv('VISIONCORTEX_DEFAULT_CONFIG', str(frozen))
    monkeypatch.delenv('VISIONCORTEX_SITE_CONFIG', raising=False)
    assert load_config(profile) == _deep_merge(baseline, profile_data)


def test_missing_explicit_site_overlay_fails_without_creating_a_fallback(tmp_path, monkeypatch):
    path = tmp_path / 'missing-site.yaml'
    monkeypatch.setenv('VISIONCORTEX_SITE_CONFIG', str(path))
    with pytest.raises(FileNotFoundError):
        load_config()
    assert not path.exists()


def test_private_site_inheritance_cannot_escape_its_directory(tmp_path, monkeypatch):
    private = tmp_path / 'private'
    private.mkdir()
    (tmp_path / 'outside.yaml').write_text('project: {}\n')
    path = private / 'site.yaml'
    path.write_text('extends: ../outside.yaml\n')
    monkeypatch.setenv('VISIONCORTEX_SITE_CONFIG', str(path))
    with pytest.raises(ValueError, match='must stay'):
        load_config()


def test_default_registry_contains_no_deployment_devices(monkeypatch):
    monkeypatch.delenv('VISIONCORTEX_DEVICE_REGISTRY', raising=False)
    registry = load_device_registry()
    assert registry['devices'] == {}
    assert registry['experiment_role_overrides'] == {}
    result = resolve_view_role(registry, 'new-session', 'new-camera', 'first')
    assert result['resolved_role'] == 'first_person'
    assert result['resolution_source'] == 'experiment_record_index'


def test_explicit_private_device_registry_retains_role_policy(tmp_path, monkeypatch):
    private = tmp_path / 'devices.json'
    private.write_text(json.dumps({
        'schema_version': 'visioncortex-device-registry/1',
        'devices': {'site-wearable': {'expected_role': 'first_person', 'role_policy': {
            'status': 'approved', 'reason': 'Reviewed site role policy',
            'source': 'site-reviewed-role-ledger',
        }}},
        'experiment_role_overrides': {},
    }))
    monkeypatch.setenv('VISIONCORTEX_DEVICE_REGISTRY', str(private))
    result = resolve_view_role(load_device_registry(), 'session', 'site-wearable', 'front')
    assert result['resolved_role'] == 'first_person'
    assert result['resolution_source'] == 'approved_device_registry_override'
    assert result['blocking_reasons'] == []


def test_explicit_device_path_takes_precedence_over_site_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('VISIONCORTEX_DEVICE_REGISTRY', str(tmp_path / 'not-present.json'))
    registry = load_device_registry(ROOT / 'examples/device-registry.example.json')
    assert set(registry['devices']) == {'wearable-01', 'fixed-01'}
