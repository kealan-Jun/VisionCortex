from __future__ import annotations

from repo_paths import ROOT

import hashlib
import importlib.util
import json
import marshal
from pathlib import Path
import struct
import subprocess
import sys
import tarfile

import pytest
import yaml




@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location("prepared_package", ROOT / "tools/build_prepared_automation_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "_implementation", module)


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / 'sealed source 中文 $ "quoted"'
    files = {
        "src/visioncortex/__init__.py": "# source snapshot\n",
        "configs/default.yaml": "models:\n  expected_class_count: 21\n",
        "configs/models/closed-set-yolo.json": '{}',
        "configs/mllm-providers.json": '{}',
        "pyproject.toml": '[project]\nname="visioncortex"\n',
        "deployment/rtx3090ti-ubuntu/_common.sh": "# shell helper\n",
        "deployment/rtx3090ti-ubuntu/render_service.py": "# service template renderer\n",
        "deployment/rtx3090ti-ubuntu/06-Run-LAN-Server.sh": "#!/usr/bin/env bash\nexit 0\n",
        "deployment/rtx3090ti-ubuntu/09-Install-Analysis-Service.sh": (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            '"$VISIONCORTEX_PYTHON" -c \'import json, os; print(json.dumps({n: os.environ[n] for n in '
            '["VISIONCORTEX_PROJECT_ROOT", "VISIONCORTEX_CONFIG", "VISIONCORTEX_SERVICE_NAME", '
            '"VISIONCORTEX_WEB_PORT", "VISIONCORTEX_WEB_AI_SETTINGS"]}))\'\n'
        ),
        "deployment/rtx3090ti-ubuntu/visioncortex-analysis.service": "[Service]\nExecStart=/bin/false\n",
        "configs/customer.yaml": (
            "device_day:\n  enabled: true\n  paused_stages: [understanding]\n"
            "collection_ingest:\n  enabled: true\n  mode: directory_metadata\n  source_root: /prepared/nas\n"
            "storage:\n  sync_to_nas: true\n  archive_root: /prepared/nas/archive\n"
            "mllm:\n  api_key_env: APPROVED_PROVIDER_KEY\n"
        ),
    }
    for name, text in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    manifest = source.parent / "SelfTestManifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "visioncortex-automation-selftest/1", "base_commit": "a" * 40,
        "files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
    }))
    return source, source / "configs/customer.yaml", tmp_path / 'candidate $(touch injected) "中文"'


def build(builder, snapshot, **kwargs):
    source, config, output = snapshot
    return builder.build_package(source, config, sys.executable, output,
                                 service_name="visioncortex-selftest-package.service", port=18012,
                                 web_ai_settings=0, **kwargs)


def run_setup(output, *args):
    return subprocess.run(["bash", str(output / "install.sh"), *args], cwd=output.parent,
                          capture_output=True, text=True, timeout=15)


def test_candidate_manifest_and_tar_are_bound_without_formal_release(builder, snapshot):
    metadata = snapshot[0] / "src/visioncortex.egg-info"
    metadata.mkdir()
    (metadata / "requires.txt").write_text("generated editable-install metadata")
    result = build(builder, snapshot)
    output = snapshot[2]
    manifest = json.loads((output / "PackageManifest.json").read_text())
    identity = json.loads((output / "App/src/visioncortex/BuildManifest.json").read_text())
    assert result["commit"] is None and result["release_ready"] is False
    assert identity["commit"] is None and identity["release_ready"] is False
    assert identity["source_digest"] == manifest["source_digest"]
    assert identity["lock_digest_scope"] == "pyproject_metadata_not_dependency_lock"
    assert manifest["base_commit_context"] == "a" * 40
    assert not any(".egg-info" in name for name in manifest["files"])
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with tarfile.open(result["archive"], "r:gz") as archive:
        assert f"{output.name}/install.sh" in archive.getnames()
        assert all("vendor/" not in name for name in archive.getnames())
    customer = yaml.safe_load((output / "App/configs/PreparedCustomer.yaml").read_text())
    assert customer["device_day"]["paused_stages"] == ["understanding"]
    assert customer["collection_ingest"]["source_root"] == "/prepared/nas"


def test_setup_verifies_and_passes_literal_prepared_paths_without_eval(builder, snapshot):
    build(builder, snapshot)
    output = snapshot[2]
    result = run_setup(output, "--setup")
    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout)
    assert environment["VISIONCORTEX_PROJECT_ROOT"] == str(output / "App")
    assert environment["VISIONCORTEX_CONFIG"] == str(output / "App/configs/PreparedCustomer.yaml")
    assert environment["VISIONCORTEX_SERVICE_NAME"] == "visioncortex-selftest-package.service"
    assert environment["VISIONCORTEX_WEB_PORT"] == "18012"
    assert environment["VISIONCORTEX_WEB_AI_SETTINGS"] == "0"
    assert not (output.parent / "injected").exists()


def test_setup_repeats_without_creating_unsealed_package_cache(builder, snapshot):
    build(builder, snapshot)
    output = snapshot[2]
    for _ in range(2):
        result = run_setup(output, "--setup")
        assert result.returncode == 0, result.stderr
        assert not list(output.rglob('*.pyc'))


@pytest.mark.parametrize("valid_header", [False, True], ids=["malformed", "matching_source_header"])
def test_setup_rejects_extra_cache_before_dispatch_even_with_valid_source_header(builder, snapshot, valid_header):
    build(builder, snapshot)
    output = snapshot[2]
    source = output / 'App/src/visioncortex/__init__.py'
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir()
    if valid_header:
        stat = source.stat()
        header = importlib.util.MAGIC_NUMBER + struct.pack('<III', 0, int(stat.st_mtime) & 0xffffffff, stat.st_size)
        cached_code = compile('value = "owned alternate cached code"\n', str(source), 'exec')
        cache.write_bytes(header + marshal.dumps(cached_code))
    else:
        cache.write_bytes(b'owned malformed test cache')
    manifest = json.loads((output / 'PackageManifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files']['App/src/visioncortex/__init__.py']
    result = run_setup(output, '--setup')
    assert result.returncode != 0
    assert 'Package file set differs from its manifest' in result.stderr
    assert not result.stdout


def test_verifier_ignores_unverified_cwd_and_pythonpath_standard_library_shadows(builder, snapshot, monkeypatch):
    build(builder, snapshot)
    output = snapshot[2]
    marker = output.parent / 'stdlib-shadow-executed'
    for name in ['json.py', 'hashlib.py']:
        (output.parent / name).write_text(
            f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n'
            'raise RuntimeError("unverified module executed")\n'
        )
    monkeypatch.setenv('PYTHONPATH', str(output.parent))
    (output / 'App/src/visioncortex/__init__.py').write_text('changed source')
    result = run_setup(output, '--setup')
    assert result.returncode != 0
    assert 'Package content changed:' in result.stderr
    assert not marker.exists()
    assert not result.stdout


@pytest.mark.parametrize("change", ["tamper", "unexpected_file", "symlink", "escape", "source_digest", "source_files", "host_escape", "host_service", "host_port"])
def test_setup_rejects_changed_or_escaping_payload_before_dispatch(builder, snapshot, change):
    build(builder, snapshot)
    output = snapshot[2]
    manifest_path = output / "PackageManifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "tamper":
        (output / "App/src/visioncortex/__init__.py").write_text("changed")
    elif change == "unexpected_file":
        (output / "App/extra.pyc").write_bytes(b"not in a permitted cache directory")
    elif change == "symlink":
        (output / "App/redirect").symlink_to(output.parent)
    elif change == "escape":
        manifest["files"]["../outside"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif change in {"source_digest", "source_files"}:
        manifest[change] = "0" * 64 if change == "source_digest" else {}
        manifest_path.write_text(json.dumps(manifest))
    else:
        host_path = output / "PreparedHost.json"
        host = json.loads(host_path.read_text())
        if change == "host_escape":
            host["config"] = "App/configs/../../outside.yaml"
        elif change == "host_service":
            host["service_name"] = "../other.service"
        else:
            host["port"] = "18012; injected"
        host_path.write_text(json.dumps(host))
        manifest["files"]["PreparedHost.json"] = hashlib.sha256(host_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    result = run_setup(output, "--setup")
    assert result.returncode != 0
    assert "服务包校验未通过" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("invalid", ["secret", "binary", "runtime", "symlink", "unsealed_change", "inheritance", "missing_seal", "relative_python", "bad_port", "bad_name", "bad_ai"])
def test_builder_rejects_unsafe_inputs_without_creating_package(builder, snapshot, invalid):
    source, config, output = snapshot
    kwargs = {}
    if invalid == "secret":
        config.write_text("mllm:\n  api_key: forbidden-literal\n")
    elif invalid == "binary":
        (source / "src/model.engine").write_bytes(b"forbidden binary")
    elif invalid == "runtime":
        (source / "src/runtime").mkdir()
        (source / "src/runtime/state.json").write_text('{}')
    elif invalid == "symlink":
        (source / "src/redirect.py").symlink_to(source / "pyproject.toml")
    elif invalid == "unsealed_change":
        (source / "src/visioncortex/__init__.py").write_text("changed snapshot")
    elif invalid == "inheritance":
        config.write_text("extends: unsealed.yaml\n")
        (config.parent / "unsealed.yaml").write_text("device_day:\n  enabled: true\n")
        manifest_path = source.parent / "SelfTestManifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["configs/customer.yaml"] = hashlib.sha256(config.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    elif invalid == "missing_seal":
        (source.parent / "SelfTestManifest.json").unlink()
    elif invalid == "relative_python":
        kwargs["python"] = "relative/python"
    elif invalid == "bad_port":
        kwargs["port"] = 65536
    elif invalid == "bad_ai":
        kwargs["web_ai_settings"] = True
    else:
        kwargs["service_name"] = "../other.service"
    options = {"python": sys.executable, "port": 8001, "service_name": "visioncortex-analysis.service"} | kwargs
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.build_package(source, config, output=output, **options)
    assert not output.exists() and not output.with_name(output.name + ".tar.gz").exists()


@pytest.mark.parametrize("args", [[], ["--unknown"], ["--setup", "extra"]])
def test_setup_rejects_unknown_arguments_without_dispatch(builder, snapshot, args):
    build(builder, snapshot)
    result = run_setup(snapshot[2], *args)
    assert result.returncode == 2
    assert result.stdout.startswith("用法：")


def test_candidate_destination_cannot_overwrite_a_previous_package(builder, snapshot):
    build(builder, snapshot)
    with pytest.raises(FileExistsError):
        build(builder, snapshot)


def test_customer_inheritance_is_sealed_and_keeps_environment_credentials(builder, snapshot):
    source, config, output = snapshot
    config.write_text("extends: default.yaml\nmllm:\n  api_key: ${APPROVED_PROVIDER_KEY}\n")
    manifest_path = source.parent / "SelfTestManifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["configs/customer.yaml"] = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    build(builder, snapshot)
    customer = yaml.safe_load((output / "App/configs/PreparedCustomer.yaml").read_text())
    assert "extends" not in customer
    assert customer["models"]["expected_class_count"] == 21
    assert customer["mllm"]["api_key"] == "${APPROVED_PROVIDER_KEY}"


@pytest.mark.parametrize("shim_text", [None, "from visioncortex import *\n"])
def test_residual_empty_source_directory_is_ignored_but_unknown_files_are_rejected(builder, snapshot, shim_text):
    source, _, output = snapshot
    compatibility = source / "src/labvision_evidence"
    compatibility.mkdir()
    if shim_text is not None:
        path = compatibility / "__init__.py"
        path.write_text(shim_text)
        manifest_path = source.parent / "SelfTestManifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    if shim_text is not None:
        with pytest.raises(ValueError, match="explicit package allowlist"):
            build(builder, snapshot)
        assert not output.exists()
    else:
        build(builder, snapshot)
        manifest = json.loads((output / "PackageManifest.json").read_text())
        assert not any("labvision_evidence" in name for name in manifest["files"])
