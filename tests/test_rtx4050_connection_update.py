"""Filesystem update/rollback tests use tiny synthetic packages, never a GPU runtime."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    builder = load(ROOT / "tools/build_rtx4050_offline_package.py", "update_builder")
    updater = load(ROOT / "deployment/rtx4050-windows/apply-update.py", "package_updater")
    base, patch = tmp_path / "base", tmp_path / "patch"
    contents = {name: b"synthetic old source" for name in [
        "tools/rtx4050_portable.py", "tools/verify_mllm_connection.py", "README.md",
        "resources/app/main.cjs", "resources/app/connection.cjs", "resources/app/setup.js",
        "python/python.exe", "VisionCortex.exe", "models/unchanged.bin",
    ]}
    contents["receipts/source-snapshot.json"] = json.dumps({"base_commit": "synthetic-sha", "source_files": [
        {"path": "models/unchanged.bin", "package_path": "models/unchanged.bin", "sha256": "synthetic"},
    ]}).encode()
    contents["BUNDLE-METADATA.json"] = b'{"windows_runtime":"NOT_PROVEN"}'
    records = []
    for name, data in contents.items():
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"path": name, "size_bytes": len(data), "sha256": updater.digest(target)})
    (base / "SHA256SUMS.json").write_text(json.dumps({"file_count": len(records), "files": records}))
    runtime = base / "Runtime/Desktop/settings.json"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("synthetic existing user configuration")
    old = {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    builder.connection_update(base, patch)
    return updater, builder, base, patch, old


def test_update_preserves_models_data_and_unchanged_manifest_then_reuses_safely(bundle):
    updater, builder, base, patch, old = bundle
    result = updater.apply(base, patch)
    assert result["status"] == "applied"
    builder.verify_package(base)
    for name in ["models/unchanged.bin", "python/python.exe", "Runtime/Desktop/settings.json"]:
        assert (base / name).read_bytes() == old[name]
    assert (base / "tools/verify_mllm_connection.py").read_bytes() == (ROOT / "tools/verify_mllm_connection.py").read_bytes()
    assert (Path(result["backup"]) / "before/SHA256SUMS.json").read_bytes() == old["SHA256SUMS.json"]
    assert updater.apply(base, patch) == {"status": "already_applied"}
    (base / "tools/verify_mllm_connection.py").write_text("damaged")
    with pytest.raises(RuntimeError, match="damaged"):
        updater.apply(base, patch)


@pytest.mark.parametrize("kind", ["base_version", "base_file", "payload"])
def test_invalid_update_never_starts_a_partial_install(bundle, kind):
    updater, _, base, patch, _ = bundle
    target = {"base_version": base / "SHA256SUMS.json", "base_file": base / "tools/verify_mllm_connection.py",
              "payload": patch / "payload/tools/verify_mllm_connection.py"}[kind]
    target.write_bytes(target.read_bytes() + b" ")
    before = {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError):
        updater.apply(base, patch)
    assert {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*") if p.is_file()} == before


def test_mid_update_replace_failure_restores_original_files(bundle, monkeypatch):
    updater, builder, base, patch, old = bundle
    real_replace = updater.os.replace
    calls = []

    def fail_third(source, target):
        calls.append(target)
        if len(calls) == 3:
            raise PermissionError("synthetic Windows locked file")
        real_replace(source, target)

    monkeypatch.setattr(updater.os, "replace", fail_third)
    with pytest.raises(PermissionError):
        updater.apply(base, patch)
    for name, data in old.items():
        assert (base / name).read_bytes() == data
    builder.verify_package(base)
    assert list((base / "Runtime/Updates").glob("*/FAILED.txt"))


def test_update_cannot_silently_change_unrelated_model_manifest(bundle):
    updater, _, base, patch, _ = bundle
    manifest = patch / "payload/SHA256SUMS.json"
    value = json.loads(manifest.read_text())
    next(item for item in value["files"] if item["path"] == "models/unchanged.bin")["sha256"] = "wrong"
    manifest.write_text(json.dumps(value))
    spec = json.loads((patch / "update.json").read_text())
    spec["updated_manifest_sha256"] = updater.digest(manifest)
    next(item for item in spec["files"] if item["path"] == "SHA256SUMS.json")["after_sha256"] = updater.digest(manifest)
    (patch / "update.json").write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="unrelated"):
        updater.apply(base, patch)
    assert not (base / "Runtime/Updates").exists()
