from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def portable():
    spec = importlib.util.spec_from_file_location("engine_reuse_portable", ROOT / "tools/rtx4050_portable.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(portable, tmp_path):
    files = []
    for name in ("python/python.exe", "src/visioncortex/cli.py", "src/visioncortex/web/app.js",
                 "models/first.pt", "models/third.pt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": portable.sha256(path)})
    manifest = {"files": files, "file_count": len(files)}
    portable.write_json(tmp_path / "SHA256SUMS.json", manifest)
    config = {"models": {"first_person": str(tmp_path / "models/first.pt"),
                         "third_person": str(tmp_path / "models/third.pt"),
                         "first_person_engine": str(tmp_path / "Runtime/Engines/first_person.engine"),
                         "third_person_engine": str(tmp_path / "Runtime/Engines/third_person.engine")},
              "performance": {"image_size": 640, "half": True, "device": 0,
                              "batch_size": 16, "engine_batch_candidates": [16, 8, 4, 2, 1],
                              "engine_dynamic": False, "engine_workspace_gib": 1.0,
                              "engine_autotune_iterations": 6,
                              "engine_autotune_max_gpu_memory_fraction": 0.78}}
    hardware = {"gpu_uuid": "test-gpu", "driver": "test-driver", "tensorrt": "10.4",
                "torch": "2.6+cu124", "python": "3.12", "ram_gib": 64}
    return tmp_path, config, hardware, manifest


def install_engines(portable, config, directory):
    result = copy.deepcopy(config)
    for role in ("first_person", "third_person"):
        engine = directory / f"{role}.engine"
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(role.encode())
        result["models"][f"{role}_engine"] = str(engine)
        portable.write_json(engine.with_suffix(".engine.build.json"), {
            "engine_sha256": portable.sha256(engine),
            "source_sha256": portable.sha256(Path(config["models"][role])),
            "image_size": 640, "selected_batch": 16, "half": True, "dynamic": False,
            "batch_candidates": [16, 8, 4, 2, 1], "workspace_gib": 1.0,
            "autotune_iterations": 6, "autotune_max_gpu_memory_fraction": 0.78})
    return result


def test_unrelated_source_and_analysis_changes_keep_cache_identity(portable, bundle):
    root, config, hardware, manifest = bundle
    original = portable.engine_cache_identity(root, config, hardware, manifest)
    manifest["files"][2]["sha256"] = "updated-ui"
    config["alignment"] = {"timestamp_sample_limit": 500}
    config["performance"]["cpu_decode_threads"] = 3
    config["mllm"] = {"api_key_env": "ANOTHER_ENVIRONMENT"}
    hardware["ram_gib"] = 32
    assert portable.engine_cache_identity(root, config, hardware, manifest) == original


@pytest.mark.parametrize("field,value", [
    ("image_size", 960), ("half", False), ("device", 1),
    ("engine_dynamic", True), ("engine_workspace_gib", 2.0),
    ("engine_batch_candidates", [8, 4, 2, 1]), ("engine_autotune_iterations", 4),
    ("engine_autotune_max_gpu_memory_fraction", 0.7)])
def test_build_parameters_invalidate_cache(portable, bundle, field, value):
    root, config, hardware, manifest = bundle
    original = portable.engine_cache_identity(root, config, hardware, manifest)
    config["performance"][field] = value
    assert portable.engine_cache_identity(root, config, hardware, manifest) != original


@pytest.mark.parametrize("field", ["gpu_uuid", "driver", "tensorrt", "torch", "python"])
def test_runtime_identity_changes_invalidate_cache(portable, bundle, field):
    root, config, hardware, manifest = bundle
    original = portable.engine_cache_identity(root, config, hardware, manifest)
    hardware[field] = "changed"
    assert portable.engine_cache_identity(root, config, hardware, manifest) != original


@pytest.mark.parametrize("index", [0, 1, 3, 4])
def test_runtime_builder_and_weight_changes_invalidate_cache(portable, bundle, index):
    root, config, hardware, manifest = bundle
    original = portable.engine_cache_identity(root, config, hardware, manifest)
    manifest["files"][index]["sha256"] = "changed"
    assert portable.engine_cache_identity(root, config, hardware, manifest) != original


def legacy_bundle(portable, bundle):
    root, config, hardware, manifest = bundle
    old_hash = portable.sha256(root / "SHA256SUMS.json")
    legacy = portable.legacy_engine_signature(config, hardware, old_hash)
    directory = root / "Runtime/Engines" / legacy
    install_engines(portable, config, directory)
    backup = root / "Runtime/Updates/source-test"
    portable.write_json(backup / "before/SHA256SUMS.json", manifest)
    portable.write_json(backup / "result.json", {"status": "applied", "base_manifest_sha256": old_hash})
    manifest["files"][2]["sha256"] = "updated-ui"
    portable.write_json(root / "SHA256SUMS.json", manifest)
    return directory, backup


def test_legacy_adoption_preserves_paths_and_survives_backup_removal(portable, bundle):
    root, config, hardware, _ = bundle
    directory, backup = legacy_bundle(portable, bundle)
    before = (directory / "first_person.engine").stat().st_mtime_ns
    assert portable.engine_cache_directory(root, config, hardware) == directory
    (backup / "before/SHA256SUMS.json").unlink()
    assert portable.engine_cache_directory(root, config, hardware) == directory
    assert (directory / "first_person.engine").stat().st_mtime_ns == before


@pytest.mark.parametrize("reason", ["runtime", "hardware", "receipt", "backup_hash", "malformed_result", "malformed_manifest", "malformed_entry"])
def test_legacy_adoption_requires_compatible_provenance(portable, bundle, reason):
    root, config, hardware, manifest = bundle
    directory, backup = legacy_bundle(portable, bundle)
    if reason == "runtime":
        manifest["files"][0]["sha256"] = "different-runtime"
        portable.write_json(root / "SHA256SUMS.json", manifest)
    elif reason == "hardware":
        hardware["gpu_uuid"] = "different-gpu"
    elif reason == "receipt":
        (directory / "first_person.engine.build.json").unlink()
        with pytest.raises(RuntimeError, match="身份回执"):
            portable.engine_cache_directory(root, config, hardware)
        return
    elif reason == "malformed_result":
        (backup / "result.json").write_text("null")
    elif reason in {"malformed_manifest", "malformed_entry"}:
        invalid = "null" if reason == "malformed_manifest" else '{"files": [{"path": null, "sha256": "invalid"}]}'
        (backup / "before/SHA256SUMS.json").write_text(invalid)
        portable.write_json(backup / "result.json", {
            "status": "applied", "base_manifest_sha256": portable.sha256(backup / "before/SHA256SUMS.json")})
    else:
        portable.write_json(backup / "result.json", {"status": "applied", "base_manifest_sha256": "wrong"})
    assert portable.engine_cache_directory(root, config, hardware) != directory


@pytest.mark.parametrize("field,value", [("half", False), ("dynamic", True),
    ("workspace_gib", 2.0), ("batch_candidates", [8]),
    ("autotune_iterations", 4), ("autotune_max_gpu_memory_fraction", 0.6)])
def test_incompatible_build_receipt_is_rejected(portable, bundle, field, value):
    root, config, _, _ = bundle
    config = install_engines(portable, config, root / "engines")
    receipt = root / "engines/first_person.engine.build.json"
    data = json.loads(receipt.read_text())
    data[field] = value
    portable.write_json(receipt, data)
    with pytest.raises(RuntimeError, match="身份校验失败"):
        portable.verify_engine_receipts(config)


def test_reuse_skips_export_subprocess_and_records_elapsed_time(portable, bundle, monkeypatch, capsys):
    root, config, _, _ = bundle
    config = install_engines(portable, config, root / "engines")
    monkeypatch.setattr(portable, "ROOT", root)
    monkeypatch.setattr(portable, "command", lambda *_: pytest.fail("cache hit must not export"))
    portable.prepare_engines(root / "config.yaml", config)
    record = json.loads((root / "Runtime/Logs/engine-preparation.json").read_text())
    assert record["mode"] == "reused"
    assert record["elapsed_seconds"] >= 0
    assert "复用" in capsys.readouterr().out


def test_missing_engine_builds_only_via_existing_command(portable, bundle, monkeypatch):
    root, config, _, _ = bundle
    config = install_engines(portable, config, root / "engines")
    missing = root / "engines/third_person.engine"
    missing.unlink()
    calls = []
    def command(path, action):
        calls.append(action)
        missing.write_bytes(b"third_person")
    monkeypatch.setattr(portable, "ROOT", root)
    monkeypatch.setattr(portable, "command", command)
    portable.prepare_engines(root / "config.yaml", config)
    assert calls == ["prepare-engine"]
    record = json.loads((root / "Runtime/Logs/engine-preparation.json").read_text())
    assert record["mode"] == "built"


def test_incomplete_export_does_not_report_ready(portable, bundle, monkeypatch):
    root, config, _, _ = bundle
    monkeypatch.setattr(portable, "ROOT", root)
    monkeypatch.setattr(portable, "command", lambda *_: None)
    with pytest.raises(RuntimeError, match="引擎"):
        portable.prepare_engines(root / "config.yaml", config)
    record = json.loads((root / "Runtime/Logs/engine-preparation.json").read_text())
    assert record["status"] == "failed"


def test_legacy_mapping_is_bound_to_verified_engine_bytes(portable, bundle):
    root, config, hardware, _ = bundle
    directory, _ = legacy_bundle(portable, bundle)
    portable.engine_cache_directory(root, config, hardware)
    engine = directory / "first_person.engine"
    engine.write_bytes(b"same model and parameters, different engine build")
    receipt_path = engine.with_suffix(".engine.build.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["engine_sha256"] = portable.sha256(engine)
    portable.write_json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="映射"):
        portable.engine_cache_directory(root, config, hardware)


def test_failed_validation_replaces_previous_success_receipt(portable, bundle, monkeypatch):
    root, config, _, _ = bundle
    config = install_engines(portable, config, root / "engines")
    monkeypatch.setattr(portable, "ROOT", root)
    portable.prepare_engines(root / "config.yaml", config)
    (root / "engines/first_person.engine").write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="身份校验失败"):
        portable.prepare_engines(root / "config.yaml", config)
    record = json.loads((root / "Runtime/Logs/engine-preparation.json").read_text())
    assert record["status"] == "failed"


def test_changed_package_cannot_reuse_stale_auxiliary_smoke_receipts(portable, bundle, monkeypatch):
    root, config, _, _ = bundle
    config = install_engines(portable, config, root / "engines")
    config_path = root / "config.yaml"
    config_path.write_text("same configuration")
    for stage in ("world", "dino", "labpics", "sam2"):
        portable.write_json(root / "engines/startup-smoke" / f"{stage}.json", {
            "model_invocation": "PROVEN", "config_sha256": portable.sha256(config_path),
            "source_manifest_sha256": portable.sha256(root / "SHA256SUMS.json")})
    monkeypatch.setattr(portable, "ROOT", root)
    calls = []
    monkeypatch.setattr(portable, "run_logged", calls.append)
    portable.ensure_model_smoke(config_path, config)
    assert calls == []
    (root / "SHA256SUMS.json").write_text("changed model or source package")
    portable.ensure_model_smoke(config_path, config)
    assert len(calls) == 4
