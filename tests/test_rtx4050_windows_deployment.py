from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

from visioncortex.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def portable():
    return load_tool("rtx4050_portable")


def test_profile_preserves_full_chain_and_has_only_local_storage():
    config = load_config(ROOT / "configs/rtx4050-6gb-windows-local.yaml")
    parent = load_config(ROOT / "configs/rtx3050-6gb-ubuntu20-production.yaml")
    assert config["mllm"]["enabled"] is True
    assert config["performance"]["tensor_rt"] == "required"
    assert not config["performance"]["concurrent_role_scanners"]
    assert config["performance"]["engine_autotune_max_gpu_memory_fraction"] == 0.78
    assert config["models"]["temporal_participant_segmentation"]["required_for_final_key_material"]
    assert config["models"]["liquid_semantic_sidecar"]["required_for_selected_actions"]
    assert config["validation"]["model_certification"]["targets"] == parent["validation"]["model_certification"]["targets"]
    for key in ("index_csv", "device_registry_path", "archive_root", "local_input_root",
                "local_runtime_root", "local_cache_root", "local_staging_root"):
        assert config["storage"][key].startswith("./Runtime")
    assert not config["storage"]["sync_to_nas"]
    assert config["storage"]["web_upload_retention_mode"] == "local_only"
    assert not config["collection_ingest"]["enabled"]
    assert config["speech_recognition"]["enabled"] is False
    assert config["speech_recognition"]["python_executable"] is None
    assert config["speech_recognition"]["model_directory"] is None
    assert config["mllm"]["workers"] == config["mllm"]["group_workers"] == 4
    assert config["performance"]["release_auxiliary_models_between_stages"]
    assert config["performance"]["auxiliary_cpu_cache_min_available_gib"] == 8
    dino = config["models"]["open_vocabulary_key_frame"]["grounding_dino_fallback"]
    liquid = config["models"]["liquid_semantic_sidecar"]
    assert dino["device"] == liquid["device"] == "cuda"
    assert dino["cuda_oom_fallback_cpu"] and liquid["cuda_oom_fallback_cpu"]
    assert not liquid["half"]


def test_old_host_overrides_cannot_redirect_portable_config(portable, tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "SHA256SUMS.json").write_text('{}')
    monkeypatch.setenv("VISIONCORTEX_DEFAULT_CONFIG", "/old-host/missing.yaml")
    monkeypatch.setenv("VISIONCORTEX_NAS_ARCHIVE_ROOT", "/old-nas/archive")
    monkeypatch.setenv("VISIONCORTEX_FIRST_PERSON_ENGINE", "/old-gpu/engine")
    monkeypatch.setenv("YOLO_CONFIG_DIR", "/old-cache")
    monkeypatch.setenv("ARK_API_KEY", "test-value")
    # Restore all environment updates made by configure_environment.
    monkeypatch.setattr(portable.os, "environ", dict(portable.os.environ))
    portable.configure_environment(tmp_path)
    path, config = portable.effective_config(tmp_path, {"gpu_uuid": "test-gpu"})
    assert portable.os.environ["ARK_API_KEY"] == "test-value"
    assert portable.os.environ["HF_HUB_OFFLINE"] == "1"
    assert config["storage"]["archive_root"] == str(tmp_path / "Runtime/Archives")
    assert str(tmp_path / "Runtime/Engines") in config["models"]["first_person_engine"]
    assert "test-value" not in path.read_text()
    assert "/old-nas" not in path.read_text()
    # Identical hardware reuses an engine; a different GPU creates a new identity.
    _, repeated = portable.effective_config(tmp_path, {"gpu_uuid": "test-gpu"})
    assert repeated["models"]["first_person_engine"] == config["models"]["first_person_engine"]
    _, changed = portable.effective_config(tmp_path, {"gpu_uuid": "other-gpu"})
    assert changed["models"]["first_person_engine"] != config["models"]["first_person_engine"]


@pytest.mark.parametrize("relative", ["../escape", "/etc/test", "C:/test", "a\\b", "a/../../b"])
def test_manifest_rejects_escaping_paths(portable, tmp_path, relative):
    with pytest.raises(RuntimeError):
        portable.safe_path(tmp_path, relative)


def test_package_tamper_and_case_collision_fail_closed(portable, tmp_path):
    asset = tmp_path / "asset.txt"
    asset.write_text("original")
    record = {"path": asset.name, "size_bytes": asset.stat().st_size, "sha256": portable.sha256(asset)}
    sums = tmp_path / "SHA256SUMS.json"
    sums.write_text(json.dumps({"file_count": 1, "files": [record]}))
    portable.verify_package(tmp_path)
    asset.write_text("tampered")
    with pytest.raises(RuntimeError, match="校验失败"):
        portable.verify_package(tmp_path)
    asset.write_text("original")
    sums.write_text(json.dumps({"file_count": 2, "files": [record, dict(record, path="ASSET.TXT")]}))
    with pytest.raises(RuntimeError, match="Duplicate"):
        portable.verify_package(tmp_path)


def test_engine_reuse_requires_model_and_engine_hashes(portable, tmp_path):
    source = tmp_path / "model.pt"
    engine = tmp_path / "model.engine"
    source.write_bytes(b"test model")
    engine.write_bytes(b"test engine")
    config = {"models": {role: str(source) for role in ("first_person", "third_person")},
              "performance": {"image_size": 640, "engine_batch_candidates": [16, 8, 4, 2, 1]}}
    for role in ("first_person", "third_person"):
        config["models"][f"{role}_engine"] = str(engine)
    with pytest.raises(RuntimeError, match="身份回执"):
        portable.verify_engine_receipts(config)
    engine.with_suffix(".engine.build.json").write_text(json.dumps({
        "engine_sha256": portable.sha256(engine), "source_sha256": portable.sha256(source),
        "image_size": 640, "selected_batch": 4}))
    portable.verify_engine_receipts(config)
    source.write_bytes(b"changed model")
    with pytest.raises(RuntimeError, match="身份校验失败"):
        portable.verify_engine_receipts(config)


def test_builder_rejects_zip_traversal(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    builder = load_tool("build_rtx4050_offline_package")
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(RuntimeError):
        builder.extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_launcher_uses_offline_runtime_and_user_bound_secret():
    script = (ROOT / "deployment/rtx4050-windows/Start-VisionCortex.ps1").read_text()
    assert "python\\python.exe" in script
    assert "ConvertFrom-SecureString" in script
    assert "ZeroFreeBSTR" in script
    assert "Get-AuthenticodeSignature" in script
    assert "Invoke-WebRequest" not in script
    assert "pip install" not in script


def test_check_only_does_not_build_engines_or_serve(portable, tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(portable, "ROOT", tmp_path)
    monkeypatch.setattr(portable.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr(sys, "argv", ["launcher", "--check-only"])
    monkeypatch.setattr(portable, "verify_package", lambda _, **kwargs: {})
    monkeypatch.setattr(portable, "configure_environment", lambda _: None)
    monkeypatch.setattr(portable, "hardware_preflight", lambda: {"gpu": "test"})
    monkeypatch.setattr(portable, "effective_config", lambda *_: (tmp_path / "config.yaml", {}))
    monkeypatch.setattr(portable, "verify_engine_receipts", lambda _: None)
    monkeypatch.setattr(portable, "command", lambda *_: pytest.fail("must not build"))
    monkeypatch.setattr(portable, "serve", lambda *_: pytest.fail("must not serve"))
    monkeypatch.chdir(tmp_path)
    assert portable.main() == 0


def test_sam_smoke_accepts_execution_receipt_without_claiming_quality(portable, tmp_path, monkeypatch):
    from visioncortex import config as config_module
    from visioncortex import local_model_acceptance, temporal_segmentation
    config_path = tmp_path / "active.yaml"
    config_path.write_text("test config")
    engine = tmp_path / "Engines/fp.engine"
    monkeypatch.setattr(config_module, "load_config", lambda _: {"models": {"first_person_engine": str(engine)}})
    monkeypatch.setattr(local_model_acceptance, "_write_bounded_clip", lambda *_: None)
    monkeypatch.setattr(temporal_segmentation, "audit_participant_continuity", lambda *_, **kwargs: ([], {"status": "completed"}))
    portable.model_smoke("sam2", config_path)
    record = json.loads((engine.parent / "startup-smoke/sam2.json").read_text())
    assert record["model_invocation"] == "PROVEN"
    assert record["real_video_quality"] == "NOT_PROVEN"
    assert record["input_kind"] == "synthetic_not_ground_truth"


def test_desktop_state_and_descendant_environment(portable, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(portable, "DESKTOP_MODE", True)
    monkeypatch.setattr(portable.os, "environ", dict(portable.os.environ))
    portable.configure_environment(tmp_path)
    assert portable.os.environ["VISIONCORTEX_DESKTOP_MODE"] == "1"
    portable.desktop_state("engines", "准备加速引擎")
    line = capsys.readouterr().out
    record = json.loads(line.removeprefix("VISIONCORTEX_DESKTOP_STATE "))
    assert record == {"stage": "engines", "message": "准备加速引擎", "status": "preparing"}


def test_windows_job_layout_and_fail_closed_on_other_platforms(monkeypatch):
    import ctypes
    lifecycle = load_tool("windows_desktop_lifecycle")
    assert ctypes.sizeof(lifecycle.ExtendedLimits) == 144
    assert lifecycle.BasicLimits.flags.offset == 16
    assert lifecycle.ExtendedLimits.io.offset == 64
    monkeypatch.setattr(lifecycle.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="Windows parent"):
        lifecycle.supervise_desktop_parent(123)


def test_windows_job_owns_descendants_and_watches_original_parent_handle(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from unittest.mock import Mock
    lifecycle = load_tool("windows_desktop_lifecycle")
    kernel = SimpleNamespace(**{name: Mock(return_value=value) for name, value in {
        "CreateJobObjectW": 101, "SetInformationJobObject": 1,
        "AssignProcessToJobObject": 1, "GetCurrentProcess": -1,
        "OpenProcess": 202, "WaitForSingleObject": 0, "CloseHandle": 1,
        "GetStdHandle": 303, "GetFileType": 3, "ReadFile": 1, "PeekNamedPipe": 1,
    }.items()})
    targets = []
    monkeypatch.setattr(lifecycle.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_, **kwargs: kernel, raising=False)
    monkeypatch.setattr(lifecycle.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: targets.append(kw["target"])))
    lifecycle.supervise_desktop_parent(123)
    kernel.OpenProcess.assert_called_once_with(0x00100000, False, 123)
    kernel.AssignProcessToJobObject.assert_called_once_with(101, -1)
    limits = kernel.SetInformationJobObject.call_args.args[2]._obj
    assert limits.basic.flags == 0x2000
    assert len(targets) == 2
    monkeypatch.setattr(lifecycle.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as exit_info:
        targets[0]()
    assert exit_info.value.code == 0
    kernel.WaitForSingleObject.assert_called_once_with(202, 500)


def test_hidden_subprocess_keeps_existing_creation_flags(monkeypatch):
    import runpy
    from types import SimpleNamespace
    received = []

    class FakePopen:
        def __init__(self, *args, **kwargs):
            received.append((args, kwargs))

    fake = SimpleNamespace(Popen=FakePopen, CREATE_NO_WINDOW=0x08000000)
    monkeypatch.setitem(sys.modules, "subprocess", fake)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("VISIONCORTEX_DESKTOP_MODE", "1")
    runpy.run_path(str(ROOT / "deployment/rtx4050-windows/desktop/sitecustomize.py"))
    fake.Popen(["ffmpeg.exe", "test"], creationflags=0x200)
    assert received[0][1]["creationflags"] == 0x08000200


def test_desktop_overlay_has_exe_entry_and_embedded_import_paths(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    builder = load_tool("build_rtx4050_offline_package")
    build = tmp_path / "build"
    output = tmp_path / "package"
    (build / "assets").mkdir(parents=True)
    with zipfile.ZipFile(build / "assets/electron-v44.2.0-win32-x64.zip", "w") as bundle:
        bundle.writestr("electron.exe", "synthetic packaging test")
        bundle.writestr("resources/default_app.asar", "unused")
    output.mkdir()
    (output / "Start-VisionCortex.bat").write_text("legacy")
    source = {"source_files": [{"path": "src/frozen.py", "sha256": "unchanged"}]}
    builder.install_desktop(build, output, source)
    assert (output / "VisionCortex.exe").is_file()
    assert not (output / "Start-VisionCortex.bat").exists()
    assert not (output / "resources/default_app.asar").exists()
    assert "../tools" in (output / "python/python312._pth").read_text().splitlines()
    assert (output / "python/Lib/sitecustomize.py").is_file()
    assert {"path": "src/frozen.py", "sha256": "unchanged"} in source["source_files"]
    for item in source["source_files"]:
        if "package_path" in item:
            assert builder.sha256(output / item["package_path"]) == item["sha256"]


def test_package_progress_is_streamed_and_does_not_bypass_hash_check(portable, tmp_path):
    asset = tmp_path / "large.bin"
    asset.write_bytes(b"x" * (9 * 1024 * 1024))
    size = asset.stat().st_size
    (tmp_path / "SHA256SUMS.json").write_text(json.dumps({"file_count": 1, "files": [
        {"path": asset.name, "size_bytes": size, "sha256": portable.sha256(asset)},
    ]}))
    progress = []
    portable.verify_package(tmp_path, progress=progress.append)
    assert progress[0]["checked_bytes"] == 0
    assert any(0 < row["checked_bytes"] < size and row["checked_files"] == 0 for row in progress)
    assert progress[-1] == {"checked_bytes": size, "total_bytes": size, "checked_files": 1, "total_files": 1}
    with asset.open("r+b") as handle:
        handle.write(b"y")
    with pytest.raises(RuntimeError, match="校验失败"):
        portable.verify_package(tmp_path, progress=progress.append)
