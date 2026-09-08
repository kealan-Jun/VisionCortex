from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from visioncortex.storage import ARCHIVE_DIRECTORIES, fixed_archive_staging_paths, initialize_nas_archive


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import desktop_storage
    return desktop_storage


def test_probe_creates_only_archive_and_removes_its_own_probe(storage, tmp_path):
    selected = tmp_path / "我的实验 data"
    selected.mkdir()
    existing = selected / "keep.txt"
    existing.write_text("preserve")
    result = storage.probe(str(selected), tmp_path / "package")
    assert result["kind"] == "local"
    assert result["write_rename_read"] == "PROVEN"
    assert result["archive_root"] == str(selected / "Archives")
    assert list((selected / "Archives").iterdir()) == []
    assert existing.read_text() == "preserve"


def test_probe_failure_does_not_accept_file_or_full_disk(storage, tmp_path, monkeypatch):
    target = tmp_path / "data"
    target.write_text("not a directory")
    with pytest.raises(ValueError, match="是文件"):
        storage.probe(str(target), tmp_path / "package")
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    with pytest.raises(OSError, match="不足"):
        storage.probe(str(tmp_path / "low-space"), tmp_path / "package")


def test_discovery_only_reads_existing_network_mappings_with_a_timeout(storage, monkeypatch):
    monkeypatch.setattr(storage.sys, "platform", "win32")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=json.dumps([
            {"DeviceID": "Z:", "ProviderName": r"\\lab-nas\experiments"},
            {"DeviceID": "Y:", "ProviderName": r"\\lab-nas\experiments"},
        ]))

    monkeypatch.setattr(storage.subprocess, "run", run)
    result = storage.discover()
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["data_root"] == r"\\lab-nas\experiments\VisionCortexData"
    assert "DriveType=4" in calls[0][0][-1]
    assert calls[0][1]["timeout"] == 6
    assert "Get-ChildItem" not in calls[0][0][-1]


def test_probe_protects_package_and_rejects_ambiguous_paths(storage, tmp_path):
    package = tmp_path / "app"
    for value in (str(package), str(package / "models"), "relative/data", "/", "abc\nxyz"):
        with pytest.raises(ValueError):
            storage.normalize_root(value, package)
    assert storage.normalize_root(str(package / "Runtime"), package) == package / "Runtime"


@pytest.mark.parametrize("source_selected", [False, True])
@pytest.mark.parametrize("kind", ["local", "network"])
def test_selection_routes_uploads_archives_staging_and_local_database(storage, tmp_path, monkeypatch, kind, source_selected):
    spec = importlib.util.spec_from_file_location("portable_storage_test", ROOT / "tools/rtx4050_portable.py")
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    package, selected = tmp_path / "package", tmp_path / "选择的数据"
    shutil.copytree(ROOT / "configs", package / "configs")
    (package / "Runtime").mkdir()
    (package / "SHA256SUMS.json").write_text("{}")
    monkeypatch.setattr(portable, "DESKTOP_MODE", True)
    monkeypatch.setattr(portable.os, "environ", dict(portable.os.environ))
    source = tmp_path / "Original recordings"
    source.mkdir()
    monkeypatch.setenv("VISIONCORTEX_DESKTOP_STORAGE", json.dumps({"data_root": str(selected), "kind": kind, "source_root": str(source) if source_selected else ""}))
    # Credential validation has separate request/receipt tests; no real key or call.
    from visioncortex import provider_connection
    monkeypatch.setattr(provider_connection, "verification_matches", lambda *_: True)
    monkeypatch.setenv("VISIONCORTEX_DESKTOP_CONNECTION", json.dumps({"connection": {
        "provider": "aliyun", "base_url": "https://example.test/v1", "model": "vision", "api_protocol": "chat_completions"
    }, "verification": {"status": "verified"}}))
    monkeypatch.setenv("MLLM_API_KEY", "synthetic-storage-test")
    portable.configure_environment(package)
    config_path, config = portable.effective_config(package, {"gpu_uuid": "synthetic-hardware"})
    assert config["collection_ingest"]["enabled"] == source_selected
    if source_selected:
        assert config["collection_ingest"]["mode"] == "directory_metadata"
        assert config["collection_ingest"]["source_root"] == str(source)
        assert config["collection_ingest"]["discover_plain_video_csv"] is True
    runtime = Path(config["storage"]["local_runtime_root"])
    if kind == "local":
        assert runtime == selected
    else:
        assert runtime.parent == package / "Runtime/NetworkStores"
        assert runtime != package / "Runtime"
    assert Path(config["storage"]["local_cache_root"]) == runtime / "Cache"
    assert Path(config["storage"]["archive_root"]) == selected / "Archives"
    assert Path(portable.os.environ["TEMP"]) == runtime / "Temp"
    assert config["storage"]["sync_to_nas"] == (kind == "network")
    assert "synthetic-storage-test" not in config_path.read_text()
    from visioncortex.device_registry import load_device_registry
    registry = load_device_registry(config["storage"]["device_registry_path"])
    assert registry["devices"] == {}
    assert registry["experiment_role_overrides"] == {}
    registered = {"schema_version": registry["schema_version"], "devices": {"my-camera": {"expected_role": "first_person"}}}
    Path(config["storage"]["device_registry_path"]).write_text(json.dumps(registered))
    portable.effective_config(package, {"gpu_uuid": "synthetic-hardware"})
    assert load_device_registry(config["storage"]["device_registry_path"])["devices"] == registered["devices"]
    fixed, staging, history = fixed_archive_staging_paths(config, "experiment-one", "run-1")
    assert fixed == selected / "Archives/experiment-one"
    assert staging == selected / "Archives/Processing/experiment-one/run-1"
    assert history.parent.parent == selected / "Archives/.VisionCortex-Run-History"
    config["storage"]["active_archive_path"] = str(staging)
    initialize_nas_archive(config, "实验一")
    assert all((staging / directory).is_dir() for directory in ARCHIVE_DIRECTORIES)
    assert (staging / "处理状态.txt").is_file()
    # Selecting a different output disk must not rebuild identical GPU engines.
    first_engine = config["models"]["first_person_engine"]
    monkeypatch.setenv("VISIONCORTEX_DESKTOP_STORAGE", json.dumps({"data_root": str(tmp_path / "other"), "kind": kind}))
    _, changed = portable.effective_config(package, {"gpu_uuid": "synthetic-hardware"})
    assert changed["models"]["first_person_engine"] == first_engine
    assert Path(changed["storage"]["local_runtime_root"]) != runtime


def test_network_queue_cannot_be_inherited_by_default_local_mode(storage, tmp_path):
    import rtx4050_portable
    first = rtx4050_portable.data_runtime_root(tmp_path, {"data_root": str(tmp_path.parent / "share-one"), "kind": "network"})
    second = rtx4050_portable.data_runtime_root(tmp_path, {"data_root": str(tmp_path.parent / "share-two"), "kind": "network"})
    local = rtx4050_portable.data_runtime_root(tmp_path, {"data_root": str(tmp_path / "Runtime"), "kind": "local"})
    assert first != second and first != local and second != local


def test_source_selection_is_read_only_and_separate_from_output(storage, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = source / "video.mp4"
    original.write_bytes(b"original")
    data = tmp_path / "data"
    result = storage.probe(str(data), tmp_path / "package", str(source))
    assert result["source_root"] == str(source)
    assert list(source.iterdir()) == [original]
    assert original.read_bytes() == b"original"
    for value in (str(data), str(tmp_path), str(data / "nested")):
        Path(value).mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="分开"):
            storage.probe(str(data), tmp_path / "package", value)
