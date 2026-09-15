from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher():
    spec = spec_from_file_location("local_experience_test", ROOT / "tools/local_experience.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_instance_storage_is_local_even_with_production_parent(launcher, tmp_path):
    from visioncortex.config import load_config

    original = load_config(ROOT / "configs/rtx3090ti-ubuntu-production.yaml")
    runtime = tmp_path / "Runtime"
    settings = launcher.local_config(original, runtime, tmp_path / "Originals", {"provider": "aliyun"})
    storage = settings["storage"]
    assert not storage["sync_to_nas"] and not storage["require_nas_source_paths"]
    assert storage["manifest_storage"] == storage["run_output_mode"] == "local"
    for key in ("index_csv", "device_registry_path", "archive_root", "local_input_root", "local_runtime_root", "local_cache_root", "local_staging_root"):
        path = Path(storage[key])
        assert path == runtime or runtime in path.parents
    assert settings["mllm"]["enabled"]
    assert settings["models"] == original["models"]
    assert settings["validation"] == original["validation"]
    assert original["storage"]["sync_to_nas"]


@pytest.mark.skipif(sys.platform == "win32", reason="Ubuntu launcher uses flock and /proc")
def test_reopening_reuses_owned_service_instead_of_starting_another(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "owned_server", lambda root: {"pid": 123, "url": "http://127.0.0.1:8123"})
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must reuse existing instance"))
    assert launcher.launch(tmp_path, True) == {"pid": 123, "url": "http://127.0.0.1:8123/#/ai-settings"}


@pytest.mark.parametrize("state", ["running", "queued", "input_preflight", "candidate_fine", "finalizing", None])
def test_stop_preserves_an_active_analysis(launcher, tmp_path, monkeypatch, state):
    monkeypatch.setattr(launcher, "owned_server", lambda root: {"pid": 123, "url": "http://127.0.0.1:8123"})
    monkeypatch.setattr(launcher, "request", lambda url: {"runs": [{"state": state}]})
    monkeypatch.setattr(launcher.os, "kill", lambda *args: pytest.fail("must not stop active work"))
    with pytest.raises(RuntimeError, match="等待"):
        launcher.stop(tmp_path)


def test_existing_prepared_version_cannot_be_rebuilt(launcher, tmp_path):
    (tmp_path / "instance.json").write_text("{}")
    with pytest.raises(RuntimeError, match="已经存在"):
        launcher.prepare(tmp_path, tmp_path)


def test_wrong_pid_cannot_be_reused_as_the_instance(launcher, tmp_path, monkeypatch):
    launcher.write_json(tmp_path / "Runtime/server.json", {"pid": 999, "url": "http://127.0.0.1:8123"})
    monkeypatch.setattr(launcher, "Path", lambda name: SimpleNamespace(read_bytes=lambda: b"other-service\0"))
    monkeypatch.setattr(launcher, "request", lambda *args: pytest.fail("must reject wrong process before probing"))
    assert launcher.owned_server(tmp_path) is None


def test_fresh_catalog_loads_and_initialization_preserves_device_roles(launcher, tmp_path):
    from visioncortex.collection_catalog import discover_collections
    from visioncortex.config import load_config
    import json

    settings = launcher.local_config(load_config(ROOT / "configs/rtx3090ti-ubuntu-local.yaml"), tmp_path / "Runtime", tmp_path / "Sources", {})
    settings["collection_ingest"]["enabled"] = False
    launcher.initialize_storage(settings)
    assert discover_collections(settings)["collections"] == []
    path = Path(settings["storage"]["device_registry_path"])
    value = json.loads(path.read_text())
    value["devices"]["camera-test"] = {"expected_role": "first_person"}
    launcher.write_json(path, value)
    before = path.read_bytes()
    launcher.initialize_storage(settings)
    assert path.read_bytes() == before
