from repo_paths import ROOT
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import sys



@pytest.fixture
def launcher():
    spec = spec_from_file_location("local_experience_test", ROOT / "tools/local_experience.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "_implementation", module)


def test_instance_storage_is_local_even_with_production_parent(launcher, tmp_path):
    from visioncortex.config import load_config

    original = load_config(ROOT / "configs/rtx3090ti-ubuntu-production.yaml")
    original["storage"].update(sync_to_nas=True, require_nas_source_paths=True)
    original["runtime"]["local_only"] = False
    original["device_day"]["enabled"] = True
    runtime = tmp_path / "Runtime"
    settings = launcher.local_config(original, runtime, tmp_path / "Originals", {"provider": "aliyun"})
    storage = settings["storage"]
    assert not storage["sync_to_nas"] and not storage["require_nas_source_paths"]
    assert storage["manifest_storage"] == storage["run_output_mode"] == "local"
    for key in ("index_csv", "device_registry_path", "archive_root", "local_input_root", "local_runtime_root", "local_cache_root", "local_staging_root"):
        path = Path(storage[key])
        assert path == runtime or runtime in path.parents
    assert settings["runtime"]["local_only"] is True
    assert settings["project"]["run_purpose"] == "analysis"
    assert settings["project"]["site_configuration_required"] is False
    assert settings["device_day"]["enabled"] is False
    assert settings["collection_ingest"]["enabled"] is False
    assert original["runtime"]["local_only"] is False
    assert original["device_day"]["enabled"] is True
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


def test_existing_prepared_version_cannot_be_rebuilt(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "REPOSITORY", tmp_path / "source-checkout")
    marker = tmp_path / "instance.json"
    marker.write_text('{"version": "original"}')
    sentinel = tmp_path / "existing-data.txt"
    sentinel.write_bytes(b"preserve existing data")
    with pytest.raises(RuntimeError, match="已经存在"):
        launcher.prepare(tmp_path, tmp_path)
    assert marker.read_text() == '{"version": "original"}'
    assert sentinel.read_bytes() == b"preserve existing data"


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



def test_local_experience_configuration_can_be_reloaded(launcher, tmp_path, monkeypatch):
    import yaml
    from visioncortex.config import load_config

    monkeypatch.delenv("VISIONCORTEX_SITE_CONFIG", raising=False)
    settings = launcher.local_config(load_config(ROOT / "configs/rtx3090ti-ubuntu-production.yaml"),
                                     tmp_path / "Runtime", tmp_path / "Originals", {})
    config = tmp_path / "instance.yaml"
    config.write_text(yaml.safe_dump(settings))
    assert load_config(config)["project"]["run_purpose"] == "analysis"
