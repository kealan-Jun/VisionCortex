from pathlib import Path
import sqlite3

import pytest

from visioncortex.benchmark_registration import registration
from visioncortex.config import load_config
from visioncortex.runtime_shutdown import snapshot
from visioncortex import local_storage
from visioncortex import model_runtime_readiness


def test_first_clone_does_not_register_a_site_benchmark():
    settings = load_config()
    assert registration(settings)["enabled"] is False
    with pytest.raises(ValueError, match="explicit site registration"):
        registration(settings, required=True)


@pytest.mark.parametrize("archive", ["../escape", "a/b", " a ", "a\\b"])
def test_registered_benchmark_rejects_ambiguous_archive_identity(archive):
    settings = {"fixed_benchmark": {"enabled": True, "experiment_id": "fixture", "archive_name": archive}}
    with pytest.raises(ValueError):
        registration(settings, required=True)


def test_incomplete_site_cannot_reserve_a_benchmark():
    settings = {"project": {"site_configuration_required": True},
                "fixed_benchmark": {"enabled": True, "experiment_id": "fixture", "archive_name": "Fixture-Benchmark"}}
    assert registration(settings)["enabled"] is False
    with pytest.raises(ValueError, match="private site"):
        registration(settings, required=True)


@pytest.mark.parametrize("endpoint", ["/api/runs", "/api/runs/from-paths", "/api/upload-sessions",
                                    "/api/collections/fixture/runs", "/api/benchmarks/fixture/runs"])
def test_template_submission_stops_before_credentials_or_storage(monkeypatch, endpoint):
    from fastapi.testclient import TestClient
    from visioncortex import api
    from visioncortex.web_api import boundary
    settings = load_config(Path("configs/rtx4060-laptop-production.yaml"))
    monkeypatch.setattr(api, "_settings", lambda: settings)
    monkeypatch.setattr(boundary, "audit_production_model_certification",
                        lambda _: pytest.fail("template touched production certification"))
    response = TestClient(api.app).post(endpoint, json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "site_configuration_required"


def test_unconfigured_device_day_does_not_create_local_queues(tmp_path):
    from visioncortex.device_day import DeviceDayRunner
    settings = load_config(Path("configs/rtx3090ti-ubuntu-production.yaml"))
    settings["storage"]["local_runtime_root"] = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="private site"):
        DeviceDayRunner(settings)
    assert not (tmp_path / "absent").exists()


def test_unconfigured_nas_template_cannot_report_analysis_ready(monkeypatch):
    from fastapi.testclient import TestClient
    from visioncortex import api
    from visioncortex.web_api import health

    settings = load_config(Path("configs/rtx4060-laptop-production.yaml"))
    settings["storage"]["sync_to_nas"] = True
    monkeypatch.setattr(api, "_settings", lambda: settings)
    monkeypatch.setattr(health, "audit_production_model_certification",
                        lambda _: pytest.fail("unconfigured template inspected models"))
    monkeypatch.setattr(health, "_nas_storage_available",
                        lambda _: pytest.fail("unconfigured template probed NAS"))
    response = TestClient(api.app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["analysis_ready"] is False
    assert "私有站点配置" in response.json()["analysis_blocker"]


@pytest.mark.parametrize("mode", [True, "true", "required"])
def test_tensor_rt_required_aliases_need_engine_files(tmp_path, monkeypatch, mode):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fake-model-not-loaded")
    monkeypatch.setattr(model_runtime_readiness.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(model_runtime_readiness.shutil, "which", lambda _: "/fixture/ffmpeg")
    settings = {"models": {"first_person": str(weights), "third_person": str(weights)},
                "performance": {"tensor_rt": mode}}
    ready, blocker = model_runtime_readiness.local_preflight(settings)
    assert ready is False and "TensorRT" in blocker


def test_shutdown_observation_does_not_create_state(tmp_path):
    root = tmp_path / "absent"
    result = snapshot({"storage": {"local_runtime_root": str(root)}}, web_runs=[])
    assert result["provable"] and result["active_tasks"] == 0
    assert not root.exists()


def test_shutdown_observation_preserves_queued_work_and_background_admission(tmp_path):
    root = tmp_path / "runtime"
    (root / "device-day").mkdir(parents=True)
    path = root / "device-day/queue-vision.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE recordings(status TEXT)")
        db.executemany("INSERT INTO recordings VALUES (?)", [("queued",), ("running",), ("completed",)])
    settings = {"storage": {"local_runtime_root": str(root)}, "device_day": {"enabled": True}}
    result = snapshot(settings, web_runs=[{"state": "queued"}])
    assert result["provable"] and result["active_tasks"] >= 4
    assert result["background_admission_enabled"]


def test_unreadable_queue_cannot_prove_safe_shutdown(tmp_path):
    (tmp_path / "device-day").mkdir()
    (tmp_path / "device-day/queue-vision.sqlite3").write_text("not a database")
    result = snapshot({"storage": {"local_runtime_root": str(tmp_path)}}, web_runs=[])
    assert result["provable"] is False and result["active_tasks"] is None


def test_local_destination_rejects_an_arbitrarily_named_network_mount(monkeypatch, tmp_path):
    monkeypatch.setattr(local_storage, "mount_filesystem_type", lambda _: "cifs")
    with pytest.raises(RuntimeError, match="non-NAS filesystem"):
        local_storage.require_local_path(tmp_path / "innocent-name")


def test_operations_page_uses_analysis_readiness():
    source = (Path(__file__).parents[1] / "src/visioncortex/web/app.js").read_text()
    assert 'health.analysis_ready === true ? "预检通过" : "待配置"' in source
    assert 'esc(health.analysis_blocker ||' in source
