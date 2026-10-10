"""Background readiness uses configuration and local status, never NAS probes."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from visioncortex import api, runtime_process
from visioncortex.automation_readiness import (
    configuration_blockers, configuration_digest, snapshot,
)


@pytest.fixture
def automation_config(tmp_path):
    return {
        "runtime": {"role": "combined", "local_only": False},
        "device_day": {"enabled": True, "paused_stages": []},
        "collection_ingest": {
            "enabled": True,
            "mode": "directory_metadata",
            "source_root": "/unmounted-readiness-fixture-nas/capture",
        },
        "storage": {
            "sync_to_nas": True,
            "archive_root": "/unmounted-readiness-fixture-nas/archive",
            "local_runtime_root": str(tmp_path / "runtime"),
        },
    }


def healthy_monitor():
    return {
        "monitor": {
            "status": "watching",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "poll_seconds": 5,
        },
        "errors": [], "discovery_lanes": [], "truncated": False,
        "recording_count": 0, "recordings": [],
    }


def configuration_identity(config):
    return {
        "config_path": "/prepared/Customer.yaml",
        "default_config_path": "/prepared/default.yaml",
        "settings_sha256": configuration_digest(config),
    }


def ready_snapshot(config, **changes):
    arguments = {
        "process_id": 123,
        "runtime_role": "combined",
        "worker": {"status": "running", "pid": 123, "role": "combined"},
        "monitor_alive": True,
        "monitor_status": "watching",
        "dispatcher_alive": True,
        "runner_initialized": True,
        "dispatcher_status": "waiting_for_nas_monitor",
        "storage_maintenance": False,
        "configuration": configuration_identity(config),
        "expected_configuration": configuration_identity(config),
        "monitor_metadata": healthy_monitor(),
    }
    arguments.update(changes)
    return snapshot(config, **arguments)


def test_configuration_accepts_prepared_automatic_service(automation_config):
    original = deepcopy(automation_config)
    assert configuration_blockers(automation_config, "combined") == []
    assert automation_config == original


@pytest.mark.parametrize(
    "section,key,value,blocker",
    [
        ("device_day", "enabled", False, "device_day_disabled"),
        ("collection_ingest", "enabled", False, "directory_ingest_disabled"),
        ("collection_ingest", "mode", "indexed_staging", "directory_ingest_disabled"),
        ("storage", "sync_to_nas", False, "nas_archive_disabled"),
        ("runtime", "local_only", True, "local_only_runtime"),
    ],
)
def test_configuration_rejects_wrong_prepared_profile(
    automation_config, section, key, value, blocker
):
    automation_config[section][key] = value
    original = deepcopy(automation_config)
    assert blocker in configuration_blockers(automation_config, "combined")
    result = ready_snapshot(automation_config)
    assert result["status"] == "not_ready" and result["ready"] is False
    assert blocker in result["blockers"]
    assert automation_config == original


@pytest.mark.parametrize("role", ["web", "worker"])
def test_configuration_rejects_service_without_combined_owner(automation_config, role):
    assert "runtime_role_not_combined" in configuration_blockers(automation_config, role)
    result = ready_snapshot(automation_config, runtime_role=role)
    assert result["ready"] is False
    assert "runtime_role_not_combined" in result["blockers"]


@pytest.mark.parametrize("status", ["running", "waiting_for_nas_monitor"])
def test_ready_snapshot_distinguishes_running_and_idle(automation_config, status):
    original = deepcopy(automation_config)
    result = ready_snapshot(automation_config, dispatcher_status=status)
    assert result["schema_version"] == "visioncortex-automation-readiness/1"
    assert result["ready"] is True and result["status"] == "ready"
    assert result["pid"] == 123 and result["runtime_role"] == "combined"
    assert result["configuration"] == configuration_identity(automation_config)
    assert result["nas_monitor"]["thread_alive"] is True
    assert result["nas_monitor"]["status"] == "watching"
    assert result["device_day"] == {
        "thread_alive": True,
        "runner_initialized": True,
        "status": status,
        "paused_stages": [],
    }
    assert result["blockers"] == []
    assert automation_config == original


@pytest.mark.parametrize(
    "changes,blocker",
    [
        ({"worker": {"status": "heartbeat_expired", "pid": 123}}, "worker_not_running"),
        ({"worker": {"status": "stopped", "pid": 123}}, "worker_not_running"),
        ({"worker": {"status": "not_started"}}, "worker_not_running"),
        ({"worker": {"status": "running", "pid": 456}}, "worker_pid_mismatch"),
        ({"monitor_alive": False}, "nas_monitor_not_running"),
        ({"monitor_status": "retrying"}, "nas_monitor_not_watching"),
        ({"monitor_status": "starting"}, "nas_monitor_not_watching"),
        ({"dispatcher_alive": False}, "device_day_not_running"),
        ({"runner_initialized": False}, "device_day_not_initialized"),
        ({"dispatcher_status": "failed"}, "device_day_dispatcher_unavailable"),
        ({"dispatcher_status": "not_started"}, "device_day_dispatcher_unavailable"),
        ({"dispatcher_status": "waiting_for_storage"}, "device_day_dispatcher_unavailable"),
        ({"storage_maintenance": True}, "storage_maintenance"),
    ],
)
def test_snapshot_rejects_unavailable_background_component(automation_config, changes, blocker):
    result = ready_snapshot(automation_config, **changes)
    assert result["status"] == "not_ready" and result["ready"] is False
    assert blocker in result["blockers"]


def test_snapshot_preserves_deliberate_stage_pause(automation_config):
    automation_config["device_day"]["paused_stages"] = ["report", "understanding"]
    result = ready_snapshot(automation_config)
    assert result["ready"] is True
    assert result["device_day"]["paused_stages"] == ["report", "understanding"]


def test_snapshot_reports_effective_preprocessing_policy_without_mutation(automation_config):
    automation_config["device_day"]["preprocessing_only"] = True
    original = deepcopy(automation_config)
    result = ready_snapshot(automation_config)
    assert result["ready"] is True
    assert result["device_day"]["paused_stages"] == ["report", "stt", "understanding"]
    assert automation_config == original


@pytest.fixture
def automation_http(monkeypatch, automation_config):
    class StatusThread:
        def __init__(self, alive=True):
            self.alive = alive

        def is_alive(self):
            return self.alive

    def forbidden(*args, **kwargs):
        pytest.fail("Automation readiness must not scan storage or start processing")

    original_is_dir, original_stat = Path.is_dir, Path.stat

    def guard_nas_is_dir(path):
        if str(path).startswith("/unmounted-readiness-fixture-nas"):
            forbidden()
        return original_is_dir(path)

    def guard_nas_stat(path, *args, **kwargs):
        if str(path).startswith("/unmounted-readiness-fixture-nas"):
            forbidden()
        return original_stat(path, *args, **kwargs)

    monitor = StatusThread()
    dispatcher = StatusThread()
    service = SimpleNamespace(
        thread=dispatcher,
        _runner=object(),
        last_result={"status": "waiting_for_nas_monitor"},
        start=forbidden,
    )
    worker = {"status": "running", "pid": os.getpid(), "role": "combined"}
    monitor_metadata = healthy_monitor()
    monkeypatch.setattr(api, "_settings", lambda: automation_config)
    monkeypatch.setattr(api, "_nas_monitor_thread", monitor)
    monkeypatch.setattr(api, "_nas_monitor_snapshot", monitor_metadata)
    monkeypatch.setattr(api, "_device_day_service", service)
    monkeypatch.setattr(api, "_start_nas_monitor", forbidden)
    monkeypatch.setattr(api, "_start_queue_worker", forbidden)
    monkeypatch.setattr(runtime_process, "worker_status", lambda config: worker)
    monkeypatch.setattr(api, "scan_recordings", forbidden)
    monkeypatch.setattr(api, "_nas_storage_available", forbidden)
    monkeypatch.setattr(Path, "is_dir", guard_nas_is_dir)
    monkeypatch.setattr(Path, "stat", guard_nas_stat)
    monkeypatch.setenv("VISIONCORTEX_WEB_ACCESS_MODE", "local")
    monkeypatch.setenv("VISIONCORTEX_STORAGE_MAINTENANCE", "0")
    monkeypatch.delenv("VISIONCORTEX_RUNTIME_ROLE", raising=False)
    monkeypatch.setenv(api.CONFIG_ENV, "/prepared/Customer.yaml")
    startup = api._automation_configuration_identity(automation_config)
    monkeypatch.setattr(api, "_automation_startup_configuration", deepcopy(startup))
    return SimpleNamespace(
        client=TestClient(api.app), monitor=monitor, dispatcher=dispatcher,
        service=service, worker=worker, config=automation_config,
        monitor_metadata=monitor_metadata, startup=startup,
    )


def test_http_readiness_returns_local_snapshot_without_starting_lifespan(automation_http):
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["ready"] is True
    assert result["pid"] == os.getpid()
    assert result["configuration"]["config_path"] == "/prepared/Customer.yaml"
    assert result["configuration"]["default_config_path"]
    assert result["configuration"]["settings_sha256"] == configuration_digest(automation_http.config)
    assert result["build"]
    assert result["nas_monitor"]["thread_alive"] is True
    assert result["device_day"]["runner_initialized"] is True


@pytest.mark.parametrize("component", ["monitor", "dispatcher", "runner", "heartbeat"])
def test_http_readiness_reports_failed_runtime_without_storage_probe(automation_http, component):
    if component == "monitor":
        automation_http.monitor.alive = False
    elif component == "dispatcher":
        automation_http.dispatcher.alive = False
    elif component == "runner":
        automation_http.service._runner = None
    else:
        automation_http.worker["status"] = "heartbeat_expired"
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["blockers"]


def test_http_maintenance_exposes_readiness_json_with_processing_blocked(
    automation_http, monkeypatch
):
    monkeypatch.setenv("VISIONCORTEX_STORAGE_MAINTENANCE", "1")
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["ready"] is False
    assert result["storage_maintenance"] is True
    assert "storage_maintenance" in result["blockers"]


def test_http_responds_with_blockers_when_monitor_retries(automation_http, monkeypatch):
    monkeypatch.setattr(api, "_nas_monitor_snapshot", {"monitor": {"status": "retrying"}})
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert "nas_monitor_not_watching" in response.json()["blockers"]


def test_http_absent_threads_do_not_claim_ready(automation_http, monkeypatch):
    monkeypatch.setattr(api, "_nas_monitor_thread", None)
    automation_http.service.thread = None
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert {"nas_monitor_not_running", "device_day_not_running"} <= set(
        response.json()["blockers"]
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("collection_ingest", "source_root", "/unmounted-readiness-fixture-nas/new-capture"),
        ("storage", "archive_root", "/unmounted-readiness-fixture-nas/new-archive"),
        ("device_day", "paused_stages", ["report"]),
    ],
)
def test_same_configuration_path_does_not_hide_changed_settings(
    automation_config, section, key, value
):
    startup = configuration_identity(automation_config)
    automation_config[section][key] = value
    result = ready_snapshot(automation_config, configuration=startup)
    assert result["ready"] is False
    assert "configuration_changed" in result["blockers"]
    assert result["configuration"] == startup


def test_dynamic_ai_selection_preserves_prepared_configuration_identity(automation_config):
    startup = configuration_identity(automation_config)
    automation_config["mllm"] = {
        "provider": "test-provider", "model": "test-model", "credential_ref": "test-reference",
    }
    original = deepcopy(automation_config)
    assert configuration_identity(automation_config) == startup
    result = ready_snapshot(automation_config, configuration=startup)
    assert result["ready"] is True
    assert automation_config == original


def test_missing_startup_configuration_is_not_ready(automation_config):
    result = ready_snapshot(automation_config, configuration={})
    assert result["ready"] is False
    assert "startup_configuration_missing" in result["blockers"]


@pytest.mark.parametrize("timestamp", [None, "not-a-time", "2026-09-01T12:00:00"])
def test_monitor_timestamp_must_be_valid_and_timezone_aware(automation_config, timestamp):
    metadata = healthy_monitor()
    if timestamp is None:
        del metadata["monitor"]["observed_at"]
    else:
        metadata["monitor"]["observed_at"] = timestamp
    result = ready_snapshot(automation_config, monitor_metadata=metadata)
    assert result["ready"] is False
    assert "nas_monitor_snapshot_stale" in result["blockers"]


def test_live_thread_with_old_success_snapshot_is_not_ready(automation_config):
    metadata = healthy_monitor()
    metadata["monitor"]["observed_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=61)
    ).isoformat()
    result = ready_snapshot(automation_config, monitor_metadata=metadata)
    assert result["ready"] is False
    assert "nas_monitor_snapshot_stale" in result["blockers"]


def test_monitor_freshness_respects_prepared_poll_interval(automation_config):
    metadata = healthy_monitor()
    metadata["monitor"]["poll_seconds"] = 30
    metadata["monitor"]["observed_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=60)
    ).isoformat()
    assert ready_snapshot(automation_config, monitor_metadata=metadata)["ready"] is True


@pytest.mark.parametrize(
    "metadata_change,blocker",
    [
        ({"errors": [{"path": "test-camera", "message": "cannot read"}]}, "nas_discovery_errors"),
        ({"truncated": True}, "nas_discovery_incomplete"),
    ],
)
def test_watching_parent_does_not_hide_discovery_failure(
    automation_config, metadata_change, blocker
):
    metadata = healthy_monitor() | metadata_change
    result = ready_snapshot(automation_config, monitor_metadata=metadata)
    assert result["ready"] is False
    assert blocker in result["blockers"]


@pytest.mark.parametrize("status", ["failed", "retrying", "slow_or_unavailable", "starting"])
def test_watching_parent_does_not_hide_failed_camera_lane(automation_config, status):
    metadata = healthy_monitor()
    metadata["discovery_lanes"] = [{
        "camera_key": "test-camera", "mode": "live", "status": status,
        "thread_alive": True, "initial_scan_completed": True,
    }]
    result = ready_snapshot(automation_config, monitor_metadata=metadata)
    assert result["ready"] is False
    assert "nas_discovery_lane_unavailable" in result["blockers"]


def test_dead_camera_lane_does_not_inherit_old_success(automation_config):
    metadata = healthy_monitor()
    metadata["discovery_lanes"] = [{
        "status": "watching", "thread_alive": False, "initial_scan_completed": True,
    }]
    result = ready_snapshot(automation_config, monitor_metadata=metadata)
    assert result["ready"] is False
    assert "nas_discovery_lane_unavailable" in result["blockers"]


@pytest.mark.parametrize("completed", [False, True])
def test_camera_rescan_requires_initial_success(automation_config, completed):
    metadata = healthy_monitor()
    metadata["discovery_lanes"] = [{
        "status": "scanning", "thread_alive": True, "initial_scan_completed": completed,
    }]
    result = ready_snapshot(automation_config, monitor_metadata=metadata)
    assert result["ready"] is completed
    assert ("nas_discovery_lane_unavailable" in result["blockers"]) is (not completed)


def test_zero_recordings_can_wait_with_background_ready(automation_config):
    metadata = healthy_monitor()
    assert metadata["recording_count"] == 0
    assert metadata["recordings"] == []
    assert ready_snapshot(automation_config, monitor_metadata=metadata)["ready"] is True


def test_http_same_path_changed_configuration_requires_restart(automation_http):
    automation_http.config["collection_ingest"]["source_root"] += "/new"
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    result = response.json()
    assert result["ready"] is False
    assert "configuration_changed" in result["blockers"]
    assert result["configuration"] == automation_http.startup


def test_http_ai_selection_can_change_without_false_restart_requirement(automation_http):
    automation_http.config["mllm"] = {"provider": "test-provider", "model": "test-model"}
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["configuration"] == automation_http.startup


def test_http_no_startup_configuration_is_not_ready(automation_http, monkeypatch):
    monkeypatch.setattr(api, "_automation_startup_configuration", {})
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert "startup_configuration_missing" in response.json()["blockers"]


@pytest.mark.parametrize("failure", ["stale", "lanes", "errors", "truncated", "initial_scan"])
def test_http_parent_watching_cannot_hide_incomplete_discovery(automation_http, failure):
    metadata = automation_http.monitor_metadata
    if failure == "stale":
        metadata["monitor"]["observed_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=61)
        ).isoformat()
    elif failure == "lanes":
        metadata["discovery_lanes"] = [{"status": "failed", "thread_alive": False}]
    elif failure == "errors":
        metadata["errors"] = [{"path": "test-camera", "message": "read unavailable"}]
    elif failure == "truncated":
        metadata["truncated"] = True
    else:
        metadata["discovery_lanes"] = [{
            "status": "scanning", "thread_alive": True, "initial_scan_completed": False,
        }]
    response = automation_http.client.get("/health/automation")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["blockers"]
