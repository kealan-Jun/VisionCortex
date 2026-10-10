"""Background startup checks from configuration and existing process state.

These checks never discover media, probe NAS directories or invoke models.
Startup readiness does not prove a recording has been processed or archived.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any

from .device_day_night_schedule import paused_stages
from .nas_recordings import enabled as directory_ingest_enabled


def configuration_digest(config: dict[str, Any]) -> str:
    # Saved AI-service selection is deliberately hot-swappable. It changes only
    # mllm; all prepared discovery/storage/processing settings remain bound.
    prepared = {key: value for key, value in config.items() if key != "mllm"}
    return hashlib.sha256(json.dumps(prepared, sort_keys=True, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def configuration_blockers(config: dict[str, Any], runtime_role: str) -> list[str]:
    """Validate the dedicated combined NAS automation service profile."""
    blockers = []
    if runtime_role != "combined":
        blockers.append("runtime_role_not_combined")
    if not (config.get("device_day") or {}).get("enabled"):
        blockers.append("device_day_disabled")
    if not directory_ingest_enabled(config):
        blockers.append("directory_ingest_disabled")
    if not (config.get("storage") or {}).get("sync_to_nas"):
        blockers.append("nas_archive_disabled")
    if (config.get("runtime") or {}).get("local_only"):
        blockers.append("local_only_runtime")
    return blockers


def snapshot(
    config: dict[str, Any],
    *,
    process_id: int,
    runtime_role: str,
    worker: dict[str, Any],
    monitor_alive: bool,
    monitor_status: str,
    dispatcher_alive: bool,
    runner_initialized: bool,
    dispatcher_status: str,
    storage_maintenance: bool,
    configuration: dict[str, Any],
    expected_configuration: dict[str, Any] | None = None,
    monitor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = configuration_blockers(config, runtime_role)
    if not configuration:
        blockers.append("startup_configuration_missing")
    elif expected_configuration is not None and configuration != expected_configuration:
        blockers.append("configuration_changed")
    if storage_maintenance:
        blockers.append("storage_maintenance")
    if worker.get("status") != "running":
        blockers.append("worker_not_running")
    if worker.get("pid") != process_id:
        blockers.append("worker_pid_mismatch")
    if not monitor_alive:
        blockers.append("nas_monitor_not_running")
    if monitor_status != "watching":
        blockers.append("nas_monitor_not_watching")
    metadata = monitor_metadata or {}
    monitor = metadata.get("monitor") or {}
    try:
        observed = datetime.fromisoformat(monitor["observed_at"])
        if observed.tzinfo is None:
            raise ValueError("Monitor timestamp requires a timezone")
        age = (datetime.now(timezone.utc) - observed).total_seconds()
        interval = float(monitor.get("poll_seconds", 30))
        if not math.isfinite(interval) or interval <= 0 or not -5 <= age <= max(15, interval * 3):
            raise ValueError("Monitor snapshot is stale")
    except (KeyError, ValueError, TypeError, OverflowError):
        blockers.append("nas_monitor_snapshot_stale")
    if metadata.get("errors"):
        blockers.append("nas_discovery_errors")
    if metadata.get("truncated"):
        blockers.append("nas_discovery_incomplete")
    lanes = metadata.get("discovery_lanes") or []
    if any(lane.get("thread_alive") is not True or
           (lane.get("status") != "watching" and not (
               lane.get("status") == "scanning" and lane.get("initial_scan_completed") is True))
           for lane in lanes):
        blockers.append("nas_discovery_lane_unavailable")
    if not dispatcher_alive:
        blockers.append("device_day_not_running")
    if not runner_initialized:
        blockers.append("device_day_not_initialized")
    if dispatcher_status not in {"waiting_for_nas_monitor", "running"}:
        blockers.append("device_day_dispatcher_unavailable")
    ready = not blockers
    return {
        "schema_version": "visioncortex-automation-readiness/1",
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "pid": process_id,
        "runtime_role": runtime_role,
        "configuration": configuration,
        "storage_maintenance": storage_maintenance,
        "worker": worker,
        "nas_monitor": {"thread_alive": monitor_alive, "status": monitor_status},
        "device_day": {
            "thread_alive": dispatcher_alive,
            "runner_initialized": runner_initialized,
            "status": dispatcher_status,
            "paused_stages": sorted(paused_stages(config)),
        },
        "blockers": blockers,
    }
