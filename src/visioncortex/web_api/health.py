"""HTTP adapters for health"""

from __future__ import annotations
from ..application.dependencies import ports
import os
from pathlib import Path
from typing import Any
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import Response
from ..provider_connection import connection_health
from ..provider_credentials import key_configured
from .. import ai_settings
from .. import speech
from ..identity import PRODUCT_NAME
from ..model_certification import audit_production_model_certification
from ..model_runtime_readiness import local_preflight
from ..nas_recordings import (
    enabled as directory_ingest_enabled,
)

from ..application import archive_read_model, runtime_host, upload_service

from fastapi import APIRouter

router = APIRouter()
_web_root = Path(__file__).parents[1] / "web"


def _nas_storage_available(settings: dict[str, Any]) -> bool:
    """A readiness probe must never create replacement storage directories."""

    try:
        return all(
            Path(settings["storage"][key]).is_dir()
            and os.access(settings["storage"][key], os.W_OK | os.X_OK)
            for key in ("archive_root", "local_cache_root")
        )
    except OSError:
        return False


def index() -> str:
    return (_web_root / "index.html").read_text(encoding="utf-8")


def favicon() -> Response:
    return Response(status_code=204)


def health() -> dict[str, Any]:
    settings = runtime_host._settings()
    queue_stats = (
        runtime_host._persistent_queue.stats()
        if runtime_host._persistent_queue is not None
        else None
    )
    upload_stats = (
        runtime_host._upload_sessions.stats()
        if runtime_host._upload_sessions is not None
        else None
    )
    upload_policy = upload_service._upload_policy(settings)
    storage_mode = "nas" if settings["storage"].get("sync_to_nas") else "local"
    archive_root = archive_read_model._archive_root(settings)
    input_mode = (
        "NAS 采集批次 / zero-copy virtual timeline"
        if storage_mode == "nas"
        else "local files / zero-copy source references"
    )
    if settings.get("project", {}).get("site_configuration_required"):
        analysis_ready = False
        analysis_blocker = "尚未提供已准备的私有站点配置"
    elif settings.get("project", {}).get("run_purpose") == "demo":
        analysis_ready = False
        analysis_blocker = "当前为本地演示模式，尚未配置真实分析环境"
    else:
        analysis_ready, analysis_blocker = local_preflight(settings) if storage_mode == "local" else (True, None)
        if storage_mode == "nas":
            try:
                audit_production_model_certification(settings)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                analysis_ready = False
                analysis_blocker = "分析模型待质量验收"
        if storage_mode == "nas" and not _nas_storage_available(settings):
            analysis_ready = False
            analysis_blocker = "NAS 归档或缓存目录不可用"
    with runtime_host._nas_monitor_lock:
        monitor_snapshot = runtime_host._nas_monitor_snapshot or {}
        monitor = dict(monitor_snapshot.get("monitor") or {})
    from ..runtime_shutdown import snapshot as shutdown_snapshot
    with runtime_host._lock:
        web_runs = [dict(run) for run in runtime_host._runs.values()]
    shutdown_safety = shutdown_snapshot(
        settings, web_runs=web_runs, gpu_busy=runtime_host._gpu_job_lock.locked()
    )
    if queue_stats is not None and shutdown_safety["provable"]:
        shutdown_safety["active_tasks"] += queue_stats.get("queued", 0) + queue_stats.get("running", 0)
    return {
        "status": "ok",
        "product_name": PRODUCT_NAME,
        "analysis_ready": analysis_ready,
        "analysis_readiness": {
            "preflight": "PARTIAL_EVIDENCE" if analysis_ready else "NOT_PROVEN",
            "actual_model_invocation": "NOT_PROVEN",
        },
        "speech_recognition": {"enabled": speech.enabled(settings)},
        "run_purpose": settings.get("project", {}).get("run_purpose", "production"),
        "analysis_blocker": analysis_blocker,
        "shutdown_safety": shutdown_safety,
        "storage_mode": storage_mode,
        "archive_label": "NAS 正式归档" if storage_mode == "nas" else "任务归档",
        "web_upload_retention_mode": settings["storage"].get(
            "web_upload_retention_mode", "local_and_nas"
        ),
        "capacity_policy": "dynamic_by_submitted_files",
        "view_count_policy": "dynamic",
        "minimum_cross_view_sources": 2,
        "max_concurrent_archive_streams": runtime_host._ARCHIVE_STREAM_LIMIT,
        "web_access": ports.validate_web_access_configuration(),
        "large_uploads": {
            "protocol": "resumable_chunks_v2",
            "compatible_protocols": ["resumable_chunks_v1"],
            "fixed_total_size_limit": False,
            "chunk_size_bytes": int(upload_policy["chunk_size_bytes"]),
            "parallel_files": int(upload_policy["parallel_files"]),
            "chunk_sha256_required": bool(upload_policy["chunk_sha256_required"]),
            "large_file_identity": "persistent_verified_chunk_tree",
            "full_file_sha256_max_bytes": int(
                upload_policy["full_file_sha256_max_bytes"]
            ),
            "storage_admission": "dynamic_free_space_with_reservation",
            "original_media_copies": 1,
            "inactive_session_ttl_seconds": float(upload_policy["session_ttl_seconds"]),
            "sessions": upload_stats,
        },
        "ark_key_configured": key_configured(settings["mllm"]),
        "mllm_key_configured": key_configured(settings["mllm"]),
        "mllm_connection": connection_health(
            settings["mllm"], key_configured(settings["mllm"])
        ),
        "ai_settings_enabled": ai_settings.enabled(),
        "mllm_enabled": bool(settings["mllm"].get("enabled")),
        "model": settings["mllm"]["model"],
        "archive_root": str(archive_root),
        "archive_available": archive_root.is_dir(),
        # Compatibility aliases for existing production Web clients. In local
        # mode the canonical archive_* fields above carry the accurate label.
        "nas_archive_root": str(archive_root),
        "nas_available": storage_mode == "nas" and archive_root.is_dir(),
        "fixed_benchmark": {
            **runtime_host.benchmark_registration(settings),
            "index_csv": str(settings["storage"]["index_csv"]),
            "input_mode": input_mode,
            "local_runtime_root": str(settings["storage"]["local_runtime_root"]),
            "local_cache_root": str(settings["storage"]["local_cache_root"]),
        },
        "collection_ingest": {
            "enabled": bool(
                (settings.get("collection_ingest") or {}).get("enabled", True)
            ),
            "mode": (
                "directory_metadata"
                if directory_ingest_enabled(settings)
                else "index_metadata_poll"
            ),
            "poll_seconds": float(
                (settings.get("collection_ingest") or {}).get("poll_seconds", 30.0)
            ),
            "recursive_nas_scan": False,
            "camera_directories": list(
                monitor_snapshot.get("camera_directories")
                or (settings.get("collection_ingest") or {}).get(
                    "camera_directories", []
                )
            ),
            "camera_directory_count": int(
                monitor_snapshot.get("camera_directory_count") or 0
            ),
            "unconfigured_camera_directories": list(
                monitor_snapshot.get("unconfigured_camera_directories") or []
            ),
            "monitor_status": monitor.get("status", "starting"),
            "monitor_observed_at": monitor.get("observed_at"),
            "monitor_consecutive_failures": int(
                monitor.get("consecutive_failures") or 0
            ),
            "index_csv": str(settings["storage"]["index_csv"]),
            "device_registry_path": settings["storage"].get("device_registry_path"),
        },
        "execution_queue": {
            "policy": "single_gpu_one_job_at_a_time",
            "persistence": "sqlite"
            if runtime_host._persistent_queue is not None
            else "not_initialized",
            "survives_web_service_restart": runtime_host._persistent_queue is not None,
            "archive_receipt_disaster_recovery": True,
            "database": (
                str(runtime_host._persistent_queue.database)
                if runtime_host._persistent_queue is not None
                else None
            ),
            "counts": queue_stats,
            "gpu_busy": bool(
                runtime_host._gpu_job_lock.locked()
                or (queue_stats is not None and queue_stats["running"] > 0)
            ),
        },
    }


def liveness():
    from ..build_identity import identity

    return {
        "status": "alive",
        "storage_maintenance": runtime_host._storage_maintenance(),
        "build": identity(),
    }


def automation_readiness() -> JSONResponse:
    """Observe background startup without scanning NAS or starting inference."""
    from ..automation_readiness import snapshot
    from ..build_identity import identity
    from ..runtime_process import role, worker_status

    settings = runtime_host._settings()
    with runtime_host._nas_monitor_lock:
        monitor_metadata = runtime_host._nas_monitor_snapshot or {}
        monitor_status = (monitor_metadata.get("monitor") or {}).get(
            "status", "starting"
        )
    monitor_thread = runtime_host._nas_monitor_thread
    dispatcher_thread = runtime_host._device_day_service.thread
    result = snapshot(
        settings,
        process_id=os.getpid(),
        runtime_role=role(settings),
        worker=worker_status(settings),
        monitor_alive=bool(monitor_thread is not None and monitor_thread.is_alive()),
        monitor_status=monitor_status,
        dispatcher_alive=bool(
            dispatcher_thread is not None and dispatcher_thread.is_alive()
        ),
        runner_initialized=runtime_host._device_day_service._runner is not None,
        dispatcher_status=runtime_host._device_day_service.last_result.get(
            "status", "not_started"
        ),
        storage_maintenance=runtime_host._storage_maintenance(),
        configuration=runtime_host._automation_startup_configuration or {},
        expected_configuration=runtime_host._automation_configuration_identity(
            settings
        ),
        monitor_metadata=monitor_metadata,
    )
    return JSONResponse(
        result | {"build": identity()}, headers={"Cache-Control": "no-store"}
    )


def runtime_status():
    from ..runtime_process import role, worker_status
    from ..runtime_control import ResourceCoordinator
    from ..build_identity import identity

    settings = runtime_host._settings()
    database = (
        Path(settings["storage"]["local_runtime_root"]) / "state" / "resources.sqlite3"
    )
    resources = ResourceCoordinator(database).snapshot() if database.is_file() else []
    return {
        "build": identity(),
        "role": role(settings),
        "worker": worker_status(settings),
        "resources": resources,
        "queue": runtime_host._persistent_queue.stats()
        if runtime_host._persistent_queue
        else {},
    }


router.get("/", response_class=HTMLResponse)(index)

router.get("/favicon.ico", include_in_schema=False)(favicon)

router.get("/api/health")(health)

router.get("/health/live")(liveness)

router.get("/health/automation")(automation_readiness)

router.get("/api/runtime")(runtime_status)
