from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import yaml
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .run_insights import hardware_summary, timing_summary, token_summary, with_operation_refreshes
from .archive_catalog import (
    archive_catalog_path,
    catalog_archive_release,
    ensure_archive_catalog,
    lightweight_release_integrity,
    list_catalog_archives,
    search_catalog_events,
)
from .collection_catalog import discover_collections, get_collection
from .collection_state import record_collection_state
from .annotation_workspace import (
    export_reviewed_ground_truth,
    load_annotation_workspace,
    record_annotation_decision,
    resolve_annotation_image,
)
from .config import load_config
from .provider_connection import connection_health
from .provider_credentials import key_configured
from . import ai_settings
from . import speech, speech_worker
from .identity import CONFIG_ENV, PRODUCT_NAME
from .device_registry import load_device_registry, resolve_view_role
from .device_day_service import DeviceDayService, install_routes as install_device_day_routes
from .knowledge_api import install_routes as install_knowledge_routes
from .input_preflight import preflight_manifest_inputs
from .input_seal import build_input_seal, verify_input_seal, write_input_seal
from .model_certification import audit_production_model_certification
from .media_preview import cached_video_poster
from .nas_recordings import (
    create_selection,
    enabled as directory_ingest_enabled,
    scan_recordings,
    selection_path,
    validate_selection,
)
from .indexing import (
    INDEX_DB_NAME,
    INDEX_MANIFEST_NAME,
    get_indexed_event,
    get_indexed_evidence,
    search_archive_index,
    search_physical_changes,
)
from .pagination import decode_cursor, encode_cursor
from .pathing import archive_contains, archive_relative_posix
from .pipeline import EvidencePipeline
from .partial_delivery import (
    component_results, partial_result_available, prioritize_retained_candidates,
    registered_partial_root, write_partial_delivery,
)
from .run_queue import DurableRunQueue, QueuedRunJob
from .schemas import RunManifest, VideoSegmentInput, ViewInput
from .storage import (
    ARCHIVE_DIRECTORIES,
    archive_promotion_in_progress,
    fixed_archive_staging_paths,
    initialize_nas_archive,
    prepare_from_nas_index,
    promote_fixed_archive,
    run_staging_roots,
    read_current_release_pointer,
    safe_archive_name,
)
from .build_identity import identity as build_identity
from .upload_sessions import StorageReservationError, UploadSessionStore
from .web_access import (
    append_access_audit,
    authenticate_basic_authorization,
    is_allowed_lan_client,
    validate_web_access_configuration,
    web_access_mode,
    web_https_required,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    if _ARCHIVE_STREAM_LIMIT_ERROR:
        raise RuntimeError(_ARCHIVE_STREAM_LIMIT_ERROR)
    validate_web_access_configuration()
    if _storage_maintenance():
        # Explicit deployment hold: do not recover receipts, discover media or
        # start any producer while the NAS is being repaired.
        yield
        return
    settings = _settings()
    _initialize_persistent_queue(settings)
    from .runtime_process import role, worker_owner
    if role(settings) == 'web':
        yield
        return
    with worker_owner(settings):
        _expire_stale_upload_sessions(settings)
        try:
            _recover_jobs_from_archive_receipts(settings)
            _recover_orphaned_tasks()
            _start_queue_worker()
            _start_nas_monitor(settings)
            _device_day_service.start()
            yield
        finally:
            _device_day_service.stop()
            _stop_nas_monitor()
            _stop_queue_worker()
            from .shared_inference import close_pools
            close_pools()


app = FastAPI(
    title=PRODUCT_NAME,
    version=build_identity()["version"],
    lifespan=_lifespan,
)
_web_root = Path(__file__).with_name("web")
app.mount("/ui", StaticFiles(directory=_web_root), name="ui")
_lock = threading.Lock()
_gpu_job_lock = threading.Lock()
_upload_finalize_lock = threading.Lock()
_nas_batch_submission_lock = threading.Lock()
_web_access_audit_lock = threading.Lock()
try:
    _ARCHIVE_STREAM_LIMIT = max(
        1, int(os.getenv("VISIONCORTEX_WEB_MAX_CONCURRENT_ARCHIVE_STREAMS", "8"))
    )
except ValueError:
    _ARCHIVE_STREAM_LIMIT = 8
    _ARCHIVE_STREAM_LIMIT_ERROR = (
        "VISIONCORTEX_WEB_MAX_CONCURRENT_ARCHIVE_STREAMS must be an integer"
    )
else:
    _ARCHIVE_STREAM_LIMIT_ERROR = None
_archive_stream_slots = threading.BoundedSemaphore(_ARCHIVE_STREAM_LIMIT)
_runs: dict[str, dict[str, Any]] = {}
_persistent_queue: DurableRunQueue | None = None
_upload_sessions: UploadSessionStore | None = None
_queue_thread: threading.Thread | None = None
_queue_stop = threading.Event()
_queue_wakeup = threading.Event()
_nas_monitor_lock = threading.Lock()
_nas_monitor_stop = threading.Event()
_nas_monitor_thread: threading.Thread | None = None
_nas_monitor_snapshot: dict[str, Any] | None = None
_queue_worker_id = f"web-{os.getpid()}-{uuid.uuid4().hex[:8]}"
_QUEUE_LEASE_SECONDS = 60.0
_QUEUE_LEASE_RENEW_SECONDS = 15.0
_QUEUE_POLL_SECONDS = 1.0
_BENCHMARK_EXPERIMENT_ID = "exp_20260810_144014_e918b762"
_BENCHMARK_ARCHIVE_NAME = (
    "CustomFlow_standard_correct_12_ABCFA_0001--exp_20260810_144014_e918b762"
)
_BENCHMARK_SUBMISSION_PROTOCOL_VERSION = 1
_PHYSICAL_ACTION_TYPES = (
    "hand_object_contact",
    "object_movement",
    "liquid_transfer",
    "container_state_change",
    "device_panel_operation",
)
_RUNTIME_HEARTBEAT_STALE_SECONDS = 90.0
_RUNTIME_HEARTBEAT_FILES = (
    "resource_telemetry_live.json",
    "source_progress.json",
    "run_metrics_live.json",
)


def _nas_monitor_receipt_path(settings: dict[str, Any]) -> Path:
    return (
        Path(settings["storage"]["local_runtime_root"])
        / "state"
        / "nas-recording-monitor.json"
    )


def _publish_nas_monitor_snapshot(
    settings: dict[str, Any], payload: dict[str, Any]
) -> None:
    global _nas_monitor_snapshot
    with _nas_monitor_lock:
        _nas_monitor_snapshot = payload
    path = _nas_monitor_receipt_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _nas_monitor_loop(settings: dict[str, Any]) -> None:
    failures = 0
    last_success: dict[str, Any] | None = None
    interval = max(
        5.0,
        float((settings.get("collection_ingest") or {}).get("poll_seconds", 30)),
    )
    camera_monitor = None
    if (settings.get("device_day") or {}).get("camera_lanes"):
        from .device_day_monitor import CameraMonitor
        camera_monitor = CameraMonitor(settings, _nas_monitor_stop, _device_day_service.observe)
    while not _nas_monitor_stop.is_set():
        observed_at = datetime.now().astimezone().isoformat()
        try:
            if camera_monitor is not None:
                inventory = camera_monitor.poll()
            else:
                discovery = copy.deepcopy(settings)
                if (settings.get("device_day") or {}).get("enabled"):
                    discovery["collection_ingest"]["max_recordings_per_camera"] = 0
                    discovery["collection_ingest"]["capture_since_date"] = settings["device_day"].get("start_date")
                if (settings.get("device_day") or {}).get("enabled"):
                    # Release each uploaded slice to preprocessing as soon as it is
                    # inspected, without waiting for every camera in the NAS scan.
                    inventory = scan_recordings(discovery, on_record=lambda record:
                        _device_day_service.observe(settings, {"recordings": [record]}))
                else:
                    inventory = scan_recordings(discovery)
            if camera_monitor is None:
                _device_day_service.observe(settings, inventory)
            failures = 0
            last_success = inventory | {
                "monitor": {
                    "status": "watching",
                    "observed_at": observed_at,
                    "poll_seconds": interval,
                    "consecutive_failures": 0,
                }
            }
            _publish_nas_monitor_snapshot(settings, last_success)
        except (OSError, ValueError, TypeError) as exc:
            failures += 1
            degraded = dict(
                last_success
                or {
                    "mode": "directory_metadata",
                    "recordings": [],
                    "recording_count": 0,
                    "batches": [],
                    "errors": [],
                    "truncated": False,
                }
            )
            degraded["monitor"] = {
                "status": "retrying",
                "observed_at": observed_at,
                "poll_seconds": interval,
                "consecutive_failures": failures,
                "message": "NAS 暂时不可用，后台会继续重试。",
                "error_type": type(exc).__name__,
            }
            try:
                _publish_nas_monitor_snapshot(settings, degraded)
            except OSError:
                with _nas_monitor_lock:
                    global _nas_monitor_snapshot
                    _nas_monitor_snapshot = degraded
        _nas_monitor_stop.wait(interval)


def _start_nas_monitor(settings: dict[str, Any]) -> None:
    global _nas_monitor_snapshot, _nas_monitor_thread
    if not directory_ingest_enabled(settings):
        return
    if _nas_monitor_thread is not None and _nas_monitor_thread.is_alive():
        return
    with _nas_monitor_lock:
        _nas_monitor_snapshot = None
    _nas_monitor_stop.clear()
    _nas_monitor_thread = threading.Thread(
        target=_nas_monitor_loop,
        args=(settings,),
        name="nas-recording-monitor",
        daemon=True,
    )
    _nas_monitor_thread.start()


def _stop_nas_monitor() -> None:
    global _nas_monitor_thread
    _nas_monitor_stop.set()
    if _nas_monitor_thread is not None:
        _nas_monitor_thread.join(timeout=5)
    _nas_monitor_thread = None


def _runtime_activity_receipt(
    status_path: Path,
    *,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Describe recent pipeline activity without parsing a concurrently written JSON file."""

    observed_at = time.time() if now_epoch is None else float(now_epoch)
    latest_path: Path | None = None
    latest_mtime: float | None = None
    for file_name in _RUNTIME_HEARTBEAT_FILES:
        candidate = status_path.parent / file_name
        try:
            candidate_mtime = candidate.stat().st_mtime
        except (FileNotFoundError, OSError):
            continue
        if latest_mtime is None or candidate_mtime > latest_mtime:
            latest_path = candidate
            latest_mtime = candidate_mtime

    if latest_path is None or latest_mtime is None:
        return {
            "active": False,
            "latest_path": None,
            "latest_mtime_epoch": None,
            "age_seconds": None,
            "stale_after_seconds": _RUNTIME_HEARTBEAT_STALE_SECONDS,
        }

    age_seconds = max(0.0, observed_at - latest_mtime)
    return {
        "active": age_seconds <= _RUNTIME_HEARTBEAT_STALE_SECONDS,
        "latest_path": latest_path.name,
        "latest_mtime_epoch": latest_mtime,
        "age_seconds": age_seconds,
        "stale_after_seconds": _RUNTIME_HEARTBEAT_STALE_SECONDS,
    }


def _recover_orphaned_tasks() -> None:
    """Mark only stale durable tasks interrupted after a Web service restart."""

    root = _archive_root()
    if not root.is_dir():
        return
    status_paths = list(root.glob("*/JSON-Config-Files/pipeline_status.json"))
    settings = _settings()
    resolved_settings = {
        **settings,
        "storage": {**settings.get("storage", {}), "archive_root": str(root)},
    }
    for staging_root in run_staging_roots(resolved_settings):
        if staging_root.is_dir():
            status_paths.extend(
                staging_root.glob("*/*/JSON-Config-Files/pipeline_status.json")
            )
    for status_path in status_paths:
        payload = _read_json(status_path, {}) or {}
        # An unreadable NAS receipt is not an interrupted task. In particular,
        # never overwrite a read failure with a fabricated recovery receipt.
        if not isinstance(payload, dict) or not payload.get("stage"):
            continue
        stage = str(payload.get("stage") or "")
        if stage in {"completed", "partial", "failed"}:
            continue
        heartbeat = _runtime_activity_receipt(status_path)
        recovery = payload.get("recovery") if isinstance(payload.get("recovery"), dict) else {}
        if heartbeat["active"]:
            previous_stage = str(recovery.get("previous_stage") or "")
            if (
                stage == "interrupted"
                and recovery.get("status") == "orphaned_after_service_restart"
                and previous_stage
                and previous_stage not in {"completed", "partial", "failed", "interrupted"}
            ):
                payload.update(
                    {
                        "stage": previous_stage,
                        "message": "后台流水线心跳仍活跃；Web 服务重启未中断正在运行的任务。",
                        "updated_at": datetime.now().astimezone().isoformat(),
                        "recovery": {
                            "status": "active_after_service_restart",
                            "resumable": False,
                            "previous_stage": previous_stage,
                            "heartbeat": heartbeat,
                        },
                    }
                )
                _write_json_atomic(status_path, payload)
            continue
        if stage == "interrupted":
            continue
        payload.update(
            {
                "stage": "interrupted",
                "message": "服务重启时发现未完成任务；检测账本与模型结果缓存保留，可使用同一输入重新启动以续跑。",
                "updated_at": datetime.now().astimezone().isoformat(),
                "recovery": {
                    "status": "orphaned_after_service_restart",
                    "resumable": True,
                    "previous_stage": stage,
                    "heartbeat": heartbeat,
                },
            }
        )
        _write_json_atomic(status_path, payload)


def _storage_maintenance() -> bool:
    return os.environ.get("VISIONCORTEX_STORAGE_MAINTENANCE", "0") == "1"


@app.middleware("http")
async def enforce_web_access(request: Request, call_next):
    """Keep the default local service open and fail closed for LAN service mode."""

    if _storage_maintenance() and request.url.path not in {"/api/device-day-progress", "/health/live"}:
        message = "修复版代码已加载。NAS 正在维护，数据读取、提交与后台处理暂不开放；历史队列保留，未触发全量重跑。"
        headers = {"Retry-After": "60", "Cache-Control": "no-store"}
        if request.url.path.startswith("/api/"):
            return JSONResponse({"status": "storage_maintenance", "detail": message}, status_code=503, headers=headers)
        return HTMLResponse("<!doctype html><meta charset='utf-8'><title>VisionCortex 维护状态</title>"
                            "<main style='max-width:760px;margin:80px auto;font:20px sans-serif'>"
                            "<h1>VisionCortex · NAS 维护中</h1><p>" + message + "</p></main>",
                            status_code=503, headers=headers)
    started = time.perf_counter()
    identity: dict[str, str] | None = None
    mode = "unknown"

    def audited(response: Response) -> Response:
        if mode == "lan" and request.url.path.startswith("/api/"):
            try:
                settings = _settings()
                audit_path = (
                    Path(settings["storage"]["local_runtime_root"])
                    / "state"
                    / f"web_access_audit-{datetime.now().astimezone():%Y-%m-%d}.jsonl"
                )
                with _web_access_audit_lock:
                    append_access_audit(
                        audit_path,
                        {
                            "schema_version": "visioncortex-web-access-audit/1",
                            "observed_at": datetime.now().astimezone().isoformat(),
                            "username": (identity or {}).get("username") or "anonymous",
                            "role": (identity or {}).get("role"),
                            "client": request.client.host if request.client else None,
                            "method": request.method,
                            "path": request.url.path,
                            "status_code": response.status_code,
                            "duration_ms": round(
                                (time.perf_counter() - started) * 1000.0, 3
                            ),
                        },
                    )
            except (OSError, KeyError, RuntimeError, TypeError):
                pass
        return response

    async def dispatch():
        from .submission import handle, eligible
        if request.method != "POST" or not eligible(request.url.path):
            return await call_next(request)
        return await handle(request, call_next, _settings())

    try:
        mode = web_access_mode()
        if mode == "local":
            identity = {"username": "local", "role": "admin"}
            request.state.web_identity = identity
            return audited(await dispatch())
        client_host = request.client.host if request.client else None
        if not is_allowed_lan_client(client_host):
            return audited(Response(
                "VisionCortex LAN access is not allowed from this network.",
                status_code=403,
                headers={"Cache-Control": "no-store"},
            ))
        if web_https_required() and request.url.scheme != "https":
            return audited(Response(
                "VisionCortex LAN access requires HTTPS.",
                status_code=426,
                headers={"Cache-Control": "no-store"},
            ))
        identity = authenticate_basic_authorization(
            request.headers.get("authorization")
        )
        if identity is None:
            return audited(Response(
                "VisionCortex login required.",
                status_code=401,
                headers={
                    "WWW-Authenticate": 'Basic realm="VisionCortex 3090 Ti", charset="UTF-8"',
                    "Cache-Control": "no-store",
                },
            ))
        if identity["role"] == "viewer" and request.method not in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            return audited(Response(
                "VisionCortex viewer accounts are read-only.",
                status_code=403,
                headers={"Cache-Control": "no-store"},
            ))
        if identity["role"] != "admin" and (
            request.url.path.startswith("/api/annotation-workspace")
            or (
                request.method == "POST"
                and request.url.path.startswith("/api/archives/")
                and request.url.path.endswith("/open")
            )
        ):
            return audited(Response(
                "VisionCortex administrator access is required.",
                status_code=403,
                headers={"Cache-Control": "no-store"},
            ))
        request.state.web_identity = identity
    except RuntimeError:
        return audited(Response(
            "VisionCortex Web access configuration is invalid.",
            status_code=503,
            headers={"Cache-Control": "no-store"},
        ))
    return audited(await dispatch())


@app.middleware("http")
async def record_web_ingest_start(request: Request, call_next):
    """Capture the request boundary before multipart parsing starts."""

    if request.method == "POST" and request.url.path == "/api/runs":
        request.state.ingest_started_perf = time.perf_counter()
        request.state.ingest_started_epoch = time.time()
        request.state.ingest_started_at = datetime.now().astimezone().isoformat()
    return await call_next(request)


@app.middleware("http")
async def require_analysis_certification(request: Request, call_next):
    path = request.url.path
    starts_analysis = (
        path in {"/api/runs", "/api/runs/from-paths", "/api/upload-sessions"}
        or bool(re.fullmatch(r"/api/(collections|benchmarks)/[^/]+/runs", path))
        or bool(re.fullmatch(r"/api/nas-batches/[^/]+/runs", path))
        or bool(re.fullmatch(r"/api/upload-sessions/[^/]+/finalize", path))
        or bool(re.fullmatch(r"/api/runs/[^/]+/retry", path))
    )
    if request.method == "POST" and starts_analysis:
        settings = _settings()
        if settings.get("mllm", {}).get("credential_ref") and not key_configured(settings["mllm"]):
            return JSONResponse(status_code=409, content={"detail": "AI 服务配置需要重新验证，请打开 AI 服务设置。"})
        if settings["storage"].get("sync_to_nas"):
            try:
                audit_production_model_certification(settings)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "model_certification_required",
                            "message": "分析模型尚未完成质量验收，当前可浏览和整理 NAS 素材。",
                        }
                    },
                )
            if not _nas_storage_available(settings):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "nas_unavailable",
                            "message": "NAS 归档或缓存目录不可用，请检查连接后重试。",
                        }
                    },
                )
    return await call_next(request)


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


@app.middleware("http")
async def limit_archive_streams(request: Request, call_next):
    """Bound concurrent NAS-backed streams for predictable multi-user playback."""

    is_stream = request.url.path in {"/api/archive-file", "/api/staging-file"} or bool(re.fullmatch(r"/api/(staging-runs|archives)/[^/]+/speech-video", request.url.path))
    if not is_stream or request.query_params.get("poster") == "true":
        return await call_next(request)
    if not _archive_stream_slots.acquire(blocking=False):
        return Response(
            "VisionCortex archive streaming is busy; retry shortly.",
            status_code=429,
            headers={"Retry-After": "2", "Cache-Control": "no-store"},
        )
    try:
        response = await call_next(request)
    except BaseException:
        _archive_stream_slots.release()
        raise
    original_iterator = response.body_iterator

    async def guarded_body():
        try:
            async for chunk in original_iterator:
                yield chunk
        finally:
            _archive_stream_slots.release()

    response.body_iterator = guarded_body()
    return response


def _safe_file_name(value: str, fallback_stem: str = "file") -> str:
    original = Path(value).name
    suffix = Path(original).suffix
    stem = original[: -len(suffix)] if suffix else original
    ascii_stem = (
        unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    )
    ascii_suffix = (
        unicodedata.normalize("NFKD", suffix).encode("ascii", "ignore").decode("ascii")
    )
    cleaned_stem = (
        re.sub(r"[^A-Za-z0-9_.-]+", "-", ascii_stem).strip(" .-")
        or re.sub(r"[^A-Za-z0-9_.-]+", "-", fallback_stem).strip(" .-")
        or "file"
    )
    cleaned_suffix = re.sub(r"[^A-Za-z0-9.]", "", ascii_suffix).lower()
    return f"{cleaned_stem[:160]}{cleaned_suffix[:20]}"


def _settings() -> dict[str, Any]:
    configured = os.getenv(CONFIG_ENV)
    settings = load_config(Path(configured)) if configured else load_config()
    return ai_settings.apply_active(settings)


_device_day_service = DeviceDayService(_settings, _gpu_job_lock)
install_device_day_routes(app, lambda: _settings(), _device_day_service)
install_knowledge_routes(app, lambda: _settings())


def _archive_root(settings: dict[str, Any] | None = None) -> Path:
    return Path((settings or _settings())["storage"]["archive_root"])


def _archive_catalog_database(
    settings: dict[str, Any] | None = None,
    archive_root: Path | None = None,
) -> Path:
    effective_settings = settings or _settings()
    root = (archive_root or _archive_root(effective_settings)).resolve()
    storage = effective_settings.get("storage") or {}
    local_runtime = Path(storage.get("local_runtime_root") or (root / ".VisionCortex-Web-Runtime"))
    return archive_catalog_path(local_runtime, root)


def _ensure_archive_read_catalog(root: Path | None = None) -> Path:
    archive_root = (root or _archive_root()).resolve()
    database = _archive_catalog_database(archive_root=archive_root)
    ensure_archive_catalog(archive_root, database)
    return database


def _queue_database_path(settings: dict[str, Any]) -> Path:
    return (
        Path(settings["storage"]["local_runtime_root"])
        / "state"
        / "web_run_queue.sqlite3"
    )


def _initialize_persistent_queue(settings: dict[str, Any]) -> None:
    global _persistent_queue, _upload_sessions

    store = DurableRunQueue(_queue_database_path(settings))
    restored_runs = store.load_runs()
    with _lock:
        _runs.clear()
        _runs.update(restored_runs)
    _persistent_queue = store
    _upload_sessions = UploadSessionStore(_queue_database_path(settings))


def _renew_queue_lease(
    store: DurableRunQueue,
    run_id: str,
    worker_id: str,
    stop: threading.Event,
) -> None:
    while not stop.wait(_QUEUE_LEASE_RENEW_SECONDS):
        if not store.renew_lease(
            run_id,
            worker_id,
            lease_seconds=_QUEUE_LEASE_SECONDS,
        ):
            return


def _dispatch_persisted_job(job: QueuedRunJob) -> None:
    from .runtime_control import execution_context
    with execution_context(job_id=job.run_id, source='offline', stop=_queue_stop):
        _dispatch_job_in_context(job)


def _dispatch_job_in_context(job: QueuedRunJob) -> None:
    payload = job.payload
    # Keep queued model/input settings immutable. Admission is a live worker
    # policy so jobs submitted before the split cannot bypass shared limits.
    settings = payload["settings"] | {"runtime": _settings().get("runtime", {})}
    if job.kind == "stage_refresh":
        from .stage_refresh import refresh
        root = _find_staging_run(settings, payload["parent_run_id"])
        if root is None:
            raise ValueError("原实验暂存目录不可用")
        _update(job.run_id, state="running", progress=0.1, message="正在刷新所选阶段")
        receipt = refresh(root, payload["scope"], settings, target=payload.get("target"), revision=payload.get("revision"))
        _update(job.run_id, state="completed", progress=1.0, message="所选阶段已刷新，原质量门保持不变", refresh_receipt=receipt)
        return
    if settings["storage"].get("sync_to_nas"):
        audit_production_model_certification(settings)
    if job.kind == "run":
        _execute_now(
            job.run_id,
            RunManifest.model_validate(payload["manifest"]),
            settings,
            Path(payload["nas_root"]),
            payload.get("ingest"),
        )
        return
    if job.kind == "fixed_benchmark":
        _execute_fixed_benchmark_now(
            job.run_id,
            settings,
            Path(payload["nas_root"]),
            payload["timing"],
        )
        return
    if job.kind == "index_collection":
        _execute_index_collection_now(
            job.run_id,
            payload["source_experiment_id"],
            payload["archive_name"],
            settings,
            Path(payload["staging_root"]),
            Path(payload["fixed_root"]),
            Path(payload["history_root"]),
            payload["timing"],
        )
        return
    raise ValueError(f"Unsupported durable queue job kind: {job.kind}")


def _queue_worker_loop(store: DurableRunQueue) -> None:
    from .runtime_control import ExecutionCancelled
    while not _queue_stop.is_set():
        job = store.claim_next(
            _queue_worker_id,
            lease_seconds=_QUEUE_LEASE_SECONDS,
        )
        if job is None:
            _queue_wakeup.wait(_QUEUE_POLL_SECONDS)
            _queue_wakeup.clear()
            continue

        existing_state = str((store.load_run(job.run_id) or {}).get("state") or "")
        if existing_state in {"completed", "partial", "failed"}:
            store.finish(job.run_id, _queue_worker_id, existing_state)
            _queue_wakeup.set()
            continue

        lease_stop = threading.Event()
        renewer = threading.Thread(
            target=_renew_queue_lease,
            args=(store, job.run_id, _queue_worker_id, lease_stop),
            name=f"visioncortex-queue-lease-{job.run_id}",
            daemon=True,
        )
        renewer.start()
        try:
            if job.reclaimed:
                _update(
                    job.run_id,
                    state="queued",
                    progress=0.0,
                    message="服务重启后已恢复任务，等待可用计算资源",
                    recovered_from_durable_queue=True,
                )
            _dispatch_persisted_job(job)
            latest = store.load_run(job.run_id) or {}
            final_state = str(latest.get("state") or "")
            final_error = latest.get("error")
            if _queue_stop.is_set() and final_state == 'failed' and str(final_error).startswith('ExecutionCancelled:'):
                store.release_for_shutdown(job.run_id, _queue_worker_id)
                continue
            if final_state not in {"completed", "partial", "failed"}:
                final_state = "failed"
                final_error = "Durable queue executor returned without a terminal run state"
                _update(job.run_id, state=final_state, progress=1.0, error=final_error)
            store.finish(
                job.run_id,
                _queue_worker_id,
                final_state,
                error=str(final_error) if final_error else None,
            )
        except ExecutionCancelled:
            store.release_for_shutdown(job.run_id, _queue_worker_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _update(job.run_id, state="failed", progress=1.0, error=error)
            store.finish(
                job.run_id,
                _queue_worker_id,
                "failed",
                error=error,
            )
        finally:
            lease_stop.set()
            renewer.join(timeout=1.0)
            _queue_wakeup.set()


def _start_queue_worker() -> None:
    global _queue_thread

    if _persistent_queue is None:
        raise RuntimeError("Durable run queue has not been initialized")
    if _queue_thread is not None and _queue_thread.is_alive():
        return
    _queue_stop.clear()
    _queue_wakeup.set()
    _queue_thread = threading.Thread(
        target=_queue_worker_loop,
        args=(_persistent_queue,),
        name="visioncortex-durable-run-queue",
        daemon=True,
    )
    _queue_thread.start()


def _stop_queue_worker() -> None:
    global _persistent_queue, _queue_thread

    _queue_stop.set()
    _queue_wakeup.set()
    if _queue_thread is not None:
        _queue_thread.join(timeout=5.0)
        if _queue_thread.is_alive():
            return
    _queue_thread = None
    _persistent_queue = None


def _schedule_job(
    background_tasks: BackgroundTasks,
    *,
    run_id: str,
    kind: str,
    payload: dict[str, Any],
    fallback: Any,
    fallback_args: tuple[Any, ...],
) -> str:
    if _persistent_queue is None:
        # Direct function calls in focused tests do not enter the ASGI lifespan.
        # A real Web server always initializes the SQLite queue before accepting requests.
        background_tasks.add_task(fallback, *fallback_args)
        return "process_memory_fallback"
    _persistent_queue.enqueue(run_id, kind, payload)
    _queue_wakeup.set()
    return "sqlite"


def _reserve_archive(settings: dict[str, Any], experiment_name: str) -> tuple[str, Path]:
    base_name = safe_archive_name(experiment_name)
    archive_root = _archive_root(settings)
    with _lock:
        archive_name = base_name
        if (archive_root / archive_name).exists():
            suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
            archive_name = f"{base_name}-{suffix}-{uuid.uuid4().hex[:4]}"
        nas_root = initialize_nas_archive(settings, archive_name)
    return archive_name, nas_root


def _prepare_formal_run_staging(
    settings: dict[str, Any],
    archive_name: str,
    run_id: str,
    fixed_root: Path,
) -> tuple[Path, Path, Path]:
    """Route every Web-created derived package through the same promotion gate."""

    expected_fixed, staging_root, history_root = fixed_archive_staging_paths(
        settings, archive_name, run_id
    )
    if expected_fixed.resolve() != fixed_root.resolve():
        raise RuntimeError(
            f"Formal archive reservation mismatch: {expected_fixed} != {fixed_root}"
        )
    settings["storage"]["active_archive_path"] = str(staging_root)
    settings["storage"]["run_output_mode"] = "nas_direct"
    settings["storage"]["formal_fixed_root"] = str(expected_fixed)
    settings["storage"]["formal_history_root"] = str(history_root)
    settings["storage"]["formal_promotion_required"] = True
    initialize_nas_archive(settings, archive_name)
    source_json = fixed_root / "JSON-Config-Files"
    if source_json.is_dir():
        shutil.copytree(
            source_json,
            staging_root / "JSON-Config-Files",
            dirs_exist_ok=True,
        )
    return expected_fixed, staging_root, history_root


def _manifest_source_receipts(
    manifest: RunManifest,
    known_sha256: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Describe manifest inputs without reopening large media files."""

    known_sha256 = known_sha256 or {}
    receipts: list[dict[str, Any]] = []
    for view in manifest.views:
        source_pairs: list[tuple[str, int | None, Path]] = []
        if view.video is not None:
            source_pairs.append(("video", None, Path(view.video)))
            if view.timestamps_csv is not None:
                source_pairs.append(("timestamp_csv", None, Path(view.timestamps_csv)))
            if view.audio is not None:
                source_pairs.append(("audio", None, Path(view.audio)))
        else:
            for ordinal, segment in enumerate(view.segments):
                source_pairs.append(("video", ordinal, Path(segment.video)))
                if segment.audio is not None:
                    source_pairs.append(("audio", ordinal, Path(segment.audio)))
                if segment.timestamps_csv is not None:
                    source_pairs.append(
                        ("timestamp_csv", ordinal, Path(segment.timestamps_csv))
                    )
        for kind, ordinal, path in source_pairs:
            stat = path.stat()
            resolved = str(path.resolve())
            sha256 = known_sha256.get(resolved) or known_sha256.get(str(path))
            receipt = {
                "kind": kind,
                "view_id": view.view_id,
                "segment_ordinal": ordinal,
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "source_fingerprint_algorithm": "path_size_mtime",
            }
            if sha256:
                receipt.update(
                    {
                        "sha256": sha256,
                        "content_hash": sha256,
                        "content_hash_algorithm": "sha256",
                    }
                )
            receipts.append(receipt)
    return receipts


def _declared_role_resolution(manifest: RunManifest, source: str) -> dict[str, Any]:
    return {
        "schema_version": "visioncortex-view-role-resolution-ledger/1",
        "experiment_id": manifest.experiment_id,
        "input_source": source,
        "status": "resolved",
        "views": [
            {
                "camera_key": view.view_id,
                "resolved_role": view.role.value,
                "resolution_source": source,
                "blocking_reasons": [],
            }
            for view in manifest.views
        ],
    }


def _require_upload_store(settings: dict[str, Any]) -> UploadSessionStore:
    global _upload_sessions

    if _upload_sessions is None:
        # Focused unit calls may not enter the ASGI lifespan. The production
        # server initializes this same store before accepting requests.
        _upload_sessions = UploadSessionStore(_queue_database_path(settings))
    return _upload_sessions


def _expire_stale_upload_sessions(settings: dict[str, Any]) -> None:
    store = _require_upload_store(settings)
    archive_root = _archive_root(settings).resolve()
    for expired in store.expire_stale():
        candidate = Path(expired["archive_root"]).resolve()
        if candidate.parent != archive_root or not candidate.name:
            continue
        if candidate.is_dir():
            try:
                shutil.rmtree(candidate)
            except OSError:
                # The remaining bytes are still reflected by disk_usage, so a
                # cleanup failure cannot over-admit the next reservation.
                continue


def _upload_policy(settings: dict[str, Any]) -> dict[str, Any]:
    configured = settings.get("web_upload") or {}
    chunk_size_mib = max(1, min(64, int(configured.get("chunk_size_mib", 16))))
    session_ttl_hours = max(1.0, float(configured.get("session_ttl_hours", 168.0)))
    processing_ratio = max(
        0.0, float(configured.get("processing_headroom_ratio", 0.35))
    )
    safety_ratio = max(0.0, float(configured.get("safety_headroom_ratio", 0.10)))
    minimum_safety_gib = max(
        0.0, float(configured.get("minimum_safety_headroom_gib", 10.0))
    )
    parallel_files = max(1, min(4, int(configured.get("parallel_files", 2))))
    full_file_sha256_max_gib = max(
        0.0, float(configured.get("full_file_sha256_max_gib", 4.0))
    )
    return {
        "chunk_size_bytes": chunk_size_mib * 1024 * 1024,
        "session_ttl_seconds": session_ttl_hours * 3600.0,
        "processing_headroom_ratio": processing_ratio,
        "safety_headroom_ratio": safety_ratio,
        "minimum_safety_bytes": math.ceil(minimum_safety_gib * 1024**3),
        "parallel_files": parallel_files,
        "full_file_sha256_max_bytes": math.ceil(full_file_sha256_max_gib * 1024**3),
        "prequeue_media_preflight_enabled": bool(
            configured.get("prequeue_media_preflight_enabled", True)
        ),
        "chunk_sha256_required": bool(
            configured.get("chunk_sha256_required", True)
        ),
    }


def _upload_retention_mode(settings: dict[str, Any]) -> str:
    if settings["storage"].get("sync_to_nas"):
        return "nas_only"
    return "local_only"


def _unique_upload_archive_name(settings: dict[str, Any], experiment_name: str) -> str:
    base_name = safe_archive_name(experiment_name)
    archive_root = _archive_root(settings)
    with _lock:
        if not (archive_root / base_name).exists():
            return base_name
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{base_name}-{suffix}-{uuid.uuid4().hex[:4]}"


def _parse_upload_session_files(
    payload: dict[str, Any],
    archive_path: Path,
    session_id: str,
    settings: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    specs = payload.get("view_specs")
    raw_files = payload.get("files")
    if not isinstance(specs, list) or not specs:
        raise HTTPException(400, "view_specs 必须是非空数组")
    if not isinstance(raw_files, list) or not raw_files:
        raise HTTPException(400, "files 必须是非空数组")

    normalized_files: list[dict[str, Any]] = []
    file_ids: set[str] = set()
    by_kind_index: dict[tuple[str, int], dict[str, Any]] = {}
    for position, item in enumerate(raw_files):
        if not isinstance(item, dict):
            raise HTTPException(400, f"files[{position}] 必须是对象")
        kind = str(item.get("kind") or "")
        if kind not in {"video", "timestamp_csv", "audio"}:
            raise HTTPException(400, f"files[{position}].kind 无效")
        try:
            file_index = int(item.get("file_index"))
            expected_bytes = int(item.get("size"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"files[{position}] 的序号或大小无效") from exc
        if file_index < 0 or expected_bytes <= 0:
            raise HTTPException(400, f"files[{position}] 的序号或大小无效")
        file_id = str(item.get("file_id") or f"{kind}-{file_index}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", file_id) or file_id in file_ids:
            raise HTTPException(400, f"files[{position}].file_id 无效或重复")
        key = (kind, file_index)
        if key in by_kind_index:
            raise HTTPException(400, f"files[{position}] 的类型与序号重复")
        normalized = {
            "file_id": file_id,
            "kind": kind,
            "file_index": file_index,
            "source_name": str(item.get("name") or f"{kind}-{file_index}"),
            "expected_bytes": expected_bytes,
        }
        file_ids.add(file_id)
        by_kind_index[key] = normalized
        normalized_files.append(normalized)

    videos = sorted(
        (item for item in normalized_files if item["kind"] == "video"),
        key=lambda item: int(item["file_index"]),
    )
    if len(videos) < 2 or [item["file_index"] for item in videos] != list(
        range(len(videos))
    ):
        raise HTTPException(400, "视频至少需要两路，且 video_index 必须从 0 连续编号")
    if len(specs) < 2:
        raise HTTPException(400, "至少需要两个视角")

    validation_views: list[ViewInput] = []
    normalized_specs: list[dict[str, Any]] = []
    used_video_indexes: set[int] = set()
    used_csv_indexes: set[int] = set()
    used_audio_indexes: set[int] = set()
    used_paths: set[Path] = set()
    try:
        registry = load_device_registry(
            ((settings or {}).get("storage") or {}).get("device_registry_path")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"设备角色注册表不可用: {exc}") from exc
    for position, raw_spec in enumerate(specs):
        if not isinstance(raw_spec, dict):
            raise HTTPException(400, f"view_specs[{position}] 必须是对象")
        try:
            calibration_hint_ms = float(raw_spec.get("calibration_hint_ms", 0.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"view_specs[{position}] 的文件映射无效") from exc
        raw_segments = raw_spec.get("segments")
        segmented_layout = raw_segments is not None
        if segmented_layout:
            if not isinstance(raw_segments, list) or not raw_segments:
                raise HTTPException(400, f"view_specs[{position}].segments 必须是非空数组")
            mapping_items = raw_segments
        else:
            mapping_items = [
                {
                    "video_index": raw_spec.get("video_index", position),
                    "csv_index": raw_spec.get("csv_index"),
                    "audio_index": raw_spec.get("audio_index"),
                    "audio_offset_ms": raw_spec.get("audio_offset_ms"),
                }
            ]
        mappings: list[tuple[int, int | None, dict[str, Any], dict[str, Any] | None]] = []
        for segment_position, raw_mapping in enumerate(mapping_items):
            if not isinstance(raw_mapping, dict):
                raise HTTPException(
                    400,
                    f"view_specs[{position}].segments[{segment_position}] 必须是对象",
                )
            try:
                video_index = int(raw_mapping.get("video_index"))
                csv_value = raw_mapping.get("csv_index")
                csv_index = int(csv_value) if csv_value is not None else None
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    400,
                    f"view_specs[{position}] 第 {segment_position + 1} 段映射无效",
                ) from exc
            video_file = by_kind_index.get(("video", video_index))
            if video_file is None:
                raise HTTPException(
                    400,
                    f"view_specs[{position}] 第 {segment_position + 1} 段引用了不存在的视频",
                )
            if video_index in used_video_indexes:
                raise HTTPException(400, "同一个视频不能映射到多个视角或分片")
            used_video_indexes.add(video_index)
            csv_file = (
                by_kind_index.get(("timestamp_csv", csv_index))
                if csv_index is not None
                else None
            )
            if csv_index is not None and csv_file is None:
                raise HTTPException(
                    400,
                    f"view_specs[{position}] 第 {segment_position + 1} 段引用了不存在的 CSV",
                )
            if csv_index is not None:
                if csv_index in used_csv_indexes:
                    raise HTTPException(400, "同一个 CSV 不能映射到多个视角或分片")
                used_csv_indexes.add(csv_index)
            mappings.append((video_index, csv_index, video_file, csv_file))
        view_id = _safe_file_name(
            str(raw_spec.get("view_id") or f"view-{position + 1:02d}"),
            f"view-{position + 1:02d}",
        )
        requested_role = str(raw_spec.get("role") or "")
        role_resolution = resolve_view_role(
            registry,
            archive_path.name,
            view_id,
            requested_role,
        )
        role_resolution["input_source"] = "browser_user_declared"
        if view_id not in (registry.get("devices") or {}):
            role_resolution["resolution_source"] = "browser_user_declared"
        elif role_resolution.get("resolution_source") == "experiment_record_index":
            role_resolution["resolution_source"] = (
                "browser_user_declared_confirmed_by_device_registry"
            )
        if role_resolution.get("blocking_reasons") or not role_resolution.get(
            "resolved_role"
        ):
            raise HTTPException(
                400,
                {
                    "message": f"机位 {view_id} 的视角角色与设备注册表冲突",
                    "view_id": view_id,
                    "blocking_reasons": role_resolution.get("blocking_reasons"),
                    "requested_role": requested_role,
                    "registry_expected_role": role_resolution.get(
                        "registry_expected_role"
                    ),
                },
            )
        role = str(role_resolution["resolved_role"])
        normalized_spec = {
            "view_id": view_id,
            "role": role,
            "calibration_hint_ms": calibration_hint_ms,
            "source_layout": "segments" if segmented_layout else "single",
            "role_resolution": role_resolution,
        }
        normalized_mappings = [
            {
                "video_index": video_index,
                **({"csv_index": csv_index} if csv_index is not None else {}),
            }
            for video_index, csv_index, _video_file, _csv_file in mappings
        ]
        audio_mappings = []
        for raw_mapping, normalized_mapping in zip(mapping_items, normalized_mappings, strict=True):
            audio_file = None
            if raw_mapping.get("audio_index") is not None:
                try:
                    audio_index = int(raw_mapping["audio_index"])
                    raw_offset = raw_mapping.get("audio_offset_ms")
                    audio_offset = float(raw_offset) if raw_offset is not None else None
                    if audio_offset is not None and not math.isfinite(audio_offset):
                        raise ValueError("nonfinite audio offset")
                except (TypeError, ValueError) as exc:
                    raise HTTPException(400, "录音映射或时间偏移无效") from exc
                audio_file = by_kind_index.get(("audio", audio_index))
                if audio_file is None or audio_index in used_audio_indexes:
                    raise HTTPException(400, "录音文件不存在或重复映射")
                used_audio_indexes.add(audio_index)
                normalized_mapping.update(audio_index=audio_index, audio_offset_ms=audio_offset)
            elif raw_mapping.get("audio_offset_ms") is not None:
                raise HTTPException(400, "录音时间偏移必须关联录音文件")
            audio_mappings.append(audio_file)
        if segmented_layout:
            normalized_spec["segments"] = normalized_mappings
        else:
            normalized_spec.update(normalized_mappings[0])
        normalized_specs.append(normalized_spec)

        for segment_position, (_video_index, _csv_index, video_file, csv_file) in enumerate(
            mappings, 1
        ):
            for file_item in (video_file, csv_file, audio_mappings[segment_position - 1]):
                if file_item is None:
                    continue
                source_name = _safe_file_name(
                    file_item["source_name"],
                    "video" if file_item["kind"] == "video" else "timestamps",
                )
                prefix = (
                    f"segment-{segment_position:04d}-video-"
                    if file_item["kind"] == "video"
                    else f"segment-{segment_position:04d}-audio-" if file_item["kind"] == "audio"
                    else f"segment-{segment_position:04d}-timestamps-"
                )
                stored_name = f"{prefix}{source_name}"
                final_path = (
                    archive_path
                    / "Original-Experiment-Videos"
                    / view_id
                    / stored_name
                )
                if final_path in used_paths:
                    raise HTTPException(400, "上传文件的目标路径发生冲突")
                used_paths.add(final_path)
                file_item.update(
                    {
                        "view_id": view_id,
                        "segment_ordinal": segment_position,
                        "stored_name": stored_name,
                        "final_path": final_path,
                        "partial_path": final_path.with_name(
                            f".{stored_name}.upload-{session_id}.partial"
                        ),
                    }
                )
        try:
            if segmented_layout:
                validation_views.append(
                    ViewInput(
                        view_id=view_id,
                        role=role,
                        segments=[
                            VideoSegmentInput(
                                video=Path(f"/{view_id}-{index:04d}.mp4"),
                                timestamps_csv=(
                                    Path(f"/{view_id}-{index:04d}.csv")
                                    if csv_file is not None
                                    else None
                                ),
                            )
                            for index, (_v, _c, _video_file, csv_file) in enumerate(
                                mappings, 1
                            )
                        ],
                        calibration_hint_ms=calibration_hint_ms,
                    )
                )
            else:
                _video_index, _csv_index, _video_file, csv_file = mappings[0]
                validation_views.append(
                    ViewInput(
                        view_id=view_id,
                        role=role,
                        video=Path(f"/{view_id}.mp4"),
                        timestamps_csv=(Path(f"/{view_id}.csv") if csv_file else None),
                        calibration_hint_ms=calibration_hint_ms,
                    )
                )
        except ValueError as exc:
            raise HTTPException(400, f"view_specs[{position}] 无效: {exc}") from exc

    unused_audio = {int(item["file_index"]) for item in normalized_files if item["kind"] == "audio"} - used_audio_indexes
    if unused_audio:
        raise HTTPException(400, "存在未映射的录音文件")
    unused_csvs = {
        int(item["file_index"])
        for item in normalized_files
        if item["kind"] == "timestamp_csv"
    } - used_csv_indexes
    if unused_csvs:
        raise HTTPException(400, f"存在未映射的 CSV: {sorted(unused_csvs)}")
    unused_videos = {int(item["file_index"]) for item in videos} - used_video_indexes
    if unused_videos:
        raise HTTPException(400, f"存在未映射的视频: {sorted(unused_videos)}")
    try:
        RunManifest(experiment_id=archive_path.name, views=validation_views)
    except ValueError as exc:
        raise HTTPException(400, f"视角配置无效: {exc}") from exc

    return normalized_specs, normalized_files, sum(
        int(item["expected_bytes"]) for item in normalized_files
    )


def _public_upload_session(session: dict[str, Any]) -> dict[str, Any]:
    files = [
        {
            "file_id": item["file_id"],
            "kind": item["kind"],
            "file_index": int(item["file_index"]),
            "name": item["source_name"],
            "size": int(item["expected_bytes"]),
            "uploaded_bytes": int(item["uploaded_bytes"]),
            "completed": bool(item.get("content_hash") or item.get("sha256")),
            "sha256": item.get("sha256"),
            "content_hash": item.get("content_hash"),
            "content_hash_algorithm": item.get("content_hash_algorithm"),
        }
        for item in session["files"]
    ]
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "archive_name": session["archive_name"],
        "retention_mode": session["retention_mode"],
        "expected_source_bytes": int(session["expected_source_bytes"]),
        "reserved_bytes": int(session["reserved_bytes"]),
        "processing_headroom_bytes": int(session["processing_headroom_bytes"]),
        "safety_headroom_bytes": int(session["safety_headroom_bytes"]),
        "uploaded_bytes": sum(int(item["uploaded_bytes"]) for item in files),
        "expires_at_epoch": float(session["expires_at"]),
        "run_id": session.get("run_id"),
        "files": files,
    }


def _load_upload_session(
    settings: dict[str, Any], session_id: str, *, refresh_expiry: bool = False
) -> dict[str, Any]:
    if refresh_expiry:
        _expire_stale_upload_sessions(settings)
    store = _require_upload_store(settings)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, "上传会话不存在")
    policy = _upload_policy(settings)
    for item in session["files"]:
        final_path = Path(item["final_path"])
        partial_path = Path(item["partial_path"])
        active_path = final_path if final_path.is_file() else partial_path
        actual_bytes = active_path.stat().st_size if active_path.is_file() else 0
        if actual_bytes > int(item["expected_bytes"]):
            raise HTTPException(500, f"服务器暂存文件超过声明大小: {item['file_id']}")
        if actual_bytes != int(item["uploaded_bytes"]):
            store.update_progress(
                session_id,
                str(item["file_id"]),
                actual_bytes,
                expires_at=time.time() + float(policy["session_ttl_seconds"]),
            )
            item["uploaded_bytes"] = actual_bytes
    if refresh_expiry and session["status"] == "open" and session["files"]:
        first = session["files"][0]
        store.update_progress(
            session_id,
            str(first["file_id"]),
            int(first["uploaded_bytes"]),
            expires_at=time.time() + float(policy["session_ttl_seconds"]),
        )
        session = store.get(session_id) or session
    return session


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _reserve_collection_archive(
    settings: dict[str, Any], experiment_name: str, run_id: str
) -> tuple[str, Path, Path, Path]:
    """Reserve a unique formal name while writing only to run staging."""

    base_name = safe_archive_name(experiment_name)
    archive_root = _archive_root(settings)
    with _lock:
        archive_name = base_name
        if (archive_root / archive_name).exists() or any(
            run.get("experiment_id") == archive_name
            and run.get("state") not in {"completed", "partial", "failed", "interrupted"}
            for run in _runs.values()
        ):
            suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
            archive_name = f"{base_name}-{suffix}-{uuid.uuid4().hex[:4]}"
        fixed_root, staging_root, history_root = fixed_archive_staging_paths(
            settings, archive_name, run_id
        )
        local_runtime_root = Path(settings["storage"]["local_runtime_root"]).resolve()
        settings["project"]["output_root"] = str(local_runtime_root / "runs" / run_id)
        settings["storage"]["active_archive_path"] = str(staging_root)
        # The configured archive may be local or network storage. Write derived
        # files directly into its staging tree in either case.
        settings["storage"]["run_output_mode"] = "nas_direct"
        initialize_nas_archive(settings, archive_name)
    return archive_name, fixed_root, staging_root, history_root


def _reserve_fixed_benchmark(settings: dict[str, Any], run_id: str) -> Path:
    fixed_root, nas_root, history_root = fixed_archive_staging_paths(
        settings, _BENCHMARK_ARCHIVE_NAME, run_id
    )
    with _lock:
        if any(
            run.get("benchmark_archive") == _BENCHMARK_ARCHIVE_NAME
            and run.get("state") not in {"completed", "partial", "failed"}
            for existing_id, run in _runs.items()
            if existing_id != run_id
        ):
            raise HTTPException(409, "六路三小时固定基准已经在运行")
        local_runtime_root = Path(settings["storage"]["local_runtime_root"]).resolve()
        settings["project"]["output_root"] = str(local_runtime_root / "runs" / run_id)
        settings["storage"]["active_archive_path"] = str(nas_root)
        initialize_nas_archive(settings, _BENCHMARK_ARCHIVE_NAME)
        _runs[run_id] = {
            "state": "reserving",
            "progress": 0.0,
            "experiment_id": _BENCHMARK_ARCHIVE_NAME,
            "benchmark_archive": _BENCHMARK_ARCHIVE_NAME,
            "nas_output": str(fixed_root),
            "nas_staging": str(nas_root),
            "nas_history": str(history_root),
        }
    return nas_root


def _update(run_id: str, **values: Any) -> None:
    with _lock:
        if _persistent_queue is not None:
            _runs[run_id] = _persistent_queue.patch_run(run_id, values)
        else:
            _runs.setdefault(run_id, {}).update(values)


def _write_fixed_benchmark_submission_receipt(
    nas_root: Path,
    run_id: str,
    request_received_at: str,
) -> Path:
    """Persist the asynchronous ownership contract before starting a long run."""

    receipt_path = nas_root / "JSON-Config-Files" / "run_submission.json"
    _write_json_atomic(
        receipt_path,
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "task": "fixed_six_view_three_hour_benchmark",
            "state": "queued",
            "submitted_at": request_received_at,
            "execution": {
                "owner": "visioncortex_web_service",
                "server_pid": os.getpid(),
                "client_process_independent": True,
                "queue_persistence": "sqlite",
                "survives_web_service_restart": True,
                "requires_web_service_alive_to_execute": True,
            },
            "monitoring": {
                "status_url": f"/api/runs/{run_id}",
                "nas_staging": str(nas_root),
                "durable_status": "JSON-Config-Files/pipeline_status.json",
            },
        },
    )
    return receipt_path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _queue_recovery_receipt_path(nas_root: Path) -> Path:
    return (
        nas_root
        / "JSON-Config-Files"
        / "Input-Manifests"
        / "queue_recovery.json"
    )


def _write_queue_recovery_receipt(
    nas_root: Path,
    *,
    run_id: str,
    state: str,
    manifest_path: Path,
    input_seal_path: Path | None,
    ingest: dict[str, Any] | None,
    error: str | None = None,
    recovered_from_archive_receipt: bool = False,
    job_kind: str = "run",
    recovery_context: dict[str, Any] | None = None,
) -> Path:
    receipt_path = _queue_recovery_receipt_path(nas_root)
    previous = _read_json(receipt_path, {}) or {}
    payload = {
        "schema_version": "visioncortex-queue-recovery/1",
        "run_id": run_id,
        "job_kind": job_kind,
        "state": state,
        "updated_at": datetime.now().astimezone().isoformat(),
        "archive_root": str(nas_root),
        "manifest_relative_path": manifest_path.relative_to(nas_root).as_posix(),
        "input_seal_relative_path": (
            input_seal_path.relative_to(nas_root).as_posix()
            if input_seal_path is not None
            else None
        ),
        "ingest": ingest,
        "recovery_context": recovery_context,
        "error": error,
        "recovered_from_archive_receipt": bool(
            recovered_from_archive_receipt
            or previous.get("recovered_from_archive_receipt")
        ),
        "recovery_count": int(previous.get("recovery_count") or 0),
    }
    if recovered_from_archive_receipt:
        payload["recovery_count"] += 1
    _write_json_atomic(receipt_path, payload)
    return receipt_path


def _update_queue_recovery_state(
    nas_root: Path,
    state: str,
    *,
    error: str | None = None,
) -> None:
    receipt_path = _queue_recovery_receipt_path(nas_root)
    payload = _read_json(receipt_path, {}) or {}
    if payload.get("schema_version") != "visioncortex-queue-recovery/1":
        return
    payload.update(
        {
            "state": state,
            "updated_at": datetime.now().astimezone().isoformat(),
            "error": error,
        }
    )
    _write_json_atomic(receipt_path, payload)


def _recover_jobs_from_archive_receipts(settings: dict[str, Any]) -> None:
    """Rebuild queued browser jobs when the local SQLite queue is lost."""

    if _persistent_queue is None:
        return
    root = _archive_root(settings).resolve()
    if not root.is_dir():
        return
    receipt_paths = sorted(
        root.glob("*/JSON-Config-Files/Input-Manifests/queue_recovery.json")
    )
    for staging_root in run_staging_roots(settings):
        if staging_root.is_dir():
            receipt_paths.extend(
                sorted(
                    staging_root.glob(
                        "*/*/JSON-Config-Files/Input-Manifests/queue_recovery.json"
                    )
                )
            )
    for receipt_path in receipt_paths:
        payload = _read_json(receipt_path, {}) or {}
        if payload.get("schema_version") != "visioncortex-queue-recovery/1":
            continue
        job_kind = str(payload.get("job_kind") or "")
        if job_kind not in {"run", "index_collection"} or payload.get(
            "state"
        ) not in {
            "queued",
            "running",
        }:
            continue
        run_id = str(payload.get("run_id") or "")
        if not run_id or _persistent_queue.get_job(run_id) is not None:
            continue
        nas_root = receipt_path.parents[2].resolve()
        if not archive_contains(nas_root, root) or nas_root == root:
            continue
        status_path = nas_root / "JSON-Config-Files" / "pipeline_status.json"
        if status_path.exists():
            pipeline_status = _read_json(status_path, {}) or {}
            if pipeline_status.get("stage") in {"completed", "partial", "failed"}:
                _update_queue_recovery_state(
                    nas_root,
                    str(pipeline_status["stage"]),
                    error=pipeline_status.get("error"),
                )
                continue
            if _runtime_activity_receipt(status_path)["active"]:
                continue
        manifest_relative = Path(str(payload.get("manifest_relative_path") or ""))
        seal_relative_value = str(payload.get("input_seal_relative_path") or "").strip()
        seal_relative = Path(seal_relative_value) if seal_relative_value else None
        if manifest_relative.is_absolute() or (
            seal_relative is not None and seal_relative.is_absolute()
        ):
            continue
        manifest_path = (nas_root / manifest_relative).resolve()
        input_seal_path = (
            (nas_root / seal_relative).resolve()
            if seal_relative is not None
            else None
        )
        try:
            manifest_path.relative_to(nas_root)
            if input_seal_path is not None:
                input_seal_path.relative_to(nas_root)
        except ValueError:
            continue
        if not manifest_path.is_file() or (
            input_seal_path is not None and not input_seal_path.is_file()
        ):
            continue
        try:
            manifest = RunManifest.model_validate(
                yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            )
            seal = (
                json.loads(input_seal_path.read_text(encoding="utf-8"))
                if input_seal_path is not None
                else None
            )
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError):
            continue
        if seal is not None and (
            not verify_input_seal(seal)
            or seal.get("experiment_id") != manifest.experiment_id
        ):
            continue
        sources_available = True
        sources = (
            seal.get("sources") or []
            if seal is not None
            else _manifest_source_receipts(manifest)
        )
        for source in sources:
            source_path = Path(str(source.get("path") or ""))
            try:
                source_stat = source_path.stat()
            except OSError:
                sources_available = False
                break
            if source.get("size_bytes") is not None and int(
                source["size_bytes"]
            ) != int(source_stat.st_size):
                sources_available = False
                break
            expected_mtime = source.get("mtime_ns")
            if expected_mtime is not None and int(expected_mtime) != int(
                source_stat.st_mtime_ns
            ):
                sources_available = False
                break
        if not sources_available:
            continue
        recovered_settings = copy.deepcopy(settings)
        recovered_settings["storage"]["active_archive_path"] = str(nas_root)
        if job_kind == "run":
            context = payload.get("recovery_context") or {}
            fixed_root_value = str(context.get("formal_fixed_root") or "").strip()
            history_root_value = str(context.get("formal_history_root") or "").strip()
            if fixed_root_value and history_root_value:
                recovered_settings["storage"].update(
                    {
                        "run_output_mode": "nas_direct",
                        "formal_fixed_root": fixed_root_value,
                        "formal_history_root": history_root_value,
                        "formal_promotion_required": True,
                    }
                )
            recovered_settings["storage"]["sync_to_nas"] = (
                str((payload.get("ingest") or {}).get("retention_mode")) == "nas_only"
            )
        run_state = {
            "state": "queued",
            "progress": 0.0,
            "message": "本地队列账本丢失后，已从归档输入封条恢复任务",
            "experiment_id": manifest.experiment_id,
            "nas_output": str(
                recovered_settings["storage"].get("formal_fixed_root") or nas_root
            ),
            "nas_staging": str(nas_root),
            "recovered_from_archive_receipt": True,
        }
        _runs[run_id] = run_state
        _persistent_queue.save_run(run_id, run_state)
        try:
            if job_kind == "run":
                job_payload = {
                    "manifest": manifest.model_dump(mode="json"),
                    "settings": recovered_settings,
                    "nas_root": str(nas_root),
                    "ingest": payload.get("ingest"),
                }
            else:
                context = payload.get("recovery_context") or {}
                required = {
                    "source_experiment_id",
                    "archive_name",
                    "staging_root",
                    "fixed_root",
                    "history_root",
                    "timing",
                }
                if not required.issubset(context):
                    continue
                job_payload = {
                    "source_experiment_id": context["source_experiment_id"],
                    "archive_name": context["archive_name"],
                    "settings": recovered_settings,
                    "staging_root": context["staging_root"],
                    "fixed_root": context["fixed_root"],
                    "history_root": context["history_root"],
                    "timing": context["timing"],
                }
            _persistent_queue.enqueue(run_id, job_kind, job_payload)
        except ValueError:
            continue
        _write_queue_recovery_receipt(
            nas_root,
            run_id=run_id,
            state="queued",
            manifest_path=manifest_path,
            input_seal_path=input_seal_path,
            ingest=payload.get("ingest"),
            recovered_from_archive_receipt=True,
            job_kind=job_kind,
            recovery_context=payload.get("recovery_context"),
        )


def _append_web_end_to_end_metrics(
    roots: list[Path], ingest: dict[str, Any], completed: bool
) -> None:
    ended_at = datetime.now().astimezone().isoformat()
    total_seconds = round(
        (
            time.time() - float(ingest["request_started_epoch"])
            if ingest.get("request_started_epoch") is not None
            else time.perf_counter() - float(ingest["request_started_perf"])
        ),
        6,
    )
    for root in dict.fromkeys(path.resolve() for path in roots):
        metrics_path = root / "JSON-Config-Files" / "delivery_metrics.json"
        metrics = _read_json(metrics_path, {}) or {}
        metrics["web_ingest"] = {
            key: value
            for key, value in ingest.items()
            if key not in {"request_started_perf", "request_started_epoch"}
        }
        metrics["web_end_to_end"] = {
            "definition": "HTTP request arrival through analysis release-gate readiness; final publication time is authoritative in the current-release pointer",
            "request_received_at": ingest["request_received_at"],
            "completed_at": ended_at,
            "total_duration_seconds": total_seconds,
            "completed": completed,
        }
        _write_json_atomic(metrics_path, metrics)


def _append_fixed_benchmark_metrics(
    roots: list[Path], timing: dict[str, Any], completed: bool
) -> None:
    ended_at = datetime.now().astimezone().isoformat()
    total_seconds = round(
        (
            time.time() - float(timing["request_started_epoch"])
            if timing.get("request_started_epoch") is not None
            else time.perf_counter() - float(timing["request_started_perf"])
        ),
        6,
    )
    for root in dict.fromkeys(path.resolve() for path in roots):
        metrics_path = root / "JSON-Config-Files" / "delivery_metrics.json"
        metrics = _read_json(metrics_path, {}) or {}
        metrics["nas_index_ingest"] = {
            key: value
            for key, value in timing.items()
            if key
            not in {
                "request_started_perf",
                "request_started_epoch",
                "request_received_at",
            }
        }
        metrics["fixed_benchmark_end_to_end"] = {
            "definition": "benchmark request through analysis release-gate readiness; final publication time is authoritative in the current-release pointer",
            "request_received_at": timing["request_received_at"],
            "completed_at": ended_at,
            "total_duration_seconds": total_seconds,
            "completed": completed,
            "experiment_id": _BENCHMARK_EXPERIMENT_ID,
            "archive_name": _BENCHMARK_ARCHIVE_NAME,
        }
        _write_json_atomic(metrics_path, metrics)


def _append_collection_index_metrics(
    roots: list[Path], timing: dict[str, Any], completed: bool
) -> None:
    ended_at = datetime.now().astimezone().isoformat()
    total_seconds = round(
        (
            time.time() - float(timing["request_started_epoch"])
            if timing.get("request_started_epoch") is not None
            else time.perf_counter() - float(timing["request_started_perf"])
        ),
        6,
    )
    for root in dict.fromkeys(path.resolve() for path in roots):
        metrics_path = root / "JSON-Config-Files" / "delivery_metrics.json"
        metrics = _read_json(metrics_path, {}) or {}
        metrics["nas_index_ingest"] = {
            key: value
            for key, value in timing.items()
            if key
            not in {
                "request_started_perf",
                "request_started_epoch",
                "request_received_at",
            }
        }
        metrics["collection_end_to_end"] = {
            "definition": "collection selection through analysis release-gate readiness; final publication time is authoritative in the current-release pointer",
            "request_received_at": timing["request_received_at"],
            "completed_at": ended_at,
            "total_duration_seconds": total_seconds,
            "completed": completed,
            "source_experiment_id": timing.get("source_experiment_id"),
            "archive_name": timing.get("archive_name"),
        }
        _write_json_atomic(metrics_path, metrics)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _pipeline_status_from_root(root: Path) -> dict[str, Any]:
    return _read_json(root / "JSON-Config-Files/pipeline_status.json", {}) or _read_json(
        root / "run_status.json", {}
    ) or {}


def _latest_result_review(root: Path, *, save=False) -> dict:
    from .result_review import inspect
    try:
        return inspect(root, save=save)
    except (OSError, ValueError, KeyError, TypeError):
        return {"available":False, "reason":"当前记录不足以执行版本与完整性检查"}


def _merged_run_metrics(root: Path) -> dict[str, Any]:
    json_root = root / "JSON-Config-Files"
    final = _read_json(json_root / "run_metrics.json", {}) or {}
    live = _read_json(json_root / "run_metrics_live.json", {}) or {}
    metrics = live if str(live.get("run_started_at", "")) > str(final.get("run_started_at", "")) else final or live
    delivery = _read_json(json_root / "delivery_metrics.json", {}) or {}
    # A retry keeps the previous receipts on disk. Its delivery timing must not
    # overwrite the newer live attempt's metrics.
    if not final or metrics.get("run_started_at") == final.get("run_started_at"):
        metrics.update(delivery)
    publication = (read_current_release_pointer(root) or {}).get("publication")
    if isinstance(publication, dict):
        metric_key = str(publication.get("metric_key") or "").strip()
        if metric_key:
            metrics[metric_key] = {
                key: value
                for key, value in publication.items()
                if key != "metric_key"
            }
    # Add operation-only refresh receipts once; never replace the original run's
    # wall clock with follow-up request time or count local reuse as new usage.
    return with_operation_refreshes(root, metrics)


def _current_attempt_started_at(status: dict[str, Any]) -> float | None:
    try:
        return datetime.fromisoformat(status["updated_at"]).timestamp() - float(status["elapsed_seconds"])
    except (KeyError, TypeError, ValueError):
        return None


def _stage_receipts_from_root(root: Path) -> list[dict[str, Any]]:
    """Return durable stage receipts without recursively walking NAS outputs."""

    receipt_root = root / "JSON-Config-Files" / "Stage-Receipts"
    if not receipt_root.is_dir():
        return []
    started_at = _current_attempt_started_at(_pipeline_status_from_root(root))
    receipts: list[dict[str, Any]] = []
    for path in sorted(receipt_root.glob("*.json")):
        payload = _read_json(path, {}) or {}
        if not payload.get("stage"):
            continue
        if started_at is not None:
            try:
                if datetime.fromisoformat(payload["completed_at"]).timestamp() < started_at - 1:
                    continue
            except (KeyError, TypeError, ValueError):
                # Undated legacy receipts cannot establish completion of a
                # timestamped attempt; the files remain available on disk.
                continue
        artifacts = []
        for raw in payload.get("artifacts") or []:
            artifact_path = Path(str(raw))
            resolved = artifact_path if artifact_path.is_absolute() else root / artifact_path
            if artifact_path.is_absolute():
                # Stage receipts are written before the staging tree is promoted.
                # Resolve their archive-relative suffix against the final archive.
                parts = artifact_path.parts
                for directory in ARCHIVE_DIRECTORIES:
                    if directory not in parts:
                        continue
                    suffix = Path(*parts[parts.index(directory) :])
                    promoted = root / suffix
                    if promoted.exists():
                        resolved = promoted
                        break
            relative_path = None
            try:
                relative_path = (
                    resolved.resolve().relative_to(root.resolve()).as_posix()
                )
            except (OSError, ValueError):
                pass
            artifacts.append(
                {
                    "path": str(raw),
                    "name": artifact_path.name,
                    "available": resolved.exists(),
                    "relative_path": relative_path,
                    "kind": "directory" if resolved.is_dir() else "file",
                }
            )
        receipts.append(
            {
                "stage": payload["stage"],
                "status": payload.get("status", "completed"),
                "completed_at": payload.get("completed_at"),
                "run_elapsed_seconds": payload.get("run_elapsed_seconds"),
                "stage_duration_seconds": payload.get("stage_duration_seconds"),
                "archive_mode": payload.get("archive_mode"),
                "artifacts": artifacts,
                "receipt": f"JSON-Config-Files/Stage-Receipts/{path.name}",
            }
        )
    return receipts


def _archive_areas_from_root(root: Path) -> list[dict[str, Any]]:
    """Expose the stable archive contract used by both staging and final output."""

    return [
        {
            "name": name,
            "path": str(root / name),
            "available": (root / name).is_dir(),
        }
        for name in ARCHIVE_DIRECTORIES
    ]


def _run_snapshot_from_root(root: Path) -> dict[str, Any]:
    json_root = root / "JSON-Config-Files"
    status = _pipeline_status_from_root(root)
    live_telemetry = _read_json(json_root / "resource_telemetry_live.json", {}) or {}
    telemetry = _read_json(json_root / "resource_telemetry.json", {}) or {}
    metrics = _merged_run_metrics(root)
    source_progress = _read_json(json_root / "source_progress.json", {}) or {}
    groups = _read_json(json_root / "experiment_group_understanding.json", {}) or {}
    if source_progress.get("views"):
        status["views"] = source_progress["views"]
    scans = {
        phase: _read_json(json_root / f"scan_runtime_{phase}.json", {}) or {}
        for phase in ("coarse", "fine_scout", "fine")
    }
    return {
        "root": str(root),
        "status": status,
        "live_telemetry": live_telemetry,
        "insights": {
            "hardware": hardware_summary(telemetry, live_telemetry, metrics.get("run_started_at")),
            "tokens": token_summary(metrics),
            "timing": timing_summary(metrics, scans),
        },
        "telemetry_summary": {
            "sample_count": telemetry.get("sample_count"),
            "sampling_interval_seconds": telemetry.get("sampling_interval_seconds"),
            "stage_summaries": telemetry.get("stage_summaries") or {},
        },
        "metrics": metrics,
        "capture_quality": _read_json(json_root / "capture_quality.json", {}) or {},
        "partial_delivery": _read_json(json_root / "partial_delivery.json", {}) or {},
        "components": component_results(root),
        "retained_experiment_count": len(groups.get("groups") or []),
        "scan_runtime": scans,
        "source_progress": source_progress,
        "stage_receipts": _stage_receipts_from_root(root),
        "archive_areas": _archive_areas_from_root(root),
        "freshness": {
            "status_updated_at": status.get("updated_at"),
            "telemetry_updated_at": live_telemetry.get("updated_at"),
            "source_progress_updated_at": source_progress.get("updated_at"),
        },
        "sources": {
            "status": "JSON-Config-Files/pipeline_status.json",
            "telemetry_live": "JSON-Config-Files/resource_telemetry_live.json",
            "telemetry_final": "JSON-Config-Files/resource_telemetry.json",
            "metrics": "JSON-Config-Files/run_metrics.json",
            "metrics_live": "JSON-Config-Files/run_metrics_live.json",
            "scan_runtime_coarse": "JSON-Config-Files/scan_runtime_coarse.json",
            "scan_runtime_fine_scout": (
                "JSON-Config-Files/scan_runtime_fine_scout.json"
            ),
            "scan_runtime_fine": "JSON-Config-Files/scan_runtime_fine.json",
            "source_progress": "JSON-Config-Files/source_progress.json",
            "stage_receipts": "JSON-Config-Files/Stage-Receipts/*.json",
        },
    }


def _hydrate_run_snapshot(run: dict[str, Any]) -> dict[str, Any]:
    if run.get("origin") == "registered_local_partial":
        root = registered_partial_root(_settings(), run)
        return {**run, "observability": _run_snapshot_from_root(root) if root else {},
                "result_available": root is not None}
    keys = (
        ("observability_root", "nas_output", "output", "nas_staging")
        if run.get("state") == "completed"
        else ("observability_root", "nas_staging", "output", "nas_output")
    )
    for key in keys:
        value = run.get(key)
        if not value:
            continue
        root = Path(value)
        if root.is_dir():
            snapshot = _run_snapshot_from_root(root)
            if run.get("state") == "queued" and run.get("attempt_history"):
                # Retrying moves old derived output out of the active view.
                # Read its retained receipt; the durable queue owns current status.
                previous_status = snapshot["status"]
                attempt = run["attempt_history"][-1].get("attempt")
                if isinstance(attempt, int) and attempt >= 0:
                    previous_status = _read_json(
                        root / "JSON-Config-Files" / "Retry-Attempts"
                        / str(attempt) / "pipeline_status.json", {},
                    ) or previous_status
                snapshot["previous_attempt_status"] = previous_status
                snapshot["status"] = {
                    "stage": "queued", "progress": 0.0,
                    "message": run.get("message"), "elapsed_seconds": 0.0,
                }
            return {**run, "observability": snapshot}
    return run


def _find_staging_run(settings: dict[str, Any], run_id: str) -> Path | None:
    record = _runs.get(run_id) or {}
    if record.get("origin") == "registered_local_partial":
        return registered_partial_root(settings, record)
    for staging_root in run_staging_roots(settings):
        # Input validation can fail before the pipeline writes its first status.
        # Use the durable job's exact directory, still constrained to this store.
        if record.get("nas_staging"):
            candidate = Path(record["nas_staging"]).resolve()
            if (candidate.name == run_id and candidate.parent.parent == staging_root.resolve()
                    and candidate.is_dir()):
                return candidate
        for path in staging_root.glob("*/*/JSON-Config-Files/pipeline_status.json"):
            archive_run_root = path.parent.parent
            if archive_run_root.name == run_id:
                return archive_run_root
    return None


def _file_url(
    archive_name: str,
    relative: str | Path,
    release_id: str | None = None,
) -> str:
    url = f"/api/archive-file?archive={quote(archive_name)}&path={quote(Path(relative).as_posix())}"
    if release_id:
        url += f"&release={quote(release_id)}"
    return url


def _staging_file_url(run_id: str, relative: str | Path) -> str:
    return (
        f"/api/staging-file?run_id={quote(run_id)}"
        f"&path={quote(Path(relative).as_posix())}"
    )


def _resolve_archive(archive_name: str) -> Path:
    root = _archive_root().resolve()
    candidate = (root / archive_name).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise HTTPException(404, "实验档案不存在")
    if archive_promotion_in_progress(candidate):
        raise HTTPException(409, "实验档案正在原子发布，请稍后重试")
    json_root = candidate / "JSON-Config-Files"
    if not (
        (json_root / "evidence_package.json").is_file()
        or (json_root / INDEX_DB_NAME).is_file()
    ):
        raise HTTPException(409, "实验档案尚未通过正式发布门禁")
    return candidate


def _resolve_staging_run(run_id: str) -> Path:
    root = _find_staging_run(_settings(), run_id)
    if root is None:
        raise HTTPException(404, "staging run_id 不存在")
    return root.resolve()


async def _save_upload_to_local_and_nas(
    upload: UploadFile,
    local_destination: Path,
    nas_destination: Path,
    *,
    retain_local_copy: bool = True,
    archive_is_network: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    if retain_local_copy:
        local_destination.parent.mkdir(parents=True, exist_ok=True)
    nas_destination.parent.mkdir(parents=True, exist_ok=True)
    local_partial = local_destination.with_name(
        f".{local_destination.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    nas_partial = nas_destination.with_name(f".{nas_destination.name}.partial-{uuid.uuid4().hex[:8]}")
    total = 0
    digest = hashlib.sha256()
    local_handle = None
    try:
        local_handle = local_partial.open("wb") if retain_local_copy else None
        with nas_partial.open("wb") as nas_handle:
            while block := await upload.read(8 * 1024 * 1024):
                if local_handle is not None:
                    local_handle.write(block)
                nas_handle.write(block)
                total += len(block)
                digest.update(block)
            if local_handle is not None:
                local_handle.flush()
                os.fsync(local_handle.fileno())
            nas_handle.flush()
            os.fsync(nas_handle.fileno())
        if local_handle is not None:
            local_handle.close()
            local_handle = None
            os.replace(local_partial, local_destination)
        os.replace(nas_partial, nas_destination)
    finally:
        if local_handle is not None:
            local_handle.close()
        for partial in (local_partial, nas_partial):
            if partial.exists():
                partial.unlink()
        await upload.close()
    duration_seconds = max(time.perf_counter() - started, 1e-9)
    return {
        "filename": upload.filename,
        "bytes": total,
        "local_path": str(local_destination) if retain_local_copy else (
            str(nas_destination) if not archive_is_network else None
        ),
        "nas_path": str(nas_destination) if archive_is_network else None,
        "analysis_path": str(local_destination if retain_local_copy else nas_destination),
        "retention_mode": ("local_and_nas" if retain_local_copy else "nas_only")
        if archive_is_network else "local_only",
        "local_write_bytes": total * (int(retain_local_copy) + int(not archive_is_network)),
        "nas_write_bytes": total if archive_is_network else 0,
        "sha256": digest.hexdigest(),
        "duration_seconds": round(duration_seconds, 6),
        "effective_source_throughput_mib_s": round(
            total / duration_seconds / (1024 * 1024), 3
        ),
    }


def _record_partial_completion(run_id: str, root: Path) -> bool:
    """Recognize only a fully written quality-attention result, never publish it."""
    if not partial_result_available(root):
        return False
    status = _pipeline_status_from_root(root)
    _update_queue_recovery_state(root, "partial")
    _update(run_id, state="partial", progress=1.0, error=None,
            message=status.get("message"), nas_staging=str(root),
            observability_root=str(root), output=None, promotion=None, archive_url=None,
            result_url=f"/#/stage/{quote(run_id)}/experiments",
            evidence_classification="PARTIAL_EVIDENCE")
    return True


def _execute_now(
    run_id: str,
    manifest: RunManifest,
    settings: dict[str, Any],
    nas_root: Path,
    ingest: dict[str, Any] | None = None,
) -> None:
    def progress(stage: str, value: float, message: str) -> None:
        if (
            stage == "completed"
            and settings.get("storage", {}).get("formal_promotion_required")
        ):
            stage = "finalizing"
            value = min(float(value), 0.999)
            message = "派生结果已通过验收，正在发布正式档案"
        display_root = str(
            settings.get("storage", {}).get("formal_fixed_root") or nas_root
        )
        _update(
            run_id,
            state=stage,
            progress=value,
            message=message,
            nas_output=display_root,
            nas_staging=str(nas_root),
        )

    try:
        _update(
            run_id,
            state="running",
            progress=0.0,
            nas_output=str(
                settings.get("storage", {}).get("formal_fixed_root") or nas_root
            ),
            nas_staging=str(nas_root),
        )
        _update_queue_recovery_state(nas_root, "running")
        output = EvidencePipeline(settings, progress).run(manifest)
        if _record_partial_completion(run_id, Path(output)):
            if ingest is not None:
                _append_web_end_to_end_metrics([Path(output)], ingest, completed=False)
                write_partial_delivery(Path(output), _merged_run_metrics(Path(output)))
            return
        formal_fixed_value = str(
            settings.get("storage", {}).get("formal_fixed_root") or ""
        ).strip()
        formal_history_value = str(
            settings.get("storage", {}).get("formal_history_root") or ""
        ).strip()
        promotion = None
        completed_root = Path(output)
        publication_context = None
        if ingest is not None:
            _append_web_end_to_end_metrics([Path(output)], ingest, completed=True)
            publication_context = {
                "metric_key": "web_end_to_end",
                "definition": "HTTP request arrival + source retention + analysis + atomic formal archive publication",
                "request_received_at": ingest.get("request_received_at"),
                "request_started_epoch": ingest.get("request_started_epoch"),
            }
        if settings.get("storage", {}).get("formal_promotion_required"):
            _update_queue_recovery_state(nas_root, "completed")
            if not formal_fixed_value or not formal_history_value:
                raise RuntimeError(
                    "Formal Web run is missing fixed/history promotion targets"
                )
            formal_fixed_root = Path(formal_fixed_value)
            promotion = promote_fixed_archive(
                Path(output),
                formal_fixed_root,
                Path(formal_history_value),
                publication_context,
            )
            completed_root = formal_fixed_root
        elif ingest is not None:
            _append_web_end_to_end_metrics(
                [completed_root], ingest, completed=True
            )
        _update(
            run_id,
            state="completed",
            progress=1.0,
            output=str(completed_root),
            nas_output=str(completed_root),
            nas_staging=str(nas_root),
            promotion=promotion,
            archive_url=f"/?archive={quote(completed_root.name)}",
        )
        if not settings.get("storage", {}).get("formal_promotion_required"):
            _update_queue_recovery_state(completed_root, "completed")
    except Exception as exc:
        if ingest is not None:
            _append_web_end_to_end_metrics([nas_root], ingest, completed=False)
        error = f"{type(exc).__name__}: {exc}"
        _update(run_id, state="failed", progress=1.0, error=error)
        _update_queue_recovery_state(nas_root, "failed", error=error)
    finally:
        upload_session_id = (ingest or {}).get("upload_session_id")
        if upload_session_id and _upload_sessions is not None:
            _upload_sessions.release(str(upload_session_id))


def _execute(
    run_id: str,
    manifest: RunManifest,
    settings: dict[str, Any],
    nas_root: Path,
    ingest: dict[str, Any] | None = None,
) -> None:
    _update(run_id, state="queued", progress=0.0, message="等待可用计算资源")
    with _gpu_job_lock:
        _execute_now(run_id, manifest, settings, nas_root, ingest)


def _execute_fixed_benchmark_now(
    run_id: str,
    settings: dict[str, Any],
    nas_root: Path,
    timing: dict[str, Any],
) -> None:
    fixed_root, _, history_root = fixed_archive_staging_paths(
        settings, _BENCHMARK_ARCHIVE_NAME, run_id
    )

    def progress(stage: str, value: float, message: str) -> None:
        if stage == "completed":
            stage = "finalizing"
            value = min(float(value), 0.999)
            message = "派生结果已通过验收，正在发布固定基准档案"
        _update(run_id, state=stage, progress=value, message=message, nas_output=str(fixed_root))

    try:
        manifest, manifest_path, ingest = prepare_from_nas_index(
            settings,
            _BENCHMARK_EXPERIMENT_ID,
            lambda message: _update(
                run_id,
                state="nas_ingest",
                progress=0.01,
                message=message,
                nas_output=str(fixed_root),
            ),
        )
        manifest_path.write_text(
            yaml.safe_dump(manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        timing.update(
            {
                "ingest_completed_at": datetime.now().astimezone().isoformat(),
                "duration_seconds": round(
                    time.perf_counter() - float(timing["request_started_perf"]), 6
                ),
                "manifest": str(manifest_path),
                "source_count": len(manifest.views),
                "input_reused": bool(
                    ingest.get("source_validation", {}).get("missing_count") == 0
                ),
                "ingest_details": ingest,
            }
        )
        _update(run_id, state="running", progress=0.02, nas_output=str(fixed_root))
        output = EvidencePipeline(settings, progress).run(manifest)
        if _record_partial_completion(run_id, Path(output)):
            _append_fixed_benchmark_metrics([Path(output)], timing, completed=False)
            write_partial_delivery(Path(output), _merged_run_metrics(Path(output)))
            return
        _append_fixed_benchmark_metrics([Path(output), nas_root], timing, completed=True)
        receipt = promote_fixed_archive(
            nas_root,
            fixed_root,
            history_root,
            {
                "metric_key": "fixed_benchmark_end_to_end",
                "definition": "benchmark request + NAS index preparation + analysis + atomic fixed archive publication",
                "request_received_at": timing.get("request_received_at"),
                "request_started_epoch": timing.get("request_started_epoch"),
                "experiment_id": _BENCHMARK_EXPERIMENT_ID,
                "archive_name": _BENCHMARK_ARCHIVE_NAME,
            },
        )
        _update(
            run_id,
            state="completed",
            progress=1.0,
            output=str(output),
            nas_output=str(fixed_root),
            nas_staging=str(nas_root),
            observability_root=str(fixed_root),
            promotion=receipt,
            archive_url=f"/#/archive/{quote(_BENCHMARK_ARCHIVE_NAME)}/experiments",
        )
    except Exception as exc:
        timing.setdefault(
            "duration_seconds",
            round(time.perf_counter() - float(timing["request_started_perf"]), 6),
        )
        with _lock:
            failure_stage = _runs.get(run_id, {}).get("state")
        timing.setdefault("failure_stage", failure_stage)
        _append_fixed_benchmark_metrics([nas_root], timing, completed=False)
        _update(run_id, state="failed", progress=1.0, error=f"{type(exc).__name__}: {exc}")


def _execute_fixed_benchmark(
    run_id: str,
    settings: dict[str, Any],
    nas_root: Path,
    timing: dict[str, Any],
) -> None:
    _update(run_id, state="queued", progress=0.0, message="等待可用计算资源")
    with _gpu_job_lock:
        _execute_fixed_benchmark_now(run_id, settings, nas_root, timing)


def _preflight_and_seal_collection_input(
    *,
    run_id: str,
    source_experiment_id: str,
    archive_name: str,
    manifest: RunManifest,
    manifest_path: Path,
    ingest: dict[str, Any],
    settings: dict[str, Any],
    staging_root: Path,
    fixed_root: Path,
    history_root: Path,
    timing: dict[str, Any],
) -> None:
    _update(
        run_id,
        state="input_preflight",
        progress=0.015,
        message="输入已封存，正在读取媒体头与时钟首尾，尚未占用 GPU",
        nas_output=str(fixed_root),
        nas_staging=str(staging_root),
    )
    input_preflight = preflight_manifest_inputs(manifest, settings)
    preflight_path = (
        staging_root
        / "JSON-Config-Files"
        / "Input-Manifests"
        / "prequeue_input_preflight.json"
    )
    _write_json_atomic(preflight_path, input_preflight)
    existing_seal_path = (
        staging_root / "JSON-Config-Files" / "Input-Manifests" / "input_seal.json"
    )
    # Local zero-copy ingestion keeps its first manifest/seal in Runtime. Bind
    # the recovery receipt to an archive-owned metadata copy, never to ../ paths.
    source_seal_path = Path(str((ingest.get("original_retention") or {}).get("input_seal") or existing_seal_path))
    existing_seal = _read_json(source_seal_path, {}) or {}
    if not verify_input_seal(existing_seal) or RunManifest.model_validate(existing_seal.get("manifest")).model_dump(mode="json") != manifest.model_dump(mode="json"):
        raise ValueError("采集批次输入封条无效或与本次清单不一致，未启动分析。")
    archived_manifest_path = staging_root / "JSON-Config-Files" / "input_manifest.yaml"
    archived_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archived_manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    refreshed_seal = build_input_seal(
        manifest,
        source_mode="nas_segmented_virtual_timeline",
        sources=list(existing_seal.get("sources") or []),
        role_resolution=existing_seal.get("role_resolution") or {},
        copied_source_bytes=0,
        preflight=input_preflight,
    )
    write_input_seal(existing_seal_path, refreshed_seal)
    _write_json_atomic(staging_root / "JSON-Config-Files" / "view_role_resolution.json",
                       refreshed_seal["role_resolution"])
    ingest["archived_input_manifest"] = str(archived_manifest_path)
    ingest.setdefault("original_retention", {}).update(
        source_input_seal=str(source_seal_path), input_seal=str(existing_seal_path))
    timing["manifest"] = str(archived_manifest_path)
    ingest["prequeue_input_preflight"] = {
        "status": input_preflight["status"],
        "receipt": str(preflight_path),
    }
    ingest.setdefault("original_retention", {})["input_seal_sha256"] = refreshed_seal[
        "seal_sha256"
    ]
    _write_queue_recovery_receipt(
        staging_root,
        run_id=run_id,
        state="running",
        manifest_path=archived_manifest_path,
        input_seal_path=existing_seal_path,
        ingest=ingest,
        job_kind="index_collection",
        recovery_context={
            "source_experiment_id": source_experiment_id,
            "archive_name": archive_name,
            "staging_root": str(staging_root),
            "fixed_root": str(fixed_root),
            "history_root": str(history_root),
            "timing": timing,
        },
    )


def _execute_index_collection_now(
    run_id: str,
    source_experiment_id: str,
    archive_name: str,
    settings: dict[str, Any],
    staging_root: Path,
    fixed_root: Path,
    history_root: Path,
    timing: dict[str, Any],
) -> None:
    def progress(stage: str, value: float, message: str) -> None:
        if stage == "completed":
            stage = "finalizing"
            value = min(float(value), 0.999)
            message = "派生结果已通过验收，正在发布采集批次正式档案"
        _update(
            run_id,
            state=stage,
            progress=value,
            message=message,
            nas_output=str(fixed_root),
            nas_staging=str(staging_root),
        )

    try:
        if directory_ingest_enabled(settings):
            validate_selection(settings, source_experiment_id)
            settings["storage"]["index_csv"] = str(
                selection_path(settings, source_experiment_id, ".csv")
            )
        record_collection_state(
            settings,
            source_experiment_id,
            archive_name=archive_name,
            run_id=run_id,
            state="processing",
            details={"staging": str(staging_root)},
        )
        manifest, manifest_path, ingest = prepare_from_nas_index(
            settings,
            source_experiment_id,
            lambda message: _update(
                run_id,
                state="nas_ingest",
                progress=0.01,
                message=message,
                nas_output=str(fixed_root),
                nas_staging=str(staging_root),
            ),
        )
        manifest_path.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        timing.update(
            {
                "ingest_completed_at": datetime.now().astimezone().isoformat(),
                "ingest_duration_seconds": round(
                    time.perf_counter() - float(timing["request_started_perf"]), 6
                ),
                "manifest": str(manifest_path),
                "source_count": len(manifest.views),
                "ingest_details": ingest,
            }
        )
        if isinstance(manifest, RunManifest):
            _preflight_and_seal_collection_input(
                run_id=run_id,
                source_experiment_id=source_experiment_id,
                archive_name=archive_name,
                manifest=manifest,
                manifest_path=manifest_path,
                ingest=ingest,
                settings=settings,
                staging_root=staging_root,
                fixed_root=fixed_root,
                history_root=history_root,
                timing=timing,
            )
        _update(
            run_id,
            state="running",
            progress=0.02,
            nas_output=str(fixed_root),
            nas_staging=str(staging_root),
        )
        with _gpu_job_lock:
            output = EvidencePipeline(settings, progress).run(manifest)
        if _record_partial_completion(run_id, Path(output)):
            _append_collection_index_metrics([Path(output)], timing, completed=False)
            write_partial_delivery(Path(output), _merged_run_metrics(Path(output)))
            record_collection_state(
                settings, source_experiment_id, archive_name=archive_name,
                run_id=run_id, state="partial",
                details={"staging": str(staging_root), "evidence_classification": "PARTIAL_EVIDENCE"},
            )
            return
        _append_collection_index_metrics(
            [Path(output), staging_root], timing, completed=True
        )
        _update_queue_recovery_state(staging_root, "completed")
        promotion = promote_fixed_archive(
            staging_root,
            fixed_root,
            history_root,
            {
                "metric_key": "collection_end_to_end",
                "definition": "collection selection + zero-copy NAS ingest + analysis + atomic formal archive publication",
                "request_received_at": timing.get("request_received_at"),
                "request_started_epoch": timing.get("request_started_epoch"),
                "source_experiment_id": timing.get("source_experiment_id"),
                "archive_name": timing.get("archive_name"),
            },
        )
        record_collection_state(
            settings,
            source_experiment_id,
            archive_name=archive_name,
            run_id=run_id,
            state="archived",
            details={
                "formal_archive": str(fixed_root),
                "promotion_verification": promotion.get("verification"),
            },
        )
        _update(
            run_id,
            state="completed",
            progress=1.0,
            output=str(fixed_root),
            nas_output=str(fixed_root),
            nas_staging=str(staging_root),
            promotion=promotion,
            archive_url=f"/#/archive/{quote(archive_name)}/experiments",
        )
    except Exception as exc:
        timing.setdefault(
            "ingest_duration_seconds",
            round(time.perf_counter() - float(timing["request_started_perf"]), 6),
        )
        with _lock:
            timing.setdefault("failure_stage", _runs.get(run_id, {}).get("state"))
        _append_collection_index_metrics([staging_root], timing, completed=False)
        try:
            record_collection_state(
                settings,
                source_experiment_id,
                archive_name=archive_name,
                run_id=run_id,
                state="failed",
                details={
                    "failure_stage": timing.get("failure_stage"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "staging": str(staging_root),
                },
            )
        except Exception as state_exc:
            timing["failure_state_record_error"] = (
                f"{type(state_exc).__name__}: {state_exc}"
            )
        error = f"{type(exc).__name__}: {exc}"
        _update_queue_recovery_state(staging_root, "failed", error=error)
        if timing.get("failure_state_record_error"):
            error += (
                "; collection_state_audit_failed="
                f"{timing['failure_state_record_error']}"
            )
        _update(
            run_id,
            state="failed",
            progress=1.0,
            error=error,
            failure_audit={
                "collection_state_recorded": not bool(
                    timing.get("failure_state_record_error")
                ),
                "collection_state_error": timing.get(
                    "failure_state_record_error"
                ),
            },
        )


def _execute_index_collection(
    run_id: str,
    source_experiment_id: str,
    archive_name: str,
    settings: dict[str, Any],
    staging_root: Path,
    fixed_root: Path,
    history_root: Path,
    timing: dict[str, Any],
) -> None:
    _update(
        run_id,
        state="queued",
        progress=0.0,
        message="等待可用计算资源",
        nas_output=str(fixed_root),
        nas_staging=str(staging_root),
    )
    _execute_index_collection_now(
        run_id,
        source_experiment_id,
        archive_name,
        settings,
        staging_root,
        fixed_root,
        history_root,
        timing,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_web_root / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


def _require_local_ai_settings(request: Request) -> None:
    if not ai_settings.enabled():
        raise HTTPException(404, "当前服务未启用本机 AI 设置。")
    identity = getattr(request.state, "web_identity", {})
    if (not request.client or request.client.host not in {"127.0.0.1", "::1"}
            or request.url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or identity.get("role") != "admin"):
        raise HTTPException(403, "请从本机工作台以管理员身份设置 AI 服务。")
    origin = request.headers.get("origin")
    expected = f"{request.url.scheme}://{request.url.netloc}"
    if (origin and origin != expected) or request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "AI 设置请求来源无效。")
    if request.method == "POST" and (origin != expected or request.headers.get("content-type", "").split(";", 1)[0] != "application/json"):
        raise HTTPException(403, "请通过本机 AI 服务设置页面提交。")


@app.get("/api/ai-settings")
def get_ai_settings(request: Request) -> JSONResponse:
    _require_local_ai_settings(request)
    try:
        result = ai_settings.public_settings(_settings())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(503, "本机 AI 配置无法读取，请检查本地配置目录权限。") from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.post("/api/ai-settings/models")
@app.post("/api/ai-settings/verify")
async def verify_ai_settings(request: Request) -> JSONResponse:
    _require_local_ai_settings(request)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 16384:
            raise HTTPException(413, "AI 配置请求过大。")
    try:
        value = json.loads(raw)
        connection = value.get("connection")
        key = value.get("api_key", "")
        if not isinstance(connection, dict) or not isinstance(key, str) or len(key) > 4096 or any(ord(c) < 32 for c in key):
            raise ValueError("invalid settings")
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(400, "请填写有效的厂商、视觉模型和 API 密钥。") from exc
    try:
        action = ai_settings.discover_available_models if request.url.path.endswith("/models") else ai_settings.verify_and_activate
        result = await run_in_threadpool(action, connection, key, _settings())
    except ai_settings.DiscoveryError as exc:
        raise HTTPException(400, str(exc)) from exc
    except BlockingIOError as exc:
        raise HTTPException(409, "已有连接正在验证，请稍候。") from exc
    except ValueError as exc:
        raise HTTPException(400, "请核对厂商、HTTPS 接口地址、视觉模型及 API 密钥。") from exc
    except (OSError, RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(503, "连接验证或配置保存未完成，请重试并检查本机配置目录权限。") from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = _settings()
    queue_stats = _persistent_queue.stats() if _persistent_queue is not None else None
    upload_stats = _upload_sessions.stats() if _upload_sessions is not None else None
    upload_policy = _upload_policy(settings)
    storage_mode = "nas" if settings["storage"].get("sync_to_nas") else "local"
    archive_root = _archive_root(settings)
    input_mode = (
        "NAS 采集批次 / zero-copy virtual timeline"
        if storage_mode == "nas"
        else "local files / zero-copy source references"
    )
    analysis_ready = True
    if storage_mode == "nas":
        try:
            audit_production_model_certification(settings)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            analysis_ready = False
    analysis_blocker = "分析模型待质量验收" if not analysis_ready else None
    if storage_mode == "nas" and not _nas_storage_available(settings):
        analysis_ready = False
        analysis_blocker = "NAS 归档或缓存目录不可用"
    with _nas_monitor_lock:
        monitor_snapshot = _nas_monitor_snapshot or {}
        monitor = dict(monitor_snapshot.get("monitor") or {})
    return {
        "status": "ok",
        "product_name": PRODUCT_NAME,
        "analysis_ready": analysis_ready,
        "speech_recognition": {"enabled": speech.enabled(settings)},
        "run_purpose": settings.get("project", {}).get(
            "run_purpose", "production"
        ),
        "analysis_blocker": analysis_blocker,
        "storage_mode": storage_mode,
        "archive_label": "NAS 正式归档" if storage_mode == "nas" else "任务归档",
        "web_upload_retention_mode": settings["storage"].get(
            "web_upload_retention_mode", "local_and_nas"
        ),
        "capacity_policy": "dynamic_by_submitted_files",
        "view_count_policy": "dynamic",
        "minimum_cross_view_sources": 2,
        "max_concurrent_archive_streams": _ARCHIVE_STREAM_LIMIT,
        "web_access": validate_web_access_configuration(),
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
        "mllm_connection": connection_health(settings["mllm"], key_configured(settings["mllm"])),
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
            "enabled": not bool(settings.get("project", {}).get("portable_desktop")),
            "experiment_id": _BENCHMARK_EXPERIMENT_ID,
            "archive_name": _BENCHMARK_ARCHIVE_NAME,
            "submission_protocol_version": _BENCHMARK_SUBMISSION_PROTOCOL_VERSION,
            "index_csv": str(settings["storage"]["index_csv"]),
            "input_mode": input_mode,
            "local_runtime_root": str(settings["storage"]["local_runtime_root"]),
            "local_cache_root": str(settings["storage"]["local_cache_root"]),
        },
        "collection_ingest": {
            "enabled": bool((settings.get("collection_ingest") or {}).get("enabled", True)),
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
            "persistence": "sqlite" if _persistent_queue is not None else "not_initialized",
            "survives_web_service_restart": _persistent_queue is not None,
            "archive_receipt_disaster_recovery": True,
            "database": (
                str(_persistent_queue.database) if _persistent_queue is not None else None
            ),
            "counts": queue_stats,
            "gpu_busy": bool(
                _gpu_job_lock.locked()
                or (queue_stats is not None and queue_stats["running"] > 0)
            ),
        },
    }


@app.get("/api/annotation-workspace")
def annotation_workspace(
    priority: str | None = None,
    review_status: str | None = None,
    q: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=200),
) -> dict[str, Any]:
    settings = _settings()
    if not bool((settings.get("developer_tools") or {}).get("yolo_annotation_workspace_enabled")):
        raise HTTPException(404, "内部 YOLO 标注工具未启用")
    try:
        return load_annotation_workspace(
            settings,
            priority=priority,
            review_status=review_status,
            query=q,
            offset=offset,
            limit=limit,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"无法读取 YOLO 标注工作台：{exc}") from exc


@app.get("/api/annotation-workspace/items/{item_id}/image")
def annotation_workspace_image(item_id: str) -> FileResponse:
    settings = _settings()
    if not bool((settings.get("developer_tools") or {}).get("yolo_annotation_workspace_enabled")):
        raise HTTPException(404, "内部 YOLO 标注工具未启用")
    try:
        path = resolve_annotation_image(settings, item_id)
    except KeyError as exc:
        raise HTTPException(404, "标注项不存在") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "badcase 图片不存在") from exc
    return FileResponse(path)


@app.post("/api/annotation-workspace/items/{item_id}/decision")
def save_annotation_workspace_decision(
    item_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    settings = _settings()
    if not bool((settings.get("developer_tools") or {}).get("yolo_annotation_workspace_enabled")):
        raise HTTPException(404, "内部 YOLO 标注工具未启用")
    try:
        decision = record_annotation_decision(settings, item_id, payload)
    except KeyError as exc:
        raise HTTPException(404, "标注项不存在") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "saved", "decision": decision}


@app.get("/api/annotation-workspace/export")
def annotation_workspace_export() -> dict[str, Any]:
    settings = _settings()
    if not bool((settings.get("developer_tools") or {}).get("yolo_annotation_workspace_enabled")):
        raise HTTPException(404, "内部 YOLO 标注工具未启用")
    try:
        return export_reviewed_ground_truth(settings)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"无法导出审核真值：{exc}") from exc


@app.get("/api/archives")
def list_archives(
    q: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    root = _archive_root()
    if not root.is_dir():
        return {
            "archive_root": str(root),
            "archives": [],
            "count": 0,
            "total_count": 0,
            "totals": {"experiments": 0, "key_events": 0},
            "next_cursor": None,
        }
    cursor_filters = {"q": q}
    decoded = None
    if cursor:
        try:
            decoded = decode_cursor(cursor, namespace="archives", filters=cursor_filters)
            if set(decoded) != {"modified_epoch_us", "name"}:
                raise ValueError("archive cursor position is invalid")
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, "无效的实验档案分页 cursor") from exc
    try:
        database = _ensure_archive_read_catalog(root)
        rows, total_count, totals = list_catalog_archives(
            database,
            query=q,
            after_modified_epoch_us=(int(decoded["modified_epoch_us"]) if decoded else None),
            after_name=(str(decoded["name"]) if decoded else None),
            limit=limit + 1,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise HTTPException(503, f"无法读取实验档案目录索引: {exc}") from exc
    has_more = len(rows) > limit
    page = rows[:limit]
    archives = [
        {
            **{key: value for key, value in row.items() if key not in {"source_revision", "modified_epoch_us", "published_epoch", "catalog_error"}},
            "has_model_understanding": bool(row["has_model_understanding"]),
            "has_daily_report": bool(row["has_daily_report"]),
            "has_evidence_index": bool(row["has_evidence_index"]),
            "formal_accuracy_claim_allowed": (
                bool(row["formal_accuracy_claim_allowed"])
                if row["formal_accuracy_claim_allowed"] is not None
                else None
            ),
            "catalog_status": "ready" if not row.get("catalog_error") else "degraded",
        }
        for row in page
    ]
    next_cursor = None
    if has_more and page:
        last = page[-1]
        next_cursor = encode_cursor(
            namespace="archives",
            position={
                "modified_epoch_us": int(last["modified_epoch_us"]),
                "name": str(last["name"]),
            },
            filters=cursor_filters,
        )
    return {
        "archive_root": str(root),
        "archives": archives,
        "count": len(archives),
        "total_count": total_count,
        "totals": totals,
        "has_more": has_more,
        "next_cursor": next_cursor,
        "canonical_source": "formal archive JSON and per-archive SQLite",
        "catalog_is_rebuildable": True,
    }


def _collection_card(collection: dict[str, Any]) -> dict[str, Any]:
    return {
        key: collection.get(key)
        for key in (
            "collection_id",
            "experiment_id",
            "display_name",
            "status",
            "sealed",
            "ready_to_analyze",
            "recording_start_time",
            "recording_end_time",
            "duration_seconds",
            "camera_count",
            "video_segment_count",
            "clock_segment_count",
            "declared_view_counts",
            "resolved_view_counts",
            "blocking_issue_count",
            "warning_count",
            "approved_override_count",
            "fingerprint_sha256",
            "processing",
        )
    } | {
        "blocking_issue_codes": sorted(
            {str(item.get("code")) for item in collection.get("blocking_issues") or []}
        ),
        "warning_codes": sorted(
            {str(item.get("code")) for item in collection.get("warnings") or []}
        ),
    }


@app.get("/api/collections")
def list_collections(
    status: str | None = None,
    q: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
    cursor: str | None = None,
) -> dict[str, Any]:
    cursor_filters = {"status": status, "q": q}
    decoded = None
    if cursor:
        try:
            decoded = decode_cursor(
                cursor,
                namespace="collections",
                filters=cursor_filters,
            )
            if set(decoded) != {"recording_start_time", "experiment_id"}:
                raise ValueError("collection cursor position is invalid")
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, "无效的采集批次分页 cursor") from exc
    try:
        payload = discover_collections(
            _settings(),
            status=status,
            query=q,
            limit=limit + 1,
            after_recording_start_time=(
                str(decoded["recording_start_time"]) if decoded else None
            ),
            after_experiment_id=(str(decoded["experiment_id"]) if decoded else None),
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"无法读取采集批次索引: {exc}") from exc
    has_more = len(payload["collections"]) > limit
    source_page = payload["collections"][:limit]
    collections = [_collection_card(item) for item in source_page]
    next_cursor = None
    if has_more and source_page:
        last = source_page[-1]
        next_cursor = encode_cursor(
            namespace="collections",
            position={
                "recording_start_time": str(last.get("recording_start_time") or ""),
                "experiment_id": str(last["experiment_id"]),
            },
            filters=cursor_filters,
        )
    counts = Counter(item["status"] for item in collections)
    processing_counts = Counter(
        str((item.get("processing") or {}).get("state") or "not_processed")
        for item in collections
    )
    return {
        "schema_version": payload["schema_version"],
        "generated_at": payload["generated_at"],
        "index": payload["index"],
        "registry": payload["registry"],
        "monitoring_policy": payload["monitoring_policy"],
        "status_counts": dict(counts),
        "processing_status_counts": dict(processing_counts),
        "collection_count": len(collections),
        "total_count": int(payload.get("total_count") or len(collections)),
        "has_more": has_more,
        "next_cursor": next_cursor,
        "collections": collections,
    }


@app.get("/api/collections/{experiment_id}")
def collection_detail(experiment_id: str) -> dict[str, Any]:
    try:
        return get_collection(_settings(), experiment_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"无法读取采集批次索引: {exc}") from exc


def _search_archive_roots(archive_name: str | None) -> list[tuple[str, Path]]:
    if archive_name:
        return [(archive_name, _resolve_archive(archive_name))]
    root = _archive_root()
    if not root.is_dir():
        return []
    return sorted(
        (
            (item.name, item)
            for item in root.iterdir()
            if item.is_dir()
            and (item / "JSON-Config-Files" / INDEX_DB_NAME).is_file()
        ),
        key=lambda item: item[0],
    )


def _decode_event_cursor(
    value: str | None, filters: dict[str, Any]
) -> tuple[str, int, str] | None:
    if not value:
        return None
    try:
        payload = decode_cursor(value, namespace="key-events", filters=filters)
        if set(payload) != {"archive_id", "peak_timestamp_us", "event_uid"}:
            raise ValueError("key-event cursor position is invalid")
        return (
            str(payload["archive_id"]),
            int(payload["peak_timestamp_us"]),
            str(payload["event_uid"]),
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, "无效的关键事件分页 cursor") from exc


def _encode_event_cursor(
    archive_id: str,
    peak_timestamp_us: int,
    event_uid: str,
    filters: dict[str, Any],
) -> str:
    return encode_cursor(
        namespace="key-events",
        position={
            "archive_id": archive_id,
            "peak_timestamp_us": int(peak_timestamp_us),
            "event_uid": event_uid,
        },
        filters=filters,
    )


def _attach_index_urls(
    archive_name: str, event: dict[str, Any]
) -> dict[str, Any]:
    result = dict(event)
    result["archive_name"] = archive_name
    release_id = str(event.get("release_id") or "") or None
    result["event_url"] = f"/api/key-events/{quote(str(event['event_uid']))}?archive={quote(archive_name)}"
    result["artifact_references"] = [
        {
            **artifact,
            "url": (
                _file_url(archive_name, artifact["path"], release_id)
                if "://" not in str(artifact.get("path") or "")
                else None
            ),
            "sidecar_url": (
                _file_url(archive_name, artifact["sidecar_path"], release_id)
                if artifact.get("sidecar_path")
                else None
            ),
        }
        for artifact in event.get("artifact_references", [])
    ]
    aligned_frame = next(
        (
            item
            for item in result["artifact_references"]
            if item.get("artifact_type") == "key_frame"
            and item.get("view_role") == "aligned_first_third"
        ),
        None,
    )
    aligned_clip = next(
        (
            item
            for item in result["artifact_references"]
            if item.get("artifact_type") == "key_clip"
            and item.get("view_role") == "aligned_first_third"
        ),
        None,
    )
    result["aligned_frame_url"] = aligned_frame.get("url") if aligned_frame else None
    result["aligned_clip_url"] = aligned_clip.get("url") if aligned_clip else None
    result["dual_view_material_ready"] = bool(aligned_frame and aligned_clip)
    provenance = result.get("provenance") or {}
    result.setdefault(
        "experiment_group",
        {
            "group_id": result.get("parent_event_id"),
            "group_uid": result.get("parent_event_uid"),
            "name": provenance.get("experiment_name") or result.get("parent_event_id"),
        },
    )
    return result


def _decode_catalog_event_cursor(
    value: str | None, filters: dict[str, Any]
) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        payload = decode_cursor(value, namespace="catalog-key-events", filters=filters)
        if set(payload) != {
            "published_epoch_us",
            "relevance_score",
            "archive_name",
            "release_id",
            "peak_timestamp_us",
            "event_uid",
        }:
            raise ValueError("catalog key-event cursor position is invalid")
        return {
            "published_epoch_us": int(payload["published_epoch_us"]),
            "relevance_score": int(payload["relevance_score"]),
            "archive_name": str(payload["archive_name"]),
            "release_id": str(payload["release_id"] or "") or None,
            "peak_timestamp_us": int(payload["peak_timestamp_us"]),
            "event_uid": str(payload["event_uid"]),
        }
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, "无效的关键事件分页 cursor") from exc


def _encode_catalog_event_cursor(event: dict[str, Any], filters: dict[str, Any]) -> str:
    return encode_cursor(
        namespace="catalog-key-events",
        position={
            "published_epoch_us": int(event.get("published_epoch_us") or 0),
            "relevance_score": int(event.get("relevance_score") or 0),
            "archive_name": str(event["archive_name"]),
            "release_id": str(event.get("release_id") or ""),
            "peak_timestamp_us": int(event.get("peak_timestamp_us") or 0),
            "event_uid": str(event["event_uid"]),
        },
        filters=filters,
    )


def _attach_staging_index_urls(
    run_id: str, event: dict[str, Any]
) -> dict[str, Any]:
    result = dict(event)
    result["staging_run_id"] = run_id
    result["event_url"] = (
        f"/api/staging-runs/{quote(run_id)}/key-events/"
        f"{quote(str(event['event_uid']))}"
    )
    result["artifact_references"] = [
        {
            **artifact,
            "url": (
                _staging_file_url(run_id, artifact["path"])
                if "://" not in str(artifact.get("path") or "")
                else None
            ),
            "sidecar_url": (
                _staging_file_url(run_id, artifact["sidecar_path"])
                if artifact.get("sidecar_path")
                else None
            ),
        }
        for artifact in event.get("artifact_references", [])
    ]
    return result


@app.get("/api/key-events")
def search_key_events(
    archive: str | None = None,
    q: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    cross_view: bool | None = None,
    material_ready: bool | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Search one or every archive without loading monolithic event JSON arrays."""

    if archive:
        _resolve_archive(archive)

    cursor_filters = {
        "archive": archive,
        "q": q,
        "action_type": action_type,
        "parent_event_id": parent_event_id,
        "cross_view": cross_view,
        "material_ready": material_ready,
        "start_us": start_us,
        "end_us": end_us,
    }
    decoded_cursor = _decode_catalog_event_cursor(cursor, cursor_filters)
    try:
        database = _ensure_archive_read_catalog()
        if decoded_cursor and catalog_archive_release(
            database, decoded_cursor["archive_name"]
        ) != decoded_cursor["release_id"]:
            raise HTTPException(
                409,
                "分页期间实验档案已发布新版本，请从第一页重新查询",
            )
        items, total_count = search_catalog_events(
            database,
            archive_name=archive,
            query=q,
            action_type=action_type,
            parent_event_id=parent_event_id,
            cross_view=cross_view,
            material_ready=material_ready,
            start_us=start_us,
            end_us=end_us,
            after_position=(
                (
                    decoded_cursor["relevance_score"],
                    decoded_cursor["published_epoch_us"],
                    decoded_cursor["archive_name"],
                    decoded_cursor["peak_timestamp_us"],
                    decoded_cursor["event_uid"],
                )
                if decoded_cursor
                else None
            ),
            limit=limit + 1,
        )
    except HTTPException:
        raise
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        raise HTTPException(503, f"无法读取关键事件目录索引: {exc}") from exc
    has_more = len(items) > limit
    page = [
        _attach_index_urls(str(item["archive_name"]), item)
        for item in items[:limit]
    ]
    next_cursor = None
    if has_more and page:
        next_cursor = _encode_catalog_event_cursor(page[-1], cursor_filters)
    return {
        "items": page,
        "count": len(page),
        "total_count": total_count,
        "next_cursor": next_cursor,
        "canonical_source": "archived JSON",
        "index_is_rebuildable": True,
    }


@app.get("/api/staging-runs/{run_id}/key-events")
def search_staging_key_events(
    run_id: str,
    q: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    cross_view: bool | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Search one isolated NAS staging run without promoting it."""

    root = _resolve_staging_run(run_id)
    cursor_filters = {
        "staging_run_id": run_id,
        "q": q,
        "action_type": action_type,
        "parent_event_id": parent_event_id,
        "cross_view": cross_view,
        "start_us": start_us,
        "end_us": end_us,
    }
    decoded_cursor = _decode_event_cursor(cursor, cursor_filters)
    manifest = _read_json(
        root / "JSON-Config-Files" / INDEX_MANIFEST_NAME, {}
    ) or {}
    archive_id = str(manifest.get("archive_id") or run_id)
    after_peak_us = None
    after_event_uid = None
    if decoded_cursor:
        cursor_archive, after_peak_us, after_event_uid = decoded_cursor
        if cursor_archive != archive_id:
            raise HTTPException(400, "分页游标与 staging run 不匹配")
    items = search_archive_index(
        root,
        query=q,
        action_type=action_type,
        parent_event_id=parent_event_id,
        cross_view=cross_view,
        start_us=start_us,
        end_us=end_us,
        after_peak_us=after_peak_us,
        after_event_uid=after_event_uid,
        limit=limit + 1,
    )
    has_more = len(items) > limit
    page = [
        _attach_staging_index_urls(run_id, item)
        for item in items[:limit]
    ]
    next_cursor = None
    if has_more and page:
        last = page[-1]
        next_cursor = _encode_event_cursor(
            archive_id,
            int(last.get("peak_timestamp_us") or 0),
            str(last["event_uid"]),
            cursor_filters,
        )
    return {
        "items": page,
        "count": len(page),
        "next_cursor": next_cursor,
        "canonical_source": "isolated staging indexed JSON",
        "index_is_rebuildable": True,
        "formal_archive_promotion": False,
    }


@app.get("/api/staging-runs/{run_id}/key-events/{event_uid}")
def staging_key_event(run_id: str, event_uid: str) -> dict[str, Any]:
    root = _resolve_staging_run(run_id)
    event = get_indexed_event(root, event_uid)
    if event is None:
        raise HTTPException(404, "staging 关键事件索引记录不存在")
    return _attach_staging_index_urls(run_id, event)


@app.get("/api/physical-changes")
def indexed_physical_changes(
    archive: str | None = None,
    object_id: str | None = None,
    object_role: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Search observed state transitions without inferring unobserved states."""

    cursor_filters = {
        "archive": archive,
        "object_id": object_id,
        "object_role": object_role,
        "action_type": action_type,
        "parent_event_id": parent_event_id,
        "start_us": start_us,
        "end_us": end_us,
    }
    decoded = None
    if cursor:
        try:
            decoded = decode_cursor(
                cursor,
                namespace="physical-changes",
                filters=cursor_filters,
            )
            if set(decoded) != {"archive_id", "peak_timestamp_us", "change_uid"}:
                raise ValueError("physical change cursor position is invalid")
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, "无效的物理状态变化分页 cursor") from exc
    collected: list[dict[str, Any]] = []
    for archive_name, root in _search_archive_roots(archive):
        manifest = _read_json(
            root / "JSON-Config-Files" / INDEX_MANIFEST_NAME, {}
        ) or {}
        archive_id = str(manifest.get("archive_id") or archive_name)
        after_peak_us = None
        after_change_uid = None
        if decoded:
            cursor_archive = str(decoded["archive_id"])
            if archive_id < cursor_archive:
                continue
            if archive_id == cursor_archive:
                after_peak_us = int(decoded["peak_timestamp_us"])
                after_change_uid = str(decoded["change_uid"])
        items = search_physical_changes(
            root,
            object_id=object_id,
            object_role=object_role,
            action_type=action_type,
            parent_event_id=parent_event_id,
            start_us=start_us,
            end_us=end_us,
            after_peak_us=after_peak_us,
            after_change_uid=after_change_uid,
            limit=limit + 1,
        )
        collected.extend(
            {
                **item,
                "archive_name": archive_name,
                "event_url": (
                    f"/api/key-events/{quote(str(item['event_uid']))}"
                    f"?archive={quote(archive_name)}"
                ),
            }
            for item in items
        )
    collected.sort(
        key=lambda item: (
            str(item["archive_id"]),
            int(item["peak_timestamp_us"]),
            str(item["change_uid"]),
        )
    )
    has_more = len(collected) > limit
    page = collected[:limit]
    next_cursor = None
    if has_more and page:
        last = page[-1]
        next_cursor = encode_cursor(
            namespace="physical-changes",
            position={
                "archive_id": str(last["archive_id"]),
                "peak_timestamp_us": int(last["peak_timestamp_us"]),
                "change_uid": str(last["change_uid"]),
            },
            filters=cursor_filters,
        )
    return {
        "items": page,
        "count": len(page),
        "next_cursor": next_cursor,
        "canonical_source": "archived key-material event JSON",
        "projection_policy": "explicit before/after differences only; no gap-state inference",
        "index_is_rebuildable": True,
    }


@app.get("/api/key-events/{event_uid}")
def indexed_key_event(event_uid: str, archive: str | None = None) -> dict[str, Any]:
    for archive_name, root in _search_archive_roots(archive):
        event = get_indexed_event(root, event_uid)
        if event is not None:
            return _attach_index_urls(archive_name, event)
    raise HTTPException(404, "关键事件索引记录不存在")


@app.get("/api/evidence/{evidence_uid:path}")
def indexed_evidence(evidence_uid: str, archive: str | None = None) -> dict[str, Any]:
    for archive_name, root in _search_archive_roots(archive):
        evidence = get_indexed_evidence(root, evidence_uid)
        if evidence is not None:
            return {
                **evidence,
                "archive_name": archive_name,
                "event_url": (
                    f"/api/key-events/{quote(str(evidence['event_uid']))}"
                    f"?archive={quote(archive_name)}"
                ),
            }
    raise HTTPException(404, "证据索引记录不存在")


def _event_has_cross_view_support(event: dict[str, Any]) -> bool:
    return any(
        association.get("both_views_support_action") is True
        for association in event.get("cross_view_associations", [])
    )


def _event_has_aligned_dual_view_material(event: dict[str, Any]) -> bool:
    if event.get("aligned_frame_url") and event.get("aligned_clip_url"):
        return True
    frame_roles = {item.get("view_role") for item in event.get("key_frames", [])}
    clip_roles = {item.get("view_role") for item in event.get("key_clips", [])}
    return "aligned_first_third" in frame_roles and "aligned_first_third" in clip_roles


def _derive_archived_quality_summary(
    package: dict[str, Any],
    key_events: list[dict[str, Any]],
    evidence_eval: dict[str, Any],
) -> dict[str, Any]:
    """Expose verified legacy evidence without inventing boundary accuracy metrics."""

    action_types = sorted(
        {
            str(event.get("action_type"))
            for event in key_events
            if event.get("action_type")
        }
    )
    cross_view_supported_count = sum(
        1 for event in key_events if _event_has_cross_view_support(event)
    )
    dual_view_material_count = sum(
        1 for event in key_events if _event_has_aligned_dual_view_material(event)
    )
    event_count = len(key_events)
    evidence_checks = [
        check
        for check in evidence_eval.get("checks", [])
        if check.get("check") == "cross_view_or_explicit_uncertainty"
    ]
    explicit_evidence_count = sum(check.get("passed") is True for check in evidence_checks)
    evaluation_passed = evidence_eval.get("passed") is True
    return {
        "schema_version": "visioncortex-archive-quality-display/1",
        "status": (
            "evidence_package_passed_no_boundary_ground_truth"
            if evaluation_passed
            else "structural_evidence_only"
        ),
        "source": "derived_from_archived_evidence",
        "display_note": (
            "历史档案没有人工边界基线；系统仅展示可从证据包复算的结构、媒体与跨视角结果，"
            "不虚构 Precision、Recall 或边界通过率。"
        ),
        "experiment_boundaries": {
            "evaluated": False,
            "reason": "no_reviewed_boundary_ground_truth_in_archive",
            "structural_group_count": len(package.get("experiment_groups", [])),
            "evidence_package_eval_passed": evaluation_passed,
        },
        "key_materials": {
            "event_count": event_count,
            "cross_view_supported_count": cross_view_supported_count,
            "cross_view_supported_rate": (
                cross_view_supported_count / event_count if event_count else None
            ),
            "dual_view_material_count": dual_view_material_count,
            "dual_view_material_rate": (
                dual_view_material_count / event_count if event_count else None
            ),
            "missing_dual_view_material_count": event_count - dual_view_material_count,
            "cross_view_or_explicit_uncertainty_count": explicit_evidence_count,
            "evidence_package_eval_passed": evaluation_passed,
            "action_types_present": action_types,
            "missing_action_types": sorted(set(_PHYSICAL_ACTION_TYPES) - set(action_types)),
            "source": "event_cross_view_associations + evidence_package_eval.json",
        },
    }


def _attach_archive_performance_display(
    metrics: dict[str, Any], acceptance: dict[str, Any]
) -> None:
    benchmark = acceptance.get("preprocessing_full_run") or {}
    clean_run = acceptance.get("clean_package_run") or {}
    current_preprocessing = (metrics.get("preprocessing_sla") or {}).get("actual_seconds")
    if benchmark.get("seconds") is not None:
        metrics["display_preprocessing_seconds"] = benchmark["seconds"]
        metrics["display_preprocessing_source"] = "full_cold_start_benchmark"
    else:
        metrics["display_preprocessing_seconds"] = current_preprocessing
        metrics["display_preprocessing_source"] = "current_run"
    reuse_note = str(clean_run.get("note") or "")
    metrics["preprocessing_display"] = {
        "full_cold_start": {
            "seconds": benchmark.get("seconds"),
            "includes": benchmark.get("includes"),
            "measured": benchmark.get("seconds") is not None,
            "source": "acceptance_report.preprocessing_full_run",
        },
        "current_run": {
            "total_seconds": metrics.get("total_duration_seconds"),
            "preprocessing_seconds": current_preprocessing,
            "reuse_note": reuse_note or None,
            "reused_validated_cv_ledgers": "reused validated cv ledgers" in reuse_note.lower(),
            "source": "run_metrics + acceptance_report.clean_package_run",
        },
    }


def _summarize_key_material_verification(
    annotation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records = [
        item for item in annotation.get("records") or [] if isinstance(item, dict)
    ]
    by_event: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        event_id = str(record.get("event_id") or "").strip()
        if event_id:
            by_event.setdefault(event_id, []).append(record)
    event_summaries: dict[str, dict[str, Any]] = {}
    model_execution_counts: dict[str, int] = {}
    for event_id, event_records in by_event.items():
        views = []
        event_models: set[str] = set()
        confidences: list[float] = []
        inference_seconds = 0.0
        model_load_seconds = 0.0
        statuses: list[str] = []
        uncertainty_reasons: set[str] = set()
        for record in event_records:
            decision = record.get("selective_verification") or {}
            status = str(decision.get("status") or "not_recorded")
            statuses.append(status)
            assessment = decision.get("assessment") or {}
            uncertainty_reasons.update(str(item) for item in assessment.get("reasons") or [])
            if status == "deferred_budget_exhausted":
                uncertainty_reasons.add(str(decision.get("reason") or status))
            models = ["closed_set_yolo_tensorrt"]
            supplement = record.get("open_vocabulary_supplement") or {}
            if supplement.get("status") == "executed":
                models.append("yolo_world_v2")
                model_load_seconds += float(supplement.get("model_load_seconds") or 0.0)
                inference_seconds += float(supplement.get("inference_seconds") or 0.0)
            grounding = supplement.get("grounding_dino_fallback") or {}
            if grounding.get("status") == "executed":
                models.append("grounding_dino_base")
                model_load_seconds += float(grounding.get("model_load_seconds") or 0.0)
                inference_seconds += float(grounding.get("inference_seconds") or 0.0)
            segmentation = record.get("temporal_participant_segmentation") or {}
            if segmentation.get("status") == "completed":
                models.append("sam2_temporal_participant")
                inference_seconds += float(segmentation.get("inference_seconds") or 0.0)
            liquid = record.get("liquid_semantic_sidecar") or {}
            if liquid.get("status") == "completed":
                models.append("labpics_pspnet_liquid_semantic")
                inference_seconds += float(liquid.get("inference_seconds") or 0.0)
            for item in record.get("rendered_detections") or []:
                if item.get("confidence") is not None:
                    confidences.append(float(item["confidence"]))
            event_models.update(models)
            for model in models:
                model_execution_counts[model] = model_execution_counts.get(model, 0) + 1
            views.append(
                {
                    "view_id": record.get("view_id"),
                    "role_label": record.get("role_label"),
                    "status": status,
                    "models": models,
                    "rendered_classes": record.get("rendered_classes") or [],
                    "rendered_detections": record.get("rendered_detections") or [],
                    "minimum_rendered_confidence": record.get(
                        "minimum_rendered_confidence"
                    ),
                    "uncertainty_reasons": sorted(
                        str(item) for item in assessment.get("reasons") or []
                    ),
                }
            )
        if "deferred_budget_exhausted" in statuses:
            overall_status = "verification_deferred_budget_exhausted"
        elif "admitted" in statuses:
            overall_status = "secondary_verification_executed"
        elif statuses and all(
            item == "skipped_clear_closed_set_evidence" for item in statuses
        ):
            overall_status = "clear_closed_set_evidence"
        else:
            overall_status = "verification_recorded"
        event_summaries[event_id] = {
            "status": overall_status,
            "models": sorted(event_models),
            "view_count": len(views),
            "views": views,
            "confidence": {
                "minimum": round(min(confidences), 6) if confidences else None,
                "maximum": round(max(confidences), 6) if confidences else None,
            },
            "timing": {
                "model_load_seconds": round(model_load_seconds, 6),
                "inference_seconds": round(inference_seconds, 6),
            },
            "uncertain": bool(uncertainty_reasons),
            "uncertainty_reasons": sorted(uncertainty_reasons),
        }
    selective = annotation.get("selective_verification") or {}
    summary = {
        "available": bool(annotation),
        "mode": annotation.get("mode"),
        "event_count": annotation.get("event_count", len(event_summaries)),
        "rendered_view_count": annotation.get("rendered_view_count", len(records)),
        "policy": selective.get("policy"),
        "decision_status_counts": selective.get("decision_status_counts") or {},
        "model_execution_counts": dict(sorted(model_execution_counts.items())),
        "timing": {
            "wall_seconds": selective.get("wall_seconds"),
            "open_vocabulary_model_load_seconds": selective.get(
                "open_vocabulary_model_load_seconds"
            ),
            "open_vocabulary_inference_seconds": selective.get(
                "open_vocabulary_inference_seconds"
            ),
            "grounding_dino_model_load_seconds": selective.get(
                "grounding_dino_model_load_seconds"
            ),
            "grounding_dino_inference_seconds": selective.get(
                "grounding_dino_inference_seconds"
            ),
        },
        "budget": selective.get("budget") or {},
        "uncertain_event_count": sum(
            item["uncertain"] for item in event_summaries.values()
        ),
        "source_copy_bytes": selective.get("source_copy_bytes", 0),
        "ark_calls": selective.get("ark_calls", 0),
        "token_usage": selective.get("token_usage", 0),
    }
    return summary, event_summaries


def _archive_links(
    archive_name: str,
    root: Path,
    release_id: str | None,
    daily_manifest: dict[str, Any],
    *,
    staging_run_id: str | None = None,
) -> dict[str, str | None]:
    def existing(relative: str | None) -> str | None:
        if not relative:
            return None
        candidate = (root / relative).resolve()
        if not archive_contains(candidate, root.resolve()) or not candidate.is_file():
            return None
        return (
            _staging_file_url(staging_run_id, relative)
            if staging_run_id else _file_url(archive_name, relative, release_id)
        )

    return {
        "experiment_understanding": existing(
            "JSON-Config-Files/Experiment-Groups-Step-Level-Analysis.json"
        ),
        "key_material_understanding": existing(
            "Key-Materials/Key-Materials-Model-Understanding.json"
        ),
        "key_material_category_index": existing(
            "Key-Materials/Key-Material-Category-Index.json"
        ),
        "metrics": existing("JSON-Config-Files/run_metrics.json"),
        "resource_telemetry": existing("JSON-Config-Files/resource_telemetry.json"),
        "resource_telemetry_journal": existing("JSON-Config-Files/resource_telemetry.jsonl"),
        "operation_review": existing("JSON-Config-Files/operation_review.json"),
        "delivery_metrics": existing("JSON-Config-Files/delivery_metrics.json"),
        "acceptance": existing("JSON-Config-Files/acceptance_report.json"),
        "quality_acceptance": existing("JSON-Config-Files/quality_acceptance.json"),
        "evidence_package_eval": existing("JSON-Config-Files/evidence_package_eval.json"),
        "key_material_recall_eval": existing(
            "JSON-Config-Files/key_material_recall_eval.json"
        ),
        "daily_report_json": existing(daily_manifest.get("json")),
        "daily_report_markdown": existing(daily_manifest.get("markdown")),
        "daily_report_html": existing(daily_manifest.get("html")),
        "daily_report_pdf": existing(daily_manifest.get("pdf")),
        "daily_report_eval": existing(daily_manifest.get("evaluation")),
        "evidence_index_manifest": existing(
            f"JSON-Config-Files/{INDEX_MANIFEST_NAME}"
        ),
        "run_provenance": existing("JSON-Config-Files/run_provenance.json"),
        "project_annotation_comparison": existing("Project-Review/Project-Annotation-Comparison.html"),
        "final_key_material_annotation": existing(
            "JSON-Config-Files/final_key_material_annotation.json"
        ),
    }


def _movement_screening_payload(root: Path, report_url: str) -> dict[str, Any] | None:
    movement_report = _read_json(root / "JSON-Config-Files/movement_visual_verification.json", {}) or {}
    movement_screening = None
    if movement_report.get("enabled"):
        movement_screening = {
            "counts": movement_report.get("counts", {}),
            "duration_seconds": movement_report.get("duration_seconds"),
            "total_candidates": len(movement_report.get("candidates", [])),
            "candidates": [
                {key: item.get(key) for key in ("candidate_id", "view_id", "start_ms", "end_ms", "objects", "status")}
                for item in movement_report.get("candidates", [])[:200]
            ],
            "report_url": report_url,
            "physical_action_confirmed": False,
        }
    return movement_screening


def _archive_summary_payload(archive_name: str, root: Path) -> dict[str, Any]:
    pointer = read_current_release_pointer(root) or {}
    release_id = str(pointer.get("release_id") or "") or None
    json_root = root / "JSON-Config-Files"
    index_manifest = _read_json(json_root / INDEX_MANIFEST_NAME, {}) or {}
    daily_manifest = _read_json(json_root / "daily_report_manifest.json", {}) or {}
    quality = _read_json(json_root / "quality_acceptance.json", {}) or {}
    metrics = _merged_run_metrics(root)
    _attach_archive_performance_display(
        metrics,
        _read_json(json_root / "acceptance_report.json", {}) or {},
    )
    counts = index_manifest.get("counts") or {}
    manifest = _read_json(json_root / "run_manifest.json", {}) or {}
    experiment_root = root / "Experiment-Clips"
    experiment_count = (
        sum(1 for item in experiment_root.iterdir() if item.is_dir())
        if experiment_root.is_dir()
        else 0
    )
    return {
        "name": archive_name,
        "path": str(_archive_root() / archive_name),
        "network_path": str(root),
        "release_id": release_id,
        "current_release": pointer or None,
        "integrity_status": (
            lightweight_release_integrity(root, pointer)
            if pointer
            else "legacy_archive_not_release_verified"
        ),
        "counts": {
            "experiments": experiment_count,
            "key_events": int(counts.get("key_events") or 0),
        },
        "quality_acceptance": quality,
        "input_view_count": len(manifest.get("views") or []) or None,
        "metrics": metrics,
        "observability": {
            "status": _pipeline_status_from_root(root),
            "stage_receipts": _stage_receipts_from_root(root),
        },
        "evidence_index": {
            **index_manifest,
            "search_url": f"/api/key-events?archive={quote(archive_name)}",
        }
        if index_manifest
        else None,
        "movement_screening": _movement_screening_payload(
            root, _file_url(archive_name, "JSON-Config-Files/movement_visual_verification.json", release_id)
        ),
        "links": _archive_links(archive_name, root, release_id, daily_manifest),
    }


def _archive_section_payload(
    archive_name: str,
    root: Path,
    *,
    section: str,
    cursor: str | None,
    limit: int,
) -> dict[str, Any]:
    if section not in {"summary", "experiments", "reports", "metrics"}:
        raise HTTPException(400, "section 必须是 summary、experiments、reports 或 metrics")
    # Incomplete legacy runs have no final index yet. Retain their already-built
    # media and failure state instead of presenting an empty successful archive.
    status = _pipeline_status_from_root(root)
    if status.get("stage") in {"failed", "interrupted"}:
        return _archive_detail_from_root(root, archive_name)
    result = _archive_summary_payload(archive_name, root)
    if section == "summary":
        return result
    json_root = root / "JSON-Config-Files"
    if section == "experiments":
        release_id = result.get("release_id")
        filters = {"archive": archive_name, "release_id": release_id}
        offset = 0
        if cursor:
            try:
                decoded = decode_cursor(
                    cursor, namespace="archive-experiments", filters=filters
                )
                if set(decoded) != {"offset"}:
                    raise ValueError("experiment cursor position is invalid")
                offset = int(decoded["offset"])
            except (ValueError, TypeError, KeyError) as exc:
                raise HTTPException(400, "无效的实验片段分页 cursor") from exc
        analysis = _read_json(
            json_root / "Experiment-Groups-Step-Level-Analysis.json", {}
        ) or {}
        groups = list(analysis.get("experiment_groups") or [])
        if not groups:
            legacy_package = _read_json(json_root / "evidence_package.json", {}) or {}
            groups = list(legacy_package.get("experiment_groups") or [])
        source_page = groups[offset : offset + limit + 1]
        has_more = len(source_page) > limit
        source_page = source_page[:limit]
        experiments = []
        for group in source_page:
            folder_name = str(group.get("archive_folder") or "")
            understanding = group.get("model_understanding") or {}
            folder = root / "Experiment-Clips" / folder_name
            aligned_candidates = (
                folder / "Aligned_First+Third.mp4",
                folder / f"{group.get('group_id')}_aligned_multiview.mp4",
            )
            aligned = next((path for path in aligned_candidates if path.is_file()), None)
            experiments.append(
                {
                    "folder": folder_name,
                    "name": group.get("experiment_name") or folder_name,
                    "continuity_type": group.get("continuity_type"),
                    "workflow_kind": group.get("workflow_kind", "unresolved"),
                    "source_archive_folders": group.get("source_archive_folders") or [],
                    "workflow_units": group.get("workflow_units") or [],
                    "completion_status": group.get("completion_status", "unreviewed"),
                    "completion_reason": group.get("completion_reason") or "",
                    "boundary_extension_requires_step_review": group.get("boundary_extension_requires_step_review", False),
                    "view_timeline": group.get("view_timeline") or [],
                    "start_ms": group.get("global_start_ms"),
                    "end_ms": group.get("global_end_ms"),
                    "speech_interpretation": understanding.get("speech_interpretation"),
                    "speech_context": understanding.get("speech_context"),
                    "summary": understanding.get("overall_summary"),
                    "steps": understanding.get("steps") or [],
                    "uncertainties": understanding.get("uncertainties") or [],
                    "first_person_video_url": (
                        _file_url(archive_name, archive_relative_posix(folder / "First-Person.mp4", root), release_id)
                        if (folder / "First-Person.mp4").is_file() else None
                    ),
                    "third_person_video_url": (
                        _file_url(archive_name, archive_relative_posix(folder / "Third-Person.mp4", root), release_id)
                        if (folder / "Third-Person.mp4").is_file() else None
                    ),
                    "aligned_video_url": (
                        _file_url(
                            archive_name,
                            Path(archive_relative_posix(aligned, root)),
                            release_id,
                        )
                        if aligned
                        else None
                    ),
                }
            )
        result.update(
            {
                "experiments": experiments,
                "experiment_groups": [
                    {
                        "group_id": group.get("group_id"),
                        "name": group.get("experiment_name")
                        or group.get("archive_folder"),
                        "folder": group.get("archive_folder"),
                        "continuity_type": group.get("continuity_type"),
                        "workflow_kind": group.get("workflow_kind", "unresolved"),
                        "source_archive_folders": group.get("source_archive_folders") or [],
                        "workflow_units": group.get("workflow_units") or [],
                        "completion_status": group.get("completion_status", "unreviewed"),
                        "completion_reason": group.get("completion_reason") or "",
                        "boundary_extension_requires_step_review": group.get("boundary_extension_requires_step_review", False),
                        "view_timeline": group.get("view_timeline") or [],
                        "start_ms": group.get("global_start_ms"),
                        "end_ms": group.get("global_end_ms"),
                        "key_event_count": len(group.get("key_event_ids") or []),
                    }
                    for group in groups
                ],
                "next_cursor": (
                    encode_cursor(
                        namespace="archive-experiments",
                        position={"offset": offset + limit},
                        filters=filters,
                    )
                    if has_more
                    else None
                ),
                "has_more": has_more,
            }
        )
        return result
    if section in {"reports", "metrics"}:
        daily_manifest = _read_json(json_root / "daily_report_manifest.json", {}) or {}
        result.update(
            {
                "daily_report_manifest": daily_manifest,
                "daily_report": (
                    _read_json(root / daily_manifest["json"], {})
                    if daily_manifest.get("json")
                    else {}
                )
                or {},
            }
        )
        if section == "reports":
            return result
    metrics = _merged_run_metrics(root)
    acceptance = _read_json(json_root / "acceptance_report.json", {}) or {}
    _attach_archive_performance_display(metrics, acceptance)
    final_annotation = _read_json(
        json_root / "final_key_material_annotation.json", {}
    ) or {}
    key_material_verification, _ = _summarize_key_material_verification(final_annotation)
    result.update(
        {
            "metrics": metrics,
            "key_material_recall_eval": _read_json(
                json_root / "key_material_recall_eval.json", {}
            )
            or {},
            "observability": _run_snapshot_from_root(root),
            "key_material_verification": key_material_verification,
        }
    )
    return result


@app.get("/api/archives/{archive_name}")
def archive_detail(
    archive_name: str,
    section: str = "all",
    cursor: str | None = None,
    limit: int = Query(default=24, ge=1, le=100),
) -> dict[str, Any]:
    root = _resolve_archive(archive_name)
    if section != "all":
        return _archive_section_payload(
            archive_name, root, section=section, cursor=cursor, limit=limit
        )
    return _archive_detail_from_root(root, archive_name)


@app.get("/api/staging-runs/{run_id}/archive")
def staging_archive_detail(run_id: str, section: str = "all") -> dict[str, Any]:
    if section not in {"all", "library-materials", "library-reports"}:
        raise HTTPException(status_code=400, detail="Unsupported staging section")
    root = _resolve_staging_run(run_id)
    record = _runs.get(run_id) or {}
    name = record.get("experiment_id") or root.parent.name
    result = _archive_detail_from_root(root, name, staging_run_id=run_id, library_section=section if section != "all" else None)
    if record.get("read_only"):
        result.update(read_only=True, retry_available=False)
    return result


def _archive_detail_from_root(
    root: Path, archive_name: str, *, staging_run_id: str | None = None,
    library_section: str | None = None
) -> dict[str, Any]:
    release_pointer = read_current_release_pointer(root) or {}
    release_id = str(release_pointer.get("release_id") or "") or None

    def file_url(name: str, relative: str | Path) -> str:
        return (
            _staging_file_url(staging_run_id, relative)
            if staging_run_id else _file_url(name, relative, release_id)
        )

    index_manifest_path = root / "JSON-Config-Files" / INDEX_MANIFEST_NAME
    index_manifest = _read_json(index_manifest_path, {}) or {}
    snapshot = ({"status": _pipeline_status_from_root(root),
                 "stage_receipts": _stage_receipts_from_root(root),
                 "partial_delivery": _read_json(root / "JSON-Config-Files/partial_delivery.json", {}) or {}}
                if library_section else _run_snapshot_from_root(root))
    status = snapshot.get("status") or {}
    active_preview = bool(staging_run_id) and status.get("stage") not in {"completed", "partial", "failed", "interrupted", "cancelled"}
    completed_stages = {
        item["stage"] for item in snapshot.get("stage_receipts", [])
        if item.get("status") == "completed"
    }
    package = {} if library_section else _read_json(root / "JSON-Config-Files" / "evidence_package.json", {}) or {}
    if active_preview:
        package = {}
    metrics = {} if library_section else _merged_run_metrics(root)
    acceptance = _read_json(root / "JSON-Config-Files" / "acceptance_report.json", {}) or {}
    quality_path = root / "JSON-Config-Files" / "quality_acceptance.json"
    quality_acceptance = _read_json(quality_path, {}) or {}
    evidence_eval_path = root / "JSON-Config-Files" / "evidence_package_eval.json"
    evidence_eval = _read_json(evidence_eval_path, {}) or {}
    recall_eval_path = (
        root / "JSON-Config-Files" / "key_material_recall_eval.json"
    )
    key_material_recall_eval = _read_json(recall_eval_path, {}) or {}
    final_annotation_path = (
        root / "JSON-Config-Files" / "final_key_material_annotation.json"
    )
    final_annotation = {} if library_section else _read_json(final_annotation_path, {}) or {}
    if active_preview:
        quality_acceptance = {}
        evidence_eval = {}
        key_material_recall_eval = {}
        if "material_refinement" not in completed_stages:
            final_annotation = {}
    key_material_verification, verification_by_event = (
        _summarize_key_material_verification(final_annotation)
    )
    _attach_archive_performance_display(metrics, acceptance)
    key_events = _read_json(
        root / "Key-Materials" / "Key-Materials-Model-Understanding.json", []
    ) or []
    if active_preview and "mllm" not in completed_stages:
        key_events = []
    daily_manifest = _read_json(
        root / "JSON-Config-Files" / "daily_report_manifest.json", {}
    ) or {}
    if active_preview:
        daily_manifest = {}
    daily_report = (
        _read_json(root / daily_manifest["json"], {})
        if daily_manifest.get("json")
        else {}
    ) or {}
    package_groups = package.get("experiment_groups", [])
    if not package_groups:
        stage_groups = _read_json(
            root / "JSON-Config-Files" / "experiment_group_understanding.json", {}
        ) or {}
        package_groups = stage_groups.get("groups", [])
        if active_preview and _current_attempt_started_at(status) is not None and "experiment_understanding" not in completed_stages:
            package_groups = []
    from .speech_refresh import apply as apply_speech_revision
    package_groups = apply_speech_revision(root, package_groups)
    from .operation_review import apply as apply_operation_revision, coverage
    package_groups = apply_operation_revision(root, package_groups)
    group_by_folder = {
        str(group.get("archive_folder")): group for group in package_groups
    }
    group_by_id = {
        str(group.get("group_id")): group
        for group in package_groups
        if group.get("group_id")
    }
    experiments = []
    experiment_root = root / "Experiment-Clips"
    if experiment_root.is_dir():
        for folder in sorted(item for item in experiment_root.iterdir() if item.is_dir()):
            if (package_groups or active_preview) and folder.name not in group_by_folder:
                continue
            group = group_by_folder.get(folder.name, {})
            understanding = group.get("model_understanding") or {}
            first_person = folder / "First-Person.mp4"
            third_person = folder / "Third-Person.mp4"
            aligned = folder / "Aligned_First+Third.mp4"
            experiments.append(
                {
                    "folder": folder.name,
                    "group_id": group.get("group_id"),
                    "name": group.get("experiment_name") or folder.name,
                    "continuity_type": group.get("continuity_type"),
                    "workflow_kind": group.get("workflow_kind", "unresolved"),
                    "source_archive_folders": group.get("source_archive_folders") or [],
                    "workflow_units": group.get("workflow_units") or [],
                    "completion_status": group.get("completion_status", "unreviewed"),
                    "completion_reason": group.get("completion_reason") or "",
                    "boundary_extension_requires_step_review": group.get("boundary_extension_requires_step_review", False),
                    "view_timeline": group.get("view_timeline") or [],
                    "start_ms": group.get("global_start_ms"),
                    "end_ms": group.get("global_end_ms"),
                    "speech_interpretation": understanding.get("speech_interpretation"),
                    "speech_context": understanding.get("speech_context"),
                    "summary": understanding.get("overall_summary"),
                    "steps": understanding.get("steps") or [],
                    "operation_coverage": coverage(group, understanding.get("steps") or []),
                    "operation_review_accepted": (understanding.get("operation_review") or {}).get("accepted"),
                    "uncertainties": understanding.get("uncertainties") or [],
                    "first_person_video_url": file_url(
                        archive_name,
                        Path(archive_relative_posix(first_person, root)),
                    )
                    if first_person.is_file()
                    else None,
                    "third_person_video_url": file_url(
                        archive_name,
                        Path(archive_relative_posix(third_person, root)),
                    )
                    if third_person.is_file()
                    else None,
                    "aligned_video_url": file_url(
                        archive_name, Path(archive_relative_posix(aligned, root))
                    )
                    if aligned.is_file()
                    else None,
                }
            )
    normalized_events = []
    for event in key_events:
        group = group_by_id.get(str(event.get("parent_event_id")), {})
        frame = next(
            (
                item for item in event.get("key_frames", [])
                if item.get("view_role") == "aligned_first_third"
            ),
            None,
        )
        clip = next(
            (
                item for item in event.get("key_clips", [])
                if item.get("view_role") == "aligned_first_third"
            ),
            None,
        )
        normalized_events.append(
            {
                **event,
                "aligned_frame_url": file_url(archive_name, frame["path"]) if frame else None,
                "aligned_clip_url": file_url(archive_name, clip["path"]) if clip else None,
                "dual_view_material_ready": bool(frame and clip),
                "verification": verification_by_event.get(
                    str(event.get("event_id") or ""),
                    {
                        "status": "not_available_historical_archive",
                        "models": [],
                        "views": [],
                        "uncertain": True,
                        "uncertainty_reasons": [
                            "final_key_material_annotation_not_available"
                        ],
                    },
                ),
                "experiment_group": {
                    "group_id": group.get("group_id") or event.get("parent_event_id"),
                    "group_uid": group.get("group_uid")
                    or event.get("parent_event_uid"),
                    "name": group.get("experiment_name") or event.get("parent_event_id"),
                    "folder": group.get("archive_folder"),
                    "continuity_type": group.get("continuity_type"),
                    "workflow_kind": group.get("workflow_kind", "unresolved"),
                    "source_archive_folders": group.get("source_archive_folders") or [],
                    "workflow_units": group.get("workflow_units") or [],
                    "completion_status": group.get("completion_status", "unreviewed"),
                    "completion_reason": group.get("completion_reason") or "",
                    "boundary_extension_requires_step_review": group.get("boundary_extension_requires_step_review", False),
                    "view_timeline": group.get("view_timeline") or [],
                    "start_ms": group.get("global_start_ms"),
                    "end_ms": group.get("global_end_ms"),
                },
            }
        )
    if not quality_acceptance:
        quality_acceptance = _derive_archived_quality_summary(
            package, normalized_events, evidence_eval
        )
    else:
        quality_acceptance.setdefault("source", "quality_acceptance.json")
    key_quality = quality_acceptance.setdefault("key_materials", {})
    dual_view_material_count = sum(
        1 for event in normalized_events if event["dual_view_material_ready"]
    )
    key_quality.setdefault("event_count", len(normalized_events))
    key_quality.setdefault("dual_view_material_count", dual_view_material_count)
    key_quality.setdefault(
        "dual_view_material_rate",
        dual_view_material_count / len(normalized_events) if normalized_events else None,
    )
    key_quality.setdefault(
        "missing_dual_view_material_count",
        len(normalized_events) - dual_view_material_count,
    )
    links = _archive_links(
        archive_name, root, release_id, daily_manifest,
        staging_run_id=staging_run_id,
    )
    preliminary_materials = []
    if not normalized_events and snapshot.get("status", {}).get("stage") != "completed":
        completed = {item.get("stage") for item in snapshot.get("stage_receipts", []) if item.get("status") == "completed"}
        category_index = _read_json(root / "Key-Materials/Key-Material-Category-Index.json", {}) or {}
        if "key_materials" in completed:
            for group in category_index.get("experiments", []):
                for category in group.get("action_categories", []):
                    for event in category.get("events", []):
                        media = {}
                        for field, source in (("frame_url", "key_frames"), ("clip_url", "key_clips")):
                            relative = (event.get(source) or {}).get("aligned_first_third")
                            if relative:
                                candidate = (root / relative).resolve()
                                if archive_contains(candidate, root.resolve()) and candidate.is_file():
                                    media[field] = file_url(archive_name, relative)
                        if media:
                            preliminary_materials.append({
                                "event_id": event.get("event_id"),
                                "group_name": group.get("experiment_name"),
                                "timestamp_ms": event.get("peak_timestamp_us", 0) / 1000,
                                "review_status": "pending_semantic_review",
                                **media,
                            })
    quarantine_materials = []
    manifest = _read_json(root / "JSON-Config-Files/run_manifest.json", {}) or {}
    role_by_view = {view.get("view_id"): view.get("role") for view in manifest.get("views", [])}
    retained_candidates = {}
    for relative_index in (
        "Key-Materials/Machine-Quarantine/Machine-Quarantine-Index.json",
        "Key-Materials/Review-Candidates/Candidate-Index.json",
    ):
        index = _read_json(root / relative_index, {}) or {}
        for candidate in index.get("candidates", []):
            if candidate.get("event_id"):
                retained_candidates[str(candidate["event_id"])] = candidate
    quarantine_index = {"candidates": list(retained_candidates.values())}
    if active_preview and "material_refinement" not in completed_stages:
        quarantine_index = {}
    for event in quarantine_index.get("candidates", []):
        media = {}
        role_media: dict[str, dict[str, Any]] = {}
        pairing = event.get("view_pairing") or {}
        for relative in event.get("media", []):
            candidate = (root / relative).resolve()
            if not archive_contains(candidate, root.resolve()) or not candidate.is_file():
                continue
            if candidate.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                field = "frame_url"
            elif candidate.suffix.lower() == ".mp4":
                field = "clip_url"
            else:
                continue
            if field not in media or "aligned" in candidate.name.lower():
                media[field] = file_url(archive_name, relative)
            if candidate.stem in {"First-Person", "Third-Person"}:
                role_media.setdefault(candidate.stem, {})[field] = file_url(archive_name, relative)
        context_media = {}
        if pairing.get("pair_evidence_status") == "context_only_missing_key_time_support":
            # A same-time context camera is not a corresponding action view.
            # Keep it accessible separately instead of presenting a false pair.
            supported = {item["view_id"] for candidates in pairing.get("candidates", {}).values()
                         for item in candidates if item.get("candidate_supported_at_key")}
            primary_role = ("Third-Person" if pairing.get("third_person_view") in supported
                            and pairing.get("first_person_view") not in supported else "First-Person")
            other_role = "Third-Person" if primary_role == "First-Person" else "First-Person"
            if role_media.get(primary_role):
                media = dict(role_media[primary_role])
                context_media = role_media.get(other_role, {})
        if media:
            group = next((item for item in package_groups
                          if item.get("global_start_ms") is not None
                          and item.get("global_end_ms") is not None
                          and item["global_start_ms"] <= event.get("key_global_ms", 0)
                          <= item["global_end_ms"]), {})
            quarantine_materials.append({
                "event_id": event.get("event_id"),
                "timestamp_ms": event.get("key_global_ms", 0),
                "start_ms": event.get("global_start_ms"),
                "end_ms": event.get("global_end_ms"),
                "cv_action_type": event.get("cv_action_type"),
                "cv_objects": event.get("cv_objects", []),
                "source_views": event.get("source_views", []),
                "view_pairing": event.get("view_pairing", {}),
                "context_media": context_media,
                "source_roles": sorted({role_by_view[view] for view in event.get("source_views", [])
                                        if role_by_view.get(view)}),
                "group_id": group.get("group_id"),
                "group_folder": group.get("archive_folder"),
                "group_name": group.get("experiment_name"),
                "review_status": "machine_quarantined",
                "evidence_classification": "PARTIAL_EVIDENCE",
                "disposition": event.get("disposition"),
                **media,
            })
    quarantine_materials, retained_review = prioritize_retained_candidates(quarantine_materials)
    movement_screening = None if library_section else _movement_screening_payload(
        root, file_url(archive_name, "JSON-Config-Files/movement_visual_verification.json")
    )
    partial_delivery = snapshot.get("partial_delivery") or {}
    if partial_delivery and (root / "Partial-Results/Partial-Evidence-Report.html").is_file():
        links["partial_report"] = file_url(
            archive_name, "Partial-Results/Partial-Evidence-Report.html"
        )
        if (root / "Partial-Results/Analysis-Result.json").is_file():
            links["partial_json"] = file_url(archive_name, "Partial-Results/Analysis-Result.json")
        for key, kind, relative in (
            ("partial_pdf", "pdf", "Partial-Results/Stage-Evidence-Report.pdf"),
            ("partial_daily_report", "daily_html", "Partial-Results/Stage-Lab-Daily-Report.html"),
        ):
            presentation = (partial_delivery.get("readable_reports") or {}).get(kind) or {}
            path = root / relative
            if (presentation.get("path") == relative and path.is_file()
                    and hashlib.sha256(path.read_bytes()).hexdigest() == presentation.get("sha256")):
                links[key] = file_url(archive_name, relative)
    result = {
        "name": archive_name,
        "path": str(root),
        "staging_run_id": staging_run_id,
        "network_path": str(root),
        "release_id": release_id,
        "integrity_status": (
            lightweight_release_integrity(root, release_pointer)
            if release_pointer
            else "legacy_archive_not_release_verified"
        ),
        "counts": {
            "experiments": len(experiments),
            "key_events": len(normalized_events),
        },
        "experiments": experiments,
        "experiment_groups": [
            {
                "group_id": group.get("group_id"),
                "name": group.get("experiment_name") or group.get("archive_folder"),
                "folder": group.get("archive_folder"),
                "continuity_type": group.get("continuity_type"),
                "workflow_kind": group.get("workflow_kind", "unresolved"),
                "source_archive_folders": group.get("source_archive_folders") or [],
                "workflow_units": group.get("workflow_units") or [],
                "completion_status": group.get("completion_status", "unreviewed"),
                "completion_reason": group.get("completion_reason") or "",
                "boundary_extension_requires_step_review": group.get("boundary_extension_requires_step_review", False),
                "view_timeline": group.get("view_timeline") or [],
                "start_ms": group.get("global_start_ms"),
                "end_ms": group.get("global_end_ms"),
                "key_event_count": len(group.get("key_event_ids", [])),
            }
            for group in package_groups
        ],
        "key_events": normalized_events,
        "preliminary_materials": preliminary_materials,
        "quarantined_materials": quarantine_materials,
        "retained_material_review": retained_review,
        "movement_screening": movement_screening,
        "partial_delivery": partial_delivery,
        "metrics": metrics,
        "quality_acceptance": quality_acceptance,
        "result_review": _latest_result_review(root) if not active_preview and not library_section else {"available":False},
        "key_material_recall_eval": key_material_recall_eval,
        "key_material_verification": key_material_verification,
        "observability": snapshot,
        "daily_report": daily_report,
        "daily_report_manifest": daily_manifest,
        "current_release": release_pointer or None,
        "evidence_index": {
            **index_manifest,
            "search_url": (f"/api/staging-runs/{quote(staging_run_id)}/key-events"
                           if staging_run_id else f"/api/key-events?archive={quote(archive_name)}"),
        } if index_manifest else None,
        "links": links,
    }

    if library_section:
        from .library_projection import project_staging_library
        return project_staging_library(result, library_section)
    return result


@app.get("/api/archive-file")
def archive_file(
    archive: str,
    path: str,
    release: str | None = None,
    poster: bool = False,
) -> FileResponse:
    root = _resolve_archive(archive).resolve()
    from .artifact_reader import resolve, storage_status
    try:
        candidate = resolve(root, path, historical=True)
    except (OSError, ValueError) as exc:
        code, message = storage_status(exc)
        raise HTTPException(code, message) from exc
    headers = {
        "Accept-Ranges": "bytes",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-cache",
    }
    if release:
        pointer = read_current_release_pointer(root) or {}
        if str(pointer.get("release_id") or "") != release:
            raise HTTPException(409, "该素材链接对应的归档版本已更新，请刷新页面")
        if lightweight_release_integrity(root, pointer) != "release_manifest_verified":
            raise HTTPException(409, "正式归档发布清单完整性校验失败")
        manifest_path = (root / str(pointer.get("release_manifest") or "")).resolve()
        if not archive_contains(manifest_path, root) or not manifest_path.is_file():
            raise HTTPException(409, "正式归档发布清单不可用")
        manifest = _read_json(manifest_path, {}) or {}
        relative = candidate.relative_to(root)
        directory = relative.parts[0] if relative.parts else ""
        inner_path = Path(*relative.parts[1:]).as_posix() if len(relative.parts) > 1 else ""
        receipt = next(
            (
                item
                for item in (manifest.get("manifests") or {}).get(directory, [])
                if str(item.get("path") or "") == inner_path
            ),
            None,
        )
        if receipt is None or int(receipt.get("size_bytes") or -1) != candidate.stat().st_size:
            raise HTTPException(409, "正式归档素材与发布清单不一致")
        digest = str(receipt.get("sha256") or "")
        headers.update(
            {
                "Cache-Control": "private, max-age=31536000, immutable",
                "ETag": f'"sha256-{digest}"',
                "X-VisionCortex-Release": release,
                "X-VisionCortex-Integrity": "release-manifest-size-matched",
            }
        )
    if poster:
        return _video_poster_response(candidate, root)
    return FileResponse(candidate, headers=headers)


@app.get("/api/staging-file")
def staging_file(run_id: str, path: str, poster: bool = False) -> FileResponse:
    root = _resolve_staging_run(run_id)
    from .artifact_reader import resolve, storage_status
    try:
        candidate = resolve(root, path, historical=True)
    except (OSError, ValueError) as exc:
        code, message = storage_status(exc)
        raise HTTPException(code, message) from exc
    if poster:
        return _video_poster_response(candidate, root)
    # Staging media can be rebuilt at the same URL. Revalidate cached ranges so
    # the browser does not combine an earlier MP4 index with newer media bytes.
    return FileResponse(candidate, headers={
        "Cache-Control": "private, no-cache",
        "X-Content-Type-Options": "nosniff",
    })



def _video_poster_response(candidate: Path, root: Path) -> FileResponse:
    relative = candidate.relative_to(root.resolve())
    if candidate.suffix.lower() != ".mp4" or relative.parts[0] not in {"Experiment-Clips", "Key-Materials"}:
        raise HTTPException(400, "仅为已有实验片段和素材生成封面")
    runtime = _settings().get("storage", {}).get("local_runtime_root")
    if not runtime:
        raise HTTPException(503, "本地封面缓存尚未配置")
    try:
        path = cached_video_poster(candidate, Path(runtime))
    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as exc:
        raise HTTPException(503, "封面暂不可用，仍可点击播放视频") from exc
    return FileResponse(path, media_type="image/jpeg", headers={
        "Cache-Control": "private, no-cache", "X-Content-Type-Options": "nosniff",
        "X-VisionCortex-Preview": "first-decoded-frame",
    })


def _folder_open_command(
    root: Path,
    *,
    os_name: str | None = None,
    platform: str | None = None,
) -> list[str]:
    effective_os_name = os.name if os_name is None else os_name
    effective_platform = sys.platform if platform is None else platform
    if effective_os_name == "nt":
        return ["explorer.exe", str(root)]
    if effective_platform == "darwin":
        return ["open", str(root)]
    opener = shutil.which("xdg-open")
    if opener:
        return [opener, str(root)]
    raise RuntimeError("No supported desktop folder opener is installed")


@app.post("/api/archives/{archive_name}/open")
def open_archive_folder(archive_name: str) -> dict[str, str]:
    root = _resolve_archive(archive_name)
    try:
        command = _folder_open_command(root)
    except RuntimeError as exc:
        raise HTTPException(501, str(exc)) from exc
    subprocess.Popen(command, start_new_session=True)
    return {"status": "opened", "path": str(root)}


@app.post("/api/staging-runs/{run_id}/open")
def open_staging_folder(run_id: str) -> dict[str, str]:
    """Open an already resolved run directory without promoting its evidence."""
    root = _resolve_staging_run(run_id)
    try:
        command = _folder_open_command(root)
    except RuntimeError as exc:
        raise HTTPException(501, str(exc)) from exc
    subprocess.Popen(command, start_new_session=True)
    return {"status": "opened", "path": str(root)}


@app.post("/api/upload-sessions", status_code=201)
def create_upload_session(payload: dict[str, Any]) -> dict[str, Any]:
    settings = _settings()
    _expire_stale_upload_sessions(settings)
    archive_root = _archive_root(settings).resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    experiment_name = str(payload.get("experiment_name") or "").strip()
    if not experiment_name:
        raise HTTPException(400, "experiment_name 不能为空")

    session_id = uuid.uuid4().hex
    archive_name = _unique_upload_archive_name(settings, experiment_name)
    archive_path = archive_root / archive_name
    specs, files, expected_source_bytes = _parse_upload_session_files(
        payload, archive_path, session_id, settings
    )
    policy = _upload_policy(settings)
    processing_headroom = math.ceil(
        expected_source_bytes * float(policy["processing_headroom_ratio"])
    )
    safety_headroom = max(
        math.ceil(expected_source_bytes * float(policy["safety_headroom_ratio"])),
        int(policy["minimum_safety_bytes"]),
    )
    disk = shutil.disk_usage(archive_root)
    store = _require_upload_store(settings)
    try:
        capacity = store.reserve(
            session_id=session_id,
            archive_name=archive_name,
            archive_root=archive_path,
            storage_key=str(archive_root),
            experiment_name=experiment_name,
            view_specs=specs,
            retention_mode=_upload_retention_mode(settings),
            expected_source_bytes=expected_source_bytes,
            processing_headroom_bytes=processing_headroom,
            safety_headroom_bytes=safety_headroom,
            free_bytes=int(disk.free),
            files=files,
            expires_at=time.time() + float(policy["session_ttl_seconds"]),
        )
    except StorageReservationError as exc:
        raise HTTPException(
            507,
            {
                "message": "NAS 可用空间不足，任务尚未创建",
                "required_bytes": exc.required_bytes,
                "available_bytes": exc.available_bytes,
                "missing_bytes": exc.missing_bytes,
                "free_bytes": exc.free_bytes,
                "already_reserved_bytes": exc.already_reserved_bytes,
            },
        ) from exc
    try:
        settings["storage"]["active_archive_path"] = str(archive_path)
        initialize_nas_archive(settings, archive_name)
    except Exception:
        store.release(session_id)
        raise
    session = store.get(session_id)
    if session is None:
        raise HTTPException(500, "上传会话创建后无法读取")
    return {
        **_public_upload_session(session),
        "chunk_size_bytes": int(policy["chunk_size_bytes"]),
        "parallel_files": int(policy["parallel_files"]),
        "chunk_sha256_required": bool(policy["chunk_sha256_required"]),
        "capacity": capacity,
        "resume_url": f"/api/upload-sessions/{session_id}",
        "finalize_url": f"/api/upload-sessions/{session_id}/finalize",
    }


@app.get("/api/upload-sessions/{session_id}")
def get_upload_session(session_id: str) -> dict[str, Any]:
    settings = _settings()
    session = _load_upload_session(settings, session_id, refresh_expiry=True)
    return {
        **_public_upload_session(session),
        "chunk_size_bytes": int(_upload_policy(settings)["chunk_size_bytes"]),
        "parallel_files": int(_upload_policy(settings)["parallel_files"]),
        "chunk_sha256_required": bool(
            _upload_policy(settings)["chunk_sha256_required"]
        ),
    }


@app.delete("/api/upload-sessions/{session_id}")
def cancel_upload_session(session_id: str) -> dict[str, Any]:
    settings = _settings()
    store = _require_upload_store(settings)
    with _upload_finalize_lock:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(404, "上传会话不存在")
        if session["status"] != "open" or session.get("run_id"):
            raise HTTPException(409, "只有尚未提交分析的上传会话可以取消")
        archive_root = _archive_root(settings).resolve()
        candidate = Path(session["archive_root"]).resolve()
        if candidate.parent != archive_root or not candidate.name:
            raise HTTPException(500, "上传会话归档路径超出允许清理范围")
        if not store.cancel(session_id):
            raise HTTPException(409, "上传会话状态已经变化，未执行清理")
        cleanup_status = "not_present"
        if candidate.is_dir():
            try:
                shutil.rmtree(candidate)
                cleanup_status = "removed"
            except OSError:
                cleanup_status = "pending_retry"
        return {
            "session_id": session_id,
            "status": "cancelled",
            "released_bytes": int(session["reserved_bytes"]),
            "archive_cleanup": cleanup_status,
        }


@app.patch("/api/upload-sessions/{session_id}/files/{file_id}")
async def upload_session_chunk(
    request: Request, session_id: str, file_id: str
) -> dict[str, Any]:
    settings = _settings()
    store = _require_upload_store(settings)
    session = _load_upload_session(settings, session_id)
    if session["status"] != "open":
        raise HTTPException(409, "上传会话已经结束，不能继续写入")
    item = next((entry for entry in session["files"] if entry["file_id"] == file_id), None)
    if item is None:
        raise HTTPException(404, "上传文件不存在")
    if item.get("content_hash") or item.get("sha256"):
        return {
            "session_id": session_id,
            "file_id": file_id,
            "uploaded_bytes": int(item["expected_bytes"]),
            "completed": True,
        }

    try:
        offset = int(request.headers.get("upload-offset", ""))
    except ValueError as exc:
        raise HTTPException(400, "Upload-Offset 请求头无效") from exc
    partial_path = Path(item["partial_path"])
    final_path = Path(item["final_path"])
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    active_path = final_path if final_path.is_file() else partial_path
    actual_offset = active_path.stat().st_size if active_path.is_file() else 0
    if offset < 0 or offset > actual_offset:
        raise HTTPException(
            409,
            f"上传位置不一致，服务器应从 {actual_offset} 字节继续",
            headers={"Upload-Offset": str(actual_offset)},
        )

    policy = _upload_policy(settings)
    maximum_chunk = int(policy["chunk_size_bytes"])
    expected_chunk_sha = request.headers.get("x-chunk-sha256")
    if expected_chunk_sha is not None:
        expected_chunk_sha = expected_chunk_sha.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_chunk_sha):
            raise HTTPException(400, "X-Chunk-SHA256 请求头无效")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise HTTPException(400, "Content-Length 请求头无效") from exc
        if declared_length <= 0 or declared_length > maximum_chunk:
            raise HTTPException(413, f"单个分块必须在 1 到 {maximum_chunk} 字节之间")
    expected_bytes = int(item["expected_bytes"])
    if offset < actual_offset:
        replay = bytearray()
        async for block in request.stream():
            if not block:
                continue
            replay.extend(block)
            if len(replay) > maximum_chunk or offset + len(replay) > actual_offset:
                raise HTTPException(409, "续传校验范围超过服务器已保存的内容")
        if not replay:
            raise HTTPException(400, "续传校验分块不能为空")
        replay_digest = hashlib.sha256(replay).hexdigest()
        if expected_chunk_sha and not hmac.compare_digest(
            expected_chunk_sha, replay_digest
        ):
            raise HTTPException(422, "续传校验分块 SHA-256 无效")
        with active_path.open("rb") as handle:
            handle.seek(offset)
            retained = handle.read(len(replay))
        if not hmac.compare_digest(bytes(replay), retained):
            raise HTTPException(409, "重新选择的文件与服务器断点内容不一致")
        return {
            "session_id": session_id,
            "file_id": file_id,
            "uploaded_bytes": actual_offset,
            "completed": actual_offset == expected_bytes,
            "replayed": True,
        }
    if actual_offset == expected_bytes:
        return {
            "session_id": session_id,
            "file_id": file_id,
            "uploaded_bytes": actual_offset,
            "completed": True,
        }
    if policy["chunk_sha256_required"] and not expected_chunk_sha:
        raise HTTPException(428, "每个新上传分块必须提供 X-Chunk-SHA256")
    digest = hashlib.sha256()
    written = 0
    try:
        with partial_path.open("ab") as handle:
            async for block in request.stream():
                if not block:
                    continue
                written += len(block)
                if written > maximum_chunk or actual_offset + written > expected_bytes:
                    raise HTTPException(413, "分块超过服务器声明的大小或文件边界")
                handle.write(block)
                digest.update(block)
            if written <= 0:
                raise HTTPException(400, "上传分块不能为空")
            handle.flush()
            os.fsync(handle.fileno())
        if expected_chunk_sha and not hmac.compare_digest(
            expected_chunk_sha, digest.hexdigest()
        ):
            raise HTTPException(422, "上传分块 SHA-256 校验失败，请重试该分块")
    except Exception:
        if partial_path.is_file():
            with partial_path.open("r+b") as handle:
                handle.truncate(actual_offset)
                handle.flush()
                os.fsync(handle.fileno())
        raise

    uploaded_bytes = actual_offset + written
    store.update_progress(
        session_id,
        file_id,
        uploaded_bytes,
        expires_at=time.time() + float(policy["session_ttl_seconds"]),
    )
    store.record_chunk(
        session_id,
        file_id,
        actual_offset,
        written,
        digest.hexdigest(),
    )
    return {
        "session_id": session_id,
        "file_id": file_id,
        "uploaded_bytes": uploaded_bytes,
        "completed": uploaded_bytes == expected_bytes,
    }


@app.post("/api/upload-sessions/{session_id}/finalize", status_code=202)
def finalize_upload_session(
    session_id: str, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    settings = _settings()
    store = _require_upload_store(settings)
    with _upload_finalize_lock:
        session = _load_upload_session(settings, session_id)
        if session["status"] in {"finalized", "released"} and session.get("run_id"):
            run_id = str(session["run_id"])
            run = _runs.get(run_id, {})
            return {
                "run_id": run_id,
                "state": run.get("state", "queued"),
                "status_url": f"/api/runs/{run_id}",
                "nas_output": str(session["archive_root"]),
                "archive_url": f"/?archive={quote(str(session['archive_name']))}",
                "queue_persistence": "sqlite",
            }
        if session["status"] != "open":
            raise HTTPException(409, "上传会话不能提交分析")
        incomplete = [
            item
            for item in session["files"]
            if int(item["uploaded_bytes"]) != int(item["expected_bytes"])
        ]
        if incomplete:
            raise HTTPException(
                409,
                {
                    "message": "仍有文件没有上传完成",
                    "files": [item["file_id"] for item in incomplete],
                },
            )

        policy = _upload_policy(settings)
        for item in session["files"]:
            if item.get("content_hash") or item.get("sha256"):
                continue
            partial_path = Path(item["partial_path"])
            final_path = Path(item["final_path"])
            source_path = final_path if final_path.is_file() else partial_path
            if not source_path.is_file() or source_path.stat().st_size != int(
                item["expected_bytes"]
            ):
                raise HTTPException(409, f"暂存文件不完整: {item['file_id']}")
            expected_bytes = int(item["expected_bytes"])
            chunk_tree = store.chunk_tree_identity(
                session_id,
                str(item["file_id"]),
                expected_bytes,
            )
            sha256 = None
            if (
                chunk_tree is None
                or expected_bytes <= int(policy["full_file_sha256_max_bytes"])
            ):
                sha256 = _sha256_path(source_path)
            content_hash = sha256 or chunk_tree
            content_hash_algorithm = (
                "sha256" if sha256 else "visioncortex-upload-chunk-tree-v1"
            )
            if not content_hash:
                raise HTTPException(409, f"上传分块完整性账本不完整: {item['file_id']}")
            if source_path == partial_path:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(partial_path, final_path)
            store.complete_file(
                session_id,
                str(item["file_id"]),
                sha256,
                content_hash=content_hash,
                content_hash_algorithm=content_hash_algorithm,
            )

        session = store.get(session_id)
        if session is None:
            raise HTTPException(500, "上传会话在完成校验后丢失")
        archive_name = str(session["archive_name"])
        nas_root = Path(session["archive_root"])
        files_by_key = {
            (str(item["kind"]), int(item["file_index"])): item
            for item in session["files"]
        }
        views: list[ViewInput] = []
        for spec in session["view_specs"]:
            if spec.get("source_layout") == "segments" or spec.get("segments"):
                views.append(
                    ViewInput(
                        view_id=str(spec["view_id"]),
                        role=spec["role"],
                        segments=[
                            VideoSegmentInput(
                                video=Path(
                                    files_by_key[
                                        ("video", int(mapping["video_index"]))
                                    ]["final_path"]
                                ),
                                timestamps_csv=(
                                    Path(
                                        files_by_key[
                                            (
                                                "timestamp_csv",
                                                int(mapping["csv_index"]),
                                            )
                                        ]["final_path"]
                                    )
                                    if mapping.get("csv_index") is not None
                                    else None
                                ),
                            )
                            for mapping in spec["segments"]
                        ],
                        calibration_hint_ms=float(
                            spec.get("calibration_hint_ms", 0.0)
                        ),
                    )
                )
            else:
                video = files_by_key[("video", int(spec["video_index"]))]
                csv_index = spec.get("csv_index")
                timestamp_csv = (
                    Path(
                        files_by_key[("timestamp_csv", int(csv_index))]["final_path"]
                    )
                    if csv_index is not None
                    else None
                )
                views.append(
                    ViewInput(
                        view_id=str(spec["view_id"]),
                        role=spec["role"],
                        video=Path(video["final_path"]),
                        timestamps_csv=timestamp_csv,
                        calibration_hint_ms=float(
                            spec.get("calibration_hint_ms", 0.0)
                        ),
                    )
                )
        for view, spec in zip(views, session["view_specs"], strict=True):
            mappings = spec.get("segments") or [spec]
            for part, mapping in zip(view.segments or [view], mappings, strict=True):
                if mapping.get("audio_index") is not None:
                    part.audio = Path(files_by_key[("audio", int(mapping["audio_index"]))]["final_path"])
                    part.audio_offset_ms = mapping.get("audio_offset_ms")
        manifest = RunManifest(experiment_id=archive_name, views=views)
        manifest_path = nas_root / "JSON-Config-Files" / "input_manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        role_resolution = {
            "schema_version": "visioncortex-view-role-resolution-ledger/1",
            "experiment_id": archive_name,
            "input_source": "browser_user_declared_with_device_registry_validation",
            "status": "resolved",
            "resolved_first_person_views": sum(
                spec["role"] == "first_person" for spec in session["view_specs"]
            ),
            "resolved_third_person_views": sum(
                spec["role"] == "third_person" for spec in session["view_specs"]
            ),
            "views": [
                spec.get("role_resolution")
                or {
                    "camera_key": spec["view_id"],
                    "resolved_role": spec["role"],
                    "status": "legacy_session_without_registry_receipt",
                }
                for spec in session["view_specs"]
            ],
        }
        role_resolution_path = (
            nas_root / "JSON-Config-Files" / "view_role_resolution.json"
        )
        _write_json_atomic(role_resolution_path, role_resolution)
        preflight_path = (
            nas_root
            / "JSON-Config-Files"
            / "Input-Manifests"
            / "prequeue_input_preflight.json"
        )
        if policy["prequeue_media_preflight_enabled"]:
            try:
                preflight = preflight_manifest_inputs(manifest, settings)
            except (OSError, RuntimeError, ValueError) as exc:
                _write_json_atomic(
                    preflight_path,
                    {
                        "schema_version": "visioncortex-prequeue-input-preflight/1",
                        "status": "failed",
                        "completed_at": datetime.now().astimezone().isoformat(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise HTTPException(
                    422,
                    {
                        "message": "输入文件未通过媒体或时钟预检，任务未进入 GPU 队列",
                        "error": f"{type(exc).__name__}: {exc}",
                        "receipt": str(preflight_path),
                    },
                ) from exc
        else:
            preflight = {
                "schema_version": "visioncortex-prequeue-input-preflight/1",
                "status": "skipped_by_configuration",
            }
        _write_json_atomic(preflight_path, preflight)

        completed_at = datetime.now().astimezone().isoformat()
        duration_seconds = max(0.0, time.time() - float(session["created_at"]))
        retention_mode = str(session["retention_mode"])
        upload_ledger = [
            {
                "filename": item["source_name"],
                "bytes": int(item["expected_bytes"]),
                "local_path": (
                    item["final_path"] if retention_mode == "local_only" else None
                ),
                "nas_path": (
                    item["final_path"] if retention_mode == "nas_only" else None
                ),
                "analysis_path": item["final_path"],
                "retention_mode": retention_mode,
                "local_write_bytes": (
                    int(item["expected_bytes"]) if retention_mode == "local_only" else 0
                ),
                "nas_write_bytes": (
                    int(item["expected_bytes"]) if retention_mode == "nas_only" else 0
                ),
                "sha256": item["sha256"],
                "content_hash": item.get("content_hash") or item.get("sha256"),
                "content_hash_algorithm": item.get("content_hash_algorithm")
                or "sha256",
                "file_id": item["file_id"],
                "kind": item["kind"],
                "view_id": item["view_id"],
                "segment_ordinal": item.get("segment_ordinal"),
            }
            for item in session["files"]
        ]
        ingest = {
            "request_started_perf": None,
            "request_started_epoch": float(session["created_at"]),
            "request_received_at": datetime.fromtimestamp(
                float(session["created_at"])
            ).astimezone().isoformat(),
            "original_retention_completed_at": completed_at,
            "duration_seconds": round(duration_seconds, 6),
            "file_count": len(upload_ledger),
            "video_count": sum(item["kind"] == "video" for item in session["files"]),
            "timestamp_csv_count": sum(
                item["kind"] == "timestamp_csv" for item in session["files"]
            ),
            "total_bytes": int(session["expected_source_bytes"]),
            "local_write_bytes": sum(
                int(item["local_write_bytes"]) for item in upload_ledger
            ),
            "nas_write_bytes": sum(int(item["nas_write_bytes"]) for item in upload_ledger),
            "retention_mode": retention_mode,
            "destinations": [
                "nas_original_experiment_videos"
                if retention_mode == "nas_only"
                else "local_archive_original_experiment_videos"
            ],
            "upload_protocol": "resumable_chunks_v2",
            "protocol_compatibility": ["resumable_chunks_v1"],
            "chunk_integrity": "persistent_sha256_ledger",
            "full_file_sha256_max_bytes": int(
                policy["full_file_sha256_max_bytes"]
            ),
            "upload_session_id": session_id,
            "storage_reservation_bytes": int(session["reserved_bytes"]),
        }
        ingest["effective_source_throughput_mib_s"] = round(
            ingest["total_bytes"] / max(duration_seconds, 1e-9) / (1024 * 1024), 3
        )
        input_seal = build_input_seal(
            manifest,
            source_mode="browser_resumable_upload",
            sources=[
                {
                    "file_id": item["file_id"],
                    "kind": item["kind"],
                    "view_id": item["view_id"],
                    "segment_ordinal": item.get("segment_ordinal"),
                    "path": item["analysis_path"],
                    "size_bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "content_hash": item["content_hash"],
                    "content_hash_algorithm": item["content_hash_algorithm"],
                }
                for item in upload_ledger
            ],
            role_resolution=role_resolution,
            copied_source_bytes=int(session["expected_source_bytes"]),
            preflight=preflight,
        )
        input_seal_path = write_input_seal(
            nas_root
            / "JSON-Config-Files"
            / "Input-Manifests"
            / "input_seal.json",
            input_seal,
        )
        ingest["input_seal"] = {
            "path": str(input_seal_path),
            "sha256": input_seal["seal_sha256"],
            "source_mode": input_seal["source_mode"],
        }
        upload_record = {
            "run_id": f"upload-{session_id[:12]}",
            "archive_name": archive_name,
            "uploaded_at": completed_at,
            "files": upload_ledger,
            "manifest": manifest.model_dump(mode="json"),
            "web_ingest": {
                key: value
                for key, value in ingest.items()
                if key not in {"request_started_perf", "request_started_epoch"}
            },
        }
        _write_json_atomic(
            nas_root / "JSON-Config-Files" / "original_upload_manifest.json",
            upload_record,
        )
        _write_json_atomic(
            nas_root
            / "JSON-Config-Files"
            / "Stage-Receipts"
            / "original_ingest.json",
            {
                "schema_version": "visioncortex-stage-receipt/1",
                "stage": "original_ingest",
                "status": "completed",
                "completed_at": completed_at,
                "run_elapsed_seconds": round(duration_seconds, 6),
                "stage_duration_seconds": round(duration_seconds, 6),
                "archive_mode": settings["storage"].get("run_output_mode", "local"),
                "archive_root": str(nas_root),
                "retention_mode": retention_mode,
                "upload_protocol": "resumable_chunks_v2",
                "source_copy_bytes": int(session["expected_source_bytes"]),
                "storage_reservation_bytes": int(session["reserved_bytes"]),
                "artifacts": [
                    "Original-Experiment-Videos",
                    "JSON-Config-Files/original_upload_manifest.json",
                    "JSON-Config-Files/Input-Manifests/input_seal.json",
                    "JSON-Config-Files/Input-Manifests/prequeue_input_preflight.json",
                    "JSON-Config-Files/view_role_resolution.json",
                ],
                "token_ledger": "JSON-Config-Files/run_metrics.json",
            },
        )

        run_id = str(upload_record["run_id"])
        settings["storage"]["sync_to_nas"] = retention_mode == "nas_only"
        fixed_root, staging_root, _history_root = _prepare_formal_run_staging(
            settings, archive_name, run_id, nas_root
        )
        queue_recovery_path = _write_queue_recovery_receipt(
            staging_root,
            run_id=run_id,
            state="queued",
            manifest_path=(
                staging_root
                / "JSON-Config-Files"
                / "input_manifest.yaml"
            ),
            input_seal_path=(
                staging_root
                / "JSON-Config-Files"
                / "Input-Manifests"
                / "input_seal.json"
            ),
            ingest=ingest,
            recovery_context={
                "formal_fixed_root": str(fixed_root),
                "formal_history_root": str(_history_root),
            },
        )
        store.assign_run(session_id, run_id)
        existing_job = (
            _persistent_queue.get_job(run_id) if _persistent_queue is not None else None
        )
        if existing_job is None:
            _update(
                run_id,
                state="queued",
                progress=0.0,
                experiment_id=manifest.experiment_id,
                nas_output=str(fixed_root),
                nas_staging=str(staging_root),
            )
            queue_persistence = _schedule_job(
                background_tasks,
                run_id=run_id,
                kind="run",
                payload={
                    "manifest": manifest.model_dump(mode="json"),
                    "settings": settings,
                    "nas_root": str(staging_root),
                    "ingest": ingest,
                },
                fallback=_execute,
                fallback_args=(run_id, manifest, settings, staging_root, ingest),
            )
        else:
            queue_persistence = "sqlite"
        try:
            store.finalize(session_id, run_id)
        except ValueError:
            refreshed = store.get(session_id)
            if refreshed is None or refreshed["status"] != "released":
                raise
        return {
            "run_id": run_id,
            "state": "queued",
            "status_url": f"/api/runs/{run_id}",
            "nas_output": str(fixed_root),
            "nas_staging": str(staging_root),
            "archive_url": f"/?archive={quote(archive_name)}",
            "queue_persistence": queue_persistence,
            "queue_disaster_recovery": "archive_receipt",
            "queue_recovery_receipt": str(queue_recovery_path),
        }


@app.post("/api/runs", status_code=202)
async def create_run(
    request: Request,
    background_tasks: BackgroundTasks,
    experiment_name: Annotated[str, Form()],
    view_specs_json: Annotated[str, Form()],
    videos: Annotated[list[UploadFile], File()],
    timestamp_csvs: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, Any]:
    try:
        specs = json.loads(view_specs_json)
        if not isinstance(specs, list) or not specs:
            raise ValueError("必须是非空数组")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, f"view_specs_json 无效: {exc}") from exc
    settings = _settings()
    archive_name, nas_root = _reserve_archive(settings, experiment_name)
    settings["storage"]["active_archive_path"] = str(nas_root)
    run_id = uuid.uuid4().hex[:12]
    local_root = Path(settings["storage"]["local_input_root"]) / archive_name / run_id
    retention_mode = str(
        settings["storage"].get("web_upload_retention_mode", "local_and_nas")
    ).lower()
    if retention_mode not in {"local_and_nas", "nas_only", "local_only"}:
        raise HTTPException(500, f"Unsupported web_upload_retention_mode: {retention_mode}")
    retain_local_copy = retention_mode == "local_and_nas"
    saved_videos: list[Path] = []
    saved_csvs: list[Path] = []
    upload_ledger: list[dict[str, Any]] = []
    try:
        for index, upload in enumerate(videos):
            spec = next((item for item in specs if int(item.get("video_index", -1)) == index), {})
            view_id = _safe_file_name(
                str(spec.get("view_id") or f"view-{index + 1:02d}"),
                f"view-{index + 1:02d}",
            )
            filename = _safe_file_name(
                upload.filename or f"video-{index + 1:02d}.mp4",
                f"video-{index + 1:02d}",
            )
            local_destination = local_root / view_id / filename
            nas_destination = nas_root / "Original-Experiment-Videos" / view_id / filename
            upload_ledger.append(
                await _save_upload_to_local_and_nas(
                    upload,
                    local_destination,
                    nas_destination,
                    retain_local_copy=retain_local_copy,
                    archive_is_network=bool(settings["storage"].get("sync_to_nas")),
                )
            )
            saved_videos.append(Path(upload_ledger[-1]["analysis_path"]))
        for index, upload in enumerate(timestamp_csvs or []):
            spec = next((item for item in specs if int(item.get("csv_index", -1)) == index), {})
            view_id = _safe_file_name(
                str(spec.get("view_id") or f"view-{index + 1:02d}"),
                f"view-{index + 1:02d}",
            )
            filename = _safe_file_name(
                upload.filename or f"timestamps-{index + 1:02d}.csv",
                f"timestamps-{index + 1:02d}",
            )
            local_destination = local_root / view_id / filename
            nas_destination = nas_root / "Original-Experiment-Videos" / view_id / filename
            upload_ledger.append(
                await _save_upload_to_local_and_nas(
                    upload,
                    local_destination,
                    nas_destination,
                    retain_local_copy=retain_local_copy,
                    archive_is_network=bool(settings["storage"].get("sync_to_nas")),
                )
            )
            saved_csvs.append(Path(upload_ledger[-1]["analysis_path"]))
        view_inputs = []
        for position, spec in enumerate(specs):
            video_index = int(spec.get("video_index", position))
            csv_index = spec.get("csv_index")
            view_inputs.append(
                ViewInput(
                    view_id=str(spec["view_id"]),
                    role=spec["role"],
                    video=saved_videos[video_index],
                    timestamps_csv=saved_csvs[int(csv_index)] if csv_index is not None else None,
                    calibration_hint_ms=float(spec.get("calibration_hint_ms", 0.0)),
                )
            )
        manifest = RunManifest(experiment_id=archive_name, views=view_inputs)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"视角映射无效: {exc}") from exc
    manifest_payload = yaml.safe_dump(
        manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
    )
    manifest_path = nas_root / "JSON-Config-Files" / "input_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_payload, encoding="utf-8")
    if retain_local_copy:
        local_manifest_path = local_root / "manifest.yaml"
        local_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        local_manifest_path.write_text(manifest_payload, encoding="utf-8")
    role_resolution = _declared_role_resolution(
        manifest, "browser_legacy_upload_declared"
    )
    _write_json_atomic(
        nas_root / "JSON-Config-Files" / "view_role_resolution.json",
        role_resolution,
    )
    sha_by_path = {
        str(Path(item["analysis_path"]).resolve()): str(item["sha256"])
        for item in upload_ledger
    }
    input_seal = build_input_seal(
        manifest,
        source_mode="browser_legacy_upload",
        sources=_manifest_source_receipts(manifest, sha_by_path),
        role_resolution=role_resolution,
        copied_source_bytes=sum(int(item["bytes"]) for item in upload_ledger),
    )
    write_input_seal(
        nas_root
        / "JSON-Config-Files"
        / "Input-Manifests"
        / "input_seal.json",
        input_seal,
    )
    upload_record = {
        "run_id": run_id,
        "archive_name": archive_name,
        "uploaded_at": datetime.now().isoformat(),
        "files": upload_ledger,
        "manifest": manifest.model_dump(mode="json"),
    }
    ingest_ended_at = datetime.now().astimezone().isoformat()
    ingest_started_perf = float(
        getattr(request.state, "ingest_started_perf", time.perf_counter())
    )
    ingest = {
        "request_started_perf": ingest_started_perf,
        "request_started_epoch": float(
            getattr(request.state, "ingest_started_epoch", time.time())
        ),
        "request_received_at": getattr(
            request.state, "ingest_started_at", datetime.now().astimezone().isoformat()
        ),
        "original_retention_completed_at": ingest_ended_at,
        "duration_seconds": round(time.perf_counter() - ingest_started_perf, 6),
        "file_count": len(upload_ledger),
        "video_count": len(saved_videos),
        "timestamp_csv_count": len(saved_csvs),
        "total_bytes": sum(int(item["bytes"]) for item in upload_ledger),
        "local_write_bytes": sum(int(item["local_write_bytes"]) for item in upload_ledger),
        "nas_write_bytes": sum(int(item["nas_write_bytes"]) for item in upload_ledger),
        "retention_mode": retention_mode,
        "destinations": (
            ["local_archive_original_experiment_videos"]
            if not settings["storage"].get("sync_to_nas") else
            ["local_input", "nas_original_experiment_videos"]
            if retain_local_copy
            else ["nas_original_experiment_videos"]
        ),
    }
    ingest["effective_source_throughput_mib_s"] = round(
        ingest["total_bytes"] / max(ingest["duration_seconds"], 1e-9) / (1024 * 1024),
        3,
    )
    upload_record["web_ingest"] = {
        key: value
        for key, value in ingest.items()
        if key not in {"request_started_perf", "request_started_epoch"}
    }
    record_path = nas_root / "JSON-Config-Files" / "original_upload_manifest.json"
    record_path.write_text(json.dumps(upload_record, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_json_atomic(
        nas_root
        / "JSON-Config-Files"
        / "Stage-Receipts"
        / "original_ingest.json",
        {
            "schema_version": "visioncortex-stage-receipt/1",
            "stage": "original_ingest",
            "status": "completed",
            "completed_at": ingest_ended_at,
            "run_elapsed_seconds": ingest["duration_seconds"],
            "stage_duration_seconds": ingest["duration_seconds"],
            "archive_mode": settings["storage"].get("run_output_mode", "local"),
            "archive_root": str(nas_root),
            "retention_mode": retention_mode,
            "source_copy_bytes": ingest["local_write_bytes"],
            "artifacts": [
                "Original-Experiment-Videos",
                "JSON-Config-Files/original_upload_manifest.json",
                "JSON-Config-Files/Input-Manifests/input_seal.json",
                "JSON-Config-Files/view_role_resolution.json",
            ],
            "token_ledger": "JSON-Config-Files/run_metrics.json",
        },
    )
    fixed_root, staging_root, _history_root = _prepare_formal_run_staging(
        settings, archive_name, run_id, nas_root
    )
    _write_queue_recovery_receipt(
        staging_root,
        run_id=run_id,
        state="queued",
        manifest_path=(
            staging_root / "JSON-Config-Files" / "input_manifest.yaml"
        ),
        input_seal_path=(
            staging_root
            / "JSON-Config-Files"
            / "Input-Manifests"
            / "input_seal.json"
        ),
        ingest=ingest,
        recovery_context={
            "formal_fixed_root": str(fixed_root),
            "formal_history_root": str(_history_root),
        },
    )
    _update(
        run_id,
        state="queued",
        progress=0.0,
        experiment_id=manifest.experiment_id,
        nas_output=str(fixed_root),
        nas_staging=str(staging_root),
    )
    queue_persistence = _schedule_job(
        background_tasks,
        run_id=run_id,
        kind="run",
        payload={
            "manifest": manifest.model_dump(mode="json"),
            "settings": settings,
            "nas_root": str(staging_root),
            "ingest": ingest,
        },
        fallback=_execute,
        fallback_args=(run_id, manifest, settings, staging_root, ingest),
    )
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "nas_output": str(fixed_root),
        "nas_staging": str(staging_root),
        "archive_url": f"/?archive={quote(archive_name)}",
        "queue_persistence": queue_persistence,
    }


@app.post("/api/benchmarks/six-view-three-hour/runs", status_code=202)
def create_fixed_benchmark_run(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Rerun the registered six-view benchmark into its fixed NAS archive."""

    request_started_perf = time.perf_counter()
    request_started_epoch = time.time()
    request_received_at = datetime.now().astimezone().isoformat()
    settings = _settings()
    if settings.get("project", {}).get("portable_desktop"):
        raise HTTPException(404, "此应用使用用户上传的视频，不包含固定六路基准数据。")
    settings["storage"]["sync_to_nas"] = True
    run_id = f"benchmark-{uuid.uuid4().hex[:10]}"
    nas_root = _reserve_fixed_benchmark(settings, run_id)

    timing = {
        "request_started_perf": request_started_perf,
        "request_started_epoch": request_started_epoch,
        "request_received_at": request_received_at,
        "experiment_id": _BENCHMARK_EXPERIMENT_ID,
        "archive_name": _BENCHMARK_ARCHIVE_NAME,
        "index_csv": str(settings["storage"]["index_csv"]),
    }
    _update(
        run_id,
        state="queued",
        progress=0.0,
        experiment_id=_BENCHMARK_ARCHIVE_NAME,
        benchmark_archive=_BENCHMARK_ARCHIVE_NAME,
        nas_output=str(_archive_root(settings) / _BENCHMARK_ARCHIVE_NAME),
        nas_staging=str(nas_root),
    )
    submission_receipt = _write_fixed_benchmark_submission_receipt(
        nas_root,
        run_id,
        request_received_at,
    )
    queue_persistence = _schedule_job(
        background_tasks,
        run_id=run_id,
        kind="fixed_benchmark",
        payload={
            "settings": settings,
            "nas_root": str(nas_root),
            "timing": timing,
        },
        fallback=_execute_fixed_benchmark,
        fallback_args=(run_id, settings, nas_root, timing),
    )
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "nas_output": str(_archive_root(settings) / _BENCHMARK_ARCHIVE_NAME),
        "nas_staging": str(nas_root),
        "archive_url": f"/#/archive/{quote(_BENCHMARK_ARCHIVE_NAME)}/experiments",
        "reused_archive": True,
        "source_count": 6,
        "execution_owner": "visioncortex_web_service",
        "client_process_independent": True,
        "submission_protocol_version": _BENCHMARK_SUBMISSION_PROTOCOL_VERSION,
        "submission_receipt": str(submission_receipt),
        "queue_persistence": queue_persistence,
    }


@app.post("/api/collections/{experiment_id}/runs", status_code=202)
def create_collection_run(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_started_perf = time.perf_counter()
    request_started_epoch = time.time()
    request_received_at = datetime.now().astimezone().isoformat()
    settings = _settings()
    try:
        collection = get_collection(settings, experiment_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"无法读取采集批次索引: {exc}") from exc
    if not collection.get("ready_to_analyze"):
        raise HTTPException(
            409,
            {
                "message": "采集批次尚未通过自动封口与视角质量门",
                "status": collection.get("status"),
                "blocking_issues": collection.get("blocking_issues"),
            },
        )
    with _lock:
        duplicate = next(
            (
                run_id
                for run_id, run in _runs.items()
                if run.get("source_collection_id") == experiment_id
                and run.get("state") not in {"completed", "partial", "failed", "interrupted"}
            ),
            None,
        )
    if duplicate:
        raise HTTPException(409, f"该采集批次已在任务 {duplicate} 中运行")

    payload = payload or {}
    start_date = str(collection.get("recording_start_time") or "")[:10].replace("-", "")
    default_name = (
        f"VisionCortex-Collection-{start_date or 'Undated'}-{experiment_id[-8:]}"
    )
    requested_name = str(payload.get("experiment_name") or default_name).strip()
    run_id = f"collection-{uuid.uuid4().hex[:10]}"
    if directory_ingest_enabled(settings):
        settings["storage"]["index_csv"] = str(
            selection_path(settings, experiment_id, ".csv")
        )
    archive_name, fixed_root, staging_root, history_root = _reserve_collection_archive(
        settings, requested_name, run_id
    )
    timing = {
        "request_started_perf": request_started_perf,
        "request_started_epoch": request_started_epoch,
        "request_received_at": request_received_at,
        "source_experiment_id": experiment_id,
        "archive_name": archive_name,
        "index_csv": str(settings["storage"]["index_csv"]),
        "collection_fingerprint_sha256": collection.get("fingerprint_sha256"),
        "source_copy_bytes": 0,
    }
    _update(
        run_id,
        state="queued",
        progress=0.0,
        experiment_id=archive_name,
        source_collection_id=experiment_id,
        nas_output=str(fixed_root),
        nas_staging=str(staging_root),
    )
    record_collection_state(
        settings,
        experiment_id,
        archive_name=archive_name,
        run_id=run_id,
        state="queued",
        details={"staging": str(staging_root), "source_copy_bytes": 0},
    )
    queue_persistence = _schedule_job(
        background_tasks,
        run_id=run_id,
        kind="index_collection",
        payload={
            "source_experiment_id": experiment_id,
            "archive_name": archive_name,
            "settings": settings,
            "staging_root": str(staging_root),
            "fixed_root": str(fixed_root),
            "history_root": str(history_root),
            "timing": timing,
        },
        fallback=_execute_index_collection,
        fallback_args=(
            run_id,
            experiment_id,
            archive_name,
            settings,
            staging_root,
            fixed_root,
            history_root,
            timing,
        ),
    )
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "source_collection_id": experiment_id,
        "source_copy_bytes": 0,
        "nas_output": str(fixed_root),
        "nas_staging": str(staging_root),
        "archive_name": archive_name,
        "archive_url": f"/#/archive/{quote(archive_name)}/experiments",
        "queue_persistence": queue_persistence,
    }


@app.post("/api/runs/from-paths", status_code=202)
def create_run_from_paths(payload: dict[str, Any], background_tasks: BackgroundTasks) -> dict[str, Any]:
    try:
        manifest = RunManifest.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    settings = _settings()
    archive_name, nas_root = _reserve_archive(settings, manifest.experiment_id)
    if archive_name != manifest.experiment_id:
        manifest = manifest.model_copy(update={"experiment_id": archive_name})
    run_id = uuid.uuid4().hex[:12]
    fixed_root, staging_root, _history_root = _prepare_formal_run_staging(
        settings, archive_name, run_id, nas_root
    )
    manifest_path = staging_root / "JSON-Config-Files" / "input_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    try:
        role_resolution = _declared_role_resolution(
            manifest, "api_from_paths_declared"
        )
        _write_json_atomic(
            staging_root / "JSON-Config-Files" / "view_role_resolution.json",
            role_resolution,
        )
        input_seal = build_input_seal(
            manifest,
            source_mode="api_from_paths",
            sources=_manifest_source_receipts(manifest),
            role_resolution=role_resolution,
            copied_source_bytes=0,
        )
        input_seal_path = write_input_seal(
            staging_root
            / "JSON-Config-Files"
            / "Input-Manifests"
            / "input_seal.json",
            input_seal,
        )
    except OSError as exc:
        raise HTTPException(422, f"输入路径不可读取: {exc}") from exc
    _write_queue_recovery_receipt(
        staging_root,
        run_id=run_id,
        state="queued",
        manifest_path=manifest_path,
        input_seal_path=input_seal_path,
        ingest=None,
        recovery_context={
            "formal_fixed_root": str(fixed_root),
            "formal_history_root": str(_history_root),
        },
    )
    _update(
        run_id,
        state="queued",
        progress=0.0,
        experiment_id=manifest.experiment_id,
        nas_output=str(fixed_root),
        nas_staging=str(staging_root),
    )
    queue_persistence = _schedule_job(
        background_tasks,
        run_id=run_id,
        kind="run",
        payload={
            "manifest": manifest.model_dump(mode="json"),
            "settings": settings,
            "nas_root": str(staging_root),
            "ingest": None,
        },
        fallback=_execute,
        fallback_args=(run_id, manifest, settings, staging_root),
    )
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "nas_output": str(fixed_root),
        "nas_staging": str(staging_root),
        "queue_persistence": queue_persistence,
    }


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    if _persistent_queue is not None:
        with _lock:
            _runs.update(_persistent_queue.load_runs())
    with _lock:
        runs = [
            _hydrate_run_snapshot({"run_id": run_id, **values})
            for run_id, values in _runs.items()
        ]
    runs.sort(key=lambda item: str(item.get("run_id")), reverse=True)
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    if _persistent_queue is not None:
        fresh = _persistent_queue.load_run(run_id)
        if fresh:
            with _lock:
                _runs[run_id] = fresh
    with _lock:
        state = dict(_runs.get(run_id, {}))
    if not state:
        root = _find_staging_run(_settings(), run_id)
        if root is None:
            raise HTTPException(404, "run_id 不存在")
        snapshot = _run_snapshot_from_root(root)
        status = snapshot.get("status") or {}
        state = {
            "state": status.get("stage", "interrupted"),
            "progress": status.get("progress", 0.0),
            "message": status.get("message"),
            "nas_staging": str(root),
            "observability": snapshot,
            "recovered_from_durable_status": True,
        }
    return _hydrate_run_snapshot({"run_id": run_id, **state})


def _retain_retry_outputs(root: Path, attempt: int) -> None:
    """Rename previous derived output into history on the same share.

    Original media and input seals remain at their frozen paths. A fresh
    attempt must not mix old clip names, quarantine files or indexes with its
    own results. No media bytes are copied or removed.
    """
    json_root = root / "JSON-Config-Files"
    history = json_root / "Retry-Attempts" / str(attempt) / "Derived"
    inputs = {"Input-Manifests", "Retry-Attempts", "input_manifest.yaml",
              "run_manifest.json", "original_upload_manifest.json"}
    sources = [root / name for name in ARCHIVE_DIRECTORIES
               if name not in {"Original-Experiment-Videos", "JSON-Config-Files"}]
    sources.extend([root / "Partial-Results", root / "run_status.json"])
    if json_root.is_dir():
        sources.extend(item for item in json_root.iterdir() if item.name not in inputs)
    moves = [(source, history / source.relative_to(root)) for source in sources if source.exists()]
    if any(destination.exists() for _, destination in moves):
        raise RuntimeError("上次复跑历史目录已存在同名产出，已停止以保留两份结果。")
    applied = []
    try:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            applied.append((source, destination))
        _write_json_atomic(history.parent / "output_relocation.json", {
            "schema_version": "visioncortex-retry-output-relocation/1",
            "previous_attempt": attempt, "source_media_moved": False,
            "copied_media_bytes": 0,
            "paths": [{"from": str(source.relative_to(root)), "to": str(destination.relative_to(root))}
                      for source, destination in applied],
        })
    except OSError:
        for source, destination in reversed(applied):
            destination.replace(source)
        raise


def _refresh_in_progress(run_id: str) -> bool:
    return any(state.get("parent_run_id") == run_id and state.get("state") in {"queued", "running"}
               for state in _runs.values())


@app.post("/api/runs/{run_id}/refresh/{scope}", status_code=202)
def refresh_stage(run_id: str, scope: str, target: str | None = None, revision: str | None = None) -> dict[str, Any]:
    if scope not in {"understanding", "reports", "timeline", "capture_quality", "search", "operations", "result_check", "gap_review"}:
        raise HTTPException(422, "请选择需要更新的阶段")
    store = _persistent_queue
    job = store.get_job(run_id) if store else None
    if not job or job["status"] not in {"failed", "completed"}:
        raise HTTPException(409, "原任务须已停止且保留持久任务记录")
    root = _find_staging_run(_settings(), run_id)
    if root is None or read_current_release_pointer(root):
        raise HTTPException(409, "仅可刷新尚未正式发布的实验")
    settings = copy.deepcopy(job["payload"]["settings"])
    if scope in {"result_check", "gap_review"}:
        from .result_review import inspect
        try:
            plan = inspect(root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, "当前保存记录不足以检查，请先核对实验产出") from exc
        if not revision or revision != plan["revision"]:
            raise HTTPException(409, "结果版本已改变，请刷新后重新检查")
        if scope == "gap_review":
            if target not in {w["window_id"] for w in plan["windows"]}:
                raise HTTPException(422, "请选择当前检查中的缺口区间")
            try:
                settings["mllm"] = ai_settings.reverified_job_mllm(settings)
            except (OSError, ValueError, RuntimeError) as exc:
                raise HTTPException(422, str(exc)) from exc
    if scope == "operations":
        from .operation_review import bindings, GROUPS, EVENTS
        bound = bindings(root)
        if not bound.get(EVENTS) or not revision or revision != bound[GROUPS]:
            raise HTTPException(422, "缺少已审核事件，或步骤版本已经变化，请刷新后重试")
        try:
            settings["mllm"] = ai_settings.reverified_job_mllm(settings)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
    if scope == "understanding":
        groups = _read_json(root / "JSON-Config-Files/experiment_group_understanding.json", {}) or {}
        if target is not None:
            from .speech_refresh import input_path
            if not revision or len(revision) != 64:
                raise HTTPException(422, "片段重算需要当前理解版本")
            if target.startswith("group:"):
                group = next((item for item in groups.get("groups", []) if "group:"+item["group_id"] == target), None)
                if group is None or not input_path(root, group["group_id"]).is_file():
                    raise HTTPException(422, "该片段没有保留可核验的理解画面，请运行完整流程生成")
            elif not target.startswith("recording:") or not target.split(":")[1].isdigit():
                raise HTTPException(422, "理解片段标识无效")
        elif groups.get("groups") or not (root / "JSON-Config-Files/speech_understanding.json").is_file():
            raise HTTPException(422, "请选择一个保留了理解画面的实验片段")
        try:
            settings["mllm"] = ai_settings.reverified_job_mllm(settings)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
    settings.setdefault("project", {})["semantic_cache_mode"] = "reuse"
    refresh_id = "refresh-" + uuid.uuid4().hex[:12]
    with _lock:
        # Parent retry and child refresh cannot alter the same artifacts together.
        if _refresh_in_progress(run_id) or store.get_job(run_id)["status"] not in {"failed", "completed"}:
            raise HTTPException(409, "此实验已有阶段任务，请等待完成")
        state = {"state": "queued", "progress": 0.0, "parent_run_id": run_id,
                 "refresh_scope": scope, "refresh_target": target, "message": "等待刷新所选阶段"}
        store.save_run(refresh_id, state)
        store.enqueue(refresh_id, "stage_refresh", {"settings": settings, "parent_run_id": run_id, "scope": scope, "target": target, "revision": revision})
        _runs[refresh_id] = state
    _queue_wakeup.set()
    return {"run_id": refresh_id, "parent_run_id": run_id, "state": "queued",
            "status_url": f"/api/runs/{refresh_id}", "scope": scope}


@app.get("/api/runs/{run_id}/recovery-plan")
def recovery_plan(run_id: str) -> dict[str, Any]:
    store = _persistent_queue
    job = store.get_job(run_id) if store else None
    if job is None:
        raise HTTPException(404, "原持久任务不存在，请重新选择原输入")
    root = _find_staging_run(_settings(), run_id)
    if root is None or read_current_release_pointer(root):
        raise HTTPException(409, "仅可恢复尚未正式发布的暂存任务")
    from .operation_review import bindings, GROUPS, EVENTS
    bound = bindings(root)
    status = _pipeline_status_from_root(root)
    stopped = job["status"] in {"failed", "completed"} and not _refresh_in_progress(run_id)
    retryable = stopped and (job["status"] == "failed" or (store.load_runs().get(run_id) or {}).get("state") == "partial")
    model = job["payload"]["settings"].get("mllm") or {}
    model_ready = False
    model_message = "原任务未启用模型理解"
    if model.get("enabled"):
        try:
            ai_settings.reverified_job_mllm(job["payload"]["settings"])
            model_ready = True
            model_message = "已保存的连接验证可用于原任务；提交时再次检查"
        except (OSError, ValueError, RuntimeError) as exc:
            model_message = str(exc)
    identity = {"run_id":run_id, "attempt":job["attempts"], "status":job["status"], "bindings":bound,
                "updated_at":status.get("updated_at"), "refresh_in_progress":_refresh_in_progress(run_id)}
    revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"run_id":run_id, "revision":revision, "group_revision":bound[GROUPS],
            "failed_stage":status.get("failed_stage") or status.get("stage"),
            "provider":model.get("provider"), "model":model.get("model"),
            "output_path":str(root), "retained_stage_count":len(_stage_receipts_from_root(root)),
            "model_ready":model_ready, "model_check_message":model_message,
            "actions": {"retry":retryable and (model_ready or not model.get("enabled")), "reports":stopped,
                        "operations":stopped and model_ready and bool(bound[GROUPS] and bound[EVENTS])},
            "cache_policy":"校验源文件、代码、配置和模型身份后复用；未通过校验的部分重新计算。",
            "retry_effect":"原输入不复制；已有派生产出移入 Retry-Attempts 历史目录，本轮重新生成可见成果。",
            "model_cost":"完整复跑与操作整理可能产生新的模型费用；缓存命中数与新增用量以执行回执为准，当前不作费用承诺。",
            "quality_policy":"操作整理复用已保存画面与已审核事件；不能补出未观察的动作或证明实验已结束。报告刷新不调用模型，未通过质量门时仅生成阶段报告。"}


@app.post("/api/runs/{run_id}/retry", status_code=202)
def retry_run(run_id: str, revision: str | None = None) -> dict[str, Any]:
    store = _persistent_queue
    if store is None:
        raise HTTPException(503, "持久任务队列未就绪")
    job = store.get_job(run_id)
    if job is None:
        raise HTTPException(404, "原任务记录不存在，请重新选择原输入")
    if job["status"] != "failed" and not (
        job["status"] == "completed" and (store.load_runs().get(run_id) or {}).get("state") == "partial"
    ):
        raise HTTPException(409, "仅可复跑已停止的未完成任务")
    root = _find_staging_run(_settings(), run_id)
    if root is None or read_current_release_pointer(root):
        raise HTTPException(409, "原任务暂存产出不可用，或已经正式发布")
    try:
        verified_mllm = ai_settings.reverified_job_mllm(job["payload"]["settings"])
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    # Preserve compact receipts before this attempt rewrites its current view.
    # No source media or derived video is copied for a retry.
    attempt_root = root / "JSON-Config-Files" / "Retry-Attempts" / str(job["attempts"])
    with _lock:
        current = store.get_job(run_id)
        if current is None or current["status"] != job["status"] or current["attempts"] != job["attempts"]:
            raise HTTPException(409, "任务状态已改变，请刷新恢复方案")
        if revision is not None and recovery_plan(run_id)["revision"] != revision:
            raise HTTPException(409, "恢复方案已过期，请刷新后重试")
        if _refresh_in_progress(run_id):
            raise HTTPException(409, "此实验正在刷新阶段，请等待完成")
        try:
            retained = [
                "JSON-Config-Files/partial_delivery.json",
                "JSON-Config-Files/run_metrics.json",
                "JSON-Config-Files/pipeline_status.json",
                "JSON-Config-Files/key_material_semantic_failures.json",
                "JSON-Config-Files/experiment_group_semantic_failures.json",
                "Partial-Results/Partial-Evidence-Report.html",
            ]
            for relative in retained:
                source = root / relative
                if source.is_file():
                    destination = attempt_root / source.name
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
            _retain_retry_outputs(root, int(job["attempts"]))
            state = store.retry_failed(run_id, verified_mllm=verified_mllm)
        except ValueError as exc:
            raise HTTPException(409, "任务已重新排队或尚未停止") from exc
        _runs[run_id] = state
    _queue_wakeup.set()
    return {"run_id": run_id, "state": "queued", "status_url": f"/api/runs/{run_id}",
            "source_copy_bytes": 0, "queue_persistence": "sqlite",
            "reuses_original_inputs": True, "cache_policy": "verified_reuse"}


def _experiment_speech_response(root: Path, name: str, staging: bool, query: str,
                                offset: int, limit: int, chunk: str | None,
                                release: str | None = None, *, fold: bool = False,
                                phrase: str | None = None, aliases: bool = True, hint: str | None = None) -> dict[str, Any]:
    pointer = read_current_release_pointer(root) or {} if not staging else {}
    current = str(pointer.get("release_id") or "") or None
    if release is not None and release != current:
        raise HTTPException(409, "实验归档已更新，请刷新页面")
    try:
        if current:
            if lightweight_release_integrity(root, pointer) != "release_manifest_verified":
                raise ValueError("invalid release manifest")
            manifest_path = (root / str(pointer.get("release_manifest") or "")).resolve()
            if not archive_contains(manifest_path, root):
                raise ValueError("invalid release path")
            for filename in ("speech.json", "speech_understanding.json", "speech_timeline.json", "speech_search.json", "speech_search_receipt.json", "capture_quality.json", "speech_group_understanding.json"):
                index = root / "JSON-Config-Files" / filename
                if index.is_file():
                    release_manifest = _read_json(manifest_path, {}) or {}
                    expected = next((item for item in release_manifest.get("manifests", {}).get("JSON-Config-Files", [])
                                     if item.get("path") == filename), None)
                    observed = speech_worker.file_record(index)
                    if not expected or observed != {"size": expected.get("size_bytes"), "sha256": expected.get("sha256")}:
                        raise ValueError("speech index does not match release")
        result = speech.archive_result(root, query, offset, limit, chunk, fold=fold, phrase=phrase, aliases=aliases, hint=hint)
        result["refresh_targets"] = []
        recording_path = root / "JSON-Config-Files/speech_understanding.json"
        if staging and recording_path.is_file():
            result["refresh_targets"] = [{"id": f"recording:{i}", "label": f"录音理解第 {i+1} 段",
                "revision": speech_worker.sha256(recording_path),
                "start_global_ms": part["speech_context"]["start_global_ms"], "end_global_ms": part["speech_context"]["end_global_ms"]}
                for i, part in enumerate((result.get("model_understanding") or {}).get("parts", [])) if part.get("speech_context")]
        groups_path = root / "JSON-Config-Files/experiment_group_understanding.json"
        if groups_path.is_file():
            from .speech_refresh import apply, input_path
            groups = apply(root, (_read_json(groups_path, {}) or {}).get("groups", []))
            result["group_understanding"] = groups
            if staging:
                result["refresh_targets"].extend({"id": "group:"+group["group_id"], "label": group.get("experiment_name") or group["group_id"],
                    "revision": speech_worker.sha256(groups_path), "start_global_ms": group["global_start_ms"], "end_global_ms": group["global_end_ms"]}
                    for group in groups if input_path(root, group["group_id"]).is_file())
        from .speech_timeline import load as load_speech_timeline
        timeline = load_speech_timeline(root)
        if timeline:
            prefix = f"/api/{'staging-runs' if staging else 'archives'}/{quote(name)}/speech-video"
            timeline["videos"] = [{key:value for key,value in video.items() if not key.startswith("_")} | {
                "url": prefix + f"?view={quote(video['view_id'])}&part={video['segment_ordinal']}&timeline={timeline['sha256']}" + (f"&release={quote(current)}" if current else "")}
                for video in timeline["videos"]]
        result["timeline"] = timeline
        for source in result["sources"]:
            for part in source["chunks"]:
                for spec in part["files"].values():
                    spec["url"] = (_staging_file_url(name, spec["path"]) if staging else
                                   _file_url(name, spec["path"], current))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "实验录音产物不可用或完整性检查未通过") from exc
    if not staging and (read_current_release_pointer(root) or {}) != pointer:
        raise HTTPException(409, "实验归档正在更新，请刷新页面")
    result["release_id"] = current
    return result


@app.get("/api/archives/{name}/speech")
def archive_speech(name: str, q: str = Query(default="", max_length=200),
                   offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500),
                   chunk: str | None = None, release: str | None = None, fold: bool = False,
                   phrase: str | None = None, aliases: bool = True, hint: str | None = None) -> dict[str, Any]:
    return _experiment_speech_response(_resolve_archive(name), name, False, q, offset, limit, chunk, release,
                                       fold=fold, phrase=phrase, aliases=aliases, hint=hint)


@app.get("/api/staging-runs/{run_id}/speech")
def staging_speech(run_id: str, q: str = Query(default="", max_length=200),
                   offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500),
                   chunk: str | None = None, fold: bool = False, phrase: str | None = None,
                   aliases: bool = True, hint: str | None = None) -> dict[str, Any]:
    return _experiment_speech_response(_resolve_staging_run(run_id), run_id, True, q, offset, limit, chunk,
                                       fold=fold, phrase=phrase, aliases=aliases, hint=hint)


@app.get("/api/speech-search")
def search_speech_library(q: str = Query(min_length=1, max_length=200),
                          archive_offset: int = Query(default=0, ge=0), aliases: bool = True) -> dict:
    roots = [(name, root, False) for name, root in _search_archive_roots(None)]
    for run_id, state in sorted(list(_runs.items())):
        if state.get("parent_run_id") or state.get("state") not in {"failed", "completed", "partial", "interrupted"}:
            continue
        try:
            roots.append((run_id, _resolve_staging_run(run_id), True))
        except HTTPException:
            continue
    rows, unavailable = [], []
    for name, root, staging in roots[archive_offset:archive_offset+10]:
        if not (root / "JSON-Config-Files/speech.json").is_file():
            continue
        try:
            if not staging:
                root = _resolve_archive(name)
            result = _experiment_speech_response(root, name, staging, q, 0, 100, None, aliases=aliases)
            for row in result["segments"]:
                params = f"chunk={quote(row['chunk_id'])}&t={row['playback_start_seconds']}"
                rows.append({**row, "experiment": root.parent.name if staging else name,
                    "run_id": name if staging else None, "experiment_matches": result["total"],
                    "href": f"#/{'stage' if staging else 'archive'}/{quote(name)}/speech?{params}"})
        except HTTPException:
            unavailable.append({"name": name, "reason": "来源或索引完整性未通过"})
    return {"segments": rows, "unavailable": unavailable, "searched_experiments": min(10, max(0, len(roots)-archive_offset)),
            "total_experiments": len(roots), "per_experiment_limit": 100,
            "next_archive_offset": archive_offset+10 if archive_offset+10 < len(roots) else None,
            "evidence_kind": "spoken_mention", "physical_action_confirmation": False}


@app.post("/api/{collection}/{name}/speech-evaluation")
def evaluate_speech(collection: str, name: str, reference: dict) -> dict:
    from .speech_search import evaluate
    if collection not in {"archives", "staging-runs"}:
        raise HTTPException(404, "实验不存在")
    staging = collection == "staging-runs"
    root = _resolve_staging_run(name) if staging else _resolve_archive(name)
    _experiment_speech_response(root, name, staging, "", 0, 1, None)
    try:
        return evaluate(root, reference)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/staging-runs/{run_id}/speech-video")
def staging_speech_video(run_id: str, view: str, timeline: str, part: int = Query(ge=0)) -> FileResponse:
    from .speech_timeline import video_path
    try:
        path = video_path(_resolve_staging_run(run_id), timeline, view, part)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "同步视频不可用或输入身份已变化") from exc
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control":"private, no-cache", "X-Content-Type-Options":"nosniff"})


@app.get("/api/archives/{name}/speech-video")
def archive_speech_video(name: str, view: str, timeline: str, part: int = Query(ge=0), release: str | None = None) -> FileResponse:
    from .speech_timeline import video_path
    root = _resolve_archive(name)
    # Reuse the release manifest check before granting original-source access.
    _experiment_speech_response(root, name, False, "", 0, 1, None, release)
    try:
        path = video_path(root, timeline, view, part)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "同步视频不可用或输入身份已变化") from exc
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control":"private, no-cache", "X-Content-Type-Options":"nosniff"})


@app.get("/api/nas-recordings")
def nas_recordings() -> dict[str, Any]:
    settings = _settings()
    if not directory_ingest_enabled(settings):
        return {"mode": "disabled", "recordings": [], "recording_count": 0}
    with _nas_monitor_lock:
        monitored = (
            json.loads(json.dumps(_nas_monitor_snapshot))
            if _nas_monitor_snapshot is not None
            else None
        )
    if monitored is not None:
        return _device_day_service.monitor_status(monitored) if (settings.get("device_day") or {}).get("enabled") else monitored
    if _nas_monitor_thread is not None and _nas_monitor_thread.is_alive():
        # The first scan can cover many recorder directories on a remote SMB
        # mount. Do not start a duplicate synchronous scan from each browser
        # refresh while the single background monitor is establishing its
        # first snapshot.
        return {
            "mode": "directory_metadata",
            "recordings": [],
            "recording_count": 0,
            "batches": [],
            "camera_directories": [],
            "camera_directory_count": 0,
            "unconfigured_camera_directories": [],
            "errors": [],
            "truncated": False,
            "source_copy_bytes": 0,
            "video_decode": False,
            "monitor": {
                "status": "starting",
                "observed_at": datetime.now().astimezone().isoformat(),
                "poll_seconds": float(
                    (settings.get("collection_ingest") or {}).get(
                        "poll_seconds", 30.0
                    )
                ),
                "consecutive_failures": 0,
            },
        }
    try:
        inventory = scan_recordings(settings)
        return inventory | {
            "monitor": {
                "status": "starting",
                "observed_at": datetime.now().astimezone().isoformat(),
                "poll_seconds": float(
                    (settings.get("collection_ingest") or {}).get(
                        "poll_seconds", 30.0
                    )
                ),
                "consecutive_failures": 0,
            }
        }
    except (OSError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc

@app.post("/api/nas-selections", status_code=201)
def select_nas_recordings(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        receipt = create_selection(_settings(), payload)
        return {"collection_id": receipt["collection_id"], "source_copy_bytes": 0}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, "无法读取 NAS 素材或保存实验清单") from exc

@app.post("/api/nas-batches/{batch_id}/runs", status_code=202)
def create_nas_batch_run(
    batch_id: str,
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn one recorder-native batch into a queued full-chain run."""

    if not re.fullmatch(r"nas-batch-[a-f0-9]{24}", batch_id):
        raise HTTPException(400, "无效的 NAS 采集批次编号")
    with _nas_batch_submission_lock:
        settings = _settings()
        try:
            inventory = scan_recordings(settings)
        except (OSError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc
        batch = next(
            (item for item in inventory.get("batches") or [] if item["batch_id"] == batch_id),
            None,
        )
        if batch is None:
            raise HTTPException(404, "NAS 采集批次不存在或内容已经变化，请刷新后重试")
        device_day_enabled = bool((settings.get("device_day") or {}).get("enabled"))
        if not (batch.get("device_day_processable") if device_day_enabled else batch.get("available")):
            raise HTTPException(
                409,
                {
                    "message": "该采集批次尚未完成",
                    "issues": batch.get("issues") or [],
                },
            )
        if device_day_enabled:
            selected_ids = {r["recording_id"] for r in batch["recordings"]}
            selected = [r for r in inventory["recordings"] if r["recording_id"] in selected_ids]
            if len(selected) != len(selected_ids):
                raise HTTPException(409, "采集分片清单已变化，请刷新后重试")
            return _device_day_service.submit(settings, selected) | {"batch_id": batch_id}
        request = payload or {}
        batch_timestamp = datetime.fromtimestamp(
            int(batch["recording_start_us"]) / 1_000_000
        ).strftime("%Y%m%d-%H%M%S")
        default_name = f"采集批次-{batch_timestamp}"
        experiment_name = str(request.get("experiment_name") or default_name).strip()
        collection_id = "nas-" + hashlib.sha256(batch_id.encode()).hexdigest()[:24]
        try:
            receipt = create_selection(
                settings,
                {
                    "experiment_name": experiment_name,
                    "recordings": [
                        {"recording_id": item["recording_id"]}
                        for item in batch["recordings"]
                    ],
                    "_collection_id": collection_id,
                    "_batch_id": batch_id,
                },
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        result = create_collection_run(
            receipt["collection_id"],
            background_tasks,
            {"experiment_name": experiment_name},
        )
        return result | {
            "batch_id": batch_id,
            "collection_id": receipt["collection_id"],
        }

@app.get("/api/model-candidates")
def model_candidates() -> dict[str, Any]:
    """Expose the committed, non-production model quality ledger to local Web."""

    registry_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "models"
        / "public-apparatus-candidates.json"
    )
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(503, "模型候选质量账本不可用") from exc
    candidates = payload.get("candidates")
    if (
        payload.get("schema_version")
        != "visioncortex-public-apparatus-candidate-registry/1"
        or not isinstance(candidates, dict)
    ):
        raise HTTPException(503, "模型候选质量账本格式无效")
    records = []
    for candidate_id, raw in candidates.items():
        if not isinstance(raw, dict):
            raise HTTPException(503, "模型候选质量账本包含无效记录")
        records.append({**raw, "candidate_id": str(candidate_id)})
    return {
        "schema_version": payload["schema_version"],
        "production_configuration_changed": False,
        "candidate_count": len(records),
        "candidates": records,
    }


@app.get('/health/live')
def liveness():
    from .build_identity import identity
    return {'status': 'alive', 'storage_maintenance': _storage_maintenance(), 'build': identity()}


@app.get('/api/runtime')
def runtime_status():
    from .runtime_process import role, worker_status
    from .runtime_control import ResourceCoordinator
    from .build_identity import identity
    settings = _settings()
    database = Path(settings['storage']['local_runtime_root']) / 'state' / 'resources.sqlite3'
    resources = ResourceCoordinator(database).snapshot() if database.is_file() else []
    return {'build': identity(), 'role': role(settings), 'worker': worker_status(settings),
            'resources': resources, 'queue': _persistent_queue.stats() if _persistent_queue else {}}
