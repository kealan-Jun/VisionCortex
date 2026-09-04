from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import re
import shutil
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

from .collection_catalog import discover_collections, get_collection
from .collection_state import record_collection_state
from .annotation_workspace import (
    export_reviewed_ground_truth,
    load_annotation_workspace,
    record_annotation_decision,
    resolve_annotation_image,
)
from .config import load_config
from .device_registry import load_device_registry, resolve_view_role
from .input_preflight import preflight_manifest_inputs
from .input_seal import build_input_seal, verify_input_seal, write_input_seal
from .model_certification import audit_production_model_certification
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
from .run_queue import DurableRunQueue, QueuedRunJob
from .schemas import RunManifest, VideoSegmentInput, ViewInput
from .storage import (
    ARCHIVE_DIRECTORIES,
    fixed_archive_staging_paths,
    initialize_nas_archive,
    prepare_from_nas_index,
    promote_fixed_archive,
    run_staging_roots,
    safe_archive_name,
)
from .upload_sessions import StorageReservationError, UploadSessionStore
from .web_access import (
    is_allowed_lan_client,
    valid_basic_authorization,
    validate_web_access_configuration,
    web_access_mode,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    validate_web_access_configuration()
    settings = _settings()
    _initialize_persistent_queue(settings)
    _expire_stale_upload_sessions(settings)
    try:
        _recover_jobs_from_archive_receipts(settings)
        _recover_orphaned_tasks()
        _start_queue_worker()
        _start_nas_monitor(settings)
        yield
    finally:
        _stop_nas_monitor()
        _stop_queue_worker()


app = FastAPI(
    title="VisionCortex Lab Evidence",
    version="0.2.0",
    lifespan=_lifespan,
)
_web_root = Path(__file__).with_name("web")
app.mount("/ui", StaticFiles(directory=_web_root), name="ui")
_lock = threading.Lock()
_gpu_job_lock = threading.Lock()
_upload_finalize_lock = threading.Lock()
_nas_batch_submission_lock = threading.Lock()
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
    while not _nas_monitor_stop.is_set():
        observed_at = datetime.now().astimezone().isoformat()
        try:
            inventory = scan_recordings(settings)
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
        stage = str(payload.get("stage") or "")
        if stage in {"completed", "failed"}:
            continue
        heartbeat = _runtime_activity_receipt(status_path)
        recovery = payload.get("recovery") if isinstance(payload.get("recovery"), dict) else {}
        if heartbeat["active"]:
            previous_stage = str(recovery.get("previous_stage") or "")
            if (
                stage == "interrupted"
                and recovery.get("status") == "orphaned_after_service_restart"
                and previous_stage
                and previous_stage not in {"completed", "failed", "interrupted"}
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


@app.middleware("http")
async def enforce_web_access(request: Request, call_next):
    """Keep the default local service open and fail closed for LAN service mode."""

    try:
        mode = web_access_mode()
        if mode == "local":
            return await call_next(request)
        client_host = request.client.host if request.client else None
        if not is_allowed_lan_client(client_host):
            return Response(
                "VisionCortex LAN access is not allowed from this network.",
                status_code=403,
                headers={"Cache-Control": "no-store"},
            )
        if not valid_basic_authorization(request.headers.get("authorization")):
            return Response(
                "VisionCortex login required.",
                status_code=401,
                headers={
                    "WWW-Authenticate": 'Basic realm="VisionCortex 3090 Ti", charset="UTF-8"',
                    "Cache-Control": "no-store",
                },
            )
    except RuntimeError:
        return Response(
            "VisionCortex Web access configuration is invalid.",
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    return await call_next(request)


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
    )
    if request.method == "POST" and starts_analysis:
        settings = _settings()
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
    configured = os.getenv("LABVISION_CONFIG")
    return load_config(Path(configured)) if configured else load_config()


def _archive_root(settings: dict[str, Any] | None = None) -> Path:
    return Path((settings or _settings())["storage"]["archive_root"])


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
    payload = job.payload
    settings = payload["settings"]
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
    while not _queue_stop.is_set():
        job = store.claim_next(
            _queue_worker_id,
            lease_seconds=_QUEUE_LEASE_SECONDS,
        )
        if job is None:
            _queue_wakeup.wait(_QUEUE_POLL_SECONDS)
            _queue_wakeup.clear()
            continue

        with _lock:
            existing_state = str((_runs.get(job.run_id) or {}).get("state") or "")
        if existing_state in {"completed", "failed"}:
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
                    message="服务重启后已恢复任务，等待 3090 Ti 继续执行",
                    recovered_from_durable_queue=True,
                )
            _dispatch_persisted_job(job)
            with _lock:
                final_state = str((_runs.get(job.run_id) or {}).get("state") or "")
                final_error = (_runs.get(job.run_id) or {}).get("error")
            if final_state not in {"completed", "failed"}:
                final_state = "failed"
                final_error = "Durable queue executor returned without a terminal run state"
                _update(job.run_id, state=final_state, progress=1.0, error=final_error)
            store.finish(
                job.run_id,
                _queue_worker_id,
                final_state,
                error=str(final_error) if final_error else None,
            )
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
        if kind not in {"video", "timestamp_csv"}:
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
        if segmented_layout:
            normalized_spec["segments"] = normalized_mappings
        else:
            normalized_spec.update(normalized_mappings[0])
        normalized_specs.append(normalized_spec)

        for segment_position, (_video_index, _csv_index, video_file, csv_file) in enumerate(
            mappings, 1
        ):
            for file_item in (video_file, csv_file):
                if file_item is None:
                    continue
                source_name = _safe_file_name(
                    file_item["source_name"],
                    "video" if file_item["kind"] == "video" else "timestamps",
                )
                prefix = (
                    f"segment-{segment_position:04d}-video-"
                    if file_item["kind"] == "video"
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
            and run.get("state") not in {"completed", "failed", "interrupted"}
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
        initialize_nas_archive(settings, archive_name)
    return archive_name, fixed_root, staging_root, history_root


def _reserve_fixed_benchmark(settings: dict[str, Any], run_id: str) -> Path:
    fixed_root, nas_root, history_root = fixed_archive_staging_paths(
        settings, _BENCHMARK_ARCHIVE_NAME, run_id
    )
    with _lock:
        if any(
            run.get("benchmark_archive") == _BENCHMARK_ARCHIVE_NAME
            and run.get("state") not in {"completed", "failed"}
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
        state = _runs.setdefault(run_id, {})
        state.update(values)
        if _persistent_queue is not None:
            _persistent_queue.save_run(run_id, state)


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
    input_seal_path: Path,
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
        "input_seal_relative_path": input_seal_path.relative_to(nas_root).as_posix(),
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
            if pipeline_status.get("stage") in {"completed", "failed"}:
                _update_queue_recovery_state(
                    nas_root,
                    str(pipeline_status["stage"]),
                    error=pipeline_status.get("error"),
                )
                continue
            if _runtime_activity_receipt(status_path)["active"]:
                continue
        manifest_relative = Path(str(payload.get("manifest_relative_path") or ""))
        seal_relative = Path(str(payload.get("input_seal_relative_path") or ""))
        if manifest_relative.is_absolute() or seal_relative.is_absolute():
            continue
        manifest_path = (nas_root / manifest_relative).resolve()
        input_seal_path = (nas_root / seal_relative).resolve()
        try:
            manifest_path.relative_to(nas_root)
            input_seal_path.relative_to(nas_root)
        except ValueError:
            continue
        if not manifest_path.is_file() or not input_seal_path.is_file():
            continue
        try:
            manifest = RunManifest.model_validate(
                yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            )
            seal = json.loads(input_seal_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError):
            continue
        if not verify_input_seal(seal) or seal.get(
            "experiment_id"
        ) != manifest.experiment_id:
            continue
        sources_available = True
        for source in seal.get("sources") or []:
            source_path = Path(str(source.get("path") or ""))
            try:
                source_stat = source_path.stat()
            except OSError:
                sources_available = False
                break
            if int(source.get("size_bytes") or -1) != int(source_stat.st_size):
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
            recovered_settings["storage"]["sync_to_nas"] = (
                str((payload.get("ingest") or {}).get("retention_mode")) == "nas_only"
            )
        run_state = {
            "state": "queued",
            "progress": 0.0,
            "message": "本地队列账本丢失后，已从归档输入封条恢复任务",
            "experiment_id": manifest.experiment_id,
            "nas_output": str(nas_root),
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
        metrics_path = root / "JSON-Config-Files" / "run_metrics.json"
        metrics = _read_json(metrics_path, {}) or {}
        metrics["web_ingest"] = {
            key: value
            for key, value in ingest.items()
            if key not in {"request_started_perf", "request_started_epoch"}
        }
        metrics["web_end_to_end"] = {
            "definition": "HTTP request arrival + multipart receive + local/NAS original retention + analysis + final NAS archive",
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
        metrics_path = root / "JSON-Config-Files" / "run_metrics.json"
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
            "definition": "benchmark request + NAS index staging/reuse + analysis + fixed NAS archive publication",
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
        metrics_path = root / "JSON-Config-Files" / "run_metrics.json"
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
            "definition": "collection card selection + zero-copy NAS index ingest + analysis + NAS archive",
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


def _stage_receipts_from_root(root: Path) -> list[dict[str, Any]]:
    """Return durable stage receipts without recursively walking NAS outputs."""

    receipt_root = root / "JSON-Config-Files" / "Stage-Receipts"
    if not receipt_root.is_dir():
        return []
    receipts: list[dict[str, Any]] = []
    for path in sorted(receipt_root.glob("*.json")):
        payload = _read_json(path, {}) or {}
        if not payload.get("stage"):
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
    status = _read_json(json_root / "pipeline_status.json", {}) or _read_json(
        root / "run_status.json", {}
    ) or {}
    live_telemetry = _read_json(json_root / "resource_telemetry_live.json", {}) or {}
    telemetry = _read_json(json_root / "resource_telemetry.json", {}) or {}
    metrics = _read_json(json_root / "run_metrics.json", {}) or _read_json(
        json_root / "run_metrics_live.json", {}
    ) or {}
    source_progress = _read_json(json_root / "source_progress.json", {}) or {}
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
        "telemetry_summary": {
            "sample_count": telemetry.get("sample_count"),
            "sampling_interval_seconds": telemetry.get("sampling_interval_seconds"),
            "stage_summaries": telemetry.get("stage_summaries") or {},
        },
        "metrics": metrics,
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
            return {**run, "observability": _run_snapshot_from_root(root)}
    return run


def _find_staging_run(settings: dict[str, Any], run_id: str) -> Path | None:
    for staging_root in run_staging_roots(settings):
        for path in staging_root.glob("*/*/JSON-Config-Files/pipeline_status.json"):
            archive_run_root = path.parent.parent
            if archive_run_root.name == run_id:
                return archive_run_root
    return None


def _file_url(archive_name: str, relative: str | Path) -> str:
    return f"/api/archive-file?archive={quote(archive_name)}&path={quote(Path(relative).as_posix())}"


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
        "local_path": str(local_destination) if retain_local_copy else None,
        "nas_path": str(nas_destination),
        "analysis_path": str(local_destination if retain_local_copy else nas_destination),
        "retention_mode": "local_and_nas" if retain_local_copy else "nas_only",
        "local_write_bytes": total if retain_local_copy else 0,
        "nas_write_bytes": total,
        "sha256": digest.hexdigest(),
        "duration_seconds": round(duration_seconds, 6),
        "effective_source_throughput_mib_s": round(
            total / duration_seconds / (1024 * 1024), 3
        ),
    }


def _execute_now(
    run_id: str,
    manifest: RunManifest,
    settings: dict[str, Any],
    nas_root: Path,
    ingest: dict[str, Any] | None = None,
) -> None:
    def progress(stage: str, value: float, message: str) -> None:
        _update(run_id, state=stage, progress=value, message=message, nas_output=str(nas_root))

    try:
        _update(run_id, state="running", progress=0.0, nas_output=str(nas_root))
        _update_queue_recovery_state(nas_root, "running")
        output = EvidencePipeline(settings, progress).run(manifest)
        if ingest is not None:
            _append_web_end_to_end_metrics([Path(output), nas_root], ingest, completed=True)
        _update(
            run_id,
            state="completed",
            progress=1.0,
            output=str(output),
            nas_output=str(nas_root),
            archive_url=f"/?archive={quote(nas_root.name)}",
        )
        _update_queue_recovery_state(nas_root, "completed")
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
    _update(run_id, state="queued", progress=0.0, message="等待 3090 Ti 计算资源")
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
        _append_fixed_benchmark_metrics([Path(output), nas_root], timing, completed=True)
        receipt = promote_fixed_archive(nas_root, fixed_root, history_root)
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
    _update(run_id, state="queued", progress=0.0, message="等待 3090 Ti 计算资源")
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
    existing_seal = _read_json(existing_seal_path, {}) or {}
    refreshed_seal = build_input_seal(
        manifest,
        source_mode="nas_segmented_virtual_timeline",
        sources=list(existing_seal.get("sources") or []),
        role_resolution=existing_seal.get("role_resolution") or {},
        copied_source_bytes=0,
        preflight=input_preflight,
    )
    write_input_seal(existing_seal_path, refreshed_seal)
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
        manifest_path=manifest_path,
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
        _append_collection_index_metrics(
            [Path(output), staging_root], timing, completed=True
        )
        promotion = promote_fixed_archive(staging_root, fixed_root, history_root)
        _update_queue_recovery_state(fixed_root, "completed")
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
        message="等待 3090 Ti 计算资源",
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
        monitor = dict((_nas_monitor_snapshot or {}).get("monitor") or {})
    return {
        "status": "ok",
        "analysis_ready": analysis_ready,
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
        "ark_key_configured": bool(os.getenv(settings["mllm"]["api_key_env"])),
        "mllm_enabled": bool(settings["mllm"].get("enabled")),
        "model": settings["mllm"]["model"],
        "archive_root": str(archive_root),
        "archive_available": archive_root.is_dir(),
        # Compatibility aliases for existing production Web clients. In local
        # mode the canonical archive_* fields above carry the accurate label.
        "nas_archive_root": str(archive_root),
        "nas_available": storage_mode == "nas" and archive_root.is_dir(),
        "fixed_benchmark": {
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
                (settings.get("collection_ingest") or {}).get(
                    "camera_directories", []
                )
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
def list_archives() -> dict[str, Any]:
    root = _archive_root()
    if not root.is_dir():
        return {"archive_root": str(root), "archives": []}
    archives = []
    for folder in sorted(
        (item for item in root.iterdir() if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        directories = {item.name for item in folder.iterdir() if item.is_dir()}
        if not directories.intersection(ARCHIVE_DIRECTORIES):
            continue
        experiment_root = folder / "Experiment-Clips"
        key_index = folder / "Key-Materials" / "Key-Materials-Model-Understanding.json"
        daily_manifest = folder / "JSON-Config-Files" / "daily_report_manifest.json"
        status = _read_json(folder / "JSON-Config-Files" / "pipeline_status.json", {}) or {}
        metrics = _read_json(folder / "JSON-Config-Files" / "run_metrics.json", {}) or {}
        archives.append(
            {
                "name": folder.name,
                "modified_at": datetime.fromtimestamp(folder.stat().st_mtime).isoformat(),
                "experiment_count": len(list(experiment_root.iterdir())) if experiment_root.is_dir() else 0,
                "key_event_count": len(_read_json(key_index, []) or []),
                "pipeline_stage": status.get("stage") or ("completed" if metrics else "archived"),
                "progress": status.get("progress"),
                "has_model_understanding": key_index.is_file(),
                "has_daily_report": daily_manifest.is_file(),
                "has_evidence_index": (
                    folder / "JSON-Config-Files" / INDEX_DB_NAME
                ).is_file(),
            }
        )
    return {"archive_root": str(root), "archives": archives}


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
    if directory_ingest_enabled(settings):
        try:
            validate_selection(settings, experiment_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        settings["storage"]["index_csv"] = str(
            selection_path(settings, experiment_id, ".csv")
        )


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
    result["event_url"] = f"/api/key-events/{quote(str(event['event_uid']))}?archive={quote(archive_name)}"
    result["artifact_references"] = [
        {
            **artifact,
            "url": (
                _file_url(archive_name, artifact["path"])
                if "://" not in str(artifact.get("path") or "")
                else None
            ),
            "sidecar_url": (
                _file_url(archive_name, artifact["sidecar_path"])
                if artifact.get("sidecar_path")
                else None
            ),
        }
        for artifact in event.get("artifact_references", [])
    ]
    return result


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
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Search one or every archive without loading monolithic event JSON arrays."""

    cursor_filters = {
        "archive": archive,
        "q": q,
        "action_type": action_type,
        "parent_event_id": parent_event_id,
        "cross_view": cross_view,
        "start_us": start_us,
        "end_us": end_us,
    }
    decoded_cursor = _decode_event_cursor(cursor, cursor_filters)
    collected: list[dict[str, Any]] = []
    for archive_name, root in _search_archive_roots(archive):
        manifest = _read_json(
            root / "JSON-Config-Files" / INDEX_MANIFEST_NAME, {}
        ) or {}
        archive_id = str(manifest.get("archive_id") or archive_name)
        after_peak_us = None
        after_event_uid = None
        if decoded_cursor:
            cursor_archive, cursor_peak, cursor_event = decoded_cursor
            if archive_id < cursor_archive:
                continue
            if archive_id == cursor_archive:
                after_peak_us = cursor_peak
                after_event_uid = cursor_event
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
        collected.extend(_attach_index_urls(archive_name, item) for item in items)
    collected.sort(
        key=lambda item: (
            str(item["archive_id"]),
            int(item.get("peak_timestamp_us") or 0),
            str(item["event_uid"]),
        )
    )
    has_more = len(collected) > limit
    page = collected[:limit]
    next_cursor = None
    if has_more and page:
        last = page[-1]
        next_cursor = _encode_event_cursor(
            str(last["archive_id"]),
            int(last.get("peak_timestamp_us") or 0),
            str(last["event_uid"]),
            cursor_filters,
        )
    return {
        "items": page,
        "count": len(page),
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

@app.get("/api/archives/{archive_name}")
def archive_detail(archive_name: str) -> dict[str, Any]:
    root = _resolve_archive(archive_name)
    return _archive_detail_from_root(root, archive_name)

@app.get("/api/staging-runs/{run_id}/archive")
def staging_archive_detail(run_id: str) -> dict[str, Any]:
    root = _resolve_staging_run(run_id)
    return _archive_detail_from_root(root, root.parent.name, staging_run_id=run_id)

def _archive_detail_from_root(
    root: Path, archive_name: str, *, staging_run_id: str | None = None
) -> dict[str, Any]:
    def file_url(name: str, relative: str | Path) -> str:
        return (
            _staging_file_url(staging_run_id, relative)
            if staging_run_id else _file_url(name, relative)
        )

    index_manifest_path = root / "JSON-Config-Files" / INDEX_MANIFEST_NAME
    index_manifest = _read_json(index_manifest_path, {}) or {}
    package = _read_json(root / "JSON-Config-Files" / "evidence_package.json", {}) or {}
    metrics = _read_json(root / "JSON-Config-Files" / "run_metrics.json", {}) or {}
    if not metrics:
        metrics = _read_json(root / "JSON-Config-Files" / "run_metrics_live.json", {}) or {}
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
    final_annotation = _read_json(final_annotation_path, {}) or {}
    key_material_verification, verification_by_event = (
        _summarize_key_material_verification(final_annotation)
    )
    _attach_archive_performance_display(metrics, acceptance)
    key_events = _read_json(
        root / "Key-Materials" / "Key-Materials-Model-Understanding.json", []
    ) or []
    daily_manifest = _read_json(
        root / "JSON-Config-Files" / "daily_report_manifest.json", {}
    ) or {}
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
            group = group_by_folder.get(folder.name, {})
            understanding = group.get("model_understanding") or {}
            first_person = folder / "First-Person.mp4"
            third_person = folder / "Third-Person.mp4"
            aligned = folder / "Aligned_First+Third.mp4"
            experiments.append(
                {
                    "folder": folder.name,
                    "name": group.get("experiment_name") or folder.name,
                    "continuity_type": group.get("continuity_type"),
                    "start_ms": group.get("global_start_ms"),
                    "end_ms": group.get("global_end_ms"),
                    "summary": understanding.get("overall_summary"),
                    "steps": understanding.get("steps") or [],
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
                    "name": group.get("experiment_name") or event.get("parent_event_id"),
                    "folder": group.get("archive_folder"),
                    "continuity_type": group.get("continuity_type"),
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
    links = {
        "experiment_understanding": file_url(
            archive_name, "JSON-Config-Files/Experiment-Groups-Step-Level-Analysis.json"
        ),
        "key_material_understanding": file_url(
            archive_name, "Key-Materials/Key-Materials-Model-Understanding.json"
        ),
        "key_material_category_index": file_url(
            archive_name, "Key-Materials/Key-Material-Category-Index.json"
        )
        if (root / "Key-Materials" / "Key-Material-Category-Index.json").is_file()
        else None,
        "metrics": file_url(archive_name, "JSON-Config-Files/run_metrics.json"),
        "acceptance": file_url(archive_name, "JSON-Config-Files/acceptance_report.json"),
        "quality_acceptance": file_url(
            archive_name, "JSON-Config-Files/quality_acceptance.json"
        ) if quality_path.is_file() else None,
        "evidence_package_eval": file_url(
            archive_name, "JSON-Config-Files/evidence_package_eval.json"
        ) if evidence_eval_path.is_file() else None,
        "key_material_recall_eval": file_url(
            archive_name, "JSON-Config-Files/key_material_recall_eval.json"
        ) if recall_eval_path.is_file() else None,
        "final_key_material_annotation": file_url(
            archive_name, "JSON-Config-Files/final_key_material_annotation.json"
        ) if final_annotation_path.is_file() else None,
        "daily_report_json": file_url(archive_name, daily_manifest["json"])
        if daily_manifest.get("json")
        else None,
        "daily_report_markdown": file_url(archive_name, daily_manifest["markdown"])
        if daily_manifest.get("markdown")
        else None,
        "daily_report_html": file_url(archive_name, daily_manifest["html"])
        if daily_manifest.get("html")
        else None,
        "daily_report_pdf": file_url(archive_name, daily_manifest["pdf"])
        if daily_manifest.get("pdf")
        else None,
        "daily_report_eval": file_url(archive_name, daily_manifest["evaluation"])
        if daily_manifest.get("evaluation")
        else None,
        "evidence_index_manifest": file_url(
            archive_name, f"JSON-Config-Files/{INDEX_MANIFEST_NAME}"
        ) if index_manifest_path.is_file() else None,
    }
    snapshot = _run_snapshot_from_root(root)
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
    return {
        "name": archive_name,
        "path": str(root),
        "staging_run_id": staging_run_id,
        "network_path": str(root),
        "experiments": experiments,
        "experiment_groups": [
            {
                "group_id": group.get("group_id"),
                "name": group.get("experiment_name") or group.get("archive_folder"),
                "folder": group.get("archive_folder"),
                "continuity_type": group.get("continuity_type"),
                "start_ms": group.get("global_start_ms"),
                "end_ms": group.get("global_end_ms"),
                "key_event_count": len(group.get("key_event_ids", [])),
            }
            for group in package_groups
        ],
        "key_events": normalized_events,
        "preliminary_materials": preliminary_materials,
        "metrics": metrics,
        "quality_acceptance": quality_acceptance,
        "key_material_recall_eval": key_material_recall_eval,
        "key_material_verification": key_material_verification,
        "observability": snapshot,
        "daily_report": daily_report,
        "daily_report_manifest": daily_manifest,
        "evidence_index": {
            **index_manifest,
            "search_url": (f"/api/staging-runs/{quote(staging_run_id)}/key-events"
                           if staging_run_id else f"/api/key-events?archive={quote(archive_name)}"),
        } if index_manifest else None,
        "links": links,
    }


@app.get("/api/archive-file")
def archive_file(archive: str, path: str) -> FileResponse:
    root = _resolve_archive(archive).resolve()
    candidate = (root / path).resolve()
    if not archive_contains(candidate, root) or not candidate.is_file():
        raise HTTPException(404, "档案文件不存在")
    return FileResponse(candidate)


@app.get("/api/staging-file")
def staging_file(run_id: str, path: str) -> FileResponse:
    root = _resolve_staging_run(run_id)
    candidate = (root / path).resolve()
    if not archive_contains(candidate, root) or not candidate.is_file():
        raise HTTPException(404, "staging 文件不存在")
    return FileResponse(candidate)


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
        settings["storage"]["active_archive_path"] = str(nas_root)
        queue_recovery_path = _write_queue_recovery_receipt(
            nas_root,
            run_id=run_id,
            state="queued",
            manifest_path=manifest_path,
            input_seal_path=input_seal_path,
            ingest=ingest,
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
                nas_output=str(nas_root),
            )
            queue_persistence = _schedule_job(
                background_tasks,
                run_id=run_id,
                kind="run",
                payload={
                    "manifest": manifest.model_dump(mode="json"),
                    "settings": settings,
                    "nas_root": str(nas_root),
                    "ingest": ingest,
                },
                fallback=_execute,
                fallback_args=(run_id, manifest, settings, nas_root, ingest),
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
            "nas_output": str(nas_root),
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
    settings["storage"]["sync_to_nas"] = True
    archive_name, nas_root = _reserve_archive(settings, experiment_name)
    settings["storage"]["active_archive_path"] = str(nas_root)
    run_id = uuid.uuid4().hex[:12]
    local_root = Path(settings["storage"]["local_input_root"]) / archive_name / run_id
    retention_mode = str(
        settings["storage"].get("web_upload_retention_mode", "local_and_nas")
    ).lower()
    if retention_mode not in {"local_and_nas", "nas_only"}:
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
    manifest_path = (
        nas_root / "JSON-Config-Files" / "input_manifest.yaml"
        if not retain_local_copy
        else local_root / "manifest.yaml"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
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
            ],
            "token_ledger": "JSON-Config-Files/run_metrics.json",
        },
    )
    _update(
        run_id,
        state="queued",
        progress=0.0,
        experiment_id=manifest.experiment_id,
        nas_output=str(nas_root),
    )
    queue_persistence = _schedule_job(
        background_tasks,
        run_id=run_id,
        kind="run",
        payload={
            "manifest": manifest.model_dump(mode="json"),
            "settings": settings,
            "nas_root": str(nas_root),
            "ingest": ingest,
        },
        fallback=_execute,
        fallback_args=(run_id, manifest, settings, nas_root, ingest),
    )
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "nas_output": str(nas_root),
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
                and run.get("state") not in {"completed", "failed", "interrupted"}
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
    settings["storage"]["sync_to_nas"] = True
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
    settings["storage"]["sync_to_nas"] = True
    archive_name, nas_root = _reserve_archive(settings, manifest.experiment_id)
    if archive_name != manifest.experiment_id:
        manifest = manifest.model_copy(update={"experiment_id": archive_name})
    settings["storage"]["active_archive_path"] = str(nas_root)
    run_id = uuid.uuid4().hex[:12]
    _update(run_id, state="queued", progress=0.0, experiment_id=manifest.experiment_id)
    queue_persistence = _schedule_job(
        background_tasks,
        run_id=run_id,
        kind="run",
        payload={
            "manifest": manifest.model_dump(mode="json"),
            "settings": settings,
            "nas_root": str(nas_root),
            "ingest": None,
        },
        fallback=_execute,
        fallback_args=(run_id, manifest, settings, nas_root),
    )
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "nas_output": str(nas_root),
        "queue_persistence": queue_persistence,
    }


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    with _lock:
        runs = [
            _hydrate_run_snapshot({"run_id": run_id, **values})
            for run_id, values in _runs.items()
        ]
    runs.sort(key=lambda item: str(item.get("run_id")), reverse=True)
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
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
        return monitored
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
        if not batch.get("available"):
            raise HTTPException(
                409,
                {
                    "message": "该采集批次尚未完成",
                    "issues": batch.get("issues") or [],
                },
            )
        request = payload or {}
        default_name = datetime.fromtimestamp(
            int(batch["recording_start_us"]) / 1_000_000
        ).strftime("采集批次-%Y%m%d-%H%M%S")
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
