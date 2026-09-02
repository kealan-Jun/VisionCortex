from __future__ import annotations

import hashlib
import json
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
from fastapi.responses import FileResponse, HTMLResponse
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
from .schemas import RunManifest, ViewInput
from .storage import (
    ARCHIVE_DIRECTORIES,
    fixed_archive_staging_paths,
    initialize_nas_archive,
    prepare_from_nas_index,
    promote_fixed_archive,
    safe_archive_name,
)
from .web_access import (
    is_allowed_lan_client,
    valid_basic_authorization,
    validate_web_access_configuration,
    web_access_mode,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    validate_web_access_configuration()
    _initialize_persistent_queue(_settings())
    try:
        _recover_orphaned_tasks()
        _start_queue_worker()
        yield
    finally:
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
_runs: dict[str, dict[str, Any]] = {}
_persistent_queue: DurableRunQueue | None = None
_queue_thread: threading.Thread | None = None
_queue_stop = threading.Event()
_queue_wakeup = threading.Event()
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
    staging_root = root / ".VisionCortex-Run-Staging"
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
    global _persistent_queue

    store = DurableRunQueue(_queue_database_path(settings))
    restored_runs = store.load_runs()
    with _lock:
        _runs.clear()
        _runs.update(restored_runs)
    _persistent_queue = store


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
            artifacts.append(
                {
                    "path": str(raw),
                    "name": artifact_path.name,
                    "available": resolved.exists(),
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
    staging_root = Path(settings["storage"]["archive_root"]) / ".VisionCortex-Run-Staging"
    if not staging_root.is_dir():
        return None
    matches = []
    for path in staging_root.glob("*/*/JSON-Config-Files/pipeline_status.json"):
        archive_run_root = path.parent.parent
        if archive_run_root.name == run_id:
            matches.append(archive_run_root)
    return matches[0] if matches else None


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
    except Exception as exc:
        if ingest is not None:
            _append_web_end_to_end_metrics([nas_root], ingest, completed=False)
        _update(run_id, state="failed", progress=1.0, error=f"{type(exc).__name__}: {exc}")


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
        _update(
            run_id,
            state="running",
            progress=0.02,
            nas_output=str(fixed_root),
            nas_staging=str(staging_root),
        )
        output = EvidencePipeline(settings, progress).run(manifest)
        _append_collection_index_metrics(
            [Path(output), staging_root], timing, completed=True
        )
        promotion = promote_fixed_archive(staging_root, fixed_root, history_root)
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
    with _gpu_job_lock:
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
    storage_mode = "nas" if settings["storage"].get("sync_to_nas") else "local_development"
    archive_root = _archive_root(settings)
    input_mode = (
        "NAS 15-minute segments / zero-copy virtual timeline"
        if storage_mode == "nas"
        else "local files / zero-copy source references"
    )
    return {
        "status": "ok",
        "storage_mode": storage_mode,
        "archive_label": "NAS 正式归档" if storage_mode == "nas" else "本地开发归档",
        "web_upload_retention_mode": settings["storage"].get(
            "web_upload_retention_mode", "local_and_nas"
        ),
        "minimum_capacity": "6 views x 3 hours",
        "view_count_policy": "dynamic",
        "minimum_cross_view_sources": 2,
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
            "mode": "index_metadata_poll",
            "poll_seconds": float(
                (settings.get("collection_ingest") or {}).get("poll_seconds", 30.0)
            ),
            "recursive_nas_scan": False,
            "index_csv": str(settings["storage"]["index_csv"]),
            "device_registry_path": settings["storage"].get("device_registry_path"),
        },
        "execution_queue": {
            "policy": "single_gpu_one_job_at_a_time",
            "persistence": "sqlite" if _persistent_queue is not None else "not_initialized",
            "survives_web_service_restart": _persistent_queue is not None,
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


@app.get("/api/archives/{archive_name}")
def archive_detail(archive_name: str) -> dict[str, Any]:
    root = _resolve_archive(archive_name)
    index_manifest_path = root / "JSON-Config-Files" / INDEX_MANIFEST_NAME
    index_manifest = _read_json(index_manifest_path, {}) or {}
    package = _read_json(root / "JSON-Config-Files" / "evidence_package.json", {}) or {}
    metrics = _read_json(root / "JSON-Config-Files" / "run_metrics.json", {}) or {}
    acceptance = _read_json(root / "JSON-Config-Files" / "acceptance_report.json", {}) or {}
    quality_path = root / "JSON-Config-Files" / "quality_acceptance.json"
    quality_acceptance = _read_json(quality_path, {}) or {}
    evidence_eval_path = root / "JSON-Config-Files" / "evidence_package_eval.json"
    evidence_eval = _read_json(evidence_eval_path, {}) or {}
    recall_eval_path = (
        root / "JSON-Config-Files" / "key_material_recall_eval.json"
    )
    key_material_recall_eval = _read_json(recall_eval_path, {}) or {}
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
                    "aligned_video_url": _file_url(
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
                "aligned_frame_url": _file_url(archive_name, frame["path"]) if frame else None,
                "aligned_clip_url": _file_url(archive_name, clip["path"]) if clip else None,
                "dual_view_material_ready": bool(frame and clip),
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
        "experiment_understanding": _file_url(
            archive_name, "JSON-Config-Files/Experiment-Groups-Step-Level-Analysis.json"
        ),
        "key_material_understanding": _file_url(
            archive_name, "Key-Materials/Key-Materials-Model-Understanding.json"
        ),
        "key_material_category_index": _file_url(
            archive_name, "Key-Materials/Key-Material-Category-Index.json"
        )
        if (root / "Key-Materials" / "Key-Material-Category-Index.json").is_file()
        else None,
        "metrics": _file_url(archive_name, "JSON-Config-Files/run_metrics.json"),
        "acceptance": _file_url(archive_name, "JSON-Config-Files/acceptance_report.json"),
        "quality_acceptance": _file_url(
            archive_name, "JSON-Config-Files/quality_acceptance.json"
        ) if quality_path.is_file() else None,
        "evidence_package_eval": _file_url(
            archive_name, "JSON-Config-Files/evidence_package_eval.json"
        ) if evidence_eval_path.is_file() else None,
        "key_material_recall_eval": _file_url(
            archive_name, "JSON-Config-Files/key_material_recall_eval.json"
        ) if recall_eval_path.is_file() else None,
        "daily_report_json": _file_url(archive_name, daily_manifest["json"])
        if daily_manifest.get("json")
        else None,
        "daily_report_markdown": _file_url(archive_name, daily_manifest["markdown"])
        if daily_manifest.get("markdown")
        else None,
        "daily_report_html": _file_url(archive_name, daily_manifest["html"])
        if daily_manifest.get("html")
        else None,
        "daily_report_pdf": _file_url(archive_name, daily_manifest["pdf"])
        if daily_manifest.get("pdf")
        else None,
        "daily_report_eval": _file_url(archive_name, daily_manifest["evaluation"])
        if daily_manifest.get("evaluation")
        else None,
        "evidence_index_manifest": _file_url(
            archive_name, f"JSON-Config-Files/{INDEX_MANIFEST_NAME}"
        ) if index_manifest_path.is_file() else None,
    }
    return {
        "name": archive_name,
        "path": str(_archive_root() / archive_name),
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
        "metrics": metrics,
        "quality_acceptance": quality_acceptance,
        "key_material_recall_eval": key_material_recall_eval,
        "observability": _run_snapshot_from_root(root),
        "daily_report": daily_report,
        "daily_report_manifest": daily_manifest,
        "evidence_index": {
            **index_manifest,
            "search_url": f"/api/key-events?archive={quote(archive_name)}",
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
