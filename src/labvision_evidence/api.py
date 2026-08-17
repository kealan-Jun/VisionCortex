from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import unicodedata
import uuid
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

from .config import load_config
from .indexing import (
    INDEX_DB_NAME,
    INDEX_MANIFEST_NAME,
    get_indexed_event,
    get_indexed_evidence,
    search_archive_index,
)
from .pipeline import EvidencePipeline
from .schemas import RunManifest, ViewInput
from .storage import (
    ARCHIVE_DIRECTORIES,
    fixed_archive_staging_paths,
    initialize_nas_archive,
    prepare_from_nas_index,
    promote_fixed_archive,
    safe_archive_name,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _recover_orphaned_tasks()
    yield


app = FastAPI(
    title="VisionCortex Lab Evidence",
    version="0.2.0",
    lifespan=_lifespan,
)
_web_root = Path(__file__).with_name("web")
app.mount("/ui", StaticFiles(directory=_web_root), name="ui")
_lock = threading.Lock()
_runs: dict[str, dict[str, Any]] = {}
_BENCHMARK_EXPERIMENT_ID = "exp_20260810_144014_e918b762"
_BENCHMARK_ARCHIVE_NAME = "Six-View-Three-Hour-Experiment-2026-08-13"
_BENCHMARK_SUBMISSION_PROTOCOL_VERSION = 1
_PHYSICAL_ACTION_TYPES = (
    "hand_object_contact",
    "object_movement",
    "liquid_transfer",
    "container_state_change",
    "device_panel_operation",
)


def _recover_orphaned_tasks() -> None:
    """Make stale durable `running` states honest after a service restart."""

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
        if payload.get("stage") in {"completed", "failed", "interrupted"}:
            continue
        payload.update(
            {
                "stage": "interrupted",
                "message": "服务重启时发现未完成任务；检测账本与模型结果缓存保留，可使用同一输入重新启动以续跑。",
                "updated_at": datetime.now().astimezone().isoformat(),
                "recovery": {
                    "status": "orphaned_after_service_restart",
                    "resumable": True,
                },
            }
        )
        _write_json_atomic(status_path, payload)


@app.middleware("http")
async def record_web_ingest_start(request: Request, call_next):
    """Capture the request boundary before multipart parsing starts."""

    if request.method == "POST" and request.url.path == "/api/runs":
        request.state.ingest_started_perf = time.perf_counter()
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
                "requires_web_service_alive": True,
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
    total_seconds = round(time.perf_counter() - float(ingest["request_started_perf"]), 6)
    for root in dict.fromkeys(path.resolve() for path in roots):
        metrics_path = root / "JSON-Config-Files" / "run_metrics.json"
        metrics = _read_json(metrics_path, {}) or {}
        metrics["web_ingest"] = {
            key: value for key, value in ingest.items() if key != "request_started_perf"
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
    total_seconds = round(time.perf_counter() - float(timing["request_started_perf"]), 6)
    for root in dict.fromkeys(path.resolve() for path in roots):
        metrics_path = root / "JSON-Config-Files" / "run_metrics.json"
        metrics = _read_json(metrics_path, {}) or {}
        metrics["nas_index_ingest"] = {
            key: value
            for key, value in timing.items()
            if key not in {"request_started_perf", "request_received_at"}
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


def _resolve_archive(archive_name: str) -> Path:
    root = _archive_root().resolve()
    candidate = (root / archive_name).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise HTTPException(404, "实验档案不存在")
    return candidate


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


def _execute(
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


def _execute_fixed_benchmark(
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
        manifest.experiment_id = _BENCHMARK_ARCHIVE_NAME
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_web_root / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = _settings()
    storage_mode = "nas" if settings["storage"].get("sync_to_nas") else "local_development"
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
        "model": settings["mllm"]["model"],
        "nas_archive_root": str(_archive_root(settings)),
        "nas_available": _archive_root(settings).is_dir(),
        "fixed_benchmark": {
            "experiment_id": _BENCHMARK_EXPERIMENT_ID,
            "archive_name": _BENCHMARK_ARCHIVE_NAME,
            "submission_protocol_version": _BENCHMARK_SUBMISSION_PROTOCOL_VERSION,
            "index_csv": str(settings["storage"]["index_csv"]),
            "input_mode": "NAS 15-minute segments / zero-copy virtual timeline",
            "local_runtime_root": str(settings["storage"]["local_runtime_root"]),
            "local_cache_root": str(settings["storage"]["local_cache_root"]),
        },
    }


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


def _decode_event_cursor(value: str | None) -> tuple[str, int, str] | None:
    if not value:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        return str(payload[0]), int(payload[1]), str(payload[2])
    except (ValueError, TypeError, IndexError, binascii.Error, json.JSONDecodeError) as exc:
        raise HTTPException(400, "无效的关键事件分页 cursor") from exc


def _encode_event_cursor(archive_id: str, peak_timestamp_us: int, event_uid: str) -> str:
    payload = json.dumps(
        [archive_id, int(peak_timestamp_us), event_uid],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


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


@app.get("/api/key-events")
def search_key_events(
    archive: str | None = None,
    q: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    cross_view: bool | None = None,
    liquid_state_status: str | None = None,
    liquid_present: bool | None = None,
    visible_flow: bool | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Search one or every archive without loading monolithic event JSON arrays."""

    decoded_cursor = _decode_event_cursor(cursor)
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
            liquid_state_status=liquid_state_status,
            liquid_present=liquid_present,
            visible_flow=visible_flow,
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
        )
    return {
        "items": page,
        "count": len(page),
        "next_cursor": next_cursor,
        "canonical_source": "archived JSON",
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
                        archive_name, aligned.relative_to(root)
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
        "metrics": _file_url(archive_name, "JSON-Config-Files/run_metrics.json"),
        "acceptance": _file_url(archive_name, "JSON-Config-Files/acceptance_report.json"),
        "quality_acceptance": _file_url(
            archive_name, "JSON-Config-Files/quality_acceptance.json"
        ) if quality_path.is_file() else None,
        "evidence_package_eval": _file_url(
            archive_name, "JSON-Config-Files/evidence_package_eval.json"
        ) if evidence_eval_path.is_file() else None,
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
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise HTTPException(404, "档案文件不存在")
    return FileResponse(candidate)


@app.post("/api/archives/{archive_name}/open")
def open_archive_folder(archive_name: str) -> dict[str, str]:
    root = _resolve_archive(archive_name)
    if os.name != "nt":
        raise HTTPException(501, "仅支持在运行服务的 Windows 机器上打开文件夹")
    subprocess.Popen(["explorer.exe", str(root)])
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
        key: value for key, value in ingest.items() if key != "request_started_perf"
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
    background_tasks.add_task(_execute, run_id, manifest, settings, nas_root, ingest)
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "nas_output": str(nas_root),
        "archive_url": f"/?archive={quote(archive_name)}",
    }


@app.post("/api/benchmarks/six-view-three-hour/runs", status_code=202)
def create_fixed_benchmark_run(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Rerun the registered six-view benchmark into its fixed NAS archive."""

    request_started_perf = time.perf_counter()
    request_received_at = datetime.now().astimezone().isoformat()
    settings = _settings()
    settings["storage"]["sync_to_nas"] = True
    run_id = f"benchmark-{uuid.uuid4().hex[:10]}"
    nas_root = _reserve_fixed_benchmark(settings, run_id)

    timing = {
        "request_started_perf": request_started_perf,
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
    background_tasks.add_task(_execute_fixed_benchmark, run_id, settings, nas_root, timing)
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
    background_tasks.add_task(_execute, run_id, manifest, settings, nas_root)
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "nas_output": str(nas_root),
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
