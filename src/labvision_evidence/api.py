from __future__ import annotations

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
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .config import load_config
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
        for phase in ("coarse", "fine")
    }
    return {
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
        "freshness": {
            "status_updated_at": status.get("updated_at"),
            "telemetry_updated_at": live_telemetry.get("updated_at"),
        },
        "sources": {
            "status": "JSON-Config-Files/pipeline_status.json",
            "telemetry_live": "JSON-Config-Files/resource_telemetry_live.json",
            "telemetry_final": "JSON-Config-Files/resource_telemetry.json",
            "metrics": "JSON-Config-Files/run_metrics.json",
            "metrics_live": "JSON-Config-Files/run_metrics_live.json",
            "scan_runtime_coarse": "JSON-Config-Files/scan_runtime_coarse.json",
            "scan_runtime_fine": "JSON-Config-Files/scan_runtime_fine.json",
            "source_progress": "JSON-Config-Files/source_progress.json",
        },
    }


def _hydrate_run_snapshot(run: dict[str, Any]) -> dict[str, Any]:
    for key in ("nas_staging", "output", "nas_output"):
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
            }
        )
    return {"archive_root": str(root), "archives": archives}


@app.get("/api/archives/{archive_name}")
def archive_detail(archive_name: str) -> dict[str, Any]:
    root = _resolve_archive(archive_name)
    package = _read_json(root / "JSON-Config-Files" / "evidence_package.json", {}) or {}
    metrics = _read_json(root / "JSON-Config-Files" / "run_metrics.json", {}) or {}
    acceptance = _read_json(root / "JSON-Config-Files" / "acceptance_report.json", {}) or {}
    quality_acceptance = _read_json(
        root / "JSON-Config-Files" / "quality_acceptance.json", {}
    ) or {}
    benchmark = acceptance.get("preprocessing_full_run") or {}
    if benchmark.get("seconds") is not None:
        metrics["display_preprocessing_seconds"] = benchmark["seconds"]
        metrics["display_preprocessing_source"] = "full_run_benchmark"
    else:
        metrics["display_preprocessing_seconds"] = (metrics.get("preprocessing_sla") or {}).get(
            "actual_seconds"
        )
        metrics["display_preprocessing_source"] = "current_run"
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
    group_by_folder = {
        str(group.get("archive_folder")): group for group in package.get("experiment_groups", [])
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
            }
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
        ),
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
    }
    return {
        "name": archive_name,
        "path": str(_archive_root() / archive_name),
        "network_path": str(root),
        "experiments": experiments,
        "key_events": normalized_events,
        "metrics": metrics,
        "quality_acceptance": quality_acceptance,
        "observability": _run_snapshot_from_root(root),
        "daily_report": daily_report,
        "daily_report_manifest": daily_manifest,
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
