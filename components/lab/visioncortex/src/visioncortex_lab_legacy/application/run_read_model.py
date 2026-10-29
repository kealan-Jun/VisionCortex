"""Task projections from durable queue and archive receipts."""

from __future__ import annotations
from .dependencies import ports
import json
from pathlib import Path
from typing import Any
from .contracts import ApplicationError
from ..run_insights import (
    hardware_summary,
    timing_summary,
    token_summary,
    with_operation_refreshes,
)
from ..partial_delivery import (
    component_results,
    registered_partial_root,
)
from ..storage import (
    ARCHIVE_DIRECTORIES,
    run_staging_roots,
)

from . import runtime_host


def _runtime_activity_receipt(
    status_path: Path,
    *,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Describe recent pipeline activity without parsing a concurrently written JSON file."""

    observed_at = ports.time.time() if now_epoch is None else float(now_epoch)
    latest_path: Path | None = None
    latest_mtime: float | None = None
    for file_name in runtime_host._RUNTIME_HEARTBEAT_FILES:
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
            "stale_after_seconds": runtime_host._RUNTIME_HEARTBEAT_STALE_SECONDS,
        }

    age_seconds = max(0.0, observed_at - latest_mtime)
    return {
        "active": age_seconds <= runtime_host._RUNTIME_HEARTBEAT_STALE_SECONDS,
        "latest_path": latest_path.name,
        "latest_mtime_epoch": latest_mtime,
        "age_seconds": age_seconds,
        "stale_after_seconds": runtime_host._RUNTIME_HEARTBEAT_STALE_SECONDS,
    }


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _pipeline_status_from_root(root: Path) -> dict[str, Any]:
    return (
        _read_json(root / "JSON-Config-Files/pipeline_status.json", {})
        or _read_json(root / "run_status.json", {})
        or {}
    )


def _latest_result_review(root: Path, *, save=False) -> dict:
    from ..result_review import inspect

    try:
        return inspect(root, save=save)
    except (OSError, ValueError, KeyError, TypeError):
        return {"available": False, "reason": "当前记录不足以执行版本与完整性检查"}


def _merged_run_metrics(root: Path) -> dict[str, Any]:
    json_root = root / "JSON-Config-Files"
    final = _read_json(json_root / "run_metrics.json", {}) or {}
    live = _read_json(json_root / "run_metrics_live.json", {}) or {}
    metrics = (
        live
        if str(live.get("run_started_at", "")) > str(final.get("run_started_at", ""))
        else final or live
    )
    delivery = _read_json(json_root / "delivery_metrics.json", {}) or {}
    # A retry keeps the previous receipts on disk. Its delivery timing must not
    # overwrite the newer live attempt's metrics.
    if not final or metrics.get("run_started_at") == final.get("run_started_at"):
        metrics.update(delivery)
    publication = (ports.read_current_release_pointer(root) or {}).get("publication")
    if isinstance(publication, dict):
        metric_key = str(publication.get("metric_key") or "").strip()
        if metric_key:
            metrics[metric_key] = {
                key: value for key, value in publication.items() if key != "metric_key"
            }
    # Add operation-only refresh receipts once; never replace the original run's
    # wall clock with follow-up request time or count local reuse as new usage.
    return with_operation_refreshes(root, metrics)


def _current_attempt_started_at(status: dict[str, Any]) -> float | None:
    try:
        return ports.datetime.fromisoformat(status["updated_at"]).timestamp() - float(
            status["elapsed_seconds"]
        )
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
                if (
                    ports.datetime.fromisoformat(payload["completed_at"]).timestamp()
                    < started_at - 1
                ):
                    continue
            except (KeyError, TypeError, ValueError):
                # Undated legacy receipts cannot establish completion of a
                # timestamped attempt; the files remain available on disk.
                continue
        artifacts = []
        for raw in payload.get("artifacts") or []:
            artifact_path = Path(str(raw))
            resolved = (
                artifact_path if artifact_path.is_absolute() else root / artifact_path
            )
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
                "reason": payload.get("reason"),
                "reused": bool(payload.get("reused")),
                "archive_status": payload.get("archive_status"),
                "completed_at": payload.get("completed_at"),
                "run_elapsed_seconds": payload.get("run_elapsed_seconds"),
                "stage_duration_seconds": payload.get("stage_duration_seconds"),
                "archive_mode": payload.get("archive_mode"),
                "artifacts": artifacts,
                "receipt": f"JSON-Config-Files/Stage-Receipts/{path.name}",
                "version_manifest": payload.get("version_manifest"),
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
            "hardware": hardware_summary(
                telemetry, live_telemetry, metrics.get("run_started_at")
            ),
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
        root = registered_partial_root(runtime_host._settings(), run)
        return {
            **run,
            "observability": _run_snapshot_from_root(root) if root else {},
            "result_available": root is not None,
        }
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
                    previous_status = (
                        _read_json(
                            root
                            / "JSON-Config-Files"
                            / "Retry-Attempts"
                            / str(attempt)
                            / "pipeline_status.json",
                            {},
                        )
                        or previous_status
                    )
                snapshot["previous_attempt_status"] = previous_status
                snapshot["status"] = {
                    "stage": "queued",
                    "progress": 0.0,
                    "message": run.get("message"),
                    "elapsed_seconds": 0.0,
                }
            return {**run, "observability": snapshot}
    return run


def _find_staging_run(settings: dict[str, Any], run_id: str) -> Path | None:
    record = runtime_host._runs.get(run_id) or {}
    if record.get("origin") == "registered_local_partial":
        return registered_partial_root(settings, record)
    for staging_root in run_staging_roots(settings):
        # Input validation can fail before the pipeline writes its first status.
        # Use the durable job's exact directory, still constrained to this store.
        if record.get("nas_staging"):
            candidate = Path(record["nas_staging"]).resolve()
            if (
                candidate.name == run_id
                and candidate.parent.parent == staging_root.resolve()
                and candidate.is_dir()
            ):
                return candidate
        for path in staging_root.glob("*/*/JSON-Config-Files/pipeline_status.json"):
            archive_run_root = path.parent.parent
            if archive_run_root.name == run_id:
                return archive_run_root
    return None


def _resolve_staging_run(run_id: str) -> Path:
    root = _find_staging_run(runtime_host._settings(), run_id)
    if root is None:
        raise ApplicationError(404, "staging run_id 不存在")
    return root.resolve()


def list_runs() -> dict[str, Any]:
    if runtime_host._persistent_queue is not None:
        with runtime_host._lock:
            runtime_host._runs.update(runtime_host._persistent_queue.load_runs())
    with runtime_host._lock:
        runs = [
            _hydrate_run_snapshot({"run_id": run_id, **values})
            for run_id, values in runtime_host._runs.items()
        ]
    runs.sort(key=lambda item: str(item.get("run_id")), reverse=True)
    return {"runs": runs}


def get_run(run_id: str) -> dict[str, Any]:
    if runtime_host._persistent_queue is not None:
        fresh = runtime_host._persistent_queue.load_run(run_id)
        if fresh:
            with runtime_host._lock:
                runtime_host._runs[run_id] = fresh
    with runtime_host._lock:
        state = dict(runtime_host._runs.get(run_id, {}))
    if not state:
        root = _find_staging_run(runtime_host._settings(), run_id)
        if root is None:
            raise ApplicationError(404, "run_id 不存在")
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


class RunReadModel:
    """Durable task snapshots without HTTP or model execution."""

    @staticmethod
    def snapshot(root: Path) -> dict[str, Any]:
        return _run_snapshot_from_root(root)

    @staticmethod
    def get(run_id: str) -> dict[str, Any]:
        return get_run(run_id)

    @staticmethod
    def list() -> dict[str, Any]:
        return list_runs()


run_read_model = RunReadModel()
