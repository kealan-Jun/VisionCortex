"""HTTP adapters for collections"""

from __future__ import annotations
from ..application.dependencies import ports
import hashlib
import json
import re
import uuid
from collections import Counter
from typing import Any
from urllib.parse import quote
from fastapi import BackgroundTasks, HTTPException, Query
from ..collection_catalog import discover_collections
from ..nas_recordings import (
    create_selection,
    enabled as directory_ingest_enabled,
    selection_path,
)
from ..pagination import decode_cursor, encode_cursor

from ..application import archive_read_model, runtime_host

from fastapi import APIRouter

router = APIRouter()


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
            runtime_host._settings(),
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


def collection_detail(experiment_id: str) -> dict[str, Any]:
    try:
        return ports.get_collection(runtime_host._settings(), experiment_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"无法读取采集批次索引: {exc}") from exc


def create_fixed_benchmark_run(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Rerun the registered six-view benchmark into its fixed NAS archive."""

    request_started_perf = ports.time.perf_counter()
    request_started_epoch = ports.time.time()
    request_received_at = ports.datetime.now().astimezone().isoformat()
    settings = runtime_host._settings()
    try:
        benchmark = runtime_host.benchmark_registration(settings, required=True)
    except ValueError as exc:
        raise HTTPException(404, "当前部署未配置固定六路基准。") from exc
    archive_name = benchmark["archive_name"]
    settings["storage"]["sync_to_nas"] = True
    run_id = f"benchmark-{uuid.uuid4().hex[:10]}"
    nas_root = runtime_host._reserve_fixed_benchmark(settings, run_id)

    timing = {
        "request_started_perf": request_started_perf,
        "request_started_epoch": request_started_epoch,
        "request_received_at": request_received_at,
        "experiment_id": benchmark["experiment_id"],
        "archive_name": archive_name,
        "index_csv": str(settings["storage"]["index_csv"]),
    }
    runtime_host._update(
        run_id,
        state="queued",
        progress=0.0,
        experiment_id=archive_name,
        benchmark_archive=archive_name,
        nas_output=str(
            archive_read_model._archive_root(settings)
            / archive_name
        ),
        nas_staging=str(nas_root),
    )
    submission_receipt = runtime_host._write_fixed_benchmark_submission_receipt(
        nas_root,
        run_id,
        request_received_at,
    )
    queue_persistence = runtime_host._schedule_job(
        background_tasks,
        run_id=run_id,
        kind="fixed_benchmark",
        payload={
            "settings": settings,
            "nas_root": str(nas_root),
            "timing": timing,
        },
        fallback=runtime_host._execute_fixed_benchmark,
        fallback_args=(run_id, settings, nas_root, timing),
    )
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "nas_output": str(
            archive_read_model._archive_root(settings)
            / archive_name
        ),
        "nas_staging": str(nas_root),
        "archive_url": f"/#/archive/{quote(archive_name)}/experiments",
        "reused_archive": True,
        "source_count": 6,
        "execution_owner": "visioncortex_web_service",
        "client_process_independent": True,
        "submission_protocol_version": runtime_host._BENCHMARK_SUBMISSION_PROTOCOL_VERSION,
        "submission_receipt": str(submission_receipt),
        "queue_persistence": queue_persistence,
    }


def create_collection_run(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_started_perf = ports.time.perf_counter()
    request_started_epoch = ports.time.time()
    request_received_at = ports.datetime.now().astimezone().isoformat()
    settings = runtime_host._settings()
    try:
        collection = ports.get_collection(settings, experiment_id)
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
    with runtime_host._lock:
        duplicate = next(
            (
                run_id
                for run_id, run in runtime_host._runs.items()
                if run.get("source_collection_id") == experiment_id
                and run.get("state")
                not in {"completed", "partial", "failed", "interrupted"}
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
    archive_name, fixed_root, staging_root, history_root = (
        runtime_host._reserve_collection_archive(settings, requested_name, run_id)
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
    runtime_host._update(
        run_id,
        state="queued",
        progress=0.0,
        experiment_id=archive_name,
        source_collection_id=experiment_id,
        nas_output=str(fixed_root),
        nas_staging=str(staging_root),
    )
    ports.record_collection_state(
        settings,
        experiment_id,
        archive_name=archive_name,
        run_id=run_id,
        state="queued",
        details={"staging": str(staging_root), "source_copy_bytes": 0},
    )
    queue_persistence = runtime_host._schedule_job(
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
        fallback=runtime_host._execute_index_collection,
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


def nas_recordings() -> dict[str, Any]:
    settings = runtime_host._settings()
    if not directory_ingest_enabled(settings):
        return {"mode": "disabled", "recordings": [], "recording_count": 0}
    with runtime_host._nas_monitor_lock:
        monitored = (
            json.loads(json.dumps(runtime_host._nas_monitor_snapshot))
            if runtime_host._nas_monitor_snapshot is not None
            else None
        )
    if monitored is not None:
        return (
            runtime_host._device_day_service.monitor_status(monitored)
            if (settings.get("device_day") or {}).get("enabled")
            else monitored
        )
    if (
        runtime_host._nas_monitor_thread is not None
        and runtime_host._nas_monitor_thread.is_alive()
    ):
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
                "observed_at": ports.datetime.now().astimezone().isoformat(),
                "poll_seconds": float(
                    (settings.get("collection_ingest") or {}).get("poll_seconds", 30.0)
                ),
                "consecutive_failures": 0,
            },
        }
    try:
        inventory = ports.scan_recordings(settings)
        return inventory | {
            "monitor": {
                "status": "starting",
                "observed_at": ports.datetime.now().astimezone().isoformat(),
                "poll_seconds": float(
                    (settings.get("collection_ingest") or {}).get("poll_seconds", 30.0)
                ),
                "consecutive_failures": 0,
            }
        }
    except (OSError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


def select_nas_recordings(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        receipt = create_selection(runtime_host._settings(), payload)
        return {"collection_id": receipt["collection_id"], "source_copy_bytes": 0}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, "无法读取 NAS 素材或保存实验清单") from exc


def create_nas_batch_run(
    batch_id: str,
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn one recorder-native batch into a queued full-chain run."""

    if not re.fullmatch(r"nas-batch-[a-f0-9]{24}", batch_id):
        raise HTTPException(400, "无效的 NAS 采集批次编号")
    with runtime_host._nas_batch_submission_lock:
        settings = runtime_host._settings()
        try:
            inventory = ports.scan_recordings(settings)
        except (OSError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc
        batch = next(
            (
                item
                for item in inventory.get("batches") or []
                if item["batch_id"] == batch_id
            ),
            None,
        )
        if batch is None:
            raise HTTPException(404, "NAS 采集批次不存在或内容已经变化，请刷新后重试")
        device_day_enabled = bool((settings.get("device_day") or {}).get("enabled"))
        if not (
            batch.get("device_day_processable")
            if device_day_enabled
            else batch.get("available")
        ):
            raise HTTPException(
                409,
                {
                    "message": "该采集批次尚不具备分析条件，请查看各项原因",
                    "issues": batch.get("issues") or [],
                },
            )
        if device_day_enabled:
            selected_ids = {r["recording_id"] for r in batch["recordings"]}
            selected = [
                r for r in inventory["recordings"] if r["recording_id"] in selected_ids
            ]
            if len(selected) != len(selected_ids):
                raise HTTPException(409, "采集分片清单已变化，请刷新后重试")
            return runtime_host._device_day_service.submit(settings, selected) | {
                "batch_id": batch_id
            }
        request = payload or {}
        batch_timestamp = ports.datetime.fromtimestamp(
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


router.get("/api/collections")(list_collections)

router.get("/api/collections/{experiment_id}")(collection_detail)

router.post("/api/benchmarks/six-view-three-hour/runs", status_code=202)(
    create_fixed_benchmark_run
)

router.post("/api/collections/{experiment_id}/runs", status_code=202)(
    create_collection_run
)

router.get("/api/nas-recordings")(nas_recordings)

router.post("/api/nas-selections", status_code=201)(select_nas_recordings)

router.post("/api/nas-batches/{batch_id}/runs", status_code=202)(create_nas_batch_run)
