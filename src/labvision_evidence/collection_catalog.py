from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .collection_state import read_collection_states
from .device_registry import load_device_registry, resolve_view_role


COLLECTION_CATALOG_SCHEMA_VERSION = "visioncortex-collection-catalog/1"
COLLECTION_RECEIPT_SCHEMA_VERSION = "visioncortex-collection-readiness/1"

_CACHE_LOCK = threading.Lock()
_INDEX_CACHE: dict[str, dict[str, Any]] = {}


def _parts(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _timestamp(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_index_rows(index_csv: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stat = index_csv.stat()
    key = str(index_csv.resolve())
    signature = (stat.st_size, stat.st_mtime_ns)
    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached and cached["signature"] == signature:
            return cached["rows"], {
                "cache_hit": True,
                "file_size_bytes": stat.st_size,
                "file_mtime_ns": stat.st_mtime_ns,
                "row_count": len(cached["rows"]),
                "parse_duration_seconds": cached["parse_duration_seconds"],
            }
    started = time.perf_counter()
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parse_duration = round(time.perf_counter() - started, 6)
    with _CACHE_LOCK:
        _INDEX_CACHE[key] = {
            "signature": signature,
            "rows": rows,
            "parse_duration_seconds": parse_duration,
        }
    return rows, {
        "cache_hit": False,
        "file_size_bytes": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns,
        "row_count": len(rows),
        "parse_duration_seconds": parse_duration,
    }


def clear_collection_catalog_cache() -> None:
    with _CACHE_LOCK:
        _INDEX_CACHE.clear()


def _fingerprint(rows: Iterable[dict[str, str]]) -> str:
    normalized = [
        {
            key: row.get(key)
            for key in (
                "experiment_id",
                "camera_key",
                "camera_view",
                "recording_start_us",
                "recording_end_us",
                "segment_count",
                "rgb_file",
                "frames_file",
                "updated_at",
                "sync_error",
            )
        }
        for row in rows
    ]
    content = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _collection_summary(
    experiment_id: str,
    rows: list[dict[str, str]],
    registry: dict[str, Any],
    index_meta: dict[str, Any],
    settle_seconds: float,
    now: datetime,
    processing_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    cameras: list[dict[str, Any]] = []
    declared_counts: Counter[str] = Counter()
    resolved_counts: Counter[str] = Counter()
    segment_total = 0
    clock_total = 0

    for row in rows:
        camera_key = str(row.get("camera_key") or "").strip()
        videos = _parts(row.get("rgb_file"))
        clocks = _parts(row.get("frames_file"))
        declared = _integer(row.get("segment_count"), len(videos))
        segment_total += len(videos)
        clock_total += len(clocks)
        receipt = resolve_view_role(
            registry, experiment_id, camera_key, row.get("camera_view")
        )
        receipts.append(receipt)
        if receipt.get("index_role"):
            declared_counts[str(receipt["index_role"])] += 1
        if receipt.get("resolved_role"):
            resolved_counts[str(receipt["resolved_role"])] += 1

        camera_issues: list[str] = []
        if not camera_key:
            camera_issues.append("camera_key_missing")
        if not videos:
            camera_issues.append("video_segments_missing")
        if declared != len(videos):
            camera_issues.append("declared_video_segment_count_mismatch")
        if clocks and len(clocks) != len(videos):
            camera_issues.append("video_clock_segment_count_mismatch")
        if not clocks:
            warnings.append(
                {
                    "code": "clock_csv_absent",
                    "camera_key": camera_key,
                    "message": "No frame clock CSV is indexed; visual alignment is required.",
                }
            )
        sync_error = str(row.get("sync_error") or "").strip()
        if sync_error:
            camera_issues.append("index_sync_error")
        for code in camera_issues:
            blocking.append(
                {
                    "code": code,
                    "camera_key": camera_key,
                    "message": sync_error if code == "index_sync_error" else code,
                }
            )
        for reason in receipt.get("blocking_reasons") or []:
            blocking.append(
                {
                    "code": reason,
                    "camera_key": camera_key,
                    "message": "View role requires registry or index correction.",
                }
            )
        if receipt.get("status") == "approved_override":
            warnings.append(
                {
                    "code": "approved_view_role_override",
                    "camera_key": camera_key,
                    "message": (receipt.get("override") or {}).get("reason"),
                }
            )
        cameras.append(
            {
                "camera_key": camera_key,
                "camera_id": row.get("camera_id"),
                "display_name": receipt.get("display_name") or camera_key,
                "index_camera_view": row.get("camera_view"),
                "resolved_role": receipt.get("resolved_role"),
                "role_status": receipt.get("status"),
                "segment_count": len(videos),
                "clock_segment_count": len(clocks),
                "recording_start_time": row.get("recording_start_time"),
                "recording_end_time": row.get("recording_end_time"),
                "sync_error": sync_error or None,
            }
        )

    if not resolved_counts.get("first_person") or not resolved_counts.get("third_person"):
        blocking.append(
            {
                "code": "dual_view_roles_incomplete",
                "camera_key": None,
                "message": "At least one resolved first-person and third-person source is required.",
            }
        )

    starts = [
        value
        for value in (_timestamp(row.get("recording_start_time")) for row in rows)
        if value is not None
    ]
    ends = [
        value
        for value in (_timestamp(row.get("recording_end_time")) for row in rows)
        if value is not None
    ]
    updated = [
        value
        for value in (_timestamp(row.get("updated_at")) for row in rows)
        if value is not None
    ]
    has_open_recording = len(ends) != len(rows)
    latest_update = max(updated) if updated else None
    if latest_update is not None:
        if latest_update.tzinfo is None:
            latest_update = latest_update.replace(tzinfo=now.tzinfo)
        settled_age = max(0.0, (now - latest_update.astimezone(now.tzinfo)).total_seconds())
    else:
        settled_age = float("inf")
    settled = settled_age >= settle_seconds

    if has_open_recording or not settled:
        status = "recording"
    elif blocking:
        status = "attention"
    else:
        status = "ready"

    start = min(starts) if starts else None
    end = max(ends) if ends else None
    duration_seconds = (end - start).total_seconds() if start and end else None
    first = rows[0]
    display_name = (
        first.get("experiment_prefix")
        or first.get("experiment_base")
        or first.get("experiment_id")
        or experiment_id
    )
    return {
        "schema_version": COLLECTION_RECEIPT_SCHEMA_VERSION,
        "collection_id": experiment_id,
        "experiment_id": experiment_id,
        "display_name": display_name,
        "status": status,
        "sealed": status in {"ready", "attention"},
        "ready_to_analyze": status == "ready",
        "recording_start_time": start.isoformat() if start else None,
        "recording_end_time": end.isoformat() if end else None,
        "duration_seconds": round(duration_seconds, 3)
        if duration_seconds is not None
        else None,
        "camera_count": len(cameras),
        "video_segment_count": segment_total,
        "clock_segment_count": clock_total,
        "declared_view_counts": dict(declared_counts),
        "resolved_view_counts": dict(resolved_counts),
        "blocking_issue_count": len(blocking),
        "warning_count": len(warnings),
        "approved_override_count": sum(
            receipt.get("status") == "approved_override" for receipt in receipts
        ),
        "blocking_issues": blocking,
        "warnings": warnings,
        "cameras": cameras,
        "view_role_resolution": receipts,
        "settlement": {
            "required_seconds": settle_seconds,
            "observed_age_seconds": None
            if settled_age == float("inf")
            else round(settled_age, 3),
            "settled": settled,
            "latest_index_update": latest_update.isoformat() if latest_update else None,
        },
        "fingerprint_sha256": _fingerprint(rows),
        "source_policy": {
            "mode": "index_metadata_only",
            "video_files_opened": 0,
            "clock_csv_files_opened": 0,
            "source_path_stats": 0,
            "source_validation_deferred_until_run": True,
        },
        "processing": processing_state
        or {
            "state": "not_processed",
            "archive_name": None,
            "run_id": None,
            "updated_at": None,
        },
        "index_snapshot": {
            "path": index_meta.get("path"),
            "file_size_bytes": index_meta.get("file_size_bytes"),
            "file_mtime_ns": index_meta.get("file_mtime_ns"),
        },
    }


def _snapshot_path(config: dict[str, Any]) -> Path:
    configured = (config.get("collection_ingest") or {}).get("snapshot_path")
    if configured:
        return Path(str(configured))
    return (
        Path(config["storage"]["local_cache_root"])
        / "Collection-Catalog"
        / "experiment_record_index.snapshot.json"
    )


def _write_snapshot(config: dict[str, Any], payload: dict[str, Any]) -> None:
    if not (config.get("collection_ingest") or {}).get("persist_snapshot", True):
        return
    path = _snapshot_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def discover_collections(
    config: dict[str, Any],
    *,
    status: str | None = None,
    query: str | None = None,
    limit: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    settings = config.get("collection_ingest") or {}
    index_csv = Path(config["storage"]["index_csv"])
    rows, index_meta = _load_index_rows(index_csv)
    index_meta["path"] = str(index_csv)
    registry = load_device_registry(config["storage"].get("device_registry_path"))
    processing_registry = (
        read_collection_states(config)
        if config.get("storage", {}).get("archive_root")
        else {"available": False, "collections": {}}
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        experiment_id = str(row.get("experiment_id") or "").strip()
        if experiment_id:
            grouped[experiment_id].append(row)
    current = now or datetime.now().astimezone()
    settle_seconds = float(settings.get("settle_seconds", 120.0))
    collections = [
        _collection_summary(
            experiment_id,
            experiment_rows,
            registry,
            index_meta,
            settle_seconds,
            current,
            (processing_registry.get("collections") or {}).get(experiment_id),
        )
        for experiment_id, experiment_rows in grouped.items()
    ]
    collections.sort(
        key=lambda item: str(item.get("recording_start_time") or ""), reverse=True
    )
    if status:
        collections = [item for item in collections if item["status"] == status]
    normalized_query = str(query or "").strip().lower()
    if normalized_query:
        collections = [
            item
            for item in collections
            if normalized_query
            in " ".join(
                [
                    str(item.get("collection_id") or ""),
                    str(item.get("display_name") or ""),
                    *(str(camera.get("camera_key") or "") for camera in item["cameras"]),
                ]
            ).lower()
        ]
    result_limit = int(limit or settings.get("max_results", 200))
    collections = collections[: max(1, min(result_limit, 1000))]
    payload = {
        "schema_version": COLLECTION_CATALOG_SCHEMA_VERSION,
        "generated_at": current.isoformat(),
        "index": index_meta,
        "registry": registry.get("provenance"),
        "processing_registry": {
            "available": processing_registry.get("available"),
            "path": processing_registry.get("path"),
            "updated_at": processing_registry.get("updated_at"),
        },
        "monitoring_policy": {
            "mode": "index_signature_poll",
            "recursive_nas_scan": False,
            "video_decode": False,
            "recommended_poll_seconds": float(settings.get("poll_seconds", 30.0)),
        },
        "collection_count": len(collections),
        "collections": collections,
    }
    if not index_meta["cache_hit"]:
        _write_snapshot(config, payload)
    return payload


def get_collection(config: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    payload = discover_collections(config, limit=1000)
    selected = next(
        (
            item
            for item in payload["collections"]
            if item["experiment_id"] == experiment_id
        ),
        None,
    )
    if selected is None:
        raise KeyError(f"experiment_id not found: {experiment_id}")
    return selected
