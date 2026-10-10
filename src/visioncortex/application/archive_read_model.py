"""Archive query projections independent of HTTP routing."""

from __future__ import annotations
from .dependencies import ports
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote
from .contracts import ApplicationError
from ..archive_catalog import (
    list_catalog_archives,
    archive_catalog_path,
    catalog_archive_release,
    ensure_archive_catalog,
    search_catalog_events,
)
from ..indexing import (
    INDEX_DB_NAME,
    INDEX_MANIFEST_NAME,
    get_indexed_event,
    get_indexed_evidence,
    search_archive_index,
    search_physical_changes,
)
from ..pagination import decode_cursor, encode_cursor
from ..pathing import archive_contains, archive_relative_posix
from ..partial_delivery import (
    prioritize_retained_candidates,
)
from ..storage import (
    archive_promotion_in_progress,
)

from . import run_read_model, runtime_host


def _archive_root(settings: dict[str, Any] | None = None) -> Path:
    return Path((settings or runtime_host._settings())["storage"]["archive_root"])


def _archive_catalog_database(
    settings: dict[str, Any] | None = None,
    archive_root: Path | None = None,
) -> Path:
    effective_settings = settings or runtime_host._settings()
    root = (archive_root or _archive_root(effective_settings)).resolve()
    storage = effective_settings.get("storage") or {}
    local_runtime = Path(
        storage.get("local_runtime_root") or (root / ".VisionCortex-Web-Runtime")
    )
    return archive_catalog_path(local_runtime, root)


def _ensure_archive_read_catalog(root: Path | None = None) -> Path:
    archive_root = (root or _archive_root()).resolve()
    database = _archive_catalog_database(archive_root=archive_root)
    ensure_archive_catalog(archive_root, database)
    return database


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
        raise ApplicationError(404, "实验档案不存在")
    if archive_promotion_in_progress(candidate):
        raise ApplicationError(409, "实验档案正在原子发布，请稍后重试")
    json_root = candidate / "JSON-Config-Files"
    if not (
        (json_root / "evidence_package.json").is_file()
        or (json_root / INDEX_DB_NAME).is_file()
    ):
        raise ApplicationError(409, "实验档案尚未通过正式发布门禁")
    return candidate


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
            if item.is_dir() and (item / "JSON-Config-Files" / INDEX_DB_NAME).is_file()
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
        raise ApplicationError(400, "无效的关键事件分页 cursor") from exc


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


def _attach_index_urls(archive_name: str, event: dict[str, Any]) -> dict[str, Any]:
    result = dict(event)
    result["archive_name"] = archive_name
    release_id = str(event.get("release_id") or "") or None
    result["event_url"] = (
        f"/api/key-events/{quote(str(event['event_uid']))}?archive={quote(archive_name)}"
    )
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
        raise ApplicationError(400, "无效的关键事件分页 cursor") from exc


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


def _attach_staging_index_urls(run_id: str, event: dict[str, Any]) -> dict[str, Any]:
    result = dict(event)
    result["staging_run_id"] = run_id
    result["event_url"] = (
        f"/api/staging-runs/{quote(run_id)}/key-events/{quote(str(event['event_uid']))}"
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


def search_key_events(
    archive: str | None = None,
    q: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    cross_view: bool | None = None,
    liquid_state_status: str | None = None,
    liquid_present: bool | None = None,
    visible_flow: bool | None = None,
    material_ready: bool | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = 50,
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
        "liquid_state_status": liquid_state_status,
        "liquid_present": liquid_present,
        "visible_flow": visible_flow,
        "material_ready": material_ready,
        "start_us": start_us,
        "end_us": end_us,
    }
    decoded_cursor = _decode_catalog_event_cursor(cursor, cursor_filters)
    try:
        database = _ensure_archive_read_catalog()
        if (
            decoded_cursor
            and catalog_archive_release(database, decoded_cursor["archive_name"])
            != decoded_cursor["release_id"]
        ):
            raise ApplicationError(
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
            liquid_state_status=liquid_state_status,
            liquid_present=liquid_present,
            visible_flow=visible_flow,
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
    except ApplicationError:
        raise
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        raise ApplicationError(503, f"无法读取关键事件目录索引: {exc}") from exc
    has_more = len(items) > limit
    page = [
        _attach_index_urls(str(item["archive_name"]), item) for item in items[:limit]
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


def search_staging_key_events(
    run_id: str,
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
    limit: int = 50,
) -> dict[str, Any]:
    """Search one isolated NAS staging run without promoting it."""

    root = run_read_model._resolve_staging_run(run_id)
    cursor_filters = {
        "staging_run_id": run_id,
        "q": q,
        "action_type": action_type,
        "parent_event_id": parent_event_id,
        "cross_view": cross_view,
        "liquid_state_status": liquid_state_status,
        "liquid_present": liquid_present,
        "visible_flow": visible_flow,
        "start_us": start_us,
        "end_us": end_us,
    }
    decoded_cursor = _decode_event_cursor(cursor, cursor_filters)
    manifest = (
        run_read_model._read_json(root / "JSON-Config-Files" / INDEX_MANIFEST_NAME, {})
        or {}
    )
    archive_id = str(manifest.get("archive_id") or run_id)
    after_peak_us = None
    after_event_uid = None
    if decoded_cursor:
        cursor_archive, after_peak_us, after_event_uid = decoded_cursor
        if cursor_archive != archive_id:
            raise ApplicationError(400, "分页游标与 staging run 不匹配")
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
    has_more = len(items) > limit
    page = [_attach_staging_index_urls(run_id, item) for item in items[:limit]]
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


def staging_key_event(run_id: str, event_uid: str) -> dict[str, Any]:
    root = run_read_model._resolve_staging_run(run_id)
    event = get_indexed_event(root, event_uid)
    if event is None:
        raise ApplicationError(404, "staging 关键事件索引记录不存在")
    return _attach_staging_index_urls(run_id, event)


def indexed_physical_changes(
    archive: str | None = None,
    object_id: str | None = None,
    object_role: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = 50,
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
            raise ApplicationError(400, "无效的物理状态变化分页 cursor") from exc
    collected: list[dict[str, Any]] = []
    for archive_name, root in _search_archive_roots(archive):
        manifest = (
            run_read_model._read_json(
                root / "JSON-Config-Files" / INDEX_MANIFEST_NAME, {}
            )
            or {}
        )
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


def indexed_key_event(event_uid: str, archive: str | None = None) -> dict[str, Any]:
    for archive_name, root in _search_archive_roots(archive):
        event = get_indexed_event(root, event_uid)
        if event is not None:
            return _attach_index_urls(archive_name, event)
    raise ApplicationError(404, "关键事件索引记录不存在")


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
    raise ApplicationError(404, "证据索引记录不存在")


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
    explicit_evidence_count = sum(
        check.get("passed") is True for check in evidence_checks
    )
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
            "missing_action_types": sorted(
                set(runtime_host._PHYSICAL_ACTION_TYPES) - set(action_types)
            ),
            "source": "event_cross_view_associations + evidence_package_eval.json",
        },
    }


def _attach_archive_performance_display(
    metrics: dict[str, Any], acceptance: dict[str, Any]
) -> None:
    benchmark = acceptance.get("preprocessing_full_run") or {}
    clean_run = acceptance.get("clean_package_run") or {}
    current_preprocessing = (metrics.get("preprocessing_sla") or {}).get(
        "actual_seconds"
    )
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
            "reused_validated_cv_ledgers": "reused validated cv ledgers"
            in reuse_note.lower(),
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
            uncertainty_reasons.update(
                str(item) for item in assessment.get("reasons") or []
            )
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
            if staging_run_id
            else _file_url(archive_name, relative, release_id)
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
        "resource_telemetry_journal": existing(
            "JSON-Config-Files/resource_telemetry.jsonl"
        ),
        "operation_review": existing("JSON-Config-Files/operation_review.json"),
        "delivery_metrics": existing("JSON-Config-Files/delivery_metrics.json"),
        "acceptance": existing("JSON-Config-Files/acceptance_report.json"),
        "quality_acceptance": existing("JSON-Config-Files/quality_acceptance.json"),
        "evidence_package_eval": existing(
            "JSON-Config-Files/evidence_package_eval.json"
        ),
        "key_material_recall_eval": existing(
            "JSON-Config-Files/key_material_recall_eval.json"
        ),
        "daily_report_json": existing(daily_manifest.get("json")),
        "daily_report_markdown": existing(daily_manifest.get("markdown")),
        "daily_report_html": existing(daily_manifest.get("html")),
        "daily_report_pdf": existing(daily_manifest.get("pdf")),
        "daily_report_eval": existing(daily_manifest.get("evaluation")),
        "evidence_index_manifest": existing(f"JSON-Config-Files/{INDEX_MANIFEST_NAME}"),
        "run_provenance": existing("JSON-Config-Files/run_provenance.json"),
        "project_annotation_comparison": existing(
            "Project-Review/Project-Annotation-Comparison.html"
        ),
        "final_key_material_annotation": existing(
            "JSON-Config-Files/final_key_material_annotation.json"
        ),
    }


def _movement_screening_payload(root: Path, report_url: str) -> dict[str, Any] | None:
    movement_report = (
        run_read_model._read_json(
            root / "JSON-Config-Files/movement_visual_verification.json", {}
        )
        or {}
    )
    movement_screening = None
    if movement_report.get("enabled"):
        movement_screening = {
            "counts": movement_report.get("counts", {}),
            "duration_seconds": movement_report.get("duration_seconds"),
            "total_candidates": len(movement_report.get("candidates", [])),
            "candidates": [
                {
                    key: item.get(key)
                    for key in (
                        "candidate_id",
                        "view_id",
                        "start_ms",
                        "end_ms",
                        "objects",
                        "status",
                    )
                }
                for item in movement_report.get("candidates", [])[:200]
            ],
            "report_url": report_url,
            "physical_action_confirmed": False,
        }
    return movement_screening


def _archive_summary_payload(archive_name: str, root: Path) -> dict[str, Any]:
    pointer = ports.read_current_release_pointer(root) or {}
    release_id = str(pointer.get("release_id") or "") or None
    json_root = root / "JSON-Config-Files"
    index_manifest = (
        run_read_model._read_json(json_root / INDEX_MANIFEST_NAME, {}) or {}
    )
    daily_manifest = (
        run_read_model._read_json(json_root / "daily_report_manifest.json", {}) or {}
    )
    quality = run_read_model._read_json(json_root / "quality_acceptance.json", {}) or {}
    metrics = run_read_model._merged_run_metrics(root)
    _attach_archive_performance_display(
        metrics,
        run_read_model._read_json(json_root / "acceptance_report.json", {}) or {},
    )
    counts = index_manifest.get("counts") or {}
    manifest = run_read_model._read_json(json_root / "run_manifest.json", {}) or {}
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
            ports.lightweight_release_integrity(root, pointer)
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
            "status": run_read_model._pipeline_status_from_root(root),
            "stage_receipts": run_read_model._stage_receipts_from_root(root),
        },
        "evidence_index": {
            **index_manifest,
            "search_url": f"/api/key-events?archive={quote(archive_name)}",
        }
        if index_manifest
        else None,
        "movement_screening": _movement_screening_payload(
            root,
            _file_url(
                archive_name,
                "JSON-Config-Files/movement_visual_verification.json",
                release_id,
            ),
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
        raise ApplicationError(
            400, "section 必须是 summary、experiments、reports 或 metrics"
        )
    # Incomplete legacy runs have no final index yet. Retain their already-built
    # media and failure state instead of presenting an empty successful archive.
    status = run_read_model._pipeline_status_from_root(root)
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
                raise ApplicationError(400, "无效的实验片段分页 cursor") from exc
        analysis = (
            run_read_model._read_json(
                json_root / "Experiment-Groups-Step-Level-Analysis.json", {}
            )
            or {}
        )
        groups = list(analysis.get("experiment_groups") or [])
        if not groups:
            legacy_package = (
                run_read_model._read_json(json_root / "evidence_package.json", {}) or {}
            )
            groups = list(legacy_package.get("experiment_groups") or [])
        from ..activity_review import assessment

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
            aligned = next(
                (path for path in aligned_candidates if path.is_file()), None
            )
            experiments.append(
                {
                    "folder": folder_name,
                    "name": group.get("experiment_name") or folder_name,
                    "activity_assessment": assessment(group),
                    "continuity_type": group.get("continuity_type"),
                    "workflow_kind": group.get("workflow_kind", "unresolved"),
                    "source_archive_folders": group.get("source_archive_folders") or [],
                    "workflow_units": group.get("workflow_units") or [],
                    "completion_status": group.get("completion_status", "unreviewed"),
                    "completion_reason": group.get("completion_reason") or "",
                    "boundary_extension_requires_step_review": group.get(
                        "boundary_extension_requires_step_review", False
                    ),
                    "view_timeline": group.get("view_timeline") or [],
                    "start_ms": group.get("global_start_ms"),
                    "end_ms": group.get("global_end_ms"),
                    "speech_interpretation": understanding.get("speech_interpretation"),
                    "speech_context": understanding.get("speech_context"),
                    "summary": understanding.get("overall_summary"),
                    "steps": understanding.get("steps") or [],
                    "uncertainties": understanding.get("uncertainties") or [],
                    "first_person_video_url": (
                        _file_url(
                            archive_name,
                            archive_relative_posix(folder / "First-Person.mp4", root),
                            release_id,
                        )
                        if (folder / "First-Person.mp4").is_file()
                        else None
                    ),
                    "third_person_video_url": (
                        _file_url(
                            archive_name,
                            archive_relative_posix(folder / "Third-Person.mp4", root),
                            release_id,
                        )
                        if (folder / "Third-Person.mp4").is_file()
                        else None
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
                        "source_archive_folders": group.get("source_archive_folders")
                        or [],
                        "workflow_units": group.get("workflow_units") or [],
                        "completion_status": group.get(
                            "completion_status", "unreviewed"
                        ),
                        "completion_reason": group.get("completion_reason") or "",
                        "boundary_extension_requires_step_review": group.get(
                            "boundary_extension_requires_step_review", False
                        ),
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
        daily_manifest = (
            run_read_model._read_json(json_root / "daily_report_manifest.json", {})
            or {}
        )
        result.update(
            {
                "daily_report_manifest": daily_manifest,
                "daily_report": (
                    run_read_model._read_json(root / daily_manifest["json"], {})
                    if daily_manifest.get("json")
                    else {}
                )
                or {},
            }
        )
        if section == "reports":
            return result
    metrics = run_read_model._merged_run_metrics(root)
    acceptance = (
        run_read_model._read_json(json_root / "acceptance_report.json", {}) or {}
    )
    _attach_archive_performance_display(metrics, acceptance)
    final_annotation = (
        run_read_model._read_json(json_root / "final_key_material_annotation.json", {})
        or {}
    )
    key_material_verification, _ = _summarize_key_material_verification(
        final_annotation
    )
    result.update(
        {
            "metrics": metrics,
            "key_material_recall_eval": run_read_model._read_json(
                json_root / "key_material_recall_eval.json", {}
            )
            or {},
            "observability": run_read_model._run_snapshot_from_root(root),
            "key_material_verification": key_material_verification,
        }
    )
    return result


def archive_detail(
    archive_name: str,
    section: str = "all",
    cursor: str | None = None,
    limit: int = 24,
) -> dict[str, Any]:
    root = _resolve_archive(archive_name)
    if section != "all":
        return _archive_section_payload(
            archive_name, root, section=section, cursor=cursor, limit=limit
        )
    return _archive_detail_from_root(root, archive_name)


def _stage_clip_group(
    folder: Path, groups: dict[str, dict], status: dict
) -> dict | None:
    started = run_read_model._current_attempt_started_at(status)
    sidecar = folder / "First-Person.json"
    if not sidecar.is_file() or (
        started is not None and sidecar.stat().st_mtime < started
    ):
        return None
    saved = run_read_model._read_json(sidecar, {}) or {}
    source = saved.get("group") or {}
    current = groups.get(str(source.get("group_id")))
    if not current or current.get("archive_folder"):
        return None
    if any(
        source.get(key) != current.get(key)
        for key in ("global_start_ms", "global_end_ms")
    ):
        return None
    if (
        saved.get("artifact_type") != "experiment_view_video"
        or source.get("archive_folder") != folder.name
    ):
        return None
    return {**current, "archive_folder": folder.name}


def _archive_detail_from_root(
    root: Path,
    archive_name: str,
    *,
    staging_run_id: str | None = None,
    library_section: str | None = None,
) -> dict[str, Any]:
    release_pointer = ports.read_current_release_pointer(root) or {}
    release_id = str(release_pointer.get("release_id") or "") or None

    def file_url(name: str, relative: str | Path) -> str:
        return (
            _staging_file_url(staging_run_id, relative)
            if staging_run_id
            else _file_url(name, relative, release_id)
        )

    index_manifest_path = root / "JSON-Config-Files" / INDEX_MANIFEST_NAME
    index_manifest = run_read_model._read_json(index_manifest_path, {}) or {}
    snapshot = (
        {
            "status": run_read_model._pipeline_status_from_root(root),
            "stage_receipts": run_read_model._stage_receipts_from_root(root),
            "partial_delivery": run_read_model._read_json(
                root / "JSON-Config-Files/partial_delivery.json", {}
            )
            or {},
        }
        if library_section
        else run_read_model._run_snapshot_from_root(root)
    )
    status = snapshot.get("status") or {}
    for receipt in snapshot.get("stage_receipts", []):
        receipt["receipt_url"] = file_url(archive_name, receipt["receipt"])
        version = receipt.get("version_manifest")
        if version and (root / version).is_file():
            receipt["version_url"] = file_url(archive_name, version)
        for artifact in receipt.get("artifacts", []):
            if (
                artifact.get("available")
                and artifact.get("relative_path")
                and artifact.get("kind") == "file"
            ):
                artifact["url"] = file_url(archive_name, artifact["relative_path"])
    active_preview = bool(staging_run_id) and status.get("stage") not in {
        "completed",
        "partial",
        "failed",
        "interrupted",
        "cancelled",
    }
    completed_stages = {
        item["stage"]
        for item in snapshot.get("stage_receipts", [])
        if item.get("status") == "completed"
    }
    package = (
        {}
        if library_section
        else run_read_model._read_json(
            root / "JSON-Config-Files" / "evidence_package.json", {}
        )
        or {}
    )
    if active_preview:
        package = {}
    metrics = {} if library_section else run_read_model._merged_run_metrics(root)
    acceptance = (
        run_read_model._read_json(
            root / "JSON-Config-Files" / "acceptance_report.json", {}
        )
        or {}
    )
    quality_path = root / "JSON-Config-Files" / "quality_acceptance.json"
    quality_acceptance = run_read_model._read_json(quality_path, {}) or {}
    evidence_eval_path = root / "JSON-Config-Files" / "evidence_package_eval.json"
    evidence_eval = run_read_model._read_json(evidence_eval_path, {}) or {}
    recall_eval_path = root / "JSON-Config-Files" / "key_material_recall_eval.json"
    key_material_recall_eval = run_read_model._read_json(recall_eval_path, {}) or {}
    final_annotation_path = (
        root / "JSON-Config-Files" / "final_key_material_annotation.json"
    )
    final_annotation = (
        {}
        if library_section
        else run_read_model._read_json(final_annotation_path, {}) or {}
    )
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
    key_events = (
        run_read_model._read_json(
            root / "Key-Materials" / "Key-Materials-Model-Understanding.json", []
        )
        or []
    )
    if active_preview and "mllm" not in completed_stages:
        key_events = []
    daily_manifest = (
        run_read_model._read_json(
            root / "JSON-Config-Files" / "daily_report_manifest.json", {}
        )
        or {}
    )
    if active_preview:
        daily_manifest = {}
    daily_report = (
        run_read_model._read_json(root / daily_manifest["json"], {})
        if daily_manifest.get("json")
        else {}
    ) or {}
    package_groups = package.get("experiment_groups", [])
    if not package_groups:
        stage_groups = (
            run_read_model._read_json(
                root / "JSON-Config-Files" / "experiment_group_understanding.json", {}
            )
            or {}
        )
        package_groups = stage_groups.get("groups", [])
        if (
            active_preview
            and run_read_model._current_attempt_started_at(status) is not None
            and "experiment_understanding" not in completed_stages
        ):
            package_groups = []
    from ..speech_refresh import apply as apply_speech_revision

    package_groups = apply_speech_revision(root, package_groups)
    from ..operation_review import apply as apply_operation_revision, coverage

    package_groups = apply_operation_revision(root, package_groups)
    from ..activity_review import (
        apply as apply_activity,
        assessment,
        counts as activity_counts,
    )

    package_groups = apply_activity(root, package_groups)
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
        for folder in sorted(
            item for item in experiment_root.iterdir() if item.is_dir()
        ):
            if staging_run_id and folder.name not in group_by_folder:
                # Older runs published their group index before clip paths
                # were assigned. Recover only matching, finished clip metadata.
                recovered = _stage_clip_group(folder, group_by_id, status)
                if recovered:
                    group_by_folder[folder.name] = recovered
            if (
                package_groups or active_preview
            ) and folder.name not in group_by_folder:
                continue
            if active_preview and "experiment_clips" not in completed_stages:
                continue  # A directory/MP4 may still be under construction.
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
                    "activity_assessment": assessment(group),
                    "continuity_type": group.get("continuity_type"),
                    "workflow_kind": group.get("workflow_kind", "unresolved"),
                    "source_archive_folders": group.get("source_archive_folders") or [],
                    "workflow_units": group.get("workflow_units") or [],
                    "completion_status": group.get("completion_status", "unreviewed"),
                    "completion_reason": group.get("completion_reason") or "",
                    "boundary_extension_requires_step_review": group.get(
                        "boundary_extension_requires_step_review", False
                    ),
                    "view_timeline": group.get("view_timeline") or [],
                    "start_ms": group.get("global_start_ms"),
                    "end_ms": group.get("global_end_ms"),
                    "speech_interpretation": understanding.get("speech_interpretation"),
                    "speech_context": understanding.get("speech_context"),
                    "summary": understanding.get("overall_summary"),
                    "steps": understanding.get("steps") or [],
                    "operation_coverage": coverage(
                        group, understanding.get("steps") or []
                    ),
                    "operation_review_accepted": (
                        understanding.get("operation_review") or {}
                    ).get("accepted"),
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
                item
                for item in event.get("key_frames", [])
                if item.get("view_role") == "aligned_first_third"
            ),
            None,
        )
        clip = next(
            (
                item
                for item in event.get("key_clips", [])
                if item.get("view_role") == "aligned_first_third"
            ),
            None,
        )
        normalized_events.append(
            {
                **event,
                "aligned_frame_url": file_url(archive_name, frame["path"])
                if frame
                else None,
                "aligned_clip_url": file_url(archive_name, clip["path"])
                if clip
                else None,
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
                    "name": group.get("experiment_name")
                    or event.get("parent_event_id"),
                    "folder": group.get("archive_folder"),
                    "continuity_type": group.get("continuity_type"),
                    "workflow_kind": group.get("workflow_kind", "unresolved"),
                    "source_archive_folders": group.get("source_archive_folders") or [],
                    "workflow_units": group.get("workflow_units") or [],
                    "completion_status": group.get("completion_status", "unreviewed"),
                    "completion_reason": group.get("completion_reason") or "",
                    "boundary_extension_requires_step_review": group.get(
                        "boundary_extension_requires_step_review", False
                    ),
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
        dual_view_material_count / len(normalized_events)
        if normalized_events
        else None,
    )
    key_quality.setdefault(
        "missing_dual_view_material_count",
        len(normalized_events) - dual_view_material_count,
    )
    links = _archive_links(
        archive_name,
        root,
        release_id,
        daily_manifest,
        staging_run_id=staging_run_id,
    )
    preliminary_materials = []
    if not normalized_events and snapshot.get("status", {}).get("stage") != "completed":
        completed = {
            item.get("stage")
            for item in snapshot.get("stage_receipts", [])
            if item.get("status") == "completed"
        }
        category_index = (
            run_read_model._read_json(
                root / "Key-Materials/Key-Material-Category-Index.json", {}
            )
            or {}
        )
        if "key_materials" in completed:
            for group in category_index.get("experiments", []):
                for category in group.get("action_categories", []):
                    for event in category.get("events", []):
                        media = {}
                        for field, source in (
                            ("frame_url", "key_frames"),
                            ("clip_url", "key_clips"),
                        ):
                            relative = (event.get(source) or {}).get(
                                "aligned_first_third"
                            )
                            if relative:
                                candidate = (root / relative).resolve()
                                if (
                                    archive_contains(candidate, root.resolve())
                                    and candidate.is_file()
                                ):
                                    media[field] = file_url(archive_name, relative)
                        if media:
                            preliminary_materials.append(
                                {
                                    "event_id": event.get("event_id"),
                                    "group_name": group.get("experiment_name"),
                                    "timestamp_ms": event.get("peak_timestamp_us", 0)
                                    / 1000,
                                    "review_status": "pending_semantic_review",
                                    **media,
                                }
                            )
    quarantine_materials = []
    manifest = (
        run_read_model._read_json(root / "JSON-Config-Files/run_manifest.json", {})
        or {}
    )
    role_by_view = {
        view.get("view_id"): view.get("role") for view in manifest.get("views", [])
    }
    retained_candidates = {}
    for relative_index in (
        "Key-Materials/Machine-Quarantine/Machine-Quarantine-Index.json",
        "Key-Materials/Review-Candidates/Candidate-Index.json",
    ):
        index = run_read_model._read_json(root / relative_index, {}) or {}
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
            if (
                not archive_contains(candidate, root.resolve())
                or not candidate.is_file()
            ):
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
                role_media.setdefault(candidate.stem, {})[field] = file_url(
                    archive_name, relative
                )
        context_media = {}
        if (
            pairing.get("pair_evidence_status")
            == "context_only_missing_key_time_support"
        ):
            # A same-time context camera is not a corresponding action view.
            # Keep it accessible separately instead of presenting a false pair.
            supported = {
                item["view_id"]
                for candidates in pairing.get("candidates", {}).values()
                for item in candidates
                if item.get("candidate_supported_at_key")
            }
            primary_role = (
                "Third-Person"
                if pairing.get("third_person_view") in supported
                and pairing.get("first_person_view") not in supported
                else "First-Person"
            )
            other_role = (
                "Third-Person" if primary_role == "First-Person" else "First-Person"
            )
            if role_media.get(primary_role):
                media = dict(role_media[primary_role])
                context_media = role_media.get(other_role, {})
        if media:
            group = next(
                (
                    item
                    for item in package_groups
                    if item.get("global_start_ms") is not None
                    and item.get("global_end_ms") is not None
                    and item["global_start_ms"]
                    <= event.get("key_global_ms", 0)
                    <= item["global_end_ms"]
                ),
                {},
            )
            quarantine_materials.append(
                {
                    "event_id": event.get("event_id"),
                    "timestamp_ms": event.get("key_global_ms", 0),
                    "start_ms": event.get("global_start_ms"),
                    "end_ms": event.get("global_end_ms"),
                    "cv_action_type": event.get("cv_action_type"),
                    "cv_objects": event.get("cv_objects", []),
                    "source_views": event.get("source_views", []),
                    "view_pairing": event.get("view_pairing", {}),
                    "context_media": context_media,
                    "source_roles": sorted(
                        {
                            role_by_view[view]
                            for view in event.get("source_views", [])
                            if role_by_view.get(view)
                        }
                    ),
                    "group_id": group.get("group_id"),
                    "group_folder": group.get("archive_folder"),
                    "group_name": group.get("experiment_name"),
                    "review_status": "machine_quarantined",
                    "evidence_classification": "PARTIAL_EVIDENCE",
                    "disposition": event.get("disposition"),
                    **media,
                }
            )
    quarantine_materials, retained_review = prioritize_retained_candidates(
        quarantine_materials
    )
    movement_screening = (
        None
        if library_section
        else _movement_screening_payload(
            root,
            file_url(
                archive_name, "JSON-Config-Files/movement_visual_verification.json"
            ),
        )
    )
    partial_delivery = snapshot.get("partial_delivery") or {}
    if (
        partial_delivery
        and (root / "Partial-Results/Partial-Evidence-Report.html").is_file()
    ):
        links["partial_report"] = file_url(
            archive_name, "Partial-Results/Partial-Evidence-Report.html"
        )
        if (root / "Partial-Results/Analysis-Result.json").is_file():
            links["partial_json"] = file_url(
                archive_name, "Partial-Results/Analysis-Result.json"
            )
        for key, kind, relative in (
            ("partial_pdf", "pdf", "Partial-Results/Stage-Evidence-Report.pdf"),
            (
                "partial_daily_report",
                "daily_html",
                "Partial-Results/Stage-Lab-Daily-Report.html",
            ),
        ):
            presentation = (partial_delivery.get("readable_reports") or {}).get(
                kind
            ) or {}
            path = root / relative
            if (
                presentation.get("path") == relative
                and path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest()
                == presentation.get("sha256")
            ):
                links[key] = file_url(archive_name, relative)
    result = {
        "name": archive_name,
        "path": str(root),
        "staging_run_id": staging_run_id,
        "network_path": str(root),
        "release_id": release_id,
        "integrity_status": (
            ports.lightweight_release_integrity(root, release_pointer)
            if release_pointer
            else "legacy_archive_not_release_verified"
        ),
        "counts": {
            **activity_counts(experiments),
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
                "boundary_extension_requires_step_review": group.get(
                    "boundary_extension_requires_step_review", False
                ),
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
        "result_review": run_read_model._latest_result_review(root)
        if not active_preview and not library_section
        else {"available": False},
        "key_material_recall_eval": key_material_recall_eval,
        "key_material_verification": key_material_verification,
        "observability": snapshot,
        "daily_report": daily_report,
        "daily_report_manifest": daily_manifest,
        "current_release": release_pointer or None,
        "evidence_index": {
            **index_manifest,
            "search_url": (
                f"/api/staging-runs/{quote(staging_run_id)}/key-events"
                if staging_run_id
                else f"/api/key-events?archive={quote(archive_name)}"
            ),
        }
        if index_manifest
        else None,
        "links": links,
    }

    if library_section:
        from ..library_projection import project_staging_library

        return project_staging_library(result, library_section)
    return result


class ArchiveReadModel:
    """Lightweight summary/section reads remain separate from full projections."""

    @staticmethod
    def summary(name: str, root: Path) -> dict[str, Any]:
        return _archive_summary_payload(name, root)

    @staticmethod
    def section(
        name: str,
        root: Path,
        section: str,
        *,
        cursor: str | None = None,
        limit: int = 24,
    ) -> dict[str, Any]:
        return _archive_section_payload(
            name, root, section=section, cursor=cursor, limit=limit
        )

    @staticmethod
    def detail(name: str, **options) -> dict[str, Any]:
        return archive_detail(name, **options)


archive_read_model = ArchiveReadModel()


def list_archives(
    q: str | None = None,
    cursor: str | None = None,
    limit: int = 50,
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
            decoded = decode_cursor(
                cursor, namespace="archives", filters=cursor_filters
            )
            if set(decoded) != {"modified_epoch_us", "name"}:
                raise ValueError("archive cursor position is invalid")
        except (ValueError, TypeError, KeyError) as exc:
            raise ApplicationError(400, "无效的实验档案分页 cursor") from exc
    try:
        database = _ensure_archive_read_catalog(root)
        rows, total_count, totals = list_catalog_archives(
            database,
            query=q,
            after_modified_epoch_us=(
                int(decoded["modified_epoch_us"]) if decoded else None
            ),
            after_name=(str(decoded["name"]) if decoded else None),
            limit=limit + 1,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise ApplicationError(503, f"无法读取实验档案目录索引: {exc}") from exc
    has_more = len(rows) > limit
    page = rows[:limit]
    archives = [
        {
            **{
                key: value
                for key, value in row.items()
                if key
                not in {
                    "source_revision",
                    "modified_epoch_us",
                    "published_epoch",
                    "catalog_error",
                }
            },
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
