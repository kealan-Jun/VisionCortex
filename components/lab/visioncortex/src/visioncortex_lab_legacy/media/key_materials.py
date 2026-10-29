"""Key material generation with independent frame/clip failure receipts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

from ..evidence.layout import ArchiveLayout
from ..material_naming import key_material_action_folder
from ..schemas import (
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    FrameEvidence,
    VideoInfo,
    ViewInput,
    set_event_admission,
)


@dataclass(frozen=True)
class KeyMaterialsServices:
    """Named execution ports; supplied explicitly at the composition boundary."""

    ParticipantVisualReviewer: Callable[..., Any]
    ViewFrameReader: Callable[..., Any]
    _artifact_json: Callable[..., Any]
    _attempt_key_material: Callable[..., Any]
    _best_event_frames_many: Callable[..., Any]
    _bounded_grounding_dino_temporal_rescue: Callable[..., Any]
    _event_participant_boxes: Callable[..., Any]
    _generate_detached_media: Callable[..., Any]
    _grounding_dino_key_frame_detections: Callable[..., Any]
    _key_material_event_folder_name: Callable[..., Any]
    _materialize_derived_media: Callable[..., Any]
    _prune_replaced_key_material_views: Callable[..., Any]
    _relative: Callable[..., Any]
    _release_auxiliary_model_caches: Callable[..., Any]
    _review_bounded_cap_keyframe: Callable[..., Any]
    _safe_folder_name: Callable[..., Any]
    _select_key_material_media_source: Callable[..., Any]
    _select_key_material_view_pair_with_peak_fallback: Callable[..., Any]
    _shared_candidate_time: Callable[..., Any]
    _write_aligned_frame: Callable[..., Any]
    create_grid_video: Callable[..., Any]
    extract_view_clip: Callable[..., Any]
    nearest_frame_evidence_many: Callable[..., Any]
    prepare_key_material_category_layout: Callable[..., Any]
    probe_video: Callable[..., Any]
    read_evidence_frame: Callable[..., Any]
    view_source_files: Callable[..., Any]
    write_annotated_frame: Callable[..., Any]
    write_json: Callable[..., Any]
    write_key_material_category_index: Callable[..., Any]


def _review_bounded_cap_keyframe(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    pair: tuple[str, str],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    *,
    services: KeyMaterialsServices,
) -> tuple[float | None, dict[str, Any]]:
    """Validate the cap instance on a few neighboring event frames.

    A missing visible cap does not justify deleting a real sequence-level
    action. Keep the current frame first, then inspect bounded alternatives;
    only the reviewer may admit an existing cap proposal. All requests share
    the final-annotation budget and content-addressed cache.
    """
    settings = config.get("key_materials", {}).get("participant_visual_review") or {}
    temporal = settings.get("temporal_keyframe_review") or {}
    if not (
        settings.get("enabled")
        and temporal.get("enabled")
        and "bottle_cap" in settings.get("classes", ["paper"])
        and "bottle_cap" in event.objects
        and config.get("mllm", {}).get("enabled")
        and (event.model_understanding or {}).get("status") == "completed"
    ):
        return None, {"status": "not_applicable"}
    maximum = int(temporal.get("max_frames_per_event", 5))
    if not 1 <= maximum <= 5:
        raise ValueError("Cap keyframe review accepts one to five candidate times")
    span = max(0.0, event.global_end_ms - event.global_start_ms)
    candidates = []
    values = (
        event.key_global_ms,
        event.key_global_ms + 0.25 * span,
        event.key_global_ms - 0.25 * span,
        event.global_end_ms,
        event.global_start_ms,
    )
    for value in values:
        timestamp = round(
            min(event.global_end_ms, max(event.global_start_ms, value)), 3
        )
        if timestamp not in candidates:
            candidates.append(timestamp)
    candidates = candidates[:maximum]
    nearest = {
        view_id: services.nearest_frame_evidence_many(
            detection_paths[view_id], candidates
        )
        for view_id in pair
    }
    by_view = {view.view_id: view for view in views}
    reviewer = services.ParticipantVisualReviewer(
        config,
        layout.work,
        layout.json_config,
        services._grounding_dino_key_frame_detections,
    )
    receipt: dict[str, Any] = {
        "status": "no_verified_candidate",
        "participant_class": "bottle_cap",
        "original_key_global_ms": float(event.key_global_ms),
        "maximum_candidate_times": maximum,
        "candidates": [],
        "full_scan_repeated": False,
        "source_copy_bytes": 0,
        "purpose": "instance-localized representative frame; not action certification",
    }
    reader = services.ViewFrameReader(max_open=2)
    try:
        for timestamp in candidates:
            prepared = []
            root = (
                layout.work
                / "participant-keyframe-candidates"
                / event.event_id
                / f"{timestamp:.3f}"
            )
            root.mkdir(parents=True, exist_ok=True)
            for role, view_id in zip(
                ("First-Person", "Third-Person"), pair, strict=True
            ):
                local_ms = transforms[view_id].to_local(timestamp)
                if not 0 <= local_ms <= infos[view_id].duration_ms:
                    break
                frame = reader.read(by_view[view_id], infos[view_id], local_ms)
                if frame is None:
                    break
                path = root / f"{role}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError("Could not retain cap keyframe candidate")
                evidence = nearest[view_id].get(timestamp)
                prepared.append(
                    {
                        "view_id": view_id,
                        "role_label": role,
                        "raw_path": path,
                        "detections": [box.model_dump() for box in evidence.detections]
                        if evidence
                        else [],
                    }
                )
            if len(prepared) != 2:
                receipt["candidates"].append(
                    {"global_ms": timestamp, "status": "missing_view"}
                )
                continue
            record, plan = reviewer.review(
                event, prepared, participant_class="bottle_cap"
            )
            visible = [view for view, item in plan.items() if item["boxes"]]
            receipt["candidates"].append(
                {
                    "global_ms": timestamp,
                    "status": record["status"],
                    "input_fingerprint": record["input_fingerprint"],
                    "localized_view_ids": visible,
                    "images": {
                        view["view_id"]: str(view["raw_path"]) for view in prepared
                    },
                }
            )
            if visible:
                receipt.update(status="selected", selected_global_ms=timestamp)
                return timestamp, receipt
    finally:
        reader.close()
    return None, receipt


def _select_key_material_media_source(
    view: ViewInput,
    info: VideoInfo,
    transform: AlignmentTransform,
    event: EvidenceEvent,
    group: ExperimentGroup,
    local_experiment_source: tuple[ViewInput, VideoInfo] | None,
    *,
    before_ms: float,
    after_ms: float,
) -> tuple[ViewInput, VideoInfo, float, float, float, dict[str, Any]]:
    """Use a derived experiment clip only when it honestly covers the event.

    Encoded experiment clips can end a fraction of a frame before the requested
    global group boundary. An event close to that tail must fall back to the
    original source instead of aborting the archive or clamping its timestamp.
    """

    clip_start_global = max(event.global_start_ms - before_ms, 0.0)
    clip_end_global = event.global_end_ms + after_ms
    original_key_ms = transform.to_local(event.key_global_ms)
    original_start_ms = max(0.0, transform.to_local(clip_start_global))
    original_end_ms = min(
        info.duration_ms,
        transform.to_local(clip_end_global),
    )
    selection: dict[str, Any] = {
        "schema_version": "visioncortex-key-material-source-selection/1",
        "selected_source": "original_source",
        "fallback_reason": "experiment_clip_unavailable",
        "requested_key_global_ms": float(event.key_global_ms),
    }
    if local_experiment_source is None:
        return (
            view,
            info,
            original_key_ms,
            original_start_ms,
            original_end_ms,
            selection,
        )

    candidate_view, candidate_info = local_experiment_source
    # Per-view experiment clips preserve source playback speed. Their zero is
    # the actual (possibly clipped) local source start, not the global start.
    source_start_ms = max(0.0, transform.to_local(group.global_start_ms))
    candidate_key_ms = transform.to_local(event.key_global_ms) - source_start_ms
    candidate_event_start_ms = (
        transform.to_local(event.global_start_ms) - source_start_ms
    )
    candidate_event_end_ms = transform.to_local(event.global_end_ms) - source_start_ms
    requested_start_ms = transform.to_local(clip_start_global) - source_start_ms
    requested_end_ms = transform.to_local(clip_end_global) - source_start_ms
    candidate_start_ms = max(
        0.0,
        requested_start_ms,
    )
    candidate_end_ms = min(
        candidate_info.duration_ms,
        requested_end_ms,
    )
    key_covered = 0.0 <= candidate_key_ms < candidate_info.duration_ms
    interval_covered = bool(
        0.0 <= candidate_event_start_ms < candidate_event_end_ms
        and candidate_event_end_ms <= candidate_info.duration_ms
    )
    selection.update(
        {
            "experiment_clip_path": str(candidate_view.video),
            "experiment_source_local_start_ms": float(source_start_ms),
            "experiment_clip_duration_ms": float(candidate_info.duration_ms),
            "experiment_relative_key_ms": float(candidate_key_ms),
            "experiment_relative_event_start_ms": float(candidate_event_start_ms),
            "experiment_relative_event_end_ms": float(candidate_event_end_ms),
            "experiment_requested_clip_start_ms": float(requested_start_ms),
            "experiment_requested_clip_end_ms": float(requested_end_ms),
            "experiment_relative_clip_start_ms": float(candidate_start_ms),
            "experiment_relative_clip_end_ms": float(candidate_end_ms),
            "experiment_clip_key_covered": key_covered,
            "experiment_clip_interval_covered": interval_covered,
        }
    )
    if key_covered and interval_covered:
        selection.update(
            {
                "selected_source": "verified_local_experiment_clip",
                "fallback_reason": None,
            }
        )
        return (
            candidate_view,
            candidate_info,
            candidate_key_ms,
            candidate_start_ms,
            candidate_end_ms,
            selection,
        )

    selection["fallback_reason"] = (
        "experiment_clip_key_timestamp_outside_media"
        if not key_covered
        else "experiment_clip_event_interval_outside_media"
    )
    return (
        view,
        info,
        original_key_ms,
        original_start_ms,
        original_end_ms,
        selection,
    )


def _prune_replaced_key_material_views(
    event: EvidenceEvent, first_view: str, third_view: str
) -> None:
    """Drop obsolete aliases after both replacement views finish extracting."""
    expected = {first_view, third_view, "aligned_first_third"}
    removed = {
        field: {
            key: value
            for key, value in getattr(event, field).items()
            if key not in expected
        }
        for field in ("key_frames", "key_clips")
    }
    if not any(removed.values()):
        return
    event.observability.setdefault("key_material_view_reference_history", []).append(
        {
            "reason": "final_material_view_pair_changed",
            "removed_aliases": removed,
            "final_view_ids": [first_view, third_view],
            "source_media_modified": False,
        }
    )
    for field in removed:
        setattr(
            event,
            field,
            {
                key: value
                for key, value in getattr(event, field).items()
                if key in expected
            },
        )


def _attempt_key_material(
    kind: str, generate: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """Contain one media failure without claiming its old output is current."""
    started = time.perf_counter()
    try:
        return {**generate(), "status": "completed", "artifact_kind": kind}
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        cv2.error,
    ) as error:
        return {
            "status": "failed",
            "artifact_kind": kind,
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "duration_seconds": round(time.perf_counter() - started, 6),
            "retryable": True,
        }


def materialize_key_materials(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    publisher: Any | None = None,
    archive_id: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    materialize_event_ids: set[str] | None = None,
    *,
    services: KeyMaterialsServices,
) -> None:
    by_view = {view.view_id: view for view in views}
    before = float(config["segmentation"]["key_clip_pre_seconds"]) * 1000.0
    after = float(config["segmentation"]["key_clip_post_seconds"]) * 1000.0
    encoder = config["performance"]["ffmpeg_video_encoder"]
    workers = max(
        1, min(2, int(config["performance"].get("materialization_workers", 2)))
    )
    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    runtime_records: list[dict[str, Any]] = []
    stage_started = time.perf_counter()
    all_accepted_events = [
        event for event in events if event.accepted and event.event_id in group_by_event
    ]
    accepted_events = [
        event
        for event in all_accepted_events
        if materialize_event_ids is None or event.event_id in materialize_event_ids
    ]
    include_empty_categories = bool(
        config.get("archive", {}).get("include_empty_action_categories", True)
    )
    services.prepare_key_material_category_layout(
        layout,
        groups,
        all_accepted_events,
        include_empty_categories=include_empty_categories,
    )
    best_frames_by_view = {
        view_id: services._best_event_frames_many(path, accepted_events, transforms)
        for view_id, path in detection_paths.items()
    }
    material_view_pairs: dict[str, tuple[str, str]] = {}
    key_frame_selection_records: list[dict[str, Any]] = []
    for event_index, event in enumerate(accepted_events):
        if progress_callback is not None:
            progress_callback(event_index, len(accepted_events))
        group = group_by_event[event.event_id]
        ranked: list[tuple[float, int, str, FrameEvidence, dict[str, Any]]] = []
        for role_rank, view_id in enumerate(
            dict.fromkeys(
                (
                    group.first_person_view,
                    group.third_person_view,
                    *event.supporting_views,
                )
            )
        ):
            selected = (best_frames_by_view.get(view_id) or {}).get(event.event_id)
            if selected is None:
                continue
            frame, score, receipt = selected
            ranked.append((score, -role_rank, view_id, frame, receipt))
        previous_key_global_ms = float(event.key_global_ms)
        if ranked:
            directly_supported_view_ids = {
                str(item)
                for item in (event.semantic_review or {}).get(
                    "directly_supported_view_ids", []
                )
            }
            directly_supported_ranked = [
                item for item in ranked if item[2] in directly_supported_view_ids
            ]
            cv_supported_ranked = [
                item for item in ranked if item[2] in event.supporting_views
            ]
            use_cv_support = (event.model_understanding or {}).get(
                "status"
            ) != "completed"
            eligible_ranked = (
                directly_supported_ranked
                or (cv_supported_ranked if use_cv_support else [])
                or ranked
            )
            score, _, source_view_id, frame, receipt = max(
                eligible_ranked,
                key=lambda item: (
                    services._shared_candidate_time(
                        event, transforms, float(item[3].global_ms)
                    ),
                    item[0],
                    item[1],
                ),
            )
            event.key_global_ms = float(frame.global_ms)
            event.observability["key_frame_selection"] = {
                "schema_version": "visioncortex-participant-key-frame-selection/1",
                "policy": "highest participant-instance coverage within accepted event bounds",
                "source_view_id": source_view_id,
                "directly_supported_view_ids": sorted(directly_supported_view_ids),
                "direct_support_priority_applied": bool(directly_supported_ranked),
                "cv_source_priority_applied": bool(
                    not directly_supported_ranked
                    and use_cv_support
                    and cv_supported_ranked
                ),
                "shared_candidate_time_priority_applied": services._shared_candidate_time(
                    event, transforms, float(frame.global_ms)
                ),
                "previous_key_global_ms": previous_key_global_ms,
                "selected_key_global_ms": event.key_global_ms,
                "selection_offset_ms": round(
                    event.key_global_ms - previous_key_global_ms, 3
                ),
                "selection_score": round(score, 6),
                "selection_receipt": receipt,
            }
        actor_required_actions = {
            ActionType.HAND_OBJECT_CONTACT,
            ActionType.CONTAINER_STATE_CHANGE,
            ActionType.DEVICE_PANEL_OPERATION,
            ActionType.PIPETTE_TRANSFER_OPERATION,
        }
        if event.action_type in actor_required_actions:
            rescue_global_ms, rescue_receipt = (
                services._bounded_grounding_dino_temporal_rescue(
                    event,
                    group,
                    views,
                    infos,
                    transforms,
                    detection_paths,
                    config,
                )
            )
            event.observability.setdefault("key_frame_selection", {})[
                "grounding_dino_temporal_rescue"
            ] = rescue_receipt
            if rescue_global_ms is not None:
                event.key_global_ms = float(rescue_global_ms)
                event.observability["key_frame_selection"].update(
                    {
                        "policy": (
                            "highest bounded grounded participant interaction "
                            "within accepted event bounds"
                        ),
                        "source_view_id": rescue_receipt.get("selected_view_id"),
                        "selected_key_global_ms": event.key_global_ms,
                        "selection_offset_ms": round(
                            event.key_global_ms - previous_key_global_ms, 3
                        ),
                        "grounding_dino_temporal_rescue_applied": True,
                    }
                )
        try:
            pair, pair_receipt = (
                services._select_key_material_view_pair_with_peak_fallback(
                    group,
                    event,
                    views,
                    infos,
                    transforms,
                    previous_key_global_ms,
                )
            )
        except ValueError as error:
            # One CV candidate can outlive a shorter physical camera tail after
            # alignment or participant-keyframe reranking.  That candidate has
            # no honest dual-role key material, but it must not abort unrelated
            # events or the experiment package.  Reject it fail-closed and let
            # semantic curation write the normal machine-quarantine receipt.
            event.key_global_ms = previous_key_global_ms
            set_event_admission(event, "rejected")
            event.uncertainty = list(
                dict.fromkeys(
                    [*event.uncertainty, "key_material_dual_role_coverage_unavailable"]
                )
            )
            event.audit_reason = (
                f"{event.audit_reason}; " if event.audit_reason else ""
            ) + "machine-quarantined: no honest dual-role key material"
            failure_receipt = {
                "schema_version": "visioncortex-key-material-view-selection/1",
                "status": "machine_quarantined_missing_dual_role_key_material",
                "evidence_classification": "PARTIAL_EVIDENCE",
                "reason": str(error),
                "key_global_ms": previous_key_global_ms,
                "timestamp_clamped": False,
                "synthetic_cross_view_evidence": False,
                "manual_fallback_required": False,
                "analysis_continuation_allowed": True,
            }
            event.observability["key_material_view_selection"] = failure_receipt
            event.semantic_review = {
                **dict(event.semantic_review or {}),
                "model_status": "not_run",
                "error_class": "key_material_dual_role_coverage_unavailable",
                "error": str(error),
                "evidence_classification": "PARTIAL_EVIDENCE",
                "retryable": False,
            }
            key_frame_selection_records.append(
                {
                    "event_id": event.event_id,
                    **dict(event.observability.get("key_frame_selection") or {}),
                    "key_material_view_selection": failure_receipt,
                }
            )
            continue
        if pair_receipt.get("key_timestamp_fallback"):
            event.observability.setdefault("key_frame_selection", {}).update(
                {
                    "selected_key_global_ms": float(event.key_global_ms),
                    "selection_offset_ms": round(
                        event.key_global_ms - previous_key_global_ms, 3
                    ),
                    "cross_view_coverage_fallback": dict(
                        pair_receipt["key_timestamp_fallback"]
                    ),
                }
            )
        material_view_pairs[event.event_id] = pair
        event.observability["key_material_view_selection"] = pair_receipt
        reviewed_key_ms, cap_review = services._review_bounded_cap_keyframe(
            layout, event, pair, views, infos, transforms, detection_paths, config
        )
        if cap_review["status"] != "not_applicable":
            event.observability.setdefault("key_frame_selection", {})[
                "cap_visual_candidate_review"
            ] = cap_review
        if reviewed_key_ms is not None:
            event.key_global_ms = reviewed_key_ms
            selection = event.observability["key_frame_selection"]
            selection["prior_cv_selection"] = {
                key: selection[key]
                for key in ("source_view_id", "selection_score", "selection_receipt")
                if key in selection
            }
            selection.pop("selection_score", None)
            event.observability["key_frame_selection"].update(
                {
                    "policy": "visually reviewed bottle-cap instance on bounded event frames",
                    "source_view_id": cap_review["candidates"][-1][
                        "localized_view_ids"
                    ][0],
                    "selection_receipt": cap_review["candidates"][-1],
                    "selected_key_global_ms": reviewed_key_ms,
                    "selection_offset_ms": round(
                        reviewed_key_ms - previous_key_global_ms, 3
                    ),
                }
            )
        key_frame_selection_records.append(
            {
                "event_id": event.event_id,
                **dict(event.observability.get("key_frame_selection") or {}),
                "key_material_view_selection": pair_receipt,
            }
        )
    accepted_events = [event for event in accepted_events if event.accepted]
    if progress_callback is not None:
        progress_callback(len(accepted_events), len(accepted_events))
    if bool(config.get("performance", {}).get("release_auxiliary_models_after_event")):
        services._release_auxiliary_model_caches()
    services.write_json(
        layout.json_config / "key_frame_selection.json",
        {
            "schema_version": "visioncortex-participant-key-frame-selection/1",
            "event_count": len(key_frame_selection_records),
            "records": key_frame_selection_records,
        },
    )
    accepted_timestamps = [event.key_global_ms for event in accepted_events]
    lookup_started = time.perf_counter()
    nearest_by_view = {
        view_id: services.nearest_frame_evidence_many(path, accepted_timestamps)
        for view_id, path in detection_paths.items()
    }
    lookup_seconds = time.perf_counter() - lookup_started
    local_experiment_sources: dict[tuple[str, str], tuple[ViewInput, VideoInfo]] = {}
    if bool(
        config.get("performance", {}).get(
            "reuse_experiment_clips_for_key_materials", True
        )
    ):
        for group in groups:
            for role_label, view_id in (
                ("First-Person", group.first_person_view),
                ("Third-Person", group.third_person_view),
            ):
                relative = group.videos.get(role_label.lower())
                if not relative:
                    continue
                path = layout.root / relative
                if not path.is_file():
                    continue
                source_view = ViewInput(
                    view_id=view_id,
                    role=by_view[view_id].role,
                    video=path,
                )
                local_experiment_sources[(group.group_id, view_id)] = (
                    source_view,
                    services.probe_video(path),
                )
    frame_reader_max_open = max(
        1,
        int(config.get("performance", {}).get("key_material_frame_reader_max_open", 2)),
    )
    # Event clips are processed in stable order, and each role maps to a
    # distinct physical view. Reusing one bounded decoder per view avoids
    # reopening the same experiment clip for every key frame while preserving
    # exact timestamp seeks and the original-source fallback path.
    frame_readers = {
        view_id: services.ViewFrameReader(max_open=frame_reader_max_open)
        for view_id in by_view
    }
    overlap_aligned = bool(
        publisher is None
        and config.get("performance", {}).get("overlap_aligned_key_materials", False)
    )
    aligned_executor = (
        ThreadPoolExecutor(
            max_workers=max(
                1,
                int(
                    config.get("performance", {}).get("aligned_key_material_workers", 1)
                ),
            ),
            thread_name_prefix="key-material-aligned",
        )
        if overlap_aligned
        else None
    )
    aligned_jobs: list[Any] = []

    def register_artifact(event, group, artifact_type, view_id, result, field):
        mapping = getattr(event, field)
        key = view_id or "aligned_first_third"
        path = result.get("frame_path" if field == "key_frames" else "clip_path")
        if result["status"] != "completed" or path is None:
            previous = mapping.pop(key, None)
            if previous:
                event.observability.setdefault(
                    "key_material_unavailable_history", []
                ).append(
                    {
                        "view_id": key,
                        "artifact_type": artifact_type,
                        "previous_path": previous,
                        "reason": result.get("error") or result.get("reason"),
                        "previous_file_retained": True,
                    }
                )
            return None
        path = Path(path)
        relative = services._relative(path, layout.root)
        mapping[key] = relative
        sidecar = path.with_suffix(".json")
        services.write_json(
            sidecar,
            services._artifact_json(
                group, event, artifact_type, relative, view_id, transforms, archive_id
            ),
        )
        if publisher is not None:
            publisher.publish_file(path)
            publisher.publish_file(sidecar)
        return path

    def materialize_aligned(
        event,
        group,
        action_folder,
        frame_dir,
        clip_dir,
        first_material_view,
        third_material_view,
        frame_paths,
        clip_paths,
        event_started,
    ):
        aligned_started = time.perf_counter()
        aligned_frame = frame_dir / "Aligned_First+Third.jpg"
        aligned_clip = clip_dir / "Aligned_First+Third.mp4"

        def generate_frame():
            services._generate_detached_media(
                aligned_frame,
                lambda: services._write_aligned_frame(
                    frame_paths["First-Person"],
                    frame_paths["Third-Person"],
                    aligned_frame,
                    (first_material_view, third_material_view),
                ),
            )
            return {
                "frame_path": aligned_frame,
                "frame_output_bytes": aligned_frame.stat().st_size,
            }

        def generate_clip():
            aligned_inputs = [
                (first_material_view, clip_paths["First-Person"]),
                (third_material_view, clip_paths["Third-Person"]),
            ]
            cache = services._materialize_derived_media(
                aligned_clip,
                "key-aligned-clip",
                {
                    "event_id": event.event_id,
                    "global_start_ms": max(event.global_start_ms - before, 0.0),
                    "global_end_ms": event.global_end_ms + after,
                    "layout": "first_person_left,third_person_right",
                    "grid_shape": "2x1@640x360_each",
                },
                [path for _, path in aligned_inputs],
                config,
                lambda: services.create_grid_video(
                    aligned_inputs, aligned_clip, encoder
                ),
                content_address_inputs=True,
            )
            return {
                "clip_path": aligned_clip,
                "clip_output_bytes": aligned_clip.stat().st_size,
                **cache,
            }

        def attempt_pair(kind, paths, generate):
            missing = sorted({"First-Person", "Third-Person"} - paths.keys())
            if missing:
                return {
                    "status": "unavailable",
                    "artifact_kind": kind,
                    "reason": "missing_source_material",
                    "missing_roles": missing,
                    "retryable": True,
                }
            return services._attempt_key_material(kind, generate)

        frame_result = attempt_pair("aligned_key_frame", frame_paths, generate_frame)
        clip_result = attempt_pair("aligned_key_clip", clip_paths, generate_clip)
        results = event.observability["key_material_materialization"]["artifacts"]
        results["aligned_first_third"] = {
            "key_frame": frame_result,
            "key_clip": clip_result,
        }
        # Do not preserve Path objects in event state/checkpoint serialization.
        results["aligned_first_third"] = {
            kind: {
                k: v for k, v in item.items() if k not in {"frame_path", "clip_path"}
            }
            for kind, item in results["aligned_first_third"].items()
        }
        complete = all(
            item["status"] == "completed"
            for view in results.values()
            for item in view.values()
        )
        event.observability["key_material_materialization"].update(
            status="completed" if complete else "partial",
            retry_required=not complete,
            analysis_continuation_allowed=True,
        )
        register_artifact(
            event,
            group,
            "aligned_first_third_key_frame",
            None,
            frame_result,
            "key_frames",
        )
        register_artifact(
            event, group, "aligned_first_third_key_clip", None, clip_result, "key_clips"
        )
        # Finalize all surviving sidecars with the same completion state and
        # current media references after both role and composition attempts.
        for field, kind in (("key_frames", "key_frame"), ("key_clips", "key_clip")):
            for view_id, relative in getattr(event, field).items():
                aligned = view_id == "aligned_first_third"
                sidecar = (layout.root / relative).with_suffix(".json")
                services.write_json(
                    sidecar,
                    services._artifact_json(
                        group,
                        event,
                        f"aligned_first_third_{kind}" if aligned else kind,
                        relative,
                        None if aligned else view_id,
                        transforms,
                        archive_id,
                    ),
                )
                if publisher is not None:
                    publisher.publish_file(sidecar)
        return {
            "event_id": event.event_id,
            "experiment_group_id": group.group_id,
            "action_type": event.action_type.value,
            "action_category_folder": action_folder,
            "role_label": "Aligned-First-Third",
            "view_id": "aligned_first_third",
            "duration_seconds": round(time.perf_counter() - aligned_started, 6),
            "frame_output_bytes": frame_result.get("frame_output_bytes", 0),
            "clip_output_bytes": clip_result.get("clip_output_bytes", 0),
            "event_wall_duration_seconds": round(
                time.perf_counter() - event_started, 6
            ),
            "overlapped_with_next_event": overlap_aligned,
            **{k: v for k, v in clip_result.items() if k.startswith("cache_")},
            "artifact_results": results["aligned_first_third"],
            "status": "completed" if complete else "partial",
        }

    selected_event_ids = {event.event_id for event in accepted_events}
    for event in events:
        if event.event_id not in selected_event_ids:
            continue
        event_started = time.perf_counter()
        group = group_by_event[event.event_id]
        first_material_view, third_material_view = material_view_pairs[event.event_id]
        folder = group.archive_folder or services._safe_folder_name(group.group_id)
        action_folder = key_material_action_folder(event.action_type)
        event_folder = services._key_material_event_folder_name(layout, folder, event)
        frame_dir = layout.key_frames / folder / action_folder / event_folder
        clip_dir = layout.key_clips / folder / action_folder / event_folder
        frame_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: dict[str, Path] = {}
        clip_paths: dict[str, Path] = {}

        def extract_role(role_label: str, view_id: str) -> dict[str, Any]:
            role_started = time.perf_counter()
            view = by_view[view_id]
            transform = transforms[view_id]
            clip_start_global = max(event.global_start_ms - before, 0.0)
            clip_end_global = event.global_end_ms + after
            local_experiment_source = local_experiment_sources.get(
                (group.group_id, view_id)
            )
            (
                material_view,
                material_info,
                local_key_ms,
                local_start,
                local_end,
                material_source_selection,
            ) = services._select_key_material_media_source(
                view,
                infos[view_id],
                transform,
                event,
                group,
                local_experiment_source,
                before_ms=before,
                after_ms=after,
            )
            material_source = str(material_source_selection["selected_source"])
            material_source_path = (
                str(material_view.video)
                if material_source == "verified_local_experiment_clip"
                else None
            )

            def generate_frame():
                if not 0.0 <= local_key_ms < material_info.duration_ms:
                    raise ValueError(
                        f"{event.event_id}/{view_id} key timestamp is outside the material source"
                    )
                frame_started = time.perf_counter()
                nearest = nearest_by_view[view_id].get(float(event.key_global_ms))
                frame, frame_provenance = services.read_evidence_frame(
                    view, infos[view_id], nearest
                )
                decoded_key_global_ms = None
                used_offset_ms = 0.0
                if frame is not None:
                    decoded_key_global_ms = transform.to_global(
                        frame_provenance["view_local_ms"]
                    )
                    if not clip_start_global <= decoded_key_global_ms < clip_end_global:
                        frame = None
                        decoded_key_global_ms = None
                        frame_provenance.update(
                            status="unverified",
                            reason="native_frame_outside_event_context",
                            detections_bound_to_pixels=False,
                        )
                    else:
                        used_offset_ms = decoded_key_global_ms - event.key_global_ms
                if frame is None:
                    frame_reader = frame_readers[view_id]
                    for offset_ms in (0.0, -100.0, 100.0, -250.0, 250.0):
                        candidate_ms = local_key_ms + offset_ms
                        if not 0.0 <= candidate_ms <= material_info.duration_ms:
                            continue
                        frame = frame_reader.read(
                            material_view, material_info, candidate_ms
                        )
                        if frame is not None:
                            used_offset_ms = offset_ms
                            break
                if frame is None:
                    raise RuntimeError(
                        f"{event.event_id}/{view_id} key frame decode failed"
                    )
                frame_seconds = time.perf_counter() - frame_started
                detected_boxes = (
                    [box.model_dump() for box in nearest.detections]
                    if nearest and frame_provenance["detections_bound_to_pixels"]
                    else []
                )
                frame_provenance["decoded_global_ms"] = decoded_key_global_ms
                frame_provenance["alignment_uncertainty_ms"] = transform.uncertainty_ms
                frame_provenance["physical_cross_view_sync_verified"] = False
                frame_provenance["display_source"] = (
                    "verified_native_source_frame"
                    if frame_provenance["detections_bound_to_pixels"]
                    else material_source
                )
                annotation_input_root = (
                    layout.work / "key-material-annotation-inputs" / event.event_id
                )
                annotation_input_root.mkdir(parents=True, exist_ok=True)
                raw_annotation_frame = annotation_input_root / f"{role_label}.jpg"
                if not cv2.imwrite(
                    str(raw_annotation_frame),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError(
                        f"Unable to retain raw key-material annotation frame: {raw_annotation_frame}"
                    )
                services.write_json(
                    annotation_input_root / f"{role_label}.json",
                    {
                        "schema_version": "visioncortex-key-material-annotation-input/2",
                        "event_id": event.event_id,
                        "view_id": view_id,
                        "role_label": role_label,
                        "requested_key_global_ms": float(event.key_global_ms),
                        "decoded_key_global_ms": decoded_key_global_ms,
                        "seek_target_global_ms": (
                            None
                            if decoded_key_global_ms is not None
                            else transform.to_global(
                                material_source_selection[
                                    "experiment_source_local_start_ms"
                                ]
                                + local_key_ms
                                + used_offset_ms
                            )
                            if material_source == "verified_local_experiment_clip"
                            else transform.to_global(local_key_ms + used_offset_ms)
                        ),
                        "key_frame_time_basis": (
                            "verified_native_frame_with_alignment_transform"
                            if frame_provenance["detections_bound_to_pixels"]
                            else "decoder_seek_target_only; actual_frame_time_unverified"
                        ),
                        "source_frame_verification": frame_provenance,
                        "image_sha256": hashlib.sha256(
                            raw_annotation_frame.read_bytes()
                        ).hexdigest(),
                        "material_source": frame_provenance["display_source"],
                        "material_source_path": (
                            str(nearest.source_frame.source_path)
                            if frame_provenance["detections_bound_to_pixels"]
                            else material_source_path
                        ),
                        "clip_material_source_selection": material_source_selection,
                        "upstream_source_files": [
                            str(path) for path in services.view_source_files(view)
                        ],
                        "frame_decode_offset_ms": used_offset_ms,
                        "detections": detected_boxes,
                        "retention": "existing_persistent_run_cache",
                        "formally_published": False,
                    },
                )
                boxes, annotation_filter = services._event_participant_boxes(
                    event, detected_boxes, view_id=view_id
                )
                annotation_filter["source_frame_verification"] = frame_provenance
                base = role_label
                frame_path = frame_dir / f"{base}.jpg"
                services._generate_detached_media(
                    frame_path,
                    lambda: services.write_annotated_frame(frame, boxes, frame_path),
                )
                return {
                    "frame_path": frame_path,
                    "frame_decode_offset_ms": used_offset_ms,
                    "frame_duration_seconds": round(frame_seconds, 6),
                    "frame_output_bytes": frame_path.stat().st_size,
                    "annotation_filter": annotation_filter,
                    "source_frame_verification": frame_provenance,
                }

            def generate_clip():
                if local_end <= local_start:
                    raise ValueError(
                        f"{event.event_id}/{view_id} key clip boundary is outside the material source"
                    )
                clip_path = clip_dir / f"{role_label}.mp4"
                clip_started = time.perf_counter()
                clip_cache = services._materialize_derived_media(
                    clip_path,
                    "key-view-clip",
                    {
                        "event_id": event.event_id,
                        "role_label": role_label,
                        "view_id": view_id,
                        "local_start_ms": local_start,
                        "local_end_ms": local_end,
                        "global_start_ms": clip_start_global,
                        "global_end_ms": clip_end_global,
                        "material_source": material_source,
                    },
                    services.view_source_files(material_view),
                    config,
                    lambda: services.extract_view_clip(
                        material_view,
                        material_info,
                        clip_path,
                        local_start,
                        local_end - local_start,
                        encoder,
                    ),
                )
                clip_seconds = time.perf_counter() - clip_started
                return {
                    "clip_path": clip_path,
                    "clip_duration_seconds": round(clip_seconds, 6),
                    "clip_source_duration_seconds": round(
                        (local_end - local_start) / 1000.0, 6
                    ),
                    "clip_output_bytes": clip_path.stat().st_size,
                    **clip_cache,
                }

            frame_result = services._attempt_key_material("key_frame", generate_frame)
            clip_result = services._attempt_key_material("key_clip", generate_clip)
            return {
                "event_id": event.event_id,
                "experiment_group_id": group.group_id,
                "action_type": event.action_type.value,
                "action_category_folder": action_folder,
                "role_label": role_label,
                "view_id": view_id,
                "material_source": material_source,
                "material_source_path": material_source_path,
                "material_source_selection": material_source_selection,
                "upstream_source_files": [
                    str(path) for path in services.view_source_files(view)
                ],
                "duration_seconds": round(time.perf_counter() - role_started, 6),
                **{
                    k: v
                    for item in (frame_result, clip_result)
                    for k, v in item.items()
                    if k
                    not in {
                        "status",
                        "artifact_kind",
                        "duration_seconds",
                        "error",
                        "error_type",
                        "retryable",
                    }
                },
                "frame_result": frame_result,
                "clip_result": clip_result,
            }

        role_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="key-material-media",
        ) as executor:
            futures = {
                role_label: executor.submit(
                    services._attempt_key_material,
                    "role",
                    lambda role_label=role_label, view_id=view_id: extract_role(
                        role_label, view_id
                    ),
                )
                for role_label, view_id in (
                    ("First-Person", first_material_view),
                    ("Third-Person", third_material_view),
                )
            }
            for role_label, future in futures.items():
                result = future.result()
                if result["status"] == "failed":
                    view_id = (
                        first_material_view
                        if role_label == "First-Person"
                        else third_material_view
                    )
                    result = {
                        "event_id": event.event_id,
                        "experiment_group_id": group.group_id,
                        "action_type": event.action_type.value,
                        "action_category_folder": action_folder,
                        "role_label": role_label,
                        "view_id": view_id,
                        "frame_result": {**result, "artifact_kind": "key_frame"},
                        "clip_result": {**result, "artifact_kind": "key_clip"},
                    }
                role_results[role_label] = result

        services._prune_replaced_key_material_views(
            event, first_material_view, third_material_view
        )
        event.observability["key_material_materialization"] = {
            "schema_version": "visioncortex-key-material-completion/1",
            "status": "in_progress",
            "analysis_continuation_allowed": True,
            "artifacts": {},
        }
        for role_label, view_id in (
            ("First-Person", first_material_view),
            ("Third-Person", third_material_view),
        ):
            result = role_results[role_label]
            frame_result, clip_result = result["frame_result"], result["clip_result"]
            artifact_results = {
                kind: {
                    k: v
                    for k, v in item.items()
                    if k not in {"frame_path", "clip_path"}
                }
                for kind, item in (
                    ("key_frame", frame_result),
                    ("key_clip", clip_result),
                )
            }
            event.observability["key_material_materialization"]["artifacts"][
                view_id
            ] = artifact_results
            annotations = event.observability.setdefault(
                "key_material_annotation",
                {
                    "schema_version": "visioncortex-key-material-annotation/1",
                    "mode": "event_participants_only",
                    "views": {},
                },
            )["views"]
            sources = event.observability.setdefault("key_frame_sources", {})
            if frame_result["status"] == "completed":
                sources[view_id] = frame_result["source_frame_verification"]
                annotations[view_id] = frame_result["annotation_filter"]
                offset = frame_result["frame_decode_offset_ms"]
                note = f"{view_id} key frame decode offset {offset:+.0f} ms"
                if offset and note not in event.uncertainty:
                    event.uncertainty.append(note)
            else:
                sources.pop(view_id, None)
                annotations.pop(view_id, None)
            frame_path = register_artifact(
                event, group, "key_frame", view_id, frame_result, "key_frames"
            )
            clip_path = register_artifact(
                event, group, "key_clip", view_id, clip_result, "key_clips"
            )
            if frame_path is not None:
                frame_paths[role_label] = frame_path
            if clip_path is not None:
                clip_paths[role_label] = clip_path
            runtime_records.append(
                {
                    **{
                        key: value
                        for key, value in result.items()
                        if key
                        not in {
                            "frame_path",
                            "clip_path",
                            "frame_result",
                            "clip_result",
                        }
                    },
                    "artifact_results": artifact_results,
                    "status": "completed"
                    if all(
                        x["status"] == "completed" for x in artifact_results.values()
                    )
                    else "partial",
                }
            )

        aligned_arguments = (
            event,
            group,
            action_folder,
            frame_dir,
            clip_dir,
            first_material_view,
            third_material_view,
            dict(frame_paths),
            dict(clip_paths),
            event_started,
        )
        if aligned_executor is None:
            runtime_records.append(materialize_aligned(*aligned_arguments))
        else:
            aligned_jobs.append(
                aligned_executor.submit(materialize_aligned, *aligned_arguments)
            )

    if aligned_executor is not None:
        try:
            runtime_records.extend(job.result() for job in aligned_jobs)
        finally:
            aligned_executor.shutdown(wait=True, cancel_futures=True)
        event_order = {
            event.event_id: index for index, event in enumerate(accepted_events)
        }
        role_order = {
            "First-Person": 0,
            "Third-Person": 1,
            "Aligned-First-Third": 2,
        }
        runtime_records.sort(
            key=lambda item: (
                event_order.get(str(item.get("event_id")), len(event_order)),
                role_order.get(str(item.get("role_label")), len(role_order)),
            )
        )
    for frame_reader in frame_readers.values():
        frame_reader.close()

    category_index_path = services.write_key_material_category_index(
        layout,
        groups,
        all_accepted_events,
        publisher=publisher,
        include_empty_categories=include_empty_categories,
    )
    runtime_path = layout.json_config / "key_material_materialization_runtime.json"
    previous_runtime: dict[str, Any] = {}
    if materialize_event_ids is not None and runtime_path.exists():
        try:
            previous_runtime = json.loads(runtime_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            previous_runtime = {}
    previous_records = [
        item
        for item in previous_runtime.get("records") or []
        if str(item.get("event_id") or "") not in selected_event_ids
    ]
    previous_passes = list(previous_runtime.get("materialization_passes") or [])
    if not previous_passes and previous_runtime:
        previous_passes = [
            {
                "scope": "initial",
                "event_ids": sorted(
                    {
                        str(item.get("event_id"))
                        for item in previous_runtime.get("records") or []
                        if item.get("event_id")
                    }
                ),
                "duration_seconds": previous_runtime.get("total_duration_seconds", 0.0),
            }
        ]
    current_duration = round(time.perf_counter() - stage_started, 6)
    materialization_passes = previous_passes + [
        {
            "scope": (
                "initial"
                if materialize_event_ids is None
                else "post_semantic_selective_refresh"
            ),
            "event_ids": sorted(selected_event_ids),
            "duration_seconds": current_duration,
        }
    ]
    combined_records = previous_records + runtime_records
    incomplete_artifacts = [
        {
            "event_id": record["event_id"],
            "view_id": record["view_id"],
            "artifact_kind": kind,
            "status": item["status"],
            "reason": item.get("error") or item.get("reason"),
            "retryable": True,
        }
        for record in combined_records
        for kind, item in (record.get("artifact_results") or {}).items()
        if item["status"] != "completed"
    ]
    materialization_passes[-1]["incomplete_artifacts"] = [
        item for item in incomplete_artifacts if item["event_id"] in selected_event_ids
    ]
    services.write_json(
        runtime_path,
        {
            "schema_version": "visioncortex-key-materialization-runtime/1",
            "status": "partial" if incomplete_artifacts else "completed",
            "incomplete_artifacts": incomplete_artifacts,
            "retry_event_ids": sorted(
                {item["event_id"] for item in incomplete_artifacts}
            ),
            "analysis_continuation_allowed": True,
            "workers": workers,
            "frame_reader_reuse": True,
            "frame_reader_count": len(frame_readers),
            "frame_reader_max_open": frame_reader_max_open,
            "total_duration_seconds": round(
                sum(
                    float(item.get("duration_seconds") or 0.0)
                    for item in materialization_passes
                ),
                6,
            ),
            "detection_ledger_lookup_seconds": round(
                float(previous_runtime.get("detection_ledger_lookup_seconds") or 0.0)
                + lookup_seconds,
                6,
            ),
            "detection_ledger_passes": int(
                previous_runtime.get("detection_ledger_passes") or 0
            )
            + len(nearest_by_view),
            "archive_hierarchy_version": "2.0.0",
            "category_index": services._relative(category_index_path, layout.root),
            "candidate_event_count": len(all_accepted_events),
            "accepted_event_count": sum(
                event.accepted for event in all_accepted_events
            ),
            "machine_quarantined_event_count": sum(
                not event.accepted for event in all_accepted_events
            ),
            "last_pass_materialized_event_count": len(accepted_events),
            "materialization_passes": materialization_passes,
            "records": combined_records,
        },
    )
    if publisher is not None:
        publisher.publish_file(runtime_path)
