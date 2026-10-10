"""Participant grounding, bounded rescue and auxiliary model residency."""

from __future__ import annotations

import gc
import math
import re
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ..open_vocabulary_runtime import serialized_open_vocabulary
from ..schemas import (
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    FrameEvidence,
    VideoInfo,
    ViewInput,
    ViewRole,
)


@dataclass(frozen=True)
class ParticipantGroundingServices:
    """Named execution ports; supplied explicitly at the composition boundary."""

    ViewFrameReader: Callable[..., Any]
    _GROUNDING_DINO_ASSET_VALIDATION: Any
    _GROUNDING_DINO_MODEL_CACHE: Any
    _OPEN_VOCABULARY_ASSET_VALIDATION: Any
    _OPEN_VOCABULARY_MODEL_CACHE: Any
    _box_aspect_ratio: Callable[..., Any]
    _box_edge_gap_norm: Callable[..., Any]
    _box_iou: Callable[..., Any]
    _canonical_grounding_label: Callable[..., Any]
    _event_key_frame_score: Callable[..., Any]
    _event_participant_boxes: Callable[..., Any]
    _filter_grounded_actor_boxes: Callable[..., Any]
    _grounding_dino_key_frame_detections: Callable[..., Any]
    _grounding_dino_key_frame_detections_once: Callable[..., Any]
    _missing_state_transition_fallback_classes: Callable[..., Any]
    _preferred_grounding_terms_by_class: Callable[..., Any]
    _release_auxiliary_model_caches: Callable[..., Any]
    _select_manipulated_object_candidate: Callable[..., Any]
    _select_state_container_candidate: Callable[..., Any]
    _sha256_file: Callable[..., Any]
    _shared_candidate_time: Callable[..., Any]
    iter_frame_evidence: Callable[..., Any]
    normalize_participant_class: Callable[..., Any]
    park_open_vocabulary_model: Callable[..., Any]
    release_liquid_semantic_model_cache: Callable[..., Any]
    release_temporal_segmentation_model_cache: Callable[..., Any]
    run_with_cuda_oom_cpu_fallback: Callable[..., Any]
    yolo_world_prediction_device: Callable[..., Any]


_OPEN_VOCABULARY_MODEL_CACHE: dict[str, Any] = {}
_OPEN_VOCABULARY_ASSET_VALIDATION: set[tuple[str, str, str, str]] = set()
_GROUNDING_DINO_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_GROUNDING_DINO_ASSET_VALIDATION: set[tuple[str, str]] = set()


def _event_participant_boxes(
    event: EvidenceEvent,
    detections: Sequence[dict[str, Any]],
    view_id: str | None = None,
    maximum_interaction_gap_norm: float = 0.08,
    *,
    services: ParticipantGroundingServices,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter a delivery frame to the objects participating in one event.

    Full detector output remains in the frame-evidence ledger. Key-material
    images are explanatory evidence, so background detections must not appear.
    """

    participant_classes = {
        services.normalize_participant_class(item)
        for item in event.objects
        if str(item).strip()
    }
    actor_classes = {"hand", "gloved_hand"}
    tool_classes = {"pipette", "spearhead"}
    vessel_classes = {
        "container",
        "beaker",
        "tube",
        "sample_bottle",
        "sample_bottle_blue",
        "reagent_bottle",
    }
    detected_classes = {
        services.normalize_participant_class(box.get("class_name"))
        for box in detections
    }
    groups: list[tuple[str, set[str]]] = []
    consumed: set[str] = set()
    if participant_classes & actor_classes:
        groups.append(("actor", actor_classes))
        consumed.update(actor_classes)
    if event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION:
        # A manipulation frame is only explanatory when it shows the operated
        # tool and its nearby vessel.  The actor is optional in the ontology but
        # is a useful visible participant whenever the detector sees it.
        if detected_classes & actor_classes:
            groups.append(("actor", actor_classes))
            consumed.update(actor_classes)
        participant_tool_classes = participant_classes & tool_classes
        participant_vessel_classes = participant_classes & vessel_classes
        if participant_tool_classes:
            groups.append(("tool", participant_tool_classes))
            consumed.update(participant_tool_classes)
        if participant_vessel_classes:
            # ``container`` is a semantic slot rather than a trained visual
            # class, and the two sample-bottle labels are detector aliases.
            # Broaden only those explicitly requested slots; never add every
            # bench vessel merely because the action is a pipette operation.
            vessel_slot_classes = set(participant_vessel_classes)
            if "container" in participant_vessel_classes:
                vessel_slot_classes.update(vessel_classes)
            if participant_vessel_classes & {
                "sample_bottle",
                "sample_bottle_blue",
            }:
                vessel_slot_classes.update({"sample_bottle", "sample_bottle_blue"})
            groups.append(("vessel", vessel_slot_classes))
            consumed.update(vessel_slot_classes)
    elif event.action_type == ActionType.CONTAINER_STATE_CHANGE:
        # Semantic review can correctly identify the physical role (the
        # bottle being opened) even when the detector uses a neighbouring
        # bottle class.  Treat those detector labels as one *container slot*,
        # then choose the single instance closest to the manipulating hand.
        # This preserves participant-only rendering without boxing every
        # bottle on a crowded bench.
        closure_classes: set[str] = set()
        container_classes: set[str] = set()
        if "bottle_cap" in participant_classes:
            closure_classes.add("bottle_cap")
            container_classes.update(
                {
                    "container",
                    "sample_bottle",
                    "sample_bottle_blue",
                    "reagent_bottle",
                }
            )
        if "tube_cap" in participant_classes:
            closure_classes.add("tube_cap")
            container_classes.update({"container", "tube"})
        if participant_classes & vessel_classes and not container_classes:
            container_classes.update(participant_classes & vessel_classes)
        if closure_classes:
            groups.append(("closure", closure_classes))
        if container_classes:
            groups.append(("container", container_classes))
        consumed.update(closure_classes | container_classes)
    elif participant_classes & tool_classes:
        groups.append(("tool", participant_classes & tool_classes))
        consumed.update(tool_classes)
    for class_name in sorted(participant_classes - consumed):
        groups.append((class_name, {class_name}))

    # Avoid duplicate semantic slots when the action-specific branch already
    # inserted an actor group.
    unique_groups: list[tuple[str, set[str]]] = []
    seen_group_names: set[str] = set()
    for name, classes in groups:
        if name in seen_group_names:
            continue
        seen_group_names.add(name)
        unique_groups.append((name, classes))
    groups = unique_groups
    boxes_by_group = [
        [
            dict(box)
            for box in detections
            if services.normalize_participant_class(box.get("class_name")) in classes
        ]
        for _, classes in groups
    ]
    relation_rejected_count = 0
    relation_gate_applied = False
    boxes_by_name = {
        group[0]: boxes for group, boxes in zip(groups, boxes_by_group, strict=True)
    }
    actor_box_pool = boxes_by_name.get("actor") or []
    actor_required_actions = {
        ActionType.HAND_OBJECT_CONTACT,
        ActionType.CONTAINER_STATE_CHANGE,
        ActionType.DEVICE_PANEL_OPERATION,
        ActionType.PIPETTE_TRANSFER_OPERATION,
    }

    def adjacent_to_any(
        box: dict[str, Any], references: Sequence[dict[str, Any]]
    ) -> bool:
        return bool(references) and min(
            services._box_edge_gap_norm(box, reference) for reference in references
        ) <= float(maximum_interaction_gap_norm)

    if event.action_type in actor_required_actions:
        relation_gate_applied = True
        for group_name, boxes in list(boxes_by_name.items()):
            if group_name == "actor":
                continue
            references = actor_box_pool
            if (
                event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION
                and group_name == "vessel"
            ):
                references = boxes_by_name.get("tool") or []
            eligible = [box for box in boxes if adjacent_to_any(box, references)]
            relation_rejected_count += len(boxes) - len(eligible)
            boxes_by_name[group_name] = eligible
    elif actor_box_pool:
        # For movement classes an actor may be occluded, but when it is visible
        # a distant static instance cannot be presented as the manipulated one.
        relation_gate_applied = True
        for group_name, boxes in list(boxes_by_name.items()):
            if group_name == "actor":
                continue
            eligible = [box for box in boxes if adjacent_to_any(box, actor_box_pool)]
            relation_rejected_count += len(boxes) - len(eligible)
            boxes_by_name[group_name] = eligible
    boxes_by_group = [boxes_by_name[group[0]] for group in groups]
    available = [
        (group, boxes)
        for group, boxes in zip(groups, boxes_by_group, strict=True)
        if boxes
    ]

    candidate_track_ids: set[int] = set()
    if view_id is not None:
        for candidate in event.candidates:
            if candidate.view_id != view_id:
                continue
            for evidence in candidate.evidence:
                for key, value in evidence.items():
                    if not key.endswith("track_id") or value is None:
                        continue
                    try:
                        candidate_track_ids.add(int(value))
                    except (TypeError, ValueError):
                        continue

    def center(box: dict[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(item) for item in box.get("xyxy_norm") or (0, 0, 0, 0))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        lx, ly = center(left)
        rx, ry = center(right)
        return math.hypot(lx - rx, ly - ry)

    best_choice: tuple[dict[str, Any], ...] = ()
    best_score = float("-inf")
    if available:
        for choice in product(*(boxes for _, boxes in available)):
            labels = [group[0] for group, _ in available]
            by_label = dict(zip(labels, choice, strict=True))
            score = sum(float(box.get("confidence") or 0.0) for box in choice)
            score += sum(
                min(float(box.get("roi_motion") or 0.0), 100.0) / 100.0
                for box in choice
            )
            score += 1.5 * sum(
                int(box.get("track_id")) in candidate_track_ids
                for box in choice
                if box.get("track_id") is not None
            )
            specific_proximity_applied = False
            if "actor" in by_label and "tool" in by_label:
                score -= 4.0 * distance(by_label["actor"], by_label["tool"])
                specific_proximity_applied = True
            if "tool" in by_label and "vessel" in by_label:
                score -= 4.0 * distance(by_label["tool"], by_label["vessel"])
                specific_proximity_applied = True
            if "actor" in by_label and "container" in by_label:
                score -= 5.0 * distance(by_label["actor"], by_label["container"])
                specific_proximity_applied = True
            if "actor" in by_label and "closure" in by_label:
                score -= 5.0 * distance(by_label["actor"], by_label["closure"])
                specific_proximity_applied = True
            if "closure" in by_label and "container" in by_label:
                score -= 3.0 * distance(by_label["closure"], by_label["container"])
                specific_proximity_applied = True
            if (
                not specific_proximity_applied
                and "actor" in by_label
                and len(choice) > 1
            ):
                score -= 4.0 * min(
                    distance(by_label["actor"], box)
                    for label, box in by_label.items()
                    if label != "actor"
                )
            elif not specific_proximity_applied and len(choice) > 1:
                score -= 2.0 * min(
                    distance(left, right)
                    for index, left in enumerate(choice)
                    for right in choice[index + 1 :]
                )
            if score > best_score:
                best_score = score
                best_choice = tuple(dict(box) for box in choice)
    rendered = list(best_choice)
    effective_participant_classes = participant_classes | {
        services.normalize_participant_class(box.get("class_name")) for box in rendered
    }
    rendered_classes = sorted(
        {
            str(box.get("class_name") or "")
            for box in rendered
            if str(box.get("class_name") or "").strip()
        }
    )
    suppressed_classes = sorted(
        {
            str(box.get("class_name") or "")
            for box in detections
            if box not in rendered and str(box.get("class_name") or "").strip()
        }
    )
    same_class_suppressed = sum(
        services.normalize_participant_class(box.get("class_name"))
        in effective_participant_classes
        and box not in rendered
        for box in detections
    )
    return rendered, {
        "mode": "event_participants_only",
        "instance_policy": "single_interacting_instance_per_semantic_slot",
        "participant_classes": sorted(effective_participant_classes),
        "participant_class_groups": [
            {"name": name, "classes": sorted(classes)} for name, classes in groups
        ],
        "detected_box_count": len(detections),
        "rendered_box_count": len(rendered),
        "suppressed_background_box_count": len(detections) - len(rendered),
        "suppressed_same_class_instance_count": same_class_suppressed,
        "rendered_classes": rendered_classes,
        "rendered_detections": [
            {
                "class_name": str(box.get("class_name") or ""),
                "confidence": round(float(box.get("confidence") or 0.0), 6),
                "detector_source": str(
                    box.get("detector_source") or "closed_set_yolo_tensorrt"
                ),
                "track_id": box.get("track_id"),
            }
            for box in rendered
        ],
        "minimum_rendered_confidence": (
            round(
                min(float(box.get("confidence") or 0.0) for box in rendered),
                6,
            )
            if rendered
            else None
        ),
        "rendered_track_ids": [
            box.get("track_id") for box in rendered if box.get("track_id") is not None
        ],
        "suppressed_background_classes": suppressed_classes,
        "extraneous_rendered_classes": sorted(
            set(rendered_classes) - effective_participant_classes
        ),
        "instance_selection_score": (
            round(best_score, 6) if math.isfinite(best_score) else None
        ),
        "interaction_relation_gate": {
            "applied": relation_gate_applied,
            "maximum_edge_gap_norm": float(maximum_interaction_gap_norm),
            "rejected_box_count": relation_rejected_count,
            "rule": (
                "actor_to_object; pipette_tool_to_vessel"
                if event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION
                else "actor_to_manipulated_object"
            ),
        },
    }


def _event_key_frame_score(
    event: EvidenceEvent,
    frame: FrameEvidence,
    *,
    services: ParticipantGroundingServices,
) -> tuple[float, dict[str, Any]]:
    detections = [box.model_dump() for box in frame.detections]
    rendered, receipt = services._event_participant_boxes(
        event, detections, view_id=frame.view_id
    )
    group_names = {
        item["name"]
        for item in receipt.get("participant_class_groups") or []
        if any(
            str(box.get("class_name") or "") in set(item.get("classes") or [])
            for box in rendered
        )
    }
    score = 100.0 * len(group_names)
    if event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION:
        score += 100.0 * int({"tool", "vessel"}.issubset(group_names))
        score += 40.0 * int("actor" in group_names)
    key_frame_phase_bias = 0.0
    if event.action_type == ActionType.CONTAINER_STATE_CHANGE:
        # Prefer an equally participant-rich frame that shows the resulting
        # open/closed state while the operator or closure is still nearby.
        # Participant coverage remains dominant; this bounded term is only a
        # temporal tie-breaker against pre-contact detector confidence.
        span_ms = max(1.0, event.global_end_ms - event.global_start_ms)
        progress = min(
            1.0,
            max(0.0, (frame.global_ms - event.global_start_ms) / span_ms),
        )
        key_frame_phase_bias = 4.0 * (1.0 - abs(progress - 0.80))
        score += key_frame_phase_bias
    score += float(receipt.get("instance_selection_score") or 0.0)
    score += min(float(frame.motion_score or 0.0), 100.0) / 100.0
    receipt["key_frame_phase_bias"] = round(key_frame_phase_bias, 6)
    return score, receipt


def _shared_candidate_time(
    event: EvidenceEvent, transforms: dict[str, AlignmentTransform], timestamp_ms: float
) -> bool:
    roles = {
        candidate.role
        for candidate in event.candidates
        if candidate.view_id in event.supporting_views
        and candidate.view_id in transforms
        and candidate.global_start_ms - transforms[candidate.view_id].uncertainty_ms
        <= timestamp_ms
        <= candidate.global_end_ms + transforms[candidate.view_id].uncertainty_ms
    }
    return {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}.issubset(roles)


def _best_event_frames_many(
    path: Path,
    events: Sequence[EvidenceEvent],
    transforms: dict[str, AlignmentTransform] | None = None,
    *,
    services: ParticipantGroundingServices,
) -> dict[str, tuple[FrameEvidence, float, dict[str, Any]] | None]:
    """Select participant-rich frames for all events in one ledger pass."""

    accepted = [event for event in events if event.accepted]
    results: dict[str, tuple[FrameEvidence, float, dict[str, Any]] | None] = {
        event.event_id: None for event in accepted
    }
    # Some callers intentionally materialize media without a persisted
    # detection ledger (for example, compatibility tests and manually supplied
    # events).  In that case retain each event's existing key timestamp; the
    # normal nearest-frame lookup below remains the single fallback source of
    # annotation data.  Production runs always provide immutable ledgers.
    if not path.is_file():
        return results
    for frame in services.iter_frame_evidence(path):
        if frame.global_ms is None:
            continue
        timestamp = float(frame.global_ms)
        for event in accepted:
            if not event.global_start_ms <= timestamp <= event.global_end_ms:
                continue
            score, receipt = services._event_key_frame_score(event, frame)
            shared_time = services._shared_candidate_time(
                event, transforms or {}, timestamp
            )
            receipt["shared_candidate_time"] = shared_time
            previous = results[event.event_id]
            if previous is None or (shared_time, score) > (
                previous[2].get("shared_candidate_time", False),
                previous[1],
            ):
                results[event.event_id] = (frame, score, receipt)
    return results


def _bounded_grounding_dino_temporal_rescue(
    event: EvidenceEvent,
    group: ExperimentGroup,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    *,
    services: ParticipantGroundingServices,
) -> tuple[float | None, dict[str, Any]]:
    """Find an explanatory participant frame without repeating a video scan.

    The immutable fine ledgers provide candidate timestamps and actor boxes.
    At most a configured number of real source frames per eligible view are
    decoded in place.  Grounding DINO is then asked only for the missing
    semantic participant classes.  No Ark request or source copy is possible
    in this helper.
    """

    settings = config.get("models", {}).get("open_vocabulary_key_frame") or {}
    fallback = dict(settings.get("grounding_dino_fallback") or {})
    # Temporal rescue runs separately from the final annotation model stack.
    # A small GPU can accelerate this phase without keeping DINO resident
    # alongside YOLO-World and SAM2 during final annotation.
    if fallback.get("temporal_rescue_device"):
        fallback["device"] = fallback["temporal_rescue_device"]
        settings = {**settings, "grounding_dino_fallback": fallback}
    if not (
        settings.get("enabled")
        and fallback.get("enabled")
        and fallback.get("temporal_rescue_enabled")
    ):
        return None, {"status": "disabled"}
    if str((event.model_understanding or {}).get("status") or "") != "completed":
        return None, {"status": "deferred_until_semantic_curation"}

    actor_classes = {"hand", "gloved_hand"}
    prompt_classes = {
        services.normalize_participant_class(canonical)
        for canonical in dict(fallback.get("prompt_map") or {}).values()
    }
    participant_classes = {
        services.normalize_participant_class(item)
        for item in event.objects
        if str(item).strip()
    }
    active_actor_classes = participant_classes & actor_classes or actor_classes
    target_classes = (participant_classes - actor_classes) & prompt_classes
    if not target_classes:
        return None, {
            "status": "not_applicable",
            "participant_classes": sorted(participant_classes),
        }

    by_view = {view.view_id: view for view in views}
    direct_views = {
        str(item)
        for item in (event.semantic_review or {}).get("directly_supported_view_ids", [])
    }
    preferred_views = [
        view_id
        for view_id in (group.first_person_view, group.third_person_view)
        if view_id in detection_paths
        and view_id in by_view
        and view_id in infos
        and view_id in transforms
    ]
    direct_preferred = [
        view_id for view_id in preferred_views if view_id in direct_views
    ]
    # Model-declared direct support is recorded, not used as an exclusion.
    # A conservative semantic review can mark a visually clearer role
    # uncertain; bounded geometry must still inspect both real role views.
    eligible_views = preferred_views
    maximum_frames = max(
        1, int(fallback.get("temporal_rescue_max_frames_per_view", 12))
    )
    interval_ms = max(
        1.0, float(fallback.get("temporal_rescue_sample_interval_ms", 500.0))
    )
    maximum_gap = float(fallback.get("temporal_rescue_max_actor_gap_norm", 0.02))
    pipette_aspect = float(settings.get("pipette_minimum_aspect_ratio", 1.7))
    dino_pipette_aspect = float(fallback.get("pipette_minimum_aspect_ratio", 1.4))
    minimum_class_confidence = {
        services.normalize_participant_class(class_name): float(limit)
        for class_name, limit in dict(
            settings.get("minimum_manipulated_object_confidence") or {}
        ).items()
    }
    pipette_center_must_overlap_actor = bool(
        settings.get("pipette_center_must_overlap_actor", False)
    )
    pipette_maximum_actor_iou = float(settings.get("pipette_maximum_actor_iou", 0.65))
    closure_maximum_gap = float(settings.get("closure_max_actor_gap_norm", 0.01))
    closure_minimum_confidence = float(settings.get("closure_minimum_confidence", 0.12))
    grounded_actor_maximum_area = float(
        settings.get("grounded_actor_maximum_box_area_norm", 0.15)
    )
    grounded_actor_confidence_ratio = float(
        settings.get("grounded_actor_confidence_ratio", 0.75)
    )
    preferred_grounding_terms = services._preferred_grounding_terms_by_class(event)
    inspected: list[dict[str, Any]] = []
    winners: list[tuple[tuple[float, ...], float, str, dict[str, Any]]] = []
    model_load_seconds = 0.0
    inference_seconds = 0.0

    for view_id in eligible_views:
        candidate_frames = [
            frame
            for frame in services.iter_frame_evidence(detection_paths[view_id])
            if frame.global_ms is not None
            and event.global_start_ms <= float(frame.global_ms) <= event.global_end_ms
        ]
        if not candidate_frames:
            inspected.append({"view_id": view_id, "status": "no_fine_ledger_frames"})
            continue
        desired_count = min(
            maximum_frames,
            max(
                2,
                int(
                    math.ceil(
                        max(0.0, event.global_end_ms - event.global_start_ms)
                        / interval_ms
                    )
                )
                + 1,
            ),
            len(candidate_frames),
        )
        if desired_count == 1:
            sampled_frames = [candidate_frames[0]]
        else:
            indices = {
                round(index * (len(candidate_frames) - 1) / (desired_count - 1))
                for index in range(desired_count)
            }
            sampled_frames = [candidate_frames[index] for index in sorted(indices)]
        current_frame = min(
            candidate_frames,
            key=lambda item: abs(float(item.global_ms) - float(event.key_global_ms)),
        )
        if current_frame not in sampled_frames:
            sampled_frames.append(current_frame)
            sampled_frames.sort(key=lambda item: float(item.global_ms))

        reader = services.ViewFrameReader(max_open=1)
        try:
            for candidate in sampled_frames:
                global_ms = float(candidate.global_ms)
                dual_role_coverage = all(
                    0.0
                    <= float(transforms[pair_view].to_local(global_ms))
                    <= float(infos[pair_view].duration_ms)
                    for pair_view in preferred_views
                )
                if not dual_role_coverage:
                    inspected.append(
                        {
                            "view_id": view_id,
                            "global_ms": global_ms,
                            "status": "rejected_no_real_dual_role_coverage",
                        }
                    )
                    continue
                local_ms = float(transforms[view_id].to_local(global_ms))
                if not 0.0 <= local_ms <= float(infos[view_id].duration_ms):
                    continue
                frame = reader.read(by_view[view_id], infos[view_id], local_ms)
                if frame is None:
                    inspected.append(
                        {
                            "view_id": view_id,
                            "global_ms": global_ms,
                            "status": "decode_failed",
                        }
                    )
                    continue
                closed_actor_boxes = [
                    box.model_dump()
                    for box in candidate.detections
                    if services.normalize_participant_class(box.class_name)
                    in actor_classes
                ]
                requested_classes = set(target_classes)
                if not closed_actor_boxes:
                    requested_classes.update(active_actor_classes)
                grounded, grounding_receipt = (
                    services._grounding_dino_key_frame_detections(
                        frame, requested_classes, settings
                    )
                )
                model_load_seconds += float(
                    grounding_receipt.get("model_load_seconds") or 0.0
                )
                inference_seconds += float(
                    grounding_receipt.get("inference_seconds") or 0.0
                )
                raw_grounded_actor_boxes = [
                    box
                    for box in grounded
                    if services.normalize_participant_class(box.get("class_name"))
                    in active_actor_classes
                ]
                actor_boxes = (
                    closed_actor_boxes
                    or services._filter_grounded_actor_boxes(
                        raw_grounded_actor_boxes,
                        maximum_area_norm=grounded_actor_maximum_area,
                        confidence_ratio=grounded_actor_confidence_ratio,
                    )
                )
                admitted_objects: list[dict[str, Any]] = []
                selection_receipts: dict[str, Any] = {}
                for canonical_class in sorted(target_classes):
                    class_maximum_gap = (
                        closure_maximum_gap
                        if canonical_class in {"bottle_cap", "tube_cap"}
                        else maximum_gap
                    )
                    class_minimum_confidence = max(
                        float(minimum_class_confidence.get(canonical_class, 0.0)),
                        (
                            closure_minimum_confidence
                            if canonical_class in {"bottle_cap", "tube_cap"}
                            else 0.0
                        ),
                    )
                    selected, selection_receipt = (
                        services._select_manipulated_object_candidate(
                            [
                                box
                                for box in grounded
                                if services.normalize_participant_class(
                                    box.get("class_name")
                                )
                                == canonical_class
                            ],
                            actor_boxes,
                            canonical_class=canonical_class,
                            maximum_actor_gap=class_maximum_gap,
                            pipette_minimum_aspect_ratio=pipette_aspect,
                            grounding_dino_pipette_minimum_aspect_ratio=(
                                dino_pipette_aspect
                            ),
                            minimum_confidence=class_minimum_confidence,
                            pipette_center_must_overlap_actor=(
                                pipette_center_must_overlap_actor
                            ),
                            pipette_maximum_actor_iou=(pipette_maximum_actor_iou),
                            preferred_grounding_terms=(
                                preferred_grounding_terms.get(canonical_class, ())
                            ),
                        )
                    )
                    selection_receipts[canonical_class] = selection_receipt
                    if selected is not None:
                        admitted_objects.append(selected)
                rendered, render_receipt = services._event_participant_boxes(
                    event,
                    [*actor_boxes, *admitted_objects],
                    view_id=view_id,
                    maximum_interaction_gap_norm=maximum_gap,
                )
                rendered_actor_boxes = [
                    box
                    for box in rendered
                    if services.normalize_participant_class(box.get("class_name"))
                    in actor_classes
                ]
                rendered_object_boxes = [
                    box
                    for box in rendered
                    if services.normalize_participant_class(box.get("class_name"))
                    not in actor_classes
                ]
                selected_object_classes = {
                    services.normalize_participant_class(box.get("class_name"))
                    for box in rendered_object_boxes
                }
                pair_visible = bool(
                    rendered_actor_boxes
                    and target_classes.issubset(selected_object_classes)
                )
                best_gap = (
                    min(
                        services._box_edge_gap_norm(obj, actor)
                        for obj in rendered_object_boxes
                        for actor in rendered_actor_boxes
                    )
                    if pair_visible
                    else None
                )
                best_confidence = max(
                    (
                        float(box.get("confidence") or 0.0)
                        for box in rendered_object_boxes
                    ),
                    default=0.0,
                )
                record = {
                    "view_id": view_id,
                    "global_ms": global_ms,
                    "status": "eligible" if pair_visible else "rejected",
                    "rendered_classes": render_receipt.get("rendered_classes"),
                    "selected_object_classes": sorted(selected_object_classes),
                    "best_object_confidence": round(best_confidence, 6),
                    "best_actor_gap_norm": (
                        round(float(best_gap), 6) if best_gap is not None else None
                    ),
                    "grounding": {
                        key: value
                        for key, value in grounding_receipt.items()
                        if key
                        in {
                            "status",
                            "raw_detection_count",
                            "admitted_area_bounded_count",
                            "rejected_box_area_count",
                        }
                    },
                    "grounded_actor_filter": {
                        "raw_candidate_count": len(raw_grounded_actor_boxes),
                        "admitted_candidate_count": (
                            len(actor_boxes) if not closed_actor_boxes else 0
                        ),
                        "closed_set_actor_used": bool(closed_actor_boxes),
                        "maximum_box_area_norm": grounded_actor_maximum_area,
                        "confidence_ratio": grounded_actor_confidence_ratio,
                    },
                    "selection": selection_receipts,
                }
                inspected.append(record)
                if pair_visible:
                    temporal_phase_score = 0.0
                    if event.action_type in {
                        ActionType.CONTAINER_STATE_CHANGE,
                        ActionType.HAND_OBJECT_CONTACT,
                    }:
                        span_ms = max(
                            1.0,
                            event.global_end_ms - event.global_start_ms,
                        )
                        progress = min(
                            1.0,
                            max(
                                0.0,
                                (global_ms - event.global_start_ms) / span_ms,
                            ),
                        )
                        target_progress = (
                            0.80
                            if event.action_type == ActionType.CONTAINER_STATE_CHANGE
                            else 0.50
                        )
                        temporal_phase_score = 1.0 - abs(progress - target_progress)
                        record["temporal_phase_score"] = round(temporal_phase_score, 6)
                    score = (
                        float(len(record["selected_object_classes"])),
                        temporal_phase_score,
                        -float(best_gap or 0.0),
                        best_confidence,
                        float(candidate.motion_score or 0.0),
                        -abs(global_ms - float(event.key_global_ms)),
                    )
                    winners.append((score, global_ms, view_id, record))
        finally:
            reader.close()

    winner = max(winners, key=lambda item: item[0], default=None)
    return (
        float(winner[1]) if winner is not None else None,
        {
            "schema_version": ("visioncortex-grounding-dino-temporal-rescue/1"),
            "status": "selected" if winner is not None else "no_eligible_frame",
            "scope": "bounded immutable-ledger candidate frames",
            "full_scan_repeated": False,
            "source_copy_bytes": 0,
            "ark_calls": 0,
            "token_usage": 0,
            "target_classes": sorted(target_classes),
            "eligible_views": eligible_views,
            "direct_support_views": direct_preferred,
            "direct_support_used_as_exclusion": False,
            "sample_interval_ms": interval_ms,
            "maximum_frames_per_view": maximum_frames,
            "inspected_frame_count": len(inspected),
            "eligible_frame_count": len(winners),
            "model_load_seconds": round(model_load_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "selected_view_id": winner[2] if winner is not None else None,
            "selected_global_ms": winner[1] if winner is not None else None,
            "selected_receipt": winner[3] if winner is not None else None,
            "inspected": inspected,
        },
    )


@serialized_open_vocabulary
def _release_auxiliary_model_caches(
    *, retain_on_cpu: bool = False, services: ParticipantGroundingServices
) -> dict[str, int]:
    """Release VRAM between models; optionally retain weights in host RAM."""

    open_vocabulary = len(services._OPEN_VOCABULARY_MODEL_CACHE)
    grounding_dino = len(services._GROUNDING_DINO_MODEL_CACHE)
    for cached in [
        *services._OPEN_VOCABULARY_MODEL_CACHE.values(),
        *services._GROUNDING_DINO_MODEL_CACHE.values(),
    ]:
        model = cached.get("model") if isinstance(cached, dict) else None
        if model is not None and hasattr(model, "to"):
            try:
                model.to("cpu")
                if hasattr(model, "predictor"):
                    model.predictor = None
                cached["device"] = "cpu"
            except (RuntimeError, TypeError, ValueError):
                pass
    if not retain_on_cpu:
        services._OPEN_VOCABULARY_MODEL_CACHE.clear()
        services._GROUNDING_DINO_MODEL_CACHE.clear()
    temporal = services.release_temporal_segmentation_model_cache(
        retain_on_cpu=retain_on_cpu
    )
    liquid = services.release_liquid_semantic_model_cache(retain_on_cpu=retain_on_cpu)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {
        "open_vocabulary": open_vocabulary,
        "grounding_dino": grounding_dino,
        "temporal_segmentation": temporal,
        "liquid_semantic": liquid,
    }


def _park_auxiliary_model_caches(
    config: dict[str, Any] | None, *, services: ParticipantGroundingServices
) -> dict[str, Any]:
    """Keep warm weights only while the configured host-memory reserve exists."""
    import psutil

    minimum_gib = float(
        ((config or {}).get("performance") or {}).get(
            "auxiliary_cpu_cache_min_available_gib", 0
        )
    )
    available = int(psutil.virtual_memory().available)
    retain = minimum_gib > 0 and available >= minimum_gib * 1024**3
    started = time.perf_counter()
    counts = services._release_auxiliary_model_caches(retain_on_cpu=retain)
    return {
        "retained_on_cpu": retain,
        "available_host_bytes": available,
        "models": counts,
        "seconds": round(time.perf_counter() - started, 6),
    }


def _box_edge_gap_norm(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return normalized rectangle edge distance (zero for touch/overlap)."""

    lx1, ly1, lx2, ly2 = (float(item) for item in left.get("xyxy_norm") or (0, 0, 0, 0))
    rx1, ry1, rx2, ry2 = (
        float(item) for item in right.get("xyxy_norm") or (0, 0, 0, 0)
    )
    dx = max(lx1 - rx2, rx1 - lx2, 0.0)
    dy = max(ly1 - ry2, ry1 - ly2, 0.0)
    return math.hypot(dx, dy)


def _box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx1, ly1, lx2, ly2 = (float(item) for item in left.get("xyxy_norm") or (0, 0, 0, 0))
    rx1, ry1, rx2, ry2 = (
        float(item) for item in right.get("xyxy_norm") or (0, 0, 0, 0)
    )
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_aspect_ratio(box: dict[str, Any]) -> float:
    """Return the orientation-independent aspect ratio of one box."""

    x1, y1, x2, y2 = (float(item) for item in box.get("xyxy_norm") or (0, 0, 0, 0))
    # Normalized x/y coordinates are not metrically comparable on a 16:9
    # image.  Grounded boxes retain the source aspect ratio so a broad balance
    # cannot pass as an elongated pipette merely because normalized height is
    # measured against fewer pixels.
    image_aspect_ratio = float(box.get("image_aspect_ratio") or 1.0)
    width = max(0.0, x2 - x1) * image_aspect_ratio
    height = max(0.0, y2 - y1)
    shorter = min(width, height)
    return max(width, height) / shorter if shorter > 0.0 else 0.0


def _filter_grounded_actor_boxes(
    boxes: Sequence[dict[str, Any]],
    *,
    maximum_area_norm: float,
    confidence_ratio: float,
) -> list[dict[str, Any]]:
    """Reject oversized/weak open-vocabulary hand unions fail-closed."""

    area_eligible = []
    for box in boxes:
        x1, y1, x2, y2 = (float(item) for item in box.get("xyxy_norm") or (0, 0, 0, 0))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= maximum_area_norm:
            area_eligible.append(box)
    if not area_eligible:
        return []
    best_confidence = max(float(box.get("confidence") or 0.0) for box in area_eligible)
    threshold = best_confidence * confidence_ratio
    return [
        box for box in area_eligible if float(box.get("confidence") or 0.0) >= threshold
    ]


def _preferred_grounding_terms_by_class(
    event: EvidenceEvent,
) -> dict[str, tuple[str, ...]]:
    """Extract only explicit participant appearance constraints from Ark text."""

    understanding = event.model_understanding or {}
    physical_change = understanding.get("physical_change") or {}
    semantic_text = " ".join(
        [
            str(understanding.get("current_step") or ""),
            " ".join(map(str, understanding.get("objects") or [])),
            " ".join(
                str(item.get("object") or "")
                for item in understanding.get("hand_object_interactions") or []
                if isinstance(item, dict)
            ),
            str(physical_change.get("before") or ""),
            str(physical_change.get("after") or ""),
        ]
    ).lower()
    preferred: dict[str, tuple[str, ...]] = {}
    if any(term in semantic_text for term in ("brown", "amber", "棕", "琥珀")):
        preferred["reagent_bottle"] = ("brown", "amber")
    # Closure colour is used only for the closure class. This avoids treating
    # the routinely blue glove as evidence for a blue bottle or cap.
    closure_colour = (
        "red"
        if "red" in semantic_text or "红" in semantic_text
        else "orange"
        if "orange" in semantic_text or "橙" in semantic_text
        else None
    )
    if closure_colour is not None:
        preferred["bottle_cap"] = (closure_colour,)
        preferred["tube_cap"] = (closure_colour,)
    return preferred


def _select_manipulated_object_candidate(
    candidates: Sequence[dict[str, Any]],
    actor_boxes: Sequence[dict[str, Any]],
    *,
    canonical_class: str,
    maximum_actor_gap: float,
    pipette_minimum_aspect_ratio: float,
    grounding_dino_pipette_minimum_aspect_ratio: float | None = None,
    minimum_confidence: float = 0.0,
    pipette_center_must_overlap_actor: bool = False,
    pipette_maximum_actor_iou: float | None = None,
    preferred_grounding_terms: Sequence[str] = (),
    services: ParticipantGroundingServices,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Choose one active object instance using contact before confidence.

    Crowded laboratory benches can contain many high-confidence static tools.
    A final participant annotation is allowed to show only the instance that
    touches or nearly touches the operator. Pipettes and tips additionally
    need an elongated box so a hand-shaped open-vocabulary false positive
    cannot satisfy the interaction gate.
    """

    shape_rejected = 0
    contact_rejected = 0
    confidence_rejected = 0
    centre_rejected = 0
    actor_overlap_rejected = 0
    eligible: list[tuple[float, dict[str, Any]]] = []
    elongated_classes = {"pipette", "spearhead"}
    for candidate in candidates:
        if float(candidate.get("confidence") or 0.0) < float(minimum_confidence):
            confidence_rejected += 1
            continue
        candidate_minimum_aspect_ratio = pipette_minimum_aspect_ratio
        if (
            str(candidate.get("detector_source") or "").startswith("grounding_dino")
            and grounding_dino_pipette_minimum_aspect_ratio is not None
        ):
            candidate_minimum_aspect_ratio = float(
                grounding_dino_pipette_minimum_aspect_ratio
            )
        if (
            canonical_class in elongated_classes
            and services._box_aspect_ratio(candidate) < candidate_minimum_aspect_ratio
        ):
            shape_rejected += 1
            continue
        relaxed_dino_shape_requires_centre = bool(
            canonical_class in elongated_classes
            and str(candidate.get("detector_source") or "").startswith("grounding_dino")
            and services._box_aspect_ratio(candidate) < pipette_minimum_aspect_ratio
        )
        if canonical_class in elongated_classes and (
            pipette_center_must_overlap_actor or relaxed_dino_shape_requires_centre
        ):
            x1, y1, x2, y2 = (
                float(item) for item in candidate.get("xyxy_norm") or (0, 0, 0, 0)
            )
            centre_x = (x1 + x2) / 2.0
            centre_y = (y1 + y2) / 2.0
            if not any(
                float((actor.get("xyxy_norm") or (0, 0, 0, 0))[0])
                <= centre_x
                <= float((actor.get("xyxy_norm") or (0, 0, 0, 0))[2])
                and float((actor.get("xyxy_norm") or (0, 0, 0, 0))[1])
                <= centre_y
                <= float((actor.get("xyxy_norm") or (0, 0, 0, 0))[3])
                for actor in actor_boxes
            ):
                centre_rejected += 1
                continue
        if (
            canonical_class in elongated_classes
            and pipette_maximum_actor_iou is not None
            and actor_boxes
            and max(services._box_iou(candidate, actor) for actor in actor_boxes)
            > float(pipette_maximum_actor_iou)
        ):
            actor_overlap_rejected += 1
            continue
        actor_gap = (
            min(services._box_edge_gap_norm(candidate, actor) for actor in actor_boxes)
            if actor_boxes
            else float("inf")
        )
        if actor_gap > maximum_actor_gap:
            contact_rejected += 1
            continue
        eligible.append((actor_gap, candidate))

    preferred_terms = tuple(
        str(item).strip().lower()
        for item in preferred_grounding_terms
        if str(item).strip()
    )
    preferred_eligible = [
        item
        for item in eligible
        if any(
            term in str(item[1].get("grounding_prompt") or "").lower()
            for term in preferred_terms
        )
    ]
    ranked_eligible = preferred_eligible or eligible
    selected_pair = min(
        ranked_eligible,
        key=lambda item: (
            item[0],
            -float(item[1].get("confidence") or 0.0),
        ),
        default=None,
    )
    selected = selected_pair[1] if selected_pair is not None else None
    return selected, {
        "rule": (
            "explicit_semantic_appearance_then_minimum_actor_edge_gap_then_confidence"
            if preferred_eligible
            else "minimum_actor_edge_gap_then_confidence"
        ),
        "canonical_class": canonical_class,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "shape_rejected_count": shape_rejected,
        "contact_rejected_count": contact_rejected,
        "confidence_rejected_count": confidence_rejected,
        "centre_rejected_count": centre_rejected,
        "actor_overlap_rejected_count": actor_overlap_rejected,
        "minimum_confidence": float(minimum_confidence),
        "preferred_grounding_terms": list(preferred_terms),
        "preferred_eligible_candidate_count": len(preferred_eligible),
        "selected_grounding_prompt": (
            str(selected.get("grounding_prompt") or "")
            if selected is not None
            else None
        ),
        "pipette_center_must_overlap_actor": bool(
            pipette_center_must_overlap_actor
            if canonical_class in elongated_classes
            else False
        ),
        "relaxed_grounding_dino_shape_requires_actor_center_overlap": True,
        "pipette_maximum_actor_iou": (
            float(pipette_maximum_actor_iou)
            if canonical_class in elongated_classes
            and pipette_maximum_actor_iou is not None
            else None
        ),
        "maximum_actor_gap_norm": maximum_actor_gap,
        "minimum_aspect_ratio": (
            pipette_minimum_aspect_ratio
            if canonical_class in elongated_classes
            else None
        ),
        "grounding_dino_minimum_aspect_ratio": (
            grounding_dino_pipette_minimum_aspect_ratio
            if canonical_class in elongated_classes
            else None
        ),
        "selected_actor_gap_norm": (
            round(float(selected_pair[0]), 6) if selected_pair is not None else None
        ),
        "fail_closed": selected is None,
    }


def _select_state_container_candidate(
    candidates: Sequence[dict[str, Any]],
    actor_boxes: Sequence[dict[str, Any]],
    *,
    view_role: str,
    state_direction: str,
    maximum_actor_gap: float,
    first_person_minimum_top_offset: float = 0.0,
    closure_boxes: Sequence[dict[str, Any]] = (),
    maximum_closure_gap: float | None = None,
    services: ParticipantGroundingServices,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select a state-bearing container without preferring background vessels.

    In a first-person post-opening frame the manipulated vessel starts at or
    below the operating hand's vertical centre.  Crowded benches often contain
    higher, unrelated open vessels that score much better in open-vocabulary
    detection.  Treat the spatial relation as a fail-closed eligibility rule,
    then rank only eligible candidates by confidence.
    """

    eligible: list[dict[str, Any]] = []
    spatial_rule = (
        view_role == "First-Person"
        and state_direction == "opening"
        and bool(actor_boxes)
    )
    # After an opening transition the closure is normally already in the
    # operator's hand and can be spatially separated from the bottle mouth.
    # Requiring closure-to-container proximity at that phase silently removes
    # the exact participant that proves the opened state.  Keep that relation
    # strict for closing/unknown transitions, while opening requires the actor
    # to remain related independently to both the closure and the container.
    closure_proximity_required = bool(closure_boxes and state_direction != "opening")
    for candidate in candidates:
        coordinates = candidate.get("xyxy_norm") or (0, 0, 0, 0)
        candidate_top = float(coordinates[1])
        for actor in actor_boxes:
            actor_coordinates = actor.get("xyxy_norm") or (0, 0, 0, 0)
            actor_centre_y = (
                float(actor_coordinates[1]) + float(actor_coordinates[3])
            ) / 2.0
            if (
                services._box_edge_gap_norm(candidate, actor) <= maximum_actor_gap
                and (
                    not closure_proximity_required
                    or maximum_closure_gap is None
                    or min(
                        services._box_edge_gap_norm(candidate, closure)
                        for closure in closure_boxes
                    )
                    <= maximum_closure_gap
                )
                and (
                    not spatial_rule
                    or candidate_top >= actor_centre_y + first_person_minimum_top_offset
                )
            ):
                eligible.append(candidate)
                break
    relation_selected = min(
        eligible,
        key=lambda box: (
            min(
                (
                    services._box_edge_gap_norm(box, closure)
                    for closure in closure_boxes
                ),
                default=0.0,
            ),
            -float(box.get("confidence") or 0.0),
        ),
        default=None,
    )
    extent_candidates: list[dict[str, Any]] = []
    if relation_selected is not None:
        selected_confidence = float(relation_selected.get("confidence") or 0.0)
        sx1, sy1, sx2, sy2 = (
            float(item) for item in relation_selected.get("xyxy_norm") or (0, 0, 0, 0)
        )
        selected_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
        for candidate in eligible:
            cx1, cy1, cx2, cy2 = (
                float(item) for item in candidate.get("xyxy_norm") or (0, 0, 0, 0)
            )
            intersection_width = max(0.0, min(sx2, cx2) - max(sx1, cx1))
            intersection_height = max(0.0, min(sy2, cy2) - max(sy1, cy1))
            intersection = intersection_width * intersection_height
            candidate_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            union = selected_area + candidate_area - intersection
            overlap = intersection / union if union > 0.0 else 0.0
            if (
                float(candidate.get("confidence") or 0.0) >= selected_confidence * 0.80
                and overlap >= 0.30
            ):
                extent_candidates.append(candidate)
        selected = max(
            extent_candidates,
            key=lambda box: (
                max(
                    0.0,
                    float((box.get("xyxy_norm") or (0, 0, 0, 0))[2])
                    - float((box.get("xyxy_norm") or (0, 0, 0, 0))[0]),
                )
                * max(
                    0.0,
                    float((box.get("xyxy_norm") or (0, 0, 0, 0))[3])
                    - float((box.get("xyxy_norm") or (0, 0, 0, 0))[1]),
                ),
                float(box.get("confidence") or 0.0),
            ),
            default=relation_selected,
        )
    else:
        selected = None
    return selected, {
        "rule": (
            "closure_proximity_then_first_person_post_opening_geometry"
            if closure_proximity_required and spatial_rule
            else "closure_proximity_then_actor_contact"
            if closure_proximity_required
            else "actor_contact_with_independently_held_closure"
            if closure_boxes and state_direction == "opening"
            else "first_person_post_opening_below_actor_centre"
            if spatial_rule
            else "actor_contact_gate_then_highest_confidence"
        ),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "extent_expansion_candidate_count": len(extent_candidates),
        "extent_confidence_ratio": 0.80,
        "extent_minimum_iou": 0.30,
        "first_person_minimum_top_offset_norm": (
            first_person_minimum_top_offset if spatial_rule else None
        ),
        "closure_candidate_count": len(closure_boxes),
        "maximum_closure_gap_norm": maximum_closure_gap,
        "selected_closure_gap_norm": (
            round(
                min(
                    services._box_edge_gap_norm(selected, closure)
                    for closure in closure_boxes
                ),
                6,
            )
            if selected is not None and closure_boxes
            else None
        ),
        "fail_closed": not bool(eligible),
    }


def _canonical_grounding_label(label: str, prompt_map: dict[str, str]) -> str | None:
    """Resolve complete prompt phrases, including same-class merged labels.

    Grounding DINO can return several activated phrases for one box, such as
    ``brown reagent bottle brown bottle``. Keep that box only when every word
    is covered by configured prompts and all possible matches name one class.
    Partial phrases and combinations of different classes remain unresolved.
    """

    tokens = tuple(re.findall(r"\w+", str(label).lower()))
    if not tokens:
        return None
    phrases = [
        (tuple(re.findall(r"\w+", str(prompt).lower())), canonical)
        for prompt, canonical in prompt_map.items()
        if str(prompt).strip()
    ]
    exact_classes = {canonical for phrase, canonical in phrases if phrase == tokens}
    if exact_classes:
        return next(iter(exact_classes)) if len(exact_classes) == 1 else None
    resolved: dict[int, set[str]] = {0: set()}
    for start in range(len(tokens)):
        if start not in resolved:
            continue
        for phrase, canonical in phrases:
            end = start + len(phrase)
            if phrase and tokens[start:end] == phrase:
                resolved.setdefault(end, set()).update(resolved[start] | {canonical})
    classes = resolved.get(len(tokens), set())
    return next(iter(classes)) if len(classes) == 1 else None


@serialized_open_vocabulary
def _grounding_dino_key_frame_detections(
    frame: np.ndarray,
    canonical_classes: set[str],
    settings: dict[str, Any],
    *,
    services: ParticipantGroundingServices,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fallback = dict(settings.get("grounding_dino_fallback") or {})
    if not fallback.get("enabled"):
        return [], {"status": "disabled"}

    def infer_once(device):
        return services._grounding_dino_key_frame_detections_once(
            frame,
            canonical_classes,
            {**settings, "grounding_dino_fallback": {**fallback, "device": device}},
        )

    def cleanup():
        model_path = str(Path(str(fallback.get("model_path") or "")).resolve())
        cached = services._GROUNDING_DINO_MODEL_CACHE.get(
            (model_path, str(fallback.get("model_sha256") or ""))
        )
        if cached is not None:
            services.park_open_vocabulary_model(cached["model"])
            cached["device"] = "cpu"

    return services.run_with_cuda_oom_cpu_fallback(
        infer_once,
        device=str(fallback.get("device") or "cuda"),
        enabled=bool(fallback.get("cuda_oom_fallback_cpu", False)),
        cleanup=cleanup,
    )


def _grounding_dino_key_frame_detections_once(
    frame: np.ndarray,
    canonical_classes: set[str],
    settings: dict[str, Any],
    *,
    services: ParticipantGroundingServices,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ground only missing participant classes with the pinned local model."""

    fallback = dict(settings.get("grounding_dino_fallback") or {})
    if not fallback.get("enabled"):
        return [], {"status": "disabled"}
    requested_classes = {
        services.normalize_participant_class(item) for item in canonical_classes
    }
    prompt_map = {
        str(prompt).strip(): services.normalize_participant_class(canonical)
        for prompt, canonical in dict(fallback.get("prompt_map") or {}).items()
        if str(prompt).strip()
        and services.normalize_participant_class(canonical) in requested_classes
    }
    if not prompt_map:
        return [], {
            "status": "not_applicable",
            "requested_classes": sorted(requested_classes),
        }
    model_path = Path(str(fallback.get("model_path") or "")).resolve()
    weights_path = model_path / "model.safetensors"
    if not model_path.is_dir() or not weights_path.is_file():
        raise RuntimeError(f"Pinned Grounding DINO model is missing: {model_path}")
    model_sha256 = str(fallback.get("model_sha256") or "")
    validation_key = (str(weights_path), model_sha256)
    if validation_key not in services._GROUNDING_DINO_ASSET_VALIDATION:
        if model_sha256 and services._sha256_file(weights_path) != model_sha256:
            raise RuntimeError(f"Grounding DINO model hash mismatch: {weights_path}")
        services._GROUNDING_DINO_ASSET_VALIDATION.add(validation_key)

    import torch
    from PIL import Image
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    device = str(fallback.get("device") or "cuda")
    # Device is mutable residency, not a second model identity. CPU recovery
    # reuses the same validated weights instead of loading another DINO copy.
    cache_key = (str(model_path), model_sha256)
    cached = services._GROUNDING_DINO_MODEL_CACHE.get(cache_key)
    model_cache_reused = cached is not None
    model_load_seconds = 0.0
    model_restore_seconds = 0.0
    if cached is None:
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float32,
        )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Grounding DINO requires CUDA but CUDA is unavailable")
        # Retain the validated CPU weights before CUDA restoration so an OOM
        # can park and reuse this same model instead of loading a second copy.
        cached = {"processor": processor, "model": model.eval(), "device": "cpu"}
        services._GROUNDING_DINO_MODEL_CACHE[cache_key] = cached
        model.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        cached["device"] = str(model.device)
    else:
        restore_started = time.perf_counter()
        cached["model"].to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        model_restore_seconds = time.perf_counter() - restore_started
        cached["device"] = str(cached["model"].device)
    processor = cached["processor"]
    model = cached["model"]
    device = str(cached["device"])

    prompts = list(prompt_map)
    prompt_text = ". ".join(prompts) + "."
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    inference_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=float(fallback.get("box_threshold", 0.20)),
        text_threshold=float(fallback.get("text_threshold", 0.20)),
        target_sizes=[image.size[::-1]],
    )[0]
    labels = result.get("text_labels") or result.get("labels") or []
    height, width = frame.shape[:2]
    maximum_area = float(fallback.get("maximum_box_area_norm", 0.50))
    maximum_class_areas = {
        services.normalize_participant_class(class_name): float(limit)
        for class_name, limit in dict(
            fallback.get("maximum_class_box_area_norm") or {}
        ).items()
    }
    admitted: list[dict[str, Any]] = []
    rejected_area = 0
    rejected_label = 0
    recovered_composite_label_count = 0
    for confidence, label, coordinates in zip(
        result["scores"], labels, result["boxes"], strict=True
    ):
        grounded_prompt = str(label).strip().strip(".")
        canonical = prompt_map.get(grounded_prompt)
        if canonical is None:
            canonical = prompt_map.get(grounded_prompt.lower())
        if canonical is None:
            canonical = services._canonical_grounding_label(grounded_prompt, prompt_map)
            if canonical is not None:
                recovered_composite_label_count += 1
        if canonical is None:
            rejected_label += 1
            continue
        x1, y1, x2, y2 = (float(item) for item in coordinates)
        normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
        area = max(0.0, normalized[2] - normalized[0]) * max(
            0.0, normalized[3] - normalized[1]
        )
        class_maximum_area = min(
            maximum_area,
            float(maximum_class_areas.get(canonical, maximum_area)),
        )
        if area <= 0.0 or area > class_maximum_area:
            rejected_area += 1
            continue
        admitted.append(
            {
                "class_name": canonical,
                "confidence": float(confidence),
                "xyxy_norm": normalized,
                "track_id": None,
                "roi_motion": 0.0,
                "detector_source": "grounding_dino_base_key_frame_fallback",
                "grounding_prompt": grounded_prompt,
                "image_aspect_ratio": width / height,
            }
        )
    return admitted, {
        "schema_version": "visioncortex-grounding-dino-key-frame/1",
        "status": "executed",
        "scope": "missing final participant classes only",
        "full_timeline_inference": False,
        "model": str(model_path),
        "model_revision": str(fallback.get("model_revision") or ""),
        "model_sha256": model_sha256,
        "requested_classes": sorted(requested_classes),
        "prompts": prompts,
        "device": device,
        "precision": "float32",
        "raw_detection_count": len(result["scores"]),
        "admitted_area_bounded_count": len(admitted),
        "rejected_box_area_count": rejected_area,
        "rejected_label_count": rejected_label,
        "recovered_composite_label_count": recovered_composite_label_count,
        "maximum_box_area_norm": maximum_area,
        "model_load_seconds": round(model_load_seconds, 6),
        "model_cache_reused": model_cache_reused,
        "model_restore_seconds": round(model_restore_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "token_usage": 0,
        "ark_calls": 0,
    }


def _missing_state_transition_fallback_classes(
    grounded: Sequence[dict[str, Any]],
    *,
    active_object_classes: set[str],
    state_prompts: set[str],
    actor_boxes: Sequence[dict[str, Any]],
    maximum_actor_gap: float,
    closure_maximum_actor_gap: float,
    closure_minimum_confidence: float,
    services: ParticipantGroundingServices,
) -> set[str]:
    """Return only missing closure/container slots for one final state frame."""

    missing = {
        canonical_class
        for canonical_class in active_object_classes & {"bottle_cap", "tube_cap"}
        if not any(
            str(box.get("class_name") or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            == canonical_class
            and float(box.get("confidence") or 0.0) >= closure_minimum_confidence
            and actor_boxes
            and min(services._box_edge_gap_norm(box, actor) for actor in actor_boxes)
            <= closure_maximum_actor_gap
            for box in grounded
        )
    }
    container_aliases = {
        "container",
        "reagent_bottle",
        "sample_bottle",
        "sample_bottle_blue",
        "tube",
    }
    requested_container_aliases = active_object_classes & container_aliases
    state_container_present = any(
        str(box.get("class_name") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        in ({"container"} | requested_container_aliases)
        and actor_boxes
        and min(services._box_edge_gap_norm(box, actor) for actor in actor_boxes)
        <= maximum_actor_gap
        for box in grounded
    )
    if state_prompts and not state_container_present:
        # Request both state-specific generic prompts and the event's explicit
        # semantic bottle/tube alias.  The contact gate chooses only the
        # manipulated physical instance on a crowded bench.
        missing.update({"container", *requested_container_aliases})
    return missing


@serialized_open_vocabulary
def _open_vocabulary_key_frame_supplement(
    frame: np.ndarray,
    event: EvidenceEvent,
    closed_set_detections: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    view_role: str,
    services: ParticipantGroundingServices,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ground missing manipulated objects only on final accepted key frames.

    This is deliberately not a video-wide detector.  It runs at most on the
    two final role frames for an accepted event, and admits a grounded object
    only when its box touches or nearly touches an actor box.  Thus an open-set
    model cannot reintroduce the crowded-background boxes that the final
    participant-only contract is designed to suppress.
    """

    settings = config.get("models", {}).get("open_vocabulary_key_frame") or {}
    if not settings.get("enabled"):
        return [], {"status": "disabled"}
    participant_classes = {
        services.normalize_participant_class(item)
        for item in event.objects
        if str(item).strip()
    }
    active_actor_classes = participant_classes & {"hand", "gloved_hand"} or {
        "hand",
        "gloved_hand",
    }
    state_transition = event.action_type == ActionType.CONTAINER_STATE_CHANGE
    supplementable_classes = {
        "pipette",
        "spearhead",
        "paper",
        "tube",
        "tube_rack",
        "balance",
        "beaker",
        "container",
        "reagent_bottle",
        "sample_bottle",
        "sample_bottle_blue",
        "bottle_cap",
        "tube_cap",
    }
    active_object_classes = participant_classes & supplementable_classes
    if not state_transition and not active_object_classes:
        return [], {"status": "not_applicable"}
    model_path = Path(str(settings.get("model_path") or "")).resolve()
    clip_path = Path(str(settings.get("clip_model_path") or "")).resolve()
    if not model_path.is_file() or not clip_path.is_file():
        raise RuntimeError(
            "Open-vocabulary key-frame assets are missing: "
            f"model={model_path} clip={clip_path}"
        )
    model_sha256 = str(settings.get("model_sha256") or "")
    clip_sha256 = str(settings.get("clip_model_sha256") or "")
    validation_key = (
        str(model_path),
        model_sha256,
        str(clip_path),
        clip_sha256,
    )
    if validation_key not in services._OPEN_VOCABULARY_ASSET_VALIDATION:
        if model_sha256 and services._sha256_file(model_path) != model_sha256:
            raise RuntimeError(f"Open-vocabulary model hash mismatch: {model_path}")
        if clip_sha256 and services._sha256_file(clip_path) != clip_sha256:
            raise RuntimeError(f"CLIP model hash mismatch: {clip_path}")
        services._OPEN_VOCABULARY_ASSET_VALIDATION.add(validation_key)
    prompt_map = dict(settings.get("prompt_map") or {})
    physical_change = (event.model_understanding or {}).get("physical_change") or {}
    after_state_text = " ".join(
        str(item or "")
        for item in (
            (
                physical_change.get("after")
                if isinstance(physical_change, dict)
                else physical_change
            ),
            (event.model_understanding or {}).get("current_step"),
        )
    ).lower()
    opening_terms = (
        "open",
        "uncapped",
        "打开",
        "开启",
        "开放",
        "敞口",
        "瓶口露出",
        "瓶口暴露",
        "瓶盖与瓶口分离",
        "瓶口敞开",
        "瓶盖已被取下",
        "瓶盖被取下",
        "瓶盖离开瓶口",
        "瓶盖从瓶口取下",
    )
    closing_terms = (
        "closed",
        "capped",
        "盖合",
        "盖住",
        "封闭",
        "密封",
        "拧紧",
    )
    state_direction = (
        "opening"
        if any(term in after_state_text for term in opening_terms)
        else (
            "closing"
            if any(term in after_state_text for term in closing_terms)
            else "unknown"
        )
    )
    opening_prompts = {
        str(item) for item in settings.get("opening_container_prompts") or []
    }
    closing_prompts = {
        str(item) for item in settings.get("closing_container_prompts") or []
    }
    state_prompts = (
        opening_prompts
        if state_direction == "opening"
        else closing_prompts
        if state_direction == "closing"
        else set()
    )
    actor_prompts = {
        prompt
        for prompt, canonical in prompt_map.items()
        if services.normalize_participant_class(canonical) in active_actor_classes
    }
    closure_prompts = {
        prompt
        for prompt, canonical in prompt_map.items()
        if str(canonical) in {"bottle_cap", "tube_cap"}
    }
    participant_prompts = {
        prompt
        for prompt, canonical in prompt_map.items()
        if services.normalize_participant_class(canonical) in active_object_classes
    }
    active_prompts = actor_prompts | participant_prompts
    if state_transition:
        active_prompts |= state_prompts | closure_prompts
    prompts = [
        str(item) for item in prompt_map if str(item).strip() and item in active_prompts
    ]
    if not prompts:
        raise RuntimeError("Open-vocabulary prompt_map is empty")
    from ..open_vocabulary_runtime import load_yolo_world_with_local_clip

    cache_key = str(model_path)
    cached = services._OPEN_VOCABULARY_MODEL_CACHE.get(cache_key)
    model_cache_hit = cached is not None
    model_load_seconds = 0.0
    if cached is None:
        model_load_started = time.perf_counter()
        cached = {
            "model": load_yolo_world_with_local_clip(settings),
            "prompts": None,
        }
        model_load_seconds = time.perf_counter() - model_load_started
        services._OPEN_VOCABULARY_MODEL_CACHE[cache_key] = cached
    model = cached["model"]
    if cached.get("prompts") != prompts:
        # Ultralytics leaves the world model on the prediction device. Its
        # CLIP tokenizer creates CPU token tensors, so changing classes after a
        # GPU prediction otherwise mixes CPU indices with CUDA embeddings.
        # Final-key-frame grounding is bounded; moving back to CPU before the
        # rare prompt change is deterministic and avoids a second model copy.
        model.to("cpu")
        model.set_classes(prompts)
        cached["prompts"] = list(prompts)

    def infer_once(device):
        inference_started = time.perf_counter()
        result = model.predict(
            frame,
            device=device,
            half=False,
            imgsz=int(settings.get("image_size", 1280)),
            conf=float(settings.get("confidence", 0.03)),
            iou=float(settings.get("iou", 0.50)),
            verbose=False,
        )[0]
        return result, {
            "inference_seconds": round(time.perf_counter() - inference_started, 6),
            "actual_device": services.yolo_world_prediction_device(model),
        }

    result, execution_receipt = services.run_with_cuda_oom_cpu_fallback(
        infer_once,
        device=settings.get("device", 0),
        enabled=bool(settings.get("cuda_oom_fallback_cpu", False)),
        cleanup=lambda: services.park_open_vocabulary_model(model),
    )
    inference_seconds = execution_receipt["inference_seconds"]
    height, width = frame.shape[:2]
    grounded: list[dict[str, Any]] = []
    for class_index, confidence, coordinates in zip(
        result.boxes.cls,
        result.boxes.conf,
        result.boxes.xyxy,
        strict=True,
    ):
        prompt = str(result.names[int(class_index)])
        canonical = str(prompt_map.get(prompt) or "").strip()
        if not canonical:
            continue
        x1, y1, x2, y2 = (float(item) for item in coordinates)
        grounded.append(
            {
                "class_name": canonical,
                "confidence": float(confidence),
                "xyxy_norm": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
                "track_id": None,
                "roi_motion": 0.0,
                "detector_source": "yolo_world_v2_key_frame_supplement",
                "grounding_prompt": prompt,
            }
        )
    actor_classes = {"hand", "gloved_hand"}
    closed_set_actor_boxes = [
        dict(box)
        for box in closed_set_detections
        if str(box.get("class_name") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        in actor_classes
    ]
    raw_grounded_actor_boxes = [
        dict(box)
        for box in grounded
        if str(box.get("class_name") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        in actor_classes
    ]
    grounded_actor_maximum_area = float(
        settings.get("grounded_actor_maximum_box_area_norm", 0.15)
    )
    grounded_actor_confidence_ratio = float(
        settings.get("grounded_actor_confidence_ratio", 0.75)
    )
    grounded_actor_boxes = services._filter_grounded_actor_boxes(
        raw_grounded_actor_boxes,
        maximum_area_norm=grounded_actor_maximum_area,
        confidence_ratio=grounded_actor_confidence_ratio,
    )
    # Prefer the task-trained closed-set actor boxes whenever they exist. Open
    # vocabulary hand prompts can expand into an oversized arm/torso region and
    # incorrectly make a static rack tool appear adjacent. Grounded actors are
    # a fallback only for views where the closed-set detector saw no hand.
    actor_boxes = closed_set_actor_boxes or grounded_actor_boxes
    grounding_dino_receipt: dict[str, Any] = {"status": "not_needed"}
    grounding_dino_minimum_aspect_ratio = float(
        (settings.get("grounding_dino_fallback") or {}).get(
            "pipette_minimum_aspect_ratio", 1.40
        )
    )
    minimum_class_confidence = {
        services.normalize_participant_class(class_name): float(limit)
        for class_name, limit in dict(
            settings.get("minimum_manipulated_object_confidence") or {}
        ).items()
    }
    pipette_center_must_overlap_actor = bool(
        settings.get("pipette_center_must_overlap_actor", False)
    )
    pipette_maximum_actor_iou = float(settings.get("pipette_maximum_actor_iou", 0.65))
    state_maximum_gap = float(settings.get("state_container_max_actor_gap_norm", 0.08))
    state_maximum_closure_gap = float(
        settings.get("state_container_max_closure_gap_norm", 0.03)
    )
    closure_maximum_gap = float(settings.get("closure_max_actor_gap_norm", 0.01))
    closure_minimum_confidence = float(settings.get("closure_minimum_confidence", 0.12))
    preferred_grounding_terms = services._preferred_grounding_terms_by_class(event)
    if state_transition:
        # YOLO-World can miss either the small closure or the state-bearing
        # bottle/tube as a whole.  A state-transition final frame is complete
        # only when both slots accompany the actor, so request every missing
        # slot in one bounded local Grounding DINO call.  This remains
        # one-frame inference: no Ark call and no full-timeline rescan.
        missing_state_classes = services._missing_state_transition_fallback_classes(
            grounded,
            active_object_classes=active_object_classes,
            state_prompts=state_prompts,
            actor_boxes=actor_boxes,
            maximum_actor_gap=state_maximum_gap,
            closure_maximum_actor_gap=closure_maximum_gap,
            closure_minimum_confidence=closure_minimum_confidence,
        )
        if missing_state_classes:
            requested_classes = set(missing_state_classes)
            if not actor_boxes:
                requested_classes.update(active_actor_classes)
            fallback_boxes, grounding_dino_receipt = (
                services._grounding_dino_key_frame_detections(
                    frame, requested_classes, settings
                )
            )
            grounded.extend(fallback_boxes)
            if not actor_boxes:
                raw_grounded_actor_boxes = [
                    dict(box)
                    for box in grounded
                    if services.normalize_participant_class(box.get("class_name"))
                    in actor_classes
                ]
                grounded_actor_boxes = services._filter_grounded_actor_boxes(
                    raw_grounded_actor_boxes,
                    maximum_area_norm=grounded_actor_maximum_area,
                    confidence_ratio=grounded_actor_confidence_ratio,
                )
                actor_boxes = grounded_actor_boxes
    else:
        provisional_maximum_gap = float(
            settings.get("manipulated_object_max_actor_gap_norm", 0.08)
        )
        provisional_pipette_aspect = float(
            settings.get("pipette_minimum_aspect_ratio", 1.7)
        )
        missing_classes: set[str] = set()
        for canonical_class in sorted(active_object_classes):
            class_maximum_gap = (
                closure_maximum_gap
                if canonical_class in {"bottle_cap", "tube_cap"}
                else provisional_maximum_gap
            )
            class_minimum_confidence = max(
                float(minimum_class_confidence.get(canonical_class, 0.0)),
                (
                    closure_minimum_confidence
                    if canonical_class in {"bottle_cap", "tube_cap"}
                    else 0.0
                ),
            )
            candidates = [
                box
                for box in grounded
                if services.normalize_participant_class(box.get("class_name"))
                == canonical_class
            ]
            selected, _ = services._select_manipulated_object_candidate(
                candidates,
                actor_boxes,
                canonical_class=canonical_class,
                maximum_actor_gap=class_maximum_gap,
                pipette_minimum_aspect_ratio=provisional_pipette_aspect,
                grounding_dino_pipette_minimum_aspect_ratio=(
                    grounding_dino_minimum_aspect_ratio
                ),
                minimum_confidence=class_minimum_confidence,
                pipette_center_must_overlap_actor=(pipette_center_must_overlap_actor),
                pipette_maximum_actor_iou=pipette_maximum_actor_iou,
                preferred_grounding_terms=preferred_grounding_terms.get(
                    canonical_class, ()
                ),
            )
            if selected is None:
                missing_classes.add(canonical_class)
        if missing_classes:
            requested_classes = set(missing_classes)
            if not actor_boxes:
                requested_classes.update(active_actor_classes)
            fallback_boxes, grounding_dino_receipt = (
                services._grounding_dino_key_frame_detections(
                    frame, requested_classes, settings
                )
            )
            grounded.extend(fallback_boxes)
            if not actor_boxes:
                raw_grounded_actor_boxes = [
                    dict(box)
                    for box in grounded
                    if services.normalize_participant_class(box.get("class_name"))
                    in actor_classes
                ]
                grounded_actor_boxes = services._filter_grounded_actor_boxes(
                    raw_grounded_actor_boxes,
                    maximum_area_norm=grounded_actor_maximum_area,
                    confidence_ratio=grounded_actor_confidence_ratio,
                )
                actor_boxes = grounded_actor_boxes
    maximum_gap = float(settings.get("max_actor_object_gap_norm", 0.02))
    first_person_minimum_top_offset = float(
        settings.get("first_person_minimum_container_top_offset_norm", 0.0)
    )
    eligible_closure_boxes = [
        box
        for box in grounded
        if services.normalize_participant_class(box.get("class_name"))
        in {"bottle_cap", "tube_cap"}
        and float(box.get("confidence") or 0.0) >= closure_minimum_confidence
        and actor_boxes
        and min(services._box_edge_gap_norm(box, actor) for actor in actor_boxes)
        <= closure_maximum_gap
    ]
    state_semantic_text = " ".join(
        str(item or "")
        for item in (
            (event.model_understanding or {}).get("current_step"),
            physical_change.get("before"),
            physical_change.get("after"),
        )
    ).lower()
    preferred_closure_color = (
        "red"
        if "red" in state_semantic_text or "红" in state_semantic_text
        else "orange"
        if "orange" in state_semantic_text or "橙" in state_semantic_text
        else "blue"
        if "blue" in state_semantic_text or "蓝" in state_semantic_text
        else None
    )
    if preferred_closure_color:
        color_specific_closures = [
            box
            for box in eligible_closure_boxes
            if preferred_closure_color in str(box.get("grounding_prompt") or "").lower()
        ]
        if color_specific_closures:
            eligible_closure_boxes = color_specific_closures
    best_state_closure: dict[str, Any] | None = None
    if state_prompts:
        state_container_classes = {
            "container",
            *(
                active_object_classes
                & {
                    "reagent_bottle",
                    "sample_bottle",
                    "sample_bottle_blue",
                    "tube",
                }
            ),
        }
        state_grounded = [
            box
            for box in grounded
            if services.normalize_participant_class(box.get("class_name"))
            in state_container_classes
        ]
        best_state_container, state_selection = (
            services._select_state_container_candidate(
                state_grounded,
                actor_boxes,
                view_role=view_role,
                state_direction=state_direction,
                maximum_actor_gap=state_maximum_gap,
                first_person_minimum_top_offset=(first_person_minimum_top_offset),
                closure_boxes=eligible_closure_boxes,
                maximum_closure_gap=state_maximum_closure_gap,
            )
        )
        if best_state_container is not None and eligible_closure_boxes:
            best_state_closure = min(
                eligible_closure_boxes,
                key=lambda box: (
                    services._box_edge_gap_norm(box, best_state_container),
                    min(
                        services._box_edge_gap_norm(box, actor) for actor in actor_boxes
                    ),
                    -float(box.get("confidence") or 0.0),
                ),
            )
        state_selection["preferred_closure_color"] = preferred_closure_color
        state_selection["selected_closure_prompt"] = (
            str(best_state_closure.get("grounding_prompt") or "")
            if best_state_closure is not None
            else None
        )
    else:
        best_state_container = None
        state_selection = {
            "rule": "not_applicable",
            "candidate_count": 0,
            "eligible_candidate_count": 0,
            "first_person_minimum_top_offset_norm": None,
            "fail_closed": False,
        }
    manipulated_object_maximum_gap = float(
        settings.get("manipulated_object_max_actor_gap_norm", 0.08)
    )
    pipette_minimum_aspect_ratio = float(
        settings.get("pipette_minimum_aspect_ratio", 1.7)
    )
    selected_manipulated_objects: dict[str, dict[str, Any]] = {}
    manipulated_object_selection: dict[str, dict[str, Any]] = {}
    if not state_transition:
        for canonical_class in sorted(active_object_classes):
            class_maximum_gap = (
                closure_maximum_gap
                if canonical_class in {"bottle_cap", "tube_cap"}
                else manipulated_object_maximum_gap
            )
            class_minimum_confidence = max(
                float(minimum_class_confidence.get(canonical_class, 0.0)),
                (
                    closure_minimum_confidence
                    if canonical_class in {"bottle_cap", "tube_cap"}
                    else 0.0
                ),
            )
            canonical_candidates = [
                box
                for box in grounded
                if services.normalize_participant_class(box.get("class_name"))
                == canonical_class
            ]
            selected, selection_receipt = services._select_manipulated_object_candidate(
                canonical_candidates,
                actor_boxes,
                canonical_class=canonical_class,
                maximum_actor_gap=class_maximum_gap,
                pipette_minimum_aspect_ratio=pipette_minimum_aspect_ratio,
                grounding_dino_pipette_minimum_aspect_ratio=(
                    grounding_dino_minimum_aspect_ratio
                ),
                minimum_confidence=class_minimum_confidence,
                pipette_center_must_overlap_actor=(pipette_center_must_overlap_actor),
                pipette_maximum_actor_iou=pipette_maximum_actor_iou,
                preferred_grounding_terms=preferred_grounding_terms.get(
                    canonical_class, ()
                ),
            )
            manipulated_object_selection[canonical_class] = selection_receipt
            if selected is not None:
                selected_manipulated_objects[canonical_class] = selected
    admitted: list[dict[str, Any]] = []
    rejected_noncontact = 0
    for box in grounded:
        canonical = str(box.get("class_name") or "")
        if canonical in actor_classes:
            if not closed_set_actor_boxes and box in actor_boxes:
                admitted.append(box)
            continue
        normalized_canonical = services.normalize_participant_class(canonical)
        if not state_transition and normalized_canonical in active_object_classes:
            if box is selected_manipulated_objects.get(normalized_canonical):
                admitted.append(box)
            else:
                rejected_noncontact += 1
            continue
        actor_gap = (
            min(services._box_edge_gap_norm(box, actor) for actor in actor_boxes)
            if actor_boxes
            else float("inf")
        )
        if canonical == "container" and state_prompts:
            if box is best_state_container and actor_gap <= state_maximum_gap:
                admitted.append(box)
            else:
                rejected_noncontact += 1
            continue
        if canonical in {"bottle_cap", "tube_cap"}:
            if state_transition and box is best_state_closure:
                admitted.append(box)
            elif not state_transition and (
                float(box.get("confidence") or 0.0) >= closure_minimum_confidence
                and actor_gap <= closure_maximum_gap
            ):
                admitted.append(box)
            else:
                rejected_noncontact += 1
            continue
        if actor_gap <= maximum_gap:
            admitted.append(box)
        else:
            rejected_noncontact += 1
    if state_transition:
        closed_set_replaced_classes = sorted(
            {
                "container",
                "beaker",
                "tube",
                "sample_bottle",
                "sample_bottle_blue",
                "reagent_bottle",
            }
        )
    else:
        closed_set_replaced_classes = sorted(active_object_classes)
    return admitted, {
        "schema_version": "visioncortex-open-vocabulary-key-frame/1",
        "status": "executed",
        "scope": "final accepted key frames only",
        "full_timeline_inference": False,
        "model": str(model_path),
        "model_sha256": model_sha256,
        "model_cache_hit": model_cache_hit,
        "model_load_seconds": round(model_load_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        **execution_receipt,
        "clip_model": str(clip_path),
        "clip_model_sha256": clip_sha256,
        "prompts": prompts,
        "raw_detection_count": len(grounded),
        "admitted_contact_detection_count": len(admitted),
        "rejected_noncontact_detection_count": rejected_noncontact,
        "max_actor_object_gap_norm": maximum_gap,
        "actor_box_source": (
            "closed_set"
            if closed_set_actor_boxes
            else "open_vocabulary_fallback"
            if grounded_actor_boxes
            else "missing"
        ),
        "actor_box_count": len(actor_boxes),
        "grounded_actor_filter": {
            "raw_candidate_count": len(raw_grounded_actor_boxes),
            "admitted_candidate_count": len(grounded_actor_boxes),
            "maximum_box_area_norm": grounded_actor_maximum_area,
            "confidence_ratio": grounded_actor_confidence_ratio,
        },
        "manipulated_object_max_actor_gap_norm": (
            manipulated_object_maximum_gap if not state_transition else None
        ),
        "pipette_minimum_aspect_ratio": (
            pipette_minimum_aspect_ratio if not state_transition else None
        ),
        "manipulated_object_selection": manipulated_object_selection,
        "grounding_dino_fallback": grounding_dino_receipt,
        "closed_set_replaced_classes": closed_set_replaced_classes,
        "state_direction": state_direction,
        "state_container_prompts": sorted(state_prompts),
        "state_container_max_actor_gap_norm": state_maximum_gap,
        "state_container_selection": state_selection,
        "view_role": view_role,
        "closure_max_actor_gap_norm": closure_maximum_gap,
        "closure_minimum_confidence": closure_minimum_confidence,
        "admitted": [
            {
                "class_name": box["class_name"],
                "confidence": round(float(box["confidence"]), 6),
                "grounding_prompt": box["grounding_prompt"],
                "xyxy_norm": [round(float(item), 6) for item in box["xyxy_norm"]],
            }
            for box in admitted
        ],
    }
