from __future__ import annotations

import itertools
import math
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .detection import iter_frame_evidence
from .candidate_index import CoarseFrameIndex, FineFrameIndex
from .decisions import decision_receipt
from .grouping import (
    event_stable_identities,
    select_formal_experiment_start_events,
)
from .ordering import (
    candidate_sort_key,
    event_sort_key,
    stable_candidate_fingerprint,
    stable_event_fingerprint,
    stable_segment_uid,
)
from .schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    BoxEvidence,
    EvidenceEvent,
    ExperimentSegment,
    FrameEvidence,
    PhysicalChange,
    ViewInput,
    ViewRole,
    event_is_formal,
    set_event_admission,
)


HAND_CLASSES = {"hand", "gloved_hand"}
NON_ACTION_CLASSES = {"lab_coat", "PPE_Storage"}
DEVICE_CLASSES = {"balance", "magnetic_stirrer"}
CONTAINER_CLASSES = {
    "beaker",
    "reagent_bottle",
    "reagent_bottle_open",
    "sample_bottle",
    "sample_bottle_blue",
    "tube",
    "container",
}
CAP_CLASSES = {"tube_cap", "bottle_cap"}
TRANSFER_TOOL_CLASSES = {"pipette", "spearhead", "spatula"}
SUPPORT_ANCHOR_CLASSES = DEVICE_CLASSES | CONTAINER_CLASSES | CAP_CLASSES | {
    "tube_rack",
    "magnetic_stir_bar",
}


def _box_center(box: BoxEvidence) -> tuple[float, float]:
    x1, y1, x2, y2 = box.xyxy_norm
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_distance(a: BoxEvidence, b: BoxEvidence) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy_norm
    bx1, by1, bx2, by2 = b.xyxy_norm
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


@dataclass
class _Observation:
    action_type: ActionType
    local_ms: float
    global_ms: float
    objects: tuple[str, ...]
    confidence: float
    evidence: dict[str, Any]
    instance_key: str | None = None


def _observation_key(
    observation: _Observation,
    *,
    instance_aware: bool = False,
) -> tuple[str, str, str]:
    relevant = [item for item in observation.objects if item not in HAND_CLASSES]
    primary = sorted(relevant)[0] if relevant else "unknown"
    instance = observation.instance_key if instance_aware else None
    return observation.action_type.value, primary, instance or "class_fallback"


def _box_area(box: BoxEvidence) -> float:
    x1, y1, x2, y2 = box.xyxy_norm
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _instance_key(box: BoxEvidence, *, prefix: str = "object") -> str | None:
    if box.track_id is None:
        return None
    return f"{prefix}:{box.class_name}:{box.track_id}"


def _instance_evidence(box: BoxEvidence) -> dict[str, Any]:
    return {
        "class_name": box.class_name,
        "track_id": box.track_id,
        "center_norm": [round(item, 6) for item in _box_center(box)],
        "area_norm": round(_box_area(box), 8),
        "appearance_signature": list(box.appearance_signature),
    }


def _container_family(class_name: str) -> str:
    if class_name in {"tube", "tube_cap"}:
        return "tube"
    if class_name in {
        "reagent_bottle",
        "reagent_bottle_open",
        "sample_bottle",
        "sample_bottle_blue",
        "bottle_cap",
    }:
        return "bottle"
    if class_name in {"beaker", "container"}:
        return "open_container"
    return class_name


def _container_state_key(box: BoxEvidence) -> str:
    center_x, center_y = _box_center(box)
    return (
        f"{_container_family(box.class_name)}:"
        f"{round(center_x, 1):.1f}:{round(center_y, 1):.1f}"
    )


def _container_state(box: BoxEvidence) -> str:
    if box.class_name == "reagent_bottle_open":
        return "open_visible"
    if box.class_name in CAP_CLASSES:
        return "cap_visible"
    return "container_visible"


def _frame_observations(
    frame: FrameEvidence,
    previous_tracks: dict[int, tuple[float, float, float]],
    cfg: dict[str, Any],
    interaction_state: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    container_states: dict[str, str] | None = None,
    movement_history: dict[int, deque[tuple[float, float, float]]] | None = None,
) -> list[_Observation]:
    assert frame.global_ms is not None
    observations: list[_Observation] = []
    hands = [box for box in frame.detections if box.class_name in HAND_CLASSES]
    objects = [box for box in frame.detections if box.class_name not in HAND_CLASSES | NON_ACTION_CLASSES]
    contact_threshold = float(cfg["contact_distance_norm"])
    interaction_state = interaction_state if interaction_state is not None else {}
    container_states = container_states if container_states is not None else {}
    approach_delta = float(cfg.get("contact_approach_delta_norm", 0.01))
    seen_pairs: set[tuple[Any, ...]] = set()
    current_container_states = {
        _container_state_key(obj): _container_state(obj)
        for obj in objects
        if obj.class_name in CONTAINER_CLASSES | CAP_CLASSES
    }

    for hand, obj in itertools.product(hands, objects):
        distance = _box_distance(hand, obj)
        pair_key = (
            hand.track_id if hand.track_id is not None else hand.class_name,
            obj.track_id if obj.track_id is not None else obj.class_name,
        )
        seen_pairs.add(pair_key)
        state = interaction_state.setdefault(
            pair_key,
            {
                "previous_gap": None,
                "approach_confirmed": False,
                "contact_frames": 0,
                "released_at_global_ms": None,
                "last_seen_global_ms": float(frame.global_ms),
            },
        )
        state["last_seen_global_ms"] = float(frame.global_ms)
        previous_gap = state.get("previous_gap")
        if previous_gap is not None and float(previous_gap) - distance >= approach_delta:
            state["approach_confirmed"] = True
        state["previous_gap"] = distance
        if distance <= contact_threshold:
            state["contact_frames"] = int(state.get("contact_frames", 0)) + 1
            confidence = min(hand.confidence, obj.confidence) * max(0.5, 1.0 - distance / max(contact_threshold, 1e-9))
            interaction_receipt = {
                "phase": "contact",
                "approach_confirmed": bool(state.get("approach_confirmed")),
                "contact_frame_count": int(state["contact_frames"]),
                "previous_release_global_ms": state.get("released_at_global_ms"),
            }
            observations.append(
                _Observation(
                    action_type=ActionType.HAND_OBJECT_CONTACT,
                    local_ms=frame.local_ms,
                    global_ms=frame.global_ms,
                    objects=tuple(sorted({hand.class_name, obj.class_name})),
                    confidence=confidence,
                    evidence={
                        "frame_index": frame.frame_index,
                        "distance_norm": round(distance, 5),
                        "hand_track_id": hand.track_id,
                        "object_track_id": obj.track_id,
                        "object_instance": _instance_evidence(obj),
                        "interaction_state": interaction_receipt,
                    },
                    instance_key=_instance_key(obj),
                )
            )
            if obj.class_name in DEVICE_CLASSES:
                observations.append(
                    _Observation(
                        action_type=ActionType.DEVICE_PANEL_OPERATION,
                        local_ms=frame.local_ms,
                        global_ms=frame.global_ms,
                        objects=(hand.class_name, obj.class_name),
                        confidence=min(1.0, confidence + 0.08),
                        evidence={
                            "frame_index": frame.frame_index,
                            "distance_norm": round(distance, 5),
                            "hand_track_id": hand.track_id,
                            "object_track_id": obj.track_id,
                            "object_instance": _instance_evidence(obj),
                            "interaction_state": interaction_receipt,
                        },
                        instance_key=_instance_key(obj, prefix="device"),
                    )
                )
            if obj.class_name in CAP_CLASSES or obj.class_name == "reagent_bottle_open":
                state_key = _container_state_key(obj)
                observations.append(
                    _Observation(
                        action_type=ActionType.CONTAINER_STATE_CHANGE,
                        local_ms=frame.local_ms,
                        global_ms=frame.global_ms,
                        objects=tuple(sorted({hand.class_name, obj.class_name})),
                        confidence=min(1.0, confidence + 0.06),
                        evidence={
                            "frame_index": frame.frame_index,
                            "state_cue": obj.class_name,
                            "state_before": container_states.get(state_key, "unknown"),
                            "state_after": current_container_states.get(
                                state_key, _container_state(obj)
                            ),
                            "state_key": state_key,
                            "hand_track_id": hand.track_id,
                            "object_track_id": obj.track_id,
                            "object_instance": _instance_evidence(obj),
                            "interaction_state": interaction_receipt,
                        },
                        instance_key=(
                            _instance_key(obj, prefix="container") or state_key
                        ),
                    )
                )
        elif int(state.get("contact_frames", 0)) > 0:
            state["released_at_global_ms"] = float(frame.global_ms)
            state["contact_frames"] = 0
            state["approach_confirmed"] = False

    for pair_key, state in interaction_state.items():
        if pair_key in seen_pairs:
            continue
        if int(state.get("contact_frames", 0)) > 0:
            state["released_at_global_ms"] = float(frame.global_ms)
        state["contact_frames"] = 0
        state["approach_confirmed"] = False
        state["previous_gap"] = None
    state_retention_ms = max(
        2000.0,
        float(cfg.get("event_merge_gap_seconds", 1.25)) * 2000.0,
    )
    for pair_key, state in list(interaction_state.items()):
        if (
            int(state.get("contact_frames", 0)) == 0
            and float(state.get("last_seen_global_ms", frame.global_ms))
            < float(frame.global_ms) - state_retention_ms
        ):
            del interaction_state[pair_key]
    maximum_state_entries = max(
        128, int(cfg.get("fine_state_max_entries", 4096))
    )
    while len(interaction_state) > maximum_state_entries:
        interaction_state.pop(next(iter(interaction_state)))
    for state_key, state_value in current_container_states.items():
        container_states.pop(state_key, None)
        container_states[state_key] = state_value
    while len(container_states) > maximum_state_entries:
        container_states.pop(next(iter(container_states)))

    window_ms = float(cfg.get("movement_window_ms", 250.0))
    movement_threshold = float(cfg.get("movement_window_threshold_norm", 0.01)
                               if movement_history is not None else cfg["movement_threshold_norm"])
    movement_samples: list[
        tuple[BoxEvidence, float, float, float, float, float, float, float]
    ] = []
    for obj in objects:
        if obj.track_id is None:
            continue
        center_x, center_y = _box_center(obj)
        previous = previous_tracks.pop(obj.track_id, None)
        previous_tracks[obj.track_id] = (center_x, center_y, frame.local_ms)
        if movement_history is not None:
            history = movement_history.setdefault(obj.track_id, deque(maxlen=64))
            while history and frame.local_ms - history[0][2] > 2000.0:
                history.popleft()
            eligible = [item for item in history if frame.local_ms - item[2] >= window_ms * 0.6]
            previous = min(eligible, key=lambda item: abs(frame.local_ms - item[2] - window_ms)) if eligible else None
            history.append((center_x, center_y, frame.local_ms))
        if previous is None:
            continue
        delta_ms = frame.local_ms - previous[2]
        if not 0.0 < delta_ms <= 2000.0:
            continue
        dx = center_x - previous[0]
        dy = center_y - previous[1]
        movement_samples.append(
            (
                obj,
                center_x,
                center_y,
                delta_ms,
                dx,
                dy,
                previous[0],
                previous[1],
            )
        )
    stale_track_cutoff_ms = float(frame.local_ms) - 2000.0
    for track_id, (_, _, last_seen_ms) in list(previous_tracks.items()):
        if float(last_seen_ms) < stale_track_cutoff_ms:
            del previous_tracks[track_id]
    while len(previous_tracks) > maximum_state_entries:
        previous_tracks.pop(next(iter(previous_tracks)))
    if movement_history is not None:
        for track_id in list(movement_history):
            if track_id not in previous_tracks:
                del movement_history[track_id]

    minimum_anchors = max(2, int(cfg.get("camera_motion_compensation_min_anchors", 2)))
    anchor_vectors = [
        (dx, dy)
        for obj, _, _, _, dx, dy, _, _ in movement_samples
        if obj.class_name in SUPPORT_ANCHOR_CLASSES
    ]
    camera_dx = float(median(item[0] for item in anchor_vectors)) if len(anchor_vectors) >= minimum_anchors else 0.0
    camera_dy = float(median(item[1] for item in anchor_vectors)) if len(anchor_vectors) >= minimum_anchors else 0.0
    camera_compensated = len(anchor_vectors) >= minimum_anchors
    affine_matrix: np.ndarray | None = None
    affine_inliers = 0
    affine_enabled = bool(cfg.get("fine_affine_object_motion_enabled", False))
    affine_samples = [
        (previous_x, previous_y, center_x, center_y)
        for obj, center_x, center_y, _, _, _, previous_x, previous_y in movement_samples
        if obj.class_name in SUPPORT_ANCHOR_CLASSES
    ]
    if affine_enabled and len(affine_samples) >= max(3, minimum_anchors):
        previous_points = np.asarray(
            [[item[0], item[1]] for item in affine_samples], dtype=np.float32
        )
        current_points = np.asarray(
            [[item[2], item[3]] for item in affine_samples], dtype=np.float32
        )
        candidate_matrix, inliers = cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=0.02,
        )
        if candidate_matrix is not None:
            scale = math.hypot(
                float(candidate_matrix[0, 0]), float(candidate_matrix[1, 0])
            )
            rotation = abs(
                math.degrees(
                    math.atan2(
                        float(candidate_matrix[1, 0]),
                        float(candidate_matrix[0, 0]),
                    )
                )
            )
            if (
                abs(scale - 1.0)
                <= float(cfg.get("motion_probe_max_scale_delta", 0.12))
                and rotation
                <= float(cfg.get("motion_probe_max_rotation_degrees", 8.0))
            ):
                affine_matrix = candidate_matrix
                affine_inliers = int(inliers.sum()) if inliers is not None else 0
    suppress_stationary_devices = bool(
        cfg.get("suppress_stationary_device_movement", True)
    )
    for (
        obj,
        center_x,
        center_y,
        delta_ms,
        dx,
        dy,
        previous_x,
        previous_y,
    ) in movement_samples:
        raw_displacement = math.hypot(dx, dy)
        if affine_matrix is not None:
            predicted_x = (
                float(affine_matrix[0, 0]) * previous_x
                + float(affine_matrix[0, 1]) * previous_y
                + float(affine_matrix[0, 2])
            )
            predicted_y = (
                float(affine_matrix[1, 0]) * previous_x
                + float(affine_matrix[1, 1]) * previous_y
                + float(affine_matrix[1, 2])
            )
            displacement = math.hypot(
                center_x - predicted_x, center_y - predicted_y
            )
            compensation_method = "anchor_affine"
        elif camera_compensated:
            displacement = math.hypot(dx - camera_dx, dy - camera_dy)
            compensation_method = "anchor_translation"
        else:
            displacement = raw_displacement
            compensation_method = "none"
        if suppress_stationary_devices and obj.class_name in DEVICE_CLASSES:
            continue
        if displacement >= movement_threshold:
            observations.append(
                _Observation(
                    action_type=ActionType.OBJECT_MOVEMENT,
                    local_ms=frame.local_ms,
                    global_ms=frame.global_ms,
                    objects=(obj.class_name,),
                    confidence=min(1.0, obj.confidence * (0.75 + displacement / max(movement_threshold, 1e-9) * 0.08)),
                    evidence={
                        "frame_index": frame.frame_index,
                        "track_id": obj.track_id,
                        "displacement_norm": round(raw_displacement, 5),
                        "camera_compensated_displacement_norm": round(displacement, 5),
                        "camera_motion_compensated": camera_compensated,
                        "camera_motion_method": compensation_method,
                        "camera_translation_norm": [round(camera_dx, 5), round(camera_dy, 5)],
                        "camera_anchor_count": len(anchor_vectors),
                        "camera_affine_inlier_count": affine_inliers,
                        "delta_ms": round(delta_ms, 3),
                        "object_instance": _instance_evidence(obj),
                    },
                    instance_key=_instance_key(obj),
                )
            )

    transfers = [box for box in objects if box.class_name in TRANSFER_TOOL_CLASSES]
    vessels = [box for box in objects if box.class_name in CONTAINER_CLASSES]
    liquid_threshold = float(cfg["liquid_motion_threshold"])
    for tool, vessel in itertools.product(transfers, vessels):
        distance = _box_distance(tool, vessel)
        roi_motion = max(tool.roi_motion, vessel.roi_motion)
        if distance <= contact_threshold * 2.0 and roi_motion >= liquid_threshold:
            observations.append(
                _Observation(
                    action_type=ActionType.LIQUID_MOVEMENT,
                    local_ms=frame.local_ms,
                    global_ms=frame.global_ms,
                    objects=tuple(sorted({tool.class_name, vessel.class_name})),
                    confidence=min(tool.confidence, vessel.confidence) * min(1.0, 0.6 + roi_motion / 30.0),
                    evidence={
                        "frame_index": frame.frame_index,
                        "distance_norm": round(distance, 5),
                        "roi_motion": round(roi_motion, 3),
                        "tool_class": tool.class_name,
                        "tool_track_id": tool.track_id,
                        "vessel_class": vessel.class_name,
                        "vessel_track_id": vessel.track_id,
                        "tool_instance": _instance_evidence(tool),
                        "vessel_instance": _instance_evidence(vessel),
                        "inference": "液体不属于21类标签；这是工具+容器+ROI运动候选，需多模态确认",
                    },
                    instance_key=(
                        f"transfer:{tool.track_id}:{vessel.track_id}"
                        if tool.track_id is not None
                        and vessel.track_id is not None
                        else None
                    ),
                )
            )
    return observations


def _merge_observations(
    observations: Sequence[_Observation],
    view: ViewInput,
    cfg: dict[str, Any],
) -> list[ActionCandidate]:
    instance_aware = bool(cfg.get("fine_instance_association_enabled", False))
    grouped: dict[tuple[str, str, str], list[_Observation]] = defaultdict(list)
    for observation in observations:
        grouped[
            _observation_key(observation, instance_aware=instance_aware)
        ].append(observation)
    candidates: list[ActionCandidate] = []
    merge_gap_ms = float(cfg["event_merge_gap_seconds"]) * 1000.0
    min_duration_ms = float(cfg["min_event_duration_seconds"]) * 1000.0
    minimum_observations = int(cfg["min_event_observations"])
    counter = 1
    for items in grouped.values():
        items.sort(key=lambda item: item.global_ms)
        runs: list[list[_Observation]] = []
        current: list[_Observation] = []
        for item in items:
            if current and item.global_ms - current[-1].global_ms > merge_gap_ms:
                runs.append(current)
                current = []
            current.append(item)
        if current:
            runs.append(current)
        for run in runs:
            duration = run[-1].global_ms - run[0].global_ms
            action_type = run[0].action_type
            enough = len(run) >= minimum_observations and duration >= min_duration_ms
            if action_type in {ActionType.CONTAINER_STATE_CHANGE, ActionType.DEVICE_PANEL_OPERATION}:
                enough = len(run) >= max(2, minimum_observations - 1)
            if not enough:
                continue
            object_names = sorted({obj for observation in run for obj in observation.objects})
            raw_confidence = sum(item.confidence for item in run) / len(run)
            persistence = min(0.15, math.log1p(len(run)) * 0.035)
            confidence = min(1.0, raw_confidence + persistence)
            key_item = max(run, key=lambda item: item.confidence)
            evidence = _representative_observation_evidence(run, key_item)
            instance_signature = _observation_instance_signature(run)
            uncertainty = _observation_state_uncertainty(run, cfg)
            candidates.append(
                ActionCandidate(
                    candidate_id=f"CAND-{view.view_id}-{counter:06d}",
                    action_type=action_type,
                    view_id=view.view_id,
                    role=view.role,
                    local_start_ms=run[0].local_ms,
                    local_end_ms=run[-1].local_ms,
                    global_start_ms=run[0].global_ms,
                    global_end_ms=run[-1].global_ms,
                    key_global_ms=key_item.global_ms,
                    objects=object_names,
                    confidence=confidence,
                    evidence=evidence,
                    uncertainty=uncertainty,
                    instance_signature=instance_signature,
                    provenance={
                        "source_stage": "candidate_fine",
                        "candidate_reducer": "batch",
                        "instance_association": bool(instance_signature),
                    },
                )
            )
            counter += 1
    return sorted(candidates, key=candidate_sort_key)


def _infer_liquid_transfer_sequences(
    observations: Sequence[_Observation],
    view: ViewInput,
    cfg: dict[str, Any],
) -> list[ActionCandidate]:
    """Use the same exclusive-contact reducer for batch and streaming paths."""
    reducer = _StreamingLiquidSequences(view, cfg)
    for timestamp, items in itertools.groupby(
        sorted(observations, key=lambda item: item.global_ms),
        key=lambda item: item.global_ms,
    ):
        reducer.expire(timestamp)
        reducer.add_frame(list(items))
    return reducer.finish()


def generate_candidates(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    frame_index: FineFrameIndex | None = None,
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    cfg = dict(config["segmentation"])
    for key in (
        "fine_streaming_candidates_enabled",
        "fine_state_max_entries",
        "fine_instance_association_enabled",
        "fine_contact_state_enabled",
        "fine_affine_object_motion_enabled",
        "motion_probe_max_scale_delta",
        "motion_probe_max_rotation_degrees",
    ):
        if key in config["performance"]:
            cfg[key] = config["performance"][key]
    for view in views:
        frames = (
            frame_index.iter_frames(view.view_id)
            if frame_index is not None
            else iter_frame_evidence(detection_paths[view.view_id])
        )
        if cfg.get("fine_streaming_candidates_enabled", False):
            candidates.extend(
                _generate_candidates_streaming(view, frames, cfg)
            )
            continue
        observations: list[_Observation] = []
        previous_tracks: dict[int, tuple[float, float, float]] = {}
        movement_history: dict[int, deque[tuple[float, float, float]]] = {}
        interaction_state: dict[tuple[Any, ...], dict[str, Any]] = {}
        container_states: dict[str, str] = {}
        for frame in frames:
            observations.extend(
                _frame_observations(
                    frame,
                    previous_tracks,
                    cfg,
                    interaction_state,
                    container_states,
                    movement_history,
                )
            )
        candidates.extend(_merge_observations(observations, view, cfg))
        candidates.extend(_infer_liquid_transfer_sequences(observations, view, cfg))
    return sorted(candidates, key=candidate_sort_key)


def generate_coarse_activity_candidates(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    frame_index: CoarseFrameIndex | None = None,
) -> list[ActionCandidate]:
    """Recall-first activity windows; the fine layer must verify contact and action type."""
    cfg = config["segmentation"]
    perf = config["performance"]
    spatial_relation = bool(
        perf.get("coarse_spatial_relation_enabled", False)
    )
    relation_gap = float(
        perf.get("coarse_hand_object_gap_norm", cfg["contact_distance_norm"])
    )
    relaxed_relation_gap = float(
        perf.get("coarse_hand_object_approach_gap_norm", relation_gap * 2.0)
    )
    approach_delta = float(
        perf.get("coarse_hand_object_approach_delta_norm", 0.025)
    )
    micro_action_guard = bool(
        perf.get("coarse_micro_action_guard_enabled", False)
    )
    micro_confidence = float(
        perf.get("coarse_micro_action_min_confidence", 0.72)
    )
    micro_half_window_ms = float(
        perf.get("coarse_micro_action_window_seconds", 4.0)
    ) * 500.0
    rolling_motion = bool(
        perf.get("coarse_rolling_motion_threshold_enabled", False)
    )
    rolling_window_ms = float(
        perf.get("coarse_rolling_motion_window_seconds", 600.0)
    ) * 1000.0
    observations_by_view: dict[str, list[_Observation]] = defaultdict(list)
    micro_observations_by_view: dict[str, list[_Observation]] = defaultdict(list)
    by_id = {view.view_id: view for view in views}
    for view in views:
        previous_pair_gaps: dict[tuple[Any, ...], float] = {}
        current_motion_bucket: int | None = None
        current_motion_scores: list[float] = []
        previous_motion_threshold = 10.0
        indexed_motion_thresholds: dict[int, float] = {}
        if rolling_motion and frame_index is not None:
            indexed_scores: dict[int, list[float]] = defaultdict(list)
            for global_ms, motion_score in frame_index.iter_motion_samples(
                view.view_id
            ):
                indexed_scores[
                    max(0, int(global_ms // max(rolling_window_ms, 1.0)))
                ].append(motion_score)
            for bucket, values in indexed_scores.items():
                scores = np.asarray(values, dtype=np.float64)
                indexed_motion_thresholds[bucket] = max(
                    float(np.percentile(scores, 80.0)),
                    float(np.median(scores) + 2.5),
                )
        source_frames = (
            frame_index.iter_frames(view.view_id, activity_only=True)
            if frame_index is not None
            else iter_frame_evidence(detection_paths[view.view_id])
        )
        for frame in source_frames:
            if frame.global_ms is None:
                continue
            bucket = max(0, int(frame.global_ms // max(rolling_window_ms, 1.0)))
            if indexed_motion_thresholds:
                previous_motion_threshold = indexed_motion_thresholds.get(
                    bucket, 10.0
                )
            elif current_motion_bucket is None:
                current_motion_bucket = bucket
            elif bucket != current_motion_bucket:
                if current_motion_scores:
                    scores = np.asarray(current_motion_scores, dtype=np.float64)
                    previous_motion_threshold = max(
                        float(np.percentile(scores, 80.0)),
                        float(np.median(scores) + 2.5),
                    )
                current_motion_scores = []
                current_motion_bucket = bucket
            if not indexed_motion_thresholds:
                current_motion_scores.append(frame.motion_score)
            hands = [box for box in frame.detections if box.class_name in HAND_CLASSES]
            objects = [
                box
                for box in frame.detections
                if box.class_name not in HAND_CLASSES | NON_ACTION_CLASSES | {"paper"}
            ]
            if hands and objects:
                legacy_hand = max(hands, key=lambda box: box.confidence)
                legacy_object = max(objects, key=lambda box: box.confidence)
                if spatial_relation:
                    best_hand, best_object = min(
                        itertools.product(hands, objects),
                        key=lambda pair: (
                            _box_distance(pair[0], pair[1]),
                            -min(pair[0].confidence, pair[1].confidence),
                        ),
                    )
                else:
                    best_hand = legacy_hand
                    best_object = legacy_object
                gap = _box_distance(best_hand, best_object)
                pair_key = (
                    best_hand.track_id
                    if best_hand.track_id is not None
                    else best_hand.class_name,
                    best_object.track_id
                    if best_object.track_id is not None
                    else best_object.class_name,
                )
                previous_gap = previous_pair_gaps.get(pair_key)
                approaching = bool(
                    previous_gap is not None
                    and previous_gap - gap >= approach_delta
                    and gap <= relaxed_relation_gap
                )
                previous_pair_gaps[pair_key] = gap
                relation_confirmed = bool(
                    not spatial_relation or gap <= relation_gap or approaching
                )
                confidence = min(best_hand.confidence, best_object.confidence)
                if spatial_relation and not relation_confirmed:
                    confidence *= 0.55
                observation = _Observation(
                    action_type=ActionType.HAND_OBJECT_CONTACT,
                    local_ms=frame.local_ms,
                    global_ms=frame.global_ms,
                    objects=tuple(
                        sorted({best_hand.class_name, best_object.class_name})
                    ),
                    confidence=confidence,
                    evidence={
                        "frame_index": frame.frame_index,
                        "coarse_activity": (
                            "hand_object_spatial_relation"
                            if relation_confirmed and spatial_relation
                            else "hand_and_lab_object_cooccurrence"
                        ),
                        "hand_object_gap_norm": round(gap, 6),
                        "approaching": approaching,
                        "relation_confirmed": relation_confirmed,
                        "legacy_cooccurrence_retained": bool(
                            spatial_relation and not relation_confirmed
                        ),
                        "hand_track_id": best_hand.track_id,
                        "object_track_id": best_object.track_id,
                    },
                )
                observations_by_view[view.view_id].append(observation)
                if spatial_relation and (
                    legacy_hand is not best_hand
                    or legacy_object is not best_object
                ):
                    observations_by_view[view.view_id].append(
                        _Observation(
                            action_type=ActionType.HAND_OBJECT_CONTACT,
                            local_ms=frame.local_ms,
                            global_ms=frame.global_ms,
                            objects=tuple(
                                sorted(
                                    {
                                        legacy_hand.class_name,
                                        legacy_object.class_name,
                                    }
                                )
                            ),
                            confidence=(
                                min(
                                    legacy_hand.confidence,
                                    legacy_object.confidence,
                                )
                                * 0.55
                            ),
                            evidence={
                                "frame_index": frame.frame_index,
                                "coarse_activity": (
                                    "hand_and_lab_object_cooccurrence"
                                ),
                                "legacy_cooccurrence_retained": True,
                                "relation_confirmed": False,
                                "hand_object_gap_norm": round(
                                    _box_distance(
                                        legacy_hand, legacy_object
                                    ),
                                    6,
                                ),
                                "hand_track_id": legacy_hand.track_id,
                                "object_track_id": legacy_object.track_id,
                            },
                        )
                    )
                if (
                    micro_action_guard
                    and relation_confirmed
                    and observation.confidence >= micro_confidence
                ):
                    micro_observations_by_view[view.view_id].append(observation)
            motion_threshold = (
                min(10.0, previous_motion_threshold) if rolling_motion else 10.0
            )
            if not hands and frame.motion_score >= motion_threshold and len(objects) >= 2:
                selected = sorted(objects, key=lambda box: box.confidence, reverse=True)[:2]
                observations_by_view[view.view_id].append(
                    _Observation(
                        action_type=ActionType.OBJECT_MOVEMENT,
                        local_ms=frame.local_ms,
                        global_ms=frame.global_ms,
                        objects=tuple(sorted(box.class_name for box in selected)),
                        confidence=min(box.confidence for box in selected) * 0.75,
                        evidence={
                            "frame_index": frame.frame_index,
                            "coarse_motion": frame.motion_score,
                            "coarse_motion_threshold": motion_threshold,
                        },
                    )
                )
    candidates: list[ActionCandidate] = []
    for view_id, observations in observations_by_view.items():
        merged = _merge_observations(observations, by_id[view_id], cfg)
        candidates.extend(merged)
        for observation in micro_observations_by_view.get(view_id, []):
            if any(
                candidate.action_type == observation.action_type
                and candidate.global_start_ms <= observation.global_ms
                <= candidate.global_end_ms
                for candidate in merged
            ):
                continue
            if any(
                candidate.view_id == view_id
                and abs(candidate.key_global_ms - observation.global_ms)
                <= micro_half_window_ms
                and candidate.action_type == observation.action_type
                for candidate in candidates
            ):
                continue
            candidates.append(
                ActionCandidate(
                    candidate_id=(
                        f"COARSE-MICRO-{view_id}-{len(candidates) + 1:06d}"
                    ),
                    action_type=observation.action_type,
                    view_id=view_id,
                    role=by_id[view_id].role,
                    local_start_ms=max(
                        0.0, observation.local_ms - micro_half_window_ms
                    ),
                    local_end_ms=observation.local_ms + micro_half_window_ms,
                    global_start_ms=max(
                        0.0, observation.global_ms - micro_half_window_ms
                    ),
                    global_end_ms=observation.global_ms + micro_half_window_ms,
                    key_global_ms=observation.global_ms,
                    objects=list(observation.objects),
                    confidence=min(0.89, observation.confidence),
                    evidence=[
                        {
                            **observation.evidence,
                            "micro_action_guard": True,
                        }
                    ],
                    uncertainty=[
                        "单帧高置信手物关系仅扩大精扫窗口，不独立确认物理动作"
                    ],
                )
            )
    return sorted(candidates, key=candidate_sort_key)


def _representative_observation_evidence(
    observations: Sequence[_Observation],
    key_item: _Observation,
) -> list[dict[str, Any]]:
    """Keep bounded evidence while always retaining start, peak and end."""

    selected = list(observations[:47])
    for item in (key_item, observations[-1]):
        if item not in selected:
            selected.append(item)
    if len(observations) > 49:
        middle = observations[len(observations) // 2]
        if middle not in selected:
            selected.append(middle)
    return [_observation_evidence(item) for item in selected[:50]]


def _observation_evidence(observation: _Observation) -> dict[str, Any]:
    return {
        **observation.evidence,
        "observation_local_ms": round(float(observation.local_ms), 3),
        "observation_global_ms": round(float(observation.global_ms), 3),
    }


def _observation_instance_signature(
    observations: Sequence[_Observation],
) -> dict[str, Any]:
    instance_keys = sorted(
        {item.instance_key for item in observations if item.instance_key}
    )
    track_ids = sorted(
        {
            int(track_id)
            for item in observations
            for key in (
                "object_track_id",
                "track_id",
                "tool_track_id",
                "vessel_track_id",
            )
            if isinstance((track_id := item.evidence.get(key)), int)
        }
    )
    instances = [
        value
        for item in observations
        for key in ("object_instance", "tool_instance", "vessel_instance")
        if isinstance((value := item.evidence.get(key)), dict)
    ]
    centers = [
        value["center_norm"]
        for value in instances
        if isinstance(value.get("center_norm"), list)
        and len(value["center_norm"]) == 2
    ]
    areas = [
        float(value["area_norm"])
        for value in instances
        if isinstance(value.get("area_norm"), (int, float))
    ]
    appearance_values = [
        list(value["appearance_signature"])
        for value in instances
        if isinstance(value.get("appearance_signature"), list)
        and value["appearance_signature"]
    ]
    appearance_signature = None
    if appearance_values:
        width = min(len(item) for item in appearance_values)
        appearance_signature = [
            round(
                float(median(item[index] for item in appearance_values)), 6
            )
            for index in range(width)
        ]
    if not instance_keys and not track_ids:
        return {}
    return {
        "schema_version": "visioncortex-object-instance-signature/1",
        "instance_keys": instance_keys,
        "track_ids": track_ids,
        "object_classes": sorted(
            {
                str(value.get("class_name"))
                for value in instances
                if value.get("class_name")
            }
        ),
        "reliable_single_instance": bool(
            len(instance_keys) == 1 or len(track_ids) == 1
        ),
        "median_center_norm": (
            [
                round(float(median(item[0] for item in centers)), 6),
                round(float(median(item[1] for item in centers)), 6),
            ]
            if centers
            else None
        ),
        "median_area_norm": (
            round(float(median(areas)), 8) if areas else None
        ),
        "appearance_signature": appearance_signature,
    }


def _observation_state_uncertainty(
    observations: Sequence[_Observation],
    cfg: dict[str, Any],
) -> list[str]:
    uncertainty: list[str] = []
    action_type = observations[0].action_type
    if (
        cfg.get("fine_contact_state_enabled", False)
        and action_type
        in {
            ActionType.HAND_OBJECT_CONTACT,
            ActionType.CONTAINER_STATE_CHANGE,
            ActionType.DEVICE_PANEL_OPERATION,
        }
        and not any(
            bool((item.evidence.get("interaction_state") or {}).get(
                "approach_confirmed"
            ))
            for item in observations
        )
    ):
        uncertainty.append(
            "未观察到完整接近过程；保留原有接触召回，不能单独确认操作"
        )
    if action_type == ActionType.CONTAINER_STATE_CHANGE:
        transitions = {
            (
                str(item.evidence.get("state_before", "unknown")),
                str(item.evidence.get("state_after", "unknown")),
            )
            for item in observations
        }
        if not any(left != "unknown" and left != right for left, right in transitions):
            uncertainty.append(
                "未形成明确的容器前后状态差异；保留状态线索候选"
            )
    return uncertainty


@dataclass
class _RunAccumulator:
    first: _Observation
    last: _Observation
    key_item: _Observation
    count: int
    confidence_sum: float
    objects: set[str]
    samples: list[_Observation]
    instance_mode: str
    release_observed_at_global_ms: float | None

    @classmethod
    def start(
        cls, observation: _Observation, *, instance_mode: str
    ) -> _RunAccumulator:
        return cls(
            first=observation,
            last=observation,
            key_item=observation,
            count=1,
            confidence_sum=float(observation.confidence),
            objects=set(observation.objects),
            samples=[observation],
            instance_mode=instance_mode,
            release_observed_at_global_ms=None,
        )

    def add(self, observation: _Observation) -> None:
        self.last = observation
        self.count += 1
        self.confidence_sum += float(observation.confidence)
        self.objects.update(observation.objects)
        if observation.confidence > self.key_item.confidence:
            self.key_item = observation
        if len(self.samples) < 47:
            self.samples.append(observation)

    def representative_observations(self) -> list[_Observation]:
        selected = list(self.samples)
        for item in (self.key_item, self.last):
            if item not in selected:
                selected.append(item)
        return selected[:50]

    def mark_release(self, global_ms: float) -> None:
        if self.release_observed_at_global_ms is None:
            self.release_observed_at_global_ms = float(global_ms)


def _candidate_from_accumulator(
    accumulator: _RunAccumulator,
    view: ViewInput,
    cfg: dict[str, Any],
) -> ActionCandidate | None:
    duration = accumulator.last.global_ms - accumulator.first.global_ms
    minimum_observations = int(cfg["min_event_observations"])
    enough = bool(
        accumulator.count >= minimum_observations
        and duration >= float(cfg["min_event_duration_seconds"]) * 1000.0
    )
    if accumulator.first.action_type in {
        ActionType.CONTAINER_STATE_CHANGE,
        ActionType.DEVICE_PANEL_OPERATION,
    }:
        enough = accumulator.count >= max(2, minimum_observations - 1)
    if not enough:
        return None
    persistence = min(0.15, math.log1p(accumulator.count) * 0.035)
    confidence = min(
        1.0,
        accumulator.confidence_sum / accumulator.count + persistence,
    )
    representatives = accumulator.representative_observations()
    signature = _observation_instance_signature(representatives)
    uncertainty = _observation_state_uncertainty(representatives, cfg)
    interaction_receipt = {
        "schema_version": "visioncortex-contact-state/1",
        "approach_observed": any(
            bool((item.evidence.get("interaction_state") or {}).get(
                "approach_confirmed"
            ))
            for item in representatives
        ),
        "contact_observation_count": accumulator.count,
        "release_observed": (
            accumulator.release_observed_at_global_ms is not None
        ),
        "release_observed_at_global_ms": (
            round(accumulator.release_observed_at_global_ms, 3)
            if accumulator.release_observed_at_global_ms is not None
            else None
        ),
        "release_inferred_from_next_observation_gap": False,
    }
    if accumulator.instance_mode == "class_fallback":
        uncertainty.append(
            "轨迹身份不足时保留原有类别级召回；不得用于证明同一物体"
        )
    return ActionCandidate(
        candidate_id="CAND-PENDING",
        action_type=accumulator.first.action_type,
        view_id=view.view_id,
        role=view.role,
        local_start_ms=accumulator.first.local_ms,
        local_end_ms=accumulator.last.local_ms,
        global_start_ms=accumulator.first.global_ms,
        global_end_ms=accumulator.last.global_ms,
        key_global_ms=accumulator.key_item.global_ms,
        objects=sorted(accumulator.objects),
        confidence=confidence,
        evidence=[_observation_evidence(item) for item in representatives],
        uncertainty=uncertainty,
        instance_signature=signature,
        provenance={
            "source_stage": "candidate_fine",
            "candidate_reducer": "streaming",
            "instance_mode": accumulator.instance_mode,
            "observation_count": accumulator.count,
            "representative_evidence_policy": "start_peak_end_bounded_50",
            "interaction_state": interaction_receipt,
        },
    )


@dataclass
class _LiquidContactRun:
    first: _Observation
    last: _Observation
    count: int
    confidence_sum: float

    @property
    def vessel_identity(self) -> tuple[str, int]:
        return (
            str(self.first.evidence.get("vessel_class") or "unknown"),
            int(self.first.evidence["vessel_track_id"]),
        )

    def add(self, observation: _Observation) -> None:
        self.last = observation
        self.count += 1
        self.confidence_sum += float(observation.confidence)


class _StreamingLiquidSequences:
    def __init__(self, view: ViewInput, cfg: dict[str, Any]):
        self.view = view
        self.maximum_gap_ms = float(
            cfg.get("liquid_transfer_max_sequence_gap_seconds", 20.0)
        ) * 1000.0
        self.contact_gap_ms = float(
            cfg.get("event_merge_gap_seconds", 1.25)
        ) * 1000.0
        self.minimum_observations = max(
            2, int(cfg.get("liquid_transfer_min_contact_observations", 2))
        )
        self.minimum_contact_ms = max(
            0.0, float(cfg.get("liquid_transfer_min_contact_duration_seconds", .2)) * 1000.0
        )
        self.active: dict[tuple[str, int], _LiquidContactRun] = {}
        self.previous: dict[tuple[str, int], _LiquidContactRun] = {}
        self.output: list[ActionCandidate] = []

    def add_frame(self, observations: Sequence[_Observation]) -> None:
        by_tool: dict[tuple[str, int], list[_Observation]] = defaultdict(list)
        for observation in observations:
            if observation.action_type == ActionType.LIQUID_MOVEMENT:
                key = self._tool_key(observation)
                if key is not None:
                    by_tool[key].append(observation)
        for key, items in by_tool.items():
            vessels = {(str(item.evidence.get("vessel_class") or "unknown"), item.evidence["vessel_track_id"])
                       for item in items}
            if len(vessels) != 1:
                # Simultaneous proximity is ambiguous, never a source/target transition.
                # Raw tool/vessel observations still enter the ordinary recall path.
                self.active.pop(key, None)
                self.previous.pop(key, None)
                continue
            self.add(max(items, key=lambda item: item.confidence))

    @staticmethod
    def _tool_key(observation: _Observation) -> tuple[str, int] | None:
        tool_class = observation.evidence.get("tool_class")
        tool_track_id = observation.evidence.get("tool_track_id")
        vessel_track_id = observation.evidence.get("vessel_track_id")
        if (
            not isinstance(tool_class, str)
            or not isinstance(tool_track_id, int)
            or not isinstance(vessel_track_id, int)
        ):
            return None
        return tool_class, tool_track_id

    def add(self, observation: _Observation) -> None:
        if observation.action_type != ActionType.LIQUID_MOVEMENT:
            return
        key = self._tool_key(observation)
        if key is None:
            return
        current = self.active.get(key)
        vessel = (
            str(observation.evidence.get("vessel_class") or "unknown"),
            int(observation.evidence["vessel_track_id"]),
        )
        if current is not None and observation.global_ms <= current.last.global_ms:
            if vessel != current.vessel_identity:
                self.active.pop(key, None)
                self.previous.pop(key, None)
            return
        if current is not None and (
            vessel != current.vessel_identity
            or observation.global_ms - current.last.global_ms
            > self.contact_gap_ms
        ):
            self._finalize(key)
            current = None
        if current is None:
            self.active[key] = _LiquidContactRun(
                first=observation,
                last=observation,
                count=1,
                confidence_sum=float(observation.confidence),
            )
        else:
            current.add(observation)

    def expire(self, global_ms: float) -> None:
        for key, current in list(self.active.items()):
            if global_ms - current.last.global_ms > self.contact_gap_ms:
                self._finalize(key)
        for key, previous in list(self.previous.items()):
            if global_ms - previous.last.global_ms > self.maximum_gap_ms:
                del self.previous[key]

    def finish(self) -> list[ActionCandidate]:
        for key in list(self.active):
            self._finalize(key)
        return self.output

    def _finalize(self, key: tuple[str, int]) -> None:
        current = self.active.pop(key)
        if (current.count < self.minimum_observations
                or current.last.global_ms - current.first.global_ms < self.minimum_contact_ms):
            self.previous.pop(key, None)
            return
        previous = self.previous.get(key)
        if previous is not None:
            gap_ms = current.first.global_ms - previous.last.global_ms
            if (
                previous.vessel_identity != current.vessel_identity
                and 0.0 < gap_ms <= self.maximum_gap_ms
            ):
                combined_count = previous.count + current.count
                confidence = min(
                    1.0,
                    (
                        previous.confidence_sum + current.confidence_sum
                    )
                    / combined_count
                    + 0.08,
                )
                self.output.append(
                    ActionCandidate(
                        candidate_id=(
                            f"TRANSFER-SEQ-{self.view.view_id}-"
                            f"{len(self.output) + 1:06d}"
                        ),
                        action_type=ActionType.LIQUID_MOVEMENT,
                        view_id=self.view.view_id,
                        role=self.view.role,
                        local_start_ms=previous.first.local_ms,
                        local_end_ms=current.last.local_ms,
                        global_start_ms=previous.first.global_ms,
                        global_end_ms=current.last.global_ms,
                        key_global_ms=(
                            previous.last.global_ms + current.first.global_ms
                        )
                        / 2.0,
                        objects=sorted(
                            {
                                key[0],
                                previous.vessel_identity[0],
                                current.vessel_identity[0],
                            }
                        ),
                        confidence=confidence,
                        evidence=[
                            {
                                "transfer_sequence": "source_transport_target",
                                "tool_class": key[0],
                                "tool_track_id": key[1],
                                "source_class": previous.vessel_identity[0],
                                "source_track_id": previous.vessel_identity[1],
                                "target_class": current.vessel_identity[0],
                                "target_track_id": current.vessel_identity[1],
                                "source_contact_end_global_ms": (
                                    previous.last.global_ms
                                ),
                                "target_contact_start_global_ms": (
                                    current.first.global_ms
                                ),
                                "transport_gap_ms": gap_ms,
                                "source_observation_count": previous.count,
                                "target_observation_count": current.count,
                                "source_contact_duration_ms": previous.last.global_ms - previous.first.global_ms,
                                "target_contact_duration_ms": current.last.global_ms - current.first.global_ms,
                                "minimum_contact_duration_ms": self.minimum_contact_ms,
                                "exclusive_contact_observations": True,
                                "contact_sequence_only": True,
                                "transport_observed": False,
                                "streaming_state_machine": True,
                            }
                        ],
                        uncertainty=[
                            "工具先后接近不同容器；中间转运、释放及液体本体尚未确认"
                        ],
                        instance_signature={
                            "schema_version": (
                                "visioncortex-object-instance-signature/1"
                            ),
                            "tool_track_id": key[1],
                            "source_track_id": previous.vessel_identity[1],
                            "target_track_id": current.vessel_identity[1],
                            "reliable_single_instance": True,
                        },
                        provenance={
                            "source_stage": "candidate_fine",
                            "candidate_reducer": "streaming_transfer_state_machine",
                        },
                    )
                )
        self.previous[key] = current


def _legacy_candidate_covered(
    legacy: ActionCandidate,
    precise: Sequence[ActionCandidate],
) -> bool:
    legacy_objects = set(legacy.objects) - HAND_CLASSES
    return any(
        item.action_type == legacy.action_type
        and bool(legacy_objects & (set(item.objects) - HAND_CLASSES))
        and item.global_start_ms <= legacy.global_end_ms
        and item.global_end_ms >= legacy.global_start_ms
        for item in precise
    )


def _generate_candidates_streaming(
    view: ViewInput,
    frames: Iterable[FrameEvidence],
    cfg: dict[str, Any],
) -> list[ActionCandidate]:
    merge_gap_ms = float(cfg["event_merge_gap_seconds"]) * 1000.0
    instance_aware = bool(cfg.get("fine_instance_association_enabled", False))
    active_precise: dict[tuple[str, str, str], _RunAccumulator] = {}
    active_legacy: dict[tuple[str, str, str], _RunAccumulator] = {}
    precise: list[ActionCandidate] = []
    legacy: list[ActionCandidate] = []
    previous_tracks: dict[int, tuple[float, float, float]] = {}
    movement_history: dict[int, deque[tuple[float, float, float]]] = {}
    interaction_state: dict[tuple[Any, ...], dict[str, Any]] = {}
    container_states: dict[str, str] = {}
    liquid_sequences = _StreamingLiquidSequences(view, cfg)

    def finalize_stale(
        active: dict[tuple[str, str, str], _RunAccumulator],
        output: list[ActionCandidate],
        global_ms: float,
    ) -> None:
        for key, accumulator in list(active.items()):
            if global_ms - accumulator.last.global_ms <= merge_gap_ms:
                continue
            candidate = _candidate_from_accumulator(accumulator, view, cfg)
            if candidate is not None:
                output.append(candidate)
            del active[key]

    def add_to(
        active: dict[tuple[str, str, str], _RunAccumulator],
        key: tuple[str, str, str],
        observation: _Observation,
        *,
        instance_mode: str,
    ) -> None:
        accumulator = active.get(key)
        if accumulator is None:
            active[key] = _RunAccumulator.start(
                observation, instance_mode=instance_mode
            )
        else:
            accumulator.add(observation)

    def mark_observed_releases(global_ms: float) -> None:
        released_objects = {
            str(pair_key[1])
            for pair_key, state in interaction_state.items()
            if state.get("released_at_global_ms") == global_ms
        }
        if not released_objects:
            return
        for active in (active_precise, active_legacy):
            for accumulator in active.values():
                object_track_id = accumulator.last.evidence.get(
                    "object_track_id"
                )
                object_instance = accumulator.last.evidence.get(
                    "object_instance"
                ) or {}
                object_class = object_instance.get("class_name")
                if (
                    str(object_track_id) in released_objects
                    or str(object_class) in released_objects
                ):
                    accumulator.mark_release(global_ms)

    for frame in frames:
        if frame.global_ms is None:
            continue
        finalize_stale(active_precise, precise, float(frame.global_ms))
        finalize_stale(active_legacy, legacy, float(frame.global_ms))
        liquid_sequences.expire(float(frame.global_ms))
        observations = _frame_observations(
            frame,
            previous_tracks,
            cfg,
            interaction_state,
            container_states,
            movement_history,
        )
        mark_observed_releases(float(frame.global_ms))
        liquid_sequences.add_frame(observations)
        for observation in observations:
            if instance_aware and observation.instance_key:
                add_to(
                    active_precise,
                    _observation_key(observation, instance_aware=True),
                    observation,
                    instance_mode="track_instance",
                )
                add_to(
                    active_legacy,
                    _observation_key(observation, instance_aware=False),
                    observation,
                    instance_mode="class_fallback",
                )
            else:
                add_to(
                    active_legacy,
                    _observation_key(observation, instance_aware=False),
                    observation,
                    instance_mode="class_fallback",
                )
    for active, output in (
        (active_precise, precise),
        (active_legacy, legacy),
    ):
        for accumulator in active.values():
            candidate = _candidate_from_accumulator(accumulator, view, cfg)
            if candidate is not None:
                output.append(candidate)

    if instance_aware:
        candidates = [
            *precise,
            *[
                item
                for item in legacy
                if not _legacy_candidate_covered(item, precise)
            ],
        ]
    else:
        candidates = legacy
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.global_start_ms,
            item.global_end_ms,
            item.action_type.value,
            item.objects,
        ),
    )
    numbered = [
        item.model_copy(
            update={"candidate_id": f"CAND-{view.view_id}-{index:06d}"}
        )
        for index, item in enumerate(ordered, 1)
    ]
    return sorted(
        [*numbered, *liquid_sequences.finish()], key=candidate_sort_key
    )


def _motion_probe_frames(
    view_id: str,
    path: Path,
    config: dict[str, Any],
    frame_index: CoarseFrameIndex | None = None,
) -> list[FrameEvidence]:
    """Load the logical probe grid, including scores shared by a coarse pass."""

    frames = list(
        frame_index.iter_frames(view_id)
        if frame_index is not None
        else iter_frame_evidence(path)
    )
    if not config["performance"].get(
        "motion_probe_use_embedded_coarse_scores", False
    ):
        return frames
    return [
        frame.model_copy(
            update={
                "motion_score": float(frame.motion_probe_score),
                "raw_motion_score": float(
                    frame.motion_probe_raw_score
                    if frame.motion_probe_raw_score is not None
                    else frame.motion_probe_score
                ),
            }
        )
        for frame in frames
        if frame.motion_probe_score is not None
    ]


def _adaptive_motion_thresholds(
    frames: Sequence[FrameEvidence], config: dict[str, Any]
) -> tuple[float, dict[int, float], float | None]:
    perf = config["performance"]
    percentile = float(perf["motion_burst_percentile"])
    scores = np.asarray([frame.motion_score for frame in frames], dtype=np.float64)
    global_threshold = max(
        float(np.percentile(scores, percentile)),
        float(np.median(scores) + 2.5),
    )
    raw_threshold: float | None = None
    if perf.get("motion_probe_legacy_raw_union_enabled", False):
        raw_scores = np.asarray(
            [frame.raw_motion_score for frame in frames], dtype=np.float64
        )
        raw_threshold = max(
            float(np.percentile(raw_scores, percentile)),
            float(np.median(raw_scores) + 2.5),
        )
    rolling: dict[int, float] = {}
    if perf.get("motion_probe_rolling_threshold_enabled", False):
        window_ms = max(
            1_000.0,
            float(perf.get("motion_probe_rolling_window_seconds", 600.0))
            * 1000.0,
        )
        grouped: dict[int, list[float]] = defaultdict(list)
        for frame in frames:
            timeline_ms = float(
                frame.global_ms if frame.global_ms is not None else frame.local_ms
            )
            grouped[max(0, int(timeline_ms // window_ms))].append(
                frame.motion_score
            )
        for bucket, values in grouped.items():
            array = np.asarray(values, dtype=np.float64)
            rolling[bucket] = max(
                float(np.percentile(array, percentile)),
                float(np.median(array) + 2.5),
            )
    return global_threshold, rolling, raw_threshold


def generate_motion_burst_candidates(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    frame_index: CoarseFrameIndex | None = None,
) -> list[ActionCandidate]:
    """Adaptive per-view motion bursts, optionally gated by detected lab objects."""
    perf = config["performance"]
    merge_gap_ms = float(perf["motion_burst_merge_gap_seconds"]) * 1000.0
    min_observations = int(perf["motion_burst_min_observations"])
    maximum_run_ms = float(
        perf.get("motion_probe_max_cluster_seconds", 0.0)
    ) * 1000.0
    require_objects = bool(perf.get("motion_probe_require_objects", True))
    candidates: list[ActionCandidate] = []
    for view in views:
        frames = _motion_probe_frames(
            view.view_id,
            detection_paths[view.view_id],
            config,
            frame_index,
        )
        if not frames:
            continue
        threshold, rolling_thresholds, raw_threshold = _adaptive_motion_thresholds(
            frames, config
        )
        rolling_window_ms = max(
            1_000.0,
            float(perf.get("motion_probe_rolling_window_seconds", 600.0))
            * 1000.0,
        )
        active: list[tuple[FrameEvidence, list[str]]] = []
        for frame in frames:
            objects = sorted(
                {
                    box.class_name
                    for box in frame.detections
                    if box.class_name not in HAND_CLASSES | NON_ACTION_CLASSES | {"paper"}
                }
            )
            timeline_ms = float(
                frame.global_ms if frame.global_ms is not None else frame.local_ms
            )
            bucket_threshold = rolling_thresholds.get(
                max(0, int(timeline_ms // rolling_window_ms)), threshold
            )
            adaptive_active = frame.motion_score >= min(
                threshold, bucket_threshold
            )
            raw_active = bool(
                raw_threshold is not None
                and frame.raw_motion_score >= raw_threshold
            )
            if (adaptive_active or raw_active) and (objects or not require_objects):
                active.append((frame, objects))
        runs: list[list[tuple[FrameEvidence, list[str]]]] = []
        current: list[tuple[FrameEvidence, list[str]]] = []
        for item in active:
            global_ms = item[0].global_ms or 0.0
            previous_ms = (current[-1][0].global_ms or 0.0) if current else None
            exceeds_run_span = bool(
                current
                and maximum_run_ms > 0.0
                and global_ms - (current[0][0].global_ms or 0.0)
                > maximum_run_ms
            )
            if current and previous_ms is not None and (
                global_ms - previous_ms > merge_gap_ms or exceeds_run_span
            ):
                runs.append(current)
                current = []
            current.append(item)
        if current:
            runs.append(current)
        for run in runs:
            if len(run) < min_observations:
                continue
            first, last = run[0][0], run[-1][0]
            assert first.global_ms is not None and last.global_ms is not None
            peak_frame, _peak_objects = max(
                run, key=lambda item: item[0].motion_score
            )
            assert peak_frame.global_ms is not None
            confidence = min(1.0, 0.55 + 0.05 * len(run) + 0.1 * peak_frame.motion_score / max(threshold, 1e-6))
            candidates.append(
                ActionCandidate(
                    candidate_id=f"BURST-{view.view_id}-{len(candidates) + 1:06d}",
                    action_type=ActionType.OBJECT_MOVEMENT,
                    view_id=view.view_id,
                    role=view.role,
                    local_start_ms=first.local_ms,
                    local_end_ms=last.local_ms,
                    global_start_ms=first.global_ms,
                    global_end_ms=last.global_ms,
                    key_global_ms=peak_frame.global_ms,
                    objects=sorted({obj for _, objects in run for obj in objects}),
                    confidence=confidence,
                    evidence=[
                        {
                            "frame_index": frame.frame_index,
                            "motion_score": frame.motion_score,
                            "threshold": threshold,
                            "rolling_threshold": rolling_thresholds.get(
                                max(
                                    0,
                                    int(
                                        float(
                                            frame.global_ms
                                            if frame.global_ms is not None
                                            else frame.local_ms
                                        )
                                        // rolling_window_ms
                                    ),
                                )
                            ),
                            "raw_motion_score": frame.raw_motion_score,
                            "raw_threshold": raw_threshold,
                            "objects": objects,
                        }
                        for frame, objects in run
                    ],
                    uncertainty=["粗层运动突发，仅用于召回精扫窗口，不作为最终动作证据"],
                )
            )
    return sorted(candidates, key=candidate_sort_key)


def _candidate_object_identity(candidate: ActionCandidate) -> set[str]:
    return {
        item
        for item in candidate.objects
        if item not in HAND_CLASSES | NON_ACTION_CLASSES
    }


def _coarse_candidates_compatible(
    left: ActionCandidate,
    right: ActionCandidate,
    *,
    temporal_margin_ms: float,
) -> bool:
    if (
        left.global_end_ms + temporal_margin_ms < right.global_start_ms
        or right.global_end_ms + temporal_margin_ms < left.global_start_ms
    ):
        return False
    left_objects = _candidate_object_identity(left)
    right_objects = _candidate_object_identity(right)
    if left_objects and right_objects:
        return bool(left_objects & right_objects)
    return bool(
        left.action_type == right.action_type
        or ActionType.OBJECT_MOVEMENT in {left.action_type, right.action_type}
    )


def _semantic_coarse_clusters(
    candidates: Sequence[ActionCandidate], temporal_margin_ms: float
) -> list[list[ActionCandidate]]:
    clusters: list[list[ActionCandidate]] = []
    for candidate in sorted(candidates, key=candidate_sort_key):
        compatible = [
            cluster
            for cluster in clusters
            if any(
                _coarse_candidates_compatible(
                    candidate,
                    member,
                    temporal_margin_ms=temporal_margin_ms,
                )
                for member in cluster
            )
        ]
        if not compatible:
            clusters.append([candidate])
            continue
        target = compatible[0]
        target.append(candidate)
        for extra in compatible[1:]:
            target.extend(extra)
            clusters.remove(extra)
    return clusters


def refine_motion_candidates_with_coarse(
    motion_candidates: Sequence[ActionCandidate],
    coarse_candidates: Sequence[ActionCandidate],
    config: dict[str, Any],
) -> tuple[list[ActionCandidate], dict[str, Any]]:
    """Conservatively tighten motion windows with bounded YOLO evidence.

    Unsupported motion candidates with object evidence are retained, and coarse
    candidates outside every motion interval are appended. Objectless motion
    intervals with no coarse YOLO match are quarantined with a receipt because
    they cannot become a physical evidence event without an observed object.
    """

    perf = config["performance"]
    association_margin_ms = float(
        perf.get("coarse_refinement_association_margin_seconds", 30.0)
    ) * 1000.0
    minimum_candidates = max(1, int(perf.get("coarse_refinement_min_candidates", 2)))
    minimum_span_ms = float(perf.get("coarse_refinement_min_span_seconds", 30.0)) * 1000.0
    quarantine_objectless = bool(
        perf.get("coarse_quarantine_objectless_unconfirmed_motion", True)
    )
    sustained_objectless_recall_ms = float(
        perf.get(
            "coarse_objectless_recall_guard_min_seconds",
            perf.get("motion_probe_primary_min_seconds", 45.0),
        )
    ) * 1000.0
    semantic_association = bool(
        perf.get("coarse_semantic_association_enabled", False)
    )
    cluster_margin_ms = float(
        perf.get("coarse_semantic_cluster_margin_seconds", 3.0)
    ) * 1000.0
    maximum_boundary_expansion_ms = float(
        perf.get("coarse_max_boundary_expansion_seconds", 0.0)
    ) * 1000.0
    uncertainty_by_view = {
        str(key): max(0.0, float(value))
        for key, value in dict(
            perf.get("candidate_alignment_uncertainty_ms_by_view") or {}
        ).items()
    }
    ordered_coarse_candidates = sorted(coarse_candidates, key=candidate_sort_key)
    used_coarse_ids: set[str] = set()
    refined: list[ActionCandidate] = []
    decisions: list[dict[str, Any]] = []

    for motion in sorted(motion_candidates, key=candidate_sort_key):
        motion_uncertainty_ms = uncertainty_by_view.get(motion.view_id, 0.0)
        matches = [
            coarse
            for coarse in ordered_coarse_candidates
            if coarse.global_end_ms
            >= motion.global_start_ms
            - association_margin_ms
            - motion_uncertainty_ms
            - uncertainty_by_view.get(coarse.view_id, 0.0)
            and coarse.global_start_ms
            <= motion.global_end_ms
            + association_margin_ms
            + motion_uncertainty_ms
            + uncertainty_by_view.get(coarse.view_id, 0.0)
        ]
        discarded_semantic_match_ids: list[str] = []
        if semantic_association and len(matches) > 1:
            clusters = _semantic_coarse_clusters(matches, cluster_margin_ms)
            motion_objects = _candidate_object_identity(motion)

            def cluster_rank(cluster: Sequence[ActionCandidate]) -> tuple[Any, ...]:
                objects = {
                    item
                    for candidate in cluster
                    for item in _candidate_object_identity(candidate)
                }
                return (
                    int(bool(motion_objects & objects)),
                    len({item.view_id for item in cluster}),
                    len(cluster),
                    sum(float(item.confidence) for item in cluster),
                    -min(item.global_start_ms for item in cluster),
                )

            selected_cluster = max(clusters, key=cluster_rank)
            selected_ids = {item.candidate_id for item in selected_cluster}
            discarded_semantic_match_ids = [
                item.candidate_id
                for item in matches
                if item.candidate_id not in selected_ids
            ]
            matches = list(selected_cluster)
        coarse_start = min(
            (item.global_start_ms for item in matches), default=motion.global_start_ms
        )
        coarse_end = max(
            (item.global_end_ms for item in matches), default=motion.global_end_ms
        )
        if maximum_boundary_expansion_ms > 0.0:
            coarse_start = max(
                coarse_start,
                motion.global_start_ms - maximum_boundary_expansion_ms,
            )
            coarse_end = min(
                coarse_end,
                motion.global_end_ms + maximum_boundary_expansion_ms,
            )
        supported = (
            len(matches) >= minimum_candidates
            and coarse_end - coarse_start >= minimum_span_ms
        )
        if not supported:
            motion_duration_ms = (
                motion.global_end_ms - motion.global_start_ms
            )
            sustained_objectless_recall_guard = bool(
                not matches
                and not motion.objects
                and motion_duration_ms >= sustained_objectless_recall_ms
            )
            if (
                quarantine_objectless
                and not matches
                and not motion.objects
                and not sustained_objectless_recall_guard
            ):
                decisions.append(
                    {
                        "motion_candidate_id": motion.candidate_id,
                        "decision": "quarantined_objectless_motion_without_coarse_yolo",
                        "coarse_candidate_ids": [],
                        "discarded_semantic_match_ids": discarded_semantic_match_ids,
                        "reason": (
                            "motion-only interval has no object identity and no "
                            "coarse YOLO match; it cannot form a physical event"
                        ),
                        "original_duration_seconds": round(
                            (motion.global_end_ms - motion.global_start_ms) / 1000.0,
                            3,
                        ),
                    }
                )
                continue
            refined.append(motion)
            decisions.append(
                {
                    "motion_candidate_id": motion.candidate_id,
                    "decision": "retained_motion_recall_guard",
                    "coarse_candidate_ids": [item.candidate_id for item in matches],
                    "discarded_semantic_match_ids": discarded_semantic_match_ids,
                    "reason": (
                        "sustained objectless sentinel motion remains a bounded "
                        "fine-scan recall guard"
                        if sustained_objectless_recall_guard
                        else "object-backed or coarse-supported motion recall guard"
                    ),
                    "original_duration_seconds": round(
                        motion_duration_ms / 1000.0,
                        3,
                    ),
                }
            )
            continue

        used_coarse_ids.update(item.candidate_id for item in matches)
        best = min(
            matches,
            key=lambda item: (-float(item.confidence), candidate_sort_key(item)),
        )
        refined.append(
            ActionCandidate(
                candidate_id=f"REFINED-{motion.candidate_id}",
                action_type=best.action_type,
                view_id=best.view_id,
                role=best.role,
                local_start_ms=min(item.local_start_ms for item in matches),
                local_end_ms=max(item.local_end_ms for item in matches),
                global_start_ms=coarse_start,
                global_end_ms=coarse_end,
                key_global_ms=best.key_global_ms,
                objects=sorted({name for item in matches for name in item.objects}),
                confidence=max(item.confidence for item in matches),
                evidence=[
                    {
                        "source": "coarse_yolo_refinement",
                        "motion_candidate_id": motion.candidate_id,
                        "coarse_candidate_ids": [item.candidate_id for item in matches],
                        "discarded_semantic_match_ids": discarded_semantic_match_ids,
                        "original_global_start_ms": motion.global_start_ms,
                        "original_global_end_ms": motion.global_end_ms,
                    }
                ],
                uncertainty=[
                    "Boundary tightened by coarse YOLO evidence; bounded fine scan and progressive cross-view audit remain mandatory."
                ],
            )
        )
        decisions.append(
            {
                "motion_candidate_id": motion.candidate_id,
                "decision": "refined_by_coarse_yolo",
                "coarse_candidate_ids": [item.candidate_id for item in matches],
                "discarded_semantic_match_ids": discarded_semantic_match_ids,
                "original_duration_seconds": round(
                    (motion.global_end_ms - motion.global_start_ms) / 1000.0, 3
                ),
                "refined_duration_seconds": round((coarse_end - coarse_start) / 1000.0, 3),
            }
        )

    unmatched = [
        item
        for item in ordered_coarse_candidates
        if item.candidate_id not in used_coarse_ids
    ]
    refined.extend(unmatched)
    refined.sort(key=candidate_sort_key)
    return refined, {
        "schema_version": "visioncortex-coarse-boundary-refinement/1",
        "motion_candidate_count": len(motion_candidates),
        "coarse_candidate_count": len(coarse_candidates),
        "refined_motion_count": sum(
            item["decision"] == "refined_by_coarse_yolo" for item in decisions
        ),
        "retained_motion_count": sum(
            item["decision"] == "retained_motion_recall_guard" for item in decisions
        ),
        "quarantined_motion_count": sum(
            item["decision"]
            == "quarantined_objectless_motion_without_coarse_yolo"
            for item in decisions
        ),
        "unmatched_coarse_candidate_count": len(unmatched),
        "output_candidate_count": len(refined),
        "output_candidates": [
            item.model_dump(mode="json") for item in refined
        ],
        "settings": {
            "association_margin_seconds": association_margin_ms / 1000.0,
            "minimum_candidates": minimum_candidates,
            "minimum_span_seconds": minimum_span_ms / 1000.0,
            "quarantine_objectless_unconfirmed_motion": quarantine_objectless,
            "objectless_recall_guard_min_seconds": (
                sustained_objectless_recall_ms / 1000.0
            ),
            "semantic_association_enabled": semantic_association,
            "semantic_cluster_margin_seconds": cluster_margin_ms / 1000.0,
            "maximum_boundary_expansion_seconds": (
                maximum_boundary_expansion_ms / 1000.0
            ),
            "alignment_uncertainty_ms_by_view": uncertainty_by_view,
        },
        "decisions": decisions,
    }


def fuse_motion_probe_candidates(
    candidates: Sequence[ActionCandidate], config: dict[str, Any]
) -> list[ActionCandidate]:
    """Fuse sentinel-view motion into recall-first global windows.

    Cross-view agreement is preferred. A sustained first-person burst is kept
    by itself so a temporarily occluded third-person sentinel cannot erase a
    real experiment.
    """

    if not candidates:
        return []
    perf = config["performance"]
    merge_gap_ms = float(perf.get("motion_probe_merge_gap_seconds", 20.0)) * 1000.0
    minimum_views = max(1, int(perf.get("motion_probe_min_views", 2)))
    primary_min_ms = float(perf.get("motion_probe_primary_min_seconds", 45.0)) * 1000.0
    maximum_cluster_ms = float(
        perf.get("motion_probe_max_cluster_seconds", 0.0)
    ) * 1000.0
    clusters: list[list[ActionCandidate]] = []
    current: list[ActionCandidate] = []
    current_end = -1.0
    for candidate in sorted(candidates, key=candidate_sort_key):
        exceeds_cluster_span = bool(
            current
            and maximum_cluster_ms > 0.0
            and candidate.global_end_ms - current[0].global_start_ms
            > maximum_cluster_ms
        )
        if current and (
            candidate.global_start_ms > current_end + merge_gap_ms
            or exceeds_cluster_span
        ):
            clusters.append(current)
            current = []
            current_end = -1.0
        current.append(candidate)
        current_end = max(current_end, candidate.global_end_ms)
    if current:
        clusters.append(current)

    fused: list[ActionCandidate] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        views = sorted({item.view_id for item in cluster})
        sustained_primary = any(
            item.role == ViewRole.FIRST_PERSON
            and item.global_end_ms - item.global_start_ms >= primary_min_ms
            for item in cluster
        )
        if len(views) < minimum_views and not sustained_primary:
            continue
        best = min(
            cluster,
            key=lambda item: (-float(item.confidence), candidate_sort_key(item)),
        )
        start_ms = min(item.global_start_ms for item in cluster)
        end_ms = max(item.global_end_ms for item in cluster)
        fused.append(
            ActionCandidate(
                candidate_id=f"MOTION-FUSED-{cluster_index:06d}",
                action_type=ActionType.OBJECT_MOVEMENT,
                view_id=best.view_id,
                role=best.role,
                local_start_ms=best.local_start_ms,
                local_end_ms=best.local_end_ms,
                global_start_ms=start_ms,
                global_end_ms=end_ms,
                key_global_ms=best.key_global_ms,
                objects=sorted({obj for item in cluster for obj in item.objects}),
                confidence=min(
                    1.0,
                    max(item.confidence for item in cluster)
                    + 0.05 * max(0, len(views) - 1),
                ),
                evidence=[
                    {
                        "source_candidate_id": item.candidate_id,
                        "view_id": item.view_id,
                        "role": item.role.value,
                        "global_start_ms": item.global_start_ms,
                        "global_end_ms": item.global_end_ms,
                        "confidence": item.confidence,
                    }
                    for item in cluster
                ],
                uncertainty=[
                    "Motion sentinel candidate only; YOLO coarse and bounded fine scans must verify it."
                ],
            )
        )
    # A silent or obstructed secondary sentinel must not force a full-timeline
    # YOLO fallback. Keep the original motion windows as a recall-first fallback.
    return fused or list(sorted(candidates, key=candidate_sort_key))


def generate_motion_safety_candidates(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    frame_index: CoarseFrameIndex | None = None,
) -> list[ActionCandidate]:
    """Select separated motion peaks when adaptive thresholding returns nothing."""

    perf = config["performance"]
    peaks_per_hour = max(1, int(perf.get("motion_probe_fallback_peaks_per_hour", 8)))
    separation_ms = float(perf.get("motion_probe_fallback_separation_seconds", 120.0)) * 1000.0
    half_window_ms = float(perf.get("motion_probe_fallback_window_seconds", 60.0)) * 500.0
    candidates: list[ActionCandidate] = []
    for view in views:
        frames = [
            frame
            for frame in _motion_probe_frames(
                view.view_id,
                detection_paths[view.view_id],
                config,
                frame_index,
            )
            if frame.global_ms is not None
        ]
        if not frames:
            continue
        duration_hours = max(1.0, (frames[-1].local_ms - frames[0].local_ms) / 3_600_000.0)
        target = max(1, math.ceil(duration_hours * peaks_per_hour))
        selected: list[FrameEvidence] = []
        for frame in sorted(frames, key=lambda item: item.motion_score, reverse=True):
            assert frame.global_ms is not None
            if any(
                abs(frame.global_ms - (item.global_ms or 0.0)) < separation_ms
                for item in selected
            ):
                continue
            selected.append(frame)
            if len(selected) >= target:
                break
        for frame in sorted(selected, key=lambda item: item.global_ms or 0.0):
            assert frame.global_ms is not None
            candidates.append(
                ActionCandidate(
                    candidate_id=f"MOTION-SAFETY-{view.view_id}-{len(candidates) + 1:06d}",
                    action_type=ActionType.OBJECT_MOVEMENT,
                    view_id=view.view_id,
                    role=view.role,
                    local_start_ms=max(0.0, frame.local_ms - half_window_ms),
                    local_end_ms=frame.local_ms + half_window_ms,
                    global_start_ms=max(0.0, frame.global_ms - half_window_ms),
                    global_end_ms=frame.global_ms + half_window_ms,
                    key_global_ms=frame.global_ms,
                    objects=[],
                    confidence=max(0.35, min(0.70, 0.35 + frame.motion_score / 100.0)),
                    evidence=[
                        {
                            "frame_index": frame.frame_index,
                            "motion_score": frame.motion_score,
                            "fallback": "separated_motion_peak",
                        }
                    ],
                    uncertainty=[
                        "Adaptive motion threshold was empty; bounded YOLO verification is mandatory."
                    ],
                )
            )
    return sorted(candidates, key=candidate_sort_key)


def select_fine_scan_views(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    coarse_candidates: Sequence[ActionCandidate],
    config: dict[str, Any],
    frame_index: CoarseFrameIndex | None = None,
) -> tuple[list[ViewInput], dict[str, dict[str, Any]]]:
    """Select expensive fine-scan views from independent coarse evidence."""
    padding = float(config["performance"]["fine_window_padding_seconds"]) * 1000.0
    global_windows = [
        (candidate.global_start_ms - padding, candidate.global_end_ms + padding)
        for candidate in coarse_candidates
    ]
    selected: list[ViewInput] = []
    report: dict[str, dict[str, Any]] = {}
    for view in views:
        if view.role == ViewRole.FIRST_PERSON:
            selected.append(view)
            report[view.view_id] = {
                "selected": True,
                "reason": "first_person primary boundary sensor",
            }
            continue
        detection_path = detection_paths.get(view.view_id)
        if detection_path is None:
            report[view.view_id] = {
                "selected": False,
                "reason": "not scanned during sentinel coarse stage",
                "anchor_frames": 0,
                "active_anchor_frames": 0,
                "anchor_classes": [],
                "motion_threshold": None,
            }
            continue
        frames = list(
            frame_index.iter_frames(
                view.view_id,
                start_ms=min((item[0] for item in global_windows), default=None),
                end_ms=max((item[1] for item in global_windows), default=None),
            )
            if frame_index is not None
            else iter_frame_evidence(detection_path)
        )
        all_motion = np.asarray([frame.motion_score for frame in frames], dtype=np.float64)
        motion_threshold = max(
            float(np.percentile(all_motion, 80.0)) if len(all_motion) else 0.0,
            float(np.median(all_motion) + 1.0) if len(all_motion) else 1.0,
        )
        anchor_frames = 0
        active_anchor_frames = 0
        anchors: set[str] = set()
        for frame in frames:
            if frame.global_ms is None or not any(start <= frame.global_ms <= end for start, end in global_windows):
                continue
            names = {box.class_name for box in frame.detections}
            frame_anchors = names & SUPPORT_ANCHOR_CLASSES
            if frame_anchors:
                anchor_frames += 1
                anchors.update(frame_anchors)
                if frame.motion_score >= motion_threshold or bool(names & HAND_CLASSES):
                    active_anchor_frames += 1
        eligible = anchor_frames >= 3 and active_anchor_frames >= 2
        if eligible:
            selected.append(view)
        report[view.view_id] = {
            "selected": eligible,
            "reason": (
                "coarse synchronized motion with container/device anchors"
                if eligible
                else "no stable synchronized container/device operation evidence"
            ),
            "anchor_frames": anchor_frames,
            "active_anchor_frames": active_anchor_frames,
            "anchor_classes": sorted(anchors),
            "motion_threshold": motion_threshold,
        }
    configured_minimum = config["performance"].get(
        "fine_min_third_person_views", 1
    )
    minimum_third_views = (
        sum(view.role == ViewRole.THIRD_PERSON for view in views)
        if configured_minimum is None
        else max(1, int(configured_minimum))
    )
    selected_third = [view for view in selected if view.role == ViewRole.THIRD_PERSON]
    if len(selected_third) < minimum_third_views:
        unselected_third = [
            view
            for view in views
            if view.role == ViewRole.THIRD_PERSON and view not in selected
        ]
        ranked = sorted(
            unselected_third,
            key=lambda view: (
                int(report[view.view_id].get("active_anchor_frames", 0)),
                int(report[view.view_id].get("anchor_frames", 0)),
            ),
            reverse=True,
        )
        for view in ranked[: minimum_third_views - len(selected_third)]:
            selected.append(view)
            report[view.view_id]["selected"] = True
            report[view.view_id]["reason"] = (
                "cross-view quality fallback: best available third-person boundary sensor"
            )
    return selected, report


def _appearance_similarity(
    left: ActionCandidate,
    right: ActionCandidate,
) -> float | None:
    left_values = left.instance_signature.get("appearance_signature")
    right_values = right.instance_signature.get("appearance_signature")
    if not isinstance(left_values, list) or not isinstance(right_values, list):
        return None
    width = min(len(left_values), len(right_values))
    if width == 0:
        return None
    left_array = np.asarray(left_values[:width], dtype=np.float64)
    right_array = np.asarray(right_values[:width], dtype=np.float64)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 1e-9:
        return None
    return float(np.dot(left_array, right_array) / denominator)


def _objects_overlap(
    left: ActionCandidate,
    right: ActionCandidate,
    config: dict[str, Any] | None = None,
) -> bool:
    a = set(left.objects) - HAND_CLASSES
    b = set(right.objects) - HAND_CLASSES
    compatible = bool(a & b)
    if not compatible and left.action_type == ActionType.LIQUID_MOVEMENT:
        # Cross-view detectors may call the same transfer tool pipette vs.
        # spearhead and the same vessel tube vs. container. Require both
        # physical families; never merge candidates merely because both are
        # labelled "liquid_movement".
        compatible = bool(
            a & TRANSFER_TOOL_CLASSES and b & TRANSFER_TOOL_CLASSES
        ) and bool(
            a & CONTAINER_CLASSES and b & CONTAINER_CLASSES
        )
    if not compatible and left.action_type == ActionType.DEVICE_PANEL_OPERATION:
        compatible = bool(a & b & DEVICE_CLASSES)
    if not compatible and left.action_type == ActionType.CONTAINER_STATE_CHANGE:
        def families(objects: set[str]) -> set[str]:
            result: set[str] = set()
            if objects & {"tube", "tube_cap"}:
                result.add("tube")
            if objects & {
                "reagent_bottle",
                "reagent_bottle_open",
                "sample_bottle",
                "sample_bottle_blue",
                "bottle_cap",
            }:
                result.add("bottle")
            if objects & {"beaker", "container"}:
                result.add("open_container")
            return result

        compatible = bool(families(a) & families(b))
    if not compatible:
        return False

    left_tracks = set(left.instance_signature.get("track_ids") or [])
    right_tracks = set(right.instance_signature.get("track_ids") or [])
    if (
        left.view_id == right.view_id
        and left_tracks
        and right_tracks
        and not bool(left_tracks & right_tracks)
    ):
        return False

    performance = (config or {}).get("performance", {})
    if (
        left.view_id != right.view_id
        and performance.get(
            "fine_instance_cross_view_conflict_quarantine_enabled", False
        )
    ):
        similarity = _appearance_similarity(left, right)
        if similarity is not None and similarity < float(
            performance.get("fine_instance_minimum_cosine_similarity", 0.25)
        ):
            return False
    return True


def _instance_association_receipt(
    candidates: Sequence[ActionCandidate],
    config: dict[str, Any],
) -> dict[str, Any]:
    pairs = []
    for left, right in itertools.combinations(candidates, 2):
        if left.view_id == right.view_id:
            continue
        similarity = _appearance_similarity(left, right)
        pairs.append(
            {
                "left_candidate_id": left.candidate_id,
                "right_candidate_id": right.candidate_id,
                "left_view_id": left.view_id,
                "right_view_id": right.view_id,
                "left_instance_signature": left.instance_signature,
                "right_instance_signature": right.instance_signature,
                "appearance_cosine_similarity": (
                    round(similarity, 6) if similarity is not None else None
                ),
                "compatible": _objects_overlap(left, right, config),
                "association_basis": (
                    "class_time_and_appearance"
                    if similarity is not None
                    else "class_time_without_cross_view_appearance"
                ),
            }
        )
    return {
        "schema_version": "visioncortex-cross-view-instance-association/1",
        "pair_count": len(pairs),
        "appearance_conflict_quarantine_enabled": bool(
            config["performance"].get(
                "fine_instance_cross_view_conflict_quarantine_enabled", False
            )
        ),
        "minimum_cosine_similarity": float(
            config["performance"].get(
                "fine_instance_minimum_cosine_similarity", 0.25
            )
        ),
        "pairs": pairs,
    }


def _candidate_alignment_profile(
    candidate: ActionCandidate,
    transform: AlignmentTransform,
    *,
    local_segments_enabled: bool,
) -> dict[str, Any]:
    if not local_segments_enabled or not transform.segment_transforms:
        return {
            "state": transform.state,
            "confidence": float(transform.confidence),
            "uncertainty_ms": max(0.0, float(transform.uncertainty_ms)),
            "available": bool(
                transform.state != "failed"
                and transform.is_available_at_global(candidate.key_global_ms)
            ),
            "segment_indexes": [],
            "basis": "view_transform",
        }
    overlapping = [
        segment
        for segment in transform.segment_transforms
        if segment.local_end_ms >= candidate.local_start_ms
        and segment.local_start_ms <= candidate.local_end_ms
    ]
    if not overlapping:
        return {
            "state": "failed",
            "confidence": 0.0,
            "uncertainty_ms": max(0.0, float(transform.uncertainty_ms)),
            "available": False,
            "segment_indexes": [],
            "basis": "no_local_alignment_segment",
        }
    states = {segment.state for segment in overlapping}
    state = (
        "failed"
        if "failed" in states
        else "uncertain"
        if "uncertain" in states
        else "aligned"
    )
    return {
        "state": state,
        "confidence": min(float(segment.confidence) for segment in overlapping),
        "uncertainty_ms": max(
            max(0.0, float(segment.uncertainty_ms))
            for segment in overlapping
        ),
        "available": state != "failed",
        "segment_indexes": [
            int(segment.segment_index) for segment in overlapping
        ],
        "basis": "candidate_local_alignment_segments",
    }


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    ordered = sorted(
        zip(values, weights, strict=True), key=lambda item: item[0]
    )
    if not ordered:
        raise ValueError("weighted quantile requires at least one value")
    total = math.fsum(max(0.0, float(weight)) for _, weight in ordered)
    if total <= 1e-9:
        return float(median(value for value, _ in ordered))
    target = min(1.0, max(0.0, float(quantile))) * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, float(weight))
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def _cluster_confidence_receipt(
    cluster: Sequence[ActionCandidate],
    profiles: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    perf = config["performance"]
    role_weights = dict(perf.get("audit_role_confidence_weights") or {})
    action_offsets = dict(perf.get("audit_action_confidence_offsets") or {})
    best_by_view: dict[str, ActionCandidate] = {}
    for candidate in cluster:
        current = best_by_view.get(candidate.view_id)
        if current is None or candidate.confidence > current.confidence:
            best_by_view[candidate.view_id] = candidate
    contributors = []
    numerator = 0.0
    denominator = 0.0
    for candidate in sorted(best_by_view.values(), key=candidate_sort_key):
        profile = profiles[id(candidate)]
        calibrated = min(
            1.0,
            max(
                0.0,
                float(candidate.confidence)
                + float(
                    action_offsets.get(candidate.action_type.value, 0.0)
                ),
            ),
        )
        role_weight = float(role_weights.get(candidate.role.value, 1.0))
        alignment_weight = max(
            0.10,
            0.25 + 0.75 * max(0.0, float(profile["confidence"])),
        )
        weight = role_weight * alignment_weight
        numerator += calibrated * weight
        denominator += weight
        contributors.append(
            {
                "candidate_id": candidate.candidate_id,
                "view_id": candidate.view_id,
                "role": candidate.role.value,
                "raw_confidence": round(float(candidate.confidence), 6),
                "calibrated_confidence": round(calibrated, 6),
                "role_weight": round(role_weight, 6),
                "alignment_weight": round(alignment_weight, 6),
                "effective_weight": round(weight, 6),
            }
        )
    base = numerator / max(denominator, 1e-9)
    views = {candidate.view_id for candidate in cluster}
    roles = {candidate.role for candidate in cluster}
    independence_bonus = min(
        float(perf.get("audit_maximum_independence_bonus", 0.12)),
        float(perf.get("audit_independent_view_bonus", 0.04))
        * max(0, len(views) - 1)
        + (
            float(perf.get("audit_dual_role_bonus", 0.04))
            if len(roles) == 2
            else 0.0
        ),
    )
    maximum_alignment_penalty = float(
        perf.get("audit_maximum_alignment_penalty", 0.08)
    )
    tolerance = max(
        1.0,
        float(config["alignment"].get("cross_view_event_max_tolerance_ms", 2500.0)),
    )
    uncertainty_ratio = math.fsum(
        min(1.0, float(profiles[id(candidate)]["uncertainty_ms"]) / tolerance)
        for candidate in best_by_view.values()
    ) / max(1, len(best_by_view))
    uncertain_state_ratio = math.fsum(
        profiles[id(candidate)]["state"] != "aligned"
        for candidate in best_by_view.values()
    ) / max(1, len(best_by_view))
    alignment_penalty = min(
        maximum_alignment_penalty,
        maximum_alignment_penalty
        * (0.70 * uncertainty_ratio + 0.30 * uncertain_state_ratio),
    )
    confidence = min(1.0, max(0.0, base + independence_bonus - alignment_penalty))
    return confidence, {
        "schema_version": "visioncortex-audit-confidence/1",
        "method": "best_per_view_alignment_weighted_independent_support",
        "base_confidence": round(base, 6),
        "independence_bonus": round(independence_bonus, 6),
        "alignment_penalty": round(alignment_penalty, 6),
        "final_confidence": round(confidence, 6),
        "contributing_view_count": len(best_by_view),
        "contributors": contributors,
    }


def audit_candidates(
    candidates: Sequence[ActionCandidate],
    transforms: dict[str, AlignmentTransform],
    config: dict[str, Any],
    frame_index: FineFrameIndex | None = None,
) -> tuple[list[EvidenceEvent], list[dict[str, Any]]]:
    motion_rejected = [
        {"candidate_id": c.candidate_id, "reason": "movement_not_supported_by_image",
         "event_id": f"REJECTED-CANDIDATE-{c.candidate_id}", "confidence": c.confidence,
         "movement_visual_verification": c.provenance["movement_visual_verification"]}
        for c in candidates if c.action_type == ActionType.OBJECT_MOVEMENT
        and c.provenance.get("movement_visual_verification", {}).get("status") in {"contradicted", "unverified"}
    ]
    rejected_motion_ids = {c["candidate_id"] for c in motion_rejected}
    candidates = [c for c in candidates if c.candidate_id not in rejected_motion_ids]
    tolerance = float(config["alignment"]["cross_view_event_tolerance_ms"])
    maximum_tolerance = max(
        tolerance,
        float(
            config["alignment"].get(
                "cross_view_event_max_tolerance_ms", tolerance
            )
        ),
    )

    perf = config["performance"]
    local_alignment_enabled = bool(
        perf.get("audit_local_alignment_enabled", False)
    )
    constrained = bool(
        perf.get("audit_constrained_clustering_enabled", False)
    )
    profiles = {
        id(candidate): _candidate_alignment_profile(
            candidate,
            transforms[candidate.view_id],
            local_segments_enabled=local_alignment_enabled,
        )
        for candidate in candidates
    }

    def pair_tolerance(left: ActionCandidate, right: ActionCandidate) -> float:
        if left.view_id == right.view_id:
            return tolerance
        left_error = float(profiles[id(left)]["uncertainty_ms"])
        right_error = float(profiles[id(right)]["uncertainty_ms"])
        propagated = math.sqrt(left_error**2 + right_error**2)
        # Cross-camera co-occurrence must overlap within measured clock error.
        # The same-view joining tolerance is not evidence that another bench
        # observed this action hundreds of milliseconds earlier or later.
        return min(maximum_tolerance, propagated)

    seg_cfg = config["segmentation"]
    ordered_candidates = sorted(candidates, key=candidate_sort_key)
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in ordered_candidates
    }
    sqlite_lookup_enabled = bool(
        perf.get("audit_sqlite_candidate_index_enabled", False)
        and frame_index is not None
    )
    if sqlite_lookup_enabled:
        assert frame_index is not None
        frame_index.replace_audit_candidates(ordered_candidates)
    bucket_ms = max(1000.0, maximum_tolerance)
    candidate_buckets: dict[int, list[ActionCandidate]] = defaultdict(list)
    for candidate in ordered_candidates:
        first_bucket = math.floor(candidate.global_start_ms / bucket_ms)
        last_bucket = math.floor(candidate.global_end_ms / bucket_ms)
        for bucket in range(first_bucket, last_bucket + 1):
            candidate_buckets[bucket].append(candidate)

    def nearby_candidates(start_ms: float, end_ms: float) -> list[ActionCandidate]:
        if sqlite_lookup_enabled:
            assert frame_index is not None
            return [
                candidate_by_id.get(item.candidate_id, item)
                for item in frame_index.iter_audit_candidates(
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            ]
        first_bucket = math.floor(start_ms / bucket_ms)
        last_bucket = math.floor(end_ms / bucket_ms)
        unique: dict[str, ActionCandidate] = {}
        for bucket in range(first_bucket, last_bucket + 1):
            for candidate in candidate_buckets.get(bucket, []):
                if (
                    candidate.global_end_ms >= start_ms
                    and candidate.global_start_ms <= end_ms
                ):
                    unique[candidate.candidate_id] = candidate
        return sorted(unique.values(), key=candidate_sort_key)

    maximum_cluster_span_ms = max(
        maximum_tolerance,
        float(perf.get("audit_max_cluster_span_seconds", 30.0)) * 1000.0,
    )
    clusters: list[list[ActionCandidate]] = []
    for candidate in ordered_candidates:
        selected: list[ActionCandidate] | None = None
        for cluster in reversed(clusters):
            if (
                candidate.global_start_ms
                - max(item.global_end_ms for item in cluster)
                > maximum_tolerance
            ):
                break
            compatible_members = [
                item
                for item in cluster
                if _objects_overlap(candidate, item, config)
            ]
            temporal_matches = [
                candidate.global_start_ms
                <= item.global_end_ms + pair_tolerance(candidate, item)
                and candidate.global_end_ms
                >= item.global_start_ms - pair_tolerance(candidate, item)
                for item in compatible_members
            ]
            temporal_match = bool(temporal_matches) and (
                all(temporal_matches) if constrained else any(temporal_matches)
            )
            complete_identity_match = bool(compatible_members) and (
                not constrained or len(compatible_members) == len(cluster)
            )
            proposed_span = max(
                candidate.global_end_ms,
                max(item.global_end_ms for item in cluster),
            ) - min(
                candidate.global_start_ms,
                min(item.global_start_ms for item in cluster),
            )
            local_alignment_available = bool(
                profiles[id(candidate)]["available"]
                and all(profiles[id(item)]["available"] for item in cluster)
            )
            if (
                cluster[0].action_type == candidate.action_type
                and temporal_match
                and complete_identity_match
                and (not local_alignment_enabled or local_alignment_available)
                and (
                    not constrained
                    or proposed_span <= maximum_cluster_span_ms
                )
            ):
                selected = cluster
                break
        if selected is None:
            clusters.append([candidate])
        else:
            selected.append(candidate)

    events: list[EvidenceEvent] = []
    rejected: list[dict[str, Any]] = list(motion_rejected)
    for index, unsorted_cluster in enumerate(clusters, 1):
        cluster = sorted(unsorted_cluster, key=candidate_sort_key)
        views = sorted({item.view_id for item in cluster})
        roles = sorted({item.role for item in cluster}, key=lambda role: role.value)
        weighted_confidence = math.fsum(item.confidence for item in cluster) / len(cluster)
        cross_view_bonus = min(0.16, 0.06 * (len(views) - 1) + (0.06 if len(roles) == 2 else 0.0))
        legacy_confidence = min(1.0, weighted_confidence + cross_view_bonus)
        if perf.get("audit_explainable_scoring_enabled", False):
            confidence, confidence_receipt = _cluster_confidence_receipt(
                cluster, profiles, config
            )
        else:
            confidence = legacy_confidence
            confidence_receipt = {
                "schema_version": "visioncortex-audit-confidence/1",
                "method": "legacy_candidate_mean_plus_cross_view_bonus",
                "base_confidence": round(weighted_confidence, 6),
                "independence_bonus": round(cross_view_bonus, 6),
                "alignment_penalty": 0.0,
                "final_confidence": round(confidence, 6),
            }
        aligned_views = sorted(
            {
                item.view_id
                for item in cluster
                if profiles[id(item)]["state"] == "aligned"
                and profiles[id(item)]["available"]
            }
        )
        both_roles = len(roles) == 2
        evidence_start_ms = min(item.global_start_ms for item in cluster)
        evidence_end_ms = max(item.global_end_ms for item in cluster)
        duration_ms = evidence_end_ms - evidence_start_ms
        if perf.get("audit_core_interval_enabled", False):
            weights = [max(0.01, float(item.confidence)) for item in cluster]
            core_start_ms = _weighted_quantile(
                [float(item.global_start_ms) for item in cluster],
                weights,
                float(perf.get("audit_core_start_quantile", 0.50)),
            )
            core_end_ms = _weighted_quantile(
                [float(item.global_end_ms) for item in cluster],
                weights,
                float(perf.get("audit_core_end_quantile", 0.50)),
            )
            if core_end_ms < core_start_ms:
                center = float(median(item.key_global_ms for item in cluster))
                core_start_ms = core_end_ms = center
        else:
            core_start_ms = evidence_start_ms
            core_end_ms = evidence_end_ms
        action_specific_thresholds = seg_cfg.get(
            "single_view_action_accept_confidence", {}
        )
        single_view_threshold = float(
            action_specific_thresholds.get(
                cluster[0].action_type.value,
                seg_cfg["single_view_accept_confidence"],
            )
        )
        single_view_strong = (
            len(views) == 1
            and confidence >= single_view_threshold
            and cluster[0].action_type
            in {
                ActionType.HAND_OBJECT_CONTACT,
                ActionType.LIQUID_MOVEMENT,
                ActionType.CONTAINER_STATE_CHANGE,
                ActionType.DEVICE_PANEL_OPERATION,
            }
        )
        semantic_context_candidate: ActionCandidate | None = None
        semantic_context_thresholds = seg_cfg.get(
            "semantic_review_cross_role_context_min_confidence", {}
        )
        semantic_context_minimum = semantic_context_thresholds.get(
            cluster[0].action_type.value
        )
        if (
            bool(seg_cfg.get("semantic_review_cross_role_context_enabled", False))
            and len(views) == 1
            and semantic_context_minimum is not None
            and confidence >= float(semantic_context_minimum)
        ):
            cluster_start_ms = min(item.global_start_ms for item in cluster)
            cluster_end_ms = max(item.global_end_ms for item in cluster)
            context_minimum = float(
                seg_cfg.get(
                    "semantic_review_cross_role_context_support_min_confidence",
                    0.55,
                )
            )
            cluster_ids = {item.candidate_id for item in cluster}
            context_candidates = [
                item
                for item in nearby_candidates(
                    cluster_start_ms - maximum_tolerance,
                    cluster_end_ms + maximum_tolerance,
                )
                if item.candidate_id not in cluster_ids
                and item.role not in roles
                and _candidate_alignment_profile(
                    item,
                    transforms[item.view_id],
                    local_segments_enabled=local_alignment_enabled,
                )["state"]
                == "aligned"
                and item.confidence >= context_minimum
                and item.global_start_ms
                <= cluster_end_ms
                + max(pair_tolerance(item, member) for member in cluster)
                and item.global_end_ms
                >= cluster_start_ms
                - max(pair_tolerance(item, member) for member in cluster)
            ]
            if context_candidates:
                semantic_context_candidate = max(
                    context_candidates,
                    key=lambda item: (
                        min(cluster_end_ms, item.global_end_ms)
                        - max(cluster_start_ms, item.global_start_ms),
                        item.confidence,
                        -abs(item.key_global_ms - median(
                            candidate.key_global_ms for candidate in cluster
                        )),
                    ),
                )
        semantic_context_admitted = semantic_context_candidate is not None
        accepted = bool(aligned_views) and (
            both_roles
            or len(views) >= 2
            or single_view_strong
            or semantic_context_admitted
        )
        if bool(seg_cfg.get("require_both_roles")) and not bool(
            seg_cfg.get("allow_strong_single_role_actions", False)
        ):
            accepted = accepted and both_roles
        formal_status = "formal" if accepted else "rejected"
        if perf.get("audit_formal_admission_status_enabled", False):
            direct_admission = bool(
                accepted
                and (
                    both_roles
                    or len(views) >= 2
                    or single_view_strong
                )
            )
            if accepted and not direct_admission and semantic_context_admitted:
                formal_status = "provisional"
            liquid_sequence_observed = any(
                evidence.get("transfer_sequence")
                == "source_transport_target"
                for item in cluster
                for evidence in item.evidence
            )
            if (
                accepted
                and cluster[0].action_type == ActionType.LIQUID_MOVEMENT
                and perf.get(
                    "audit_liquid_sequence_formal_gate_enabled", False
                )
                and not liquid_sequence_observed
            ):
                formal_status = "provisional"
        uncertainty: list[str] = []
        if not both_roles:
            uncertainty.append("该动作没有同时获得第一与第三人称支持")
        if semantic_context_candidate is not None:
            uncertainty.append(
                "对侧视角仅提供同时动作上下文，不直接证明候选类别；"
                f"必须经语义模型确认：{semantic_context_candidate.candidate_id}"
            )
        uncertain_alignments = [view_id for view_id in views if transforms[view_id].state != "aligned"]
        if uncertain_alignments:
            uncertainty.append(f"以下视角对齐不确定: {', '.join(uncertain_alignments)}")
        if cluster[0].action_type == ActionType.LIQUID_MOVEMENT:
            uncertainty.append("液体移动由传统CV候选提出，最终语义需多模态模型确认")
        if formal_status == "formal":
            if both_roles:
                reason = "第一/第三人称同类动作候选在时钟误差内重叠；跨机位实体对应仍待核验"
            elif len(views) >= 2:
                reason = "至少两路同角色视角的同类动作候选在时钟误差内重叠；跨机位实体对应仍待核验"
            elif semantic_context_candidate is not None:
                reason = (
                    "单路状态线索与对侧同时动作上下文通过语义召回门控；"
                    "候选事实仍需多模态确认"
                )
            else:
                reason = "单路持续强物理证据通过门控；未强制其他空/无效视角产出"
        elif formal_status == "provisional":
            reason = "候选进入待语义复核区；未获得正式实验边界资格"
        else:
            reason = "候选缺少足够的跨视角或持续强物理证据"
        observability: dict[str, Any] = {
            "alignment_association": {
                "schema_version": "visioncortex-alignment-association/1",
                "base_tolerance_ms": tolerance,
                "maximum_tolerance_ms": maximum_tolerance,
                "cross_view_temporal_policy": "measured_clock_uncertainty_only",
                "effective_cross_view_tolerance_ms": max(
                    (pair_tolerance(left, right) for left in cluster for right in cluster
                     if left.view_id != right.view_id), default=0.0
                ),
                "view_uncertainty_ms": {
                    view_id: transforms[view_id].uncertainty_ms
                    for view_id in views
                },
                "effective_cluster_tolerance_ms": max(
                    (
                        pair_tolerance(left, right)
                        for left in cluster
                        for right in cluster
                    ),
                    default=tolerance,
                ),
            },
            "object_instance_association": _instance_association_receipt(
                cluster, config
            ),
            "candidate_alignment": {
                "schema_version": "visioncortex-candidate-local-alignment/1",
                "local_segment_enforced": local_alignment_enabled,
                "candidates": {
                    item.candidate_id: profiles[id(item)] for item in cluster
                },
            },
            "confidence_fusion": confidence_receipt,
            "event_boundary": {
                "schema_version": "visioncortex-event-boundary/1",
                "core_global_start_ms": core_start_ms,
                "core_global_end_ms": core_end_ms,
                "evidence_global_start_ms": evidence_start_ms,
                "evidence_global_end_ms": evidence_end_ms,
                "core_drives_formal_segmentation": bool(
                    perf.get("audit_core_interval_enabled", False)
                ),
            },
            "audit_lookup": {
                "strategy": (
                    "sqlite_time_index"
                    if sqlite_lookup_enabled
                    else "bounded_time_buckets"
                ),
                "bucket_ms": None if sqlite_lookup_enabled else bucket_ms,
            },
        }
        if semantic_context_candidate is not None:
            observability["semantic_recall_admission"] = {
                "schema_version": "visioncortex-semantic-recall-admission/1",
                "mode": "single_view_state_plus_cross_role_activity",
                "candidate_action_directly_confirmed": False,
                "mandatory_semantic_review": True,
                "context_candidate_id": semantic_context_candidate.candidate_id,
                "context_view_id": semantic_context_candidate.view_id,
                "context_role": semantic_context_candidate.role.value,
                "context_action_type": semantic_context_candidate.action_type.value,
                "context_confidence": semantic_context_candidate.confidence,
                "context_global_start_ms": semantic_context_candidate.global_start_ms,
                "context_global_end_ms": semantic_context_candidate.global_end_ms,
            }
        event = EvidenceEvent(
            event_id=f"EVT-{index:06d}",
            action_type=cluster[0].action_type,
            global_start_ms=core_start_ms,
            global_end_ms=core_end_ms,
            key_global_ms=float(median(item.key_global_ms for item in cluster)),
            objects=sorted({obj for item in cluster for obj in item.objects}),
            confidence=confidence,
            accepted=formal_status != "rejected",
            formal_admission_status=formal_status,
            audit_reason=reason,
            supporting_views=views,
            supporting_roles=roles,
            candidates=cluster,
            uncertainty=uncertainty,
            observability=observability,
            core_global_start_ms=core_start_ms,
            core_global_end_ms=core_end_ms,
            evidence_global_start_ms=evidence_start_ms,
            evidence_global_end_ms=evidence_end_ms,
        )
        event.event_fingerprint = stable_event_fingerprint(event)
        if perf.get("audit_stable_event_ids_enabled", False):
            event.event_id = f"EVT-{event.event_fingerprint[:16].upper()}"
        events.append(event)
        if formal_status == "rejected":
            rejected.append(
                {
                    "event_id": event.event_id,
                    "candidate_ids": [item.candidate_id for item in cluster],
                    "reason": reason,
                    "confidence": confidence,
                    "duration_ms": duration_ms,
                    "formal_admission_status": formal_status,
                    "event_fingerprint": event.event_fingerprint,
                }
            )
    return events, rejected


def refine_liquid_events_with_context(
    events: Sequence[EvidenceEvent],
    detection_paths: dict[str, Path],
    context_ms: float = 1000.0,
    frame_index: FineFrameIndex | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Reject single-view liquid hypotheses without nearby visible hand evidence.

    Liquid is not a YOLO class in the 21-class models. A tool/vessel ROI-motion
    coincidence is therefore recall-only. Cross-view matches remain accepted;
    single-view candidates must additionally show a hand in the temporal
    neighborhood before they may influence experiment bounds.
    """
    performance = (config or {}).get("performance", {})
    spatial_enabled = bool(
        performance.get("audit_liquid_spatial_context_enabled", False)
    )
    maximum_gap = float(
        performance.get("audit_liquid_context_hand_object_gap_norm", 0.08)
    )
    minimum_frames = max(
        1,
        int(performance.get("audit_liquid_context_minimum_frames", 2)),
    )
    indexes: dict[str, tuple[list[float], list[FrameEvidence]]] = {}
    rejected: list[dict[str, Any]] = []
    for event in events:
        if not event.accepted or event.action_type != ActionType.LIQUID_MOVEMENT:
            continue
        supported: list[str] = []
        hand_counts: dict[str, int] = {}
        spatial_counts: dict[str, int] = {}
        for view_id in event.supporting_views:
            view_candidates = [
                candidate
                for candidate in event.candidates
                if candidate.view_id == view_id
            ]
            track_ids = {
                int(value)
                for candidate in view_candidates
                for evidence in candidate.evidence
                for key in (
                    "object_track_id",
                    "tool_track_id",
                    "vessel_track_id",
                    "source_track_id",
                    "target_track_id",
                )
                if isinstance((value := evidence.get(key)), int)
            }
            object_classes = set(event.objects) - HAND_CLASSES
            if frame_index is not None:
                nearby_frames = list(
                    frame_index.iter_global_frames(
                        view_id,
                        start_ms=event.global_start_ms - context_ms,
                        end_ms=event.global_end_ms + context_ms,
                    )
                )
            else:
                if view_id not in indexes:
                    frames = [
                        frame
                        for frame in iter_frame_evidence(
                            detection_paths[view_id]
                        )
                        if frame.global_ms is not None
                    ]
                    indexes[view_id] = (
                        [float(frame.global_ms) for frame in frames],
                        frames,
                    )
                times, frames = indexes[view_id]
                left = bisect_left(
                    times, event.global_start_ms - context_ms
                )
                right = bisect_left(
                    times, event.global_end_ms + context_ms
                )
                nearby_frames = frames[left:right]
            hand_count = 0
            spatial_count = 0
            for frame in nearby_frames:
                hands = [
                    box
                    for box in frame.detections
                    if box.class_name in HAND_CLASSES
                ]
                if hands:
                    hand_count += 1
                relevant_objects = [
                    box
                    for box in frame.detections
                    if box.class_name in object_classes
                    and (
                        not track_ids
                        or (
                            box.track_id is not None
                            and int(box.track_id) in track_ids
                        )
                    )
                ]
                if hands and relevant_objects and any(
                    _box_distance(hand, obj) <= maximum_gap
                    for hand in hands
                    for obj in relevant_objects
                ):
                    spatial_count += 1
            hand_counts[view_id] = hand_count
            spatial_counts[view_id] = spatial_count
            qualifying_count = spatial_count if spatial_enabled else hand_count
            if qualifying_count >= (minimum_frames if spatial_enabled else 1):
                supported.append(view_id)
        event.observability["liquid_context"] = {
            "schema_version": "visioncortex-liquid-spatial-context/1",
            "spatial_association_required": spatial_enabled,
            "maximum_hand_object_gap_norm": maximum_gap,
            "minimum_supporting_frames": minimum_frames,
            "hand_frame_counts": hand_counts,
            "spatial_support_frame_counts": spatial_counts,
        }
        if supported:
            removed_views = sorted(set(event.supporting_views) - set(supported))
            if removed_views:
                retained_candidates = [item for item in event.candidates if item.view_id in supported]
                previous_confidence = event.confidence
                event.confidence = min(event.confidence, max(
                    (item.confidence for item in retained_candidates), default=0.0
                ))
                # A removed source invalidates the earlier consensus/admission.
                # Keep the hypothesis available, but require a fresh semantic decision.
                set_event_admission(event, "provisional")
                event.observability["liquid_context"].update({
                    "removed_supporting_views": removed_views,
                    "previous_confidence": previous_confidence,
                    "retained_confidence": event.confidence,
                    "confidence_policy": "cap_to_retained_source_confidence",
                    "admission_after_support_filter": "provisional",
                })
                event.audit_reason = "液体候选部分视角未通过手部空间核验，原跨视角裁决失效，保留待核验"
                event.uncertainty.append("支持来源减少后须重新裁决，不能沿用此前跨视角一致结论")
            event.supporting_views = supported
            event.supporting_roles = sorted(
                {candidate.role for candidate in event.candidates if candidate.view_id in supported},
                key=lambda role: role.value,
            )
            event.audit_reason += (
                f"；液体候选手部上下文={hand_counts}，"
                f"手-工具/容器空间支持={spatial_counts}"
            )
            continue
        set_event_admission(event, "rejected")
        event.audit_reason = (
            "液体运动候选的全部支持视角均缺少满足门槛的邻近手-工具/容器证据，"
            "不能用于实验边界"
        )
        event.uncertainty.append("保留为 CV 候选，未进入关键素材；液体语义不得仅由 ROI 运动推断")
        rejected.append(
            {
                "event_id": event.event_id,
                "candidate_ids": [item.candidate_id for item in event.candidates],
                "reason": event.audit_reason,
                "confidence": event.confidence,
                "duration_ms": event.global_end_ms - event.global_start_ms,
                "formal_admission_status": event.formal_admission_status,
                "hand_frame_counts": hand_counts,
                "spatial_support_frame_counts": spatial_counts,
            }
        )
    return rejected


def _split_rich_repeated_primary_sequences(
    groups: Sequence[Sequence[EvidenceEvent]],
    config: dict[str, Any],
    decision_receipts: list[dict[str, Any]] | None = None,
) -> list[list[EvidenceEvent]]:
    """Split two complete workflows that one coarse envelope fused together.

    Coarse motion recall intentionally favors wide envelopes.  A fixed event
    gap cannot distinguish a long pause inside one workflow from two adjacent
    workflows, so this rule is deliberately stricter: both sides must be rich,
    both roles must remain observable on each side, and both sides must repeat
    the same configured primary process action.  The largest eligible inactive
    interval is selected deterministically and the rule is applied recursively.
    """

    continuity_cfg = config["continuity"]
    minimum_gap_ms = float(
        continuity_cfg.get("atomic_sequence_split_min_gap_seconds", 5.0)
    ) * 1000.0
    gap_by_action_ms = {
        str(name): float(value) * 1000.0
        for name, value in (
            continuity_cfg.get(
                "atomic_sequence_split_min_gap_seconds_by_action"
            )
            or {}
        ).items()
    }
    minimum_events = max(
        1,
        int(continuity_cfg.get("atomic_sequence_split_min_events_per_side", 4)),
    )
    primary_action_names = {
        str(value)
        for value in continuity_cfg.get(
            "atomic_fragment_repeated_primary_actions",
            [ActionType.LIQUID_MOVEMENT.value],
        )
    }
    required_roles = {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}

    def primary_actions(items: Sequence[EvidenceEvent]) -> set[str]:
        return {
            event.action_type.value
            for event in items
            if event.action_type.value in primary_action_names
        }

    def observed_roles(items: Sequence[EvidenceEvent]) -> set[ViewRole]:
        return {role for event in items for role in event.supporting_roles}

    def split_one(items: Sequence[EvidenceEvent]) -> list[list[EvidenceEvent]]:
        ordered = sorted(items, key=event_sort_key)
        candidates: list[
            tuple[
                float,
                int,
                list[EvidenceEvent],
                list[EvidenceEvent],
                list[str],
                float,
            ]
        ] = []
        for position in range(minimum_events, len(ordered) - minimum_events + 1):
            left = ordered[:position]
            right = ordered[position:]
            left_end_ms = max(event.global_end_ms for event in left)
            right_start_ms = min(event.global_start_ms for event in right)
            inactive_gap_ms = right_start_ms - left_end_ms
            repeated_primary_actions = sorted(
                primary_actions(left) & primary_actions(right)
            )
            if not repeated_primary_actions:
                continue
            required_gap_ms = max(
                [minimum_gap_ms]
                + [gap_by_action_ms.get(action, minimum_gap_ms) for action in repeated_primary_actions]
            )
            if inactive_gap_ms < required_gap_ms:
                continue
            if not (
                required_roles <= observed_roles(left)
                and required_roles <= observed_roles(right)
            ):
                continue
            candidates.append(
                (
                    inactive_gap_ms,
                    position,
                    left,
                    right,
                    repeated_primary_actions,
                    required_gap_ms,
                )
            )
        if not candidates:
            return [ordered]

        # Prefer the strongest physical inactivity boundary.  For an exact tie,
        # the earlier boundary wins so the result is stable across input order.
        inactive_gap_ms, position, left, right, repeated, required_gap_ms = max(
            candidates,
            key=lambda item: (item[0] / max(item[5], 1.0), item[0], -item[1]),
        )
        if decision_receipts is not None:
            decision_receipts.append(
                decision_receipt(
                    decision_type="raw_atomic_sequence_split",
                    rule_id="QF1-RICH-REPEATED-PRIMARY-SEQUENCE-SPLIT",
                    verdict="split",
                    subject_ids=[event.event_id for event in ordered],
                    reason_codes=[
                        "rich_dual_view_sequences_repeat_primary_action_across_inactive_gap"
                    ],
                    facts={
                        "split_position": position,
                        "inactive_gap_ms": inactive_gap_ms,
                        "left_event_ids": [event.event_id for event in left],
                        "right_event_ids": [event.event_id for event in right],
                        "left_event_fingerprints": [
                            stable_event_fingerprint(event) for event in left
                        ],
                        "right_event_fingerprints": [
                            stable_event_fingerprint(event) for event in right
                        ],
                        "left_roles": sorted(role.value for role in observed_roles(left)),
                        "right_roles": sorted(role.value for role in observed_roles(right)),
                        "repeated_primary_actions": repeated,
                        "required_gap_ms_for_repeated_actions": required_gap_ms,
                    },
                    thresholds={
                        "minimum_inactive_gap_ms": minimum_gap_ms,
                        "minimum_inactive_gap_ms_by_action": gap_by_action_ms,
                        "minimum_events_per_side": minimum_events,
                        "required_roles_per_side": sorted(
                            role.value for role in required_roles
                        ),
                        "configured_primary_actions": sorted(primary_action_names),
                    },
                    evidence_refs=[event.event_id for event in ordered],
                    legacy={
                        "decision": "split_fused_complete_action_sequences",
                    },
                )
            )
        return [*split_one(left), *split_one(right)]

    split_groups: list[list[EvidenceEvent]] = []
    for group in groups:
        split_groups.extend(split_one(group))
    return split_groups


def _event_core_interval(event: EvidenceEvent) -> tuple[float, float]:
    """Return a valid core interval, tolerating legacy ``model_copy`` fixtures."""

    core_start = event.core_global_start_ms
    core_end = event.core_global_end_ms
    if core_start is None or core_end is None:
        return float(event.global_start_ms), float(event.global_end_ms)
    core_start = float(core_start)
    core_end = float(core_end)
    if (
        core_end < core_start
        or core_end < event.global_start_ms
        or core_start > event.global_end_ms
    ):
        return float(event.global_start_ms), float(event.global_end_ms)
    return core_start, core_end


def build_experiment_segments(
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    config: dict[str, Any],
    coarse_windows: Sequence[ActionCandidate] | None = None,
    decision_receipts: list[dict[str, Any]] | None = None,
) -> list[ExperimentSegment]:
    component_only = sorted(
        (
            event
            for event in events
            if event_is_formal(event)
            and str(
                ((event.state_machine or {}).get("publication") or {}).get(
                    "status"
                )
                or ""
            )
            == "component_only"
        ),
        key=event_sort_key,
    )
    # Higher-level state-machine actions keep their lower-level contacts and
    # movements in the audit ledger, but those duplicate components must not
    # open, extend, bridge, or merge experiment boundaries.  Otherwise a long
    # static hand/tool proximity can fuse two independently completed workflows
    # even though its containing liquid/panel action has already superseded it.
    accepted = sorted(
        (
            event
            for event in events
            if event_is_formal(event) and event not in component_only
        ),
        key=event_sort_key,
    )
    if component_only and decision_receipts is not None:
        decision_receipts.append(
            decision_receipt(
                decision_type="component_publication_boundary_quarantine",
                rule_id="QF1-COMPONENT-ONLY-NONBOUNDING",
                verdict="quarantined",
                subject_ids=[event.event_id for event in component_only],
                reason_codes=["superseded_components_cannot_define_boundaries"],
                facts={
                    "component_event_ids": [
                        event.event_id for event in component_only
                    ],
                    "superseding_event_ids": sorted(
                        {
                            str(
                                (
                                    (event.state_machine or {}).get(
                                        "publication"
                                    )
                                    or {}
                                ).get("suppressed_by_event_id")
                            )
                            for event in component_only
                            if (
                                (event.state_machine or {}).get("publication")
                                or {}
                            ).get("suppressed_by_event_id")
                        }
                    ),
                    "formal_membership_changed": True,
                    "audit_ledger_membership_changed": False,
                },
                evidence_refs=[event.event_id for event in component_only],
            )
        )
    if not accepted:
        return []
    cfg = config["segmentation"]
    default_gap_ms = float(cfg["experiment_gap_seconds"]) * 1000.0
    action_gap_seconds = {
        str(name): float(value)
        for name, value in (
            cfg.get("experiment_gap_seconds_by_action") or {}
        ).items()
    }
    identity_idle_bridge_enabled = bool(
        cfg.get("identity_idle_bridge_enabled", True)
    )
    identity_idle_bridge_max_gap_ms = float(
        cfg.get("identity_idle_bridge_max_gap_seconds", 300.0)
    ) * 1000.0
    identity_idle_bridge_minimum = max(
        2,
        int(cfg.get("identity_idle_bridge_min_shared_identities", 2)),
    )
    groups: list[list[EvidenceEvent]] = []
    ordered_windows = (
        sorted(coarse_windows, key=candidate_sort_key) if coarse_windows else []
    )
    if ordered_windows:
        # A coarse motion burst is the experiment-level temporal envelope. Fine
        # actions may legitimately contain long pauses (incubation, reading a
        # balance, changing tools), so a fixed short event gap must not split one
        # bounded experiment. Padding is used only for assigning fine evidence;
        # the final boundary still comes from accepted fine events below.
        padding_ms = float(config["performance"]["fine_window_padding_seconds"]) * 1000.0
        buckets: list[list[EvidenceEvent]] = [[] for _ in ordered_windows]
        unassigned: list[EvidenceEvent] = []
        active_windows: list[tuple[int, ActionCandidate]] = []
        window_cursor = 0
        for event in accepted:
            event_start_ms, event_end_ms = _event_core_interval(event)
            while (
                window_cursor < len(ordered_windows)
                and ordered_windows[window_cursor].global_start_ms - padding_ms
                <= event_end_ms
            ):
                active_windows.append(
                    (window_cursor, ordered_windows[window_cursor])
                )
                window_cursor += 1
            active_windows = [
                item
                for item in active_windows
                if item[1].global_end_ms + padding_ms >= event_start_ms
            ]
            matches = [
                (index, window)
                for index, window in active_windows
                if window.global_start_ms - padding_ms <= event_end_ms
                and window.global_end_ms + padding_ms >= event_start_ms
            ]
            if not matches:
                unassigned.append(event)
                continue
            event_span_ms = max(1.0, event_end_ms - event_start_ms)
            best, _best_window = max(
                matches,
                key=lambda item: (
                    max(
                        0.0,
                        min(event_end_ms, item[1].global_end_ms)
                        - max(event_start_ms, item[1].global_start_ms),
                    )
                    / event_span_ms,
                    int(item[1].view_id in event.supporting_views),
                    -abs(
                        (event_start_ms + event_end_ms) / 2.0
                        - (
                            item[1].global_start_ms
                            + item[1].global_end_ms
                        )
                        / 2.0
                    ),
                    stable_candidate_fingerprint(item[1]),
                ),
            )
            buckets[best].append(event)

        def effective_gap_ms(
            previous: EvidenceEvent,
            current: EvidenceEvent,
        ) -> float:
            return max(
                default_gap_ms,
                action_gap_seconds.get(previous.action_type.value, 0.0)
                * 1000.0,
                action_gap_seconds.get(current.action_type.value, 0.0)
                * 1000.0,
            )

        def can_bridge_identity_idle(
            current: Sequence[EvidenceEvent],
            event: EvidenceEvent,
            gap_ms: float,
            *,
            same_recalled_window: bool,
        ) -> tuple[bool, list[tuple[str, str, int]]]:
            if (
                not identity_idle_bridge_enabled
                or gap_ms < 0.0
                or gap_ms > identity_idle_bridge_max_gap_ms
            ):
                return False, []
            current_identities = {
                identity
                for prior in current
                for identity in event_stable_identities(prior)
            }
            shared_identities = sorted(
                current_identities & event_stable_identities(event)
            )
            if len(shared_identities) < identity_idle_bridge_minimum:
                return False, shared_identities
            both_roles = {
                ViewRole.FIRST_PERSON,
                ViewRole.THIRD_PERSON,
            }
            current_roles = {
                role for prior in current for role in prior.supporting_roles
            }
            if not (
                both_roles.issubset(current_roles)
                and both_roles.issubset(set(event.supporting_roles))
            ):
                return False, shared_identities
            if same_recalled_window:
                return True, shared_identities
            # Without a recalled window, extend only an action whose observed
            # state explicitly ended incomplete. A completed stationary bottle
            # must not hold an experiment open by identity alone.
            prior_state = str(
                (current[-1].state_machine or {}).get("lifecycle_state") or ""
            )
            return prior_state == "incomplete_end", shared_identities

        def append_temporal_groups(
            items: Sequence[EvidenceEvent],
            *,
            same_recalled_window: bool,
        ) -> None:
            current: list[EvidenceEvent] = []
            current_end_ms = 0.0
            for event in sorted(items, key=event_sort_key):
                event_start_ms, event_end_ms = _event_core_interval(event)
                observed_gap_ms = event_start_ms - current_end_ms if current else 0.0
                gap_limit_ms = (
                    effective_gap_ms(current[-1], event)
                    if current
                    else default_gap_ms
                )
                bridged = False
                shared_identities: list[tuple[str, str, int]] = []
                if current and observed_gap_ms > gap_limit_ms:
                    bridged, shared_identities = can_bridge_identity_idle(
                        current,
                        event,
                        observed_gap_ms,
                        same_recalled_window=same_recalled_window,
                    )
                if current and observed_gap_ms > gap_limit_ms and not bridged:
                    groups.append(current)
                    current = []
                    current_end_ms = 0.0
                elif bridged and decision_receipts is not None:
                    decision_receipts.append(
                        decision_receipt(
                            decision_type="adaptive_experiment_idle_bridge",
                            rule_id="QF1-STABLE-IDENTITY-IDLE-BRIDGE",
                            verdict="accepted",
                            subject_ids=[current[-1].event_id, event.event_id],
                            reason_codes=[
                                "same_recalled_window_and_multiple_stable_identities"
                                if same_recalled_window
                                else "incomplete_action_and_multiple_stable_identities"
                            ],
                            facts={
                                "gap_ms": observed_gap_ms,
                                "shared_track_identities": [
                                    {
                                        "view_id": view_id,
                                        "object": object_name,
                                        "track_id": track_id,
                                    }
                                    for view_id, object_name, track_id in shared_identities
                                ],
                                "same_recalled_window": same_recalled_window,
                            },
                            thresholds={
                                "ordinary_gap_limit_ms": gap_limit_ms,
                                "identity_idle_bridge_maximum_gap_ms": (
                                    identity_idle_bridge_max_gap_ms
                                ),
                                "minimum_shared_identities": (
                                    identity_idle_bridge_minimum
                                ),
                            },
                            evidence_refs=[current[-1].event_id, event.event_id],
                        )
                    )
                current.append(event)
                current_end_ms = max(current_end_ms, event_end_ms)
            if current:
                groups.append(current)

        for bucket in buckets:
            append_temporal_groups(bucket, same_recalled_window=True)
        # Retain independently strong evidence that falls outside a recalled
        # window, but group it conservatively by the legacy event-gap rule.
        append_temporal_groups(unassigned, same_recalled_window=False)
        groups.sort(
            key=lambda group: (
                min(event.global_start_ms for event in group),
                tuple(stable_event_fingerprint(event) for event in sorted(group, key=event_sort_key)),
            )
        )
    else:
        def no_window_gap_ms(previous: EvidenceEvent, current: EvidenceEvent) -> float:
            return max(
                default_gap_ms,
                action_gap_seconds.get(previous.action_type.value, 0.0) * 1000.0,
                action_gap_seconds.get(current.action_type.value, 0.0) * 1000.0,
            )

        current: list[EvidenceEvent] = []
        current_end_ms = 0.0
        for event in accepted:
            event_start_ms, event_end_ms = _event_core_interval(event)
            if current:
                observed_gap_ms = event_start_ms - current_end_ms
                gap_limit_ms = no_window_gap_ms(current[-1], event)
                current_identities = {
                    identity
                    for prior in current
                    for identity in event_stable_identities(prior)
                }
                shared_identities = sorted(
                    current_identities & event_stable_identities(event)
                )
                prior_incomplete = str(
                    (current[-1].state_machine or {}).get("lifecycle_state") or ""
                ) == "incomplete_end"
                identity_bridge = bool(
                    identity_idle_bridge_enabled
                    and observed_gap_ms <= identity_idle_bridge_max_gap_ms
                    and len(shared_identities) >= identity_idle_bridge_minimum
                    and prior_incomplete
                    and {
                        ViewRole.FIRST_PERSON,
                        ViewRole.THIRD_PERSON,
                    }.issubset(set(event.supporting_roles))
                )
                if observed_gap_ms > gap_limit_ms and not identity_bridge:
                    groups.append(current)
                    current = []
                    current_end_ms = 0.0
                elif observed_gap_ms > gap_limit_ms and identity_bridge and decision_receipts is not None:
                    decision_receipts.append(
                        decision_receipt(
                            decision_type="adaptive_experiment_idle_bridge",
                            rule_id="QF1-STABLE-IDENTITY-IDLE-BRIDGE",
                            verdict="accepted",
                            subject_ids=[current[-1].event_id, event.event_id],
                            reason_codes=[
                                "incomplete_action_and_multiple_stable_identities"
                            ],
                            facts={
                                "gap_ms": observed_gap_ms,
                                "same_recalled_window": False,
                            },
                            thresholds={
                                "ordinary_gap_limit_ms": gap_limit_ms,
                                "identity_idle_bridge_maximum_gap_ms": (
                                    identity_idle_bridge_max_gap_ms
                                ),
                            },
                            evidence_refs=[current[-1].event_id, event.event_id],
                        )
                    )
            current.append(event)
            current_end_ms = max(current_end_ms, event_end_ms)
        if current:
            groups.append(current)

    groups = _split_rich_repeated_primary_sequences(
        groups,
        config,
        decision_receipts=decision_receipts,
    )
    groups.sort(
        key=lambda group: (
            min(event.global_start_ms for event in group),
            tuple(
                stable_event_fingerprint(event)
                for event in sorted(group, key=event_sort_key)
            ),
        )
    )

    def core_start_ms(event: EvidenceEvent) -> float:
        return _event_core_interval(event)[0]

    def core_end_ms(event: EvidenceEvent) -> float:
        return _event_core_interval(event)[1]

    by_event = {event.event_id: event for event in events}
    segments: list[ExperimentSegment] = []
    for index, unsorted_group in enumerate(groups, 1):
        segment_id = f"EXP-{index:04d}"
        group = sorted(unsorted_group, key=event_sort_key)
        # A segment must start on a physically meaningful operation anchor.
        # Single-view liquid hypotheses and movement of fixed equipment may be
        # useful context, but are too noisy to pull the experiment boundary
        # earlier by themselves.
        start_anchors = select_formal_experiment_start_events(group, config)
        raw_start = min(
            core_start_ms(event) for event in (start_anchors or group)
        )
        leading_context: list[EvidenceEvent] = []
        if start_anchors and bool(
            cfg.get("accepted_leading_context_enabled", False)
        ):
            # A full-timeline scan often observes a reliable single-role
            # preparation chain before the first dual-role opener.  Recover
            # that boundary evidence only when it is already accepted,
            # publication-primary (not semantic-provisional), and connected
            # to the opener by a tight temporal chain.  This changes the clip
            # boundary and membership but never promotes rejected evidence.
            maximum_leading_gap_ms = float(
                cfg.get("accepted_leading_context_max_gap_seconds", 10.0)
            ) * 1000.0
            maximum_leading_extension_ms = float(
                cfg.get("accepted_leading_context_max_extension_seconds", 90.0)
            ) * 1000.0
            lower_limit_ms = max(0.0, raw_start - maximum_leading_extension_ms)
            cursor_ms = raw_start
            eligible_leading = [
                event
                for event in group
                if core_start_ms(event) < raw_start
                and core_end_ms(event) >= lower_limit_ms
                and str(
                    ((event.state_machine or {}).get("publication") or {}).get(
                        "status"
                    )
                    or "primary"
                )
                == "primary"
                and not bool(
                    ((event.observability or {}).get("semantic_recall_admission") or {}).get(
                        "mandatory_semantic_review"
                    )
                )
            ]
            for event in sorted(
                eligible_leading,
                key=lambda item: (core_end_ms(item), core_start_ms(item)),
                reverse=True,
            ):
                if core_start_ms(event) >= cursor_ms:
                    continue
                if core_end_ms(event) < cursor_ms - maximum_leading_gap_ms:
                    break
                leading_context.append(event)
                cursor_ms = min(cursor_ms, core_start_ms(event))
            if leading_context:
                leading_context.sort(key=event_sort_key)
                raw_start = min(
                    raw_start,
                    min(core_start_ms(event) for event in leading_context),
                )
        start = max(0.0, raw_start - float(cfg["experiment_pre_roll_seconds"]) * 1000.0)
        # Accepted recall evidence may precede the first reliable operation
        # anchor. Keep it in the global audit ledger, but do not attach an event
        # that ends before the bounded clip starts to this experiment.
        bounded_group = [event for event in group if core_end_ms(event) >= start]
        raw_end = max(core_end_ms(event) for event in bounded_group)
        boundary_context_receipt: dict[str, Any] | None = None
        core_roles = {
            role for event in bounded_group for role in event.supporting_roles
        }
        if ordered_windows and core_roles == {
            ViewRole.FIRST_PERSON,
            ViewRole.THIRD_PERSON,
        }:
            midpoint = (raw_start + raw_end) / 2.0
            matching_windows = [
                window
                for window in ordered_windows
                if window.global_start_ms <= midpoint <= window.global_end_ms
            ]
            if matching_windows:
                boundary_window = min(
                    matching_windows,
                    key=lambda window: (
                        abs(
                            midpoint
                            - (
                                window.global_start_ms + window.global_end_ms
                            )
                            / 2.0
                        ),
                        candidate_sort_key(window),
                    ),
                )
                later_core_in_window = any(
                    other is not unsorted_group
                    and min(item.global_start_ms for item in other) > raw_end
                    and boundary_window.global_start_ms
                    <= (
                        min(item.global_start_ms for item in other)
                        + max(item.global_end_ms for item in other)
                    )
                    / 2.0
                    <= boundary_window.global_end_ms
                    for other in groups
                )
                activation_gap_ms = float(
                    cfg.get("boundary_context_activation_gap_seconds", 30.0)
                ) * 1000.0
                if (
                    not later_core_in_window
                    and boundary_window.global_end_ms - raw_end >= activation_gap_ms
                ):
                    bridge_confidence = float(
                        cfg.get("boundary_context_bridge_confidence", 0.50)
                    )
                    extension_confidence = float(
                        cfg.get("boundary_context_min_confidence", 0.65)
                    )
                    maximum_gap_ms = float(
                        cfg.get("boundary_context_max_gap_seconds", 10.0)
                    ) * 1000.0
                    maximum_extension_ms = float(
                        cfg.get("boundary_context_max_extension_seconds", 90.0)
                    ) * 1000.0
                    cleanup_objects = {
                        "brush",
                        "cleaning_tool",
                        "sink",
                        "wash_bottle",
                        "waste_container",
                    }
                    cursor = raw_end
                    supported_end = raw_end
                    limit = min(
                        raw_end + maximum_extension_ms,
                        boundary_window.global_end_ms,
                    )
                    context_chain: list[EvidenceEvent] = []
                    strong_context: list[EvidenceEvent] = []
                    for context in sorted(events, key=event_sort_key):
                        if context.accepted or context.global_end_ms <= cursor:
                            continue
                        if context.global_start_ms > limit:
                            break
                        if context.global_start_ms > cursor + maximum_gap_ms:
                            break
                        context_roles = set(context.supporting_roles)
                        cleanup_evidence = sorted(
                            set(context.objects) & cleanup_objects
                        )
                        if (
                            len(context_roles) != 1
                            or context.confidence < bridge_confidence
                            or not cleanup_evidence
                        ):
                            continue
                        context_chain.append(context)
                        cursor = min(limit, max(cursor, context.global_end_ms))
                        if context.confidence >= extension_confidence:
                            strong_context.append(context)
                            supported_end = max(supported_end, cursor)
                    if supported_end > raw_end and strong_context:
                        previous_end = raw_end
                        raw_end = supported_end
                        boundary_context_receipt = decision_receipt(
                            decision_type="raw_boundary_context_extension",
                            rule_id="QF1-SINGLE-VIEW-BOUNDARY-CONTEXT",
                            verdict="accepted",
                            subject_ids=[segment_id],
                            reason_codes=[
                                "single_role_cleanup_chain_inside_recalled_boundary"
                            ],
                            facts={
                                "previous_raw_end_ms": previous_end,
                                "extended_raw_end_ms": raw_end,
                                "context_event_ids": [
                                    event.event_id for event in context_chain
                                ],
                                "context_event_fingerprints": [
                                    stable_event_fingerprint(event)
                                    for event in context_chain
                                ],
                                "strong_context_event_ids": [
                                    event.event_id for event in strong_context
                                ],
                                "cleanup_objects": sorted(
                                    {
                                        obj
                                        for event in context_chain
                                        for obj in event.objects
                                        if obj in cleanup_objects
                                    }
                                ),
                                "boundary_candidate_id": boundary_window.candidate_id,
                                "boundary_candidate_fingerprint": (
                                    stable_candidate_fingerprint(boundary_window)
                                ),
                                "formal_membership_changed": False,
                                "boundary_clipped_to_recalled_window": (
                                    raw_end == boundary_window.global_end_ms
                                ),
                            },
                            thresholds={
                                "activation_gap_ms": activation_gap_ms,
                                "bridge_confidence": bridge_confidence,
                                "extension_confidence": extension_confidence,
                                "maximum_context_gap_ms": maximum_gap_ms,
                                "maximum_extension_ms": maximum_extension_ms,
                            },
                            evidence_refs=[
                                *[event.event_id for event in bounded_group],
                                *[event.event_id for event in context_chain],
                            ],
                            legacy={
                                "segment_id": segment_id,
                                "decision": "extended_raw_boundary_from_cleanup_context",
                            },
                        )
        end = raw_end + float(cfg["experiment_post_roll_seconds"]) * 1000.0
        minimum = float(cfg["min_experiment_seconds"]) * 1000.0
        if end - start < minimum:
            padding = (minimum - (end - start)) / 2.0
            start, end = max(0.0, start - padding), end + padding
        candidate_counts = defaultdict(int)
        direct_evidence_ms = defaultdict(float)
        semantic_review_admitted_views: set[str] = set()
        for event in bounded_group:
            semantic_admission = (event.observability or {}).get(
                "semantic_recall_admission"
            )
            for view_id in event.supporting_views:
                candidate_counts[view_id] += 1
                direct_evidence_ms[view_id] += max(
                    125.0,
                    max(
                        (
                            item.global_end_ms - item.global_start_ms
                            for item in event.candidates
                            if item.view_id == view_id
                        ),
                        default=0.0,
                    ),
                )
                if (
                    isinstance(semantic_admission, dict)
                    and semantic_admission.get("mandatory_semantic_review") is True
                ):
                    semantic_review_admitted_views.add(view_id)
            if (
                isinstance(semantic_admission, dict)
                and semantic_admission.get("mandatory_semantic_review") is True
                and str(semantic_admission.get("context_view_id") or "")
            ):
                context_view_id = str(semantic_admission["context_view_id"])
                context_duration_ms = max(
                    125.0,
                    float(semantic_admission.get("context_global_end_ms") or 0.0)
                    - float(
                        semantic_admission.get("context_global_start_ms") or 0.0
                    ),
                )
                candidate_counts[context_view_id] += 1
                direct_evidence_ms[context_view_id] += context_duration_ms
                semantic_review_admitted_views.add(context_view_id)
        duration_minutes = max((end - start) / 60_000.0, 1e-6)
        threshold = float(cfg["min_view_action_density_per_minute"])
        sparse_view_enabled = bool(
            cfg.get("long_experiment_sparse_view_enabled", True)
        )
        sparse_minimum_duration_ms = float(
            cfg.get("long_experiment_sparse_view_min_duration_seconds", 120.0)
        ) * 1000.0
        sparse_minimum_events = max(
            2,
            int(cfg.get("long_experiment_sparse_view_min_events", 2)),
        )
        sparse_minimum_evidence_ms = float(
            cfg.get("long_experiment_sparse_view_min_evidence_seconds", 1.0)
        ) * 1000.0
        participating = sorted(
            view_id
            for view_id, count in candidate_counts.items()
            if (
                count / duration_minutes >= threshold
                or (
                    sparse_view_enabled
                    and end - start >= sparse_minimum_duration_ms
                    and count >= sparse_minimum_events
                    and direct_evidence_ms[view_id]
                    >= sparse_minimum_evidence_ms
                )
            )
            and (
                direct_evidence_ms[view_id] >= 500.0
                or view_id in semantic_review_admitted_views
            )
        )
        rejected_views = {
            view.view_id: "该边界内没有通过审计的实验动作，或动作密度不足"
            for view in views
            if view.view_id not in participating
        }
        if not participating:
            continue
        micro_segments = []
        for position, event in enumerate(bounded_group):
            micro_segments.append(
                {
                    "micro_segment_id": f"MICRO-{index:04d}-{position + 1:04d}",
                    "start_global_ms": core_start_ms(event),
                    "end_global_ms": core_end_ms(event),
                    "action_type": event.action_type.value,
                    "objects": event.objects,
                    "evidence_event_id": event.event_id,
                    "view_alignment_state": "aligned"
                    if all(event.supporting_views)
                    else "uncertain",
                    "next_event_id": (
                        bounded_group[position + 1].event_id
                        if position + 1 < len(bounded_group)
                        else None
                    ),
                    "uncertainty": event.uncertainty,
                }
            )
        segment = ExperimentSegment(
                segment_id=segment_id,
                global_start_ms=start,
                global_end_ms=end,
                event_ids=[event.event_id for event in bounded_group],
                participating_views=participating,
                rejected_views=rejected_views,
                micro_segments=micro_segments,
            )
        segment.segment_uid = stable_segment_uid(segment, by_event)
        segments.append(segment)
        if decision_receipts is not None:
            if boundary_context_receipt is not None:
                decision_receipts.append(boundary_context_receipt)
            start_sources = sorted(
                (
                    event
                    for event in (
                        [*leading_context, *start_anchors]
                        if start_anchors
                        else group
                    )
                    if core_start_ms(event) == raw_start
                ),
                key=event_sort_key,
            )
            end_sources = sorted(
                (
                    event
                    for event in bounded_group
                    if core_end_ms(event) == raw_end
                ),
                key=event_sort_key,
            )
            reason_codes = [
                "start_from_accepted_operation_anchor",
                (
                    "end_from_explicit_qf1_receipt"
                    if boundary_context_receipt is not None
                    else "end_from_accepted_event"
                ),
            ]
            if leading_context:
                reason_codes.append(
                    "accepted_primary_leading_context_chain_applied"
                )
            direct_start = max(
                0.0,
                raw_start - float(cfg["experiment_pre_roll_seconds"]) * 1000.0,
            )
            direct_end = raw_end + float(cfg["experiment_post_roll_seconds"]) * 1000.0
            if start < direct_start or end > direct_end:
                reason_codes.append("minimum_duration_padding_applied")
            decision_receipts.append(
                decision_receipt(
                    decision_type="raw_segment_boundary",
                    rule_id="QF1-RAW-ACCEPTED-EVENT-BOUNDARY",
                    verdict="accepted",
                    subject_ids=[segment_id],
                    reason_codes=reason_codes,
                    facts={
                        "raw_start_ms": raw_start,
                        "raw_end_ms": raw_end,
                        "final_start_ms": start,
                        "final_end_ms": end,
                        "start_source_event_ids": [event.event_id for event in start_sources],
                        "start_source_event_fingerprints": [
                            stable_event_fingerprint(event) for event in start_sources
                        ],
                        "end_source_event_ids": [event.event_id for event in end_sources],
                        "end_source_event_fingerprints": [
                            stable_event_fingerprint(event) for event in end_sources
                        ],
                        "end_source_decision_ids": (
                            [boundary_context_receipt["decision_id"]]
                            if boundary_context_receipt is not None
                            else []
                        ),
                        "leading_context_event_ids": [
                            event.event_id for event in leading_context
                        ],
                        "leading_context_event_fingerprints": [
                            stable_event_fingerprint(event)
                            for event in leading_context
                        ],
                        "accepted_event_ids": [event.event_id for event in bounded_group],
                        "accepted_event_fingerprints": [
                            stable_event_fingerprint(event) for event in bounded_group
                        ],
                        "coarse_window_fingerprints": sorted(
                            stable_candidate_fingerprint(window)
                            for window in ordered_windows
                            if window.global_end_ms >= raw_start
                            and window.global_start_ms <= raw_end
                        ),
                        "implicit_motion_window_extension": False,
                    },
                    thresholds={
                        "pre_roll_ms": float(cfg["experiment_pre_roll_seconds"]) * 1000.0,
                        "post_roll_ms": float(cfg["experiment_post_roll_seconds"]) * 1000.0,
                        "minimum_duration_ms": minimum,
                        "accepted_leading_context_max_gap_ms": float(
                            cfg.get(
                                "accepted_leading_context_max_gap_seconds",
                                10.0,
                            )
                        )
                        * 1000.0,
                        "accepted_leading_context_max_extension_ms": float(
                            cfg.get(
                                "accepted_leading_context_max_extension_seconds",
                                90.0,
                            )
                        )
                        * 1000.0,
                    },
                    evidence_refs=[event.event_id for event in bounded_group],
                    legacy={
                        "segment_id": segment_id,
                        "decision": "bounded_from_accepted_events",
                    },
                )
            )
    return segments


def build_physical_change_log(events: Iterable[EvidenceEvent]) -> list[PhysicalChange]:
    mapping = {
        ActionType.HAND_OBJECT_CONTACT: "contact_started",
        ActionType.OBJECT_MOVEMENT: "object_moved",
        ActionType.LIQUID_MOVEMENT: "liquid_transferred",
        ActionType.CONTAINER_STATE_CHANGE: "container_state_changed",
        ActionType.DEVICE_PANEL_OPERATION: "device_operated",
        ActionType.PIPETTE_TRANSFER_OPERATION: "pipette_transfer_operated",
    }
    changes: list[PhysicalChange] = []
    for event in events:
        if not event_is_formal(event):
            continue
        before, after = None, None
        if event.action_type == ActionType.CONTAINER_STATE_CHANGE:
            before, after = "closed_or_unknown", "open_closed_or_cap_changed"
        changes.append(
            PhysicalChange(
                change_id=f"CHANGE-{len(changes) + 1:06d}",
                event_id=event.event_id,
                global_ms=event.key_global_ms,
                change_type=mapping[event.action_type],
                object_names=event.objects,
                before_state=before,
                after_state=after,
                supporting_views=event.supporting_views,
                confidence=event.confidence,
                uncertainty=event.uncertainty,
            )
        )
    return changes
