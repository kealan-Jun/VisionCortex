"""Frame-level observation reduction, geometry and interaction state."""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

import cv2
import numpy as np

from ..schemas import ActionType, BoxEvidence, FrameEvidence

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
SUPPORT_ANCHOR_CLASSES = (
    DEVICE_CLASSES
    | CONTAINER_CLASSES
    | CAP_CLASSES
    | {
        "tube_rack",
        "magnetic_stir_bar",
    }
)
_ACTIVITY_PHYSICAL_TYPES = frozenset(
    {
        "hand_object_contact",
        "object_movement",
        "liquid_movement",
        "container_state_change",
        "device_panel_operation",
    }
)


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
    objects = [
        box
        for box in frame.detections
        if box.class_name not in HAND_CLASSES | NON_ACTION_CLASSES
    ]
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
        if (
            previous_gap is not None
            and float(previous_gap) - distance >= approach_delta
        ):
            state["approach_confirmed"] = True
        state["previous_gap"] = distance
        if distance <= contact_threshold:
            state["contact_frames"] = int(state.get("contact_frames", 0)) + 1
            confidence = min(hand.confidence, obj.confidence) * max(
                0.5, 1.0 - distance / max(contact_threshold, 1e-9)
            )
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
    maximum_state_entries = max(128, int(cfg.get("fine_state_max_entries", 4096)))
    while len(interaction_state) > maximum_state_entries:
        interaction_state.pop(next(iter(interaction_state)))
    for state_key, state_value in current_container_states.items():
        container_states.pop(state_key, None)
        container_states[state_key] = state_value
    while len(container_states) > maximum_state_entries:
        container_states.pop(next(iter(container_states)))

    window_ms = float(cfg.get("movement_window_ms", 250.0))
    movement_threshold = float(
        cfg.get("movement_window_threshold_norm", 0.01)
        if movement_history is not None
        else cfg["movement_threshold_norm"]
    )
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
            eligible = [
                item for item in history if frame.local_ms - item[2] >= window_ms * 0.6
            ]
            previous = (
                min(
                    eligible, key=lambda item: abs(frame.local_ms - item[2] - window_ms)
                )
                if eligible
                else None
            )
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
    camera_dx = (
        float(median(item[0] for item in anchor_vectors))
        if len(anchor_vectors) >= minimum_anchors
        else 0.0
    )
    camera_dy = (
        float(median(item[1] for item in anchor_vectors))
        if len(anchor_vectors) >= minimum_anchors
        else 0.0
    )
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
            if abs(scale - 1.0) <= float(
                cfg.get("motion_probe_max_scale_delta", 0.12)
            ) and rotation <= float(cfg.get("motion_probe_max_rotation_degrees", 8.0)):
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
            displacement = math.hypot(center_x - predicted_x, center_y - predicted_y)
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
                    confidence=min(
                        1.0,
                        obj.confidence
                        * (0.75 + displacement / max(movement_threshold, 1e-9) * 0.08),
                    ),
                    evidence={
                        "frame_index": frame.frame_index,
                        "track_id": obj.track_id,
                        "displacement_norm": round(raw_displacement, 5),
                        "camera_compensated_displacement_norm": round(displacement, 5),
                        "camera_motion_compensated": camera_compensated,
                        "camera_motion_method": compensation_method,
                        "camera_translation_norm": [
                            round(camera_dx, 5),
                            round(camera_dy, 5),
                        ],
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
                    confidence=min(tool.confidence, vessel.confidence)
                    * min(1.0, 0.6 + roi_motion / 30.0),
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
                        if tool.track_id is not None and vessel.track_id is not None
                        else None
                    ),
                )
            )
    return observations


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
        if isinstance(value.get("center_norm"), list) and len(value["center_norm"]) == 2
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
            round(float(median(item[index] for item in appearance_values)), 6)
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
        "median_area_norm": (round(float(median(areas)), 8) if areas else None),
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
            bool(
                (item.evidence.get("interaction_state") or {}).get("approach_confirmed")
            )
            for item in observations
        )
    ):
        uncertainty.append("未观察到完整接近过程；保留原有接触召回，不能单独确认操作")
    if action_type == ActionType.CONTAINER_STATE_CHANGE:
        transitions = {
            (
                str(item.evidence.get("state_before", "unknown")),
                str(item.evidence.get("state_after", "unknown")),
            )
            for item in observations
        }
        if not any(left != "unknown" and left != right for left, right in transitions):
            uncertainty.append("未形成明确的容器前后状态差异；保留状态线索候选")
    return uncertainty
