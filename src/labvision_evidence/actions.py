from __future__ import annotations

import itertools
import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np

from .detection import iter_frame_evidence
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


def _observation_key(observation: _Observation) -> tuple[str, str]:
    relevant = [item for item in observation.objects if item not in HAND_CLASSES]
    primary = sorted(relevant)[0] if relevant else "unknown"
    return observation.action_type.value, primary


def _frame_observations(
    frame: FrameEvidence,
    previous_tracks: dict[int, tuple[float, float, float]],
    cfg: dict[str, Any],
) -> list[_Observation]:
    assert frame.global_ms is not None
    observations: list[_Observation] = []
    hands = [box for box in frame.detections if box.class_name in HAND_CLASSES]
    objects = [box for box in frame.detections if box.class_name not in HAND_CLASSES | NON_ACTION_CLASSES]
    contact_threshold = float(cfg["contact_distance_norm"])

    for hand, obj in itertools.product(hands, objects):
        distance = _box_distance(hand, obj)
        if distance <= contact_threshold:
            confidence = min(hand.confidence, obj.confidence) * max(0.5, 1.0 - distance / max(contact_threshold, 1e-9))
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
                    },
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
                        evidence={"frame_index": frame.frame_index, "distance_norm": round(distance, 5)},
                    )
                )
            if obj.class_name in CAP_CLASSES or obj.class_name == "reagent_bottle_open":
                observations.append(
                    _Observation(
                        action_type=ActionType.CONTAINER_STATE_CHANGE,
                        local_ms=frame.local_ms,
                        global_ms=frame.global_ms,
                        objects=tuple(sorted({hand.class_name, obj.class_name})),
                        confidence=min(1.0, confidence + 0.06),
                        evidence={"frame_index": frame.frame_index, "state_cue": obj.class_name},
                    )
                )

    movement_threshold = float(cfg["movement_threshold_norm"])
    for obj in objects:
        if obj.track_id is None:
            continue
        center_x, center_y = _box_center(obj)
        previous = previous_tracks.get(obj.track_id)
        previous_tracks[obj.track_id] = (center_x, center_y, frame.local_ms)
        if previous is None:
            continue
        delta_ms = frame.local_ms - previous[2]
        if not 0.0 < delta_ms <= 2000.0:
            continue
        displacement = math.hypot(center_x - previous[0], center_y - previous[1])
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
                        "displacement_norm": round(displacement, 5),
                        "delta_ms": round(delta_ms, 3),
                    },
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
                        "inference": "液体不属于21类标签；这是工具+容器+ROI运动候选，需多模态确认",
                    },
                )
            )
    return observations


def _merge_observations(
    observations: Sequence[_Observation],
    view: ViewInput,
    cfg: dict[str, Any],
) -> list[ActionCandidate]:
    grouped: dict[tuple[str, str], list[_Observation]] = defaultdict(list)
    for observation in observations:
        grouped[_observation_key(observation)].append(observation)
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
                    evidence=[item.evidence for item in run[:50]],
                )
            )
            counter += 1
    return sorted(candidates, key=lambda candidate: candidate.global_start_ms)


def generate_candidates(
    views: Sequence[ViewInput], detection_paths: dict[str, Path], config: dict[str, Any]
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    cfg = config["segmentation"]
    for view in views:
        observations: list[_Observation] = []
        previous_tracks: dict[int, tuple[float, float, float]] = {}
        for frame in iter_frame_evidence(detection_paths[view.view_id]):
            observations.extend(_frame_observations(frame, previous_tracks, cfg))
        candidates.extend(_merge_observations(observations, view, cfg))
    return sorted(candidates, key=lambda candidate: candidate.global_start_ms)


def generate_coarse_activity_candidates(
    views: Sequence[ViewInput], detection_paths: dict[str, Path], config: dict[str, Any]
) -> list[ActionCandidate]:
    """Recall-first activity windows; the fine layer must verify contact and action type."""
    cfg = config["segmentation"]
    observations_by_view: dict[str, list[_Observation]] = defaultdict(list)
    by_id = {view.view_id: view for view in views}
    for view in views:
        for frame in iter_frame_evidence(detection_paths[view.view_id]):
            if frame.global_ms is None:
                continue
            hands = [box for box in frame.detections if box.class_name in HAND_CLASSES]
            objects = [
                box
                for box in frame.detections
                if box.class_name not in HAND_CLASSES | NON_ACTION_CLASSES | {"paper"}
            ]
            if hands and objects:
                best_hand = max(hands, key=lambda box: box.confidence)
                best_object = max(objects, key=lambda box: box.confidence)
                observations_by_view[view.view_id].append(
                    _Observation(
                        action_type=ActionType.HAND_OBJECT_CONTACT,
                        local_ms=frame.local_ms,
                        global_ms=frame.global_ms,
                        objects=tuple(sorted({best_hand.class_name, best_object.class_name})),
                        confidence=min(best_hand.confidence, best_object.confidence),
                        evidence={
                            "frame_index": frame.frame_index,
                            "coarse_activity": "hand_and_lab_object_cooccurrence",
                        },
                    )
                )
            elif frame.motion_score >= 10.0 and len(objects) >= 2:
                selected = sorted(objects, key=lambda box: box.confidence, reverse=True)[:2]
                observations_by_view[view.view_id].append(
                    _Observation(
                        action_type=ActionType.OBJECT_MOVEMENT,
                        local_ms=frame.local_ms,
                        global_ms=frame.global_ms,
                        objects=tuple(sorted(box.class_name for box in selected)),
                        confidence=min(box.confidence for box in selected) * 0.75,
                        evidence={"frame_index": frame.frame_index, "coarse_motion": frame.motion_score},
                    )
                )
    candidates: list[ActionCandidate] = []
    for view_id, observations in observations_by_view.items():
        candidates.extend(_merge_observations(observations, by_id[view_id], cfg))
    return sorted(candidates, key=lambda candidate: candidate.global_start_ms)


def generate_motion_burst_candidates(
    views: Sequence[ViewInput], detection_paths: dict[str, Path], config: dict[str, Any]
) -> list[ActionCandidate]:
    """Adaptive per-view motion bursts gated by laboratory object presence."""
    perf = config["performance"]
    percentile = float(perf["motion_burst_percentile"])
    merge_gap_ms = float(perf["motion_burst_merge_gap_seconds"]) * 1000.0
    min_observations = int(perf["motion_burst_min_observations"])
    candidates: list[ActionCandidate] = []
    for view in views:
        frames = list(iter_frame_evidence(detection_paths[view.view_id]))
        if not frames:
            continue
        scores = np.asarray([frame.motion_score for frame in frames], dtype=np.float64)
        threshold = max(float(np.percentile(scores, percentile)), float(np.median(scores) + 2.5))
        active: list[tuple[FrameEvidence, list[str]]] = []
        for frame in frames:
            objects = sorted(
                {
                    box.class_name
                    for box in frame.detections
                    if box.class_name not in HAND_CLASSES | NON_ACTION_CLASSES | {"paper"}
                }
            )
            if frame.motion_score >= threshold and objects:
                active.append((frame, objects))
        runs: list[list[tuple[FrameEvidence, list[str]]]] = []
        current: list[tuple[FrameEvidence, list[str]]] = []
        for item in active:
            global_ms = item[0].global_ms or 0.0
            previous_ms = (current[-1][0].global_ms or 0.0) if current else None
            if current and previous_ms is not None and global_ms - previous_ms > merge_gap_ms:
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
            peak_frame, peak_objects = max(run, key=lambda item: item[0].motion_score)
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
                            "objects": objects,
                        }
                        for frame, objects in run
                    ],
                    uncertainty=["粗层运动突发，仅用于召回精扫窗口，不作为最终动作证据"],
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.global_start_ms)


def select_fine_scan_views(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    coarse_candidates: Sequence[ActionCandidate],
    config: dict[str, Any],
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
        frames = list(iter_frame_evidence(detection_paths[view.view_id]))
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
    return selected, report


def _objects_overlap(left: ActionCandidate, right: ActionCandidate) -> bool:
    a = set(left.objects) - HAND_CLASSES
    b = set(right.objects) - HAND_CLASSES
    return bool(a & b) or left.action_type in {
        ActionType.LIQUID_MOVEMENT,
        ActionType.DEVICE_PANEL_OPERATION,
        ActionType.CONTAINER_STATE_CHANGE,
    }


def audit_candidates(
    candidates: Sequence[ActionCandidate],
    transforms: dict[str, AlignmentTransform],
    config: dict[str, Any],
) -> tuple[list[EvidenceEvent], list[dict[str, Any]]]:
    tolerance = float(config["alignment"]["cross_view_event_tolerance_ms"])
    seg_cfg = config["segmentation"]
    clusters: list[list[ActionCandidate]] = []
    for candidate in candidates:
        selected: list[ActionCandidate] | None = None
        for cluster in reversed(clusters):
            if candidate.global_start_ms - max(item.global_end_ms for item in cluster) > tolerance:
                break
            if cluster[0].action_type == candidate.action_type and any(
                _objects_overlap(candidate, item) for item in cluster
            ):
                selected = cluster
                break
        if selected is None:
            clusters.append([candidate])
        else:
            selected.append(candidate)

    events: list[EvidenceEvent] = []
    rejected: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, 1):
        views = sorted({item.view_id for item in cluster})
        roles = sorted({item.role for item in cluster}, key=lambda role: role.value)
        weighted_confidence = sum(item.confidence for item in cluster) / len(cluster)
        cross_view_bonus = min(0.16, 0.06 * (len(views) - 1) + (0.06 if len(roles) == 2 else 0.0))
        confidence = min(1.0, weighted_confidence + cross_view_bonus)
        aligned_views = [view_id for view_id in views if transforms[view_id].state == "aligned"]
        both_roles = len(roles) == 2
        duration_ms = max(item.global_end_ms for item in cluster) - min(item.global_start_ms for item in cluster)
        single_view_strong = (
            len(views) == 1
            and confidence >= float(seg_cfg["single_view_accept_confidence"])
            and cluster[0].action_type
            in {
                ActionType.HAND_OBJECT_CONTACT,
                ActionType.LIQUID_MOVEMENT,
                ActionType.CONTAINER_STATE_CHANGE,
                ActionType.DEVICE_PANEL_OPERATION,
            }
        )
        accepted = bool(aligned_views) and (both_roles or len(views) >= 2 or single_view_strong)
        if bool(seg_cfg.get("require_both_roles")):
            accepted = accepted and both_roles
        uncertainty: list[str] = []
        if not both_roles:
            uncertainty.append("该动作没有同时获得第一与第三人称支持")
        uncertain_alignments = [view_id for view_id in views if transforms[view_id].state != "aligned"]
        if uncertain_alignments:
            uncertainty.append(f"以下视角对齐不确定: {', '.join(uncertain_alignments)}")
        if cluster[0].action_type == ActionType.LIQUID_MOVEMENT:
            uncertainty.append("液体移动由传统CV候选提出，最终语义需多模态模型确认")
        if accepted:
            if both_roles:
                reason = "第一/第三人称动作、对象和全局时间窗一致"
            elif len(views) >= 2:
                reason = "至少两路同角色视角动作、对象和全局时间窗一致"
            else:
                reason = "单路持续强物理证据通过门控；未强制其他空/无效视角产出"
        else:
            reason = "候选缺少足够的跨视角或持续强物理证据"
        event = EvidenceEvent(
            event_id=f"EVT-{index:06d}",
            action_type=cluster[0].action_type,
            global_start_ms=min(item.global_start_ms for item in cluster),
            global_end_ms=max(item.global_end_ms for item in cluster),
            key_global_ms=float(median(item.key_global_ms for item in cluster)),
            objects=sorted({obj for item in cluster for obj in item.objects}),
            confidence=confidence,
            accepted=accepted,
            audit_reason=reason,
            supporting_views=views,
            supporting_roles=roles,
            candidates=cluster,
            uncertainty=uncertainty,
        )
        events.append(event)
        if not accepted:
            rejected.append(
                {
                    "event_id": event.event_id,
                    "candidate_ids": [item.candidate_id for item in cluster],
                    "reason": reason,
                    "confidence": confidence,
                    "duration_ms": duration_ms,
                }
            )
    return events, rejected


def refine_liquid_events_with_context(
    events: Sequence[EvidenceEvent],
    detection_paths: dict[str, Path],
    context_ms: float = 1000.0,
) -> list[dict[str, Any]]:
    """Reject single-view liquid hypotheses without nearby visible hand evidence.

    Liquid is not a YOLO class in the 21-class models. A tool/vessel ROI-motion
    coincidence is therefore recall-only. Cross-view matches remain accepted;
    single-view candidates must additionally show a hand in the temporal
    neighborhood before they may influence experiment bounds.
    """
    indexes: dict[str, tuple[list[float], list[FrameEvidence]]] = {}
    rejected: list[dict[str, Any]] = []
    for event in events:
        if not event.accepted or event.action_type != ActionType.LIQUID_MOVEMENT:
            continue
        supported: list[str] = []
        hand_counts: dict[str, int] = {}
        for view_id in event.supporting_views:
            if view_id not in indexes:
                frames = [
                    frame
                    for frame in iter_frame_evidence(detection_paths[view_id])
                    if frame.global_ms is not None
                ]
                indexes[view_id] = ([float(frame.global_ms) for frame in frames], frames)
            times, frames = indexes[view_id]
            left = bisect_left(times, event.global_start_ms - context_ms)
            right = bisect_left(times, event.global_end_ms + context_ms)
            count = sum(
                1
                for frame in frames[left:right]
                if any(box.class_name in HAND_CLASSES for box in frame.detections)
            )
            hand_counts[view_id] = count
            if count:
                supported.append(view_id)
        if supported:
            event.supporting_views = supported
            event.supporting_roles = sorted(
                {candidate.role for candidate in event.candidates if candidate.view_id in supported},
                key=lambda role: role.value,
            )
            event.audit_reason += f"；液体候选手部上下文={hand_counts}"
            continue
        event.accepted = False
        event.audit_reason = "液体运动候选的全部支持视角均缺少邻近手部证据，不能用于实验边界"
        event.uncertainty.append("保留为 CV 候选，未进入关键素材；液体语义不得仅由 ROI 运动推断")
        rejected.append(
            {
                "event_id": event.event_id,
                "candidate_ids": [item.candidate_id for item in event.candidates],
                "reason": event.audit_reason,
                "confidence": event.confidence,
                "duration_ms": event.global_end_ms - event.global_start_ms,
            }
        )
    return rejected


def build_experiment_segments(
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    config: dict[str, Any],
    coarse_windows: Sequence[ActionCandidate] | None = None,
) -> list[ExperimentSegment]:
    accepted = sorted((event for event in events if event.accepted), key=lambda event: event.global_start_ms)
    if not accepted:
        return []
    cfg = config["segmentation"]
    gap_ms = float(cfg["experiment_gap_seconds"]) * 1000.0
    groups: list[list[EvidenceEvent]] = []
    if coarse_windows:
        # A coarse motion burst is the experiment-level temporal envelope. Fine
        # actions may legitimately contain long pauses (incubation, reading a
        # balance, changing tools), so a fixed short event gap must not split one
        # bounded experiment. Padding is used only for assigning fine evidence;
        # the final boundary still comes from accepted fine events below.
        padding_ms = float(config["performance"]["fine_window_padding_seconds"]) * 1000.0
        buckets: list[list[EvidenceEvent]] = [[] for _ in coarse_windows]
        unassigned: list[EvidenceEvent] = []
        for event in accepted:
            midpoint = (event.global_start_ms + event.global_end_ms) / 2.0
            matches = [
                index
                for index, window in enumerate(coarse_windows)
                if window.global_start_ms - padding_ms <= midpoint <= window.global_end_ms + padding_ms
            ]
            if not matches:
                unassigned.append(event)
                continue
            best = min(
                matches,
                key=lambda index: abs(
                    midpoint
                    - (coarse_windows[index].global_start_ms + coarse_windows[index].global_end_ms) / 2.0
                ),
            )
            buckets[best].append(event)
        for bucket in buckets:
            current: list[EvidenceEvent] = []
            for event in sorted(bucket, key=lambda item: item.global_start_ms):
                if current and event.global_start_ms - current[-1].global_end_ms > gap_ms:
                    groups.append(current)
                    current = []
                current.append(event)
            if current:
                groups.append(current)
        # Retain independently strong evidence that falls outside a recalled
        # window, but group it conservatively by the legacy event-gap rule.
        current: list[EvidenceEvent] = []
        for event in unassigned:
            if current and event.global_start_ms - current[-1].global_end_ms > gap_ms:
                groups.append(current)
                current = []
            current.append(event)
        if current:
            groups.append(current)
        groups.sort(key=lambda group: min(event.global_start_ms for event in group))
    else:
        current = []
        for event in accepted:
            if current and event.global_start_ms - current[-1].global_end_ms > gap_ms:
                groups.append(current)
                current = []
            current.append(event)
        if current:
            groups.append(current)

    segments: list[ExperimentSegment] = []
    for index, group in enumerate(groups, 1):
        # A segment must start on a physically meaningful operation anchor.
        # Single-view liquid hypotheses and movement of fixed equipment may be
        # useful context, but are too noisy to pull the experiment boundary
        # earlier by themselves.
        start_anchors = []
        for event in group:
            if event.action_type in {
                ActionType.HAND_OBJECT_CONTACT,
                ActionType.CONTAINER_STATE_CHANGE,
                ActionType.DEVICE_PANEL_OPERATION,
            }:
                start_anchors.append(event)
            elif event.action_type == ActionType.LIQUID_MOVEMENT and len(event.supporting_views) > 1:
                start_anchors.append(event)
            elif event.action_type == ActionType.OBJECT_MOVEMENT and not (
                set(event.objects) & {"balance", "magnetic_stirrer", "computer"}
            ):
                start_anchors.append(event)
        raw_start = min(
            event.global_start_ms for event in (start_anchors or group)
        )
        raw_end = max(event.global_end_ms for event in group)
        start = max(0.0, raw_start - float(cfg["experiment_pre_roll_seconds"]) * 1000.0)
        end = raw_end + float(cfg["experiment_post_roll_seconds"]) * 1000.0
        minimum = float(cfg["min_experiment_seconds"]) * 1000.0
        if end - start < minimum:
            padding = (minimum - (end - start)) / 2.0
            start, end = max(0.0, start - padding), end + padding
        candidate_counts = defaultdict(int)
        direct_evidence_ms = defaultdict(float)
        for event in group:
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
        duration_minutes = max((end - start) / 60_000.0, 1e-6)
        threshold = float(cfg["min_view_action_density_per_minute"])
        participating = sorted(
            view_id
            for view_id, count in candidate_counts.items()
            if count / duration_minutes >= threshold and direct_evidence_ms[view_id] >= 500.0
        )
        rejected_views = {
            view.view_id: "该边界内没有通过审计的实验动作，或动作密度不足"
            for view in views
            if view.view_id not in participating
        }
        if not participating:
            continue
        micro_segments = []
        for position, event in enumerate(group):
            micro_segments.append(
                {
                    "micro_segment_id": f"MICRO-{index:04d}-{position + 1:04d}",
                    "start_global_ms": event.global_start_ms,
                    "end_global_ms": event.global_end_ms,
                    "action_type": event.action_type.value,
                    "objects": event.objects,
                    "evidence_event_id": event.event_id,
                    "view_alignment_state": "aligned"
                    if all(event.supporting_views)
                    else "uncertain",
                    "next_event_id": group[position + 1].event_id if position + 1 < len(group) else None,
                    "uncertainty": event.uncertainty,
                }
            )
        segments.append(
            ExperimentSegment(
                segment_id=f"EXP-{index:04d}",
                global_start_ms=start,
                global_end_ms=end,
                event_ids=[event.event_id for event in group],
                participating_views=participating,
                rejected_views=rejected_views,
                micro_segments=micro_segments,
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
    }
    changes: list[PhysicalChange] = []
    for event in events:
        if not event.accepted:
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
