"""Recall-only motion probes and fine-view selection."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..candidate_index import CoarseFrameIndex
from ..detection import iter_frame_evidence
from ..ordering import candidate_sort_key
from ..schemas import ActionCandidate, ActionType, FrameEvidence, ViewInput, ViewRole
from .observations import HAND_CLASSES, NON_ACTION_CLASSES, SUPPORT_ANCHOR_CLASSES


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
    if not config["performance"].get("motion_probe_use_embedded_coarse_scores", False):
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
            float(perf.get("motion_probe_rolling_window_seconds", 600.0)) * 1000.0,
        )
        grouped: dict[int, list[float]] = defaultdict(list)
        for frame in frames:
            timeline_ms = float(
                frame.global_ms if frame.global_ms is not None else frame.local_ms
            )
            grouped[max(0, int(timeline_ms // window_ms))].append(frame.motion_score)
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
    maximum_run_ms = float(perf.get("motion_probe_max_cluster_seconds", 0.0)) * 1000.0
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
            float(perf.get("motion_probe_rolling_window_seconds", 600.0)) * 1000.0,
        )
        active: list[tuple[FrameEvidence, list[str]]] = []
        for frame in frames:
            objects = sorted(
                {
                    box.class_name
                    for box in frame.detections
                    if box.class_name
                    not in HAND_CLASSES | NON_ACTION_CLASSES | {"paper"}
                }
            )
            timeline_ms = float(
                frame.global_ms if frame.global_ms is not None else frame.local_ms
            )
            bucket_threshold = rolling_thresholds.get(
                max(0, int(timeline_ms // rolling_window_ms)), threshold
            )
            adaptive_active = frame.motion_score >= min(threshold, bucket_threshold)
            raw_active = bool(
                raw_threshold is not None and frame.raw_motion_score >= raw_threshold
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
                and global_ms - (current[0][0].global_ms or 0.0) > maximum_run_ms
            )
            if (
                current
                and previous_ms is not None
                and (global_ms - previous_ms > merge_gap_ms or exceeds_run_span)
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
            peak_frame, _peak_objects = max(run, key=lambda item: item[0].motion_score)
            assert peak_frame.global_ms is not None
            confidence = min(
                1.0,
                0.55
                + 0.05 * len(run)
                + 0.1 * peak_frame.motion_score / max(threshold, 1e-6),
            )
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
                    uncertainty=[
                        "粗层运动突发，仅用于召回精扫窗口，不作为最终动作证据"
                    ],
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
    association_margin_ms = (
        float(perf.get("coarse_refinement_association_margin_seconds", 30.0)) * 1000.0
    )
    minimum_candidates = max(1, int(perf.get("coarse_refinement_min_candidates", 2)))
    minimum_span_ms = (
        float(perf.get("coarse_refinement_min_span_seconds", 30.0)) * 1000.0
    )
    quarantine_objectless = bool(
        perf.get("coarse_quarantine_objectless_unconfirmed_motion", True)
    )
    sustained_objectless_recall_ms = (
        float(
            perf.get(
                "coarse_objectless_recall_guard_min_seconds",
                perf.get("motion_probe_primary_min_seconds", 45.0),
            )
        )
        * 1000.0
    )
    semantic_association = bool(perf.get("coarse_semantic_association_enabled", False))
    cluster_margin_ms = (
        float(perf.get("coarse_semantic_cluster_margin_seconds", 3.0)) * 1000.0
    )
    maximum_boundary_expansion_ms = (
        float(perf.get("coarse_max_boundary_expansion_seconds", 0.0)) * 1000.0
    )
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
            motion_duration_ms = motion.global_end_ms - motion.global_start_ms
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
                "refined_duration_seconds": round(
                    (coarse_end - coarse_start) / 1000.0, 3
                ),
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
            item["decision"] == "quarantined_objectless_motion_without_coarse_yolo"
            for item in decisions
        ),
        "unmatched_coarse_candidate_count": len(unmatched),
        "output_candidate_count": len(refined),
        "output_candidates": [item.model_dump(mode="json") for item in refined],
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
    maximum_cluster_ms = (
        float(perf.get("motion_probe_max_cluster_seconds", 0.0)) * 1000.0
    )
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
    separation_ms = (
        float(perf.get("motion_probe_fallback_separation_seconds", 120.0)) * 1000.0
    )
    half_window_ms = (
        float(perf.get("motion_probe_fallback_window_seconds", 60.0)) * 500.0
    )
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
        duration_hours = max(
            1.0, (frames[-1].local_ms - frames[0].local_ms) / 3_600_000.0
        )
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
        all_motion = np.asarray(
            [frame.motion_score for frame in frames], dtype=np.float64
        )
        motion_threshold = max(
            float(np.percentile(all_motion, 80.0)) if len(all_motion) else 0.0,
            float(np.median(all_motion) + 1.0) if len(all_motion) else 1.0,
        )
        anchor_frames = 0
        active_anchor_frames = 0
        anchors: set[str] = set()
        for frame in frames:
            if frame.global_ms is None or not any(
                start <= frame.global_ms <= end for start, end in global_windows
            ):
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
    configured_minimum = config["performance"].get("fine_min_third_person_views", 1)
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
