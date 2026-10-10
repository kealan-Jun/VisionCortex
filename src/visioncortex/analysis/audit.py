"""Cross-view candidate adjudication and bounded liquid-context review."""

from __future__ import annotations

import itertools
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np

from ..candidate_index import FineFrameIndex
from ..detection import iter_frame_evidence
from ..ordering import candidate_sort_key, stable_event_fingerprint
from ..schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    FrameEvidence,
    set_event_admission,
)
from .observations import (
    CONTAINER_CLASSES,
    DEVICE_CLASSES,
    HAND_CLASSES,
    TRANSFER_TOOL_CLASSES,
    _box_distance,
)


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
        ) and bool(a & CONTAINER_CLASSES and b & CONTAINER_CLASSES)
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
    if left.view_id != right.view_id and performance.get(
        "fine_instance_cross_view_conflict_quarantine_enabled", False
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
            config["performance"].get("fine_instance_minimum_cosine_similarity", 0.25)
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
            max(0.0, float(segment.uncertainty_ms)) for segment in overlapping
        ),
        "available": state != "failed",
        "segment_indexes": [int(segment.segment_index) for segment in overlapping],
        "basis": "candidate_local_alignment_segments",
    }


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
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
                + float(action_offsets.get(candidate.action_type.value, 0.0)),
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
        float(perf.get("audit_independent_view_bonus", 0.04)) * max(0, len(views) - 1)
        + (float(perf.get("audit_dual_role_bonus", 0.04)) if len(roles) == 2 else 0.0),
    )
    maximum_alignment_penalty = float(perf.get("audit_maximum_alignment_penalty", 0.08))
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
        {
            "candidate_id": c.candidate_id,
            "reason": "movement_not_supported_by_image",
            "event_id": f"REJECTED-CANDIDATE-{c.candidate_id}",
            "confidence": c.confidence,
            "movement_visual_verification": c.provenance[
                "movement_visual_verification"
            ],
        }
        for c in candidates
        if c.action_type == ActionType.OBJECT_MOVEMENT
        and c.provenance.get("movement_visual_verification", {}).get("status")
        in {"contradicted", "unverified"}
    ]
    rejected_motion_ids = {c["candidate_id"] for c in motion_rejected}
    candidates = [c for c in candidates if c.candidate_id not in rejected_motion_ids]
    tolerance = float(config["alignment"]["cross_view_event_tolerance_ms"])
    maximum_tolerance = max(
        tolerance,
        float(config["alignment"].get("cross_view_event_max_tolerance_ms", tolerance)),
    )

    perf = config["performance"]
    local_alignment_enabled = bool(perf.get("audit_local_alignment_enabled", False))
    constrained = bool(perf.get("audit_constrained_clustering_enabled", False))
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
                candidate.global_start_ms - max(item.global_end_ms for item in cluster)
                > maximum_tolerance
            ):
                break
            compatible_members = [
                item for item in cluster if _objects_overlap(candidate, item, config)
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
                and (not constrained or proposed_span <= maximum_cluster_span_ms)
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
        weighted_confidence = math.fsum(item.confidence for item in cluster) / len(
            cluster
        )
        cross_view_bonus = min(
            0.16, 0.06 * (len(views) - 1) + (0.06 if len(roles) == 2 else 0.0)
        )
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
                        -abs(
                            item.key_global_ms
                            - median(candidate.key_global_ms for candidate in cluster)
                        ),
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
                accepted and (both_roles or len(views) >= 2 or single_view_strong)
            )
            if accepted and not direct_admission and semantic_context_admitted:
                formal_status = "provisional"
            liquid_sequence_observed = any(
                evidence.get("transfer_sequence") == "source_transport_target"
                for item in cluster
                for evidence in item.evidence
            )
            if (
                accepted
                and cluster[0].action_type == ActionType.LIQUID_MOVEMENT
                and perf.get("audit_liquid_sequence_formal_gate_enabled", False)
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
        uncertain_alignments = [
            view_id for view_id in views if transforms[view_id].state != "aligned"
        ]
        if uncertain_alignments:
            uncertainty.append(f"以下视角对齐不确定: {', '.join(uncertain_alignments)}")
        if cluster[0].action_type == ActionType.LIQUID_MOVEMENT:
            uncertainty.append("液体移动由传统CV候选提出，最终语义需多模态模型确认")
        if formal_status == "formal":
            if both_roles:
                reason = (
                    "第一/第三人称同类动作候选在时钟误差内重叠；跨机位实体对应仍待核验"
                )
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
                    (
                        pair_tolerance(left, right)
                        for left in cluster
                        for right in cluster
                        if left.view_id != right.view_id
                    ),
                    default=0.0,
                ),
                "view_uncertainty_ms": {
                    view_id: transforms[view_id].uncertainty_ms for view_id in views
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
                        for frame in iter_frame_evidence(detection_paths[view_id])
                        if frame.global_ms is not None
                    ]
                    indexes[view_id] = (
                        [float(frame.global_ms) for frame in frames],
                        frames,
                    )
                times, frames = indexes[view_id]
                left = bisect_left(times, event.global_start_ms - context_ms)
                right = bisect_left(times, event.global_end_ms + context_ms)
                nearby_frames = frames[left:right]
            hand_count = 0
            spatial_count = 0
            for frame in nearby_frames:
                hands = [
                    box for box in frame.detections if box.class_name in HAND_CLASSES
                ]
                if hands:
                    hand_count += 1
                relevant_objects = [
                    box
                    for box in frame.detections
                    if box.class_name in object_classes
                    and (
                        not track_ids
                        or (box.track_id is not None and int(box.track_id) in track_ids)
                    )
                ]
                if (
                    hands
                    and relevant_objects
                    and any(
                        _box_distance(hand, obj) <= maximum_gap
                        for hand in hands
                        for obj in relevant_objects
                    )
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
                retained_candidates = [
                    item for item in event.candidates if item.view_id in supported
                ]
                previous_confidence = event.confidence
                event.confidence = min(
                    event.confidence,
                    max((item.confidence for item in retained_candidates), default=0.0),
                )
                # A removed source invalidates the earlier consensus/admission.
                # Keep the hypothesis available, but require a fresh semantic decision.
                set_event_admission(event, "provisional")
                event.observability["liquid_context"].update(
                    {
                        "removed_supporting_views": removed_views,
                        "previous_confidence": previous_confidence,
                        "retained_confidence": event.confidence,
                        "confidence_policy": "cap_to_retained_source_confidence",
                        "admission_after_support_filter": "provisional",
                    }
                )
                event.audit_reason = (
                    "液体候选部分视角未通过手部空间核验，原跨视角裁决失效，保留待核验"
                )
                event.uncertainty.append(
                    "支持来源减少后须重新裁决，不能沿用此前跨视角一致结论"
                )
            event.supporting_views = supported
            event.supporting_roles = sorted(
                {
                    candidate.role
                    for candidate in event.candidates
                    if candidate.view_id in supported
                },
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
        event.uncertainty.append(
            "保留为 CV 候选，未进入关键素材；液体语义不得仅由 ROI 运动推断"
        )
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
