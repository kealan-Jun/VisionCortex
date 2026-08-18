from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from .schemas import ActionType, EvidenceEvent, ViewRole


HAND_CLASSES = {"hand", "gloved_hand"}
DEVICE_CLASSES = {"balance", "magnetic_stirrer"}
CAP_OR_OPEN_CLASSES = {"tube_cap", "bottle_cap", "reagent_bottle_open"}
TRANSFER_TOOL_CLASSES = {"pipette", "spearhead", "spatula"}
CONTAINER_CLASSES = {
    "beaker",
    "reagent_bottle",
    "reagent_bottle_open",
    "sample_bottle",
    "sample_bottle_blue",
    "tube",
    "container",
}


def _flatten_evidence(event: EvidenceEvent) -> list[dict[str, Any]]:
    return [
        item
        for candidate in event.candidates
        for item in candidate.evidence
        if isinstance(item, dict)
    ]


def _number_values(items: Iterable[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for item in items:
        for key in keys:
            value = item.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
                break
    return values


def build_action_observability(event: EvidenceEvent) -> dict[str, Any]:
    """Describe what the fixed 21-class CV stack can and cannot prove.

    This receipt deliberately does not change ``event.accepted``. It separates
    object/geometry evidence from state/action semantics so downstream MLLM
    review cannot silently turn an inference into an observed fact.
    """

    evidence = _flatten_evidence(event)
    objects = set(event.objects)
    non_hand_objects = sorted(objects - HAND_CLASSES)
    track_ids = {
        f"{candidate.view_id}:{key}:{item[key]}"
        for candidate in event.candidates
        for item in candidate.evidence
        if isinstance(item, dict)
        for key in (
            "track_id",
            "object_track_id",
            "tool_track_id",
            "vessel_track_id",
            "container_track_id",
        )
        if item.get(key) is not None
    }
    distances = _number_values(evidence, "distance_norm")
    raw_displacements = _number_values(evidence, "displacement_norm")
    compensated_displacements = _number_values(
        evidence, "camera_compensated_displacement_norm"
    )
    roi_motion = _number_values(evidence, "roi_motion")
    state_cues = sorted(
        {
            str(item["state_cue"])
            for item in evidence
            if item.get("state_cue") is not None
        }
    )
    transfer_sequences = [
        item for item in evidence if item.get("transfer_sequence") is not None
    ]
    role_values = {
        role.value if isinstance(role, ViewRole) else str(role)
        for role in event.supporting_roles
    }
    both_roles = {ViewRole.FIRST_PERSON.value, ViewRole.THIRD_PERSON.value}.issubset(
        role_values
    )

    signals: list[str] = []
    missing: list[str] = []
    support_level = "insufficient"
    review_priority = "required"

    if event.action_type == ActionType.OBJECT_MOVEMENT:
        if track_ids and (compensated_displacements or raw_displacements):
            support_level = "direct_cv"
            review_priority = "optional"
            signals.append("tracked_object_displacement")
            if compensated_displacements:
                signals.append("camera_motion_compensated")
        else:
            missing.extend(["stable_object_track", "measured_displacement"])
    elif event.action_type == ActionType.HAND_OBJECT_CONTACT:
        if distances and objects & HAND_CLASSES and non_hand_objects:
            support_level = "indirect_cv"
            review_priority = "recommended"
            signals.append("hand_object_box_proximity")
        else:
            missing.append("hand_object_spatial_relation")
        missing.append("true_surface_contact_or_grasp_state")
    elif event.action_type == ActionType.LIQUID_MOVEMENT:
        if transfer_sequences:
            support_level = "indirect_cv"
            signals.append("source_transport_target_sequence")
        elif (
            objects & TRANSFER_TOOL_CLASSES
            and objects & CONTAINER_CLASSES
            and roi_motion
        ):
            support_level = "indirect_cv"
            signals.append("tool_container_proximity_with_roi_motion")
        else:
            missing.append("tool_container_motion_sequence")
        missing.extend(["visible_liquid_region", "liquid_level_or_flow_change"])
    elif event.action_type == ActionType.CONTAINER_STATE_CHANGE:
        if state_cues or objects & CAP_OR_OPEN_CLASSES:
            support_level = "indirect_cv"
            signals.append("cap_or_open_container_cue")
        else:
            missing.append("cap_open_close_transition_cue")
        missing.append("before_after_container_state")
    elif event.action_type == ActionType.DEVICE_PANEL_OPERATION:
        if distances and objects & DEVICE_CLASSES and objects & HAND_CLASSES:
            support_level = "indirect_cv"
            signals.append("hand_device_box_proximity")
        else:
            missing.append("hand_device_spatial_relation")
        missing.extend(["button_or_knob_identity", "control_or_display_state_change"])

    if not both_roles:
        missing.append("dual_role_action_support")

    return {
        "schema_version": "visioncortex-action-observability/1",
        "event_id": event.event_id,
        "action_type": event.action_type.value,
        "cv_ontology_support": support_level,
        "semantic_review_priority": review_priority,
        "can_cv_directly_prove_action": support_level == "direct_cv",
        "can_define_boundary_without_semantic_promotion": bool(
            event.accepted and (support_level == "direct_cv" or both_roles)
        ),
        "signals": sorted(set(signals)),
        "unmet_visual_requirements": sorted(set(missing)),
        "objects": sorted(objects),
        "non_hand_objects": non_hand_objects,
        "supporting_views": sorted(event.supporting_views),
        "supporting_roles": sorted(role_values),
        "both_roles": both_roles,
        "measurements": {
            "candidate_count": len(event.candidates),
            "evidence_observation_count": len(evidence),
            "stable_track_token_count": len(track_ids),
            "minimum_box_distance_norm": min(distances) if distances else None,
            "maximum_raw_displacement_norm": (
                max(raw_displacements) if raw_displacements else None
            ),
            "maximum_camera_compensated_displacement_norm": (
                max(compensated_displacements)
                if compensated_displacements
                else None
            ),
            "maximum_roi_motion": max(roi_motion) if roi_motion else None,
            "state_cues": state_cues,
            "transfer_sequence_count": len(transfer_sequences),
        },
    }


def attach_action_observability(
    events: Sequence[EvidenceEvent],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for event in events:
        receipt = build_action_observability(event)
        event.observability = receipt
        receipts.append(receipt)
    return receipts


def build_semantic_review_plan(
    events: Sequence[EvidenceEvent], config: dict[str, Any]
) -> dict[str, Any]:
    """Create a bounded MLLM review queue without making an API call.

    Accepted events remain accepted. Rejected events remain rejected. The plan
    makes future semantic fusion explicit, budgeted and auditable.
    """

    cfg = config.get("mllm", {})
    max_events = max(0, int(cfg.get("candidate_review_max_events", 120)))
    include_rejected = bool(cfg.get("candidate_review_include_rejected", True))
    minimum_rejected_confidence = float(
        cfg.get("candidate_review_rejected_min_confidence", 0.75)
    )
    rank = {"required": 0, "recommended": 1, "optional": 2}
    eligible: list[EvidenceEvent] = []
    skipped: list[dict[str, Any]] = []
    for event in events:
        receipt = event.observability or build_action_observability(event)
        event.observability = receipt
        priority = str(receipt["semantic_review_priority"])
        if priority == "optional" and event.accepted:
            skipped.append({"event_id": event.event_id, "reason": "direct_cv_optional"})
            continue
        if not event.accepted and (
            not include_rejected or event.confidence < minimum_rejected_confidence
        ):
            skipped.append(
                {
                    "event_id": event.event_id,
                    "reason": "rejected_below_semantic_recall_floor",
                }
            )
            continue
        eligible.append(event)

    eligible.sort(
        key=lambda event: (
            rank.get(str(event.observability.get("semantic_review_priority")), 9),
            0 if not event.accepted else 1,
            -float(event.confidence),
            float(event.key_global_ms),
        )
    )
    selected = eligible[:max_events]
    overflow = eligible[max_events:]
    action_counts = Counter(event.action_type.value for event in selected)
    return {
        "schema_version": "visioncortex-semantic-review-plan/1",
        "policy": "cv_recall_first_mllm_bounded_adjudication",
        "model_may_mutate_cv_acceptance": False,
        "model_may_confirm_relabel_or_mark_uncertain": True,
        "temporal_evidence_required": True,
        "dual_role_media_required": True,
        "selected_count": len(selected),
        "eligible_count": len(eligible),
        "overflow_count": len(overflow),
        "max_events": max_events,
        "action_type_counts": dict(sorted(action_counts.items())),
        "selected": [
            {
                "event_id": event.event_id,
                "action_type": event.action_type.value,
                "accepted_by_cv": event.accepted,
                "confidence": event.confidence,
                "priority": event.observability["semantic_review_priority"],
                "cv_ontology_support": event.observability["cv_ontology_support"],
                "unmet_visual_requirements": event.observability[
                    "unmet_visual_requirements"
                ],
                "global_start_ms": event.global_start_ms,
                "global_end_ms": event.global_end_ms,
                "key_global_ms": event.key_global_ms,
                "supporting_views": event.supporting_views,
                "supporting_roles": [
                    role.value if isinstance(role, ViewRole) else str(role)
                    for role in event.supporting_roles
                ],
            }
            for event in selected
        ],
        "skipped": [
            *skipped,
            *(
                {"event_id": event.event_id, "reason": "semantic_budget_overflow"}
                for event in overflow
            ),
        ],
    }


def record_semantic_review(
    event: EvidenceEvent, result: dict[str, Any]
) -> dict[str, Any]:
    """Attach a model verdict without allowing it to rewrite CV history."""

    known_actions = {item.value for item in ActionType}
    status = str(result.get("status") or "unknown")
    confirmed = str(result.get("action_type_confirmed") or "unknown")
    confidence = float(result.get("confidence") or 0.0)
    consistency = str(result.get("cross_view_consistency") or "unreviewed")
    if status != "completed":
        verdict = "model_unavailable"
    elif confirmed == event.action_type.value and confidence >= 0.55:
        verdict = "confirmed"
    elif confirmed in known_actions and confidence >= 0.55:
        verdict = "relabel_suggested"
    else:
        verdict = "uncertain"
    receipt = {
        "schema_version": "visioncortex-semantic-review-receipt/1",
        "event_id": event.event_id,
        "cv_action_type": event.action_type.value,
        "cv_accepted_before_review": event.accepted,
        "cv_accepted_after_review": event.accepted,
        "model_mutated_cv_acceptance": False,
        "model_status": status,
        "model": result.get("model"),
        "model_action_type": confirmed,
        "model_confidence": confidence,
        "cross_view_consistency": consistency,
        "verdict": verdict,
        "usage": result.get("usage") or {},
        "latency_seconds": result.get("latency_seconds"),
        "unmet_visual_requirements": (
            event.observability.get("unmet_visual_requirements", [])
            if event.observability
            else []
        ),
    }
    event.semantic_review = receipt
    return receipt
