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

LIQUID_DIRECT_PROOF_TYPES = {
    "visible_liquid_flow",
    "visible_liquid_level_change",
    "pour_with_visible_liquid_change",
    "pipette_closed_transfer_cycle",
}
PIPETTE_OPERATION_PROOF_TYPE = "pipette_operational_transfer_chain"


def semantic_action_proof_contradictions(
    action_type: ActionType,
    result: dict[str, Any] | None,
    observability: dict[str, Any] | None = None,
) -> list[str]:
    """Return deterministic reasons a semantic action cannot be published.

    Liquid transfer is deliberately fail-closed. The fixed CV ontology can
    propose a tool/container sequence, but a tool merely entering a vessel is
    not evidence that liquid actually moved. A completed semantic response
    must therefore carry one of the explicit, auditable proof types below.
    """

    if action_type == ActionType.PIPETTE_TRANSFER_OPERATION:
        payload = result or {}
        proof = payload.get("action_proof")
        if not isinstance(proof, dict):
            return ["missing_structured_pipette_operation_proof"]
        contradictions: list[str] = []
        proof_type = str(proof.get("proof_type") or "").strip().lower()
        if proof_type != PIPETTE_OPERATION_PROOF_TYPE:
            contradictions.append(
                f"inadmissible_pipette_operation_proof_type:{proof_type or 'missing'}"
            )
        required_phases = (
            "source_contact_visible",
            "withdrawal_or_transport_visible",
            "target_contact_visible",
        )
        if not all(proof.get(field) is True for field in required_phases):
            contradictions.append("pipette_operational_transfer_chain_incomplete")
        target_support = [
            item
            for item in (payload.get("confirmed_action_support_by_view") or [])
            if isinstance(item, dict)
            and item.get("supports_confirmed_action") is True
            and float(item.get("confidence") or 0.0) >= 0.65
            and str(item.get("view_id") or "").strip()
        ]
        if not target_support:
            contradictions.append("pipette_operation_lacks_direct_view_support")
        return list(dict.fromkeys(contradictions))

    if action_type == ActionType.CONTAINER_STATE_CHANGE:
        payload = result or {}
        proof = payload.get("action_proof")
        if not isinstance(proof, dict):
            return ["missing_structured_container_state_proof"]
        contradictions: list[str] = []
        if proof.get("container_before_state_visible") is not True:
            contradictions.append("container_before_state_not_visible")
        if proof.get("container_after_state_visible") is not True:
            contradictions.append("container_after_state_not_visible")
        if proof.get("container_state_transition_completed") is not True:
            contradictions.append("container_state_transition_not_completed")
        return contradictions

    if action_type != ActionType.LIQUID_MOVEMENT:
        return []
    payload = result or {}
    proof = payload.get("action_proof")
    if not isinstance(proof, dict):
        return ["missing_structured_liquid_action_proof"]

    proof_type = str(proof.get("proof_type") or "").strip().lower()
    contradictions: list[str] = []
    if proof_type not in LIQUID_DIRECT_PROOF_TYPES:
        contradictions.append(
            f"inadmissible_liquid_proof_type:{proof_type or 'missing'}"
        )
    elif proof_type in {
        "visible_liquid_flow",
        "visible_liquid_level_change",
        "pour_with_visible_liquid_change",
    } and proof.get("visible_liquid_or_level_change") is not True:
        contradictions.append("liquid_visual_change_not_affirmed")
    elif proof_type == "pipette_closed_transfer_cycle":
        sequence_receipt = (observability or {}).get("measurements") or {}
        model_dual_role_sequence = (
            proof.get("dual_role_cv_sequence_verified") is True
        )
        receipt_dual_role_sequence = (
            sequence_receipt.get("dual_role_transfer_sequence_verified") is True
        )
        if model_dual_role_sequence and not receipt_dual_role_sequence:
            contradictions.append("unverified_dual_role_transfer_sequence_claim")
        required_visible_phases = all(
            proof.get(field) is True
            for field in (
                "source_contact_visible",
                "withdrawal_or_transport_visible",
                "target_contact_visible",
            )
        )
        release_proof = proof.get("release_or_plunger_change_visible") is True
        verified_dual_role_path = bool(
            model_dual_role_sequence and receipt_dual_role_sequence
        )
        if not required_visible_phases or not (
            release_proof or verified_dual_role_path
        ):
            contradictions.append("pipette_transfer_cycle_incomplete")

    uncertainty_text = " ".join(
        str(item) for item in (payload.get("uncertainties") or [])
    ).lower()
    explicit_denials = (
        "无法确认液体是否实际被吸取",
        "无法确认液体是否实际被转移",
        "无法确认实际吸取",
        "无法确认实际转移",
        "cannot confirm actual liquid",
        "cannot confirm whether liquid was actually",
        "unable to confirm actual liquid",
    )
    if any(marker in uncertainty_text for marker in explicit_denials):
        contradictions.append("model_explicitly_cannot_confirm_liquid_transfer")
    return list(dict.fromkeys(contradictions))


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


def _repeated_pipette_path_receipt(
    sequences: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a conservative receipt for repeated source/target pipetting.

    Detector tracking can emit several near-identical transfer candidates for
    one physical transition.  Count only temporally separated transitions
    between the same stable, unordered container pair in the same view.  This
    remains an operational-path candidate: semantic review must independently
    see the source contact, transport and target contact before publication.
    """

    grouped: dict[
        tuple[str, str, tuple[tuple[str, str], tuple[str, str]]],
        list[dict[str, Any]],
    ] = {}
    raw_counts: Counter[
        tuple[str, str, tuple[tuple[str, str], tuple[str, str]]]
    ] = Counter()
    for item in sequences:
        if item.get("tool_class") != "pipette":
            continue
        gap = item.get("transport_gap_ms")
        transition_ms = item.get("source_contact_end_global_ms")
        if not isinstance(gap, (int, float)) or not isinstance(
            transition_ms, (int, float)
        ):
            continue
        # Zero-gap candidates are ambiguous box overlap; very long gaps do not
        # establish one continuous source-to-target operation.
        if not 100.0 <= float(gap) <= 5_000.0:
            continue
        pair = tuple(
            sorted(
                (
                    (
                        str(item.get("source_class") or ""),
                        str(item.get("source_track_id")),
                    ),
                    (
                        str(item.get("target_class") or ""),
                        str(item.get("target_track_id")),
                    ),
                )
            )
        )
        key = (str(item.get("view_id") or ""), str(item.get("role") or ""), pair)
        grouped.setdefault(key, []).append(item)
        raw_counts[key] += 1

    ranked: list[dict[str, Any]] = []
    for key, items in grouped.items():
        cycle_clusters: list[list[dict[str, Any]]] = []
        for item in sorted(
            items, key=lambda value: float(value["source_contact_end_global_ms"])
        ):
            timestamp = float(item["source_contact_end_global_ms"])
            if not cycle_clusters or timestamp - float(
                cycle_clusters[-1][-1]["source_contact_end_global_ms"]
            ) > 350.0:
                cycle_clusters.append([item])
            else:
                cycle_clusters[-1].append(item)
        # Pick the clearest temporal bridge from each near-duplicate cluster.
        # The source/target identity is already identical by grouping key.
        representatives = [
            max(
                cluster,
                key=lambda value: (
                    float(value.get("transport_gap_ms") or 0.0),
                    int(value.get("source_observation_count") or 0)
                    + int(value.get("target_observation_count") or 0),
                ),
            )
            for cluster in cycle_clusters
        ]
        distinct_cycles = [
            float(item["source_contact_end_global_ms"])
            for item in representatives
        ]
        span_ms = (
            distinct_cycles[-1] - distinct_cycles[0]
            if len(distinct_cycles) >= 2
            else 0.0
        )
        ranked.append(
            {
                "view_id": key[0],
                "role": key[1],
                "container_pair": [
                    {"class": class_name, "track_id": track_id}
                    for class_name, track_id in key[2]
                ],
                "raw_sequence_count": raw_counts[key],
                "distinct_cycle_count": len(distinct_cycles),
                "cycle_transition_global_ms": distinct_cycles,
                "cycle_anchors": [
                    {
                        "cycle_index": index,
                        "source_contact_global_ms": max(
                            0.0,
                            float(item["source_contact_end_global_ms"]) - 100.0,
                        ),
                        "transport_global_ms": (
                            float(item["source_contact_end_global_ms"])
                            + float(item["target_contact_start_global_ms"])
                        )
                        / 2.0,
                        "target_contact_global_ms": float(
                            item["target_contact_start_global_ms"]
                        )
                        + 100.0,
                    }
                    for index, item in enumerate(representatives, start=1)
                ],
                "cycle_span_ms": span_ms,
            }
        )
    if not ranked:
        return None
    ranked.sort(
        key=lambda item: (
            -int(item["distinct_cycle_count"]),
            -float(item["cycle_span_ms"]),
            -int(item["raw_sequence_count"]),
            str(item["view_id"]),
            str(item["container_pair"]),
        )
    )
    best = ranked[0]
    best["candidate"] = bool(
        int(best["distinct_cycle_count"]) >= 3
        and float(best["cycle_span_ms"]) >= 2_000.0
    )
    best["candidate_rule"] = (
        "same_view_same_stable_container_pair; pipette; transport_gap_ms="
        "[100,5000]; >=3 transitions separated_by>350ms; span>=2000ms"
    )
    best["limitation"] = (
        "tracking fragments cannot prove repeated physical cycles or liquid movement; "
        "use only to choose the dense-review primary view"
    )
    return best


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
    admissible_transfer_sequences: list[dict[str, Any]] = []
    for candidate in event.candidates:
        role = (
            candidate.role.value
            if isinstance(candidate.role, ViewRole)
            else str(candidate.role)
        )
        for item in candidate.evidence:
            if not isinstance(item, dict) or item.get("transfer_sequence") != (
                "source_transport_target"
            ):
                continue
            tool_class = str(item.get("tool_class") or "")
            source_class = str(item.get("source_class") or "")
            target_class = str(item.get("target_class") or "")
            source_track_id = item.get("source_track_id")
            target_track_id = item.get("target_track_id")
            if not (
                tool_class in TRANSFER_TOOL_CLASSES
                and source_class in CONTAINER_CLASSES
                and target_class in CONTAINER_CLASSES
                and source_track_id is not None
                and target_track_id is not None
                and source_track_id != target_track_id
                and int(item.get("source_observation_count") or 0) >= 2
                and int(item.get("target_observation_count") or 0) >= 2
            ):
                continue
            admissible_transfer_sequences.append(
                {
                    "view_id": candidate.view_id,
                    "role": role,
                    "tool_class": tool_class,
                    "tool_track_id": item.get("tool_track_id"),
                    "source_class": source_class,
                    "source_track_id": source_track_id,
                    "target_class": target_class,
                    "target_track_id": target_track_id,
                    "source_contact_end_global_ms": item.get(
                        "source_contact_end_global_ms"
                    ),
                    "target_contact_start_global_ms": item.get(
                        "target_contact_start_global_ms"
                    ),
                    "transport_gap_ms": item.get("transport_gap_ms"),
                    "source_observation_count": int(
                        item.get("source_observation_count") or 0
                    ),
                    "target_observation_count": int(
                        item.get("target_observation_count") or 0
                    ),
                }
            )
    admissible_transfer_roles = {
        item["role"] for item in admissible_transfer_sequences
    }
    dual_role_transfer_sequence_verified = {
        ViewRole.FIRST_PERSON.value,
        ViewRole.THIRD_PERSON.value,
    }.issubset(admissible_transfer_roles)
    repeated_pipette_path = _repeated_pipette_path_receipt(
        admissible_transfer_sequences
    )
    repeated_pipette_transfer_path_candidate = bool(
        repeated_pipette_path and repeated_pipette_path.get("candidate") is True
    )
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
    elif event.action_type in {
        ActionType.LIQUID_MOVEMENT,
        ActionType.PIPETTE_TRANSFER_OPERATION,
    }:
        if transfer_sequences:
            support_level = "indirect_cv"
            signals.append("source_transport_target_sequence")
            if dual_role_transfer_sequence_verified:
                signals.append("dual_role_distinct_container_transfer_sequence")
            if repeated_pipette_transfer_path_candidate:
                signals.append("repeated_pipette_path_candidate_for_dense_review")
        elif (
            objects & TRANSFER_TOOL_CLASSES
            and objects & CONTAINER_CLASSES
            and roi_motion
        ):
            support_level = "indirect_cv"
            signals.append("tool_container_proximity_with_roi_motion")
        else:
            missing.append("tool_container_motion_sequence")
        if event.action_type == ActionType.LIQUID_MOVEMENT:
            missing.extend(["visible_liquid_region", "liquid_level_or_flow_change"])
        else:
            missing.extend(
                [
                    "visible_source_contact",
                    "visible_withdrawal_or_transport",
                    "visible_distinct_target_contact",
                ]
            )
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
        "schema_version": "visioncortex-action-observability/3",
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
            "admissible_transfer_sequence_count": len(
                admissible_transfer_sequences
            ),
            "dual_role_transfer_sequence_verified": (
                dual_role_transfer_sequence_verified
            ),
            "repeated_pipette_transfer_path_candidate": (
                repeated_pipette_transfer_path_candidate
            ),
            "repeated_pipette_transfer_path_verified": False,
            "repeated_pipette_path": repeated_pipette_path,
            "admissible_transfer_sequences": admissible_transfer_sequences,
        },
    }


def attach_action_observability(
    events: Sequence[EvidenceEvent],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for event in events:
        semantic_recall_admission = (event.observability or {}).get(
            "semantic_recall_admission"
        )
        receipt = build_action_observability(event)
        if isinstance(semantic_recall_admission, dict):
            receipt["semantic_recall_admission"] = semantic_recall_admission
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
    model_evidence_verdict = str(
        result.get("evidence_verdict") or "unknown"
    ).strip().lower()
    confidence = float(result.get("confidence") or 0.0)
    consistency = str(result.get("cross_view_consistency") or "unreviewed")
    candidate_support_by_view = [
        item
        for item in (result.get("candidate_action_support_by_view") or [])
        if isinstance(item, dict)
    ]
    confirmed_action_support_by_view = [
        item
        for item in (result.get("confirmed_action_support_by_view") or [])
        if isinstance(item, dict)
    ]
    support_for_confirmed_action = (
        confirmed_action_support_by_view
        if confirmed in known_actions and confirmed_action_support_by_view
        else candidate_support_by_view
    )
    directly_supported_views = [
        str(item.get("view_id") or "")
        for item in support_for_confirmed_action
        if (
            item.get("supports_candidate_action") is True
            or item.get("supports_confirmed_action") is True
        )
        and float(item.get("confidence") or 0.0) >= 0.55
        and str(item.get("view_id") or "").strip()
    ]
    proposed_action = (
        ActionType(confirmed) if confirmed in known_actions else event.action_type
    )
    proof_contradictions = semantic_action_proof_contradictions(
        proposed_action,
        result,
        event.observability,
    )
    if proof_contradictions:
        directly_supported_views = []
    if status != "completed":
        verdict = "model_unavailable"
    elif proof_contradictions:
        verdict = "uncertain"
    elif (
        directly_supported_views
        and confirmed in known_actions
        and confirmed != event.action_type.value
    ):
        # ``evidence_verdict`` evaluates the supplied CV candidate.  Rejecting
        # that candidate is fully compatible with directly proving a safer,
        # different action in ``action_type_confirmed``.  Keep the two verdict
        # scopes separate and let curation apply its stricter relabel gate.
        verdict = "relabel_suggested"
    elif directly_supported_views and model_evidence_verdict != "rejected":
        # The response contract defines these booleans specifically against
        # the supplied CV candidate. Cross-view subject conflict remains in a
        # separate field and must not erase a directly visible single-view
        # action. Raw model fields are retained below for audit.
        verdict = "confirmed"
    elif model_evidence_verdict in {"uncertain", "rejected"}:
        verdict = model_evidence_verdict
    elif model_evidence_verdict == "relabel_suggested":
        verdict = "relabel_suggested"
    elif (
        model_evidence_verdict in {"confirmed", "unknown", ""}
        and confirmed == event.action_type.value
        and confidence >= 0.55
    ):
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
        "model_evidence_verdict": model_evidence_verdict,
        "model_confidence": confidence,
        "cross_view_consistency": consistency,
        "candidate_action_support_by_view": candidate_support_by_view,
        "confirmed_action_support_by_view": confirmed_action_support_by_view,
        "directly_supported_view_ids": directly_supported_views,
        "confirmation_scope": (
            "at_least_one_view_direct_candidate_action"
            if directly_supported_views
            else "whole_model_response"
        ),
        "semantic_proof_contradictions": proof_contradictions,
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
