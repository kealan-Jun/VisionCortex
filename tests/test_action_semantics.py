from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visioncortex.action_semantics import (
    attach_action_observability,
    build_semantic_review_plan,
    record_semantic_review,
)
from visioncortex.actions import (
    _Observation,
    _frame_observations,
    _infer_liquid_transfer_sequences,
    audit_candidates,
)
from visioncortex.archive import extract_temporal_review_frames
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    BoxEvidence,
    EvidenceEvent,
    FrameEvidence,
    ViewInput,
    ViewRole,
)


def _candidate(
    candidate_id: str,
    action_type: ActionType,
    view_id: str,
    role: ViewRole,
    objects: list[str],
    *,
    confidence: float = 0.9,
    evidence: list[dict] | None = None,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        view_id=view_id,
        role=role,
        local_start_ms=10_000,
        local_end_ms=11_000,
        global_start_ms=10_000,
        global_end_ms=11_000,
        key_global_ms=10_500,
        objects=objects,
        confidence=confidence,
        evidence=evidence or [],
    )


def _event(
    event_id: str,
    action_type: ActionType,
    candidates: list[ActionCandidate],
    *,
    accepted: bool = True,
    confidence: float = 0.9,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=event_id,
        action_type=action_type,
        global_start_ms=10_000,
        global_end_ms=11_000,
        key_global_ms=10_500,
        objects=sorted({item for candidate in candidates for item in candidate.objects}),
        confidence=confidence,
        accepted=accepted,
        audit_reason="test",
        supporting_views=sorted({candidate.view_id for candidate in candidates}),
        supporting_roles=sorted(
            {candidate.role for candidate in candidates}, key=lambda item: item.value
        ),
        candidates=candidates,
    )


def test_observability_separates_direct_movement_from_indirect_liquid(default_config):
    movement = _event(
        "MOVE",
        ActionType.OBJECT_MOVEMENT,
        [
            _candidate(
                "MOVE-FP",
                ActionType.OBJECT_MOVEMENT,
                "fp",
                ViewRole.FIRST_PERSON,
                ["tube"],
                evidence=[
                    {
                        "track_id": 7,
                        "displacement_norm": 0.08,
                        "camera_compensated_displacement_norm": 0.05,
                    }
                ],
            ),
            _candidate(
                "MOVE-TP",
                ActionType.OBJECT_MOVEMENT,
                "tp",
                ViewRole.THIRD_PERSON,
                ["tube"],
                evidence=[{"track_id": 8, "displacement_norm": 0.06}],
            ),
        ],
    )
    liquid = _event(
        "LIQUID",
        ActionType.LIQUID_MOVEMENT,
        [
            _candidate(
                "LIQUID-FP",
                ActionType.LIQUID_MOVEMENT,
                "fp",
                ViewRole.FIRST_PERSON,
                ["pipette", "tube"],
                evidence=[{"distance_norm": 0.01, "roi_motion": 8.0}],
            )
        ],
    )

    receipts = attach_action_observability([movement, liquid])
    plan = build_semantic_review_plan([movement, liquid], default_config)

    assert receipts[0]["cv_ontology_support"] == "direct_cv"
    assert receipts[0]["semantic_review_priority"] == "optional"
    assert receipts[1]["cv_ontology_support"] == "indirect_cv"
    assert "visible_liquid_region" in receipts[1]["unmet_visual_requirements"]
    assert [item["event_id"] for item in plan["selected"]] == ["LIQUID"]
    assert plan["model_may_mutate_cv_acceptance"] is False
    assert liquid.accepted is True


def test_observability_records_dual_role_distinct_container_transfer_path():
    def transfer_candidate(
        candidate_id: str,
        view_id: str,
        role: ViewRole,
        source_track_id: int,
        target_track_id: int,
    ) -> ActionCandidate:
        return _candidate(
            candidate_id,
            ActionType.LIQUID_MOVEMENT,
            view_id,
            role,
            ["pipette", "sample_bottle", "tube"],
            evidence=[
                {
                    "transfer_sequence": "source_transport_target",
                    "tool_class": "pipette",
                    "tool_track_id": 7,
                    "source_class": "sample_bottle",
                    "source_track_id": source_track_id,
                    "target_class": "tube",
                    "target_track_id": target_track_id,
                    "source_contact_end_global_ms": 10_200,
                    "target_contact_start_global_ms": 10_700,
                    "transport_gap_ms": 500,
                    "source_observation_count": 3,
                    "target_observation_count": 4,
                }
            ],
        )

    event = _event(
        "LIQUID-DUAL",
        ActionType.LIQUID_MOVEMENT,
        [
            transfer_candidate("FP", "fp", ViewRole.FIRST_PERSON, 10, 11),
            transfer_candidate("TP", "tp", ViewRole.THIRD_PERSON, 20, 21),
        ],
    )

    receipt = attach_action_observability([event])[0]

    assert receipt["measurements"]["admissible_transfer_sequence_count"] == 2
    assert receipt["measurements"]["dual_role_transfer_sequence_verified"] is True
    assert "dual_role_distinct_container_transfer_sequence" in receipt["signals"]


def test_observability_records_repeated_tracks_as_review_candidate_not_proof():
    def sequence(source_end_ms: float, *, gap_ms: float = 300.0) -> dict:
        return {
            "transfer_sequence": "source_transport_target",
            "tool_class": "pipette",
            "tool_track_id": 7,
            "source_class": "sample_bottle",
            "source_track_id": 10,
            "target_class": "beaker",
            "target_track_id": 11,
            "source_contact_end_global_ms": source_end_ms,
            "target_contact_start_global_ms": source_end_ms + gap_ms,
            "transport_gap_ms": gap_ms,
            "source_observation_count": 3,
            "target_observation_count": 3,
        }

    event = _event(
        "LIQUID-REPEATED",
        ActionType.LIQUID_MOVEMENT,
        [
            _candidate(
                "FP",
                ActionType.LIQUID_MOVEMENT,
                "fp",
                ViewRole.FIRST_PERSON,
                ["pipette", "sample_bottle", "beaker"],
                evidence=[
                    sequence(10_000),
                    sequence(10_100),  # duplicate detector candidate, not a cycle
                    sequence(11_100),
                    sequence(12_200),
                ],
            )
        ],
    )

    receipt = attach_action_observability([event])[0]

    path = receipt["measurements"]["repeated_pipette_path"]
    assert path["raw_sequence_count"] == 4
    assert path["distinct_cycle_count"] == 3
    assert path["cycle_span_ms"] == 2_200
    assert path["candidate"] is True
    assert receipt["measurements"][
        "repeated_pipette_transfer_path_verified"
    ] is False
    assert receipt["measurements"][
        "repeated_pipette_transfer_path_candidate"
    ] is True
    assert "repeated_pipette_path_candidate_for_dense_review" in receipt["signals"]


def test_repeated_pipette_path_rejects_one_transition_with_track_duplicates():
    evidence = []
    for source_end_ms in (10_000.0, 10_100.0, 10_200.0, 10_300.0):
        evidence.append(
            {
                "transfer_sequence": "source_transport_target",
                "tool_class": "pipette",
                "tool_track_id": 7,
                "source_class": "sample_bottle",
                "source_track_id": 10,
                "target_class": "beaker",
                "target_track_id": 11,
                "source_contact_end_global_ms": source_end_ms,
                "target_contact_start_global_ms": source_end_ms + 200,
                "transport_gap_ms": 200,
                "source_observation_count": 3,
                "target_observation_count": 3,
            }
        )
    event = _event(
        "LIQUID-DUPLICATES",
        ActionType.LIQUID_MOVEMENT,
        [
            _candidate(
                "FP",
                ActionType.LIQUID_MOVEMENT,
                "fp",
                ViewRole.FIRST_PERSON,
                ["pipette", "sample_bottle", "beaker"],
                evidence=evidence,
            )
        ],
    )

    receipt = attach_action_observability([event])[0]

    assert receipt["measurements"][
        "repeated_pipette_transfer_path_verified"
    ] is False
    assert receipt["measurements"][
        "repeated_pipette_transfer_path_candidate"
    ] is False
    assert receipt["measurements"]["repeated_pipette_path"][
        "distinct_cycle_count"
    ] == 1


def test_rejected_high_confidence_candidate_enters_bounded_semantic_recall_plan(
    default_config,
):
    candidate = _candidate(
        "CONTACT-FP",
        ActionType.HAND_OBJECT_CONTACT,
        "fp",
        ViewRole.FIRST_PERSON,
        ["gloved_hand", "paper"],
        evidence=[{"distance_norm": 0.0, "object_track_id": 22}],
    )
    event = _event(
        "CONTACT",
        ActionType.HAND_OBJECT_CONTACT,
        [candidate],
        accepted=False,
        confidence=0.88,
    )
    attach_action_observability([event])

    plan = build_semantic_review_plan([event], default_config)

    assert plan["selected"][0]["accepted_by_cv"] is False
    assert event.accepted is False


def test_semantic_review_can_suggest_relabel_without_mutating_cv_acceptance(
    default_config,
):
    candidate = _candidate(
        "CONTACT-FP",
        ActionType.HAND_OBJECT_CONTACT,
        "fp",
        ViewRole.FIRST_PERSON,
        ["gloved_hand", "tube"],
        evidence=[{"distance_norm": 0.0}],
    )
    event = _event("CONTACT", ActionType.HAND_OBJECT_CONTACT, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "container_state_change",
            "confidence": 0.8,
            "cross_view_consistency": "partial",
            "action_proof": {
                "proof_type": "direct_other",
                "container_before_state_visible": True,
                "container_after_state_visible": True,
                "container_state_transition_completed": True,
            },
            "usage": {"total_tokens": 10},
        },
    )

    assert receipt["verdict"] == "relabel_suggested"
    assert receipt["model_mutated_cv_acceptance"] is False
    assert event.accepted is True
    assert event.semantic_review == receipt


def test_container_state_without_completed_before_after_proof_fails_closed(
    default_config,
):
    candidate = _candidate(
        "CAP-TP",
        ActionType.CONTAINER_STATE_CHANGE,
        "tp",
        ViewRole.THIRD_PERSON,
        ["gloved_hand", "bottle_cap"],
    )
    event = _event("CAP", ActionType.CONTAINER_STATE_CHANGE, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "container_state_change",
            "evidence_verdict": "confirmed",
            "confidence": 0.9,
            "cross_view_consistency": "partial",
            "candidate_action_support_by_view": [
                {
                    "view_id": "tp",
                    "supports_candidate_action": True,
                    "confidence": 0.9,
                }
            ],
            "action_proof": {
                "proof_type": "direct_other",
                "container_before_state_visible": True,
                "container_after_state_visible": False,
                "container_state_transition_completed": False,
            },
        },
    )

    assert receipt["verdict"] == "uncertain"
    assert receipt["semantic_proof_contradictions"] == [
        "container_after_state_not_visible",
        "container_state_transition_not_completed",
    ]


def test_completed_container_before_after_transition_can_be_confirmed(
    default_config,
):
    candidate = _candidate(
        "CAP-TP",
        ActionType.CONTAINER_STATE_CHANGE,
        "tp",
        ViewRole.THIRD_PERSON,
        ["gloved_hand", "bottle_cap"],
    )
    event = _event("CAP", ActionType.CONTAINER_STATE_CHANGE, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "container_state_change",
            "evidence_verdict": "confirmed",
            "confidence": 0.9,
            "cross_view_consistency": "partial",
            "candidate_action_support_by_view": [
                {
                    "view_id": "tp",
                    "supports_candidate_action": True,
                    "confidence": 0.9,
                }
            ],
            "action_proof": {
                "proof_type": "direct_other",
                "container_before_state_visible": True,
                "container_after_state_visible": True,
                "container_state_transition_completed": True,
            },
        },
    )

    assert receipt["verdict"] == "confirmed"
    assert receipt["semantic_proof_contradictions"] == []


def test_semantic_review_respects_explicit_uncertain_evidence_verdict(
    default_config,
):
    candidate = _candidate(
        "LIQUID-FP",
        ActionType.LIQUID_MOVEMENT,
        "fp",
        ViewRole.FIRST_PERSON,
        ["pipette", "tube"],
    )
    event = _event("LIQUID", ActionType.LIQUID_MOVEMENT, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "liquid_movement",
            "evidence_verdict": "uncertain",
            "confidence": 0.9,
            "cross_view_consistency": "partial",
        },
    )

    assert receipt["verdict"] == "uncertain"
    assert receipt["model_evidence_verdict"] == "uncertain"
    assert event.accepted is True


def test_liquid_posture_only_support_fails_closed_even_if_model_confirms(
    default_config,
):
    candidate = _candidate(
        "LIQUID-TP",
        ActionType.LIQUID_MOVEMENT,
        "tp",
        ViewRole.THIRD_PERSON,
        ["pipette", "sample_bottle"],
    )
    event = _event("LIQUID", ActionType.LIQUID_MOVEMENT, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "liquid_movement",
            "evidence_verdict": "confirmed",
            "confidence": 0.9,
            "cross_view_consistency": "partial",
            "candidate_action_support_by_view": [
                {
                    "view_id": "tp",
                    "supports_candidate_action": True,
                    "confidence": 0.9,
                    "reason": "Pipette tip enters the bottle mouth.",
                }
            ],
            "action_proof": {
                "proof_type": "posture_only",
                "visible_liquid_or_level_change": False,
            },
            "uncertainties": ["无法确认液体是否实际被吸取/转移"],
        },
    )

    assert receipt["verdict"] == "uncertain"
    assert receipt["directly_supported_view_ids"] == []
    assert receipt["semantic_proof_contradictions"] == [
        "inadmissible_liquid_proof_type:posture_only",
        "model_explicitly_cannot_confirm_liquid_transfer",
    ]


def test_liquid_closed_transfer_cycle_can_be_confirmed(default_config):
    candidate = _candidate(
        "LIQUID-TP",
        ActionType.LIQUID_MOVEMENT,
        "tp",
        ViewRole.THIRD_PERSON,
        ["pipette", "sample_bottle", "tube"],
    )
    event = _event("LIQUID", ActionType.LIQUID_MOVEMENT, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "liquid_movement",
            "evidence_verdict": "confirmed",
            "confidence": 0.9,
            "cross_view_consistency": "partial",
            "candidate_action_support_by_view": [
                {
                    "view_id": "tp",
                    "supports_candidate_action": True,
                    "confidence": 0.9,
                    "reason": "Complete source-to-target transfer cycle is visible.",
                }
            ],
            "action_proof": {
                "proof_type": "pipette_closed_transfer_cycle",
                "source_contact_visible": True,
                "withdrawal_or_transport_visible": True,
                "target_contact_visible": True,
                "release_or_plunger_change_visible": True,
            },
        },
    )

    assert receipt["verdict"] == "confirmed"
    assert receipt["directly_supported_view_ids"] == ["tp"]
    assert receipt["semantic_proof_contradictions"] == []


def test_liquid_dual_role_cv_path_can_substitute_for_hidden_plunger(
    default_config,
):
    candidates = []
    for view_id, role, source_track, target_track in (
        ("fp", ViewRole.FIRST_PERSON, 10, 11),
        ("tp", ViewRole.THIRD_PERSON, 20, 21),
    ):
        candidates.append(
            _candidate(
                f"LIQUID-{view_id}",
                ActionType.LIQUID_MOVEMENT,
                view_id,
                role,
                ["pipette", "sample_bottle", "tube"],
                evidence=[
                    {
                        "transfer_sequence": "source_transport_target",
                        "tool_class": "pipette",
                        "tool_track_id": 7,
                        "source_class": "sample_bottle",
                        "source_track_id": source_track,
                        "target_class": "tube",
                        "target_track_id": target_track,
                        "source_observation_count": 2,
                        "target_observation_count": 2,
                    }
                ],
            )
        )
    event = _event("LIQUID", ActionType.LIQUID_MOVEMENT, candidates)
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "liquid_movement",
            "evidence_verdict": "confirmed",
            "confidence": 0.9,
            "cross_view_consistency": "consistent",
            "action_proof": {
                "proof_type": "pipette_closed_transfer_cycle",
                "source_contact_visible": True,
                "withdrawal_or_transport_visible": True,
                "target_contact_visible": True,
                "release_or_plunger_change_visible": False,
                "dual_role_cv_sequence_verified": True,
            },
        },
    )

    assert receipt["verdict"] == "confirmed"
    assert receipt["semantic_proof_contradictions"] == []


def test_repeated_pipette_path_is_not_an_admissible_liquid_proof(
    default_config,
):
    evidence = []
    for source_end_ms in (10_000.0, 11_100.0, 12_200.0):
        evidence.append(
            {
                "transfer_sequence": "source_transport_target",
                "tool_class": "pipette",
                "tool_track_id": 7,
                "source_class": "sample_bottle",
                "source_track_id": 10,
                "target_class": "beaker",
                "target_track_id": 11,
                "source_contact_end_global_ms": source_end_ms,
                "target_contact_start_global_ms": source_end_ms + 300,
                "transport_gap_ms": 300,
                "source_observation_count": 3,
                "target_observation_count": 3,
            }
        )
    event = _event(
        "LIQUID-REPEATED",
        ActionType.LIQUID_MOVEMENT,
        [
            _candidate(
                "FP",
                ActionType.LIQUID_MOVEMENT,
                "fp",
                ViewRole.FIRST_PERSON,
                ["pipette", "sample_bottle", "beaker"],
                evidence=evidence,
            )
        ],
    )
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "liquid_movement",
            "evidence_verdict": "confirmed",
            "confidence": 0.9,
            "cross_view_consistency": "single_view",
            "action_proof": {
                "proof_type": "pipette_repeated_source_target_cycles",
                "source_contact_visible": True,
                "withdrawal_or_transport_visible": True,
                "target_contact_visible": True,
                "release_or_plunger_change_visible": False,
                "repeated_source_target_cycles_visible": True,
            },
        },
    )

    assert receipt["verdict"] == "uncertain"
    assert receipt["semantic_proof_contradictions"] == [
        "inadmissible_liquid_proof_type:pipette_repeated_source_target_cycles",
    ]


def test_visible_pipette_chain_relabels_without_claiming_visible_liquid(
    default_config,
):
    candidate = _candidate(
        "LIQUID-FP",
        ActionType.LIQUID_MOVEMENT,
        "fp",
        ViewRole.FIRST_PERSON,
        ["pipette", "tube", "beaker"],
    )
    event = _event("LIQUID", ActionType.LIQUID_MOVEMENT, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "pipette_transfer_operation",
            "evidence_verdict": "relabel_suggested",
            "confidence": 0.82,
            "cross_view_consistency": "conflict",
            "candidate_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_candidate_action": False,
                    "confidence": 0.2,
                    "reason": "No visible liquid change.",
                }
            ],
            "confirmed_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_confirmed_action": True,
                    "confidence": 0.86,
                    "reason": "Source, transport and distinct target are visible.",
                }
            ],
            "action_proof": {
                "proof_type": "pipette_operational_transfer_chain",
                "visible_liquid_or_level_change": False,
                "source_contact_visible": True,
                "withdrawal_or_transport_visible": True,
                "target_contact_visible": True,
                "release_or_plunger_change_visible": False,
            },
            "uncertainties": ["无法确认液体是否实际被转移"],
        },
    )

    assert receipt["verdict"] == "relabel_suggested"
    assert receipt["directly_supported_view_ids"] == ["fp"]
    assert receipt["semantic_proof_contradictions"] == []


def test_pipette_operation_relabel_requires_direct_target_view_support(
    default_config,
):
    event = _event(
        "LIQUID",
        ActionType.LIQUID_MOVEMENT,
        [
            _candidate(
                "LIQUID-FP",
                ActionType.LIQUID_MOVEMENT,
                "fp",
                ViewRole.FIRST_PERSON,
                ["pipette", "tube", "beaker"],
            )
        ],
    )
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "action_type_confirmed": "pipette_transfer_operation",
            "evidence_verdict": "relabel_suggested",
            "confidence": 0.9,
            "action_proof": {
                "proof_type": "pipette_operational_transfer_chain",
                "source_contact_visible": True,
                "withdrawal_or_transport_visible": True,
                "target_contact_visible": True,
            },
        },
    )

    assert receipt["verdict"] == "uncertain"
    assert receipt["semantic_proof_contradictions"] == [
        "pipette_operation_lacks_direct_view_support"
    ]


def test_semantic_review_keeps_direct_single_view_support_separate_from_conflict(
    default_config,
):
    candidate = _candidate(
        "CONTACT-TP",
        ActionType.HAND_OBJECT_CONTACT,
        "tp",
        ViewRole.THIRD_PERSON,
        ["gloved_hand", "bottle_cap"],
    )
    event = _event("CONTACT", ActionType.HAND_OBJECT_CONTACT, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "unknown",
            "evidence_verdict": "uncertain",
            "confidence": 0.35,
            "cross_view_consistency": "conflict",
            "candidate_action_support_by_view": [
                {
                    "view_id": "tp",
                    "supports_candidate_action": True,
                    "confidence": 0.9,
                    "reason": "The hand visibly grasps the cap.",
                },
                {
                    "view_id": "fp",
                    "supports_candidate_action": False,
                    "confidence": 0.9,
                    "reason": "A different concurrent action is visible.",
                },
            ],
        },
    )

    assert receipt["verdict"] == "confirmed"
    assert receipt["directly_supported_view_ids"] == ["tp"]
    assert receipt["cross_view_consistency"] == "conflict"
    assert receipt["confirmation_scope"] == (
        "at_least_one_view_direct_candidate_action"
    )


def test_same_action_direct_views_follow_confirmed_support_when_available(
    default_config,
):
    candidate = _candidate(
        "MOVE-FP",
        ActionType.OBJECT_MOVEMENT,
        "fp",
        ViewRole.FIRST_PERSON,
        ["pipette"],
    )
    event = _event("MOVE", ActionType.OBJECT_MOVEMENT, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "object_movement",
            "evidence_verdict": "confirmed",
            "confidence": 0.9,
            "cross_view_consistency": "conflict",
            "candidate_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_candidate_action": True,
                    "confidence": 0.9,
                },
                {
                    "view_id": "tp",
                    "supports_candidate_action": True,
                    "confidence": 0.8,
                },
            ],
            "confirmed_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_confirmed_action": True,
                    "confidence": 0.9,
                },
                {
                    "view_id": "tp",
                    "supports_confirmed_action": False,
                    "confidence": 0.8,
                },
            ],
        },
    )

    assert receipt["directly_supported_view_ids"] == ["fp"]


def test_camera_translation_is_removed_before_object_movement(default_config):
    cfg = default_config["segmentation"]
    previous_tracks: dict[int, tuple[float, float, float]] = {}

    def box(class_name: str, track_id: int, x: float) -> BoxEvidence:
        return BoxEvidence(
            class_id=0,
            class_name=class_name,
            confidence=0.9,
            xyxy_norm=(x, 0.2, x + 0.1, 0.3),
            track_id=track_id,
        )

    first = FrameEvidence(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        frame_index=1,
        local_ms=0,
        global_ms=0,
        width=640,
        height=360,
        detections=[box("beaker", 1, 0.1), box("tube", 2, 0.4), box("balance", 3, 0.7)],
    )
    second = first.model_copy(
        update={
            "frame_index": 2,
            "local_ms": 100,
            "global_ms": 100,
            "detections": [
                box("beaker", 1, 0.2),
                box("tube", 2, 0.5),
                box("balance", 3, 0.8),
            ],
        }
    )

    _frame_observations(first, previous_tracks, cfg)
    observations = _frame_observations(second, previous_tracks, cfg)

    assert not any(item.action_type == ActionType.OBJECT_MOVEMENT for item in observations)


def test_tracked_source_transport_target_sequence_adds_liquid_recall_candidate(
    default_config,
):
    observations = []
    for local_ms in (1_000.0, 1_100.0, 1_200.0):
        observations.append(
            _Observation(
                action_type=ActionType.LIQUID_MOVEMENT,
                local_ms=local_ms,
                global_ms=local_ms,
                objects=("pipette", "sample_bottle"),
                confidence=0.8,
                evidence={
                    "tool_class": "pipette",
                    "tool_track_id": 7,
                    "vessel_class": "sample_bottle",
                    "vessel_track_id": 11,
                },
            )
        )
    for local_ms in (2_000.0, 2_100.0, 2_200.0):
        observations.append(
            _Observation(
                action_type=ActionType.LIQUID_MOVEMENT,
                local_ms=local_ms,
                global_ms=local_ms,
                objects=("pipette", "tube"),
                confidence=0.82,
                evidence={
                    "tool_class": "pipette",
                    "tool_track_id": 7,
                    "vessel_class": "tube",
                    "vessel_track_id": 12,
                },
            )
        )
    view = ViewInput(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        video=Path("fp.mp4"),
    )

    candidates = _infer_liquid_transfer_sequences(
        observations, view, default_config["segmentation"]
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.objects == ["pipette", "sample_bottle", "tube"]
    assert candidate.evidence[0]["transfer_sequence"] == "source_transport_target"
    assert candidate.evidence[0]["source_track_id"] == 11
    assert candidate.evidence[0]["target_track_id"] == 12
    assert "液体本体" in candidate.uncertainty[0]


def test_unrelated_panel_and_container_candidates_do_not_cross_view_merge(default_config):
    transforms = {
        view_id: AlignmentTransform(
            view_id=view_id,
            reference_view_id="fp",
            confidence=0.9,
            state="aligned",
        )
        for view_id in ("fp", "tp")
    }
    candidates = [
        _candidate(
            "PANEL-FP",
            ActionType.DEVICE_PANEL_OPERATION,
            "fp",
            ViewRole.FIRST_PERSON,
            ["hand", "balance"],
        ),
        _candidate(
            "PANEL-TP",
            ActionType.DEVICE_PANEL_OPERATION,
            "tp",
            ViewRole.THIRD_PERSON,
            ["hand", "magnetic_stirrer"],
        ),
        _candidate(
            "STATE-FP",
            ActionType.CONTAINER_STATE_CHANGE,
            "fp",
            ViewRole.FIRST_PERSON,
            ["hand", "tube_cap"],
        ),
        _candidate(
            "STATE-TP",
            ActionType.CONTAINER_STATE_CHANGE,
            "tp",
            ViewRole.THIRD_PERSON,
            ["hand", "bottle_cap"],
        ),
    ]

    events, _ = audit_candidates(candidates, transforms, default_config)

    assert len(events) == 4
    assert all(len(event.candidates) == 1 for event in events)


def test_temporal_review_reuses_short_key_clip(tmp_path: Path):
    clip = tmp_path / "key.avi"
    writer = cv2.VideoWriter(
        str(clip),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
    writer.release()

    samples = extract_temporal_review_frames(clip, tmp_path / "samples", "fp01", 3)

    assert len(samples) == 3
    assert [label.split("temporal_phase=", 1)[1].split(";", 1)[0] for label, _ in samples] == [
        "clip_early", "clip_middle", "clip_late",
    ]
    assert all("sample_scope=clip_timeline" in label and "nominal_clip_time_ms=" in label for label, _ in samples)
    assert all(path.is_file() and path.stat().st_size > 0 for _, path in samples)


def test_five_frame_temporal_review_preserves_named_phases(tmp_path: Path):
    clip = tmp_path / "liquid-key.avi"
    writer = cv2.VideoWriter(
        str(clip),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(50):
        writer.write(np.full((48, 64, 3), index * 3, dtype=np.uint8))
    writer.release()

    samples = extract_temporal_review_frames(
        clip, tmp_path / "liquid-samples", "fp01", 5
    )

    assert [label.split("temporal_phase=", 1)[1].split(";", 1)[0] for label, _ in samples] == [
        "clip_early",
        "clip_mid_early",
        "clip_middle",
        "clip_mid_late",
        "clip_late",
    ]
