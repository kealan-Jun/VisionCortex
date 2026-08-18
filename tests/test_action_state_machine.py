from __future__ import annotations

from labvision_evidence.action_state_machine import (
    attach_continuous_action_states,
    build_event_state_receipt,
)
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ViewRole,
)


def _candidate(
    candidate_id: str,
    action_type: ActionType,
    objects: list[str],
    evidence: list[dict],
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=1_000,
        local_end_ms=2_000,
        global_start_ms=1_000,
        global_end_ms=2_000,
        key_global_ms=1_500,
        objects=objects,
        confidence=0.9,
        evidence=evidence,
    )


def _event(event_id: str, candidate: ActionCandidate) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=event_id,
        action_type=candidate.action_type,
        global_start_ms=candidate.global_start_ms,
        global_end_ms=candidate.global_end_ms,
        key_global_ms=candidate.key_global_ms,
        objects=candidate.objects,
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[candidate],
    )


def test_complete_transfer_sequence_builds_all_required_phases(default_config):
    event = _event(
        "TRANSFER",
        _candidate(
            "C1",
            ActionType.LIQUID_MOVEMENT,
            ["pipette", "sample_bottle", "tube"],
            [
                {
                    "transfer_sequence": "source_transport_target",
                    "tool_track_id": 7,
                    "source_track_id": 11,
                    "target_track_id": 12,
                }
            ],
        ),
    )

    receipt = build_event_state_receipt(event, default_config)

    assert receipt["lifecycle_state"] == "completed"
    assert receipt["action_subtype"] == "pipette_transfer"
    assert receipt["phases"] == receipt["required_phases"]
    assert receipt["scores"]["phase_completeness"] == 1.0
    assert receipt["object_identity"]["identity_status"] == "tracked"


def test_source_only_liquid_candidate_is_not_promoted_to_complete(default_config):
    event = _event(
        "SOURCE-ONLY",
        _candidate(
            "C2",
            ActionType.LIQUID_MOVEMENT,
            ["pipette", "sample_bottle"],
            [{"distance_norm": 0.01, "roi_motion": 9.0}],
        ),
    )

    receipt = build_event_state_receipt(event, default_config)

    assert receipt["lifecycle_state"] == "incomplete_end"
    assert "target_contact" in receipt["missing_phases"]
    assert receipt["state_after"]["target"] == "not_observed"


def test_higher_level_transfer_suppresses_duplicate_contact_publication(default_config):
    transfer = _event(
        "TRANSFER",
        _candidate(
            "C3",
            ActionType.LIQUID_MOVEMENT,
            ["pipette", "tube"],
            [{"transfer_sequence": "source_transport_target", "tool_track_id": 1}],
        ),
    )
    contact = _event(
        "CONTACT",
        _candidate(
            "C4",
            ActionType.HAND_OBJECT_CONTACT,
            ["gloved_hand", "pipette"],
            [{"distance_norm": 0.0, "object_track_id": 1}],
        ),
    )

    ledger = attach_continuous_action_states([transfer, contact], default_config)

    assert transfer.accepted is True and contact.accepted is True
    assert contact.state_machine["publication"]["status"] == "component_only"
    assert contact.state_machine["publication"]["suppressed_by_event_id"] == "TRANSFER"
    assert ledger["cv_acceptance_mutated"] is False
