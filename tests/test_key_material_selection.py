from labvision_evidence.grouping import select_key_events
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    ViewRole,
)


def _event(index: int, start_ms: float) -> EvidenceEvent:
    candidate = ActionCandidate(
        candidate_id=f"C-{index}",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=start_ms,
        local_end_ms=start_ms + 900,
        global_start_ms=start_ms,
        global_end_ms=start_ms + 900,
        key_global_ms=start_ms + 450,
        objects=["pipette", "tube"],
        confidence=0.9,
    )
    return EvidenceEvent(
        event_id=f"E-{index}",
        action_type=candidate.action_type,
        global_start_ms=candidate.global_start_ms,
        global_end_ms=candidate.global_end_ms,
        key_global_ms=candidate.key_global_ms,
        objects=candidate.objects,
        confidence=candidate.confidence,
        accepted=True,
        audit_reason="cross-view confirmed",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[candidate],
    )


def test_repeated_physical_actions_seconds_apart_are_not_class_deduplicated(
    default_config,
):
    events = [_event(1, 10_000), _event(2, 13_000), _event(3, 16_000)]
    segment = ExperimentSegment(
        segment_id="EXP-1",
        global_start_ms=8_000,
        global_end_ms=20_000,
        event_ids=[event.event_id for event in events],
        participating_views=["fp", "tp"],
    )
    group = ExperimentGroup(
        group_id="GROUP-1",
        continuity_type="independent",
        atomic_experiment_ids=[segment.segment_id],
        global_start_ms=segment.global_start_ms,
        global_end_ms=segment.global_end_ms,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
    )

    selected = select_key_events([group], [segment], events, default_config)

    assert [event.event_id for event in selected] == ["E-1", "E-2", "E-3"]
