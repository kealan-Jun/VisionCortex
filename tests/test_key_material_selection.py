from visioncortex.grouping import select_key_events
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    ViewRole,
)


def _event(
    index: int,
    start_ms: float,
    *,
    duration_ms: float = 900.0,
    confidence: float = 0.9,
) -> EvidenceEvent:
    candidate = ActionCandidate(
        candidate_id=f"C-{index}",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=start_ms,
        local_end_ms=start_ms + duration_ms,
        global_start_ms=start_ms,
        global_end_ms=start_ms + duration_ms,
        key_global_ms=start_ms + duration_ms / 2.0,
        objects=["pipette", "tube"],
        confidence=confidence,
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


def _context(events):
    segment = ExperimentSegment(
        segment_id="EXP-1",
        global_start_ms=min(event.global_start_ms for event in events) - 1000,
        global_end_ms=max(event.global_end_ms for event in events) + 1000,
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
    return group, segment


def test_nearby_non_overlapping_actions_are_not_deduplicated(default_config):
    events = [_event(1, 10_000), _event(2, 10_950)]
    group, segment = _context(events)
    receipts = []

    selected = select_key_events(
        [group], [segment], events, default_config, decision_receipts=receipts
    )

    assert [event.event_id for event in selected] == ["E-1", "E-2"]
    assert len(receipts) == 2
    assert all(item["selected"] for item in receipts)
    assert all(item["decision"] == "selected" for item in receipts)


def test_overlapping_nearby_actions_keep_higher_confidence_with_receipt(
    default_config,
):
    events = [
        _event(1, 10_000, confidence=0.8),
        _event(2, 10_800, confidence=0.95),
    ]
    group, segment = _context(events)
    receipts = []

    selected = select_key_events(
        [group], [segment], events, default_config, decision_receipts=receipts
    )
    by_id = {item["event_id"]: item for item in receipts}

    assert [event.event_id for event in selected] == ["E-2"]
    assert by_id["E-1"]["decision"] == "duplicate_replaced_by_higher_confidence"
    assert by_id["E-1"]["competitor_event_id"] == "E-2"
    assert by_id["E-1"]["interval_overlap_ms"] == 100.0
    assert by_id["E-2"]["selected"] is True


def test_overlapping_same_class_different_tracks_remain_distinct(
    default_config,
):
    events = [_event(1, 10_000, confidence=0.8), _event(2, 10_800, confidence=0.95)]
    for event, track_id in zip(events, (101, 202), strict=True):
        event.candidates[0].evidence = [
            {"object_name": "tube", "track_id": track_id}
        ]
    group, segment = _context(events)
    receipts = []

    selected = select_key_events(
        [group], [segment], events, default_config, decision_receipts=receipts
    )
    comparison = next(
        item
        for item in receipts
        if item["event_id"] == "E-2"
    )["facts"]["dedup_comparisons"][0]

    assert [event.event_id for event in selected] == ["E-1", "E-2"]
    assert comparison["stable_identity_available"] is True
    assert comparison["stable_identity_conflict"] is True
    assert comparison["is_duplicate"] is False


def test_overlapping_same_track_is_deduplicated(default_config):
    events = [_event(1, 10_000, confidence=0.8), _event(2, 10_800, confidence=0.95)]
    for event in events:
        event.candidates[0].evidence = [
            {"object_name": "tube", "track_id": 101}
        ]
    group, segment = _context(events)
    receipts = []

    selected = select_key_events(
        [group], [segment], events, default_config, decision_receipts=receipts
    )

    assert [event.event_id for event in selected] == ["E-2"]
    dropped = next(item for item in receipts if item["event_id"] == "E-1")
    assert dropped["shared_stable_identities"] == [("fp", "tube", 101)]


def test_hand_only_overlap_cannot_deduplicate_different_manipulated_objects(
    default_config,
):
    events = [
        _event(1, 10_000, confidence=0.8).model_copy(
            update={"objects": ["gloved_hand", "paper"]}
        ),
        _event(2, 10_800, confidence=0.95).model_copy(
            update={"objects": ["gloved_hand", "balance"]}
        ),
    ]
    group, segment = _context(events)
    receipts = []

    selected = select_key_events(
        [group], [segment], events, default_config, decision_receipts=receipts
    )
    by_id = {item["event_id"]: item for item in receipts}

    assert [event.event_id for event in selected] == ["E-1", "E-2"]
    assert by_id["E-1"]["selected"] is True
    assert by_id["E-2"]["selected"] is True
    assert by_id["E-2"]["shared_objects"] == []
    assert by_id["E-2"]["facts"]["dedup_comparisons"][0][
        "shared_actor_objects"
    ] == ["gloved_hand"]
    assert by_id["E-2"]["facts"]["dedup_comparisons"][0][
        "is_duplicate"
    ] is False
    assert by_id["E-2"]["receipt_schema_version"] == (
        "visioncortex-decision-receipt/1.0.0"
    )


def test_coverage_budget_preserves_timeline_end_instead_of_confidence_only(
    default_config,
):
    default_config["key_materials"][
        "max_per_action_type_per_atomic_experiment"
    ] = 3
    default_config["key_materials"]["adaptive_coverage_budget_enabled"] = False
    events = [
        _event(1, 1_000, confidence=0.99),
        _event(2, 2_000, confidence=0.98),
        _event(3, 3_000, confidence=0.97),
        _event(4, 400_000, confidence=0.60),
        _event(5, 800_000, confidence=0.20),
    ]
    group, segment = _context(events)
    receipts = []

    selected = select_key_events(
        [group], [segment], events, default_config, decision_receipts=receipts
    )

    selected_ids = {event.event_id for event in selected}
    assert selected_ids == {"E-1", "E-4", "E-5"}
    end_receipt = next(item for item in receipts if item["event_id"] == "E-5")
    assert "timeline_end_coverage" in end_receipt["facts"][
        "coverage_selection_reasons"
    ]


def test_long_experiment_grows_bounded_key_material_budget(default_config):
    key_cfg = default_config["key_materials"]
    key_cfg["max_per_action_type_per_atomic_experiment"] = 2
    key_cfg["max_per_action_type_ceiling"] = 6
    key_cfg["adaptive_budget_step_seconds"] = 600
    key_cfg["adaptive_per_action_type_additional_per_step"] = 1
    events = [
        _event(index, index * 300_000, confidence=0.9 - index * 0.01)
        for index in range(1, 6)
    ]
    group, segment = _context(events)
    segment.global_start_ms = 0
    segment.global_end_ms = 1_800_000
    group.global_start_ms = 0
    group.global_end_ms = 1_800_000

    selected = select_key_events(
        [group], [segment], events, default_config
    )

    assert len(selected) == 4
