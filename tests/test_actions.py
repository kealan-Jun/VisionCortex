from pathlib import Path

from labvision_evidence.actions import audit_candidates, build_experiment_segments
from labvision_evidence.grouping import (
    build_experiment_groups,
    normalize_experiment_segments,
    prepare_formal_experiment_segments,
)
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ViewInput,
    ViewRole,
    ExperimentSegment,
)


def _candidate(view_id: str, role: ViewRole, start: float) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=f"CAND-{view_id}",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        view_id=view_id,
        role=role,
        local_start_ms=start,
        local_end_ms=start + 1200,
        global_start_ms=start,
        global_end_ms=start + 1200,
        key_global_ms=start + 500,
        objects=["gloved_hand", "pipette"],
        confidence=0.88,
    )


def test_only_views_with_valid_evidence_participate(default_config):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
        ViewInput(view_id="empty01", role=ViewRole.THIRD_PERSON, video=Path("c.mp4")),
    ]
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp01",
            confidence=0.9,
            state="aligned",
        )
        for view in views
    }
    events, _ = audit_candidates(
        [_candidate("fp01", ViewRole.FIRST_PERSON, 10_000), _candidate("tp01", ViewRole.THIRD_PERSON, 10_100)],
        transforms,
        default_config,
    )
    segments = build_experiment_segments(events, views, default_config)
    assert segments[0].participating_views == ["fp01", "tp01"]
    assert "empty01" in segments[0].rejected_views


def test_component_only_events_remain_audited_but_cannot_bridge_boundaries(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def event(event_id, start, end, action, publication="primary"):
        candidates = [
            ActionCandidate(
                candidate_id=f"{event_id}-{view_id}",
                action_type=action,
                view_id=view_id,
                role=role,
                local_start_ms=start,
                local_end_ms=end,
                global_start_ms=start,
                global_end_ms=end,
                key_global_ms=(start + end) / 2,
                objects=["gloved_hand", "reagent_bottle"],
                confidence=0.9,
            )
            for view_id, role in (
                ("fp01", ViewRole.FIRST_PERSON),
                ("tp01", ViewRole.THIRD_PERSON),
            )
        ]
        return EvidenceEvent(
            event_id=event_id,
            action_type=action,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2,
            objects=["gloved_hand", "reagent_bottle"],
            confidence=0.9,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=candidates,
            state_machine={
                "publication": {
                    "status": publication,
                    "suppressed_by_event_id": (
                        "PRIMARY-LEFT" if publication == "component_only" else None
                    ),
                }
            },
        )

    events = [
        event("PRIMARY-LEFT", 10_000, 12_000, ActionType.LIQUID_MOVEMENT),
        event(
            "STATIC-CONTACT-COMPONENT",
            11_000,
            55_000,
            ActionType.HAND_OBJECT_CONTACT,
            publication="component_only",
        ),
        event("PRIMARY-RIGHT", 70_000, 72_000, ActionType.HAND_OBJECT_CONTACT),
    ]
    receipts = []

    segments = build_experiment_segments(
        events, views, default_config, decision_receipts=receipts
    )

    assert [segment.event_ids for segment in segments] == [
        ["PRIMARY-LEFT"],
        ["PRIMARY-RIGHT"],
    ]
    quarantine = next(
        receipt
        for receipt in receipts
        if receipt["decision_type"]
        == "component_publication_boundary_quarantine"
    )
    assert quarantine["verdict"] == "quarantined"
    assert quarantine["facts"]["audit_ledger_membership_changed"] is False


def test_accepted_primary_leading_chain_extends_full_timeline_boundary(
    default_config,
):
    default_config["segmentation"]["accepted_leading_context_enabled"] = True
    default_config["segmentation"][
        "accepted_leading_context_max_gap_seconds"
    ] = 10.0
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    leading = EvidenceEvent(
        event_id="LEADING-PRIMARY",
        action_type=ActionType.DEVICE_PANEL_OPERATION,
        global_start_ms=70_000,
        global_end_ms=80_000,
        key_global_ms=75_000,
        objects=["balance", "gloved_hand"],
        confidence=0.9,
        accepted=True,
        audit_reason="strong third-person preparation",
        supporting_views=["tp01"],
        supporting_roles=[ViewRole.THIRD_PERSON],
        candidates=[
            ActionCandidate(
                candidate_id="LEADING-TP",
                action_type=ActionType.DEVICE_PANEL_OPERATION,
                view_id="tp01",
                role=ViewRole.THIRD_PERSON,
                local_start_ms=70_000,
                local_end_ms=80_000,
                global_start_ms=70_000,
                global_end_ms=80_000,
                key_global_ms=75_000,
                objects=["balance", "gloved_hand"],
                confidence=0.9,
            )
        ],
        state_machine={"publication": {"status": "primary"}},
    )
    opener = leading.model_copy(
        update={
            "event_id": "DUAL-OPENER",
            "global_start_ms": 85_000,
            "global_end_ms": 90_000,
            "key_global_ms": 87_500,
            "objects": ["gloved_hand", "spatula"],
            "supporting_views": ["fp01", "tp01"],
            "supporting_roles": [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            "candidates": [
                ActionCandidate(
                    candidate_id=f"OPENER-{view_id}",
                    action_type=ActionType.DEVICE_PANEL_OPERATION,
                    view_id=view_id,
                    role=role,
                    local_start_ms=85_000,
                    local_end_ms=90_000,
                    global_start_ms=85_000,
                    global_end_ms=90_000,
                    key_global_ms=87_500,
                    objects=["gloved_hand", "spatula"],
                    confidence=0.9,
                )
                for view_id, role in (
                    ("fp01", ViewRole.FIRST_PERSON),
                    ("tp01", ViewRole.THIRD_PERSON),
                )
            ],
        }
    )
    receipts = []

    segments = build_experiment_segments(
        [leading, opener], views, default_config, decision_receipts=receipts
    )

    assert len(segments) == 1
    assert segments[0].global_start_ms == 68_000
    assert segments[0].event_ids == ["LEADING-PRIMARY", "DUAL-OPENER"]
    boundary = next(
        receipt
        for receipt in receipts
        if receipt["decision_type"] == "raw_segment_boundary"
    )
    assert boundary["facts"]["leading_context_event_ids"] == [
        "LEADING-PRIMARY"
    ]


def test_fused_rich_repeated_primary_sequences_split_at_largest_inactive_gap(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def event(event_id, start, end, action):
        return EvidenceEvent(
            event_id=event_id,
            action_type=action,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2.0,
            objects=["gloved_hand", "pipette", "tube"],
            confidence=0.9,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=[],
        )

    events = [
        event("LEFT-1", 10_000, 14_000, ActionType.LIQUID_MOVEMENT),
        event("LEFT-2", 17_000, 21_000, ActionType.HAND_OBJECT_CONTACT),
        event("LEFT-3", 24_000, 28_000, ActionType.HAND_OBJECT_CONTACT),
        event("LEFT-4", 31_000, 35_000, ActionType.LIQUID_MOVEMENT),
        event("RIGHT-1", 45_000, 49_000, ActionType.HAND_OBJECT_CONTACT),
        event("RIGHT-2", 52_000, 56_000, ActionType.LIQUID_MOVEMENT),
        event("RIGHT-3", 59_000, 63_000, ActionType.HAND_OBJECT_CONTACT),
        event("RIGHT-4", 66_000, 70_000, ActionType.LIQUID_MOVEMENT),
    ]
    receipts = []

    segments = build_experiment_segments(
        events,
        views,
        default_config,
        coarse_windows=[
            ActionCandidate(
                candidate_id="COARSE-ONE-WIDE-ENVELOPE",
                action_type=ActionType.OBJECT_MOVEMENT,
                view_id="fp01",
                role=ViewRole.FIRST_PERSON,
                local_start_ms=0.0,
                local_end_ms=80_000.0,
                global_start_ms=0.0,
                global_end_ms=80_000.0,
                key_global_ms=40_000.0,
                objects=["motion"],
                confidence=0.9,
            )
        ],
        decision_receipts=receipts,
    )

    assert [segment.event_ids for segment in segments] == [
        ["LEFT-1", "LEFT-2", "LEFT-3", "LEFT-4"],
        ["RIGHT-1", "RIGHT-2", "RIGHT-3", "RIGHT-4"],
    ]
    split_receipt = next(
        receipt
        for receipt in receipts
        if receipt["decision_type"] == "raw_atomic_sequence_split"
    )
    assert split_receipt["verdict"] == "split"
    assert split_receipt["facts"]["inactive_gap_ms"] == 10_000
    assert split_receipt["facts"]["repeated_primary_actions"] == [
        "liquid_movement"
    ]


def test_eight_point_six_second_pause_does_not_split_one_rich_primary_sequence(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def event(event_id, start, end, action):
        return EvidenceEvent(
            event_id=event_id,
            action_type=action,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2.0,
            objects=["gloved_hand", "pipette", "tube"],
            confidence=0.9,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=[],
        )

    events = [
        event("LEFT-1", 10_000, 14_000, ActionType.LIQUID_MOVEMENT),
        event("LEFT-2", 17_000, 21_000, ActionType.HAND_OBJECT_CONTACT),
        event("LEFT-3", 24_000, 28_000, ActionType.HAND_OBJECT_CONTACT),
        event("LEFT-4", 31_000, 35_000, ActionType.LIQUID_MOVEMENT),
        event("RIGHT-1", 43_600, 47_600, ActionType.HAND_OBJECT_CONTACT),
        event("RIGHT-2", 50_600, 54_600, ActionType.LIQUID_MOVEMENT),
        event("RIGHT-3", 57_600, 61_600, ActionType.HAND_OBJECT_CONTACT),
        event("RIGHT-4", 64_600, 68_600, ActionType.LIQUID_MOVEMENT),
    ]
    receipts = []
    coarse = ActionCandidate(
        candidate_id="COARSE-ONE-WIDE-ENVELOPE",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=0.0,
        local_end_ms=80_000.0,
        global_start_ms=0.0,
        global_end_ms=80_000.0,
        key_global_ms=40_000.0,
        objects=["motion"],
        confidence=0.9,
    )

    segments = build_experiment_segments(
        events,
        views,
        default_config,
        coarse_windows=[coarse],
        decision_receipts=receipts,
    )

    assert len(segments) == 1
    assert segments[0].event_ids == [event.event_id for event in events]
    assert not any(
        receipt["decision_type"] == "raw_atomic_sequence_split"
        for receipt in receipts
    )


def test_strong_single_role_action_is_accepted_with_paired_media_policy(default_config):
    transforms = {
        "fp01": AlignmentTransform(
            view_id="fp01",
            reference_view_id="fp01",
            confidence=0.95,
            state="aligned",
        ),
        "tp01": AlignmentTransform(
            view_id="tp01",
            reference_view_id="fp01",
            confidence=0.90,
            state="aligned",
        ),
    }

    events, rejected = audit_candidates(
        [_candidate("fp01", ViewRole.FIRST_PERSON, 10_000)],
        transforms,
        default_config,
    )

    assert len(events) == 1
    assert events[0].accepted is True
    assert events[0].supporting_roles == [ViewRole.FIRST_PERSON]
    assert rejected == []


def test_single_view_container_state_uses_direct_transition_threshold(
    default_config,
):
    transforms = {
        "fp01": AlignmentTransform(
            view_id="fp01",
            reference_view_id="fp01",
            confidence=0.95,
            state="aligned",
        )
    }

    def candidate(action, candidate_id):
        return ActionCandidate(
            candidate_id=candidate_id,
            action_type=action,
            view_id="fp01",
            role=ViewRole.FIRST_PERSON,
            local_start_ms=10_000,
            local_end_ms=11_200,
            global_start_ms=10_000,
            global_end_ms=11_200,
            key_global_ms=10_600,
            objects=["gloved_hand", "tube_cap"],
            confidence=0.72,
        )

    state_events, _ = audit_candidates(
        [candidate(ActionType.CONTAINER_STATE_CHANGE, "STATE")],
        transforms,
        default_config,
    )
    contact_events, _ = audit_candidates(
        [candidate(ActionType.HAND_OBJECT_CONTACT, "CONTACT")],
        transforms,
        default_config,
    )

    assert state_events[0].accepted is True
    assert contact_events[0].accepted is False


def test_cross_role_context_admits_short_state_candidate_for_semantic_review(
    default_config,
):
    transforms = {
        view_id: AlignmentTransform(
            view_id=view_id,
            reference_view_id="fp01",
            confidence=0.95,
            state="aligned",
        )
        for view_id in ("fp01", "tp01")
    }
    state = ActionCandidate(
        candidate_id="STATE-TP",
        action_type=ActionType.CONTAINER_STATE_CHANGE,
        view_id="tp01",
        role=ViewRole.THIRD_PERSON,
        local_start_ms=6_000,
        local_end_ms=6_200,
        global_start_ms=6_000,
        global_end_ms=6_200,
        key_global_ms=6_100,
        objects=["gloved_hand", "bottle_cap"],
        confidence=0.39,
    )
    movement = ActionCandidate(
        candidate_id="MOVE-FP",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=4_800,
        local_end_ms=7_900,
        global_start_ms=4_800,
        global_end_ms=7_900,
        key_global_ms=7_300,
        objects=["pipette"],
        confidence=0.60,
    )

    events, rejected = audit_candidates([movement, state], transforms, default_config)
    state_event = next(
        event for event in events if event.action_type == ActionType.CONTAINER_STATE_CHANGE
    )

    assert state_event.accepted is True
    assert state_event.supporting_views == ["tp01"]
    assert state_event.supporting_roles == [ViewRole.THIRD_PERSON]
    assert "候选事实仍需多模态确认" in state_event.audit_reason
    assert any("MOVE-FP" in item for item in state_event.uncertainty)
    assert state_event.observability["semantic_recall_admission"][
        "candidate_action_directly_confirmed"
    ] is False
    assert all(item["event_id"] != state_event.event_id for item in rejected)

    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    segments = build_experiment_segments(events, views, default_config)
    assert len(segments) == 1
    assert segments[0].participating_views == ["fp01", "tp01"]
    formal, receipts = prepare_formal_experiment_segments(
        segments,
        events,
        views,
        [],
        default_config,
    )
    assert len(formal) == 1
    assert any(
        item.get("decision") == "promoted_provisional_semantic_review"
        and item.get("candidate_action_directly_confirmed") is False
        for item in receipts
    )


def test_production_semantic_review_requires_independent_dual_movement_anchor(
    default_config,
):
    default_config["segmentation"][
        "semantic_review_requires_dual_role_movement_anchor"
    ] = True
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    event = EvidenceEvent(
        event_id="STATE-PROVISIONAL",
        action_type=ActionType.CONTAINER_STATE_CHANGE,
        global_start_ms=10_000,
        global_end_ms=11_000,
        key_global_ms=10_500,
        objects=["gloved_hand", "bottle_cap"],
        confidence=0.6,
        accepted=True,
        audit_reason="single-role hypothesis plus opposite-role activity",
        supporting_views=["tp01"],
        supporting_roles=[ViewRole.THIRD_PERSON],
        candidates=[],
        observability={
            "semantic_review_priority": "required",
            "can_cv_directly_prove_action": False,
            "semantic_recall_admission": {
                "mandatory_semantic_review": True,
                "candidate_action_directly_confirmed": False,
                "context_view_id": "fp01",
                "context_global_start_ms": 9_800,
                "context_global_end_ms": 11_200,
            },
        },
        state_machine={
            "lifecycle_state": "incomplete_end",
            "publication": {"status": "provisional_semantic_review"},
        },
    )
    segment = ExperimentSegment(
        segment_id="EXP-PROVISIONAL",
        global_start_ms=8_000,
        global_end_ms=14_000,
        event_ids=[event.event_id],
        participating_views=["fp01", "tp01"],
    )

    formal, receipts = prepare_formal_experiment_segments(
        [segment], [event], views, [], default_config
    )

    assert formal == []
    quarantine = next(
        item
        for item in receipts
        if item.get("decision") == "quarantined_missing_corroborated_opener"
    )
    assert quarantine[
        "semantic_review_requires_dual_role_movement_anchor"
    ] is True
    assert quarantine["required_semantic_event_ids"] == [event.event_id]


def test_cross_view_movement_with_required_semantic_event_gets_model_review(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    movement = EvidenceEvent(
        event_id="MOVE-DUAL",
        action_type=ActionType.OBJECT_MOVEMENT,
        global_start_ms=8_000,
        global_end_ms=9_600,
        key_global_ms=8_850,
        objects=["tube"],
        confidence=0.62,
        accepted=True,
        audit_reason="cross-view direct movement",
        supporting_views=["fp01", "tp01"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        observability={
            "cv_ontology_support": "direct_cv",
            "semantic_review_priority": "optional",
            "can_cv_directly_prove_action": True,
        },
        state_machine={"lifecycle_state": "completed"},
    )
    semantic = EvidenceEvent(
        event_id="SEMANTIC-REQUIRED",
        action_type=ActionType.LIQUID_MOVEMENT,
        global_start_ms=20_100,
        global_end_ms=21_900,
        key_global_ms=21_100,
        objects=["pipette", "sample_bottle"],
        confidence=0.61,
        accepted=True,
        audit_reason="single-view high-risk hypothesis",
        supporting_views=["fp01"],
        supporting_roles=[ViewRole.FIRST_PERSON],
        candidates=[],
        observability={
            "cv_ontology_support": "indirect_cv",
            "semantic_review_priority": "required",
            "can_cv_directly_prove_action": False,
        },
        state_machine={"lifecycle_state": "completed"},
    )
    segment = ExperimentSegment(
        segment_id="EXP-SEMANTIC-RESCUE",
        global_start_ms=6_000,
        global_end_ms=24_900,
        event_ids=[movement.event_id, semantic.event_id],
        participating_views=["fp01", "tp01"],
    )

    formal, receipts = prepare_formal_experiment_segments(
        [segment], [movement, semantic], views, [], default_config
    )

    assert [item.segment_id for item in formal] == [segment.segment_id]
    promoted = next(
        item
        for item in receipts
        if item.get("decision") == "promoted_provisional_semantic_review"
    )
    assert promoted["candidate_action_directly_confirmed"] is False
    assert promoted["provisional_movement_opener_event_ids"] == ["MOVE-DUAL"]
    assert promoted["required_semantic_event_ids"] == ["SEMANTIC-REQUIRED"]


def test_cross_view_movement_alone_stays_quarantined_without_model_review(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    movement = EvidenceEvent(
        event_id="MOVE-ONLY",
        action_type=ActionType.OBJECT_MOVEMENT,
        global_start_ms=8_000,
        global_end_ms=9_600,
        key_global_ms=8_850,
        objects=["tube"],
        confidence=0.62,
        accepted=True,
        audit_reason="cross-view direct movement",
        supporting_views=["fp01", "tp01"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        observability={
            "cv_ontology_support": "direct_cv",
            "semantic_review_priority": "optional",
            "can_cv_directly_prove_action": True,
        },
        state_machine={"lifecycle_state": "completed"},
    )
    segment = ExperimentSegment(
        segment_id="EXP-MOVEMENT-ONLY",
        global_start_ms=6_000,
        global_end_ms=12_600,
        event_ids=[movement.event_id],
        participating_views=["fp01", "tp01"],
    )

    formal, receipts = prepare_formal_experiment_segments(
        [segment], [movement], views, [], default_config
    )

    assert formal == []
    assert any(
        item.get("decision") == "quarantined_missing_corroborated_opener"
        for item in receipts
    )


def test_below_floor_state_candidate_with_context_stays_rejected(
    default_config,
):
    transforms = {
        view_id: AlignmentTransform(
            view_id=view_id,
            reference_view_id="fp01",
            confidence=0.95,
            state="aligned",
        )
        for view_id in ("fp01", "tp01")
    }
    state = ActionCandidate(
        candidate_id="STATE-TP",
        action_type=ActionType.CONTAINER_STATE_CHANGE,
        view_id="tp01",
        role=ViewRole.THIRD_PERSON,
        local_start_ms=6_000,
        local_end_ms=6_200,
        global_start_ms=6_000,
        global_end_ms=6_200,
        key_global_ms=6_100,
        objects=["gloved_hand", "bottle_cap"],
        confidence=0.37,
    )
    movement = ActionCandidate(
        candidate_id="MOVE-FP",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=4_800,
        local_end_ms=7_900,
        global_start_ms=4_800,
        global_end_ms=7_900,
        key_global_ms=7_300,
        objects=["pipette"],
        confidence=0.60,
    )

    events, rejected = audit_candidates(
        [movement, state], transforms, default_config
    )
    state_event = next(
        event
        for event in events
        if event.action_type == ActionType.CONTAINER_STATE_CHANGE
    )

    assert state_event.accepted is False
    assert "semantic_recall_admission" not in (state_event.observability or {})
    assert any(item["event_id"] == state_event.event_id for item in rejected)


def test_short_state_candidate_without_cross_role_context_stays_rejected(
    default_config,
):
    transforms = {
        "tp01": AlignmentTransform(
            view_id="tp01",
            reference_view_id="tp01",
            confidence=0.95,
            state="aligned",
        )
    }
    state = ActionCandidate(
        candidate_id="STATE-TP",
        action_type=ActionType.CONTAINER_STATE_CHANGE,
        view_id="tp01",
        role=ViewRole.THIRD_PERSON,
        local_start_ms=6_000,
        local_end_ms=6_200,
        global_start_ms=6_000,
        global_end_ms=6_200,
        key_global_ms=6_100,
        objects=["gloved_hand", "bottle_cap"],
        confidence=0.46,
    )

    events, rejected = audit_candidates([state], transforms, default_config)

    assert events[0].accepted is False
    assert rejected[0]["event_id"] == events[0].event_id


def test_single_view_tail_extends_only_an_existing_cross_view_boundary(default_config):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp01",
            confidence=0.9,
            state="aligned",
        )
        for view in views
    }
    core, _ = audit_candidates(
        [
            _candidate("fp01", ViewRole.FIRST_PERSON, 100_000),
            _candidate("tp01", ViewRole.THIRD_PERSON, 100_100),
        ],
        transforms,
        default_config,
    )
    bridge = EvidenceEvent(
        event_id="TAIL-BRIDGE",
        action_type=ActionType.OBJECT_MOVEMENT,
        global_start_ms=110_000,
        global_end_ms=125_000,
        key_global_ms=117_000,
        objects=["cleaning_tool"],
        confidence=0.55,
        accepted=False,
        audit_reason="single-view context only",
        supporting_views=["fp01"],
        supporting_roles=[ViewRole.FIRST_PERSON],
        candidates=[],
    )
    tail = bridge.model_copy(
        update={
            "event_id": "TAIL-STRONG",
            "global_start_ms": 126_000,
            "global_end_ms": 135_000,
            "key_global_ms": 130_000,
            "confidence": 0.82,
        }
    )
    coarse_window = ActionCandidate(
        candidate_id="COARSE-1",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=90_000,
        local_end_ms=160_000,
        global_start_ms=90_000,
        global_end_ms=160_000,
        key_global_ms=120_000,
        objects=["lab_bench"],
        confidence=0.9,
    )

    segments = build_experiment_segments(
        [*core, bridge, tail],
        views,
        default_config,
        coarse_windows=[coarse_window],
    )

    assert len(segments) == 1
    assert segments[0].global_end_ms == 138_000
    assert "TAIL-BRIDGE" not in segments[0].event_ids
    assert "TAIL-STRONG" not in segments[0].event_ids


def test_unrelated_single_view_tail_does_not_extend_boundary(default_config):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp01",
            confidence=0.9,
            state="aligned",
        )
        for view in views
    }
    core, _ = audit_candidates(
        [
            _candidate("fp01", ViewRole.FIRST_PERSON, 100_000),
            _candidate("tp01", ViewRole.THIRD_PERSON, 100_100),
        ],
        transforms,
        default_config,
    )
    unrelated = EvidenceEvent(
        event_id="TAIL-PAPER",
        action_type=ActionType.OBJECT_MOVEMENT,
        global_start_ms=104_000,
        global_end_ms=110_000,
        key_global_ms=106_000,
        objects=["paper"],
        confidence=0.9,
        accepted=False,
        audit_reason="single-view context only",
        supporting_views=["fp01"],
        supporting_roles=[ViewRole.FIRST_PERSON],
        candidates=[],
    )
    coarse_window = ActionCandidate(
        candidate_id="COARSE-TAIL",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=90_000,
        local_end_ms=160_000,
        global_start_ms=90_000,
        global_end_ms=160_000,
        key_global_ms=120_000,
        objects=["lab_bench"],
        confidence=0.9,
    )

    segments = build_experiment_segments(
        [*core, unrelated], views, default_config, coarse_windows=[coarse_window]
    )

    assert segments[0].global_end_ms == 104_300


def test_overlapping_windows_collapse_and_short_equipment_prelude_is_suppressed(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def evidence(event_id, action, start, end, objects):
        return EvidenceEvent(
            event_id=event_id,
            action_type=action,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2,
            objects=objects,
            confidence=0.9,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=[],
        )

    events = [
        evidence("CORE-1", ActionType.HAND_OBJECT_CONTACT, 100_000, 110_000, ["pipette"]),
        evidence("CORE-2", ActionType.OBJECT_MOVEMENT, 108_000, 120_000, ["tube"]),
        evidence("PREP", ActionType.OBJECT_MOVEMENT, 200_000, 203_000, ["balance"]),
        evidence("WEIGH", ActionType.DEVICE_PANEL_OPERATION, 216_000, 250_000, ["balance"]),
    ]
    segments = [
        ExperimentSegment(
            segment_id="EXP-1",
            global_start_ms=98_000,
            global_end_ms=113_000,
            event_ids=["CORE-1"],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-2",
            global_start_ms=108_000,
            global_end_ms=123_000,
            event_ids=["CORE-2"],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-PREP",
            global_start_ms=198_000,
            global_end_ms=206_000,
            event_ids=["PREP"],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-WEIGH",
            global_start_ms=216_000,
            global_end_ms=253_000,
            event_ids=["WEIGH"],
            participating_views=["fp01", "tp01"],
        ),
    ]

    normalized = normalize_experiment_segments(segments, events, views, default_config)

    assert [segment.segment_id for segment in normalized] == ["EXP-1", "EXP-WEIGH"]
    assert normalized[0].event_ids == ["CORE-1", "CORE-2"]
    assert normalized[0].global_end_ms == 123_000


def test_short_dual_view_fragments_with_carried_objects_form_one_atomic_experiment(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def evidence(event_id, action, start, end, objects):
        track_ids = {
            "sample_bottle": 101,
            "tube": 102,
            "tube_rack": 103,
            "pipette": 104,
        }
        candidates = [
            ActionCandidate(
                candidate_id=f"{event_id}-{obj}",
                action_type=action,
                view_id="fp01",
                role=ViewRole.FIRST_PERSON,
                local_start_ms=start,
                local_end_ms=end,
                global_start_ms=start,
                global_end_ms=end,
                key_global_ms=(start + end) / 2,
                objects=[obj],
                confidence=0.8,
                evidence=[{"track_id": track_ids[obj]}],
            )
            for obj in objects
            if obj in track_ids
        ]
        return EvidenceEvent(
            event_id=event_id,
            action_type=action,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2,
            objects=objects,
            confidence=0.8,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=candidates,
        )

    events = [
        evidence(
            "TRANSFER",
            ActionType.LIQUID_MOVEMENT,
            2_000,
            47_000,
            ["pipette", "sample_bottle", "tube", "tube_rack"],
        ),
        evidence(
            "CONTACT",
            ActionType.HAND_OBJECT_CONTACT,
            63_600,
            97_000,
            ["gloved_hand", "sample_bottle", "tube", "tube_rack"],
        ),
    ]
    segments = [
        ExperimentSegment(
            segment_id="EXP-1",
            global_start_ms=0,
            global_end_ms=50_000,
            event_ids=["TRANSFER"],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-2",
            global_start_ms=61_600,
            global_end_ms=100_000,
            event_ids=["CONTACT"],
            participating_views=["fp01", "tp01"],
        ),
    ]

    normalized = normalize_experiment_segments(segments, events, views, default_config)
    groups = build_experiment_groups(normalized, events, views, default_config)

    assert len(normalized) == 1
    assert normalized[0].event_ids == ["TRANSFER", "CONTACT"]
    assert len(groups) == 1
    assert groups[0].continuity_type == "independent"
    assert groups[0].atomic_experiment_ids == ["EXP-1"]


def test_explicit_state_transition_remains_a_continuous_two_atomic_chain(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def evidence(event_id, action, start, end):
        candidates = [
            ActionCandidate(
                candidate_id=f"{event_id}-{obj}",
                action_type=action,
                view_id="fp01",
                role=ViewRole.FIRST_PERSON,
                local_start_ms=start,
                local_end_ms=end,
                global_start_ms=start,
                global_end_ms=end,
                key_global_ms=(start + end) / 2,
                objects=[obj],
                confidence=0.9,
                evidence=[{"track_id": track_id}],
            )
            for obj, track_id in (("sample_bottle", 201), ("tube", 202))
        ]
        return EvidenceEvent(
            event_id=event_id,
            action_type=action,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2,
            objects=["sample_bottle", "tube"],
            confidence=0.9,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=candidates,
        )

    events = [
        evidence("STATE", ActionType.CONTAINER_STATE_CHANGE, 10_000, 20_000),
        evidence("TRANSFER", ActionType.LIQUID_MOVEMENT, 34_600, 45_000),
    ]
    segments = [
        ExperimentSegment(
            segment_id="EXP-1",
            global_start_ms=8_000,
            global_end_ms=23_000,
            event_ids=["STATE"],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-2",
            global_start_ms=34_600,
            global_end_ms=48_000,
            event_ids=["TRANSFER"],
            participating_views=["fp01", "tp01"],
        ),
    ]

    normalized = normalize_experiment_segments(segments, events, views, default_config)
    groups = build_experiment_groups(normalized, events, views, default_config)

    assert [segment.segment_id for segment in normalized] == ["EXP-1", "EXP-2"]
    assert len(groups) == 1
    assert groups[0].continuity_type == "continuous"
    assert groups[0].atomic_experiment_ids == ["EXP-1", "EXP-2"]


def test_stationary_track_identity_across_long_gap_requires_context_chain(
    default_config,
):
    default_config["continuity"]["stable_identity_max_gap_seconds"] = 10.0
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def event(event_id, start, end):
        candidates = [
            ActionCandidate(
                candidate_id=f"{event_id}-{name}",
                action_type=ActionType.HAND_OBJECT_CONTACT,
                view_id="tp01",
                role=ViewRole.THIRD_PERSON,
                local_start_ms=start,
                local_end_ms=end,
                global_start_ms=start,
                global_end_ms=end,
                key_global_ms=(start + end) / 2,
                objects=[name],
                confidence=0.9,
                evidence=[{"track_id": track_id}],
            )
            for name, track_id in (("reagent_bottle", 11), ("tube_rack", 12))
        ]
        return EvidenceEvent(
            event_id=event_id,
            action_type=ActionType.HAND_OBJECT_CONTACT,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2,
            objects=["gloved_hand", "reagent_bottle", "tube_rack"],
            confidence=0.9,
            accepted=True,
            audit_reason="same stationary identities",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=candidates,
        )

    events = [event("LEFT", 10_000, 20_000), event("RIGHT", 45_000, 55_000)]
    segments = [
        ExperimentSegment(
            segment_id="EXP-LEFT",
            global_start_ms=8_000,
            global_end_ms=23_000,
            event_ids=["LEFT"],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-RIGHT",
            global_start_ms=43_000,
            global_end_ms=58_000,
            event_ids=["RIGHT"],
            participating_views=["fp01", "tp01"],
        ),
    ]
    receipts = []

    groups = build_experiment_groups(
        segments,
        events,
        views,
        default_config,
        decision_receipts=receipts,
    )

    assert len(groups) == 2
    edge = next(
        receipt
        for receipt in receipts
        if receipt["decision_type"] == "experiment_continuity_edge"
    )
    assert edge["verdict"] == "rejected"
    assert edge["reason_codes"] == ["identity_gap_requires_context_chain"]


def test_rich_adjacent_sequences_remain_two_continuous_atomic_experiments(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def event(event_id, start, end, action):
        candidates = [
            ActionCandidate(
                candidate_id=f"{event_id}-{obj}",
                action_type=action,
                view_id="fp01",
                role=ViewRole.FIRST_PERSON,
                local_start_ms=start,
                local_end_ms=end,
                global_start_ms=start,
                global_end_ms=end,
                key_global_ms=(start + end) / 2,
                objects=[obj],
                confidence=0.85,
                evidence=[{"track_id": track_id}],
            )
            for obj, track_id in (("sample_bottle", 401), ("tube", 402))
        ]
        return EvidenceEvent(
            event_id=event_id,
            action_type=action,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2,
            objects=["gloved_hand", "sample_bottle", "tube"],
            confidence=0.85,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=candidates,
        )

    events = [
        event("LEFT-1", 10_000, 14_000, ActionType.HAND_OBJECT_CONTACT),
        event("LEFT-2", 17_000, 21_000, ActionType.LIQUID_MOVEMENT),
        event("LEFT-3", 24_000, 28_000, ActionType.HAND_OBJECT_CONTACT),
        event("LEFT-4", 31_000, 35_000, ActionType.LIQUID_MOVEMENT),
        event("RIGHT-1", 45_000, 49_000, ActionType.HAND_OBJECT_CONTACT),
        event("RIGHT-2", 52_000, 56_000, ActionType.LIQUID_MOVEMENT),
        event("RIGHT-3", 59_000, 63_000, ActionType.HAND_OBJECT_CONTACT),
        event("RIGHT-4", 66_000, 70_000, ActionType.LIQUID_MOVEMENT),
    ]
    segments = [
        ExperimentSegment(
            segment_id="EXP-LEFT",
            global_start_ms=8_000,
            global_end_ms=37_000,
            event_ids=[item.event_id for item in events[:4]],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-RIGHT",
            global_start_ms=43_000,
            global_end_ms=73_000,
            event_ids=[item.event_id for item in events[4:]],
            participating_views=["fp01", "tp01"],
        ),
    ]
    receipts = []

    normalized = normalize_experiment_segments(
        segments,
        events,
        views,
        default_config,
        decision_receipts=receipts,
    )
    groups = build_experiment_groups(
        normalized,
        events,
        views,
        default_config,
    )

    assert [item.segment_id for item in normalized] == ["EXP-LEFT", "EXP-RIGHT"]
    assert len(groups) == 1
    assert groups[0].continuity_type == "continuous"
    assert groups[0].atomic_experiment_ids == ["EXP-LEFT", "EXP-RIGHT"]
    fragment_receipt = next(
        item for item in receipts if item["decision_type"] == "atomic_fragment_merge"
    )
    assert fragment_receipt["verdict"] == "rejected"
    assert fragment_receipt["reason_codes"] == [
        "complete_action_sequences_not_fragments"
    ]


def test_single_carried_object_does_not_join_independent_atomic_experiments(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def event(event_id, start, end, objects):
        track_ids = {"tube": 301, "tube_rack": 302}
        candidates = [
            ActionCandidate(
                candidate_id=f"{event_id}-{obj}",
                action_type=ActionType.OBJECT_MOVEMENT,
                view_id="fp01",
                role=ViewRole.FIRST_PERSON,
                local_start_ms=start,
                local_end_ms=end,
                global_start_ms=start,
                global_end_ms=end,
                key_global_ms=(start + end) / 2.0,
                objects=[obj],
                confidence=0.9,
                evidence=[{"track_id": track_ids[obj]}],
            )
            for obj in objects
            if obj in track_ids
        ]
        return EvidenceEvent(
            event_id=event_id,
            action_type=ActionType.OBJECT_MOVEMENT,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2.0,
            objects=objects,
            confidence=0.9,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=candidates,
        )

    single_object_events = [
        event("LEFT", 10_000, 20_000, ["tube"]),
        event("RIGHT", 37_500, 45_000, ["tube"]),
    ]
    segments = [
        ExperimentSegment(
            segment_id="EXP-1",
            global_start_ms=8_000,
            global_end_ms=20_000,
            event_ids=["LEFT"],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-2",
            global_start_ms=37_500,
            global_end_ms=48_000,
            event_ids=["RIGHT"],
            participating_views=["fp01", "tp01"],
        ),
    ]

    groups = build_experiment_groups(
        segments, single_object_events, views, default_config
    )

    assert len(groups) == 2
    assert all(group.continuity_type == "independent" for group in groups)

    two_object_events = [
        event("LEFT", 10_000, 20_000, ["tube", "tube_rack"]),
        event("RIGHT", 92_700, 100_000, ["tube", "tube_rack"]),
    ]
    two_object_segments = [
        segments[0].model_copy(update={"event_ids": ["LEFT"]}),
        segments[1].model_copy(
            update={
                "global_start_ms": 92_700,
                "global_end_ms": 103_000,
                "event_ids": ["RIGHT"],
            }
        ),
    ]

    groups = build_experiment_groups(
        two_object_segments, two_object_events, views, default_config
    )

    assert len(groups) == 1
    assert groups[0].continuity_type == "continuous"
    assert len(groups[0].atomic_experiment_ids) == 2


def test_incomplete_liquid_hypothesis_cannot_pull_start_before_direct_contact(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]

    def evidence(event_id, action, start, end, objects):
        candidates = [
            ActionCandidate(
                candidate_id=f"{event_id}-{view_id}",
                action_type=action,
                view_id=view_id,
                role=role,
                local_start_ms=start,
                local_end_ms=end,
                global_start_ms=start,
                global_end_ms=end,
                key_global_ms=(start + end) / 2,
                objects=objects,
                confidence=0.8,
            )
            for view_id, role in (
                ("fp01", ViewRole.FIRST_PERSON),
                ("tp01", ViewRole.THIRD_PERSON),
            )
        ]
        return EvidenceEvent(
            event_id=event_id,
            action_type=action,
            global_start_ms=start,
            global_end_ms=end,
            key_global_ms=(start + end) / 2,
            objects=objects,
            confidence=0.8,
            accepted=True,
            audit_reason="cross-view evidence",
            supporting_views=["fp01", "tp01"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=candidates,
        )

    liquid = evidence(
        "LIQUID-TIP-ONLY",
        ActionType.LIQUID_MOVEMENT,
        100_000,
        101_800,
        ["spearhead", "sample_bottle", "tube"],
    )
    contact = evidence(
        "DIRECT-CONTACT",
        ActionType.HAND_OBJECT_CONTACT,
        110_900,
        113_900,
        ["gloved_hand", "tube"],
    )
    coarse_window = ActionCandidate(
        candidate_id="COARSE",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=90_000,
        local_end_ms=120_000,
        global_start_ms=90_000,
        global_end_ms=120_000,
        key_global_ms=105_000,
        objects=["tube"],
        confidence=0.9,
    )

    segments = build_experiment_segments(
        [liquid, contact],
        views,
        default_config,
        coarse_windows=[coarse_window],
    )

    assert len(segments) == 1
    assert segments[0].global_start_ms == 108_900
    assert segments[0].event_ids == ["DIRECT-CONTACT"]

    legacy_segment = ExperimentSegment(
        segment_id="EXP-LEGACY",
        global_start_ms=98_000,
        global_end_ms=116_900,
        event_ids=["LIQUID-TIP-ONLY", "DIRECT-CONTACT"],
        participating_views=["fp01", "tp01"],
        micro_segments=[
            {"evidence_event_id": "LIQUID-TIP-ONLY", "start_global_ms": 100_000},
            {"evidence_event_id": "DIRECT-CONTACT", "start_global_ms": 110_900},
        ],
    )
    replayed = normalize_experiment_segments(
        [legacy_segment], [liquid, contact], views, default_config
    )
    assert replayed[0].global_start_ms == 108_900
    assert replayed[0].event_ids == ["DIRECT-CONTACT"]
    assert [
        item["evidence_event_id"] for item in replayed[0].micro_segments
    ] == ["DIRECT-CONTACT"]

    complete_liquid = liquid.model_copy(
        update={
            "event_id": "LIQUID-WITH-PIPETTE",
            "objects": ["pipette", "spearhead", "sample_bottle", "tube"],
        }
    )
    complete_segments = build_experiment_segments(
        [complete_liquid, contact],
        views,
        default_config,
        coarse_windows=[coarse_window],
    )
    assert complete_segments[0].global_start_ms == 98_000
    assert complete_segments[0].event_ids == ["LIQUID-WITH-PIPETTE", "DIRECT-CONTACT"]


def test_single_role_prelude_cannot_open_dual_view_experiment(default_config):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    prelude = EvidenceEvent(
        event_id="TP-PRELUDE",
        action_type=ActionType.DEVICE_PANEL_OPERATION,
        global_start_ms=100_000,
        global_end_ms=100_200,
        key_global_ms=100_100,
        objects=["balance", "hand"],
        confidence=0.88,
        accepted=True,
        audit_reason="strong third-person-only prelude",
        supporting_views=["tp01"],
        supporting_roles=[ViewRole.THIRD_PERSON],
        candidates=[],
    )
    core = prelude.model_copy(
        update={
            "event_id": "DUAL-CORE",
            "global_start_ms": 110_000,
            "global_end_ms": 112_000,
            "key_global_ms": 111_000,
            "supporting_views": ["fp01", "tp01"],
            "supporting_roles": [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            "audit_reason": "cross-view balance operation",
            "candidates": [
                ActionCandidate(
                    candidate_id=f"CORE-{view_id}",
                    action_type=ActionType.DEVICE_PANEL_OPERATION,
                    view_id=view_id,
                    role=role,
                    local_start_ms=110_000,
                    local_end_ms=112_000,
                    global_start_ms=110_000,
                    global_end_ms=112_000,
                    key_global_ms=111_000,
                    objects=["balance", "hand"],
                    confidence=0.88,
                )
                for view_id, role in (
                    ("fp01", ViewRole.FIRST_PERSON),
                    ("tp01", ViewRole.THIRD_PERSON),
                )
            ],
        }
    )
    coarse = ActionCandidate(
        candidate_id="COARSE",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=90_000,
        local_end_ms=130_000,
        global_start_ms=90_000,
        global_end_ms=130_000,
        key_global_ms=110_000,
        objects=["balance"],
        confidence=0.9,
    )

    segments = build_experiment_segments(
        [prelude, core], views, default_config, coarse_windows=[coarse]
    )
    normalized = normalize_experiment_segments(
        segments, [prelude, core], views, default_config
    )

    assert len(normalized) == 1
    assert normalized[0].global_start_ms == 108_000
    assert normalized[0].event_ids == ["DUAL-CORE"]


def test_formal_promotion_quarantines_leading_third_person_only_segment(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    third_only = EvidenceEvent(
        event_id="LEADING-TP",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=233_800,
        global_end_ms=246_900,
        key_global_ms=240_000,
        objects=["hand", "spatula"],
        confidence=0.83,
        accepted=True,
        audit_reason="strong third-person-only context",
        supporting_views=["tp01"],
        supporting_roles=[ViewRole.THIRD_PERSON],
        candidates=[],
    )
    dual = third_only.model_copy(
        update={
            "event_id": "DUAL-EXPERIMENT",
            "global_start_ms": 301_100,
            "global_end_ms": 420_900,
            "key_global_ms": 350_000,
            "objects": ["pipette", "tube"],
            "supporting_views": ["fp01", "tp01"],
            "supporting_roles": [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        }
    )
    segments = [
        ExperimentSegment(
            segment_id="EXP-LEADING",
            global_start_ms=233_800,
            global_end_ms=249_900,
            event_ids=["LEADING-TP"],
            participating_views=["tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-DUAL",
            global_start_ms=299_100,
            global_end_ms=423_900,
            event_ids=["DUAL-EXPERIMENT"],
            participating_views=["fp01", "tp01"],
        ),
    ]
    coarse = ActionCandidate(
        candidate_id="COARSE",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=290_000,
        local_end_ms=440_000,
        global_start_ms=290_000,
        global_end_ms=440_000,
        key_global_ms=365_000,
        objects=["tube"],
        confidence=0.9,
    )

    formal, receipts = prepare_formal_experiment_segments(
        segments, [third_only, dual], views, [coarse], default_config
    )
    groups = build_experiment_groups(formal, [third_only, dual], views, default_config)

    assert [segment.segment_id for segment in formal] == ["EXP-DUAL"]
    assert receipts[0]["decision"] == "quarantined_missing_dual_view"
    assert receipts[1]["decision"] == "promoted_dual_view"
    assert len(groups) == 1
    assert groups[0].group_id == "GROUP-0001"


def test_formal_promotion_attaches_connected_first_person_cleanup_tail(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    core = EvidenceEvent(
        event_id="CORE",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=100_000,
        global_end_ms=110_000,
        key_global_ms=105_000,
        objects=["pipette", "tube"],
        confidence=0.9,
        accepted=True,
        audit_reason="cross-view core",
        supporting_views=["fp01", "tp01"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    tail = EvidenceEvent(
        event_id="CLEANUP-TAIL",
        action_type=ActionType.OBJECT_MOVEMENT,
        global_start_ms=114_000,
        global_end_ms=118_000,
        key_global_ms=116_000,
        objects=["cleaning_tool"],
        confidence=0.82,
        accepted=True,
        audit_reason="first-person cleanup tail",
        supporting_views=["fp01"],
        supporting_roles=[ViewRole.FIRST_PERSON],
        candidates=[],
    )
    segments = [
        ExperimentSegment(
            segment_id="EXP-CORE",
            global_start_ms=98_000,
            global_end_ms=113_000,
            event_ids=["CORE"],
            participating_views=["fp01", "tp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-TAIL",
            global_start_ms=114_000,
            global_end_ms=121_000,
            event_ids=["CLEANUP-TAIL"],
            participating_views=["fp01"],
        ),
    ]
    coarse = ActionCandidate(
        candidate_id="COARSE",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=90_000,
        local_end_ms=150_000,
        global_start_ms=90_000,
        global_end_ms=150_000,
        key_global_ms=120_000,
        objects=["tube"],
        confidence=0.9,
    )

    formal, receipts = prepare_formal_experiment_segments(
        segments, [core, tail], views, [coarse], default_config
    )

    assert len(formal) == 1
    assert formal[0].segment_id == "EXP-CORE"
    assert formal[0].global_end_ms == 121_000
    assert formal[0].event_ids == ["CORE", "CLEANUP-TAIL"]
    assert receipts[-1]["decision"] == "attached_first_person_tail"
    assert receipts[-1]["cleanup_objects"] == ["cleaning_tool"]


def test_formal_promotion_uses_fp_only_context_as_graph_edge_without_event_leakage(
    default_config,
):
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp_left", role=ViewRole.THIRD_PERSON, video=Path("left.mp4")),
        ViewInput(view_id="tp_right", role=ViewRole.THIRD_PERSON, video=Path("right.mp4")),
    ]

    def event(
        event_id: str,
        start_ms: float,
        end_ms: float,
        objects: list[str],
        supporting_views: list[str],
        supporting_roles: list[ViewRole],
    ) -> EvidenceEvent:
        return EvidenceEvent(
            event_id=event_id,
            action_type=ActionType.HAND_OBJECT_CONTACT,
            global_start_ms=start_ms,
            global_end_ms=end_ms,
            key_global_ms=(start_ms + end_ms) / 2.0,
            objects=objects,
            confidence=0.9,
            accepted=True,
            audit_reason="test evidence",
            supporting_views=supporting_views,
            supporting_roles=supporting_roles,
            candidates=[],
        )

    left_event = event(
        "LEFT-DUAL",
        309_000,
        329_000,
        ["paper", "gloved_hand"],
        ["fp01", "tp_left"],
        [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
    )
    context_event = event(
        "CONTEXT-FP-ONLY",
        356_600,
        358_100,
        ["bottle_cap", "gloved_hand"],
        ["fp01"],
        [ViewRole.FIRST_PERSON],
    )
    right_event = event(
        "RIGHT-DUAL",
        410_700,
        411_500,
        ["sample_bottle", "gloved_hand"],
        ["fp01", "tp_right"],
        [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
    )
    segments = [
        ExperimentSegment(
            segment_id="EXP-LEFT",
            global_start_ms=307_200,
            global_end_ms=332_400,
            event_ids=[left_event.event_id],
            participating_views=["fp01", "tp_left"],
        ),
        ExperimentSegment(
            segment_id="EXP-CONTEXT",
            global_start_ms=354_600,
            global_end_ms=361_100,
            event_ids=[context_event.event_id],
            participating_views=["fp01"],
        ),
        ExperimentSegment(
            segment_id="EXP-RIGHT",
            global_start_ms=408_700,
            global_end_ms=414_500,
            event_ids=[right_event.event_id],
            participating_views=["fp01", "tp_right"],
        ),
    ]
    coarse = ActionCandidate(
        candidate_id="MOTION-FUSED-000001",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=290_000,
        local_end_ms=440_000,
        global_start_ms=290_000,
        global_end_ms=440_000,
        key_global_ms=365_000,
        objects=[],
        confidence=0.9,
    )

    formal, receipts = prepare_formal_experiment_segments(
        segments,
        [left_event, context_event, right_event],
        views,
        [coarse],
        default_config,
    )
    groups = build_experiment_groups(
        formal,
        [left_event, context_event, right_event],
        views,
        default_config,
    )

    assert len(formal) == 1
    assert formal[0].segment_id == "EXP-LEFT"
    # The quarantined event contributes no boundary. The shared independent
    # motion envelope 290-440 s is contracted by the existing 10 s context
    # guard, producing a bounded 300-430 s formal clip.
    assert formal[0].global_start_ms == 300_000
    assert formal[0].global_end_ms == 430_000
    assert formal[0].event_ids == ["LEFT-DUAL", "RIGHT-DUAL"]
    assert "CONTEXT-FP-ONLY" not in formal[0].event_ids
    assert [item["decision"] for item in receipts] == [
        "promoted_dual_view",
        "quarantined_continuity_bridge",
        "merged_dual_view_fragment",
    ]
    assert receipts[1]["semantic_object_bridge_proven"] is False
    assert receipts[1]["shared_boundary_ids"] == ["MOTION-FUSED-000001"]
    assert receipts[1]["coarse_context_start_ms"] == 300_000
    assert receipts[1]["coarse_context_end_ms"] == 430_000
    assert len(groups) == 1
    assert groups[0].continuity_type == "independent"
    assert groups[0].atomic_experiment_ids == ["EXP-LEFT"]


def test_formal_promotion_does_not_bridge_across_different_coarse_boundaries(
    default_config,
):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=Path("tp.mp4")),
    ]
    dual = EvidenceEvent(
        event_id="LEFT",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=10_000,
        global_end_ms=12_000,
        key_global_ms=11_000,
        objects=["paper"],
        confidence=0.9,
        accepted=True,
        audit_reason="left",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    context = dual.model_copy(
        update={
            "event_id": "CONTEXT",
            "global_start_ms": 20_000,
            "global_end_ms": 22_000,
            "key_global_ms": 21_000,
            "supporting_views": ["fp"],
            "supporting_roles": [ViewRole.FIRST_PERSON],
        }
    )
    right = dual.model_copy(
        update={
            "event_id": "RIGHT",
            "global_start_ms": 30_000,
            "global_end_ms": 32_000,
            "key_global_ms": 31_000,
        }
    )
    segments = [
        ExperimentSegment(
            segment_id="LEFT-SEG",
            global_start_ms=9_000,
            global_end_ms=13_000,
            event_ids=["LEFT"],
            participating_views=["fp", "tp"],
        ),
        ExperimentSegment(
            segment_id="CONTEXT-SEG",
            global_start_ms=19_000,
            global_end_ms=23_000,
            event_ids=["CONTEXT"],
            participating_views=["fp"],
        ),
        ExperimentSegment(
            segment_id="RIGHT-SEG",
            global_start_ms=29_000,
            global_end_ms=33_000,
            event_ids=["RIGHT"],
            participating_views=["fp", "tp"],
        ),
    ]
    coarse = [
        ActionCandidate(
            candidate_id="LEFT-WINDOW",
            action_type=ActionType.OBJECT_MOVEMENT,
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            local_start_ms=0,
            local_end_ms=24_000,
            global_start_ms=0,
            global_end_ms=24_000,
            key_global_ms=12_000,
            objects=[],
            confidence=0.9,
        ),
        ActionCandidate(
            candidate_id="RIGHT-WINDOW",
            action_type=ActionType.OBJECT_MOVEMENT,
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            local_start_ms=25_000,
            local_end_ms=40_000,
            global_start_ms=25_000,
            global_end_ms=40_000,
            key_global_ms=32_000,
            objects=[],
            confidence=0.9,
        ),
    ]

    formal, receipts = prepare_formal_experiment_segments(
        segments, [dual, context, right], views, coarse, default_config
    )

    assert [segment.segment_id for segment in formal] == ["LEFT-SEG", "RIGHT-SEG"]
    assert "quarantined_continuity_bridge" not in {
        item["decision"] for item in receipts
    }
