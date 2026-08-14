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
            candidates=[],
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
            candidates=[],
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
