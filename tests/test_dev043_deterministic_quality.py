import json
from itertools import permutations
from pathlib import Path

from labvision_evidence.actions import (
    audit_candidates,
    build_experiment_segments,
    refine_motion_candidates_with_coarse,
)
from labvision_evidence.grouping import build_experiment_groups
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentSegment,
    ViewInput,
    ViewRole,
)


def _views() -> list[ViewInput]:
    return [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=Path("tp.mp4")),
    ]


def _candidate(
    candidate_id: str,
    action: ActionType,
    view_id: str,
    role: ViewRole,
    start_ms: float,
    end_ms: float,
    objects: list[str],
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=action,
        view_id=view_id,
        role=role,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=objects,
        confidence=0.9,
    )


def _event(
    event_id: str,
    action: ActionType,
    start_ms: float,
    end_ms: float,
    objects: list[str],
    *,
    accepted: bool = True,
    views: list[str] | None = None,
    roles: list[ViewRole] | None = None,
) -> EvidenceEvent:
    supporting_views = views or ["fp", "tp"]
    supporting_roles = roles or [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON]
    candidates = [
        _candidate(
            f"{event_id}-{view_id}",
            action,
            view_id,
            role,
            start_ms,
            end_ms,
            objects,
        )
        for view_id, role in zip(supporting_views, supporting_roles, strict=True)
    ]
    return EvidenceEvent(
        event_id=event_id,
        action_type=action,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=objects,
        confidence=0.9,
        accepted=accepted,
        audit_reason="deterministic fixture",
        supporting_views=supporting_views,
        supporting_roles=supporting_roles,
        candidates=candidates,
    )


def _json(value) -> str:
    if isinstance(value, list) and value and hasattr(value[0], "model_dump"):
        value = [item.model_dump(mode="json") for item in value]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_candidate_audit_is_byte_stable_under_input_permutations(default_config):
    candidates = [
        _candidate(
            "FP-CONTACT",
            ActionType.HAND_OBJECT_CONTACT,
            "fp",
            ViewRole.FIRST_PERSON,
            10_000.0,
            12_000.0,
            ["gloved_hand", "pipette"],
        ),
        _candidate(
            "TP-CONTACT",
            ActionType.HAND_OBJECT_CONTACT,
            "tp",
            ViewRole.THIRD_PERSON,
            10_050.0,
            12_050.0,
            ["hand", "pipette"],
        ),
        _candidate(
            "FP-PANEL",
            ActionType.DEVICE_PANEL_OPERATION,
            "fp",
            ViewRole.FIRST_PERSON,
            30_000.0,
            31_500.0,
            ["gloved_hand", "balance"],
        ),
    ]
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            confidence=0.95,
            state="aligned",
        )
        for view in _views()
    }

    outputs = []
    for candidate_order in permutations(candidates):
        events, rejected = audit_candidates(
            list(candidate_order), transforms, default_config
        )
        outputs.append((_json(events), _json(rejected)))

    assert len(set(outputs)) == 1


def test_coarse_boundary_refinement_is_stable_under_input_permutations(
    default_config,
):
    motion = _candidate(
        "MOTION",
        ActionType.OBJECT_MOVEMENT,
        "fp",
        ViewRole.FIRST_PERSON,
        10_000.0,
        70_000.0,
        ["lab_bench"],
    )
    coarse = [
        _candidate(
            "COARSE-TP",
            ActionType.HAND_OBJECT_CONTACT,
            "tp",
            ViewRole.THIRD_PERSON,
            15_000.0,
            45_000.0,
            ["gloved_hand", "tube"],
        ),
        _candidate(
            "COARSE-FP",
            ActionType.HAND_OBJECT_CONTACT,
            "fp",
            ViewRole.FIRST_PERSON,
            40_000.0,
            75_000.0,
            ["gloved_hand", "tube"],
        ),
    ]

    outputs = []
    for coarse_order in (coarse, list(reversed(coarse))):
        refined, report = refine_motion_candidates_with_coarse(
            [motion], coarse_order, default_config
        )
        outputs.append((_json(refined), _json(report)))

    assert len(set(outputs)) == 1


def test_raw_segment_boundary_and_receipts_are_stable_under_permutations(
    default_config,
):
    core = _event(
        "DUAL-CORE",
        ActionType.HAND_OBJECT_CONTACT,
        10_000.0,
        30_000.0,
        ["gloved_hand", "pipette", "tube"],
    )
    ordinary_context = _event(
        "REJECTED-PIPETTE-CONTEXT",
        ActionType.OBJECT_MOVEMENT,
        31_000.0,
        40_000.0,
        ["gloved_hand", "pipette", "tube"],
        accepted=False,
        views=["fp"],
        roles=[ViewRole.FIRST_PERSON],
    )
    unrelated_context = _event(
        "REJECTED-BALANCE-CONTEXT",
        ActionType.OBJECT_MOVEMENT,
        41_000.0,
        45_000.0,
        ["gloved_hand", "balance"],
        accepted=False,
        views=["tp"],
        roles=[ViewRole.THIRD_PERSON],
    )
    windows = [
        _candidate(
            "WINDOW-LATE",
            ActionType.OBJECT_MOVEMENT,
            "fp",
            ViewRole.FIRST_PERSON,
            8_000.0,
            60_000.0,
            [],
        ),
        _candidate(
            "WINDOW-EARLY",
            ActionType.OBJECT_MOVEMENT,
            "tp",
            ViewRole.THIRD_PERSON,
            5_000.0,
            55_000.0,
            [],
        ),
    ]

    outputs = []
    for event_order in (
        [core, ordinary_context, unrelated_context],
        [unrelated_context, ordinary_context, core],
    ):
        for window_order in (windows, list(reversed(windows))):
            receipts: list[dict] = []
            segments = build_experiment_segments(
                event_order,
                _views(),
                default_config,
                coarse_windows=window_order,
                decision_receipts=receipts,
            )
            outputs.append((_json(segments), _json(receipts)))
            assert segments[0].global_end_ms == 33_000.0
            boundary = next(
                item
                for item in receipts
                if item.get("rule_id") == "QF1-RAW-ACCEPTED-EVENT-BOUNDARY"
            )
            assert boundary["facts"]["end_source_event_ids"] == ["DUAL-CORE"]
            assert boundary["facts"]["end_source_decision_ids"] == []
            assert boundary["facts"]["implicit_motion_window_extension"] is False

    assert len(set(outputs)) == 1


def test_cleanup_context_requires_an_explicit_qf1_boundary_receipt(default_config):
    core = _event(
        "DUAL-CORE",
        ActionType.HAND_OBJECT_CONTACT,
        10_000.0,
        30_000.0,
        ["gloved_hand", "beaker"],
    )
    cleanup = _event(
        "REJECTED-CLEANUP-CONTEXT",
        ActionType.OBJECT_MOVEMENT,
        31_000.0,
        40_000.0,
        ["gloved_hand", "cleaning_tool"],
        accepted=False,
        views=["fp"],
        roles=[ViewRole.FIRST_PERSON],
    )
    coarse = _candidate(
        "WINDOW",
        ActionType.OBJECT_MOVEMENT,
        "fp",
        ViewRole.FIRST_PERSON,
        5_000.0,
        60_000.0,
        [],
    )
    receipts: list[dict] = []

    segments = build_experiment_segments(
        [cleanup, core],
        _views(),
        default_config,
        coarse_windows=[coarse],
        decision_receipts=receipts,
    )

    assert segments[0].global_end_ms == 43_000.0
    assert cleanup.event_id not in segments[0].event_ids
    extension = next(
        item
        for item in receipts
        if item.get("decision_type") == "raw_boundary_context_extension"
    )
    boundary = next(
        item
        for item in receipts
        if item.get("rule_id") == "QF1-RAW-ACCEPTED-EVENT-BOUNDARY"
    )
    assert extension["verdict"] == "accepted"
    assert extension["facts"]["formal_membership_changed"] is False
    assert boundary["facts"]["end_source_event_ids"] == []
    assert boundary["facts"]["end_source_decision_ids"] == [
        extension["decision_id"]
    ]


def test_qf2_base_and_context_fallback_have_independent_stable_receipts(
    default_config,
):
    left = _event(
        "LEFT",
        ActionType.HAND_OBJECT_CONTACT,
        10_000.0,
        20_000.0,
        ["gloved_hand", "sample_bottle_blue", "tube", "tube_rack"],
    )
    right = _event(
        "RIGHT",
        ActionType.LIQUID_MOVEMENT,
        90_000.0,
        100_000.0,
        ["sample_bottle", "tube", "beaker"],
    )
    bottle_context = _event(
        "CONTEXT-BOTTLE",
        ActionType.HAND_OBJECT_CONTACT,
        30_000.0,
        35_000.0,
        ["gloved_hand", "sample_bottle_blue"],
        accepted=False,
        views=["fp"],
        roles=[ViewRole.FIRST_PERSON],
    )
    tube_context = _event(
        "CONTEXT-TUBE",
        ActionType.CONTAINER_STATE_CHANGE,
        60_000.0,
        70_000.0,
        ["gloved_hand", "tube"],
        accepted=False,
        views=["fp"],
        roles=[ViewRole.FIRST_PERSON],
    )
    base_segments = [
        ExperimentSegment(
            segment_id="LEFT-SEG",
            global_start_ms=8_000.0,
            global_end_ms=20_000.0,
            event_ids=[left.event_id],
            participating_views=["fp", "tp"],
        ),
        ExperimentSegment(
            segment_id="RIGHT-SEG",
            global_start_ms=90_000.0,
            global_end_ms=103_000.0,
            event_ids=[right.event_id],
            participating_views=["fp", "tp"],
        ),
    ]
    coarse = _candidate(
        "MOTION-G4",
        ActionType.OBJECT_MOVEMENT,
        "fp",
        ViewRole.FIRST_PERSON,
        0.0,
        120_000.0,
        [],
    )

    outputs = []
    for segments, events in (
        (
            base_segments,
            [left, bottle_context, tube_context, right],
        ),
        (
            list(reversed(base_segments)),
            [right, tube_context, left, bottle_context],
        ),
    ):
        receipts: list[dict] = []
        groups = build_experiment_groups(
            [segment.model_copy(deep=True) for segment in segments],
            events,
            _views(),
            default_config,
            decision_receipts=receipts,
            coarse_windows=[coarse],
        )
        outputs.append((_json(groups), _json(receipts)))
        assert len(groups) == 1
        assert groups[0].continuity_type == "continuous"
        assert groups[0].atomic_experiment_ids == ["LEFT-SEG", "RIGHT-SEG"]
        base = next(
            item for item in receipts if item.get("rule_id") == "QF2-STABLE-OBJECT-IDENTITY"
        )
        fallback = next(
            item
            for item in receipts
            if item.get("rule_id") == "QF2-QUARANTINED-CONTEXT-CHAIN"
        )
        assert base["verdict"] == "rejected"
        assert fallback["verdict"] == "accepted"
        assert fallback["facts"]["formal_membership_changed"] is False
        assert not set(fallback["facts"]["context_event_ids"]) & set(
            groups[0].key_event_ids
        )

    assert len(set(outputs)) == 1
