import time
from pathlib import Path

from visioncortex.archive import ArchiveLayout
from visioncortex.grouping import (
    build_experiment_groups,
    normalize_experiment_segments,
    prepare_formal_experiment_segments,
    select_formal_experiment_start_events,
)
from visioncortex.pipeline import EvidencePipeline
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    VideoInfo,
    ViewInput,
    ViewRole,
)


def _views() -> list[ViewInput]:
    return [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=Path("tp.mp4")),
    ]


def _tracked_candidate(
    candidate_id: str,
    action: ActionType,
    start_ms: float,
    end_ms: float,
    object_name: str,
    track_id: int,
    *,
    view_id: str = "fp",
    role: ViewRole = ViewRole.FIRST_PERSON,
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
        objects=[object_name],
        confidence=0.9,
        evidence=[{"track_id": track_id}],
    )


def _event(
    event_id: str,
    action: ActionType,
    start_ms: float,
    end_ms: float,
    objects: list[str],
    *,
    views: list[str] | None = None,
    roles: list[ViewRole] | None = None,
    tracks: dict[str, int] | None = None,
    confidence: float = 0.9,
) -> EvidenceEvent:
    supporting_views = views or ["fp", "tp"]
    supporting_roles = roles or [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON]
    candidates = [
        _tracked_candidate(
            f"{event_id}-{object_name}",
            action,
            start_ms,
            end_ms,
            object_name,
            track_id,
        )
        for object_name, track_id in (tracks or {}).items()
    ]
    return EvidenceEvent(
        event_id=event_id,
        action_type=action,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=objects,
        confidence=confidence,
        accepted=True,
        audit_reason="fixture evidence",
        supporting_views=supporting_views,
        supporting_roles=supporting_roles,
        candidates=candidates,
    )


def test_g1_single_view_context_extends_boundary_without_membership_leak(
    default_config,
):
    context = _event(
        "EVT-G1-CONTEXT",
        ActionType.HAND_OBJECT_CONTACT,
        302_900.0,
        305_100.0,
        ["gloved_hand", "paper"],
        views=["tp"],
        roles=[ViewRole.THIRD_PERSON],
        confidence=0.956781,
    )
    anchor = _event(
        "EVT-G1-DUAL",
        ActionType.HAND_OBJECT_CONTACT,
        309_800.0,
        331_100.0,
        ["gloved_hand", "paper"],
    )
    segments = [
        ExperimentSegment(
            segment_id="EXP-G1-CONTEXT",
            global_start_ms=300_900.0,
            global_end_ms=308_100.0,
            event_ids=[context.event_id],
            participating_views=["tp"],
        ),
        ExperimentSegment(
            segment_id="EXP-G1-DUAL",
            global_start_ms=307_800.0,
            global_end_ms=334_100.0,
            event_ids=[anchor.event_id],
            participating_views=["fp", "tp"],
        ),
    ]
    coarse = ActionCandidate(
        candidate_id="MOTION-G1",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=290_000.0,
        local_end_ms=440_000.0,
        global_start_ms=290_000.0,
        global_end_ms=440_000.0,
        key_global_ms=365_000.0,
        objects=[],
        confidence=0.9,
    )

    formal, receipts = prepare_formal_experiment_segments(
        segments, [context, anchor], _views(), [coarse], default_config
    )

    assert len(formal) == 1
    assert formal[0].global_start_ms == 300_900.0
    assert formal[0].event_ids == [anchor.event_id]
    assert context.event_id not in formal[0].event_ids
    receipt = next(
        item
        for item in receipts
        if item.get("event_id") == context.event_id
        and item.get("rule_id") == "QF1-SINGLE-VIEW-BOUNDARY-CONTEXT"
    )
    assert receipt["verdict"] == "accepted"
    assert receipt["facts"]["formal_membership_changed"] is False


def test_verified_recall_boundary_restores_only_accepted_opener_clipped_events(
    default_config,
):
    early_single = _event(
        "EARLY-SINGLE",
        ActionType.DEVICE_PANEL_OPERATION,
        60_000.0,
        65_000.0,
        ["balance", "gloved_hand"],
        views=["tp"],
        roles=[ViewRole.THIRD_PERSON],
        confidence=0.9,
    )
    early_dual = _event(
        "EARLY-DUAL",
        ActionType.OBJECT_MOVEMENT,
        75_000.0,
        85_000.0,
        ["paper"],
    )
    core = _event(
        "CORE",
        ActionType.HAND_OBJECT_CONTACT,
        100_000.0,
        110_000.0,
        ["gloved_hand", "paper"],
    )
    rejected = _event(
        "REJECTED",
        ActionType.HAND_OBJECT_CONTACT,
        88_000.0,
        92_000.0,
        ["gloved_hand", "paper"],
    )
    rejected.accepted = False
    segment = ExperimentSegment(
        segment_id="EXP-CORE",
        global_start_ms=50_000.0,
        global_end_ms=113_000.0,
        event_ids=[core.event_id],
        participating_views=["fp", "tp"],
    )
    coarse = ActionCandidate(
        candidate_id="RECALL",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=50_000.0,
        local_end_ms=120_000.0,
        global_start_ms=50_000.0,
        global_end_ms=120_000.0,
        key_global_ms=85_000.0,
        objects=["paper", "balance"],
        confidence=0.9,
    )

    formal, receipts = prepare_formal_experiment_segments(
        [segment],
        [early_single, early_dual, rejected, core],
        _views(),
        [coarse],
        default_config,
    )

    assert formal[0].event_ids == ["EARLY-SINGLE", "EARLY-DUAL", "CORE"]
    assert "REJECTED" not in formal[0].event_ids
    receipt = next(
        item
        for item in receipts
        if item.get("decision_type") == "recalled_boundary_membership_recovery"
    )
    assert receipt["facts"]["formal_membership_changed"] is True
    assert receipt["facts"]["rejected_events_promoted"] is False


def test_g3_class_overlap_without_stable_track_identity_is_not_continuous(
    default_config,
):
    left = _event(
        "EVT-G3-LEFT",
        ActionType.HAND_OBJECT_CONTACT,
        10_000.0,
        20_000.0,
        ["gloved_hand", "sample_bottle", "tube", "tube_rack"],
        tracks={"sample_bottle": 1314, "tube_rack": 1338},
    )
    right = _event(
        "EVT-G3-RIGHT",
        ActionType.HAND_OBJECT_CONTACT,
        23_900.0,
        30_000.0,
        ["gloved_hand", "sample_bottle", "tube", "tube_rack"],
        tracks={"sample_bottle": 1481, "tube_rack": 1387},
    )
    segments = [
        ExperimentSegment(
            segment_id="EXP-G3-LEFT",
            global_start_ms=8_000.0,
            global_end_ms=20_000.0,
            event_ids=[left.event_id],
            participating_views=["fp", "tp"],
        ),
        ExperimentSegment(
            segment_id="EXP-G3-RIGHT",
            global_start_ms=23_900.0,
            global_end_ms=33_000.0,
            event_ids=[right.event_id],
            participating_views=["fp", "tp"],
        ),
    ]
    receipts: list[dict] = []

    groups = build_experiment_groups(
        segments,
        [left, right],
        _views(),
        default_config,
        decision_receipts=receipts,
    )

    assert len(groups) == 2
    assert all(group.continuity_type == "independent" for group in groups)
    assert receipts[0]["verdict"] == "rejected"
    assert receipts[0]["reason_codes"] == [
        "class_overlap_without_stable_identity"
    ]
    assert receipts[0]["facts"]["shared_object_labels"] == [
        "sample_bottle",
        "tube",
        "tube_rack",
    ]


def test_g4_rejected_fp_context_can_bridge_two_dual_atomic_fragments(
    default_config,
):
    left = _event(
        "EVT-G4-LEFT",
        ActionType.HAND_OBJECT_CONTACT,
        10_000.0,
        20_000.0,
        ["gloved_hand", "sample_bottle_blue", "tube", "tube_rack"],
    )
    right = _event(
        "EVT-G4-RIGHT",
        ActionType.LIQUID_MOVEMENT,
        90_000.0,
        100_000.0,
        ["sample_bottle", "tube", "beaker"],
    )
    bottle_context = _event(
        "EVT-G4-CONTEXT-BOTTLE",
        ActionType.HAND_OBJECT_CONTACT,
        30_000.0,
        35_000.0,
        ["gloved_hand", "sample_bottle_blue"],
        views=["fp"],
        roles=[ViewRole.FIRST_PERSON],
    )
    tube_context = _event(
        "EVT-G4-CONTEXT-TUBE",
        ActionType.CONTAINER_STATE_CHANGE,
        60_000.0,
        70_000.0,
        ["gloved_hand", "tube"],
        views=["fp"],
        roles=[ViewRole.FIRST_PERSON],
    )
    bottle_context.accepted = False
    tube_context.accepted = False
    segments = [
        ExperimentSegment(
            segment_id="EXP-G4-LEFT",
            global_start_ms=8_000.0,
            global_end_ms=20_000.0,
            event_ids=[left.event_id],
            participating_views=["fp", "tp"],
        ),
        ExperimentSegment(
            segment_id="EXP-G4-RIGHT",
            global_start_ms=90_000.0,
            global_end_ms=103_000.0,
            event_ids=[right.event_id],
            participating_views=["fp", "tp"],
        ),
    ]
    coarse = ActionCandidate(
        candidate_id="MOTION-G4",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=0.0,
        local_end_ms=120_000.0,
        global_start_ms=0.0,
        global_end_ms=120_000.0,
        key_global_ms=60_000.0,
        objects=[],
        confidence=0.9,
    )
    receipts: list[dict] = []

    groups = build_experiment_groups(
        segments,
        [left, bottle_context, tube_context, right],
        _views(),
        default_config,
        decision_receipts=receipts,
        coarse_windows=[coarse],
    )

    assert len(groups) == 1
    assert groups[0].continuity_type == "continuous"
    assert groups[0].atomic_experiment_ids == ["EXP-G4-LEFT", "EXP-G4-RIGHT"]
    assert bottle_context.event_id not in groups[0].key_event_ids
    receipt = next(
        item
        for item in receipts
        if item.get("rule_id") == "QF2-QUARANTINED-CONTEXT-CHAIN"
    )
    assert receipt["verdict"] == "accepted"
    assert receipt["facts"]["formal_membership_changed"] is False
    assert receipt["facts"]["left_context_object_families"] == [
        "sample_bottle",
        "tube",
    ]


def test_dense_rejected_context_can_bridge_adjacent_coarse_windows(
    default_config,
):
    left = _event(
        "LEFT",
        ActionType.HAND_OBJECT_CONTACT,
        10_000.0,
        20_000.0,
        ["gloved_hand", "sample_bottle", "tube"],
    )
    right = _event(
        "RIGHT",
        ActionType.LIQUID_MOVEMENT,
        90_000.0,
        100_000.0,
        ["sample_bottle", "tube", "beaker"],
    )
    context_events = [
        _event(
            f"CONTEXT-{index}",
            ActionType.HAND_OBJECT_CONTACT,
            start,
            start + 5_000.0,
            ["gloved_hand", "sample_bottle", "tube"],
            views=["fp"],
            roles=[ViewRole.FIRST_PERSON],
        )
        for index, start in enumerate(
            [30_000.0, 45_000.0, 60_000.0, 75_000.0], 1
        )
    ]
    for event in context_events:
        event.accepted = False
    segments = [
        ExperimentSegment(
            segment_id="EXP-LEFT",
            global_start_ms=8_000.0,
            global_end_ms=20_000.0,
            event_ids=[left.event_id],
            participating_views=["fp", "tp"],
        ),
        ExperimentSegment(
            segment_id="EXP-RIGHT",
            global_start_ms=90_000.0,
            global_end_ms=103_000.0,
            event_ids=[right.event_id],
            participating_views=["fp", "tp"],
        ),
    ]
    coarse_windows = [
        ActionCandidate(
            candidate_id="COARSE-LEFT",
            action_type=ActionType.OBJECT_MOVEMENT,
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            local_start_ms=0.0,
            local_end_ms=50_000.0,
            global_start_ms=0.0,
            global_end_ms=50_000.0,
            key_global_ms=25_000.0,
            objects=[],
            confidence=0.9,
        ),
        ActionCandidate(
            candidate_id="COARSE-RIGHT",
            action_type=ActionType.OBJECT_MOVEMENT,
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            local_start_ms=50_001.0,
            local_end_ms=120_000.0,
            global_start_ms=50_001.0,
            global_end_ms=120_000.0,
            key_global_ms=85_000.0,
            objects=[],
            confidence=0.9,
        ),
    ]
    receipts: list[dict] = []

    groups = build_experiment_groups(
        segments,
        [left, *context_events, right],
        _views(),
        default_config,
        decision_receipts=receipts,
        coarse_windows=coarse_windows,
    )

    assert len(groups) == 1
    assert groups[0].continuity_type == "continuous"
    receipt = next(
        item
        for item in receipts
        if item.get("rule_id") == "QF2-QUARANTINED-CONTEXT-CHAIN"
    )
    assert receipt["verdict"] == "accepted"
    assert receipt["facts"]["shared_boundary_ids"] == []
    assert receipt["facts"]["strong_cross_boundary_context"] is True


def test_g1_single_view_tail_extends_boundary_without_membership_leak(
    default_config,
):
    dual = _event(
        "EVT-G1-DUAL",
        ActionType.DEVICE_PANEL_OPERATION,
        390_000.0,
        397_000.0,
        ["balance", "gloved_hand", "paper"],
    )
    tail = _event(
        "EVT-G1-TAIL",
        ActionType.DEVICE_PANEL_OPERATION,
        426_900.0,
        427_400.0,
        ["balance", "gloved_hand", "paper"],
        views=["tp"],
        roles=[ViewRole.THIRD_PERSON],
    )
    segments = [
        ExperimentSegment(
            segment_id="EXP-G1-DUAL",
            global_start_ms=307_800.0,
            global_end_ms=400_100.0,
            event_ids=[dual.event_id],
            participating_views=["fp", "tp"],
        ),
        ExperimentSegment(
            segment_id="EXP-G1-TAIL",
            global_start_ms=424_900.0,
            global_end_ms=430_400.0,
            event_ids=[tail.event_id],
            participating_views=["tp"],
        ),
    ]
    coarse = ActionCandidate(
        candidate_id="MOTION-G1",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=290_000.0,
        local_end_ms=450_000.0,
        global_start_ms=290_000.0,
        global_end_ms=450_000.0,
        key_global_ms=365_000.0,
        objects=[],
        confidence=0.9,
    )

    formal, receipts = prepare_formal_experiment_segments(
        segments, [dual, tail], _views(), [coarse], default_config
    )

    assert len(formal) == 1
    assert formal[0].global_end_ms == 430_400.0
    assert formal[0].event_ids == [dual.event_id]
    assert tail.event_id not in formal[0].event_ids
    receipt = next(
        item
        for item in receipts
        if item.get("decision_type") == "trailing_boundary_context"
    )
    assert receipt["verdict"] == "accepted"
    assert receipt["facts"]["formal_membership_changed"] is False
    assert receipt["facts"]["shared_non_hand_objects"] == ["balance", "paper"]


def test_rejected_fp_context_with_only_one_object_family_cannot_bridge(
    default_config,
):
    left = _event(
        "LEFT",
        ActionType.HAND_OBJECT_CONTACT,
        10_000.0,
        20_000.0,
        ["gloved_hand", "tube"],
    )
    right = _event(
        "RIGHT",
        ActionType.HAND_OBJECT_CONTACT,
        90_000.0,
        100_000.0,
        ["gloved_hand", "tube"],
    )
    context = _event(
        "CONTEXT",
        ActionType.CONTAINER_STATE_CHANGE,
        40_000.0,
        60_000.0,
        ["gloved_hand", "tube"],
        views=["fp"],
        roles=[ViewRole.FIRST_PERSON],
    )
    context.accepted = False
    segments = [
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
    coarse = ActionCandidate(
        candidate_id="MOTION",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=0.0,
        local_end_ms=120_000.0,
        global_start_ms=0.0,
        global_end_ms=120_000.0,
        key_global_ms=60_000.0,
        objects=[],
        confidence=0.9,
    )

    groups = build_experiment_groups(
        segments,
        [left, context, right],
        _views(),
        default_config,
        coarse_windows=[coarse],
    )

    assert len(groups) == 2
    assert all(group.continuity_type == "independent" for group in groups)


def test_g4_late_fp_cluster_outside_early_group_uses_refined_target_fallback(
    default_config,
):
    pipeline = EvidencePipeline(default_config)
    views = [
        *_views(),
        ViewInput(
            view_id="tp2", role=ViewRole.THIRD_PERSON, video=Path("tp2.mp4")
        ),
    ]
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=8_000_000.0,
            fps=30.0,
            width=16,
            height=16,
            frame_count=240_000,
        )
        for view in views
    }
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            state="aligned",
            confidence=1.0,
        )
        for view in views
    }
    early = _event(
        "EVT-G4-EARLY",
        ActionType.HAND_OBJECT_CONTACT,
        7_071_000.0,
        7_074_000.0,
        ["gloved_hand", "tube"],
    )
    segment = ExperimentSegment(
        segment_id="EXP-G4-EARLY",
        global_start_ms=7_069_600.0,
        global_end_ms=7_093_200.0,
        event_ids=[early.event_id],
        participating_views=["fp", "tp"],
    )
    group = ExperimentGroup(
        group_id="G4",
        continuity_type="independent",
        atomic_experiment_ids=[segment.segment_id],
        global_start_ms=segment.global_start_ms,
        global_end_ms=segment.global_end_ms,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="early dual anchor only",
    )
    late = [
        _tracked_candidate(
            f"G4-LATE-{index}",
            ActionType.HAND_OBJECT_CONTACT,
            start,
            start + 2_000.0,
            "tube",
            700 + index,
        )
        for index, start in enumerate(
            (7_100_000.0, 7_114_000.0, 7_128_000.0), 1
        )
    ]
    boundary = ActionCandidate(
        candidate_id="MOTION-G4",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=7_051_000.0,
        local_end_ms=7_131_000.0,
        global_start_ms=7_051_000.0,
        global_end_ms=7_131_000.0,
        key_global_ms=7_091_000.0,
        objects=[],
        confidence=0.9,
    )

    plan = pipeline._group_local_recall_plan(
        [group],
        [segment],
        [early],
        late,
        views,
        {view.view_id: {} for view in views},
        {
            "fp": [(7_051_000.0, 7_131_000.0)],
            "tp": [(7_051_000.0, 7_131_000.0)],
        },
        infos,
        transforms,
        boundary_candidates=[boundary],
    )

    assert plan["complete"] is False
    selected = plan["selected_plans"][0]
    assert selected["view_id"] == "tp2"
    assert selected["refined_target_end_ms"] == 7_131_000.0
    assert selected["windows"][-1][1] == 7_130_000.0
    assert plan["groups"][0]["global_end_ms"] == 7_093_200.0
    assert plan["groups"][0]["unresolved_temporal_cluster_count"] == 3
    assert plan["groups"][0]["selection_mode"] == "zero_prior_quality_fallback"


def test_g5_uncorroborated_balance_movement_cannot_open_experiment(default_config):
    movement = _event(
        "EVT-G5-BALANCE-MOVE",
        ActionType.OBJECT_MOVEMENT,
        11_731_000.0,
        11_734_000.0,
        ["balance"],
    )
    paper = _event(
        "EVT-G5-PAPER",
        ActionType.HAND_OBJECT_CONTACT,
        11_751_000.0,
        11_760_000.0,
        ["gloved_hand", "paper"],
    )
    receipts: list[dict] = []

    openers = select_formal_experiment_start_events(
        [movement, paper], default_config, decision_receipts=receipts
    )

    assert [event.event_id for event in openers] == [paper.event_id]
    movement_receipt = next(
        item for item in receipts if item.get("event_id") == movement.event_id
    )
    assert movement_receipt["verdict"] == "deferred"
    assert movement_receipt["facts"]["shared_non_hand_objects"] == []

    segment = ExperimentSegment(
        segment_id="EXP-G5",
        global_start_ms=11_729_000.0,
        global_end_ms=11_763_000.0,
        event_ids=[movement.event_id, paper.event_id],
        participating_views=["fp", "tp"],
    )
    normalized = normalize_experiment_segments(
        [segment], [movement, paper], _views(), default_config
    )
    assert normalized[0].global_start_ms == 11_749_000.0


def test_pipeline_metrics_remain_callable_after_quality_planning(default_config):
    pipeline = EvidencePipeline(default_config)
    pipeline._run_started_iso = "2026-08-17T00:00:00+00:00"
    pipeline._run_started_perf = time.perf_counter()

    metrics = pipeline._metrics([], [])

    assert metrics["tokens"]["run_total"]["total_tokens"] is None
    assert "preprocessing_sla" in metrics


def test_unresolved_cross_view_clusters_are_retained_without_failing_run(
    default_config, tmp_path
):
    pipeline = EvidencePipeline(default_config)
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    progressive_report = {
        "quality_complete": False,
        "unresolved_candidate_ids": ["G4-LATE-1"],
        "group_local_recall": {
            "completeness_gate": {"quality_complete": False}
        },
    }

    report = pipeline._run_boundary_precheck(
        layout, [], progressive_report=progressive_report
    )

    assert report["status"] == "partial"
    assert report["passed"] is False
    assert report["blocking_failure"] is False
    assert report["analysis_continuation_allowed"] is True
    assert report["evidence_classification"] == "PARTIAL_EVIDENCE"
