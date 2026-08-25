from pathlib import Path

import pytest

from labvision_evidence.grouping import (
    prepare_formal_experiment_segments,
    select_formal_experiment_start_events,
)
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ExperimentSegment,
    ViewInput,
    ViewRole,
)


def _views() -> list[ViewInput]:
    return [
        ViewInput(
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            video=Path("fp.mp4"),
        ),
        ViewInput(
            view_id="tp",
            role=ViewRole.THIRD_PERSON,
            video=Path("tp.mp4"),
        ),
    ]


def _event(
    event_id: str,
    start_ms: float,
    end_ms: float,
    confidence: float,
    *,
    accepted: bool = False,
) -> EvidenceEvent:
    roles = (
        [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON]
        if accepted
        else [ViewRole.FIRST_PERSON]
    )
    views = ["fp", "tp"] if accepted else ["fp"]
    return EvidenceEvent(
        event_id=event_id,
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=["gloved_hand", "sample_bottle"],
        confidence=confidence,
        accepted=accepted,
        audit_reason="dual core" if accepted else "single-role context",
        supporting_views=views,
        supporting_roles=roles,
        candidates=[],
    )


def _window(start_ms: float, end_ms: float) -> ActionCandidate:
    return ActionCandidate(
        candidate_id="MOTION-OBJECT-VERIFIED",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=["sample_bottle"],
        confidence=1.0,
    )


def test_dual_role_contact_overlapping_movement_corroborates_opener(
    default_config,
):
    movement = _event("MOVE", 309_200.0, 314_500.0, 0.73, accepted=True)
    movement.action_type = ActionType.OBJECT_MOVEMENT
    movement.objects = ["weighing_paper"]
    contact = _event("CONTACT", 309_800.0, 331_100.0, 0.79, accepted=True)
    contact.objects = ["gloved_hand", "weighing_paper"]

    selected = select_formal_experiment_start_events(
        [movement, contact], default_config
    )

    assert [event.event_id for event in selected] == ["MOVE", "CONTACT"]


def test_rejected_direct_cv_chain_extends_boundaries_without_membership(
    default_config,
):
    core = _event("CORE", 170_900.0, 250_500.0, 0.90, accepted=True)
    context = [
        _event("LEAD-WEAK", 155_300.0, 156_100.0, 0.357),
        _event("LEAD-STRONG", 165_500.0, 172_300.0, 0.672),
        _event("TAIL-01", 248_000.0, 255_000.0, 0.54),
        _event("TAIL-02", 260_000.0, 268_000.0, 0.55),
        _event("TAIL-03", 276_000.0, 284_000.0, 0.55),
        _event("TAIL-04", 292_000.0, 300_000.0, 0.55),
        _event("TAIL-05", 308_000.0, 316_000.0, 0.55),
        _event("TAIL-STRONG", 324_000.0, 332_500.0, 0.79),
    ]
    segment = ExperimentSegment(
        segment_id="EXP-0001",
        global_start_ms=168_900.0,
        global_end_ms=253_500.0,
        event_ids=[core.event_id],
        participating_views=["fp", "tp"],
    )

    formal, receipts = prepare_formal_experiment_segments(
        [segment],
        [*context, core],
        _views(),
        [_window(10_000.0, 320_000.0)],
        default_config,
    )

    assert formal[0].global_start_ms == pytest.approx(153_300.0)
    assert formal[0].global_end_ms == pytest.approx(335_500.0)
    assert formal[0].event_ids == ["CORE"]
    receipt = next(
        item
        for item in receipts
        if item.get("decision_type") == "formal_boundary_context_extension"
    )
    assert receipt["facts"]["formal_membership_changed"] is False
    assert receipt["facts"]["start_source"] == "rejected_direct_cv_chain"
    assert receipt["facts"]["end_source"] == "rejected_direct_cv_chain"


def test_object_verified_motion_start_snap_needs_context_chain(
    default_config,
):
    core = _event("CORE", 5_499_100.0, 5_536_800.0, 0.90, accepted=True)
    context = [
        _event("NEAR-WINDOW-START", 5_439_400.0, 5_440_000.0, 0.448),
        _event("LEAD-01", 5_465_800.0, 5_471_000.0, 0.595),
        _event("LEAD-02", 5_480_400.0, 5_488_400.0, 0.55),
        _event("LEAD-03", 5_490_600.0, 5_495_100.0, 0.55),
        _event("TAIL-STRONG", 5_538_600.0, 5_541_000.0, 0.661),
        _event("TAIL-01", 5_543_000.0, 5_547_000.0, 0.55),
        _event("TAIL-02", 5_548_000.0, 5_552_000.0, 0.55),
        _event("TAIL-03", 5_553_000.0, 5_559_000.0, 0.55),
    ]
    segment = ExperimentSegment(
        segment_id="EXP-0001",
        global_start_ms=5_497_100.0,
        global_end_ms=5_539_800.0,
        event_ids=[core.event_id],
        participating_views=["fp", "tp"],
    )
    window = _window(5_393_166.667, 5_553_166.667)

    formal, receipts = prepare_formal_experiment_segments(
        [segment], [*context, core], _views(), [window], default_config
    )

    assert formal[0].global_start_ms == pytest.approx(5_393_166.667)
    assert formal[0].global_end_ms == pytest.approx(5_544_000.0)
    assert formal[0].event_ids == ["CORE"]
    receipt = next(
        item
        for item in receipts
        if item.get("decision_type") == "formal_boundary_context_extension"
    )
    assert receipt["facts"]["start_source"] == (
        "object_verified_recalled_window_start"
    )
    assert receipt["facts"]["end_source"] == "rejected_direct_cv_chain"
