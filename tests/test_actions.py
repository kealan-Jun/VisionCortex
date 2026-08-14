from pathlib import Path

from labvision_evidence.actions import audit_candidates, build_experiment_segments
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ViewInput,
    ViewRole,
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
