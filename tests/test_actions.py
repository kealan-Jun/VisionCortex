from pathlib import Path

from labvision_evidence.actions import audit_candidates, build_experiment_segments
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
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

