import json
from pathlib import Path

from labvision_evidence.schemas import ActionType, EvidenceEvent, ExperimentGroup, ViewRole
from labvision_evidence.validation import validate_experiment_and_material_quality


def group(group_id: str, start_ms: float, end_ms: float, continuity: str, atomic_count: int):
    return ExperimentGroup(
        group_id=group_id,
        continuity_type=continuity,
        atomic_experiment_ids=[f"EXP-{index + 1}" for index in range(atomic_count)],
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        participating_views=["fp01", "tp01"],
        first_person_view="fp01",
        third_person_view="tp01",
        continuity_reason="test",
    )


def event(index: int, action_type: ActionType):
    return EvidenceEvent(
        event_id=f"EVT-{index:03d}",
        action_type=action_type,
        global_start_ms=index * 1000.0,
        global_end_ms=index * 1000.0 + 500.0,
        key_global_ms=index * 1000.0 + 250.0,
        objects=["object"],
        confidence=0.9,
        accepted=True,
        audit_reason="cross-view evidence",
        supporting_views=["fp01", "tp01"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        key_frames={"fp01": "fp.jpg", "tp01": "tp.jpg", "aligned_first_third": "aligned.jpg"},
        key_clips={"fp01": "fp.mp4", "tp01": "tp.mp4", "aligned_first_third": "aligned.mp4"},
        model_understanding={"status": "completed"},
    )


def test_reviewed_baseline_matches_five_groups_and_five_material_types():
    baseline = json.loads(
        Path("configs/acceptance/six-view-three-hour-reviewed-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    groups = [
        group("GROUP-0001", 299250, 429875, "independent", 1),
        group("GROUP-0002", 1684250, 1789750, "independent", 1),
        group("GROUP-0003", 4159250, 4202250, "independent", 1),
        group("GROUP-0004", 7069375, 7172375, "continuous", 2),
        group("GROUP-0005", 11743750, 11841000, "independent", 1),
    ]
    events = [event(index, action) for index, action in enumerate(ActionType, 1)]

    report = validate_experiment_and_material_quality(groups, events, baseline)

    assert report["passed"] is True
    assert report["status"] == "passed"
    assert report["experiment_boundaries"]["precision"] == 1.0
    assert report["experiment_boundaries"]["recall"] == 1.0
    assert report["experiment_boundaries"]["continuity_accuracy"] == 1.0
    assert report["key_materials"]["missing_action_types"] == []


def test_quality_acceptance_reports_boundary_and_continuity_regression():
    baseline = {
        "authority": "test",
        "experiments": [
            {
                "baseline_id": "expected-1",
                "start_seconds": 10,
                "end_seconds": 20,
                "continuity_type": "continuous",
                "atomic_experiment_count": 2,
            }
        ],
    }
    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 15000, 25000, "independent", 1)],
        [event(index, action) for index, action in enumerate(ActionType, 1)],
        baseline,
        boundary_match_iou=0.30,
        max_start_error_seconds=2,
        max_end_error_seconds=2,
    )

    match = report["experiment_boundaries"]["matches"][0]
    assert report["passed"] is False
    assert match["start_error_seconds"] == 5.0
    assert match["end_error_seconds"] == 5.0
    assert match["continuity_correct"] is False
    assert match["atomic_count_correct"] is False


def test_structural_only_when_no_reviewed_baseline_is_available():
    events = [event(index, action) for index, action in enumerate(ActionType, 1)]
    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)], events, None
    )

    assert report["status"] == "structural_only"
    assert report["experiment_boundaries"]["evaluated"] is False
    assert report["key_materials"]["passed"] is True
