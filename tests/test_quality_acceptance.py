import json
from pathlib import Path

from visioncortex.schemas import ActionType, EvidenceEvent, ExperimentGroup, ViewRole
from visioncortex.validation import validate_experiment_and_material_quality


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


def test_reviewed_baseline_matches_five_groups_and_required_material_types():
    baseline = json.loads(
        Path("configs/acceptance/six-view-three-hour-reviewed-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    groups = [
        group("GROUP-0001", 299250, 429875, "independent", 1),
        group("GROUP-0002", 1684250, 1789750, "independent", 1),
        group("GROUP-0003", 4159250, 4202250, "independent", 1),
        group("GROUP-0004", 7057000, 7172375, "continuous", 2),
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


def test_incomplete_material_attempt_cannot_pass_with_stale_complete_aliases():
    item = event(1, ActionType.OBJECT_MOVEMENT)
    item.observability["key_material_materialization"] = {"status": "partial"}
    report = validate_experiment_and_material_quality([], [item], None)
    assert report["key_materials"]["passed"] is False
    assert report["key_materials"]["media_complete_count"] == 0


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


def test_natural_experiment_does_not_require_all_action_categories():
    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [event(1, ActionType.HAND_OBJECT_CONTACT)],
        None,
    )

    assert report["status"] == "structural_only"
    assert report["key_materials"]["passed"] is True
    assert report["key_materials"]["required_action_types"] == []
    assert report["key_materials"]["missing_action_types"] == []
    assert "liquid_movement" in report["key_materials"]["unobserved_action_types"]


def test_segmentation_integrity_rejects_duplicate_uid_and_uncovered_view_pair():
    first = group("GROUP-1", 1_000, 2_000, "independent", 1)
    second = group("GROUP-2", 3_000, 4_000, "independent", 1)
    first.group_uid = "GRP-duplicate"
    second.group_uid = "GRP-duplicate"
    candidate = event(1, ActionType.HAND_OBJECT_CONTACT)
    first.key_event_ids = [candidate.event_id]
    first.third_person_view = "tp02"

    report = validate_experiment_and_material_quality(
        [first, second], [candidate], None
    )

    integrity = report["segmentation_integrity"]
    assert integrity["passed"] is False
    assert integrity["stable_group_uid_gate_passed"] is False
    assert integrity["canonical_pair_coverage_passed"] is False
    assert report["passed"] is False
    assert report["key_materials"]["category_coverage_is_acceptance_gate"] is False


def test_semantically_unconfirmed_material_fails_quality_gate():
    candidate = event(1, ActionType.HAND_OBJECT_CONTACT)
    candidate.semantic_review = {
        "verdict": "rejected",
        "model_status": "completed",
    }

    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
    )

    assert report["passed"] is False
    assert report["status"] == "structural_only"
    assert report["key_materials"]["passed"] is False
    assert report["key_materials"]["semantic_unconfirmed_count"] == 1


def test_required_participant_only_annotation_cannot_be_absent():
    candidate = event(1, ActionType.HAND_OBJECT_CONTACT)

    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
        require_participant_only_annotations=True,
    )

    assert report["passed"] is False
    assert report["key_materials"]["passed"] is False
    assert report["key_materials"]["participant_only_annotation_required"] is True
    assert report["key_materials"]["participant_only_annotation_count"] == 0
    assert report["key_materials"]["participant_only_annotation_gate_passed"] is False


def test_container_state_annotation_requires_actor_closure_and_container_same_view():
    candidate = event(1, ActionType.CONTAINER_STATE_CHANGE)
    candidate.objects = ["gloved_hand", "bottle_cap", "reagent_bottle"]
    candidate.observability["key_material_annotation"] = {
        "mode": "event_participants_only",
        "views": {
            "fp01": {
                "rendered_classes": ["gloved_hand"],
                "extraneous_rendered_classes": [],
            },
            "tp01": {
                "rendered_classes": ["gloved_hand"],
                "extraneous_rendered_classes": [],
            },
        },
    }

    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
    )

    assert report["passed"] is False
    assert (
        report["key_materials"]["interaction_pair_annotation_gate_passed"]
        is False
    )
    assert (
        report["key_materials"]["events"][0][
            "actor_and_manipulated_object_visible"
        ]
        is False
    )

    candidate.observability["key_material_annotation"]["views"]["fp01"][
        "rendered_classes"
    ] = ["gloved_hand", "container"]
    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
    )

    assert report["passed"] is False
    assert (
        report["key_materials"]["interaction_pair_annotation_gate_passed"]
        is True
    )
    assert report["key_materials"]["events"][0][
        "action_participant_visibility_rule"
    ] == "actor_closure_container_same_view"
    assert report["key_materials"]["events"][0][
        "state_transition_complete_view_ids"
    ] == []

    candidate.observability["key_material_annotation"]["views"]["fp01"][
        "rendered_classes"
    ] = ["gloved_hand", "bottle_cap", "container"]
    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
    )

    assert report["passed"] is True
    assert report["key_materials"]["events"][0][
        "state_transition_complete_view_ids"
    ] == ["fp01"]

    candidate.observability["key_material_annotation"]["views"]["fp01"][
        "rendered_classes"
    ] = ["gloved_hand"]
    candidate.observability["key_material_annotation"]["views"]["tp01"][
        "rendered_classes"
    ] = ["container"]
    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
    )

    assert report["passed"] is False
    assert report["key_materials"]["events"][0][
        "interaction_pair_view_ids"
    ] == []


def test_device_panel_annotation_requires_device_identity_in_same_view():
    candidate = event(1, ActionType.DEVICE_PANEL_OPERATION)
    candidate.objects = ["gloved_hand", "tube", "tube_rack"]
    candidate.observability["key_material_annotation"] = {
        "mode": "event_participants_only",
        "views": {
            "fp01": {
                "rendered_classes": ["gloved_hand", "tube"],
                "extraneous_rendered_classes": [],
            },
            "tp01": {
                "rendered_classes": ["gloved_hand", "tube_rack"],
                "extraneous_rendered_classes": [],
            },
        },
    }

    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
    )

    check = report["key_materials"]["events"][0]
    assert report["passed"] is False
    assert check["action_participant_visibility_rule"] == "actor_device_same_view"
    assert check["action_participant_missing_slots"] == ["event_device_object"]

    candidate.objects = ["gloved_hand", "balance"]
    candidate.observability["key_material_annotation"]["views"]["fp01"][
        "rendered_classes"
    ] = ["gloved_hand", "balance"]
    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
    )

    check = report["key_materials"]["events"][0]
    assert report["passed"] is True
    assert check["action_participant_complete_view_ids"] == ["fp01"]


def test_object_movement_allows_actor_occlusion_when_object_is_visible():
    candidate = event(1, ActionType.OBJECT_MOVEMENT)
    candidate.objects = ["pipette"]
    candidate.observability["key_material_annotation"] = {
        "mode": "event_participants_only",
        "views": {
            "fp01": {
                "rendered_classes": ["pipette"],
                "extraneous_rendered_classes": [],
            },
            "tp01": {
                "rendered_classes": [],
                "extraneous_rendered_classes": [],
            },
        },
    }

    report = validate_experiment_and_material_quality(
        [group("GROUP-1", 1000, 2000, "independent", 1)],
        [candidate],
        None,
    )

    assert report["passed"] is True
    assert (
        report["key_materials"]["interaction_pair_annotation_gate_passed"]
        is False
    )
    assert (
        report["key_materials"]["action_participant_visibility_gate_passed"]
        is True
    )
    assert (
        report["key_materials"]["events"][0][
            "action_participant_visibility_rule"
        ]
        == "manipulated_object"
    )
