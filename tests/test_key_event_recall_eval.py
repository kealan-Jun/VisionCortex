from labvision_evidence.schemas import (
    ActionType,
    EvidenceEvent,
    ViewRole,
)
from labvision_evidence.validation import evaluate_key_event_recall


def prediction(
    event_id: str,
    action_type: ActionType,
    start_ms: float,
    end_ms: float,
    objects: list[str],
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=event_id,
        action_type=action_type,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=objects,
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )


def test_key_event_recall_is_one_to_one_object_aware_and_coverage_scoped():
    predictions = [
        prediction(
            "P-1",
            ActionType.LIQUID_MOVEMENT,
            10_000,
            20_000,
            ["pipette", "tube"],
        ),
        prediction(
            "P-WRONG-OBJECT",
            ActionType.HAND_OBJECT_CONTACT,
            30_000,
            40_000,
            ["gloved_hand", "balance"],
        ),
        prediction(
            "P-OUTSIDE",
            ActionType.OBJECT_MOVEMENT,
            100_000,
            110_000,
            ["tube"],
        ),
    ]
    ground_truth = {
        "ground_truth_id": "reviewed-test",
        "authority": "human_reviewed",
        "source_duration_seconds": 120,
        "labeled_windows": [{"start_seconds": 0, "end_seconds": 60}],
        "events": [
            {
                "event_id": "GT-1",
                "action_type": "liquid_transfer",
                "start_seconds": 10,
                "end_seconds": 20,
                "objects": ["tube"],
            },
            {
                "event_id": "GT-2",
                "action_type": "hand_object_contact",
                "start_seconds": 30,
                "end_seconds": 40,
                "objects": ["paper"],
            },
            {
                "event_id": "GT-UNCERTAIN",
                "action_type": "object_movement",
                "start_seconds": 45,
                "end_seconds": 46,
                "objects": ["tube"],
                "uncertain": True,
            },
        ],
    }

    report = evaluate_key_event_recall(predictions, ground_truth)
    at_half = next(
        item
        for item in report["threshold_results"]
        if item["temporal_iou_threshold"] == 0.5
    )

    assert report["evaluated"] is True
    assert report["annotation_coverage"]["coverage_ratio"] == 0.5
    assert report["excluded_prediction_ids_outside_labeled_windows"] == [
        "P-OUTSIDE"
    ]
    assert report["excluded_uncertain_ground_truth_event_ids"] == [
        "GT-UNCERTAIN"
    ]
    assert at_half["true_positives"] == 1
    assert at_half["false_positives"] == 1
    assert at_half["false_negatives"] == 1
    assert at_half["precision"] == 0.5
    assert at_half["recall"] == 0.5
    assert at_half["matches"][0]["action_type"] == "liquid_movement"


def test_key_event_recall_is_explicitly_not_evaluated_without_labels():
    report = evaluate_key_event_recall([], None)
    assert report["status"] == "not_evaluated"
    assert report["evaluated"] is False
