import json
from pathlib import Path

import pytest

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


def test_recall_evaluation_canonicalizes_lab_object_aliases():
    report = evaluate_key_event_recall(
        [
            prediction(
                "P-PAPER",
                ActionType.HAND_OBJECT_CONTACT,
                10_000,
                20_000,
                ["gloved_hand", "paper"],
            )
        ],
        {
            "ground_truth_id": "object-alias-test",
            "events": [
                {
                    "event_id": "GT-PAPER",
                    "action_type": "hand_object_contact",
                    "start_seconds": 10,
                    "end_seconds": 20,
                    "objects": ["weighing_paper"],
                }
            ],
        },
    )

    at_half = report["threshold_results"][1]
    assert at_half["true_positives"] == 1
    assert at_half["matches"][0]["ground_truth_objects"] == [
        "weighing_paper"
    ]
    assert at_half["matches"][0]["prediction_objects"] == [
        "weighing_paper"
    ]


def test_recall_evaluation_uses_maximum_cardinality_not_greedy_pairing():
    predictions = [
        prediction(
            "P-WIDE",
            ActionType.OBJECT_MOVEMENT,
            0,
            14_000,
            ["tube"],
        ),
        prediction(
            "P-LEFT",
            ActionType.OBJECT_MOVEMENT,
            0,
            6_000,
            ["tube"],
        ),
    ]
    report = evaluate_key_event_recall(
        predictions,
        {
            "ground_truth_id": "maximum-matching-test",
            "events": [
                {
                    "event_id": "GT-LEFT",
                    "action_type": "object_movement",
                    "start_seconds": 0,
                    "end_seconds": 10,
                    "objects": ["tube"],
                },
                {
                    "event_id": "GT-RIGHT",
                    "action_type": "object_movement",
                    "start_seconds": 8,
                    "end_seconds": 18,
                    "objects": ["tube"],
                },
            ],
        },
        thresholds=(0.30,),
    )

    result = report["threshold_results"][0]
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision_confidence_interval"]["lower"] < 1.0
    assert result["recall_confidence_interval"]["lower"] < 1.0
    assert report["small_sample_warning"] is True


def test_annotation_coverage_merges_overlapping_windows():
    report = evaluate_key_event_recall(
        [
            prediction(
                "P-1",
                ActionType.OBJECT_MOVEMENT,
                5_000,
                6_000,
                ["tube"],
            )
        ],
        {
            "ground_truth_id": "coverage-test",
            "source_duration_seconds": 20,
            "labeled_windows": [
                {"start_seconds": 0, "end_seconds": 10},
                {"start_seconds": 5, "end_seconds": 15},
            ],
            "events": [
                {
                    "event_id": "GT-1",
                    "action_type": "object_movement",
                    "start_seconds": 5,
                    "end_seconds": 6,
                    "objects": ["tube"],
                }
            ],
        },
    )

    coverage = report["annotation_coverage"]
    assert coverage["labeled_duration_ms"] == 15_000
    assert coverage["coverage_ratio"] == 0.75
    assert coverage["labeled_windows"] == [
        {"start_ms": 0.0, "end_ms": 15_000.0}
    ]


def test_recall_evaluation_rejects_duplicate_reviewed_event_ids():
    with pytest.raises(ValueError, match="duplicate event_id"):
        evaluate_key_event_recall(
            [],
            {
                "events": [
                    {
                        "event_id": "GT-DUPLICATE",
                        "action_type": "object_movement",
                        "start_seconds": 1,
                        "end_seconds": 2,
                    },
                    {
                        "event_id": "GT-DUPLICATE",
                        "action_type": "object_movement",
                        "start_seconds": 3,
                        "end_seconds": 4,
                    },
                ]
            },
        )


def test_documented_ground_truth_example_is_executable():
    ground_truth = json.loads(
        Path(
            "configs/evaluation/key-event-ground-truth.example.json"
        ).read_text(encoding="utf-8")
    )

    report = evaluate_key_event_recall([], ground_truth)

    assert report["evaluated"] is True
    assert report["ground_truth_validation"]["status"] == "valid"
    assert report["ground_truth_event_count"] == 1
    assert report["excluded_uncertain_ground_truth_event_ids"] == [
        "GT-000002"
    ]
