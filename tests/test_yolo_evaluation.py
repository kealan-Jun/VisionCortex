from __future__ import annotations

from labvision_evidence.yolo_evaluation import evaluate_yolo_predictions


def test_per_class_metrics_require_box_iou_and_class_match():
    ground_truth = {
        "images": [
            {"image_id": "i1", "event_id": "e1", "role": "first_person", "frame_index": 1},
            {"image_id": "i2", "event_id": "e2", "role": "third_person", "frame_index": 2},
        ],
        "annotations": [
            {"image_id": "i1", "class_name": "bottle_cap", "xyxy": [0, 0, 10, 10]},
            {"image_id": "i2", "class_name": "bottle_cap", "xyxy": [0, 0, 10, 10]},
        ],
    }
    predictions = [
        {
            "event_id": "e1",
            "role": "first_person",
            "frame_index": 1,
            "detections": [
                {"class_name": "bottle_cap", "confidence": 0.9, "xyxy": [0, 0, 10, 10]},
                {"class_name": "tube_cap", "confidence": 0.8, "xyxy": [0, 0, 10, 10]},
            ],
        },
        {"event_id": "e2", "role": "third_person", "frame_index": 2, "detections": []},
    ]

    report = evaluate_yolo_predictions(predictions, ground_truth)

    bottle = report["per_class"]["bottle_cap"]
    assert bottle["true_positive"] == 1
    assert bottle["false_negative"] == 1
    assert bottle["precision"] == 1.0
    assert bottle["recall"] == 0.5
    assert report["per_class"]["tube_cap"]["false_positive"] == 1
    assert report["status"] == "completed"


def test_empty_ground_truth_cannot_claim_formal_metrics():
    report = evaluate_yolo_predictions([], {"images": [], "annotations": []})
    assert report["status"] == "not_evaluated_no_ground_truth"


def test_prediction_below_confidence_threshold_remains_false_negative():
    ground_truth = {
        "images": [
            {"image_id": "i1", "event_id": "e1", "role": "first_person", "frame_index": 1}
        ],
        "annotations": [
            {"image_id": "i1", "class_name": "bottle_cap", "xyxy": [0, 0, 10, 10]}
        ],
    }
    predictions = [
        {
            "event_id": "e1",
            "role": "first_person",
            "frame_index": 1,
            "detections": [
                {"class_name": "bottle_cap", "confidence": 0.2, "xyxy": [0, 0, 10, 10]}
            ],
        }
    ]

    report = evaluate_yolo_predictions(
        predictions, ground_truth, confidence_threshold=0.25
    )

    bottle = report["per_class"]["bottle_cap"]
    assert bottle["true_positive"] == 0
    assert bottle["false_negative"] == 1
    assert any(item["kind"] == "false_negative" for item in report["errors"])
