from __future__ import annotations

import json

import pytest
import yaml

from labvision_evidence.yolo_calibration import (
    _select_threshold,
    audit_yolo_dataset_integrity,
    evaluate_yolo_with_calibrated_thresholds,
)
from labvision_evidence.yolo_evaluation import evaluate_yolo_predictions


def _dataset(tmp_path, *, duplicate_test_content: bool = False):
    root = tmp_path / "dataset"
    classes = ["hand", "pipette"]
    (root / "dataset.yaml").parent.mkdir(parents=True)
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "hand", 1: "pipette"},
            }
        ),
        encoding="utf-8",
    )
    (root / "dataset-receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-mapped-public-yolo-union/1",
                "status": "completed",
                "truth_status": "public_human_annotations",
                "classes": classes,
                "source_copy_bytes": 0,
                "nas_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    contents = {"train": b"train", "val": b"val", "test": b"test"}
    if duplicate_test_content:
        contents["test"] = contents["val"]
    for index, split in enumerate(("train", "val", "test")):
        image = root / "images" / split / f"{split}.jpg"
        label = root / "labels" / split / f"{split}.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        image.write_bytes(contents[split])
        label.write_text(f"{index % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return root


def test_dataset_integrity_audit_hashes_unique_sources_without_copying(tmp_path):
    root = _dataset(tmp_path)

    report = audit_yolo_dataset_integrity(root, tmp_path / "audit")

    assert report["passed"] is True
    assert report["unique_source_image_count"] == 3
    assert report["cross_split_content_hash_count"] == 0
    assert report["source_copy_bytes"] == 0
    assert report["nas_accessed"] is False


def test_dataset_integrity_audit_fails_closed_on_cross_split_content(tmp_path):
    root = _dataset(tmp_path, duplicate_test_content=True)

    report = audit_yolo_dataset_integrity(root, tmp_path / "audit")

    assert report["passed"] is False
    assert report["status"] == "failed_split_leakage"
    assert report["cross_split_content_hash_count"] == 1


def test_class_specific_threshold_changes_only_requested_operating_point():
    truth = {
        "schema_version": "visioncortex-yolo-box-ground-truth/1",
        "images": [
            {"image_id": "image-1", "event_id": "event", "role": "fp", "frame_index": 1}
        ],
        "annotations": [
            {
                "annotation_id": "truth-1",
                "image_id": "image-1",
                "class_name": "hand",
                "xyxy": [0.0, 0.0, 10.0, 10.0],
            }
        ],
    }
    predictions = [
        {
            "event_id": "event",
            "role": "fp",
            "frame_index": 1,
            "detections": [
                {"class_name": "hand", "confidence": 0.4, "xyxy": [0.0, 0.0, 10.0, 10.0]},
                {"class_name": "hand", "confidence": 0.3, "xyxy": [20.0, 20.0, 30.0, 30.0]},
            ],
        }
    ]

    report = evaluate_yolo_predictions(
        predictions,
        truth,
        confidence_threshold=0.25,
        class_confidence_thresholds={"hand": 0.35},
        iou_thresholds=(0.5,),
    )

    assert report["per_class"]["hand"]["precision"] == 1.0
    assert report["per_class"]["hand"]["recall"] == 1.0
    assert report["class_confidence_thresholds"] == {"hand": 0.35}


def test_threshold_selection_prefers_highest_recall_after_gate_passes():
    selected = _select_threshold(
        [
            {"threshold": 0.1, "precision": 0.8, "recall": 0.95, "f1": 0.87},
            {"threshold": 0.2, "precision": 0.86, "recall": 0.82, "f1": 0.84},
            {"threshold": 0.3, "precision": 0.9, "recall": 0.8, "f1": 0.847},
        ],
        0.85,
        0.8,
    )

    assert selected["threshold"] == 0.2
    assert selected["gate_passed"] is True
    assert (
        selected["selection_reason"]
        == "highest_recall_meeting_precision_and_recall_gate"
    )


def test_invalid_class_threshold_is_rejected():
    with pytest.raises(ValueError, match="class confidence"):
        evaluate_yolo_predictions(
            [],
            {"schema_version": "visioncortex-yolo-box-ground-truth/1", "images": [], "annotations": []},
            class_confidence_thresholds={"hand": 1.1},
        )


def test_invalid_test_exposure_status_is_rejected_before_inference(tmp_path):
    root = _dataset(tmp_path)
    model = tmp_path / "model.pt"
    calibration = tmp_path / "calibration.json"
    model.write_bytes(b"model")
    calibration.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="test_exposure_status"):
        evaluate_yolo_with_calibrated_thresholds(
            root,
            model,
            calibration,
            tmp_path / "evaluation",
            test_exposure_status="unknown",
        )
