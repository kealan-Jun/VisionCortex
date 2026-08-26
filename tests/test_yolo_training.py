from __future__ import annotations

import json

import pytest

from labvision_evidence.yolo_training import build_yolo_training_dataset


def _truth(tmp_path, *, truth_status=None):
    image_root = tmp_path / "source-images"
    image_root.mkdir()
    (image_root / "train.jpg").write_bytes(b"train-image")
    (image_root / "val.jpg").write_bytes(b"val-image")
    payload = {
        "schema_version": "visioncortex-yolo-box-ground-truth/1.0.0",
        "dataset_id": "dataset-1",
        "classes": ["paper"],
        "images": [
            {
                "image_id": "train-1",
                "file_name": "train.jpg",
                "width": 100,
                "height": 50,
                "split": "train",
            },
            {
                "image_id": "val-1",
                "file_name": "val.jpg",
                "width": 100,
                "height": 50,
                "split": "val",
            },
        ],
        "annotations": [
            {
                "annotation_id": "ann-1",
                "image_id": "train-1",
                "class_name": "paper",
                "xyxy": [10, 5, 50, 25],
                "review_status": "double_checked",
            }
        ],
    }
    if truth_status:
        payload["truth_status"] = truth_status
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, image_root


def test_build_training_dataset_links_images_without_copying(tmp_path):
    truth, image_root = _truth(tmp_path)
    output = tmp_path / "dataset"

    report = build_yolo_training_dataset(truth, image_root, output)

    assert report["source_copy_bytes"] == 0
    assert report["split_image_counts"] == {"train": 1, "val": 1, "test": 0}
    assert (output / "images" / "train" / "train-1.jpg").is_symlink()
    assert (output / "labels" / "train" / "train-1.txt").read_text().strip() == (
        "0 0.30000000 0.30000000 0.40000000 0.40000000"
    )
    receipt = json.loads((output / "dataset-receipt.json").read_text())
    assert receipt["truth_status"] == "reviewed_ground_truth"


def test_build_training_dataset_rejects_pseudo_truth(tmp_path):
    truth, image_root = _truth(tmp_path, truth_status="pseudo_labels_not_ground_truth")

    with pytest.raises(ValueError, match="Pseudo-labels"):
        build_yolo_training_dataset(truth, image_root, tmp_path / "dataset")


def test_build_training_dataset_refuses_existing_output(tmp_path):
    truth, image_root = _truth(tmp_path)
    output = tmp_path / "dataset"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        build_yolo_training_dataset(truth, image_root, output)
