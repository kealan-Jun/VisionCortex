from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from labvision_evidence.yolo_training import (
    build_mapped_public_yolo_union,
    build_public_yolo_training_view,
    build_yolo_training_dataset,
    evaluate_yolo_model_on_human_truth,
    train_yolo_model,
)


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


def _public_yolo_fixture(tmp_path):
    source = tmp_path / "public-source"
    (source / "Classes.names").parent.mkdir(parents=True)
    (source / "Classes.names").write_text("hand\npipette\n", encoding="utf-8")
    for split in ("Train", "Valid", "Test"):
        images = source / split / "Images"
        labels = source / split / "Labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"{split.lower()}.jpg").write_bytes(
            f"{split}-image".encode()
        )
        (labels / f"{split.lower()}.txt").write_text(
            "1 0.500000 0.500000 0.250000 0.500000\n", encoding="utf-8"
        )
    receipt = tmp_path / "dataset-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-public-dataset-receipt/1",
                "status": "completed",
                "dataset_id": "WasedaChemicalApparatus",
                "title": "Annotated Chemical Apparatus Image Dataset",
                "license": "CC-BY-4.0",
                "source_record": "https://data.mendeley.com/datasets/8p2hvgdvpn/1",
                "nas_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return source, receipt


def test_public_yolo_training_view_validates_labels_and_links_without_copying(tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    output = tmp_path / "training-view"

    report = build_public_yolo_training_view(source, receipt, output)

    assert report["truth_status"] == "public_human_annotations"
    assert report["split_image_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["split_annotation_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["class_annotation_counts"] == {"hand": 0, "pipette": 3}
    assert report["source_copy_bytes"] == 0
    assert report["nas_accessed"] is False
    assert not (output / "images" / "train").is_symlink()
    assert (output / "images" / "train" / "train.jpg").is_symlink()
    assert not (output / "labels" / "val" / "valid.txt").is_symlink()
    assert report["source_annotation_formats"] == {
        "box": 3,
        "polygon_converted_to_bounding_box": 0,
    }


def test_public_yolo_training_view_converts_polygon_to_bounding_box(tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    (source / "Train" / "Labels" / "train.txt").write_text(
        "1 0.2 0.3 0.8 0.3 0.8 0.9 0.2 0.9\n", encoding="utf-8"
    )
    output = tmp_path / "training-view"

    report = build_public_yolo_training_view(source, receipt, output)

    assert report["source_annotation_formats"] == {
        "box": 2,
        "polygon_converted_to_bounding_box": 1,
    }
    assert (output / "labels" / "train" / "train.txt").read_text().strip() == (
        "1 0.50000000 0.60000000 0.60000000 0.60000000"
    )


def test_public_yolo_training_view_rejects_invalid_box(tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    (source / "Test" / "Labels" / "test.txt").write_text(
        "1 0.5 0.5 1.5 0.2\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="box is invalid"):
        build_public_yolo_training_view(source, receipt, tmp_path / "training-view")


def test_public_yolo_training_view_accepts_standard_data_yaml(tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    (source / "Classes.names").unlink()
    (source / "data.yaml").write_text(
        "names: [hand, pipette]\nnc: 2\n", encoding="utf-8"
    )

    report = build_public_yolo_training_view(
        source, receipt, tmp_path / "training-view"
    )

    assert report["classes"] == ["hand", "pipette"]


def test_mapped_public_union_uses_target_ontology_and_train_oversampling(tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    view = tmp_path / "public-view"
    build_public_yolo_training_view(source, receipt, view)
    target_classes = ["hand", "pipette", *[f"class-{index}" for index in range(19)]]
    target_registry = tmp_path / "target-registry.json"
    target_registry.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-closed-set-model-registry/1",
                "ontology": {"class_count": 21, "classes": target_classes},
            }
        ),
        encoding="utf-8",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-public-yolo-ontology-map/1",
                "train_oversample_factors": {"pipette": 3},
                "datasets": {
                    "fixture": {
                        "class_mapping": {"hand": "hand", "pipette": "pipette"}
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_mapped_public_yolo_union(
        {"fixture": view}, mapping, target_registry, tmp_path / "union"
    )

    assert report["class_count"] == 21
    assert report["split_image_counts"] == {"train": 3, "val": 1, "test": 1}
    assert report["underlying_image_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["class_annotation_counts"]["pipette"] == 5
    assert report["source_copy_bytes"] == 0
    assert all((tmp_path / "union" / "images" / "train").iterdir())
    assert all(path.is_symlink() for path in (tmp_path / "union" / "images" / "train").iterdir())


def test_mapped_public_union_removes_cross_split_content_leakage(tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    (source / "Valid" / "Images" / "valid.jpg").write_bytes(
        (source / "Train" / "Images" / "train.jpg").read_bytes()
    )
    (source / "Train" / "Images" / "train-unique.jpg").write_bytes(b"unique-train")
    (source / "Train" / "Labels" / "train-unique.txt").write_text(
        "1 0.5 0.5 0.25 0.5\n", encoding="utf-8"
    )
    view = tmp_path / "public-view"
    build_public_yolo_training_view(source, receipt, view)
    target_classes = ["hand", "pipette", *[f"class-{index}" for index in range(19)]]
    target_registry = tmp_path / "target-registry.json"
    target_registry.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-closed-set-model-registry/1",
                "ontology": {"class_count": 21, "classes": target_classes},
            }
        ),
        encoding="utf-8",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-public-yolo-ontology-map/1",
                "train_oversample_factors": {"pipette": 3},
                "datasets": {
                    "fixture": {
                        "class_mapping": {"hand": "hand", "pipette": "pipette"}
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_mapped_public_yolo_union(
        {"fixture": view}, mapping, target_registry, tmp_path / "union"
    )

    assert report["raw_mapped_image_counts"] == {"train": 2, "val": 1, "test": 1}
    assert report["underlying_image_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["excluded_cross_split_duplicate_image_count"] == 1
    assert report["cross_split_content_policy"] == "keep_test_then_val_then_train"


def test_human_truth_evaluation_records_numpy_metrics(monkeypatch, tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    dataset = tmp_path / "training-view"
    build_public_yolo_training_view(source, receipt, dataset)
    model = tmp_path / "candidate.pt"
    model.write_bytes(b"candidate")
    output = tmp_path / "evaluation"

    class FakeYOLO:
        def __init__(self, model_path):
            assert model_path == str(model)

        def val(self, **kwargs):
            assert kwargs["split"] == "test"
            (tmp_path / kwargs["name"]).mkdir()
            return SimpleNamespace(
                results_dict={"metrics/precision(B)": np.float32(0.875)},
                names={0: "hand", 1: "pipette", 2: "paper"},
                box=SimpleNamespace(
                    ap_class_index=np.asarray([0, 1]),
                    p=np.asarray([0.8, 0.9]),
                    r=np.asarray([0.7, 0.85]),
                    f1=np.asarray([0.746667, 0.874286]),
                    ap50=np.asarray([0.7, 0.9]),
                    # Ultralytics fills an absent class slot with the overall
                    # mAP. The receipt must still expose it as null.
                    maps=np.asarray([0.5, 0.75, 0.625]),
                ),
            )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    report = evaluate_yolo_model_on_human_truth(dataset, model, output)

    assert report["metrics"] == {"metrics/precision(B)": pytest.approx(0.875)}
    assert report["per_class"] == [
        {
            "class_id": 0,
            "class_name": "hand",
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.746667,
            "map50": 0.7,
            "map50_95": 0.5,
        },
        {
            "class_id": 1,
            "class_name": "pipette",
            "precision": 0.9,
            "recall": 0.85,
            "f1": 0.874286,
            "map50": 0.9,
            "map50_95": 0.75,
        },
        {
            "class_id": 2,
            "class_name": "paper",
            "precision": None,
            "recall": None,
            "f1": None,
            "map50": None,
            "map50_95": None,
        },
    ]
    assert report["production_certified"] is False
    assert report["nas_accessed"] is False


def test_training_records_metrics_and_resource_telemetry(monkeypatch, tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    dataset = tmp_path / "training-view"
    build_public_yolo_training_view(source, receipt, dataset)
    dataset_receipt_path = dataset / "dataset-receipt.json"
    dataset_receipt = json.loads(dataset_receipt_path.read_text(encoding="utf-8"))
    dataset_receipt["schema_version"] = "visioncortex-mapped-public-yolo-union/1"
    dataset_receipt_path.write_text(json.dumps(dataset_receipt), encoding="utf-8")
    integrity_audit = tmp_path / "dataset-integrity-audit.json"
    integrity_audit.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-yolo-dataset-integrity-audit/1",
                "passed": True,
                "dataset_receipt_sha256": hashlib.sha256(
                    dataset_receipt_path.read_bytes()
                ).hexdigest(),
                "cross_split_content_hash_count": 0,
                "nas_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base")
    output = tmp_path / "candidate"

    class FakeMonitor:
        def __init__(self, destination, interval_seconds):
            assert interval_seconds == 0.25
            self.destination = destination

        def start(self):
            return None

        def set_stage(self, stage):
            assert stage == "public_yolo_training"

        def stop(self):
            payload = {"stage_summaries": {"public_yolo_training": {}}}
            self.destination.write_text(json.dumps(payload), encoding="utf-8")
            self.destination.with_name(
                f"{self.destination.stem}_live.json"
            ).write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            return payload

    class FakeYOLO:
        def __init__(self, model_path):
            assert model_path == str(base_model)
            self.trainer = None

        def add_callback(self, event, callback):
            assert event == "on_train_epoch_end"
            self.callback = callback

        def train(self, **kwargs):
            assert "time" not in kwargs
            assert kwargs["optimizer"] == "AdamW"
            assert kwargs["lr0"] == 0.001
            assert kwargs["lrf"] == 0.05
            assert kwargs["cos_lr"] is True
            assert kwargs["warmup_epochs"] == 1.0
            assert kwargs["close_mosaic"] == 1
            run = Path(kwargs["project"]) / kwargs["name"]
            (run / "weights").mkdir(parents=True)
            (run / "weights" / "best.pt").write_bytes(b"best")
            (run / "results.csv").write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50-95(B)\n"
                "1,0.8,0.7,0.6\n",
                encoding="utf-8",
            )
            self.trainer = SimpleNamespace(
                optimizer=SimpleNamespace(
                    param_groups=[{"initial_lr": 0.0004, "lr": 0.0001}]
                )
            )
            return SimpleNamespace(save_dir=run)

    monkeypatch.setattr(
        "labvision_evidence.yolo_training.ResourceMonitor", FakeMonitor
    )
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    report = train_yolo_model(
        dataset,
        base_model,
        output,
        epochs=1,
        max_hours=0.1,
        optimizer="AdamW",
        learning_rate=0.001,
        final_learning_rate_fraction=0.05,
        cosine_schedule=True,
        warmup_epochs=1.0,
        close_mosaic=1,
        dataset_integrity_audit=integrity_audit,
    )

    assert report["completed_epoch_count"] == 1
    assert report["final_validation_metrics"] == {
        "metrics/precision(B)": 0.8,
        "metrics/recall(B)": 0.7,
        "metrics/mAP50-95(B)": 0.6,
    }
    assert report["best_validation_completed_epoch_number"] == 1
    assert report["best_validation_csv_epoch"] == 1
    assert report["best_validation_metrics"]["metrics/mAP50-95(B)"] == 0.6
    assert report["optimizer"] == "AdamW"
    assert report["requested_optimizer"] == "AdamW"
    assert report["effective_optimizer"] == "SimpleNamespace"
    assert report["effective_initial_learning_rates"] == [0.0004]
    assert report["optimizer_auto_may_override_requested_learning_rate"] is False
    assert report["learning_rate"] == 0.001
    assert report["final_learning_rate_fraction"] == 0.05
    assert report["cosine_schedule"] is True
    assert report["warmup_epochs"] == 1.0
    assert report["close_mosaic"] == 1
    assert report["dataset_integrity_gate_passed"] is True
    assert report["dataset_integrity_audit"] == str(integrity_audit)
    assert (output / "resource-telemetry.json").is_file()
    assert (output / "resource-telemetry_live.json").is_file()
    assert report["production_certified"] is False
    assert report["source_copy_bytes"] == 0
    assert report["nas_accessed"] is False


def test_training_rejects_public_union_without_integrity_audit(tmp_path):
    source, receipt = _public_yolo_fixture(tmp_path)
    dataset = tmp_path / "training-view"
    build_public_yolo_training_view(source, receipt, dataset)
    dataset_receipt_path = dataset / "dataset-receipt.json"
    dataset_receipt = json.loads(dataset_receipt_path.read_text(encoding="utf-8"))
    dataset_receipt["schema_version"] = "visioncortex-mapped-public-yolo-union/1"
    dataset_receipt_path.write_text(json.dumps(dataset_receipt), encoding="utf-8")
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base")

    with pytest.raises(RuntimeError, match="integrity audit"):
        train_yolo_model(
            dataset,
            base_model,
            tmp_path / "candidate",
            epochs=1,
            close_mosaic=1,
        )
