from __future__ import annotations

import csv
import hashlib
import json
import math
import numbers
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .telemetry import ResourceMonitor


DATASET_SCHEMA = "visioncortex-yolo-training-dataset/1"
TRAINING_SCHEMA = "visioncortex-yolo-training-run/1"
EVALUATION_SCHEMA = "visioncortex-yolo-human-truth-evaluation/1"
PUBLIC_DATASET_SCHEMA = "visioncortex-public-yolo-training-dataset/1"
TRAINING_TRUTH_STATUSES = frozenset(
    {"reviewed_ground_truth", "public_human_annotations"}
)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return normalized.strip(".-") or "image"


def _validate_review_authority(payload: dict[str, Any]) -> None:
    schema = str(payload.get("schema_version") or "")
    if not schema.startswith("visioncortex-yolo-"):
        raise ValueError("Training input must declare a VisionCortex YOLO schema")
    truth_status = str(payload.get("truth_status") or "").lower()
    if "pseudo" in truth_status or "pseudo" in schema:
        raise ValueError("Pseudo-labels cannot be used as reviewed training truth")


def _find_single_directory(root: Path, name: str) -> Path:
    matches = sorted(
        item
        for item in root.rglob("*")
        if item.is_dir() and item.name.casefold() == name.casefold()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one public YOLO {name} directory under {root}, got {len(matches)}"
        )
    return matches[0]


def _find_child_directory(root: Path, name: str) -> Path:
    matches = [
        item
        for item in root.iterdir()
        if item.is_dir() and item.name.casefold() == name.casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {name} directory under public split {root}"
        )
    return matches[0]


def _public_yolo_classes(source_root: Path) -> tuple[Path, list[str]]:
    candidates = sorted(
        item
        for item in source_root.rglob("*")
        if item.is_file()
        and item.name.casefold() in {"classes.names", "classes.txt"}
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one Classes.names or classes.txt in public YOLO dataset"
        )
    classes = [
        line.strip()
        for line in candidates[0].read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not classes or len(classes) != len(set(classes)):
        raise RuntimeError("Public YOLO class list is empty or contains duplicates")
    return candidates[0], classes


def _validate_public_yolo_split(
    split_root: Path,
    classes: list[str],
) -> dict[str, Any]:
    images_root = _find_child_directory(split_root, "Images")
    labels_root = _find_child_directory(split_root, "Labels")
    images = sorted(
        item
        for item in images_root.rglob("*")
        if item.is_file() and item.suffix.casefold() in _IMAGE_SUFFIXES
    )
    labels = sorted(item for item in labels_root.rglob("*.txt") if item.is_file())
    images_by_stem = {
        item.relative_to(images_root).with_suffix(""): item for item in images
    }
    labels_by_stem = {
        item.relative_to(labels_root).with_suffix(""): item for item in labels
    }
    if not images or images_by_stem.keys() != labels_by_stem.keys():
        raise RuntimeError(
            f"Public YOLO image/label mismatch in {split_root}: "
            f"images={len(images_by_stem)} labels={len(labels_by_stem)}"
        )
    annotation_count = 0
    class_counts = {name: 0 for name in classes}
    for label in labels:
        for line_number, row in enumerate(
            label.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            values = row.split()
            if len(values) != 5:
                raise RuntimeError(
                    f"Invalid public YOLO row: {label}:{line_number}"
                )
            try:
                class_id = int(values[0])
                coordinates = [float(item) for item in values[1:]]
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid public YOLO value: {label}:{line_number}"
                ) from exc
            if class_id < 0 or class_id >= len(classes):
                raise RuntimeError(
                    f"Public YOLO class id is out of range: {label}:{line_number}"
                )
            if (
                not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates)
                or coordinates[2] <= 0.0
                or coordinates[3] <= 0.0
            ):
                raise RuntimeError(
                    f"Public YOLO box is invalid: {label}:{line_number}"
                )
            annotation_count += 1
            class_counts[classes[class_id]] += 1
    return {
        "images_root": images_root,
        "labels_root": labels_root,
        "image_count": len(images),
        "annotation_count": annotation_count,
        "class_annotation_counts": class_counts,
        "file_pairs": [
            {
                "image": images_by_stem[stem],
                "image_relative": images_by_stem[stem].relative_to(images_root),
                "label": labels_by_stem[stem],
                "label_relative": labels_by_stem[stem].relative_to(labels_root),
            }
            for stem in sorted(images_by_stem)
        ],
    }


def build_public_yolo_training_view(
    source_root: Path,
    dataset_receipt_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate public human YOLO labels and expose a zero-copy training view."""

    source_root = source_root.resolve()
    dataset_receipt_path = dataset_receipt_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Public YOLO training output already exists: {output}")
    if not source_root.is_dir() or not dataset_receipt_path.is_file():
        raise FileNotFoundError("Public YOLO source or acquisition receipt is missing")
    acquisition = json.loads(dataset_receipt_path.read_text(encoding="utf-8-sig"))
    if (
        acquisition.get("schema_version") != "visioncortex-public-dataset-receipt/1"
        or acquisition.get("status") != "completed"
        or acquisition.get("nas_accessed") is not False
    ):
        raise RuntimeError("Public dataset acquisition receipt is not trusted")
    if str(acquisition.get("license") or "") != "CC-BY-4.0":
        raise RuntimeError("Public YOLO training requires the pinned CC-BY-4.0 source")
    _, classes = _public_yolo_classes(source_root)
    source_splits = {
        "train": _find_single_directory(source_root, "Train"),
        "val": _find_single_directory(source_root, "Valid"),
        "test": _find_single_directory(source_root, "Test"),
    }
    split_receipts = {
        split: _validate_public_yolo_split(split_root, classes)
        for split, split_root in source_splits.items()
    }
    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    try:
        for split, split_receipt in split_receipts.items():
            image_destination_root = temporary / "images" / split
            label_destination_root = temporary / "labels" / split
            for pair in split_receipt["file_pairs"]:
                image_destination = image_destination_root / pair["image_relative"]
                label_destination = label_destination_root / pair["label_relative"]
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(pair["image"], image_destination)
                os.symlink(pair["label"], label_destination)
        data_yaml = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(classes)},
        }
        (temporary / "dataset.yaml").write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        receipt = {
            "schema_version": PUBLIC_DATASET_SCHEMA,
            "status": "completed",
            "truth_status": "public_human_annotations",
            "dataset_id": acquisition.get("dataset_id"),
            "title": acquisition.get("title"),
            "license": acquisition.get("license"),
            "source_record": acquisition.get("source_record"),
            "source_dataset_receipt": str(dataset_receipt_path),
            "source_dataset_receipt_sha256": _sha256(dataset_receipt_path),
            "class_count": len(classes),
            "classes": classes,
            "split_image_counts": {
                split: item["image_count"] for split, item in split_receipts.items()
            },
            "split_annotation_counts": {
                split: item["annotation_count"] for split, item in split_receipts.items()
            },
            "class_annotation_counts": {
                class_name: sum(
                    item["class_annotation_counts"][class_name]
                    for item in split_receipts.values()
                )
                for class_name in classes
            },
            "link_mode": "file_symlink",
            "source_copy_bytes": 0,
            "nas_accessed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "dataset-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
        return {
            **receipt,
            "output": str(output),
            "dataset_yaml": str(output / "dataset.yaml"),
        }
    except Exception:
        raise


def build_yolo_training_dataset(
    ground_truth_path: Path,
    image_root: Path,
    output: Path,
    *,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    """Materialize YOLO labels while linking, never copying, source images."""

    if link_mode not in {"symlink", "hardlink"}:
        raise ValueError("YOLO training dataset link_mode must be symlink or hardlink")
    ground_truth_path = ground_truth_path.resolve()
    image_root = image_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Training dataset output already exists: {output}")
    payload = json.loads(ground_truth_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("YOLO training ground truth must be a JSON object")
    _validate_review_authority(payload)
    classes = [str(item).strip() for item in payload.get("classes") or []]
    if not classes or any(not item for item in classes):
        raise ValueError("YOLO training ground truth must declare non-empty classes")
    if len(set(classes)) != len(classes):
        raise ValueError("YOLO training classes contain duplicates")
    class_ids = {name: index for index, name in enumerate(classes)}
    images = payload.get("images") or []
    image_by_id: dict[str, dict[str, Any]] = {}
    for item in images:
        image_id = str(item.get("image_id") or "").strip()
        if not image_id or image_id in image_by_id:
            raise ValueError("Training images require unique non-empty image_id values")
        split = str(item.get("split") or "").strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Training image {image_id} has invalid split: {split}")
        width, height = int(item.get("width") or 0), int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Training image {image_id} has invalid dimensions")
        image_by_id[image_id] = dict(item)

    annotations_by_image: dict[str, list[dict[str, Any]]] = {
        image_id: [] for image_id in image_by_id
    }
    for annotation in payload.get("annotations") or []:
        image_id = str(annotation.get("image_id") or "")
        if image_id not in image_by_id:
            raise ValueError(f"Training annotation references unknown image: {image_id}")
        class_name = str(annotation.get("class_name") or "").strip()
        if class_name not in class_ids:
            raise ValueError(f"Training annotation uses unknown class: {class_name}")
        review_status = str(annotation.get("review_status") or "").strip().lower()
        if review_status and review_status not in {
            "reviewed",
            "double_checked",
            "adjudicated",
            "confirmed",
        }:
            raise ValueError(
                f"Training annotation is not reviewed: {annotation.get('annotation_id')}"
            )
        box = annotation.get("xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Training annotation xyxy must contain four values")
        coordinates = [float(value) for value in box]
        if (
            not all(math.isfinite(value) for value in coordinates)
            or coordinates[2] <= coordinates[0]
            or coordinates[3] <= coordinates[1]
        ):
            raise ValueError("Training annotation contains invalid geometry")
        annotations_by_image[image_id].append(
            {**annotation, "class_name": class_name, "xyxy": coordinates}
        )

    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    split_images: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    linked_sources = []
    try:
        for split in split_images:
            (temporary / "images" / split).mkdir(parents=True, exist_ok=True)
            (temporary / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_id, item in sorted(image_by_id.items()):
            source_value = item.get("image_path") or item.get("file_name")
            source = Path(str(source_value or ""))
            if not source.is_absolute():
                source = image_root / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Training image is missing: {source}")
            expected_hash = str(item.get("sha256") or "").lower()
            actual_hash = _sha256(source)
            if expected_hash and actual_hash != expected_hash:
                raise RuntimeError(f"Training image hash mismatch: {source}")
            split = str(item["split"]).lower()
            destination_name = f"{_safe_name(image_id)}{source.suffix.lower() or '.jpg'}"
            image_destination = temporary / "images" / split / destination_name
            if link_mode == "symlink":
                os.symlink(source, image_destination)
            else:
                os.link(source, image_destination)
            width, height = int(item["width"]), int(item["height"])
            label_rows = []
            for annotation in annotations_by_image[image_id]:
                x1, y1, x2, y2 = annotation["xyxy"]
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    raise ValueError(f"Training box lies outside image {image_id}")
                label_rows.append(
                    " ".join(
                        (
                            str(class_ids[annotation["class_name"]]),
                            f"{((x1 + x2) / 2) / width:.8f}",
                            f"{((y1 + y2) / 2) / height:.8f}",
                            f"{(x2 - x1) / width:.8f}",
                            f"{(y2 - y1) / height:.8f}",
                        )
                    )
                )
            label_destination = temporary / "labels" / split / f"{Path(destination_name).stem}.txt"
            label_destination.write_text(
                "\n".join(label_rows) + ("\n" if label_rows else ""),
                encoding="utf-8",
            )
            split_images[split].append(str(Path("images") / split / destination_name))
            linked_sources.append(
                {
                    "image_id": image_id,
                    "source": str(source),
                    "sha256": actual_hash,
                    "split": split,
                    "annotation_count": len(annotations_by_image[image_id]),
                }
            )
        if not split_images["train"] or not split_images["val"]:
            raise ValueError("Training dataset requires non-empty train and val splits")
        data_yaml = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test" if split_images["test"] else None,
            "names": {index: name for index, name in enumerate(classes)},
        }
        (temporary / "dataset.yaml").write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        receipt = {
            "schema_version": DATASET_SCHEMA,
            "status": "completed",
            "truth_status": "reviewed_ground_truth",
            "dataset_id": payload.get("dataset_id"),
            "source_ground_truth": str(ground_truth_path),
            "source_ground_truth_sha256": _sha256(ground_truth_path),
            "class_count": len(classes),
            "classes": classes,
            "image_count": len(image_by_id),
            "annotation_count": sum(len(items) for items in annotations_by_image.values()),
            "split_image_counts": {key: len(value) for key, value in split_images.items()},
            "link_mode": link_mode,
            "source_copy_bytes": 0,
            "linked_sources": linked_sources,
            "nas_accessed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "dataset-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
        return {**receipt, "output": str(output), "dataset_yaml": str(output / "dataset.yaml")}
    except Exception:
        # Leave an explicit partial directory for audit and recovery; never
        # erase user data or silently reuse an incomplete training dataset.
        raise


def train_yolo_model(
    dataset_root: Path,
    base_model: Path,
    output: Path,
    *,
    epochs: int = 100,
    image_size: int = 1280,
    batch: int = 8,
    device: str = "0",
    patience: int = 20,
    max_hours: float = 2.0,
    workers: int = 8,
) -> dict[str, Any]:
    """Run a real Ultralytics training job from reviewed ground truth only."""

    dataset_root = dataset_root.resolve()
    base_model = base_model.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"YOLO training output already exists: {output}")
    if (
        epochs < 1
        or image_size < 64
        or batch < 1
        or patience < 0
        or max_hours <= 0.0
        or max_hours > 24.0
        or workers < 0
    ):
        raise ValueError("YOLO training limits are invalid")
    receipt_path = dataset_root / "dataset-receipt.json"
    dataset_yaml = dataset_root / "dataset.yaml"
    if not receipt_path.is_file() or not dataset_yaml.is_file():
        raise FileNotFoundError("YOLO training dataset receipt or dataset.yaml is missing")
    dataset_receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (
        dataset_receipt.get("schema_version")
        not in {DATASET_SCHEMA, PUBLIC_DATASET_SCHEMA}
        or dataset_receipt.get("status") != "completed"
        or dataset_receipt.get("truth_status") not in TRAINING_TRUTH_STATUSES
        or dataset_receipt.get("nas_accessed") is not False
    ):
        raise RuntimeError("YOLO training dataset is not trusted human ground truth")
    if not base_model.is_file():
        raise FileNotFoundError(f"YOLO base model is missing: {base_model}")

    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    model = YOLO(str(base_model))
    if not hasattr(model, "add_callback"):
        raise RuntimeError("Ultralytics model does not support bounded training callbacks")
    deadline_monotonic = time.monotonic() + max_hours * 3600.0

    def enforce_wall_time_bound(trainer: Any) -> None:
        if time.monotonic() >= deadline_monotonic:
            trainer.stop = True

    model.add_callback("on_train_epoch_end", enforce_wall_time_bound)
    temporary_telemetry = output.parent / (
        f".{output.name}-resource-telemetry-{uuid.uuid4().hex[:8]}.json"
    )
    monitor = ResourceMonitor(temporary_telemetry, interval_seconds=0.25)
    monitor.start()
    monitor.set_stage("public_yolo_training")
    try:
        result = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            device=device,
            project=str(output.parent),
            name=output.name,
            exist_ok=False,
            plots=True,
            verbose=True,
            patience=patience,
            workers=workers,
        )
    finally:
        telemetry = monitor.stop()
    best = output / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"YOLO training completed without best.pt: {best}")
    telemetry_path = output / "resource-telemetry.json"
    telemetry_live_path = output / "resource-telemetry_live.json"
    os.replace(temporary_telemetry, telemetry_path)
    os.replace(
        temporary_telemetry.with_name(f"{temporary_telemetry.stem}_live.json"),
        telemetry_live_path,
    )
    ended = datetime.now(timezone.utc)
    results_csv = output / "results.csv"
    metric_rows: list[dict[str, str]] = []
    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8-sig") as handle:
            metric_rows = [
                {str(key).strip(): str(value).strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    final_metrics = {
        key: float(value)
        for key, value in (metric_rows[-1] if metric_rows else {}).items()
        if key.startswith("metrics/") and value
    }
    map50_key = "metrics/mAP50(B)"
    map_key = "metrics/mAP50-95(B)"
    best_metric_row = (
        max(
            metric_rows,
            key=lambda row: (
                0.1 * float(row.get(map50_key) or "-inf")
                + 0.9 * float(row.get(map_key) or "-inf")
            ),
        )
        if metric_rows
        else {}
    )
    best_metrics = {
        key: float(value)
        for key, value in best_metric_row.items()
        if key.startswith("metrics/") and value
    }
    best_csv_epoch = (
        int(float(best_metric_row["epoch"]))
        if best_metric_row.get("epoch") is not None
        else None
    )
    receipt = {
        "schema_version": TRAINING_SCHEMA,
        "status": "completed",
        "truth_status": dataset_receipt.get("truth_status"),
        "dataset_receipt": str(receipt_path),
        "dataset_receipt_sha256": _sha256(receipt_path),
        "base_model": str(base_model),
        "base_model_sha256": _sha256(base_model),
        "best_model": str(best),
        "best_model_sha256": _sha256(best),
        "epochs": epochs,
        "image_size": image_size,
        "batch": batch,
        "device": device,
        "patience": patience,
        "max_hours": max_hours,
        "wall_time_bound_enforcement": "epoch_boundary_callback",
        "workers": workers,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "completed_epoch_count": len(metric_rows),
        "final_validation_metrics": final_metrics,
        "best_validation_metric": (
            "ultralytics_detection_fitness="
            "0.1*metrics/mAP50(B)+0.9*metrics/mAP50-95(B)"
        ),
        "best_validation_csv_epoch": best_csv_epoch,
        "best_validation_completed_epoch_number": (
            best_csv_epoch + 1 if best_csv_epoch is not None else None
        ),
        "best_validation_metrics": best_metrics,
        "results_csv": str(results_csv) if results_csv.is_file() else None,
        "trainer_save_dir": str(getattr(result, "save_dir", output)),
        "resource_telemetry": str(telemetry_path),
        "resource_summary": telemetry.get("stage_summaries") or {},
        "telemetry_health": telemetry.get("monitor_health") or {},
        "production_certified": False,
        "certification_required_before_deployment": True,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
    }
    (output / "visioncortex-training-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt


def evaluate_yolo_model_on_human_truth(
    dataset_root: Path,
    model_path: Path,
    output: Path,
    *,
    split: str = "test",
    image_size: int = 640,
    batch: int = 32,
    device: str = "0",
    workers: int = 8,
) -> dict[str, Any]:
    """Evaluate a candidate only against a held-out trusted human-label split."""

    dataset_root = dataset_root.resolve()
    model_path = model_path.resolve()
    output = output.resolve()
    if split not in {"val", "test"}:
        raise ValueError("YOLO human-truth evaluation split must be val or test")
    if image_size < 64 or batch < 1 or workers < 0:
        raise ValueError("YOLO human-truth evaluation limits are invalid")
    if output.exists():
        raise FileExistsError(f"YOLO evaluation output already exists: {output}")
    receipt_path = dataset_root / "dataset-receipt.json"
    dataset_yaml = dataset_root / "dataset.yaml"
    if not receipt_path.is_file() or not dataset_yaml.is_file():
        raise FileNotFoundError("YOLO evaluation dataset receipt or YAML is missing")
    dataset_receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (
        dataset_receipt.get("schema_version")
        not in {DATASET_SCHEMA, PUBLIC_DATASET_SCHEMA}
        or dataset_receipt.get("status") != "completed"
        or dataset_receipt.get("truth_status") not in TRAINING_TRUTH_STATUSES
        or dataset_receipt.get("nas_accessed") is not False
    ):
        raise RuntimeError("YOLO evaluation dataset is not trusted human ground truth")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO evaluation model is missing: {model_path}")

    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    temporary_telemetry = output.parent / (
        f".{output.name}-resource-telemetry-{uuid.uuid4().hex[:8]}.json"
    )
    monitor = ResourceMonitor(temporary_telemetry, interval_seconds=0.25)
    monitor.start()
    monitor.set_stage("public_yolo_human_truth_evaluation")
    try:
        metrics = YOLO(str(model_path)).val(
            data=str(dataset_yaml),
            split=split,
            imgsz=image_size,
            batch=batch,
            device=device,
            workers=workers,
            project=str(output.parent),
            name=output.name,
            exist_ok=False,
            plots=True,
            verbose=True,
        )
    finally:
        telemetry = monitor.stop()
    ended = datetime.now(timezone.utc)
    if not output.is_dir():
        raise RuntimeError(f"YOLO evaluation did not create output: {output}")
    telemetry_path = output / "resource-telemetry.json"
    telemetry_live_path = output / "resource-telemetry_live.json"
    os.replace(temporary_telemetry, telemetry_path)
    os.replace(
        temporary_telemetry.with_name(f"{temporary_telemetry.stem}_live.json"),
        telemetry_live_path,
    )
    results = {
        str(key): float(value)
        for key, value in dict(getattr(metrics, "results_dict", {}) or {}).items()
        if isinstance(value, numbers.Real)
    }
    model_names = {
        int(key): str(value)
        for key, value in dict(getattr(metrics, "names", {}) or {}).items()
    }
    box_metrics = getattr(metrics, "box", None)
    raw_class_maps = getattr(box_metrics, "maps", None)
    class_maps = list(raw_class_maps) if raw_class_maps is not None else []
    raw_class_indexes = getattr(box_metrics, "ap_class_index", None)
    class_indexes = list(raw_class_indexes) if raw_class_indexes is not None else []
    class_positions = {
        int(class_id): position for position, class_id in enumerate(class_indexes)
    }

    def class_metric(name: str, class_id: int) -> float | None:
        raw_values = getattr(box_metrics, name, None)
        values = list(raw_values) if raw_values is not None else []
        position = class_positions.get(class_id, class_id)
        if position >= len(values):
            return None
        return round(float(values[position]), 6)

    per_class = [
        {
            "class_id": class_id,
            "class_name": model_names.get(class_id, str(class_id)),
            "precision": class_metric("p", class_id),
            "recall": class_metric("r", class_id),
            "f1": class_metric("f1", class_id),
            "map50": class_metric("ap50", class_id),
            "map50_95": (
                round(float(class_maps[class_id]), 6)
                if class_id < len(class_maps)
                else None
            ),
        }
        for class_id in sorted(model_names)
    ]
    receipt = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "completed",
        "truth_status": dataset_receipt.get("truth_status"),
        "dataset_receipt": str(receipt_path),
        "dataset_receipt_sha256": _sha256(receipt_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "split": split,
        "image_size": image_size,
        "batch": batch,
        "device": device,
        "workers": workers,
        "metrics": results,
        "per_class": per_class,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "resource_telemetry": str(telemetry_path),
        "resource_summary": telemetry.get("stage_summaries") or {},
        "telemetry_health": telemetry.get("monitor_health") or {},
        "production_certified": False,
        "certification_required_before_deployment": True,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
    }
    (output / "visioncortex-evaluation-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt
