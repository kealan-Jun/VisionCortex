from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DATASET_SCHEMA = "visioncortex-yolo-training-dataset/1"
TRAINING_SCHEMA = "visioncortex-yolo-training-run/1"


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
) -> dict[str, Any]:
    """Run a real Ultralytics training job from reviewed ground truth only."""

    dataset_root = dataset_root.resolve()
    base_model = base_model.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"YOLO training output already exists: {output}")
    if epochs < 1 or image_size < 64 or batch < 1:
        raise ValueError("YOLO training epochs, image_size, and batch must be positive")
    receipt_path = dataset_root / "dataset-receipt.json"
    dataset_yaml = dataset_root / "dataset.yaml"
    if not receipt_path.is_file() or not dataset_yaml.is_file():
        raise FileNotFoundError("YOLO training dataset receipt or dataset.yaml is missing")
    dataset_receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (
        dataset_receipt.get("schema_version") != DATASET_SCHEMA
        or dataset_receipt.get("status") != "completed"
        or dataset_receipt.get("truth_status") != "reviewed_ground_truth"
    ):
        raise RuntimeError("YOLO training dataset is not reviewed ground truth")
    if not base_model.is_file():
        raise FileNotFoundError(f"YOLO base model is missing: {base_model}")

    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    model = YOLO(str(base_model))
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
    )
    best = output / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"YOLO training completed without best.pt: {best}")
    ended = datetime.now(timezone.utc)
    receipt = {
        "schema_version": TRAINING_SCHEMA,
        "status": "completed",
        "truth_status": "reviewed_ground_truth",
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
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "trainer_save_dir": str(getattr(result, "save_dir", output)),
        "production_certified": False,
        "certification_required_before_deployment": True,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
    }
    (output / "visioncortex-training-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt
