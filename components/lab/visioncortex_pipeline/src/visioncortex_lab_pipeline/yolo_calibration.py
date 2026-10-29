from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .telemetry import ResourceMonitor
from .yolo_evaluation import evaluate_yolo_predictions
from .yolo_training import (
    DATASET_SCHEMA,
    PUBLIC_DATASET_SCHEMA,
    PUBLIC_UNION_SCHEMA,
    TRAINING_TRUTH_STATUSES,
)


AUDIT_SCHEMA = "visioncortex-yolo-dataset-integrity-audit/1"
CALIBRATION_SCHEMA = "visioncortex-yolo-confidence-calibration/1"
CALIBRATED_EVALUATION_SCHEMA = "visioncortex-yolo-calibrated-evaluation/1"
TEST_EXPOSURE_STATUSES = {
    "first_use_independent",
    "repeat_comparative_benchmark",
}
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_LOW_CONFIDENCE_MAX_BATCH = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_dataset(dataset_root: Path) -> tuple[Path, dict[str, Any], list[str]]:
    root = dataset_root.resolve()
    receipt_path = root / "dataset-receipt.json"
    yaml_path = root / "dataset.yaml"
    if not receipt_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError("YOLO dataset receipt or dataset.yaml is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (
        receipt.get("schema_version")
        not in {DATASET_SCHEMA, PUBLIC_DATASET_SCHEMA, PUBLIC_UNION_SCHEMA}
        or receipt.get("status") != "completed"
        or receipt.get("truth_status") not in TRAINING_TRUTH_STATUSES
        or receipt.get("nas_accessed") is not False
        or int(receipt.get("source_copy_bytes") or 0) != 0
    ):
        raise RuntimeError("YOLO dataset is not trusted zero-copy human ground truth")
    yaml_payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    names = yaml_payload.get("names")
    if isinstance(names, list):
        classes = [str(item).strip() for item in names]
    elif isinstance(names, dict):
        indexed = {int(key): str(value).strip() for key, value in names.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise RuntimeError("YOLO dataset class ids are not contiguous")
        classes = [indexed[index] for index in range(len(indexed))]
    else:
        raise RuntimeError("YOLO dataset.yaml does not declare names")
    if (
        not classes
        or any(not item for item in classes)
        or len(classes) != len(set(classes))
        or receipt.get("classes") != classes
    ):
        raise RuntimeError("YOLO dataset class ontology does not match its receipt")
    return root, receipt, classes


def _split_pairs(root: Path, split: str) -> list[tuple[str, Path, Path]]:
    images_root = root / "images" / split
    labels_root = root / "labels" / split
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(f"YOLO dataset split is missing: {split}")
    images: dict[Path, Path] = {}
    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in _IMAGE_SUFFIXES:
            continue
        stem = path.relative_to(images_root).with_suffix("")
        if stem in images:
            raise RuntimeError(f"Duplicate YOLO image stem: {split}/{stem}")
        images[stem] = path
    labels = {
        path.relative_to(labels_root).with_suffix(""): path
        for path in labels_root.rglob("*.txt")
        if path.is_file()
    }
    if not images or images.keys() != labels.keys():
        raise RuntimeError(f"YOLO image/label mismatch: {split}")
    return [
        (stem.as_posix(), images[stem], labels[stem]) for stem in sorted(images)
    ]


def _label_rows(path: Path, class_count: int) -> list[tuple[int, list[float]]]:
    rows = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        values = raw.split()
        if len(values) != 5:
            raise RuntimeError(f"Invalid YOLO label row: {path}:{line_number}")
        try:
            class_id = int(values[0])
            coordinates = [float(item) for item in values[1:]]
        except ValueError as exc:
            raise RuntimeError(f"Invalid YOLO label value: {path}:{line_number}") from exc
        if (
            class_id < 0
            or class_id >= class_count
            or not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in coordinates)
            or coordinates[2] <= 0.0
            or coordinates[3] <= 0.0
        ):
            raise RuntimeError(f"Invalid YOLO label geometry: {path}:{line_number}")
        rows.append((class_id, coordinates))
    if not rows:
        raise RuntimeError(f"Mapped public YOLO label is empty: {path}")
    return rows


def audit_yolo_dataset_integrity(
    dataset_root: Path,
    output: Path,
    *,
    focus_classes: Iterable[str] = ("hand", "pipette"),
) -> dict[str, Any]:
    """Hash every unique image and fail closed on train/val/test leakage."""

    root, dataset_receipt, classes = _load_dataset(dataset_root)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"YOLO integrity audit output already exists: {output}")
    focus = tuple(dict.fromkeys(str(item) for item in focus_classes))
    if not focus or any(item not in classes for item in focus):
        raise ValueError("YOLO audit focus classes must exist in the dataset ontology")
    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.mkdir(parents=True)
    started = time.monotonic()
    source_hash_cache: dict[Path, str] = {}
    digest_splits: dict[str, set[str]] = {}
    digest_members: dict[str, set[tuple[str, str]]] = {}
    target_splits: dict[Path, set[str]] = {}
    split_image_counts: dict[str, int] = {}
    split_unique_source_counts: dict[str, int] = {}
    split_annotation_counts: dict[str, int] = {}
    focus_annotation_counts: dict[str, dict[str, int]] = {}
    repeated_train_source_count = 0
    for split in ("train", "val", "test"):
        pairs = _split_pairs(root, split)
        split_image_counts[split] = len(pairs)
        split_annotation_counts[split] = 0
        focus_annotation_counts[split] = {name: 0 for name in focus}
        sources_in_split: set[Path] = set()
        for _, image_path, label_path in pairs:
            source = image_path.resolve(strict=True)
            if not source.is_file():
                raise RuntimeError(f"YOLO source image is not a regular file: {source}")
            sources_in_split.add(source)
            target_splits.setdefault(source, set()).add(split)
            digest = source_hash_cache.get(source)
            if digest is None:
                digest = _sha256(source)
                source_hash_cache[source] = digest
            digest_splits.setdefault(digest, set()).add(split)
            digest_members.setdefault(digest, set()).add((split, str(source)))
            rows = _label_rows(label_path, len(classes))
            split_annotation_counts[split] += len(rows)
            for class_id, _ in rows:
                name = classes[class_id]
                if name in focus_annotation_counts[split]:
                    focus_annotation_counts[split][name] += 1
        split_unique_source_counts[split] = len(sources_in_split)
        if split == "train":
            repeated_train_source_count = len(pairs) - len(sources_in_split)
    cross_split_source_targets = [
        {"source": str(source), "splits": sorted(splits)}
        for source, splits in sorted(target_splits.items(), key=lambda item: str(item[0]))
        if len(splits) > 1
    ]
    cross_split_content_hashes = [
        {
            "sha256": digest,
            "splits": sorted(splits),
            "members": [
                {"split": split, "source": source}
                for split, source in sorted(digest_members[digest])
            ],
        }
        for digest, splits in sorted(digest_splits.items())
        if len(splits) > 1
    ]
    passed = not cross_split_source_targets and not cross_split_content_hashes
    receipt_path = root / "dataset-receipt.json"
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "status": "completed" if passed else "failed_split_leakage",
        "passed": passed,
        "dataset_root": str(root),
        "dataset_receipt": str(receipt_path),
        "dataset_receipt_sha256": _sha256(receipt_path),
        "truth_status": dataset_receipt.get("truth_status"),
        "classes": classes,
        "focus_classes": list(focus),
        "split_image_counts": split_image_counts,
        "split_unique_source_counts": split_unique_source_counts,
        "split_annotation_counts": split_annotation_counts,
        "focus_annotation_counts": focus_annotation_counts,
        "unique_source_image_count": len(source_hash_cache),
        "repeated_train_source_count": repeated_train_source_count,
        "cross_split_source_target_count": len(cross_split_source_targets),
        "cross_split_source_targets": cross_split_source_targets,
        "cross_split_content_hash_count": len(cross_split_content_hashes),
        "cross_split_content_hashes": cross_split_content_hashes,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(temporary / "dataset-integrity-audit.json", payload)
    os.replace(temporary, output)
    return payload


def _predict_split(
    dataset_root: Path,
    model_path: Path,
    split: str,
    output: Path,
    *,
    image_size: int,
    batch: int,
    device: str,
    confidence_floor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    root, _, classes = _load_dataset(dataset_root)
    pairs = _split_pairs(root, split)
    pair_by_source: dict[Path, tuple[str, Path]] = {}
    for stem, image_path, label_path in pairs:
        source = image_path.resolve(strict=True)
        if source in pair_by_source:
            raise RuntimeError(f"Duplicate source image in calibration split: {source}")
        pair_by_source[source] = (stem, label_path)
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model_names = [str(model.names[index]) for index in sorted(model.names)]
    if model_names != classes:
        raise RuntimeError("YOLO candidate ontology does not match calibration dataset")
    predictions: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen: set[Path] = set()
    # Very low confidence floors intentionally retain many boxes for threshold
    # calibration.  Ultralytics NMS can otherwise require several GiB of
    # temporary CUDA memory for a large batch.  Calibration is an accuracy
    # workflow, not a throughput benchmark, so bound only this raw-candidate
    # batch while preserving every image and prediction.
    effective_batch = (
        min(batch, _LOW_CONFIDENCE_MAX_BATCH)
        if confidence_floor < 0.05
        else batch
    )
    prediction_sources = [str(image_path) for _, image_path, _ in pairs]

    def _batched_results() -> Iterable[tuple[Path, Any]]:
        # Passing the whole source list to one Ultralytics stream still lets its
        # predictor retain large intermediate buffers across heterogeneous
        # source groups.  Explicitly bound each call so CUDA working memory is
        # released between chunks; this does not drop or resample any image.
        for start in range(0, len(prediction_sources), effective_batch):
            chunk = prediction_sources[start : start + effective_batch]
            chunk_results = model.predict(
                source=chunk,
                stream=True,
                imgsz=image_size,
                batch=effective_batch,
                device=device,
                conf=confidence_floor,
                iou=0.70,
                max_det=300,
                save=False,
                verbose=False,
            )
            # Ultralytics assigns generic result.path values (image0.jpg, ...)
            # when source is a Python path list.  Its result order is the input
            # order, so pair them strictly and retain the audited absolute
            # source path instead of trusting that generic display name.
            yield from zip(
                (Path(item).resolve(strict=True) for item in chunk),
                chunk_results,
                strict=True,
            )

    results = _batched_results()
    for source, result in results:
        if source not in pair_by_source or source in seen:
            raise RuntimeError(f"Unexpected YOLO prediction source: {source}")
        seen.add(source)
        stem, label_path = pair_by_source[source]
        image_id = f"{split}:{stem}"
        height, width = [int(item) for item in result.orig_shape]
        images.append(
            {
                "image_id": image_id,
                "event_id": image_id,
                "role": split,
                "frame_index": 0,
                "source_image": str(source),
                "width": width,
                "height": height,
            }
        )
        for index, (class_id, xywh) in enumerate(
            _label_rows(label_path, len(classes)), start=1
        ):
            center_x, center_y, box_width, box_height = xywh
            xyxy = [
                max(0.0, (center_x - box_width / 2.0) * width),
                max(0.0, (center_y - box_height / 2.0) * height),
                min(float(width), (center_x + box_width / 2.0) * width),
                min(float(height), (center_y + box_height / 2.0) * height),
            ]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                raise RuntimeError(f"YOLO label collapses at image boundary: {label_path}")
            annotations.append(
                {
                    "annotation_id": f"{image_id}:{index}",
                    "image_id": image_id,
                    "class_name": classes[class_id],
                    "xyxy": xyxy,
                }
            )
        detections = []
        boxes = result.boxes
        if boxes is not None:
            xyxy_rows = boxes.xyxy.detach().cpu().tolist()
            confidence_rows = boxes.conf.detach().cpu().tolist()
            class_rows = boxes.cls.detach().cpu().tolist()
            for xyxy, confidence, raw_class_id in zip(
                xyxy_rows, confidence_rows, class_rows, strict=True
            ):
                class_id = int(raw_class_id)
                detections.append(
                    {
                        "class_name": classes[class_id],
                        "confidence": float(confidence),
                        "xyxy": [float(item) for item in xyxy],
                    }
                )
        predictions.append(
            {
                "event_id": image_id,
                "role": split,
                "frame_index": 0,
                "source_image": str(source),
                "detections": detections,
            }
        )
    if len(seen) != len(pairs):
        raise RuntimeError("YOLO prediction stream did not return every split image")
    ground_truth = {
        "schema_version": "visioncortex-yolo-calibration-ground-truth/1",
        "dataset_id": str(root),
        "images": images,
        "annotations": annotations,
    }
    predictions_path = output / "raw-predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in predictions
        ),
        encoding="utf-8",
    )
    ground_truth_path = output / "ground-truth.json"
    ground_truth_path.write_text(
        json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    artifacts = {
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "ground_truth": str(ground_truth_path),
        "ground_truth_sha256": _sha256(ground_truth_path),
        "requested_batch": batch,
        "effective_batch": effective_batch,
        "low_confidence_batch_cap": _LOW_CONFIDENCE_MAX_BATCH,
        "prediction_chunk_count": math.ceil(len(prediction_sources) / effective_batch),
    }
    return predictions, ground_truth, artifacts


def _metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "confidence_threshold": report["confidence_threshold"],
        "class_confidence_thresholds": report["class_confidence_thresholds"],
        "image_count": report["image_count"],
        "ground_truth_instance_count": report["ground_truth_instance_count"],
        "micro": report["micro"],
        "macro": report["macro"],
        "per_class": report["per_class"],
    }


def _select_threshold(
    rows: list[dict[str, Any]], minimum_precision: float, minimum_recall: float
) -> dict[str, Any]:
    passing = [
        row
        for row in rows
        if row["precision"] >= minimum_precision and row["recall"] >= minimum_recall
    ]
    if passing:
        selected = max(
            passing,
            # Candidate generation is recall-oriented: once the explicit
            # precision floor is satisfied, retain the operating point that
            # misses the fewest objects. The participant/relation layer later
            # suppresses non-interacting boxes. An F1-first selector can
            # silently choose a higher threshold and discard valid hands or
            # tools despite a lower, still-compliant validation point.
            key=lambda row: (
                row["recall"],
                row["precision"],
                row["f1"],
                -row["threshold"],
            ),
        )
        return {
            **selected,
            "gate_passed": True,
            "selection_reason": "highest_recall_meeting_precision_and_recall_gate",
        }
    selected = max(
        rows,
        key=lambda row: (
            min(
                row["precision"] / max(minimum_precision, 1e-12),
                row["recall"] / max(minimum_recall, 1e-12),
            ),
            row["f1"],
            row["precision"],
            row["recall"],
        ),
    )
    return {**selected, "gate_passed": False, "selection_reason": "closest_balanced_point_below_gate"}


def calibrate_yolo_confidence_thresholds(
    dataset_root: Path,
    model_path: Path,
    audit_receipt_path: Path,
    output: Path,
    *,
    target_classes: Iterable[str] = ("hand", "pipette"),
    thresholds: Iterable[float] = (
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ),
    minimum_precision: float = 0.85,
    minimum_recall: float = 0.80,
    default_threshold: float = 0.25,
    annotation_gap_confidence: float = 0.80,
    image_size: int = 640,
    batch: int = 32,
    device: str = "0",
) -> dict[str, Any]:
    """Choose class thresholds on validation truth without inspecting test labels."""

    root, _, classes = _load_dataset(dataset_root)
    model_path = model_path.resolve()
    audit_receipt_path = audit_receipt_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"YOLO calibration output already exists: {output}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO calibration model is missing: {model_path}")
    audit = json.loads(audit_receipt_path.read_text(encoding="utf-8"))
    receipt_path = root / "dataset-receipt.json"
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("dataset_receipt_sha256") != _sha256(receipt_path)
        or int(audit.get("cross_split_content_hash_count") or 0) != 0
        or int(audit.get("source_copy_bytes") or 0) != 0
        or audit.get("nas_accessed") is not False
    ):
        raise RuntimeError("A passing matching dataset integrity audit is required")
    targets = tuple(dict.fromkeys(str(item) for item in target_classes))
    candidates = sorted({round(float(item), 6) for item in thresholds})
    if (
        not targets
        or any(item not in classes for item in targets)
        or not candidates
        or any(not 0.0 < item <= 1.0 for item in candidates)
        or not 0.0 <= default_threshold <= 1.0
        or not 0.0 <= annotation_gap_confidence <= 1.0
        or not 0.0 <= minimum_precision <= 1.0
        or not 0.0 <= minimum_recall <= 1.0
        or image_size < 64
        or batch < 1
    ):
        raise ValueError("YOLO calibration limits are invalid")
    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.mkdir(parents=True)
    telemetry_temp = output.parent / f".{output.name}-telemetry-{uuid.uuid4().hex[:8]}.json"
    monitor = ResourceMonitor(telemetry_temp, interval_seconds=0.25)
    monitor.start()
    monitor.set_stage("yolo_validation_threshold_calibration")
    started = datetime.now(timezone.utc)
    try:
        predictions, ground_truth, artifacts = _predict_split(
            root,
            model_path,
            "val",
            temporary,
            image_size=image_size,
            batch=batch,
            device=device,
            confidence_floor=min(candidates),
        )
    finally:
        telemetry = monitor.stop()
    ended = datetime.now(timezone.utc)
    os.replace(telemetry_temp, temporary / "resource-telemetry.json")
    os.replace(
        telemetry_temp.with_name(f"{telemetry_temp.stem}_live.json"),
        temporary / "resource-telemetry_live.json",
    )
    sweep: dict[str, list[dict[str, Any]]] = {name: [] for name in targets}
    for threshold in candidates:
        report = evaluate_yolo_predictions(
            predictions,
            ground_truth,
            confidence_threshold=default_threshold,
            class_confidence_thresholds={name: threshold for name in targets},
            iou_thresholds=(0.5,),
        )
        for name in targets:
            metrics = report["per_class"][name]
            sweep[name].append(
                {
                    "threshold": threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "true_positive": metrics["true_positive"],
                    "false_positive": metrics["false_positive"],
                    "false_negative": metrics["false_negative"],
                }
            )
    selections = {
        name: _select_threshold(sweep[name], minimum_precision, minimum_recall)
        for name in targets
    }
    selected_thresholds = {
        name: float(selection["threshold"]) for name, selection in selections.items()
    }
    selected_report = evaluate_yolo_predictions(
        predictions,
        ground_truth,
        confidence_threshold=default_threshold,
        class_confidence_thresholds=selected_thresholds,
    )
    gap_candidates = sorted(
        (
            {
                **error,
                "status": "annotation_gap_candidate_not_ground_truth",
            }
            for error in selected_report["errors"]
            if error.get("kind") == "false_positive"
            and error.get("class_name") in targets
            and float(error.get("confidence") or 0.0) >= annotation_gap_confidence
        ),
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )[:500]
    gap_path = temporary / "annotation-gap-candidates.json"
    gap_path.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-yolo-annotation-gap-candidates/1",
                "truth_status": "candidate_not_ground_truth",
                "selection_split": "val",
                "minimum_confidence": annotation_gap_confidence,
                "candidate_count": len(gap_candidates),
                "candidates": gap_candidates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": CALIBRATION_SCHEMA,
        "status": "completed",
        "truth_status": "public_human_annotations",
        "selection_split": "val",
        "test_labels_inspected": False,
        "dataset_root": str(root),
        "dataset_receipt": str(receipt_path),
        "dataset_receipt_sha256": _sha256(receipt_path),
        "dataset_integrity_audit": str(audit_receipt_path),
        "dataset_integrity_audit_sha256": _sha256(audit_receipt_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "target_classes": list(targets),
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "default_threshold": default_threshold,
        "threshold_sweep": sweep,
        "selections": selections,
        "selected_class_thresholds": selected_thresholds,
        "validation_gate_passed": all(
            item["gate_passed"] for item in selections.values()
        ),
        "selected_validation_metrics": _metric_summary(selected_report),
        "annotation_gap_candidates": str(output / gap_path.name),
        "annotation_gap_candidate_count": len(gap_candidates),
        "raw_artifacts": {
            key: str(output / Path(value).name) if key in {"predictions", "ground_truth"} else value
            for key, value in artifacts.items()
        },
        "image_size": image_size,
        "requested_batch": batch,
        "effective_batch": artifacts["effective_batch"],
        "device": device,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "resource_telemetry": str(output / "resource-telemetry.json"),
        "resource_summary": telemetry.get("stage_summaries") or {},
        "telemetry_health": telemetry.get("monitor_health") or {},
        "production_certified": False,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(temporary / "threshold-calibration.json", payload)
    os.replace(temporary, output)
    return payload


def evaluate_yolo_with_calibrated_thresholds(
    dataset_root: Path,
    model_path: Path,
    calibration_receipt_path: Path,
    output: Path,
    *,
    image_size: int = 640,
    batch: int = 32,
    device: str = "0",
    test_exposure_status: str = "repeat_comparative_benchmark",
) -> dict[str, Any]:
    """Apply validation-frozen thresholds to test truth with exposure provenance."""

    root, _, classes = _load_dataset(dataset_root)
    model_path = model_path.resolve()
    calibration_receipt_path = calibration_receipt_path.resolve()
    output = output.resolve()
    if test_exposure_status not in TEST_EXPOSURE_STATUSES:
        raise ValueError(
            "test_exposure_status must be first_use_independent or "
            "repeat_comparative_benchmark"
        )
    if output.exists():
        raise FileExistsError(f"YOLO calibrated evaluation output exists: {output}")
    calibration = json.loads(calibration_receipt_path.read_text(encoding="utf-8"))
    receipt_path = root / "dataset-receipt.json"
    raw_audit_path = calibration.get("dataset_integrity_audit")
    audit_path = (
        Path(str(raw_audit_path)).resolve()
        if isinstance(raw_audit_path, str) and raw_audit_path
        else None
    )
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path is not None and audit_path.is_file()
        else {}
    )
    thresholds = {
        str(name): float(value)
        for name, value in (calibration.get("selected_class_thresholds") or {}).items()
    }
    targets = [str(item) for item in calibration.get("target_classes") or []]
    if (
        calibration.get("schema_version") != CALIBRATION_SCHEMA
        or calibration.get("status") != "completed"
        or calibration.get("selection_split") != "val"
        or calibration.get("test_labels_inspected") is not False
        or calibration.get("dataset_receipt_sha256") != _sha256(receipt_path)
        or calibration.get("model_sha256") != _sha256(model_path)
        or audit_path is None
        or not audit_path.is_file()
        or calibration.get("dataset_integrity_audit_sha256")
        != _sha256(audit_path)
        or audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("dataset_receipt_sha256") != _sha256(receipt_path)
        or int(audit.get("cross_split_content_hash_count") or 0) != 0
        or int(audit.get("source_copy_bytes") or 0) != 0
        or audit.get("nas_accessed") is not False
        or not targets
        or any(name not in classes or name not in thresholds for name in targets)
    ):
        raise RuntimeError("Matching validation-only YOLO calibration is required")
    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.mkdir(parents=True)
    telemetry_temp = output.parent / f".{output.name}-telemetry-{uuid.uuid4().hex[:8]}.json"
    monitor = ResourceMonitor(telemetry_temp, interval_seconds=0.25)
    monitor.start()
    monitor.set_stage(
        "yolo_calibrated_independent_test"
        if test_exposure_status == "first_use_independent"
        else "yolo_calibrated_repeat_comparative_test"
    )
    started = datetime.now(timezone.utc)
    try:
        predictions, ground_truth, artifacts = _predict_split(
            root,
            model_path,
            "test",
            temporary,
            image_size=image_size,
            batch=batch,
            device=device,
            confidence_floor=min(
                [
                    float(calibration.get("default_threshold", 0.25)),
                    *thresholds.values(),
                ]
            ),
        )
    finally:
        telemetry = monitor.stop()
    ended = datetime.now(timezone.utc)
    os.replace(telemetry_temp, temporary / "resource-telemetry.json")
    os.replace(
        telemetry_temp.with_name(f"{telemetry_temp.stem}_live.json"),
        temporary / "resource-telemetry_live.json",
    )
    report = evaluate_yolo_predictions(
        predictions,
        ground_truth,
        confidence_threshold=float(calibration.get("default_threshold", 0.25)),
        class_confidence_thresholds=thresholds,
    )
    minimum_precision = float(calibration.get("minimum_precision", 1.0))
    minimum_recall = float(calibration.get("minimum_recall", 1.0))
    class_gate_checks = [
        {
            "class_name": name,
            "threshold": thresholds[name],
            "precision": report["per_class"][name]["precision"],
            "recall": report["per_class"][name]["recall"],
            "minimum_precision": minimum_precision,
            "minimum_recall": minimum_recall,
            "passed": bool(
                report["per_class"][name]["precision"] >= minimum_precision
                and report["per_class"][name]["recall"] >= minimum_recall
            ),
        }
        for name in targets
    ]
    payload = {
        "schema_version": CALIBRATED_EVALUATION_SCHEMA,
        "status": "completed",
        "truth_status": "public_human_annotations",
        "evaluation_split": "test",
        "threshold_selection_split": "val",
        "thresholds_frozen_before_test": True,
        "test_exposure_status": test_exposure_status,
        "independent_test_claim_allowed": (
            test_exposure_status == "first_use_independent"
        ),
        "dataset_root": str(root),
        "dataset_receipt": str(receipt_path),
        "dataset_receipt_sha256": _sha256(receipt_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "calibration_receipt": str(calibration_receipt_path),
        "calibration_receipt_sha256": _sha256(calibration_receipt_path),
        "selected_class_thresholds": thresholds,
        "metrics": _metric_summary(report),
        "class_gate_checks": class_gate_checks,
        "public_operating_point_gate_passed": all(
            item["passed"] for item in class_gate_checks
        ),
        "raw_artifacts": {
            key: str(output / Path(value).name) if key in {"predictions", "ground_truth"} else value
            for key, value in artifacts.items()
        },
        "image_size": image_size,
        "requested_batch": batch,
        "effective_batch": artifacts["effective_batch"],
        "device": device,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "resource_telemetry": str(output / "resource-telemetry.json"),
        "resource_summary": telemetry.get("stage_summaries") or {},
        "telemetry_health": telemetry.get("monitor_health") or {},
        "production_certified": False,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(temporary / "calibrated-test-evaluation.json", payload)
    os.replace(temporary, output)
    return payload
