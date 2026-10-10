from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .telemetry import ResourceMonitor
from .yolo_calibration import (
    AUDIT_SCHEMA,
    _LOW_CONFIDENCE_MAX_BATCH,
    _label_rows,
    _load_dataset,
    _select_threshold,
    _split_pairs,
)
from .yolo_evaluation import evaluate_yolo_predictions


YOLO_WORLD_CALIBRATION_SCHEMA = "visioncortex-yolo-world-calibration/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _canonical_nms(
    detections: Iterable[dict[str, Any]], *, iou_threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Deduplicate synonymous prompts after mapping to canonical classes."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("YOLO-World canonical NMS IoU must be between 0 and 1")
    kept: list[dict[str, Any]] = []
    for detection in sorted(
        (dict(item) for item in detections),
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    ):
        box = [float(item) for item in detection.get("xyxy") or []]
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise RuntimeError("YOLO-World returned an invalid detection box")
        if any(
            item.get("class_name") == detection.get("class_name")
            and _iou([float(value) for value in item["xyxy"]], box) > iou_threshold
            for item in kept
        ):
            continue
        detection["xyxy"] = box
        kept.append(detection)
    return kept


def calibrate_yolo_world_prompts(
    dataset_root: Path,
    model_path: Path,
    audit_receipt_path: Path,
    output: Path,
    *,
    prompt_map: Mapping[str, str],
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
    image_size: int = 640,
    batch: int = 16,
    device: str = "0",
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Measure open-vocabulary prompts on validation truth before integration."""

    root, _, classes = _load_dataset(dataset_root)
    model_path = model_path.resolve()
    audit_receipt_path = audit_receipt_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"YOLO-World calibration output exists: {output}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO-World model is missing: {model_path}")
    dataset_receipt_path = root / "dataset-receipt.json"
    audit = json.loads(audit_receipt_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("dataset_receipt_sha256") != _sha256(dataset_receipt_path)
        or int(audit.get("cross_split_content_hash_count") or 0) != 0
        or int(audit.get("source_copy_bytes") or 0) != 0
        or audit.get("nas_accessed") is not False
    ):
        raise RuntimeError("A passing matching dataset integrity audit is required")
    targets = tuple(dict.fromkeys(str(item).strip() for item in target_classes))
    prompts = tuple(str(item).strip() for item in prompt_map)
    canonical_by_prompt = {
        str(prompt).strip(): str(canonical).strip()
        for prompt, canonical in prompt_map.items()
    }
    candidates = sorted({round(float(item), 6) for item in thresholds})
    if (
        not targets
        or any(target not in classes for target in targets)
        or not prompts
        or any(not prompt or not canonical_by_prompt[prompt] for prompt in prompts)
        or any(canonical_by_prompt[prompt] not in targets for prompt in prompts)
        or not set(targets).issubset(canonical_by_prompt.values())
        or not candidates
        or any(not 0.0 < item <= 1.0 for item in candidates)
        or not 0.0 <= minimum_precision <= 1.0
        or not 0.0 <= minimum_recall <= 1.0
        or image_size < 64
        or batch < 1
        or not 0.0 <= iou_threshold <= 1.0
    ):
        raise ValueError("YOLO-World calibration inputs are invalid")

    pairs = _split_pairs(root, "val")
    effective_batch = (
        min(batch, _LOW_CONFIDENCE_MAX_BATCH)
        if min(candidates) < 0.05
        else batch
    )
    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.mkdir(parents=True)
    telemetry_temp = output.parent / f".{output.name}-telemetry-{uuid.uuid4().hex[:8]}.json"
    monitor = ResourceMonitor(telemetry_temp, interval_seconds=0.25)
    monitor.start()
    monitor.set_stage("yolo_world_validation_prompt_calibration")
    started = datetime.now(timezone.utc)
    predictions: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    try:
        from ultralytics import YOLOWorld

        model = YOLOWorld(str(model_path))
        model.set_classes(list(prompts))
        for start in range(0, len(pairs), effective_batch):
            chunk = pairs[start : start + effective_batch]
            sources = [str(image.resolve(strict=True)) for _, image, _ in chunk]
            results = model.predict(
                source=sources,
                stream=True,
                imgsz=image_size,
                batch=effective_batch,
                device=device,
                conf=min(candidates),
                iou=iou_threshold,
                max_det=300,
                save=False,
                verbose=False,
            )
            for (stem, source_image, label_path), result in zip(
                chunk, results, strict=True
            ):
                image_id = f"val:{stem}"
                height, width = [int(item) for item in result.orig_shape]
                images.append(
                    {
                        "image_id": image_id,
                        "event_id": image_id,
                        "role": "val",
                        "frame_index": 0,
                        "source_image": str(source_image.resolve(strict=True)),
                        "width": width,
                        "height": height,
                    }
                )
                for row_index, (class_id, xywh) in enumerate(
                    _label_rows(label_path, len(classes)), start=1
                ):
                    class_name = classes[class_id]
                    if class_name not in targets:
                        continue
                    center_x, center_y, box_width, box_height = xywh
                    annotations.append(
                        {
                            "annotation_id": f"{image_id}:{row_index}",
                            "image_id": image_id,
                            "class_name": class_name,
                            "xyxy": [
                                max(0.0, (center_x - box_width / 2.0) * width),
                                max(0.0, (center_y - box_height / 2.0) * height),
                                min(float(width), (center_x + box_width / 2.0) * width),
                                min(float(height), (center_y + box_height / 2.0) * height),
                            ],
                        }
                    )
                detections = []
                boxes = result.boxes
                if boxes is not None:
                    for xyxy, confidence, raw_class_id in zip(
                        boxes.xyxy.detach().cpu().tolist(),
                        boxes.conf.detach().cpu().tolist(),
                        boxes.cls.detach().cpu().tolist(),
                        strict=True,
                    ):
                        prompt = str(result.names[int(raw_class_id)])
                        canonical = canonical_by_prompt.get(prompt)
                        if canonical is None:
                            raise RuntimeError(
                                f"YOLO-World returned an unknown prompt: {prompt}"
                            )
                        detections.append(
                            {
                                "class_name": canonical,
                                "confidence": float(confidence),
                                "xyxy": [float(item) for item in xyxy],
                                "grounding_prompt": prompt,
                                "detector_source": "yolo_world_v2_validation",
                            }
                        )
                predictions.append(
                    {
                        "event_id": image_id,
                        "role": "val",
                        "frame_index": 0,
                        "source_image": str(source_image.resolve(strict=True)),
                        "detections": _canonical_nms(
                            detections, iou_threshold=iou_threshold
                        ),
                    }
                )
    finally:
        telemetry = monitor.stop()
    ended = datetime.now(timezone.utc)
    os.replace(telemetry_temp, temporary / "resource-telemetry.json")
    live_path = telemetry_temp.with_name(f"{telemetry_temp.stem}_live.json")
    if live_path.is_file():
        os.replace(live_path, temporary / "resource-telemetry_live.json")

    truth = {
        "schema_version": "visioncortex-yolo-world-ground-truth/1",
        "dataset_id": str(root),
        "images": images,
        "annotations": annotations,
    }
    raw_predictions_path = temporary / "raw-predictions.jsonl"
    raw_predictions_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in predictions
        ),
        encoding="utf-8",
    )
    truth_path = temporary / "ground-truth.json"
    truth_path.write_text(
        json.dumps(truth, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    sweep: dict[str, list[dict[str, Any]]] = {name: [] for name in targets}
    for target in targets:
        target_truth = {
            **truth,
            "annotations": [
                item for item in annotations if item["class_name"] == target
            ],
        }
        target_predictions = [
            {
                **item,
                "detections": [
                    detection
                    for detection in item["detections"]
                    if detection["class_name"] == target
                ],
            }
            for item in predictions
        ]
        for threshold in candidates:
            report = evaluate_yolo_predictions(
                target_predictions,
                target_truth,
                confidence_threshold=threshold,
                iou_thresholds=(0.5,),
            )
            metrics = report["per_class"][target]
            sweep[target].append(
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
    payload = {
        "schema_version": YOLO_WORLD_CALIBRATION_SCHEMA,
        "status": "completed",
        "selection_split": "val",
        "test_labels_inspected": False,
        "integration_status": "measurement_only_not_integrated",
        "dataset_root": str(root),
        "dataset_receipt": str(dataset_receipt_path),
        "dataset_receipt_sha256": _sha256(dataset_receipt_path),
        "dataset_integrity_audit": str(audit_receipt_path),
        "dataset_integrity_audit_sha256": _sha256(audit_receipt_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "prompt_map": canonical_by_prompt,
        "target_classes": list(targets),
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "threshold_sweep": sweep,
        "selections": selections,
        "validation_gate_passed": all(
            selection["gate_passed"] for selection in selections.values()
        ),
        "image_size": image_size,
        "requested_batch": batch,
        "effective_batch": effective_batch,
        "low_confidence_batch_cap": _LOW_CONFIDENCE_MAX_BATCH,
        "prediction_chunk_count": (
            len(pairs) + effective_batch - 1
        ) // effective_batch,
        "device": device,
        "iou_threshold": iou_threshold,
        "image_count": len(images),
        "ground_truth_instance_count": len(annotations),
        "raw_predictions": str(output / raw_predictions_path.name),
        "raw_predictions_sha256": _sha256(raw_predictions_path),
        "ground_truth": str(output / truth_path.name),
        "ground_truth_sha256": _sha256(truth_path),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "resource_telemetry": str(output / "resource-telemetry.json"),
        "resource_summary": telemetry.get("stage_summaries") or {},
        "telemetry_health": telemetry.get("monitor_health") or {},
        "production_certified": False,
        "production_configuration_changed": False,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = temporary / "yolo-world-calibration.json"
    receipt_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, output)
    return payload
