from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _frame_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["event_id"]),
        str(record["role"]),
        int(record["frame_index"]),
    )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _average_precision(points: list[tuple[float, int]], gt_count: int) -> float:
    if gt_count <= 0:
        return 0.0
    true_positive = 0
    false_positive = 0
    curve: list[tuple[float, float]] = []
    for _, label in sorted(points, key=lambda item: item[0], reverse=True):
        if label:
            true_positive += 1
        else:
            false_positive += 1
        recall = true_positive / gt_count
        precision = true_positive / max(1, true_positive + false_positive)
        curve.append((recall, precision))
    interpolated = []
    for step in range(101):
        target_recall = step / 100.0
        interpolated.append(
            max((precision for recall, precision in curve if recall >= target_recall), default=0.0)
        )
    return sum(interpolated) / len(interpolated)


def _class_threshold_metrics(
    class_name: str,
    frame_predictions: dict[tuple[str, str, int], list[dict[str, Any]]],
    frame_ground_truth: dict[tuple[str, str, int], list[dict[str, Any]]],
    *,
    iou_threshold: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    gt_by_frame = {
        key: [item for item in values if item["class_name"] == class_name]
        for key, values in frame_ground_truth.items()
    }
    gt_count = sum(len(items) for items in gt_by_frame.values())
    matched: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    threshold_matched: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    predictions = []
    for key, values in frame_predictions.items():
        for item in values:
            if item["class_name"] == class_name:
                predictions.append((float(item["confidence"]), key, item))
    points: list[tuple[float, int]] = []
    errors: list[dict[str, Any]] = []
    for confidence, key, prediction in sorted(predictions, reverse=True, key=lambda item: item[0]):
        best_index = None
        best_iou = 0.0
        for index, truth in enumerate(gt_by_frame.get(key, [])):
            if index in matched[key]:
                continue
            overlap = _iou(prediction["xyxy"], truth["xyxy"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        is_match = best_index is not None and best_iou >= iou_threshold
        if is_match:
            matched[key].add(int(best_index))
            if confidence >= confidence_threshold:
                threshold_matched[key].add(int(best_index))
        points.append((confidence, int(is_match)))
        if not is_match and confidence >= confidence_threshold:
            errors.append(
                {
                    "kind": "false_positive",
                    "class_name": class_name,
                    "event_id": key[0],
                    "role": key[1],
                    "frame_index": key[2],
                    "confidence": confidence,
                    "best_iou": round(best_iou, 6),
                    "xyxy": prediction["xyxy"],
                }
            )
    threshold_points = [item for item in points if item[0] >= confidence_threshold]
    true_positive = sum(label for _, label in threshold_points)
    false_positive = len(threshold_points) - true_positive
    false_negative = gt_count - true_positive
    for key, truths in gt_by_frame.items():
        for index, truth in enumerate(truths):
            if index not in threshold_matched[key]:
                errors.append(
                    {
                        "kind": "false_negative",
                        "class_name": class_name,
                        "event_id": key[0],
                        "role": key[1],
                        "frame_index": key[2],
                        "xyxy": truth["xyxy"],
                    }
                )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "gt_count": gt_count,
        "prediction_count": len(predictions),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "ap": _average_precision(points, gt_count),
        "errors": errors,
    }


def evaluate_yolo_predictions(
    predictions: Iterable[dict[str, Any]],
    ground_truth: dict[str, Any],
    *,
    confidence_threshold: float = 0.25,
    iou_thresholds: tuple[float, ...] = tuple(round(0.5 + 0.05 * index, 2) for index in range(10)),
) -> dict[str, Any]:
    images = {
        str(image["image_id"]): image for image in ground_truth.get("images", [])
    }
    frame_ground_truth: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth.get("annotations", []):
        image = images[str(annotation["image_id"])]
        frame_ground_truth[_frame_key(image)].append(
            {
                "class_name": str(annotation["class_name"]),
                "xyxy": [float(value) for value in annotation["xyxy"]],
            }
        )
    frame_predictions: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in predictions:
        frame_predictions[_frame_key(record)].extend(record.get("detections") or [])

    classes = sorted(
        {
            *[item["class_name"] for values in frame_ground_truth.values() for item in values],
            *[item["class_name"] for values in frame_predictions.values() for item in values],
        }
    )
    per_class: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    for class_name in classes:
        threshold_results = {
            str(iou): _class_threshold_metrics(
                class_name,
                frame_predictions,
                frame_ground_truth,
                iou_threshold=iou,
                confidence_threshold=confidence_threshold,
            )
            for iou in iou_thresholds
        }
        at_50 = threshold_results[str(iou_thresholds[0])]
        ap_values = [value["ap"] for value in threshold_results.values()]
        per_class[class_name] = {
            key: at_50[key]
            for key in (
                "gt_count",
                "prediction_count",
                "true_positive",
                "false_positive",
                "false_negative",
                "precision",
                "recall",
                "f1",
            )
        }
        per_class[class_name]["ap50"] = at_50["ap"]
        per_class[class_name]["ap50_95"] = sum(ap_values) / max(1, len(ap_values))
        all_errors.extend(at_50["errors"])

    supported = [value for value in per_class.values() if value["gt_count"] > 0]
    totals = {
        key: sum(int(value[key]) for value in per_class.values())
        for key in ("true_positive", "false_positive", "false_negative")
    }
    micro_precision = totals["true_positive"] / max(
        1, totals["true_positive"] + totals["false_positive"]
    )
    micro_recall = totals["true_positive"] / max(
        1, totals["true_positive"] + totals["false_negative"]
    )
    return {
        "schema_version": "visioncortex-yolo-ground-truth-evaluation/1.0.0",
        "status": "completed" if images and supported else "not_evaluated_no_ground_truth",
        "confidence_threshold": confidence_threshold,
        "iou_thresholds": list(iou_thresholds),
        "image_count": len(images),
        "class_count": len(classes),
        "supported_class_count": len(supported),
        "micro": {
            **totals,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": 2.0 * micro_precision * micro_recall / max(
                1e-12, micro_precision + micro_recall
            ),
        },
        "macro": {
            "precision": sum(value["precision"] for value in supported) / max(1, len(supported)),
            "recall": sum(value["recall"] for value in supported) / max(1, len(supported)),
            "f1": sum(value["f1"] for value in supported) / max(1, len(supported)),
            "map50": sum(value["ap50"] for value in supported) / max(1, len(supported)),
            "map50_95": sum(value["ap50_95"] for value in supported) / max(1, len(supported)),
        },
        "per_class": per_class,
        "errors": all_errors,
    }


def evaluate_files(
    predictions_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    *,
    confidence_threshold: float = 0.25,
) -> dict[str, Any]:
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    report = evaluate_yolo_predictions(
        load_predictions(predictions_path),
        ground_truth,
        confidence_threshold=confidence_threshold,
    )
    report["provenance"] = {
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "ground_truth_path": str(ground_truth_path),
        "ground_truth_sha256": _sha256(ground_truth_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
