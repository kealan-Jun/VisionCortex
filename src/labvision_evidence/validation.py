from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .schemas import ExperimentGroup, ViewInput


def _iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def validate_against_sidecars(
    views: Sequence[ViewInput], groups: Sequence[ExperimentGroup], sidecar_name: str
) -> dict[str, Any] | None:
    annotations: dict[str, dict[str, Any]] = {}
    for view in views:
        sidecar = view.video.parent / sidecar_name
        if sidecar.is_file():
            annotations[view.view_id] = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    if not annotations:
        return None
    predicted_views = {view_id for group in groups for view_id in group.participating_views}
    gt_views = {
        view_id
        for view_id, payload in annotations.items()
        if payload.get("evidence_status") == "valid_experiment_evidence" and payload.get("event")
    }
    true_positive = predicted_views & gt_views
    precision = len(true_positive) / len(predicted_views) if predicted_views else 0.0
    recall = len(true_positive) / len(gt_views) if gt_views else 1.0
    per_view: dict[str, Any] = {}
    all_ious: list[float] = []
    for view in views:
        payload = annotations.get(view.view_id, {})
        ground_truth = [
            (float(item["start_seconds"]) * 1000.0, float(item["end_seconds"]) * 1000.0)
            for item in payload.get("event", [])
        ]
        predicted = [
            (group.global_start_ms, group.global_end_ms)
            for group in groups
            if view.view_id in group.participating_views
        ]
        matches = []
        for gt_index, gt in enumerate(ground_truth):
            scores = [_iou(gt, item) for item in predicted]
            best = max(scores, default=0.0)
            matches.append({"ground_truth_index": gt_index, "best_interval_iou": best})
            all_ious.append(best)
        per_view[view.view_id] = {
            "ground_truth_valid": view.view_id in gt_views,
            "predicted_valid": view.view_id in predicted_views,
            "ground_truth_interval_count": len(ground_truth),
            "predicted_interval_count": len(predicted),
            "matches": matches,
        }
    return {
        "annotation_usage": "evaluation_only_not_inference",
        "ground_truth_valid_views": sorted(gt_views),
        "predicted_valid_views": sorted(predicted_views),
        "view_selection_precision": precision,
        "view_selection_recall": recall,
        "mean_best_boundary_iou": sum(all_ious) / len(all_ious) if all_ious else None,
        "per_view": per_view,
    }
