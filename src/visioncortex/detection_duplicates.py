"""Explicit per-role box suppression, with the complete pre-tracking evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .schemas import (
    BoxEvidence,
    DetectionDuplicateSuppression,
    DuplicateBoxRemoval,
    ViewRole,
)


def _threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Duplicate suppression IoU must be a number in (0, 1]")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("Duplicate suppression IoU must be a number in (0, 1]")
    return threshold


def duplicate_suppression_policies(config: Mapping[str, Any]) -> dict[ViewRole, dict]:
    configured = config.get("models", {}).get("duplicate_suppression_iou_by_role", {})
    if not isinstance(configured, Mapping):
        raise ValueError("models.duplicate_suppression_iou_by_role must be a role mapping")
    policies = {}
    for name, value in configured.items():
        try:
            role = ViewRole(name)
        except ValueError as exc:
            raise ValueError(f"Unknown duplicate suppression role: {name}") from exc
        policies[role] = {
            "schema_version": "visioncortex-detection-duplicate-suppression/1",
            "iou_threshold": _threshold(value),
        }
    return policies


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    area = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0.0 else 0.0


def suppress_duplicate_boxes(
    boxes: Sequence[BoxEvidence], iou_threshold: float
) -> tuple[list[BoxEvidence], DetectionDuplicateSuppression]:
    threshold = _threshold(iou_threshold)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy_norm
        if (
            not math.isfinite(box.confidence)
            or not 0.0 <= box.confidence <= 1.0
            or not all(math.isfinite(v) for v in box.xyxy_norm)
            or not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0)
        ):
            raise ValueError("Duplicate suppression requires finite, valid normalized boxes")
    kept: list[int] = []
    removals = []
    for index in sorted(range(len(boxes)), key=lambda i: (-boxes[i].confidence, i)):
        box = boxes[index]
        for retained in kept:
            other = boxes[retained]
            overlap = _iou(box.xyxy_norm, other.xyxy_norm)
            if (
                box.class_id == other.class_id
                and box.class_name == other.class_name
                and overlap >= threshold
            ):
                removals.append(DuplicateBoxRemoval(
                    removed_input_index=index, retained_input_index=retained, iou=overlap
                ))
                break
        else:
            kept.append(index)
    kept.sort()
    audit = DetectionDuplicateSuppression(
        iou_threshold=threshold,
        # Tracking assigns IDs in place; the raw model output must stay unchanged.
        raw_detections=[box.model_copy(deep=True) for box in boxes],
        retained_input_indices=kept,
        removals=removals,
    )
    return [boxes[index] for index in kept], audit
