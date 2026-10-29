from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Sequence


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def build_consensus_box_labels(
    observations: Sequence[dict[str, Any]],
    *,
    minimum_model_families: int = 2,
    minimum_iou: float = 0.50,
    minimum_confidence: float = 0.10,
) -> dict[str, Any]:
    """Create auditable pseudo-labels without ever calling them ground truth."""

    if minimum_model_families < 2:
        raise ValueError("Pseudo-label consensus requires at least two model families")
    if not 0 < minimum_iou <= 1:
        raise ValueError("Pseudo-label consensus IoU must be in (0, 1]")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("Pseudo-label confidence must be in [0, 1]")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    invalid_observations = []
    for observation in observations:
        family = str(observation.get("model_family") or "").strip()
        class_name = str(observation.get("class_name") or "").strip()
        context_id = next(
            (
                str(observation.get(name)).strip()
                for name in (
                    "context_id",
                    "sample_id",
                    "image_id",
                    "frame_id",
                    "artifact_id",
                )
                if observation.get(name) is not None
                and str(observation.get(name)).strip()
            ),
            "",
        )
        box = observation.get("xyxy_norm")
        confidence = float(observation.get("confidence") or 0.0)
        if (
            not context_id
            or not family
            or not class_name
            or not isinstance(box, (list, tuple))
            or len(box) != 4
        ):
            invalid_observations.append(
                {**observation, "reason": "missing_or_invalid_consensus_identity"}
            )
            continue
        coordinates = [float(value) for value in box]
        if (
            not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates)
            or coordinates[2] <= coordinates[0]
            or coordinates[3] <= coordinates[1]
            or not math.isfinite(confidence)
            or confidence < minimum_confidence
            or confidence > 1
        ):
            invalid_observations.append(
                {**observation, "reason": "invalid_box_or_confidence"}
            )
            continue
        grouped[(context_id, class_name)].append(
            {
                **observation,
                "context_id": context_id,
                "model_family": family,
                "class_name": class_name,
                "xyxy_norm": coordinates,
                "confidence": confidence,
            }
        )
    accepted = []
    rejected = []
    for (context_id, class_name), candidates in sorted(grouped.items()):
        remaining = list(candidates)
        while remaining:
            seed = max(remaining, key=lambda item: item["confidence"])
            cluster = [
                item
                for item in remaining
                if _iou(seed["xyxy_norm"], item["xyxy_norm"]) >= minimum_iou
            ]
            by_family: dict[str, dict[str, Any]] = {}
            for item in cluster:
                previous = by_family.get(item["model_family"])
                if previous is None or item["confidence"] > previous["confidence"]:
                    by_family[item["model_family"]] = item
            support = [by_family[name] for name in sorted(by_family)]
            families = [item["model_family"] for item in support]
            suppressed = [item for item in cluster if item not in support]
            record = {
                "context_id": context_id,
                "class_name": class_name,
                "model_families": families,
                "support_count": len(support),
                "mean_confidence": round(
                    sum(item["confidence"] for item in support) / len(support), 6
                ),
                "xyxy_norm": [
                    round(
                        sum(item["xyxy_norm"][index] for item in support)
                        / len(support),
                        8,
                    )
                    for index in range(4)
                ],
                "source_observations": support,
                "suppressed_same_family_observations": suppressed,
            }
            if len(families) >= minimum_model_families:
                accepted.append(record)
            else:
                rejected.append({**record, "reason": "insufficient_independent_models"})
            clustered_ids = {id(item) for item in cluster}
            remaining = [item for item in remaining if id(item) not in clustered_ids]
    return {
        "schema_version": "visioncortex-consensus-pseudo-labels/1",
        "status": "completed",
        "truth_status": "pseudo_labels_not_ground_truth",
        "policy": "independent model-family spatial consensus; rejected candidates retained",
        "minimum_model_families": minimum_model_families,
        "minimum_iou": minimum_iou,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected) + len(invalid_observations),
        "accepted": accepted,
        "rejected": rejected,
        "invalid_observations": invalid_observations,
    }
