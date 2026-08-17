from __future__ import annotations

import math
from typing import Any, Sequence

from .schemas import EvidenceEvent


ACTION_TYPE_ALIASES = {
    "hand_object_contact": "hand_object_contact",
    "object_movement": "object_movement",
    "liquid_movement": "liquid_movement",
    "liquid_transfer": "liquid_movement",
    "container_state_change": "container_state_change",
    "device_panel_operation": "device_panel_operation",
    "panel_operation": "device_panel_operation",
}

DEFAULT_OBJECT_ALIASES = {
    "paper": "weighing_paper",
    "weighing_paper": "weighing_paper",
    "称量纸": "weighing_paper",
    "spearhead": "pipette_tip",
    "pipette_tip": "pipette_tip",
    "pipette-tip": "pipette_tip",
    "枪头": "pipette_tip",
    "吸头": "pipette_tip",
    "reagent_bottle_open": "reagent_bottle",
    "reagent_bottle": "reagent_bottle",
    "试剂瓶": "reagent_bottle",
    "sample_bottle": "sample_bottle",
    "tube_rack": "tube_rack",
    "tube-rack": "tube_rack",
    "离心管架": "tube_rack",
    "balance": "balance",
    "analytical_balance": "balance",
    "分析天平": "balance",
}

IGNORED_ACTOR_OBJECTS = {
    "hand",
    "gloved_hand",
    "gloved-hand",
    "手",
    "戴手套的手",
}


def _iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _canonical_action_type(value: Any) -> str:
    raw = str(getattr(value, "value", value)).strip().lower()
    return ACTION_TYPE_ALIASES.get(raw, raw)


def _label_interval_ms(item: dict[str, Any]) -> tuple[float, float]:
    def value(name: str) -> float:
        if item.get(f"{name}_ms") is not None:
            return float(item[f"{name}_ms"])
        if item.get(f"{name}_us") is not None:
            return float(item[f"{name}_us"]) / 1000.0
        return float(item[f"{name}_seconds"]) * 1000.0

    start_ms, end_ms = value("start"), value("end")
    if end_ms <= start_ms:
        raise ValueError(
            f"Reviewed interval must have end > start: {start_ms}..{end_ms}"
        )
    return start_ms, end_ms


def _label_peak_ms(
    item: dict[str, Any], start_ms: float, end_ms: float
) -> float:
    if item.get("peak_timestamp_ms") is not None:
        return float(item["peak_timestamp_ms"])
    if item.get("peak_timestamp_us") is not None:
        return float(item["peak_timestamp_us"]) / 1000.0
    if item.get("peak_timestamp_seconds") is not None:
        return float(item["peak_timestamp_seconds"]) * 1000.0
    return (start_ms + end_ms) / 2.0


def _event_object_values(item: dict[str, Any]) -> list[str]:
    objects = item.get("objects")
    if objects is None:
        return []
    if isinstance(objects, list):
        if any(not isinstance(value, str) for value in objects):
            raise ValueError("objects array must contain only strings")
        return objects
    if isinstance(objects, dict):
        values: list[str] = []
        for value in objects.values():
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                values.extend(value)
            elif value is not None:
                raise ValueError(
                    "objects mapping values must be strings or string arrays"
                )
        return values
    raise ValueError("objects must be a string array or role-to-object mapping")


def _event_review_status(item: dict[str, Any]) -> str:
    decision = item.get("decision")
    decision_status = (
        decision.get("status") if isinstance(decision, dict) else None
    )
    return str(item.get("status") or decision_status or "").strip().lower()


def _canonical_object(value: Any, aliases: dict[str, str]) -> str:
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = aliases.get(raw, raw)
    return str(canonical).strip().lower().replace("-", "_").replace(" ", "_")


def _normalized_object_set(
    values: Sequence[Any], aliases: dict[str, str]
) -> set[str]:
    normalized = {_canonical_object(value, aliases) for value in values}
    ignored = {_canonical_object(value, aliases) for value in IGNORED_ACTOR_OBJECTS}
    return {item for item in normalized - ignored if item}


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start_ms, end_ms in sorted(intervals):
        if not merged or start_ms > merged[-1][1]:
            merged.append([start_ms, end_ms])
        else:
            merged[-1][1] = max(merged[-1][1], end_ms)
    return [(item[0], item[1]) for item in merged]


def _wilson_interval(successes: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "method": "wilson_95_percent",
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _maximum_cardinality_matches(
    labels: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Find a deterministic maximum-cardinality bipartite matching."""

    neighbors: dict[int, list[tuple[int, float]]] = {}
    for label_index, label in enumerate(labels):
        options: list[tuple[int, float]] = []
        for prediction_index, prediction in enumerate(predictions):
            if prediction["action_type"] != label["action_type"]:
                continue
            temporal_iou = _iou(
                (label["start_ms"], label["end_ms"]),
                (prediction["start_ms"], prediction["end_ms"]),
            )
            if temporal_iou < threshold:
                continue
            label_objects = set(label["objects"])
            prediction_objects = set(prediction["objects"])
            if label_objects and not label_objects.intersection(
                prediction_objects
            ):
                continue
            options.append((prediction_index, temporal_iou))
        neighbors[label_index] = sorted(
            options,
            key=lambda item: (
                -item[1],
                predictions[item[0]]["event_id"],
            ),
        )

    prediction_to_label: dict[int, int] = {}

    def augment(label_index: int, seen_predictions: set[int]) -> bool:
        for prediction_index, _ in neighbors[label_index]:
            if prediction_index in seen_predictions:
                continue
            seen_predictions.add(prediction_index)
            previous_label = prediction_to_label.get(prediction_index)
            if previous_label is None or augment(
                previous_label, seen_predictions
            ):
                prediction_to_label[prediction_index] = label_index
                return True
        return False

    label_order = sorted(
        range(len(labels)),
        key=lambda index: (
            len(neighbors[index]),
            labels[index]["event_id"],
        ),
    )
    for label_index in label_order:
        augment(label_index, set())

    matches = []
    for prediction_index, label_index in sorted(
        prediction_to_label.items(),
        key=lambda item: labels[item[1]]["event_id"],
    ):
        label = labels[label_index]
        prediction = predictions[prediction_index]
        temporal_iou = _iou(
            (label["start_ms"], label["end_ms"]),
            (prediction["start_ms"], prediction["end_ms"]),
        )
        matches.append(
            {
                "ground_truth_event_id": label["event_id"],
                "prediction_event_id": prediction["event_id"],
                "action_type": label["action_type"],
                "temporal_iou": round(temporal_iou, 6),
                "object_supported": True,
                "ground_truth_objects": label["objects"],
                "prediction_objects": prediction["objects"],
            }
        )
    return matches


def _metric_summary(
    true_positives: int, false_positives: int, false_negatives: int
) -> dict[str, Any]:
    predicted = true_positives + false_positives
    reviewed = true_positives + false_negatives
    precision = true_positives / predicted if predicted else None
    recall = true_positives / reviewed if reviewed else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
        if precision is not None and recall is not None
        else None
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "precision_confidence_interval": _wilson_interval(
            true_positives, predicted
        ),
        "recall": recall,
        "recall_confidence_interval": _wilson_interval(
            true_positives, reviewed
        ),
        "f1": f1,
    }


def _validate_ground_truth_events(
    labels: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    supported_actions = set(ACTION_TYPE_ALIASES.values())
    event_ids: set[str] = set()
    for index, item in enumerate(labels, 1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Reviewed ground-truth event {index} must be a JSON object"
            )
        event_id = str(
            item.get("event_id") or item.get("label_id") or ""
        ).strip()
        if not event_id:
            raise ValueError(
                f"Reviewed ground-truth event {index} has no event_id"
            )
        if event_id in event_ids:
            raise ValueError(
                f"Reviewed ground truth contains duplicate event_id: {event_id}"
            )
        event_ids.add(event_id)
        action_type = _canonical_action_type(item.get("action_type"))
        if action_type not in supported_actions:
            raise ValueError(
                f"Reviewed event {event_id} has unsupported action_type: "
                f"{item.get('action_type')}"
            )
        try:
            _event_object_values(item)
        except ValueError as error:
            raise ValueError(f"Reviewed event {event_id} {error}") from error
        _label_interval_ms(item)
    return {
        "status": "valid",
        "event_count": len(labels),
        "unique_event_id_count": len(event_ids),
        "supported_canonical_action_types": sorted(supported_actions),
    }


def evaluate_key_event_recall(
    predictions: Sequence[EvidenceEvent],
    ground_truth: dict[str, Any] | None,
    *,
    thresholds: Sequence[float] = (0.30, 0.50, 0.70),
) -> dict[str, Any]:
    """Create an evaluation-only, reproducible key-event recall receipt."""

    if not ground_truth:
        return {
            "schema_version": "visioncortex-key-event-recall-eval/1.1.0",
            "status": "not_evaluated",
            "evaluated": False,
            "reason": "reviewed_key_event_ground_truth_not_configured",
            "prediction_count": len(predictions),
            "thresholds": [float(item) for item in thresholds],
        }

    object_aliases = dict(DEFAULT_OBJECT_ALIASES)
    object_aliases.update(
        {
            _canonical_object(alias, {}): _canonical_object(canonical, {})
            for alias, canonical in (
                ground_truth.get("object_aliases") or {}
            ).items()
        }
    )
    labels = list(ground_truth.get("events") or [])
    ground_truth_validation = _validate_ground_truth_events(labels)
    labeled_windows = _merge_intervals(
        [
            _label_interval_ms(item)
            for item in ground_truth.get("labeled_windows") or []
        ]
    )
    source_duration_ms = ground_truth.get("source_duration_ms")
    if (
        source_duration_ms is None
        and ground_truth.get("source_duration_seconds") is not None
    ):
        source_duration_ms = float(ground_truth["source_duration_seconds"]) * 1000.0
    if (
        source_duration_ms is None
        and ground_truth.get("source_duration_us") is not None
    ):
        source_duration_ms = float(ground_truth["source_duration_us"]) / 1000.0

    def in_coverage(start_ms: float, end_ms: float, peak_ms: float) -> bool:
        if not labeled_windows:
            return True
        return any(
            window_start <= peak_ms <= window_end
            or max(start_ms, window_start) < min(end_ms, window_end)
            for window_start, window_end in labeled_windows
        )

    normalized_labels: list[dict[str, Any]] = []
    excluded_uncertain_labels: list[str] = []
    excluded_rejected_labels: list[str] = []
    excluded_labels_outside_coverage: list[str] = []
    for index, item in enumerate(labels, 1):
        label_id = str(
            item.get("event_id") or item.get("label_id") or f"GT-{index:05d}"
        )
        review_status = _event_review_status(item)
        if bool(item.get("uncertain")) or review_status == "uncertain":
            excluded_uncertain_labels.append(label_id)
            continue
        if review_status == "rejected":
            excluded_rejected_labels.append(label_id)
            continue
        start_ms, end_ms = _label_interval_ms(item)
        peak_ms = _label_peak_ms(item, start_ms, end_ms)
        if not in_coverage(start_ms, end_ms, peak_ms):
            excluded_labels_outside_coverage.append(label_id)
            continue
        normalized_labels.append(
            {
                "event_id": label_id,
                "action_type": _canonical_action_type(item.get("action_type")),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "objects": sorted(
                    _normalized_object_set(
                        _event_object_values(item), object_aliases
                    )
                ),
            }
        )

    normalized_predictions: list[dict[str, Any]] = []
    excluded_prediction_ids: list[str] = []
    for event in predictions:
        if not event.accepted:
            continue
        if not in_coverage(
            event.global_start_ms, event.global_end_ms, event.key_global_ms
        ):
            excluded_prediction_ids.append(event.event_id)
            continue
        normalized_predictions.append(
            {
                "event_id": event.event_id,
                "action_type": _canonical_action_type(event.action_type),
                "start_ms": float(event.global_start_ms),
                "end_ms": float(event.global_end_ms),
                "objects": sorted(
                    _normalized_object_set(event.objects, object_aliases)
                ),
            }
        )

    if not normalized_labels:
        return {
            "schema_version": "visioncortex-key-event-recall-eval/1.1.0",
            "status": "not_evaluated",
            "evaluated": False,
            "reason": "reviewed_ground_truth_contains_no_eligible_events",
            "prediction_count_in_scope": len(normalized_predictions),
            "excluded_uncertain_ground_truth_event_ids": (
                excluded_uncertain_labels
            ),
            "excluded_rejected_ground_truth_event_ids": (
                excluded_rejected_labels
            ),
            "excluded_ground_truth_event_ids_outside_labeled_windows": (
                excluded_labels_outside_coverage
            ),
            "ground_truth_validation": ground_truth_validation,
        }

    action_types = sorted(
        {
            item["action_type"]
            for item in [*normalized_labels, *normalized_predictions]
            if item["action_type"]
        }
    )
    threshold_reports: list[dict[str, Any]] = []
    for threshold in thresholds:
        matches = _maximum_cardinality_matches(
            normalized_labels,
            normalized_predictions,
            float(threshold),
        )
        matched_labels = {
            item["ground_truth_event_id"] for item in matches
        }
        matched_predictions = {
            item["prediction_event_id"] for item in matches
        }
        per_class = []
        for action_type in action_types:
            ground_truth_ids = {
                item["event_id"]
                for item in normalized_labels
                if item["action_type"] == action_type
            }
            prediction_ids = {
                item["event_id"]
                for item in normalized_predictions
                if item["action_type"] == action_type
            }
            true_positives = len(ground_truth_ids & matched_labels)
            metrics = _metric_summary(
                true_positives,
                len(prediction_ids - matched_predictions),
                len(ground_truth_ids - matched_labels),
            )
            per_class.append(
                {
                    "action_type": action_type,
                    **metrics,
                    "small_sample_warning": len(ground_truth_ids) < 10,
                }
            )
        threshold_reports.append(
            {
                "temporal_iou_threshold": float(threshold),
                **_metric_summary(
                    len(matches),
                    len(normalized_predictions) - len(matches),
                    len(normalized_labels) - len(matches),
                ),
                "matches": matches,
                "per_class": per_class,
            }
        )

    labeled_duration_ms = sum(
        max(0.0, end - start) for start, end in labeled_windows
    )
    annotation_coverage_ratio = (
        min(1.0, labeled_duration_ms / float(source_duration_ms))
        if source_duration_ms and labeled_windows
        else 1.0
        if not labeled_windows
        else None
    )
    return {
        "schema_version": "visioncortex-key-event-recall-eval/1.1.0",
        "status": "evaluated",
        "evaluated": True,
        "authority": ground_truth.get("authority"),
        "ground_truth_id": ground_truth.get("ground_truth_id"),
        "ground_truth_validation": ground_truth_validation,
        "matching_policy": {
            "assignment": (
                "maximum_cardinality_one_to_one_within_action_class"
            ),
            "action_aliases": ACTION_TYPE_ALIASES,
            "object_aliases": object_aliases,
            "object_rule": (
                "reviewed object labels require at least one canonical "
                "non-hand object overlap"
            ),
            "uncertain_labels": "excluded",
            "rejected_labels": "excluded",
            "predictions_outside_labeled_windows": "excluded",
            "alignment": (
                "uses reviewed aligned-global timestamps; no offset is "
                "estimated from predictions"
            ),
        },
        "annotation_coverage": {
            "labeled_windows": [
                {"start_ms": start, "end_ms": end}
                for start, end in labeled_windows
            ],
            "labeled_duration_ms": labeled_duration_ms if labeled_windows else None,
            "source_duration_ms": source_duration_ms,
            "coverage_ratio": annotation_coverage_ratio,
        },
        "ground_truth_event_count": len(normalized_labels),
        "prediction_count_in_scope": len(normalized_predictions),
        "small_sample_warning": len(normalized_labels) < 30,
        "excluded_uncertain_ground_truth_event_ids": excluded_uncertain_labels,
        "excluded_rejected_ground_truth_event_ids": excluded_rejected_labels,
        "excluded_ground_truth_event_ids_outside_labeled_windows": (
            excluded_labels_outside_coverage
        ),
        "excluded_prediction_ids_outside_labeled_windows": (
            excluded_prediction_ids
        ),
        "threshold_results": threshold_reports,
    }
