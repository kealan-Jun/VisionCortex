from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .schemas import ActionType, EvidenceEvent, ExperimentGroup, ViewInput


def _iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _optimal_interval_assignment(
    expected: Sequence[dict[str, Any]], predicted: Sequence[ExperimentGroup]
) -> list[tuple[int, int, float]]:
    """Return the globally best one-to-one expected/predicted interval assignment."""

    if not expected or not predicted:
        return []
    scores = [
        [
            _iou(
                (float(item["start_ms"]), float(item["end_ms"])),
                (group.global_start_ms, group.global_end_ms),
            )
            for group in predicted
        ]
        for item in expected
    ]
    if len(predicted) > 20:
        # This pipeline normally yields only a handful of bounded experiments.
        # Keep pathological inputs bounded while preserving one-to-one matching.
        remaining = set(range(len(predicted)))
        matches = []
        for expected_index, row in enumerate(scores):
            if not remaining:
                break
            predicted_index = max(remaining, key=lambda index: row[index])
            remaining.remove(predicted_index)
            matches.append((expected_index, predicted_index, row[predicted_index]))
        return matches

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(expected_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int, float], ...]]:
        if expected_index >= len(expected):
            return 0.0, ()
        best_score, best_matches = solve(expected_index + 1, used_mask)
        for predicted_index in range(len(predicted)):
            if used_mask & (1 << predicted_index):
                continue
            tail_score, tail_matches = solve(
                expected_index + 1, used_mask | (1 << predicted_index)
            )
            candidate_score = scores[expected_index][predicted_index] + tail_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_matches = (
                    (expected_index, predicted_index, scores[expected_index][predicted_index]),
                    *tail_matches,
                )
        return best_score, best_matches

    return list(solve(0, 0)[1])


def validate_experiment_and_material_quality(
    groups: Sequence[ExperimentGroup],
    key_events: Sequence[EvidenceEvent],
    baseline: dict[str, Any] | None,
    *,
    boundary_match_iou: float = 0.50,
    max_start_error_seconds: float = 8.0,
    max_end_error_seconds: float = 8.0,
    minimum_cross_view_event_rate: float = 0.25,
) -> dict[str, Any]:
    """Evaluate bounded experiments and key material without leaking labels into inference.

    ``baseline`` is optional and must be a separately reviewed artifact. When it is
    absent, structural/material checks still run and boundary quality remains
    explicitly unevaluated instead of being presented as ground truth.
    """

    expected = list((baseline or {}).get("experiments") or [])

    def boundary_ms(item: dict[str, Any], name: str) -> float:
        milliseconds = item.get(f"{name}_ms")
        if milliseconds is not None:
            return float(milliseconds)
        return float(item[f"{name}_seconds"]) * 1000.0

    normalized_expected = [
        {
            "baseline_id": str(item.get("baseline_id") or f"baseline-{index + 1:03d}"),
            "name": item.get("name"),
            "start_ms": boundary_ms(item, "start"),
            "end_ms": boundary_ms(item, "end"),
            "continuity_type": str(item.get("continuity_type") or "independent"),
            "atomic_experiment_count": int(
                item.get("atomic_experiment_count")
                or len(item.get("atomic_experiment_ids") or [])
                or 1
            ),
        }
        for index, item in enumerate(expected)
    ]
    assignments = _optimal_interval_assignment(normalized_expected, groups)
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    matches: list[dict[str, Any]] = []
    for expected_index, predicted_index, interval_iou in assignments:
        if interval_iou < boundary_match_iou:
            continue
        expected_item = normalized_expected[expected_index]
        predicted_item = groups[predicted_index]
        start_error_seconds = (
            predicted_item.global_start_ms - expected_item["start_ms"]
        ) / 1000.0
        end_error_seconds = (
            predicted_item.global_end_ms - expected_item["end_ms"]
        ) / 1000.0
        continuity_correct = predicted_item.continuity_type == expected_item["continuity_type"]
        atomic_count_correct = (
            len(predicted_item.atomic_experiment_ids)
            == expected_item["atomic_experiment_count"]
        )
        boundary_within_tolerance = (
            abs(start_error_seconds) <= max_start_error_seconds
            and abs(end_error_seconds) <= max_end_error_seconds
        )
        matched_expected.add(expected_index)
        matched_predicted.add(predicted_index)
        matches.append(
            {
                "baseline_id": expected_item["baseline_id"],
                "expected_name": expected_item["name"],
                "predicted_group_id": predicted_item.group_id,
                "predicted_name": predicted_item.experiment_name,
                "interval_iou": round(interval_iou, 6),
                "start_error_seconds": round(start_error_seconds, 6),
                "end_error_seconds": round(end_error_seconds, 6),
                "boundary_within_tolerance": boundary_within_tolerance,
                "expected_continuity_type": expected_item["continuity_type"],
                "predicted_continuity_type": predicted_item.continuity_type,
                "continuity_correct": continuity_correct,
                "expected_atomic_experiment_count": expected_item["atomic_experiment_count"],
                "predicted_atomic_experiment_count": len(predicted_item.atomic_experiment_ids),
                "atomic_count_correct": atomic_count_correct,
                "passed": boundary_within_tolerance and continuity_correct and atomic_count_correct,
            }
        )

    false_negatives = [
        item["baseline_id"]
        for index, item in enumerate(normalized_expected)
        if index not in matched_expected
    ]
    false_positives = [
        group.group_id for index, group in enumerate(groups) if index not in matched_predicted
    ]
    true_positive_count = len(matches)
    precision = true_positive_count / len(groups) if groups else (1.0 if not expected else 0.0)
    recall = (
        true_positive_count / len(normalized_expected) if normalized_expected else None
    )
    boundary_pass_rate = (
        sum(bool(item["boundary_within_tolerance"]) for item in matches) / len(matches)
        if matches
        else None
    )
    continuity_accuracy = (
        sum(bool(item["continuity_correct"] and item["atomic_count_correct"]) for item in matches)
        / len(matches)
        if matches
        else None
    )

    action_counts = {action.value: 0 for action in ActionType}
    cross_view_count = 0
    confirmed_count = 0
    model_completed_count = 0
    media_complete_count = 0
    auditable_count = 0
    per_event: list[dict[str, Any]] = []
    for event in key_events:
        action_counts[event.action_type.value] += 1
        both_roles = {role.value for role in event.supporting_roles} >= {
            "first_person",
            "third_person",
        }
        cross_view_count += int(both_roles)
        understanding = event.model_understanding or {}
        model_completed = understanding.get("status") == "completed"
        model_completed_count += int(model_completed)
        confirmed = event.accepted and event.confidence >= 0.5
        confirmed_count += int(confirmed)
        media_complete = len(event.key_frames) == 3 and len(event.key_clips) == 3
        media_complete_count += int(media_complete)
        auditable = both_roles or bool(event.uncertainty)
        auditable_count += int(auditable)
        per_event.append(
            {
                "event_id": event.event_id,
                "action_type": event.action_type.value,
                "accepted": event.accepted,
                "confidence": event.confidence,
                "cross_view_supported": both_roles,
                "media_complete": media_complete,
                "model_understanding_completed": model_completed,
                "cross_view_or_explicit_uncertainty": auditable,
                "uncertainty": event.uncertainty,
            }
        )
    event_count = len(key_events)
    cross_view_rate = cross_view_count / event_count if event_count else 0.0
    missing_action_types = [name for name, count in action_counts.items() if count == 0]
    materials_passed = bool(event_count) and not missing_action_types and all(
        (
            media_complete_count == event_count,
            model_completed_count == event_count,
            auditable_count == event_count,
            cross_view_rate >= minimum_cross_view_event_rate,
        )
    )
    boundary_evaluated = bool(normalized_expected)
    boundary_passed = bool(boundary_evaluated) and all(
        (
            not false_negatives,
            not false_positives,
            len(matches) == len(normalized_expected),
            all(item["passed"] for item in matches),
        )
    )
    return {
        "schema_version": "visioncortex-quality-acceptance/1",
        "status": (
            "passed"
            if boundary_evaluated and boundary_passed and materials_passed
            else "failed"
            if boundary_evaluated
            else "structural_only"
        ),
        "baseline": {
            "available": boundary_evaluated,
            "authority": (baseline or {}).get("authority"),
            "baseline_id": (baseline or {}).get("baseline_id"),
            "usage": "evaluation_only_not_inference" if boundary_evaluated else None,
            "expected_experiment_count": len(normalized_expected) if boundary_evaluated else None,
        },
        "thresholds": {
            "boundary_match_iou": boundary_match_iou,
            "max_start_error_seconds": max_start_error_seconds,
            "max_end_error_seconds": max_end_error_seconds,
            "minimum_cross_view_event_rate": minimum_cross_view_event_rate,
        },
        "experiment_boundaries": {
            "evaluated": boundary_evaluated,
            "passed": boundary_passed if boundary_evaluated else None,
            "predicted_experiment_count": len(groups),
            "precision": precision if boundary_evaluated else None,
            "recall": recall,
            "boundary_pass_rate": boundary_pass_rate,
            "continuity_accuracy": continuity_accuracy,
            "false_negative_baseline_ids": false_negatives,
            "false_positive_group_ids": false_positives,
            "matches": matches,
        },
        "key_materials": {
            "passed": materials_passed,
            "event_count": event_count,
            "action_counts": action_counts,
            "missing_action_types": missing_action_types,
            "confirmed_count": confirmed_count,
            "media_complete_count": media_complete_count,
            "model_understanding_completed_count": model_completed_count,
            "cross_view_or_explicit_uncertainty_count": auditable_count,
            "cross_view_supported_count": cross_view_count,
            "cross_view_supported_rate": cross_view_rate,
            "events": per_event,
        },
        "passed": boundary_passed and materials_passed if boundary_evaluated else materials_passed,
    }


def validate_against_sidecars(
    views: Sequence[ViewInput], groups: Sequence[ExperimentGroup], sidecar_name: str
) -> dict[str, Any] | None:
    annotations: dict[str, dict[str, Any]] = {}
    for view in views:
        source = view.video if view.video is not None else view.segments[0].video
        sidecar = source.parent / sidecar_name
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
