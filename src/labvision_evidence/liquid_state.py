from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .schemas import (
    LiquidBenchmarkManifest,
    LiquidBenchmarkSample,
    LiquidExpertSample,
    LiquidFlowDirection,
    LiquidStateEvidence,
    LiquidStateStatus,
    LiquidViewObservation,
    NormalizedPoint,
)


class LiquidStateBackendUnavailable(RuntimeError):
    """Raised when an opt-in backend was requested but is not registered."""


class LiquidStateExpert(Protocol):
    """Small adapter surface for an external segmentation/state model."""

    name: str

    def analyze(
        self, samples: Sequence[LiquidExpertSample]
    ) -> dict[str, LiquidStateEvidence]: ...

    def receipt(self) -> dict[str, Any]: ...


class DisabledLiquidStateExpert:
    """Zero-cost default used until a deployment explicitly opts in."""

    name = "disabled"

    def __init__(self) -> None:
        self._requested_events: set[str] = set()

    def analyze(
        self, samples: Sequence[LiquidExpertSample]
    ) -> dict[str, LiquidStateEvidence]:
        requested = {sample.event_id for sample in samples}
        self._requested_events.update(requested)
        return {
            event_id: LiquidStateEvidence(
                status=LiquidStateStatus.NOT_EVALUATED,
                backend=self.name,
                model_receipt={
                    "backend": self.name,
                    "invocations": 0,
                    "input_frames": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "reason": "liquid-state specialist is disabled",
                },
            )
            for event_id in sorted(requested)
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "enabled": False,
            "requested_events": len(self._requested_events),
            "invocations": 0,
            "input_frames": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }


LiquidBackendFactory = Callable[[Mapping[str, Any]], LiquidStateExpert]
_BACKEND_FACTORIES: dict[str, LiquidBackendFactory] = {}


def register_liquid_state_backend(
    name: str, factory: LiquidBackendFactory, *, replace: bool = False
) -> None:
    """Register a deployment-owned adapter without adding a hard ML dependency."""

    normalized = name.strip().lower()
    if not normalized or normalized == "disabled":
        raise ValueError("backend name must be non-empty and cannot be 'disabled'")
    if normalized in _BACKEND_FACTORIES and not replace:
        raise ValueError(f"liquid-state backend is already registered: {normalized}")
    _BACKEND_FACTORIES[normalized] = factory


def build_liquid_state_expert(config: Mapping[str, Any] | None) -> LiquidStateExpert:
    """Build an explicit adapter; disabled remains the safe zero-work default."""

    settings = dict(config or {})
    enabled = bool(settings.get("enabled", False))
    backend = str(settings.get("backend") or "disabled").strip().lower()
    if not enabled or backend == "disabled":
        return DisabledLiquidStateExpert()
    factory = _BACKEND_FACTORIES.get(backend)
    if factory is None:
        raise LiquidStateBackendUnavailable(
            f"Liquid-state backend '{backend}' is enabled but not registered. "
            "Install and register a deployment adapter before starting a production run."
        )
    return factory(settings)


def _stable_split(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _artifact_paths(payload: Mapping[str, Any], field: str) -> list[str]:
    result = []
    for item in payload.get(field) or []:
        path = item.get("path") if isinstance(item, Mapping) else item
        if path:
            result.append(str(path))
    return list(dict.fromkeys(result))


def build_liquid_benchmark_manifest(
    events: Sequence[Mapping[str, Any]],
    *,
    benchmark_id: str,
    source_archive: str,
    annotations: Mapping[str, LiquidStateEvidence | Mapping[str, Any]] | None = None,
    split_by_group: Mapping[str, str] | None = None,
    include_action_types: set[str] | None = None,
) -> LiquidBenchmarkManifest:
    """Build a deterministic, references-only golden-set manifest.

    Splits are assigned by experiment group, not by frame, so adjacent material
    from the same physical procedure cannot leak across train/evaluation sets.
    """

    annotations = annotations or {}
    split_by_group = split_by_group or {}
    samples: list[LiquidBenchmarkSample] = []
    for payload in sorted(events, key=lambda item: str(item.get("event_id") or "")):
        event_id = str(payload.get("event_id") or "").strip()
        group_id = str(payload.get("parent_event_id") or "unassigned").strip()
        action_type = str(payload.get("action_type") or "")
        if not event_id or (include_action_types and action_type not in include_action_types):
            continue
        split = str(split_by_group.get(group_id) or _stable_split(group_id))
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported benchmark split for {group_id}: {split}")
        raw_truth = annotations.get(event_id)
        ground_truth = (
            raw_truth
            if isinstance(raw_truth, LiquidStateEvidence)
            else LiquidStateEvidence.model_validate(raw_truth)
            if raw_truth is not None
            else None
        )
        identity = f"{source_archive}|{group_id}|{event_id}"
        sample_id = f"liquid-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
        samples.append(
            LiquidBenchmarkSample(
                sample_id=sample_id,
                event_id=event_id,
                group_id=group_id,
                split=split,
                image_refs=_artifact_paths(payload, "key_frames"),
                clip_refs=_artifact_paths(payload, "key_clips"),
                ground_truth=ground_truth,
                tags=list(
                    dict.fromkeys(
                        [
                            action_type,
                            str(payload.get("action_subtype") or ""),
                            "annotated" if ground_truth else "unannotated",
                        ]
                    )
                ),
                metadata={
                    "peak_timestamp_us": int(payload.get("peak_timestamp_us") or 0),
                    "decision_status": str(
                        (payload.get("decision") or {}).get("status") or ""
                    ),
                },
            )
        )
    return LiquidBenchmarkManifest(
        benchmark_id=benchmark_id,
        source_archive=source_archive,
        samples=samples,
    )


def write_liquid_benchmark_manifest(
    path: Path, manifest: LiquidBenchmarkManifest
) -> None:
    """Atomically persist a manifest without copying referenced media."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _boolean_metrics(
    expected: Mapping[str, bool | None], predicted: Mapping[str, bool | None]
) -> dict[str, Any]:
    tp = fp = fn = tn = covered = 0
    for key, truth in expected.items():
        if truth is None:
            continue
        prediction = predicted.get(key)
        if prediction is not None:
            covered += 1
        if truth is True and prediction is True:
            tp += 1
        elif truth is True:
            fn += 1
        elif prediction is True:
            fp += 1
        elif prediction is False:
            tn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "eligible": sum(value is not None for value in expected.values()),
        "covered": covered,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _observation_key(event_id: str, item: LiquidViewObservation) -> str:
    return f"{event_id}|{item.view_id}|{item.timestamp_us}|{item.container_id or ''}"


def _state_values(evidence: LiquidStateEvidence) -> dict[str, float | None]:
    return {
        name: state.fill_ratio if state is not None else None
        for name, state in {
            "source_before": evidence.source_before,
            "source_after": evidence.source_after,
            "target_before": evidence.target_before,
            "target_after": evidence.target_after,
        }.items()
    }


def _resample_polyline(points: Sequence[NormalizedPoint], count: int = 32) -> list[tuple[float, float]]:
    if not points:
        return []
    raw = [(item.x, item.y) for item in points]
    if len(raw) == 1:
        return [raw[0]] * count
    lengths = [0.0]
    for first, second in zip(raw, raw[1:]):
        lengths.append(lengths[-1] + math.dist(first, second))
    if lengths[-1] == 0.0:
        return [raw[0]] * count
    result: list[tuple[float, float]] = []
    segment = 0
    for ordinal in range(count):
        target = lengths[-1] * ordinal / max(1, count - 1)
        while segment + 1 < len(lengths) and lengths[segment + 1] < target:
            segment += 1
        if segment + 1 >= len(raw):
            result.append(raw[-1])
            continue
        span = lengths[segment + 1] - lengths[segment]
        fraction = (target - lengths[segment]) / span if span else 0.0
        x = raw[segment][0] + (raw[segment + 1][0] - raw[segment][0]) * fraction
        y = raw[segment][1] + (raw[segment + 1][1] - raw[segment][1]) * fraction
        result.append((x, y))
    return result


def evaluate_liquid_state_predictions(
    ground_truth: Mapping[str, LiquidStateEvidence],
    predictions: Mapping[str, LiquidStateEvidence],
) -> dict[str, Any]:
    """Evaluate state, flow, fill-level and meniscus evidence without model calls."""

    truth_observations = {
        _observation_key(event_id, item): item
        for event_id, evidence in ground_truth.items()
        for item in evidence.per_view_observations
    }
    predicted_observations = {
        _observation_key(event_id, item): item
        for event_id, evidence in predictions.items()
        for item in evidence.per_view_observations
    }
    presence = _boolean_metrics(
        {key: item.liquid_present for key, item in truth_observations.items()},
        {
            key: item.liquid_present
            for key, item in predicted_observations.items()
        },
    )
    visible_flow = _boolean_metrics(
        {
            **{event_id: item.visible_flow for event_id, item in ground_truth.items()},
            **{
                key: item.visible_flow
                for key, item in truth_observations.items()
            },
        },
        {
            **{event_id: item.visible_flow for event_id, item in predictions.items()},
            **{
                key: item.visible_flow
                for key, item in predicted_observations.items()
            },
        },
    )

    fill_errors: list[float] = []
    for event_id, truth in ground_truth.items():
        predicted = predictions.get(event_id)
        if predicted is None:
            continue
        expected_states = _state_values(truth)
        predicted_states = _state_values(predicted)
        for key, value in expected_states.items():
            other = predicted_states.get(key)
            if value is not None and other is not None:
                fill_errors.append(abs(value - other))
    for key, truth in truth_observations.items():
        predicted = predicted_observations.get(key)
        if truth.fill_ratio is not None and predicted and predicted.fill_ratio is not None:
            fill_errors.append(abs(truth.fill_ratio - predicted.fill_ratio))

    meniscus_distances: list[float] = []
    for key, truth in truth_observations.items():
        predicted = predicted_observations.get(key)
        if not predicted or not truth.meniscus_polyline or not predicted.meniscus_polyline:
            continue
        expected_line = _resample_polyline(truth.meniscus_polyline)
        predicted_line = _resample_polyline(predicted.meniscus_polyline)
        meniscus_distances.append(
            sum(math.dist(first, second) for first, second in zip(expected_line, predicted_line))
            / len(expected_line)
        )

    event_ids = sorted(ground_truth)
    status_correct = sum(
        predictions.get(event_id) is not None
        and predictions[event_id].status == ground_truth[event_id].status
        for event_id in event_ids
    )
    direction_eligible = [
        event_id
        for event_id in event_ids
        if ground_truth[event_id].flow_direction != LiquidFlowDirection.UNKNOWN
    ]
    direction_correct = sum(
        predictions.get(event_id) is not None
        and predictions[event_id].flow_direction == ground_truth[event_id].flow_direction
        for event_id in direction_eligible
    )
    change_eligible = [
        event_id
        for event_id in event_ids
        if ground_truth[event_id].state_change_confirmed is not None
    ]
    change_correct = sum(
        predictions.get(event_id) is not None
        and predictions[event_id].state_change_confirmed
        == ground_truth[event_id].state_change_confirmed
        for event_id in change_eligible
    )
    uncertainty_eligible = [
        event_id
        for event_id in event_ids
        if ground_truth[event_id].status
        in {LiquidStateStatus.UNCERTAIN, LiquidStateStatus.OCCLUDED}
    ]
    false_confirmations = sum(
        predictions.get(event_id) is not None
        and predictions[event_id].status == LiquidStateStatus.OBSERVED
        for event_id in uncertainty_eligible
    )
    return {
        "schema_version": "visioncortex-liquid-evaluation/1.0.0",
        "events": {
            "ground_truth": len(ground_truth),
            "predicted": len(predictions),
            "covered": sum(event_id in predictions for event_id in event_ids),
        },
        "status_accuracy": _safe_div(status_correct, len(event_ids)),
        "liquid_presence": presence,
        "visible_flow": visible_flow,
        "flow_direction": {
            "eligible": len(direction_eligible),
            "correct": direction_correct,
            "accuracy": _safe_div(direction_correct, len(direction_eligible)),
        },
        "state_change": {
            "eligible": len(change_eligible),
            "correct": change_correct,
            "accuracy": _safe_div(change_correct, len(change_eligible)),
        },
        "fill_ratio": {
            "paired_observations": len(fill_errors),
            "mae": sum(fill_errors) / len(fill_errors) if fill_errors else None,
        },
        "meniscus": {
            "paired_polylines": len(meniscus_distances),
            "mean_normalized_distance": (
                sum(meniscus_distances) / len(meniscus_distances)
                if meniscus_distances
                else None
            ),
        },
        "uncertainty_guard": {
            "eligible_uncertain_or_occluded": len(uncertainty_eligible),
            "false_observed_confirmations": false_confirmations,
            "false_confirmation_rate": _safe_div(
                false_confirmations, len(uncertainty_eligible)
            ),
        },
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reason": "deterministic local evaluator; no model call",
        },
    }
