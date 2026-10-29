"""Candidate accumulation with streaming and historical compatibility modes."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..candidate_index import CoarseFrameIndex, FineFrameIndex
from ..detection import iter_frame_evidence
from ..ordering import candidate_sort_key
from ..schemas import ActionCandidate, ActionType, FrameEvidence, ViewInput
from .observations import (
    HAND_CLASSES,
    NON_ACTION_CLASSES,
    _box_distance,
    _frame_observations,
    _Observation,
    _observation_evidence,
    _observation_instance_signature,
    _observation_key,
    _observation_state_uncertainty,
    _representative_observation_evidence,
)


def _merge_observations(
    observations: Sequence[_Observation],
    view: ViewInput,
    cfg: dict[str, Any],
) -> list[ActionCandidate]:
    instance_aware = bool(cfg.get("fine_instance_association_enabled", False))
    grouped: dict[tuple[str, str, str], list[_Observation]] = defaultdict(list)
    for observation in observations:
        grouped[_observation_key(observation, instance_aware=instance_aware)].append(
            observation
        )
    candidates: list[ActionCandidate] = []
    merge_gap_ms = float(cfg["event_merge_gap_seconds"]) * 1000.0
    min_duration_ms = float(cfg["min_event_duration_seconds"]) * 1000.0
    minimum_observations = int(cfg["min_event_observations"])
    counter = 1
    for items in grouped.values():
        items.sort(key=lambda item: item.global_ms)
        runs: list[list[_Observation]] = []
        current: list[_Observation] = []
        for item in items:
            if current and item.global_ms - current[-1].global_ms > merge_gap_ms:
                runs.append(current)
                current = []
            current.append(item)
        if current:
            runs.append(current)
        for run in runs:
            duration = run[-1].global_ms - run[0].global_ms
            action_type = run[0].action_type
            enough = len(run) >= minimum_observations and duration >= min_duration_ms
            if action_type in {
                ActionType.CONTAINER_STATE_CHANGE,
                ActionType.DEVICE_PANEL_OPERATION,
            }:
                enough = len(run) >= max(2, minimum_observations - 1)
            if not enough:
                continue
            object_names = sorted(
                {obj for observation in run for obj in observation.objects}
            )
            raw_confidence = sum(item.confidence for item in run) / len(run)
            persistence = min(0.15, math.log1p(len(run)) * 0.035)
            confidence = min(1.0, raw_confidence + persistence)
            key_item = max(run, key=lambda item: item.confidence)
            evidence = _representative_observation_evidence(run, key_item)
            instance_signature = _observation_instance_signature(run)
            uncertainty = _observation_state_uncertainty(run, cfg)
            candidates.append(
                ActionCandidate(
                    candidate_id=f"CAND-{view.view_id}-{counter:06d}",
                    action_type=action_type,
                    view_id=view.view_id,
                    role=view.role,
                    local_start_ms=run[0].local_ms,
                    local_end_ms=run[-1].local_ms,
                    global_start_ms=run[0].global_ms,
                    global_end_ms=run[-1].global_ms,
                    key_global_ms=key_item.global_ms,
                    objects=object_names,
                    confidence=confidence,
                    evidence=evidence,
                    uncertainty=uncertainty,
                    instance_signature=instance_signature,
                    provenance={
                        "source_stage": "candidate_fine",
                        "candidate_reducer": "batch",
                        "instance_association": bool(instance_signature),
                    },
                )
            )
            counter += 1
    return sorted(candidates, key=candidate_sort_key)


def _infer_liquid_transfer_sequences(
    observations: Sequence[_Observation],
    view: ViewInput,
    cfg: dict[str, Any],
) -> list[ActionCandidate]:
    """Use the same exclusive-contact reducer for batch and streaming paths."""
    reducer = _StreamingLiquidSequences(view, cfg)
    for timestamp, items in itertools.groupby(
        sorted(observations, key=lambda item: item.global_ms),
        key=lambda item: item.global_ms,
    ):
        reducer.expire(timestamp)
        reducer.add_frame(list(items))
    return reducer.finish()


def generate_candidates(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    frame_index: FineFrameIndex | None = None,
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    cfg = dict(config["segmentation"])
    for key in (
        "fine_streaming_candidates_enabled",
        "fine_state_max_entries",
        "fine_instance_association_enabled",
        "fine_contact_state_enabled",
        "fine_affine_object_motion_enabled",
        "motion_probe_max_scale_delta",
        "motion_probe_max_rotation_degrees",
    ):
        if key in config["performance"]:
            cfg[key] = config["performance"][key]
    for view in views:
        frames = (
            frame_index.iter_frames(view.view_id)
            if frame_index is not None
            else iter_frame_evidence(detection_paths[view.view_id])
        )
        if cfg.get("fine_streaming_candidates_enabled", False):
            candidates.extend(_generate_candidates_streaming(view, frames, cfg))
            continue
        observations: list[_Observation] = []
        previous_tracks: dict[int, tuple[float, float, float]] = {}
        movement_history: dict[int, deque[tuple[float, float, float]]] = {}
        interaction_state: dict[tuple[Any, ...], dict[str, Any]] = {}
        container_states: dict[str, str] = {}
        for frame in frames:
            observations.extend(
                _frame_observations(
                    frame,
                    previous_tracks,
                    cfg,
                    interaction_state,
                    container_states,
                    movement_history,
                )
            )
        candidates.extend(_merge_observations(observations, view, cfg))
        candidates.extend(_infer_liquid_transfer_sequences(observations, view, cfg))
    return sorted(candidates, key=candidate_sort_key)


def generate_coarse_activity_candidates(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    frame_index: CoarseFrameIndex | None = None,
) -> list[ActionCandidate]:
    """Recall-first activity windows; the fine layer must verify contact and action type."""
    cfg = config["segmentation"]
    perf = config["performance"]
    spatial_relation = bool(perf.get("coarse_spatial_relation_enabled", False))
    relation_gap = float(
        perf.get("coarse_hand_object_gap_norm", cfg["contact_distance_norm"])
    )
    relaxed_relation_gap = float(
        perf.get("coarse_hand_object_approach_gap_norm", relation_gap * 2.0)
    )
    approach_delta = float(perf.get("coarse_hand_object_approach_delta_norm", 0.025))
    micro_action_guard = bool(perf.get("coarse_micro_action_guard_enabled", False))
    micro_confidence = float(perf.get("coarse_micro_action_min_confidence", 0.72))
    micro_half_window_ms = (
        float(perf.get("coarse_micro_action_window_seconds", 4.0)) * 500.0
    )
    rolling_motion = bool(perf.get("coarse_rolling_motion_threshold_enabled", False))
    rolling_window_ms = (
        float(perf.get("coarse_rolling_motion_window_seconds", 600.0)) * 1000.0
    )
    observations_by_view: dict[str, list[_Observation]] = defaultdict(list)
    micro_observations_by_view: dict[str, list[_Observation]] = defaultdict(list)
    by_id = {view.view_id: view for view in views}
    for view in views:
        previous_pair_gaps: dict[tuple[Any, ...], float] = {}
        current_motion_bucket: int | None = None
        current_motion_scores: list[float] = []
        previous_motion_threshold = 10.0
        indexed_motion_thresholds: dict[int, float] = {}
        if rolling_motion and frame_index is not None:
            indexed_scores: dict[int, list[float]] = defaultdict(list)
            for global_ms, motion_score in frame_index.iter_motion_samples(
                view.view_id
            ):
                indexed_scores[
                    max(0, int(global_ms // max(rolling_window_ms, 1.0)))
                ].append(motion_score)
            for bucket, values in indexed_scores.items():
                scores = np.asarray(values, dtype=np.float64)
                indexed_motion_thresholds[bucket] = max(
                    float(np.percentile(scores, 80.0)),
                    float(np.median(scores) + 2.5),
                )
        source_frames = (
            frame_index.iter_frames(view.view_id, activity_only=True)
            if frame_index is not None
            else iter_frame_evidence(detection_paths[view.view_id])
        )
        for frame in source_frames:
            if frame.global_ms is None:
                continue
            bucket = max(0, int(frame.global_ms // max(rolling_window_ms, 1.0)))
            if indexed_motion_thresholds:
                previous_motion_threshold = indexed_motion_thresholds.get(bucket, 10.0)
            elif current_motion_bucket is None:
                current_motion_bucket = bucket
            elif bucket != current_motion_bucket:
                if current_motion_scores:
                    scores = np.asarray(current_motion_scores, dtype=np.float64)
                    previous_motion_threshold = max(
                        float(np.percentile(scores, 80.0)),
                        float(np.median(scores) + 2.5),
                    )
                current_motion_scores = []
                current_motion_bucket = bucket
            if not indexed_motion_thresholds:
                current_motion_scores.append(frame.motion_score)
            hands = [box for box in frame.detections if box.class_name in HAND_CLASSES]
            objects = [
                box
                for box in frame.detections
                if box.class_name not in HAND_CLASSES | NON_ACTION_CLASSES | {"paper"}
            ]
            if hands and objects:
                legacy_hand = max(hands, key=lambda box: box.confidence)
                legacy_object = max(objects, key=lambda box: box.confidence)
                if spatial_relation:
                    best_hand, best_object = min(
                        itertools.product(hands, objects),
                        key=lambda pair: (
                            _box_distance(pair[0], pair[1]),
                            -min(pair[0].confidence, pair[1].confidence),
                        ),
                    )
                else:
                    best_hand = legacy_hand
                    best_object = legacy_object
                gap = _box_distance(best_hand, best_object)
                pair_key = (
                    best_hand.track_id
                    if best_hand.track_id is not None
                    else best_hand.class_name,
                    best_object.track_id
                    if best_object.track_id is not None
                    else best_object.class_name,
                )
                previous_gap = previous_pair_gaps.get(pair_key)
                approaching = bool(
                    previous_gap is not None
                    and previous_gap - gap >= approach_delta
                    and gap <= relaxed_relation_gap
                )
                previous_pair_gaps[pair_key] = gap
                relation_confirmed = bool(
                    not spatial_relation or gap <= relation_gap or approaching
                )
                confidence = min(best_hand.confidence, best_object.confidence)
                if spatial_relation and not relation_confirmed:
                    confidence *= 0.55
                observation = _Observation(
                    action_type=ActionType.HAND_OBJECT_CONTACT,
                    local_ms=frame.local_ms,
                    global_ms=frame.global_ms,
                    objects=tuple(
                        sorted({best_hand.class_name, best_object.class_name})
                    ),
                    confidence=confidence,
                    evidence={
                        "frame_index": frame.frame_index,
                        "coarse_activity": (
                            "hand_object_spatial_relation"
                            if relation_confirmed and spatial_relation
                            else "hand_and_lab_object_cooccurrence"
                        ),
                        "hand_object_gap_norm": round(gap, 6),
                        "approaching": approaching,
                        "relation_confirmed": relation_confirmed,
                        "legacy_cooccurrence_retained": bool(
                            spatial_relation and not relation_confirmed
                        ),
                        "hand_track_id": best_hand.track_id,
                        "object_track_id": best_object.track_id,
                    },
                )
                observations_by_view[view.view_id].append(observation)
                if spatial_relation and (
                    legacy_hand is not best_hand or legacy_object is not best_object
                ):
                    observations_by_view[view.view_id].append(
                        _Observation(
                            action_type=ActionType.HAND_OBJECT_CONTACT,
                            local_ms=frame.local_ms,
                            global_ms=frame.global_ms,
                            objects=tuple(
                                sorted(
                                    {
                                        legacy_hand.class_name,
                                        legacy_object.class_name,
                                    }
                                )
                            ),
                            confidence=(
                                min(
                                    legacy_hand.confidence,
                                    legacy_object.confidence,
                                )
                                * 0.55
                            ),
                            evidence={
                                "frame_index": frame.frame_index,
                                "coarse_activity": ("hand_and_lab_object_cooccurrence"),
                                "legacy_cooccurrence_retained": True,
                                "relation_confirmed": False,
                                "hand_object_gap_norm": round(
                                    _box_distance(legacy_hand, legacy_object),
                                    6,
                                ),
                                "hand_track_id": legacy_hand.track_id,
                                "object_track_id": legacy_object.track_id,
                            },
                        )
                    )
                if (
                    micro_action_guard
                    and relation_confirmed
                    and observation.confidence >= micro_confidence
                ):
                    micro_observations_by_view[view.view_id].append(observation)
            motion_threshold = (
                min(10.0, previous_motion_threshold) if rolling_motion else 10.0
            )
            if (
                not hands
                and frame.motion_score >= motion_threshold
                and len(objects) >= 2
            ):
                selected = sorted(
                    objects, key=lambda box: box.confidence, reverse=True
                )[:2]
                observations_by_view[view.view_id].append(
                    _Observation(
                        action_type=ActionType.OBJECT_MOVEMENT,
                        local_ms=frame.local_ms,
                        global_ms=frame.global_ms,
                        objects=tuple(sorted(box.class_name for box in selected)),
                        confidence=min(box.confidence for box in selected) * 0.75,
                        evidence={
                            "frame_index": frame.frame_index,
                            "coarse_motion": frame.motion_score,
                            "coarse_motion_threshold": motion_threshold,
                        },
                    )
                )
    candidates: list[ActionCandidate] = []
    for view_id, observations in observations_by_view.items():
        merged = _merge_observations(observations, by_id[view_id], cfg)
        candidates.extend(merged)
        for observation in micro_observations_by_view.get(view_id, []):
            if any(
                candidate.action_type == observation.action_type
                and candidate.global_start_ms
                <= observation.global_ms
                <= candidate.global_end_ms
                for candidate in merged
            ):
                continue
            if any(
                candidate.view_id == view_id
                and abs(candidate.key_global_ms - observation.global_ms)
                <= micro_half_window_ms
                and candidate.action_type == observation.action_type
                for candidate in candidates
            ):
                continue
            candidates.append(
                ActionCandidate(
                    candidate_id=(f"COARSE-MICRO-{view_id}-{len(candidates) + 1:06d}"),
                    action_type=observation.action_type,
                    view_id=view_id,
                    role=by_id[view_id].role,
                    local_start_ms=max(
                        0.0, observation.local_ms - micro_half_window_ms
                    ),
                    local_end_ms=observation.local_ms + micro_half_window_ms,
                    global_start_ms=max(
                        0.0, observation.global_ms - micro_half_window_ms
                    ),
                    global_end_ms=observation.global_ms + micro_half_window_ms,
                    key_global_ms=observation.global_ms,
                    objects=list(observation.objects),
                    confidence=min(0.89, observation.confidence),
                    evidence=[
                        {
                            **observation.evidence,
                            "micro_action_guard": True,
                        }
                    ],
                    uncertainty=[
                        "单帧高置信手物关系仅扩大精扫窗口，不独立确认物理动作"
                    ],
                )
            )
    return sorted(candidates, key=candidate_sort_key)


@dataclass
class _RunAccumulator:
    first: _Observation
    last: _Observation
    key_item: _Observation
    count: int
    confidence_sum: float
    objects: set[str]
    samples: list[_Observation]
    instance_mode: str
    release_observed_at_global_ms: float | None

    @classmethod
    def start(cls, observation: _Observation, *, instance_mode: str) -> _RunAccumulator:
        return cls(
            first=observation,
            last=observation,
            key_item=observation,
            count=1,
            confidence_sum=float(observation.confidence),
            objects=set(observation.objects),
            samples=[observation],
            instance_mode=instance_mode,
            release_observed_at_global_ms=None,
        )

    def add(self, observation: _Observation) -> None:
        self.last = observation
        self.count += 1
        self.confidence_sum += float(observation.confidence)
        self.objects.update(observation.objects)
        if observation.confidence > self.key_item.confidence:
            self.key_item = observation
        if len(self.samples) < 47:
            self.samples.append(observation)

    def representative_observations(self) -> list[_Observation]:
        selected = list(self.samples)
        for item in (self.key_item, self.last):
            if item not in selected:
                selected.append(item)
        return selected[:50]

    def mark_release(self, global_ms: float) -> None:
        if self.release_observed_at_global_ms is None:
            self.release_observed_at_global_ms = float(global_ms)


def _candidate_from_accumulator(
    accumulator: _RunAccumulator,
    view: ViewInput,
    cfg: dict[str, Any],
) -> ActionCandidate | None:
    duration = accumulator.last.global_ms - accumulator.first.global_ms
    minimum_observations = int(cfg["min_event_observations"])
    enough = bool(
        accumulator.count >= minimum_observations
        and duration >= float(cfg["min_event_duration_seconds"]) * 1000.0
    )
    if accumulator.first.action_type in {
        ActionType.CONTAINER_STATE_CHANGE,
        ActionType.DEVICE_PANEL_OPERATION,
    }:
        enough = accumulator.count >= max(2, minimum_observations - 1)
    if not enough:
        return None
    persistence = min(0.15, math.log1p(accumulator.count) * 0.035)
    confidence = min(
        1.0,
        accumulator.confidence_sum / accumulator.count + persistence,
    )
    representatives = accumulator.representative_observations()
    signature = _observation_instance_signature(representatives)
    uncertainty = _observation_state_uncertainty(representatives, cfg)
    interaction_receipt = {
        "schema_version": "visioncortex-contact-state/1",
        "approach_observed": any(
            bool(
                (item.evidence.get("interaction_state") or {}).get("approach_confirmed")
            )
            for item in representatives
        ),
        "contact_observation_count": accumulator.count,
        "release_observed": (accumulator.release_observed_at_global_ms is not None),
        "release_observed_at_global_ms": (
            round(accumulator.release_observed_at_global_ms, 3)
            if accumulator.release_observed_at_global_ms is not None
            else None
        ),
        "release_inferred_from_next_observation_gap": False,
    }
    if accumulator.instance_mode == "class_fallback":
        uncertainty.append("轨迹身份不足时保留原有类别级召回；不得用于证明同一物体")
    return ActionCandidate(
        candidate_id="CAND-PENDING",
        action_type=accumulator.first.action_type,
        view_id=view.view_id,
        role=view.role,
        local_start_ms=accumulator.first.local_ms,
        local_end_ms=accumulator.last.local_ms,
        global_start_ms=accumulator.first.global_ms,
        global_end_ms=accumulator.last.global_ms,
        key_global_ms=accumulator.key_item.global_ms,
        objects=sorted(accumulator.objects),
        confidence=confidence,
        evidence=[_observation_evidence(item) for item in representatives],
        uncertainty=uncertainty,
        instance_signature=signature,
        provenance={
            "source_stage": "candidate_fine",
            "candidate_reducer": "streaming",
            "instance_mode": accumulator.instance_mode,
            "observation_count": accumulator.count,
            "representative_evidence_policy": "start_peak_end_bounded_50",
            "interaction_state": interaction_receipt,
        },
    )


@dataclass
class _LiquidContactRun:
    first: _Observation
    last: _Observation
    count: int
    confidence_sum: float

    @property
    def vessel_identity(self) -> tuple[str, int]:
        return (
            str(self.first.evidence.get("vessel_class") or "unknown"),
            int(self.first.evidence["vessel_track_id"]),
        )

    def add(self, observation: _Observation) -> None:
        self.last = observation
        self.count += 1
        self.confidence_sum += float(observation.confidence)


class _StreamingLiquidSequences:
    def __init__(self, view: ViewInput, cfg: dict[str, Any]):
        self.view = view
        self.maximum_gap_ms = (
            float(cfg.get("liquid_transfer_max_sequence_gap_seconds", 20.0)) * 1000.0
        )
        self.contact_gap_ms = float(cfg.get("event_merge_gap_seconds", 1.25)) * 1000.0
        self.minimum_observations = max(
            2, int(cfg.get("liquid_transfer_min_contact_observations", 2))
        )
        self.minimum_contact_ms = max(
            0.0,
            float(cfg.get("liquid_transfer_min_contact_duration_seconds", 0.2))
            * 1000.0,
        )
        self.active: dict[tuple[str, int], _LiquidContactRun] = {}
        self.previous: dict[tuple[str, int], _LiquidContactRun] = {}
        self.output: list[ActionCandidate] = []

    def add_frame(self, observations: Sequence[_Observation]) -> None:
        by_tool: dict[tuple[str, int], list[_Observation]] = defaultdict(list)
        for observation in observations:
            if observation.action_type == ActionType.LIQUID_MOVEMENT:
                key = self._tool_key(observation)
                if key is not None:
                    by_tool[key].append(observation)
        for key, items in by_tool.items():
            vessels = {
                (
                    str(item.evidence.get("vessel_class") or "unknown"),
                    item.evidence["vessel_track_id"],
                )
                for item in items
            }
            if len(vessels) != 1:
                # Simultaneous proximity is ambiguous, never a source/target transition.
                # Raw tool/vessel observations still enter the ordinary recall path.
                self.active.pop(key, None)
                self.previous.pop(key, None)
                continue
            self.add(max(items, key=lambda item: item.confidence))

    @staticmethod
    def _tool_key(observation: _Observation) -> tuple[str, int] | None:
        tool_class = observation.evidence.get("tool_class")
        tool_track_id = observation.evidence.get("tool_track_id")
        vessel_track_id = observation.evidence.get("vessel_track_id")
        if (
            not isinstance(tool_class, str)
            or not isinstance(tool_track_id, int)
            or not isinstance(vessel_track_id, int)
        ):
            return None
        return tool_class, tool_track_id

    def add(self, observation: _Observation) -> None:
        if observation.action_type != ActionType.LIQUID_MOVEMENT:
            return
        key = self._tool_key(observation)
        if key is None:
            return
        current = self.active.get(key)
        vessel = (
            str(observation.evidence.get("vessel_class") or "unknown"),
            int(observation.evidence["vessel_track_id"]),
        )
        if current is not None and observation.global_ms <= current.last.global_ms:
            if vessel != current.vessel_identity:
                self.active.pop(key, None)
                self.previous.pop(key, None)
            return
        if current is not None and (
            vessel != current.vessel_identity
            or observation.global_ms - current.last.global_ms > self.contact_gap_ms
        ):
            self._finalize(key)
            current = None
        if current is None:
            self.active[key] = _LiquidContactRun(
                first=observation,
                last=observation,
                count=1,
                confidence_sum=float(observation.confidence),
            )
        else:
            current.add(observation)

    def expire(self, global_ms: float) -> None:
        for key, current in list(self.active.items()):
            if global_ms - current.last.global_ms > self.contact_gap_ms:
                self._finalize(key)
        for key, previous in list(self.previous.items()):
            if global_ms - previous.last.global_ms > self.maximum_gap_ms:
                del self.previous[key]

    def finish(self) -> list[ActionCandidate]:
        for key in list(self.active):
            self._finalize(key)
        return self.output

    def _finalize(self, key: tuple[str, int]) -> None:
        current = self.active.pop(key)
        if (
            current.count < self.minimum_observations
            or current.last.global_ms - current.first.global_ms
            < self.minimum_contact_ms
        ):
            self.previous.pop(key, None)
            return
        previous = self.previous.get(key)
        if previous is not None:
            gap_ms = current.first.global_ms - previous.last.global_ms
            if (
                previous.vessel_identity != current.vessel_identity
                and 0.0 < gap_ms <= self.maximum_gap_ms
            ):
                combined_count = previous.count + current.count
                confidence = min(
                    1.0,
                    (previous.confidence_sum + current.confidence_sum) / combined_count
                    + 0.08,
                )
                self.output.append(
                    ActionCandidate(
                        candidate_id=(
                            f"TRANSFER-SEQ-{self.view.view_id}-"
                            f"{len(self.output) + 1:06d}"
                        ),
                        action_type=ActionType.LIQUID_MOVEMENT,
                        view_id=self.view.view_id,
                        role=self.view.role,
                        local_start_ms=previous.first.local_ms,
                        local_end_ms=current.last.local_ms,
                        global_start_ms=previous.first.global_ms,
                        global_end_ms=current.last.global_ms,
                        key_global_ms=(
                            previous.last.global_ms + current.first.global_ms
                        )
                        / 2.0,
                        objects=sorted(
                            {
                                key[0],
                                previous.vessel_identity[0],
                                current.vessel_identity[0],
                            }
                        ),
                        confidence=confidence,
                        evidence=[
                            {
                                "transfer_sequence": "source_transport_target",
                                "tool_class": key[0],
                                "tool_track_id": key[1],
                                "source_class": previous.vessel_identity[0],
                                "source_track_id": previous.vessel_identity[1],
                                "target_class": current.vessel_identity[0],
                                "target_track_id": current.vessel_identity[1],
                                "source_contact_end_global_ms": (
                                    previous.last.global_ms
                                ),
                                "target_contact_start_global_ms": (
                                    current.first.global_ms
                                ),
                                "transport_gap_ms": gap_ms,
                                "source_observation_count": previous.count,
                                "target_observation_count": current.count,
                                "source_contact_duration_ms": previous.last.global_ms
                                - previous.first.global_ms,
                                "target_contact_duration_ms": current.last.global_ms
                                - current.first.global_ms,
                                "minimum_contact_duration_ms": self.minimum_contact_ms,
                                "exclusive_contact_observations": True,
                                "contact_sequence_only": True,
                                "transport_observed": False,
                                "streaming_state_machine": True,
                            }
                        ],
                        uncertainty=[
                            "工具先后接近不同容器；中间转运、释放及液体本体尚未确认"
                        ],
                        instance_signature={
                            "schema_version": (
                                "visioncortex-object-instance-signature/1"
                            ),
                            "tool_track_id": key[1],
                            "source_track_id": previous.vessel_identity[1],
                            "target_track_id": current.vessel_identity[1],
                            "reliable_single_instance": True,
                        },
                        provenance={
                            "source_stage": "candidate_fine",
                            "candidate_reducer": "streaming_transfer_state_machine",
                        },
                    )
                )
        self.previous[key] = current


def _legacy_candidate_covered(
    legacy: ActionCandidate,
    precise: Sequence[ActionCandidate],
) -> bool:
    legacy_objects = set(legacy.objects) - HAND_CLASSES
    return any(
        item.action_type == legacy.action_type
        and bool(legacy_objects & (set(item.objects) - HAND_CLASSES))
        and item.global_start_ms <= legacy.global_end_ms
        and item.global_end_ms >= legacy.global_start_ms
        for item in precise
    )


def _generate_candidates_streaming(
    view: ViewInput,
    frames: Iterable[FrameEvidence],
    cfg: dict[str, Any],
) -> list[ActionCandidate]:
    merge_gap_ms = float(cfg["event_merge_gap_seconds"]) * 1000.0
    instance_aware = bool(cfg.get("fine_instance_association_enabled", False))
    active_precise: dict[tuple[str, str, str], _RunAccumulator] = {}
    active_legacy: dict[tuple[str, str, str], _RunAccumulator] = {}
    precise: list[ActionCandidate] = []
    legacy: list[ActionCandidate] = []
    previous_tracks: dict[int, tuple[float, float, float]] = {}
    movement_history: dict[int, deque[tuple[float, float, float]]] = {}
    interaction_state: dict[tuple[Any, ...], dict[str, Any]] = {}
    container_states: dict[str, str] = {}
    liquid_sequences = _StreamingLiquidSequences(view, cfg)

    def finalize_stale(
        active: dict[tuple[str, str, str], _RunAccumulator],
        output: list[ActionCandidate],
        global_ms: float,
    ) -> None:
        for key, accumulator in list(active.items()):
            if global_ms - accumulator.last.global_ms <= merge_gap_ms:
                continue
            candidate = _candidate_from_accumulator(accumulator, view, cfg)
            if candidate is not None:
                output.append(candidate)
            del active[key]

    def add_to(
        active: dict[tuple[str, str, str], _RunAccumulator],
        key: tuple[str, str, str],
        observation: _Observation,
        *,
        instance_mode: str,
    ) -> None:
        accumulator = active.get(key)
        if accumulator is None:
            active[key] = _RunAccumulator.start(
                observation, instance_mode=instance_mode
            )
        else:
            accumulator.add(observation)

    def mark_observed_releases(global_ms: float) -> None:
        released_objects = {
            str(pair_key[1])
            for pair_key, state in interaction_state.items()
            if state.get("released_at_global_ms") == global_ms
        }
        if not released_objects:
            return
        for active in (active_precise, active_legacy):
            for accumulator in active.values():
                object_track_id = accumulator.last.evidence.get("object_track_id")
                object_instance = accumulator.last.evidence.get("object_instance") or {}
                object_class = object_instance.get("class_name")
                if (
                    str(object_track_id) in released_objects
                    or str(object_class) in released_objects
                ):
                    accumulator.mark_release(global_ms)

    for frame in frames:
        if frame.global_ms is None:
            continue
        finalize_stale(active_precise, precise, float(frame.global_ms))
        finalize_stale(active_legacy, legacy, float(frame.global_ms))
        liquid_sequences.expire(float(frame.global_ms))
        observations = _frame_observations(
            frame,
            previous_tracks,
            cfg,
            interaction_state,
            container_states,
            movement_history,
        )
        mark_observed_releases(float(frame.global_ms))
        liquid_sequences.add_frame(observations)
        for observation in observations:
            if instance_aware and observation.instance_key:
                add_to(
                    active_precise,
                    _observation_key(observation, instance_aware=True),
                    observation,
                    instance_mode="track_instance",
                )
                add_to(
                    active_legacy,
                    _observation_key(observation, instance_aware=False),
                    observation,
                    instance_mode="class_fallback",
                )
            else:
                add_to(
                    active_legacy,
                    _observation_key(observation, instance_aware=False),
                    observation,
                    instance_mode="class_fallback",
                )
    for active, output in (
        (active_precise, precise),
        (active_legacy, legacy),
    ):
        for accumulator in active.values():
            candidate = _candidate_from_accumulator(accumulator, view, cfg)
            if candidate is not None:
                output.append(candidate)

    if instance_aware:
        candidates = [
            *precise,
            *[item for item in legacy if not _legacy_candidate_covered(item, precise)],
        ]
    else:
        candidates = legacy
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.global_start_ms,
            item.global_end_ms,
            item.action_type.value,
            item.objects,
        ),
    )
    numbered = [
        item.model_copy(update={"candidate_id": f"CAND-{view.view_id}-{index:06d}"})
        for index, item in enumerate(ordered, 1)
    ]
    return sorted([*numbered, *liquid_sequences.finish()], key=candidate_sort_key)
