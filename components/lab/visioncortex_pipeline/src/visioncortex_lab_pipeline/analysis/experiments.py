"""Shared experiment composition after input-specific event auditing.

Recorder clock placement and offline full-timeline routing belong to their
adapters. This module owns the common segmentation/grouping/key-selection
sequence and keeps every stage's decision receipts separate.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..schemas import (
    ActionCandidate,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    ViewInput,
)


@dataclass(frozen=True)
class ExperimentAnalysisOperations:
    """Explicit algorithm bindings, also used by compatibility entry points."""

    build_segments: Callable[..., list[ExperimentSegment]]
    normalize_segments: Callable[..., list[ExperimentSegment]]
    prepare_formal_segments: Callable[
        ..., tuple[list[ExperimentSegment], list[dict[str, Any]]]
    ]
    build_groups: Callable[..., list[ExperimentGroup]]
    select_keys: Callable[..., list[EvidenceEvent]]


@dataclass
class ExperimentAnalysisResult:
    raw_segments: list[ExperimentSegment]
    normalized_segments: list[ExperimentSegment]
    segments: list[ExperimentSegment]
    groups: list[ExperimentGroup]
    selected_key_events: list[EvidenceEvent]
    boundary_decisions: list[dict[str, Any]] = field(default_factory=list)
    normalization_decisions: list[dict[str, Any]] = field(default_factory=list)
    formal_decisions: list[dict[str, Any]] = field(default_factory=list)
    continuity_decisions: list[dict[str, Any]] = field(default_factory=list)
    selection_decisions: list[dict[str, Any]] = field(default_factory=list)


def compose_experiments(
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    coarse_candidates: Sequence[ActionCandidate],
    config: dict[str, Any],
    *,
    operations: ExperimentAnalysisOperations,
    boundary_decisions: list[dict[str, Any]] | None = None,
    collect_decisions: bool = True,
) -> ExperimentAnalysisResult:
    """Apply the same ordered algorithms without changing adapter policy."""
    boundaries = boundary_decisions if boundary_decisions is not None else []
    normalization: list[dict[str, Any]] = []
    continuity: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []

    def receipts(target):
        # Progressive preview historically omits optional receipt collectors;
        # preserve that invocation policy while sharing the same composition.
        return {"decision_receipts": target} if collect_decisions else {}

    raw = operations.build_segments(
        events,
        views,
        config,
        coarse_windows=coarse_candidates,
        **receipts(boundaries),
    )
    normalized = operations.normalize_segments(
        raw, events, views, config, **receipts(normalization)
    )
    segments, formal = operations.prepare_formal_segments(
        normalized, events, views, coarse_candidates, config
    )
    groups = operations.build_groups(
        segments,
        events,
        views,
        config,
        **receipts(continuity),
        coarse_windows=coarse_candidates,
    )
    keys = operations.select_keys(
        groups, segments, events, config, **receipts(selection)
    )
    return ExperimentAnalysisResult(
        raw,
        normalized,
        segments,
        groups,
        keys,
        boundaries,
        normalization,
        formal,
        continuity,
        selection,
    )
