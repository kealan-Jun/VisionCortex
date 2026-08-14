from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .schemas import EvidenceEvent, ExperimentGroup, ExperimentSegment, ViewInput, ViewRole


CONTINUITY_OBJECTS = {
    "beaker",
    "container",
    "reagent_bottle",
    "reagent_bottle_open",
    "sample_bottle",
    "sample_bottle_blue",
    "tube",
    "tube_rack",
    "magnetic_stir_bar",
}


def _segment_events(
    segment: ExperimentSegment, by_event: dict[str, EvidenceEvent]
) -> list[EvidenceEvent]:
    return [by_event[event_id] for event_id in segment.event_ids if event_id in by_event]


def _continuity_evidence(
    left: ExperimentSegment,
    right: ExperimentSegment,
    by_event: dict[str, EvidenceEvent],
    roles: dict[str, ViewRole],
    config: dict[str, Any],
) -> tuple[bool, str]:
    cfg = config["continuity"]
    gap_ms = right.global_start_ms - left.global_end_ms
    max_gap_ms = float(cfg["max_gap_seconds"]) * 1000.0
    if gap_ms < 0:
        return True, "原子实验边界重叠，属于同一连续活动链"
    if gap_ms > max_gap_ms:
        return False, f"间隔 {gap_ms / 1000.0:.1f}s 超过连续实验阈值 {max_gap_ms / 1000.0:.1f}s"

    shared_views = sorted(set(left.participating_views) & set(right.participating_views))
    shared_third = [view for view in shared_views if roles.get(view) == ViewRole.THIRD_PERSON]
    if bool(cfg.get("require_shared_third_view", True)) and not shared_third:
        return False, "前后原子实验没有共同的有效第三人称证据视角"

    left_events = _segment_events(left, by_event)
    right_events = _segment_events(right, by_event)
    left_objects = {obj for event in left_events for obj in event.objects} & CONTINUITY_OBJECTS
    right_objects = {obj for event in right_events for obj in event.objects} & CONTINUITY_OBJECTS
    shared_objects = sorted(left_objects & right_objects)
    if len(shared_objects) < int(cfg.get("minimum_shared_objects", 1)):
        return False, "时间接近但没有容器/样品对象承接，判定为独立实验"
    return (
        True,
        f"间隔 {gap_ms / 1000.0:.1f}s，共同视角={shared_views}，承接对象={shared_objects}",
    )


def _select_view_pair(
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
) -> tuple[str, str]:
    roles = {view.view_id: view.role for view in views}
    event_ids = {event_id for segment in segments for event_id in segment.event_ids}
    scores: Counter[str] = Counter()
    for event in events:
        if event.event_id not in event_ids or not event.accepted:
            continue
        for view_id in event.supporting_views:
            scores[view_id] += max(1, round(event.confidence * 10))
    participating = {view for segment in segments for view in segment.participating_views}
    first_candidates = [view for view in participating if roles.get(view) == ViewRole.FIRST_PERSON]
    third_candidates = [view for view in participating if roles.get(view) == ViewRole.THIRD_PERSON]
    if not first_candidates or not third_candidates:
        raise ValueError("实验组必须同时具有第一人称和第三人称有效视角")
    first = max(first_candidates, key=lambda view: (scores[view], view))
    third = max(third_candidates, key=lambda view: (scores[view], view))
    return first, third


def build_experiment_groups(
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    config: dict[str, Any],
) -> list[ExperimentGroup]:
    """Group atomic experiments only when temporal and physical continuity agree."""
    ordered = sorted(segments, key=lambda segment: segment.global_start_ms)
    if not ordered:
        return []
    by_event = {event.event_id: event for event in events}
    roles = {view.view_id: view.role for view in views}
    chains: list[tuple[list[ExperimentSegment], list[str]]] = []
    current = [ordered[0]]
    reasons: list[str] = []
    for segment in ordered[1:]:
        continuous, reason = _continuity_evidence(current[-1], segment, by_event, roles, config)
        if continuous:
            current.append(segment)
            reasons.append(reason)
        else:
            chains.append((current, reasons + [f"与下一原子实验分开：{reason}"]))
            current, reasons = [segment], []
    chains.append((current, reasons + ["时间轴末尾，无后续原子实验"]))

    groups: list[ExperimentGroup] = []
    for index, (chain, chain_reasons) in enumerate(chains, 1):
        first, third = _select_view_pair(chain, events, views)
        group_id = f"GROUP-{index:04d}"
        for segment in chain:
            segment.group_id = group_id
        groups.append(
            ExperimentGroup(
                group_id=group_id,
                continuity_type="continuous" if len(chain) > 1 else "independent",
                atomic_experiment_ids=[segment.segment_id for segment in chain],
                global_start_ms=min(segment.global_start_ms for segment in chain),
                global_end_ms=max(segment.global_end_ms for segment in chain),
                participating_views=sorted({first, third}),
                first_person_view=first,
                third_person_view=third,
                continuity_reason="；".join(chain_reasons),
            )
        )
    return groups


def select_key_events(
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    config: dict[str, Any],
) -> list[EvidenceEvent]:
    """Select representative, non-duplicate physical actions for user delivery."""
    by_segment = {segment.segment_id: segment for segment in segments}
    by_event = {event.event_id: event for event in events}
    cfg = config["key_materials"]
    # Deduplicate repeated evidence for the *same physical action*, not every
    # occurrence of an action class within a broad time window. Pipetting,
    # weighing and cap operations often repeat several times only seconds
    # apart and each state transition is useful evidence.
    separation_ms = float(cfg["minimum_separation_seconds"]) * 1000.0
    overlap_ratio = float(cfg.get("duplicate_interval_overlap_ratio", 0.50))
    max_per_type = int(cfg["max_per_action_type_per_atomic_experiment"])
    selected: list[EvidenceEvent] = []
    for group in groups:
        group_selected: list[EvidenceEvent] = []
        for segment_id in group.atomic_experiment_ids:
            segment = by_segment[segment_id]
            candidates = sorted(
                (
                    by_event[event_id]
                    for event_id in segment.event_ids
                    if event_id in by_event and by_event[event_id].accepted
                ),
                key=lambda event: event.key_global_ms,
            )
            per_type: dict[str, list[EvidenceEvent]] = {}
            for event in candidates:
                bucket = per_type.setdefault(event.action_type.value, [])
                nearby = next(
                    (
                        existing
                        for existing in bucket
                        if bool(set(existing.objects) & set(event.objects))
                        and (
                            abs(existing.key_global_ms - event.key_global_ms)
                            < separation_ms
                            or (
                                max(
                                    0.0,
                                    min(existing.global_end_ms, event.global_end_ms)
                                    - max(existing.global_start_ms, event.global_start_ms),
                                )
                                / max(
                                    1.0,
                                    min(
                                        existing.global_end_ms
                                        - existing.global_start_ms,
                                        event.global_end_ms - event.global_start_ms,
                                    ),
                                )
                                >= overlap_ratio
                            )
                        )
                    ),
                    None,
                )
                if nearby is not None:
                    if event.confidence > nearby.confidence:
                        bucket[bucket.index(nearby)] = event
                    continue
                bucket.append(event)
            for bucket in per_type.values():
                if len(bucket) > max_per_type:
                    bucket = sorted(bucket, key=lambda event: event.confidence, reverse=True)[:max_per_type]
                group_selected.extend(bucket)
        group_selected.sort(key=lambda event: event.key_global_ms)
        max_total = int(cfg["max_per_experiment_group"])
        if len(group_selected) > max_total:
            mandatory = []
            for action_type in sorted({event.action_type for event in group_selected}, key=lambda item: item.value):
                mandatory.append(
                    max(
                        (event for event in group_selected if event.action_type == action_type),
                        key=lambda event: event.confidence,
                    )
                )
            remaining = [event for event in group_selected if event not in mandatory]
            remaining.sort(key=lambda event: event.confidence, reverse=True)
            group_selected = sorted(
                mandatory + remaining[: max(0, max_total - len(mandatory))],
                key=lambda event: event.key_global_ms,
            )
        group.key_event_ids = [event.event_id for event in group_selected]
        selected.extend(group_selected)
    return selected
