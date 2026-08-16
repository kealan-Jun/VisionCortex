from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    ViewInput,
    ViewRole,
)


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

FIXED_EQUIPMENT_OBJECTS = {
    "balance",
    "magnetic_stirrer",
    "computer",
}


def is_experiment_start_anchor(
    event: EvidenceEvent, config: dict[str, Any]
) -> bool:
    """Return whether accepted evidence is strong enough to open a clip."""

    if event.action_type in {
        ActionType.HAND_OBJECT_CONTACT,
        ActionType.CONTAINER_STATE_CHANGE,
        ActionType.DEVICE_PANEL_OPERATION,
    }:
        return True
    if event.action_type == ActionType.LIQUID_MOVEMENT:
        required = set(
            config["segmentation"].get(
                "liquid_start_anchor_required_objects", ["pipette"]
            )
        )
        return len(event.supporting_views) > 1 and (
            not required or bool(set(event.objects) & required)
        )
    return event.action_type == ActionType.OBJECT_MOVEMENT and not bool(
        set(event.objects) & FIXED_EQUIPMENT_OBJECTS
    )


def select_formal_experiment_start_events(
    events: Sequence[EvidenceEvent], config: dict[str, Any]
) -> list[EvidenceEvent]:
    """Select events allowed to open a formal dual-view experiment.

    A confident single-view prelude remains useful audit evidence, but it must
    not pull the delivered clip ahead of the first corroborated first/third-
    person operation. If no canonical action anchor exists, the earliest
    accepted dual-role event is the conservative fallback.
    """

    required_roles = {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}
    dual_role = [
        event
        for event in events
        if event.accepted and required_roles.issubset(set(event.supporting_roles))
    ]
    anchored = [
        event for event in dual_role if is_experiment_start_anchor(event, config)
    ]
    return anchored or dual_role


def normalize_experiment_segments(
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    config: dict[str, Any],
) -> list[ExperimentSegment]:
    """Collapse duplicate overlapping windows and suppress equipment-only preludes.

    Coarse activity windows can overlap and independently collect the same fine
    action chain.  Those windows are one atomic experiment, not a continuous
    chain.  A very short fixed-equipment movement immediately before a much
    richer experiment is retained as contextual evidence in the event ledger,
    but is not promoted as a standalone bounded experiment.
    """

    by_event = {event.event_id: event for event in events}
    segmentation = config["segmentation"]
    post_roll_ms = float(segmentation["experiment_post_roll_seconds"]) * 1000.0
    bridge_confidence = float(segmentation.get("boundary_context_bridge_confidence", 0.50))
    extension_confidence = float(segmentation.get("boundary_context_min_confidence", 0.65))
    maximum_gap_ms = float(segmentation.get("boundary_context_max_gap_seconds", 10.0)) * 1000.0
    cleanup_objects = {
        "brush",
        "cleaning_tool",
        "sink",
        "wash_bottle",
        "waste_container",
    }

    def trim_unrelated_head(segment: ExperimentSegment) -> ExperimentSegment:
        core_events = sorted(
            _segment_events(segment, by_event),
            key=lambda item: item.global_start_ms,
        )
        anchors = select_formal_experiment_start_events(core_events, config)
        if not anchors:
            return segment
        bounded_start = max(
            0.0,
            min(event.global_start_ms for event in anchors)
            - float(segmentation["experiment_pre_roll_seconds"]) * 1000.0,
        )
        if bounded_start <= segment.global_start_ms:
            return segment
        retained_ids = {
            event.event_id for event in core_events if event.global_end_ms >= bounded_start
        }
        return segment.model_copy(
            update={
                "global_start_ms": bounded_start,
                "event_ids": [
                    event_id for event_id in segment.event_ids if event_id in retained_ids
                ],
                "micro_segments": [
                    item
                    for item in segment.micro_segments
                    if item.get("evidence_event_id") in retained_ids
                ],
            }
        )

    def trim_unrelated_tail(segment: ExperimentSegment) -> ExperimentSegment:
        core_events = _segment_events(segment, by_event)
        if not core_events:
            return segment
        raw_end = max(event.global_end_ms for event in core_events)
        direct_end = raw_end + post_roll_ms
        if segment.global_end_ms <= direct_end:
            return segment
        connected_objects = {obj for event in core_events for obj in event.objects}
        cursor = raw_end
        supported_end = raw_end
        limit = max(raw_end, segment.global_end_ms - post_roll_ms)
        core_ids = {event.event_id for event in core_events}
        for context in sorted(events, key=lambda item: item.global_start_ms):
            if context.event_id in core_ids or context.global_end_ms <= cursor:
                continue
            if context.global_start_ms > limit:
                break
            if context.global_start_ms > cursor + maximum_gap_ms:
                break
            physically_connected = bool(
                set(context.objects) & (connected_objects | cleanup_objects)
            ) or context.action_type in {
                ActionType.CONTAINER_STATE_CHANGE,
                ActionType.DEVICE_PANEL_OPERATION,
            }
            if (
                context.confidence < bridge_confidence
                or not context.objects
                or not physically_connected
            ):
                continue
            cursor = min(limit, max(cursor, context.global_end_ms))
            connected_objects.update(context.objects)
            if context.confidence >= extension_confidence:
                supported_end = max(supported_end, cursor)
        return segment.model_copy(
            update={"global_end_ms": min(segment.global_end_ms, supported_end + post_roll_ms)}
        )

    ordered = sorted(
        (trim_unrelated_tail(trim_unrelated_head(segment)) for segment in segments),
        key=lambda item: (item.global_start_ms, item.global_end_ms),
    )
    if not ordered:
        return []
    roles = {view.view_id: view.role for view in views}
    continuity_cfg = config["continuity"]

    def has_shared_dual_view(left: ExperimentSegment, right: ExperimentSegment) -> bool:
        shared = set(left.participating_views) & set(right.participating_views)
        return any(roles.get(view_id) == ViewRole.FIRST_PERSON for view_id in shared) and any(
            roles.get(view_id) == ViewRole.THIRD_PERSON for view_id in shared
        )

    def is_same_atomic_fragment(
        left: ExperimentSegment, right: ExperimentSegment
    ) -> bool:
        gap_ms = right.global_start_ms - left.global_end_ms
        maximum_gap_ms = float(
            continuity_cfg.get("atomic_fragment_merge_max_gap_seconds", 20.0)
        ) * 1000.0
        if gap_ms < 0.0 or gap_ms > maximum_gap_ms:
            return False
        left_events = _segment_events(left, by_event)
        right_events = _segment_events(right, by_event)
        blocking_actions = {
            ActionType.CONTAINER_STATE_CHANGE,
            ActionType.DEVICE_PANEL_OPERATION,
        }
        if any(
            event.action_type in blocking_actions
            or bool(set(event.objects) & FIXED_EQUIPMENT_OBJECTS)
            for event in [*left_events, *right_events]
        ):
            return False
        left_objects = {
            obj for event in left_events for obj in event.objects
        } & CONTINUITY_OBJECTS
        right_objects = {
            obj for event in right_events for obj in event.objects
        } & CONTINUITY_OBJECTS
        minimum_shared_objects = max(
            1,
            int(continuity_cfg.get("atomic_fragment_min_shared_objects", 2)),
        )
        return (
            has_shared_dual_view(left, right)
            and len(left_objects & right_objects) >= minimum_shared_objects
        )

    consolidated: list[ExperimentSegment] = []
    for segment in ordered:
        left = consolidated[-1] if consolidated else None
        if left is not None and has_shared_dual_view(left, segment) and (
            segment.global_start_ms <= left.global_end_ms
            or is_same_atomic_fragment(left, segment)
        ):
            merged_events = list(dict.fromkeys([*left.event_ids, *segment.event_ids]))
            merged_micro = sorted(
                [*left.micro_segments, *segment.micro_segments],
                key=lambda item: float(item.get("start_global_ms", 0.0)),
            )
            consolidated[-1] = left.model_copy(
                update={
                    "global_start_ms": min(left.global_start_ms, segment.global_start_ms),
                    "global_end_ms": max(left.global_end_ms, segment.global_end_ms),
                    "event_ids": merged_events,
                    "participating_views": sorted(
                        set(left.participating_views) | set(segment.participating_views)
                    ),
                    "rejected_views": {
                        key: value
                        for key, value in left.rejected_views.items()
                        if key in segment.rejected_views
                    },
                    "micro_segments": merged_micro,
                }
            )
        else:
            consolidated.append(segment)

    cfg = continuity_cfg
    max_gap_ms = float(cfg.get("preparation_bridge_max_gap_seconds", 20.0)) * 1000.0
    max_duration_ms = float(cfg.get("preparation_segment_max_seconds", 20.0)) * 1000.0
    following_min_ms = float(cfg.get("preparation_following_min_seconds", 30.0)) * 1000.0
    max_events = int(cfg.get("preparation_segment_max_events", 2))
    filtered: list[ExperimentSegment] = []
    for index, segment in enumerate(consolidated):
        following = consolidated[index + 1] if index + 1 < len(consolidated) else None
        segment_events = _segment_events(segment, by_event)
        following_events = _segment_events(following, by_event) if following else []
        segment_objects = {obj for event in segment_events for obj in event.objects}
        following_objects = {obj for event in following_events for obj in event.objects}
        equipment_only_prelude = bool(segment_events) and all(
            event.action_type == ActionType.OBJECT_MOVEMENT
            and set(event.objects) <= FIXED_EQUIPMENT_OBJECTS
            for event in segment_events
        )
        suppress = bool(
            following
            and has_shared_dual_view(segment, following)
            and 0.0 <= following.global_start_ms - segment.global_end_ms <= max_gap_ms
            and segment.global_end_ms - segment.global_start_ms <= max_duration_ms
            and following.global_end_ms - following.global_start_ms >= following_min_ms
            and len(segment_events) <= max_events
            and equipment_only_prelude
            and bool(segment_objects & following_objects & FIXED_EQUIPMENT_OBJECTS)
        )
        if not suppress:
            filtered.append(segment)
    return filtered


def _segment_events(
    segment: ExperimentSegment, by_event: dict[str, EvidenceEvent]
) -> list[EvidenceEvent]:
    return [by_event[event_id] for event_id in segment.event_ids if event_id in by_event]


def prepare_formal_experiment_segments(
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    coarse_windows: Sequence[ActionCandidate],
    config: dict[str, Any],
) -> tuple[list[ExperimentSegment], list[dict[str, Any]]]:
    """Promote only dual-view experiments and conservatively attach FP tails.

    Standalone single-role activity is retained in the event/audit ledger but
    quarantined from formal clips. A following first-person-only cleanup tail
    may extend the preceding dual-view segment only inside the same recalled
    coarse boundary with tight temporal and physical continuity.
    """

    by_event = {event.event_id: event for event in events}
    roles = {view.view_id: view.role for view in views}
    required_roles = {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}
    segmentation = config["segmentation"]
    maximum_gap_ms = (
        float(segmentation.get("boundary_context_max_gap_seconds", 10.0)) * 1000.0
    )
    maximum_extension_ms = (
        float(segmentation.get("boundary_context_max_extension_seconds", 90.0))
        * 1000.0
    )
    cleanup_objects = {
        "brush",
        "cleaning_tool",
        "sink",
        "wash_bottle",
        "waste_container",
    }
    actor_objects = {"hand", "gloved_hand", "lab_coat"}
    continuity = config["continuity"]
    bridge_step_gap_ms = (
        float(
            continuity.get(
                "quarantined_atomic_bridge_max_step_gap_seconds", 60.0
            )
        )
        * 1000.0
    )
    bridge_span_ms = (
        float(
            continuity.get("quarantined_atomic_bridge_max_span_seconds", 180.0)
        )
        * 1000.0
    )

    def segment_roles(segment: ExperimentSegment) -> set[ViewRole]:
        return {
            roles[view_id]
            for view_id in segment.participating_views
            if view_id in roles
        }

    def boundary_ids(segment: ExperimentSegment) -> set[str]:
        midpoint = (segment.global_start_ms + segment.global_end_ms) / 2.0
        return {
            window.candidate_id
            for window in coarse_windows
            if window.global_start_ms <= midpoint <= window.global_end_ms
        }

    def event_objects(segment: ExperimentSegment) -> set[str]:
        return {
            obj
            for event in _segment_events(segment, by_event)
            for obj in event.objects
            if obj not in actor_objects
        }

    def first_person_views(segment: ExperimentSegment) -> set[str]:
        return {
            view_id
            for view_id in segment.participating_views
            if roles.get(view_id) == ViewRole.FIRST_PERSON
        }

    def quarantined_atomic_bridge(
        left: ExperimentSegment,
        context: ExperimentSegment,
        right: ExperimentSegment,
    ) -> dict[str, Any] | None:
        """Prove a graph-only bridge without promoting its single-view events."""

        if not (
            required_roles.issubset(segment_roles(left))
            and segment_roles(context) == {ViewRole.FIRST_PERSON}
            and required_roles.issubset(segment_roles(right))
        ):
            return None
        left_gap_ms = context.global_start_ms - left.global_end_ms
        right_gap_ms = right.global_start_ms - context.global_end_ms
        outer_span_ms = right.global_end_ms - left.global_start_ms
        if not (
            0.0 <= left_gap_ms <= bridge_step_gap_ms
            and 0.0 <= right_gap_ms <= bridge_step_gap_ms
            and 0.0 < outer_span_ms <= bridge_span_ms
        ):
            return None
        shared_first = (
            first_person_views(left)
            & first_person_views(context)
            & first_person_views(right)
        )
        shared_boundaries = (
            boundary_ids(left) & boundary_ids(context) & boundary_ids(right)
        )
        context_events = _segment_events(context, by_event)
        if not shared_first or not shared_boundaries or not any(
            event.accepted for event in context_events
        ):
            return None
        shared_windows = [
            window
            for window in coarse_windows
            if window.candidate_id in shared_boundaries
        ]
        context_guard_ms = maximum_gap_ms
        coarse_context_start_ms = max(
            window.global_start_ms for window in shared_windows
        ) + context_guard_ms
        coarse_context_end_ms = min(
            window.global_end_ms for window in shared_windows
        ) - context_guard_ms
        if coarse_context_end_ms <= coarse_context_start_ms:
            return None
        return {
            "shared_first_person_views": sorted(shared_first),
            "shared_boundary_ids": sorted(shared_boundaries),
            "left_gap_ms": left_gap_ms,
            "right_gap_ms": right_gap_ms,
            "outer_span_ms": outer_span_ms,
            "coarse_context_start_ms": coarse_context_start_ms,
            "coarse_context_end_ms": coarse_context_end_ms,
            "coarse_context_guard_ms": context_guard_ms,
            "semantic_object_bridge_proven": bool(
                event_objects(left)
                & event_objects(context)
                & event_objects(right)
            ),
        }

    def merge_dual_fragments(
        left: ExperimentSegment,
        right: ExperimentSegment,
        bridge: dict[str, Any],
    ) -> ExperimentSegment:
        event_ids = list(dict.fromkeys([*left.event_ids, *right.event_ids]))
        micro_segments = sorted(
            [*left.micro_segments, *right.micro_segments],
            key=lambda item: float(item.get("start_global_ms", 0.0)),
        )
        participating_views = sorted(
            set(left.participating_views) | set(right.participating_views)
        )
        return left.model_copy(
            update={
                "global_start_ms": max(
                    0.0,
                    left.global_start_ms - maximum_extension_ms,
                    min(
                        left.global_start_ms,
                        float(bridge["coarse_context_start_ms"]),
                    ),
                ),
                "global_end_ms": min(
                    right.global_end_ms + maximum_extension_ms,
                    max(
                        right.global_end_ms,
                        float(bridge["coarse_context_end_ms"]),
                    ),
                ),
                "event_ids": event_ids,
                "participating_views": participating_views,
                "rejected_views": {
                    key: value
                    for key, value in left.rejected_views.items()
                    if key in right.rejected_views
                },
                "micro_segments": micro_segments,
            }
        )

    promoted: list[ExperimentSegment] = []
    receipts: list[dict[str, Any]] = []
    previous_input_attachable = False
    ordered_segments = sorted(
        segments, key=lambda item: (item.global_start_ms, item.global_end_ms)
    )
    index = 0
    while index < len(ordered_segments):
        segment = ordered_segments[index]
        if index + 2 < len(ordered_segments):
            context = ordered_segments[index + 1]
            right = ordered_segments[index + 2]
            bridge = quarantined_atomic_bridge(segment, context, right)
            if bridge is not None:
                merged = merge_dual_fragments(segment, right, bridge)
                context_event_ids = set(context.event_ids)
                if context_event_ids & set(merged.event_ids):
                    raise ValueError(
                        "quarantined continuity events leaked into a formal segment"
                    )
                promoted.append(merged)
                receipts.extend(
                    [
                        {
                            "segment_id": segment.segment_id,
                            "decision": "promoted_dual_view",
                            "roles": sorted(
                                role.value for role in segment_roles(segment)
                            ),
                        },
                        {
                            "segment_id": context.segment_id,
                            "decision": "quarantined_continuity_bridge",
                            "roles": sorted(
                                role.value for role in segment_roles(context)
                            ),
                            "global_start_ms": context.global_start_ms,
                            "global_end_ms": context.global_end_ms,
                            "event_ids": list(context.event_ids),
                            "bridged_left_segment_id": segment.segment_id,
                            "bridged_right_segment_id": right.segment_id,
                            **bridge,
                        },
                        {
                            "segment_id": right.segment_id,
                            "decision": "merged_dual_view_fragment",
                            "merged_into_segment_id": segment.segment_id,
                            "roles": sorted(
                                role.value for role in segment_roles(right)
                            ),
                            **bridge,
                        },
                    ]
                )
                previous_input_attachable = True
                index += 3
                continue

        current_roles = segment_roles(segment)
        if required_roles.issubset(current_roles):
            promoted.append(segment)
            previous_input_attachable = True
            receipts.append(
                {
                    "segment_id": segment.segment_id,
                    "decision": "promoted_dual_view",
                    "roles": sorted(role.value for role in current_roles),
                }
            )
            index += 1
            continue

        previous = promoted[-1] if promoted and previous_input_attachable else None
        gap_ms = (
            segment.global_start_ms - previous.global_end_ms if previous else None
        )
        extension_ms = (
            segment.global_end_ms - previous.global_end_ms if previous else None
        )
        shared_boundaries = (
            boundary_ids(previous) & boundary_ids(segment) if previous else set()
        )
        previous_objects = event_objects(previous) if previous else set()
        tail_objects = event_objects(segment)
        shared_objects = previous_objects & tail_objects
        cleanup_evidence = tail_objects & cleanup_objects
        eligible_tail = bool(
            previous
            and current_roles == {ViewRole.FIRST_PERSON}
            and gap_ms is not None
            and -maximum_gap_ms <= gap_ms <= maximum_gap_ms
            and extension_ms is not None
            and extension_ms <= maximum_extension_ms
            and shared_boundaries
            and (shared_objects or cleanup_evidence)
        )
        if eligible_tail:
            merged_event_ids = list(
                dict.fromkeys([*previous.event_ids, *segment.event_ids])
            )
            merged_micro = sorted(
                [*previous.micro_segments, *segment.micro_segments],
                key=lambda item: float(item.get("start_global_ms", 0.0)),
            )
            merged_views = sorted(
                set(previous.participating_views) | set(segment.participating_views)
            )
            promoted[-1] = previous.model_copy(
                update={
                    "global_end_ms": max(
                        previous.global_end_ms, segment.global_end_ms
                    ),
                    "event_ids": merged_event_ids,
                    "participating_views": merged_views,
                    "rejected_views": {
                        key: value
                        for key, value in previous.rejected_views.items()
                        if key not in merged_views
                    },
                    "micro_segments": merged_micro,
                }
            )
            receipts.append(
                {
                    "segment_id": segment.segment_id,
                    "decision": "attached_first_person_tail",
                    "attached_to_segment_id": previous.segment_id,
                    "gap_ms": gap_ms,
                    "extension_ms": extension_ms,
                    "shared_boundary_ids": sorted(shared_boundaries),
                    "shared_objects": sorted(shared_objects),
                    "cleanup_objects": sorted(cleanup_evidence),
                }
            )
            previous_input_attachable = True
            index += 1
            continue

        receipts.append(
            {
                "segment_id": segment.segment_id,
                "decision": "quarantined_missing_dual_view",
                "roles": sorted(role.value for role in current_roles),
                "global_start_ms": segment.global_start_ms,
                "global_end_ms": segment.global_end_ms,
                "event_ids": segment.event_ids,
                "candidate_tail_gap_ms": gap_ms,
                "shared_boundary_ids": sorted(shared_boundaries),
                "shared_objects": sorted(shared_objects),
                "cleanup_objects": sorted(cleanup_evidence),
            }
        )
        previous_input_attachable = False
        index += 1

    return promoted, receipts


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
    decision_receipts: list[dict[str, Any]] | None = None,
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
    decisions: dict[str, dict[str, Any]] = {}

    def ensure_decision(
        event: EvidenceEvent,
        group: ExperimentGroup,
        segment_id: str,
    ) -> dict[str, Any]:
        return decisions.setdefault(
            event.event_id,
            {
                "event_id": event.event_id,
                "group_id": group.group_id,
                "segment_id": segment_id,
                "action_type": event.action_type.value,
                "objects": list(event.objects),
                "global_start_ms": event.global_start_ms,
                "global_end_ms": event.global_end_ms,
                "peak_timestamp_ms": event.key_global_ms,
                "confidence": event.confidence,
                "selected": False,
                "decision": "pending",
                "competitor_event_id": None,
                "shared_objects": [],
                "peak_distance_ms": None,
                "interval_overlap_ms": None,
                "interval_overlap_ratio": None,
                "minimum_separation_ms": separation_ms,
                "legacy_overlap_ratio_threshold_not_used": overlap_ratio,
                "requires_positive_interval_overlap": True,
            },
        )

    def duplicate_metrics(
        existing: EvidenceEvent,
        event: EvidenceEvent,
    ) -> dict[str, Any]:
        shared_objects = sorted(set(existing.objects) & set(event.objects))
        interval_overlap_ms = max(
            0.0,
            min(existing.global_end_ms, event.global_end_ms)
            - max(existing.global_start_ms, event.global_start_ms),
        )
        shorter_duration_ms = max(
            1.0,
            min(
                existing.global_end_ms - existing.global_start_ms,
                event.global_end_ms - event.global_start_ms,
            ),
        )
        return {
            "shared_objects": shared_objects,
            "peak_distance_ms": abs(
                existing.key_global_ms - event.key_global_ms
            ),
            "interval_overlap_ms": interval_overlap_ms,
            "interval_overlap_ratio": interval_overlap_ms
            / shorter_duration_ms,
        }

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
                event_decision = ensure_decision(event, group, segment_id)
                bucket = per_type.setdefault(event.action_type.value, [])
                duplicate: tuple[EvidenceEvent, dict[str, Any]] | None = None
                for existing in bucket:
                    metrics = duplicate_metrics(existing, event)
                    is_duplicate = bool(metrics["shared_objects"]) and bool(
                        metrics["interval_overlap_ms"] > 0.0
                        and metrics["peak_distance_ms"] < separation_ms
                    )
                    if is_duplicate:
                        duplicate = (existing, metrics)
                        break
                if duplicate is not None:
                    nearby, metrics = duplicate
                    event_decision.update(metrics)
                    event_decision["competitor_event_id"] = nearby.event_id
                    if event.confidence > nearby.confidence:
                        bucket[bucket.index(nearby)] = event
                        replaced = ensure_decision(nearby, group, segment_id)
                        replaced.update(metrics)
                        replaced.update(
                            {
                                "selected": False,
                                "decision": "duplicate_replaced_by_higher_confidence",
                                "competitor_event_id": event.event_id,
                            }
                        )
                        event_decision["decision"] = "pending_after_replacement"
                    else:
                        event_decision.update(
                            {
                                "selected": False,
                                "decision": "duplicate_lower_or_equal_confidence",
                            }
                        )
                    continue
                bucket.append(event)
            for bucket in per_type.values():
                if len(bucket) > max_per_type:
                    ranked = sorted(
                        bucket, key=lambda event: event.confidence, reverse=True
                    )
                    retained_ids = {
                        event.event_id for event in ranked[:max_per_type]
                    }
                    for dropped in ranked[max_per_type:]:
                        ensure_decision(dropped, group, segment_id).update(
                            {
                                "selected": False,
                                "decision": "per_action_type_cap",
                                "retained_event_ids": sorted(retained_ids),
                                "cap": max_per_type,
                            }
                        )
                    bucket = ranked[:max_per_type]
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
            retained = mandatory + remaining[: max(0, max_total - len(mandatory))]
            retained_ids = {event.event_id for event in retained}
            for dropped in group_selected:
                if dropped.event_id not in retained_ids:
                    decisions[dropped.event_id].update(
                        {
                            "selected": False,
                            "decision": "per_experiment_group_cap",
                            "retained_event_ids": sorted(retained_ids),
                            "cap": max_total,
                        }
                    )
            group_selected = sorted(
                retained,
                key=lambda event: event.key_global_ms,
            )
        for event in group_selected:
            decisions[event.event_id].update(
                {
                    "selected": True,
                    "decision": "selected",
                }
            )
        group.key_event_ids = [event.event_id for event in group_selected]
        selected.extend(group_selected)
    if decision_receipts is not None:
        decision_receipts.extend(
            sorted(
                decisions.values(),
                key=lambda item: (
                    str(item["group_id"]),
                    str(item["segment_id"]),
                    float(item["peak_timestamp_ms"]),
                    str(item["event_id"]),
                ),
            )
        )
    return selected
