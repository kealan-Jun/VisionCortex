from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .decisions import decision_receipt
from .ordering import (
    event_sort_key,
    segment_sort_key,
    stable_event_fingerprint,
    stable_segment_fingerprint,
)
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

ACTOR_OBJECTS = {"hand", "gloved_hand", "lab_coat"}


def _non_actor_objects(event: EvidenceEvent) -> set[str]:
    return set(event.objects) - ACTOR_OBJECTS


def _candidate_track_identities(candidate: ActionCandidate) -> set[tuple[str, str, int]]:
    """Return object identities that are explicitly bound to a tracker ID.

    Tracker IDs are scoped to one view.  We therefore require the same view,
    normalized object class, and ID on both sides of a continuity edge.  A
    class name by itself is intentionally not treated as physical identity.
    """

    objects = sorted(set(candidate.objects) - ACTOR_OBJECTS)
    identities: set[tuple[str, str, int]] = set()
    for evidence in candidate.evidence:
        explicit_object = evidence.get("object_class_name") or evidence.get(
            "object_name"
        )
        bound_objects = (
            [str(explicit_object)]
            if explicit_object in objects
            else objects
            if len(objects) == 1
            else []
        )
        track_ids = [
            evidence.get(key)
            for key in (
                "object_track_id",
                "track_id",
                "container_track_id",
                "tool_track_id",
            )
            if evidence.get(key) is not None
        ]
        for object_name in bound_objects:
            for track_id in track_ids:
                try:
                    identities.add(
                        (candidate.view_id, object_name, int(track_id))
                    )
                except (TypeError, ValueError):
                    continue
    return identities


def _candidate_recall_bounds(candidate: ActionCandidate) -> tuple[float, float]:
    """Return the recall envelope retained by coarse-refinement receipts.

    Coarse YOLO may tighten the candidate used for fine routing, but the
    original motion envelope remains the audited recall boundary. Formal
    context and continuity checks must not silently discard that envelope.
    """

    start_ms = candidate.global_start_ms
    end_ms = candidate.global_end_ms
    for item in candidate.evidence:
        if item.get("source") != "coarse_yolo_refinement":
            continue
        original_start = item.get("original_global_start_ms")
        original_end = item.get("original_global_end_ms")
        if original_start is None or original_end is None:
            continue
        start_ms = min(start_ms, float(original_start))
        end_ms = max(end_ms, float(original_end))
    return start_ms, end_ms


def _event_track_identities(event: EvidenceEvent) -> set[tuple[str, str, int]]:
    return {
        identity
        for candidate in event.candidates
        for identity in _candidate_track_identities(candidate)
    }


def _segment_track_identities(
    segment: ExperimentSegment, by_event: dict[str, EvidenceEvent]
) -> set[tuple[str, str, int]]:
    return {
        identity
        for event in _segment_events(segment, by_event)
        for identity in _event_track_identities(event)
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
    # Movement by itself is context, not proof that an experiment started.  It
    # becomes an opener only when the selector below finds a later accepted
    # dual-role operation on the same manipulated object.
    return False


def select_formal_experiment_start_events(
    events: Sequence[EvidenceEvent],
    config: dict[str, Any],
    decision_receipts: list[dict[str, Any]] | None = None,
) -> list[EvidenceEvent]:
    """Select events allowed to open a formal dual-view experiment.

    A confident single-view prelude remains useful audit evidence, but it must
    not pull the delivered clip ahead of the first corroborated first/third-
    person operation. Object movement is allowed only when a later accepted
    dual-role action corroborates the same non-hand manipulated object.
    """

    required_roles = {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}
    dual_role = [
        event
        for event in events
        if event.accepted and required_roles.issubset(set(event.supporting_roles))
    ]
    canonical = [
        event for event in dual_role if is_experiment_start_anchor(event, config)
    ]
    corroboration_gap_ms = float(
        config["segmentation"].get(
            "movement_opener_corroboration_seconds", 30.0
        )
    ) * 1000.0
    corroborated_movement: list[EvidenceEvent] = []
    for event in sorted(dual_role, key=event_sort_key):
        if event.action_type != ActionType.OBJECT_MOVEMENT:
            if decision_receipts is not None:
                accepted = event in canonical
                decision_receipts.append(
                    decision_receipt(
                        decision_type="experiment_opener",
                        rule_id="QF4-CANONICAL-OPENER",
                        verdict="accepted" if accepted else "deferred",
                        subject_ids=[event.event_id],
                        reason_codes=[
                            "canonical_physical_anchor"
                            if accepted
                            else "noncanonical_action"
                        ],
                        facts={
                            "action_type": event.action_type.value,
                            "objects": sorted(_non_actor_objects(event)),
                        },
                        legacy={
                            "event_id": event.event_id,
                            "decision": (
                                "accepted_canonical_opener"
                                if accepted
                                else "deferred_noncanonical_opener"
                            ),
                        },
                    )
                )
            continue
        manipulated = _non_actor_objects(event)
        corroborator = next(
            (
                later
                for later in sorted(dual_role, key=event_sort_key)
                if later.event_id != event.event_id
                and later.action_type != ActionType.OBJECT_MOVEMENT
                # Corroborating contact often overlaps the tracked movement;
                # requiring it to begin only after movement_end discarded a
                # valid opener by a few hundred milliseconds.
                and later.global_end_ms >= event.global_start_ms
                and later.global_start_ms - event.global_end_ms
                <= corroboration_gap_ms
                and bool(manipulated & _non_actor_objects(later))
            ),
            None,
        )
        if corroborator is not None:
            corroborated_movement.append(event)
        if decision_receipts is not None:
            shared_objects = sorted(
                manipulated & _non_actor_objects(corroborator)
                if corroborator is not None
                else set()
            )
            decision_receipts.append(
                decision_receipt(
                    decision_type="experiment_opener",
                    rule_id="QF4-MOVEMENT-CORROBORATION",
                    verdict="accepted" if corroborator else "deferred",
                    subject_ids=[
                        event.event_id,
                        *([corroborator.event_id] if corroborator else []),
                    ],
                    reason_codes=[
                        "later_same_object_dual_role_action"
                        if corroborator
                        else "missing_later_same_object_dual_role_action"
                    ],
                    facts={
                        "action_type": event.action_type.value,
                        "objects": sorted(manipulated),
                        "corroborating_event_id": (
                            corroborator.event_id if corroborator else None
                        ),
                        "shared_non_hand_objects": shared_objects,
                    },
                    thresholds={
                        "maximum_corroboration_gap_ms": corroboration_gap_ms,
                    },
                    evidence_refs=[event.event_id],
                    legacy={
                        "event_id": event.event_id,
                        "decision": (
                            "accepted_corroborated_movement_opener"
                            if corroborator
                            else "deferred_uncorroborated_movement_opener"
                        ),
                        "corroborating_event_id": (
                            corroborator.event_id if corroborator else None
                        ),
                        "shared_objects": shared_objects,
                    },
                )
            )
    return sorted(
        [*canonical, *corroborated_movement],
        key=event_sort_key,
    )


def normalize_experiment_segments(
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    config: dict[str, Any],
    decision_receipts: list[dict[str, Any]] | None = None,
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
            key=event_sort_key,
        )
        anchors = select_formal_experiment_start_events(core_events, config)
        if not anchors:
            return segment
        bounded_start = max(
            0.0,
            min(event.global_start_ms for event in anchors)
            - float(segmentation["experiment_pre_roll_seconds"]) * 1000.0,
        )
        if bool(segmentation.get("accepted_leading_context_enabled", False)):
            anchor_start_ms = min(
                event.global_start_ms for event in anchors
            )
            maximum_gap_ms = float(
                segmentation.get(
                    "accepted_leading_context_max_gap_seconds", 10.0
                )
            ) * 1000.0
            maximum_extension_ms = float(
                segmentation.get(
                    "accepted_leading_context_max_extension_seconds", 90.0
                )
            ) * 1000.0
            lower_limit_ms = max(0.0, anchor_start_ms - maximum_extension_ms)
            cursor_ms = anchor_start_ms
            leading_context: list[EvidenceEvent] = []
            eligible_leading = [
                event
                for event in core_events
                if event.accepted
                and event.global_start_ms < anchor_start_ms
                and event.global_end_ms >= lower_limit_ms
                and str(
                    ((event.state_machine or {}).get("publication") or {}).get(
                        "status"
                    )
                    or "primary"
                )
                == "primary"
                and not bool(
                    (
                        (event.observability or {}).get(
                            "semantic_recall_admission"
                        )
                        or {}
                    ).get("mandatory_semantic_review")
                )
            ]
            for event in sorted(
                eligible_leading,
                key=lambda item: (item.global_end_ms, item.global_start_ms),
                reverse=True,
            ):
                if event.global_start_ms >= cursor_ms:
                    continue
                if event.global_end_ms < cursor_ms - maximum_gap_ms:
                    break
                leading_context.append(event)
                cursor_ms = min(cursor_ms, event.global_start_ms)
            if leading_context:
                bounded_start = min(
                    bounded_start,
                    max(
                        0.0,
                        min(event.global_start_ms for event in leading_context)
                        - float(segmentation["experiment_pre_roll_seconds"])
                        * 1000.0,
                    ),
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
        for context in sorted(events, key=event_sort_key):
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
        key=lambda item: segment_sort_key(item, by_event),
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
        blocking_transition_evidence = any(
            event.action_type in blocking_actions
            or bool(set(event.objects) & FIXED_EQUIPMENT_OBJECTS)
            for event in [*left_events, *right_events]
        )
        left_objects = {
            obj for event in left_events for obj in event.objects
        } & CONTINUITY_OBJECTS
        right_objects = {
            obj for event in right_events for obj in event.objects
        } & CONTINUITY_OBJECTS
        shared_object_labels = sorted(left_objects & right_objects)
        minimum_shared_objects = max(
            1,
            int(continuity_cfg.get("atomic_fragment_min_shared_objects", 2)),
        )
        shared_identities = sorted(
            _segment_track_identities(left, by_event)
            & _segment_track_identities(right, by_event)
        )
        minimum_shared_identities = max(
            1,
            int(
                continuity_cfg.get(
                    "atomic_fragment_min_shared_object_identities", 1
                )
            ),
        )
        maximum_events_per_fragment = max(
            1,
            int(
                continuity_cfg.get(
                    "atomic_fragment_max_events_per_side", 3
                )
            ),
        )
        fragment_size_within_limit = (
            len(left_events) <= maximum_events_per_fragment
            and len(right_events) <= maximum_events_per_fragment
        )
        primary_action_names = {
            str(value)
            for value in continuity_cfg.get(
                "atomic_fragment_repeated_primary_actions",
                [ActionType.LIQUID_MOVEMENT.value],
            )
        }
        left_primary_actions = {
            event.action_type.value
            for event in left_events
            if event.action_type.value in primary_action_names
        }
        right_primary_actions = {
            event.action_type.value
            for event in right_events
            if event.action_type.value in primary_action_names
        }
        repeated_primary_actions = sorted(
            left_primary_actions & right_primary_actions
        )
        raw_inactive_gap_ms = (
            min(event.global_start_ms for event in right_events)
            - max(event.global_end_ms for event in left_events)
        )
        sequence_split_minimum_gap_ms = float(
            continuity_cfg.get("atomic_sequence_split_min_gap_seconds", 8.0)
        ) * 1000.0
        same_unsplit_sequence = bool(
            repeated_primary_actions
            and raw_inactive_gap_ms < sequence_split_minimum_gap_ms
        )
        complete_action_sequences = (
            not fragment_size_within_limit
            and bool(repeated_primary_actions)
            and not same_unsplit_sequence
        )
        accepted = (
            has_shared_dual_view(left, right)
            and (
                same_unsplit_sequence
                or (
                    not blocking_transition_evidence
                    and not complete_action_sequences
                    and len(shared_object_labels) >= minimum_shared_objects
                    and len(shared_identities) >= minimum_shared_identities
                )
            )
        )
        if decision_receipts is not None:
            decision_receipts.append(
                decision_receipt(
                    decision_type="atomic_fragment_merge",
                    rule_id="QF2-STABLE-OBJECT-IDENTITY",
                    verdict="merged" if accepted else "rejected",
                    subject_ids=[left.segment_id, right.segment_id],
                    reason_codes=[
                        "repeated_primary_sequence_below_split_gap"
                        if accepted and same_unsplit_sequence
                        else "stable_object_identity_proven"
                        if accepted
                        else "complete_action_sequences_not_fragments"
                        if not fragment_size_within_limit
                        else "class_overlap_without_stable_identity"
                    ],
                    facts={
                        "gap_ms": gap_ms,
                        "left_event_count": len(left_events),
                        "right_event_count": len(right_events),
                        "fragment_size_within_limit": fragment_size_within_limit,
                        "repeated_primary_actions": repeated_primary_actions,
                        "raw_inactive_gap_ms": raw_inactive_gap_ms,
                        "same_unsplit_sequence": same_unsplit_sequence,
                        "blocking_transition_evidence": (
                            blocking_transition_evidence
                        ),
                        "complete_action_sequences": complete_action_sequences,
                        "shared_object_labels": shared_object_labels,
                        "shared_track_identities": [
                            {
                                "view_id": view_id,
                                "object": object_name,
                                "track_id": track_id,
                            }
                            for view_id, object_name, track_id in shared_identities
                        ],
                    },
                    thresholds={
                        "maximum_gap_ms": maximum_gap_ms,
                        "minimum_shared_object_labels": minimum_shared_objects,
                        "minimum_shared_object_identities": minimum_shared_identities,
                        "maximum_events_per_fragment_side": (
                            maximum_events_per_fragment
                        ),
                        "sequence_split_minimum_gap_ms": (
                            sequence_split_minimum_gap_ms
                        ),
                    },
                    evidence_refs=[
                        *left.event_ids,
                        *right.event_ids,
                    ],
                    legacy={
                        "left_segment_id": left.segment_id,
                        "right_segment_id": right.segment_id,
                        "decision": (
                            "merged_atomic_fragment"
                            if accepted
                            else "rejected_atomic_fragment_merge"
                        ),
                    },
                )
            )
        return accepted

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
    # Evidence absent from every raw segment was clipped by the conservative
    # opener rule, rather than explicitly quarantined as its own raw segment.
    source_segment_event_ids = {
        event_id for segment in segments for event_id in segment.event_ids
    }
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
            if _candidate_recall_bounds(window)[0]
            <= midpoint
            <= _candidate_recall_bounds(window)[1]
        }

    def event_boundary_ids(event: EvidenceEvent) -> set[str]:
        return {
            window.candidate_id
            for window in coarse_windows
            if _candidate_recall_bounds(window)[1] >= event.global_start_ms
            and _candidate_recall_bounds(window)[0] <= event.global_end_ms
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
            _candidate_recall_bounds(window)[0] for window in shared_windows
        ) + context_guard_ms
        coarse_context_end_ms = min(
            _candidate_recall_bounds(window)[1] for window in shared_windows
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
        segments,
        key=lambda item: segment_sort_key(item, by_event),
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
            segment_event_items = _segment_events(segment, by_event)
            opener_receipts: list[dict[str, Any]] = []
            formal_openers = select_formal_experiment_start_events(
                segment_event_items,
                config,
                decision_receipts=opener_receipts,
            )
            if not formal_openers:
                semantic_review_openers = [
                    event
                    for event in segment_event_items
                    if isinstance(
                        (event.observability or {}).get(
                            "semantic_recall_admission"
                        ),
                        dict,
                    )
                    and (event.observability or {})[
                        "semantic_recall_admission"
                    ].get("mandatory_semantic_review")
                    is True
                ]
                required_semantic_events = [
                    event
                    for event in segment_event_items
                    if event.accepted
                    and str(
                        (event.observability or {}).get(
                            "semantic_review_priority"
                        )
                        or ""
                    )
                    == "required"
                    and (event.observability or {}).get(
                        "can_cv_directly_prove_action"
                    )
                    is False
                ]
                movement_review_minimum_confidence = float(
                    segmentation.get(
                        "semantic_movement_opener_min_confidence", 0.55
                    )
                )
                provisional_movement_openers = [
                    event
                    for event in segment_event_items
                    if event.accepted
                    and event.action_type == ActionType.OBJECT_MOVEMENT
                    and required_roles.issubset(set(event.supporting_roles))
                    and str(
                        (event.observability or {}).get(
                            "cv_ontology_support"
                        )
                        or ""
                    )
                    == "direct_cv"
                    and str(
                        (event.state_machine or {}).get("lifecycle_state")
                        or ""
                    )
                    == "completed"
                    and event.confidence
                    >= movement_review_minimum_confidence
                ]
                semantic_review_requires_dual_movement = bool(
                    segmentation.get(
                        "semantic_review_requires_dual_role_movement_anchor",
                        False,
                    )
                )
                if required_semantic_events and provisional_movement_openers:
                    # Small closures are a known closed-set blind spot.  A
                    # completed cross-view movement proves real manipulation,
                    # while the independent required-review event makes the
                    # action class explicitly unresolved.  Admit the segment
                    # only to the real semantic review stage; this is not a
                    # final-action confirmation and every downstream quality
                    # and participant-visibility gate remains fail-closed.
                    semantic_review_openers = list(
                        {
                            event.event_id: event
                            for event in [
                                *semantic_review_openers,
                                *provisional_movement_openers,
                            ]
                        }.values()
                    )
                elif semantic_review_requires_dual_movement:
                    # A same-time opposite-role activity candidate is useful
                    # recall evidence, but it is not proof that the proposed
                    # state/liquid/panel action belongs to a real experiment.
                    # Production therefore requires an independently completed
                    # dual-role displacement before spending semantic calls on
                    # a provisional-only segment.  All hypotheses stay in the
                    # audit ledger; only formal promotion is quarantined.
                    semantic_review_openers = []
                if semantic_review_openers:
                    promoted.append(segment)
                    previous_input_attachable = True
                    receipts.extend(opener_receipts)
                    receipts.append(
                        {
                            "segment_id": segment.segment_id,
                            "decision": (
                                "promoted_provisional_semantic_review"
                            ),
                            "roles": sorted(
                                role.value for role in current_roles
                            ),
                            "event_ids": list(segment.event_ids),
                            "semantic_review_opener_event_ids": [
                                event.event_id
                                for event in semantic_review_openers
                            ],
                            "required_semantic_event_ids": [
                                event.event_id
                                for event in required_semantic_events
                            ],
                            "provisional_movement_opener_event_ids": [
                                event.event_id
                                for event in provisional_movement_openers
                            ],
                            "provisional_movement_min_confidence": (
                                movement_review_minimum_confidence
                            ),
                            "semantic_review_requires_dual_role_movement_anchor": (
                                semantic_review_requires_dual_movement
                            ),
                            "candidate_action_directly_confirmed": False,
                            "formal_delivery_requires_semantic_confirmation": (
                                True
                            ),
                        }
                    )
                    index += 1
                    continue
                receipts.extend(opener_receipts)
                receipts.append(
                    {
                        "segment_id": segment.segment_id,
                        "decision": "quarantined_missing_corroborated_opener",
                        "roles": sorted(role.value for role in current_roles),
                        "event_ids": list(segment.event_ids),
                        "required_semantic_event_ids": [
                            event.event_id for event in required_semantic_events
                        ],
                        "provisional_movement_opener_event_ids": [
                            event.event_id
                            for event in provisional_movement_openers
                        ],
                        "semantic_review_requires_dual_role_movement_anchor": (
                            semantic_review_requires_dual_movement
                        ),
                    }
                )
                previous_input_attachable = False
                index += 1
                continue
            promoted.append(segment)
            previous_input_attachable = True
            receipts.append(
                {
                    "segment_id": segment.segment_id,
                    "decision": "promoted_dual_view",
                    "roles": sorted(role.value for role in current_roles),
                }
            )
            receipts.extend(opener_receipts)
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
        tail_context_max_gap_ms = (
            float(
                segmentation.get("boundary_tail_context_max_gap_seconds", 30.0)
            )
            * 1000.0
        )
        tail_context_min_shared_objects = max(
            2,
            int(
                segmentation.get(
                    "boundary_tail_context_min_shared_objects", 2
                )
            ),
        )
        eligible_boundary_tail_context = bool(
            previous
            and len(current_roles) == 1
            and gap_ms is not None
            and 0.0 <= gap_ms <= tail_context_max_gap_ms
            and extension_ms is not None
            and extension_ms <= maximum_extension_ms
            and shared_boundaries
            and len(shared_objects) >= tail_context_min_shared_objects
            and any(event.accepted for event in _segment_events(segment, by_event))
        )
        if eligible_boundary_tail_context:
            promoted[-1] = previous.model_copy(
                update={
                    "global_end_ms": max(
                        previous.global_end_ms, segment.global_end_ms
                    )
                }
            )
            receipts.append(
                decision_receipt(
                    decision_type="trailing_boundary_context",
                    rule_id="QF1-SINGLE-VIEW-BOUNDARY-CONTEXT",
                    verdict="accepted",
                    subject_ids=[previous.segment_id, segment.segment_id],
                    reason_codes=["same_boundary_multi_object_tail_continuity"],
                    facts={
                        "context_segment_id": segment.segment_id,
                        "context_roles": sorted(
                            role.value for role in current_roles
                        ),
                        "gap_ms": gap_ms,
                        "extension_ms": extension_ms,
                        "shared_boundary_ids": sorted(shared_boundaries),
                        "shared_non_hand_objects": sorted(shared_objects),
                        "context_event_ids": list(segment.event_ids),
                        "formal_membership_changed": False,
                    },
                    thresholds={
                        "maximum_gap_ms": tail_context_max_gap_ms,
                        "minimum_shared_objects": tail_context_min_shared_objects,
                        "maximum_extension_ms": maximum_extension_ms,
                    },
                    evidence_refs=[*previous.event_ids, *segment.event_ids],
                    legacy={
                        "segment_id": segment.segment_id,
                        "decision": "extended_boundary_from_single_view_tail_context",
                        "attached_to_segment_id": previous.segment_id,
                    },
                )
            )
            previous_input_attachable = True
            index += 1
            continue
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

    pre_roll_ms = (
        float(segmentation.get("experiment_pre_roll_seconds", 2.0)) * 1000.0
    )
    extended: list[ExperimentSegment] = []
    for segment in promoted:
        formal_events = sorted(
            _segment_events(segment, by_event),
            key=event_sort_key,
        )
        openers = select_formal_experiment_start_events(formal_events, config)
        if not openers:
            extended.append(segment)
            continue
        anchor = openers[0]
        segment_boundaries = boundary_ids(segment) | event_boundary_ids(anchor)
        considered = [
            event
            for event in events
            if event.accepted
            and event.event_id not in segment.event_ids
            and len(set(event.supporting_roles)) == 1
            and 0.0
            <= anchor.global_start_ms - event.global_end_ms
            <= maximum_gap_ms
            and bool(segment_boundaries & event_boundary_ids(event))
        ]
        eligible: list[EvidenceEvent] = []
        for context in sorted(considered, key=event_sort_key):
            shared_objects = sorted(
                _non_actor_objects(context) & _non_actor_objects(anchor)
            )
            same_action = context.action_type == anchor.action_type
            confident = context.confidence >= float(
                segmentation.get("boundary_context_min_confidence", 0.65)
            )
            accepted_context = bool(shared_objects and same_action and confident)
            if accepted_context:
                eligible.append(context)
            receipts.append(
                decision_receipt(
                    decision_type="leading_boundary_context",
                    rule_id="QF1-SINGLE-VIEW-BOUNDARY-CONTEXT",
                    verdict="accepted" if accepted_context else "rejected",
                    subject_ids=[segment.segment_id, context.event_id, anchor.event_id],
                    reason_codes=[
                        "same_boundary_action_object_continuity"
                        if accepted_context
                        else "insufficient_action_object_confidence_continuity"
                    ],
                    facts={
                        "context_event_id": context.event_id,
                        "anchor_event_id": anchor.event_id,
                        "context_roles": sorted(
                            role.value for role in set(context.supporting_roles)
                        ),
                        "gap_ms": anchor.global_start_ms - context.global_end_ms,
                        "same_action_type": same_action,
                        "shared_non_hand_objects": shared_objects,
                        "context_confidence": context.confidence,
                        "shared_boundary_ids": sorted(
                            segment_boundaries & event_boundary_ids(context)
                        ),
                        "formal_membership_changed": False,
                    },
                    thresholds={
                        "maximum_gap_ms": maximum_gap_ms,
                        "minimum_confidence": float(
                            segmentation.get(
                                "boundary_context_min_confidence", 0.65
                            )
                        ),
                        "maximum_extension_ms": maximum_extension_ms,
                    },
                    evidence_refs=[context.event_id, anchor.event_id],
                    legacy={
                        "segment_id": segment.segment_id,
                        "event_id": context.event_id,
                        "decision": (
                            "extended_boundary_from_single_view_context"
                            if accepted_context
                            else "rejected_single_view_boundary_context"
                        ),
                    },
                )
            )
        if not eligible:
            extended.append(segment)
            continue
        earliest_context = min(eligible, key=event_sort_key)
        shared_window_starts = [
            window.global_start_ms
            for window in coarse_windows
            if window.candidate_id
            in (segment_boundaries & event_boundary_ids(earliest_context))
        ]
        lower_boundary = max(shared_window_starts) if shared_window_starts else 0.0
        extended_start = max(
            0.0,
            lower_boundary,
            segment.global_start_ms - maximum_extension_ms,
            earliest_context.global_start_ms - pre_roll_ms,
        )
        extended.append(
            segment.model_copy(
                update={
                    "global_start_ms": min(segment.global_start_ms, extended_start)
                }
            )
        )

    if bool(segmentation.get("rejected_boundary_context_enabled", True)) and len(
        extended
    ) <= int(segmentation.get("rejected_boundary_context_max_formal_segments", 3)):
        # A dual-view core proves that an experiment exists.  Rejected
        # single-role events may still carry direct temporal evidence for its
        # preparation/cleanup, provided they form a short-gap chain inside the
        # same object-verified motion envelope.  This changes clip boundaries
        # only: rejected events never enter event_ids or key-material selection.
        minimum_context_confidence = float(
            segmentation.get("rejected_boundary_context_min_confidence", 0.30)
        )
        strong_context_confidence = float(
            segmentation.get("rejected_boundary_context_strong_confidence", 0.65)
        )
        context_gap_ms = float(
            segmentation.get("rejected_boundary_context_max_gap_seconds", 10.0)
        ) * 1000.0
        context_extension_ms = float(
            segmentation.get(
                "rejected_boundary_context_max_extension_seconds", 180.0
            )
        ) * 1000.0
        weak_fringe_ms = float(
            segmentation.get("rejected_boundary_context_weak_fringe_seconds", 10.5)
        ) * 1000.0
        evidence_overflow_ms = float(
            segmentation.get(
                "rejected_boundary_context_evidence_overflow_seconds", 15.0
            )
        ) * 1000.0
        start_snap_ms = float(
            segmentation.get("recalled_boundary_start_snap_seconds", 60.0)
        ) * 1000.0
        start_bridge_gap_ms = float(
            segmentation.get(
                "recalled_boundary_start_bridge_gap_seconds", 30.0
            )
        ) * 1000.0
        post_roll_ms = float(
            segmentation.get("experiment_post_roll_seconds", 3.0)
        ) * 1000.0
        boundary_action_types = {
            ActionType.OBJECT_MOVEMENT,
            ActionType.HAND_OBJECT_CONTACT,
            ActionType.CONTAINER_STATE_CHANGE,
        }

        def core_bounds(item: ExperimentSegment) -> tuple[float, float]:
            member_events = [
                by_event[event_id]
                for event_id in item.event_ids
                if event_id in by_event and by_event[event_id].accepted
            ]
            if not member_events:
                return item.global_start_ms, item.global_end_ms
            return (
                min(event.global_start_ms for event in member_events),
                max(event.global_end_ms for event in member_events),
            )

        def recalled_window(item: ExperimentSegment) -> ActionCandidate | None:
            core_start_ms, core_end_ms = core_bounds(item)
            midpoint = (core_start_ms + core_end_ms) / 2.0
            matches = [
                window
                for window in coarse_windows
                if _candidate_recall_bounds(window)[0]
                <= midpoint
                <= _candidate_recall_bounds(window)[1]
            ]
            if not matches:
                return None
            return min(
                matches,
                key=lambda window: (
                    abs(
                        midpoint
                        - sum(_candidate_recall_bounds(window)) / 2.0
                    ),
                    window.candidate_id,
                ),
            )

        ordered_extended = sorted(
            extended, key=lambda item: (*core_bounds(item), item.segment_id)
        )
        assignments = {
            item.segment_id: recalled_window(item) for item in ordered_extended
        }
        context_extended: list[ExperimentSegment] = []
        for position, segment in enumerate(ordered_extended):
            window = assignments[segment.segment_id]
            if window is None:
                context_extended.append(segment)
                continue
            verified_boundary_objects = (
                set(window.objects)
                | {
                    obj
                    for event in _segment_events(segment, by_event)
                    if event.accepted
                    for obj in event.objects
                }
            ) - ACTOR_OBJECTS
            if not verified_boundary_objects:
                context_extended.append(segment)
                continue
            window_recall_start_ms, window_recall_end_ms = (
                _candidate_recall_bounds(window)
            )
            core_start_ms, core_end_ms = core_bounds(segment)
            peers = [
                item
                for item in ordered_extended
                if assignments[item.segment_id] is not None
                and assignments[item.segment_id].candidate_id == window.candidate_id
            ]
            peer_index = next(
                index
                for index, item in enumerate(peers)
                if item.segment_id == segment.segment_id
            )
            previous_core_end_ms = (
                core_bounds(peers[peer_index - 1])[1]
                if peer_index > 0
                else None
            )
            next_core_start_ms = (
                core_bounds(peers[peer_index + 1])[0]
                if peer_index + 1 < len(peers)
                else None
            )
            lower_limit_ms = max(
                0.0,
                core_start_ms - context_extension_ms,
                window_recall_start_ms - evidence_overflow_ms,
                previous_core_end_ms if previous_core_end_ms is not None else 0.0,
            )
            upper_limit_ms = min(
                core_end_ms + context_extension_ms,
                window_recall_end_ms + evidence_overflow_ms,
                next_core_start_ms
                if next_core_start_ms is not None
                else float("inf"),
            )
            eligible = [
                event
                for event in events
                if not event.accepted
                and event.event_id not in segment.event_ids
                and len(set(event.supporting_roles)) == 1
                and event.action_type in boundary_action_types
                and bool(_non_actor_objects(event))
                and event.confidence >= minimum_context_confidence
                and event.global_end_ms >= lower_limit_ms
                and event.global_start_ms <= upper_limit_ms
            ]

            leading_chain: list[EvidenceEvent] = []
            leading_cursor_ms = core_start_ms
            for event in sorted(
                eligible,
                key=lambda item: (item.global_end_ms, item.global_start_ms),
                reverse=True,
            ):
                if event.global_start_ms >= leading_cursor_ms:
                    continue
                if event.global_end_ms < leading_cursor_ms - context_gap_ms:
                    break
                leading_chain.append(event)
                leading_cursor_ms = max(
                    lower_limit_ms,
                    min(leading_cursor_ms, event.global_start_ms),
                )

            trailing_chain: list[EvidenceEvent] = []
            trailing_cursor_ms = core_end_ms
            for event in sorted(
                eligible,
                key=lambda item: (item.global_start_ms, item.global_end_ms),
            ):
                if event.global_end_ms <= trailing_cursor_ms:
                    continue
                if event.global_start_ms > trailing_cursor_ms + context_gap_ms:
                    break
                trailing_chain.append(event)
                trailing_cursor_ms = min(
                    upper_limit_ms,
                    max(trailing_cursor_ms, event.global_end_ms),
                )

            strong_leading = [
                event
                for event in leading_chain
                if event.confidence >= strong_context_confidence
            ]
            strong_trailing = [
                event
                for event in trailing_chain
                if event.confidence >= strong_context_confidence
            ]
            proposed_start_ms = segment.global_start_ms
            proposed_end_ms = segment.global_end_ms
            start_source = "unchanged"
            end_source = "unchanged"

            if strong_leading:
                weak_start_ms = min(
                    event.global_start_ms for event in leading_chain
                )
                strong_start_event = min(
                    strong_leading, key=event_sort_key
                )
                if strong_start_event.action_type == ActionType.OBJECT_MOVEMENT:
                    # A tracked displacement has a crisp onset. It does not
                    # justify the weak fringe or generic clip pre-roll that a
                    # hand/contact sequence can legitimately require.
                    supported_start_ms = strong_start_event.global_start_ms
                    supported_pre_roll_ms = 0.0
                else:
                    supported_start_ms = max(
                        weak_start_ms,
                        strong_start_event.global_start_ms - weak_fringe_ms,
                    )
                    supported_pre_roll_ms = pre_roll_ms
                proposed_start_ms = min(
                    proposed_start_ms,
                    max(
                        lower_limit_ms,
                        supported_start_ms - supported_pre_roll_ms,
                    ),
                )
                start_source = "rejected_direct_cv_chain"

            near_window_start = [
                event
                for event in eligible
                if window_recall_start_ms <= event.global_start_ms
                and event.global_start_ms - window_recall_start_ms <= start_snap_ms
            ]
            start_bridge_cursor_ms = max(
                (event.global_end_ms for event in near_window_start),
                default=window_recall_start_ms,
            )
            if near_window_start:
                for event in sorted(eligible, key=event_sort_key):
                    if event.global_end_ms <= start_bridge_cursor_ms:
                        continue
                    if (
                        event.global_start_ms
                        > start_bridge_cursor_ms + start_bridge_gap_ms
                    ):
                        break
                    start_bridge_cursor_ms = max(
                        start_bridge_cursor_ms, event.global_end_ms
                    )
            start_bridge_reaches_core = bool(
                near_window_start
                and core_start_ms
                <= start_bridge_cursor_ms + start_bridge_gap_ms
            )
            if (
                peer_index == 0
                and start_bridge_reaches_core
                and core_start_ms - window_recall_start_ms >= start_snap_ms
                and core_start_ms - window_recall_start_ms <= context_extension_ms
            ):
                proposed_start_ms = min(
                    proposed_start_ms, max(0.0, window_recall_start_ms)
                )
                start_source = "object_verified_recalled_window_start"

            if strong_trailing:
                strong_end_ms = max(
                    event.global_end_ms for event in strong_trailing
                )
                # Weak observations can prove continuity between strong
                # events, but the final boundary itself must end on strong
                # direct evidence. This blocks long first-person-only tails.
                supported_end_ms = strong_end_ms
                end_source = "rejected_direct_cv_chain"
                proposed_end_ms = max(
                    proposed_end_ms,
                    min(upper_limit_ms, supported_end_ms) + post_roll_ms,
                )

            changed = bool(
                proposed_start_ms < segment.global_start_ms - 1e-6
                or proposed_end_ms > segment.global_end_ms + 1e-6
            )
            if changed:
                segment = segment.model_copy(
                    update={
                        "global_start_ms": proposed_start_ms,
                        "global_end_ms": proposed_end_ms,
                    }
                )
                receipts.append(
                    decision_receipt(
                        decision_type="formal_boundary_context_extension",
                        rule_id="QF1-REJECTED-DIRECT-CV-BOUNDARY-CONTEXT",
                        verdict="accepted",
                        subject_ids=[segment.segment_id],
                        reason_codes=[
                            "same_recalled_window_short_gap_direct_cv_chain"
                        ],
                        facts={
                            "previous_start_ms": next(
                                item.global_start_ms
                                for item in ordered_extended
                                if item.segment_id == segment.segment_id
                            ),
                            "previous_end_ms": next(
                                item.global_end_ms
                                for item in ordered_extended
                                if item.segment_id == segment.segment_id
                            ),
                            "extended_start_ms": proposed_start_ms,
                            "extended_end_ms": proposed_end_ms,
                            "start_source": start_source,
                            "end_source": end_source,
                            "leading_context_event_ids": [
                                event.event_id for event in leading_chain
                            ],
                            "trailing_context_event_ids": [
                                event.event_id for event in trailing_chain
                            ],
                            "strong_leading_event_ids": [
                                event.event_id for event in strong_leading
                            ],
                            "strong_trailing_event_ids": [
                                event.event_id for event in strong_trailing
                            ],
                            "boundary_candidate_id": window.candidate_id,
                            "boundary_candidate_objects": sorted(window.objects),
                            "boundary_verified_objects": sorted(
                                verified_boundary_objects
                            ),
                            "boundary_object_verification_source": (
                                "coarse_candidate_and_formal_dual_events"
                                if window.objects
                                else "formal_dual_events_after_sustained_motion_recall"
                            ),
                            "boundary_candidate_refined_start_ms": (
                                window.global_start_ms
                            ),
                            "boundary_candidate_refined_end_ms": (
                                window.global_end_ms
                            ),
                            "boundary_candidate_recall_start_ms": (
                                window_recall_start_ms
                            ),
                            "boundary_candidate_recall_end_ms": (
                                window_recall_end_ms
                            ),
                            "formal_membership_changed": False,
                        },
                        thresholds={
                            "minimum_context_confidence": minimum_context_confidence,
                            "strong_context_confidence": strong_context_confidence,
                            "maximum_context_gap_ms": context_gap_ms,
                            "maximum_extension_ms": context_extension_ms,
                            "weak_fringe_ms": weak_fringe_ms,
                            "start_snap_ms": start_snap_ms,
                            "start_bridge_gap_ms": start_bridge_gap_ms,
                        },
                        evidence_refs=[
                            *segment.event_ids,
                            *[event.event_id for event in leading_chain],
                            *[event.event_id for event in trailing_chain],
                        ],
                        legacy={
                            "segment_id": segment.segment_id,
                            "decision": "extended_boundary_from_rejected_cv_context",
                        },
                    )
                )
            context_extended.append(segment)
        extended = context_extended

    if bool(
        segmentation.get("recalled_boundary_membership_recovery_enabled", True)
    ):
        # A recalled boundary can safely move a formal clip earlier after the
        # raw opener filter removed preceding events. Recover only evidence CV
        # already accepted, absent from all raw segments, and connected to the
        # dual-view core by a short-gap chain inside the same recall envelope.
        # Explicitly quarantined raw-segment evidence remains excluded.
        allowed_action_types = {
            ActionType(str(value))
            for value in segmentation.get(
                "recalled_boundary_membership_action_types",
                [
                    ActionType.HAND_OBJECT_CONTACT.value,
                    ActionType.OBJECT_MOVEMENT.value,
                    ActionType.CONTAINER_STATE_CHANGE.value,
                    ActionType.DEVICE_PANEL_OPERATION.value,
                ],
            )
        }
        maximum_step_gap_ms = float(
            segmentation.get(
                "recalled_boundary_membership_max_step_gap_seconds", 30.0
            )
        ) * 1000.0
        single_role_minimum_confidence = float(
            segmentation.get(
                "recalled_boundary_membership_single_role_min_confidence", 0.78
            )
        )
        existing_formal_event_ids = {
            event_id for segment in extended for event_id in segment.event_ids
        }
        membership_recovered: list[ExperimentSegment] = []
        for segment in extended:
            member_events = sorted(
                _segment_events(segment, by_event), key=event_sort_key
            )
            if not member_events:
                membership_recovered.append(segment)
                continue
            core_start_ms = min(event.global_start_ms for event in member_events)
            shared_boundary_ids = boundary_ids(segment)
            eligible = [
                event
                for event in events
                if event.accepted
                and event.event_id not in source_segment_event_ids
                and event.event_id not in existing_formal_event_ids
                and event.action_type in allowed_action_types
                and segment.global_start_ms <= event.global_start_ms
                and event.global_end_ms <= core_start_ms
                and bool(shared_boundary_ids & event_boundary_ids(event))
                and (
                    required_roles.issubset(set(event.supporting_roles))
                    or event.confidence >= single_role_minimum_confidence
                )
            ]
            recovered_chain: list[EvidenceEvent] = []
            cursor_ms = core_start_ms
            for event in sorted(
                eligible,
                key=lambda item: (item.global_end_ms, item.global_start_ms),
                reverse=True,
            ):
                if event.global_start_ms >= cursor_ms:
                    continue
                if event.global_end_ms < cursor_ms - maximum_step_gap_ms:
                    break
                recovered_chain.append(event)
                cursor_ms = min(cursor_ms, event.global_start_ms)
            if not recovered_chain:
                membership_recovered.append(segment)
                continue

            recovered_chain.sort(key=event_sort_key)
            recovered_ids = [event.event_id for event in recovered_chain]
            combined_ids = [
                event.event_id
                for event in sorted(
                    [*recovered_chain, *member_events], key=event_sort_key
                )
            ]
            recovered_micro = [
                {
                    "micro_segment_id": (
                        f"RECOVERED-{segment.segment_id}-{position + 1:04d}"
                    ),
                    "start_global_ms": event.global_start_ms,
                    "end_global_ms": event.global_end_ms,
                    "action_type": event.action_type.value,
                    "objects": event.objects,
                    "evidence_event_id": event.event_id,
                    "view_alignment_state": "aligned",
                    "next_event_id": None,
                    "uncertainty": event.uncertainty,
                }
                for position, event in enumerate(recovered_chain)
            ]
            combined_micro = sorted(
                [
                    *recovered_micro,
                    *(dict(item) for item in segment.micro_segments),
                ],
                key=lambda item: float(item.get("start_global_ms", 0.0)),
            )
            for position, item in enumerate(combined_micro):
                item["next_event_id"] = (
                    combined_micro[position + 1].get("evidence_event_id")
                    if position + 1 < len(combined_micro)
                    else None
                )
            recovered_views = sorted(
                {
                    *segment.participating_views,
                    *(
                        view_id
                        for event in recovered_chain
                        for view_id in event.supporting_views
                    ),
                }
            )
            recovered_segment = segment.model_copy(
                update={
                    "event_ids": combined_ids,
                    "participating_views": recovered_views,
                    "rejected_views": {
                        key: value
                        for key, value in segment.rejected_views.items()
                        if key not in recovered_views
                    },
                    "micro_segments": combined_micro,
                }
            )
            membership_recovered.append(recovered_segment)
            existing_formal_event_ids.update(recovered_ids)
            receipts.append(
                decision_receipt(
                    decision_type="recalled_boundary_membership_recovery",
                    rule_id=(
                        "QF1-ACCEPTED-EVIDENCE-INSIDE-VERIFIED-RECALL-BOUNDARY"
                    ),
                    verdict="accepted",
                    subject_ids=[segment.segment_id, *recovered_ids],
                    reason_codes=[
                        "accepted_events_clipped_by_opener_recovered_as_short_gap_chain"
                    ],
                    facts={
                        "previous_event_ids": list(segment.event_ids),
                        "recovered_event_ids": recovered_ids,
                        "recovered_event_fingerprints": [
                            stable_event_fingerprint(event)
                            for event in recovered_chain
                        ],
                        "final_event_ids": combined_ids,
                        "shared_boundary_ids": sorted(shared_boundary_ids),
                        "formal_membership_changed": True,
                        "rejected_events_promoted": False,
                        "raw_segment_quarantine_bypassed": False,
                    },
                    thresholds={
                        "maximum_step_gap_ms": maximum_step_gap_ms,
                        "single_role_minimum_confidence": (
                            single_role_minimum_confidence
                        ),
                        "allowed_action_types": sorted(
                            action.value for action in allowed_action_types
                        ),
                    },
                    evidence_refs=[*segment.event_ids, *recovered_ids],
                    legacy={
                        "segment_id": segment.segment_id,
                        "decision": (
                            "recovered_accepted_recall_boundary_membership"
                        ),
                    },
                )
            )
        extended = membership_recovered

    normalized_receipts: list[dict[str, Any]] = []
    for item in receipts:
        if item.get("receipt_schema_version"):
            normalized_receipts.append(item)
            continue
        segment_id = str(item.get("segment_id") or "unknown-segment")
        decision = str(item.get("decision") or "unspecified")
        normalized_receipts.append(
            decision_receipt(
                decision_type="formal_segment_membership",
                rule_id="FORMAL-DUAL-VIEW-MEMBERSHIP",
                verdict=(
                    "accepted"
                    if decision.startswith(("promoted", "attached", "merged"))
                    else "quarantined"
                ),
                subject_ids=[segment_id],
                reason_codes=[decision],
                facts={
                    key: value
                    for key, value in item.items()
                    if key not in {"segment_id", "decision"}
                },
                evidence_refs=item.get("event_ids") or [],
                legacy=item,
            )
        )
    return extended, normalized_receipts


def _continuity_evidence(
    left: ExperimentSegment,
    right: ExperimentSegment,
    by_event: dict[str, EvidenceEvent],
    roles: dict[str, ViewRole],
    config: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    cfg = config["continuity"]
    gap_ms = right.global_start_ms - left.global_end_ms
    max_gap_ms = float(cfg["max_gap_seconds"]) * 1000.0
    if gap_ms < 0:
        return True, "原子实验边界重叠，属于同一连续活动链", {
            "gap_ms": gap_ms,
            "continuity_basis": "temporal_overlap",
            "shared_object_labels": [],
            "shared_track_identities": [],
        }
    if gap_ms > max_gap_ms:
        return False, f"间隔 {gap_ms / 1000.0:.1f}s 超过连续实验阈值 {max_gap_ms / 1000.0:.1f}s", {
            "gap_ms": gap_ms,
            "continuity_basis": "gap_exceeded",
            "shared_object_labels": [],
            "shared_track_identities": [],
        }
    stable_identity_max_gap_ms = float(
        cfg.get("stable_identity_max_gap_seconds", cfg["max_gap_seconds"])
    ) * 1000.0
    if gap_ms > stable_identity_max_gap_ms:
        # A tracker can retain the ID of a bottle, rack, or balance that sits
        # motionless on the bench for minutes.  Identity persistence alone is
        # therefore not a cross-workflow continuity proof.  Longer gaps remain
        # eligible for the independent quarantined-context-chain rule below.
        return False, (
            f"间隔 {gap_ms / 1000.0:.1f}s 超过稳定身份直连阈值 "
            f"{stable_identity_max_gap_ms / 1000.0:.1f}s，需要跨缺口动作链证明"
        ), {
            "gap_ms": gap_ms,
            "continuity_basis": "identity_gap_requires_context_chain",
            "stable_identity_maximum_gap_ms": stable_identity_max_gap_ms,
            "shared_object_labels": [],
            "shared_track_identities": [],
        }

    shared_views = sorted(set(left.participating_views) & set(right.participating_views))
    shared_third = [view for view in shared_views if roles.get(view) == ViewRole.THIRD_PERSON]
    if bool(cfg.get("require_shared_third_view", True)) and not shared_third:
        return False, "前后原子实验没有共同的有效第三人称证据视角", {
            "gap_ms": gap_ms,
            "continuity_basis": "missing_shared_third_view",
            "shared_views": shared_views,
            "shared_object_labels": [],
            "shared_track_identities": [],
        }

    left_events = _segment_events(left, by_event)
    right_events = _segment_events(right, by_event)
    left_objects = {obj for event in left_events for obj in event.objects} & CONTINUITY_OBJECTS
    right_objects = {obj for event in right_events for obj in event.objects} & CONTINUITY_OBJECTS
    shared_objects = sorted(left_objects & right_objects)
    if len(shared_objects) < int(cfg.get("minimum_shared_objects", 1)):
        return False, "时间接近但没有足够的容器/样品对象承接，判定为独立实验", {
            "gap_ms": gap_ms,
            "continuity_basis": "insufficient_object_labels",
            "shared_views": shared_views,
            "shared_object_labels": shared_objects,
            "shared_track_identities": [],
        }
    shared_identities = sorted(
        _segment_track_identities(left, by_event)
        & _segment_track_identities(right, by_event)
    )
    minimum_shared_identities = max(
        1, int(cfg.get("minimum_shared_object_identities", 1))
    )
    if len(shared_identities) < minimum_shared_identities:
        return False, "对象类别相同但缺少稳定轨迹身份承接，判定为独立实验", {
            "gap_ms": gap_ms,
            "continuity_basis": "class_overlap_without_stable_identity",
            "shared_views": shared_views,
            "shared_object_labels": shared_objects,
            "shared_track_identities": [],
        }
    return (
        True,
        f"间隔 {gap_ms / 1000.0:.1f}s，共同视角={shared_views}，稳定对象身份={shared_identities}",
        {
            "gap_ms": gap_ms,
            "continuity_basis": "stable_object_identity",
            "shared_views": shared_views,
            "shared_object_labels": shared_objects,
            "shared_track_identities": [
                {
                    "view_id": view_id,
                    "object": object_name,
                    "track_id": track_id,
                }
                for view_id, object_name, track_id in shared_identities
            ],
        },
    )


def _continuity_object_families(events: Sequence[EvidenceEvent]) -> set[str]:
    """Normalize label variants used only for conservative continuity proof."""

    aliases = {
        "sample_bottle_blue": "sample_bottle",
        "reagent_bottle_open": "reagent_bottle",
    }
    return {
        aliases.get(obj, obj)
        for event in events
        for obj in event.objects
        if obj in CONTINUITY_OBJECTS
    }


def _quarantined_context_continuity_evidence(
    left: ExperimentSegment,
    right: ExperimentSegment,
    by_event: dict[str, EvidenceEvent],
    roles: dict[str, ViewRole],
    coarse_windows: Sequence[ActionCandidate],
    config: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Prove continuity through rejected FP context without promoting it.

    This is deliberately stricter than a temporal-gap merge. Both formal
    fragments must share the delivered FP/TP pair and one recalled coarse
    boundary. Rejected first-person physical events must form a bounded chain
    across the gap and carry at least two normalized object families from each
    formal fragment. The context IDs are written to the decision receipt only;
    they never become formal group members or key materials.
    """

    cfg = config["continuity"]
    gap_ms = right.global_start_ms - left.global_end_ms
    maximum_span_ms = float(
        cfg.get("quarantined_atomic_bridge_max_span_seconds", 180.0)
    ) * 1000.0
    maximum_step_ms = float(
        cfg.get("quarantined_atomic_bridge_max_step_gap_seconds", 60.0)
    ) * 1000.0
    minimum_shared_objects = max(
        2, int(cfg.get("quarantined_context_bridge_min_shared_objects", 2))
    )
    minimum_context_events = max(
        2, int(cfg.get("quarantined_context_bridge_min_events", 2))
    )
    shared_views = set(left.participating_views) & set(right.participating_views)
    shared_first = sorted(
        view_id for view_id in shared_views if roles.get(view_id) == ViewRole.FIRST_PERSON
    )
    shared_third = sorted(
        view_id for view_id in shared_views if roles.get(view_id) == ViewRole.THIRD_PERSON
    )

    def boundary_ids(segment: ExperimentSegment) -> set[str]:
        midpoint = (segment.global_start_ms + segment.global_end_ms) / 2.0
        return {
            window.candidate_id
            for window in coarse_windows
            if window.global_start_ms <= midpoint <= window.global_end_ms
        }

    shared_boundaries = sorted(boundary_ids(left) & boundary_ids(right))
    eligible_actions = {
        ActionType.HAND_OBJECT_CONTACT,
        ActionType.LIQUID_MOVEMENT,
        ActionType.CONTAINER_STATE_CHANGE,
        ActionType.DEVICE_PANEL_OPERATION,
    }
    context_events = sorted(
        (
            event
            for event in by_event.values()
            if not event.accepted
            and ViewRole.FIRST_PERSON in event.supporting_roles
            and ViewRole.THIRD_PERSON not in event.supporting_roles
            and event.action_type in eligible_actions
            and event.global_end_ms >= left.global_end_ms
            and event.global_start_ms <= right.global_start_ms
        ),
        key=event_sort_key,
    )
    chain_gaps: list[float] = []
    if context_events:
        chain_gaps.append(context_events[0].global_start_ms - left.global_end_ms)
        prior_end = context_events[0].global_end_ms
        for event in context_events[1:]:
            chain_gaps.append(max(0.0, event.global_start_ms - prior_end))
            prior_end = max(prior_end, event.global_end_ms)
        chain_gaps.append(max(0.0, right.global_start_ms - prior_end))
    maximum_observed_step_ms = max(chain_gaps, default=float("inf"))

    left_events = _segment_events(left, by_event)
    right_events = _segment_events(right, by_event)
    left_families = _continuity_object_families(left_events)
    right_families = _continuity_object_families(right_events)
    context_families = _continuity_object_families(context_events)
    left_context_families = sorted(left_families & context_families)
    right_context_families = sorted(right_families & context_families)
    cross_boundary_enabled = bool(
        cfg.get("quarantined_cross_boundary_bridge_enabled", True)
    )
    cross_boundary_maximum_span_ms = float(
        cfg.get("quarantined_cross_boundary_bridge_max_span_seconds", 90.0)
    ) * 1000.0
    cross_boundary_maximum_step_ms = float(
        cfg.get("quarantined_cross_boundary_bridge_max_step_gap_seconds", 15.0)
    ) * 1000.0
    cross_boundary_minimum_events = max(
        4,
        int(cfg.get("quarantined_cross_boundary_bridge_min_events", 4)),
    )
    strong_cross_boundary_context = bool(
        cross_boundary_enabled
        and not shared_boundaries
        and 0.0 <= gap_ms <= cross_boundary_maximum_span_ms
        and len(context_events) >= cross_boundary_minimum_events
        and maximum_observed_step_ms <= cross_boundary_maximum_step_ms
        and len(left_context_families) >= minimum_shared_objects
        and len(right_context_families) >= minimum_shared_objects
    )
    criteria = {
        "gap_within_span": 0.0 <= gap_ms <= maximum_span_ms,
        "shared_first_person_view": bool(shared_first),
        "shared_third_person_view": bool(shared_third),
        "shared_boundary": bool(shared_boundaries),
        "shared_boundary_or_strong_cross_boundary_context": bool(
            shared_boundaries or strong_cross_boundary_context
        ),
        "enough_context_events": len(context_events) >= minimum_context_events,
        "context_step_within_limit": maximum_observed_step_ms <= maximum_step_ms,
        "left_object_family_support": (
            len(left_context_families) >= minimum_shared_objects
        ),
        "right_object_family_support": (
            len(right_context_families) >= minimum_shared_objects
        ),
    }
    accepted = all(
        value
        for key, value in criteria.items()
        if key != "shared_boundary"
    )
    facts = {
        "gap_ms": gap_ms,
        "continuity_basis": "quarantined_first_person_context_chain",
        "shared_views": sorted(shared_views),
        "shared_first_person_views": shared_first,
        "shared_third_person_views": shared_third,
        "shared_boundary_ids": shared_boundaries,
        "strong_cross_boundary_context": strong_cross_boundary_context,
        "context_event_ids": [event.event_id for event in context_events],
        "context_event_fingerprints": [
            stable_event_fingerprint(event) for event in context_events
        ],
        "context_event_count": len(context_events),
        "maximum_context_step_ms": maximum_observed_step_ms,
        "left_context_object_families": left_context_families,
        "right_context_object_families": right_context_families,
        "formal_membership_changed": False,
        "minimum_context_events": minimum_context_events,
        "minimum_shared_object_families": minimum_shared_objects,
        "maximum_step_ms": maximum_step_ms,
        "maximum_span_ms": maximum_span_ms,
        "cross_boundary_maximum_span_ms": cross_boundary_maximum_span_ms,
        "cross_boundary_maximum_step_ms": cross_boundary_maximum_step_ms,
        "cross_boundary_minimum_events": cross_boundary_minimum_events,
        "left_segment_fingerprint": stable_segment_fingerprint(left, by_event),
        "right_segment_fingerprint": stable_segment_fingerprint(right, by_event),
        "criteria": criteria,
    }
    if accepted:
        return (
            True,
            "双视角原子片段由第一人称弱证据链连续承接；弱证据仅作图边",
            facts,
        )
    return False, "弱证据上下文不足以证明连续实验", facts


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
    decision_receipts: list[dict[str, Any]] | None = None,
    coarse_windows: Sequence[ActionCandidate] | None = None,
) -> list[ExperimentGroup]:
    """Group atomic experiments only when temporal and physical continuity agree."""
    by_event = {event.event_id: event for event in events}
    ordered = sorted(
        segments,
        key=lambda segment: segment_sort_key(segment, by_event),
    )
    if not ordered:
        return []
    roles = {view.view_id: view.role for view in views}
    chains: list[tuple[list[ExperimentSegment], list[str]]] = []
    current = [ordered[0]]
    reasons: list[str] = []
    for segment in ordered[1:]:
        left = current[-1]
        base_continuous, base_reason, base_facts = _continuity_evidence(
            left, segment, by_event, roles, config
        )
        cfg = config["continuity"]
        segment_facts = {
            "left_segment_fingerprint": stable_segment_fingerprint(left, by_event),
            "right_segment_fingerprint": stable_segment_fingerprint(segment, by_event),
        }
        base_facts = {**base_facts, **segment_facts}
        if decision_receipts is not None:
            decision_receipts.append(
                decision_receipt(
                    decision_type="experiment_continuity_edge",
                    rule_id="QF2-STABLE-OBJECT-IDENTITY",
                    verdict="accepted" if base_continuous else "rejected",
                    subject_ids=[left.segment_id, segment.segment_id],
                    reason_codes=[str(base_facts.get("continuity_basis"))],
                    facts=base_facts,
                    thresholds={
                        "maximum_gap_ms": float(cfg["max_gap_seconds"]) * 1000.0,
                        "stable_identity_maximum_gap_ms": float(
                            cfg.get(
                                "stable_identity_max_gap_seconds",
                                cfg["max_gap_seconds"],
                            )
                        )
                        * 1000.0,
                        "minimum_shared_object_labels": int(
                            cfg.get("minimum_shared_objects", 1)
                        ),
                        "minimum_shared_object_identities": int(
                            cfg.get("minimum_shared_object_identities", 1)
                        ),
                        "require_shared_third_view": bool(
                            cfg.get("require_shared_third_view", True)
                        ),
                    },
                    evidence_refs=[*left.event_ids, *segment.event_ids],
                    legacy={
                        "left_segment_id": left.segment_id,
                        "right_segment_id": segment.segment_id,
                        "decision": (
                            "accepted_continuity_edge"
                            if base_continuous
                            else "rejected_continuity_edge"
                        ),
                        "reason": base_reason,
                    },
                )
            )

        continuous = base_continuous
        reason = base_reason
        if not base_continuous and coarse_windows:
            fallback_continuous, fallback_reason, fallback_facts = (
                _quarantined_context_continuity_evidence(
                    left,
                    segment,
                    by_event,
                    roles,
                    coarse_windows,
                    config,
                )
            )
            if decision_receipts is not None:
                decision_receipts.append(
                    decision_receipt(
                        decision_type="experiment_continuity_fallback",
                        rule_id="QF2-QUARANTINED-CONTEXT-CHAIN",
                        verdict=(
                            "accepted" if fallback_continuous else "rejected"
                        ),
                        subject_ids=[left.segment_id, segment.segment_id],
                        reason_codes=[
                            "quarantined_context_chain_satisfied"
                            if fallback_continuous
                            else "quarantined_context_chain_insufficient"
                        ],
                        facts=fallback_facts,
                        thresholds={
                            "maximum_span_ms": float(
                                cfg.get(
                                    "quarantined_atomic_bridge_max_span_seconds",
                                    180.0,
                                )
                            )
                            * 1000.0,
                            "maximum_step_ms": float(
                                cfg.get(
                                    "quarantined_atomic_bridge_max_step_gap_seconds",
                                    60.0,
                                )
                            )
                            * 1000.0,
                            "minimum_context_events": max(
                                2,
                                int(
                                    cfg.get(
                                        "quarantined_context_bridge_min_events",
                                        2,
                                    )
                                ),
                            ),
                            "minimum_shared_object_families": max(
                                2,
                                int(
                                    cfg.get(
                                        "quarantined_context_bridge_min_shared_objects",
                                        2,
                                    )
                                ),
                            ),
                            "require_shared_boundary_unless_strong_cross_boundary_context": True,
                            "require_shared_first_person_view": True,
                            "require_shared_third_person_view": True,
                            "cross_boundary_maximum_span_ms": float(
                                cfg.get(
                                    "quarantined_cross_boundary_bridge_max_span_seconds",
                                    90.0,
                                )
                            )
                            * 1000.0,
                            "cross_boundary_maximum_step_ms": float(
                                cfg.get(
                                    "quarantined_cross_boundary_bridge_max_step_gap_seconds",
                                    15.0,
                                )
                            )
                            * 1000.0,
                            "cross_boundary_minimum_events": max(
                                4,
                                int(
                                    cfg.get(
                                        "quarantined_cross_boundary_bridge_min_events",
                                        4,
                                    )
                                ),
                            ),
                        },
                        evidence_refs=[
                            *left.event_ids,
                            *fallback_facts.get("context_event_ids", []),
                            *segment.event_ids,
                        ],
                        legacy={
                            "left_segment_id": left.segment_id,
                            "right_segment_id": segment.segment_id,
                            "decision": (
                                "accepted_quarantined_context_bridge"
                                if fallback_continuous
                                else "rejected_quarantined_context_bridge"
                            ),
                            "reason": fallback_reason,
                        },
                    )
                )
            if fallback_continuous:
                continuous = True
                reason = fallback_reason
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
                "shared_actor_objects": [],
                "peak_distance_ms": None,
                "interval_overlap_ms": None,
                "interval_overlap_ratio": None,
                "minimum_separation_ms": separation_ms,
                "legacy_overlap_ratio_threshold_not_used": overlap_ratio,
                "requires_positive_interval_overlap": True,
                "dedup_comparisons": [],
            },
        )

    def duplicate_metrics(
        existing: EvidenceEvent,
        event: EvidenceEvent,
    ) -> dict[str, Any]:
        shared_all = set(existing.objects) & set(event.objects)
        shared_objects = sorted(shared_all - ACTOR_OBJECTS)
        shared_actor_objects = sorted(shared_all & ACTOR_OBJECTS)
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
            "shared_actor_objects": shared_actor_objects,
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
                    event_decision["dedup_comparisons"].append(
                        {
                            "competitor_event_id": existing.event_id,
                            **metrics,
                            "is_duplicate": is_duplicate,
                            "reason": (
                                "shared_non_hand_object_with_overlapping_interval"
                                if is_duplicate
                                else "missing_shared_non_hand_object_or_interval_overlap"
                            ),
                        }
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
                (
                    decision_receipt(
                        decision_type="key_material_selection",
                        rule_id="QF5-NON-HAND-OBJECT-DEDUPLICATION",
                        verdict=(
                            "selected" if item.get("selected") else "dropped"
                        ),
                        subject_ids=[
                            str(item["event_id"]),
                            *(
                                [str(item["competitor_event_id"])]
                                if item.get("competitor_event_id")
                                else []
                            ),
                        ],
                        reason_codes=[str(item.get("decision") or "unknown")],
                        facts={
                            "action_type": item.get("action_type"),
                            "objects": item.get("objects"),
                            "shared_non_hand_objects": item.get(
                                "shared_objects", []
                            ),
                            "shared_actor_objects": item.get(
                                "shared_actor_objects", []
                            ),
                            "peak_distance_ms": item.get("peak_distance_ms"),
                            "interval_overlap_ms": item.get(
                                "interval_overlap_ms"
                            ),
                            "interval_overlap_ratio": item.get(
                                "interval_overlap_ratio"
                            ),
                            "dedup_comparisons": item.get(
                                "dedup_comparisons", []
                            ),
                        },
                        thresholds={
                            "minimum_separation_ms": separation_ms,
                            "requires_positive_interval_overlap": True,
                            "requires_shared_non_hand_object": True,
                        },
                        evidence_refs=[str(item["event_id"])],
                        legacy=item,
                    )
                    for item in decisions.values()
                ),
                key=lambda item: (
                    str(item["group_id"]),
                    str(item["segment_id"]),
                    float(item["peak_timestamp_ms"]),
                    str(item["event_id"]),
                ),
            )
        )
    return selected
