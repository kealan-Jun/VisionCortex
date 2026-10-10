"""Experiment boundaries, activity intervals and physical-change projection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from ..decisions import decision_receipt
from ..grouping import event_stable_identities, select_formal_experiment_start_events
from ..ordering import (
    candidate_sort_key,
    event_sort_key,
    stable_candidate_fingerprint,
    stable_event_fingerprint,
    stable_segment_uid,
)
from ..schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ExperimentSegment,
    PhysicalChange,
    ViewInput,
    ViewRole,
    event_is_formal,
)
from .observations import _ACTIVITY_PHYSICAL_TYPES, HAND_CLASSES, NON_ACTION_CLASSES


def _split_rich_repeated_primary_sequences(
    groups: Sequence[Sequence[EvidenceEvent]],
    config: dict[str, Any],
    decision_receipts: list[dict[str, Any]] | None = None,
) -> list[list[EvidenceEvent]]:
    """Split two complete workflows that one coarse envelope fused together.

    Coarse motion recall intentionally favors wide envelopes.  A fixed event
    gap cannot distinguish a long pause inside one workflow from two adjacent
    workflows, so this rule is deliberately stricter: both sides must be rich,
    both roles must remain observable on each side, and both sides must repeat
    the same configured primary process action.  The largest eligible inactive
    interval is selected deterministically and the rule is applied recursively.
    """

    continuity_cfg = config["continuity"]
    minimum_gap_ms = (
        float(continuity_cfg.get("atomic_sequence_split_min_gap_seconds", 5.0)) * 1000.0
    )
    gap_by_action_ms = {
        str(name): float(value) * 1000.0
        for name, value in (
            continuity_cfg.get("atomic_sequence_split_min_gap_seconds_by_action") or {}
        ).items()
    }
    minimum_events = max(
        1,
        int(continuity_cfg.get("atomic_sequence_split_min_events_per_side", 4)),
    )
    primary_action_names = {
        str(value)
        for value in continuity_cfg.get(
            "atomic_fragment_repeated_primary_actions",
            [ActionType.LIQUID_MOVEMENT.value],
        )
    }
    required_roles = {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}

    def primary_actions(items: Sequence[EvidenceEvent]) -> set[str]:
        return {
            event.action_type.value
            for event in items
            if event.action_type.value in primary_action_names
        }

    def observed_roles(items: Sequence[EvidenceEvent]) -> set[ViewRole]:
        return {role for event in items for role in event.supporting_roles}

    def split_one(items: Sequence[EvidenceEvent]) -> list[list[EvidenceEvent]]:
        ordered = sorted(items, key=event_sort_key)
        candidates: list[
            tuple[
                float,
                int,
                list[EvidenceEvent],
                list[EvidenceEvent],
                list[str],
                float,
            ]
        ] = []
        for position in range(minimum_events, len(ordered) - minimum_events + 1):
            left = ordered[:position]
            right = ordered[position:]
            left_end_ms = max(event.global_end_ms for event in left)
            right_start_ms = min(event.global_start_ms for event in right)
            inactive_gap_ms = right_start_ms - left_end_ms
            repeated_primary_actions = sorted(
                primary_actions(left) & primary_actions(right)
            )
            if not repeated_primary_actions:
                continue
            required_gap_ms = max(
                [minimum_gap_ms]
                + [
                    gap_by_action_ms.get(action, minimum_gap_ms)
                    for action in repeated_primary_actions
                ]
            )
            if inactive_gap_ms < required_gap_ms:
                continue
            if not (
                required_roles <= observed_roles(left)
                and required_roles <= observed_roles(right)
            ):
                continue
            candidates.append(
                (
                    inactive_gap_ms,
                    position,
                    left,
                    right,
                    repeated_primary_actions,
                    required_gap_ms,
                )
            )
        if not candidates:
            return [ordered]

        # Prefer the strongest physical inactivity boundary.  For an exact tie,
        # the earlier boundary wins so the result is stable across input order.
        inactive_gap_ms, position, left, right, repeated, required_gap_ms = max(
            candidates,
            key=lambda item: (item[0] / max(item[5], 1.0), item[0], -item[1]),
        )
        if decision_receipts is not None:
            decision_receipts.append(
                decision_receipt(
                    decision_type="raw_atomic_sequence_split",
                    rule_id="QF1-RICH-REPEATED-PRIMARY-SEQUENCE-SPLIT",
                    verdict="split",
                    subject_ids=[event.event_id for event in ordered],
                    reason_codes=[
                        "rich_dual_view_sequences_repeat_primary_action_across_inactive_gap"
                    ],
                    facts={
                        "split_position": position,
                        "inactive_gap_ms": inactive_gap_ms,
                        "left_event_ids": [event.event_id for event in left],
                        "right_event_ids": [event.event_id for event in right],
                        "left_event_fingerprints": [
                            stable_event_fingerprint(event) for event in left
                        ],
                        "right_event_fingerprints": [
                            stable_event_fingerprint(event) for event in right
                        ],
                        "left_roles": sorted(
                            role.value for role in observed_roles(left)
                        ),
                        "right_roles": sorted(
                            role.value for role in observed_roles(right)
                        ),
                        "repeated_primary_actions": repeated,
                        "required_gap_ms_for_repeated_actions": required_gap_ms,
                    },
                    thresholds={
                        "minimum_inactive_gap_ms": minimum_gap_ms,
                        "minimum_inactive_gap_ms_by_action": gap_by_action_ms,
                        "minimum_events_per_side": minimum_events,
                        "required_roles_per_side": sorted(
                            role.value for role in required_roles
                        ),
                        "configured_primary_actions": sorted(primary_action_names),
                    },
                    evidence_refs=[event.event_id for event in ordered],
                    legacy={
                        "decision": "split_fused_complete_action_sequences",
                    },
                )
            )
        return [*split_one(left), *split_one(right)]

    split_groups: list[list[EvidenceEvent]] = []
    for group in groups:
        split_groups.extend(split_one(group))
    return split_groups


def _event_core_interval(event: EvidenceEvent) -> tuple[float, float]:
    """Return a valid core interval, tolerating legacy ``model_copy`` fixtures."""

    core_start = event.core_global_start_ms
    core_end = event.core_global_end_ms
    if core_start is None or core_end is None:
        return float(event.global_start_ms), float(event.global_end_ms)
    core_start = float(core_start)
    core_end = float(core_end)
    if (
        core_end < core_start
        or core_end < event.global_start_ms
        or core_start > event.global_end_ms
    ):
        return float(event.global_start_ms), float(event.global_end_ms)
    return core_start, core_end


def build_experiment_segments(
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    config: dict[str, Any],
    coarse_windows: Sequence[ActionCandidate] | None = None,
    decision_receipts: list[dict[str, Any]] | None = None,
) -> list[ExperimentSegment]:
    component_only = sorted(
        (
            event
            for event in events
            if event_is_formal(event)
            and str(
                ((event.state_machine or {}).get("publication") or {}).get("status")
                or ""
            )
            == "component_only"
        ),
        key=event_sort_key,
    )
    # Higher-level state-machine actions keep their lower-level contacts and
    # movements in the audit ledger, but those duplicate components must not
    # open, extend, bridge, or merge experiment boundaries.  Otherwise a long
    # static hand/tool proximity can fuse two independently completed workflows
    # even though its containing liquid/panel action has already superseded it.
    accepted = sorted(
        (
            event
            for event in events
            if event_is_formal(event) and event not in component_only
        ),
        key=event_sort_key,
    )
    if component_only and decision_receipts is not None:
        decision_receipts.append(
            decision_receipt(
                decision_type="component_publication_boundary_quarantine",
                rule_id="QF1-COMPONENT-ONLY-NONBOUNDING",
                verdict="quarantined",
                subject_ids=[event.event_id for event in component_only],
                reason_codes=["superseded_components_cannot_define_boundaries"],
                facts={
                    "component_event_ids": [event.event_id for event in component_only],
                    "superseding_event_ids": sorted(
                        {
                            str(
                                (
                                    (event.state_machine or {}).get("publication") or {}
                                ).get("suppressed_by_event_id")
                            )
                            for event in component_only
                            if (
                                (event.state_machine or {}).get("publication") or {}
                            ).get("suppressed_by_event_id")
                        }
                    ),
                    "formal_membership_changed": True,
                    "audit_ledger_membership_changed": False,
                },
                evidence_refs=[event.event_id for event in component_only],
            )
        )
    if not accepted:
        return []
    cfg = config["segmentation"]
    default_gap_ms = float(cfg["experiment_gap_seconds"]) * 1000.0
    action_gap_seconds = {
        str(name): float(value)
        for name, value in (cfg.get("experiment_gap_seconds_by_action") or {}).items()
    }
    identity_idle_bridge_enabled = bool(cfg.get("identity_idle_bridge_enabled", True))
    identity_idle_bridge_max_gap_ms = (
        float(cfg.get("identity_idle_bridge_max_gap_seconds", 300.0)) * 1000.0
    )
    identity_idle_bridge_minimum = max(
        2,
        int(cfg.get("identity_idle_bridge_min_shared_identities", 2)),
    )
    groups: list[list[EvidenceEvent]] = []
    ordered_windows = (
        sorted(coarse_windows, key=candidate_sort_key) if coarse_windows else []
    )
    if ordered_windows:
        # A coarse motion burst is the experiment-level temporal envelope. Fine
        # actions may legitimately contain long pauses (incubation, reading a
        # balance, changing tools), so a fixed short event gap must not split one
        # bounded experiment. Padding is used only for assigning fine evidence;
        # the final boundary still comes from accepted fine events below.
        padding_ms = (
            float(config["performance"]["fine_window_padding_seconds"]) * 1000.0
        )
        buckets: list[list[EvidenceEvent]] = [[] for _ in ordered_windows]
        unassigned: list[EvidenceEvent] = []
        active_windows: list[tuple[int, ActionCandidate]] = []
        window_cursor = 0
        for event in accepted:
            event_start_ms, event_end_ms = _event_core_interval(event)
            while (
                window_cursor < len(ordered_windows)
                and ordered_windows[window_cursor].global_start_ms - padding_ms
                <= event_end_ms
            ):
                active_windows.append((window_cursor, ordered_windows[window_cursor]))
                window_cursor += 1
            active_windows = [
                item
                for item in active_windows
                if item[1].global_end_ms + padding_ms >= event_start_ms
            ]
            matches = [
                (index, window)
                for index, window in active_windows
                if window.global_start_ms - padding_ms <= event_end_ms
                and window.global_end_ms + padding_ms >= event_start_ms
            ]
            if not matches:
                unassigned.append(event)
                continue
            event_span_ms = max(1.0, event_end_ms - event_start_ms)
            best, _best_window = max(
                matches,
                key=lambda item: (
                    max(
                        0.0,
                        min(event_end_ms, item[1].global_end_ms)
                        - max(event_start_ms, item[1].global_start_ms),
                    )
                    / event_span_ms,
                    int(item[1].view_id in event.supporting_views),
                    -abs(
                        (event_start_ms + event_end_ms) / 2.0
                        - (item[1].global_start_ms + item[1].global_end_ms) / 2.0
                    ),
                    stable_candidate_fingerprint(item[1]),
                ),
            )
            buckets[best].append(event)

        def effective_gap_ms(
            previous: EvidenceEvent,
            current: EvidenceEvent,
        ) -> float:
            return max(
                default_gap_ms,
                action_gap_seconds.get(previous.action_type.value, 0.0) * 1000.0,
                action_gap_seconds.get(current.action_type.value, 0.0) * 1000.0,
            )

        def can_bridge_identity_idle(
            current: Sequence[EvidenceEvent],
            event: EvidenceEvent,
            gap_ms: float,
            *,
            same_recalled_window: bool,
        ) -> tuple[bool, list[tuple[str, str, int]]]:
            if (
                not identity_idle_bridge_enabled
                or gap_ms < 0.0
                or gap_ms > identity_idle_bridge_max_gap_ms
            ):
                return False, []
            current_identities = {
                identity
                for prior in current
                for identity in event_stable_identities(prior)
            }
            shared_identities = sorted(
                current_identities & event_stable_identities(event)
            )
            if len(shared_identities) < identity_idle_bridge_minimum:
                return False, shared_identities
            both_roles = {
                ViewRole.FIRST_PERSON,
                ViewRole.THIRD_PERSON,
            }
            current_roles = {
                role for prior in current for role in prior.supporting_roles
            }
            if not (
                both_roles.issubset(current_roles)
                and both_roles.issubset(set(event.supporting_roles))
            ):
                return False, shared_identities
            if same_recalled_window:
                return True, shared_identities
            # Without a recalled window, extend only an action whose observed
            # state explicitly ended incomplete. A completed stationary bottle
            # must not hold an experiment open by identity alone.
            prior_state = str(
                (current[-1].state_machine or {}).get("lifecycle_state") or ""
            )
            return prior_state == "incomplete_end", shared_identities

        def append_temporal_groups(
            items: Sequence[EvidenceEvent],
            *,
            same_recalled_window: bool,
        ) -> None:
            current: list[EvidenceEvent] = []
            current_end_ms = 0.0
            for event in sorted(items, key=event_sort_key):
                event_start_ms, event_end_ms = _event_core_interval(event)
                observed_gap_ms = event_start_ms - current_end_ms if current else 0.0
                gap_limit_ms = (
                    effective_gap_ms(current[-1], event) if current else default_gap_ms
                )
                bridged = False
                shared_identities: list[tuple[str, str, int]] = []
                if current and observed_gap_ms > gap_limit_ms:
                    bridged, shared_identities = can_bridge_identity_idle(
                        current,
                        event,
                        observed_gap_ms,
                        same_recalled_window=same_recalled_window,
                    )
                if current and observed_gap_ms > gap_limit_ms and not bridged:
                    groups.append(current)
                    current = []
                    current_end_ms = 0.0
                elif bridged and decision_receipts is not None:
                    decision_receipts.append(
                        decision_receipt(
                            decision_type="adaptive_experiment_idle_bridge",
                            rule_id="QF1-STABLE-IDENTITY-IDLE-BRIDGE",
                            verdict="accepted",
                            subject_ids=[current[-1].event_id, event.event_id],
                            reason_codes=[
                                "same_recalled_window_and_multiple_stable_identities"
                                if same_recalled_window
                                else "incomplete_action_and_multiple_stable_identities"
                            ],
                            facts={
                                "gap_ms": observed_gap_ms,
                                "shared_track_identities": [
                                    {
                                        "view_id": view_id,
                                        "object": object_name,
                                        "track_id": track_id,
                                    }
                                    for view_id, object_name, track_id in shared_identities
                                ],
                                "same_recalled_window": same_recalled_window,
                            },
                            thresholds={
                                "ordinary_gap_limit_ms": gap_limit_ms,
                                "identity_idle_bridge_maximum_gap_ms": (
                                    identity_idle_bridge_max_gap_ms
                                ),
                                "minimum_shared_identities": (
                                    identity_idle_bridge_minimum
                                ),
                            },
                            evidence_refs=[current[-1].event_id, event.event_id],
                        )
                    )
                current.append(event)
                current_end_ms = max(current_end_ms, event_end_ms)
            if current:
                groups.append(current)

        for bucket in buckets:
            append_temporal_groups(bucket, same_recalled_window=True)
        # Retain independently strong evidence that falls outside a recalled
        # window, but group it conservatively by the legacy event-gap rule.
        append_temporal_groups(unassigned, same_recalled_window=False)
        groups.sort(
            key=lambda group: (
                min(event.global_start_ms for event in group),
                tuple(
                    stable_event_fingerprint(event)
                    for event in sorted(group, key=event_sort_key)
                ),
            )
        )
    else:

        def no_window_gap_ms(previous: EvidenceEvent, current: EvidenceEvent) -> float:
            return max(
                default_gap_ms,
                action_gap_seconds.get(previous.action_type.value, 0.0) * 1000.0,
                action_gap_seconds.get(current.action_type.value, 0.0) * 1000.0,
            )

        current: list[EvidenceEvent] = []
        current_end_ms = 0.0
        for event in accepted:
            event_start_ms, event_end_ms = _event_core_interval(event)
            if current:
                observed_gap_ms = event_start_ms - current_end_ms
                gap_limit_ms = no_window_gap_ms(current[-1], event)
                current_identities = {
                    identity
                    for prior in current
                    for identity in event_stable_identities(prior)
                }
                shared_identities = sorted(
                    current_identities & event_stable_identities(event)
                )
                prior_incomplete = (
                    str((current[-1].state_machine or {}).get("lifecycle_state") or "")
                    == "incomplete_end"
                )
                identity_bridge = bool(
                    identity_idle_bridge_enabled
                    and observed_gap_ms <= identity_idle_bridge_max_gap_ms
                    and len(shared_identities) >= identity_idle_bridge_minimum
                    and prior_incomplete
                    and {
                        ViewRole.FIRST_PERSON,
                        ViewRole.THIRD_PERSON,
                    }.issubset(set(event.supporting_roles))
                )
                if observed_gap_ms > gap_limit_ms and not identity_bridge:
                    groups.append(current)
                    current = []
                    current_end_ms = 0.0
                elif (
                    observed_gap_ms > gap_limit_ms
                    and identity_bridge
                    and decision_receipts is not None
                ):
                    decision_receipts.append(
                        decision_receipt(
                            decision_type="adaptive_experiment_idle_bridge",
                            rule_id="QF1-STABLE-IDENTITY-IDLE-BRIDGE",
                            verdict="accepted",
                            subject_ids=[current[-1].event_id, event.event_id],
                            reason_codes=[
                                "incomplete_action_and_multiple_stable_identities"
                            ],
                            facts={
                                "gap_ms": observed_gap_ms,
                                "same_recalled_window": False,
                            },
                            thresholds={
                                "ordinary_gap_limit_ms": gap_limit_ms,
                                "identity_idle_bridge_maximum_gap_ms": (
                                    identity_idle_bridge_max_gap_ms
                                ),
                            },
                            evidence_refs=[current[-1].event_id, event.event_id],
                        )
                    )
            current.append(event)
            current_end_ms = max(current_end_ms, event_end_ms)
        if current:
            groups.append(current)

    groups = _split_rich_repeated_primary_sequences(
        groups,
        config,
        decision_receipts=decision_receipts,
    )
    groups.sort(
        key=lambda group: (
            min(event.global_start_ms for event in group),
            tuple(
                stable_event_fingerprint(event)
                for event in sorted(group, key=event_sort_key)
            ),
        )
    )

    def core_start_ms(event: EvidenceEvent) -> float:
        return _event_core_interval(event)[0]

    def core_end_ms(event: EvidenceEvent) -> float:
        return _event_core_interval(event)[1]

    by_event = {event.event_id: event for event in events}
    segments: list[ExperimentSegment] = []
    for index, unsorted_group in enumerate(groups, 1):
        segment_id = f"EXP-{index:04d}"
        group = sorted(unsorted_group, key=event_sort_key)
        # A segment must start on a physically meaningful operation anchor.
        # Single-view liquid hypotheses and movement of fixed equipment may be
        # useful context, but are too noisy to pull the experiment boundary
        # earlier by themselves.
        start_anchors = select_formal_experiment_start_events(group, config)
        raw_start = min(core_start_ms(event) for event in (start_anchors or group))
        leading_context: list[EvidenceEvent] = []
        if start_anchors and bool(cfg.get("accepted_leading_context_enabled", False)):
            # A full-timeline scan often observes a reliable single-role
            # preparation chain before the first dual-role opener.  Recover
            # that boundary evidence only when it is already accepted,
            # publication-primary (not semantic-provisional), and connected
            # to the opener by a tight temporal chain.  This changes the clip
            # boundary and membership but never promotes rejected evidence.
            maximum_leading_gap_ms = (
                float(cfg.get("accepted_leading_context_max_gap_seconds", 10.0))
                * 1000.0
            )
            maximum_leading_extension_ms = (
                float(cfg.get("accepted_leading_context_max_extension_seconds", 90.0))
                * 1000.0
            )
            lower_limit_ms = max(0.0, raw_start - maximum_leading_extension_ms)
            cursor_ms = raw_start
            eligible_leading = [
                event
                for event in group
                if core_start_ms(event) < raw_start
                and core_end_ms(event) >= lower_limit_ms
                and str(
                    ((event.state_machine or {}).get("publication") or {}).get("status")
                    or "primary"
                )
                == "primary"
                and not bool(
                    (
                        (event.observability or {}).get("semantic_recall_admission")
                        or {}
                    ).get("mandatory_semantic_review")
                )
            ]
            for event in sorted(
                eligible_leading,
                key=lambda item: (core_end_ms(item), core_start_ms(item)),
                reverse=True,
            ):
                if core_start_ms(event) >= cursor_ms:
                    continue
                if core_end_ms(event) < cursor_ms - maximum_leading_gap_ms:
                    break
                leading_context.append(event)
                cursor_ms = min(cursor_ms, core_start_ms(event))
            if leading_context:
                leading_context.sort(key=event_sort_key)
                raw_start = min(
                    raw_start,
                    min(core_start_ms(event) for event in leading_context),
                )
        start = max(0.0, raw_start - float(cfg["experiment_pre_roll_seconds"]) * 1000.0)
        # Accepted recall evidence may precede the first reliable operation
        # anchor. Keep it in the global audit ledger, but do not attach an event
        # that ends before the bounded clip starts to this experiment.
        bounded_group = [event for event in group if core_end_ms(event) >= start]
        raw_end = max(core_end_ms(event) for event in bounded_group)
        boundary_context_receipt: dict[str, Any] | None = None
        core_roles = {
            role for event in bounded_group for role in event.supporting_roles
        }
        if ordered_windows and core_roles == {
            ViewRole.FIRST_PERSON,
            ViewRole.THIRD_PERSON,
        }:
            midpoint = (raw_start + raw_end) / 2.0
            matching_windows = [
                window
                for window in ordered_windows
                if window.global_start_ms <= midpoint <= window.global_end_ms
            ]
            if matching_windows:
                boundary_window = min(
                    matching_windows,
                    key=lambda window: (
                        abs(
                            midpoint
                            - (window.global_start_ms + window.global_end_ms) / 2.0
                        ),
                        candidate_sort_key(window),
                    ),
                )
                later_core_in_window = any(
                    other is not unsorted_group
                    and min(item.global_start_ms for item in other) > raw_end
                    and boundary_window.global_start_ms
                    <= (
                        min(item.global_start_ms for item in other)
                        + max(item.global_end_ms for item in other)
                    )
                    / 2.0
                    <= boundary_window.global_end_ms
                    for other in groups
                )
                activation_gap_ms = (
                    float(cfg.get("boundary_context_activation_gap_seconds", 30.0))
                    * 1000.0
                )
                if (
                    not later_core_in_window
                    and boundary_window.global_end_ms - raw_end >= activation_gap_ms
                ):
                    bridge_confidence = float(
                        cfg.get("boundary_context_bridge_confidence", 0.50)
                    )
                    extension_confidence = float(
                        cfg.get("boundary_context_min_confidence", 0.65)
                    )
                    maximum_gap_ms = (
                        float(cfg.get("boundary_context_max_gap_seconds", 10.0))
                        * 1000.0
                    )
                    maximum_extension_ms = (
                        float(cfg.get("boundary_context_max_extension_seconds", 90.0))
                        * 1000.0
                    )
                    cleanup_objects = {
                        "brush",
                        "cleaning_tool",
                        "sink",
                        "wash_bottle",
                        "waste_container",
                    }
                    cursor = raw_end
                    supported_end = raw_end
                    limit = min(
                        raw_end + maximum_extension_ms,
                        boundary_window.global_end_ms,
                    )
                    context_chain: list[EvidenceEvent] = []
                    strong_context: list[EvidenceEvent] = []
                    for context in sorted(events, key=event_sort_key):
                        if context.accepted or context.global_end_ms <= cursor:
                            continue
                        if context.global_start_ms > limit:
                            break
                        if context.global_start_ms > cursor + maximum_gap_ms:
                            break
                        context_roles = set(context.supporting_roles)
                        cleanup_evidence = sorted(
                            set(context.objects) & cleanup_objects
                        )
                        if (
                            len(context_roles) != 1
                            or context.confidence < bridge_confidence
                            or not cleanup_evidence
                        ):
                            continue
                        context_chain.append(context)
                        cursor = min(limit, max(cursor, context.global_end_ms))
                        if context.confidence >= extension_confidence:
                            strong_context.append(context)
                            supported_end = max(supported_end, cursor)
                    if supported_end > raw_end and strong_context:
                        previous_end = raw_end
                        raw_end = supported_end
                        boundary_context_receipt = decision_receipt(
                            decision_type="raw_boundary_context_extension",
                            rule_id="QF1-SINGLE-VIEW-BOUNDARY-CONTEXT",
                            verdict="accepted",
                            subject_ids=[segment_id],
                            reason_codes=[
                                "single_role_cleanup_chain_inside_recalled_boundary"
                            ],
                            facts={
                                "previous_raw_end_ms": previous_end,
                                "extended_raw_end_ms": raw_end,
                                "context_event_ids": [
                                    event.event_id for event in context_chain
                                ],
                                "context_event_fingerprints": [
                                    stable_event_fingerprint(event)
                                    for event in context_chain
                                ],
                                "strong_context_event_ids": [
                                    event.event_id for event in strong_context
                                ],
                                "cleanup_objects": sorted(
                                    {
                                        obj
                                        for event in context_chain
                                        for obj in event.objects
                                        if obj in cleanup_objects
                                    }
                                ),
                                "boundary_candidate_id": boundary_window.candidate_id,
                                "boundary_candidate_fingerprint": (
                                    stable_candidate_fingerprint(boundary_window)
                                ),
                                "formal_membership_changed": False,
                                "boundary_clipped_to_recalled_window": (
                                    raw_end == boundary_window.global_end_ms
                                ),
                            },
                            thresholds={
                                "activation_gap_ms": activation_gap_ms,
                                "bridge_confidence": bridge_confidence,
                                "extension_confidence": extension_confidence,
                                "maximum_context_gap_ms": maximum_gap_ms,
                                "maximum_extension_ms": maximum_extension_ms,
                            },
                            evidence_refs=[
                                *[event.event_id for event in bounded_group],
                                *[event.event_id for event in context_chain],
                            ],
                            legacy={
                                "segment_id": segment_id,
                                "decision": "extended_raw_boundary_from_cleanup_context",
                            },
                        )
        end = raw_end + float(cfg["experiment_post_roll_seconds"]) * 1000.0
        minimum = float(cfg["min_experiment_seconds"]) * 1000.0
        if end - start < minimum:
            padding = (minimum - (end - start)) / 2.0
            start, end = max(0.0, start - padding), end + padding
        candidate_counts = defaultdict(int)
        direct_evidence_ms = defaultdict(float)
        semantic_review_admitted_views: set[str] = set()
        for event in bounded_group:
            semantic_admission = (event.observability or {}).get(
                "semantic_recall_admission"
            )
            for view_id in event.supporting_views:
                candidate_counts[view_id] += 1
                direct_evidence_ms[view_id] += max(
                    125.0,
                    max(
                        (
                            item.global_end_ms - item.global_start_ms
                            for item in event.candidates
                            if item.view_id == view_id
                        ),
                        default=0.0,
                    ),
                )
                if (
                    isinstance(semantic_admission, dict)
                    and semantic_admission.get("mandatory_semantic_review") is True
                ):
                    semantic_review_admitted_views.add(view_id)
            if (
                isinstance(semantic_admission, dict)
                and semantic_admission.get("mandatory_semantic_review") is True
                and str(semantic_admission.get("context_view_id") or "")
            ):
                context_view_id = str(semantic_admission["context_view_id"])
                context_duration_ms = max(
                    125.0,
                    float(semantic_admission.get("context_global_end_ms") or 0.0)
                    - float(semantic_admission.get("context_global_start_ms") or 0.0),
                )
                candidate_counts[context_view_id] += 1
                direct_evidence_ms[context_view_id] += context_duration_ms
                semantic_review_admitted_views.add(context_view_id)
        duration_minutes = max((end - start) / 60_000.0, 1e-6)
        threshold = float(cfg["min_view_action_density_per_minute"])
        sparse_view_enabled = bool(cfg.get("long_experiment_sparse_view_enabled", True))
        sparse_minimum_duration_ms = (
            float(cfg.get("long_experiment_sparse_view_min_duration_seconds", 120.0))
            * 1000.0
        )
        sparse_minimum_events = max(
            2,
            int(cfg.get("long_experiment_sparse_view_min_events", 2)),
        )
        sparse_minimum_evidence_ms = (
            float(cfg.get("long_experiment_sparse_view_min_evidence_seconds", 1.0))
            * 1000.0
        )
        participating = sorted(
            view_id
            for view_id, count in candidate_counts.items()
            if (
                count / duration_minutes >= threshold
                or (
                    sparse_view_enabled
                    and end - start >= sparse_minimum_duration_ms
                    and count >= sparse_minimum_events
                    and direct_evidence_ms[view_id] >= sparse_minimum_evidence_ms
                )
            )
            and (
                direct_evidence_ms[view_id] >= 500.0
                or view_id in semantic_review_admitted_views
            )
        )
        rejected_views = {
            view.view_id: "该边界内没有通过审计的实验动作，或动作密度不足"
            for view in views
            if view.view_id not in participating
        }
        if not participating:
            continue
        micro_segments = []
        for position, event in enumerate(bounded_group):
            micro_segments.append(
                {
                    "micro_segment_id": f"MICRO-{index:04d}-{position + 1:04d}",
                    "start_global_ms": core_start_ms(event),
                    "end_global_ms": core_end_ms(event),
                    "action_type": event.action_type.value,
                    "objects": event.objects,
                    "evidence_event_id": event.event_id,
                    "view_alignment_state": "aligned"
                    if all(event.supporting_views)
                    else "uncertain",
                    "next_event_id": (
                        bounded_group[position + 1].event_id
                        if position + 1 < len(bounded_group)
                        else None
                    ),
                    "uncertainty": event.uncertainty,
                }
            )
        segment = ExperimentSegment(
            segment_id=segment_id,
            global_start_ms=start,
            global_end_ms=end,
            event_ids=[event.event_id for event in bounded_group],
            participating_views=participating,
            rejected_views=rejected_views,
            micro_segments=micro_segments,
        )
        segment.segment_uid = stable_segment_uid(segment, by_event)
        segments.append(segment)
        if decision_receipts is not None:
            if boundary_context_receipt is not None:
                decision_receipts.append(boundary_context_receipt)
            start_sources = sorted(
                (
                    event
                    for event in (
                        [*leading_context, *start_anchors] if start_anchors else group
                    )
                    if core_start_ms(event) == raw_start
                ),
                key=event_sort_key,
            )
            end_sources = sorted(
                (event for event in bounded_group if core_end_ms(event) == raw_end),
                key=event_sort_key,
            )
            reason_codes = [
                "start_from_accepted_operation_anchor",
                (
                    "end_from_explicit_qf1_receipt"
                    if boundary_context_receipt is not None
                    else "end_from_accepted_event"
                ),
            ]
            if leading_context:
                reason_codes.append("accepted_primary_leading_context_chain_applied")
            direct_start = max(
                0.0,
                raw_start - float(cfg["experiment_pre_roll_seconds"]) * 1000.0,
            )
            direct_end = raw_end + float(cfg["experiment_post_roll_seconds"]) * 1000.0
            if start < direct_start or end > direct_end:
                reason_codes.append("minimum_duration_padding_applied")
            decision_receipts.append(
                decision_receipt(
                    decision_type="raw_segment_boundary",
                    rule_id="QF1-RAW-ACCEPTED-EVENT-BOUNDARY",
                    verdict="accepted",
                    subject_ids=[segment_id],
                    reason_codes=reason_codes,
                    facts={
                        "raw_start_ms": raw_start,
                        "raw_end_ms": raw_end,
                        "final_start_ms": start,
                        "final_end_ms": end,
                        "start_source_event_ids": [
                            event.event_id for event in start_sources
                        ],
                        "start_source_event_fingerprints": [
                            stable_event_fingerprint(event) for event in start_sources
                        ],
                        "end_source_event_ids": [
                            event.event_id for event in end_sources
                        ],
                        "end_source_event_fingerprints": [
                            stable_event_fingerprint(event) for event in end_sources
                        ],
                        "end_source_decision_ids": (
                            [boundary_context_receipt["decision_id"]]
                            if boundary_context_receipt is not None
                            else []
                        ),
                        "leading_context_event_ids": [
                            event.event_id for event in leading_context
                        ],
                        "leading_context_event_fingerprints": [
                            stable_event_fingerprint(event) for event in leading_context
                        ],
                        "accepted_event_ids": [
                            event.event_id for event in bounded_group
                        ],
                        "accepted_event_fingerprints": [
                            stable_event_fingerprint(event) for event in bounded_group
                        ],
                        "coarse_window_fingerprints": sorted(
                            stable_candidate_fingerprint(window)
                            for window in ordered_windows
                            if window.global_end_ms >= raw_start
                            and window.global_start_ms <= raw_end
                        ),
                        "implicit_motion_window_extension": False,
                    },
                    thresholds={
                        "pre_roll_ms": float(cfg["experiment_pre_roll_seconds"])
                        * 1000.0,
                        "post_roll_ms": float(cfg["experiment_post_roll_seconds"])
                        * 1000.0,
                        "minimum_duration_ms": minimum,
                        "accepted_leading_context_max_gap_ms": float(
                            cfg.get(
                                "accepted_leading_context_max_gap_seconds",
                                10.0,
                            )
                        )
                        * 1000.0,
                        "accepted_leading_context_max_extension_ms": float(
                            cfg.get(
                                "accepted_leading_context_max_extension_seconds",
                                90.0,
                            )
                        )
                        * 1000.0,
                    },
                    evidence_refs=[event.event_id for event in bounded_group],
                    legacy={
                        "segment_id": segment_id,
                        "decision": "bounded_from_accepted_events",
                    },
                )
            )
    return segments


def build_physical_change_log(events: Iterable[EvidenceEvent]) -> list[PhysicalChange]:
    mapping = {
        ActionType.HAND_OBJECT_CONTACT: "contact_started",
        ActionType.OBJECT_MOVEMENT: "object_moved",
        ActionType.LIQUID_MOVEMENT: "liquid_transferred",
        ActionType.CONTAINER_STATE_CHANGE: "container_state_changed",
        ActionType.DEVICE_PANEL_OPERATION: "device_operated",
        ActionType.PIPETTE_TRANSFER_OPERATION: "pipette_transfer_operated",
    }
    changes: list[PhysicalChange] = []
    for event in events:
        if not event_is_formal(event):
            continue
        before, after = None, None
        if event.action_type == ActionType.CONTAINER_STATE_CHANGE:
            before, after = "closed_or_unknown", "open_closed_or_cap_changed"
        changes.append(
            PhysicalChange(
                change_id=f"CHANGE-{len(changes) + 1:06d}",
                event_id=event.event_id,
                global_ms=event.key_global_ms,
                change_type=mapping[event.action_type],
                object_names=event.objects,
                before_state=before,
                after_state=after,
                supporting_views=event.supporting_views,
                confidence=event.confidence,
                uncertainty=event.uncertainty,
            )
        )
    return changes


def fine_scan_windows(candidates, infos, transforms, config, padding_seconds=None):
    if bool(
        config["performance"].get("exhaustive_negative_audit_full_timeline", False)
        or config["performance"].get("exhaustive_full_timeline_scan", False)
    ):
        return {
            view_id: [(0.0, float(info.duration_ms))]
            for view_id, info in infos.items()
            if float(info.duration_ms) > 0.0
        }
    padding = (
        float(
            config["performance"]["fine_window_padding_seconds"]
            if padding_seconds is None
            else padding_seconds
        )
        * 1000.0
    )
    merge_gap = max(
        0.0,
        float(config["performance"].get("fine_window_merge_gap_seconds", 0.0)) * 1000.0,
    )
    candidate_list = list(candidates)
    risk_extra_by_id: dict[str, float] = {}
    perf = config["performance"]
    if perf.get("fine_risk_window_expansion_enabled", False):
        configured_actions = {
            str(item).strip()
            for item in perf.get("fine_risk_action_types", [])
            if str(item).strip()
        }
        low_confidence_threshold = float(
            perf.get("fine_risk_low_confidence_threshold", 0.70)
        )
        extra_padding_ms = max(
            0.0,
            float(perf.get("fine_risk_extra_padding_seconds", 30.0)) * 1000.0,
        )
        conflict_gap_ms = max(
            0.0,
            float(perf.get("fine_risk_conflict_gap_seconds", 15.0)) * 1000.0,
        )
        for candidate in candidate_list:
            if (
                candidate.action_type.value in configured_actions
                or float(candidate.confidence) < low_confidence_threshold
            ):
                risk_extra_by_id[candidate.candidate_id] = extra_padding_ms
        active_candidates: list[ActionCandidate] = []
        for current in sorted(
            candidate_list,
            key=lambda item: (item.global_start_ms, item.global_end_ms),
        ):
            active_candidates = [
                previous
                for previous in active_candidates
                if previous.global_end_ms + conflict_gap_ms >= current.global_start_ms
            ]
            for previous in active_candidates:
                if previous.action_type == current.action_type:
                    continue
                risk_extra_by_id[previous.candidate_id] = extra_padding_ms
                risk_extra_by_id[current.candidate_id] = extra_padding_ms
            active_candidates.append(current)
    grouped: dict[str, list[tuple[float, float]]] = {view_id: [] for view_id in infos}
    for candidate in candidate_list:
        candidate_padding = padding + risk_extra_by_id.get(candidate.candidate_id, 0.0)
        for view_id, info in infos.items():
            alignment_extra = 0.0
            transform = transforms[view_id]
            if perf.get("fine_low_alignment_extra_padding_enabled", False) and (
                transform.state != "aligned"
                or float(transform.confidence)
                < float(perf.get("fine_low_alignment_confidence_threshold", 0.80))
            ):
                alignment_extra = max(
                    0.0,
                    float(perf.get("fine_low_alignment_extra_padding_seconds", 30.0))
                    * 1000.0,
                )
            global_start = (
                candidate.global_start_ms - candidate_padding - alignment_extra
            )
            global_end = candidate.global_end_ms + candidate_padding + alignment_extra
            local_start = max(0.0, transforms[view_id].to_local(global_start))
            local_end = min(info.duration_ms, transforms[view_id].to_local(global_end))
            if local_end > local_start:
                grouped[view_id].append((local_start, local_end))
    merged: dict[str, list[tuple[float, float]]] = {}
    for view_id, windows in grouped.items():
        result: list[list[float]] = []
        for start, end in sorted(windows):
            if result and start <= result[-1][1] + merge_gap:
                result[-1][1] = max(result[-1][1], end)
            else:
                result.append([start, end])
        merged[view_id] = [(item[0], item[1]) for item in result]
    return merged


def merge_activity_intervals(values, lower, upper, padding=0.0):
    result = []
    for start, end in sorted(
        (max(lower, a - padding), min(upper, b + padding)) for a, b in values
    ):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _activity_objects(candidates):
    """Use the same laboratory-context exclusions for seeds and continuity."""
    ignored = HAND_CLASSES | NON_ACTION_CLASSES | {"paper", "unknown"}
    return {obj for candidate in candidates for obj in candidate.objects} - ignored


def activity_seed_intervals(events, coarse_candidates, scan_windows, config):
    """Retain visible device activity without promoting formal physical actions.

    Formal event/group acceptance may require another camera or semantic proof.
    Preprocessing instead needs bounded, fine-supported interaction intervals.
    Coarse-only detections, weak contact and stationary proximity remain out.
    """
    cfg = config["segmentation"]
    strong = float(cfg.get("rejected_boundary_context_strong_confidence", 0.65))
    minimum_ms = float(cfg["min_event_duration_seconds"]) * 1000
    minimum_observations = int(cfg["min_event_observations"])
    pre = float(cfg["key_clip_pre_seconds"]) * 1000
    post = float(cfg["key_clip_post_seconds"]) * 1000
    intervals, decisions = [], []
    for event in events:
        if event.action_type.value not in _ACTIVITY_PHYSICAL_TYPES:
            continue
        evidence = [row for candidate in event.candidates for row in candidate.evidence]
        transition = any(
            row.get("interaction_state", {}).get("approach_confirmed")
            or row.get("interaction_state", {}).get("previous_release_global_ms")
            is not None
            for row in evidence
        )
        fine_contact = (
            event.action_type.value == "hand_object_contact"
            and event.confidence >= strong
            and event.global_end_ms - event.global_start_ms >= minimum_ms
            and len(
                {
                    row.get("frame_index")
                    for row in evidence
                    if row.get("frame_index") is not None
                }
            )
            >= minimum_observations
            and transition
        )
        if not (event.accepted or fine_contact):
            continue
        if not _activity_objects(event.candidates):
            decisions.append(
                {
                    "event_id": event.event_id,
                    "start_ms": event.global_start_ms,
                    "end_ms": event.global_end_ms,
                    "basis": "non_laboratory_context_cannot_seed_activity",
                    "objects": sorted(
                        {
                            obj
                            for candidate in event.candidates
                            for obj in candidate.objects
                        }
                    ),
                    "activity_seed_eligible": False,
                    "scope": "single_device_activity_screening",
                    "event_acceptance_mutated": False,
                    "physical_action_confirmed": False,
                }
            )
            continue
        for start, end in scan_windows:
            if start <= event.global_start_ms and end >= event.global_end_ms:
                support = [
                    candidate.candidate_id
                    for candidate in coarse_candidates
                    if getattr(candidate, "global_start_ms", candidate.local_start_ms)
                    <= end
                    and getattr(candidate, "global_end_ms", candidate.local_end_ms)
                    >= start
                ]
                if not support:
                    continue
                interval = (
                    max(start, event.global_start_ms - pre),
                    min(end, event.global_end_ms + post),
                )
                intervals.append(interval)
                decisions.append(
                    {
                        "event_id": event.event_id,
                        "coarse_candidate_ids": support,
                        "start_ms": interval[0],
                        "end_ms": interval[1],
                        "basis": "accepted_fine_event"
                        if event.accepted
                        else "persistent_fine_contact_with_transition",
                        "confidence": event.confidence,
                        "observation_count": len(evidence),
                        "scope": "single_device_activity_screening",
                        "event_acceptance_mutated": False,
                        "physical_action_confirmed": False,
                    }
                )
                break
    return intervals, decisions


def build_activity_intervals(events, coarse_candidates, scan_windows, config):
    """Keep an anchored operation episode, including supported short pauses.

    Fine interactions bound the output. Object-backed coarse motion can bridge
    two fine interactions, but cannot start/end an episode or open one alone.
    Neither context membership nor continuity promotes formal action evidence.
    """
    seeds, seed_decisions = activity_seed_intervals(
        events, coarse_candidates, scan_windows, config
    )
    if not seeds:
        return seeds, seed_decisions
    cfg = config["segmentation"]
    nodes = []
    for event in events:
        if (
            event.action_type.value not in _ACTIVITY_PHYSICAL_TYPES
            or event.confidence < float(cfg["rejected_boundary_context_min_confidence"])
        ):
            continue
        objects = _activity_objects(event.candidates)
        evidence = [row for candidate in event.candidates for row in candidate.evidence]
        if (
            not objects
            or len(
                {
                    row.get("frame_index")
                    for row in evidence
                    if row.get("frame_index") is not None
                }
            )
            < 2
        ):
            continue
        identities = {
            f"{kind}:{row[kind + '_track_id']}"
            for row in evidence
            for kind in ("hand", "object")
            if row.get(kind + "_track_id") is not None
        }
        nodes.append(
            {
                "id": event.event_id,
                "kind": "fine",
                "start": event.global_start_ms,
                "end": event.global_end_ms,
                "objects": objects,
                "identities": identities,
            }
        )
    for candidate in coarse_candidates:
        # Spatial co-occurrence is not continuity through an unobserved pause.
        motion = any(
            "coarse_motion" in row or "motion_score" in row
            for row in candidate.evidence
        )
        objects = _activity_objects([candidate])
        if motion and objects:
            nodes.append(
                {
                    "id": candidate.candidate_id,
                    "kind": "coarse_motion",
                    "start": getattr(
                        candidate, "global_start_ms", candidate.local_start_ms
                    ),
                    "end": getattr(candidate, "global_end_ms", candidate.local_end_ms),
                    "objects": objects,
                    "identities": set(),
                }
            )
    gap = float(cfg["experiment_gap_seconds"]) * 1000
    extension = float(cfg["rejected_boundary_context_max_extension_seconds"]) * 1000
    pre = float(cfg["experiment_pre_roll_seconds"]) * 1000
    post = float(cfg["experiment_post_roll_seconds"]) * 1000
    intervals, decisions = list(seeds), list(seed_decisions)
    for seed in seed_decisions:
        if seed.get("activity_seed_eligible") is False:
            continue
        anchor = next((node for node in nodes if node["id"] == seed["event_id"]), None)
        if anchor is None:
            continue
        bounds = next(
            (
                (start, end)
                for start, end in scan_windows
                if start <= anchor["start"] <= anchor["end"] <= end
            ),
            None,
        )
        if bounds is None:
            continue
        eligible = [
            node
            for node in nodes
            if bounds[0] <= node["start"] <= node["end"] <= bounds[1]
            and node["start"] >= anchor["start"] - extension
            and node["end"] <= anchor["end"] + extension
        ]
        reached, frontier, edges = {anchor["id"]}, [anchor], []
        while frontier:
            left = frontier.pop()
            for right in eligible:
                distance = max(
                    0, right["start"] - left["end"], left["start"] - right["end"]
                )
                shared = left["objects"] & right["objects"]
                tracked = left["identities"] & right["identities"]
                if right["id"] in reached or distance > gap or not (shared or tracked):
                    continue
                reached.add(right["id"])
                frontier.append(right)
                edges.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "gap_ms": distance,
                        "shared_objects": sorted(shared),
                        "shared_fine_tracks": sorted(tracked),
                        "physical_identity_verified": False,
                    }
                )
        fine = [
            node
            for node in eligible
            if node["id"] in reached and node["kind"] == "fine"
        ]
        if len(fine) < 2:
            continue
        start = max(bounds[0], min(node["start"] for node in fine) - pre)
        end = min(bounds[1], max(node["end"] for node in fine) + post)
        intervals.append((start, end))
        decisions.append(
            {
                "basis": "fine_anchored_operation_continuity",
                "event_id": anchor["id"],
                "start_ms": start,
                "end_ms": end,
                "fine_event_ids": sorted(node["id"] for node in fine),
                "coarse_bridge_ids": sorted(
                    node["id"]
                    for node in eligible
                    if node["id"] in reached
                    and node["kind"] == "coarse_motion"
                    and start <= node["start"] <= node["end"] <= end
                ),
                "edges": edges,
                "maximum_gap_ms": gap,
                "maximum_extension_ms": extension,
                "scope": "single_device_activity_screening",
                "evidence_status": "PARTIAL_EVIDENCE",
                "event_acceptance_mutated": False,
                "physical_action_confirmed": False,
            }
        )
    joined = []
    for start, end in merge_activity_intervals(
        intervals, min(a for a, _ in scan_windows), max(b for _, b in scan_windows)
    ):
        if (
            joined
            and start - joined[-1][1] <= float(cfg["event_merge_gap_seconds"]) * 1000
        ):
            decisions.append(
                {
                    "basis": "fine_sampling_gap_inside_activity",
                    "start_ms": joined[-1][1],
                    "end_ms": start,
                    "physical_action_confirmed": False,
                }
            )
            joined[-1] = (joined[-1][0], end)
        else:
            joined.append((start, end))
    return joined, decisions


def build_view_activity_intervals(
    events, coarse_candidates, views, transforms, scan_windows, config
):
    """Project shared activity analysis into each camera's actual scanned time.

    Each camera is evaluated from its own fine observations. A scene in another
    camera cannot independently label this view as active. Global timestamps
    are converted using the same alignment transforms as candidate auditing.
    """
    results = {}
    for view in views:
        transform = transforms[view.view_id]
        local_windows = scan_windows.get(view.view_id, [])
        global_windows = [
            (transform.to_global(start), transform.to_global(end))
            for start, end in local_windows
        ]
        own_events = []
        for event in events:
            candidates = [c for c in event.candidates if c.view_id == view.view_id]
            if not candidates:
                continue
            own_events.append(
                event.model_copy(
                    update={
                        "candidates": candidates,
                        "global_start_ms": min(c.global_start_ms for c in candidates),
                        "global_end_ms": max(c.global_end_ms for c in candidates),
                    }
                )
            )
        intervals, decisions = build_activity_intervals(
            own_events,
            [c for c in coarse_candidates if c.view_id == view.view_id],
            global_windows,
            config,
        )
        results[view.view_id] = {
            "intervals": [
                (transform.to_local(start), transform.to_local(end))
                for start, end in intervals
            ],
            "decisions": decisions,
            "algorithm": "visioncortex.actions.build_activity_intervals",
            "decision_time_basis": "aligned_global_ms",
            "interval_time_basis": "view_local_ms",
            "physical_action_confirmed": False,
            "evidence_status": "PARTIAL_EVIDENCE",
        }
    return results
