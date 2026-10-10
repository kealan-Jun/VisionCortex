from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from .schemas import ActionType, EvidenceEvent, ViewRole, event_is_formal


HAND_CLASSES = {"hand", "gloved_hand"}
TOOL_CLASSES = {"pipette", "spearhead", "spatula", "magnetic_stir_bar"}
CONTAINER_CLASSES = {
    "beaker",
    "container",
    "reagent_bottle",
    "reagent_bottle_open",
    "sample_bottle",
    "sample_bottle_blue",
    "tube",
}
DEVICE_CLASSES = {"balance", "magnetic_stirrer"}


def _normalized_action(action_type: ActionType) -> str:
    if action_type == ActionType.LIQUID_MOVEMENT:
        return "liquid_transfer"
    return action_type.value


def _track_tokens(event: EvidenceEvent) -> list[str]:
    tokens: set[str] = set()
    for candidate in event.candidates:
        for evidence in candidate.evidence:
            for key in (
                "track_id",
                "object_track_id",
                "tool_track_id",
                "vessel_track_id",
                "source_track_id",
                "target_track_id",
                "container_track_id",
            ):
                value = evidence.get(key)
                if isinstance(value, int):
                    tokens.add(f"{candidate.view_id}:{key}:{value}")
    return sorted(tokens)


def _evidence_flags(event: EvidenceEvent) -> dict[str, bool]:
    evidence = [item for candidate in event.candidates for item in candidate.evidence]
    return {
        "contact": any(item.get("distance_norm") is not None for item in evidence),
        "movement": any(
            item.get("camera_compensated_displacement_norm") is not None
            or item.get("displacement_norm") is not None
            for item in evidence
        ),
        "state_change": any(
            item.get("state_change") is not None
            or item.get("before_state") is not None
            or item.get("after_state") is not None
            for item in evidence
        ),
        "transfer_sequence": any(
            item.get("transfer_sequence") == "source_transport_target"
            for item in evidence
        ),
        "roi_motion": any(item.get("roi_motion") is not None for item in evidence),
        "panel_control": any(
            item.get("control_id") is not None
            or item.get("panel_roi") is not None
            or item.get("display_state_change") is not None
            for item in evidence
        ),
        "release": any(
            item.get("relation") in {"release", "separated"}
            or item.get("phase") in {
                "release",
                "target_release",
                "source_withdraw",
                "target_withdraw",
            }
            for item in evidence
        )
        or any(
            bool(
                ((candidate.provenance or {}).get("interaction_state") or {}).get(
                    "release_observed"
                )
            )
            for candidate in event.candidates
        ),
    }


def _observed_transition_trace(
    event: EvidenceEvent, present: Sequence[str]
) -> list[dict[str, Any]]:
    timed_evidence = [
        (float(timestamp), evidence)
        for candidate in event.candidates
        for evidence in candidate.evidence
        if isinstance(
            (timestamp := evidence.get("observation_global_ms")),
            (int, float),
        )
    ]

    def observed_time(predicate, *, latest: bool = False) -> float | None:
        values = [
            timestamp
            for timestamp, evidence in timed_evidence
            if predicate(evidence)
        ]
        if not values:
            return None
        return max(values) if latest else min(values)

    phase_times: dict[str, tuple[float, str]] = {}
    if event.action_type == ActionType.HAND_OBJECT_CONTACT:
        approach = observed_time(
            lambda evidence: bool(
                (evidence.get("interaction_state") or {}).get(
                    "approach_confirmed"
                )
            )
        )
        contact = observed_time(
            lambda evidence: evidence.get("distance_norm") is not None
        )
        if approach is not None:
            phase_times["approach"] = (approach, "observed_frame")
        if contact is not None:
            phase_times["contact"] = (contact, "observed_frame")
            phase_times["manipulation"] = (
                float(event.key_global_ms),
                "observed_peak_candidate",
            )
        release_values = [
            float(value)
            for candidate in event.candidates
            if isinstance(
                (
                    value := (
                        (candidate.provenance or {}).get("interaction_state")
                        or {}
                    ).get("release_observed_at_global_ms")
                ),
                (int, float),
            )
        ]
        if release_values:
            phase_times["release"] = (
                min(release_values),
                "observed_separation_frame",
            )
    elif event.action_type == ActionType.OBJECT_MOVEMENT:
        movement_start = observed_time(
            lambda evidence: evidence.get("camera_compensated_displacement_norm")
            is not None
            or evidence.get("displacement_norm") is not None
        )
        movement_end = observed_time(
            lambda evidence: evidence.get("camera_compensated_displacement_norm")
            is not None
            or evidence.get("displacement_norm") is not None,
            latest=True,
        )
        if movement_start is not None:
            phase_times["movement_start"] = (
                movement_start,
                "observed_frame",
            )
            phase_times["transport"] = (
                float(event.key_global_ms),
                "observed_peak_candidate",
            )
            phase_times["stable_before"] = (
                movement_start,
                "inferred_from_first_motion_frame",
            )
        if movement_end is not None and "stable_after" in present:
            phase_times["stable_after"] = (
                movement_end,
                "observed_release_or_motion_end",
            )
    elif event.action_type in {
        ActionType.LIQUID_MOVEMENT,
        ActionType.PIPETTE_TRANSFER_OPERATION,
    }:
        sequences = [
            evidence
            for candidate in event.candidates
            for evidence in candidate.evidence
            if evidence.get("transfer_sequence") == "source_transport_target"
        ]
        if sequences:
            source_ends = [
                float(item["source_contact_end_global_ms"])
                for item in sequences
                if isinstance(
                    item.get("source_contact_end_global_ms"), (int, float)
                )
            ]
            target_starts = [
                float(item["target_contact_start_global_ms"])
                for item in sequences
                if isinstance(
                    item.get("target_contact_start_global_ms"), (int, float)
                )
            ]
            if source_ends and target_starts:
                source_end = min(source_ends)
                target_start = min(target_starts)
                phase_times.update(
                    {
                        "source_approach": (
                            float(event.core_global_start_ms),
                            "inferred_from_candidate_boundary",
                        ),
                        "source_contact": (source_end, "observed_contact_run_end"),
                        "source_withdraw": (source_end, "observed_contact_run_end"),
                        "transport": (
                            (source_end + target_start) / 2.0,
                            "inferred_between_observed_contacts",
                        ),
                        "target_contact": (
                            target_start,
                            "observed_contact_run_start",
                        ),
                        "target_release": (
                            float(event.core_global_end_ms),
                            "inferred_from_candidate_boundary",
                        ),
                    }
                )
    elif event.action_type == ActionType.CONTAINER_STATE_CHANGE:
        first = observed_time(lambda _evidence: True)
        last = observed_time(lambda _evidence: True, latest=True)
        if first is not None:
            phase_times["state_before"] = (first, "observed_frame")
            phase_times["operation"] = (
                float(event.key_global_ms),
                "observed_peak_candidate",
            )
        if last is not None and "state_after" in present:
            phase_times["state_after"] = (last, "observed_frame")
            phase_times["state_hold"] = (last, "observed_frame")
    else:
        contact = observed_time(
            lambda evidence: evidence.get("distance_norm") is not None
        )
        if contact is not None:
            phase_times["panel_approach"] = (
                contact,
                "inferred_from_first_contact_frame",
            )
            phase_times["panel_contact"] = (contact, "observed_frame")
        control = observed_time(
            lambda evidence: evidence.get("control_id") is not None
            or evidence.get("panel_roi") is not None
            or evidence.get("display_state_change") is not None
        )
        if control is not None:
            phase_times["control_change"] = (control, "observed_frame")

    trace = []
    for phase in present:
        timestamp, source = phase_times.get(
            phase,
            (
                float(event.key_global_ms),
                "inferred_from_candidate_interval",
            ),
        )
        trace.append(
            {
                "phase": phase,
                "timestamp_us": round(timestamp * 1000.0),
                "source": source,
                "observed": source.startswith("observed"),
            }
        )
    return sorted(trace, key=lambda item: (item["timestamp_us"], item["phase"]))


def _phase_contract(
    event: EvidenceEvent,
) -> tuple[str, list[str], list[str], dict[str, str], dict[str, str]]:
    objects = set(event.objects)
    flags = _evidence_flags(event)
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    if event.action_type == ActionType.HAND_OBJECT_CONTACT:
        subtype = "tool_use" if objects & TOOL_CLASSES else "grasp_or_touch"
        required = ["approach", "contact", "manipulation", "release"]
        present = ["approach"]
        if flags["contact"]:
            present.append("contact")
        if flags["movement"] or flags["contact"]:
            present.append("manipulation")
        if flags["release"]:
            present.append("release")
        before = {"hand_object_relation": "separate_or_approaching"}
        after = {
            "hand_object_relation": (
                "released" if flags["release"] else "observation_ended_while_related"
            )
        }
    elif event.action_type == ActionType.OBJECT_MOVEMENT:
        subtype = "pick_and_place" if objects & HAND_CLASSES else "reposition"
        required = ["stable_before", "movement_start", "transport", "stable_after"]
        present = ["stable_before", "movement_start", "transport"] if flags["movement"] else []
        if flags["release"]:
            present.append("stable_after")
        before = {"object_motion": "stable"}
        after = {"object_motion": "stable" if flags["release"] else "movement_observation_ended"}
    elif event.action_type in {
        ActionType.LIQUID_MOVEMENT,
        ActionType.PIPETTE_TRANSFER_OPERATION,
    }:
        subtype = (
            "pipette_transfer"
            if objects & {"pipette", "spearhead"}
            else "vessel_pour"
        )
        required = [
            "source_approach",
            "source_contact",
            "source_withdraw",
            "transport",
            "target_contact",
            "target_release",
        ]
        present = []
        if flags["transfer_sequence"]:
            contact_only = any(
                evidence.get("contact_sequence_only") is True
                for candidate in event.candidates for evidence in candidate.evidence
                if evidence.get("transfer_sequence") == "source_transport_target"
            )
            present = ["source_contact", "target_contact"] if contact_only else list(required)
        elif flags["contact"] or flags["roi_motion"]:
            present = ["source_approach", "source_contact"]
        before = {
            "tool": "outside_source",
            "source": "not_interacting",
            "target": "not_interacting",
        }
        operational_only = (
            event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION
        )
        after = {
            "tool": (
                "withdrawn_from_target"
                if flags["transfer_sequence"]
                else "source_side_only_or_unknown"
            ),
            "source": (
                "tool_contact_observed"
                if operational_only and present
                else "contents_interacted"
                if present
                else "unknown"
            ),
            "target": (
                "tool_contact_observed"
                if operational_only and flags["transfer_sequence"]
                else "contents_interacted"
                if flags["transfer_sequence"]
                else "not_observed"
            ),
        }
        if flags["transfer_sequence"] and "transport" not in present:
            after = {
                "tool": "target_proximity_only_release_unknown",
                "source": "tool_proximity_observed_contents_unknown",
                "target": "tool_proximity_observed_contents_unknown",
            }
    elif event.action_type == ActionType.CONTAINER_STATE_CHANGE:
        if "bottle_cap" in objects or "tube_cap" in objects:
            subtype = "cap_remove_or_replace"
        elif "reagent_bottle_open" in objects:
            subtype = "open_state_transition"
        else:
            subtype = "container_state_transition"
        required = ["state_before", "operation", "state_after", "state_hold"]
        present = ["state_before", "operation"]
        semantic_proof = (event.model_understanding or {}).get("action_proof") or {}
        semantic_transition_complete = bool(
            semantic_proof.get("container_before_state_visible") is True
            and semantic_proof.get("container_after_state_visible") is True
            and semantic_proof.get("container_state_transition_completed") is True
        )
        if flags["state_change"] or semantic_transition_complete:
            present.extend(["state_after", "state_hold"])
        before = {"container": "observed_before_state"}
        after = {
            "container": (
                "observed_changed_state"
                if "state_after" in present
                else "after_state_not_observed"
            )
        }
    else:
        subtype = "device_control_operation"
        required = ["panel_approach", "panel_contact", "control_change", "withdraw"]
        present = ["panel_approach"]
        if flags["contact"]:
            present.append("panel_contact")
        if flags["panel_control"]:
            present.append("control_change")
        if flags["release"]:
            present.append("withdraw")
        before = {"device_control": "not_interacting"}
        after = {
            "device_control": (
                "changed" if flags["panel_control"] else "change_not_directly_observed"
            )
        }
    return subtype, required, list(dict.fromkeys(present)), before, after


def build_event_state_receipt(
    event: EvidenceEvent, config: dict[str, Any]
) -> dict[str, Any]:
    subtype, required, present, before, after = _phase_contract(event)
    state_cfg = config.get("action_state_machine", {})
    minimum_complete = float(state_cfg.get("minimum_phase_completeness", 0.75))
    completeness = len(set(present) & set(required)) / max(1, len(required))
    both_roles = {
        role.value if isinstance(role, ViewRole) else str(role)
        for role in event.supporting_roles
    } == {ViewRole.FIRST_PERSON.value, ViewRole.THIRD_PERSON.value}
    tracks = _track_tokens(event)
    if not event.accepted:
        lifecycle = "rejected"
        terminal_reason = event.audit_reason
    elif not event_is_formal(event):
        lifecycle = "provisional"
        terminal_reason = "semantic_adjudication_required_before_formal_use"
    elif completeness >= minimum_complete:
        lifecycle = "completed"
        terminal_reason = "required_phases_observed"
    else:
        lifecycle = "incomplete_end"
        terminal_reason = "observation_ended_before_all_required_phases"

    start_us = round(event.global_start_ms * 1000.0)
    end_us = round(event.global_end_ms * 1000.0)
    span_us = max(1, end_us - start_us)
    if config.get("performance", {}).get(
        "action_state_observed_timestamps_enabled", False
    ):
        trace = _observed_transition_trace(event, present)
        trace_policy = "observed_timestamps_with_explicit_inference_labels"
    else:
        trace = [
            {
                "phase": phase,
                "timestamp_us": round(
                    start_us + span_us * index / max(1, len(present) - 1)
                ),
                "source": "deterministic_cv_state_receipt",
            }
            for index, phase in enumerate(present)
        ]
        trace_policy = "legacy_evenly_distributed_phase_receipt"
    non_hand = sorted(set(event.objects) - HAND_CLASSES)
    return {
        "schema_version": "visioncortex-continuous-action-state/1.0.0",
        "event_id": event.event_id,
        "action_type": _normalized_action(event.action_type),
        "action_subtype": subtype,
        "time_basis": "aligned_global_timeline_microseconds",
        "start_us": start_us,
        "end_us": end_us,
        "peak_timestamp_us": round(event.key_global_ms * 1000.0),
        "lifecycle_state": lifecycle,
        "terminal_reason": terminal_reason,
        "required_phases": required,
        "phases": present,
        "missing_phases": [phase for phase in required if phase not in present],
        "transition_trace": trace,
        "transition_trace_policy": trace_policy,
        "state_before": before,
        "state_after": after,
        "object_identity": {
            "object_classes": non_hand,
            "track_tokens": tracks,
            "identity_status": "tracked" if tracks else "class_only",
        },
        "cross_view": {
            "supporting_views": sorted(event.supporting_views),
            "supporting_roles": sorted(
                role.value if isinstance(role, ViewRole) else str(role)
                for role in event.supporting_roles
            ),
            "both_roles": both_roles,
        },
        "scores": {
            "temporal_continuity": round(min(1.0, 0.45 + event.confidence * 0.55), 4),
            "object_identity": round(0.9 if tracks else min(0.7, 0.3 + 0.1 * len(non_hand)), 4),
            "phase_completeness": round(completeness, 4),
            "cross_view_support": 0.95 if both_roles else 0.45,
            "state_change_support": round(
                1.0 if "state_after" in present or (
                    _evidence_flags(event)["transfer_sequence"] and "transport" in present
                ) else 0.3,
                4,
            ),
            "contradiction_penalty": 0.0,
        },
        "publication": {
            "status": (
                "primary"
                if event_is_formal(event)
                else "provisional_semantic_review"
                if event.accepted
                else "internal_candidate"
            ),
            "suppressed_by_event_id": None,
            "reason": None,
        },
    }


def _overlaps(left: EvidenceEvent, right: EvidenceEvent) -> bool:
    return min(left.global_end_ms, right.global_end_ms) >= max(
        left.global_start_ms, right.global_start_ms
    )


def attach_continuous_action_states(
    events: Sequence[EvidenceEvent], config: dict[str, Any]
) -> dict[str, Any]:
    """Attach auditable state receipts without mutating the frozen CV decision.

    Higher-level events suppress duplicate publication of their component
    contact/movement evidence, but the lower-level observations remain indexed.
    """

    for event in events:
        event.state_machine = build_event_state_receipt(event, config)
        semantic_recall = (event.observability or {}).get(
            "semantic_recall_admission"
        )
        if (
            isinstance(semantic_recall, dict)
            and semantic_recall.get("mandatory_semantic_review") is True
            and semantic_recall.get("candidate_action_directly_confirmed")
            is not True
        ):
            event.state_machine["publication"] = {
                "status": "provisional_semantic_review",
                "suppressed_by_event_id": None,
                "reason": "independent_event_level_state_proof_required",
            }

    primary_types = {
        ActionType.LIQUID_MOVEMENT,
        ActionType.DEVICE_PANEL_OPERATION,
    }
    for component in events:
        if not component.accepted or component.action_type not in {
            ActionType.HAND_OBJECT_CONTACT,
            ActionType.OBJECT_MOVEMENT,
        }:
            continue
        component_objects = set(component.objects) - HAND_CLASSES
        for primary in events:
            # Provisional semantic candidates are deliberately kept visible
            # as review material. They cannot suppress a separately observed
            # contact/movement event before formal admission.
            if (
                not event_is_formal(primary)
                or primary.action_type not in primary_types
                or primary.state_machine["publication"]["status"] != "primary"
            ):
                continue
            if not _overlaps(component, primary):
                continue
            primary_objects = set(primary.objects) - HAND_CLASSES
            if not component_objects & primary_objects:
                continue
            component.state_machine["publication"] = {
                "status": "component_only",
                "suppressed_by_event_id": primary.event_id,
                "reason": "higher_level_action_contains_same_physical_evidence",
            }
            break

    state_counts = Counter(
        event.state_machine.get("lifecycle_state", "unknown") for event in events
    )
    publication_counts = Counter(
        event.state_machine.get("publication", {}).get("status", "unknown")
        for event in events
    )
    fragments: dict[tuple[str, tuple[str, ...]], list[EvidenceEvent]] = defaultdict(list)
    for event in events:
        if not event_is_formal(event):
            continue
        key = (
            _normalized_action(event.action_type),
            tuple(sorted(set(event.objects) - HAND_CLASSES)),
        )
        fragments[key].append(event)
    fragment_counts = [len(group) for group in fragments.values()]
    return {
        "schema_version": "visioncortex-continuous-action-state-ledger/1.0.0",
        "policy": "timestamp_based_shadow_gate_preserves_cv_acceptance",
        "cv_acceptance_mutated": False,
        "event_count": len(events),
        "lifecycle_counts": dict(sorted(state_counts.items())),
        "publication_counts": dict(sorted(publication_counts.items())),
        "fragmentation": {
            "identity_group_count": len(fragment_counts),
            "event_count": sum(fragment_counts),
            "mean_events_per_identity_group": round(
                sum(fragment_counts) / max(1, len(fragment_counts)), 4
            ),
            "maximum_events_per_identity_group": max(fragment_counts, default=0),
        },
        "receipts": [event.state_machine for event in events],
    }
