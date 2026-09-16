"""Keep present actions, surrounding observations and later evidence separate."""
from __future__ import annotations

from copy import deepcopy

from .schemas import event_is_formal


def next_operation(last_event, events, current_ids):
    """An observed next step needs a later admitted event, not a self-reference.

    The original semantic statement is retained even when its position in the
    sequence cannot be verified. It must not expand the current step's evidence.
    """
    model = last_event.model_understanding or {}
    source = deepcopy(model.get("next_step_evidence") or {})
    status = source.get("status", "unknown")
    if status in {"predicted", "inferred"}:
        return {"next_step": model.get("next_step") or "未知",
                "next_step_status": "inferred",
                "next_step_evidence": {**source, "status": "inferred"}}
    if status != "observed":
        return {"next_step": "未知", "next_step_status": "unknown",
                "next_step_evidence": {**source, "status": "unknown"}}
    by_id = {event.event_id: event for event in events}
    ids = source.get("evidence_event_ids") or []
    valid = bool(ids) and len(ids) == len(set(ids)) and all(
        identity not in current_ids
        and identity in by_id
        and event_is_formal(by_id[identity])
        and (by_id[identity].model_understanding or {}).get("status") == "completed"
        and by_id[identity].global_start_ms >= last_event.global_end_ms
        for identity in ids
    )
    if valid:
        return {"next_step": model.get("next_step") or "未知",
                "next_step_status": "observed", "next_step_evidence": source}
    return {"next_step": "后续操作尚未定位到独立的时间证据",
            "next_step_status": "unknown",
            "next_step_evidence": {"status": "unknown", "evidence_event_ids": [],
                                   "reason": "missing_later_admitted_event"},
            "source_next_step": {"text": model.get("next_step"), "evidence": source}}


def timing_scope(events):
    """CV event intervals locate evidence; they are not full operation bounds."""
    return {"status": "PARTIAL_EVIDENCE", "basis": "adjudicated_event_intervals",
            "complete_operation_boundaries_proven": False,
            "missing_gate": "operation_onset_and_completion_frame_review",
            "event_intervals": [
                {"event_id": e.event_id, "start_global_ms": e.global_start_ms,
                 "end_global_ms": e.global_end_ms, "key_global_ms": e.key_global_ms}
                for e in events]}


def organization_metadata(group, events):
    """Do not present a clip-wide narrative as a precisely timed atomic step."""
    return {"group_id": group.group_id, "completion_status": group.completion_status,
            "time_scope": timing_scope(events),
            "events": [{
                "event_id": e.event_id, "action_type": e.action_type.value,
                "start_ms": e.global_start_ms, "end_ms": e.global_end_ms,
                "supporting_views": list(e.supporting_views),
                "reviewed_operation": {
                    k: (e.model_understanding or {}).get(k) for k in (
                        "operation_title", "current_step", "physical_change", "uncertainties")},
                "action_review": {
                    k: (e.model_understanding or {}).get(k) for k in (
                        "action_type_confirmed", "action_proof", "evidence_verdict",
                        "confirmed_action_support_by_view", "cross_view_consistency",
                        "temporal_support", "selected_keyframe_observations")},
                "following_context_not_current_evidence": {
                    "text": (e.model_understanding or {}).get("next_step"),
                    "evidence": (e.model_understanding or {}).get("next_step_evidence")},
            } for e in events]}
