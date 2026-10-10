import copy

import pytest

from visioncortex.operation_review import expand_steps, validate_steps
from visioncortex.pipeline import refine_groups_from_final_events, validate_final_step_action_consistency
from visioncortex.schemas import ActionType, EvidenceEvent, ExperimentGroup
from visioncortex.step_evidence import next_operation, organization_metadata


def event(identity, start, end, action=ActionType.HAND_OBJECT_CONTACT):
    return EvidenceEvent(event_id=identity, action_type=action,
                         global_start_ms=start, global_end_ms=end, key_global_ms=start,
                         objects=["bottle"], confidence=.9, accepted=True,
                         audit_reason="deterministic fixture", candidates=[],
                         supporting_views=["fp", "tp"],
                         supporting_roles=["first_person", "third_person"],
                         model_understanding={"status": "completed", "current_step": "握持瓶身"})


def group():
    return ExperimentGroup(group_id="group", global_start_ms=0, global_end_ms=30000,
                           continuity_type="independent", atomic_experiment_ids=[],
                           participating_views=["fp", "tp"], first_person_view="fp",
                           third_person_view="tp", continuity_reason="test",
                           key_event_ids=["a", "b"])


def test_later_evidence_never_becomes_current_step_support():
    current, later = event("a", 1000, 2000), event("b", 5000, 6000, ActionType.CONTAINER_STATE_CHANGE)
    current.model_understanding.update(next_step="旋开瓶盖", next_step_evidence={
        "status": "observed", "evidence_event_ids": ["b"]})
    g = group()
    original = copy.deepcopy(current.model_understanding)
    refine_groups_from_final_events([g], [current, later])
    step = g.model_understanding["steps"][0]
    assert step["supporting_event_ids"] == ["a"]
    assert step["next_step_evidence"]["evidence_event_ids"] == ["b"]
    assert step["next_step_status"] == "observed"
    assert step["time_scope"]["complete_operation_boundaries_proven"] is False
    assert current.model_understanding == original


@pytest.mark.parametrize("fault", ["self", "missing", "overlap", "rejected", "incomplete", "empty", "duplicate"])
def test_next_operation_needs_a_distinct_later_admitted_event(fault):
    current, later = event("a", 1000, 2000), event("b", 5000, 6000)
    ids = ["b"]
    if fault == "self":
        ids = ["a"]
    elif fault == "missing":
        ids = ["absent"]
    elif fault == "overlap":
        later.global_start_ms = 1900
    elif fault == "rejected":
        later.accepted = False
    elif fault == "incomplete":
        later.model_understanding["status"] = "failed"
    elif fault == "empty":
        ids = []
    elif fault == "duplicate":
        ids *= 2
    current.model_understanding.update(next_step="拿起瓶子", next_step_evidence={
        "status": "observed", "evidence_event_ids": ids})
    before = current.model_dump()
    result = next_operation(current, [current, later], {"a"})
    assert result["next_step_status"] == "unknown"
    assert result["source_next_step"]["text"] == "拿起瓶子"
    assert result["source_next_step"]["evidence"]["evidence_event_ids"] == ids
    assert current.model_dump() == before


def test_other_steps_cannot_lend_action_admission_to_current_prose():
    a, b = event("a", 1000, 2000), event("b", 5000, 6000, ActionType.CONTAINER_STATE_CHANGE)
    g = group()
    g.model_understanding = {"steps": [{"step_index": 1, "supporting_event_ids": ["a"],
                                          "current_step": "旋开瓶盖"}]}
    assert not validate_final_step_action_consistency([g], [a, b])["passed"]
    g.model_understanding["steps"][0]["supporting_event_ids"] = ["b"]
    assert validate_final_step_action_consistency([g], [a, b])["passed"]


def test_no_reference_cannot_borrow_from_multi_event_group():
    g = group()
    g.model_understanding = {"steps": [{"current_step": "旋开瓶盖"}]}
    assert not validate_final_step_action_consistency([g], [
        event("a", 1000, 2000), event("b", 5000, 6000, ActionType.CONTAINER_STATE_CHANGE)
    ])["passed"]


def test_merged_step_views_must_support_each_referenced_event():
    a, b = event("a", 1000, 2000), event("b", 1900, 2200)
    b.supporting_views = ["fp"]
    raw = {"status": "completed", "steps": [{"operation_title": "握持瓶身",
        "current_step": "握持瓶身", "observed_result": "瓶子仍在手中", "supporting_event_ids": ["a", "b"]}]}
    expanded = expand_steps([a, b], raw)
    expanded["steps"][0]["supporting_views"] = ["tp"]
    with pytest.raises(ValueError, match="机位"):
        validate_steps(group(), [a, b], expanded)


def test_clip_context_is_retained_without_becoming_current_action_outcome():
    a = event("a", 1000, 2000, ActionType.DEVICE_PANEL_OPERATION)
    a.model_understanding.update(current_step="按触面板，随后折纸再拿起瓶子",
        physical_change={"before": "手按触面板", "after": "纸被折叠、瓶子在手中"},
        action_proof={"reason": "可见面板接触"}, temporal_support={"late": "折纸"})
    metadata = organization_metadata(group(), [a])
    assert metadata["events"][0]["action_review"]["action_proof"]["reason"] == "可见面板接触"
    row = expand_steps([a], {"status": "completed", "steps": [{
        "operation_title": "按触天平面板", "current_step": "手指按触天平面板按键区域。",
        "observed_result": "显示读数变化未确认。", "supporting_event_ids": ["a"]}]})["steps"][0]
    assert "折纸" not in row["physical_change"]
    assert row["source_operation_records"][0]["physical_change"] == a.model_understanding["physical_change"]
    assert row["source_operation_records"][0]["current_step"] == a.model_understanding["current_step"]


def test_result_text_cannot_introduce_unadmitted_functional_action():
    a = event("a", 1000, 2000)
    expanded = expand_steps([a], {"status": "completed", "steps": [{
        "operation_title": "握持瓶身", "current_step": "握持瓶身",
        "observed_result": "旋开瓶盖", "supporting_event_ids": ["a"]}]})
    with pytest.raises(ValueError, match="不一致"):
        validate_steps(group(), [a], expanded)


def test_merged_next_step_uses_latest_end_and_sees_later_allocation():
    a, b, later = event("a", 1000, 2400), event("b", 1900, 2000), event("c", 5000, 6000)
    a.model_understanding.update(next_step="后续握瓶", next_step_evidence={
        "status": "observed", "evidence_event_ids": ["c"]})
    expanded = expand_steps([a, b], {"steps": [{"operation_title": "握瓶",
        "current_step": "握持瓶身", "supporting_event_ids": ["a", "b"]}]}, [a, b, later])
    assert expanded["steps"][0]["next_step_status"] == "observed"
    assert expanded["steps"][0]["next_step_evidence"]["evidence_event_ids"] == ["c"]
