import copy
import json

import pytest

from visioncortex.operation_review import apply, bindings, coverage, validate_steps, CONTROL, GROUPS, EVENTS
from visioncortex.schemas import ActionType, EvidenceEvent, ExperimentGroup


def event(identity="e1", start=1000, end=2000, action=ActionType.OBJECT_MOVEMENT):
    return EvidenceEvent(event_id=identity, action_type=action, global_start_ms=start,
        global_end_ms=end, key_global_ms=(start+end)/2, objects=["bottle"], confidence=.9,
        accepted=True, supporting_roles=["first_person","third_person"], candidates=[], audit_reason="test", supporting_views=["fp", "tp"],
        model_understanding={"status":"completed", "current_step":"拿起瓶子"})


def group():
    return ExperimentGroup(group_id="g1", continuity_type="independent", atomic_experiment_ids=[],
        global_start_ms=0, global_end_ms=30000, participating_views=["fp","tp"],
        first_person_view="fp", third_person_view="tp", continuity_reason="unreviewed",
        key_event_ids=["e1","e2"], completion_status="ongoing_at_recording_end")


def step(ids=None, start=1000, end=2000, title="拿起瓶子"):
    return {"supporting_event_ids":ids or ["e1"], "start_global_ms":start,"end_global_ms":end,
            "operation_title":title,"current_step":title,"physical_change":"瓶子离开桌面",
            "supporting_views":["fp"], "next_step":"未知", "next_step_status":"unknown"}


def test_operation_merges_evidence_without_changing_source_or_completion():
    events=[event(),event("e2",1900,2400)]
    before=[e.model_dump() for e in events]
    result=validate_steps(group(),events,{"status":"completed","steps":[step(["e1","e2"],end=2400)]})
    assert len(result)==1 and result[0]["evidence_record_count"]==2
    assert len(result[0]["evidence_intervals"])==2
    assert before==[e.model_dump() for e in events]
    assert group().completion_status=="ongoing_at_recording_end"


@pytest.mark.parametrize("steps",[[step()],[step(),step()],[step(["fake"])],[step(start=0)]])
def test_rejects_missing_duplicate_unknown_or_expanded_evidence(steps):
    with pytest.raises(ValueError):
        validate_steps(group(),[event(),event("e2",2100,2400)],{"status":"completed","steps":steps})


def test_distant_repeated_operations_do_not_merge():
    with pytest.raises(ValueError,match="未观察区间"):
        validate_steps(group(),[event(),event("e2",10000,11000)],{"status":"completed",
            "steps":[step(["e1","e2"],end=11000)]})


def test_cannot_borrow_other_steps_functional_action_evidence():
    events=[event(),event("e2",2100,2400,ActionType.CONTAINER_STATE_CHANGE)]
    with pytest.raises(ValueError,match="不一致"):
        validate_steps(group(),events,{"status":"completed","steps":[step(title="旋开瓶盖"),
            step(["e2"],2100,2400,title="旋开瓶盖")]})


def test_coverage_is_not_claimed_operation_recall_and_zero_time_is_valid():
    g=group().model_dump()
    result=coverage(g,[step(start=0,end=1000),step(start=15000,end=20000)])
    assert result["review_intervals"]==[{"start_ms":1000.,"end_ms":15000.,"seconds":14.},
                                          {"start_ms":20000.,"end_ms":30000.,"seconds":10.}]
    assert result["all_operator_steps_proven"] is False
    assert result["completion_status"]=="ongoing_at_recording_end"


@pytest.mark.parametrize("state", ["unreviewed", "unresolved", "ongoing_at_recording_end"])
def test_final_step_pass_does_not_turn_request_completion_into_experiment_completion(state):
    from visioncortex.pipeline import refine_groups_from_final_events
    g=group()
    g.completion_status=state
    g.model_understanding={"status":"completed","boundary_assessment":{"end_complete":True}}
    refine_groups_from_final_events([g], [])
    assert g.model_understanding['boundary_assessment']['end_complete'] is False
    assert g.model_understanding['pre_curation_understanding']['boundary_assessment']['end_complete'] is True
    assert g.completion_status==state


def test_overlay_is_bound_to_current_evidence_and_does_not_rewrite_files(tmp_path):
    for name in (GROUPS,EVENTS):
        p=tmp_path/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('{}')
    g=group().model_dump()
    g['model_understanding']={'steps':[step()]}
    revised=copy.deepcopy(g)
    revised['model_understanding'].update(steps=[step(title="放下瓶子")],overall_summary="具体操作",
        operation_review={"accepted":True},operation_coverage={"status":"PARTIAL_EVIDENCE"})
    (tmp_path/CONTROL).write_text(json.dumps({"bindings":bindings(tmp_path),"groups":{"g1":revised}}))
    assert apply(tmp_path,[g])[0]['model_understanding']['steps'][0]['operation_title']=="放下瓶子"
    assert g['model_understanding']['steps'][0]['operation_title']=="拿起瓶子"
    (tmp_path/EVENTS).write_text('{"new":true}')
    assert apply(tmp_path,[g])==[g]


def test_prose_response_keeps_timing_views_and_outcome_from_event_ledger():
    from visioncortex.operation_review import expand_steps
    events = [event(), event("e2", 1900, 2400)]
    events[1].supporting_views = ["tp"]
    events[1].model_understanding.update(
        physical_change={"before": "瓶子在台面", "after": "瓶子在手中"},
        next_step="放下瓶子", next_step_evidence={"status": "predicted"})
    original = [e.model_dump() for e in events]
    result = expand_steps(events, {"status": "completed", "steps": [{
        "operation_title": "拿起瓶子", "current_step": "手握瓶身并将瓶子抬离台面。",
        "supporting_event_ids": ["e2", "e1"]}]})
    row = validate_steps(group(), events, result)[0]
    assert (row["start_global_ms"], row["end_global_ms"]) == (1000, 2400)
    assert row["supporting_views"] == ["tp"]
    assert row["next_step_status"] == "predicted"
    assert row["source_operation_records"][0]["current_step"] == "拿起瓶子"
    assert [e.model_dump() for e in events] == original


def test_operation_response_contract_cannot_rewrite_experiment_boundary():
    from pydantic import ValidationError
    from visioncortex.mllm import _validate_response_payload
    payload = {"steps": [{"operation_title": "拿起瓶子", "current_step": "抬离台面。",
                           "supporting_event_ids": ["e1"]}]}
    assert _validate_response_payload(payload, "operations") == payload
    with pytest.raises(ValidationError):
        _validate_response_payload({**payload, "boundary_assessment": {"end_complete": True}}, "operations")


def test_allocations_share_one_bounded_pool_and_verified_results_are_reused(tmp_path, default_config, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from visioncortex import operation_review as module

    config = copy.deepcopy(default_config)
    config["storage"]["local_cache_root"] = str(tmp_path / "cache")
    config["mllm"].update(group_workers=2, max_images_per_group=2)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"test frame bytes, not real media")
    events = [event(f"e{i}", 1000 + i * 4000, 2000 + i * 4000) for i in range(8)]
    for e in events:
        e.key_frames = {"fp": "frame.jpg"}
    groups = []
    for i in range(2):
        g = group()
        g.group_id = f"g{i}"
        g.key_event_ids = [e.event_id for e in events[i*4:(i+1)*4]]
        groups.append(g)
    pool_limits, requests = [], []
    barrier = threading.Barrier(2)

    def pool(*, max_workers):
        pool_limits.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    class Analyzer:
        def __init__(self, config):
            pass

        def _call(self, prompt, metadata, images, *, max_images, response_kind):
            assert max_images == 2 and response_kind == "operations"
            requests.append(metadata)
            barrier.wait(timeout=5)
            return {"status": "completed", "attempts": 1, "steps": [
                {"operation_title": "拿起瓶子", "current_step": "手握瓶子并抬离台面。",
                 "supporting_event_ids": [e["event_id"]]} for e in metadata["events"]]}

        def close(self):
            pass

    monkeypatch.setattr(module, "ThreadPoolExecutor", pool)
    monkeypatch.setattr(module, "ArkAnalyzer", Analyzer)
    records = module.review(tmp_path, groups, events, config)
    assert pool_limits == [2] and len(requests) == 4
    assert all(r["accepted"] for r in records)
    for g in groups:
        assert g.completion_status == "ongoing_at_recording_end"
        assert len(g.model_understanding["steps"]) == 4
    again = module.review(tmp_path, groups, events, config)
    assert len(requests) == 4
    assert all(c["cache_reused"] for r in again for c in r["calls"])
    assert len(list((tmp_path / "JSON-Config-Files/Operation-Reviews/Allocations").glob("*.json"))) == 4
    # Accepted source-identical responses survive a validator-only update even
    # when an earlier local verdict prevented semantic cache admission.
    from visioncortex import archive
    monkeypatch.setattr(archive, "_read_semantic_cache", lambda *_: None)
    for path in (tmp_path / "JSON-Config-Files/Operation-Reviews/Allocations").glob("*.json"):
        data = json.loads(path.read_text())
        data.update(organization_accepted=False, validation_error="older local check")
        path.write_text(json.dumps(data))
    revised = module.review(tmp_path, groups, events, config)
    assert len(requests) == 4
    assert all(c["cache_reused"] and c["organization_accepted"] and "validation_error" not in c
               for r in revised for c in r["calls"])
