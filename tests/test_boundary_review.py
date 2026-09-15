from copy import deepcopy

import pytest

from visioncortex.boundary_review import join_allowed, reconcile_groups, review_times
from visioncortex.mllm import ArkAnalyzer, GROUP_SYSTEM_PROMPT
from visioncortex.schemas import ActionCandidate, ActionType, EvidenceEvent, ExperimentGroup, ExperimentSegment, ViewRole, ViewInput


def groups():
    return [ExperimentGroup(
        group_id=f"GROUP-{i:04d}", group_uid=f"uid-{i}", continuity_type="independent",
        atomic_experiment_ids=[f"EXP-{i}"], global_start_ms=start, global_end_ms=end,
        participating_views=["fp", "tp"], first_person_view="fp", third_person_view="tp",
        continuity_reason="gap exceeded, not completion evidence",
        videos={"first-person": f"old-{i}.mp4"}, archive_folder=f"old-{i}",
        model_understanding={"boundary_assessment": {"end_complete": True}},
    ) for i, (start, end) in enumerate([(477092, 577050), (588200, 725700)], 1)]


def review():
    return {
        "left_group_uid": "uid-1", "right_group_uid": "uid-2", "minimum_confidence": 0.85,
        "frames": [{"frame_id": "F001", "global_ms": 576000},
                   {"frame_id": "F002", "global_ms": 580000},
                   {"frame_id": "F003", "global_ms": 590000}],
        "result": {"status": "completed", "decision": {
            "relation": "same_experiment", "basis": "preparation_to_execution",
            "same_operator": True, "workstation_changed": False, "left_experiment_complete": False,
            "left_unit_name": "称量准备", "right_unit_name": "固体称量", "handoff_frame_ids": [],
            "before_frame_ids": ["F001"], "after_frame_ids": ["F002", "F003"],
            "observation": "Same paper is handled then put into balance; preparation continues",
            "confidence": 0.96, "uncertainties": [],
        }},
    }


def test_review_samples_both_cut_edges_and_the_previously_omitted_gap():
    left, right = groups()
    times = review_times(left, right)
    assert len(times) == 20
    assert left.global_end_ms in times and right.global_start_ms in times
    assert any(left.global_end_ms < t < right.global_start_ms for t in times)
    assert min(times) < left.global_end_ms and max(times) > right.global_start_ms


@pytest.mark.parametrize("changes", [
    {"relation": "uncertain"}, {"basis": "insufficient_evidence"},
    {"left_experiment_complete": True}, {"same_operator": False},
    {"confidence": 0.84}, {"after_frame_ids": ["F002"]},
    {"after_frame_ids": ["invented-frame"]}, {"before_frame_ids": ["F003"]},
])
def test_no_join_on_missing_continuity_or_invalid_frame_references(changes):
    left, right = groups()
    r = review()
    r["result"]["decision"].update(changes)
    assert not join_allowed(left, right, r["result"], r["frames"])


def test_different_experimenters_or_workstations_never_join_by_time_alone():
    left, right = groups()
    r = review()
    for field in ("first_person_view", "third_person_view"):
        other = right.model_copy(update={field: "another-camera"})
        assert not join_allowed(left, other, r["result"], r["frames"])
    assert not join_allowed(left, right, {"status": "failed"}, r["frames"])


def test_join_keeps_atomic_membership_gap_and_invalidates_old_video_and_narrative():
    original = groups()
    segments = [ExperimentSegment(segment_id=f"EXP-{i}", global_start_ms=g.global_start_ms,
                                  global_end_ms=g.global_end_ms, event_ids=[f"event-{i}"],
                                  participating_views=["fp", "tp"])
                for i, g in enumerate(original, 1)]
    before = deepcopy([s.event_ids for s in segments])
    result = reconcile_groups(original, segments, [], [review()])
    assert len(result) == 1
    merged = result[0]
    assert (merged.global_start_ms, merged.global_end_ms) == (477092, 725700)
    assert merged.atomic_experiment_ids == ["EXP-1", "EXP-2"]
    assert [s.event_ids for s in segments] == before
    assert all(s.group_id == merged.group_id for s in segments)
    assert merged.model_understanding is None
    assert merged.archive_folder is None and merged.videos == {} and merged.video_json == {}
    assert original[0].videos == {"first-person": "old-1.mp4"}


def test_uncertain_review_preserves_original_cut_without_inventing_a_completion():
    original = groups()
    r = review()
    r["result"]["decision"]["relation"] = "uncertain"
    segments = [ExperimentSegment(segment_id=f"EXP-{i}", global_start_ms=g.global_start_ms,
                                  global_end_ms=g.global_end_ms, event_ids=[],
                                  participating_views=["fp", "tp"])
                for i, g in enumerate(original, 1)]
    result = reconcile_groups(original, segments, [], [r])
    assert len(result) == 2
    assert result[0].boundary_reviews[0]["result"]["decision"]["relation"] == "uncertain"


def test_group_prompt_does_not_feed_cv_separation_back_as_semantic_proof():
    analyzer = object.__new__(ArkAnalyzer)
    analyzer.config = {"max_images_per_group": 12, "max_images_per_event": 12}
    captured = {}
    analyzer._call = lambda prompt, metadata, *args, **kwargs: captured.update(metadata)
    analyzer.analyze_group(groups()[0], [], [], [])
    assert "rule_based_continuity_reason" not in captured
    assert "rule_based_continuity_type" not in captured
    assert "end_complete 必须为 false" in GROUP_SYSTEM_PROMPT


def test_known_incomplete_experiment_cannot_pass_boundary_integrity():
    from visioncortex.validation import validate_experiment_and_material_quality

    g = groups()[0]
    g.model_understanding = {"boundary_assessment": {"end_complete": False,
                                                    "localized_rescan_needed": True}}
    report = validate_experiment_and_material_quality([g], [], None)
    integrity = report["segmentation_integrity"]
    assert integrity["passed"] is False
    assert integrity["boundary_completion_passed"] is False
    assert "experiment_end_incomplete" in integrity["incomplete_boundaries"][0]["reasons"]


def test_cross_station_sample_handoff_forms_workflow_with_distinct_units():
    left, right = groups()
    right.third_person_view = "pipette-camera"
    right.participating_views = ["fp", "pipette-camera"]
    r = review()
    r["result"]["decision"].update(relation="continuous_workflow", basis="sample_handoff",
        workstation_changed=True, left_experiment_complete=True, handoff_frame_ids=["F002"],
        left_unit_name="配液", right_unit_name="移液")
    assert join_allowed(left, right, r["result"], r["frames"])
    segments = [ExperimentSegment(segment_id=f"EXP-{i}", global_start_ms=g.global_start_ms,
                                  global_end_ms=g.global_end_ms, event_ids=[],
                                  participating_views=g.participating_views)
                for i, g in enumerate([left, right], 1)]
    merged = reconcile_groups([left, right], segments, [], [r])[0]
    assert merged.workflow_kind == "continuous_workflow"
    assert [u["name"] for u in merged.workflow_units] == ["配液", "移液"]
    assert any(row["third_person_view"] is None for row in merged.view_timeline)
    assert merged.view_timeline[-1]["third_person_view"] == "pipette-camera"
    r["result"]["decision"]["handoff_frame_ids"] = []
    assert not join_allowed(left, right, r["result"], r["frames"])


def test_interleaved_experimenters_can_never_be_merged_with_each_other():
    left, right = groups()
    other = left.model_copy(deep=True)
    other.group_uid = "someone-else"
    other.first_person_view = "other-wearable"
    other.atomic_experiment_ids = ["EXP-other"]
    segments = [ExperimentSegment(segment_id=g.atomic_experiment_ids[0],
                                  global_start_ms=g.global_start_ms, global_end_ms=g.global_end_ms,
                                  event_ids=[], participating_views=g.participating_views)
                for g in [left, other, right]]
    output = reconcile_groups([left, other, right], segments, [], [review()])
    assert len(output) == 2
    assert output[0].atomic_experiment_ids == ["EXP-1", "EXP-2"]
    assert output[1].atomic_experiment_ids == ["EXP-other"]


def tail_review():
    return {"left_end_ms": 725700, "kind": "tail", "frames": [
        {"frame_id": "F001", "global_ms": 720000},
        {"frame_id": "F020", "global_ms": 899673}],
        "result": {"status": "completed", "decision": {
            "state": "ongoing", "continuation_observed": True,
            "frame_ids": ["F001", "F020"], "last_continuation_frame_id": "F020",
            "completion_frame_id": None, "observation": "Same container carried then pipetting continues",
            "confidence": .95, "uncertainties": []}}}


def test_source_end_retains_tail_but_does_not_claim_experiment_finished():
    from visioncortex.boundary_review import apply_tail_review
    g = groups()[1]
    g.view_timeline = [{"start_ms": g.global_start_ms, "end_ms": g.global_end_ms,
                        "third_person_view": "tp"}]
    assert apply_tail_review(g, tail_review(), 899773)
    assert g.global_end_ms == 899773
    assert g.completion_status == "ongoing_at_recording_end"
    assert g.view_timeline[-1]["third_person_view"] is None


@pytest.mark.parametrize("changes", [{"state": "uncertain"}, {"confidence": .7},
    {"frame_ids": ["invented"]}, {"last_continuation_frame_id": "F001"},
    {"state": "completed", "completion_frame_id": None}])
def test_failed_tail_evidence_cannot_extend_or_complete(changes):
    from visioncortex.boundary_review import apply_tail_review
    g = groups()[1]
    r = tail_review()
    r["result"]["decision"].update(changes)
    assert not apply_tail_review(g, r, 899773)
    assert g.global_end_ms == 725700
    assert g.completion_status == "unresolved"


def test_video_route_requires_complete_ordered_coverage_and_marks_transfers():
    from visioncortex.workflow_video import routed_intervals
    g = groups()[1]
    g.view_timeline = [{"start_ms": g.global_start_ms, "end_ms": 600000, "third_person_view": "tp"},
                       {"start_ms": 600000, "end_ms": 610000, "third_person_view": None},
                       {"start_ms": 610000, "end_ms": g.global_end_ms, "third_person_view": "new-tp"}]
    assert [r["third_person_view"] for r in routed_intervals(g)] == ["tp", None, "new-tp"]
    g.view_timeline[1]["start_ms"] += 100
    with pytest.raises(ValueError, match="without overlaps or gaps"):
        routed_intervals(g)


def test_group_view_timeline_uses_formal_paired_events_and_leaves_unpaired_time_unrouted():
    from visioncortex.grouping import _build_view_timeline

    group = groups()[1].model_copy(update={
        "global_start_ms": 477092.0,
        "global_end_ms": 902700.0,
        "first_person_view": "fp",
        "third_person_view": "wrong-tp",
    })
    segment = ExperimentSegment(
        segment_id="EXP-2", global_start_ms=group.global_start_ms,
        global_end_ms=group.global_end_ms, event_ids=["paired", "unpaired"],
        participating_views=["fp", "right-tp"],
    )
    candidate = ActionCandidate(
        candidate_id="C-paired", action_type=ActionType.HAND_OBJECT_CONTACT,
        view_id="fp", role=ViewRole.FIRST_PERSON, local_start_ms=838950,
        local_end_ms=841800, global_start_ms=838950, global_end_ms=841800,
        key_global_ms=840000, objects=["beaker"], confidence=.9,
        evidence=[],
    )
    paired = EvidenceEvent(
        event_id="paired", action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=838950, global_end_ms=841800, key_global_ms=840000,
        objects=["gloved_hand", "beaker"], confidence=.9, accepted=True,
        audit_reason="test", supporting_views=["fp", "right-tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[candidate],
    )
    unpaired = paired.model_copy(update={
        "event_id": "unpaired", "global_start_ms": 600000.0,
        "global_end_ms": 610000.0, "key_global_ms": 605000.0,
        "supporting_views": ["fp"], "supporting_roles": [ViewRole.FIRST_PERSON],
    })
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video="fp.mp4"),
        ViewInput(view_id="right-tp", role=ViewRole.THIRD_PERSON, video="tp.mp4"),
    ]

    timeline = _build_view_timeline(group, [segment], [paired, unpaired], views)

    assert timeline[0]["third_person_view"] is None
    assert any(row["third_person_view"] == "right-tp" for row in timeline)
    routed = next(row for row in timeline if row["start_ms"] == 838950.0)
    assert routed["end_ms"] == 841800.0
    assert routed["evidence_event_ids"] == ["paired"]


def test_future_run_reviews_cross_station_chain_and_continues_to_recording_end(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from visioncortex import boundary_review as module
    from visioncortex.archive import ArchiveLayout
    from visioncortex.schemas import AlignmentTransform

    original = groups()
    original[1].third_person_view = "pipette-camera"
    original[1].participating_views = ["fp", "pipette-camera"]
    segments = [ExperimentSegment(segment_id=f"EXP-{i}", global_start_ms=g.global_start_ms,
                                  global_end_ms=g.global_end_ms, event_ids=[],
                                  participating_views=g.participating_views)
                for i, g in enumerate(original, 1)]
    calls = []
    def fake_review(layout, analyzer, config, view_ids, views, infos, transforms, times,
                    metadata, prompt, schema, identity):
        calls.append((identity["kind"], view_ids))
        if identity["kind"] == "neighbor":
            r = review()
            r["result"]["decision"].update(relation="continuous_workflow", basis="sample_handoff",
                workstation_changed=True, handoff_frame_ids=["F002"], left_unit_name="配液", right_unit_name="移液")
        else:
            r = tail_review()
            r["frames"][0]["global_ms"] = times[0]
            r["frames"][-1]["global_ms"] = times[-1]
        return {**r, **identity, "review_id": str(len(calls))}
    monkeypatch.setattr(module, "_review", fake_review)
    monkeypatch.setattr(module, "ArkAnalyzer", lambda config: SimpleNamespace(close=lambda: None))
    layout = ArchiveLayout(tmp_path)
    layout.create()
    infos = {"fp": SimpleNamespace(duration_ms=899773)}
    transforms = {"fp": AlignmentTransform(view_id="fp", reference_view_id="fp")}
    output = module.review_experiment_boundaries(layout, original, segments, [], [], infos, transforms,
        {"mllm": {"enabled": True, "boundary_review": {"enabled": True}}})
    assert len(output) == 1
    assert output[0].global_end_ms == 899773
    assert output[0].workflow_kind == "continuous_workflow"
    assert output[0].completion_status == "ongoing_at_recording_end"
    assert calls[0] == ("neighbor", ["fp", "tp", "pipette-camera"])
    assert [c[0] for c in calls] == ["neighbor", "tail", "tail"]
    assert len(output[0].workflow_units) == 2


def test_an_already_full_length_candidate_still_requires_observed_end_review():
    from visioncortex.boundary_review import apply_tail_review
    g = groups()[1]
    g.global_end_ms = 899773
    assert apply_tail_review(g, tail_review(), 899773)
    assert g.completion_status == "ongoing_at_recording_end"


def test_semantic_units_are_separate_from_cv_fragments_and_cannot_escape_video():
    from visioncortex.boundary_review import apply_semantic_units
    g = groups()[1]
    g.workflow_kind = "continuous_workflow"
    original_ids = list(g.atomic_experiment_ids)
    model = {"status": "completed", "atomic_experiments": [
        {"name": "配液", "start_global_ms": 588200, "end_global_ms": 620000},
        {"name": "移液", "start_global_ms": 630000, "end_global_ms": 725700}]}
    assert apply_semantic_units(g, model)
    assert [u["name"] for u in g.workflow_units] == ["配液", "移液"]
    assert g.atomic_experiment_ids == original_ids
    assert not any(u["completion_observed"] for u in g.workflow_units)
    model["atomic_experiments"][-1]["end_global_ms"] = 900000
    assert not apply_semantic_units(g, model)
    assert g.workflow_units[-1]["end_ms"] == 725700


def test_multiple_semantic_units_inside_one_cv_group_can_form_a_workflow():
    from visioncortex.boundary_review import apply_semantic_units
    g = groups()[1]
    model = {"status": "completed", "continuity_type_confirmed": "continuous",
             "atomic_experiments": [
                 {"name": "配液", "start_global_ms": 588200, "end_global_ms": 620000},
                 {"name": "移液", "start_global_ms": 630000, "end_global_ms": 725700}]}
    assert apply_semantic_units(g, model)
    assert g.workflow_kind == "continuous_workflow"
    assert g.continuity_type == "continuous"
    assert g.atomic_experiment_ids == ["EXP-2"]


def test_new_continuity_review_requires_specific_object_states_and_valid_sides():
    left, right = groups()
    r = review()
    r['result']['continuity_evidence_version'] = 3
    assert not join_allowed(left, right, r['result'], r['frames'])
    link = {'object_description': '同一张称量纸', 'before_state': '双手展开',
            'after_state': '继续折叠后放入天平', 'before_frame_ids': ['F001'],
            'after_frame_ids': ['F003'], 'handoff_frame_ids': []}
    r['result']['decision']['object_links'] = [link]
    assert join_allowed(left, right, r['result'], r['frames'])
    link['after_frame_ids'] = ['F002']
    assert not join_allowed(left, right, r['result'], r['frames'])
    link['after_frame_ids'] = ['unknown']
    assert not join_allowed(left, right, r['result'], r['frames'])


def test_cross_station_object_link_must_include_the_same_handoff_evidence():
    left, right = groups()
    right.third_person_view = 'pipette-camera'
    r = review()
    r['result']['continuity_evidence_version'] = 3
    r['result']['decision'].update(relation='continuous_workflow', basis='sample_handoff',
        workstation_changed=True, handoff_frame_ids=['F002'], object_links=[{
            'object_description':'被携带的容器','before_state':'装好液体','after_state':'放在移液台',
            'before_frame_ids':['F001'],'after_frame_ids':['F003'],'handoff_frame_ids':[]}])
    assert not join_allowed(left, right, r['result'], r['frames'])
    r['result']['decision']['object_links'][0]['handoff_frame_ids'] = ['F002']
    assert join_allowed(left, right, r['result'], r['frames'])
    r['result']['decision']['same_operator'] = False
    assert not join_allowed(left, right, r['result'], r['frames'])


def test_segment_seams_follow_piecewise_recording_clocks_and_get_both_sides():
    from types import SimpleNamespace
    from visioncortex.schemas import AlignmentTransform, AlignmentSegmentTransform
    from visioncortex.boundary_review import recording_seams, seam_review_times
    infos = {'fp':SimpleNamespace(segments=[
        SimpleNamespace(virtual_start_ms=0,virtual_end_ms=900000),
        SimpleNamespace(virtual_start_ms=900000,virtual_end_ms=1800000)])}
    transforms = {'fp':AlignmentTransform(view_id='fp',reference_view_id='fp',segment_transforms=[
        AlignmentSegmentTransform(segment_index=0,local_start_ms=0,local_end_ms=900000,state='aligned'),
        AlignmentSegmentTransform(segment_index=1,local_start_ms=900000,local_end_ms=1800000,offset_ms=3000,state='aligned')])}
    seams = recording_seams(['fp'],infos,transforms,895000,907000)
    assert seams == [{'view_id':'fp','left_segment_index':0,'right_segment_index':1,
                      'before_ms':899750,'after_ms':903250}]
    times = seam_review_times(list(range(895000,907001,1000)),seams,{'proposed_end_ms':900000})
    assert {899750,903250,900000,895000,907000} <= set(times)
    assert len(times) == 13
    assert not recording_seams(['fp'],infos,transforms,0,850000)


def test_too_many_mandatory_boundaries_never_silently_drop_evidence():
    from visioncortex.boundary_review import seam_review_times
    seams = [{'before_ms':i*10+1,'after_ms':i*10+2} for i in range(17)]
    with pytest.raises(ValueError,match='budget'):
        seam_review_times(list(range(32)),seams,{})


def test_tracking_keeps_open_recording_end_and_different_operator_routes_separate():
    from visioncortex.boundary_review import workflow_tracking
    a,b = groups()
    a.completion_status = 'ongoing_at_recording_end'
    b.first_person_view = 'another-operator'
    rows = workflow_tracking([a,b],{}, {})
    assert rows[0]['awaiting_next_recording'] is True
    assert rows[1]['previous_unresolved_group_uid'] is None
    b.first_person_view = a.first_person_view
    rows = workflow_tracking([a,b],{}, {})
    assert rows[1]['previous_unresolved_group_uid'] == a.group_uid
    assert rows[1]['cross_task_stitching_applied'] is False
    assert a.global_end_ms == 577050
