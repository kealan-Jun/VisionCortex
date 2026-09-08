from copy import deepcopy

import pytest

from visioncortex.boundary_review import join_allowed, reconcile_groups, review_times
from visioncortex.mllm import ArkAnalyzer, GROUP_SYSTEM_PROMPT
from visioncortex.schemas import ExperimentGroup, ExperimentSegment


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
            "same_operator_and_workstation": True, "left_experiment_complete": False,
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
    {"left_experiment_complete": True}, {"same_operator_and_workstation": False},
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
