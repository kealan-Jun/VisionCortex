from copy import deepcopy
from types import SimpleNamespace

import pytest

from visioncortex import boundary_review as boundary
from visioncortex.archive import ArchiveLayout
from visioncortex.schemas import AlignmentTransform, ExperimentGroup, ExperimentSegment


def group():
    return ExperimentGroup(
        group_id="GROUP-1", group_uid="original-group", continuity_type="independent",
        atomic_experiment_ids=["EXP-1"], global_start_ms=0, global_end_ms=20000,
        participating_views=["fp", "tp"], first_person_view="fp", third_person_view="tp",
        continuity_reason="candidate cut, not a completed experiment",
        view_timeline=[{"start_ms": 0, "end_ms": 20000, "third_person_view": "tp"}],
    )


def tail_record(times=(10000, 20000, 30000, 35000, 45000)):
    frames = [{"frame_id": f"F{i}", "global_ms": t,
               "sources": [{"view_id": "fp"}, {"view_id": "tp"}]}
              for i, t in enumerate(times)]
    return {"left_end_ms": 20000, "kind": "tail", "frames": frames,
            "result": {"status": "completed", "decision": {
                "state": "completed", "continuation_observed": True,
                "frame_ids": [f["frame_id"] for f in frames],
                "last_continuation_frame_id": None, "completion_frame_id": "F2",
                "observation": "Observable workflow closure, followed by unrelated activity",
                "confidence": .96, "uncertainties": []}}}


def reviewed_tail(times=(10000, 20000, 30000, 35000, 45000)):
    record = tail_record(times)
    record["request_metadata"] = {"continuity_anchor_ms": 20000,
                                  "completion_followup_ms": 12000}
    record["result"]["continuity_evidence_version"] = 4
    record["result"]["decision"].update(
        same_operator=True, same_workstation=True,
        completion_scope="workflow", post_completion_state="no_related_continuation",
        post_completion_frame_ids=["F3", "F4"],
        object_links=[{"object_description": "the handled sample vessel",
                       "before_state": "still being used", "after_state": "cleared and stored",
                       "before_frame_ids": ["F1"], "after_frame_ids": ["F2"],
                       "handoff_frame_ids": []}],
    )
    return record


def test_legacy_last_frame_completion_cannot_close_a_workflow():
    record = tail_record()
    record["result"]["decision"]["completion_frame_id"] = "F4"
    value = group()
    assert not boundary.apply_tail_review(value, record, 90000)
    assert value.global_end_ms == 20000
    assert value.completion_status == "unresolved"


def test_closure_requires_followup_but_does_not_include_unrelated_followup_in_clip():
    value = group()
    assert boundary.apply_tail_review(value, reviewed_tail(), 90000)
    assert value.completion_status == "observed_complete"
    assert value.global_end_ms == 31000
    assert value.boundary_extension_requires_step_review is True


@pytest.mark.parametrize("changes", [
    {"completion_scope": "operation"}, {"completion_scope": "experiment_unit"},
    {"same_operator": False}, {"object_links": []},
    {"post_completion_state": "related_continuation"},
    {"post_completion_state": "uncertain"},
    {"post_completion_frame_ids": ["F3"]},
    {"post_completion_frame_ids": ["F2", "F4"]},
    {"post_completion_frame_ids": ["F3", "invented"]},
    {"post_completion_frame_ids": ["F4", "F4"]},
    {"completion_frame_id": "F4"},
])
def test_one_step_or_unit_completion_cannot_close_the_continuous_experiment(changes):
    record = reviewed_tail()
    record["result"]["decision"].update(changes)
    value = group()
    assert not boundary.apply_tail_review(value, record, 90000)
    assert value.global_end_ms == 20000


def test_completion_followup_must_cover_the_required_time_and_latest_sample():
    record = reviewed_tail((10000, 20000, 30000, 33000, 35000))
    assert not boundary.apply_tail_review(group(), record, 90000)
    record = reviewed_tail()
    record["frames"].append({"frame_id": "F5", "global_ms": 55000,
                             "sources": [{"view_id": "fp"}]})
    assert not boundary.apply_tail_review(group(), record, 90000)


def test_closure_padding_never_overlaps_the_next_independent_experiment():
    value = group()
    record = reviewed_tail()
    record["review_limit_ms"] = 30500
    assert boundary.apply_tail_review(value, record, 90000)
    assert value.global_end_ms == 30500


def run_tail(tmp_path, monkeypatch, decisions, *, budget=4, uncertain_budget=2,
             result_status="completed", source_end=100000):
    value = group()
    layout = ArchiveLayout(tmp_path)
    layout.create()
    calls = []

    def review(layout, analyzer, config, view_ids, views, infos, transforms, times,
               metadata, prompt, schema, identity):
        calls.append({"times": times, "metadata": metadata})
        state = decisions[min(len(calls) - 1, len(decisions) - 1)]
        record = reviewed_tail((times[0], metadata["continuity_anchor_ms"],
                                times[-1] - 2000, times[-1] - 1000, times[-1]))
        record.update(identity, review_id=str(len(calls)), request_metadata=metadata)
        record["result"]["status"] = result_status
        decision = record["result"]["decision"]
        decision.update(state=state, completion_scope="none", post_completion_state="not_observed",
                        post_completion_frame_ids=[], completion_frame_id=None,
                        last_continuation_frame_id="F4", continuation_observed=state == "ongoing")
        decision["object_links"][0]["after_frame_ids"] = ["F4"]
        decision["object_links"][0]["after_state"] = "the same vessel is still in use"
        return record

    monkeypatch.setattr(boundary, "_review", review)
    monkeypatch.setattr(boundary, "ArkAnalyzer", lambda config: SimpleNamespace(close=lambda: None))
    result = boundary.review_experiment_boundaries(
        layout, [value], [ExperimentSegment(segment_id="EXP-1", global_start_ms=0,
            global_end_ms=20000, event_ids=[], participating_views=["fp", "tp"])], [], [],
        {"fp": SimpleNamespace(duration_ms=source_end)},
        {"fp": AlignmentTransform(view_id="fp", reference_view_id="fp")},
        {"mllm": {"enabled": True, "boundary_review": {"enabled": True,
            "tail_window_seconds": 40, "maximum_tail_windows": budget,
            "maximum_uncertain_tail_windows": uncertain_budget}}},
    )
    return result[0], calls


def test_short_pause_does_not_prevent_later_evidenced_continuation(tmp_path, monkeypatch):
    value, calls = run_tail(tmp_path, monkeypatch, ["uncertain", "ongoing"])
    assert len(calls) == 2
    assert calls[1]["times"][-1] > calls[0]["times"][-1]
    # The review must bridge the original cut and the uncertain interval.
    assert calls[1]["metadata"]["continuity_anchor_ms"] == 20000
    assert calls[1]["times"][0] == calls[0]["times"][0]
    assert calls[0]["times"][-1] in calls[1]["times"]
    assert value.global_end_ms == 100000
    assert value.completion_status == "ongoing_at_recording_end"


def test_uncertainty_budget_preserves_candidate_end_and_records_stop_reason(tmp_path, monkeypatch):
    value, calls = run_tail(tmp_path, monkeypatch, ["uncertain"], uncertain_budget=1)
    assert len(calls) == 2
    assert value.global_end_ms == 20000
    assert value.completion_status == "unresolved"
    assert value.boundary_reviews[-1]["stop_reason"] == "uncertainty_budget_exhausted"


def test_an_independent_new_workflow_stops_extension(tmp_path, monkeypatch):
    value, calls = run_tail(tmp_path, monkeypatch, ["different_workflow", "ongoing"])
    assert len(calls) == 1
    assert value.global_end_ms == 20000
    assert value.completion_status == "unresolved"


def test_provider_failure_does_not_trigger_paid_lookahead(tmp_path, monkeypatch):
    value, calls = run_tail(tmp_path, monkeypatch, ["uncertain"], result_status="failed")
    assert len(calls) == 1
    assert value.global_end_ms == 20000
    assert value.boundary_reviews[-1]["stop_reason"] == "review_failed"


def test_an_already_full_length_candidate_gets_a_lookback_anchor(tmp_path, monkeypatch):
    value, calls = run_tail(tmp_path, monkeypatch, ["ongoing"], source_end=20000)
    assert len(calls) == 1
    assert calls[0]["metadata"]["continuity_anchor_ms"] < 20000
    assert value.global_end_ms == 20000
    assert value.completion_status == "ongoing_at_recording_end"


def test_disabling_uncertain_lookahead_keeps_the_existing_budget_control(tmp_path, monkeypatch):
    value, calls = run_tail(tmp_path, monkeypatch, ["uncertain"], uncertain_budget=0)
    assert len(calls) == 1
    assert value.global_end_ms == 20000


@pytest.mark.parametrize("field,value", [("global_ms", float("nan")), ("frame_id", "F1")])
def test_invalid_frame_times_or_duplicate_ids_cannot_close_a_workflow(field, value):
    record = reviewed_tail()
    record["frames"][-1][field] = value
    assert not boundary.apply_tail_review(group(), record, 90000)


@pytest.mark.parametrize("missing_field", [None, "completion_scope"])
def test_real_request_adapter_uses_new_cache_version_and_requires_new_response_fields(
    tmp_path, monkeypatch, missing_field,
):
    import numpy as np
    from visioncortex import archive

    class Reader:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            return np.zeros((20, 30, 3), dtype=np.uint8)

    kinds = []
    monkeypatch.setattr(boundary, "ViewFrameReader", Reader)
    monkeypatch.setattr(archive, "_semantic_fingerprint",
                        lambda kind, *args: kinds.append(kind) or "new-tail-fingerprint")
    monkeypatch.setattr(archive, "_semantic_cache_path", lambda *args: tmp_path / "cache.json")
    monkeypatch.setattr(archive, "_semantic_cache_reads_enabled", lambda config: False)
    monkeypatch.setattr(archive, "_write_semantic_cache", lambda path, fingerprint, result: result)
    response = boundary.TailDecision.model_validate(reviewed_tail()["result"]["decision"]).model_dump()
    if missing_field:
        del response[missing_field]
    layout = ArchiveLayout(tmp_path)
    layout.create()
    record = boundary._review(
        layout, SimpleNamespace(_call=lambda *args, **kwargs: {"status": "completed", **response}),
        {}, ["fp"], [SimpleNamespace(view_id="fp")],
        {"fp": SimpleNamespace(duration_ms=90000, segments=[])},
        {"fp": AlignmentTransform(view_id="fp", reference_view_id="fp")},
        [10000, 20000, 30000, 35000, 45000],
        {"proposed_end_ms": 20000, "continuity_anchor_ms": 20000},
        boundary.TAIL_PROMPT, boundary.TailDecision,
        {"kind": "tail", "left_group_uid": "a", "right_group_uid": ""},
    )
    assert kinds == ["experiment-boundary-v4"]
    assert record["tail_evidence_version"] == 4
    assert record["result"]["continuity_evidence_version"] == 4
    assert record["result"]["status"] == ("invalid_boundary_response" if missing_field else "completed")


def test_cross_station_tail_requires_object_handoff_and_actor_continuity():
    record = reviewed_tail()
    d = record["result"]["decision"]
    d.update(state="ongoing", same_workstation=False, completion_scope="none",
             post_completion_state="not_observed", post_completion_frame_ids=[],
             last_continuation_frame_id="F4", completion_frame_id=None)
    d["object_links"][0]["after_frame_ids"] = ["F4"]
    assert not boundary.apply_tail_review(group(), record, 45000)
    d["object_links"][0]["handoff_frame_ids"] = ["F2"]
    assert boundary.apply_tail_review(group(), record, 45000)
    invalid = deepcopy(record)
    invalid["result"]["decision"]["object_links"][0]["handoff_frame_ids"] = ["wrong"]
    assert not boundary.apply_tail_review(group(), invalid, 45000)
