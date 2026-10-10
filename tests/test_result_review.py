import json

import pytest

from visioncortex.operation_review import GROUPS, EVENTS
from visioncortex.result_review import inspect, validate_observations, digest
from visioncortex.schemas import ExperimentGroup, EvidenceEvent


def dataset(root):
    event = EvidenceEvent(
        event_id="e1",
        action_type="hand_object_contact",
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["bottle"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_roles=["first_person"],
        supporting_views=["fp"],
        candidates=[],
    )
    group = ExperimentGroup(
        group_id="g1",
        global_start_ms=0,
        global_end_ms=100000,
        first_person_view="fp",
        third_person_view="tp",
        participating_views=["fp", "tp"],
        continuity_type="continuous",
        continuity_reason="test",
        atomic_experiment_ids=[],
        completion_status="ongoing_at_recording_end",
        key_event_ids=["e1"],
        model_understanding={
            "status": "completed",
            "steps": [
                {
                    "start_global_ms": 1000,
                    "end_global_ms": 2000,
                    "current_step": "手接触瓶子",
                    "supporting_event_ids": ["e1"],
                }
            ],
        },
    )
    for name, value in (
        (GROUPS, {"groups": [group.model_dump(mode="json")]}),
        (EVENTS, {"events": [event.model_dump(mode="json")]}),
        ("JSON-Config-Files/quality_acceptance.json", {"passed": False}),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return group


def test_current_check_is_version_bound_and_does_not_promote_quality(tmp_path):
    dataset(tmp_path)
    quality = (tmp_path / "JSON-Config-Files/quality_acceptance.json").read_bytes()
    before = inspect(tmp_path)
    assert before["latest_check_current"] is False
    result = inspect(tmp_path, save=True)
    assert result["step_consistency_passed"] is True
    assert result["all_operations_found"] is None
    assert result["full_quality_rechecked"] is False
    assert result["formal_archive_promotion_allowed"] is False
    assert {f["kind"] for f in result["findings"]} == {
        "unrecorded_interval",
        "unfinished_boundary",
    }
    assert all(w["end_ms"] - w["start_ms"] <= 30000 for w in result["windows"])
    assert inspect(tmp_path)["latest_check_current"] is True
    groups = json.loads((tmp_path / GROUPS).read_text())
    groups["groups"][0]["model_understanding"]["steps"][0]["current_step"] = (
        "手按下按钮并确认读数"
    )
    (tmp_path / GROUPS).write_text(json.dumps(groups))
    revised = inspect(tmp_path)
    assert revised["revision"] != result["revision"]
    assert revised["latest_check_current"] is False
    assert revised["step_consistency_passed"] is False
    assert (
        tmp_path / "JSON-Config-Files/quality_acceptance.json"
    ).read_bytes() == quality


def test_no_cross_operator_continuity_or_silent_missing_reference(tmp_path):
    group = dataset(tmp_path)
    another = group.model_copy(deep=True)
    another.group_id = "g2"
    another.first_person_view = "other"
    another.global_start_ms = 110000
    another.global_end_ms = 120000
    group.model_understanding["steps"][0]["supporting_event_ids"] = ["unknown"]
    (tmp_path / GROUPS).write_text(
        json.dumps({"groups": [g.model_dump(mode="json") for g in [group, another]]})
    )
    result = inspect(tmp_path)
    assert any(f["kind"] == "references" for f in result["findings"])
    assert not any(f["kind"] == "continuity_unverified" for f in result["findings"])


@pytest.mark.parametrize("ids", [["F001", "unknown"], ["F001", "F001"]])
def test_gap_observations_require_real_distinct_frame_references(ids):
    with pytest.raises(ValueError):
        validate_observations(
            {
                "observations": [
                    {"title": "扶瓶", "description": "扶住瓶子", "frame_ids": ids}
                ],
                "limitation": "稀疏采样",
            },
            [{"frame_id": "F001", "global_ms": 1000}],
        )


def test_selected_gap_rejects_stale_version_before_media_or_model(
    tmp_path, default_config
):
    from visioncortex.result_review import investigate

    dataset(tmp_path)
    with pytest.raises(ValueError, match="版本"):
        investigate(tmp_path, default_config, "x", digest({}))


def test_stage_check_has_no_model_invocations_and_keeps_quality(
    tmp_path, default_config
):
    from visioncortex.stage_refresh import refresh

    dataset(tmp_path)
    revision = inspect(tmp_path)["revision"]
    result = refresh(tmp_path, "result_check", default_config, revision=revision)
    assert result["model_invocations"] == result["source_media_decodes"] == 0
    assert result["quality_gate_unchanged"] is True
    assert result["dependencies"]["step_consistency_passed"] is True


def test_duplicate_evidence_is_not_silently_counted_as_another_step(tmp_path):
    group = dataset(tmp_path)
    group.model_understanding["steps"].append(
        dict(group.model_understanding["steps"][0])
    )
    (tmp_path / GROUPS).write_text(
        json.dumps({"groups": [group.model_dump(mode="json")]})
    )
    result = inspect(tmp_path)
    assert result["step_consistency_passed"] is False
    assert next(f for f in result["findings"] if f["kind"] == "references")[
        "duplicate"
    ] == ["e1"]


@pytest.mark.parametrize("change_image", [False, True])
def test_gap_review_retains_receipts_without_admitting_actions(
    tmp_path, default_config, monkeypatch, change_image
):
    import numpy as np
    from visioncortex.result_review import investigate
    from visioncortex.mllm import ArkAnalyzer

    group = dataset(tmp_path)
    group.videos = {"aligned_first_third": "clip.mp4"}
    (tmp_path / "clip.mp4").write_bytes(b"fake file used only with mock decoder")
    (tmp_path / GROUPS).write_text(
        json.dumps({"groups": [group.model_dump(mode="json")]})
    )
    original = (tmp_path / GROUPS).read_bytes()

    from visioncortex.source_frames import SampledFrame
    from types import SimpleNamespace
    from visioncortex.schemas import SourceFrameIdentity

    def samples(path, info, start, end, **kwargs):
        for index in range(8):
            ms = start + (end - start) * index / 8
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            identity = SourceFrameIdentity(
                source_path=path,
                source_size_bytes=path.stat().st_size,
                source_mtime_ns=path.stat().st_mtime_ns,
                status="resolved",
                packet_position=index,
                source_pts=int(ms),
                time_base="1/1000",
                decoded_pixels_sha256="0" * 64,
            )
            yield SampledFrame(index, ms, frame, identity)

    monkeypatch.setattr(
        "visioncortex.video_io.probe_video", lambda path: SimpleNamespace(width=8)
    )
    monkeypatch.setattr("visioncortex.video_io.iter_sampled_frames", samples)
    monkeypatch.setattr(
        "visioncortex.archive._semantic_cache_path",
        lambda *args: tmp_path / "cache.json",
    )
    monkeypatch.setattr(
        "visioncortex.archive._semantic_cache_reads_enabled", lambda *args: False
    )
    monkeypatch.setattr(ArkAnalyzer, "__init__", lambda *args: None)
    monkeypatch.setattr(ArkAnalyzer, "close", lambda *args: None)

    def call(self, prompt, metadata, images, **kwargs):
        assert len(images) == 8 and kwargs["max_images"] == 8
        if change_image:
            images[0][1].write_bytes(b"changed during request")
        return {
            "status": "completed",
            "observations": [
                {
                    "title": "扶住瓶子",
                    "description": "手扶瓶子并保持位置",
                    "frame_ids": ["F001", "F002"],
                }
            ],
            "limitation": "稀疏画面",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(ArkAnalyzer, "_call", call)
    plan = inspect(tmp_path)
    result = investigate(
        tmp_path, default_config, plan["windows"][0]["window_id"], plan["revision"]
    )
    assert result["status"] == (
        "unverified" if change_image else "supplementary_observations"
    )
    assert result["formal_action_admission"] is False
    assert result["source_video_decodes"] == 0 and result["derived_frame_reads"] == 8
    assert result["result"]["usage"]["total_tokens"] == 10
    assert (tmp_path / GROUPS).read_bytes() == original
    assert inspect(tmp_path)["revision"] == plan["revision"]
