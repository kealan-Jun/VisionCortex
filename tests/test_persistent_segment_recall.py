from pathlib import Path

import pytest

from labvision_evidence.config import load_config
from labvision_evidence.pipeline import (
    EvidencePipeline,
    _merge_frame_evidence_ledgers,
)
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    BoxEvidence,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    FrameEvidence,
    VideoInfo,
    VideoSegmentInfo,
    ViewInput,
    ViewRole,
)
from labvision_evidence.video_io import (
    _aligned_grid_start_ms,
    _selected_session_timestamps,
    plan_physical_segment_decode_sessions,
)


def test_physical_segment_sessions_collapse_repeated_references_without_widening():
    segments = [
        VideoSegmentInfo(
            path=Path("segment-00.mp4"),
            virtual_start_ms=0.0,
            virtual_end_ms=900_000.0,
            frame_start_index=0,
            duration_ms=900_000.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=27_000,
            size_bytes=1,
        ),
        VideoSegmentInfo(
            path=Path("segment-01.mp4"),
            virtual_start_ms=900_000.0,
            virtual_end_ms=1_800_000.0,
            frame_start_index=27_000,
            duration_ms=900_000.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=27_000,
            size_bytes=1,
        ),
    ]
    info = VideoInfo(
        path=segments[0].path,
        duration_ms=1_800_000.0,
        fps=30.0,
        width=1920,
        height=1080,
        frame_count=54_000,
        size_bytes=2,
        segments=segments,
    )

    sessions = plan_physical_segment_decode_sessions(
        info,
        [
            (100_000.0, 120_000.0),
            (110_000.0, 130_000.0),
            (300_000.0, 310_000.0),
            (895_000.0, 905_000.0),
        ],
    )

    assert len(sessions) == 2
    assert sessions[0].path == Path("segment-00.mp4")
    assert sessions[0].target_virtual_windows == (
        (100_000.0, 130_000.0),
        (300_000.0, 310_000.0),
        (895_000.0, 900_000.0),
    )
    assert sessions[0].virtual_start_ms == 100_000.0
    assert sessions[0].virtual_end_ms == 900_000.0
    assert sessions[0].selected_duration_ms == 45_000.0
    assert sessions[1].target_virtual_windows == ((900_000.0, 905_000.0),)


def test_persistent_session_timestamp_count_matches_ffmpeg_round_near_endpoint():
    """DEV-027: 205.633333 seconds at 10 FPS must be 2056, not ceil=2057."""

    start_ms = 6_996_209.421143
    end_ms = 7_201_842.754476
    timestamps = _selected_session_timestamps(
        start_ms,
        end_ms,
        [(start_ms, end_ms)],
        sample_fps=10.0,
    )

    assert len(timestamps) == 2056
    assert timestamps[0] == start_ms
    assert timestamps[-1] == start_ms + 205_500.0


def test_persistent_sessions_share_one_alignment_anchored_global_sampling_grid():
    transform = AlignmentTransform(
        view_id="tp",
        reference_view_id="fp",
        scale=1.0,
        offset_ms=34.159,
        state="aligned",
    )
    local_origin_ms = transform.to_local(0.0)

    first = _aligned_grid_start_ms(1_671_773.0, local_origin_ms, 100.0)
    second = _aligned_grid_start_ms(1_800_642.0, local_origin_ms, 100.0)

    assert transform.to_global(first) % 100.0 == pytest.approx(0.0, abs=1e-6)
    assert transform.to_global(second) % 100.0 == pytest.approx(0.0, abs=1e-6)


def _candidate(
    candidate_id: str,
    start_ms: float,
    end_ms: float,
    action_type: ActionType,
    objects: list[str],
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=objects,
        confidence=0.9,
    )


def test_group_local_recall_selects_unscanned_evidence_producing_view(
    default_config,
):
    default_config["performance"].update(
        {
            "fine_group_recall_min_unresolved_anchors": 3,
            "fine_group_recall_padding_seconds": 0.0,
        }
    )
    pipeline = EvidencePipeline(default_config)
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(
            view_id="b439", role=ViewRole.THIRD_PERSON, video=Path("b439.mp4")
        ),
        ViewInput(
            view_id="d12", role=ViewRole.THIRD_PERSON, video=Path("d12.mp4")
        ),
        ViewInput(view_id="rk", role=ViewRole.THIRD_PERSON, video=Path("rk.mp4")),
    ]
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=500_000.0,
            fps=30.0,
            width=16,
            height=16,
            frame_count=15_000,
        )
        for view in views
    }
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            state="aligned",
            confidence=1.0,
        )
        for view in views
    }
    cross = EvidenceEvent(
        event_id="E-CROSS",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=305_000.0,
        global_end_ms=310_000.0,
        key_global_ms=307_500.0,
        objects=["gloved_hand", "balance"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "rk"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    segment = ExperimentSegment(
        segment_id="EXP-1",
        global_start_ms=300_000.0,
        global_end_ms=430_000.0,
        event_ids=[cross.event_id],
        participating_views=["fp", "b439", "rk"],
    )
    group = ExperimentGroup(
        group_id="G1",
        continuity_type="independent",
        atomic_experiment_ids=[segment.segment_id],
        global_start_ms=300_000.0,
        global_end_ms=430_000.0,
        participating_views=["fp", "b439"],
        first_person_view="fp",
        third_person_view="b439",
        continuity_reason="test",
    )
    candidates = [
        _candidate(
            "C-COVERED",
            305_000.0,
            310_000.0,
            ActionType.HAND_OBJECT_CONTACT,
            ["gloved_hand", "balance"],
        ),
        _candidate(
            "C-1",
            333_100.0,
            346_100.0,
            ActionType.OBJECT_MOVEMENT,
            ["paper"],
        ),
        _candidate(
            "C-2",
            356_600.0,
            381_500.0,
            ActionType.CONTAINER_STATE_CHANGE,
            ["bottle_cap"],
        ),
        _candidate(
            "C-3",
            393_600.0,
            396_100.0,
            ActionType.DEVICE_PANEL_OPERATION,
            ["balance"],
        ),
    ]
    actual_windows = {
        "fp": [(215_000.0, 455_000.0)],
        "b439": [(283_100.0, 445_600.0)],
        "d12": [(283_100.0, 329_500.0)],
        "rk": [(283_100.0, 329_500.0)],
    }

    plan = pipeline._group_local_recall_plan(
        [group],
        [segment],
        [cross],
        candidates,
        views,
        {view.view_id: {} for view in views},
        actual_windows,
        infos,
        transforms,
    )

    assert plan["complete"] is False
    assert plan["selected_plans"][0]["view_id"] == "rk"
    assert plan["selected_plans"][0]["windows"] == [
        (333_100.0, 346_100.0),
        (356_600.0, 381_500.0),
        (393_600.0, 396_100.0),
    ]
    choices = plan["groups"][0]["ranked_view_choices"]
    assert [item["view_id"] for item in choices] == ["rk", "d12"]
    assert all(item["view_id"] != "b439" for item in choices)


def test_group_local_recall_skips_zero_prior_views_for_complete_dual_view_group(
    default_config,
):
    pipeline = EvidencePipeline(default_config)
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(
            view_id="b439", role=ViewRole.THIRD_PERSON, video=Path("b439.mp4")
        ),
        ViewInput(view_id="d12", role=ViewRole.THIRD_PERSON, video=Path("d12.mp4")),
        ViewInput(view_id="rk", role=ViewRole.THIRD_PERSON, video=Path("rk.mp4")),
    ]
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=500_000.0,
            fps=30.0,
            width=16,
            height=16,
            frame_count=15_000,
        )
        for view in views
    }
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            state="aligned",
            confidence=1.0,
        )
        for view in views
    }
    event = EvidenceEvent(
        event_id="E-CROSS",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=300_000.0,
        global_end_ms=310_000.0,
        key_global_ms=305_000.0,
        objects=["gloved_hand", "tube"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "b439"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    segment = ExperimentSegment(
        segment_id="EXP-1",
        global_start_ms=300_000.0,
        global_end_ms=430_000.0,
        event_ids=[event.event_id],
        participating_views=["fp", "b439"],
    )
    group = ExperimentGroup(
        group_id="G1",
        continuity_type="independent",
        atomic_experiment_ids=[segment.segment_id],
        global_start_ms=300_000.0,
        global_end_ms=430_000.0,
        participating_views=["fp", "b439"],
        first_person_view="fp",
        third_person_view="b439",
        continuity_reason="test",
    )
    unresolved = [
        _candidate(
            f"C-{index}",
            330_000.0 + index * 10_000.0,
            335_000.0 + index * 10_000.0,
            ActionType.OBJECT_MOVEMENT,
            ["tube"],
        )
        for index in range(3)
    ]

    plan = pipeline._group_local_recall_plan(
        [group],
        [segment],
        [event],
        unresolved,
        views,
        {view.view_id: {} for view in views},
        {"fp": [(300_000.0, 430_000.0)], "b439": [(300_000.0, 430_000.0)]},
        infos,
        transforms,
    )

    assert plan["complete"] is True
    assert plan["selected_plans"] == []
    assert plan["groups"][0]["status"] == "complete_no_positive_recall_prior"
    assert all(
        item["positive_recall_prior"] is False
        for item in plan["groups"][0]["ranked_view_choices"]
    )


def test_detection_ledger_merge_deduplicates_frames_and_namespaces_tracks(
    tmp_path,
):
    existing = tmp_path / "existing.jsonl"
    supplement = tmp_path / "supplement.jsonl"
    merged = tmp_path / "merged.jsonl"

    def frame(local_ms: float, track_id: int) -> FrameEvidence:
        return FrameEvidence(
            view_id="rk",
            role=ViewRole.THIRD_PERSON,
            frame_index=int(local_ms),
            local_ms=local_ms,
            global_ms=local_ms,
            width=16,
            height=16,
            detections=[
                BoxEvidence(
                    class_id=1,
                    class_name="balance",
                    confidence=0.9,
                    xyxy_norm=(0.1, 0.1, 0.2, 0.2),
                    track_id=track_id,
                )
            ],
        )

    existing.write_text(
        "\n".join(
            [frame(100.0, 1).model_dump_json(), frame(200.0, 2).model_dump_json()]
        )
        + "\n",
        encoding="utf-8",
    )
    supplement.write_text(
        "\n".join(
            [frame(200.0, 3).model_dump_json(), frame(300.0, 4).model_dump_json()]
        )
        + "\n",
        encoding="utf-8",
    )

    report = _merge_frame_evidence_ledgers(
        existing,
        supplement,
        merged,
        track_id_namespace=4,
    )
    frames = [
        FrameEvidence.model_validate_json(line)
        for line in merged.read_text(encoding="utf-8").splitlines()
    ]

    assert report["existing_frames"] == 2
    assert report["supplement_frames"] == 2
    assert report["merged_frames"] == 3
    assert report["deduplicated_frames"] == 1
    assert [item.local_ms for item in frames] == [100.0, 200.0, 300.0]
    assert frames[1].detections[0].track_id == 4_000_003
    assert frames[2].detections[0].track_id == 4_000_004


def test_rtx_profile_enables_persistent_decode_and_local_recall():
    config = load_config(Path("configs/rtx4060-laptop-production.yaml"))

    assert config["performance"]["fine_persistent_segment_decode"] is True
    assert config["performance"]["fine_group_local_recall_enabled"] is True
    assert config["performance"]["fine_group_recall_min_unresolved_anchors"] == 3
    assert config["performance"]["fine_group_recall_padding_seconds"] == 0.0
