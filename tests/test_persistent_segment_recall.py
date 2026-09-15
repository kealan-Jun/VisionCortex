import io
import shutil
import subprocess
from pathlib import Path

import pytest

from visioncortex.config import load_config
from visioncortex.pipeline import (
    EvidencePipeline,
    _merge_frame_evidence_ledgers,
)
from visioncortex.detection import _accept_unique_frame_timestamp
from visioncortex.schemas import (
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
from visioncortex.video_io import (
    _aligned_grid_start_ms,
    _ffmpeg_multi_window_iterator,
    _is_reconcilable_terminal_eof_shortfall,
    _selected_session_frame_indices,
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


def test_distant_windows_split_without_changing_selected_timestamps():
    segment = VideoSegmentInfo(
        path=Path("segment.mp4"),
        virtual_start_ms=0.0,
        virtual_end_ms=900_000.0,
        frame_start_index=0,
        duration_ms=900_000.0,
        fps=30.0,
        width=1920,
        height=1080,
        frame_count=27_000,
        size_bytes=1,
    )
    info = VideoInfo(
        path=segment.path,
        duration_ms=900_000.0,
        fps=30.0,
        width=1920,
        height=1080,
        frame_count=27_000,
        size_bytes=1,
        segments=[segment],
    )
    windows = [(100_000.0, 110_000.0), (125_000.0, 130_000.0), (800_000.0, 805_000.0)]

    sessions = plan_physical_segment_decode_sessions(
        info, windows, max_gap_ms=30_000.0
    )

    assert len(sessions) == 2
    assert sessions[0].target_virtual_windows == (
        (100_000.0, 110_000.0),
        (125_000.0, 130_000.0),
    )
    assert sessions[1].target_virtual_windows == ((800_000.0, 805_000.0),)
    assert sum(session.selected_duration_ms for session in sessions) == 20_000.0
    selected_after_split = [
        timestamp
        for session in sessions
        for timestamp in _selected_session_timestamps(
            session.source_start_ms,
            session.source_end_ms,
            session.target_source_windows,
            sample_fps=10.0,
        )
    ]
    selected_without_split = _selected_session_timestamps(
        100_000.0,
        805_000.0,
        windows,
        sample_fps=10.0,
    )
    assert selected_after_split == selected_without_split


def test_physical_session_excludes_wall_clock_gap_after_media_eof():
    segment = VideoSegmentInfo(
        path=Path("paused-recorder.mp4"),
        timestamps_csv=Path("paused-recorder-frames.csv"),
        virtual_start_ms=0.0,
        virtual_end_ms=332_800.0,
        frame_start_index=0,
        duration_ms=332_800.0,
        fps=30.0,
        width=1280,
        height=800,
        frame_count=9_984,
        size_bytes=1,
        source_clock_duration_ms=410_402.117,
    )
    info = VideoInfo(
        path=segment.path,
        duration_ms=410_402.117,
        fps=30.0,
        width=1280,
        height=800,
        frame_count=9_984,
        size_bytes=1,
        segments=[segment],
    )

    sessions = plan_physical_segment_decode_sessions(info, [(0.0, 405_000.0)])

    assert len(sessions) == 1
    assert sessions[0].source_end_ms == 332_800.0
    assert sessions[0].target_source_windows == ((0.0, 332_800.0),)
    assert len(
        _selected_session_timestamps(
            sessions[0].source_start_ms,
            sessions[0].source_end_ms,
            sessions[0].target_source_windows,
            sample_fps=10.0,
        )
    ) == 3_328


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


def test_only_single_terminal_eof_shortfall_is_reconcilable():
    assert _is_reconcilable_terminal_eof_shortfall([0, 1, 2, 3], 3, 4) is True
    assert _is_reconcilable_terminal_eof_shortfall([0, 1, 2, 3], 2, 4) is False
    assert _is_reconcilable_terminal_eof_shortfall([0, 1, 2], 2, 4) is False
    assert _is_reconcilable_terminal_eof_shortfall([1], 0, 2) is False


def test_bounded_multi_frame_terminal_eof_shortfall_requires_full_session():
    assert (
        _is_reconcilable_terminal_eof_shortfall(
            list(range(10)), 6, 10, maximum_shortfall_frames=4
        )
        is True
    )
    assert (
        _is_reconcilable_terminal_eof_shortfall(
            list(range(10)), 6, 10, maximum_shortfall_frames=3
        )
        is False
    )
    assert (
        _is_reconcilable_terminal_eof_shortfall(
            [0, 1, 8, 9], 2, 10, maximum_shortfall_frames=2
        )
        is False
    )


def test_persistent_ffmpeg_fills_and_records_one_terminal_eof_shortfall(
    monkeypatch,
):
    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(bytes(2 * 2 * 3 * 3))
            self.stderr = io.BytesIO()
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "visioncortex.video_io.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    info = VideoInfo(
        path=Path("terminal-eof.mp4"),
        duration_ms=400.0,
        fps=30.0,
        width=2,
        height=2,
        frame_count=12,
    )
    receipt = {}

    frames = list(
        _ffmpeg_multi_window_iterator(
            info.path,
            info,
            0.0,
            400.0,
            [(0.0, 400.0)],
            10.0,
            2,
            None,
            1,
            False,
            receipt,
        )
    )

    assert len(frames) == 4
    assert receipt["expected_frame_count"] == 4
    assert receipt["decoded_frame_count"] == 3
    assert receipt["actual_frame_count"] == 4
    assert receipt["pre_fill_frame_accounting_mismatch"] == -1
    assert receipt["frame_accounting_mismatch"] == 0
    assert receipt["frame_accounting_reconciled"] is True
    assert receipt["terminal_eof_shortfall_frames"] == 1
    assert receipt["terminal_eof_fill_frames"] == 1
    assert receipt["terminal_eof_fill_duration_ms"] == 100.0
    assert frames[-1][1] == 300.0
    assert (frames[-1][2] == frames[-2][2]).all()


def test_persistent_ffmpeg_fills_seven_frame_terminal_tail_within_one_second(
    monkeypatch,
):
    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(bytes(2 * 2 * 3 * 3))
            self.stderr = io.BytesIO()
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "visioncortex.video_io.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    info = VideoInfo(
        path=Path("terminal-eof-seven.mp4"),
        duration_ms=1_000.0,
        fps=30.0,
        width=2,
        height=2,
        frame_count=30,
    )
    receipt = {}

    frames = list(
        _ffmpeg_multi_window_iterator(
            info.path,
            info,
            0.0,
            1_000.0,
            [(0.0, 1_000.0)],
            10.0,
            2,
            None,
            1,
            False,
            receipt,
        )
    )

    assert len(frames) == 10
    assert receipt["decoded_frame_count"] == 3
    assert receipt["terminal_eof_fill_frames"] == 7
    assert receipt["terminal_eof_fill_duration_ms"] == 700.0
    assert receipt["frame_accounting_mismatch"] == 0


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


def test_discontinuous_windows_compile_to_exact_integer_fps_indices():
    indices = _selected_session_frame_indices(
        0.0,
        2_000.0,
        [(0.0, 450.0), (1_000.0002, 1_150.0)],
        sample_fps=9.99999993,
    )

    assert indices == [0, 1, 2, 3, 4, 11]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is optional")
def test_non_identity_discontinuous_ffmpeg_windows_do_not_emit_extra_frame(
    tmp_path,
):
    """DEV-030: rounded ``t`` bounds emitted 7 frames for a 6-frame ledger."""

    path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=16x16:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    info = VideoInfo(
        path=path,
        duration_ms=2_000.0,
        fps=30.0,
        width=16,
        height=16,
        frame_count=60,
        size_bytes=path.stat().st_size,
    )
    transform = AlignmentTransform(
        view_id="tp",
        reference_view_id="fp",
        scale=0.999999993,
        offset_ms=75.491930160,
        state="aligned",
    )
    effective_local_fps = 10.0 * transform.scale
    receipt = {}

    frames = list(
        _ffmpeg_multi_window_iterator(
            path,
            info,
            0.0,
            2_000.0,
            [(0.0, 450.0), (1_000.0002, 1_150.0)],
            effective_local_fps,
            16,
            None,
            1,
            False,
            receipt,
        )
    )

    assert len(frames) == 6
    assert receipt["expected_frame_count"] == 6
    assert receipt["actual_frame_count"] == 6
    assert receipt["frame_accounting_mismatch"] == 0
    assert receipt["frame_selection_policy"] == "post_fps_integer_indices_half_open"


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


def test_group_local_recall_uses_zero_prior_quality_fallback_for_unresolved_cluster(
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

    assert plan["complete"] is False
    assert plan["selected_plans"][0]["view_id"] == "d12"
    assert plan["groups"][0]["status"] == "needs_group_local_recall"
    assert plan["groups"][0]["selection_mode"] == "zero_prior_quality_fallback"
    assert plan["groups"][0]["unresolved_temporal_cluster_count"] == 1
    assert all(
        item["positive_recall_prior"] is False
        for item in plan["groups"][0]["ranked_view_choices"]
    )
    assert plan["decision_receipts"][0]["rule_id"] == (
        "QF3-TEMPORAL-CLUSTER-COMPLETENESS"
    )


def test_group_local_recall_never_rescans_clipped_media_tail(default_config):
    default_config["performance"]["fine_group_recall_min_unresolved_anchors"] = 3
    pipeline = EvidencePipeline(default_config)
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp-covered", role=ViewRole.THIRD_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp-unscanned", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
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
        global_start_ms=470_000.0,
        global_end_ms=471_000.0,
        key_global_ms=470_500.0,
        objects=["gloved_hand", "tube"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp-covered"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    segment = ExperimentSegment(
        segment_id="EXP-1",
        global_start_ms=470_000.0,
        global_end_ms=500_250.0,
        event_ids=[event.event_id],
        participating_views=["fp", "tp-covered"],
    )
    group = ExperimentGroup(
        group_id="G1",
        continuity_type="independent",
        atomic_experiment_ids=[segment.segment_id],
        global_start_ms=470_000.0,
        global_end_ms=500_250.0,
        participating_views=["fp", "tp-covered"],
        first_person_view="fp",
        third_person_view="tp-covered",
        continuity_reason="test",
    )
    unresolved = [
        _candidate(
            f"C-{index}",
            499_000.0 + index * 100.0,
            500_200.0 + index * 10.0,
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
        {
            "fp": [(470_000.0, 500_000.0)],
            "tp-covered": [(499_000.0, 500_000.0)],
        },
        infos,
        transforms,
    )

    assert plan["complete"] is False
    assert plan["selected_plans"][0]["view_id"] == "tp-unscanned"
    assert all(
        item["view_id"] != "tp-covered"
        for item in plan["groups"][0]["ranked_view_choices"]
    )


def test_group_local_recall_closes_each_temporal_cluster_with_cross_view_proof(
    default_config,
):
    default_config["performance"]["fine_group_recall_min_unresolved_anchors"] = 3
    pipeline = EvidencePipeline(default_config)
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=Path("tp.mp4")),
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
    cross_view = EvidenceEvent(
        event_id="E-CROSS",
        action_type=ActionType.DEVICE_PANEL_OPERATION,
        global_start_ms=350_000.0,
        global_end_ms=351_000.0,
        key_global_ms=350_500.0,
        objects=["balance", "gloved_hand"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    segment = ExperimentSegment(
        segment_id="EXP-1",
        global_start_ms=340_000.0,
        global_end_ms=370_000.0,
        event_ids=[cross_view.event_id],
        participating_views=["fp", "tp"],
    )
    group = ExperimentGroup(
        group_id="G1",
        continuity_type="independent",
        atomic_experiment_ids=[segment.segment_id],
        global_start_ms=340_000.0,
        global_end_ms=370_000.0,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
    )
    # These detector fragments deliberately disagree with the accepted event's
    # action/object labels.  They form one temporal cluster that already has
    # direct dual-role proof and therefore must not trigger redundant rescans.
    fragments = [
        _candidate(
            f"C-{index}",
            345_000.0 + index * 2_000.0,
            352_000.0 + index * 2_000.0,
            ActionType.OBJECT_MOVEMENT,
            ["tube"],
        )
        for index in range(3)
    ]

    plan = pipeline._group_local_recall_plan(
        [group],
        [segment],
        [cross_view],
        fragments,
        views,
        {view.view_id: {} for view in views},
        {"fp": [(340_000.0, 370_000.0)], "tp": [(340_000.0, 370_000.0)]},
        infos,
        transforms,
    )

    assert plan["complete"] is True
    assert plan["quality_complete"] is True
    assert plan["selected_plans"] == []
    report = plan["groups"][0]
    assert report["raw_candidate_level_unresolved_anchor_count"] == 3
    assert report["unresolved_anchor_count"] == 0
    assert report["cross_view_supported_temporal_cluster_count"] == 1
    assert report["cross_view_supported_temporal_clusters"][0][
        "cross_view_event_ids"
    ] == ["E-CROSS"]


@pytest.mark.parametrize(
    ("long_matching_fragments", "unrelated_view_fully_scanned", "distant_support_only"),
    [(False, False, False), (True, False, False),
     (False, True, False), (True, True, False), (False, False, True)],
)
def test_group_recall_does_not_bridge_unscanned_middle(
    default_config, long_matching_fragments, unrelated_view_fully_scanned,
    distant_support_only,
):
    """An early event cannot prove coverage of the later weighing interval."""
    default_config["performance"]["fine_group_recall_min_unresolved_anchors"] = 3
    default_config["performance"]["fine_group_recall_padding_seconds"] = 0
    pipeline = EvidencePipeline(default_config)
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=Path("tp.mp4")),
        ViewInput(view_id="other", role=ViewRole.THIRD_PERSON, video=Path("other.mp4")),
    ]
    infos = {
        v.view_id: VideoInfo(path=v.video, duration_ms=1_000_000.0,
                             fps=30.0, width=16, height=16, frame_count=30_000)
        for v in views
    }
    transforms = {
        v.view_id: AlignmentTransform(view_id=v.view_id, reference_view_id="fp",
                                      state="aligned", confidence=1.0)
        for v in views
    }
    event = EvidenceEvent(
        event_id="EARLY", action_type=ActionType.DEVICE_PANEL_OPERATION,
        global_start_ms=456_000.0, global_end_ms=457_000.0,
        key_global_ms=456_500.0, objects=["balance"], confidence=0.9,
        accepted=True, audit_reason="test", supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON], candidates=[],
    )
    segment = ExperimentSegment(
        segment_id="EXP", global_start_ms=445_000.0, global_end_ms=900_000.0,
        event_ids=[event.event_id], participating_views=["fp", "tp"],
    )
    group = ExperimentGroup(
        group_id="G", continuity_type="independent", atomic_experiment_ids=["EXP"],
        global_start_ms=445_000.0, global_end_ms=900_000.0,
        participating_views=["fp", "tp"], first_person_view="fp",
        third_person_view="tp", continuity_reason="test",
    )
    if long_matching_fragments:
        # Even direct same-label overlap is insufficient for a long candidate.
        fragments = [
            _candidate(f"C-{i}", 445_000.0 + i * 5_000, 900_000.0,
                       ActionType.DEVICE_PANEL_OPERATION, ["balance"])
            for i in range(3)
        ]
    else:
        # Many short fragments form a transitive cluster across the same gap.
        fragments = [
            _candidate(f"C-{i}", float(start), float(start + 10_000),
                       ActionType.OBJECT_MOVEMENT, ["tube"])
            for i, start in enumerate(range(445_000, 891_000, 5_000))
        ]
    coverage = {"fp": [(445_000.0, 900_000.0)],
                "tp": [(445_000.0, 500_000.0), (619_050.0, 900_000.0)]}
    if distant_support_only:
        # The early event's bench was scanned, but it has no event near the
        # later weighing action. A different, unscanned bench must be recalled.
        coverage["tp"] = [(445_000.0, 900_000.0)]
    if unrelated_view_fully_scanned:
        coverage["other"] = [(445_000.0, 900_000.0)]
    plan = pipeline._group_local_recall_plan(
        [group], [segment], [event], fragments, views,
        {v.view_id: {} for v in views}, coverage, infos, transforms,
    )
    assert plan["complete"] is False
    assert plan["quality_complete"] is False
    selected = plan["selected_plans"][0]
    assert selected["view_id"] == ("other" if distant_support_only else "tp")
    if distant_support_only:
        assert any(start <= 600_000.0 <= end for start, end in selected["windows"])
    else:
        assert selected["windows"] == [(500_000.0, 619_050.0)]
    assert plan["groups"][0]["unresolved_anchor_count"] >= 3


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


def test_scan_timestamp_gate_deduplicates_adjacent_physical_segment_boundary():
    emitted: dict[str, set[int]] = {}

    assert _accept_unique_frame_timestamp(emitted, "fp", 900_000.0) is True
    assert _accept_unique_frame_timestamp(emitted, "fp", 900_000.0) is False
    assert _accept_unique_frame_timestamp(emitted, "tp", 900_000.0) is True
    assert _accept_unique_frame_timestamp(emitted, "fp", 900_000.001) is True
    assert len(emitted["fp"]) == 2


def test_rtx_profile_enables_persistent_decode_and_local_recall():
    config = load_config(Path("configs/rtx4060-laptop-production.yaml"))

    assert config["performance"]["fine_persistent_segment_decode"] is True
    assert config["performance"]["fine_group_local_recall_enabled"] is True
    assert config["performance"]["fine_group_recall_min_unresolved_anchors"] == 3
    assert config["performance"]["fine_group_recall_padding_seconds"] == 0.0
