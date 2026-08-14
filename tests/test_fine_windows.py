from pathlib import Path

from labvision_evidence.pipeline import EvidencePipeline
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    VideoInfo,
    ViewRole,
)


def _candidate(candidate_id: str, start_ms: float, end_ms: float) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=["tube"],
        confidence=0.9,
    )


def test_fine_windows_do_not_bridge_a_second_padding_gap(default_config):
    default_config["performance"]["fine_window_padding_seconds"] = 10.0
    default_config["performance"]["fine_window_merge_gap_seconds"] = 0.0
    pipeline = EvidencePipeline(default_config)
    infos = {
        "fp01": VideoInfo(
            path=Path("fp.mp4"),
            duration_ms=120_000,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3600,
            size_bytes=1,
        )
    }
    transforms = {
        "fp01": AlignmentTransform(
            view_id="fp01",
            reference_view_id="fp01",
            confidence=1.0,
            state="aligned",
        )
    }

    windows = pipeline._fine_windows(
        [
            _candidate("A", 20_000, 30_000),
            _candidate("B", 55_000, 65_000),
        ],
        infos,
        transforms,
    )

    assert windows["fp01"] == [(10_000.0, 40_000.0), (45_000.0, 75_000.0)]


def test_fine_windows_merge_overlapping_padded_intervals(default_config):
    default_config["performance"]["fine_window_padding_seconds"] = 10.0
    default_config["performance"]["fine_window_merge_gap_seconds"] = 0.0
    pipeline = EvidencePipeline(default_config)
    infos = {
        "fp01": VideoInfo(
            path=Path("fp.mp4"),
            duration_ms=120_000,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3600,
            size_bytes=1,
        )
    }
    transforms = {
        "fp01": AlignmentTransform(
            view_id="fp01",
            reference_view_id="fp01",
            confidence=1.0,
            state="aligned",
        )
    }

    windows = pipeline._fine_windows(
        [
            _candidate("A", 20_000, 30_000),
            _candidate("B", 35_000, 45_000),
        ],
        infos,
        transforms,
    )

    assert windows["fp01"] == [(10_000.0, 55_000.0)]
