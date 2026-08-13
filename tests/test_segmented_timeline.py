from pathlib import Path

import cv2
import numpy as np
import pytest

from labvision_evidence.schemas import VideoSegmentInput, ViewInput, ViewRole
from labvision_evidence.video_io import (
    extract_view_clip,
    iter_view_sampled_frames,
    probe_video,
    probe_view,
)


def _write_video(path: Path, color: tuple[int, int, int], seconds: int = 2) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48)
    )
    assert writer.isOpened()
    for _ in range(seconds * 5):
        writer.write(np.full((48, 64, 3), color, dtype=np.uint8))
    writer.release()


def test_segmented_view_is_zero_copy_virtual_timeline(tmp_path):
    first = tmp_path / "part-01.mp4"
    second = tmp_path / "part-02.mp4"
    _write_video(first, (0, 0, 255))
    _write_video(second, (0, 255, 0))
    view = ViewInput(
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        segments=[VideoSegmentInput(video=first), VideoSegmentInput(video=second)],
    )

    info = probe_view(view)

    assert len(info.segments) == 2
    assert info.duration_ms == pytest.approx(
        probe_video(first).duration_ms + probe_video(second).duration_ms, abs=20
    )
    assert info.segments[1].virtual_start_ms == pytest.approx(
        info.segments[0].virtual_end_ms
    )
    assert sorted(item.name for item in tmp_path.iterdir()) == ["part-01.mp4", "part-02.mp4"]


def test_segmented_sampling_and_export_are_continuous(tmp_path):
    first = tmp_path / "part-01.mp4"
    second = tmp_path / "part-02.mp4"
    destination = tmp_path / "bounded-experiment.mp4"
    _write_video(first, (0, 0, 255))
    _write_video(second, (0, 255, 0))
    view = ViewInput(
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        segments=[VideoSegmentInput(video=first), VideoSegmentInput(video=second)],
    )
    info = probe_view(view)
    boundary = info.segments[0].virtual_end_ms

    sampled = list(
        iter_view_sampled_frames(
            view,
            info,
            boundary - 1000,
            boundary + 1000,
            sample_fps=1.0,
            max_width=64,
            hwaccel=None,
        )
    )
    extract_view_clip(
        view,
        info,
        destination,
        boundary - 1000,
        2000,
        preferred_encoder="libx264",
    )

    assert len(sampled) >= 2
    assert sampled[0][1] < boundary <= sampled[-1][1]
    assert destination.is_file()
    exported = probe_video(destination)
    assert exported.duration_ms == pytest.approx(2000, abs=300)
