import time
from pathlib import Path

from labvision_evidence import video_io
from labvision_evidence.schemas import (
    VideoInfo,
    VideoSegmentInfo,
    VideoSegmentInput,
    ViewInput,
    ViewRole,
)


def _view_and_info():
    path = Path("sentinel.mp4")
    return (
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=path),
        VideoInfo(
            path=path,
            duration_ms=120_000,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3600,
            size_bytes=1,
        ),
    )


def test_sparse_decode_benchmark_selects_fastest_usable_strategy(monkeypatch):
    view, info = _view_and_info()

    def fake_frames(*_args, **_kwargs):
        strategy = _args[-2]
        time.sleep(0.001 if strategy == "indexed_seek" else 0.006)
        for index in range(6):
            yield index, index * 10_000.0, None

    monkeypatch.setattr(video_io, "iter_view_sampled_frames", fake_frames)

    report = video_io.benchmark_sparse_decode_strategy(
        view,
        info,
        sample_fps=0.1,
        max_width=416,
        hwaccel="cuda",
        decoder_threads=8,
        cuda_scale=True,
        benchmark_seconds=60.0,
    )

    assert report["selected_strategy"] == "indexed_seek"
    assert all(item["usable"] for item in report["strategies"])


def test_sparse_decode_benchmark_uses_surviving_strategy(monkeypatch):
    view, info = _view_and_info()

    def fake_frames(*_args, **_kwargs):
        strategy = _args[-2]
        if strategy == "indexed_seek":
            raise RuntimeError("seek index unavailable")
        for index in range(6):
            yield index, index * 10_000.0, None

    monkeypatch.setattr(video_io, "iter_view_sampled_frames", fake_frames)

    report = video_io.benchmark_sparse_decode_strategy(
        view,
        info,
        sample_fps=0.1,
        max_width=416,
        hwaccel="cuda",
        decoder_threads=8,
        cuda_scale=True,
        benchmark_seconds=60.0,
    )

    assert report["selected_strategy"] == "sequential_keyframes"
    indexed = next(
        item for item in report["strategies"] if item["strategy"] == "indexed_seek"
    )
    assert indexed["usable"] is False
    assert "seek index unavailable" in indexed["error"]


def test_segmented_sparse_decode_benchmark_retries_partial_smb_read(monkeypatch):
    first = Path("segment-1.mp4")
    second = Path("segment-2.mp4")
    view = ViewInput(
        view_id="fp01",
        role=ViewRole.FIRST_PERSON,
        segments=[VideoSegmentInput(video=first), VideoSegmentInput(video=second)],
    )
    info = VideoInfo(
        path=first,
        duration_ms=120_000,
        fps=30.0,
        width=1920,
        height=1080,
        frame_count=3600,
        size_bytes=2,
        segments=[
            VideoSegmentInfo(
                path=first,
                virtual_start_ms=0,
                virtual_end_ms=20_000,
                frame_start_index=0,
                duration_ms=20_000,
                fps=30.0,
                width=1920,
                height=1080,
                frame_count=600,
                size_bytes=1,
            ),
            VideoSegmentInfo(
                path=second,
                virtual_start_ms=20_000,
                virtual_end_ms=120_000,
                frame_start_index=600,
                duration_ms=100_000,
                fps=30.0,
                width=1920,
                height=1080,
                frame_count=3000,
                size_bytes=1,
            ),
        ],
    )
    calls = 0

    def transient_frames(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        # Both strategies in the first attempt see only the first physical
        # segment. The bounded retry gets full sample coverage.
        count = 2 if calls <= 2 else 6
        for index in range(count):
            yield index, index * 10_000.0, None

    monkeypatch.setattr(video_io, "iter_view_sampled_frames", transient_frames)
    monkeypatch.setattr(video_io.time, "sleep", lambda _seconds: None)

    report = video_io.benchmark_sparse_decode_strategy(
        view,
        info,
        sample_fps=0.1,
        max_width=416,
        hwaccel="cuda",
        decoder_threads=8,
        cuda_scale=True,
        benchmark_seconds=60.0,
    )

    assert report["attempt_count"] == 2
    assert report["transient_retry_used"] is True
    assert len(report["strategies"]) == 4
    assert all(not item["usable"] for item in report["strategies"][:2])
    assert all(item["usable"] for item in report["strategies"][2:])
