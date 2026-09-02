from pathlib import Path

import pytest

from labvision_evidence import input_preflight
from labvision_evidence.schemas import (
    RunManifest,
    TimestampPoint,
    VideoInfo,
    ViewInput,
)


def _manifest(tmp_path: Path) -> RunManifest:
    return RunManifest(
        experiment_id="preflight",
        views=[
            ViewInput(
                view_id="first",
                role="first_person",
                video=tmp_path / "first.mp4",
                timestamps_csv=tmp_path / "first.csv",
            ),
            ViewInput(
                view_id="third",
                role="third_person",
                video=tmp_path / "third.mp4",
                timestamps_csv=tmp_path / "third.csv",
            ),
        ],
    )


def _infos(manifest: RunManifest) -> dict[str, VideoInfo]:
    return {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=4_000,
            fps=25,
            width=1920,
            height=1080,
            frame_count=100,
            size_bytes=1024,
            media_timing_source="test_probe",
        )
        for view in manifest.views
    }


def test_prequeue_preflight_proves_common_clock_coverage(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        input_preflight,
        "probe_views",
        lambda _views, workers, prefer_clock_metadata: _infos(manifest),
    )

    def endpoints(path: Path, _fps: float):
        start = 1_000 if path.name == "first.csv" else 1_500
        return [
            TimestampPoint(frame_index=0, local_ms=0, source_ms=start),
            TimestampPoint(frame_index=99, local_ms=3_960, source_ms=start + 4_000),
        ]

    monkeypatch.setattr(input_preflight, "read_timestamp_csv_endpoints", endpoints)

    receipt = input_preflight.preflight_manifest_inputs(manifest, {})

    assert receipt["status"] == "passed"
    assert receipt["common_clock_overlap"]["start_ms"] == 1_500
    assert receipt["common_clock_overlap"]["end_ms"] == 5_000
    assert receipt["video_segment_count"] == 2


def test_prequeue_preflight_blocks_views_without_clock_overlap(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        input_preflight,
        "probe_views",
        lambda _views, workers, prefer_clock_metadata: _infos(manifest),
    )

    def endpoints(path: Path, _fps: float):
        start = 1_000 if path.name == "first.csv" else 10_000
        return [
            TimestampPoint(frame_index=0, local_ms=0, source_ms=start),
            TimestampPoint(frame_index=99, local_ms=3_960, source_ms=start + 4_000),
        ]

    monkeypatch.setattr(input_preflight, "read_timestamp_csv_endpoints", endpoints)

    with pytest.raises(ValueError, match="没有共同绝对时钟覆盖"):
        input_preflight.preflight_manifest_inputs(manifest, {})
