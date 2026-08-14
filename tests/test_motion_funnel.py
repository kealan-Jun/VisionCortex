import json
from pathlib import Path

from labvision_evidence.actions import fuse_motion_probe_candidates
from labvision_evidence.alignment import _absolute_clock_transform
from labvision_evidence.detection import _engine_build_batch, _tensorrt_plan_and_metadata
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    TimestampPoint,
    VideoInfo,
    VideoSegmentInput,
    ViewInput,
    ViewRole,
)
from labvision_evidence.video_io import probe_views


def _motion(view_id: str, role: ViewRole, start_ms: float, end_ms: float) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=f"M-{view_id}-{start_ms}",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id=view_id,
        role=role,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=[],
        confidence=0.8,
    )


def test_motion_probe_prefers_cross_view_and_keeps_sustained_primary(default_config):
    candidates = [
        _motion("fp", ViewRole.FIRST_PERSON, 10_000, 25_000),
        _motion("tp", ViewRole.THIRD_PERSON, 12_000, 24_000),
        _motion("fp", ViewRole.FIRST_PERSON, 100_000, 160_000),
        _motion("fp", ViewRole.FIRST_PERSON, 300_000, 310_000),
    ]
    fused = fuse_motion_probe_candidates(candidates, default_config)
    assert len(fused) == 2
    assert fused[0].global_start_ms == 10_000
    assert fused[0].global_end_ms == 25_000
    assert {item["view_id"] for item in fused[0].evidence} == {"fp", "tp"}
    assert fused[1].global_start_ms == 100_000
    assert fused[1].global_end_ms == 160_000


def test_ultralytics_engine_metadata_caps_runtime_batch(tmp_path):
    metadata = json.dumps({"batch": 8, "imgsz": 640}).encode("utf-8")
    plan = b"serialized-tensorrt-plan"
    engine = tmp_path / "best.engine"
    engine.write_bytes(len(metadata).to_bytes(4, "little") + metadata + plan)
    extracted, parsed, container = _tensorrt_plan_and_metadata(engine)
    assert extracted == plan
    assert parsed["batch"] == 8
    assert container == "ultralytics"
    assert _engine_build_batch(engine) == 8


def test_parallel_segment_probe_preserves_virtual_order(monkeypatch):
    view = ViewInput(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        segments=[
            VideoSegmentInput(video=Path("segment-2.mp4")),
            VideoSegmentInput(video=Path("segment-1.mp4")),
        ],
    )

    def fake_probe(path: Path) -> VideoInfo:
        duration = 2_000.0 if path.name == "segment-2.mp4" else 1_000.0
        return VideoInfo(
            path=path,
            duration_ms=duration,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=int(duration / 1000.0 * 30),
            size_bytes=100,
        )

    monkeypatch.setattr("labvision_evidence.video_io.probe_video", fake_probe)
    info = probe_views([view], workers=2)["fp"]
    assert [segment.path.name for segment in info.segments] == [
        "segment-2.mp4",
        "segment-1.mp4",
    ]
    assert [segment.virtual_start_ms for segment in info.segments] == [0.0, 2_000.0]
    assert info.duration_ms == 3_000.0


def test_absolute_clock_fit_does_not_require_simultaneous_segment_boundaries():
    reference = [
        TimestampPoint(frame_index=0, local_ms=0.0, source_ms=1_000_000.0),
        TimestampPoint(frame_index=1, local_ms=900_000.0, source_ms=1_900_000.0),
        TimestampPoint(frame_index=2, local_ms=1_800_000.0, source_ms=2_800_000.0),
    ]
    target = [
        TimestampPoint(frame_index=0, local_ms=0.0, source_ms=1_000_075.0),
        TimestampPoint(frame_index=1, local_ms=903_000.0, source_ms=1_903_075.0),
        TimestampPoint(frame_index=2, local_ms=1_797_000.0, source_ms=2_797_075.0),
    ]
    result = _absolute_clock_transform(reference, target, max_drift_ppm=2_500.0)
    assert result is not None
    scale, offset, rmse, count = result
    assert count == 3
    assert abs(scale - 1.0) < 1e-6
    assert abs(offset - 75.0) < 0.01
    assert rmse < 0.01
