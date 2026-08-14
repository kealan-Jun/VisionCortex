import json
import queue
import time
from pathlib import Path

import numpy as np

from labvision_evidence import video_io
from labvision_evidence.actions import fuse_motion_probe_candidates
from labvision_evidence.alignment import _absolute_clock_transform
from labvision_evidence.detection import (
    ChunkEnd,
    FramePacket,
    ProducerEnd,
    RoleScanner,
    _engine_build_batch,
    _producer,
    _tensorrt_plan_and_metadata,
    scan_videos,
)
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    TimestampPoint,
    VideoInfo,
    VideoSegmentInfo,
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


def test_inference_batch_crosses_chunk_boundary_without_losing_checkpoint(
    monkeypatch, tmp_path, default_config
):
    view = ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=Path("unused.mp4"))
    info = VideoInfo(
        path=Path("unused.mp4"),
        duration_ms=2_000.0,
        fps=30.0,
        width=8,
        height=8,
        frame_count=60,
        size_bytes=100,
    )
    transform = AlignmentTransform(
        view_id="tp", reference_view_id="tp", state="aligned", confidence=1.0
    )

    class FakeScanner:
        calls: list[int] = []

        def __init__(self, *_args, **_kwargs):
            self.model_path = Path("fake.engine")
            self.batch_size = 4
            self.engine_build_batch = 4
            self.last_inference_batch_sizes = []

        def infer(self, packets):
            self.last_inference_batch_sizes = [len(packets)]
            self.calls.append(len(packets))
            return [[] for _ in packets]

        def close(self):
            pass

    def fake_producer(view, _info, output, *_args):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        gray = np.zeros((8, 8), dtype=np.uint8)
        for chunk in range(2):
            for offset in range(2):
                output.put(
                    FramePacket(
                        view=view,
                        frame_index=chunk * 2 + offset,
                        local_ms=float(chunk * 1_000 + offset * 100),
                        frame=frame,
                        gray=gray,
                        previous_gray=None,
                        motion_score=0.0,
                    )
                )
            output.put(ChunkEnd(view_id=view.view_id, chunk_index=chunk, total_chunks=2))
        output.put(ProducerEnd(view_id=view.view_id))

    monkeypatch.setattr("labvision_evidence.detection.RoleScanner", FakeScanner)
    monkeypatch.setattr("labvision_evidence.detection._producer", fake_producer)
    default_config["performance"]["coarse_batch_size"] = 4
    default_config["performance"]["inference_batch_wait_ms"] = 100
    paths = scan_videos(
        [view],
        {"tp": info},
        {"tp": transform},
        tmp_path,
        default_config,
        phase="coarse",
    )

    assert FakeScanner.calls == [4]
    assert len(paths["tp"].read_text(encoding="utf-8").splitlines()) == 4
    checkpoint = json.loads((tmp_path / "tp.checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["completed_chunks"] == [0, 1]
    runtime = json.loads(
        (tmp_path / "runtime_coarse_third_person.json").read_text(encoding="utf-8")
    )
    assert runtime["effective_batch_fill_ratio"] == 1.0
    assert runtime["full_batch_flushes"] == 1


def test_motion_probe_can_run_yolo_on_the_same_sampled_frames(
    monkeypatch, tmp_path, default_config
):
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("unused.mp4"))
    info = VideoInfo(
        path=Path("unused.mp4"),
        duration_ms=2_000.0,
        fps=30.0,
        width=8,
        height=8,
        frame_count=60,
        size_bytes=100,
    )
    transform = AlignmentTransform(
        view_id="fp", reference_view_id="fp", state="aligned", confidence=1.0
    )

    class FakeScanner:
        calls: list[int] = []

        def __init__(self, *_args, **_kwargs):
            self.model_path = Path("fake.engine")
            self.batch_size = 4
            self.engine_build_batch = 4
            self.last_inference_batch_sizes = []

        def infer(self, packets):
            self.last_inference_batch_sizes = [len(packets)]
            self.calls.append(len(packets))
            return [[] for _ in packets]

        def close(self):
            pass

    def fake_producer(view, _info, output, *_args):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        gray = np.zeros((8, 8), dtype=np.uint8)
        for index in range(4):
            output.put(
                FramePacket(
                    view=view,
                    frame_index=index,
                    local_ms=float(index * 100),
                    frame=frame,
                    gray=gray,
                    previous_gray=None,
                    motion_score=float(index),
                )
            )
        output.put(ChunkEnd(view_id=view.view_id, chunk_index=0, total_chunks=1))
        output.put(ProducerEnd(view_id=view.view_id))

    monkeypatch.setattr("labvision_evidence.detection.RoleScanner", FakeScanner)
    monkeypatch.setattr("labvision_evidence.detection._producer", fake_producer)
    default_config["performance"]["motion_probe_run_yolo"] = True
    default_config["performance"]["motion_probe_batch_size"] = 4

    scan_videos(
        [view],
        {"fp": info},
        {"fp": transform},
        tmp_path,
        default_config,
        phase="motion_probe",
    )

    runtime = json.loads(
        (tmp_path / "runtime_motion_probe_first_person.json").read_text(
            encoding="utf-8"
        )
    )
    assert FakeScanner.calls == [4]
    assert runtime["backend"] != "motion_only"
    assert runtime["motion_sample_count"] == 4
    assert runtime["inference_frame_count"] == 4


def test_role_scanner_records_oom_batch_contraction(default_config):
    class Prediction:
        boxes = None

    class FakeModel:
        def predict(self, *, source, **_kwargs):
            if len(source) > 4:
                raise RuntimeError("CUDA out of memory")
            return [Prediction() for _ in source]

    scanner = RoleScanner.__new__(RoleScanner)
    scanner.config = default_config
    scanner.model = FakeModel()
    scanner.names = {}
    scanner.batch_size = 8
    scanner.initial_batch_size = 8
    scanner.batch_contractions = []
    scanner.last_inference_batch_sizes = []
    scanner.image_size = 640
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    gray = np.zeros((8, 8), dtype=np.uint8)
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4"))
    packets = [
        FramePacket(
            view=view,
            frame_index=index,
            local_ms=float(index),
            frame=frame,
            gray=gray,
            previous_gray=None,
            motion_score=0.0,
        )
        for index in range(8)
    ]

    assert scanner.infer(packets) == [[] for _ in packets]
    assert scanner.batch_size == 4
    assert scanner.last_inference_batch_sizes == [4, 4]
    assert scanner.batch_contractions == [
        {"from_batch_size": 8, "to_batch_size": 4}
    ]


def test_parallel_motion_probe_preserves_segment_order(monkeypatch, default_config):
    view = ViewInput(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        segments=[VideoSegmentInput(video=Path(f"segment-{index}.mp4")) for index in range(3)],
    )
    segments = [
        VideoSegmentInfo(
            path=item.video,
            virtual_start_ms=index * 1_000.0,
            virtual_end_ms=(index + 1) * 1_000.0,
            frame_start_index=index * 30,
            duration_ms=1_000.0,
            fps=30.0,
            width=8,
            height=8,
            frame_count=30,
            size_bytes=100,
        )
        for index, item in enumerate(view.segments)
    ]
    info = VideoInfo(
        path=segments[0].path,
        duration_ms=3_000.0,
        fps=30.0,
        width=8,
        height=8,
        frame_count=90,
        size_bytes=300,
        segments=segments,
    )

    def fake_frames(_view, _info, start_ms, *_args, **_kwargs):
        time.sleep((2_000.0 - start_ms) / 100_000.0)
        yield int(start_ms / 1_000.0), start_ms, np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr("labvision_evidence.detection.iter_view_sampled_frames", fake_frames)
    default_config["performance"]["motion_probe_segment_workers"] = 3
    default_config["performance"]["synchronized_segment_waves"] = False
    output: queue.Queue = queue.Queue()
    _producer(
        view,
        info,
        output,
        set(),
        default_config,
        None,
        0.25,
        96,
        True,
        "cpu",
        0.25,
        (64, 36),
        None,
    )
    items = []
    while not output.empty():
        items.append(output.get())
    assert [item.local_ms for item in items if isinstance(item, FramePacket)] == [
        0.0,
        1_000.0,
        2_000.0,
    ]
    assert [item.chunk_index for item in items if isinstance(item, ChunkEnd)] == [0, 1, 2]


def test_sequential_sparse_strategy_bypasses_random_indexed_seeks(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    info = VideoInfo(
        path=source,
        duration_ms=60_000,
        fps=30,
        width=1280,
        height=720,
        frame_count=1800,
    )
    indexed_calls = []
    ffmpeg_calls = []

    def fake_indexed(*args, **kwargs):
        indexed_calls.append(True)
        yield from ()

    def fake_ffmpeg(*args, **kwargs):
        ffmpeg_calls.append(args)
        yield 0, 0.0, np.zeros((36, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(video_io, "_opencv_indexed_seek_iterator", fake_indexed)
    monkeypatch.setattr(video_io, "_ffmpeg_frame_iterator", fake_ffmpeg)
    monkeypatch.setattr(video_io.shutil, "which", lambda name: "ffmpeg")

    frames = list(
        video_io.iter_sampled_frames(
            source,
            info,
            0.0,
            60_000.0,
            0.1,
            416,
            "cuda",
            True,
            None,
            "sequential_keyframes",
        )
    )

    assert len(frames) == 1
    assert not indexed_calls
    assert len(ffmpeg_calls) == 1


def test_short_microbatch_does_not_permanently_contract_engine_capacity(monkeypatch, default_config):
    scanner = RoleScanner.__new__(RoleScanner)
    scanner.role = ViewRole.FIRST_PERSON
    scanner.config = default_config
    scanner.model_path = Path("model.engine")
    scanner.names = {0: "hand"}
    scanner.requested_batch_size = 8
    scanner.engine_build_batch = 8
    scanner.batch_size = 8
    scanner.initial_batch_size = 8
    scanner.batch_contractions = []
    scanner.last_inference_batch_sizes = []
    scanner.image_size = 416

    class EmptyPrediction:
        boxes = None

    class FakeModel:
        def predict(self, source, **kwargs):
            return [EmptyPrediction() for _ in source]

    scanner.model = FakeModel()
    packet = FramePacket(
        view=ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        frame_index=0,
        local_ms=0.0,
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        gray=np.zeros((8, 8), dtype=np.uint8),
        previous_gray=None,
        motion_score=0.0,
    )

    scanner.infer([packet])

    assert scanner.last_inference_batch_sizes == [1]
    assert scanner.batch_size == 8
    assert scanner.batch_contractions == []
