from __future__ import annotations

import gc
import json
import math
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .schemas import AlignmentTransform, BoxEvidence, FrameEvidence, VideoInfo, ViewInput, ViewRole
from .video_io import iter_view_sampled_frames


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


@dataclass
class _Track:
    track_id: int
    class_name: str
    box: tuple[float, float, float, float]
    last_ms: float
    hits: int = 1


class ByteSortTracker:
    """Dependency-light ByteTrack/SORT-style two-pass IoU tracker.

    High-confidence detections associate first; remaining tracks may associate with
    low-confidence boxes. This retains short occlusions without importing scipy.
    """

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        iou_threshold: float = 0.2,
        max_age_ms: float = 1750.0,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        self.max_age_ms = max_age_ms
        self.next_id = 1
        self.tracks: dict[int, _Track] = {}

    def _associate(self, track_ids: list[int], detections: list[BoxEvidence]) -> tuple[set[int], set[int]]:
        options: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            track = self.tracks[track_id]
            for det_index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                score = _iou(track.box, detection.xyxy_norm)
                if score >= self.iou_threshold:
                    options.append((score, track_id, det_index))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, det_index in sorted(options, reverse=True):
            if track_id in matched_tracks or det_index in matched_detections:
                continue
            detections[det_index].track_id = track_id
            matched_tracks.add(track_id)
            matched_detections.add(det_index)
        return matched_tracks, matched_detections

    def update(self, detections: list[BoxEvidence], local_ms: float) -> list[BoxEvidence]:
        self.tracks = {
            key: value for key, value in self.tracks.items() if local_ms - value.last_ms <= self.max_age_ms
        }
        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.confidence < self.high_threshold]
        active = list(self.tracks)
        matched_tracks, matched_high = self._associate(active, high)
        remaining_tracks = [track_id for track_id in active if track_id not in matched_tracks]
        matched_low_tracks, _ = self._associate(remaining_tracks, low)
        matched_tracks |= matched_low_tracks
        for index, detection in enumerate(high):
            if index not in matched_high:
                detection.track_id = self.next_id
                self.next_id += 1
        for detection in high + low:
            if detection.track_id is None:
                continue
            previous = self.tracks.get(detection.track_id)
            self.tracks[detection.track_id] = _Track(
                track_id=detection.track_id,
                class_name=detection.class_name,
                box=detection.xyxy_norm,
                last_ms=local_ms,
                hits=(previous.hits + 1) if previous else 1,
            )
        return high + [d for d in low if d.track_id is not None]


@dataclass
class FramePacket:
    view: ViewInput
    frame_index: int
    local_ms: float
    frame: np.ndarray
    gray: np.ndarray
    previous_gray: np.ndarray | None
    motion_score: float


@dataclass
class ChunkEnd:
    view_id: str
    chunk_index: int
    total_chunks: int


@dataclass
class ProducerEnd:
    view_id: str


@dataclass
class ProducerError:
    view_id: str
    message: str


def _read_checkpoint(path: Path, output_path: Path | None = None) -> set[int]:
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed = {int(item) for item in payload.get("completed_chunks", [])}
        if not completed:
            return set()
        if output_path is None or not output_path.is_file():
            return set()
        expected_size = payload.get("output_size_bytes")
        if expected_size is not None and output_path.stat().st_size != int(expected_size):
            return set()
        if output_path.stat().st_size <= 0:
            return set()
        return completed
    except (json.JSONDecodeError, OSError, ValueError):
        return set()


def _write_checkpoint(path: Path, completed: set[int], output_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-detection-checkpoint/2",
                "completed_chunks": sorted(completed),
                "output_path": str(output_path),
                "output_size_bytes": output_path.stat().st_size if output_path.is_file() else 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _producer(
    view: ViewInput,
    info: VideoInfo,
    output_queue: queue.Queue[Any],
    completed_chunks: set[int],
    config: dict[str, Any],
    windows: list[tuple[float, float]] | None,
    sample_fps: float,
    max_width: int,
    keyframes_only: bool,
    decode_backend: str,
    motion_probe_fps: float,
    motion_signature_size: tuple[int, int],
    wave_barrier: threading.Barrier | None,
) -> None:
    perf = config["performance"]
    chunk_ms = float(perf["chunk_seconds"]) * 1000.0
    spans = windows if windows is not None else [(0.0, info.duration_ms)]
    if windows is None and info.segments and perf.get("synchronized_segment_waves"):
        work_units = [
            (segment.virtual_start_ms, segment.virtual_end_ms) for segment in info.segments
        ]
    else:
        work_units: list[tuple[float, float]] = []
        for span_start, span_end in spans:
            cursor = max(0.0, span_start)
            bounded_end = min(info.duration_ms, span_end)
            while cursor < bounded_end:
                unit_end = min(bounded_end, cursor + chunk_ms)
                work_units.append((cursor, unit_end))
                cursor = unit_end
    previous_gray: np.ndarray | None = None
    previous_signature: np.ndarray | None = None
    decode_fps = max(sample_fps, motion_probe_fps)
    sample_period_ms = 1000.0 / max(sample_fps, 1e-9)
    activity_path = output_queue.activity_path if hasattr(output_queue, "activity_path") else None

    def activity(event: str, chunk_index: int | None = None) -> None:
        if activity_path is None:
            return
        payload = {
            "timestamp": time.time(),
            "event": event,
            "view_id": view.view_id,
            "role": view.role.value,
            "decode_backend": decode_backend,
            "chunk_index": chunk_index,
            "segment_path": (
                str(info.segments[chunk_index].path)
                if chunk_index is not None and info.segments and chunk_index < len(info.segments)
                else None
            ),
        }
        with output_queue.activity_lock:
            with activity_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    try:
        for chunk_index, (start_ms, end_ms) in enumerate(work_units):
            activity("source_unit_started", chunk_index)
            if chunk_index in completed_chunks:
                activity("source_unit_reused", chunk_index)
                output_queue.put(
                    ChunkEnd(
                        view_id=view.view_id,
                        chunk_index=chunk_index,
                        total_chunks=len(work_units),
                    )
                )
                if wave_barrier is not None:
                    wave_barrier.wait(
                        timeout=float(perf.get("segment_wave_timeout_seconds", 3600))
                    )
                continue
            next_yolo_ms = start_ms
            for frame_index, local_ms, frame in iter_view_sampled_frames(
                view,
                info,
                start_ms,
                end_ms,
                decode_fps,
                max_width,
                "cuda" if decode_backend == "cuda" else None,
                keyframes_only,
                int(perf.get("cpu_decode_threads", 0)) if decode_backend == "cpu" else None,
            ):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                signature = cv2.resize(gray, motion_signature_size, interpolation=cv2.INTER_AREA)
                motion = (
                    float(cv2.absdiff(signature, previous_signature).mean())
                    if previous_signature is not None
                    else 0.0
                )
                previous_signature = signature
                if local_ms + 0.5 < next_yolo_ms:
                    previous_gray = gray
                    continue
                output_queue.put(
                    FramePacket(
                        view=view,
                        frame_index=frame_index,
                        local_ms=local_ms,
                        frame=frame,
                        gray=gray,
                        previous_gray=previous_gray,
                        motion_score=motion,
                    )
                )
                previous_gray = gray
                while next_yolo_ms <= local_ms + 0.5:
                    next_yolo_ms += sample_period_ms
            output_queue.put(
                ChunkEnd(
                    view_id=view.view_id,
                    chunk_index=chunk_index,
                    total_chunks=len(work_units),
                )
            )
            activity("source_unit_completed", chunk_index)
            if wave_barrier is not None:
                wave_barrier.wait(timeout=float(perf.get("segment_wave_timeout_seconds", 3600)))
    except Exception as exc:  # producer errors must cross the thread boundary
        if wave_barrier is not None:
            try:
                wave_barrier.abort()
            except threading.BrokenBarrierError:
                pass
        output_queue.put(ProducerError(view_id=view.view_id, message=f"{type(exc).__name__}: {exc}"))
    finally:
        activity("source_worker_ended")
        output_queue.put(ProducerEnd(view_id=view.view_id))


def _roi_motion(previous: np.ndarray | None, current: np.ndarray, box: Sequence[float]) -> float:
    if previous is None or previous.shape != current.shape:
        return 0.0
    height, width = current.shape
    x1, y1, x2, y2 = box
    left, top = max(0, int(x1 * width)), max(0, int(y1 * height))
    right, bottom = min(width, int(math.ceil(x2 * width))), min(height, int(math.ceil(y2 * height)))
    if right - left < 2 or bottom - top < 2:
        return 0.0
    return float(cv2.absdiff(current[top:bottom, left:right], previous[top:bottom, left:right]).mean())


def _select_model_path(role: ViewRole, config: dict[str, Any]) -> Path:
    models = config["models"]
    role_name = role.value
    engine = Path(models[f"{role_name}_engine"])
    mode = str(config["performance"].get("tensor_rt", "auto")).lower()
    if mode in {"auto", "true", "required"} and engine.is_file():
        return engine
    if mode == "required":
        raise FileNotFoundError(f"TensorRT engine 不存在: {engine}；请先运行 prepare-engine")
    return Path(models[role_name])


def _tensorrt_plan_and_metadata(path: Path) -> tuple[bytes, dict[str, Any], str]:
    """Return a raw TensorRT plan from either a raw or Ultralytics-wrapped engine."""

    payload = path.read_bytes()
    if len(payload) < 5:
        return payload, {}, "raw"
    metadata_size = int.from_bytes(payload[:4], byteorder="little", signed=False)
    if metadata_size <= 0 or metadata_size > min(len(payload) - 4, 1024 * 1024):
        return payload, {}, "raw"
    try:
        metadata = json.loads(payload[4 : 4 + metadata_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return payload, {}, "raw"
    plan = payload[4 + metadata_size :]
    if not plan:
        return payload, {}, "raw"
    return plan, metadata if isinstance(metadata, dict) else {}, "ultralytics"


def _metadata_batch(metadata: dict[str, Any]) -> int | None:
    containers = [metadata]
    for key in ("args", "export", "engine"):
        nested = metadata.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in ("batch", "batch_size", "max_batch_size"):
            value = container.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return None


def _profile_batch(engine: Any) -> int | None:
    """Read the maximum explicit batch from the first TensorRT input profile."""

    try:
        tensor_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        input_name = next(
            name
            for name in tensor_names
            if str(engine.get_tensor_mode(name)).upper().endswith("INPUT")
        )
        _minimum, _optimum, maximum = engine.get_tensor_profile_shape(input_name, 0)
        return int(maximum[0]) if maximum and int(maximum[0]) > 0 else None
    except (AttributeError, RuntimeError, StopIteration, TypeError, ValueError):
        return None


def _engine_build_batch(path: Path) -> int | None:
    if path.suffix.lower() != ".engine" or not path.is_file():
        return None
    _plan, metadata, _container = _tensorrt_plan_and_metadata(path)
    return _metadata_batch(metadata)


def validate_models(config: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    expected = int(config["models"]["expected_class_count"])
    report: dict[str, Any] = {}
    class_sets: dict[str, list[str]] = {}
    for role in ViewRole:
        path = Path(config["models"][role.value])
        if not path.is_file():
            raise FileNotFoundError(f"{role.value} 模型不存在: {path}")
        model = YOLO(str(path))
        names = [str(model.names[index]).replace("-", "_") for index in sorted(model.names)]
        if len(names) != expected:
            raise ValueError(f"{path} 类别数为 {len(names)}，期望 {expected}")
        class_sets[role.value] = names
        report[role.value] = {"path": str(path), "class_count": len(names), "classes": names}
    if class_sets[ViewRole.FIRST_PERSON.value] != class_sets[ViewRole.THIRD_PERSON.value]:
        raise ValueError("第一/第三人称模型的规范化类别表不一致")
    mode = str(config["performance"].get("tensor_rt", "auto")).lower()
    runtime: dict[str, Any] = {"mode": mode, "roles": {}}
    if mode in {"required", "true"}:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT is required but the Python tensorrt package is unavailable"
            ) from exc
        runtime["tensorrt_version"] = trt.__version__
        logger = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(logger)
        for role in ViewRole:
            engine_path = Path(config["models"][f"{role.value}_engine"])
            if not engine_path.is_file():
                raise FileNotFoundError(
                    f"TensorRT engine missing for {role.value}: {engine_path}"
                )
            plan, metadata, container = _tensorrt_plan_and_metadata(engine_path)
            engine = trt_runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError(f"TensorRT engine cannot be deserialized: {engine_path}")
            build_batch = _metadata_batch(metadata) or _profile_batch(engine)
            runtime["roles"][role.value] = {
                "backend": "TensorRT",
                "engine": str(engine_path),
                "bytes": engine_path.stat().st_size,
                "deserialized": True,
                "container": container,
                "build_batch": build_batch,
            }
    else:
        for role in ViewRole:
            selected = _select_model_path(role, config)
            runtime["roles"][role.value] = {
                "backend": "TensorRT" if selected.suffix.lower() == ".engine" else "PyTorch",
                "path": str(selected),
            }
    report["runtime"] = runtime
    report["consistent"] = True
    return report


class RoleScanner:
    def __init__(
        self,
        role: ViewRole,
        config: dict[str, Any],
        image_size: int | None = None,
        batch_size: int | None = None,
    ):
        from ultralytics import YOLO

        self.role = role
        self.config = config
        self.model_path = _select_model_path(role, config)
        self.model = YOLO(str(self.model_path))
        self.names = {int(key): str(value).replace("-", "_") for key, value in self.model.names.items()}
        expected = int(config["models"]["expected_class_count"])
        if len(self.names) != expected:
            raise ValueError(f"{self.model_path} 不是 {expected} 类模型")
        self.requested_batch_size = int(batch_size or config["performance"]["batch_size"])
        self.engine_build_batch = _engine_build_batch(self.model_path)
        self.batch_size = min(
            self.requested_batch_size,
            self.engine_build_batch or self.requested_batch_size,
        )
        self.image_size = int(image_size or config["performance"]["image_size"])

    def close(self) -> None:
        del self.model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def infer(self, packets: Sequence[FramePacket]) -> list[list[BoxEvidence]]:
        perf, model_cfg = self.config["performance"], self.config["models"]
        batch_size = min(self.batch_size, len(packets))
        while True:
            try:
                results: list[list[BoxEvidence]] = []
                for start in range(0, len(packets), batch_size):
                    sub_batch = packets[start : start + batch_size]
                    predictions = self.model.predict(
                        source=[packet.frame for packet in sub_batch],
                        imgsz=self.image_size,
                        conf=float(model_cfg["confidence"]),
                        iou=float(model_cfg["iou"]),
                        max_det=int(model_cfg["max_detections"]),
                        device=perf["device"],
                        half=bool(perf["half"]),
                        verbose=False,
                    )
                    for packet, prediction in zip(sub_batch, predictions, strict=True):
                        boxes: list[BoxEvidence] = []
                        if prediction.boxes is not None:
                            xyxy = prediction.boxes.xyxyn.detach().cpu().numpy()
                            confidences = prediction.boxes.conf.detach().cpu().numpy()
                            classes = prediction.boxes.cls.detach().cpu().numpy().astype(int)
                            for coords, confidence, class_id in zip(xyxy, confidences, classes, strict=True):
                                coords_tuple = tuple(float(np.clip(item, 0.0, 1.0)) for item in coords)
                                boxes.append(
                                    BoxEvidence(
                                        class_id=int(class_id),
                                        class_name=self.names[int(class_id)],
                                        confidence=float(confidence),
                                        xyxy_norm=coords_tuple,
                                        roi_motion=_roi_motion(packet.previous_gray, packet.gray, coords_tuple),
                                    )
                                )
                        results.append(boxes)
                self.batch_size = batch_size
                return results
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass


def scan_videos(
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    work_dir: Path,
    config: dict[str, Any],
    windows: dict[str, list[tuple[float, float]]] | None = None,
    sample_fps: float | None = None,
    image_size: int | None = None,
    keyframes_only: bool = False,
    phase: str = "fine",
    decode_backends: dict[str, str] | None = None,
    wave_barrier: threading.Barrier | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {view.view_id: work_dir / f"{view.view_id}.detections.jsonl" for view in views}
    checkpoint_paths = {view.view_id: work_dir / f"{view.view_id}.checkpoint.json" for view in views}
    completed = {
        view.view_id: _read_checkpoint(
            checkpoint_paths[view.view_id], output_paths[view.view_id]
        )
        for view in views
    }
    for view in views:
        output_path = output_paths[view.view_id]
        if not completed[view.view_id] and output_path.is_file():
            # A ledger without a matching durable checkpoint is not safe to
            # append to: rerunning into it would duplicate frame evidence.
            output_path.unlink()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(config["performance"]["max_gpu_memory_fraction"]), 0)
            torch.backends.cudnn.benchmark = True
    except (ImportError, RuntimeError):
        pass

    effective_fps = float(sample_fps if sample_fps is not None else config["performance"]["detection_fps"])
    effective_image_size = int(image_size if image_size is not None else config["performance"]["image_size"])
    phase_batch_size = int(
        config["performance"].get(
            f"{phase}_batch_size", config["performance"].get("batch_size", 16)
        )
    )
    # Motion discovery is a separate, cheap stage. Coarse/fine scans must not
    # decode extra frames that are discarded before YOLO.
    probe_fps = effective_fps
    signature = config["performance"].get("motion_signature_size", [64, 36])
    signature_size = (int(signature[0]), int(signature[1]))
    for role in ViewRole:
        role_views = [view for view in views if view.role == role]
        if not role_views:
            continue
        motion_only = phase == "motion_probe"
        scanner = None if motion_only else RoleScanner(
            role, config, effective_image_size, phase_batch_size
        )
        runtime_report = {
            "phase": phase,
            "role": role.value,
            "model_path": str(scanner.model_path) if scanner is not None else None,
            "backend": (
                "motion_only"
                if scanner is None
                else "TensorRT" if scanner.model_path.suffix.lower() == ".engine" else "PyTorch"
            ),
            "requested_batch_size": phase_batch_size,
            "effective_batch_size": scanner.batch_size if scanner is not None else 0,
            "engine_build_batch": scanner.engine_build_batch if scanner is not None else None,
            "image_size": effective_image_size,
            "yolo_sample_fps": effective_fps,
            "motion_probe_fps": probe_fps,
            "decode_backends": {
                view.view_id: (decode_backends or {}).get(
                    view.view_id,
                    "cuda" if config["performance"].get("ffmpeg_hwaccel") else "cpu",
                )
                for view in role_views
            },
            "active_source_workers": len(role_views),
            "synchronized_segment_waves": wave_barrier is not None,
        }
        runtime_path = work_dir / f"runtime_{phase}_{role.value}.json"
        trackers = {
            view.view_id: ByteSortTracker(max_age_ms=max(1750.0, 1500.0 / effective_fps))
            for view in role_views
        } if scanner is not None else {}
        writers = {
            view.view_id: output_paths[view.view_id].open("a", encoding="utf-8", buffering=1024 * 1024)
            for view in role_views
        }
        queue_depth = int(
            config["performance"].get(
                "decode_queue_depth", config["performance"].get("frame_queue_size", 64)
            )
        )
        frame_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_depth))
        frame_queue.activity_path = work_dir / f"source_activity_{phase}_{role.value}.jsonl"
        frame_queue.activity_lock = threading.Lock()
        threads = [
            threading.Thread(
                target=_producer,
                args=(
                    view,
                    infos[view.view_id],
                    frame_queue,
                    completed[view.view_id],
                    config,
                    None if windows is None else windows.get(view.view_id, []),
                    effective_fps,
                    effective_image_size,
                    keyframes_only,
                    (decode_backends or {}).get(
                        view.view_id,
                        "cuda" if config["performance"].get("ffmpeg_hwaccel") else "cpu",
                    ),
                    probe_fps,
                    signature_size,
                    wave_barrier,
                ),
                name=f"decode-{view.view_id}",
                daemon=True,
            )
            for view in role_views
        ]
        for thread in threads:
            thread.start()

        ended: set[str] = set()
        batch: list[FramePacket] = []
        errors: list[str] = []
        batch_sizes: list[int] = []
        motion_sample_count = 0
        max_queue_size = 0

        def flush() -> None:
            nonlocal motion_sample_count
            if not batch:
                return
            if scanner is None:
                inferred = [[] for _ in batch]
                motion_sample_count += len(batch)
            else:
                batch_sizes.append(len(batch))
                inferred = scanner.infer(batch)
            for packet, boxes in zip(batch, inferred, strict=True):
                tracked = (
                    trackers[packet.view.view_id].update(boxes, packet.local_ms)
                    if scanner is not None
                    else []
                )
                evidence = FrameEvidence(
                    view_id=packet.view.view_id,
                    role=packet.view.role,
                    frame_index=packet.frame_index,
                    local_ms=packet.local_ms,
                    global_ms=transforms[packet.view.view_id].to_global(packet.local_ms),
                    width=packet.frame.shape[1],
                    height=packet.frame.shape[0],
                    motion_score=packet.motion_score,
                    detections=tracked,
                )
                writers[packet.view.view_id].write(evidence.model_dump_json() + "\n")
            batch.clear()

        try:
            while len(ended) < len(role_views):
                item = frame_queue.get()
                max_queue_size = max(max_queue_size, frame_queue.qsize())
                if isinstance(item, FramePacket):
                    batch.append(item)
                    if scanner is None or len(batch) >= scanner.batch_size:
                        flush()
                else:
                    flush()
                    if isinstance(item, ChunkEnd):
                        writers[item.view_id].flush()
                        completed[item.view_id].add(item.chunk_index)
                        _write_checkpoint(
                            checkpoint_paths[item.view_id],
                            completed[item.view_id],
                            output_paths[item.view_id],
                        )
                        if progress_callback is not None:
                            progress_callback(
                                item.view_id,
                                len(completed[item.view_id]),
                                item.total_chunks,
                            )
                    elif isinstance(item, ProducerError):
                        errors.append(f"{item.view_id}: {item.message}")
                    elif isinstance(item, ProducerEnd):
                        ended.add(item.view_id)
            flush()
            if errors:
                raise RuntimeError("；".join(errors))
        finally:
            for thread in threads:
                thread.join(timeout=5.0)
            for writer in writers.values():
                writer.close()
            runtime_report.update(
                {
                    "completed_source_workers": len(ended),
                    "inference_call_count": len(batch_sizes),
                    "inference_frame_count": sum(batch_sizes),
                    "motion_sample_count": motion_sample_count,
                    "actual_batch_size_min": min(batch_sizes) if batch_sizes else 0,
                    "actual_batch_size_max": max(batch_sizes) if batch_sizes else 0,
                    "actual_batch_size_mean": (
                        round(sum(batch_sizes) / len(batch_sizes), 4) if batch_sizes else 0.0
                    ),
                    "requested_batch_fill_ratio": (
                        round(sum(batch_sizes) / len(batch_sizes) / phase_batch_size, 4)
                        if batch_sizes
                        else 0.0
                    ),
                    "effective_batch_fill_ratio": (
                        round(sum(batch_sizes) / len(batch_sizes) / scanner.batch_size, 4)
                        if batch_sizes and scanner is not None
                        else 0.0
                    ),
                    "max_observed_queue_depth": max_queue_size,
                    "configured_queue_depth": queue_depth,
                }
            )
            runtime_path.write_text(
                json.dumps(runtime_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if scanner is not None:
                scanner.close()
    return output_paths


def iter_frame_evidence(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield FrameEvidence.model_validate_json(line)


def nearest_frame_evidence(path: Path, global_ms: float, tolerance_ms: float = 1500.0) -> FrameEvidence | None:
    best: FrameEvidence | None = None
    best_distance = float("inf")
    for frame in iter_frame_evidence(path):
        if frame.global_ms is None:
            continue
        distance = abs(frame.global_ms - global_ms)
        if distance < best_distance:
            best, best_distance = frame, distance
        if frame.global_ms > global_ms + tolerance_ms:
            break
    return best if best_distance <= tolerance_ms else None
