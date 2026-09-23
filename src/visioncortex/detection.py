from __future__ import annotations

import gc
import hashlib
import json
import math
import queue
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .detection_duplicates import duplicate_suppression_policies, suppress_duplicate_boxes
from .cuda_decode_admission import CudaDecodeAdmission
from .detection_inference import prediction_branches, prediction_contract
from .performance_stages import StageTimings, MeasuredWriter, prediction_timings
from .key_material_verification import validate_selective_key_material_verification
from .schemas import AlignmentTransform, BoxEvidence, FrameEvidence, SourceFrameIdentity, VideoInfo, ViewInput, ViewRole
from .source_frames import SOURCE_FRAME_CONTRACT
from .liquid_semantic import validate_liquid_semantic_runtime
from .temporal_segmentation import validate_temporal_segmentation_runtime
from .video_io import (
    PhysicalSegmentDecodeSession,
    iter_physical_segment_session_frames,
    iter_view_sampled_frames,
    plan_physical_segment_decode_sessions,
)


def _accept_unique_frame_timestamp(
    emitted: dict[str, set[int]], view_id: str, local_ms: float
) -> bool:
    """Accept one ledger row per view and microsecond-normalized timestamp."""

    key = round(float(local_ms) * 1000.0)
    view_keys = emitted.setdefault(view_id, set())
    if key in view_keys:
        return False
    view_keys.add(key)
    return True


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
    velocity_x_per_ms: float = 0.0
    velocity_y_per_ms: float = 0.0


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
        motion_prediction_enabled: bool = False,
        maximum_center_distance: float = 0.18,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        self.max_age_ms = max_age_ms
        self.motion_prediction_enabled = motion_prediction_enabled
        self.maximum_center_distance = maximum_center_distance
        self.next_id = 1
        self.tracks: dict[int, _Track] = {}

    @staticmethod
    def _center(box: Sequence[float]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _predicted_box(
        self, track: _Track, local_ms: float
    ) -> tuple[float, float, float, float]:
        if not self.motion_prediction_enabled:
            return track.box
        delta_ms = max(0.0, local_ms - track.last_ms)
        shift_x = track.velocity_x_per_ms * delta_ms
        shift_y = track.velocity_y_per_ms * delta_ms
        return (
            track.box[0] + shift_x,
            track.box[1] + shift_y,
            track.box[2] + shift_x,
            track.box[3] + shift_y,
        )

    def _associate(
        self,
        track_ids: list[int],
        detections: list[BoxEvidence],
        local_ms: float,
    ) -> tuple[set[int], set[int]]:
        options: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            track = self.tracks[track_id]
            predicted = self._predicted_box(track, local_ms)
            predicted_center = self._center(predicted)
            for det_index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                overlap = _iou(predicted, detection.xyxy_norm)
                detection_center = self._center(detection.xyxy_norm)
                center_distance = float(
                    np.hypot(
                        predicted_center[0] - detection_center[0],
                        predicted_center[1] - detection_center[1],
                    )
                )
                center_score = max(
                    0.0,
                    1.0
                    - center_distance / max(self.maximum_center_distance, 1e-9),
                )
                if overlap >= self.iou_threshold:
                    options.append((1.0 + overlap, track_id, det_index))
                elif self.motion_prediction_enabled and center_score > 0.0:
                    options.append((center_score, track_id, det_index))
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
        matched_tracks, matched_high = self._associate(active, high, local_ms)
        remaining_tracks = [track_id for track_id in active if track_id not in matched_tracks]
        matched_low_tracks, _ = self._associate(remaining_tracks, low, local_ms)
        matched_tracks |= matched_low_tracks
        for index, detection in enumerate(high):
            if index not in matched_high:
                detection.track_id = self.next_id
                self.next_id += 1
        for detection in high + low:
            if detection.track_id is None:
                continue
            previous = self.tracks.get(detection.track_id)
            velocity_x = 0.0
            velocity_y = 0.0
            if previous is not None and local_ms > previous.last_ms:
                previous_center = self._center(previous.box)
                current_center = self._center(detection.xyxy_norm)
                delta_ms = local_ms - previous.last_ms
                measured_x = (current_center[0] - previous_center[0]) / delta_ms
                measured_y = (current_center[1] - previous_center[1]) / delta_ms
                velocity_x = 0.65 * previous.velocity_x_per_ms + 0.35 * measured_x
                velocity_y = 0.65 * previous.velocity_y_per_ms + 0.35 * measured_y
            self.tracks[detection.track_id] = _Track(
                track_id=detection.track_id,
                class_name=detection.class_name,
                box=detection.xyxy_norm,
                last_ms=local_ms,
                hits=(previous.hits + 1) if previous else 1,
                velocity_x_per_ms=velocity_x,
                velocity_y_per_ms=velocity_y,
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
    raw_motion_score: float = 0.0
    camera_motion_compensated: bool = False
    camera_shift_norm: float = 0.0
    quality_fallback_applied: bool = False
    camera_motion_method: str | None = None
    motion_quality_state: str = "usable"
    motion_probe_score: float | None = None
    motion_probe_raw_score: float | None = None
    source_frame: SourceFrameIdentity | None = None


@dataclass(frozen=True)
class _MotionScoreDetails:
    raw: float
    effective: float
    compensated: bool
    shift_ratio: float
    method: str | None = None
    rejection_reason: str | None = None


def _illumination_robust_absdiff(
    reference: np.ndarray, aligned: np.ndarray
) -> float:
    """Measure residual motion after removing a whole-frame brightness shift."""

    delta = aligned.astype(np.float32) - reference.astype(np.float32)
    delta -= float(np.median(delta))
    return float(np.mean(np.abs(delta)))


def _camera_compensated_motion_details(
    previous_signature: np.ndarray | None,
    signature: np.ndarray,
    *,
    enabled: bool,
    minimum_response: float,
    maximum_shift_ratio: float,
    affine_enabled: bool = False,
    maximum_rotation_degrees: float = 8.0,
    maximum_scale_delta: float = 0.12,
) -> _MotionScoreDetails:
    if previous_signature is None:
        return _MotionScoreDetails(0.0, 0.0, False, 0.0)
    raw_motion = float(cv2.absdiff(signature, previous_signature).mean())
    if not enabled:
        return _MotionScoreDetails(raw_motion, raw_motion, False, 0.0)

    height, width = signature.shape[:2]
    diagonal = max(float(np.hypot(width, height)), 1.0)
    best_score = raw_motion
    best_shift_ratio = 0.0
    best_method: str | None = None
    rejection_reason: str | None = None
    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            previous_signature.astype(np.float32),
            signature.astype(np.float32),
        )
        shift_ratio = float(np.hypot(shift_x, shift_y) / diagonal)
        if (
            np.isfinite(response)
            and np.isfinite(shift_ratio)
            and response >= minimum_response
            and shift_ratio <= maximum_shift_ratio
        ):
            aligned = cv2.warpAffine(
                signature,
                np.asarray(
                    [[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]],
                    dtype=np.float32,
                ),
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            best_score = min(
                raw_motion,
                _illumination_robust_absdiff(previous_signature, aligned),
            )
            best_shift_ratio = shift_ratio
            best_method = "translation"
        else:
            rejection_reason = "translation_estimate_out_of_bounds"
    except cv2.error:
        rejection_reason = "translation_estimate_failed"

    if affine_enabled:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            correlation, warp = cv2.findTransformECC(
                previous_signature.astype(np.float32) / 255.0,
                signature.astype(np.float32) / 255.0,
                warp,
                cv2.MOTION_AFFINE,
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    1e-4,
                ),
            )
            linear = warp[:, :2].astype(np.float64)
            scale_x = float(np.linalg.norm(linear[:, 0]))
            scale_y = float(np.linalg.norm(linear[:, 1]))
            rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
            translation_ratio = float(np.linalg.norm(warp[:, 2]) / diagonal)
            bounded = bool(
                np.isfinite(correlation)
                and correlation >= minimum_response
                and abs(rotation) <= maximum_rotation_degrees
                and abs(scale_x - 1.0) <= maximum_scale_delta
                and abs(scale_y - 1.0) <= maximum_scale_delta
                and translation_ratio <= maximum_shift_ratio
                and np.linalg.det(linear) > 0.0
            )
            if bounded:
                aligned = cv2.warpAffine(
                    signature,
                    warp,
                    (width, height),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT,
                )
                affine_score = _illumination_robust_absdiff(
                    previous_signature, aligned
                )
                if affine_score < best_score:
                    best_score = affine_score
                    best_shift_ratio = translation_ratio
                    best_method = "bounded_affine"
            elif best_method is None:
                rejection_reason = "affine_estimate_out_of_bounds"
        except cv2.error:
            if best_method is None:
                rejection_reason = "affine_estimate_failed"

    return _MotionScoreDetails(
        raw=raw_motion,
        effective=best_score,
        compensated=best_method is not None,
        shift_ratio=best_shift_ratio,
        method=best_method,
        rejection_reason=rejection_reason,
    )


def _camera_compensated_motion_score(
    previous_signature: np.ndarray | None,
    signature: np.ndarray,
    *,
    enabled: bool,
    minimum_response: float,
    maximum_shift_ratio: float,
    affine_enabled: bool = False,
    maximum_rotation_degrees: float = 8.0,
    maximum_scale_delta: float = 0.12,
) -> tuple[float, float, bool, float]:
    """Return raw and camera-compensated motion without changing legacy callers.

    Phase correlation estimates the dominant whole-frame translation.  The
    compensated value is admitted only when the estimate is reliable and
    bounded; otherwise the established absolute-difference score is retained.
    """

    details = _camera_compensated_motion_details(
        previous_signature,
        signature,
        enabled=enabled,
        minimum_response=minimum_response,
        maximum_shift_ratio=maximum_shift_ratio,
        affine_enabled=affine_enabled,
        maximum_rotation_degrees=maximum_rotation_degrees,
        maximum_scale_delta=maximum_scale_delta,
    )
    return (
        details.raw,
        details.effective,
        details.compensated,
        details.shift_ratio,
    )


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


@dataclass
class _DecodedUnitEnd:
    pass


@dataclass
class _DecodedUnitError:
    message: str


def _read_checkpoint(
    path: Path, output_path: Path | None = None, *, duplicate_policy: dict | None = None,
    prediction_policy: dict | None = None,
    source_frame_contract: str | None = None,
) -> set[int]:
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_frame_contract") != source_frame_contract:
            raise RuntimeError("Source frame identity contract changed; use a new scan directory")
        if payload.get("prediction_policy") != prediction_policy:
            raise RuntimeError("Detection prediction policy changed; use a new scan directory")
        completed = {int(item) for item in payload.get("completed_chunks", [])}
        if not completed:
            return set()
        if output_path is None or not output_path.is_file():
            return set()
        if payload.get("duplicate_suppression_policy") != duplicate_policy:
            # Reject before truncating the uncommitted tail or deleting a ledger.
            raise RuntimeError(
                "Detection duplicate suppression policy changed; use a new scan directory"
            )
        expected_size = payload.get("output_size_bytes")
        if expected_size is not None:
            expected = int(expected_size)
            actual = output_path.stat().st_size
            if actual < expected:
                return set()
            if actual > expected:
                # Frames beyond the last committed ChunkEnd are uncheckpointed.
                # Roll only that partial tail back instead of discarding every
                # previously durable work unit.
                with output_path.open("r+b") as handle:
                    handle.truncate(expected)
        if output_path.stat().st_size <= 0:
            return set()
        return completed
    except (json.JSONDecodeError, OSError, ValueError):
        return set()


def _write_checkpoint(
    path: Path, completed: set[int], output_path: Path, *, duplicate_policy: dict | None = None,
    prediction_policy: dict | None = None,
    source_frame_contract: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-detection-checkpoint/2",
                "completed_chunks": sorted(completed),
                "output_path": str(output_path),
                "output_size_bytes": output_path.stat().st_size if output_path.is_file() else 0,
                "duplicate_suppression_policy": duplicate_policy,
                "prediction_policy": prediction_policy,
                "source_frame_contract": source_frame_contract,
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
    phase: str = "fine",
    alignment_transform: AlignmentTransform | None = None,
    decode_worker_override: int | None = None,
) -> None:
    perf = config["performance"]
    decoder_admission = CudaDecodeAdmission.from_config(config)
    component_timings = getattr(output_queue, "component_timings", StageTimings())
    chunk_ms = float(perf.get(f"{phase}_chunk_seconds", perf["chunk_seconds"])) * 1000.0
    spans = windows if windows is not None else [(0.0, info.duration_ms)]
    persistent_sessions: list[PhysicalSegmentDecodeSession] = []
    persistent_segment_decode = bool(
        phase == "fine"
        and windows is not None
        and info.segments
        and perf.get("fine_persistent_segment_decode", False)
    )
    if persistent_segment_decode:
        configured_session_gap_seconds = perf.get(
            "fine_persistent_session_max_gap_seconds"
        )
        persistent_sessions = plan_physical_segment_decode_sessions(
            info,
            spans,
            max_gap_ms=(
                float(configured_session_gap_seconds) * 1000.0
                if configured_session_gap_seconds is not None
                else None
            ),
        )
        work_units = [
            (session.virtual_start_ms, session.virtual_end_ms)
            for session in persistent_sessions
        ]
    elif (
        windows is None
        and info.segments
        and (
            perf.get("synchronized_segment_waves")
            or (keyframes_only and int(perf.get("motion_probe_segment_workers", 1)) > 1)
        )
    ):
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
    previous_probe_signature: np.ndarray | None = None
    next_shared_probe_ms: float | None = None
    decode_fps = max(sample_fps, motion_probe_fps)
    sample_period_ms = 1000.0 / max(sample_fps, 1e-9)
    transform = alignment_transform or AlignmentTransform(
        view_id=view.view_id,
        reference_view_id=view.view_id,
        state="aligned",
    )
    if transform.scale <= 0.0:
        raise ValueError(f"{view.view_id}: alignment scale must be positive")
    global_decode_period_ms = 1000.0 / max(decode_fps, 1e-9)
    local_decode_period_ms = global_decode_period_ms / transform.scale
    local_grid_origin_ms = transform.to_local(0.0)
    activity_path = output_queue.activity_path if hasattr(output_queue, "activity_path") else None
    session_decode_receipts: dict[int, dict[str, Any]] = {}

    def source_unit_paths(chunk_index: int | None) -> list[str]:
        if chunk_index is None or not 0 <= chunk_index < len(work_units):
            return []
        if persistent_sessions:
            return [str(persistent_sessions[chunk_index].path)]
        unit_start, unit_end = work_units[chunk_index]
        if not info.segments:
            return [str(view.video)] if view.video is not None else []
        return [
            str(segment.path)
            for segment in info.segments
            if min(unit_end, segment.virtual_end_ms)
            > max(unit_start, segment.virtual_start_ms)
        ]

    def activity(event: str, chunk_index: int | None = None) -> None:
        if activity_path is None:
            return
        paths = source_unit_paths(chunk_index)
        unit_window = (
            work_units[chunk_index]
            if chunk_index is not None and 0 <= chunk_index < len(work_units)
            else None
        )
        payload = {
            "timestamp": time.time(),
            "event": event,
            "view_id": view.view_id,
            "role": view.role.value,
            "decode_backend": decode_backend,
            "chunk_index": chunk_index,
            "source_unit_start_ms": unit_window[0] if unit_window else None,
            "source_unit_end_ms": unit_window[1] if unit_window else None,
            "segment_path": paths[0] if len(paths) == 1 else None,
            "segment_paths": paths,
            "decoder_session_mode": (
                "persistent_physical_segment"
                if persistent_sessions
                else "window_or_chunk"
            ),
        }
        decoder_receipt = session_decode_receipts.get(chunk_index)
        if decoder_receipt:
            payload["decoder_receipt"] = decoder_receipt
            payload["actual_decode_backend"] = decoder_receipt.get("actual_decoder_backend")
        if persistent_sessions and chunk_index is not None:
            session = persistent_sessions[chunk_index]
            decoder_receipt = session_decode_receipts.get(chunk_index)
            if decoder_receipt and decoder_receipt.get(
                "aligned_session_start_virtual_ms"
            ) is not None:
                first_global_ms = transform.to_global(
                    float(decoder_receipt["aligned_session_start_virtual_ms"])
                )
                phase_ms = first_global_ms % global_decode_period_ms
                decoder_receipt.update(
                    {
                        "first_sample_global_ms": first_global_ms,
                        "global_sampling_phase_ms": phase_ms,
                        "global_sampling_phase_error_ms": min(
                            phase_ms, global_decode_period_ms - phase_ms
                        ),
                    }
                )
            payload.update(
                {
                    "target_window_count": len(session.target_virtual_windows),
                    "target_windows_ms": [list(item) for item in session.target_virtual_windows],
                    "selected_duration_ms": session.selected_duration_ms,
                    "decode_session_span_ms": (
                        session.virtual_end_ms - session.virtual_start_ms
                    ),
                    "avoided_physical_reopens": max(
                        0, len(session.target_virtual_windows) - 1
                    ),
                    "decoder_receipt": decoder_receipt,
                }
            )
        with output_queue.activity_lock:
            with activity_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def raw_decoded_frames(
        start_ms: float, end_ms: float, chunk_index: int | None = None
    ) -> Iterable[tuple[int, float, np.ndarray]]:
        receipt = session_decode_receipts.setdefault(chunk_index, {})
        if persistent_sessions:
            if chunk_index is None:
                raise ValueError("persistent decode requires a source unit index")
            receipt.update(
                {
                    "sampling_grid_mode": "aligned_global_timeline",
                    "global_sampling_period_ms": global_decode_period_ms,
                    "local_sampling_period_ms": local_decode_period_ms,
                    "local_grid_origin_ms": local_grid_origin_ms,
                    "alignment_scale": transform.scale,
                    "alignment_offset_ms": transform.offset_ms,
                    "visual_correction_ms": transform.visual_correction_ms,
                    "tracker_state_policy": "per_view_pass_monotonic_timeline",
                    "motion_reference_policy": "previous_emitted_frame",
                }
            )
            return iter_physical_segment_session_frames(
                info,
                persistent_sessions[chunk_index],
                decode_fps,
                max_width,
                "cuda" if decode_backend == "cuda" else None,
                int(perf.get("cpu_decode_threads", 2)),
                bool(perf.get("ffmpeg_cuda_scale", False))
                and decode_backend == "cuda",
                receipt,
                local_grid_origin_ms,
                local_decode_period_ms,
                decoder_admission=decoder_admission,
            )
        return iter_view_sampled_frames(
            view,
            info,
            start_ms,
            end_ms,
            decode_fps,
            max_width,
            "cuda" if decode_backend == "cuda" else None,
            keyframes_only,
            int(perf.get("cpu_decode_threads", 2)),
            str(perf.get("motion_probe_sparse_strategy", "indexed_seek")),
            bool(perf.get("ffmpeg_cuda_scale", False)) and decode_backend == "cuda",
            decoder_admission=decoder_admission,
            receipt=receipt,
        )

    def iter_decoded_frames(start_ms, end_ms, chunk_index=None):
        return component_timings.frames(raw_decoded_frames(start_ms, end_ms, chunk_index))

    def decoded_frames(
        start_ms: float, end_ms: float, chunk_index: int | None = None
    ) -> list[tuple[int, float, np.ndarray]]:
        return list(iter_decoded_frames(start_ms, end_ms, chunk_index))

    def emit_frames(frames: Iterable[tuple[int, float, np.ndarray]], start_ms: float) -> None:
        nonlocal previous_gray, previous_signature
        nonlocal previous_probe_signature, next_shared_probe_ms
        next_yolo_ms = start_ms
        shared_motion_probe = bool(
            phase == "coarse"
            and perf.get("coarse_shared_motion_probe_enabled", False)
        )
        probe_period_ms = 1000.0 / max(motion_probe_fps, 1e-9)
        if shared_motion_probe and next_shared_probe_ms is None:
            next_shared_probe_ms = start_ms
        for decoded in frames:
            frame_index, local_ms, frame = decoded
            preparation_started = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            signature = cv2.resize(gray, motion_signature_size, interpolation=cv2.INTER_AREA)
            compensation_enabled = bool(
                phase == "motion_probe"
                and perf.get("motion_probe_camera_motion_compensation", False)
            )
            details = _camera_compensated_motion_details(
                previous_signature,
                signature,
                enabled=compensation_enabled,
                minimum_response=float(
                    perf.get("motion_probe_camera_motion_min_response", 0.15)
                ),
                maximum_shift_ratio=float(
                    perf.get("motion_probe_camera_motion_max_shift_ratio", 0.35)
                ),
                affine_enabled=bool(
                    perf.get("motion_probe_affine_compensation_enabled", False)
                ),
                maximum_rotation_degrees=float(
                    perf.get("motion_probe_max_rotation_degrees", 8.0)
                ),
                maximum_scale_delta=float(
                    perf.get("motion_probe_max_scale_delta", 0.12)
                ),
            )
            raw_motion = details.raw
            motion = details.effective
            compensated = details.compensated
            shift_ratio = details.shift_ratio
            compensation_method = details.method
            blur_score = float(cv2.Laplacian(signature, cv2.CV_64F).var())
            intensity_mean = float(signature.mean())
            intensity_std = float(signature.std())
            unusable = bool(
                blur_score
                < float(
                    perf.get("motion_probe_first_person_min_laplacian_variance", 8.0)
                )
                or intensity_mean
                < float(perf.get("motion_probe_first_person_min_intensity", 10.0))
                or intensity_std
                < float(perf.get("motion_probe_first_person_min_intensity_std", 4.0))
            )
            motion_quality_state = "degraded" if unusable else "usable"
            quality_fallback_applied = False
            if (
                phase == "motion_probe"
                and view.role == ViewRole.FIRST_PERSON
                and perf.get("motion_probe_first_person_quality_fallback", False)
                and previous_signature is not None
                and unusable
            ):
                fallback_motion = raw_motion * float(
                    perf.get("motion_probe_first_person_raw_motion_weight", 0.25)
                )
                quality_fallback_applied = fallback_motion > motion
                motion = max(motion, fallback_motion)

            probe_motion_score: float | None = None
            probe_raw_motion_score: float | None = None
            if (
                shared_motion_probe
                and next_shared_probe_ms is not None
                and local_ms + 0.5 >= next_shared_probe_ms
            ):
                probe_details = _camera_compensated_motion_details(
                    previous_probe_signature,
                    signature,
                    enabled=bool(
                        perf.get("motion_probe_camera_motion_compensation", False)
                    ),
                    minimum_response=float(
                        perf.get("motion_probe_camera_motion_min_response", 0.15)
                    ),
                    maximum_shift_ratio=float(
                        perf.get("motion_probe_camera_motion_max_shift_ratio", 0.35)
                    ),
                    affine_enabled=bool(
                        perf.get("motion_probe_affine_compensation_enabled", False)
                    ),
                    maximum_rotation_degrees=float(
                        perf.get("motion_probe_max_rotation_degrees", 8.0)
                    ),
                    maximum_scale_delta=float(
                        perf.get("motion_probe_max_scale_delta", 0.12)
                    ),
                )
                probe_motion_score = probe_details.effective
                probe_raw_motion_score = probe_details.raw
                if (
                    view.role == ViewRole.FIRST_PERSON
                    and perf.get("motion_probe_first_person_quality_fallback", False)
                    and previous_probe_signature is not None
                    and unusable
                ):
                    fallback_motion = probe_details.raw * float(
                        perf.get("motion_probe_first_person_raw_motion_weight", 0.25)
                    )
                    quality_fallback_applied = (
                        quality_fallback_applied
                        or fallback_motion > probe_motion_score
                    )
                    probe_motion_score = max(probe_motion_score, fallback_motion)
                previous_probe_signature = signature
                compensated = probe_details.compensated
                shift_ratio = probe_details.shift_ratio
                compensation_method = probe_details.method
                while next_shared_probe_ms <= local_ms + 0.5:
                    next_shared_probe_ms += probe_period_ms

            previous_signature = signature
            if not persistent_sessions and local_ms + 0.5 < next_yolo_ms:
                previous_gray = gray
                component_timings.add("frame_preparation_seconds", time.perf_counter()-preparation_started)
                continue
            component_timings.add("frame_preparation_seconds", time.perf_counter()-preparation_started)
            emit_started = time.perf_counter()
            output_queue.put(
                FramePacket(
                    view=view,
                    frame_index=frame_index,
                    local_ms=local_ms,
                    frame=frame,
                    gray=gray,
                    previous_gray=previous_gray,
                    motion_score=motion,
                    raw_motion_score=raw_motion,
                    camera_motion_compensated=compensated,
                    camera_shift_norm=shift_ratio,
                    quality_fallback_applied=quality_fallback_applied,
                    camera_motion_method=compensation_method,
                    motion_quality_state=motion_quality_state,
                    motion_probe_score=probe_motion_score,
                    motion_probe_raw_score=probe_raw_motion_score,
                    source_frame=getattr(decoded, "source_frame", None),
                )
            )
            component_timings.add("producer_emit_seconds", time.perf_counter()-emit_started)
            previous_gray = gray
            if not persistent_sessions:
                while next_yolo_ms <= local_ms + 0.5:
                    next_yolo_ms += sample_period_ms

    def finish_unit(chunk_index: int, total_chunks: int) -> None:
        output_queue.put(
            ChunkEnd(
                view_id=view.view_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
        )
        activity("source_unit_completed", chunk_index)

    parallel_probe_workers = max(
        1,
        int(perf.get("motion_probe_segment_workers", 1)),
    )
    parallel_probe = (
        keyframes_only
        and bool(info.segments)
        and windows is None
        and wave_barrier is None
        and parallel_probe_workers > 1
    )
    configured_decode_workers = max(
        1,
        int(
            perf.get(
                "fine_first_person_decode_workers"
                if view.role == ViewRole.FIRST_PERSON
                else "fine_third_person_decode_workers",
                1,
            )
        ),
    )
    ordered_decode_workers = max(
        1,
        int(
            configured_decode_workers
            if decode_worker_override is None
            else decode_worker_override
        ),
    )
    ordered_decode = (
        phase == "fine"
        and windows is not None
        and wave_barrier is None
        and ordered_decode_workers > 1
        and bool(work_units)
    )
    try:
        if parallel_probe:
            futures: dict[int, Any] = {}
            with ThreadPoolExecutor(
                max_workers=min(parallel_probe_workers, len(work_units)),
                thread_name_prefix=f"probe-segment-{view.view_id}",
            ) as executor:
                for chunk_index, (start_ms, end_ms) in enumerate(work_units):
                    activity("source_unit_started", chunk_index)
                    if chunk_index not in completed_chunks:
                        futures[chunk_index] = executor.submit(
                            decoded_frames, start_ms, end_ms, chunk_index
                        )
                for chunk_index, (start_ms, _end_ms) in enumerate(work_units):
                    if chunk_index in completed_chunks:
                        activity("source_unit_reused", chunk_index)
                        output_queue.put(
                            ChunkEnd(
                                view_id=view.view_id,
                                chunk_index=chunk_index,
                                total_chunks=len(work_units),
                            )
                        )
                        continue
                    emit_frames(futures[chunk_index].result(), start_ms)
                    finish_unit(chunk_index, len(work_units))
            return

        if ordered_decode:
            prefetch_frames = max(
                1, int(perf.get("fine_decode_prefetch_frames", 12))
            )
            stop_event = threading.Event()
            unit_queues: dict[int, queue.Queue[Any]] = {}

            def put_until_stopped(target: queue.Queue[Any], item: Any) -> bool:
                while not stop_event.is_set():
                    try:
                        target.put(item, timeout=0.1)
                        return True
                    except queue.Full:
                        continue
                return False

            def decode_unit(
                chunk_index: int,
                start_ms: float,
                end_ms: float,
                target: queue.Queue[Any],
            ) -> None:
                activity("source_unit_started", chunk_index)
                try:
                    with closing(iter_decoded_frames(start_ms, end_ms, chunk_index)) as frames:
                        for decoded in frames:
                            if not put_until_stopped(target, decoded):
                                return
                except Exception as exc:
                    put_until_stopped(
                        target,
                        _DecodedUnitError(f"{type(exc).__name__}: {exc}"),
                    )
                finally:
                    put_until_stopped(target, _DecodedUnitEnd())

            def ordered_frames(target: queue.Queue[Any]):
                while True:
                    item = target.get()
                    if isinstance(item, _DecodedUnitEnd):
                        return
                    if isinstance(item, _DecodedUnitError):
                        raise RuntimeError(item.message)
                    yield item

            executor = ThreadPoolExecutor(
                max_workers=min(ordered_decode_workers, len(work_units)),
                thread_name_prefix=f"fine-prefetch-{view.view_id}",
            )
            try:
                futures: dict[int, Any] = {}
                for chunk_index, (start_ms, end_ms) in enumerate(work_units):
                    if chunk_index in completed_chunks:
                        continue
                    target: queue.Queue[Any] = queue.Queue(maxsize=prefetch_frames)
                    unit_queues[chunk_index] = target
                    futures[chunk_index] = executor.submit(
                        decode_unit,
                        chunk_index,
                        start_ms,
                        end_ms,
                        target,
                    )
                for chunk_index, (start_ms, _end_ms) in enumerate(work_units):
                    if chunk_index in completed_chunks:
                        activity("source_unit_reused", chunk_index)
                        finish_unit(chunk_index, len(work_units))
                        continue
                    emit_frames(ordered_frames(unit_queues[chunk_index]), start_ms)
                    futures[chunk_index].result()
                    finish_unit(chunk_index, len(work_units))
            except Exception:
                stop_event.set()
                raise
            finally:
                stop_event.set()
                executor.shutdown(wait=True, cancel_futures=True)
            return

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
            with closing(iter_decoded_frames(start_ms, end_ms, chunk_index)) as frames:
                emit_frames(frames, start_ms)
            finish_unit(chunk_index, len(work_units))
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


def _roi_appearance_signature(
    frame: np.ndarray,
    box: Sequence[float],
) -> tuple[float, ...]:
    """Return a compact color signature for contradiction-only association."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(0, int(x1 * width)), max(0, int(y1 * height))
    right = min(width, int(math.ceil(x2 * width)))
    bottom = min(height, int(math.ceil(y2 * height)))
    if right - left < 2 or bottom - top < 2:
        return ()
    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    means, deviations = cv2.meanStdDev(hsv)
    scales = (179.0, 255.0, 255.0)
    return tuple(
        round(float(value) / scales[index], 6)
        for values in (means, deviations)
        for index, value in enumerate(values.reshape(-1))
    )


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


def _metadata_dynamic(metadata: dict[str, Any]) -> bool | None:
    containers = [metadata]
    for key in ("args", "export", "engine"):
        nested = metadata.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        value = container.get("dynamic")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
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


def _engine_requires_exact_batch(path: Path) -> bool:
    if path.suffix.lower() != ".engine" or not path.is_file():
        return False
    _plan, metadata, _container = _tensorrt_plan_and_metadata(path)
    return (
        _metadata_dynamic(metadata) is False and _metadata_batch(metadata) is not None
    )


def _validate_pinned_asset(
    name: str, path: Path, expected_sha256: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} model artifact is missing: {path}")
    expected = str(expected_sha256 or "").strip().lower()
    if not expected:
        raise RuntimeError(f"{name} model SHA-256 is not configured")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{name} model hash mismatch: expected={expected} actual={actual}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_open_vocabulary_runtime(config: dict[str, Any]) -> dict[str, Any]:
    settings = (config.get("models") or {}).get("open_vocabulary_key_frame") or {}
    if not settings.get("enabled"):
        return {"enabled": False, "status": "disabled"}
    from ultralytics import YOLOWorld  # noqa: F401

    model = _validate_pinned_asset(
        "YOLO-World",
        Path(str(settings.get("model_path") or "")).resolve(),
        str(settings.get("model_sha256") or ""),
    )
    clip = _validate_pinned_asset(
        "CLIP",
        Path(str(settings.get("clip_model_path") or "")).resolve(),
        str(settings.get("clip_model_sha256") or ""),
    )
    fallback = dict(settings.get("grounding_dino_fallback") or {})
    grounding_dino: dict[str, Any] = {"enabled": False, "status": "disabled"}
    if fallback.get("enabled"):
        from transformers import (  # noqa: F401
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        root = Path(str(fallback.get("model_path") or "")).resolve()
        for required_name in ("config.json", "preprocessor_config.json"):
            required = root / required_name
            if not required.is_file():
                raise FileNotFoundError(
                    f"Grounding DINO runtime file is missing: {required}"
                )
        grounding_dino = {
            "enabled": True,
            "status": "validated",
            "model_revision": str(fallback.get("model_revision") or ""),
            **_validate_pinned_asset(
                "Grounding DINO",
                root / "model.safetensors",
                str(fallback.get("model_sha256") or ""),
            ),
        }
    return {
        "enabled": True,
        "status": "validated",
        "scope": "accepted_final_key_frames_and_bounded_temporal_rescue",
        "full_timeline_inference": False,
        "yolo_world": model,
        "clip": clip,
        "grounding_dino": grounding_dino,
    }


def _normalized_class_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        try:
            ordered = [
                value
                for _, value in sorted(
                    names.items(), key=lambda item: int(item[0])
                )
            ]
        except (TypeError, ValueError):
            ordered = [
                value
                for _, value in sorted(
                    names.items(), key=lambda item: str(item[0])
                )
            ]
    elif isinstance(names, (list, tuple)):
        ordered = list(names)
    else:
        return []
    return [str(value).replace("-", "_") for value in ordered]


def _validate_class_names(names: list[str], expected: int, source: Path) -> None:
    if len(names) != expected:
        raise ValueError(f"{source} 类别数为 {len(names)}，期望 {expected}")
    if len(set(names)) != len(names):
        raise ValueError(f"{source} 的规范化类别表包含重复类别")


def _role_class_names(config: dict, role: ViewRole) -> list[str] | None:
    """Allow explicit append-only ontology growth without coupling role upgrades."""
    configured = config["models"].get("class_names_by_role")
    if configured is None:
        return None
    if not isinstance(configured, dict) or set(configured) != {r.value for r in ViewRole}:
        raise ValueError("class_names_by_role must explicitly define both view roles")
    normalized = {}
    for name, values in configured.items():
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("Role class names must be nonempty ordered string lists")
        normalized[name] = [v.replace("-", "_") for v in values]
        if len(set(normalized[name])) != len(values):
            raise ValueError("Role ontology contains duplicate normalized class names")
    first, third = (normalized[r.value] for r in ViewRole)
    common = min(len(first), len(third))
    if first[:common] != third[:common]:
        raise ValueError("Role ontologies must preserve common class IDs; only append-only growth is allowed")
    return normalized[role.value]


def _validate_role_class_names(names: list[str], config: dict, role: ViewRole, source: Path) -> None:
    configured = _role_class_names(config, role)
    expected = len(configured) if configured is not None else int(config["models"]["expected_class_count"])
    _validate_class_names(names, expected, source)
    if configured is not None and names != configured:
        raise ValueError(f"{role.value} model classes differ from the explicit role ontology")


def validate_models(config: dict[str, Any]) -> dict[str, Any]:
    mode = str(config["performance"].get("tensor_rt", "auto")).lower()
    report: dict[str, Any] = {}
    class_sets: dict[str, list[str]] = {}
    for role in ViewRole:
        path = Path(config["models"][role.value])
        if not path.is_file():
            if mode not in {"required", "true"}:
                raise FileNotFoundError(f"{role.value} 模型不存在: {path}")
            report[role.value] = {
                "path": str(path),
                "source_model_available": False,
                "validation_source": "tensorrt_engine_metadata",
            }
            continue
        from ultralytics import YOLO

        model = YOLO(str(path))
        names = _normalized_class_names(model.names)
        _validate_role_class_names(names, config, role, path)
        class_sets[role.value] = names
        report[role.value] = {
            "path": str(path),
            "source_model_available": True,
            "validation_source": "pytorch_model",
            "class_count": len(names),
            "classes": names,
        }
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
            engine_names = _normalized_class_names(metadata.get("names"))
            if not engine_names:
                raise RuntimeError(
                    f"TensorRT engine has no embedded class metadata: {engine_path}"
                )
            _validate_role_class_names(engine_names, config, role, engine_path)
            source_names = class_sets.get(role.value)
            if source_names is not None and source_names != engine_names:
                raise ValueError(
                    f"{role.value} TensorRT 引擎类别表与源模型不一致"
                )
            class_sets[role.value] = engine_names
            report[role.value].update(
                {
                    "validation_source": (
                        "pytorch_model_and_tensorrt_engine_metadata"
                        if source_names is not None
                        else "tensorrt_engine_metadata"
                    ),
                    "class_count": len(engine_names),
                    "classes": engine_names,
                }
            )
            build_batch = _metadata_batch(metadata) or _profile_batch(engine)
            runtime["roles"][role.value] = {
                "backend": "TensorRT",
                "engine": str(engine_path),
                "bytes": engine_path.stat().st_size,
                "sha256": _sha256_file(engine_path),
                "deserialized": True,
                "container": container,
                "build_batch": build_batch,
                "class_count": len(engine_names),
                "classes": engine_names,
            }
    else:
        for role in ViewRole:
            selected = _select_model_path(role, config)
            runtime["roles"][role.value] = {
                "backend": "TensorRT" if selected.suffix.lower() == ".engine" else "PyTorch",
                "path": str(selected),
            }
    if (_role_class_names(config, ViewRole.FIRST_PERSON) is None
            and class_sets[ViewRole.FIRST_PERSON.value] != class_sets[ViewRole.THIRD_PERSON.value]):
        raise ValueError("第一/第三人称模型的规范化类别表不一致")
    runtime["temporal_participant_segmentation"] = (
        validate_temporal_segmentation_runtime(config)
    )
    runtime["open_vocabulary_key_frame"] = _validate_open_vocabulary_runtime(
        config
    )
    runtime["liquid_semantic_sidecar"] = validate_liquid_semantic_runtime(config)
    runtime["selective_key_material_verification"] = (
        validate_selective_key_material_verification(
            dict(
                (config.get("key_materials") or {}).get("selective_verification") or {}
            )
        )
    )
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
        appearance_enabled: bool = False,
    ):
        from ultralytics import YOLO

        self.role = role
        self.config = config
        self.model_path = _select_model_path(role, config)
        branch = prediction_branches(config).get(role)
        self.prediction_policy = (
            prediction_contract(branch, self.model_path, config,
                                int(image_size or config["performance"]["image_size"]))
            if branch is not None else None
        )
        self.prediction_end2end = None if branch is None else branch == "one2one"
        self.last_prediction_end2end = None
        self.requested_batch_size = int(
            batch_size or config["performance"]["batch_size"]
        )
        self.engine_build_batch = _engine_build_batch(self.model_path)
        self.engine_requires_exact_batch = _engine_requires_exact_batch(self.model_path)
        self.batch_size = min(
            self.requested_batch_size,
            self.engine_build_batch or self.requested_batch_size,
        )
        self.initial_batch_size = self.batch_size
        self.batch_contractions: list[dict[str, int]] = []
        self.last_inference_batch_sizes: list[int] = []
        self.last_engine_batch_sizes: list[int] = []
        self.exact_batch_padding_frames = 0
        self.image_size = int(image_size or config["performance"]["image_size"])
        self.appearance_enabled = bool(appearance_enabled)
        self.names = {}
        self.model = None
        self._prepared = False
        self._closed = False
        self.initialization_phase = "model_wrapper"
        self.initialization_failure_phase = None
        try:
            self.model = YOLO(str(self.model_path))
            if branch is not None:
                head = self.model.model.model[-1]
                if any(getattr(head, name, None) is None for name in (
                    "cv2", "cv3", "one2one_cv2", "one2one_cv3"
                )):
                    raise ValueError("Selected model does not contain both candidate prediction branches")
                self.model.model.end2end = self.prediction_end2end
        except BaseException:
            self.close()
            raise

    def _prediction_options(self) -> dict[str, Any]:
        perf, model_cfg = self.config["performance"], self.config["models"]
        options = {
            "imgsz": self.image_size,
            "conf": float(model_cfg["confidence"]),
            "iou": float(model_cfg["iou"]),
            "max_det": int(model_cfg["max_detections"]),
            "device": perf["device"],
            "half": bool(perf["half"]),
            "verbose": False,
        }
        if self.prediction_end2end is not None:
            options["end2end"] = self.prediction_end2end
        return options

    def prepare(self) -> None:
        """Initialize and warm the persistent predictor in its owning thread.

        Ultralytics Model.names constructs a temporary predictor for engines;
        using it before Model.predict would allocate a second context later.
        Mirror predict's setup and first-call warmup without emitting evidence.
        """
        if self._closed:
            raise RuntimeError("Inference scanner is already closed")
        if self._prepared:
            return
        import traceback

        predictor = None
        try:
            self.initialization_phase = "predictor_setup"
            import torch
            from ultralytics.utils.checks import check_imgsz

            options = {**self.model.overrides, "conf": .25, "batch": 1,
                       "save": False, "mode": "predict", "rect": True,
                       **self._prediction_options()}
            predictor = self.model._smart_load("predictor")(
                overrides=options, _callbacks=self.model.callbacks)
            self.model.predictor = predictor
            predictor.setup_model(model=self.model.model, verbose=False)
            self.initialization_phase = "class_names"
            self.names = {int(key): str(value).replace("-", "_")
                          for key, value in predictor.model.names.items()}
            _validate_role_class_names(_normalized_class_names(self.names), self.config,
                                       self.role, self.model_path)
            self.initialization_phase = "prediction_branch"
            if self.prediction_end2end is not None:
                observed = getattr(predictor.model, "end2end", None)
                if type(observed) is not bool or observed != self.prediction_end2end:
                    raise RuntimeError("Prediction backend did not honor the configured branch")
            self.initialization_phase = "image_size"
            # setup_model replaces args.imgsz with static export metadata, but
            # the first predict call supplies the requested size again. Refuse
            # a mismatch before reporting ready instead of warming one shape
            # and failing the first real frame with another shape.
            size = check_imgsz(self.image_size, stride=predictor.model.stride, min_dim=2)
            if (predictor.model.format == "engine"
                    and not getattr(predictor.model, "dynamic", False)
                    and hasattr(predictor.model, "imgsz")):
                exported = list(predictor.model.imgsz)
                if list(size) != exported:
                    raise ValueError(
                        f"Requested inference image size {list(size)} does not match static engine export {exported}"
                    )
            self.initialization_phase = "warmup"
            # Match BasePredictor.setup_source/stream_inference. No source
            # frame, detector result, batch counter or evidence is produced.
            warmup_batch = (1 if predictor.model.format in {"pt", "triton"}
                            else self.engine_build_batch if self.engine_requires_exact_batch
                            else self.batch_size)
            with torch.inference_mode():
                predictor.model.warmup(imgsz=(warmup_batch, predictor.model.channels, *size))
            if predictor.device.type == "cuda":
                torch.cuda.synchronize(predictor.device)
            predictor.done_warmup = True
            self._prepared = True
            self.initialization_phase = "ready"
        except BaseException as exc:
            self.initialization_failure_phase = self.initialization_phase
            # A Future retains the exception for its waiter. Release backend
            # frame locals before close() collects and empties the CUDA cache.
            traceback.clear_frames(exc.__traceback__)
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Scanner cleanup failed: {type(cleanup_error).__name__}")
            predictor = None
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._prepared = False
        if self.model is not None:
            predictor = getattr(self.model, "predictor", None)
            if predictor is not None:
                predictor.model = None
                self.model.predictor = None
            self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def infer(self, packets: Sequence[FramePacket]) -> list[list[BoxEvidence]]:
        from .runtime_control import check_cancelled
        check_cancelled()
        if not packets:
            return []
        self.prepare()
        # A short queue flush is not a memory-pressure signal. Keep the engine
        # capacity unchanged unless inference actually raises CUDA OOM.
        batch_size = min(self.batch_size, len(packets))
        if not hasattr(self, "component_timings"):
            self.component_timings = StageTimings()
        while True:
            try:
                results: list[list[BoxEvidence]] = []
                actual_batch_sizes: list[int] = []
                engine_batch_sizes: list[int] = []
                for start in range(0, len(packets), batch_size):
                    sub_batch = packets[start : start + batch_size]
                    actual_batch_sizes.append(len(sub_batch))
                    execution_batch = list(sub_batch)
                    if (
                        getattr(self, "engine_requires_exact_batch", False)
                        and self.engine_build_batch
                        and len(execution_batch) < self.engine_build_batch
                    ):
                        padding = self.engine_build_batch - len(execution_batch)
                        execution_batch.extend([execution_batch[-1]] * padding)
                        self.exact_batch_padding_frames = (
                            int(getattr(self, "exact_batch_padding_frames", 0))
                            + padding
                        )
                    engine_batch_sizes.append(len(execution_batch))
                    expected_end2end = getattr(self, "prediction_end2end", None)
                    predictions = self.model.predict(
                        source=[packet.frame for packet in execution_batch],
                        **self._prediction_options(),
                    )
                    prediction_timings(predictions, self.component_timings)
                    if expected_end2end is not None:
                        observed = getattr(self.model.predictor.model, "end2end", None)
                        if type(observed) is not bool or observed != expected_end2end:
                            raise RuntimeError("Prediction backend did not honor the configured branch")
                        self.last_prediction_end2end = observed
                    if len(predictions) < len(sub_batch):
                        raise RuntimeError(
                            "TensorRT inference returned fewer predictions than source frames"
                        )
                    for packet, prediction in zip(
                        sub_batch, predictions[: len(sub_batch)], strict=True
                    ):
                        boxes: list[BoxEvidence] = []
                        if prediction.boxes is not None:
                            xyxy = np.clip(prediction.boxes.xyxyn.detach().cpu().numpy(), 0.0, 1.0)
                            confidences = prediction.boxes.conf.detach().cpu().numpy()
                            classes = (
                                prediction.boxes.cls.detach().cpu().numpy().astype(int)
                            )
                            for coords, confidence, class_id in zip(
                                xyxy, confidences, classes, strict=True
                            ):
                                coords_tuple = tuple(
                                    float(item) for item in coords
                                )
                                boxes.append(
                                    BoxEvidence(
                                        class_id=int(class_id),
                                        class_name=self.names[int(class_id)],
                                        confidence=float(confidence),
                                        xyxy_norm=coords_tuple,
                                        roi_motion=_roi_motion(
                                            packet.previous_gray,
                                            packet.gray,
                                            coords_tuple,
                                        ),
                                        appearance_signature=(
                                            _roi_appearance_signature(
                                                packet.frame, coords_tuple
                                            )
                                            if self.appearance_enabled
                                            else ()
                                        ),
                                    )
                                )
                        results.append(boxes)
                self.last_inference_batch_sizes = actual_batch_sizes
                self.last_engine_batch_sizes = engine_batch_sizes
                return results
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size <= 1:
                    raise
                previous_batch_size = batch_size
                batch_size = max(1, batch_size // 2)
                self.batch_size = min(self.batch_size, batch_size)
                self.batch_contractions.append(
                    {
                        "from_batch_size": previous_batch_size,
                        "to_batch_size": batch_size,
                    }
                )
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
    scanner_id: str | None = None,
) -> dict[str, Path]:
    policies = duplicate_suppression_policies(config)
    branches = prediction_branches(config)
    effective_image_size = int(image_size if image_size is not None else config["performance"]["image_size"])
    if phase == "motion_probe" and not config["performance"].get("motion_probe_run_yolo", False):
        policies = {}
        branches = {}
    prediction_policies = {
        role: prediction_contract(branch, _select_model_path(role, config), config, effective_image_size)
        for role, branch in branches.items() if any(view.role == role for view in views)
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {view.view_id: work_dir / f"{view.view_id}.detections.jsonl" for view in views}
    checkpoint_paths = {view.view_id: work_dir / f"{view.view_id}.checkpoint.json" for view in views}
    completed = {
        view.view_id: _read_checkpoint(
            checkpoint_paths[view.view_id], output_paths[view.view_id],
            duplicate_policy=policies.get(view.role),
            prediction_policy=prediction_policies.get(view.role),
            source_frame_contract=SOURCE_FRAME_CONTRACT,
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
        duplicate_policy = policies.get(role)
        role_decode_slot_budget = max(
            len(role_views),
            int(config["performance"].get("fine_active_decode_slots", len(role_views))),
        )
        maximum_per_source = max(
            1,
            int(
                config["performance"].get(
                    "fine_first_person_decode_workers"
                    if role == ViewRole.FIRST_PERSON
                    else "fine_third_person_decode_workers",
                    1,
                )
            ),
        )
        source_decode_workers = {view.view_id: 1 for view in role_views}
        remaining_decode_slots = max(0, role_decode_slot_budget - len(role_views))
        if phase == "fine":
            for view in role_views:
                if remaining_decode_slots <= 0:
                    break
                added = min(maximum_per_source - 1, remaining_decode_slots)
                source_decode_workers[view.view_id] += added
                remaining_decode_slots -= added
        role_started = time.perf_counter()
        motion_only = phase == "motion_probe" and not bool(
            config["performance"].get("motion_probe_run_yolo", False)
        )
        model_load_started = time.perf_counter()
        from .shared_inference import acquire_scanner
        scanner = None if motion_only else acquire_scanner(
            RoleScanner, role,
            config,
            effective_image_size,
            phase_batch_size,
            appearance_enabled=bool(
                phase == "fine"
                and config["performance"].get(
                    "fine_instance_appearance_enabled", False
                )
            ),
        )
        if scanner is not None and getattr(scanner, "prediction_policy", None) != prediction_policies.get(role):
            raise RuntimeError("Prediction identity changed between checkpoint verification and model startup")
        model_load_seconds = time.perf_counter() - model_load_started
        runtime_report = {
            "phase": phase,
            "role": role.value,
            "scanner_id": scanner_id,
            "model_path": str(scanner.model_path) if scanner is not None else None,
            "prediction_policy": prediction_policies.get(role),
            "backend": (
                "motion_only"
                if scanner is None
                else "TensorRT" if scanner.model_path.suffix.lower() == ".engine" else "PyTorch"
            ),
            "requested_batch_size": phase_batch_size,
            "effective_batch_size": scanner.batch_size if scanner is not None else 0,
            "initial_effective_batch_size": (
                getattr(scanner, "initial_batch_size", scanner.batch_size)
                if scanner is not None
                else 0
            ),
            "engine_build_batch": scanner.engine_build_batch if scanner is not None else None,
            "image_size": effective_image_size,
            "yolo_sample_fps": effective_fps,
            "motion_probe_fps": probe_fps,
            "motion_probe_segment_workers": int(
                config["performance"].get("motion_probe_segment_workers", 1)
            ),
            "active_decode_slot_budget": (
                role_decode_slot_budget if phase == "fine" else len(role_views)
            ),
            "ordered_source_decode_workers": (
                source_decode_workers
                if phase == "fine"
                else {view.view_id: 1 for view in role_views}
            ),
            "total_ordered_source_decode_workers": (
                sum(source_decode_workers.values())
                if phase == "fine"
                else len(role_views)
            ),
            "phase_chunk_seconds": float(
                config["performance"].get(
                    f"{phase}_chunk_seconds",
                    config["performance"]["chunk_seconds"],
                )
            ),
            "sparse_decode_strategy": str(
                config["performance"].get("motion_probe_sparse_strategy", "indexed_seek")
            ),
            "ffmpeg_cuda_scale": bool(
                config["performance"].get("ffmpeg_cuda_scale", False)
            ),
            "persistent_physical_segment_decode": bool(
                phase == "fine"
                and config["performance"].get(
                    "fine_persistent_segment_decode", False
                )
            ),
            "sampling_grid_by_view": {
                view.view_id: {
                    "mode": (
                        "aligned_global_timeline"
                        if phase == "fine"
                        and config["performance"].get(
                            "fine_persistent_segment_decode", False
                        )
                        and windows is not None
                        and bool(infos[view.view_id].segments)
                        else "decoder_default"
                    ),
                    "global_period_ms": 1000.0 / max(
                        effective_fps,
                        float(config["performance"].get("motion_probe_fps", 0.0)),
                        1e-9,
                    ),
                    "local_period_ms": (
                        1000.0
                        / max(
                            effective_fps,
                            float(
                                config["performance"].get("motion_probe_fps", 0.0)
                            ),
                            1e-9,
                        )
                        / transforms[view.view_id].scale
                    ),
                    "local_origin_ms": transforms[view.view_id].to_local(0.0),
                    "alignment_scale": transforms[view.view_id].scale,
                    "alignment_offset_ms": transforms[view.view_id].offset_ms,
                    "visual_correction_ms": transforms[
                        view.view_id
                    ].visual_correction_ms,
                }
                for view in role_views
            },
            "decode_backends": {
                view.view_id: (decode_backends or {}).get(
                    view.view_id,
                    "cuda" if config["performance"].get("ffmpeg_hwaccel") else "cpu",
                )
                for view in role_views
            },
            "active_source_workers": len(role_views),
            "synchronized_segment_waves": wave_barrier is not None,
            "model_load_seconds": round(model_load_seconds, 6),
        }
        scanner_suffix = f"_{scanner_id}" if scanner_id else ""
        runtime_path = work_dir / f"runtime_{phase}_{role.value}{scanner_suffix}.json"
        tracker_motion_prediction = config["performance"].get(
            "fine_tracker_motion_prediction_enabled"
            if phase == "fine"
            else "coarse_tracker_motion_prediction_enabled"
        )
        if tracker_motion_prediction is None:
            tracker_motion_prediction = config["performance"].get(
                "coarse_tracker_motion_prediction_enabled", False
            )
        tracker_center_distance = config["performance"].get(
            "fine_tracker_maximum_center_distance"
            if phase == "fine"
            else "coarse_tracker_maximum_center_distance"
        )
        if tracker_center_distance is None:
            tracker_center_distance = config["performance"].get(
                "coarse_tracker_maximum_center_distance", 0.18
            )
        trackers = {
            view.view_id: ByteSortTracker(
                max_age_ms=max(1750.0, 1500.0 / effective_fps),
                motion_prediction_enabled=bool(tracker_motion_prediction),
                maximum_center_distance=float(tracker_center_distance),
            )
            for view in role_views
        } if scanner is not None else {}
        component_timings = StageTimings()
        writers = {
            view.view_id: MeasuredWriter(output_paths[view.view_id].open("a", encoding="utf-8", buffering=1024 * 1024), component_timings)
            for view in role_views
        }
        emitted_timestamp_keys: dict[str, set[int]] = {
            view.view_id: set() for view in role_views
        }
        for view in role_views:
            if not completed[view.view_id] or not output_paths[view.view_id].is_file():
                continue
            for previous in iter_frame_evidence(output_paths[view.view_id]):
                _accept_unique_frame_timestamp(
                    emitted_timestamp_keys, view.view_id, previous.local_ms
                )
        duplicate_timestamp_frames: dict[str, int] = {
            view.view_id: 0 for view in role_views
        }
        queue_depth = int(
            config["performance"].get(
                "decode_queue_depth", config["performance"].get("frame_queue_size", 64)
            )
        )
        frame_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_depth))
        frame_queue.activity_path = (
            work_dir
            / f"source_activity_{phase}_{role.value}{scanner_suffix}.jsonl"
        )
        frame_queue.component_timings = component_timings
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
                    phase,
                    transforms[view.view_id],
                    source_decode_workers[view.view_id],
                ),
                name=f"decode-{view.view_id}",
                daemon=True,
            )
            for view in role_views
        ]
        for thread in threads:
            thread.start()

        ended: set[str] = set()
        pending_items: list[Any] = []
        pending_frame_count = 0
        errors: list[str] = []
        batch_sizes: list[int] = []
        motion_sample_count = 0
        max_queue_size = 0
        microbatch_timeout_flushes = 0
        full_batch_flushes = 0
        control_only_flushes = 0
        batch_started_at: float | None = None
        received_ends: set[str] = set()
        queue_wait_seconds = 0.0
        inference_seconds = 0.0
        postprocess_seconds = 0.0
        raw_motion_score_sum = 0.0
        effective_motion_score_sum = 0.0
        compensated_motion_frame_count = 0
        quality_fallback_frame_count = 0
        maximum_camera_shift_norm = 0.0
        motion_compensation_methods: dict[str, int] = {}
        degraded_motion_frame_count = 0
        raw_detection_count = 0
        suppressed_detection_count = 0
        suppression_frame_count = 0
        batch_wait_seconds = max(
            0.001,
            float(config["performance"].get("inference_batch_wait_ms", 25.0)) / 1000.0,
        )

        def flush_pending(reason: str) -> None:
            nonlocal pending_frame_count, motion_sample_count, batch_started_at
            nonlocal microbatch_timeout_flushes, full_batch_flushes, control_only_flushes
            nonlocal inference_seconds, postprocess_seconds
            nonlocal raw_motion_score_sum, effective_motion_score_sum
            nonlocal compensated_motion_frame_count, quality_fallback_frame_count
            nonlocal maximum_camera_shift_norm, degraded_motion_frame_count
            nonlocal raw_detection_count, suppressed_detection_count, suppression_frame_count
            if not pending_items:
                return
            frames = [item for item in pending_items if isinstance(item, FramePacket)]
            motion_sample_count += sum(
                phase == "motion_probe" or item.motion_probe_score is not None
                for item in frames
            )
            if scanner is None:
                inferred = [[] for _ in frames]
            else:
                inference_started = time.perf_counter()
                inferred = scanner.infer(frames) if frames else []
                inference_seconds += time.perf_counter() - inference_started
                if frames:
                    batch_sizes.extend(scanner.last_inference_batch_sizes)
                    from .device_day_activity import counted
                    counted(phase, len(frames))
            if reason == "full_batch":
                full_batch_flushes += 1
            elif reason == "timeout":
                microbatch_timeout_flushes += 1
            elif not frames:
                control_only_flushes += 1
            inferred_iter = iter(inferred)
            postprocess_started = time.perf_counter()
            for item in pending_items:
                if isinstance(item, FramePacket):
                    boxes = next(inferred_iter)
                    if not _accept_unique_frame_timestamp(
                        emitted_timestamp_keys,
                        item.view.view_id,
                        item.local_ms,
                    ):
                        duplicate_timestamp_frames[item.view.view_id] += 1
                        continue
                    is_motion_sample = bool(
                        phase == "motion_probe" or item.motion_probe_score is not None
                    )
                    if is_motion_sample:
                        raw_motion_score_sum += float(
                            item.motion_probe_raw_score
                            if item.motion_probe_raw_score is not None
                            else item.raw_motion_score
                        )
                        effective_motion_score_sum += float(
                            item.motion_probe_score
                            if item.motion_probe_score is not None
                            else item.motion_score
                        )
                        compensated_motion_frame_count += int(
                            item.camera_motion_compensated
                        )
                        quality_fallback_frame_count += int(
                            item.quality_fallback_applied
                        )
                        degraded_motion_frame_count += int(
                            item.motion_quality_state != "usable"
                        )
                        maximum_camera_shift_norm = max(
                            maximum_camera_shift_norm,
                            float(item.camera_shift_norm),
                        )
                        if item.camera_motion_method:
                            motion_compensation_methods[item.camera_motion_method] = (
                                motion_compensation_methods.get(
                                    item.camera_motion_method, 0
                                )
                                + 1
                            )
                    suppression_audit = None
                    if scanner is not None and duplicate_policy is not None:
                        raw_detection_count += len(boxes)
                        boxes, suppression_audit = suppress_duplicate_boxes(
                            boxes, duplicate_policy["iou_threshold"]
                        )
                        suppressed_detection_count += len(suppression_audit.removals)
                        suppression_frame_count += bool(suppression_audit.removals)
                    tracked = (
                        trackers[item.view.view_id].update(boxes, item.local_ms)
                        if scanner is not None
                        else []
                    )
                    evidence = FrameEvidence(
                        view_id=item.view.view_id,
                        role=item.view.role,
                        frame_index=item.frame_index,
                        local_ms=item.local_ms,
                        global_ms=transforms[item.view.view_id].to_global(item.local_ms),
                        width=item.frame.shape[1],
                        height=item.frame.shape[0],
                        motion_score=item.motion_score,
                        raw_motion_score=item.raw_motion_score,
                        motion_probe_score=(
                            item.motion_score
                            if phase == "motion_probe"
                            else item.motion_probe_score
                        ),
                        motion_probe_raw_score=(
                            item.raw_motion_score
                            if phase == "motion_probe"
                            else item.motion_probe_raw_score
                        ),
                        camera_motion_compensated=item.camera_motion_compensated,
                        camera_motion_method=item.camera_motion_method,
                        motion_quality_state=item.motion_quality_state,
                        detections=tracked,
                        duplicate_suppression=suppression_audit,
                        source_frame=item.source_frame,
                    )
                    writers[item.view.view_id].write(evidence.model_dump_json() + "\n")
                elif isinstance(item, ChunkEnd):
                    writers[item.view_id].flush()
                    completed[item.view_id].add(item.chunk_index)
                    _write_checkpoint(
                        checkpoint_paths[item.view_id],
                        completed[item.view_id],
                        output_paths[item.view_id],
                        duplicate_policy=duplicate_policy,
                        prediction_policy=prediction_policies.get(role),
                        source_frame_contract=SOURCE_FRAME_CONTRACT,
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
            postprocess_seconds += time.perf_counter() - postprocess_started
            pending_items.clear()
            pending_frame_count = 0
            batch_started_at = None

        try:
            while len(received_ends) < len(role_views):
                timeout = None
                if scanner is not None and pending_frame_count and batch_started_at is not None:
                    timeout = max(0.0, batch_wait_seconds - (time.perf_counter() - batch_started_at))
                try:
                    queue_wait_started = time.perf_counter()
                    item = frame_queue.get(timeout=timeout)
                    queue_wait_seconds += time.perf_counter() - queue_wait_started
                except queue.Empty:
                    queue_wait_seconds += time.perf_counter() - queue_wait_started
                    flush_pending("timeout")
                    continue
                max_queue_size = max(max_queue_size, frame_queue.qsize())
                pending_items.append(item)
                if isinstance(item, FramePacket):
                    pending_frame_count += 1
                    if batch_started_at is None:
                        batch_started_at = time.perf_counter()
                    if scanner is None:
                        flush_pending("motion_frame")
                    elif pending_frame_count >= scanner.batch_size:
                        flush_pending("full_batch")
                else:
                    if isinstance(item, ProducerEnd):
                        received_ends.add(item.view_id)
                    if pending_frame_count == 0:
                        flush_pending("control_only")
            flush_pending("producer_end")
            if errors:
                raise RuntimeError("；".join(errors))
        finally:
            for thread in threads:
                thread.join(timeout=5.0)
            processing_write_seconds = component_timings.snapshot().get("ledger_write_seconds", 0)
            for writer in writers.values():
                writer.close()
            if hasattr(scanner, "shared_statistics"):
                runtime_report["shared_inference"] = scanner.shared_statistics()
                runtime_report["inference_statistics_scope"] = "caller_submissions_not_physical_gpu_batches"
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
                    "final_effective_batch_size": (
                        scanner.batch_size if scanner is not None else 0
                    ),
                    "oom_batch_contraction_count": (
                        len(getattr(scanner, "batch_contractions", []))
                        if scanner is not None
                        else 0
                    ),
                    "oom_batch_contractions": (
                        list(getattr(scanner, "batch_contractions", []))
                        if scanner is not None
                        else []
                    ),
                    "max_observed_queue_depth": max_queue_size,
                    "configured_queue_depth": queue_depth,
                    "inference_batch_wait_ms": round(batch_wait_seconds * 1000.0, 3),
                    "duplicate_timestamp_frames_removed": sum(
                        duplicate_timestamp_frames.values()
                    ),
                    "duplicate_timestamp_frames_removed_by_view": dict(
                        duplicate_timestamp_frames
                    ),
                    "unique_output_timestamps_by_view": {
                        view_id: len(keys)
                        for view_id, keys in emitted_timestamp_keys.items()
                    },
                    "full_batch_flushes": full_batch_flushes,
                    "microbatch_timeout_flushes": microbatch_timeout_flushes,
                    "control_only_flushes": control_only_flushes,
                    "queue_wait_seconds": round(queue_wait_seconds, 6),
                    "inference_seconds": round(inference_seconds, 6),
                    "observed_prediction_end2end": getattr(scanner, "last_prediction_end2end", None),
                    "tracking_and_ledger_seconds": round(postprocess_seconds, 6),
                    "component_timings": {
                        "schema_version": "visioncortex-component-timing/1",
                        "scope": "overlapping worker sums; model profile includes padding; buffered writes, not fsync",
                        "read_decode_separate": False,
                        **component_timings.snapshot(),
                        **(scanner.component_timings.snapshot() if hasattr(scanner, "component_timings") else {}),
                        "tracking_serialization_seconds": round(max(0.0, postprocess_seconds-processing_write_seconds), 6),
                    },
                    "duplicate_suppression": {
                        "policy": duplicate_policy,
                        "input_detections": raw_detection_count,
                        "suppressed_detections": suppressed_detection_count,
                        "frames_with_suppression": suppression_frame_count,
                        "scope": "current_invocation_before_tracking",
                    },
                    "role_total_seconds": round(time.perf_counter() - role_started, 6),
                }
            )
            if (
                phase == "motion_probe"
                or config["performance"].get(
                    "coarse_shared_motion_probe_enabled", False
                )
            ) and (
                config["performance"].get(
                    "motion_probe_camera_motion_compensation", False
                )
                or config["performance"].get(
                    "motion_probe_first_person_quality_fallback", False
                )
            ):
                sample_count = motion_sample_count
                runtime_report["motion_scoring"] = {
                    "camera_motion_compensation_enabled": bool(
                        config["performance"].get(
                            "motion_probe_camera_motion_compensation", False
                        )
                    ),
                    "sample_count": sample_count,
                    "raw_motion_score_mean": round(
                        raw_motion_score_sum / max(1, sample_count), 6
                    ),
                    "effective_motion_score_mean": round(
                        effective_motion_score_sum / max(1, sample_count), 6
                    ),
                    "compensated_frame_count": compensated_motion_frame_count,
                    "first_person_quality_fallback_frame_count": (
                        quality_fallback_frame_count
                    ),
                    "degraded_quality_frame_count": degraded_motion_frame_count,
                    "compensation_methods": dict(
                        sorted(motion_compensation_methods.items())
                    ),
                    "shared_with_coarse_decode": bool(
                        phase == "coarse"
                        and config["performance"].get(
                            "coarse_shared_motion_probe_enabled", False
                        )
                    ),
                    "maximum_camera_shift_norm": round(
                        maximum_camera_shift_norm, 6
                    ),
                }
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


def nearest_frame_evidence_many(
    path: Path,
    global_timestamps_ms: Sequence[float],
    tolerance_ms: float = 1500.0,
) -> dict[float, FrameEvidence | None]:
    """Resolve sorted timestamp queries with one sequential ledger pass."""

    queries = sorted({float(value) for value in global_timestamps_ms})
    results: dict[float, FrameEvidence | None] = {value: None for value in queries}
    if not queries:
        return results
    frames = (frame for frame in iter_frame_evidence(path) if frame.global_ms is not None)
    previous: FrameEvidence | None = None
    current = next(frames, None)
    for query in queries:
        while current is not None and float(current.global_ms) < query:
            previous = current
            current = next(frames, None)
        candidates = [frame for frame in (previous, current) if frame is not None]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda frame: abs(float(frame.global_ms) - query))
        if abs(float(nearest.global_ms) - query) <= tolerance_ms:
            results[query] = nearest
    return results
