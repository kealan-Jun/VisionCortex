from __future__ import annotations

import csv
import math
import time
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .schemas import (
    AlignmentSegmentTransform,
    AlignmentTransform,
    TimestampPoint,
    VideoInfo,
    ViewInput,
    ViewRole,
)
from .storage import read_source_file_edges
from .video_io import view_motion_signature, virtual_frame_index_at


FRAME_COLUMNS = (
    "rgb_video_frame_index",
    "video_frame_index",
    "frame_index",
    "frame",
    "frame_id",
    "index",
    "seq",
)
LOCAL_US_COLUMNS = ("local_time_us", "local_timestamp_us", "video_timestamp_us", "pts_us")
LOCAL_MS_COLUMNS = ("local_timestamp_ms", "video_timestamp_ms", "pts_ms", "timestamp_ms")
LOCAL_S_COLUMNS = ("local_timestamp_s", "video_timestamp_s", "pts_time", "timestamp_s", "time_s")
# Prefer the clock-synchronised global timestamp. frame_system_timestamp_us is
# the sender's raw system clock and can carry a per-device offset even when the
# recorder reports clock_sync_valid=1.
SOURCE_US_COLUMNS = ("global_timestamp_us", "frame_system_timestamp_us", "wallclock_us", "epoch_us")
SOURCE_MS_COLUMNS = ("global_timestamp_ms", "wallclock_ms", "epoch_ms", "capture_timestamp_ms")
SOURCE_COLUMNS = ("global_timestamp", "wallclock", "capture_timestamp", "datetime", "iso_time")
CLOCK_SYNC_COLUMNS = ("clock_sync_valid", "time_sync_valid", "ntp_sync_valid", "sync_valid")


def _first(row: dict[str, str], names: Sequence[str]) -> tuple[str | None, str | None]:
    lowered = {key.strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = lowered.get(name)
        if value not in (None, ""):
            return name, value.strip()
    return None, None


def _parse_datetime_ms(value: str) -> float:
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned).timestamp() * 1000.0
    except ValueError:
        return float(value)


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "valid", "locked", "synchronized"}:
        return True
    if normalized in {"0", "false", "no", "n", "invalid", "unlocked", "unsynchronized"}:
        return False
    return None


def _timestamp_point(row: dict[str, str], row_number: int, fps: float) -> TimestampPoint:
    _, frame_text = _first(row, FRAME_COLUMNS)
    frame_index = int(float(frame_text)) if frame_text is not None else row_number
    local_name, local_text = _first(row, LOCAL_US_COLUMNS)
    source_name, source_text = _first(row, SOURCE_US_COLUMNS)
    if local_text is not None:
        local_ms = float(local_text) / 1000.0
    else:
        local_name, local_text = _first(row, LOCAL_MS_COLUMNS)
        if local_text is not None:
            local_ms = float(local_text)
        else:
            local_name, local_text = _first(row, LOCAL_S_COLUMNS)
            local_ms = (
                float(local_text) * 1000.0
                if local_text is not None
                else frame_index * 1000.0 / max(fps, 1e-9)
            )
    if source_text is not None:
        source_ms = float(source_text) / 1000.0
    else:
        source_name, source_text = _first(row, SOURCE_MS_COLUMNS)
        if source_text is not None:
            source_ms = float(source_text)
        else:
            source_name, source_text = _first(row, SOURCE_COLUMNS)
            source_ms = _parse_datetime_ms(source_text) if source_text is not None else None
    _, clock_sync_text = _first(row, CLOCK_SYNC_COLUMNS)
    clock_sync_valid = _parse_optional_bool(clock_sync_text)
    raw_source_ms = source_ms
    if clock_sync_valid is False or (
        source_name == "frame_system_timestamp_us" and clock_sync_valid is not True
    ):
        source_ms = None
    # Capture exports can put epoch wall-clock values in local_time_us. Keep
    # playback on the media frame timeline even when a separate, preferred
    # capture-clock column is also present.
    if abs(local_ms) > 10_000_000_000:
        if (
            source_ms is None
            and clock_sync_valid is None
            and source_name == "frame_system_timestamp_us"
        ):
            # Some capture exports omit clock_sync_valid while providing both
            # an epoch local clock and the sender system clock. That paired
            # shape is distinct from an unqualified raw system clock alone.
            source_ms = raw_source_ms
        elif source_ms is None and clock_sync_valid is not False:
            source_ms = local_ms
            source_name = local_name
        local_ms = frame_index * 1000.0 / max(fps, 1e-9)
    return TimestampPoint(
        frame_index=frame_index,
        local_ms=local_ms,
        source_ms=source_ms,
        clock_sync_valid=clock_sync_valid,
        source_column=source_name,
        row_number=row_number,
    )


def read_timestamp_csv_endpoints(path: Path, fps: float) -> list[TimestampPoint]:
    """Read the first/last recorded RGB rows without scanning a multi-million-row CSV."""

    head, tail, size = read_source_file_edges(path)
    head_lines = head.decode("utf-8-sig", errors="replace").splitlines()
    tail_lines = tail.decode("utf-8", errors="replace").splitlines()
    if len(head_lines) < 2:
        raise ValueError(f"时间戳 CSV 至少需要两行: {path}")
    fieldnames = next(csv.reader([head_lines[0]]))
    lowered = [item.strip().lower() for item in fieldnames]
    rgb_index = lowered.index("rgb_recorded") if "rgb_recorded" in lowered else None

    def endpoint(
        lines: Sequence[str], *, reverse: bool = False
    ) -> TimestampPoint | None:
        indexes = range(len(lines) - 1, -1, -1) if reverse else range(len(lines))
        for row_number in indexes:
            line = lines[row_number]
            values = next(csv.reader([line]), [])
            if not values or values[0].strip().lower() == fieldnames[0].strip().lower():
                continue
            if rgb_index is not None and (
                rgb_index >= len(values)
                or values[rgb_index].strip().lower() not in {"1", "true", "yes"}
            ):
                continue
            row = {
                key: values[index] if index < len(values) else ""
                for index, key in enumerate(fieldnames)
            }
            try:
                return _timestamp_point(row, row_number, fps)
            except (TypeError, ValueError):
                continue
        return None

    first = endpoint(head_lines[1:])
    last = endpoint(
        tail_lines[1:] if size > len(tail) else tail_lines,
        reverse=True,
    )
    if first is None or last is None:
        raise ValueError(f"时间戳 CSV 缺少可用RGB首尾记录: {path}")
    endpoints = [first, last]
    endpoints.sort(key=lambda point: point.local_ms)
    return endpoints


def read_timestamp_csv(path: Path, fps: float, max_points: int = 50_000) -> list[TimestampPoint]:
    if not path.is_file():
        raise FileNotFoundError(f"时间戳 CSV 不存在: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as counter:
        row_count = max(0, sum(1 for _ in counter) - 1)
    stride = max(1, math.ceil(row_count / max_points))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect=dialect)
        fieldnames = next(reader, None)
        if not fieldnames:
            raise ValueError(f"时间戳 CSV 缺少表头: {path}")
        lowered_fields = [item.strip().lower() for item in fieldnames]
        rgb_recorded_index = (
            lowered_fields.index("rgb_recorded") if "rgb_recorded" in lowered_fields else None
        )
        points: list[TimestampPoint] = []
        for row_number, values in enumerate(reader):
            if (
                rgb_recorded_index is not None
                and (
                    rgb_recorded_index >= len(values)
                    or values[rgb_recorded_index].strip().lower() not in {"1", "true", "yes"}
                )
            ):
                continue
            if row_number % stride and row_number + 1 < row_count:
                continue
            row = {
                key: values[index] if index < len(values) else ""
                for index, key in enumerate(fieldnames)
            }
            points.append(_timestamp_point(row, row_number, fps))
    if len(points) < 2:
        raise ValueError(f"时间戳 CSV 至少需要两行: {path}")
    points.sort(key=lambda point: point.local_ms)
    return points


def read_timestamp_csv_bounded(
    path: Path,
    fps: float,
    sample_count: int = 5,
    full_read_limit_bytes: int = 2 * 1024 * 1024,
) -> list[TimestampPoint]:
    """Read bounded, distributed clock samples without streaming a huge NAS CSV."""

    if not path.is_file():
        raise FileNotFoundError(f"时间戳 CSV 不存在: {path}")
    sample_count = max(2, int(sample_count))
    size = path.stat().st_size
    if size <= full_read_limit_bytes:
        try:
            return read_timestamp_csv(path, fps, max_points=sample_count)
        except ValueError as error:
            if not str(error).startswith("时间戳 CSV 至少需要两行:"):
                raise
            # Sparse RGB rows can all fall between the original row strides.
            # Retry only that failed bounded sample; keep successful reads exact.
            # The file-size gate bounds this full read, including mixed depth rows.
            points = read_timestamp_csv(path, fps, max_points=size + 1)
            indices = np.linspace(0, len(points) - 1, min(sample_count, len(points)), dtype=int)
            return [points[int(index)] for index in indices]

    endpoints = read_timestamp_csv_endpoints(path, fps)
    with path.open("rb") as handle:
        header_bytes = handle.readline()
        header_text = header_bytes.decode("utf-8-sig", errors="replace").strip("\r\n")
        try:
            dialect = csv.Sniffer().sniff(header_text, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        fieldnames = next(csv.reader([header_text], dialect), None)
        if not fieldnames:
            raise ValueError(f"时间戳 CSV 缺少表头: {path}")
        lowered_fields = [item.strip().lower() for item in fieldnames]
        rgb_index = (
            lowered_fields.index("rgb_recorded")
            if "rgb_recorded" in lowered_fields
            else None
        )
        points = list(endpoints)
        interior_count = max(0, sample_count - 2)
        offsets = (
            np.linspace(len(header_bytes), max(len(header_bytes), size - 1), interior_count + 2)[1:-1]
            if interior_count
            else []
        )
        for offset in offsets:
            handle.seek(int(offset))
            if offset > len(header_bytes):
                handle.readline()
            for _ in range(64):
                raw = handle.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace").strip("\r\n")
                if not text:
                    continue
                values = next(csv.reader([text], dialect), [])
                if not values:
                    continue
                if rgb_index is not None and (
                    rgb_index >= len(values)
                    or values[rgb_index].strip().lower() not in {"1", "true", "yes"}
                ):
                    continue
                row = {
                    key: values[index] if index < len(values) else ""
                    for index, key in enumerate(fieldnames)
                }
                try:
                    points.append(_timestamp_point(row, -1, fps))
                except (TypeError, ValueError):
                    continue
                break
    unique = {
        (point.frame_index, round(point.local_ms, 6), point.source_ms): point
        for point in points
    }
    result = sorted(unique.values(), key=lambda point: point.local_ms)
    if len(result) < 2:
        raise ValueError(f"时间戳 CSV 缺少足够的有界采样点: {path}")
    return result


def read_video_timestamp_csv(path: Path, info: VideoInfo, *, sample_count: int = 5,
                             full_read_limit_bytes: int = 2 * 1024 * 1024) -> list[TimestampPoint]:
    """Bind recorder frame ordinals to native PTS before clock fitting.

    Recorder CSV local_time_us is a wall clock. Average-FPS conversion loses
    internal recording gaps, even when the first and last timestamps match.
    Generic timestamp formats retain their existing explicit media timeline.
    """
    points = read_timestamp_csv_bounded(path, info.fps, sample_count, full_read_limit_bytes)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not {"rgb_video_frame_index", "rgb_recorded"}.issubset(header):
        return points
    from .source_frames import SourceFrameTrace
    trace = SourceFrameTrace(info.path, 0, info.duration_ms, 2)
    if trace.failure or not trace.positions or not trace.time_base:
        raise ValueError("Recorder clock requires native video frame timestamps")
    from fractions import Fraction
    native = {ordinal: float(pts * Fraction(trace.time_base) * 1000) - trace.origin_ms
              for rows in trace.positions.values() for pts, ordinal in rows if ordinal is not None}
    result = []
    for point in points:
        if point.frame_index not in native:
            raise ValueError("Recorder frame index has no native video timestamp")
        result.append(point.model_copy(update={"local_ms": native[point.frame_index]}))
    return result


def audit_timestamp_series(
    points: Sequence[TimestampPoint],
    *,
    jump_threshold_ms: float,
    max_rate_error_ppm: float,
) -> dict[str, object]:
    """Return bounded clock-quality evidence without promoting endpoints to certainty."""

    ordered = sorted(points, key=lambda point: point.local_ms)
    anomalies: list[str] = []
    valid_source = [
        point
        for point in ordered
        if point.source_ms is not None and point.clock_sync_valid is not False
    ]
    if any(
        current.local_ms <= previous.local_ms
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        anomalies.append("local_clock_not_strictly_increasing")
    invalid_sync_count = sum(point.clock_sync_valid is False for point in ordered)
    if invalid_sync_count:
        anomalies.append(f"clock_sync_invalid_samples:{invalid_sync_count}")
    jump_limit = max(1.0, float(jump_threshold_ms))
    rate_limit = max(0.0, float(max_rate_error_ppm)) / 1_000_000.0
    for previous, current in zip(valid_source, valid_source[1:], strict=False):
        local_delta = current.local_ms - previous.local_ms
        source_delta = float(current.source_ms) - float(previous.source_ms)
        if source_delta <= 0:
            anomalies.append("absolute_clock_not_strictly_increasing")
            continue
        if local_delta <= 0:
            continue
        residual = source_delta - local_delta
        if abs(residual) > max(jump_limit, local_delta * rate_limit):
            anomalies.append(
                "clock_step_ms:"
                f"{round(residual, 3)}@{round(previous.local_ms, 3)}-{round(current.local_ms, 3)}"
            )
    return {
        "sample_count": len(ordered),
        "valid_source_count": len(valid_source),
        "valid_source_ratio": len(valid_source) / max(1, len(ordered)),
        "anomalies": list(dict.fromkeys(anomalies)),
    }


def synthetic_timestamps(info: VideoInfo, sample_fps: float = 2.0) -> list[TimestampPoint]:
    count = max(2, int(math.floor(info.duration_ms / 1000.0 * sample_fps)) + 1)
    return [
        TimestampPoint(
            frame_index=int(round(index / sample_fps * info.fps)),
            local_ms=index * 1000.0 / sample_fps,
        )
        for index in range(count)
    ]


def _nearest_pairs(
    reference: Sequence[TimestampPoint],
    target: Sequence[TimestampPoint],
    tolerance_ms: float,
    *,
    allow_index_fallback: bool = False,
) -> list[tuple[float, float]]:
    ref_has_source = sum(point.source_ms is not None for point in reference) >= 2
    target_has_source = sum(point.source_ms is not None for point in target) >= 2
    if not (ref_has_source and target_has_source):
        if not allow_index_fallback:
            return []
        count = min(len(reference), len(target), 20_000)
        if count < 2:
            return []
        ref_indexes = np.linspace(0, len(reference) - 1, count).astype(int)
        target_indexes = np.linspace(0, len(target) - 1, count).astype(int)
        return [(target[j].local_ms, reference[i].local_ms) for i, j in zip(ref_indexes, target_indexes, strict=True)]

    reference_source = [(point.source_ms, point.local_ms) for point in reference if point.source_ms is not None]
    reference_source.sort(key=lambda item: item[0])
    keys = [item[0] for item in reference_source]
    pairs: list[tuple[float, float]] = []
    for point in target:
        if point.source_ms is None:
            continue
        insertion = bisect_left(keys, point.source_ms)
        candidates = []
        if insertion < len(keys):
            candidates.append(reference_source[insertion])
        if insertion:
            candidates.append(reference_source[insertion - 1])
        if not candidates:
            continue
        source_ms, reference_local_ms = min(candidates, key=lambda item: abs(item[0] - point.source_ms))
        if abs(source_ms - point.source_ms) <= tolerance_ms:
            pairs.append((point.local_ms, reference_local_ms))
    return pairs


def _robust_affine(pairs: Sequence[tuple[float, float]], max_drift_ppm: float) -> tuple[float, float, float]:
    values = np.asarray(pairs, dtype=np.float64)
    if len(values) < 2:
        return 1.0, 0.0, float("inf")
    keep = np.ones(len(values), dtype=bool)
    scale, offset = 1.0, float(np.median(values[:, 1] - values[:, 0]))
    for _ in range(4):
        x, y = values[keep, 0], values[keep, 1]
        if len(x) < 2 or np.ptp(x) < 1.0:
            break
        scale, offset = np.polyfit(x, y, 1)
        scale = float(np.clip(scale, 1.0 - max_drift_ppm / 1_000_000.0, 1.0 + max_drift_ppm / 1_000_000.0))
        residuals = values[:, 1] - (scale * values[:, 0] + offset)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        threshold = max(10.0, 4.5 * 1.4826 * mad)
        keep = np.abs(residuals - median) <= threshold
    residuals = values[keep, 1] - (scale * values[keep, 0] + offset)
    rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else float("inf")
    return float(scale), float(offset), rmse


def _absolute_clock_transform(
    reference: Sequence[TimestampPoint],
    target: Sequence[TimestampPoint],
    max_drift_ppm: float,
) -> tuple[float, float, float, int] | None:
    """Map target local time to reference local time through recorder clocks.

    Segment endpoints from different cameras do not have to occur within the
    nearest-neighbour tolerance (one recorder can close a segment seconds later).
    Both CSVs nevertheless carry the same absolute system clock. Fitting each
    local timeline against that clock avoids a dense multi-million-row read and
    is more accurate than pairing non-simultaneous segment boundaries.
    """

    def fit(points: Sequence[TimestampPoint]) -> tuple[float, float, float, int] | None:
        usable = [point for point in points if point.source_ms is not None]
        if len(usable) < 2:
            return None
        origin_local = usable[0].local_ms
        origin_source = float(usable[0].source_ms)
        pairs = [
            (point.local_ms - origin_local, float(point.source_ms) - origin_source)
            for point in usable
        ]
        scale, relative_offset, rmse = _robust_affine(pairs, max_drift_ppm)
        intercept = origin_source + relative_offset - scale * origin_local
        return scale, intercept, rmse, len(usable)

    reference_fit = fit(reference)
    target_fit = fit(target)
    if reference_fit is None or target_fit is None:
        return None
    reference_scale, reference_intercept, reference_rmse, reference_count = reference_fit
    target_scale, target_intercept, target_rmse, target_count = target_fit
    scale = target_scale / max(reference_scale, 1e-12)
    offset = (target_intercept - reference_intercept) / max(reference_scale, 1e-12)
    rmse = math.sqrt(reference_rmse**2 + target_rmse**2)
    return scale, offset, rmse, min(reference_count, target_count)


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 3:
        return -1.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-9 else -1.0


def visual_anchor_calibration(
    reference_view: ViewInput,
    target_view: ViewInput,
    reference_transform: AlignmentTransform,
    target_transform: AlignmentTransform,
    reference_info: VideoInfo,
    target_info: VideoInfo,
    search_seconds: float,
    anchor_count: int,
    anchor_duration_seconds: float,
    fps: float,
    signature_cache: dict[tuple[str, tuple[float, ...]], np.ndarray] | None = None,
) -> tuple[float, float, list[dict[str, float]]]:
    overlap_start = max(reference_transform.to_global(0.0), target_transform.to_global(0.0))
    overlap_end = min(
        reference_transform.to_global(reference_info.duration_ms),
        target_transform.to_global(target_info.duration_ms),
    )
    margin_ms = (anchor_duration_seconds / 2.0 + search_seconds + 1.0) * 1000.0
    if overlap_end - overlap_start <= margin_ms * 2:
        return 0.0, 0.0, []
    # Recorder MP4s are split every 15 minutes. Seeking to an arbitrary point in a
    # long-GOP segment can make FFmpeg decode hundreds of megabytes before it
    # reaches the requested frame, especially over SMB. Prefer points immediately
    # after well-separated segment boundaries: this still audits offset/drift at
    # multiple positions on the global timeline while keeping every decode local.
    boundary_centers: list[float] = []
    if reference_info.segments:
        for segment in reference_info.segments:
            candidate = reference_transform.to_global(segment.virtual_start_ms) + margin_ms
            if overlap_start + margin_ms <= candidate <= overlap_end - margin_ms:
                boundary_centers.append(candidate)
    if len(boundary_centers) >= anchor_count:
        indexes = np.linspace(0, len(boundary_centers) - 1, anchor_count).round().astype(int)
        centers = np.asarray([boundary_centers[index] for index in indexes], dtype=np.float64)
    else:
        centers = np.linspace(overlap_start + margin_ms, overlap_end - margin_ms, anchor_count)
    half_count = max(2, int(round(anchor_duration_seconds * fps / 2.0)))
    max_lag = max(1, int(round(search_seconds * fps)))
    corrections: list[tuple[float, float, float]] = []
    details: list[dict[str, float]] = []

    def motion_signature(
        view: ViewInput,
        info: VideoInfo,
        local_times_ms: Sequence[float],
    ) -> np.ndarray:
        key = (
            view.view_id,
            tuple(round(float(value), 3) for value in local_times_ms),
        )
        if signature_cache is not None and key in signature_cache:
            return signature_cache[key]
        signal = view_motion_signature(view, info, local_times_ms)
        if signature_cache is not None:
            signature_cache[key] = signal
        return signal

    for center in centers:
        reference_globals = center + np.arange(-half_count, half_count + 1) * 1000.0 / fps
        extended_globals = center + np.arange(-half_count - max_lag, half_count + max_lag + 1) * 1000.0 / fps
        ref_times = [reference_transform.to_local(float(value)) for value in reference_globals]
        target_times = [target_transform.to_local(float(value)) for value in extended_globals]
        ref_signal = motion_signature(reference_view, reference_info, ref_times)
        target_signal = motion_signature(target_view, target_info, target_times)
        scores = [
            _normalized_correlation(ref_signal, target_signal[lag : lag + len(ref_signal)])
            for lag in range(max_lag * 2 + 1)
        ]
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        ranked = np.sort(np.asarray(scores, dtype=np.float64))
        second_score = float(ranked[-2]) if len(ranked) > 1 else -1.0
        peak_margin = best_score - second_score
        delta_ms = (best_index - max_lag) * 1000.0 / fps
        correction_ms = -delta_ms
        confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
        at_search_edge = best_index in {0, len(scores) - 1}
        reliable = best_score >= 0.25 and peak_margin >= 0.01 and not at_search_edge
        details.append(
            {
                "global_center_ms": float(center),
                "correction_ms": correction_ms,
                "correlation": best_score,
                "second_best_correlation": second_score,
                "peak_margin": peak_margin,
                "at_search_edge": at_search_edge,
                "reliable": reliable,
                "confidence": confidence,
            }
        )
        if reliable:
            corrections.append((float(center), correction_ms, confidence))
    # Visual correction is only allowed to override a high-quality CSV fit when
    # at least two independent anchors agree. One static cross-view coincidence
    # must never move a three-hour timeline by seconds.
    if len(corrections) < 2:
        return 0.0, 0.0, details
    values = np.asarray([item[1] for item in corrections], dtype=np.float64)
    correction = float(np.median(values))
    agreement_ms = float(np.max(np.abs(values - correction)))
    if agreement_ms > max(500.0, 1000.0 / fps * 2.0):
        return 0.0, 0.0, details
    confidence = float(np.median([item[2] for item in corrections]))
    return correction, confidence, details


def build_alignments(
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    config: dict,
) -> tuple[dict[str, AlignmentTransform], dict[str, list[TimestampPoint]]]:
    build_started = time.perf_counter()
    align_cfg = config["alignment"]
    preferred_reference = next(
        (view for view in views if view.view_id == align_cfg["reference_view"]),
        None,
    )
    if preferred_reference is None:
        preferred_reference = next(
            view for view in views if view.role.value == align_cfg["reference_view"]
        )
    sampling_mode = str(align_cfg.get("csv_sampling_mode", "bounded_sparse"))
    samples_per_segment = max(2, int(align_cfg.get("csv_samples_per_segment", 5)))
    clock_jobs: dict[Path, float] = {}
    clock_infos = {}
    for view in views:
        info = infos[view.view_id]
        if info.segments:
            for segment in info.segments:
                if segment.timestamps_csv:
                    clock_jobs[segment.timestamps_csv] = segment.fps
                    clock_infos[segment.timestamps_csv] = segment
        elif view.timestamps_csv:
            clock_jobs[view.timestamps_csv] = info.fps
            clock_infos[view.timestamps_csv] = info
    clock_cache: dict[Path, list[TimestampPoint]] = {}
    clock_sampling_started = time.perf_counter()
    if clock_jobs:
        reader = (
            read_timestamp_csv_endpoints
            if sampling_mode == "legacy_endpoints"
            else lambda path, fps: read_video_timestamp_csv(
                path,
                clock_infos[path],
                sample_count=samples_per_segment,
                full_read_limit_bytes=int(
                    align_cfg.get("csv_bounded_full_read_limit_bytes", 2 * 1024 * 1024)
                ),
            )
        )
        with ThreadPoolExecutor(
            max_workers=max(
                1,
                min(int(align_cfg.get("csv_read_workers", 12)), len(clock_jobs)),
            ),
            thread_name_prefix="clock-samples",
        ) as executor:
            futures = {
                executor.submit(reader, path, fps): path
                for path, fps in clock_jobs.items()
            }
            for future in as_completed(futures):
                clock_cache[futures[future]] = future.result()
    clock_sampling_seconds = time.perf_counter() - clock_sampling_started

    jump_threshold_ms = float(align_cfg.get("clock_jump_threshold_ms", 500.0))
    rate_error_ppm = float(align_cfg.get("clock_rate_error_ppm", 10_000.0))
    series: dict[str, list[TimestampPoint]] = {}
    segment_series_by_view: dict[str, list[list[TimestampPoint]]] = {}
    segment_audits_by_view: dict[str, list[dict[str, object]]] = {}
    for view in views:
        info = infos[view.view_id]
        if view.segments and info.segments:
            points: list[TimestampPoint] = []
            raw_segment_series: list[list[TimestampPoint] | None] = []
            for segment in info.segments:
                if segment.timestamps_csv:
                    raw_segment_series.append(clock_cache[segment.timestamps_csv])
                else:
                    raw_segment_series.append(None)
            source_origins = [
                point.source_ms
                for source_points in raw_segment_series
                if source_points
                for point in source_points
                if point.source_ms is not None
            ]
            base_source_ms = source_origins[0] if source_origins else None
            previous_start_ms = -1.0
            shifted_segments: list[list[TimestampPoint]] = []
            segment_audits: list[dict[str, object]] = []
            for segment, source_points in zip(
                info.segments, raw_segment_series, strict=True
            ):
                placement_anomalies: list[str] = []
                if source_points and base_source_ms is not None and source_points[0].source_ms is not None:
                    clock_start_ms = source_points[0].source_ms - base_source_ms
                    # Reject corrupt clock jumps but preserve real recorder gaps/overlaps.
                    maximum_shift_ms = float(
                        align_cfg.get("maximum_segment_clock_shift_ms", 300_000.0)
                    )
                    if clock_start_ms < previous_start_ms:
                        placement_anomalies.append("segment_absolute_clock_regression")
                    elif abs(clock_start_ms - segment.virtual_start_ms) > maximum_shift_ms:
                        placement_anomalies.append(
                            "segment_clock_shift_exceeds_limit_ms:"
                            f"{round(clock_start_ms - segment.virtual_start_ms, 3)}"
                        )
                    else:
                        segment.virtual_start_ms = max(0.0, clock_start_ms)
                        segment.virtual_end_ms = segment.virtual_start_ms + segment.duration_ms
                previous_start_ms = segment.virtual_start_ms
                if source_points:
                    origin = source_points[0].local_ms
                    shifted = [
                        TimestampPoint(
                            frame_index=segment.frame_start_index + point.frame_index,
                            local_ms=segment.virtual_start_ms + point.local_ms - origin,
                            source_ms=point.source_ms,
                            clock_sync_valid=point.clock_sync_valid,
                            source_column=point.source_column,
                            row_number=point.row_number,
                        )
                        for point in source_points
                    ]
                else:
                    shifted = [
                        TimestampPoint(
                            frame_index=segment.frame_start_index + point.frame_index,
                            local_ms=segment.virtual_start_ms + point.local_ms,
                            source_ms=point.source_ms,
                            source_column="synthetic_video_timeline",
                        )
                        for point in synthetic_timestamps(
                            VideoInfo(
                                path=segment.path,
                                duration_ms=segment.duration_ms,
                                fps=segment.fps,
                                width=segment.width,
                                height=segment.height,
                                frame_count=segment.frame_count,
                                size_bytes=segment.size_bytes,
                            ),
                            sample_fps=align_cfg["aligned_timestamps_fps"],
                        )
                    ]
                shifted_segments.append(shifted)
                points.extend(shifted)
                audit = audit_timestamp_series(
                    shifted,
                    jump_threshold_ms=jump_threshold_ms,
                    max_rate_error_ppm=rate_error_ppm,
                )
                audit["anomalies"] = [
                    *audit["anomalies"],
                    *placement_anomalies,
                ]
                segment_audits.append(audit)
            info.duration_ms = max(segment.virtual_end_ms for segment in info.segments)
            series[view.view_id] = points
            segment_series_by_view[view.view_id] = shifted_segments
            segment_audits_by_view[view.view_id] = segment_audits
        else:
            points = (
                clock_cache[view.timestamps_csv]
                if view.timestamps_csv
                else synthetic_timestamps(info, sample_fps=align_cfg["aligned_timestamps_fps"])
            )
            series[view.view_id] = points
            segment_series_by_view[view.view_id] = [points]
            segment_audits_by_view[view.view_id] = [
                audit_timestamp_series(
                    points,
                    jump_threshold_ms=jump_threshold_ms,
                    max_rate_error_ppm=rate_error_ppm,
                )
            ]

    def view_clock_score(view: ViewInput) -> float:
        audits = segment_audits_by_view[view.view_id]
        sample_count = sum(int(item["sample_count"]) for item in audits)
        valid_count = sum(int(item["valid_source_count"]) for item in audits)
        segment_coverage = sum(
            int(item["valid_source_count"]) >= min(3, samples_per_segment)
            for item in audits
        ) / max(1, len(audits))
        clean_ratio = 1.0 - min(
            1.0,
            sum(len(item["anomalies"]) for item in audits) / max(1, len(audits) * 2),
        )
        valid_ratio = valid_count / max(1, sample_count)
        preference = (
            float(align_cfg.get("reference_preference_bonus", 0.15))
            if view.view_id == preferred_reference.view_id
            else 0.0
        )
        return 0.45 * valid_ratio + 0.30 * segment_coverage + 0.25 * clean_ratio + preference

    reference = (
        preferred_reference
        if str(align_cfg.get("reference_strategy", "best_clock_quality")) == "configured"
        else max(views, key=lambda item: (view_clock_score(item), item.view_id == preferred_reference.view_id))
    )

    def combined_audit(view_id: str) -> tuple[int, int, list[str]]:
        audits = segment_audits_by_view[view_id]
        sample_count = sum(int(item["sample_count"]) for item in audits)
        valid_count = sum(int(item["valid_source_count"]) for item in audits)
        anomalies = [
            f"segment_{index}:{anomaly}"
            for index, audit in enumerate(audits)
            for anomaly in audit["anomalies"]
        ]
        return sample_count, valid_count, anomalies

    reference_samples, reference_valid, reference_anomalies = combined_audit(
        reference.view_id
    )
    shared_preparation_seconds = time.perf_counter() - build_started
    ref_transform = AlignmentTransform(
        view_id=reference.view_id,
        reference_view_id=reference.view_id,
        scale=1.0,
        offset_ms=reference.calibration_hint_ms,
        csv_match_count=len(series[reference.view_id]),
        csv_match_ratio=1.0,
        csv_rmse_ms=0.0,
        visual_confidence=1.0,
        confidence=1.0,
        state="aligned",
        alignment_basis="reference_local_timeline",
        clock_sample_count=reference_samples,
        clock_valid_sample_count=reference_valid,
        clock_anomalies=reference_anomalies,
        uncertainty_ms=0.0,
        local_coverage_start_ms=0.0,
        local_coverage_end_ms=infos[reference.view_id].duration_ms,
        runtime={
            "shared_clock_sampling_seconds": round(clock_sampling_seconds, 6),
            "shared_preparation_seconds": round(shared_preparation_seconds, 6),
            "visual_audit_seconds": 0.0,
        },
    )
    reference_segments = infos[reference.view_id].segments or [
        type(
            "SingleSegment",
            (),
            {
                "virtual_start_ms": 0.0,
                "virtual_end_ms": infos[reference.view_id].duration_ms,
            },
        )()
    ]
    ref_transform.segment_transforms = [
        AlignmentSegmentTransform(
            segment_index=index,
            local_start_ms=float(segment.virtual_start_ms),
            local_end_ms=float(segment.virtual_end_ms),
            scale=1.0,
            offset_ms=reference.calibration_hint_ms,
            csv_sample_count=int(segment_audits_by_view[reference.view_id][index]["sample_count"]),
            csv_rmse_ms=0.0,
            confidence=1.0,
            uncertainty_ms=0.0,
            state="aligned",
            clock_anomalies=list(
                segment_audits_by_view[reference.view_id][index]["anomalies"]
            ),
        )
        for index, segment in enumerate(reference_segments)
    ]
    transforms = {reference.view_id: ref_transform}
    visual_signature_cache: dict[
        tuple[str, tuple[float, ...]], np.ndarray
    ] = {}
    for view in views:
        if view.view_id == reference.view_id:
            continue
        view_fit_started = time.perf_counter()
        sample_count, valid_count, clock_anomalies = combined_audit(view.view_id)
        has_absolute_clock = (
            reference_valid >= 2 and valid_count >= 2
        )
        pairs = _nearest_pairs(
            series[reference.view_id],
            series[view.view_id],
            float(align_cfg["nearest_tolerance_ms"]),
        )
        scale, offset, rmse = _robust_affine(pairs, float(align_cfg["max_drift_ppm"]))
        alignment_basis = "nearest_absolute_clock"
        if has_absolute_clock:
            absolute_fit = _absolute_clock_transform(
                series[reference.view_id],
                series[view.view_id],
                float(align_cfg["max_drift_ppm"]),
            )
            if absolute_fit is not None:
                scale, offset, rmse, absolute_count = absolute_fit
                pairs = [(0.0, 0.0)] * absolute_count
                alignment_basis = "absolute_clock_affine"
        if not pairs and bool(align_cfg.get("local_timeline_fallback_enabled", True)):
            pairs = _nearest_pairs(
                series[reference.view_id],
                series[view.view_id],
                float(align_cfg["nearest_tolerance_ms"]),
                allow_index_fallback=True,
            )
            scale, offset, rmse = _robust_affine(
                pairs, float(align_cfg["max_drift_ppm"])
            )
            alignment_basis = "local_timeline_assumption"
        offset += reference.calibration_hint_ms + view.calibration_hint_ms
        ratio = len(pairs) / max(1, min(len(series[reference.view_id]), len(series[view.view_id])))
        audits = segment_audits_by_view[view.view_id]
        segment_coverage = sum(
            int(item["valid_source_count"]) >= min(3, samples_per_segment)
            for item in audits
        ) / max(1, len(audits))
        evidence_density = min(
            1.0,
            valid_count / max(1, len(audits) * samples_per_segment),
        )
        anomaly_score = max(0.0, 1.0 - 0.20 * len(clock_anomalies))
        transform = AlignmentTransform(
            view_id=view.view_id,
            reference_view_id=reference.view_id,
            scale=scale,
            offset_ms=offset,
            csv_match_count=len(pairs),
            csv_match_ratio=ratio,
            csv_rmse_ms=None if not math.isfinite(rmse) else rmse,
            alignment_basis=alignment_basis,
            clock_sample_count=sample_count,
            clock_valid_sample_count=valid_count,
            clock_anomalies=clock_anomalies,
            local_coverage_start_ms=0.0,
            local_coverage_end_ms=infos[view.view_id].duration_ms,
        )
        match_score = min(
            1.0,
            ratio / max(float(align_cfg["minimum_match_ratio"]), 1e-9),
        )
        rmse_score = math.exp(
            -(
                (transform.csv_rmse_ms or 500.0)
                / max(float(align_cfg["nearest_tolerance_ms"]), 1.0)
            )
        )
        csv_confidence = (
            0.30 * match_score
            + 0.25 * rmse_score
            + 0.20 * evidence_density
            + 0.15 * segment_coverage
            + 0.10 * anomaly_score
        )
        if alignment_basis == "local_timeline_assumption":
            csv_confidence = min(
                csv_confidence,
                float(align_cfg.get("local_timeline_confidence_cap", 0.65)),
            )
        high_clock_confidence = csv_confidence >= float(
            align_cfg.get("visual_audit_csv_confidence_threshold", 0.80)
        )
        should_run_visual = bool(
            align_cfg.get("mandatory_visual_audit", True)
        ) or not high_clock_confidence
        visual_audit_started = time.perf_counter()
        if should_run_visual:
            if high_clock_confidence:
                visual_args = (
                    float(align_cfg.get("visual_audit_search_seconds", 1.0)),
                    int(align_cfg.get("visual_audit_anchor_count", 3)),
                    float(align_cfg.get("visual_audit_duration_seconds", 2.0)),
                    float(align_cfg.get("visual_audit_fps", 2.0)),
                )
            else:
                visual_args = (
                    float(align_cfg["visual_anchor_search_seconds"]),
                    int(align_cfg["visual_anchor_count"]),
                    float(align_cfg["visual_anchor_duration_seconds"]),
                    float(align_cfg["visual_anchor_fps"]),
                )
            try:
                correction, visual_confidence, details = visual_anchor_calibration(
                    reference,
                    view,
                    ref_transform,
                    transform,
                    infos[reference.view_id],
                    infos[view.view_id],
                    *visual_args,
                    signature_cache=visual_signature_cache,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                correction, visual_confidence = 0.0, 0.0
                details = [
                    {
                        "audit_failed": True,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ]
        else:
            correction, visual_confidence = 0.0, 0.0
            details = [{"deferred": True, "reason": "visual_audit_disabled"}]
        visual_audit_seconds = time.perf_counter() - visual_audit_started
        transform.visual_correction_ms = correction
        transform.visual_confidence = visual_confidence
        transform.anchor_details = details
        reliable_visual = sum(bool(item.get("reliable")) for item in details) >= 2
        if reliable_visual:
            transform.alignment_basis += "+visual_anchor"
        transform.confidence = float(
            min(1.0, 0.85 * csv_confidence + 0.15 * visual_confidence)
            if reliable_visual
            else csv_confidence
        )
        base_uncertainty = max(
            float(align_cfg.get("minimum_alignment_uncertainty_ms", 25.0)),
            (transform.csv_rmse_ms or float(align_cfg["nearest_tolerance_ms"])) * 2.0,
        )
        if alignment_basis == "local_timeline_assumption":
            base_uncertainty = max(
                base_uncertainty,
                float(align_cfg.get("local_timeline_uncertainty_ms", 1_000.0)),
            )
        if clock_anomalies:
            base_uncertainty += min(
                float(align_cfg.get("maximum_alignment_uncertainty_ms", 5_000.0)),
                len(clock_anomalies) * jump_threshold_ms,
            )
        transform.uncertainty_ms = min(
            base_uncertainty,
            float(align_cfg.get("maximum_alignment_uncertainty_ms", 5_000.0)),
        )
        if transform.confidence >= 0.65:
            transform.state = "aligned"
        elif pairs or details:
            transform.state = "uncertain"
            transform.failure_reason = "时间戳或视觉锚点置信度不足，已保留粗对齐并标注不确定"
        else:
            transform.state = "failed"
            transform.failure_reason = "无可用 CSV 匹配且视觉锚点失败"

        info_segments = infos[view.view_id].segments or [
            type(
                "SingleSegment",
                (),
                {
                    "virtual_start_ms": 0.0,
                    "virtual_end_ms": infos[view.view_id].duration_ms,
                },
            )()
        ]
        segment_fit_started = time.perf_counter()
        segment_transforms: list[AlignmentSegmentTransform] = []
        for segment_index, (segment, segment_points, segment_audit) in enumerate(
            zip(
                info_segments,
                segment_series_by_view[view.view_id],
                segment_audits_by_view[view.view_id],
                strict=True,
            )
        ):
            segment_anomalies = list(segment_audit["anomalies"])
            segment_fit = _absolute_clock_transform(
                series[reference.view_id],
                segment_points,
                float(align_cfg["max_drift_ppm"]),
            )
            if segment_fit is not None:
                segment_scale, segment_offset, segment_rmse, segment_matches = segment_fit
                segment_offset += reference.calibration_hint_ms + view.calibration_hint_ms
                segment_density = min(
                    1.0,
                    int(segment_audit["valid_source_count"]) / samples_per_segment,
                )
                segment_confidence = (
                    0.25
                    * math.exp(
                        -segment_rmse
                        / max(float(align_cfg["nearest_tolerance_ms"]), 1.0)
                    )
                    + 0.40 * segment_density
                    + 0.20 * float(segment_matches >= 3)
                    + 0.15 * float(not segment_anomalies)
                )
                segment_uncertainty = max(
                    float(align_cfg.get("minimum_alignment_uncertainty_ms", 25.0)),
                    segment_rmse * 2.0,
                )
                segment_state = (
                    "aligned" if segment_confidence >= 0.65 else "uncertain"
                )
                if any(
                    item
                    in {
                        "absolute_clock_not_strictly_increasing",
                        "segment_absolute_clock_regression",
                    }
                    for item in segment_anomalies
                ):
                    segment_state = "failed"
            else:
                segment_scale = transform.scale
                segment_offset = transform.offset_ms
                segment_rmse = transform.csv_rmse_ms
                segment_confidence = transform.confidence
                segment_uncertainty = transform.uncertainty_ms
                segment_state = transform.state
            segment_transforms.append(
                AlignmentSegmentTransform(
                    segment_index=segment_index,
                    local_start_ms=float(segment.virtual_start_ms),
                    local_end_ms=float(segment.virtual_end_ms),
                    scale=segment_scale,
                    offset_ms=segment_offset,
                    csv_sample_count=int(segment_audit["sample_count"]),
                    csv_rmse_ms=segment_rmse,
                    confidence=segment_confidence,
                    uncertainty_ms=min(
                        segment_uncertainty,
                        float(
                            align_cfg.get(
                                "maximum_alignment_uncertainty_ms", 5_000.0
                            )
                        ),
                    ),
                    state=segment_state,
                    clock_anomalies=segment_anomalies,
                )
            )
        transform.segment_transforms = segment_transforms
        transform.runtime = {
            "visual_audit_seconds": round(visual_audit_seconds, 6),
            "segment_fit_seconds": round(
                time.perf_counter() - segment_fit_started, 6
            ),
            "total_view_fit_seconds": round(
                time.perf_counter() - view_fit_started, 6
            ),
            "shared_signature_cache_entries": len(visual_signature_cache),
        }
        transforms[view.view_id] = transform
    return transforms, series


def _available_global_intervals(
    transform: AlignmentTransform,
    info: VideoInfo,
    *,
    aligned_only: bool = False,
) -> list[tuple[float, float]]:
    if transform.state == "failed" and not transform.segment_transforms:
        return []
    if transform.segment_transforms:
        intervals = [
            tuple(
                sorted(
                    (
                        segment.to_global(segment.local_start_ms)
                        + transform.visual_correction_ms,
                        segment.to_global(segment.local_end_ms)
                        + transform.visual_correction_ms,
                    )
                )
            )
            for segment in transform.segment_transforms
            if segment.state == "aligned"
            or (not aligned_only and segment.state != "failed")
        ]
    else:
        intervals = [
            tuple(
                sorted(
                    (
                        transform.to_global(0.0),
                        transform.to_global(info.duration_ms),
                    )
                )
            )
        ]
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def alignment_quality_report(
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    config: dict,
) -> dict[str, object]:
    """Evaluate formal-evidence readiness while allowing isolated-view quarantine."""

    align_cfg = config["alignment"]
    errors: list[str] = []
    warnings: list[str] = []
    view_receipts: list[dict[str, object]] = []
    role_intervals: dict[ViewRole, list[tuple[float, float]]] = {
        ViewRole.FIRST_PERSON: [],
        ViewRole.THIRD_PERSON: [],
    }
    for view in views:
        transform = transforms[view.view_id]
        intervals = _available_global_intervals(transform, infos[view.view_id])
        formal_intervals = _available_global_intervals(
            transform,
            infos[view.view_id],
            aligned_only=not bool(
                align_cfg.get("allow_uncertain_segment_formal_evidence", True)
            ),
        )
        failed_segments = [
            item.segment_index
            for item in transform.segment_transforms
            if item.state == "failed"
        ]
        local_assumption_without_visual = (
            transform.alignment_basis == "local_timeline_assumption"
        )
        formal_eligible = transform.state == "aligned" and not (
            local_assumption_without_visual
            and not bool(
                align_cfg.get("allow_local_timeline_formal_evidence", True)
            )
        )
        if formal_eligible:
            role_intervals[view.role].extend(formal_intervals)
            if not formal_intervals:
                warnings.append(f"{view.view_id}:no_formal_segment_coverage")
        else:
            warnings.append(
                f"{view.view_id}:alignment_not_formal:{transform.state}:"
                f"{transform.alignment_basis}"
            )
        if local_assumption_without_visual:
            warnings.append(f"{view.view_id}:local_timeline_assumption")
        if failed_segments:
            warnings.append(
                f"{view.view_id}:quarantined_segments="
                + ",".join(str(index) for index in failed_segments)
            )
        view_receipts.append(
            {
                "view_id": view.view_id,
                "role": view.role.value,
                "state": transform.state,
                "confidence": transform.confidence,
                "uncertainty_ms": transform.uncertainty_ms,
                "alignment_basis": transform.alignment_basis,
                "formal_alignment_eligible": formal_eligible,
                "available_intervals": [
                    {"start_ms": start, "end_ms": end, "duration_ms": end - start}
                    for start, end in intervals
                ],
                "quarantined_segment_indices": failed_segments,
            }
        )

    def merge_intervals(
        intervals: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    role_intervals = {
        role: merge_intervals(intervals)
        for role, intervals in role_intervals.items()
    }
    for role in (ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON):
        if not role_intervals[role]:
            errors.append(f"no_aligned_{role.value}_coverage")
    common_role_overlap_ms = 0.0
    first_intervals = role_intervals[ViewRole.FIRST_PERSON]
    third_intervals = role_intervals[ViewRole.THIRD_PERSON]
    first_index = 0
    third_index = 0
    while first_index < len(first_intervals) and third_index < len(third_intervals):
        first_start, first_end = first_intervals[first_index]
        third_start, third_end = third_intervals[third_index]
        common_role_overlap_ms += max(
            0.0,
            min(first_end, third_end) - max(first_start, third_start),
        )
        if first_end <= third_end:
            first_index += 1
        else:
            third_index += 1
    minimum_overlap_ms = float(
        align_cfg.get("minimum_required_role_overlap_ms", 1_000.0)
    )
    if common_role_overlap_ms < minimum_overlap_ms:
        errors.append(
            "insufficient_first_third_person_overlap:"
            f"{round(common_role_overlap_ms, 3)}<{round(minimum_overlap_ms, 3)}"
        )
    reference_view_id = next(iter(transforms.values())).reference_view_id
    reference_intervals = _available_global_intervals(
        transforms[reference_view_id], infos[reference_view_id]
    )
    master_start_ms = min((item[0] for item in reference_intervals), default=None)
    master_end_ms = max((item[1] for item in reference_intervals), default=None)
    master_duration_ms = (
        master_end_ms - master_start_ms
        if master_start_ms is not None and master_end_ms is not None
        else 0.0
    )
    minimum_coverage_ratio = float(
        align_cfg.get("view_coverage_warning_ratio", 0.999)
    )
    if master_duration_ms > 0:
        for receipt in view_receipts:
            available_duration_ms = sum(
                float(interval["duration_ms"])
                for interval in receipt["available_intervals"]
            )
            coverage_ratio = available_duration_ms / master_duration_ms
            receipt["master_timeline_coverage_ratio"] = coverage_ratio
            if coverage_ratio < minimum_coverage_ratio:
                warnings.append(
                    f"{receipt['view_id']}:partial_master_timeline_coverage:"
                    f"{round(coverage_ratio, 6)}"
                )
    degraded = bool(warnings)
    return {
        "schema_version": "visioncortex-alignment-quality-gate/1",
        "status": "failed" if errors else ("passed_degraded" if degraded else "passed"),
        "formal_evidence_ready": not errors,
        "reference_view_id": reference_view_id,
        "master_timeline": {
            "start_ms": master_start_ms,
            "end_ms": master_end_ms,
            "duration_ms": (
                master_duration_ms
            ),
            "policy": "reference_timeline_with_per_view_availability",
        },
        "common_first_third_person_overlap_ms": common_role_overlap_ms,
        "minimum_required_role_overlap_ms": minimum_overlap_ms,
        "errors": errors,
        "warnings": warnings,
        "views": view_receipts,
    }


def iter_aligned_rows(
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    output_fps: float,
) -> Iterable[dict[str, float | int | str | bool]]:
    reference_view_id = next(iter(transforms.values())).reference_view_id
    reference_intervals = _available_global_intervals(
        transforms[reference_view_id], infos[reference_view_id]
    )
    if not reference_intervals:
        return
    start = min(item[0] for item in reference_intervals)
    end = max(item[1] for item in reference_intervals)
    step = 1000.0 / output_fps
    index = 0
    global_ms = start
    while global_ms <= end:
        row: dict[str, float | int | str | bool] = {
            "alignment_index": index,
            "global_ms": round(global_ms, 3),
        }
        for view in views:
            transform = transforms[view.view_id]
            available = transform.is_available_at_global(global_ms)
            row[f"{view.view_id}_available"] = available
            if available:
                local_ms = transform.to_local(global_ms)
                row[f"{view.view_id}_local_ms"] = round(local_ms, 3)
                row[f"{view.view_id}_frame"] = virtual_frame_index_at(
                    infos[view.view_id], local_ms
                )
            else:
                row[f"{view.view_id}_local_ms"] = ""
                row[f"{view.view_id}_frame"] = ""
        yield row
        index += 1
        global_ms += step
