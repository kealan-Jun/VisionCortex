from __future__ import annotations

import csv
import math
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .schemas import AlignmentTransform, TimestampPoint, VideoInfo, ViewInput
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


def _timestamp_point(row: dict[str, str], row_number: int, fps: float) -> TimestampPoint:
    _, frame_text = _first(row, FRAME_COLUMNS)
    frame_index = int(float(frame_text)) if frame_text is not None else row_number
    local_name, local_text = _first(row, LOCAL_US_COLUMNS)
    _source_name, source_text = _first(row, SOURCE_US_COLUMNS)
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
        _source_name, source_text = _first(row, SOURCE_MS_COLUMNS)
        if source_text is not None:
            source_ms = float(source_text)
        else:
            _source_name, source_text = _first(row, SOURCE_COLUMNS)
            source_ms = _parse_datetime_ms(source_text) if source_text is not None else None
    # Capture exports can put epoch wall-clock values in local_time_us.
    # Playback stays on the media frame timeline even when a separate,
    # preferred capture-clock column is also present.
    if abs(local_ms) > 10_000_000_000:
        if source_ms is None:
            source_ms = local_ms
        local_ms = frame_index * 1000.0 / max(fps, 1e-9)
    return TimestampPoint(frame_index=frame_index, local_ms=local_ms, source_ms=source_ms)


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
    reference: Sequence[TimestampPoint], target: Sequence[TimestampPoint], tolerance_ms: float
) -> list[tuple[float, float]]:
    ref_has_source = sum(point.source_ms is not None for point in reference) >= 2
    target_has_source = sum(point.source_ms is not None for point in target) >= 2
    if not (ref_has_source and target_has_source):
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
    for center in centers:
        reference_globals = center + np.arange(-half_count, half_count + 1) * 1000.0 / fps
        extended_globals = center + np.arange(-half_count - max_lag, half_count + max_lag + 1) * 1000.0 / fps
        ref_times = [reference_transform.to_local(float(value)) for value in reference_globals]
        target_times = [target_transform.to_local(float(value)) for value in extended_globals]
        ref_signal = view_motion_signature(reference_view, reference_info, ref_times)
        target_signal = view_motion_signature(target_view, target_info, target_times)
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
    align_cfg = config["alignment"]
    reference = next(
        (view for view in views if view.view_id == align_cfg["reference_view"]),
        next(view for view in views if view.role.value == align_cfg["reference_view"]),
    )
    endpoint_cache: dict[Path, list[TimestampPoint]] = {}
    if bool(align_cfg.get("csv_segment_endpoints_only", True)):
        endpoint_jobs: dict[Path, float] = {}
        for view in views:
            info = infos[view.view_id]
            for segment in info.segments:
                if segment.timestamps_csv:
                    endpoint_jobs[segment.timestamps_csv] = segment.fps
        if endpoint_jobs:
            with ThreadPoolExecutor(
                max_workers=max(
                    1,
                    min(
                        int(align_cfg.get("csv_read_workers", 12)),
                        len(endpoint_jobs),
                    ),
                ),
                thread_name_prefix="clock-endpoints",
            ) as executor:
                futures = {
                    executor.submit(read_timestamp_csv_endpoints, path, fps): path
                    for path, fps in endpoint_jobs.items()
                }
                for future in as_completed(futures):
                    endpoint_cache[futures[future]] = future.result()

    series: dict[str, list[TimestampPoint]] = {}
    for view in views:
        info = infos[view.view_id]
        if view.segments and info.segments:
            points: list[TimestampPoint] = []
            segment_series: list[list[TimestampPoint] | None] = []
            max_view_points = max(2_000, int(align_cfg.get("max_csv_points_per_view", 30_000)))
            points_per_segment = max(512, math.ceil(max_view_points / len(info.segments)))
            for segment in info.segments:
                if segment.timestamps_csv:
                    segment_series.append(
                        endpoint_cache.get(segment.timestamps_csv)
                        or read_timestamp_csv_endpoints(segment.timestamps_csv, segment.fps)
                        if bool(align_cfg.get("csv_segment_endpoints_only", True))
                        else read_timestamp_csv(
                            segment.timestamps_csv,
                            segment.fps,
                            max_points=points_per_segment,
                        )
                    )
                else:
                    segment_series.append(None)
            source_origins = [
                point.source_ms
                for source_points in segment_series
                if source_points
                for point in source_points[:1]
                if point.source_ms is not None
            ]
            base_source_ms = source_origins[0] if source_origins else None
            previous_start_ms = -1.0
            for segment, source_points in zip(info.segments, segment_series, strict=True):
                if source_points and base_source_ms is not None and source_points[0].source_ms is not None:
                    clock_start_ms = source_points[0].source_ms - base_source_ms
                    # Reject corrupt clock jumps but preserve real recorder gaps/overlaps.
                    if (
                        clock_start_ms >= previous_start_ms
                        and abs(clock_start_ms - segment.virtual_start_ms) <= 300_000.0
                    ):
                        segment.virtual_start_ms = max(0.0, clock_start_ms)
                        segment.virtual_end_ms = segment.virtual_start_ms + segment.duration_ms
                previous_start_ms = segment.virtual_start_ms
                if source_points:
                    origin = source_points[0].local_ms
                    points.extend(
                        TimestampPoint(
                            frame_index=segment.frame_start_index + point.frame_index,
                            local_ms=segment.virtual_start_ms + point.local_ms - origin,
                            source_ms=point.source_ms,
                        )
                        for point in source_points
                    )
                else:
                    points.extend(
                        TimestampPoint(
                            frame_index=segment.frame_start_index + point.frame_index,
                            local_ms=segment.virtual_start_ms + point.local_ms,
                            source_ms=point.source_ms,
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
                    )
            info.duration_ms = max(segment.virtual_end_ms for segment in info.segments)
            series[view.view_id] = points
        else:
            series[view.view_id] = (
                read_timestamp_csv(
                    view.timestamps_csv,
                    info.fps,
                    max_points=max(2_000, int(align_cfg.get("max_csv_points_per_view", 30_000))),
                )
                if view.timestamps_csv
                else synthetic_timestamps(info, sample_fps=align_cfg["aligned_timestamps_fps"])
            )
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
    )
    transforms = {reference.view_id: ref_transform}
    for view in views:
        if view.view_id == reference.view_id:
            continue
        pairs = _nearest_pairs(
            series[reference.view_id], series[view.view_id], float(align_cfg["nearest_tolerance_ms"])
        )
        scale, offset, rmse = _robust_affine(pairs, float(align_cfg["max_drift_ppm"]))
        if bool(align_cfg.get("csv_segment_endpoints_only", True)):
            absolute_fit = _absolute_clock_transform(
                series[reference.view_id],
                series[view.view_id],
                float(align_cfg["max_drift_ppm"]),
            )
            if absolute_fit is not None:
                scale, offset, rmse, absolute_count = absolute_fit
                pairs = [(0.0, 0.0)] * absolute_count
        offset += view.calibration_hint_ms
        ratio = len(pairs) / max(1, min(len(series[reference.view_id]), len(series[view.view_id])))
        transform = AlignmentTransform(
            view_id=view.view_id,
            reference_view_id=reference.view_id,
            scale=scale,
            offset_ms=offset,
            csv_match_count=len(pairs),
            csv_match_ratio=ratio,
            csv_rmse_ms=None if not math.isfinite(rmse) else rmse,
        )
        csv_score = min(1.0, ratio / max(float(align_cfg["minimum_match_ratio"]), 1e-9))
        rmse_score = math.exp(
            -(
                (transform.csv_rmse_ms or 500.0)
                / max(float(align_cfg["nearest_tolerance_ms"]), 1.0)
            )
        )
        csv_confidence = 0.75 * csv_score + 0.25 * rmse_score
        use_short_visual_audit = csv_confidence >= float(
            align_cfg.get("visual_audit_csv_confidence_threshold", 0.80)
        )
        if use_short_visual_audit:
            # Do not open the same large SMB MP4s a second time merely to audit a
            # CSV solution that is already strong. The bounded all-view YOLO/fine
            # scan later in the pipeline supplies the visual cross-view evidence
            # on the exact candidate windows. This turns visual calibration into
            # reuse of mandatory work instead of a separate full-input seek pass.
            correction, visual_confidence = 0.0, 0.0
            details = [
                {
                    "deferred": True,
                    "reason": "high_confidence_csv_reuses_bounded_cross_view_scan",
                    "csv_confidence": csv_confidence,
                }
            ]
        else:
            correction, visual_confidence, details = visual_anchor_calibration(
                reference,
                view,
                ref_transform,
                transform,
                infos[reference.view_id],
                infos[view.view_id],
                float(align_cfg["visual_anchor_search_seconds"]),
                int(align_cfg["visual_anchor_count"]),
                float(align_cfg["visual_anchor_duration_seconds"]),
                float(align_cfg["visual_anchor_fps"]),
            )
        transform.visual_correction_ms = correction
        transform.visual_confidence = visual_confidence
        transform.anchor_details = details
        transform.confidence = float(
            0.75 * csv_score + 0.25 * rmse_score
            if use_short_visual_audit
            else 0.65 * csv_score + 0.20 * rmse_score + 0.15 * visual_confidence
        )
        if transform.confidence >= 0.65:
            transform.state = "aligned"
        elif pairs or details:
            transform.state = "uncertain"
            transform.failure_reason = "时间戳或视觉锚点置信度不足，已保留粗对齐并标注不确定"
        else:
            transform.state = "failed"
            transform.failure_reason = "无可用 CSV 匹配且视觉锚点失败"
        transforms[view.view_id] = transform
    return transforms, series


def iter_aligned_rows(
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    output_fps: float,
) -> Iterable[dict[str, float | int | str]]:
    start = max(transform.to_global(0.0) for transform in transforms.values())
    end = min(transforms[view.view_id].to_global(infos[view.view_id].duration_ms) for view in views)
    step = 1000.0 / output_fps
    index = 0
    global_ms = start
    while global_ms <= end:
        row: dict[str, float | int | str] = {"alignment_index": index, "global_ms": round(global_ms, 3)}
        for view in views:
            local_ms = transforms[view.view_id].to_local(global_ms)
            row[f"{view.view_id}_local_ms"] = round(local_ms, 3)
            row[f"{view.view_id}_frame"] = virtual_frame_index_at(
                infos[view.view_id], local_ms
            )
        yield row
        index += 1
        global_ms += step
