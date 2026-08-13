from __future__ import annotations

import csv
import math
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .schemas import AlignmentTransform, TimestampPoint, VideoInfo, ViewInput
from .video_io import motion_signature


FRAME_COLUMNS = ("video_frame_index", "frame_index", "frame", "frame_id", "index", "seq")
LOCAL_US_COLUMNS = ("local_time_us", "local_timestamp_us", "video_timestamp_us", "pts_us")
LOCAL_MS_COLUMNS = ("local_timestamp_ms", "video_timestamp_ms", "pts_ms", "timestamp_ms")
LOCAL_S_COLUMNS = ("local_timestamp_s", "video_timestamp_s", "pts_time", "timestamp_s", "time_s")
SOURCE_US_COLUMNS = ("frame_system_timestamp_us", "global_timestamp_us", "wallclock_us", "epoch_us")
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
        reader = csv.DictReader(handle, dialect=dialect)
        points: list[TimestampPoint] = []
        for row_number, row in enumerate(reader):
            if row_number % stride and row_number + 1 < row_count:
                continue
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
                    local_ms = float(local_text) * 1000.0 if local_text is not None else frame_index * 1000.0 / max(fps, 1e-9)
            if source_text is not None:
                source_ms = float(source_text) / 1000.0
            else:
                source_name, source_text = _first(row, SOURCE_MS_COLUMNS)
                if source_text is not None:
                    source_ms = float(source_text)
                else:
                    source_name, source_text = _first(row, SOURCE_COLUMNS)
                    source_ms = _parse_datetime_ms(source_text) if source_text is not None else None
            if source_ms is None and local_name == "timestamp_ms" and abs(local_ms) > 10_000_000_000:
                source_ms = local_ms
                local_ms = frame_index * 1000.0 / max(fps, 1e-9)
            if source_ms is None and local_name == "timestamp_s" and abs(local_ms) > 10_000_000_000:
                source_ms = local_ms
                local_ms = frame_index * 1000.0 / max(fps, 1e-9)
            points.append(TimestampPoint(frame_index=frame_index, local_ms=local_ms, source_ms=source_ms))
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


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 3:
        return -1.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-9 else -1.0


def visual_anchor_calibration(
    reference_video: Path,
    target_video: Path,
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
        ref_signal = motion_signature(reference_video, ref_times)
        target_signal = motion_signature(target_video, target_times)
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
    series: dict[str, list[TimestampPoint]] = {}
    for view in views:
        series[view.view_id] = (
            read_timestamp_csv(view.timestamps_csv, infos[view.view_id].fps)
            if view.timestamps_csv
            else synthetic_timestamps(infos[view.view_id], sample_fps=align_cfg["aligned_timestamps_fps"])
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
        correction, visual_confidence, details = visual_anchor_calibration(
            reference.video,
            view.video,
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
        csv_score = min(1.0, ratio / max(float(align_cfg["minimum_match_ratio"]), 1e-9))
        rmse_score = math.exp(-((transform.csv_rmse_ms or 500.0) / max(float(align_cfg["nearest_tolerance_ms"]), 1.0)))
        transform.confidence = float(0.65 * csv_score + 0.20 * rmse_score + 0.15 * visual_confidence)
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
            row[f"{view.view_id}_frame"] = int(round(local_ms / 1000.0 * infos[view.view_id].fps))
        yield row
        index += 1
        global_ms += step
