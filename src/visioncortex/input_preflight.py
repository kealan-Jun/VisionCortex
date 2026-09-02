from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .alignment import read_timestamp_csv_endpoints
from .schemas import RunManifest, VideoInfo, ViewInput
from .video_io import probe_views


def _view_sources(view: ViewInput) -> list[tuple[Path, Path | None]]:
    if view.segments:
        return [(segment.video, segment.timestamps_csv) for segment in view.segments]
    assert view.video is not None
    return [(view.video, view.timestamps_csv)]


def _segment_infos(info: VideoInfo) -> list[VideoInfo | Any]:
    return list(info.segments) if info.segments else [info]


def preflight_manifest_inputs(
    manifest: RunManifest,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Probe media and bounded CSV endpoints before a job can occupy the GPU queue."""

    started = time.perf_counter()
    performance = config.get("performance") or {}
    upload = config.get("web_upload") or {}
    workers = max(
        1,
        min(
            12,
            int(
                upload.get(
                    "prequeue_probe_workers",
                    performance.get("ffprobe_workers", 4),
                )
            ),
        ),
    )
    media_probe_started = time.perf_counter()
    infos = probe_views(manifest.views, workers=workers, prefer_clock_metadata=True)
    media_probe_seconds = time.perf_counter() - media_probe_started
    clock_preflight_started = time.perf_counter()
    views: list[dict[str, Any]] = []
    warnings: list[str] = []
    clock_overlap_tolerance_ms = max(
        0.0, float(upload.get("prequeue_clock_overlap_tolerance_ms", 2_500.0))
    )
    source_windows: list[tuple[str, float, float]] = []
    for view in manifest.views:
        info = infos[view.view_id]
        sources = _view_sources(view)
        segment_infos = _segment_infos(info)
        if len(sources) != len(segment_infos):
            raise ValueError(f"媒体探测分片数量不一致: {view.view_id}")
        segment_receipts: list[dict[str, Any]] = []
        previous_source_end: float | None = None
        view_source_starts: list[float] = []
        view_source_ends: list[float] = []
        for ordinal, ((video, clock), segment_info) in enumerate(
            zip(sources, segment_infos, strict=True), 1
        ):
            duration_ms = float(segment_info.duration_ms)
            fps = float(segment_info.fps)
            width = int(segment_info.width)
            height = int(segment_info.height)
            frame_count = int(segment_info.frame_count)
            if duration_ms <= 0 or fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
                raise ValueError(
                    f"视频不可用于分析: {view.view_id} 第 {ordinal} 段 "
                    f"duration={duration_ms}, fps={fps}, size={width}x{height}, frames={frame_count}"
                )
            clock_receipt: dict[str, Any] | None = None
            if clock is not None:
                endpoints = read_timestamp_csv_endpoints(clock, fps)
                first, last = endpoints[0], endpoints[-1]
                if last.local_ms < first.local_ms or last.frame_index < first.frame_index:
                    raise ValueError(
                        f"时间戳 CSV 首尾顺序无效: {view.view_id} 第 {ordinal} 段"
                    )
                source_start = (
                    float(first.source_ms) if first.source_ms is not None else None
                )
                source_end = (
                    float(last.source_ms) if last.source_ms is not None else None
                )
                if source_start is not None and source_end is not None:
                    if source_end < source_start:
                        raise ValueError(
                            f"时间戳 CSV 绝对时钟倒退: {view.view_id} 第 {ordinal} 段"
                        )
                    if (
                        previous_source_end is not None
                        and source_start + clock_overlap_tolerance_ms
                        < previous_source_end
                    ):
                        raise ValueError(
                            f"分片时钟顺序重叠或倒退: {view.view_id} 第 {ordinal} 段"
                        )
                    if previous_source_end is not None and source_start < previous_source_end:
                        warnings.append(
                            f"{view.view_id}: segment_{ordinal}_clock_overlap_within_tolerance"
                        )
                    previous_source_end = source_end
                    view_source_starts.append(source_start)
                    view_source_ends.append(source_end)
                clock_receipt = {
                    "path": str(clock),
                    "first_frame_index": int(first.frame_index),
                    "last_frame_index": int(last.frame_index),
                    "first_local_ms": float(first.local_ms),
                    "last_local_ms": float(last.local_ms),
                    "source_start_ms": source_start,
                    "source_end_ms": source_end,
                }
            segment_receipts.append(
                {
                    "ordinal": ordinal,
                    "video": str(video),
                    "duration_ms": duration_ms,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "frame_count": frame_count,
                    "media_timing_source": segment_info.media_timing_source,
                    "clock": clock_receipt,
                }
            )
        if view_source_starts and view_source_ends:
            source_windows.append(
                (view.view_id, min(view_source_starts), max(view_source_ends))
            )
        else:
            warnings.append(f"{view.view_id}: no_absolute_clock_coverage")
        views.append(
            {
                "view_id": view.view_id,
                "role": view.role.value,
                "segment_count": len(segment_receipts),
                "duration_ms": float(info.duration_ms),
                "segments": segment_receipts,
            }
        )
    common_clock_overlap: dict[str, Any] | None = None
    if len(source_windows) >= 2:
        overlap_start = max(window[1] for window in source_windows)
        overlap_end = min(window[2] for window in source_windows)
        if overlap_end <= overlap_start:
            raise ValueError(
                "各视角时间戳 CSV 没有共同绝对时钟覆盖，不能进入 GPU 队列"
            )
        common_clock_overlap = {
            "start_ms": overlap_start,
            "end_ms": overlap_end,
            "duration_ms": overlap_end - overlap_start,
            "view_count": len(source_windows),
        }
    elif source_windows:
        warnings.append("only_one_view_has_absolute_clock_coverage")
    else:
        warnings.append("visual_alignment_required_no_absolute_clock_coverage")
    clock_preflight_seconds = time.perf_counter() - clock_preflight_started
    return {
        "schema_version": "visioncortex-prequeue-input-preflight/1",
        "status": "passed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "probe_workers": workers,
        "runtime": {
            "media_probe_seconds": round(media_probe_seconds, 6),
            "clock_preflight_seconds": round(clock_preflight_seconds, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
        },
        "clock_overlap_tolerance_ms": clock_overlap_tolerance_ms,
        "view_count": len(views),
        "video_segment_count": sum(view["segment_count"] for view in views),
        "clock_file_count": sum(
            segment["clock"] is not None
            for view in views
            for segment in view["segments"]
        ),
        "common_clock_overlap": common_clock_overlap,
        "warnings": warnings,
        "views": views,
    }
