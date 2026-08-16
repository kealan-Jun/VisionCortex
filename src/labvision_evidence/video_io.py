from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import VideoInfo, VideoSegmentInfo, ViewInput
from .storage import read_source_file_edges


@dataclass(frozen=True)
class PhysicalSegmentDecodeSession:
    """One physical media open serving several disjoint virtual-timeline windows."""

    segment_index: int
    path: Path
    virtual_start_ms: float
    virtual_end_ms: float
    source_start_ms: float
    source_end_ms: float
    target_virtual_windows: tuple[tuple[float, float], ...]
    target_source_windows: tuple[tuple[float, float], ...]

    @property
    def selected_duration_ms(self) -> float:
        return sum(end - start for start, end in self.target_virtual_windows)


def _run(command: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, capture_output=True, check=False, timeout=timeout)


def probe_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise FileNotFoundError(f"视频不存在: {path}")
    if shutil.which("ffprobe"):
        result = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        if result.returncode == 0:
            payload = json.loads(result.stdout.decode("utf-8", errors="replace"))
            stream = payload["streams"][0]
            numerator, denominator = (stream.get("avg_frame_rate") or "0/1").split("/")
            fps = float(numerator) / max(float(denominator), 1.0)
            duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
            frame_count = int(stream.get("nb_frames") or round(duration * fps))
            return VideoInfo(
                path=path,
                duration_ms=duration * 1000.0,
                fps=fps,
                width=int(stream["width"]),
                height=int(stream["height"]),
                frame_count=frame_count,
                size_bytes=path.stat().st_size,
            )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取视频: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return VideoInfo(
        path=path,
        duration_ms=(frames / max(fps, 1e-9)) * 1000.0,
        fps=fps,
        width=width,
        height=height,
        frame_count=frames,
        size_bytes=path.stat().st_size,
    )


def view_source_files(view: ViewInput) -> list[Path]:
    if view.segments:
        return [segment.video for segment in view.segments]
    assert view.video is not None
    return [view.video]


def view_timestamp_files(view: ViewInput) -> list[Path]:
    if view.segments:
        return [segment.timestamps_csv for segment in view.segments if segment.timestamps_csv]
    return [view.timestamps_csv] if view.timestamps_csv else []


def _build_segmented_view_info(view: ViewInput, probed: Sequence[VideoInfo]) -> VideoInfo:
    first = probed[0]
    virtual_start_ms = 0.0
    frame_start = 0
    segment_infos: list[VideoSegmentInfo] = []
    for source, media in zip(view.segments, probed, strict=True):
        if (media.width, media.height) != (first.width, first.height):
            raise ValueError(
                f"segment resolution changed in {view.view_id}: "
                f"{media.path} is {media.width}x{media.height}, expected {first.width}x{first.height}"
            )
        if abs(media.fps - first.fps) > 0.05:
            raise ValueError(
                f"segment frame rate changed in {view.view_id}: "
                f"{media.path} is {media.fps}, expected {first.fps}"
            )
        segment_infos.append(
            VideoSegmentInfo(
                path=media.path,
                timestamps_csv=source.timestamps_csv,
                virtual_start_ms=virtual_start_ms,
                virtual_end_ms=virtual_start_ms + media.duration_ms,
                frame_start_index=frame_start,
                duration_ms=media.duration_ms,
                fps=media.fps,
                width=media.width,
                height=media.height,
                frame_count=media.frame_count,
                size_bytes=media.size_bytes,
            )
        )
        virtual_start_ms += media.duration_ms
        frame_start += media.frame_count
    return VideoInfo(
        path=first.path,
        duration_ms=virtual_start_ms,
        fps=first.fps,
        width=first.width,
        height=first.height,
        frame_count=frame_start,
        size_bytes=sum(item.size_bytes for item in segment_infos),
        segments=segment_infos,
    )


def probe_view(view: ViewInput) -> VideoInfo:
    """Probe a single file or build a zero-copy virtual timeline from segments."""

    if not view.segments:
        assert view.video is not None
        return probe_video(view.video)
    return _build_segmented_view_info(
        view, [probe_video(segment.video) for segment in view.segments]
    )


def _clock_metadata_video_info(video: Path, clock: Path | None) -> VideoInfo | None:
    """Build exact RGB media timing from a recorder CSV without scanning the MP4."""

    if clock is None:
        return None
    try:
        head, tail, size = read_source_file_edges(clock)
        head_lines = head.decode("utf-8-sig", errors="replace").splitlines()
        tail_lines = tail.decode("utf-8", errors="replace").splitlines()
        if len(head_lines) < 2:
            return None
        fieldnames = next(csv.reader([head_lines[0]]))
        indexes = {name.strip(): index for index, name in enumerate(fieldnames)}
        required = {
            "rgb_video_frame_index",
            "rgb_recorded",
            "width",
            "height",
        }
        if not required.issubset(indexes):
            return None

        def rgb_endpoint(lines: Sequence[str], *, reverse: bool = False) -> list[str] | None:
            candidates = reversed(lines) if reverse else iter(lines)
            for line in candidates:
                values = next(csv.reader([line]), [])
                if len(values) <= max(indexes.values()):
                    continue
                if values[indexes["rgb_recorded"]].strip().lower() not in {"1", "true", "yes"}:
                    continue
                if not values[indexes["rgb_video_frame_index"]].strip():
                    continue
                return values
            return None

        first = rgb_endpoint(head_lines[1:])
        last = rgb_endpoint(
            tail_lines[1:] if size > len(tail) else tail_lines,
            reverse=True,
        )
        if first is None or last is None:
            return None

        def number(row: list[str], name: str) -> float:
            return float(row[indexes[name]])

        first_index = int(number(first, "rgb_video_frame_index"))
        last_index = int(number(last, "rgb_video_frame_index"))
        frame_count = last_index - first_index + 1
        clock_name = next(
            (
                name
                for name in ("global_timestamp_us", "rgb_system_timestamp_us", "local_time_us")
                if name in indexes
                and first[indexes[name]].strip()
                and last[indexes[name]].strip()
            ),
            None,
        )
        elapsed_seconds = (
            max(0.0, (number(last, clock_name) - number(first, clock_name)) / 1_000_000.0)
            if clock_name is not None
            else 0.0
        )
        fps_index = indexes.get("rgb_actual_fps")
        fps_values = [
            float(row[fps_index])
            for row in (first, last)
            if fps_index is not None and row[fps_index].strip()
        ]
        fps = fps_values[-1] if fps_values else 0.0
        if fps <= 0 and elapsed_seconds > 0 and frame_count > 1:
            fps = (frame_count - 1) / elapsed_seconds
        if fps <= 0 or frame_count <= 1:
            return None
        nearest_integer_fps = round(fps)
        if nearest_integer_fps > 0 and abs(fps - nearest_integer_fps) <= 0.2:
            fps = float(nearest_integer_fps)
        duration_seconds = frame_count / fps
        return VideoInfo(
            path=video,
            duration_ms=duration_seconds * 1000.0,
            fps=fps,
            width=int(number(first, "width")),
            height=int(number(first, "height")),
            frame_count=frame_count,
            size_bytes=video.stat().st_size,
        )
    except (OSError, UnicodeError, csv.Error, ValueError, IndexError):
        return None


def probe_views(
    views: Sequence[ViewInput],
    workers: int = 12,
    prefer_clock_metadata: bool = True,
) -> dict[str, VideoInfo]:
    """Probe recorder segments concurrently instead of 15 serial ffprobes per view."""

    jobs: list[tuple[str, int, Path, Path | None]] = []
    for view in views:
        if view.segments:
            jobs.extend(
                (view.view_id, index, segment.video, segment.timestamps_csv)
                for index, segment in enumerate(view.segments)
            )
        else:
            assert view.video is not None
            jobs.append((view.view_id, 0, view.video, view.timestamps_csv))

    def inspect(video: Path, clock: Path | None) -> VideoInfo:
        if prefer_clock_metadata:
            from_clock = _clock_metadata_video_info(video, clock)
            if from_clock is not None:
                return from_clock
        return probe_video(video)

    probed: dict[str, dict[int, VideoInfo]] = {view.view_id: {} for view in views}
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(workers), len(jobs))),
        thread_name_prefix="ffprobe",
    ) as executor:
        futures = {
            executor.submit(inspect, path, clock): (view_id, index)
            for view_id, index, path, clock in jobs
        }
        for future in as_completed(futures):
            view_id, index = futures[future]
            probed[view_id][index] = future.result()
    result: dict[str, VideoInfo] = {}
    for view in views:
        ordered = [probed[view.view_id][index] for index in sorted(probed[view.view_id])]
        result[view.view_id] = (
            _build_segmented_view_info(view, ordered) if view.segments else ordered[0]
        )
    return result


def estimate_disk_need(infos: Sequence[VideoInfo]) -> int:
    """Temporary JSONL, frames and clips; source files themselves are not copied."""
    source_bytes = sum(info.size_bytes for info in infos)
    duration_hours = sum(info.duration_ms for info in infos) / 3_600_000.0
    candidate_bytes = int(duration_hours * 650 * 1024 * 1024)
    delivery_bytes = int(source_bytes * 0.35)
    return candidate_bytes + delivery_bytes + 2 * 1024**3


def check_disk_capacity(output_root: Path, infos: Sequence[VideoInfo]) -> dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    required = estimate_disk_need(infos)
    if usage.free < required:
        raise RuntimeError(
            f"输出盘空间不足: 预计至少需要 {required / 1024**3:.1f} GiB，"
            f"当前可用 {usage.free / 1024**3:.1f} GiB"
        )
    return {"required_bytes": required, "free_bytes": usage.free}


def _scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if width <= max_width:
        return width - width % 2, height - height % 2
    scaled_h = int(round(height * max_width / width))
    return max_width - max_width % 2, scaled_h - scaled_h % 2


def _ffmpeg_frame_iterator(
    path: Path,
    info: VideoInfo,
    start_ms: float,
    end_ms: float,
    sample_fps: float,
    max_width: int,
    hwaccel: str | None,
    keyframes_only: bool,
    decoder_threads: int | None,
    cuda_scale: bool = False,
) -> Iterator[tuple[int, float, np.ndarray]]:
    width, height = _scaled_size(info.width, info.height, max_width)
    use_cuda_scale = bool(cuda_scale and hwaccel == "cuda")
    filter_graph = (
        f"fps={sample_fps:.8f},scale_cuda={width}:{height}:format=nv12,"
        "hwdownload,format=nv12,format=bgr24"
        if use_cuda_scale
        else f"fps={sample_fps:.8f},scale={width}:{height}"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        command += ["-hwaccel", hwaccel]
        if use_cuda_scale:
            command += ["-hwaccel_output_format", "cuda"]
    elif decoder_threads:
        command += ["-threads", str(max(1, decoder_threads))]
    if keyframes_only:
        command += ["-skip_frame", "nokey"]
    command += [
        "-ss",
        f"{start_ms / 1000.0:.6f}",
        "-i",
        str(path),
        "-t",
        f"{max(0.0, end_ms - start_ms) / 1000.0:.6f}",
        "-vf",
        filter_graph,
        "-an",
        "-sn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    frame_bytes = width * height * 3
    index = 0
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            local_ms = start_ms + index * 1000.0 / sample_fps
            frame_index = int(round(local_ms * info.fps / 1000.0))
            yield frame_index, local_ms, frame
            index += 1
    finally:
        process.stdout.close()
        process.wait()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
    if process.returncode not in (0, None):
        message = stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"FFmpeg 抽帧失败: {message}")


def _merge_windows(
    windows: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(
        (float(start), float(end)) for start, end in windows if end > start
    ):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def plan_physical_segment_decode_sessions(
    info: VideoInfo,
    windows: Sequence[tuple[float, float]],
) -> list[PhysicalSegmentDecodeSession]:
    """Group target windows by recorder MP4 without widening the inference set."""

    if not info.segments:
        raise ValueError("physical segment sessions require segmented video info")
    sessions: list[PhysicalSegmentDecodeSession] = []
    for segment_index, segment in enumerate(info.segments):
        virtual_windows = _merge_windows(
            [
                (
                    max(float(start), segment.virtual_start_ms),
                    min(float(end), segment.virtual_end_ms),
                )
                for start, end in windows
                if min(float(end), segment.virtual_end_ms)
                > max(float(start), segment.virtual_start_ms)
            ]
        )
        if not virtual_windows:
            continue
        source_windows = tuple(
            (
                start - segment.virtual_start_ms,
                end - segment.virtual_start_ms,
            )
            for start, end in virtual_windows
        )
        sessions.append(
            PhysicalSegmentDecodeSession(
                segment_index=segment_index,
                path=segment.path,
                virtual_start_ms=virtual_windows[0][0],
                virtual_end_ms=virtual_windows[-1][1],
                source_start_ms=source_windows[0][0],
                source_end_ms=source_windows[-1][1],
                target_virtual_windows=tuple(virtual_windows),
                target_source_windows=source_windows,
            )
        )
    return sessions


def _selected_session_timestamps(
    start_ms: float,
    end_ms: float,
    windows: Sequence[tuple[float, float]],
    sample_fps: float,
) -> list[float]:
    period_ms = 1000.0 / max(sample_fps, 1e-9)
    return [
        start_ms + index * period_ms
        for index in _selected_session_frame_indices(
            start_ms, end_ms, windows, sample_fps
        )
    ]


def _selected_session_frame_indices(
    start_ms: float,
    end_ms: float,
    windows: Sequence[tuple[float, float]],
    sample_fps: float,
) -> list[int]:
    """Return the exact post-``fps`` frame indices selected by the ledger.

    FFmpeg's ``select`` time variable is quantized to its filter time base. A
    six-decimal floating window boundary can therefore include or exclude one
    different frame than the Python half-open comparison, especially after a
    non-identity clock transform. Selecting the integer ``n`` produced by the
    immediately preceding ``fps`` filter makes media and ledger use one
    deterministic decision.
    """

    period_ms = 1000.0 / max(sample_fps, 1e-9)
    # The persistent FFmpeg path uses fps=...:round=near:eof_action=round.
    # Mirror that positive-duration rounding exactly.  Using ceil here made a
    # fractional endpoint (205.633333 s at 10 FPS) demand 2057 ledger rows even
    # though FFmpeg correctly emits 2056, aborting an otherwise valid session.
    scaled_frame_count = max(0.0, (end_ms - start_ms) / period_ms)
    frame_count = int(math.floor(scaled_frame_count + 0.5))
    return [
        index
        for index in range(frame_count)
        if any(
            window_start - 1e-6 <= start_ms + index * period_ms < window_end - 1e-6
            for window_start, window_end in windows
        )
    ]


def _consecutive_index_ranges(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    range_start = previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
            continue
        ranges.append((range_start, previous))
        range_start = previous = current
    ranges.append((range_start, previous))
    return ranges


def _aligned_grid_start_ms(
    start_ms: float,
    grid_origin_ms: float,
    period_ms: float,
) -> float:
    """Return the first half-open sampling-grid timestamp at or after start."""

    if period_ms <= 0.0:
        raise ValueError("sampling grid period must be positive")
    grid_index = math.ceil((start_ms - grid_origin_ms) / period_ms - 1e-9)
    aligned = grid_origin_ms + grid_index * period_ms
    if aligned < start_ms - 1e-6:
        aligned += period_ms
    return aligned


def _ffmpeg_multi_window_iterator(
    path: Path,
    info: VideoInfo,
    start_ms: float,
    end_ms: float,
    windows: Sequence[tuple[float, float]],
    sample_fps: float,
    max_width: int,
    hwaccel: str | None,
    decoder_threads: int | None,
    cuda_scale: bool = False,
    receipt: dict[str, Any] | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Decode one physical span once and emit only its requested 10 FPS windows."""

    normalized = _merge_windows(
        [
            (max(start_ms, start), min(end_ms, end))
            for start, end in windows
            if min(end_ms, end) > max(start_ms, start)
        ]
    )
    if not normalized:
        return
    width, height = _scaled_size(info.width, info.height, max_width)
    use_cuda_scale = bool(cuda_scale and hwaccel == "cuda")
    expected_indices = _selected_session_frame_indices(
        start_ms, end_ms, normalized, sample_fps
    )
    expected_timestamps = _selected_session_timestamps(
        start_ms, end_ms, normalized, sample_fps
    )
    index_ranges = _consecutive_index_ranges(expected_indices)
    if receipt is not None:
        receipt.update(
            {
                "fps_rounding_policy": "round=near:eof_action=round",
                "frame_selection_policy": "post_fps_integer_indices_half_open",
                "expected_frame_index_ranges": [list(item) for item in index_ranges],
                "expected_frame_count": len(expected_timestamps),
                "actual_frame_count": 0,
                "frame_accounting_mismatch": None,
            }
        )
    if not expected_indices:
        if receipt is not None:
            receipt.update(
                {
                    "actual_frame_count": 0,
                    "frame_accounting_mismatch": 0,
                }
            )
        return
    select_expression = "+".join(
        (
            f"eq(n\\,{range_start})"
            if range_start == range_end
            else f"between(n\\,{range_start}\\,{range_end})"
        )
        for range_start, range_end in index_ranges
    )
    filter_graph = (
        "setpts=PTS-STARTPTS,"
        f"fps={sample_fps:.8f}:round=near:eof_action=round,"
        f"select={select_expression}"
    )
    filter_graph += (
        f",scale_cuda={width}:{height}:format=nv12,hwdownload,format=nv12,format=bgr24"
        if use_cuda_scale
        else f",scale={width}:{height}"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        command += ["-hwaccel", hwaccel]
        if use_cuda_scale:
            command += ["-hwaccel_output_format", "cuda"]
    elif decoder_threads:
        command += ["-threads", str(max(1, decoder_threads))]
    command += [
        "-ss",
        f"{start_ms / 1000.0:.6f}",
        "-i",
        str(path),
        "-t",
        f"{max(0.0, end_ms - start_ms) / 1000.0:.6f}",
        "-vf",
        filter_graph,
        "-an",
        "-sn",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    frame_bytes = width * height * 3
    emitted = 0
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            if emitted >= len(expected_timestamps):
                if receipt is not None:
                    receipt.update(
                        {
                            "actual_frame_count": emitted + 1,
                            "frame_accounting_mismatch": emitted + 1
                            - len(expected_timestamps),
                        }
                    )
                raise RuntimeError("persistent FFmpeg session emitted unexpected extra frames")
            local_ms = expected_timestamps[emitted]
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield int(round(local_ms * info.fps / 1000.0)), local_ms, frame
            emitted += 1
    finally:
        process.stdout.close()
        process.wait()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
    if process.returncode not in (0, None):
        message = stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"FFmpeg 持久分片抽帧失败: {message}")
    if receipt is not None:
        receipt.update(
            {
                "actual_frame_count": emitted,
                "frame_accounting_mismatch": emitted - len(expected_timestamps),
            }
        )
    if emitted != len(expected_timestamps):
        raise RuntimeError(
            "persistent FFmpeg session frame accounting mismatch: "
            f"expected={len(expected_timestamps)} actual={emitted}"
        )


def _opencv_frame_iterator(
    path: Path,
    info: VideoInfo,
    start_ms: float,
    end_ms: float,
    sample_fps: float,
    max_width: int,
) -> Iterator[tuple[int, float, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, start_ms)
    next_sample_ms = start_ms
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            local_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not math.isfinite(local_ms) or local_ms <= 0:
                frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                local_ms = frame_index * 1000.0 / max(info.fps, 1e-9)
            if local_ms >= end_ms:
                break
            if local_ms + 0.5 < next_sample_ms:
                continue
            if info.width > max_width:
                width, height = _scaled_size(info.width, info.height, max_width)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame_index = int(round(local_ms * info.fps / 1000.0))
            yield frame_index, local_ms, frame
            next_sample_ms += 1000.0 / sample_fps
    finally:
        capture.release()


def _opencv_indexed_seek_iterator(
    path: Path,
    info: VideoInfo,
    start_ms: float,
    end_ms: float,
    sample_fps: float,
    max_width: int,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Read sparse probe frames through the MP4 seek index.

    Sequential keyframe demux still transfers almost the whole compressed file
    from NAS. For a low-rate motion probe, explicit indexed seeks fetch only the
    nearby GOPs and are substantially faster on the validated SMB recorder data.
    """

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    requested_ms = max(0.0, start_ms)
    period_ms = 1000.0 / max(sample_fps, 1e-9)
    yielded = 0
    try:
        while requested_ms < end_ms:
            capture.set(cv2.CAP_PROP_POS_MSEC, requested_ms)
            ok, frame = capture.read()
            if ok:
                if info.width > max_width:
                    width, height = _scaled_size(info.width, info.height, max_width)
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                yield (
                    int(round(requested_ms * info.fps / 1000.0)),
                    requested_ms,
                    frame,
                )
                yielded += 1
            requested_ms += period_ms
    finally:
        capture.release()
    if yielded == 0:
        raise RuntimeError(f"稀疏索引抽帧未返回图像: {path}")


def iter_sampled_frames(
    path: Path,
    info: VideoInfo,
    start_ms: float,
    end_ms: float,
    sample_fps: float,
    max_width: int,
    hwaccel: str | None = "cuda",
    keyframes_only: bool = False,
    decoder_threads: int | None = None,
    sparse_strategy: str = "indexed_seek",
    cuda_scale: bool = False,
) -> Iterator[tuple[int, float, np.ndarray]]:
    if sparse_strategy not in {"indexed_seek", "sequential_keyframes"}:
        raise ValueError(f"unsupported sparse decode strategy: {sparse_strategy}")
    if keyframes_only and sample_fps <= 1.0 and sparse_strategy == "indexed_seek":
        try:
            yield from _opencv_indexed_seek_iterator(
                path, info, start_ms, end_ms, sample_fps, max_width
            )
            return
        except RuntimeError:
            # Preserve the sequential FFmpeg path as a compatibility fallback
            # for containers whose seek index is unavailable or damaged.
            pass
    if shutil.which("ffmpeg"):
        try:
            yield from _ffmpeg_frame_iterator(
                path, info, start_ms, end_ms, sample_fps, max_width, hwaccel, keyframes_only,
                decoder_threads, cuda_scale,
            )
            return
        except RuntimeError:
            if hwaccel:
                if cuda_scale:
                    try:
                        # Some Windows FFmpeg/CUDA combinations decode on the
                        # GPU but do not expose scale_cuda. Preserve hardware
                        # decode before falling all the way back to CPU.
                        yield from _ffmpeg_frame_iterator(
                            path,
                            info,
                            start_ms,
                            end_ms,
                            sample_fps,
                            max_width,
                            hwaccel,
                            keyframes_only,
                            decoder_threads,
                            False,
                        )
                        return
                    except RuntimeError:
                        pass
                try:
                    yield from _ffmpeg_frame_iterator(
                        path, info, start_ms, end_ms, sample_fps, max_width, None, keyframes_only,
                        decoder_threads, False,
                    )
                    return
                except RuntimeError:
                    pass
    yield from _opencv_frame_iterator(path, info, start_ms, end_ms, sample_fps, max_width)


def iter_physical_segment_session_frames(
    info: VideoInfo,
    session: PhysicalSegmentDecodeSession,
    sample_fps: float,
    max_width: int,
    hwaccel: str | None = "cuda",
    decoder_threads: int | None = None,
    cuda_scale: bool = False,
    receipt: dict[str, Any] | None = None,
    sampling_grid_origin_ms: float | None = None,
    sampling_grid_period_ms: float | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Serve disjoint target windows with one physical MP4 decoder session."""

    segment = info.segments[session.segment_index]
    source_info = VideoInfo(
        path=segment.path,
        duration_ms=segment.duration_ms,
        fps=segment.fps,
        width=segment.width,
        height=segment.height,
        frame_count=segment.frame_count,
        size_bytes=segment.size_bytes,
    )
    effective_sample_fps = sample_fps
    aligned_virtual_start_ms = session.virtual_start_ms
    if sampling_grid_origin_ms is not None and sampling_grid_period_ms is not None:
        aligned_virtual_start_ms = _aligned_grid_start_ms(
            session.virtual_start_ms,
            sampling_grid_origin_ms,
            sampling_grid_period_ms,
        )
        effective_sample_fps = 1000.0 / sampling_grid_period_ms
    aligned_source_start_ms = (
        aligned_virtual_start_ms - segment.virtual_start_ms
    )
    if receipt is not None:
        receipt.update(
            {
                "sampling_grid_mode": (
                    "aligned_global_timeline"
                    if sampling_grid_origin_ms is not None
                    and sampling_grid_period_ms is not None
                    else "session_start"
                ),
                "sampling_grid_origin_virtual_ms": sampling_grid_origin_ms,
                "sampling_grid_period_local_ms": sampling_grid_period_ms,
                "unaligned_session_start_virtual_ms": session.virtual_start_ms,
                "aligned_session_start_virtual_ms": aligned_virtual_start_ms,
                "sampling_phase_adjustment_ms": (
                    aligned_virtual_start_ms - session.virtual_start_ms
                ),
            }
        )
    if aligned_source_start_ms >= session.source_end_ms - 1e-6:
        if receipt is not None:
            receipt.update(
                {
                    "expected_frame_count": 0,
                    "actual_frame_count": 0,
                    "frame_accounting_mismatch": 0,
                }
            )
        return

    def convert(
        frames: Iterator[tuple[int, float, np.ndarray]],
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        for frame_index, source_ms, frame in frames:
            yield (
                segment.frame_start_index + frame_index,
                segment.virtual_start_ms + source_ms,
                frame,
            )

    if shutil.which("ffmpeg"):
        attempts = [(hwaccel, cuda_scale)]
        if hwaccel and cuda_scale:
            attempts.append((hwaccel, False))
        if hwaccel:
            attempts.append((None, False))
        for attempt_hwaccel, attempt_cuda_scale in attempts:
            emitted = False
            try:
                for item in convert(
                    _ffmpeg_multi_window_iterator(
                        segment.path,
                        source_info,
                        aligned_source_start_ms,
                        session.source_end_ms,
                        session.target_source_windows,
                        effective_sample_fps,
                        max_width,
                        attempt_hwaccel,
                        decoder_threads,
                        attempt_cuda_scale,
                        receipt,
                    )
                ):
                    emitted = True
                    yield item
                if receipt is not None:
                    receipt.update(
                        {
                            "actual_decoder_session_mode": (
                                "ffmpeg_persistent_physical_segment"
                            ),
                            "actual_hwaccel": attempt_hwaccel,
                            "actual_cuda_scale": bool(attempt_cuda_scale),
                            "fallback_reopened_windows": 0,
                        }
                    )
                return
            except RuntimeError as exc:
                if receipt is not None:
                    receipt.setdefault("attempt_errors", []).append(
                        {
                            "hwaccel": attempt_hwaccel,
                            "cuda_scale": bool(attempt_cuda_scale),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if emitted:
                    raise
                continue

    # Compatibility fallback preserves exact windows even when the local FFmpeg
    # build cannot run a multi-window filter. It may reopen the file and is
    # explicitly visible in the runtime receipt as a fallback mode.
    if receipt is not None:
        receipt.update(
            {
                "actual_decoder_session_mode": "compatibility_per_window_fallback",
                "actual_hwaccel": None,
                "actual_cuda_scale": False,
                "fallback_reopened_windows": len(session.target_source_windows),
            }
        )
    fallback_expected_frames = 0
    fallback_actual_frames = 0
    for source_start, source_end in session.target_source_windows:
        aligned_source_start = source_start
        if sampling_grid_origin_ms is not None and sampling_grid_period_ms is not None:
            aligned_virtual_start = _aligned_grid_start_ms(
                segment.virtual_start_ms + source_start,
                sampling_grid_origin_ms,
                sampling_grid_period_ms,
            )
            aligned_source_start = aligned_virtual_start - segment.virtual_start_ms
        if aligned_source_start >= source_end - 1e-6:
            continue
        fallback_expected_frames += len(
            _selected_session_timestamps(
                aligned_source_start,
                source_end,
                [(aligned_source_start, source_end)],
                effective_sample_fps,
            )
        )
        for item in convert(
            iter_sampled_frames(
                segment.path,
                source_info,
                aligned_source_start,
                source_end,
                effective_sample_fps,
                max_width,
                None,
                False,
                decoder_threads,
                "indexed_seek",
                False,
            )
        ):
            fallback_actual_frames += 1
            yield item
    if receipt is not None:
        receipt.update(
            {
                "expected_frame_count": fallback_expected_frames,
                "actual_frame_count": fallback_actual_frames,
                "frame_accounting_mismatch": fallback_actual_frames
                - fallback_expected_frames,
            }
        )
    if fallback_actual_frames != fallback_expected_frames:
        raise RuntimeError(
            "persistent compatibility fallback frame accounting mismatch: "
            f"expected={fallback_expected_frames} actual={fallback_actual_frames}"
        )


def iter_view_sampled_frames(
    view: ViewInput,
    info: VideoInfo,
    start_ms: float,
    end_ms: float,
    sample_fps: float,
    max_width: int,
    hwaccel: str | None = "cuda",
    keyframes_only: bool = False,
    decoder_threads: int | None = None,
    sparse_strategy: str = "indexed_seek",
    cuda_scale: bool = False,
) -> Iterator[tuple[int, float, np.ndarray]]:
    if not info.segments:
        assert view.video is not None
        yield from iter_sampled_frames(
            view.video, info, start_ms, end_ms, sample_fps, max_width, hwaccel,
            keyframes_only, decoder_threads, sparse_strategy, cuda_scale,
        )
        return
    for segment in info.segments:
        overlap_start = max(start_ms, segment.virtual_start_ms)
        overlap_end = min(end_ms, segment.virtual_end_ms)
        if overlap_end <= overlap_start:
            continue
        source_info = VideoInfo(
            path=segment.path,
            duration_ms=segment.duration_ms,
            fps=segment.fps,
            width=segment.width,
            height=segment.height,
            frame_count=segment.frame_count,
            size_bytes=segment.size_bytes,
        )
        segment_start = overlap_start - segment.virtual_start_ms
        segment_end = overlap_end - segment.virtual_start_ms
        for frame_index, source_ms, frame in iter_sampled_frames(
            segment.path, source_info, segment_start, segment_end, sample_fps, max_width,
            hwaccel, keyframes_only, decoder_threads, sparse_strategy, cuda_scale,
        ):
            yield (
                segment.frame_start_index + frame_index,
                segment.virtual_start_ms + source_ms,
                frame,
            )


def benchmark_sparse_decode_strategy(
    view: ViewInput,
    info: VideoInfo,
    *,
    sample_fps: float,
    max_width: int,
    hwaccel: str | None,
    decoder_threads: int | None,
    cuda_scale: bool,
    benchmark_seconds: float,
) -> dict[str, Any]:
    """Choose a sparse decoder from a short read of the active storage path."""

    duration_ms = min(
        float(info.duration_ms),
        max(10.0, float(benchmark_seconds)) * 1000.0,
    )
    if duration_ms <= 0.0:
        raise ValueError(f"view has no benchmarkable duration: {view.view_id}")
    expected_frames = max(1, int(math.floor(duration_ms * sample_fps / 1000.0)))
    reports: list[dict[str, Any]] = []
    for strategy in ("indexed_seek", "sequential_keyframes"):
        started = time.perf_counter()
        frame_count = 0
        error: str | None = None
        try:
            for _frame_index, _local_ms, _frame in iter_view_sampled_frames(
                view,
                info,
                0.0,
                duration_ms,
                sample_fps,
                max_width,
                hwaccel,
                True,
                decoder_threads,
                strategy,
                cuda_scale,
            ):
                frame_count += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed_seconds = time.perf_counter() - started
        sample_ratio = min(1.0, frame_count / expected_frames)
        usable = error is None and frame_count > 0 and sample_ratio >= 0.50
        # Penalize strategies that return too few samples. A fast but sparse
        # decoder is not a valid motion sentinel.
        score_seconds = (
            elapsed_seconds / max(sample_ratio, 0.01) if usable else None
        )
        reports.append(
            {
                "strategy": strategy,
                "usable": usable,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "frame_count": frame_count,
                "expected_frames": expected_frames,
                "sample_ratio": round(sample_ratio, 6),
                "score_seconds": (
                    round(score_seconds, 6) if score_seconds is not None else None
                ),
                "error": error,
            }
        )
    usable_reports = [item for item in reports if item["usable"]]
    if not usable_reports:
        raise RuntimeError(
            f"no sparse decode strategy is usable for {view.view_id}: {reports}"
        )
    selected = min(
        usable_reports,
        key=lambda item: float(item["score_seconds"]),
    )
    return {
        "schema_version": "visioncortex-sparse-decode-benchmark/1",
        "view_id": view.view_id,
        "source_mode": "segmented_virtual_timeline" if info.segments else "continuous_file",
        "benchmark_seconds": round(duration_ms / 1000.0, 3),
        "sample_fps": sample_fps,
        "max_width": max_width,
        "selected_strategy": selected["strategy"],
        "strategies": reports,
    }


def read_frame_at(path: Path, local_ms: float) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, local_ms))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def read_view_frame_at(view: ViewInput, info: VideoInfo, local_ms: float) -> np.ndarray | None:
    if not info.segments:
        assert view.video is not None
        return read_frame_at(view.video, local_ms)
    for segment in info.segments:
        if segment.virtual_start_ms <= local_ms <= segment.virtual_end_ms:
            return read_frame_at(segment.path, local_ms - segment.virtual_start_ms)
    return None


class ViewFrameReader:
    """Reuse a small LRU set of decoder handles for nearby storyboard/key frames."""

    def __init__(self, max_open: int = 2):
        self.max_open = max(1, int(max_open))
        self._captures: OrderedDict[Path, cv2.VideoCapture] = OrderedDict()

    @staticmethod
    def _source(view: ViewInput, info: VideoInfo, local_ms: float) -> tuple[Path, float] | None:
        if not info.segments:
            assert view.video is not None
            return view.video, local_ms
        for segment in info.segments:
            if segment.virtual_start_ms <= local_ms <= segment.virtual_end_ms:
                return segment.path, local_ms - segment.virtual_start_ms
        return None

    def read(self, view: ViewInput, info: VideoInfo, local_ms: float) -> np.ndarray | None:
        source = self._source(view, info, local_ms)
        if source is None:
            return None
        path, source_ms = source
        capture = self._captures.pop(path, None)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                return None
        self._captures[path] = capture
        while len(self._captures) > self.max_open:
            _old_path, old_capture = self._captures.popitem(last=False)
            old_capture.release()
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, source_ms))
        ok, frame = capture.read()
        return frame if ok else None

    def close(self) -> None:
        captures = getattr(self, "_captures", None)
        if captures is None:
            return
        for capture in captures.values():
            capture.release()
        captures.clear()

    def __enter__(self) -> "ViewFrameReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def motion_signature(path: Path, local_times_ms: Sequence[float]) -> np.ndarray:
    values: list[float] = []
    previous: np.ndarray | None = None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return np.zeros(len(local_times_ms), dtype=np.float32)
    try:
        last_requested: float | None = None
        for local_ms in local_times_ms:
            requested = max(0.0, local_ms)
            if last_requested is None or requested < last_requested or requested - last_requested > 1500.0:
                capture.set(cv2.CAP_PROP_POS_MSEC, requested)
                ok, frame = capture.read()
            else:
                ok, frame = capture.read()
                while ok and float(capture.get(cv2.CAP_PROP_POS_MSEC)) + 2.0 < requested:
                    ok, frame = capture.read()
            if not ok:
                values.append(0.0)
                previous = None
                continue
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if previous is None:
                values.append(0.0)
            else:
                values.append(float(cv2.absdiff(gray, previous).mean()))
            previous = gray
            last_requested = requested
    finally:
        capture.release()
    return np.asarray(values, dtype=np.float32)


def view_motion_signature(
    view: ViewInput, info: VideoInfo, local_times_ms: Sequence[float]
) -> np.ndarray:
    if not info.segments:
        assert view.video is not None
        return motion_signature(view.video, local_times_ms)
    values = np.zeros(len(local_times_ms), dtype=np.float32)
    grouped: dict[int, list[tuple[int, float]]] = {}
    for index, local_ms in enumerate(local_times_ms):
        for segment_index, segment in enumerate(info.segments):
            if segment.virtual_start_ms <= local_ms <= segment.virtual_end_ms:
                grouped.setdefault(segment_index, []).append(
                    (index, local_ms - segment.virtual_start_ms)
                )
                break
    for segment_index, requests in grouped.items():
        segment = info.segments[segment_index]
        signature = motion_signature(segment.path, [item[1] for item in requests])
        for (output_index, _), value in zip(requests, signature, strict=True):
            values[output_index] = value
    return values


@lru_cache(maxsize=8)
def _encoder_available(name: str) -> bool:
    result = _run(["ffmpeg", "-hide_banner", "-encoders"])
    return result.returncode == 0 and name.encode() in result.stdout


@lru_cache(maxsize=8)
def _encoder_usable(name: str) -> bool:
    """Probe the complete encoder path, including the installed GPU driver.

    ``ffmpeg -encoders`` only proves that the FFmpeg binary was compiled with an
    encoder.  NVENC can still fail at runtime when the binary requires a newer
    NVENC API than the NVIDIA driver provides.  A one-frame encode catches that
    incompatibility before a multi-hour pipeline reaches materialization.
    """

    if not _encoder_available(name):
        return False
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.04",
            "-frames:v",
            "1",
            "-c:v",
            name,
            "-f",
            "null",
            "-",
        ],
        timeout=20.0,
    )
    return result.returncode == 0


def select_video_encoder(preferred_encoder: str = "h264_nvenc") -> str:
    if _encoder_usable(preferred_encoder):
        return preferred_encoder
    if preferred_encoder != "libx264" and _encoder_usable("libx264"):
        return "libx264"
    raise RuntimeError(
        f"No usable video encoder is available (preferred={preferred_encoder!r}, "
        "fallback='libx264')"
    )


def video_encoder_preflight(preferred_encoder: str = "h264_nvenc") -> dict[str, Any]:
    preferred_listed = _encoder_available(preferred_encoder)
    preferred_usable = _encoder_usable(preferred_encoder)
    selected = select_video_encoder(preferred_encoder)
    return {
        "requested_encoder": preferred_encoder,
        "requested_encoder_listed": preferred_listed,
        "requested_encoder_usable": preferred_usable,
        "selected_encoder": selected,
        "software_fallback_active": selected != preferred_encoder,
        "probe": "one_frame_lavfi_encode",
        "reason": (
            None
            if selected == preferred_encoder
            else "preferred encoder is unavailable or incompatible with the active driver"
        ),
    }


def extract_clip(
    source: Path,
    destination: Path,
    start_ms: float,
    duration_ms: float,
    preferred_encoder: str = "h264_nvenc",
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoder = select_video_encoder(preferred_encoder)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_ms) / 1000.0:.6f}",
        "-i",
        str(source),
        "-t",
        f"{max(1.0, duration_ms) / 1000.0:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        encoder,
        "-preset",
        "p4" if encoder == "h264_nvenc" else "veryfast",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = _run(command)
    if result.returncode != 0 and encoder != "libx264":
        command[command.index(encoder)] = "libx264"
        preset_index = command.index("p4")
        command[preset_index] = "veryfast"
        result = _run(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-2000:])


def extract_view_clip(
    view: ViewInput,
    info: VideoInfo,
    destination: Path,
    start_ms: float,
    duration_ms: float,
    preferred_encoder: str = "h264_nvenc",
) -> None:
    """Export one continuous clip, joining only the source segments it intersects."""

    if not info.segments:
        assert view.video is not None
        extract_clip(view.video, destination, start_ms, duration_ms, preferred_encoder)
        return
    end_ms = min(info.duration_ms, start_ms + duration_ms)
    overlaps = [
        segment for segment in info.segments
        if segment.virtual_end_ms > start_ms and segment.virtual_start_ms < end_ms
    ]
    if not overlaps:
        raise ValueError(f"clip window is outside {view.view_id}: {start_ms}..{end_ms}")
    if len(overlaps) == 1:
        segment = overlaps[0]
        extract_clip(
            segment.path,
            destination,
            max(0.0, start_ms - segment.virtual_start_ms),
            end_ms - start_ms,
            preferred_encoder,
        )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="visioncortex-clip-") as temporary:
        temporary_root = Path(temporary)
        parts: list[Path] = []
        for index, segment in enumerate(overlaps):
            overlap_start = max(start_ms, segment.virtual_start_ms)
            overlap_end = min(end_ms, segment.virtual_end_ms)
            part = temporary_root / f"part-{index:03d}.mp4"
            extract_clip(
                segment.path,
                part,
                overlap_start - segment.virtual_start_ms,
                overlap_end - overlap_start,
                preferred_encoder,
            )
            parts.append(part)
        concat_list = temporary_root / "parts.txt"
        concat_list.write_text(
            "\n".join("file '" + str(path).replace("'", "'\\''") + "'" for path in parts) + "\n",
            encoding="utf-8",
        )
        result = _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
                "-safe", "0", "-i", str(concat_list), "-c", "copy", "-movflags",
                "+faststart", str(destination),
            ]
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-2000:])


def create_grid_video(
    clips: Sequence[tuple[str, Path]],
    destination: Path,
    preferred_encoder: str = "h264_nvenc",
) -> None:
    if not clips:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = 3 if len(clips) >= 3 else len(clips)
    rows = math.ceil(len(clips) / columns)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for _, clip in clips:
        command += ["-i", str(clip)]
    filters: list[str] = []
    for index in range(len(clips)):
        filters.append(f"[{index}:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2[v{index}]")
    layout = "|".join(f"{(i % columns) * 640}_{(i // columns) * 360}" for i in range(len(clips)))
    inputs = "".join(f"[v{i}]" for i in range(len(clips)))
    filters.append(f"{inputs}xstack=inputs={len(clips)}:layout={layout}:fill=black[vout]")
    encoder = select_video_encoder(preferred_encoder)
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
        "-an",
        "-c:v",
        encoder,
        "-preset",
        "p4" if encoder == "h264_nvenc" else "veryfast",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = _run(command)
    if result.returncode != 0 and encoder != "libx264":
        command[command.index(encoder)] = "libx264"
        command[command.index("p4")] = "veryfast"
        result = _run(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-2000:])


def write_annotated_frame(frame: np.ndarray, boxes: Sequence[dict[str, Any]], destination: Path) -> None:
    height, width = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy_norm"]
        p1, p2 = (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))
        cv2.rectangle(frame, p1, p2, (0, 220, 255), 2)
        label = f"{box['class_name']} {box['confidence']:.2f}"
        cv2.putText(frame, label, (p1[0], max(16, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"关键帧写入失败: {destination}")
