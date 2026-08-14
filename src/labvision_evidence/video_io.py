from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import VideoInfo, VideoSegmentInfo, ViewInput


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

    if clock is None or not clock.is_file():
        return None
    try:
        with clock.open("rb") as handle:
            head = handle.read(512 * 1024)
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 512 * 1024))
            tail = handle.read()
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

        def rgb_rows(lines: Sequence[str]) -> list[list[str]]:
            rows = []
            for line in lines:
                values = next(csv.reader([line]), [])
                if len(values) <= max(indexes.values()):
                    continue
                if values[indexes["rgb_recorded"]].strip().lower() not in {"1", "true", "yes"}:
                    continue
                if not values[indexes["rgb_video_frame_index"]].strip():
                    continue
                rows.append(values)
            return rows

        first_rows = rgb_rows(head_lines[1:])
        last_rows = rgb_rows(tail_lines[1:] if size > len(tail) else tail_lines)
        if not first_rows or not last_rows:
            return None
        first, last = first_rows[0], last_rows[-1]

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
) -> Iterator[tuple[int, float, np.ndarray]]:
    width, height = _scaled_size(info.width, info.height, max_width)
    filter_graph = f"fps={sample_fps:.8f},scale={width}:{height}"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        command += ["-hwaccel", hwaccel]
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
) -> Iterator[tuple[int, float, np.ndarray]]:
    if keyframes_only and sample_fps <= 1.0:
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
                decoder_threads,
            )
            return
        except RuntimeError:
            if hwaccel:
                try:
                    yield from _ffmpeg_frame_iterator(
                        path, info, start_ms, end_ms, sample_fps, max_width, None, keyframes_only,
                        decoder_threads,
                    )
                    return
                except RuntimeError:
                    pass
    yield from _opencv_frame_iterator(path, info, start_ms, end_ms, sample_fps, max_width)


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
) -> Iterator[tuple[int, float, np.ndarray]]:
    if not info.segments:
        assert view.video is not None
        yield from iter_sampled_frames(
            view.video, info, start_ms, end_ms, sample_fps, max_width, hwaccel,
            keyframes_only, decoder_threads,
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
            hwaccel, keyframes_only, decoder_threads,
        ):
            yield (
                segment.frame_start_index + frame_index,
                segment.virtual_start_ms + source_ms,
                frame,
            )


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


def extract_clip(
    source: Path,
    destination: Path,
    start_ms: float,
    duration_ms: float,
    preferred_encoder: str = "h264_nvenc",
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoder = preferred_encoder if _encoder_available(preferred_encoder) else "libx264"
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


def create_grid_video(clips: Sequence[tuple[str, Path]], destination: Path) -> None:
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
    encoder = "h264_nvenc" if _encoder_available("h264_nvenc") else "libx264"
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
