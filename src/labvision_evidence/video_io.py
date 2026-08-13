from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
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


def probe_view(view: ViewInput) -> VideoInfo:
    """Probe a single file or build a zero-copy virtual timeline from segments."""

    if not view.segments:
        assert view.video is not None
        return probe_video(view.video)
    probed = [probe_video(segment.video) for segment in view.segments]
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
