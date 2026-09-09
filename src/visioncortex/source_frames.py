"""Link selected decoder frames to native PTS and packet positions.

Legacy FFmpeg exposes packet positions through showinfo; FFmpeg 6+ exposes
original decoder PTS through pre-encoding statistics. Resolve either against ffprobe's
decoded-frame ledger, never average FPS or pixel similarity. Missing or
non-unique identities remain unresolved.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import shutil
import subprocess
import threading
from collections import deque
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .schemas import FrameEvidence, SourceFrameIdentity, VideoInfo, ViewInput


SOURCE_FRAME_CONTRACT = "decoder-position-pts/2"
SOURCE_FRAME_FILTER = "showinfo@source_identity=checksum=0"
_FRAME_LINE = re.compile(r"\bn:\s*(\d+)\b")
_POSITION = re.compile(r"\bpos:\s*(-?\d+)\b")
_PTS = re.compile(r"\bpts:\s*(-?\d+)\b")
_TIME_BASE = re.compile(r"config in time_base:\s*(\d+/\d+)")
_ENCODER_LINE = re.compile(r"^VC_SOURCE (\d+) (-?\d+) (\d+/\d+)\s*$")


@lru_cache(maxsize=8)
def _encoder_stats_supported(executable):
    if not executable:
        return False
    try:
        version = subprocess.run([executable, "-version"], capture_output=True, check=True, timeout=10)
        match = re.search(rb"ffmpeg version (?:n)?(\d+)\.", version.stdout)
        return bool(match and int(match[1]) >= 6)
    except (OSError, subprocess.SubprocessError):
        return False


def _encoder_stats_options():
    return ["-stats_enc_pre", "pipe:2", "-stats_enc_pre_fmt", "VC_SOURCE {n} {ptsi} {tbi}"]


def _native_rows(path, start, end, *, timeout=120):
    return json.loads(subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-skip_loop_filter", "all", "-skip_idct", "all", "-threads", "2",
        "-read_intervals", f"{start:.9f}%{end:.9f}",
        "-show_entries", "frame=best_effort_timestamp,pkt_pos", "-of", "json", str(path),
    ], capture_output=True, check=True, timeout=timeout).stdout).get("frames", [])


def read_evidence_frame(
    view: ViewInput, info: VideoInfo, evidence: FrameEvidence | None,
) -> tuple[np.ndarray | None, dict]:
    """Reproduce the ledger's actual BGR image, never attach boxes to a seek target.

    Bounded CPU decoding must match both native identity and the recorded model
    input pixels. Unsupported decoder/scaler combinations remain unverified.
    The returned image uses the ledger resolution, not the source resolution.
    """
    receipt = {
        "schema_version": "visioncortex-key-frame-source/1",
        "status": "unverified", "reason": "source_frame_identity_unavailable",
        "view_id": view.view_id, "role": view.role.value,
        "detections_bound_to_pixels": False,
    }

    def unavailable(reason):
        return None, {**receipt, "reason": reason}

    if evidence is None or evidence.source_frame is None:
        return unavailable("source_frame_identity_unavailable")
    identity = evidence.source_frame
    receipt["source_frame"] = identity.model_dump(mode="json")
    receipt["ledger_local_ms"] = evidence.local_ms
    receipt["ledger_global_ms"] = evidence.global_ms
    if evidence.view_id != view.view_id or evidence.role != view.role:
        return unavailable("ledger_view_or_role_mismatch")
    if identity.status != "resolved":
        return unavailable("source_frame_identity_" + identity.status)
    path = identity.source_path.resolve()
    sources = [segment.video.resolve() for segment in view.segments] if view.segments else [view.video.resolve()]
    if sources.count(path) != 1:
        return unavailable("source_not_unique_in_view")
    virtual_start_ms = 0.0
    if view.segments:
        segments = [segment for segment in info.segments if segment.path.resolve() == path]
        if len(segments) != 1:
            return unavailable("source_segment_mapping_unavailable")
        virtual_start_ms = segments[0].virtual_start_ms
    if not 0 < evidence.width <= 16384 or not 0 < evidence.height <= 16384:
        return unavailable("invalid_ledger_dimensions")
    try:
        def unchanged():
            stat = path.stat()
            return (stat.st_size, stat.st_mtime_ns) == (identity.source_size_bytes, identity.source_mtime_ns)

        if not unchanged():
            return unavailable("source_file_changed")
        metadata = json.loads(subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=time_base:format=start_time", "-of", "json", str(path),
        ], capture_output=True, check=True, timeout=20).stdout)
        time_base = Fraction(metadata["streams"][0]["time_base"])
        if time_base != Fraction(identity.time_base):
            return unavailable("source_time_base_changed")
        origin = Fraction(str(metadata.get("format", {}).get("start_time", "0")))
        source_seconds = identity.source_pts * time_base - origin
        if source_seconds < 0:
            return unavailable("source_pts_precedes_media_origin")
        seek_seconds = max(0.0, float(source_seconds) - 1.0)
        # NVDEC returns NV12; at native size its BGR conversion can differ
        # from software YUV420P. Try that CPU conversion only after a pixel
        # mismatch, retaining the same exact PTS, packet and digest checks.
        encoder_stats = _encoder_stats_supported(shutil.which("ffmpeg"))
        from .video_io import _ffmpeg_passthrough_arguments
        passthrough = _ffmpeg_passthrough_arguments()
        if encoder_stats:
            rows = _native_rows(path, float(source_seconds + origin) - min(1.0, float(source_seconds)),
                                float(source_seconds + origin) + 1.1, timeout=20)
            matches = [row for row in rows if str(row.get("best_effort_timestamp")) == str(identity.source_pts)]
            if len(matches) != 1 or str(matches[0].get("pkt_pos")) != str(identity.packet_position):
                return unavailable("decoded_native_identity_mismatch")
        for format_filter in ("", "format=nv12,"):
            filters = (
                f"select=eq(pts\\,{identity.source_pts}),{SOURCE_FRAME_FILTER},{format_filter}"
                f"scale={evidence.width}:{evidence.height}"
            )
            # Input -t bounds the read even if the requested PTS is absent. copyts
            # preserves a nonzero source origin for the exact native-PTS selector.
            decoded = subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin", "-copyts",
                "-threads", "2", "-ss", f"{seek_seconds:.9f}", "-t", "2.1",
                "-i", str(path), "-map", "0:v:0", "-vf", filters,
                "-an", "-sn", *passthrough, "-frames:v", "1",
                *(_encoder_stats_options() if encoder_stats else []),
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
            ], capture_output=True, check=True, timeout=20)
            records = []
            filter_time_bases = []
            for line in decoded.stderr.decode("utf-8", errors="replace").splitlines():
                if "showinfo@source_identity" not in line:
                    continue
                match = _TIME_BASE.search(line)
                if match:
                    filter_time_bases.append(Fraction(match[1]))
                if _FRAME_LINE.search(line):
                    pts, position = _PTS.search(line), _POSITION.search(line)
                    records.append((int(pts[1]) if pts else None, int(position[1]) if position else None))
            if encoder_stats:
                mux_records = [_ENCODER_LINE.fullmatch(line.strip()) for line in decoded.stderr.decode("utf-8", errors="replace").splitlines()]
                mux_records = [m for m in mux_records if m]
                if (len(mux_records) != 1 or int(mux_records[0][1]) != 0
                        or int(mux_records[0][2]) != identity.source_pts
                        or Fraction(mux_records[0][3]) != time_base):
                    return unavailable("decoded_native_identity_mismatch")
            elif filter_time_bases != [time_base] or records != [(identity.source_pts, identity.packet_position)]:
                return unavailable("decoded_native_identity_mismatch")
            if len(decoded.stdout) != evidence.width * evidence.height * 3:
                return unavailable("decoded_frame_size_mismatch")
            digest = hashlib.sha256(decoded.stdout).hexdigest()
            if not unchanged():
                return unavailable("source_file_changed_during_decode")
            if digest == identity.decoded_pixels_sha256:
                break
        else:
            return unavailable("decoded_pixels_do_not_match_ledger")
        frame = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape((evidence.height, evidence.width, 3)).copy()
        return frame, {
            **receipt, "status": "verified", "reason": None,
            "detections_bound_to_pixels": True, "decoded_pixels_sha256": digest,
            "width": evidence.width, "height": evidence.height,
            "resolution_basis": "ledger_model_input",
            "reproduction_pixel_format": "nv12" if format_filter else "decoder_default",
            "source_media_ms": float(source_seconds * 1000),
            "source_format_start_ms": float(origin * 1000),
            "view_local_ms": virtual_start_ms + float(source_seconds * 1000),
            "time_basis": "native_pts_minus_format_origin_plus_segment_offset",
        }
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError):
        return unavailable("native_frame_decode_unavailable")


class SampledFrame(tuple):
    """Keep the existing three-value iterator API and carry provenance with it."""

    source_frame: SourceFrameIdentity | None

    def __new__(cls, index, local_ms, frame, source_frame=None):
        result = super().__new__(cls, (index, local_ms, frame))
        result.source_frame = source_frame
        return result


def retime_sampled_frame(item, index: int, local_ms: float) -> SampledFrame:
    return SampledFrame(index, local_ms, item[2], getattr(item, "source_frame", None))


class SourceFrameTrace:
    def __init__(self, path: Path, start_ms: float, end_ms: float, sample_fps: float):
        self.path = path
        self.enabled = path.is_file() and shutil.which("ffprobe") is not None
        self.records: queue.Queue[tuple[int, int] | None] = queue.Queue()
        self.messages: deque[bytes] = deque(maxlen=40)
        self.thread: threading.Thread | None = None
        self.ended = False
        self.positions: dict[int, list[tuple[int, int | None]]] = {}
        self.native_pts: dict[int, list[int]] = {}
        self.encoder_stats = self.enabled and _encoder_stats_supported(shutil.which("ffmpeg"))
        self.timestamp_offset = 0
        self.time_base: str | None = None
        self.failure: str | None = None
        self.stat = path.stat() if path.is_file() else None
        if not self.enabled:
            return
        try:
            metadata = self._probe(["-show_entries", "stream=time_base:format=start_time"])
            self.time_base = metadata["streams"][0]["time_base"]
            if Fraction(self.time_base) <= 0:
                raise ValueError("invalid native time base")
            origin_fraction = Fraction(str(metadata.get("format", {}).get("start_time", "0")))
            origin = float(origin_fraction)
            # Match the CLI's microsecond seek and nearest timebase rescale.
            offset = (origin_fraction + Fraction(f"{start_ms / 1000:.6f}")) / Fraction(self.time_base)
            self.timestamp_offset = int(offset + Fraction(1, 2)) if offset >= 0 else -int(-offset + Fraction(1, 2))
            start = origin + max(0.0, start_ms / 1000.0)
            end = origin + end_ms / 1000.0 + max(1.0, 1.0 / sample_fps)
            for index, row in enumerate(_native_rows(path, start, end)):
                if row.get("pkt_pos") in (None, "N/A") or row.get("best_effort_timestamp") is None:
                    continue
                position = int(row["pkt_pos"])
                if position >= 0:
                    pts = int(row["best_effort_timestamp"])
                    self.positions.setdefault(position, []).append((pts, index if start_ms == 0 else None))
                    self.native_pts.setdefault(pts, []).append(position)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (self.stat.st_size, self.stat.st_mtime_ns):
                self.positions.clear()
                self.failure = "source_changed_during_native_frame_probe"
        except (OSError, subprocess.SubprocessError, KeyError, ValueError, TypeError, ZeroDivisionError):
            self.positions.clear()
            self.failure = "native_frame_probe_unavailable"

    def input_options(self):
        return ["-copyts"] if self.encoder_stats else []

    def output_options(self):
        return _encoder_stats_options() if self.encoder_stats else []

    def filter_prefix(self):
        # Preserve native decoder PTS in pre-encoding statistics while presenting the
        # original zero-based timestamps to the existing FPS filter.
        return f"setpts=PTS-({self.timestamp_offset})," if self.encoder_stats else ""

    def _probe(self, arguments: list[str]) -> dict:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", *arguments,
             "-of", "json", str(self.path)],
            capture_output=True, check=True, timeout=120,
        )
        return json.loads(result.stdout)

    def start(self, stderr: BinaryIO | None) -> None:
        if not self.enabled or stderr is None:
            return

        def consume():
            try:
                for line in iter(stderr.readline, b""):
                    text = line.decode("utf-8", errors="replace")
                    if self.encoder_stats:
                        # Human progress uses carriage returns on the same
                        # stderr line; the following encoder record is independent.
                        match = _ENCODER_LINE.fullmatch(text.rstrip("\r\n").rsplit("\r", 1)[-1].strip())
                        if match:
                            try:
                                valid = (self.failure is None and self.time_base is not None
                                         and Fraction(match[3]) == Fraction(self.time_base)
                                         and int(match[2]) != 9223372036854775807)
                            except (ValueError, ZeroDivisionError):
                                valid = False
                            candidates = self.native_pts.get(int(match[2]), []) if valid else []
                            position = candidates[0] if len(candidates) == 1 else -1
                            self.records.put((int(match[1]), position))
                            continue
                    elif b"showinfo@source_identity" in line:
                        match = _FRAME_LINE.search(text)
                        if match:
                            position = _POSITION.search(text)
                            self.records.put((int(match[1]), int(position[1]) if position else -1))
                            continue
                    self.messages.append(line[-2048:])
            finally:
                self.records.put(None)

        self.thread = threading.Thread(target=consume, name="source-frame-trace", daemon=True)
        self.thread.start()

    def identity(self, index: int, frame: np.ndarray) -> SourceFrameIdentity | None:
        if not self.enabled or self.stat is None:
            return None
        position = None
        reason = self.failure
        if not self.ended:
            try:
                record = self.records.get(timeout=10)
            except queue.Empty:
                record = None
                reason = "decoder_frame_trace_timeout"
            if record is None:
                self.ended = True
                reason = reason or "decoder_frame_trace_unavailable"
            elif record[0] != index:
                self.ended = True
                reason = "decoder_frame_trace_index_mismatch"
            elif record[1] >= 0:
                position = record[1]
            else:
                reason = reason or "decoder_frame_position_unavailable"
        matches = self.positions.get(position, [])
        status = "unavailable"
        pts = native_index = time_base = None
        if reason is None and len(matches) == 1:
            status = "resolved"
            pts, native_index = matches[0]
            time_base = self.time_base
        elif reason is None and len(matches) > 1:
            status, reason = "ambiguous", "decoder_position_matches_multiple_native_frames"
        else:
            reason = reason or "decoder_position_has_no_native_frame_match"
        return SourceFrameIdentity(
            source_path=self.path.resolve(), source_size_bytes=self.stat.st_size,
            source_mtime_ns=self.stat.st_mtime_ns, status=status,
            packet_position=position, source_pts=pts, time_base=time_base,
            source_frame_index=native_index,
            decoded_pixels_sha256=hashlib.sha256(frame.tobytes()).hexdigest(), reason=reason,
        )

    def finish(self, stderr: BinaryIO | None) -> bytes:
        try:
            if self.thread is not None:
                self.thread.join(timeout=10)
                if self.thread.is_alive():
                    raise RuntimeError("Source frame metadata reader did not finish")
                after = self.path.stat()
                if (after.st_size, after.st_mtime_ns) != (self.stat.st_size, self.stat.st_mtime_ns):
                    raise RuntimeError("Source video changed during frame decoding")
                return b"".join(self.messages)
            return stderr.read() if stderr is not None else b""
        finally:
            if stderr is not None:
                stderr.close()
