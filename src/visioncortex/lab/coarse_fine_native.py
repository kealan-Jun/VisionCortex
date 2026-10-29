"""Bounded CPU audit of the original pixels/PTS delivered to a fine consumer.

No resized or sampled fallback exists. This proves local decode delivery, not
that a remote model subsequently consumed every supplied frame or read it right.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

from .coarse_fine_research import FineTask, Packet, StageClocks
from .domain import fingerprint
from .subprocesses import run_bounded


class NativeCapacityError(ValueError):
    """Return needs_attention; do not retry an identical oversized request."""


@dataclass(frozen=True)
class NativeBudget:
    max_frames: int = 4096
    max_pixels_per_frame: int = 9000000
    max_total_pixels: int = 1000000000
    max_source_frames: int = 60000
    max_source_bytes: int = 8 * 1024**3
    probe_max_bytes: int = 16 * 1024**2
    probe_timeout_seconds: int = 300

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('Explicit finite native input bounds required')


@dataclass(frozen=True)
class Frame:
    media_id: str
    source_sha256: str
    source_frame_index: int
    source_pts_seconds: str
    width: int
    height: int
    rgb_sha256: str


@dataclass(frozen=True)
class NativeReceipt:
    task_id: str
    frames: tuple[Frame, ...]
    source_spans_sha256: str
    input_sha256: str
    decoded_frame_count: int
    delivered_frame_count: int
    spatial_policy: str = 'native_decoder_rgb_no_resize'
    temporal_policy: str = 'all_frames_with_pts_in_half_open_source_spans'
    model_consumption: str = 'NOT_ATTESTED_BY_LOCAL_DECODE'

    def validate(self, task):
        if (self.spatial_policy != 'native_decoder_rgb_no_resize' or
                self.temporal_policy != 'all_frames_with_pts_in_half_open_source_spans' or
                self.model_consumption != 'NOT_ATTESTED_BY_LOCAL_DECODE'):
            raise ValueError('Local decoder receipt cannot assert a changed input policy or model attestation')
        spans_sha = fingerprint([asdict(s) for s in task.source_spans])
        if self.task_id != task.task_id or self.source_spans_sha256 != spans_sha or self.delivered_frame_count != len(self.frames):
            raise ValueError('Native receipt does not match the requested evidence')
        if not self.frames or self.input_sha256 != fingerprint([asdict(f) for f in self.frames]):
            raise ValueError('Missing or modified local decoder input receipt')
        if type(self.decoded_frame_count) is not int or self.decoded_frame_count < self.delivered_frame_count:
            raise ValueError('Actual decoded count cannot be below delivered frame count')
        for frame in self.frames:
            at = Fraction(frame.source_pts_seconds) * 1000000
            if not any(frame.media_id == s.media_id and frame.source_sha256 == s.source_sha256 and
                       s.start_us <= at < s.end_us for s in task.source_spans):
                raise ValueError('Frame is outside this fine request')


def _hash(path, maximum):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            size += len(block)
            if size > maximum:
                raise NativeCapacityError('source_size_capacity')
            digest.update(block)
    return digest.hexdigest()


def audit_native(task: FineTask, packets: dict[str, Packet], source_paths, *, source_guard,
                 consume_frame, budget=None, clocks=None):
    """Guard and hash original files, then synchronously deliver native RGB frames.

    source_guard(path, packet) must validate the actual allowed SSD/NAS location
    before any source access. Pass the existing verified NAS/Media guard at live
    integration; an unmounted path is never opened or made as a fallback here.
    consume_frame(frame, rgb) must finish copying/encoding before returning. The
    read-only RGB array is transient and must not be retained as an unbounded list.
    """
    if source_guard is None or consume_frame is None:
        raise ValueError('Explicit source guard and bounded synchronous consumer required')
    if task.status != 'pending' or task.attention_reasons:
        raise ValueError('A blocked fine task cannot silently change input policy')
    if fingerprint([asdict(packets[s.media_id]) for s in task.source_spans]) != task.source_packets_sha256:
        raise ValueError('Changed source capture identity or time mapping')
    import cv2

    budget, clocks = budget or NativeBudget(), clocks or StageClocks()
    manifests, expected_count, pixels = [], 0, 0
    for span in task.source_spans:
        packet = packets[span.media_id]
        if span.source_sha256 != packet.source_sha256 or span.end_us > packet.duration_us:
            raise ValueError('Changed fine source')
        path = Path(source_paths[span.media_id])
        source_guard(path, packet)
        with clocks.stage('read', stream_id=task.stream_id):
            before = path.stat()
            if _hash(path, budget.max_source_bytes) != span.source_sha256:
                raise ValueError('Source SHA differs from pinned original')
        with clocks.stage('decode', stream_id=task.stream_id):
            raw = run_bounded(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                               '-show_entries', 'stream=time_base,start_pts:frame=best_effort_timestamp,width,height',
                               '-of', 'json', str(path)], '', env=os.environ.copy(),
                              timeout=budget.probe_timeout_seconds, max_bytes=budget.probe_max_bytes)
            manifest = json.loads(raw)
        stream, frames = manifest['streams'][0], manifest.get('frames', [])
        if not frames or len(frames) > budget.max_source_frames:
            raise NativeCapacityError('source_frame_manifest_capacity_or_empty')
        base = Fraction(stream['time_base'])
        origin = stream.get('start_pts', frames[0]['best_effort_timestamp'])
        times = [Fraction(f['best_effort_timestamp'] - origin) * base for f in frames]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Non-monotonic original decode PTS; no silent reordering')
        selected = [i for i, at in enumerate(times) if span.start_us <= at * 1000000 < span.end_us]
        if not selected:
            raise ValueError('No decoded frame in required fine evidence')
        for i in selected:
            count = int(frames[i]['width']) * int(frames[i]['height'])
            if count > budget.max_pixels_per_frame:
                raise NativeCapacityError('native_pixels_capacity')
            pixels += count
        expected_count += len(selected)
        if expected_count > budget.max_frames or pixels > budget.max_total_pixels:
            raise NativeCapacityError('all_frame_context_capacity')
        manifests.append((span, path, before, frames, times, set(selected)))
    traces, decoded = [], 0
    for span, path, before, manifest, times, selected in manifests:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError('Original codec could not open')
        try:
            with clocks.stage('decode', stream_id=task.stream_id):
                for index in range(max(selected) + 1):
                    ok, bgr = capture.read()
                    if not ok:
                        raise ValueError('Abnormal original decode before required evidence end')
                    decoded += 1
                    if bgr.shape[:2] != (manifest[index]['height'], manifest[index]['width']):
                        raise ValueError('Original frame geometry differs from timestamp manifest')
                    if index in selected:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        rgb.setflags(write=False)
                        trace = Frame(span.media_id, span.source_sha256, index, str(times[index]),
                                      rgb.shape[1], rgb.shape[0], hashlib.sha256(rgb.tobytes()).hexdigest())
                        tick = time.monotonic()
                        consume_frame(trace, rgb)
                        clocks.records.append(dict(stage='encode', mode='cold', stream_id=task.stream_id,
                                                  elapsed_seconds=time.monotonic() - tick, outcome='returned',
                                                  gpu_synchronizer_supplied=False,
                                                  timing_scope='consumer_callback_cpu_wall_not_gpu_completion'))
                        traces.append(trace)
        finally:
            capture.release()
        source_guard(path, packets[span.media_id])
        with clocks.stage('read', stream_id=task.stream_id):
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or _hash(path, budget.max_source_bytes) != span.source_sha256:
                raise ValueError('Original source changed while decoding')
    if len(traces) != expected_count:
        raise ValueError('Native decoded evidence delivery count mismatch')
    receipt = NativeReceipt(task.task_id, tuple(traces), fingerprint([asdict(s) for s in task.source_spans]),
                            fingerprint([asdict(f) for f in traces]), decoded, len(traces))
    receipt.validate(task)
    return receipt
