"""Opt-in source-pinned video OCR; original PTS decode and bounded ROI evidence.

The decoder uses an explicitly configured existing local Python/PyAV runtime.
It starts no service and never copies/re-encodes original media. Visual confirmed
results remain candidate observations, never business confirmations.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import shutil
import struct
import subprocess
import time

from .domain import Conflict, fingerprint
from .gauge import exact
from .media import hash_file
from .ocr_fields import ROLES, tokens
from .storage import is_nas, verify_root, verify_required_roots

SCHEMA = "vc-lab-temporal-video-ocr/1"
CACHE_BYTES = 64 * 1024**2
OUTPUT_BYTES = 128 * 1024**2


def jsonb_stable(value):
    """Preserve JSON numeric meaning through PostgreSQL JSONB canonicalization.

    JSONB normalizes negative zero and exponent-form integral floats. Normalize
    these before fingerprinting; never coerce exact PTS/Fraction strings, and
    reject nonfinite numbers rather than replacing or concealing them.
    """
    if isinstance(value, dict):
        return {k: jsonb_stable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonb_stable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TemporalVideoRejected("Nonfinite temporal result number")
        return int(value) if value.is_integer() else value
    return value


class TemporalVideoRejected(ValueError):
    """A fixed input/configuration failure must not be retried identically."""


class TemporalVideoCapacityExceeded(TemporalVideoRejected):
    pass


def _text(value, name, limit=128):
    if not isinstance(value, str) or not value or len(value) > limit:
        raise TemporalVideoRejected("Explicit bounded " + name + " required")
    return value


def validate_registration(document, *, media=None, asset=None):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA
        or document.get("enabled") is not True
    ):
        raise TemporalVideoRejected(
            "Explicit enabled temporal video OCR registration required"
        )
    if len(json.dumps(document, ensure_ascii=False).encode()) > 128 * 1024:
        raise TemporalVideoRejected("Temporal registration exceeds its bound")
    _text(document.get("revision"), "registration revision", 48)
    _text(document.get("source_stream_key"), "source stream")
    if (
        not re.fullmatch("[0-9a-f]{64}", document.get("source_sha256", ""))
        or type(document.get("media_version")) is not int
    ):
        raise TemporalVideoRejected("Source hash and media version required")
    if document["media_version"] < 1:
        raise TemporalVideoRejected("Invalid media version")
    rotation = document.get("rotation_degrees_clockwise")
    if type(rotation) is not int or rotation not in {0, 180}:
        raise TemporalVideoRejected("Explicit 0/180 degree source orientation required")
    _text(document.get("orientation_basis"), "source orientation basis", 512)
    if document.get("world_clock_status") not in {"verified", "uncalibrated"}:
        raise TemporalVideoRejected("Explicit world-clock verification status required")
    if document.get("time_basis") != "original_media_pts":
        raise TemporalVideoRejected("Temporal OCR must preserve original media PTS")
    intervals = document.get("source_intervals")
    if not isinstance(intervals, list) or not 1 <= len(intervals) <= 32:
        raise TemporalVideoRejected("One to 32 explicit source intervals required")
    frontier, total = Decimal(0), Decimal(0)
    for interval in intervals:
        lo, hi = exact(interval["start_seconds"]), exact(interval["end_seconds"])
        if not 0 <= lo < hi or hi - lo > 10 or lo < frontier:
            raise TemporalVideoRejected(
                "Intervals must be ordered, non-overlapping and at most 10 seconds"
            )
        frontier, total = hi, total + hi - lo
        if media is not None and hi > media["duration_seconds"]:
            raise TemporalVideoRejected("Temporal interval escaped registered media")
    if total > 120:
        raise TemporalVideoRejected("Temporal task source duration exceeds 120 seconds")
    reference = document.get("source_tracking_reference")
    if reference is not None:
        if (not isinstance(reference, dict)
                or reference.get("schema_version") != "vc-source-panel-reference/1"
                or reference.get("source_sha256") != document["source_sha256"]
                or reference.get("media_version") != document["media_version"]
                or type(reference.get("pts")) is not int
                or not re.fullmatch("[0-9a-f]{64}", reference.get("source_frame_pixels_sha256", ""))):
            raise TemporalVideoRejected("Pinned same-source panel geometry reference required")
        try:
            position = Fraction(reference["media_offset_seconds"])
            time_base = Fraction(reference["time_base"])
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            raise TemporalVideoRejected("Exact reference frame time required") from exc
        if not 0 <= position < Fraction(intervals[0]["start_seconds"]) or time_base <= 0:
            raise TemporalVideoRejected("Geometry reference must precede the requested source window")
    fields = document.get("fields")
    if not isinstance(fields, list) or not 1 <= len(fields) <= 16:
        raise TemporalVideoRejected("One to 16 explicit registered fields required")
    keys = set()
    tracked_fields = 0
    dynamic = document.get("dynamic_context")
    if dynamic is not None:
        if not isinstance(dynamic, dict) or dynamic.get("enabled") is not True:
            raise TemporalVideoRejected("Dynamic context must be explicitly enabled")
        if not document.get("source_session_key"):
            raise TemporalVideoRejected("Dynamic context requires the source session")
        if dynamic.get("revision") is not None and (
            type(dynamic["revision"]) is not int or dynamic["revision"] < 1
        ):
            raise TemporalVideoRejected("Invalid pinned dynamic context revision")
    producer = document.get("dynamic_video")
    if producer is not None and (
        not isinstance(producer, dict)
        or producer.get("schema_version") != "vc-lab-dynamic-video/1"
        or producer.get("enabled") is not True
        or not dynamic
        or dynamic.get("revision") is not None
        or dynamic.get("known_at") is not None
        or producer.get("batch_frames") != 1
        or type(producer.get("max_frames")) is not int
        or not 1 <= producer["max_frames"] <= 600
        or type(producer.get("max_buffer_bytes")) is not int
        or not 1 <= producer["max_buffer_bytes"] <= CACHE_BYTES // 2
    ):
        raise TemporalVideoRejected(
            "Pixel producer requires explicit dynamic context, committed single-frame batches and half-cache bound"
        )
    for spec in fields:
        if (type(spec.get("use_source_reference", False)) is not bool
                or spec.get("use_source_reference") and (
                    reference is None or not spec.get("source_tracking")
                    or not isinstance(spec.get("layout", {}).get("numeric_field_box"), list))):
            raise TemporalVideoRejected("Source-reference tracking requires explicit critical field geometry")
        if type(spec.get("source_tracking", False)) is not bool:
            raise TemporalVideoRejected("Explicit source tracking boolean required")
        if (type(spec.get("source_stable_refinement", False)) is not bool
                or spec.get("source_stable_refinement") and not spec.get("source_tracking")):
            raise TemporalVideoRejected("Stable refinement requires explicit source tracking")
        if (type(spec.get("source_geometry_translation", False)) is not bool
                or spec.get("source_geometry_translation") and not spec.get("source_tracking")):
            raise TemporalVideoRejected("LED pose correction requires explicit source tracking")
        tracked_fields += int(spec.get("source_tracking", False))
        if tracked_fields > 4:
            raise TemporalVideoCapacityExceeded("At most four bounded source panel tracks")
        key = tuple(
            _text(spec.get(n), n) for n in ("screen_id", "field_key", "profile_id")
        )
        if key in keys or spec.get("role") not in ROLES:
            raise TemporalVideoRejected(
                "Unique screen/field/profile and explicit role required"
            )
        keys.add(key)
        if dynamic:
            _text(spec.get("dynamic_panel_id"), "dynamic panel")
            _text(spec.get("dynamic_track_id"), "dynamic track")
        box = spec.get("roi_box")
        if not isinstance(box, list) or len(box) != 4:
            raise TemporalVideoRejected("An upright normalized field ROI is required")
        x1, y1, x2, y2 = map(exact, box)
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise TemporalVideoRejected("Temporal field ROI escaped the source")
        layout = spec.get("layout")
        if (
            not isinstance(layout, dict)
            or not layout.get("revision")
            or layout.get("display_type") not in {"seven_segment", "lcd"}
        ):
            raise TemporalVideoRejected("Explicit versioned display profile required")
        if spec.get("unit") and spec.get("unit_basis") not in {
            "verified_on_panel",
            "verified_instrument_specification",
        }:
            raise TemporalVideoRejected("Field units require a verified basis")
    if media is not None:
        if (
            media["kind"] != "video"
            or media["stream_key"] != document["source_stream_key"]
            or (media["session_key"] or None) != document.get("source_session_key")
        ):
            raise TemporalVideoRejected(
                "Temporal OCR registration differs from its video source/session"
            )
    if asset is not None and (
        asset["version"] != document["media_version"]
        or asset["verified_sha256"] != document["source_sha256"]
        or asset["evidence_status"] != "available"
    ):
        raise Conflict("Temporal OCR source version/hash is unavailable or changed")
    return document


def resolve_dynamic_context(
    query, field, ref, registration, start, end, *, revision=None
):
    """Resolve this panel over its actual support range; focus is never identity.

    The dynamic API uses media-relative seconds. Original PTS/time_base and the
    stream origin remain separately pinned in each temporal source reference.
    """
    config = registration.get("dynamic_context")
    if not config:
        return dict(
            enabled=False,
            association_status="registered_not_physically_verified"
            if field.get("instrument_id")
            else "unknown",
            instrument_id=field.get("instrument_id"),
            business_context_status="unknown",
        )
    if not callable(query):
        raise TemporalVideoRejected(
            "Configured dynamic context requires its existing query API"
        )
    kwargs = {
        k: config[k] for k in ("revision", "known_at") if config.get(k) is not None
    }
    if revision is not None:
        kwargs["revision"] = revision
    response = query(
        ref["media_id"],
        registration["source_session_key"],
        str(start),
        str(end),
        **kwargs,
    )
    if (
        not isinstance(response, dict)
        or response.get("schema_version") != "vc-lab-dynamic-context/1"
        or len(json.dumps(response, ensure_ascii=False).encode()) > 128 * 1024
    ):
        raise TemporalVideoRejected("Invalid bounded dynamic context response")
    if response.get("status") != "unknown" and (
        str(response.get("media_id")) != str(ref["media_id"])
        or response.get("source_session_key") != registration["source_session_key"]
        or response.get("source_stream_key") != registration["source_stream_key"]
    ):
        raise TemporalVideoRejected(
            "Dynamic context escaped the registered source/session"
        )
    panels = [
        p
        for p in response.get("per_panel", [])
        if p.get("panel_id") == field["dynamic_panel_id"]
        and p.get("track_id") == field["dynamic_track_id"]
    ]
    if len(panels) > 1:
        raise TemporalVideoRejected("Ambiguous dynamic panel/track identity")
    panel = panels[0] if panels else {}
    supported = (
        panel.get("physical_association_status") == "spatial_identity_supported"
        and bool(panel.get("ocr_identity_key"))
        and panel.get("instrument_id") is not None
    )
    if (
        supported
        and field.get("instrument_id")
        and panel["instrument_id"] != field["instrument_id"]
    ):
        supported = False
    notifications = [
        n
        for n in response.get("ocr_notifications", [])
        if n.get("track_id") == field["dynamic_track_id"]
        and n.get("panel_id") == field["dynamic_panel_id"]
        and n.get("ocr_reset") is True
        and Fraction(n["source_seconds"]) <= Fraction(end)
    ]
    return dict(
        enabled=True,
        schema_version=response["schema_version"],
        revision=response.get("revision"),
        revision_id=response.get("revision_id"),
        knowledge_processed_at=response.get("knowledge_processed_at"),
        requested_media_offset_range=[str(start), str(end)],
        source_time_conversion="original_pts_times_time_base_minus_stream_origin",
        panel_id=field["dynamic_panel_id"],
        track_id=field["dynamic_track_id"],
        epoch=panel.get("epoch"),
        association_status="spatial_identity_supported" if supported else "uncertain",
        instrument_id=panel.get("instrument_id") if supported else None,
        ocr_identity_key=panel.get("ocr_identity_key") if supported else None,
        business_context_status=panel.get("business_context_status", "unknown"),
        current_focus_used_for_identity=False,
        reset_notifications=notifications[-32:],
        segments=panel.get("segments", []),
        experimental_or_material_completion_implied=False,
    )


# A fixed child program: only source path and bounds are data; no shell or user code.
_DECODER = r"""
import av,json,struct,sys
from fractions import Fraction
av.logging.set_level(av.logging.ERROR)
p,start,end,limit=sys.argv[1:];start,end=Fraction(start),Fraction(end);limit=int(limit)
with av.open(p) as c:
 s=c.streams.video[0];s.codec_context.thread_count=1
 tb=Fraction(s.time_base);origin=Fraction(s.start_time or 0)*tb
 if start>0:c.seek(int((origin+start)/tb),stream=s,backward=True,any_frame=False)
 count=0
 for frame in c.decode(s):
  if frame.pts is None:raise ValueError('Decoded frame PTS missing')
  ftb=Fraction(frame.time_base);offset=Fraction(frame.pts)*ftb-origin
  if offset<start:continue
  if offset>=end:break
  count+=1
  if count>limit:raise ValueError('Frame bound exceeded')
  if frame.width*frame.height>16000000:raise ValueError('Frame pixel bound exceeded')
  raw=frame.to_ndarray(format='bgr24').tobytes()
  meta=json.dumps(dict(pts=frame.pts,time_base=str(ftb),media_offset=str(offset),stream_origin=str(origin),width=frame.width,height=frame.height)).encode()
  sys.stdout.buffer.write(struct.pack('!II',len(meta),len(raw))+meta+raw);sys.stdout.buffer.flush()
 if not count:raise ValueError('No original PTS frames in registered interval')
"""


def validate_provider(document):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != "vc-lab-temporal-video-provider/1"
        or document.get("enabled") is not True
    ):
        raise TemporalVideoRejected("Temporal video provider is not explicitly enabled")
    _text(document.get("revision"), "temporal provider revision")
    executable = Path(document.get("decoder_python", ""))
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or str(executable.resolve()).startswith(("/mnt/", "/media/"))
    ):
        raise TemporalVideoRejected(
            "An existing local decoder Python interpreter is required"
        )
    for key, default, lower, upper in [
        ("max_input_frames", 600, 1, 600),
        ("decoder_wait_seconds", 30, 1, 180),
        ("max_execution_seconds", 300, 1, 1800),
    ]:
        value = document.get(key, default)
        if type(value) is not int or not lower <= value <= upper:
            raise TemporalVideoRejected("Invalid temporal decoder bound")
    return document


@contextmanager
def source_guard(path, settings):
    path = Path(path)
    if path.absolute().resolve() != path.absolute():
        raise TemporalVideoRejected("Temporal source cannot traverse symbolic links")
    if (
        shutil.disk_usage("/").free < 50 * 1024**3
        or shutil.disk_usage(settings.runtime_root).free < 80 * 1024**3
    ):
        raise TemporalVideoCapacityExceeded(
            "System/data filesystem free-space guard reached"
        )
    if is_nas(path):
        verify_root(path.parent, settings.nas_marker_sha256, project_destination=False)
        verify_required_roots(settings.nas_marker_sha256)
    elif str(path.absolute()).startswith(("/mnt/", "/media/")):
        raise TemporalVideoRejected("Unverified source mount")
    if path.is_symlink() or not path.is_file():
        raise TemporalVideoRejected("Original regular source file required")
    # Check all ancestors within the source location; an absent marker is not a seal.
    for parent in (path.parent, *path.parent.parents):
        marker = parent / ".uploading"
        if marker.exists() or marker.is_symlink():
            raise TemporalVideoRejected("Temporal source is still uploading")
    yield
    if is_nas(path):
        verify_root(path.parent, settings.nas_marker_sha256, project_destination=False)
        verify_required_roots(settings.nas_marker_sha256)


def decode_frames(path, start, end, provider):
    """Yield original PTS frames from a bounded, owned PyAV child; no fps filter."""
    validate_provider(provider)
    import numpy as np

    limit = provider.get("max_input_frames", 600)
    process = subprocess.Popen(
        [
            provider["decoder_python"],
            "-I",
            "-u",
            "-c",
            _DECODER,
            str(path),
            str(start),
            str(end),
            str(limit),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:

        def read(count, *, eof=False):
            chunks = bytearray()
            deadline = time.monotonic() + provider.get("decoder_wait_seconds", 30)
            with selectors.DefaultSelector() as poll:
                poll.register(process.stdout, selectors.EVENT_READ)
                while len(chunks) < count:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TemporalVideoRejected(
                            "Original frame decoder wait timed out"
                        )
                    if not poll.select(min(0.5, remaining)):
                        continue
                    block = os.read(
                        process.stdout.fileno(), min(65536, count - len(chunks))
                    )
                    if not block:
                        if eof and not chunks:
                            return None
                        raise TemporalVideoRejected("Truncated temporal decoder frame")
                    chunks.extend(block)
            return bytes(chunks)

        count = 0
        while True:
            header = read(8, eof=True)
            if header is None:
                break
            meta_size, pixel_size = struct.unpack("!II", header)
            if not 0 < meta_size <= 4096 or not 0 < pixel_size <= 48_000_000:
                raise TemporalVideoCapacityExceeded(
                    "Decoder frame exceeds its byte bound"
                )
            meta = json.loads(read(meta_size))
            if (
                type(meta.get("pts")) is not int
                or type(meta.get("width")) is not int
                or type(meta.get("height")) is not int
                or meta["width"] * meta["height"] * 3 != pixel_size
            ):
                raise TemporalVideoRejected(
                    "Decoder frame metadata differs from its pixels"
                )
            count += 1
            if count > limit:
                raise TemporalVideoCapacityExceeded(
                    "Original frame count exceeds bound"
                )
            raw = read(pixel_size)
            image = np.frombuffer(raw, dtype=np.uint8).reshape(
                meta["height"], meta["width"], 3
            )
            yield image, meta
        if process.wait(timeout=5):
            raise TemporalVideoRejected("Original PTS decoder failed")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()


def geometry_reference(path, spec, registration, provider):
    """Decode one pinned earlier original frame for stable geometry only.

    The reference never enters OCR, numeric evidence or event source times.
    Dynamic pixels are excluded by the existing source tracker.
    """
    import numpy as np

    offset = Fraction(spec["media_offset_seconds"])
    with closing(decode_frames(path, str(offset), str(offset + Fraction(1, 1000000)),
                               dict(provider, max_input_frames=1))) as frames:
        try:
            image, meta = next(frames)
        except StopIteration as exc:
            raise TemporalVideoRejected("Pinned reference frame unavailable") from exc
    if (image.shape[0] * image.shape[1] > 1048576
            or meta["pts"] != spec["pts"]
            or Fraction(meta["time_base"]) != Fraction(spec["time_base"])
            or Fraction(meta["media_offset"]) != offset
            or hashlib.sha256(image.tobytes()).hexdigest() != spec["source_frame_pixels_sha256"]):
        raise TemporalVideoRejected("Original geometry reference identity changed")
    upright = (np.ascontiguousarray(image[::-1, ::-1])
               if registration["rotation_degrees_clockwise"] == 180 else image)
    return upright, dict(spec, use="stable_structure_only", OCR_invocations=0,
                          numerical_evidence=False, physical_identity_inferred=False,
                          retained_bytes=upright.nbytes)


def context_identity(context):
    return fingerprint({k: context.get(k) for k in (
        "association_status", "instrument_id", "epoch", "ocr_identity_key", "reset_notifications"
    )})


class TemporalVideo:
    def __init__(self, settings):
        self.settings = settings

    def run(
        self,
        path,
        ref,
        registration,
        infer,
        provider,
        *,
        window_index=0,
        query_context=None,
        frame_producer=None,
    ):
        """Shared actual runner/replay entry; infer returns existing OCR result/receipt."""
        import cv2
        import numpy as np
        from .temporal_display import TemporalConfig, TemporalDisplay

        validate_registration(registration)
        validate_provider(provider)
        if bool(registration.get("dynamic_video")) != (frame_producer is not None):
            raise TemporalVideoRejected(
                "Registered pixel producer must be supplied before decoding"
            )
        if frame_producer is not None and any(
            not callable(getattr(frame_producer, k, None))
            for k in ("observe", "flush", "receipt")
        ):
            raise TemporalVideoRejected("Real pixel producer interface required")
        if registration.get("dynamic_context") and not callable(query_context):
            raise TemporalVideoRejected(
                "Dynamic query API required before source decoding"
            )
        if (
            ref["sha256"] != registration["source_sha256"]
            or ref["media_version"] != registration["media_version"]
        ):
            raise Conflict("Temporal source reference differs from registration")
        start, end = exact(ref["start_seconds"]), exact(ref["end_seconds"])
        if not any(
            start == exact(i["start_seconds"]) and end == exact(i["end_seconds"])
            for i in registration["source_intervals"]
        ):
            raise TemporalVideoRejected("Unregistered temporal source interval")
        started = time.monotonic()
        deadline = started + provider.get("max_execution_seconds", 300)
        receipts = []
        outputs = {}
        contexts = {}
        events = []
        last_pts = None
        count = 0
        track_epoch = 0
        roi_peak = 0
        actual_buffer_peak = 0
        combined_buffer_peak = 0
        decode_seconds = 0
        algorithm_seconds = 0
        ocr_seconds = 0
        core_budget = CACHE_BYTES // 2 if frame_producer is not None else CACHE_BYTES
        tracker_budget = 16 * 1024**2 if any(f.get("source_tracking") for f in registration["fields"]) else 0
        core_budget -= tracker_budget
        trackers = {}
        tracking_peak = tracking_failures = 0
        tracked_frames = geometry_pending = reacquisitions = 0
        engine = TemporalDisplay(
            TemporalConfig(max_buffer_bytes=core_budget, max_window_seconds=1.0)
        )
        original_stat = None
        reference_image = reference_receipt = None
        reference_contexts = {}
        with source_guard(path, self.settings):
            before = hash_file(Path(path), self.settings.max_native_bytes)
            if before[0] != ref["sha256"]:
                raise Conflict("Original temporal source changed")
            original_stat = Path(path).stat()
            if registration.get("source_tracking_reference") is not None:
                tick = time.monotonic()
                reference_image, reference_receipt = geometry_reference(path,
                    registration["source_tracking_reference"], registration, provider)
                decode_seconds += time.monotonic() - tick
            with closing(
                decode_frames(path, str(start), str(end), provider)
            ) as iterator:
                while True:
                    if time.monotonic() > deadline:
                        raise TemporalVideoCapacityExceeded(
                            "Temporal execution wall-time bound reached"
                        )
                    tick = time.monotonic()
                    try:
                        image, meta = next(iterator)
                    except StopIteration:
                        break
                    decode_seconds += time.monotonic() - tick
                    count += 1
                    pts = Fraction(meta["pts"]) * Fraction(meta["time_base"])
                    if last_pts is not None and pts <= last_pts:
                        iterator.close()
                        raise TemporalVideoRejected(
                            "Duplicate or out-of-order original frame PTS"
                        )
                    offset = Fraction(meta["media_offset"])
                    if not Fraction(start) <= offset < Fraction(end):
                        raise TemporalVideoRejected(
                            "Decoded PTS escaped the registered source interval"
                        )
                    gap = last_pts is not None and pts - last_pts > Fraction("0.2")
                    if gap:
                        events.extend(
                            self._event(x, ref, registration, query_context)
                            for x in outputs.values()
                        )
                        outputs.clear()
                        contexts.clear()
                        track_epoch += 1
                        engine = TemporalDisplay(
                            TemporalConfig(
                                max_buffer_bytes=core_budget, max_window_seconds=1.0
                            )
                        )
                    last_pts = pts
                    source_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    h, w = image.shape[:2]
                    matrix = (
                        np.array(
                            [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=float
                        )
                        if registration["rotation_degrees_clockwise"] == 180
                        else np.eye(3)
                    )
                    upright = (
                        np.ascontiguousarray(image[::-1, ::-1])
                        if registration["rotation_degrees_clockwise"] == 180
                        else image
                    )
                    if frame_producer is not None:
                        try:
                            marker = frame_producer.observe(
                                upright,
                                dict(
                                    pts=meta["pts"],
                                    time_base=meta["time_base"],
                                    stream_origin=meta["stream_origin"],
                                    media_offset=str(offset),
                                    source_sha256=ref["sha256"],
                                    original_bgr_sha256=source_hash,
                                    source_to_upright=matrix.tolist(),
                                    upright_rgb_sha256=hashlib.sha256(
                                        upright[:, :, ::-1].tobytes()
                                    ).hexdigest(),
                                ),
                            )
                        except ValueError as exc:
                            raise TemporalVideoRejected(
                                "Real dynamic pixel producer rejected this frame"
                            ) from exc
                        if (
                            not isinstance(marker, dict)
                            or marker.get("status") != "committed"
                            or marker.get("pending_frames") != 0
                        ):
                            raise TemporalVideoRejected(
                                "Dynamic pixels must commit before this frame's OCR identity query"
                            )
                        used = frame_producer.receipt().get("buffer_bytes")
                        if (
                            type(used) is not int
                            or used < 0
                            or used > CACHE_BYTES // 2
                            or used + engine.buffer_bytes > CACHE_BYTES
                        ):
                            raise TemporalVideoCapacityExceeded(
                                "Combined producer and temporal ROI cache exceeds 64 MiB"
                            )
                        combined_buffer_peak = max(
                            combined_buffer_peak, used + engine.buffer_bytes
                        )
                    estimated = 0
                    for field in registration["fields"]:
                        x1, y1, x2, y2 = map(float, field["roi_box"])
                        estimated += (
                            max(1, math.ceil((x2 - x1) * w))
                            * max(1, math.ceil((y2 - y1) * h))
                            * 3
                            * 30
                        )
                    if estimated > core_budget:
                        raise TemporalVideoCapacityExceeded(
                            "Registered ROI ring would exceed its temporal core cache budget"
                        )
                    roi_peak = max(roi_peak, estimated)
                    for field in registration["fields"]:
                        key = (
                            field["screen_id"],
                            field["field_key"],
                            field["profile_id"],
                        )
                        context = resolve_dynamic_context(
                            query_context,
                            field,
                            ref,
                            registration,
                            offset,
                            offset,
                            revision=marker.get("revision")
                            if frame_producer is not None
                            else None,
                        )
                        physical_key = context_identity(context)
                        previous = contexts.get(key)
                        reset = (
                            previous is not None
                            and previous["physical_key"] != physical_key
                        )
                        if reset:
                            if key in outputs:
                                events.append(
                                    self._event(
                                        outputs.pop(key),
                                        ref,
                                        registration,
                                        query_context,
                                    )
                                )
                            engine.reset(
                                previous["identity"],
                                reason="dynamic_panel_identity_or_epoch_changed",
                            )
                        x1, y1, x2, y2 = map(float, field["roi_box"])
                        left, top = int(x1 * w), int(y1 * h)
                        right, bottom = math.ceil(x2 * w), math.ceil(y2 * h)
                        roi = np.ascontiguousarray(upright[top:bottom, left:right])
                        active_layout = field["layout"]
                        tracked_registration = None
                        tracking_failure = None
                        if field.get("source_tracking"):
                            from .source_tracking import SourcePanelTracker
                            use_reference = field.get("use_source_reference", False)
                            if use_reference and key not in reference_contexts:
                                pos = Fraction(reference_receipt["media_offset_seconds"])
                                reference_contexts[key] = context_identity(resolve_dynamic_context(
                                        query_context, field, ref, registration, pos, pos,
                                        revision=marker.get("revision") if frame_producer is not None else None,
                                    ))
                            if key not in trackers or trackers[key][0] != physical_key:
                                trackers[key] = (physical_key, SourcePanelTracker(
                                    reference_image if use_reference else upright,
                                    field["roi_box"], field["layout"]["dynamic_boxes"],
                                    refine_stable_translation=field.get("source_stable_refinement", False),
                                    **(dict(required_boxes=[field["layout"]["numeric_field_box"]])
                                       if use_reference else {}),
                                ))
                                trackers[key][1].layout = None
                                if use_reference:
                                    trackers[key][1].current_points = None
                            tracker = trackers[key][1]
                            if gap:
                                tracker.current_points = None
                            tracked = (dict(status="pending", reason="reference_panel_context_changed")
                                       if use_reference and reference_contexts[key] != physical_key
                                       else tracker.observe(upright))
                            used = sum(t.buffer_bytes for _, t in trackers.values()) + (
                                reference_image.nbytes if reference_image is not None else 0)
                            tracking_peak = max(tracking_peak, used)
                            if used > tracker_budget:
                                raise TemporalVideoCapacityExceeded("Source tracker memory bound exceeded")
                            if tracked["status"] == "tracked":
                                tracked_frames += 1
                                reacquisitions += int(tracked["registration"]["reacquired_after_loss"])
                                roi = tracked["image"]
                                left, top, right, bottom = tracked["box"]
                                tracked_registration = tracked["registration"]
                                if field.get("source_geometry_translation"):
                                    if tracker.layout is None:
                                        from .emission_geometry import reanchor
                                        rectified = cv2.warpPerspective(roi,
                                            np.asarray(tracked_registration["matrix_current_to_reference"]),
                                            (tracker.reference_shape[1], tracker.reference_shape[0]))
                                        tracker.layout, geometry = reanchor(rectified, field["layout"])
                                        if tracker.layout is None:
                                            tracking_failure = "initial_LED_geometry_pending:" + geometry["reason"]
                                            geometry_pending += 1
                                        else:
                                            tracker.layout["initial_geometry_observation"].update(
                                                original_source_bgr_sha256=source_hash,
                                                source_pts=meta["pts"], source_time_base=meta["time_base"],
                                                geometry_input_was_rectified=True,
                                                OCR_input_was_rectified=False)
                                    if tracker.layout is not None:
                                        active_layout = tracker.layout
                            else:
                                tracking_failure = tracked["reason"]
                        if not roi.size or roi.shape[0] * roi.shape[1] > 262144:
                            raise TemporalVideoCapacityExceeded(
                                "Temporal field ROI exceeds native pixel bound"
                            )
                        identity = dict(
                            device_id=context.get("instrument_id")
                            or registration["source_stream_key"],
                            screen_id=field["screen_id"],
                            field_id=field["field_key"],
                            track_session_id=fingerprint(
                                dict(
                                    source=ref["sha256"],
                                    session=registration.get("source_session_key"),
                                    window=window_index,
                                    profile=field["profile_id"],
                                    source_gap_epoch=track_epoch,
                                    dynamic_physical_key=physical_key,
                                )
                            ),
                        )
                        contexts[key] = dict(
                            physical_key=physical_key, identity=identity
                        )
                        if gap:
                            engine.reset(identity, reason="original_pts_gap")
                        transform = (
                            np.array(
                                [[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=float
                            )
                            @ matrix
                        ).tolist()
                        source_ref = dict(
                            source_sha256=ref["sha256"],
                            pts=meta["pts"],
                            time_base=meta["time_base"],
                            media_id=ref["media_id"],
                            media_offset_seconds=str(offset),
                            source_frame_pixels_sha256=source_hash,
                            **(dict(upright_frame_pixels_sha256=hashlib.sha256(upright.tobytes()).hexdigest())
                               if field.get("source_tracking") else {}),
                            source_frame_pixels_format="bgr24",
                            source_to_roi=transform,
                            source_to_upright=matrix.tolist(),
                            stream_origin=meta["stream_origin"],
                            dynamic_physical_key=physical_key,
                            dynamic_context_revision=context.get("revision"),
                        )

                        def ocr(pixels, *, anchor_source=None):
                            nonlocal ocr_seconds
                            tick = time.monotonic()
                            ok, encoded = cv2.imencode(".png", pixels)
                            if not ok or encoded.nbytes > 12 * 1024**2:
                                raise TemporalVideoCapacityExceeded(
                                    "Temporal ROI OCR payload exceeds bound"
                                )
                            raw = encoded.tobytes()
                            result, execution = infer(raw)
                            parsed = tokens(result)
                            texts = [p["text"] for p in parsed if p["text"].strip()]
                            receipts.append(
                                dict(
                                    execution=execution,
                                    derivation=dict(
                                        kind="temporal_field_roi",
                                        source=anchor_source if anchor_source is not None else source_ref,
                                        derived_sha256=hashlib.sha256(raw).hexdigest(),
                                        derived_bytes=len(raw),
                                        field_key=field["field_key"],
                                        raw_tokens=parsed,
                                    ),
                                )
                            )
                            ocr_seconds += time.monotonic() - tick
                            return dict(
                                raw_text=" ".join(texts),
                                candidates=texts,
                                score=min(
                                    (float(p["score"]) for p in parsed), default=0
                                ),
                                unit=None,
                            )

                        tick = time.monotonic()
                        if tracking_failure:
                            tracking_failures += 1
                            engine.reset(identity, reason="source_panel_tracking_lost")
                            result = engine._rejected(identity, source_ref,
                                "source_panel_tracking_lost", [tracking_failure])
                        else:
                            result = engine.observe(
                                roi,
                                identity=identity,
                                source_ref=source_ref,
                                layout=active_layout,
                                ocr=ocr if infer is not None else None,
                                visibility=None,
                                **(dict(source_registration=tracked_registration)
                                   if tracked_registration is not None else {}),
                            )
                        algorithm_seconds += time.monotonic() - tick
                        actual_buffer_peak = max(
                            actual_buffer_peak, engine.buffer_bytes
                        )
                        if frame_producer is not None:
                            used = frame_producer.receipt()["buffer_bytes"]
                            if used + engine.buffer_bytes > CACHE_BYTES:
                                raise TemporalVideoCapacityExceeded(
                                    "Combined producer and temporal ROI cache exceeds 64 MiB"
                                )
                            combined_buffer_peak = max(
                                combined_buffer_peak, used + engine.buffer_bytes
                            )
                        if result.get("status") not in {
                            "confirmed",
                            "uncertain",
                            "unreadable",
                        }:
                            raise TemporalVideoRejected(
                                "Invalid temporal algorithm status"
                            )
                        signature = fingerprint(
                            {
                                k: result.get(k)
                                for k in (
                                    "status",
                                    "value_text",
                                    "candidates",
                                    "method",
                                    "observation_time_start",
                                )
                            }
                        )
                        if key in outputs and outputs[key]["signature"] != signature:
                            events.append(
                                self._event(
                                    outputs[key], ref, registration, query_context
                                )
                            )
                        outputs[key] = dict(
                            signature=signature,
                            result=result,
                            field=field,
                            source_ref=source_ref,
                            dynamic_context=context,
                        )
                        if len(events) > 450:
                            raise TemporalVideoCapacityExceeded(
                                "Temporal state transitions exceed task result bound"
                            )
                    if (
                        sum(
                            len(json.dumps(x, ensure_ascii=False).encode())
                            for x in [events, receipts, outputs_to_list(outputs)]
                        )
                        > 6 * 1024**2
                    ):
                        raise TemporalVideoCapacityExceeded(
                            "Temporal window evidence exceeds checkpoint bound"
                        )
                    del upright, image
            if frame_producer is not None:
                try:
                    marker = frame_producer.flush()
                except ValueError as exc:
                    raise TemporalVideoRejected(
                        "Real dynamic pixel producer could not finalize this window"
                    ) from exc
                if not isinstance(marker, dict) or marker.get("pending_frames") != 0:
                    raise TemporalVideoRejected(
                        "Pixel producer has uncommitted frames at window end"
                    )
            events.extend(
                self._event(x, ref, registration, query_context)
                for x in outputs.values()
            )
            if before != hash_file(Path(path), self.settings.max_native_bytes):
                raise Conflict("Original temporal source changed during decode")
        receipt = dict(
            schema_version="vc-lab-temporal-video-execution/1",
            source=ref,
            source_stat=dict(
                bytes=original_stat.st_size, mtime_ns=original_stat.st_mtime_ns
            ),
            source_orientation=dict(
                rotation_degrees_clockwise=registration["rotation_degrees_clockwise"],
                basis=registration["orientation_basis"],
            ),
            registration_fingerprint=fingerprint(registration),
            provider_fingerprint=fingerprint(provider),
            original_pts_preserved=True,
            original_media_copied=False,
            **(dict(source_tracking=dict(peak_buffer_bytes=tracking_peak,
                                         limit_bytes=tracker_budget,
                                         pending_frame_fields=tracking_failures,
                                         tracked_frame_fields=tracked_frames,
                                         geometry_pending_frame_fields=geometry_pending,
                                         reacquisitions=reacquisitions,
                                         physical_identity_inferred=False))
               if tracker_budget else {}),
            ocr_provider_invocations=sum(1 for r in receipts if r.get("execution")),
            ocr_mode="not_invoked_geometry_only"
            if infer is None
            else "existing_provider_callback",
            input_frame_count=count,
            **(dict(geometry_reference=reference_receipt, additional_reference_frames=1)
               if reference_receipt is not None else {}),
            roi_ring_upper_bound_bytes=roi_peak,
            actual_roi_buffer_peak_bytes=actual_buffer_peak,
            combined_producer_roi_peak_bytes=combined_buffer_peak
            if frame_producer is not None
            else actual_buffer_peak,
            dynamic_video=frame_producer.receipt()
            if frame_producer is not None
            else None,
            roi_cache_limit_bytes=CACHE_BYTES,
            roi_cache_retention_seconds=86400,
            output_limit_bytes=OUTPUT_BYTES,
            timing=dict(
                decode_wait_seconds=decode_seconds,
                algorithm_including_ocr_seconds=algorithm_seconds,
                ocr_seconds=ocr_seconds,
                total_seconds=time.monotonic() - started,
            ),
        )
        receipts.append(dict(temporal_execution=receipt))
        return jsonb_stable(events), jsonb_stable(receipts)

    @staticmethod
    def _event(output, ref, registration, query_context=None):
        result, field = output["result"], output["field"]
        source_refs = result.get("source_refs") or [output["source_ref"]]
        offsets = [
            Fraction(r["media_offset_seconds"])
            for r in source_refs
            if r.get("media_offset_seconds") is not None
        ]
        if not offsets:
            raise TemporalVideoRejected("Temporal output lacks original media offsets")
        if any(
            r.get("source_sha256") != ref["sha256"]
            or r.get("media_id") != ref["media_id"]
            for r in source_refs
        ):
            raise TemporalVideoRejected("Temporal result references a different source")
        if min(offsets) < Fraction(ref["start_seconds"]) or max(offsets) >= Fraction(
            ref["end_seconds"]
        ):
            raise TemporalVideoRejected(
                "Temporal support escaped the original source window"
            )
        if result["status"] == "confirmed" and (
            result.get("gates", {}).get("confirmed") is not True
            or not isinstance(result.get("value_text"), str)
        ):
            raise TemporalVideoRejected(
                "Temporal confirmation lacks computed visual gates"
            )
        lo, hi = min(offsets), max(offsets)
        context = resolve_dynamic_context(
            query_context,
            field,
            ref,
            registration,
            lo,
            hi,
            revision=output.get("dynamic_context", {}).get("revision"),
        )
        # A supported frame has a media position; do not invent the following duration.
        lo_text = str(
            (Decimal(lo.numerator) / Decimal(lo.denominator)).quantize(
                Decimal(".000001"), rounding=ROUND_FLOOR
            )
        )
        hi_text = str(
            (Decimal(hi.numerator) / Decimal(hi.denominator)).quantize(
                Decimal(".000001"), rounding=ROUND_CEILING
            )
        )
        if exact(hi_text) == exact(lo_text):
            # Timeline evidence spans for video must be positive. A tiny declared
            # media interval contains that exact PTS without asserting persistence.
            hi_text = str(
                min(exact(ref["end_seconds"]), exact(lo_text) + Decimal(".000001"))
            )
        value = result.get("value_text") if result["status"] == "confirmed" else None
        candidate = dict(
            field_key=field["field_key"],
            role=field["role"],
            channel=field.get("channel", ""),
            unit=field.get("unit"),
            unit_basis=field.get("unit_basis"),
            value=value,
            value_text=value,
            status=result["status"],
            tentative_value=result.get("numeric_candidate"),
            candidates=result.get("candidates", []),
            raw_text=result.get("raw_text", ""),
            method=result.get("method"),
            temporal_display=result,
            screen_id=field["screen_id"],
            profile_id=field["profile_id"],
            auto_confirmed=False,
            accuracy="NOT_PROVEN",
            reading_status=result.get("reading_status", result["status"]),
            association_status=context["association_status"],
            dynamic_context=context,
        )
        shown = value or result.get("display_text") or result.get("numeric_candidate")
        text = f"时序面板候选 {field['field_key']}：{shown if shown is not None else '读数未确定'}；{result['status']}"
        return dict(
            kind="reading",
            text=text,
            spans=[
                dict(
                    media_id=ref["media_id"], start_seconds=lo_text, end_seconds=hi_text
                )
            ],
            instrument_id=context.get("instrument_id"),
            fields=[candidate],
            payload=dict(
                field_mapping="temporal_registered_video",
                temporal_display=result,
                registration_revision=registration["revision"],
                observation_time_basis="original_media_pts",
                evidence_span_resolution="microsecond_envelope_exact_fraction_pts_in_source_refs",
                temporal_world_clock_status=registration["world_clock_status"],
                business_confirmation=False,
                external_confirmation="not_provided",
                association_status=context["association_status"],
                dynamic_context=context,
            ),
        )


def outputs_to_list(outputs):
    return list(outputs.values())
