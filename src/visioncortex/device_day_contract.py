"""Frozen device/day archive layout; no model, NAS discovery or deletion on import."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .stage_execution import STAGES as EXECUTION_STAGES

VERSION = "visioncortex-device-day/1"
TIMEZONE = "Asia/Shanghai"
CHUNK_SECONDS = 900
DIRECTORIES = (
    "MetaVideo",
    "ProcessedClips",
    "MultimodalUnderstanding",
    "LaboratoryDailyReport",
    "Comment",
)
ACTIVITY_LABELS = {"active": "有实验活动", "inactive": "无实验活动"}
ACTIVITY_FOLDERS = {"active": "ExperimentActivity", "inactive": "NoExperimentActivity"}
PHYSICAL_ACTION_TYPES = frozenset({
    "hand_object_contact", "object_movement", "liquid_movement",
    "container_state_change", "device_panel_operation",
})
STAGES = ("retention", "vision", "stt", "understanding", "report")
DEPENDENCIES = {name: EXECUTION_STAGES[name].parents for name in STAGES}


def digest(value: Any) -> str:
    result = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
    for block in _json_blocks(value, encoder):
        result.update(block)
    return result.hexdigest()


def _json_blocks(value, encoder, block_bytes=1024 * 1024):
    """Bound serialization copies; the caller's object and one string remain owned.

    Compact digest subtrees use the stdlib C encoder. Pretty output retains
    iterencode's exact whitespace. One individual large string can exceed the
    block limit, as before; neither path duplicates the whole serialized day.
    """
    buffer = bytearray()
    for chunk in _json_fragments(value, encoder):
        encoded = chunk.encode('utf-8')
        if len(encoded) >= block_bytes:
            if buffer:
                yield bytes(buffer)
                buffer.clear()
            yield encoded
        else:
            buffer.extend(encoded)
            if len(buffer) >= block_bytes:
                yield bytes(buffer)
                buffer.clear()
    if buffer:
        yield bytes(buffer)


def _json_subtree_fits(value, ancestors, *, budget=16384, nodes=512):
    """Bound temporary C output without a second traversal of the whole day."""
    pending = [value]
    while pending:
        item = pending.pop()
        nodes -= 1
        budget -= 8
        if nodes < 0 or budget < 0:
            return False
        if isinstance(item, str):
            budget -= len(item) * 6  # covers UTF-8 and every JSON escape
        elif isinstance(item, (list, tuple, dict)):
            if type(item) not in (list, tuple, dict) or id(item) in ancestors or len(item) > nodes:
                return False
            if isinstance(item, dict):
                pending.extend(item)
                pending.extend(item.values())
            else:
                pending.extend(item)
        elif isinstance(item, int):
            budget -= item.bit_length() // 3 + 1
        elif isinstance(item, float):
            budget -= 24  # longest finite float spelling plus punctuation
    # Cycles exhaust the node budget above; the streaming walker reports them.
    return budget >= 0


def _json_fragments(value, encoder):
    # Pretty-printing cannot use the stdlib C encoder on Python 3.12. Preserve
    # that proven stream and custom encoder behavior without extra traversal.
    if (type(encoder) is not json.JSONEncoder or encoder.indent is not None
            or encoder.item_separator != ',' or not encoder.check_circular
            or encoder.ensure_ascii or encoder.key_separator not in (':', ': ')
            or getattr(encoder.default, '__func__', None) is not json.JSONEncoder.default):
        yield from encoder.iterencode(value)
        return
    ancestors = set()
    def fragments(item):
        container = isinstance(item, (list, tuple, dict))
        if container and type(item) not in (list, tuple, dict):
            yield from encoder.iterencode(item)
            return
        if container and id(item) in ancestors:
            raise ValueError('Circular reference detected')
        if not container or _json_subtree_fits(item, ancestors):
            yield encoder.encode(item)  # indent=None selects the CPython C encoder
            return
        ancestors.add(id(item))
        try:
            first = True
            yield '{' if isinstance(item, dict) else '['
            if isinstance(item, dict):
                items = sorted(item.items()) if encoder.sort_keys else item.items()
                for key, child in items:
                    if not isinstance(key, (str, int, float, bool, type(None))):
                        if encoder.skipkeys:
                            continue
                        raise TypeError('keys must be str, int, float, bool or None, '
                                        f'not {key.__class__.__name__}')
                    # Let the same C encoder perform the stdlib's exact key
                    # coercion, float spelling, escaping and NaN validation.
                    key_text = encoder.encode({key: None})[1:-(len(encoder.key_separator) + 5)]
                    yield ('' if first else ',') + key_text + encoder.key_separator
                    first = False
                    yield from fragments(child)
            else:
                for offset in range(0, len(item), 8):
                    group = item[offset:offset + 8]
                    if len(group) > 1 and _json_subtree_fits(group, ancestors):
                        # Encode several adjacent small records in one C call.
                        text = encoder.encode(group)
                        yield ('' if first else ',') + text[1:-1]
                        first = False
                    else:
                        for child in group:
                            yield '' if first else ','
                            first = False
                            yield from fragments(child)
            yield '}' if isinstance(item, dict) else ']'
        finally:
            ancestors.remove(id(item))
    yield from fragments(value)


def media_sidecar_name(source: Path) -> str:
    """Readable file names; source identity remains in the retained mapping."""
    stem = "".join(word.capitalize() for word in re.split(r"[ _-]+", source.stem))
    return stem + source.suffix.lower()


def media_time_name(milliseconds: float) -> str:
    value = max(0, round(milliseconds))
    seconds, millis = divmod(value, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}.{millis:03d}"


def media_interval_name(start_ms: float, end_ms: float) -> str:
    return f"{media_time_name(start_ms)}_{media_time_name(end_ms)}"


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, allow_nan=False)
    def blocks():
        yield from _json_blocks(value, encoder)
        yield b'\n'
    _atomic_blocks(path, blocks())


def atomic_bytes(path: Path, value: bytes) -> None:
    _atomic_blocks(path, (value,))


def _atomic_blocks(path, blocks):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as handle:
            for block in blocks:
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_child(root: Path, relative: str) -> Path:
    """All public artifact references are archive-relative POSIX paths."""
    if not relative or "\\" in relative or ":" in relative:
        raise ValueError("Invalid archive-relative reference")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Invalid archive-relative reference")
    if relative.startswith(('Comment/Stt/', 'MultimodalUnderstanding/ClipUnderstanding/')):
        from .device_day_content_paths import aliases, relocated
        target = relocated(relative, aliases(root))
        if target != relative:
            return safe_child(root, target)
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact reference escapes archive")
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Symlink is not an archive artifact")
    return path


def archive_name(camera_key: str, start_us: int) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", camera_key):
        raise ValueError("Invalid camera key")
    if start_us <= 0:
        raise ValueError("Recording requires an absolute capture timestamp")
    date = datetime.fromtimestamp(start_us / 1e6, ZoneInfo(TIMEZONE)).date().isoformat()
    return f"{date}_{camera_key}"


def validate_archive_name(name: str) -> str:
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_([A-Za-z0-9][A-Za-z0-9_.-]{0,127})", name)
    if not match:
        raise ValueError("Invalid device/day archive name")
    datetime.strptime(match[1], "%Y-%m-%d")
    return name


def artifact(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    checked = safe_child(root, relative)
    return {"path": relative, "size_bytes": checked.stat().st_size,
            "sha256": file_hash(checked)}


def verify_artifact(root: Path, reference: dict[str, Any]) -> bool:
    try:
        path = safe_child(root, reference["path"])
        return (path.is_file() and path.stat().st_size == reference["size_bytes"]
                and file_hash(path) == reference["sha256"])
    except (KeyError, TypeError, ValueError, OSError):
        return False


class DeviceDayLayout:
    def __init__(self, archive_root: Path, camera_key: str, start_us: int, backend_root: Path | None = None):
        self.name = archive_name(camera_key, start_us)
        self.root = safe_child(archive_root, self.name)
        self.raw, self.processed, self.understanding, self.reports, self.comments = (
            self.root / item for item in DIRECTORIES
        )
        self.index = self.processed / "Index.json"
        self.backend_root = backend_root

    @property
    def receipts(self) -> Path:
        if self.backend_root is None:
            raise ValueError("Stage receipts require the configured backend storage root")
        return safe_child(self.backend_root, f"device-day-receipts/{self.name}")

    def backend_artifact(self, path: Path) -> dict[str, Any]:
        if self.backend_root is None:
            raise ValueError("Missing backend storage root")
        return artifact(self.backend_root, path) | {"storage_root": "local_cache_root"}

    def source_path(self, recording: dict, kind: str, source: Path) -> Path:
        def label(value):
            return datetime.fromtimestamp(value / 1e6, ZoneInfo(TIMEZONE)).strftime("%H-%M-%S")
        stem = f"{label(recording['recording_start_us'])}_{label(recording['recording_end_us'])}"
        if kind == "video":
            return self.raw / f"{stem}.mp4"
        if kind == "audio_audio":
            return self.raw / "Audio" / f"{stem}{source.suffix}"
        return self.raw / "Metadata" / stem / ("Audio" if kind.startswith("audio_") else "Video") / media_sidecar_name(source)

    def create(self) -> None:
        for name in DIRECTORIES:
            safe_child(self.root, name).mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        value = path.relative_to(self.root).as_posix()
        safe_child(self.root, value)
        return value


def validate_config(config: dict[str, Any]) -> None:
    settings = config.get("device_day") or {}
    if settings.get("schema_version", VERSION) != VERSION:
        raise ValueError("Device/day schema_version is frozen")
    chunk_seconds = settings.get("chunk_seconds", CHUNK_SECONDS)
    if isinstance(chunk_seconds, bool) or not isinstance(chunk_seconds, (int, float)) or not 0 < chunk_seconds <= CHUNK_SECONDS:
        raise ValueError("Device/day scan window must be positive and at most 900 seconds; source duration is unrestricted")
    if settings.get("timezone", TIMEZONE) != TIMEZONE:
        raise ValueError("Device/day timezone is frozen at Asia/Shanghai")
    if settings.get("delete_capture_sources", False):
        raise ValueError("Capture source deletion is suspended by user instruction")
    for name in ("vision_workers", "stt_workers", "understanding_workers", "inactive_frames", "active_frames_per_window"):
        value = settings.get(name, 2 if name.endswith("workers") else 8)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"device_day.{name} must be a positive integer")
    if settings.get("enabled"):
        storage = config.get("storage") or {}
        ingest = config.get("collection_ingest") or {}
        if not ingest.get("enabled") or not ingest.get("source_root"):
            raise ValueError("Device/day capture requires an explicit enabled collection source")
        for name in ("archive_root", "local_runtime_root", "local_cache_root"):
            if not storage.get(name):
                raise ValueError(f"Device/day requires storage.{name}")


def validate_segment(record: dict[str, Any]) -> None:
    for key in ("segment_id", "recording_id", "activity", "start_ms", "end_ms",
                "source_ref", "key_frames", "scene_frames", "evidence_status", "physical_action_confirmed"):
        if key not in record:
            raise ValueError(f"Segment missing {key}")
    if record["activity"] not in ACTIVITY_LABELS:
        raise ValueError("Only resolved activity results may be published as segments")
    if not 0 <= record["start_ms"] < record["end_ms"]:
        raise ValueError("Invalid segment time interval")
    if record["physical_action_confirmed"] is not False:
        raise ValueError("Single-camera activity screening cannot confirm physical actions")
    if record["evidence_status"] != "PARTIAL_EVIDENCE":
        raise ValueError("Device activity screening is partial evidence")
    if not record["scene_frames"]:
        raise ValueError("Every interval requires traceable scene samples")
    if record["activity"] == "inactive" and record["key_frames"]:
        raise ValueError("Inactive intervals cannot publish action keyframes")
    if any(f.get("frame_kind") != "scene_sample" for f in record["scene_frames"]):
        raise ValueError("Scene samples must identify their frame kind")
    for frame in record["key_frames"]:
        if (frame.get("frame_kind") != "action_keyframe" or frame.get("action_type") not in PHYSICAL_ACTION_TYPES
                or not frame.get("event_id") or not frame.get("selected_event_digest")):
            raise ValueError("Action keyframes require a selected five-class physical event")
