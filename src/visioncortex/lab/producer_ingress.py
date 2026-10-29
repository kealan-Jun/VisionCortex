"""Opt-in admission from the existing producer inventory, without a new receiver.

The inventory and producer archive are read-only. A local checkpoint excludes
already-finalized history on first activation. Only existing exact retained clips
can become complete-clip model inputs; capture packages are never such clips.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import subprocess
import tempfile
import time

from .components import inspect_recorder, pipeline_module
from .domain import Conflict, clean, fingerprint
from .ingest import Ingest
from .media import hash_file, safe_path
from .native_refresh import _directory
from .retained_clips import RetainedClips
from .storage import publication_config
from .timeline import Timeline, offset_time

SCHEMA = "vc-lab-producer-ingress/1"
ENV = "VC_LAB_PRODUCER_INGRESS_MANIFEST"
_STATE_LIMIT = 8 * 1024**2
_REMOTE = {"cifs", "smb3", "nfs", "nfs4", "fuse.sshfs"}


def _local(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute() or str(path).startswith(("/mnt/", "/media/")):
        raise ValueError("Producer state/inventory must be explicit local storage")
    mount = json.loads(
        subprocess.check_output(
            ["findmnt", "-J", "-T", str(path), "-o", "FSTYPE"], text=True, timeout=10
        )
    )
    if not mount.get("filesystems") or any(
        x.get("fstype") in _REMOTE for x in mount["filesystems"]
    ):
        raise ValueError("Producer state/inventory cannot be on a network filesystem")
    if (
        path.is_symlink()
        or path.resolve() != path
        or not (path.is_dir() if directory else path.is_file())
    ):
        raise ValueError("Existing canonical producer local file/directory required")
    return path


def _relative(value, *, empty=False):
    if value == "" and empty:
        return ""
    if (
        type(value) is not str
        or not value
        or len(value) > 2000
        or PurePosixPath(value).is_absolute()
        or "\\" in value
        or any(p in {"", ".", ".."} for p in value.split("/"))
    ):
        raise ValueError("Normalized producer-relative path required")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate producer configuration key")
        result[key] = value
    return result


def load_manifest(settings):
    value = getattr(settings, "producer_ingress_manifest", None)
    if value is None:
        return None
    path = _local(value)
    if (
        path.stat().st_uid != os.getuid()
        or stat.S_IMODE(path.stat().st_mode) & 0o077
        or path.stat().st_size > 65536
    ):
        raise ValueError("Private bounded owner-only producer manifest required")
    raw = path.read_bytes()
    doc = json.loads(raw, object_pairs_hook=_pairs)
    allowed = {
        "schema_version",
        "enabled",
        "revision",
        "inventory_path",
        "source_root",
        "inventory_relative_prefix",
        "mode",
        "recording_ids",
        "max_inventory_rows",
        "max_inventory_bytes",
        "max_admissions_per_tick",
        "max_index_bytes",
        "max_index_segments",
        "max_clips_per_tick",
        "actor",
        "reason",
    }
    if (
        not isinstance(doc, dict)
        or set(doc) - allowed
        or doc.get("schema_version") != SCHEMA
        or type(doc.get("enabled")) is not bool
    ):
        raise ValueError("Explicit versioned producer manifest required")
    if not doc["enabled"]:
        return None
    for key in ("revision", "actor", "reason"):
        if type(doc.get(key)) is not str or not 1 <= len(doc[key]) <= 200:
            raise ValueError("Explicit producer revision/actor/reason required")
    inventory = _local(doc["inventory_path"])
    expected = settings.source_root or settings.runtime_root / "incoming"
    if doc.get("source_root") != str(expected):
        raise ValueError("Producer source root differs from the exact application root")
    _relative(doc.get("inventory_relative_prefix", ""), empty=True)
    if doc.get("mode") not in {"new_only", "explicit_ids"}:
        raise ValueError("Producer mode must be new_only or explicit_ids")
    ids = doc.get("recording_ids", [])
    if (
        not isinstance(ids, list)
        or len(ids) > 32
        or any(type(x) is not str or not 1 <= len(x) <= 128 for x in ids)
        or len(set(ids)) != len(ids)
        or (doc["mode"] == "explicit_ids") != bool(ids)
    ):
        raise ValueError(
            "Only an explicit bounded recording ID list may backfill history"
        )
    bounds = {
        "max_inventory_rows": (10000, 1, 10000),
        "max_inventory_bytes": (16 * 1024**2, 1024, 16 * 1024**2),
        "max_admissions_per_tick": (4, 1, 16),
        "max_index_bytes": (32 * 1024**2, 1024, 32 * 1024**2),
        "max_index_segments": (2048, 1, 4096),
        "max_clips_per_tick": (4, 1, 8),
    }
    for key, (default, lo, hi) in bounds.items():
        doc.setdefault(key, default)
        if type(doc[key]) is not int or not lo <= doc[key] <= hi:
            raise ValueError("Producer resource bound is invalid: " + key)
    return dict(
        doc,
        inventory_path=str(inventory),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


def inventory_snapshot(doc):
    """Use the real inventory schema in one read-only SQLite transaction."""
    path = _local(doc["inventory_path"])
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        count, size = db.execute(
            "SELECT count(*),coalesce(sum(length(payload)),0) FROM observations"
        ).fetchone()
        if count > doc["max_inventory_rows"] or size > doc["max_inventory_bytes"]:
            raise Conflict(
                "Existing producer inventory exceeds the declared bounded snapshot"
            )
        rows = []
        for identity, payload in db.execute(
            "SELECT id,payload FROM observations ORDER BY id"
        ):
            if len(payload.encode()) > 256 * 1024:
                raise Conflict("Producer inventory record exceeds bound")
            row = json.loads(payload)
            if not isinstance(row, dict) or row.get("recording_id") != identity:
                raise Conflict("Producer inventory identity is inconsistent")
            rows.append(row)
        revision = db.execute(
            "SELECT value FROM metadata WHERE key='revision'"
        ).fetchone()
    return rows, revision[0] if revision else None


def _save(path, value):
    raw = (
        json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    if len(raw) > _STATE_LIMIT:
        raise Conflict("Producer checkpoint exceeds bound")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".", delete=False) as out:
        temporary = Path(out.name)
        os.chmod(temporary, 0o600)
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def _read(root, relative, guard, limit):
    parts = PurePosixPath(_relative(relative)).parts
    with _directory(root, "/".join(parts[:-1]), guard) as parent:
        if parent is None:
            return None
        guard.validate()
        try:
            fd = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
        except FileNotFoundError:
            guard.validate()
            return None
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise Conflict("Producer JSON exceeds regular-file bound")
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > limit or (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise Conflict("Producer JSON changed while reading")
        guard.validate()
        return raw


def selected_mapping(entry, original_relative, archive, parent):
    """Accept an existing exact one-source mapping, never a tolerance guess."""
    source = entry.get("source_ref") or {}
    if not isinstance(source, dict):
        raise Conflict("Existing retained source reference must be an object")
    relative = _relative(source.get("path"))
    if relative.startswith(("MetaData/", "MetaVideo/")):
        relative = archive + "/" + relative
    if (
        relative != original_relative
        or source.get("sha256") != parent["verified_sha256"]
    ):
        raise Conflict("Retained source path/SHA differs from the admitted original")
    mapping, video = entry.get("source_mapping"), entry.get("video")
    if not isinstance(mapping, dict) or not isinstance(video, dict):
        raise Conflict("Existing retained clip and exact source mapping required")
    values = []
    for key in (
        "source_start_ms",
        "source_end_ms",
        "derived_start_ms",
        "derived_end_ms",
    ):
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise Conflict("Retained mapping has unknown boundaries")
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise Conflict("Retained mapping has invalid boundaries") from exc
        if not number.is_finite():
            raise Conflict("Retained mapping has nonfinite boundaries")
        values.append(number / 1000)
    lo, hi, child_lo, child_hi = values
    if (
        not 0 <= lo < hi <= parent["duration_seconds"]
        or child_lo != 0
        or child_hi != hi - lo
    ):
        raise Conflict("Retained mapping is not an exact contiguous source interval")
    if (
        type(video.get("sha256")) is not str
        or not re.fullmatch("[0-9a-f]{64}", video["sha256"])
        or type(video.get("size_bytes")) is not int
    ):
        raise Conflict("Retained clip requires an existing exact SHA/size")
    path = _relative(video.get("path"))
    if not path.startswith("ProcessedClips/") or not path.endswith(".mp4"):
        raise Conflict("Retained video must remain in its existing ProcessedClips path")
    return lo, hi, archive + "/" + path, video


class ProducerIngress:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings

    def _camera(self, code):
        if not isinstance(code, str) or not 1 <= len(code) <= 200:
            raise ValueError("Producer camera identity is unknown")
        with self.core.db.transaction(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s",
                (self.core.customer, code),
            ).fetchone()
        return row or self.core.register(
            "producer-camera:" + fingerprint(code), "capture_device", code
        )

    def _parent(self, capture_id):
        with self.core.db.transaction(readonly=True) as conn:
            return conn.execute(
                """SELECT m.*,a.verified_sha256,a.version FROM hj_timeline_media m
                JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
                WHERE m.customer_id=%s AND m.capture_id=%s AND m.kind='video' """,
                (self.core.customer, capture_id),
            ).fetchone()

    def _clips(self, row, admitted, original, root, guard, doc, previous=None):
        parent = self._parent(admitted["capture_id"])
        if not parent:
            raise Conflict("Admitted original timeline media is unavailable")
        archive = original.split("/")[0]
        index_relative = archive + "/ProcessedClips/Index.json"
        raw = _read(root, index_relative, guard, doc["max_index_bytes"])
        previous = previous or {}
        known = list(previous.get("registered_selections", []))
        if raw is None:
            return dict(
                status="pending_existing_retained_index",
                registered=0,
                registered_selections=known,
            )
        digest = hashlib.sha256(raw).hexdigest()
        same = previous.get("index_sha256") == digest
        if same and previous.get("status") != "partial_bounded_retained_scan":
            return dict(previous, registered=0, unchanged_index=True)
        index = json.loads(raw)
        if not isinstance(index, dict):
            raise Conflict("Existing retained index must be an object")
        segments = index.get("segments")
        if (
            index.get("archive") != archive
            or not isinstance(segments, list)
            or len(segments) > doc["max_index_segments"]
        ):
            raise Conflict(
                "Existing retained index has invalid scope or exceeds bounds"
            )
        choices = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise Conflict("Invalid existing retained segment")
            source = segment.get("source_ref") or {}
            if not isinstance(source, dict) or type(source.get("path", "")) is not str:
                raise Conflict("Invalid existing retained source reference")
            path = source.get("path", "")
            if path.startswith(("MetaData/", "MetaVideo/")):
                path = archive + "/" + path
            if path != original:
                continue
            parts = segment.get("retention_intervals", [segment])
            if not isinstance(parts, list):
                raise Conflict("Invalid retained interval list")
            for part in parts:
                if (
                    not isinstance(part, dict)
                    or len(choices) >= doc["max_index_segments"]
                ):
                    raise Conflict("Retained interval list exceeds bound")
                choices.append(dict(part, source_ref=part.get("source_ref") or source))
        cursor = previous.get("cursor", 0) if same else 0
        pending = list(previous.get("pending", [])) if same else []
        count = attempts = 0
        while cursor < len(choices) and attempts < doc["max_clips_per_tick"]:
            entry = choices[cursor]
            cursor += 1
            if not entry.get("video") or not entry.get("source_mapping"):
                if len(pending) < 64:
                    pending.append("clip_or_exact_mapping_missing")
                continue
            try:
                lo, hi, clip_relative, video = selected_mapping(
                    entry, original, archive, parent
                )
                selection = fingerprint(
                    dict(
                        original=original,
                        source_sha256=parent["verified_sha256"],
                        video=video,
                        mapping=entry["source_mapping"],
                    )
                )
                if selection in known:
                    continue
                attempts += 1
                guard.validate()
                path = safe_path(root, clip_relative)
                actual, size = hash_file(
                    path, min(self.settings.max_capture_bytes, 64 * 1024**2)
                )
                guard.validate()
                if (actual, size) != (video["sha256"], video["size_bytes"]):
                    raise Conflict("Existing retained clip bytes changed")
                receipt = dict(
                    capture_id="retained:" + selection,
                    source_instance=parent["stream_key"],
                    source_revision=selection,
                    camera_id=row["camera_key"],
                    session_id=parent["session_key"],
                    kind="video",
                    relative_path=clip_relative,
                    sha256=actual,
                    captured_at=clean(offset_time(parent["start_at"], lo)),
                    time_basis=parent["time_basis"],
                    start_seconds="0",
                    end_seconds=str(hi - lo),
                    source_kind=self.settings.source_kind,
                    retained_selection=dict(
                        selection_key=selection,
                        existing_mapping=entry["source_mapping"],
                    ),
                )
                clip = Ingest(self.core, self.settings).accept(receipt)
                if clip.get("media_id") is None:
                    with self.core.db.transaction(readonly=True) as conn:
                        clip["media_id"] = conn.execute(
                            "SELECT id FROM media_asset WHERE customer_id=%s AND capture_id=%s",
                            (self.core.customer, clip["capture_id"]),
                        ).fetchone()["id"]
                RetainedClips(Timeline(self.core), self.settings).register(
                    "producer-retained:" + selection,
                    str(clip["media_id"]),
                    [
                        dict(
                            media_id=str(parent["media_id"]),
                            start_seconds=str(lo),
                            end_seconds=str(hi),
                        )
                    ],
                    selection,
                    actor=doc["actor"],
                    reason=doc["reason"],
                )
                guard.validate()
                known.append(selection)
                count += 1
            except (Conflict, ValueError, OSError) as exc:
                if len(pending) < 64:
                    pending.append(type(exc).__name__ + ":" + str(exc)[:160])
        return dict(
            status="partial_bounded_retained_scan"
            if cursor < len(choices)
            else "existing_clips_registered"
            if known
            else "pending_existing_exact_clip",
            registered=count,
            pending=pending,
            index_sha256=digest,
            cursor=cursor,
            registered_selections=known,
        )

    def tick(self):
        doc = load_manifest(self.settings)
        if doc is None:
            return dict(enabled=False)
        if self.settings.processing_profile != "timeline-v1":
            raise Conflict("Producer ingress requires the existing timeline-v1 profile")
        _local(self.settings.runtime_root, directory=True)
        folder = self.settings.runtime_root / "producer-ingress"
        folder.mkdir(exist_ok=True)
        _local(folder, directory=True)
        lock = os.open(folder / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock, "a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            rows, revision = inventory_snapshot(doc)
            identity = dict(
                customer_id=str(self.core.customer),
                inventory_path=doc["inventory_path"],
                source_root=doc["source_root"],
                inventory_relative_prefix=doc.get("inventory_relative_prefix", ""),
            )
            state_path = folder / "State.json"
            if state_path.exists():
                _local(state_path)
                if state_path.stat().st_size > _STATE_LIMIT:
                    raise Conflict("Producer checkpoint exceeds bound")
                state = json.loads(state_path.read_text())
                if state.get("identity") != identity:
                    raise Conflict(
                        "Producer checkpoint belongs to another customer or source"
                    )
            else:
                state = dict(
                    schema_version=SCHEMA,
                    identity=identity,
                    baseline_finalized=sorted(
                        r["recording_id"] for r in rows if r.get("processable") is True
                    ),
                    admitted={},
                    first_inventory_revision=revision,
                )
                _save(state_path, state)
                if doc["mode"] == "new_only":
                    return dict(
                        enabled=True,
                        status="baseline_recorded_no_history_admitted",
                        baseline_finalized=len(state["baseline_finalized"]),
                        admitted=0,
                        inventory_revision=revision,
                    )
            selected = (
                [r for r in rows if r["recording_id"] in doc["recording_ids"]]
                if doc["mode"] == "explicit_ids"
                else [
                    r
                    for r in rows
                    if r["recording_id"] not in state["baseline_finalized"]
                ]
            )
            selected.sort(
                key=lambda r: (r.get("recording_start_us") or 0, r["recording_id"]),
                reverse=True,
            )
            result = dict(
                enabled=True,
                admitted=0,
                duplicates=0,
                clips=0,
                pending=[],
                inventory_revision=revision,
                source_kind=self.settings.source_kind,
                person_identity=None,
                whole_capture_is_retained_clip=False,
            )
            eligible = []
            for row in selected:
                if row.get("processable") is not True:
                    continue
                previous = state["admitted"].get(row["recording_id"])
                if previous and previous["inventory_source_signature"] != row.get(
                    "source_signature"
                ):
                    result["pending"].append(
                        dict(
                            recording_id=row["recording_id"],
                            reason="source_revision_changed_explicit_review_required",
                        )
                    )
                    continue
                if previous is None or time.time() >= previous.get(
                    "clips_next_check", 0
                ):
                    eligible.append(row)
            eligible.sort(
                key=lambda r: (
                    r["recording_id"] not in state["admitted"],
                    r.get("recording_start_us") or 0,
                ),
                reverse=True,
            )
            eligible = eligible[: doc["max_admissions_per_tick"]]
            if not eligible:
                return result
            guard_module = pipeline_module("device_day_write_guard")
            with guard_module.storage_guard(publication_config(self.settings)) as guard:
                root = self.settings.incoming
                for row in eligible:
                    try:
                        relative = _relative(row.get("relative_path"))
                        prefix = doc.get("inventory_relative_prefix", "")
                        if prefix:
                            if not relative.startswith(prefix + "/"):
                                raise Conflict(
                                    "Inventory record is outside the explicitly pinned source prefix"
                                )
                            relative = relative[len(prefix) + 1 :]
                        if not relative.endswith("rgb.mp4") or not any(
                            p in {"MetaData", "MetaVideo"} for p in relative.split("/")
                        ):
                            raise Conflict(
                                "Inventory entry is not an existing native RGB recording"
                            )
                        previous = state["admitted"].get(row["recording_id"])
                        if previous is not None:
                            clips = self._clips(
                                row,
                                previous,
                                relative,
                                root,
                                guard,
                                doc,
                                previous.get("clips"),
                            )
                            previous.update(
                                clips=clips,
                                clips_next_check=time.time()
                                + (
                                    0
                                    if clips["status"]
                                    == "partial_bounded_retained_scan"
                                    else 30
                                ),
                            )
                            _save(state_path, state)
                            result["duplicates"] += 1
                            result["clips"] += clips["registered"]
                            continue
                        guard.validate()
                        video = safe_path(root, relative)
                        fresh = inspect_recorder(root, video, time.time(), 0.5)
                        if fresh.get("processable") is not True or any(
                            str(fresh.get(k)) != str(row.get(k))
                            for k in (
                                "camera_key",
                                "recording_session_id",
                                "recording_start_us",
                                "recording_end_us",
                            )
                        ):
                            raise Conflict(
                                "Finalized recorder identity/clock no longer matches inventory"
                            )
                        self._camera(fresh["camera_key"])
                        ready = video.with_name(
                            video.name[: -len("rgb.mp4")] + "recording_ready.json"
                        )
                        admitted = Ingest(self.core, self.settings).native_receipt(
                            ready
                        )[0]
                        guard.validate()
                        duplicate = row["recording_id"] in state["admitted"]
                        clips = self._clips(fresh, admitted, relative, root, guard, doc)
                        state["admitted"][row["recording_id"]] = dict(
                            inventory_source_signature=row.get("source_signature"),
                            native_source_signature=fresh["source_signature"],
                            capture_id=admitted["capture_id"],
                            original_inventory_recording_id=row["recording_id"],
                            camera_key=fresh["camera_key"],
                            session_id=fresh["recording_session_id"],
                            clips=clips,
                            clips_next_check=time.time()
                            + (
                                0
                                if clips["status"] == "partial_bounded_retained_scan"
                                else 30
                            ),
                            manifest_sha256=doc["manifest_sha256"],
                            processed_at=datetime.now(timezone.utc).isoformat(),
                        )
                        _save(state_path, state)
                        result["duplicates" if duplicate else "admitted"] += 1
                        result["clips"] += clips["registered"]
                        if clips["status"].startswith("pending"):
                            result["pending"].append(
                                dict(
                                    recording_id=row["recording_id"],
                                    reason=clips["status"],
                                )
                            )
                    except (Conflict, ValueError, OSError, KeyError) as exc:
                        result["pending"].append(
                            dict(
                                recording_id=row["recording_id"],
                                reason=type(exc).__name__ + ":" + str(exc)[:200],
                            )
                        )
                guard.validate()
            return clean(result)
