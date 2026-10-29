"""Opt-in manual LiveScanner -> existing CaptureReceipt discovery queue.

This is a source publisher, not an identity resolver. Raw QR text is untrusted
diagnostic evidence; the ordinary QR worker must decode the saved pixels again.
No configuration means no filesystem, network, model, or background activity.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import UUID, uuid4

from .components import pipeline_module
from .domain import Conflict, fingerprint
from .media import safe_path
from .retention import encoded
from .storage import is_nas, storage_binding, verify_root

SCHEMA = "vc-lab-scan-source/1"
ENV = "VC_LAB_SCAN_SOURCE_MANIFEST"
MAX_STATE_BYTES = 2 * 1024**2


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate scan source JSON key")
        result[key] = value
    return result


def _json(raw):
    return json.loads(
        raw,
        object_pairs_hook=_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON")
        ),
    )


def _local(path):
    p = Path(path)
    if (
        not p.is_absolute()
        or str(p) != str(path)
        or ".." in p.parts
        or is_nas(p)
        or p.resolve() != p
    ):
        raise ValueError(
            "Scan source private configuration/state requires canonical local paths"
        )
    return p


def load_publisher_from_environment():
    name = os.environ.get(ENV)
    if not name:
        return None
    path = _local(name)
    if not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("Bounded private scan source manifest required")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Scan source manifest grew beyond its bound")
    document = _json(raw)
    if (
        type(document) is not dict
        or document.get("schema_version") != SCHEMA
        or type(document.get("enabled")) is not bool
    ):
        raise ValueError("Explicit versioned scan source manifest required")
    if not document["enabled"]:
        return None
    return ScanSourcePublisher(document)


class ScanSourcePublisher:
    def __init__(self, document):
        required = {
            "schema_version",
            "enabled",
            "runtime_root",
            "incoming_root",
            "max_bytes",
            "max_frame_bytes",
            "max_events",
            "retention_seconds",
            "source_kind",
            "cameras",
        }
        if (
            type(document) is not dict
            or set(document) != required
            or document.get("schema_version") != SCHEMA
            or document.get("enabled") is not True
        ):
            raise ValueError("Complete opt-in scan source configuration required")
        for name, low, high in [
            ("max_bytes", 1024, 64 * 1024**2),
            ("max_frame_bytes", 1024, 16 * 1024**2),
            ("max_events", 1, 600),
            ("retention_seconds", 1, 86400),
        ]:
            if type(document[name]) is not int or not low <= document[name] <= high:
                raise ValueError("Explicit bounded scan source capacity required")
        if document["max_frame_bytes"] > document["max_bytes"] or document[
            "source_kind"
        ] not in {"fixture", "customer_capture", "derived_replay"}:
            raise ValueError("Invalid scan source byte bound or source category")
        self.runtime = _local(document["runtime_root"])
        root = Path(document["incoming_root"])
        if (
            not root.is_absolute()
            or str(root) != document["incoming_root"]
            or ".." in root.parts
        ):
            raise ValueError("Exact incoming root required")
        self.root = root
        if not is_nas(root) and not root.is_relative_to(self.runtime):
            raise ValueError(
                "Local scan drops must stay within their bounded runtime root"
            )
        cameras = document["cameras"]
        if type(cameras) is not list or not 1 <= len(cameras) <= 7:
            raise ValueError("Explicit camera allowlist required")
        self.cameras = {}
        for camera in cameras:
            if (
                type(camera) is not dict
                or set(camera) != {"camera_id", "source_instance", "scan_kind"}
                or camera.get("scan_kind") not in {"equipment", "material"}
            ):
                raise ValueError(
                    "Each source requires its own explicit camera and scan kind"
                )
            for name in ("camera_id", "source_instance"):
                if (
                    type(camera[name]) is not str
                    or not camera[name]
                    or len(camera[name].encode()) > 120
                ):
                    raise ValueError("Bounded camera/source identity required")
            if camera["camera_id"] in self.cameras:
                raise ValueError("Duplicate scan source camera")
            self.cameras[camera["camera_id"]] = dict(camera)
        self.config = dict(document)
        self.config_sha256 = fingerprint(document)
        self.lock = threading.RLock()

    @contextmanager
    def _state(self):
        # No state is created until a permitted manual session supplies a frame.
        runtime = _local(str(self.runtime))
        runtime.mkdir(parents=True, exist_ok=True)
        folder = runtime / "scan-source-state"
        if folder.is_symlink():
            raise Conflict("Scan source state path cannot be a link")
        folder.mkdir(exist_ok=True)
        with os.fdopen(
            os.open(
                folder / "publisher.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            ),
            "r+",
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = folder / "publisher.json"
            if path.exists():
                if path.is_symlink() or path.stat().st_size > MAX_STATE_BYTES:
                    raise Conflict("Scan source state exceeds its bound or is unsafe")
                state = _json(path.read_bytes())
                if state.get("config_sha256") != self.config_sha256:
                    raise Conflict(
                        "Scan source configuration changed; reconcile the old drop queue explicitly"
                    )
            else:
                state = dict(
                    config_sha256=self.config_sha256,
                    created_at=time.time(),
                    sessions={},
                    captures={},
                    bytes=0,
                )
            if time.time() - state["created_at"] > self.config["retention_seconds"]:
                raise Conflict(
                    "Scan source retention deadline reached; preserve and reconcile drops before continuing"
                )
            yield state
            self._save_state(state)

    def _save_state(self, state):
        raw = encoded(state)
        if len(raw) > MAX_STATE_BYTES:
            raise Conflict("Scan source state capacity reached")
        self._local_write(self.runtime / "scan-source-state/publisher.json", raw)

    @staticmethod
    def _local_write(path, raw):
        temporary = path.with_name("." + path.name + "." + uuid4().hex + ".partial")
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self, relative, limit, guard):
        path = safe_path(self.root, relative, must_exist=False)
        guard.validate()
        mount, actual = guard.mount_for(path)
        parent = (
            guard.directory(mount, actual.relative_to(mount.path).parts[:-1])
            if mount
            else None
        )
        try:
            try:
                fd = os.open(
                    actual.name if parent is not None else actual,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent,
                )
            except FileNotFoundError:
                return None
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                raw = stream.read(limit + 1)
                after = os.fstat(stream.fileno())
            if len(raw) > limit or (
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise Conflict("Scan source artifact grew or changed while reading")
            guard.validate()
            return raw
        finally:
            if parent is not None:
                os.close(parent)

    def _immutable(self, relative, raw, guard):
        previous = self._read(relative, len(raw), guard)
        if previous is not None:
            if previous != raw:
                raise Conflict(
                    "Physical scan source already has different immutable bytes"
                )
            return
        path = safe_path(self.root, relative, must_exist=False)
        if not guard.atomic(path, [raw]):
            if is_nas(path):
                raise OSError("NAS scan drop has no pinned mount; no fallback allowed")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._local_write(path, raw)
        if self._read(relative, len(raw), guard) != raw:
            raise Conflict("Scan source atomic read-back failed")

    def publish(self, frame_bgr, metadata, session, decoded):
        # The callback is only reached from the explicitly started LiveScanner.
        # A wearer label never becomes a PostgreSQL person identity.
        if session.get("owner") in {None, "", "automation"}:
            return dict(status="not_manual_session", published=False)
        camera = session.get("camera_id")
        if camera not in self.cameras:
            return dict(status="camera_not_allowed", published=False)
        session_id = str(UUID(session["session_id"]))
        if (
            metadata.get("source_camera_key") != camera
            or type(metadata.get("decoded_frame_id")) is not int
            or metadata["decoded_frame_id"] < 1
        ):
            raise ValueError(
                "Exact source camera and stable native frame identity required"
            )
        if (
            type(session.get("operator")) is not str
            or not session["operator"].strip()
            or len(session["operator"].encode()) > 320
            or type(session.get("owner")) is not str
            or len(session["owner"].encode()) > 120
            or session.get("status") not in {"scanning", "waiting_camera"}
        ):
            raise ValueError("Explicit active operator session required")
        import cv2
        import numpy as np

        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.dtype != np.uint8
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.nbytes > self.config["max_frame_bytes"]
        ):
            raise ValueError("Bounded original BGR frame required")
        values = decoded[0]
        if (
            type(values) is not list
            or len(values) > 64
            or any(
                type(v) is not str or not v or len(v.encode()) > 4096 for v in values
            )
        ):
            raise ValueError("Bounded raw decoded QR strings required")
        values = sorted(set(values))
        stable = {
            k: metadata.get(k)
            for k in (
                "sender_id",
                "camera_id",
                "decoded_frame_id",
                "sequence",
                "timestamp_us",
                "global_timestamp_us",
                "stream",
                "device_boot_id",
                "media_session_id",
            )
        }
        if any(
            v is not None
            and (type(v) not in {str, int, bool} or len(str(v).encode()) > 160)
            for v in stable.values()
        ):
            raise ValueError("Bounded original frame metadata required")
        epoch = {
            k: stable[k]
            for k in ("device_boot_id", "media_session_id")
            if stable[k] is not None
        }
        physical = fingerprint(dict(session_id=session_id, camera=camera, frame=stable))
        session_key = fingerprint(dict(session_id=session_id, camera=camera))
        rgb_sha = hashlib.sha256(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).tobytes()
        ).hexdigest()
        success, png = cv2.imencode(".png", frame_bgr)
        if not success or png.nbytes > self.config["max_frame_bytes"]:
            raise ValueError("Lossless scan evidence exceeded its byte bound")
        raw = png.tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        time_basis = dict(
            basis="receiver_frame_clock_unknown",
            clock_sync_valid=False,
            original_frame_clock={
                k: metadata.get(k)
                for k in (
                    "timestamp_us",
                    "global_timestamp_us",
                    "effective_timestamp_us",
                    "clock_sync_valid",
                    "clock_uncertainty_us",
                )
            },
        )
        captured_at = None
        global_us = metadata.get("global_timestamp_us")
        if (
            metadata.get("clock_sync_valid") is True
            and type(global_us) is int
            and global_us > 0
            and type(metadata.get("clock_uncertainty_us")) is int
            and metadata["clock_uncertainty_us"] == 0
        ):
            captured_at = (
                datetime(1970, 1, 1, tzinfo=timezone.utc)
                + timedelta(microseconds=global_us)
            ).isoformat()
            time_basis.update(
                basis="synchronized_source_frame",
                clock_sync_valid=True,
                uncertainty_us=0,
            )
        source = self.cameras[camera]
        receipt = dict(
            capture_id=physical,
            source_instance=source["source_instance"],
            source_revision="1",
            camera_id=camera,
            session_id=session_id,
            kind="photo",
            captured_at=captured_at,
            time_basis=time_basis,
            source_kind=self.config["source_kind"],
            sha256=digest,
            scan_intent=dict(
                intent="operator_scan",
                scan_kind=source["scan_kind"],
                actor="manual LiveScanner session",
                reason="explicit non-automation scan session",
            ),
            scan_source=dict(
                schema_version=SCHEMA,
                config_sha256=self.config_sha256,
                physical_source_key=physical,
                frame_metadata=stable,
                source_epoch=epoch or None,
                source_epoch_status="provided" if epoch else "unknown",
                original_rgb_sha256=rgb_sha,
                png_sha256=digest,
                decoded_values=values,
                operator_label_unverified=session["operator"],
                session_owner=session["owner"],
                person_id=None,
                wearer_is_not_person=True,
                decoder_result_is_authoritative=False,
            ),
        )
        with self.lock, self._state() as state:
            old = state["captures"].get(physical)
            was_committed = bool(old and old.get("committed"))
            received_at = (
                old["received_at"] if old else datetime.now(timezone.utc).isoformat()
            )
            from .archive_layout import day_directory

            directory = day_directory(camera, captured_at or received_at)
            relative = f"{directory}/MetaData/VideoEvidence/{physical}/Frame.png"
            receipt["relative_path"] = relative
            receipt["scan_source"].update(
                received_at=received_at,
                archive_day_basis="source_capture_clock"
                if captured_at
                else "receipt_day_not_event_time",
            )
            body = encoded(receipt)
            if (
                old is not None
                and old["receipt_sha256"] != hashlib.sha256(body).hexdigest()
            ):
                raise Conflict(
                    "Same physical scan frame has changed pixels or source metadata"
                )
            previous = state["sessions"].get(session_key)
            if (
                old is None
                and previous
                and previous["values"] == values
                and previous["epoch"] == epoch
            ):
                return dict(
                    status="same_code_in_session",
                    published=False,
                    capture_id=previous["capture_id"],
                )
            total = len(raw) + len(body)
            if old is None and (
                len(state["captures"]) >= self.config["max_events"]
                or len(state["sessions"]) >= 600
                or state["bytes"] + total > self.config["max_bytes"]
            ):
                raise Conflict("Bounded scan source drop capacity reached")
            if (
                shutil.disk_usage(self.runtime).free < 80 * 1024**3
                or shutil.disk_usage("/").free < 50 * 1024**3
            ):
                raise Conflict("Scan source local free-space floor reached")
            if old is None:
                # Reserve bytes durably before writing any frame. An interrupted
                # attempt keeps its reservation, preventing orphaned drops from
                # silently escaping the configured capacity.
                state["captures"][physical] = dict(
                    receipt_sha256=hashlib.sha256(body).hexdigest(),
                    bytes=total,
                    committed=False,
                    received_at=received_at,
                )
                state["bytes"] += total
                self._save_state(state)
            # Validate a NAS root and its current marker before *any* path access;
            # reads and writes below use the existing pinned mount dirfd guard.
            root = verify_root(self.root)
            if not root.is_dir():
                raise OSError("Explicit existing incoming root required")
            guard_module = pipeline_module("device_day_write_guard")
            config = storage_binding(root)
            config["storage"] = dict(
                archive_root=str(root), local_runtime_root=str(self.runtime)
            )
            with guard_module.storage_guard(config) as guard:
                if is_nas(root):
                    from .archive_layout import verify_archive_contract

                    identity_file = verify_archive_contract(root)
                    guard.validate()
                    # Also read the validated contract through the pinned mount,
                    # so a detached local namesake cannot stand in for it.
                    contract = self._read(
                        identity_file.relative_to(root).as_posix(), 2 * 1024**2, guard
                    )
                    if contract is None:
                        raise OSError("Pinned archive contract unavailable")
                # The receipt is the commit marker. Interrupted image-only drops
                # are ignored by discovery and can be resumed byte-identically.
                self._immutable(relative, raw, guard)
                self._immutable(
                    str(Path(relative).with_name("CaptureReceipt.json")), body, guard
                )
            if old is None or not old.get("committed"):
                state["captures"][physical]["committed"] = True
                state["sessions"][session_key] = dict(
                    values=values, epoch=epoch, capture_id=physical
                )
            return dict(
                status="duplicate"
                if was_committed
                else "resumed"
                if old
                else "published",
                published=True,
                capture_id=physical,
                relative_path=relative,
                source_sha256=digest,
                event_time_verified=captured_at is not None,
            )
