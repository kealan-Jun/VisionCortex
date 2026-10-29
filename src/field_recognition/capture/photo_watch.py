"""Read-only watch of the configured camera's photographs, with optional binding."""

import copy
import json
import os
import re
import threading
import time
from datetime import datetime

from fastapi import HTTPException

POLL_SECONDS = 0.5
STABLE_SECONDS = 0.5


class SavedPhotoWatcher:
    def __init__(self, core, *, clock=time.monotonic):
        self.core, self.clock = core, clock
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker = None
        self.workers = []
        self.fresh_count = 0
        self.pending = {}
        self.binding_id = None
        self.state = {"status": "disabled", "binding_id": None, "last_job_id": None}
        with core["db"]() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS photo_watch_seen(binding_id TEXT, path TEXT, signature TEXT, "
                "status TEXT, job_id TEXT, detail TEXT, observed_at TEXT, PRIMARY KEY(binding_id,path))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS photo_watch_files(camera TEXT, path TEXT, signature TEXT, "
                "status TEXT, job_id TEXT, detail TEXT, observed_at TEXT, PRIMARY KEY(camera,path))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS photo_watch_windows(camera TEXT PRIMARY KEY, started_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS photo_ingest_queue(camera TEXT,path TEXT,signature TEXT,document TEXT NOT NULL,PRIMARY KEY(camera,path))"
            )
            for row in conn.execute(
                "SELECT path,signature,document FROM photo_ingest_queue WHERE camera=?",
                (self.target(),),
            ):
                item = json.loads(row["document"])
                self.pending[row["path"]] = (row["signature"], self.clock())

    def enabled(self):
        config = self.core.get("config")
        if config is not None:
            return config.saved_photo_watch_enabled
        return os.environ.get("FIELD_SAVED_PHOTO_WATCH_ENABLED", "0").lower() in {
            "1",
            "true",
            "yes",
        }

    def target(self):
        receiver = self.core["receiver_camera"]
        config = self.core.get("config")
        if config is not None:
            return config.camera_id or (receiver.target if receiver else None)
        return receiver.target if receiver else os.environ.get("FIELD_CAMERA_ID")

    def configured_root(self):
        roots = self.configured_roots()
        return roots[0] if roots else None

    def configured_roots(self):
        from field_recognition.runtime.production_config import ProductionConfig

        config = self.core.get("config")
        return (
            config if config is not None else ProductionConfig.from_environment()
        ).saved_photo_roots

    def timezone(self):
        config = self.core.get("config")
        return (
            config.saved_photo_timezone
            if config is not None
            else os.environ.get("FIELD_SAVED_PHOTO_TIMEZONE", "Asia/Shanghai")
        )

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
        with self.core["db"]() as conn:
            items = [
                json.loads(r[0])
                for r in conn.execute(
                    "SELECT document FROM photo_ingest_queue WHERE camera=?", (self.target(),)
                )
            ]
            queued = len(items)
        oldest = min((i["first_observed_at"] for i in items), default=None)
        return result | {
            "enabled": self.enabled(),
            "configured": bool(self.configured_root()),
            "source_roots": list(self.configured_roots()),
            "persisted_pending_files": queued,
            "pending_files": queued,
            "retrying_files": sum(i.get("attempts", 0) > 0 for i in items),
            "ready_files": sum(
                bool(i.get("stable_at")) and i.get("retry_after", 0) <= time.time() for i in items
            ),
            "oldest_pending_at": oldest,
            "oldest_pending_seconds": max(
                0, time.time() - datetime.fromisoformat(oldest).timestamp()
            )
            if oldest
            else 0,
            "scheduling": "three_fresh_then_one_backlog",
            "poll_seconds": POLL_SECONDS,
            "stable_seconds": STABLE_SECONDS,
            "reads_existing_photos_only": True,
        }

    def start(self):
        if not self.enabled():
            return
        for name, action, interval in [
            ("discover", lambda: self.discover(recent=True), POLL_SECONDS),
            ("backfill", lambda: self.discover(recent=False), 30),
            ("import", self.ingest, POLL_SECONDS),
        ]:
            thread = threading.Thread(
                target=self._loop, args=(action, interval), name="voice-photo-" + name, daemon=True
            )
            self.workers.append(thread)
            thread.start()

    def close(self):
        self.stop.set()
        for thread in self.workers:
            thread.join(timeout=0.5)

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)

    def _loop(self, action, interval):
        while not self.stop.is_set():
            try:
                action()
            except (OSError, ValueError, KeyError):
                self.update(status="storage_unavailable")
            except Exception:
                self.update(status="watch_error")
            self.stop.wait(interval)

    def active_bindings(self):
        target = self.target()
        with self.core["db"]() as conn:
            rows = conn.execute(
                "SELECT document FROM bindings WHERE camera=? AND ended IS NULL", (target,)
            ).fetchall()
        bindings = []
        for row in rows:
            binding = json.loads(row["document"])
            if self.core["current_readout_binding"](binding):
                bindings.append(binding)
        return bindings

    def step(self):
        """Synchronous bounded cycle for callers/tests; service loops are independent."""
        self.discover(recent=False)
        self.ingest()

    def discover(self, *, recent=True):
        from field_recognition.runtime.source_io import call, SourceError

        if not self.enabled() or self.stop.is_set():
            return
        configured = self.configured_root()
        camera = self.target()
        if not configured:
            self.update(status="storage_unconfigured")
            return
        if not camera or not re.fullmatch(r"[a-zA-Z0-9_-]+", camera):
            self.update(status="invalid_camera_directory")
            return
        with self.core["db"]() as conn:
            settings = conn.execute(
                "SELECT document FROM automation_settings WHERE camera=?", (camera,)
            ).fetchone()
            registered = json.loads(settings["document"]).get("registered_at") if settings else None
            conn.execute(
                "INSERT OR IGNORE INTO photo_watch_windows VALUES(?,?)",
                (camera, registered or self.core["now"]()),
            )
            window = conn.execute(
                "SELECT started_at FROM photo_watch_windows WHERE camera=?", (camera,)
            ).fetchone()[0]
        try:
            found = call(
                {
                    "action": "discover",
                    "root": configured,
                    "roots": self.configured_roots(),
                    "camera": camera,
                    "since": window,
                    "zone": self.timezone(),
                    "recent": recent,
                }
            )
        except SourceError as exc:
            self.update(
                **(
                    {"status": "storage_unavailable", "discovery_error": str(exc)}
                    if recent
                    else {"backfill_error": str(exc)}
                )
            )
            return
        with self.lock, self.core["db"]() as conn:
            seen = {
                r["path"]: r["signature"]
                for r in conn.execute(
                    "SELECT path,signature FROM photo_watch_files WHERE camera=?", (camera,)
                )
            }
            prefix = camera + "/"
            seen.update(
                {
                    r["path"]: r["signature"]
                    for r in conn.execute(
                        "SELECT path,signature FROM photo_watch_seen WHERE substr(path,1,?)=?",
                        (len(prefix), prefix),
                    )
                }
            )
            for relative, signature, written in found["files"]:
                if seen.get(relative) == signature:
                    continue
                row = conn.execute(
                    "SELECT signature,document FROM photo_ingest_queue WHERE camera=? AND path=?",
                    (camera, relative),
                ).fetchone()
                prior = self.pending.get(relative)
                if not prior or prior[0] != signature:
                    self.pending[relative] = (signature, self.clock())
                item = (
                    json.loads(row["document"])
                    if row and row["signature"] == signature
                    else {
                        "attempts": 0,
                        "retry_after": 0,
                        "first_observed_at": self.core["now"](),
                        "is_backfill": time.time() - written > 30,
                        "stable_at": None,
                    }
                )
                item["written_at_epoch"] = written
                if (
                    not item.get("stable_at")
                    and self.clock() - self.pending[relative][1] >= STABLE_SECONDS
                ):
                    item["stable_at"] = self.core["now"]()
                raw = json.dumps(item)
                if not row or row["signature"] != signature or row["document"] != raw:
                    conn.execute(
                        "INSERT OR REPLACE INTO photo_ingest_queue VALUES(?,?,?,?)",
                        (camera, relative, signature, raw),
                    )
            count = conn.execute(
                "SELECT count(*) FROM photo_watch_files WHERE camera=? AND status='submitted'",
                (camera,),
            ).fetchone()[0]
        self.update(
            status="waiting_photo" if found.get("missing_camera") else "watching",
            camera_id=camera,
            watch_started_at=window,
            last_checked_at=self.core["now"](),
            submitted_files=count,
            discovery_error=None,
            source_root_status=found.get("source_roots", []),
            **({"backfill_error": None} if not recent else {}),
        )

    def ingest(self):
        if not self.enabled() or self.stop.is_set():
            return
        camera = self.target()
        with self.core["db"]() as conn:
            if (
                conn.execute(
                    "SELECT count(*) FROM jobs WHERE status IN ('queued','running') AND coalesce(json_extract(document,'$.request_trigger'),'')!='video_stream'"
                ).fetchone()[0]
                >= 4
            ):
                self.update(status="waiting_queue")
                return
            rows = [
                dict(r) | {"item": json.loads(r["document"])}
                for r in conn.execute("SELECT * FROM photo_ingest_queue WHERE camera=?", (camera,))
            ]
        eligible = [
            r
            for r in rows
            if r["item"].get("stable_at") and r["item"].get("retry_after", 0) <= time.time()
        ]
        if not eligible:
            return
        fresh = [
            r
            for r in eligible
            if not r["item"]["is_backfill"] and r["item"].get("attempts", 0) == 0
        ]
        older = [r for r in eligible if r not in fresh]
        # Three fresh files followed by one backlog/retry slot prevents starvation.
        if fresh and (self.fresh_count < 3 or not older):
            row = max(fresh, key=lambda r: (r["item"].get("written_at_epoch", 0), r["path"]))
            self.fresh_count += 1
        else:
            row = min(older or fresh, key=lambda r: r["item"]["first_observed_at"])
            self.fresh_count = 0
        relative, signature, item = row["path"], row["signature"], row["item"]
        self.update(importing_file=relative, import_started_at=self.core["now"]())
        try:
            body = self.core["SavedPhotoRequest"](image_path=relative)
            job = self.core["read_saved_panel"](
                body,
                trigger="voice_photo_directory",
                observation={
                    "first_observed_at": item["first_observed_at"],
                    "stable_at": item["stable_at"],
                    "is_backfill": item["is_backfill"],
                    "watch_poll_seconds": POLL_SECONDS,
                    "file_stable_seconds": STABLE_SECONDS,
                    "ingest_retry_count": item.get("attempts", 0),
                    "ingest_last_error": item.get("last_error"),
                    "ingest_first_failed_at": item.get("first_failed_at"),
                    "ingest_last_failed_at": item.get("last_failed_at"),
                },
            )
        except (HTTPException, OSError) as exc:
            code = exc.status_code if isinstance(exc, HTTPException) else 503
            detail = str(exc.detail) if isinstance(exc, HTTPException) else type(exc).__name__
            attempts = item.get("attempts", 0) + 1
            delay = 1 if code == 429 else min(300, 3 * 2 ** min(attempts - 1, 7))
            item.update(
                attempts=attempts,
                retry_after=time.time() + delay,
                last_error=detail,
                last_failed_at=self.core["now"](),
                error_code=code,
            )
            item.setdefault("first_failed_at", self.core["now"]())
            with self.lock, self.core["db"]() as conn:
                # A newer discovery signature must not be overwritten by a late failure.
                conn.execute(
                    "UPDATE photo_ingest_queue SET document=? WHERE camera=? AND path=? AND signature=?",
                    (json.dumps(item), camera, relative, signature),
                )
            self.update(last_import_error=detail, importing_file=None)
            return
        with self.lock, self.core["db"]() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO photo_watch_files VALUES(?,?,?,?,?,?,?)",
                (camera, relative, signature, "submitted", job["job_id"], None, self.core["now"]()),
            )
            conn.execute(
                "DELETE FROM photo_ingest_queue WHERE camera=? AND path=? AND signature=?",
                (camera, relative, signature),
            )
            self.pending.pop(relative, None)
            count = conn.execute(
                "SELECT count(*) FROM photo_watch_files WHERE camera=? AND status='submitted'",
                (camera,),
            ).fetchone()[0]
        self.update(
            last_photo=relative,
            last_job_id=job["job_id"],
            last_photo_status="submitted",
            detail=None,
            importing_file=None,
            last_import_error=None,
            last_submitted_at=self.core["now"](),
            submitted_files=count,
        )
