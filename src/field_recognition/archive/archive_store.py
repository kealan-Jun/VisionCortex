"""Transactional receipt outbox and immutable, content-addressed NAS archival."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from field_recognition.measurements.readout_timing import elapsed_ms
from field_recognition.archive.archive_progress import advance
from field_recognition.runtime.paths import source_path


TABLES = {
    "scans": "id",
    "bindings": "id",
    "scene_visits": "id",
    "jobs": "id",
    "automation_settings": "camera",
    "photo_measurements": "id",
    "experiment_records": "id",
}
TABLES["binding_handoffs"] = "id"
EVENT_ENTITIES = (
    "scans",
    "bindings",
    "scene_visits",
    "binding_handoffs",
    "jobs",
    "photo_measurements",
    "experiment_records",
)


def publication_timing(row):
    """Per-version measurements; old receipts cannot supply missing stages."""
    keys = row.keys()
    started = row["publication_started_at"] if "publication_started_at" in keys else None
    duration = row["receipt_publication_ms"] if "receipt_publication_ms" in keys else None
    wait = elapsed_ms(row["recorded_at"], started)
    return {
        "publication_started_at": started,
        "archive_queue_wait_ms": wait,
        "receipt_publication_ms": duration,
        "duration_clock_basis": {
            **(
                {"archive_queue_wait_ms": "wall_clock_timestamp_difference"}
                if wait is not None
                else {}
            ),
            **({"receipt_publication_ms": "process_monotonic"} if duration is not None else {}),
        },
        "receipt_publication_scope": "source_artifact_validation_and_immutable_receipt_publication",
    }


def receipt_status(conn, entity, entity_id, source_written_at=None):
    row = conn.execute(
        "SELECT * FROM archive_outbox WHERE entity=? AND entity_id=? ORDER BY seq DESC LIMIT 1",
        (entity, entity_id),
    ).fetchone()
    if not row:
        return {"status": "not_queued"}
    keys = row.keys()
    published_at = row["published_at"] if "published_at" in keys else None
    readable_at = row["readable_at"] if "readable_at" in keys else None
    readability_basis = row["readability_basis"] if "readability_basis" in keys else None
    archived_at = row["archived_at"]
    # Keep the legacy archived_at contract while exposing the explicit
    # publication/readability milestones. Older rows intentionally stay null.
    queue_end = published_at or archived_at
    return {
        "status": "archived" if archived_at else "pending",
        "sequence": row["seq"],
        "queued_at": row["recorded_at"],
        "archived_at": archived_at,
        "published_at": published_at,
        "readable_at": readable_at,
        "archive_published_at": published_at,
        "archive_readable_at": readable_at,
        "readability_basis": readability_basis,
        "receipt_path": row["receipt_path"],
        "archive_queue_ms": elapsed_ms(row["recorded_at"], queue_end),
        "write_to_archive_ms": elapsed_ms(source_written_at, queue_end),
        "archive_publication_ms": elapsed_ms(row["recorded_at"], published_at),
        "archive_readability_ms": elapsed_ms(published_at, readable_at),
    } | publication_timing(row)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def replace_view(path, data):
    """Atomically refresh a derived view; primary receipt/object files use immutable_write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    advance()


def immutable_write(path, data):
    """Publish a complete file; an existing different receipt is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Archive content conflict")
        advance()
        return
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        # There is one writer per database. An overlapping retry may only publish
        # the identical content-addressed file; never a different revision.
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Archive content conflict")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    advance()


class ArchiveStore:
    def __init__(self, core):
        self.core = core
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.worker = None
        # Three current-event selections share the writer with one oldest due
        # historical version. Retain the quota across batch boundaries.
        self.priority_selections = 0
        self.browse_digests = {}
        self.status = {"status": "disabled", "last_error": None, "last_archived_at": None}
        self.index_version = (
            "4:"
            + hashlib.sha256(
                b"".join(
                    source_path(p).read_bytes()
                    for p in (
                        "archive_catalog.py",
                        "archive_events.py",
                        "evidence_timeline.py",
                        "archive_paths.py",
                        "archive_migration.py",
                        "field_recognition/archive/source_layout.py",
                        "field_recognition/archive/archive_recovery.py",
                        "measurement_records.py",
                        "field_recognition/measurements/readout_rules.py",
                        "static/archive.html",
                        "static/archive.js",
                        "static/view-utils.js",
                    )
                )
                + canonical(
                    {
                        "layout": core["config"].archive_layout if core.get("config") else "events",
                        "source_roots": core["config"].saved_photo_roots
                        if core.get("config")
                        else (),
                    }
                )
            ).hexdigest()[:16]
        )
        with core["db"]() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS archive_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS archive_outbox(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT NOT NULL, entity_id TEXT NOT NULL,
                    document TEXT NOT NULL, recorded_at TEXT NOT NULL, archived_at TEXT, receipt_path TEXT,
                    published_at TEXT, readable_at TEXT, readability_basis TEXT,
                    publication_started_at TEXT, receipt_publication_ms REAL,
                    retry_after REAL NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT);
                CREATE INDEX IF NOT EXISTS archive_entity_version ON archive_outbox(entity,entity_id,seq);
            """)
            # Several managed roles initialize this ledger together. Serialize
            # column inspection and extension so they cannot add the same column.
            conn.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(archive_outbox)")}
            for name, declaration in [
                ("retry_after", "REAL NOT NULL DEFAULT 0"),
                ("attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("last_error", "TEXT"),
            ]:
                if name not in columns:
                    conn.execute(f"ALTER TABLE archive_outbox ADD COLUMN {name} {declaration}")
            for name in ("published_at", "readable_at", "readability_basis"):
                if name not in columns:
                    conn.execute(f"ALTER TABLE archive_outbox ADD COLUMN {name} TEXT")
            for name, declaration in [
                ("publication_started_at", "TEXT"),
                ("receipt_publication_ms", "REAL"),
            ]:
                if name not in columns:
                    conn.execute(f"ALTER TABLE archive_outbox ADD COLUMN {name} {declaration}")
            # Legacy ledgers acquire retry_after above before its index exists.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS archive_pending_due ON archive_outbox(retry_after,seq) "
                "WHERE archived_at IS NULL"
            )
            conn.execute(
                "INSERT OR IGNORE INTO archive_meta VALUES(?,?)",
                ("source_instance", str(uuid.uuid4())),
            )
            self.instance = conn.execute(
                "SELECT value FROM archive_meta WHERE key='source_instance'"
            ).fetchone()[0]
            for table, key in TABLES.items():
                if not conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone():
                    continue
                document = (
                    "json_set(NEW.document,'$.status',NEW.status)"
                    if table == "jobs"
                    else "NEW.document"
                )
                changed = "NEW.document != OLD.document" + (
                    " OR NEW.status != OLD.status" if table == "jobs" else ""
                )
                for action, suffix in [("INSERT", ""), ("UPDATE", " WHEN " + changed)]:
                    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS archive_{table}_{action.lower()}
                        AFTER {action} ON {table}{suffix} BEGIN
                        INSERT INTO archive_outbox(entity,entity_id,document,recorded_at)
                        VALUES('{table}',NEW.{key},{document},strftime('%Y-%m-%dT%H:%M:%fZ','now')); END""")
            if not conn.execute("SELECT 1 FROM archive_meta WHERE key='backfilled'").fetchone():
                for table, key in TABLES.items():
                    if not conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                    ).fetchone():
                        continue
                    document = (
                        "json_set(document,'$.status',status)" if table == "jobs" else "document"
                    )
                    conn.execute(
                        f"""INSERT INTO archive_outbox(entity,entity_id,document,recorded_at)
                        SELECT ?,{key},{document},? FROM {table}""",
                        (table, core["now"]()),
                    )
                conn.execute("INSERT INTO archive_meta VALUES('backfilled','1')")

        from field_recognition.archive.archive_integrity import ArchiveIntegrity

        self.integrity = ArchiveIntegrity(self)

    def enabled(self):
        config = self.core.get("config")
        if config is not None:
            return config.archive_enabled
        return os.environ.get("FIELD_ARCHIVE_ENABLED", "0").lower() in {"1", "true", "yes"}

    def service_role(self):
        config = self.core.get("config")
        return (
            config.service_role
            if config is not None
            else self.core.get("SERVICE_ROLE") or os.environ.get("FIELD_SERVICE_ROLE", "combined")
        )

    def configured_root(self):
        config = self.core.get("config")
        return config.archive_root if config is not None else os.environ.get("FIELD_ARCHIVE_ROOT")

    def job_receipt(self, document):
        with self.core["db"]() as conn:
            return receipt_status(
                conn,
                "jobs",
                document["job_id"],
                (document.get("timing") or {}).get("source_written_at"),
            )

    def snapshot(self):
        with self.core["db"]() as conn:
            pending, archived = conn.execute(
                "SELECT count(*) FILTER (WHERE archived_at IS NULL), "
                "count(*) FILTER (WHERE archived_at IS NOT NULL) FROM archive_outbox"
            ).fetchone()
            indexed = dict(
                conn.execute(
                    "SELECT key,value FROM archive_meta WHERE key IN ('index_count','index_version')"
                )
            )
        config = self.core.get("config")
        managed = (
            config.production_enabled
            if config is not None
            else self.core.get("RUNTIME_ENABLED", True)
        )
        if managed and self.service_role() in {"api", "capture", "ocr"}:
            with self.core["db"]() as conn:
                runtime = conn.execute(
                    "SELECT document,updated_at FROM runtime_status WHERE name='archive'"
                ).fetchone()
            with self.lock:
                if runtime and time.time() - runtime["updated_at"] < 30:
                    self.status.update(json.loads(runtime["document"]))
                elif self.enabled():
                    self.status.update(status="retrying", last_error="archive_worker_unavailable")
        with self.lock:
            status = dict(self.status)
        return status | {
            "enabled": self.enabled(),
            "pending_receipts": pending,
            "archived_receipts": archived,
            "navigation_pending": self.enabled()
            and (
                indexed.get("index_count") != str(archived)
                or indexed.get("index_version") != self.index_version
            ),
            "root": self.configured_root(),
            "source_instance": self.instance,
            "ordinary_video_frames_saved": False,
            "integrity": self.integrity.snapshot(),
        }

    def _root(self, *, create=True):
        configured = self.configured_root()
        if not configured:
            raise ValueError("Archive root is not configured")
        root = Path(configured).resolve()
        # A pre-existing mount can be required so a missing NAS never fills local /mnt.
        config = self.core.get("config")
        required = (
            config.archive_mount if config is not None else os.environ.get("FIELD_ARCHIVE_MOUNT")
        )
        if required:
            mount = Path(required).resolve()
            if not os.path.ismount(mount):
                raise OSError("Archive mount unavailable")
            root.relative_to(mount)
        if str(root).startswith("/mnt/") and not required:
            raise ValueError("NAS archive requires FIELD_ARCHIVE_MOUNT")
        if root.name != "FieldRecognitionArchive" or "VisionCortexExperimentArchive" in root.parts:
            raise ValueError("Archive must use its own FieldRecognitionArchive directory")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        elif not root.is_dir():
            raise FileNotFoundError("Archive directory unavailable")
        return root

    def _artifact(self, root, relative, expected, *, paths=None):
        if not re.fullmatch(r"[a-f0-9]{64}", expected or ""):
            raise ValueError("Missing artifact SHA-256")
        data_root = self.core["DATA"].resolve()
        source = (data_root / relative).resolve(strict=True)
        source.relative_to(data_root)
        # A decoded 16-megapixel RGB PNG can exceed the compressed upload limit.
        if not source.is_file() or source.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Invalid archive artifact")
        reserve = max(0, int(os.environ.get("FIELD_ARCHIVE_MIN_FREE_BYTES", "268435456")))
        if shutil.disk_usage(root).free < source.stat().st_size + reserve:
            raise OSError("Archive capacity below reserve; evidence remains queued locally")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Artifact SHA-256 mismatch")
        destination = Path("Objects") / expected[:2] / (expected + source.suffix)
        from field_recognition.archive.archive_paths import ArchivePaths

        if paths is None:
            return ArchivePaths(root).asset(destination.as_posix(), raw)
        return paths.asset(destination.as_posix(), raw, save=False)

    def _capture_artifacts(self, root, capture, *, paths=None):
        from field_recognition.archive.source_layout import source_relative

        if paths is not None and paths.layout == "metadata":
            relative = source_relative(paths.root, capture)
            if relative:
                original = paths.adopt_source(relative, capture["source_sha256"])
                return {
                    "original": original,
                    "image_status": "local_normalized_png_derived_from_retained_original",
                }
        ident = str(uuid.UUID(capture["capture_id"]))
        result = {
            "image": self._artifact(
                root, Path("Images") / (ident + ".png"), capture["image_sha256"], paths=paths
            )
        }
        if capture.get("original_blob"):
            result["original"] = self._artifact(
                root, capture["original_blob"], capture["source_sha256"], paths=paths
            )
        else:
            result["original_status"] = "not_retained_by_earlier_version"
        return result

    def publish(self, row):
        from field_recognition.archive.archive_paths import ArchivePaths

        root = self._root()
        # Reuse this publication's ledger, never a previous publication's
        # object bytes or validation. Every artifact still uses immutable_write.
        config = self.core.get("config")
        paths = ArchivePaths(
            root,
            layout=config.archive_layout if config else "events",
            source_roots=config.saved_photo_roots if config else (),
        )
        doc = json.loads(row["document"])
        artifacts = {}
        capture_id = (
            doc.get("capture_id") if row["entity"] in {"scans", "jobs"} else doc.get("scan_id")
        )
        if capture_id:
            if row["entity"] == "scans":
                capture = doc
            else:
                with self.core["db"]() as conn:
                    capture_row = conn.execute(
                        "SELECT document FROM scans WHERE id=?", (capture_id,)
                    ).fetchone()
                if not capture_row:
                    raise ValueError("Source capture receipt missing")
                capture = json.loads(capture_row[0])
            artifacts = self._capture_artifacts(root, capture, paths=paths)
        full_source_view = (
            (
                artifacts.get("original", {}).get("source_owned")
                and doc.get("crop") == [0, 0, capture.get("width"), capture.get("height")]
            )
            if capture_id
            else False
        )
        if full_source_view and doc.get("crop_image_sha256"):
            artifacts["panel_status"] = "full_frame_crop_derived_locally_from_retained_original"
        if doc.get("crop_image_sha256") and not full_source_view:
            ident = str(uuid.UUID(doc["job_id"]))
            artifacts["panel"] = self._artifact(
                root, Path("Images") / (ident + ".png"), doc["crop_image_sha256"], paths=paths
            )
        for region in doc.get("panel_regions", []):
            if region.get("image_sha256"):
                ident = str(uuid.UUID(region["evidence_id"]))
                artifacts["panel_" + ident] = self._artifact(
                    root, Path("Images") / (ident + ".png"), region["image_sha256"], paths=paths
                )
        if row["entity"] in {"photo_measurements", "experiment_records"}:
            for source in doc.get("sources", []):
                for kind, artifact in self._capture_artifacts(root, source, paths=paths).items():
                    artifacts[source["capture_id"] + "_" + kind] = artifact
                for region in source.get("panel_regions", []):
                    if region.get("image_sha256"):
                        ident = str(uuid.UUID(region["evidence_id"]))
                        artifacts["panel_" + ident] = self._artifact(
                            root,
                            Path("Images") / (ident + ".png"),
                            region["image_sha256"],
                            paths=paths,
                        )
        receipt = {
            "schema": "field-recognition-receipt/1",
            "source_instance": self.instance,
            "sequence": row["seq"],
            "entity": row["entity"],
            "entity_id": row["entity_id"],
            "recorded_at": row["recorded_at"],
            "document": doc,
            "artifacts": artifacts,
            "physical_action_confirmed": False,
        }
        raw = canonical(receipt)
        digest = hashlib.sha256(raw).hexdigest()
        camera = doc.get("camera_id") or (
            row["entity_id"] if row["entity"] == "automation_settings" else "Unassigned"
        )
        if not re.fullmatch("[A-Za-z0-9_-]{1,100}", camera):
            camera = "Camera-" + hashlib.sha256(camera.encode()).hexdigest()[:16]
        timestamp = self.event_time(doc, row["recorded_at"])
        day = (
            datetime.fromisoformat(timestamp)
            .astimezone(ZoneInfo("Asia/Shanghai"))
            .date()
            .isoformat()
        )
        relative = (
            Path(".System/Receipts")
            / camera
            / day
            / self.instance
            / f"{row['seq']:012d}-{digest}.json"
        )
        paths.save()  # The receipt must never acknowledge an unpersisted artifact location.
        immutable_write(root / relative, raw)
        return relative.as_posix()

    @staticmethod
    def event_time(doc, fallback):
        return next(
            (
                doc.get(key)
                for key in (
                    "finished_at",
                    "ended_at",
                    "started_at",
                    "submitted_at",
                    "received_at",
                    "registered_at",
                )
                if doc.get(key)
            ),
            fallback,
        )

    def event_dependencies(self, conn, row):
        """Latest immutable snapshots needed before publishing this event.

        Old progress versions remain queued independently. IDs from the current
        business tables discover siblings, but their documents never substitute
        for the outbox snapshots being published.
        """
        tables = {
            item[0] for item in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        wanted, todo, captures, measurements = {}, [], set(), set()
        complete = True

        def add(entity, ident):
            nonlocal complete
            if not ident or (entity, ident) in wanted:
                return
            latest = conn.execute(
                "SELECT * FROM archive_outbox WHERE entity=? AND entity_id=? "
                "ORDER BY seq DESC LIMIT 1",
                (entity, ident),
            ).fetchone()
            if latest is None:
                complete = False
                return
            wanted[(entity, ident)] = latest
            todo.append(latest)

        add(row["entity"], row["entity_id"])
        while todo:
            current = todo.pop()
            doc = json.loads(current["document"])
            entity = current["entity"]
            capture, measurement = None, None
            if entity in {"bindings", "scene_visits", "binding_handoffs"}:
                capture = doc.get("scan_id")
            elif entity == "scans":
                capture = doc.get("capture_id") or current["entity_id"]
            elif entity == "jobs":
                capture, measurement = doc.get("capture_id"), doc.get("measurement_id")
                for previous in (doc.get("recognition_root_job_id"), doc.get("supersedes_job_id")):
                    add("jobs", previous)
            elif entity in {"photo_measurements", "experiment_records"}:
                measurement = doc.get("measurement_id")
                for job_id in doc.get("job_ids", []) + doc.get("superseded_job_ids", []):
                    add("jobs", job_id)
                for source in doc.get("sources", []):
                    add("scans", source.get("capture_id"))
            if capture and capture not in captures:
                captures.add(capture)
                add("scans", capture)
                if "jobs" in tables:
                    for sibling in conn.execute(
                        "SELECT id FROM jobs WHERE json_extract(document,'$.capture_id')=?",
                        (capture,),
                    ):
                        add("jobs", sibling["id"])
            if measurement and measurement not in measurements:
                measurements.add(measurement)
                add("photo_measurements", measurement)
                if "experiment_records" in tables:
                    for record in conn.execute(
                        "SELECT id FROM experiment_records WHERE json_extract(document,'$.measurement_id')=?",
                        (measurement,),
                    ):
                        add("experiment_records", record["id"])
        return list(wanted.values()) if complete else None

    def _next_receipt(self, conn):
        """Reselect after every row; newest results cannot wait behind a batch."""
        due = time.time()
        oldest = conn.execute(
            "SELECT * FROM archive_outbox WHERE archived_at IS NULL AND retry_after<=? "
            "ORDER BY seq LIMIT 1",
            (due,),
        ).fetchone()
        if oldest is None:
            return None, False
        if self.priority_selections < 3:
            from field_recognition.archive.archive_events import processing_outcome

            roots = conn.execute(
                """SELECT * FROM archive_outbox AS candidate
                WHERE (archived_at IS NULL OR readable_at IS NULL) AND retry_after<=?
                  AND entity IN ('jobs','photo_measurements','experiment_records')
                  AND (json_extract(document,'$.status') IN ('completed','failed','cancelled','interrupted','confirmed')
                    OR (entity='jobs' AND json_extract(document,'$.status')='running'
                      AND json_extract(document,'$.reading_mode')='parallel_photo'
                      AND json_extract(document,'$.result_available')=1))
                  AND NOT EXISTS (SELECT 1 FROM archive_outbox AS newer
                    WHERE newer.entity=candidate.entity AND newer.entity_id=candidate.entity_id AND newer.seq>candidate.seq)
                ORDER BY seq""",
                (due,),
            ).fetchall()
            for root in roots:
                if (
                    root["entity"] == "jobs"
                    and not processing_outcome(json.loads(root["document"]))["terminal"]
                ):
                    from field_recognition.measurements.reading_results import result_ready

                    if not result_ready(json.loads(root["document"])):
                        continue
                dependencies = self.event_dependencies(conn, root)
                if dependencies is None:
                    continue
                waiting = [item for item in dependencies if item["archived_at"] is None]
                # Respect failed dependencies' backoff; another event or the
                # oldest historical receipt can continue making progress.
                if not waiting or any(item["retry_after"] > due for item in waiting):
                    continue
                order = {"scans": 0, "jobs": 1, "photo_measurements": 2, "experiment_records": 3}
                selected = min(
                    waiting, key=lambda item: (order.get(item["entity"], 4), item["seq"])
                )
                self.priority_selections += 1
                return selected, True
        self.priority_selections = 0
        return oldest, False

    def _publish_ready_events(self):
        """Independent event failures never stop receipt or other event delivery."""
        from field_recognition.archive.archive_migration import publish_events

        with self.core["db"]() as conn:
            confirmation = conn.execute(
                "SELECT value FROM archive_meta WHERE key='event_confirmation_pending'"
            ).fetchone()
            confirmations = set(json.loads(confirmation[0])) if confirmation else set()
            event_marks = ",".join("?" for _ in EVENT_ENTITIES)
            pending = conn.execute(
                "SELECT * FROM archive_outbox WHERE published_at IS NOT NULL "
                f"AND readable_at IS NULL AND entity IN ({event_marks}) ORDER BY seq",
                EVENT_ENTITIES,
            ).fetchall()
            if confirmations:
                marks = ",".join("?" for _ in confirmations)
                known = {row["seq"] for row in pending}
                pending += [
                    row
                    for row in conn.execute(
                        f"SELECT * FROM archive_outbox WHERE seq IN ({marks}) "
                        "AND published_at IS NOT NULL",
                        tuple(confirmations),
                    )
                    if row["seq"] not in known
                ]
            groups, checked_entities = [], {}
            for row in pending:
                key = (row["entity"], row["entity_id"])
                if key not in checked_entities:
                    dependencies = self.event_dependencies(conn, row)
                    checked_entities[key] = (
                        {(item["entity"], item["entity_id"]) for item in dependencies}
                        if dependencies is not None
                        and all(item["archived_at"] is not None for item in dependencies)
                        else None
                    )
                keys = checked_entities[key]
                if keys is None:
                    continue
                # Sibling/retry/group dependencies may overlap without having
                # identical roots. Merge their connected closures as one unit.
                merged = {"keys": set(keys), "sequences": {row["seq"]}}
                while True:
                    overlap = next(
                        (group for group in groups if group["keys"] & merged["keys"]), None
                    )
                    if overlap is None:
                        break
                    merged["keys"].update(overlap["keys"])
                    merged["sequences"].update(overlap["sequences"])
                    groups.remove(overlap)
                groups.append(merged)
            retries = dict(
                conn.execute("SELECT key,value FROM archive_meta WHERE key GLOB 'view_retry:*'")
            )
        for group in sorted(groups, key=lambda value: min(value["sequences"])):
            if self.stop.is_set():
                return
            key = "view_retry:" + hashlib.sha256(canonical(sorted(group["keys"]))).hexdigest()
            previous = json.loads(retries[key]) if key in retries else {}
            if previous.get("retry_after", 0) > time.time():
                continue
            if not self.integrity.lock.acquire(timeout=5):
                with self.lock:
                    self.status.update(status="waiting_integrity")
                return
            try:
                with self.lock:
                    self.status.update(status="indexing")
                try:
                    publish_events(self, sequences=group["sequences"])
                except Exception as exc:
                    attempts = previous.get("attempts", 0) + 1
                    failure = {
                        "error": type(exc).__name__,
                        "attempts": attempts,
                        "retry_after": time.time() + min(900, 15 * 2 ** min(attempts - 1, 6)),
                        "recorded_at": self.core["now"](),
                        "sequences": sorted(group["sequences"]),
                    }
                    with self.core["db"]() as conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO archive_meta VALUES(?,?)",
                            (key, json.dumps(failure)),
                        )
                    with self.lock:
                        self.status.update(status="retrying", last_error=type(exc).__name__)
                else:
                    with self.core["db"]() as conn:
                        conn.execute("DELETE FROM archive_meta WHERE key=?", (key,))
                        # Membership can grow between attempts; successful
                        # delivery also clears older obligations for these seeds.
                        for old_key, raw in retries.items():
                            sequences = set(json.loads(raw).get("sequences", []))
                            if sequences and sequences <= group["sequences"]:
                                conn.execute("DELETE FROM archive_meta WHERE key=?", (old_key,))
            finally:
                self.integrity.lock.release()

    def write_index(self):
        from field_recognition.archive.archive_migration import publish_events, rebuild
        from field_recognition.archive.archive_progress import checkpoint

        with self.lock:
            self.status.update(status="indexing")
        # The verifier cannot observe a file between path resolution and rename.
        if not self.integrity.lock.acquire(timeout=5):
            with self.lock:
                self.status.update(status="waiting_integrity")
            return None
        try:
            started = time.monotonic()
            publish_events(self)

            class NewReceipts(Exception):
                pass

            def yield_to_receipts():
                with self.core["db"]() as conn:
                    waiting = conn.execute(
                        "SELECT 1 FROM archive_outbox WHERE archived_at IS NULL AND retry_after<=? LIMIT 1",
                        (time.time(),),
                    ).fetchone()
                if self.stop.is_set() or waiting:
                    raise NewReceipts()

            try:
                yield_to_receipts()
                with self.core["db"]() as conn:
                    rows = conn.execute(
                        "SELECT * FROM archive_outbox WHERE archived_at IS NOT NULL ORDER BY seq DESC"
                    ).fetchall()
                yield_to_receipts()
                with checkpoint(yield_to_receipts):
                    result = rebuild(self, rows)
            except NewReceipts:
                return None
            with self.lock:
                self.status["last_view_rebuild"] = {
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                    "clock_basis": "process_monotonic",
                    "scope": "whole_archive_derived_views",
                    **result,
                }
            return result
        finally:
            self.integrity.lock.release()

    def step(self, batch_size=20):
        if not self.enabled():
            return
        with self.core["db"]() as conn:
            waiting_view = conn.execute(
                "SELECT 1 FROM archive_outbox WHERE published_at IS NOT NULL "
                "AND readable_at IS NULL LIMIT 1"
            ).fetchone()
            confirmation = conn.execute(
                "SELECT value FROM archive_meta WHERE key='event_confirmation_pending'"
            ).fetchone()
        if waiting_view or (confirmation and json.loads(confirmation[0])):
            self._publish_ready_events()
        for _ in range(batch_size):
            if self.stop.is_set():
                return
            with self.core["db"]() as conn:
                row, priority = self._next_receipt(conn)
            if row is None:
                self._root()  # Report a disconnected target even when the queue is empty.
                break
            try:
                publication_started_at = self.core["now"]()
                publication_started = time.monotonic()
                relative = self.publish(row)
                publication_ms = round((time.monotonic() - publication_started) * 1000, 2)
            except Exception as exc:
                with self.core["db"]() as conn:
                    conn.execute(
                        "UPDATE archive_outbox SET attempts=attempts+1,last_error=?,retry_after=? WHERE seq=?",
                        (
                            type(exc).__name__,
                            time.time() + min(900, 15 * 2 ** min(row["attempts"], 6)),
                            row["seq"],
                        ),
                    )
                with self.lock:
                    self.status.update(status="retrying", last_error=type(exc).__name__)
                continue
            timestamp = self.core["now"]()
            with self.core["db"]() as conn:
                conn.execute(
                    "UPDATE archive_outbox SET archived_at=?,published_at=?,receipt_path=?,last_error=NULL,"
                    "publication_started_at=?,receipt_publication_ms=? WHERE seq=?",
                    (
                        timestamp,
                        timestamp,
                        relative,
                        publication_started_at,
                        publication_ms,
                        row["seq"],
                    ),
                )
            with self.lock:
                self.status.update(last_archived_at=timestamp)
                doc = json.loads(row["document"])
                self.status["last_receipt_publication"] = {
                    "sequence": row["seq"],
                    "entity": row["entity"],
                    "entity_id": row["entity_id"],
                    "task_id": doc.get("job_id"),
                    "group_id": (doc.get("measurement_context") or {}).get("burst_id")
                    or doc.get("measurement_id"),
                    "duration_ms": publication_ms,
                    "clock_basis": "process_monotonic",
                    "scope": "receipt_and_artifacts",
                }
            if priority:
                self._publish_ready_events()
        self.integrity.publish_pending()
        with self.core["db"]() as conn:
            # Include publications whose view pass was interrupted in an earlier
            # step. Legacy receipts have no measured published_at and stay null.
            published_sequences = [
                r[0]
                for r in conn.execute(
                    "SELECT seq FROM archive_outbox WHERE published_at IS NOT NULL AND readable_at IS NULL"
                )
            ]
            archived = conn.execute(
                "SELECT count(*) FROM archive_outbox WHERE archived_at IS NOT NULL"
            ).fetchone()[0]
            indexed = conn.execute(
                "SELECT value FROM archive_meta WHERE key='index_count'"
            ).fetchone()
            version = conn.execute(
                "SELECT value FROM archive_meta WHERE key='index_version'"
            ).fetchone()
            audit = conn.execute(
                "SELECT value FROM archive_meta WHERE key='index_integrity'"
            ).fetchone()
            confirmation = conn.execute(
                "SELECT value FROM archive_meta WHERE key='event_confirmation_pending'"
            ).fetchone()
            pending_receipts = conn.execute(
                "SELECT 1 FROM archive_outbox WHERE archived_at IS NULL LIMIT 1"
            ).fetchone()
        root = self._root()
        index_ready = False
        if (
            published_sequences
            or (confirmation and json.loads(confirmation[0]))
            or not indexed
            or int(indexed[0]) != archived
            or not version
            or version[0] != self.index_version
            or not audit
            or audit[0] != (self.integrity.snapshot()["report_path"] or "")
            or not (root / "Readme.html").is_file()
            or not (root / ".System/Audit.html").is_file()
            or not (root / ".System/Index.json").is_file()
            or not (root / ".System/MigrationMap.json").is_file()
        ):
            if (
                pending_receipts
                or published_sequences
                or (confirmation and json.loads(confirmation[0]))
            ):
                self._publish_ready_events()
            with self.core["db"]() as conn:
                event_marks = ",".join("?" for _ in EVENT_ENTITIES)
                remaining_views = conn.execute(
                    "SELECT 1 FROM archive_outbox WHERE published_at IS NOT NULL "
                    f"AND readable_at IS NULL AND entity IN ({event_marks}) LIMIT 1",
                    EVENT_ENTITIES,
                ).fetchone()
                remaining_confirmation = conn.execute(
                    "SELECT value FROM archive_meta WHERE key='event_confirmation_pending'"
                ).fetchone()
                pending_receipts = conn.execute(
                    "SELECT 1 FROM archive_outbox WHERE archived_at IS NULL LIMIT 1"
                ).fetchone()
            if (
                not pending_receipts
                and not remaining_views
                and not (remaining_confirmation and json.loads(remaining_confirmation[0]))
            ):
                index_ready = self.write_index() is not None
        if index_ready and published_sequences:
            readable_at = self.core["now"]()
            placeholders = ",".join("?" for _ in published_sequences)
            with self.core["db"]() as conn:
                updated = conn.execute(
                    f"""UPDATE archive_outbox SET readable_at=?,readability_basis=?
                                 WHERE seq IN ({placeholders}) AND published_at IS NOT NULL
                                   AND readable_at IS NULL AND entity='automation_settings'""",
                    (readable_at, "archive_writer_derived_views_published", *published_sequences),
                ).rowcount
                # Force a second derived-view pass so Evidence.json and the
                # index expose the readability confirmation just established.
                if updated:
                    conn.execute(
                        "INSERT OR REPLACE INTO archive_meta VALUES('index_version','readability_metadata_pending')"
                    )
            if updated:
                self.write_index()
        with self.core["db"]() as conn:
            failed = conn.execute(
                "SELECT last_error FROM archive_outbox WHERE archived_at IS NULL AND last_error IS NOT NULL LIMIT 1"
            ).fetchone()
            view_retries = [
                json.loads(row[0])
                for row in conn.execute(
                    "SELECT value FROM archive_meta WHERE key GLOB 'view_retry:*'"
                )
            ]
        with self.lock:
            error = failed[0] if failed else view_retries[0]["error"] if view_retries else None
            self.status.update(
                status="retrying" if error else "ready",
                last_error=error,
                pending_event_view_retries=len(view_retries),
            )

    def start(self):
        if self.enabled() and self.service_role() in {"combined", "archive"}:
            self.worker = threading.Thread(target=self._run, name="receipt-archive", daemon=True)
            self.worker.start()
            self.integrity.start()

    def next_poll_seconds(self):
        # Drain due batches immediately. Deferred failures retain their backoff;
        # an unavailable target never causes a busy retry loop.
        if not self.enabled():
            return 3
        with self.core["db"]() as conn:
            due = conn.execute(
                "SELECT 1 FROM archive_outbox WHERE archived_at IS NULL AND retry_after<=? LIMIT 1",
                (time.time(),),
            ).fetchone()
        return 0 if due else 0.5

    def _run(self):
        while not self.stop.is_set():
            delay = 3
            try:
                self.step()
                delay = self.next_poll_seconds()
            except Exception as exc:
                with self.lock:
                    self.status.update(status="retrying", last_error=type(exc).__name__)
            self.stop.wait(delay)

    def close(self):
        self.integrity.close()
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=3)
