"""Input readiness is independent of task execution and historical success."""

import json
import time
from pathlib import Path
from .sqlite_store import connection


class Availability:
    def __init__(self, root):
        self.path = Path(root) / "InputAvailability.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS inputs(id TEXT PRIMARY KEY, signature TEXT,
                state TEXT, observed REAL, reason TEXT)""")

    def mark(self, record, state, *, observed=None, reason=""):
        self.mark_many([(record, state, observed, reason)])

    def mark_many(self, entries):
        values = []
        for record, state, observed, reason in entries:
            if state not in {"ready", "missing", "waiting", "unavailable"}:
                raise ValueError("Invalid input state")
            values.append(
                (
                    record["recording_id"],
                    record.get("source_signature"),
                    state,
                    time.time() if observed is None else observed,
                    reason,
                )
            )
        with connection(self.path) as db:
            db.executemany(
                """INSERT INTO inputs VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                signature=excluded.signature,state=excluded.state,observed=excluded.observed,reason=excluded.reason
                WHERE excluded.observed>inputs.observed""",
                values,
            )

    def states(self):
        with connection(self.path, readonly=True) as db:
            return {r["id"]: dict(r) for r in db.execute("SELECT * FROM inputs")}

    def migrate_legacy(self, queue):
        # One-time import of old, explicit unavailable observations. No stat of
        # the NAS and no rewriting completed model receipts.
        with queue.connect() as db:
            rows = list(
                db.execute(
                    "SELECT payload,updated_at,result FROM recordings WHERE status!='completed'"
                )
            )
        entries = []
        for row in rows:
            record = json.loads(row["payload"])
            result = json.loads(row["result"] or "{}")
            if (
                result.get("error_type") == "FileNotFoundError"
                or record.get("available") is False
                and record.get("processable") is False
            ):
                entries.append(
                    (
                        record,
                        "missing"
                        if result.get("error_type") == "FileNotFoundError"
                        else "waiting",
                        row["updated_at"],
                        "legacy_unavailable_observation",
                    )
                )
        self.mark_many(entries)


class Reconciler:
    """Bounded metadata checks on a separate lane, never in a progress request."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config["storage"]["local_runtime_root"]) / "device-day"
        self.cursor = ""
        self.migrated = False

    def tick(self):
        from .device_day_queue import DeviceDayQueue

        path = self.root / "queue-retention.sqlite3"
        if not path.is_file():
            return
        queue = DeviceDayQueue(path)
        availability = Availability(self.root)
        if not self.migrated:
            availability.migrate_legacy(queue)
            self.migrated = True
        with queue.connect() as db:
            rows = list(
                db.execute(
                    "SELECT * FROM recordings WHERE status NOT IN ('completed','running') "
                    "AND recording_id>? ORDER BY recording_id LIMIT 24",
                    (self.cursor,),
                )
            )
        if not rows:
            self.cursor = ""
        for row in rows:
            self.cursor = row["recording_id"]
            record = json.loads(row["payload"])
            video = record.get("video_path")
            if not video:
                continue
            try:
                info = Path(video).stat()
                if info.st_size <= 0:
                    availability.mark(record, "waiting", reason="empty_capture")
                    continue
                # A present file is not necessarily a closed, compatible input.
                from .nas_recordings import _inspect

                source = Path(self.config["collection_ingest"]["source_root"])
                fresh = _inspect(
                    source,
                    Path(video),
                    time.time(),
                    float(self.config["collection_ingest"].get("settle_seconds", 5)),
                )
                if fresh.get("processable"):
                    availability.mark(fresh, "ready", reason="capture_reinspected")
                    from .observed_inventory import observe

                    observe(self.root, {"recordings": [fresh]})
                else:
                    availability.mark(record, "waiting", reason="capture_not_closed")
            except FileNotFoundError:
                # Retention recovery separately verifies archive-only originals.
                # Missing capture never authorizes changing a successful receipt.
                availability.mark(
                    record,
                    "missing",
                    reason="capture_missing_pending_archive_verification",
                )
            except (OSError, ValueError):
                availability.mark(
                    record, "unavailable", reason="input_inspection_unavailable"
                )
        # A verified retained original remains usable after capture replacement.
        states = availability.states()
        with queue.connect() as db:
            completed = list(
                db.execute("SELECT payload FROM recordings WHERE status='completed'")
            )
        for row in completed:
            record = json.loads(row["payload"])
            if states.get(record["recording_id"], {}).get("state", "ready") != "ready":
                availability.mark(
                    record, "ready", reason="verified_retention_completed"
                )
