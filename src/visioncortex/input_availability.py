"""Input readiness is independent of task execution and historical success."""

import json
import time
from pathlib import Path
from .sqlite_store import connection


def configured_record(config, record):
    """Restore the explicit camera binding after metadata-only reinspection.

    No camera-prefix inference: unregistered cameras still need a binding.
    Do not mutate an inventory or an actively leased worker's snapshot.
    """
    role = config.get("collection_ingest", {}).get("camera_role_map", {}).get(
        record.get("camera_key")
    )
    return record | {"configured_role": role} if role else record


class Availability:
    def __init__(self, root):
        self.path = Path(root) / "InputAvailability.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            # Discovery writes and scheduler snapshots use separate connections.
            # Rollback journaling lets even a read snapshot block publication.
            db.execute('PRAGMA journal_mode=WAL')
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
        if not values:
            return
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
        self.last_priority_check = {}

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
        states = availability.states()
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
        # Recheck transient I/O failures promptly without starving the bounded
        # historical sweep or repeatedly hitting the same unavailable input.
        current = time.time()
        priority = [key for key, state in states.items()
                    if state['state'] == 'unavailable'
                    and current - self.last_priority_check.get(key, 0) >= 30][:8]
        with queue.connect() as db:
            extra = list(db.execute(
                "SELECT * FROM recordings WHERE status NOT IN ('completed','running') "
                "AND recording_id IN (SELECT value FROM json_each(?))", (json.dumps(priority),)))
        for key in priority:
            self.last_priority_check[key] = current
        if rows:
            self.cursor = rows[-1]['recording_id']
        rows = list({row['recording_id']: row for row in rows + extra}.values())
        source = Path(self.config["collection_ingest"]["source_root"])
        try:
            # A mount outage is not proof that every source video was removed.
            if not source.is_dir():
                return
        except OSError:
            return
        for row in rows:
            record = json.loads(row["payload"])
            video = record.get("video_path")
            if not video:
                continue
            inspecting_metadata = False
            try:
                info = Path(video).stat()
                if info.st_size <= 0:
                    availability.mark(record, "waiting", reason="empty_capture")
                    continue
                # A present file is not necessarily a closed, compatible input.
                from .nas_recordings import _inspect

                inspecting_metadata = True
                fresh = _inspect(
                    source,
                    Path(video),
                    time.time(),
                    float(self.config["collection_ingest"].get("settle_seconds", 5)),
                )
                fresh = configured_record(self.config, fresh)
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
                    "waiting" if inspecting_metadata else "missing",
                    reason="capture_metadata_missing" if inspecting_metadata else
                    "capture_missing_pending_archive_verification",
                )
            except (OSError, ValueError) as exc:
                availability.mark(
                    record, "unavailable", reason=f"input_inspection_unavailable:{type(exc).__name__}:errno={getattr(exc, 'errno', None)}"
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
