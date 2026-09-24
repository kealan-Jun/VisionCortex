"""Durable FIFO backlog; discovery limits never remove already enqueued work."""
from __future__ import annotations

import json
from contextlib import contextmanager
import logging
import sqlite3
import threading
import time
from pathlib import Path


class DeviceDayQueue:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS recordings (
              recording_id TEXT PRIMARY KEY, revision TEXT NOT NULL, payload TEXT NOT NULL,
              status TEXT NOT NULL, queued_at REAL NOT NULL, updated_at REAL NOT NULL,
              lease_owner TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
              result TEXT, completed_at REAL, wall_seconds REAL);
            CREATE TABLE IF NOT EXISTS camera_dispatch (camera_key TEXT PRIMARY KEY, last_claim REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS device_day_fifo ON recordings(status, queued_at);
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {r[1] for r in db.execute("PRAGMA table_info(recordings)")}
            from .task_events import initialize
            initialize(db)
            if "input_status" not in columns:
                db.execute("ALTER TABLE recordings ADD COLUMN input_status TEXT NOT NULL DEFAULT 'ready'")
            if "started_at" not in columns:
                db.execute("ALTER TABLE recordings ADD COLUMN started_at REAL")

    def connect(self):
        from .sqlite_store import connection
        return connection(self.path)

    def enqueue(self, recording, revision):
        observed = time.time()
        payload = json.dumps(recording, ensure_ascii=False)
        status = "queued" if recording.get("configured_role") in {"first_person", "third_person"} else "needs_camera_role"
        with self.connect() as db:
            changed = db.execute("""INSERT INTO recordings(recording_id,revision,payload,status,queued_at,updated_at)
              VALUES(?,?,?,?,?,?) ON CONFLICT(recording_id) DO UPDATE SET
              revision=excluded.revision,payload=excluded.payload,status=excluded.status,
              queued_at=excluded.queued_at,updated_at=excluded.updated_at,lease_owner=NULL,
              lease_until=NULL,result=NULL,completed_at=NULL,wall_seconds=NULL,started_at=NULL,attempts=0,
              input_status=CASE WHEN recordings.revision!=excluded.revision THEN 'ready' ELSE recordings.input_status END
              WHERE (recordings.revision != excluded.revision OR
                (recordings.status='needs_camera_role' AND excluded.status='queued')) AND
                (recordings.status != 'running' OR COALESCE(recordings.lease_until,0) < ?)
            """, (recording["recording_id"], revision, payload, status, observed, observed, observed)).rowcount
            if changed:
                from .task_events import append
                append(db, recording["recording_id"], status, revision=revision, data={"camera": recording.get("camera_key"), "date": recording.get("archive_date")})
            # Visual revisions intentionally exclude scheduling/audio metadata.
            # Keep the claim payload aligned with the inventory used to verify
            # prerequisites, even when the visual work itself is unchanged.
            # Never replace an actively leased worker's input snapshot.
            db.execute("""UPDATE recordings SET payload=?
              WHERE recording_id=? AND revision=? AND payload!=? AND
                (status!='running' OR COALESCE(lease_until,0)<?)""",
                       (payload, recording["recording_id"], revision, payload, observed))

    def pending(self):
        with self.connect() as db:
            return [json.loads(row["payload"]) for row in db.execute(
                "SELECT payload FROM recordings WHERE status != 'completed'")]

    def has_processing_work(self, max_attempts=0):
        """Keep expensive background audits behind ready stage work."""
        with self.connect() as db:
            return db.execute(
                "SELECT 1 FROM recordings WHERE input_status='ready' AND "
                "(status IN ('queued','running') OR (status='failed' AND attempts<?)) LIMIT 1",
                (max_attempts,),
            ).fetchone() is not None

    def revise_verified_completion(self, recording, previous_revision, revision):
        """Keep completion only after the caller revalidates the current receipt.

        Compare-and-swap prevents a concurrent claim or source revision from
        being overwritten. This operation does not manufacture a new result.
        """
        with self.connect() as db:
            return db.execute(
                "UPDATE recordings SET revision=?,payload=?,updated_at=? "
                "WHERE recording_id=? AND revision=? AND status='completed'",
                (revision, json.dumps(recording, ensure_ascii=False), time.time(),
                 recording['recording_id'], previous_revision),
            ).rowcount == 1

    def claim(self, owner, *, retry=False, date=None, exclude=(), allowed=None, camera_serial=False,
              camera_limit=1, max_attempts=None):
        if isinstance(camera_limit, bool) or not isinstance(camera_limit, int) or camera_limit < 1:
            raise ValueError('Camera concurrency must be a positive integer')
        current = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Filter and order in SQLite before loading one payload. The old
            # implementation decoded every pending JSON for every camera slot.
            # json_each keeps large allowlists below SQLite's parameter limit.
            conditions = ["(r.status='queued' OR (r.status='running' AND COALESCE(r.lease_until,0)<?)"
                          + (" OR (r.status='failed' AND (? IS NULL OR r.attempts<?)))" if retry else ")")]
            conditions.append("r.input_status='ready'")
            parameters = [current]
            if retry:
                parameters.extend([max_attempts, max_attempts])
            if exclude:
                conditions.append("r.recording_id NOT IN (SELECT value FROM json_each(?))")
                parameters.append(json.dumps(list(exclude)))
            if allowed is not None:
                conditions.append("r.recording_id IN (SELECT value FROM json_each(?))")
                parameters.append(json.dumps(list(allowed)))
            if date is not None:
                conditions.append("json_extract(r.payload,'$.archive_date')=?")
                parameters.append(date)
            if camera_serial:
                # Compute occupied camera slots once, rather than re-reading
                # every running JSON payload for each item in a large backlog.
                conditions.append("""COALESCE(json_extract(r.payload,'$.camera_key'),'') NOT IN (
                  SELECT COALESCE(json_extract(busy.payload,'$.camera_key'),'') FROM recordings busy
                  WHERE busy.status='running' AND busy.lease_until>=?
                  GROUP BY COALESCE(json_extract(busy.payload,'$.camera_key'),'') HAVING COUNT(*)>=?)""")
                parameters.extend((current, camera_limit))
            ordering = "COALESCE(json_extract(r.payload,'$.processing_priority'),0),"
            if self.path.stem == 'queue-retention':
                ordering = ("CASE WHEN json_extract(r.payload,'$.archive_urgent') THEN 0 "
                            "WHEN json_extract(r.payload,'$.archive_aged') THEN 1 ELSE 2 END,"
                            "COALESCE(json_extract(r.payload,'$.archive_deadline'),9e99),r.queued_at," + ordering)
            if camera_serial:
                ordering += ("COALESCE((SELECT last_claim FROM camera_dispatch "
                             "WHERE camera_key=json_extract(r.payload,'$.camera_key')),0),"
                             "CASE WHEN COALESCE(json_extract(r.payload,'$.processing_priority'),0)<0 "
                             "THEN -COALESCE(json_extract(r.payload,'$.recording_start_us'),0) "
                             "ELSE COALESCE(json_extract(r.payload,'$.recording_start_us'),0) END,")
            ordering += "r.queued_at,r.recording_id"
            row = db.execute("SELECT r.* FROM recordings r WHERE " + " AND ".join(conditions)
                             + " ORDER BY " + ordering + " LIMIT 1", parameters).fetchone()
            if row is None:
                return None
            if camera_serial:
                camera = json.loads(row["payload"]).get("camera_key")
                if camera:
                    db.execute("INSERT INTO camera_dispatch(camera_key,last_claim) VALUES(?,?) "
                               "ON CONFLICT(camera_key) DO UPDATE SET last_claim=excluded.last_claim",
                               (camera, current))
            db.execute("""UPDATE recordings SET status='running',lease_owner=?,lease_until=?,
              attempts=attempts+1,updated_at=?,started_at=? WHERE recording_id=?""",
                       (owner, current + 90, current, current, row["recording_id"]))
            from .task_events import append
            append(db, row['recording_id'], 'running', revision=row['revision'], attempt=row['attempts']+1)
            return json.loads(row["payload"])

    def renew(self, owner):
        with self.connect() as db:
            db.execute("UPDATE recordings SET lease_until=? WHERE status='running' AND lease_owner=?",
                       (time.time() + 90, owner))

    @contextmanager
    def heartbeat(self, owner, interval=10):
        """Renew while result publication or queue summaries occupy the caller."""
        stopped = threading.Event()
        def renew():
            while not stopped.wait(interval):
                try:
                    self.renew(owner)
                except sqlite3.OperationalError:
                    logging.getLogger(__name__).warning('Queue lease renewal delayed: %s', self.path.name)
        thread = threading.Thread(target=renew, name='device-queue-lease', daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=31)

    def finish(self, owner, recording_id, result, seconds):
        if result.get('status') == 'waiting_for_prerequisite':
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT revision,attempts FROM recordings WHERE recording_id=? AND lease_owner=?',
                                 (recording_id, owner)).fetchone()
                if row:
                    db.execute("UPDATE recordings SET status='waiting_for_prerequisite',result=?,"
                               "attempts=MAX(0,attempts-1),lease_owner=NULL,lease_until=NULL,updated_at=? "
                               "WHERE recording_id=? AND lease_owner=?",
                               (json.dumps(result), time.time(), recording_id, owner))
                    from .task_events import append
                    append(db, recording_id, 'waiting_for_prerequisite', revision=row['revision'],
                           attempt=row['attempts'], data={'prerequisite_stage': result.get('prerequisite_stage')})
            return
        if result.get("status") in {"running_elsewhere", "waiting_for_publication", "waiting_for_provider", "cancelled"}:
            # Lock contention is not a model failure. Return the slice to
            # scheduling; completed model receipts remain available for reuse.
            with self.connect() as db:
                db.execute("""UPDATE recordings SET status='queued',result=?,
                  attempts=MAX(0,attempts-1),lease_owner=NULL,lease_until=NULL,
                  updated_at=? WHERE recording_id=? AND lease_owner=?""",
                           (json.dumps(result, ensure_ascii=False), time.time(), recording_id, owner))
            return
        completed = result.get("status") == "completed"
        status = "completed" if completed else "failed"
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision,attempts,payload FROM recordings WHERE recording_id=? AND lease_owner=?', (recording_id, owner)).fetchone()
            db.execute("""UPDATE recordings SET status=?,result=?,completed_at=?,wall_seconds=?,
              lease_owner=NULL,lease_until=NULL,updated_at=? WHERE recording_id=? AND lease_owner=?""",
                       (status, json.dumps(result, ensure_ascii=False), time.time() if completed else None,
                        seconds, time.time(), recording_id, owner))
            if row:
                from .task_events import append
                append(db, recording_id, status, revision=row['revision'], attempt=row['attempts'], data={
                    key: result[key] for key in ('archive', 'stage', 'error_type', 'component_timings', 'measured_frame_counts') if key in result})
                if self.path.stem == 'queue-retention' and result.get('input_binding_version') != 1 and result.get('error_type') in {'FileNotFoundError', 'PermissionError', 'OSError'}:
                    state = 'missing' if result['error_type'] == 'FileNotFoundError' else 'unavailable'
                    db.execute('UPDATE recordings SET input_status=? WHERE recording_id=?', (state, recording_id))
        if row and self.path.stem == 'queue-retention' and result.get('input_binding_version') != 1 and result.get('error_type') in {'FileNotFoundError', 'PermissionError', 'OSError'}:
            from .input_availability import Availability
            record = json.loads(row['payload'])
            Availability(self.path.parent).mark(record, state, reason=result['error_type'])

    def wait_for_prerequisite(self, recording_id, revision, prerequisite):
        """CAS a queued or precisely identified legacy failure into a wait.

        Only the final historical attempt is known to be a prerequisite error.
        Refund that one attempt, preserving any earlier genuine failure budget.
        """
        from .device_day_prerequisites import legacy_prerequisite_failure
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM recordings WHERE recording_id=? AND revision=?',
                             (recording_id, revision)).fetchone()
            expired = bool(row and row['status'] == 'running' and (row['lease_until'] or 0) < time.time())
            if not row or row['status'] not in {'queued', 'failed'} and not expired:
                return False
            previous = json.loads(row['result'] or '{}')
            failed = row['status'] == 'failed'
            if failed and legacy_prerequisite_failure(previous) != prerequisite:
                return False
            result = {'status': 'waiting_for_prerequisite', 'recording_id': recording_id,
                      'prerequisite_stage': prerequisite}
            if failed:
                result['previous_failure'] = previous
            db.execute("UPDATE recordings SET status='waiting_for_prerequisite',result=?,"
                       "attempts=MAX(0,attempts-?),lease_owner=NULL,lease_until=NULL,updated_at=? "
                       "WHERE recording_id=? AND revision=?",
                       (json.dumps(result), int(failed), time.time(), recording_id, revision))
            from .task_events import append
            append(db, recording_id, 'waiting_for_prerequisite', revision=revision,
                   attempt=row['attempts'], data={'prerequisite_stage': prerequisite,
                   'legacy_failure_refunded': failed, 'refunded_attempts': int(failed),
                   'expired_lease_recovered': expired})
            return True

    def migrate_prerequisite_failures(self):
        from .device_day_prerequisites import legacy_prerequisite_failure
        with self.connect() as db:
            rows = list(db.execute("SELECT recording_id,revision,result FROM recordings WHERE status='failed' "
                                   "AND json_extract(result,'$.error_type')='ValueError'"))
        for row in rows:
            prerequisite = legacy_prerequisite_failure(json.loads(row['result'] or '{}'))
            if prerequisite:
                self.wait_for_prerequisite(row['recording_id'], row['revision'], prerequisite)

    def resume_prerequisite(self, recording_id, revision):
        """Caller has verified every exact prerequisite receipt and artifact."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            changed = db.execute("UPDATE recordings SET status='queued',updated_at=? "
                                 "WHERE recording_id=? AND revision=? AND status='waiting_for_prerequisite'",
                                 (time.time(), recording_id, revision)).rowcount
            if changed:
                from .task_events import append
                append(db, recording_id, 'prerequisite_ready', revision=revision,
                       data={'all_prerequisites_verified': True})
            return bool(changed)

    def sync_availability(self, states):
        with self.connect() as db:
            db.executemany("UPDATE recordings SET input_status=? WHERE recording_id=? AND status NOT IN ('completed','running') AND input_status!=? AND json_extract(payload,'$.source_signature') IS ?",
                           [(s['state'], key, s['state'], s.get('signature')) for key,s in states.items()])

    def snapshot(self):
        with self.connect() as db:
            current = time.time()
            # Expired leases are waiting for recovery, not live concurrency.
            counts = {row["effective_status"]: row["n"] for row in db.execute("""
              SELECT CASE WHEN status NOT IN ('completed','running') AND input_status!='ready' THEN 'input_'||input_status
                WHEN status='running' AND COALESCE(lease_until,0)<?
                THEN 'queued' ELSE status END effective_status, COUNT(*) n
              FROM recordings GROUP BY effective_status""", (current,))}
            expired = db.execute("SELECT COUNT(*) FROM recordings WHERE status='running' AND COALESCE(lease_until,0)<?",
                                 (current,)).fetchone()[0]
            prerequisite_waits = {row['stage'] or 'unknown': row['n'] for row in db.execute(
                "SELECT json_extract(result,'$.prerequisite_stage') AS stage,COUNT(*) AS n "
                "FROM recordings WHERE status='waiting_for_prerequisite' GROUP BY stage")}
            queued = db.execute("""SELECT MIN(queued_at) oldest,
              COALESCE(SUM(COALESCE(json_extract(payload,'$.duration_seconds'),0)),0) capture_seconds,
              COALESCE(SUM(json_extract(payload,'$.media_duration_seconds')),0) media_seconds,
              COALESCE(SUM(json_extract(payload,'$.media_duration_seconds') IS NULL),0) unknown
              FROM recordings WHERE (status='queued' AND input_status='ready') OR (status='running' AND COALESCE(lease_until,0)<?)""",
                                (current,)).fetchone()
            oldest = queued["oldest"]
            completed = db.execute("SELECT result,wall_seconds FROM recordings WHERE status='completed' ORDER BY completed_at DESC LIMIT 100").fetchall()
        known = [(json.loads(r["result"] or "{}").get("media_duration_seconds"), r["wall_seconds"]) for r in completed]
        measured = [(float(media), float(wall)) for media, wall in known if media is not None and wall and wall > 0]
        return {"counts": counts,
                "prerequisite_wait_count": sum(prerequisite_waits.values()),
                "prerequisite_waits_by_stage": prerequisite_waits,
                "expired_lease_count": expired,
                "queued_capture_window_seconds": queued["capture_seconds"],
                "queued_media_seconds": queued["media_seconds"] if not queued["unknown"] else None,
                "queued_media_duration_unknown_count": queued["unknown"],
                "oldest_wait_seconds": max(0, time.time() - oldest) if oldest else 0,
                "backlog_warning": bool(oldest and time.time() - oldest > 900),
                "completed_sample_count": len(completed), "measured_media_sample_count": len(measured),
                "media_seconds_per_worker_second": sum(m for m, _ in measured) / sum(w for _, w in measured) if measured else None,
                "capacity_claim": "measured_worker_service_rate_not_8_to_12_hour_stability_proof"}
