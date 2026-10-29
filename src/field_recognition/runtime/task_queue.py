"""SQLite job leases. Every publish is fenced by a unique claim token."""

import json
import random
import time
import uuid
from datetime import datetime, timezone

from field_recognition.runtime.database import Database


class LeaseLost(RuntimeError):
    pass


class QueueFull(RuntimeError):
    pass


def utc(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


class TaskQueue:
    def __init__(self, db: Database, worker_id=None):
        self.db = db
        self.worker_id = worker_id or "worker-" + uuid.uuid4().hex
        self.lease_seconds = 60
        self.heartbeat_interval = 15
        self.max_retries = 3  # three retries AFTER the first attempt
        self.base_backoff_seconds = 5
        self.claims = {}

    def enqueue(self, document, conn=None, *, max_pending=1000, per_camera=200):
        if conn is None:
            with self.db.transaction("IMMEDIATE") as tx:
                return self.enqueue(document, tx, max_pending=max_pending, per_camera=per_camera)
        camera = document.get("camera_id") or "unknown"
        existing = conn.execute(
            "SELECT document FROM jobs WHERE id=?", (document["job_id"],)
        ).fetchone()
        if existing:
            return json.loads(existing[0])
        if (
            conn.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            >= max_pending
        ):
            raise QueueFull("Global job queue is full")
        if (
            conn.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','running') AND camera_id=?",
                (camera,),
            ).fetchone()[0]
            >= per_camera
        ):
            raise QueueFull("Camera job queue is full")
        document = dict(document, task_type="ocr", status="queued")
        conn.execute(
            "INSERT INTO jobs(id,status,document,camera_id,priority) VALUES(?,?,?,?,?)",
            (
                document["job_id"],
                "queued",
                json.dumps(document),
                camera,
                20 if document.get("request_trigger") == "video_stream" else 10,
            ),
        )
        return document

    def claim_task(self, task_type=None):
        now = time.time()
        with self.db.transaction("IMMEDIATE") as conn:
            self._expire(conn, now)
            filters = (
                "j.status='queued' AND j.next_attempt_at<=? AND j.attempt_count<=j.max_retries"
            )
            args = [now]
            if task_type:
                filters += " AND coalesce(json_extract(j.document,'$.task_type'),'ocr')=?"
                args.append(task_type)
            row = conn.execute(
                f"""SELECT j.* FROM jobs j LEFT JOIN camera_schedule c ON j.camera_id=c.camera_id
                WHERE {filters} ORDER BY coalesce(c.last_claimed,0),j.priority,j.rowid LIMIT 1""",
                args,
            ).fetchone()
            if row is None:
                return None
            token = self.worker_id + ":" + uuid.uuid4().hex
            generation = row["lease_generation"] + 1
            doc = json.loads(row["document"])
            doc.update(
                job_id=row["id"],
                status="running",
                claimed_at=utc(now),
                lease_holder=token,
                lease_generation=generation,
                lease_expires_at=utc(now + self.lease_seconds),
                attempt_count=row["attempt_count"] + 1,
                retry_count=row["retry_count"],
            )
            conn.execute(
                """UPDATE jobs SET status='running',lease_holder=?,lease_generation=?,lease_expires_at=?,
                attempt_count=attempt_count+1,document=? WHERE id=?""",
                (token, generation, doc["lease_expires_at"], json.dumps(doc), row["id"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO camera_schedule VALUES(?,?)",
                (row["camera_id"] or "unknown", now),
            )
            self.claims[row["id"]] = token
            return doc

    def _expire(self, conn, now):
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status='running' AND (lease_expires_at IS NULL OR lease_expires_at<=?)",
            (utc(now),),
        ).fetchall()
        for row in rows:
            terminal = row["attempt_count"] > row["max_retries"]
            doc = json.loads(row["document"])
            history = doc.setdefault("failure_history", [])
            history.append(
                {"error": "lease_expired", "failed_at": utc(now), "attempt": row["attempt_count"]}
            )
            state = "failed" if terminal else "queued"
            doc.update(
                status=state,
                phase="dead_letter" if terminal else "lease_recovery",
                last_error="lease_expired",
            )
            if terminal:
                doc["finished_at"] = utc(now)
            conn.execute(
                """UPDATE jobs SET status=?,document=?,lease_holder=NULL,lease_expires_at=NULL,
                retry_count=?,next_attempt_at=? WHERE id=?""",
                (state, json.dumps(doc), max(0, row["attempt_count"]), now, row["id"]),
            )
        return len(rows)

    def recover_expired_leases(self):
        with self.db.transaction("IMMEDIATE") as conn:
            return self._expire(conn, time.time())

    def recover_interrupted(self):
        """Resume only photos explicitly marked interrupted by a clean shutdown."""
        with self.db.transaction("IMMEDIATE") as conn:
            rows = conn.execute("""SELECT * FROM jobs WHERE status='interrupted'
                AND lease_holder IS NULL AND json_extract(document,'$.resume_pending')=1
                AND coalesce(json_extract(document,'$.request_trigger'),'')!='video_stream'""").fetchall()
            for row in rows:
                doc = json.loads(row["document"])
                doc.update(
                    status="queued",
                    phase="restart_recovery",
                    resume_pending=False,
                    resumed_at=utc(),
                    attempt_count=max(0, row["attempt_count"] - 1),
                )
                for key in ("lease_holder", "lease_expires_at", "finished_at"):
                    doc.pop(key, None)
                conn.execute(
                    """UPDATE jobs SET status='queued',document=?,next_attempt_at=0,
                    attempt_count=max(0,attempt_count-1) WHERE id=?""",
                    (json.dumps(doc), row["id"]),
                )
            return len(rows)

    def _owned(self, conn, job_id, token=None):
        token = token or self.claims.get(job_id)
        row = conn.execute(
            "SELECT * FROM jobs WHERE id=? AND status='running' AND lease_holder=? AND lease_expires_at>?",
            (job_id, token, utc()),
        ).fetchone()
        if row is None:
            raise LeaseLost(job_id)
        return row

    def renew_lease(self, job_id, lease_holder=None):
        try:
            with self.db.transaction("IMMEDIATE") as conn:
                self._owned(conn, job_id, lease_holder)
                conn.execute(
                    "UPDATE jobs SET lease_expires_at=? WHERE id=?",
                    (utc(time.time() + self.lease_seconds), job_id),
                )
            return True
        except LeaseLost:
            return False

    def publish(self, document, *, refresh=None):
        job_id = document["job_id"]
        with self.db.transaction("IMMEDIATE") as conn:
            self._owned(conn, job_id, document.get("lease_holder"))
            state = document["status"]
            terminal = state in {"completed", "failed", "cancelled", "interrupted"}
            conn.execute(
                "UPDATE jobs SET status=?,document=? WHERE id=?",
                (state, json.dumps(document), job_id),
            )
            if terminal:
                conn.execute(
                    "UPDATE jobs SET lease_holder=NULL,lease_expires_at=NULL WHERE id=?", (job_id,)
                )
            if refresh:
                refresh(conn)
        if terminal:
            self.claims.pop(job_id, None)
        return document

    def complete_task(self, job_id, result):
        with self.db.connection() as conn:
            row = self._owned(conn, job_id)
            doc = json.loads(row["document"])
        doc.update(result, status="completed", finished_at=utc())
        return self.publish(doc)

    def fail_task(self, job_id, error, retry=True, *, document=None, refresh=None):
        with self.db.transaction("IMMEDIATE") as conn:
            row = self._owned(conn, job_id, document.get("lease_holder") if document else None)
            doc = json.loads(row["document"])
            if document:
                doc.update(document)
            now = time.time()
            doc.setdefault("failure_history", []).append(
                {"error": error, "failed_at": utc(now), "attempt": row["attempt_count"]}
            )
            again = retry and row["attempt_count"] <= row["max_retries"]
            state = "queued" if again else "failed"
            doc.update(
                status=state, last_error=error, phase="retry_wait" if again else "dead_letter"
            )
            if not again:
                doc["finished_at"] = utc(now)
            delay = min(
                300, self.base_backoff_seconds * 2 ** max(0, row["attempt_count"] - 1)
            ) * random.uniform(0.8, 1.2)
            conn.execute(
                """UPDATE jobs SET status=?,document=?,retry_count=?,next_attempt_at=?,
                lease_holder=NULL,lease_expires_at=NULL WHERE id=?""",
                (state, json.dumps(doc), row["attempt_count"], now + delay if again else 0, job_id),
            )
            if refresh:
                refresh(conn)
        self.claims.pop(job_id, None)

    def defer_task(self, job_id, *, document=None, delay=15, error="InferenceUnavailable"):
        """Dependency unavailability is a wait, not an exhausted recognition attempt."""
        with self.db.transaction("IMMEDIATE") as conn:
            row = self._owned(conn, job_id, document.get("lease_holder") if document else None)
            doc = json.loads(row["document"])
            if document:
                doc.update(document)
            doc.update(
                status="queued",
                phase="waiting_service",
                last_error=error,
                dependency_wait_count=doc.get("dependency_wait_count", 0) + 1,
                dependency_last_wait_at=utc(),
                attempt_count=max(0, row["attempt_count"] - 1),
            )
            for key in ("finished_at", "lease_holder", "lease_expires_at"):
                doc.pop(key, None)
            conn.execute(
                """UPDATE jobs SET status='queued',document=?,next_attempt_at=?,
                attempt_count=max(0,attempt_count-1),lease_holder=NULL,lease_expires_at=NULL WHERE id=?""",
                (json.dumps(doc), time.time() + delay, job_id),
            )
        self.claims.pop(job_id, None)

    def cancel_task(self, job_id):
        with self.db.transaction("IMMEDIATE") as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return
            if row["status"] == "running":
                self._owned(conn, job_id)
            elif row["status"] != "queued":
                return
            doc = json.loads(row["document"])
            doc.update(status="cancelled", finished_at=utc())
            conn.execute(
                "UPDATE jobs SET status='cancelled',document=?,lease_holder=NULL,lease_expires_at=NULL WHERE id=?",
                (json.dumps(doc), job_id),
            )

    def replay(self, job_id, actor, reason):
        if not actor or not reason.strip():
            raise ValueError("Replay requires an authorized actor and reason")
        with self.db.transaction("IMMEDIATE") as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id=? AND status='failed'", (job_id,)
            ).fetchone()
            if not row:
                raise ValueError("Only a failed task can be replayed")
            doc = json.loads(row["document"])
            if doc.get("measurement_id"):
                measurement = conn.execute(
                    "SELECT status FROM photo_measurements WHERE id=?", (doc["measurement_id"],)
                ).fetchone()
                if measurement and measurement[0] in {"confirmed", "rejected"}:
                    raise ValueError("A finalized measurement cannot be replayed")
            doc.setdefault("replays", []).append(
                {
                    "actor": actor,
                    "reason": reason,
                    "at": utc(),
                    "previous_document": {k: v for k, v in doc.items() if k != "replays"},
                }
            )
            doc.update(status="queued", phase="replayed")
            conn.execute(
                "UPDATE jobs SET status='queued',document=?,attempt_count=0,retry_count=0,next_attempt_at=0 WHERE id=?",
                (json.dumps(doc), job_id),
            )
            return doc

    def get_queue_stats(self):
        with self.db.connection() as conn:
            stats = {
                s: conn.execute("SELECT count(*) FROM jobs WHERE status=?", (s,)).fetchone()[0]
                for s in ("queued", "running", "completed", "failed", "cancelled", "interrupted")
            }
            stats["active_workers"] = conn.execute(
                "SELECT count(DISTINCT lease_holder) FROM jobs WHERE status='running' AND lease_expires_at>?",
                (utc(),),
            ).fetchone()[0]
            stats["retry_queue"] = conn.execute(
                "SELECT count(*) FROM jobs WHERE status='queued' AND retry_count>0"
            ).fetchone()[0]
        return stats

    def cleanup_completed_tasks(self, before, limit=1000):
        raise ValueError(
            "Business task receipts are retained; deletion requires a separate retention workflow"
        )
