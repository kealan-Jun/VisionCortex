from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueuedRunJob:
    run_id: str
    kind: str
    payload: dict[str, Any]
    sequence: int
    attempts: int
    reclaimed: bool


class DurableRunQueue:
    """SQLite-backed, cross-process queue for the single production GPU."""

    def __init__(self, database: Path):
        self.database = database.resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    def _connect(self):
        from .sqlite_store import connection
        return connection(self.database)

    def _initialize(self) -> None:
        with self._connect() as connection:
            from .task_events import initialize
            initialize(connection)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS run_records (
                    run_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed')
                    ),
                    queued_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    finished_at REAL,
                    last_error TEXT,
                    FOREIGN KEY (run_id) REFERENCES run_records(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_run_jobs_claim
                    ON run_jobs(status, lease_expires_at, sequence);
                """
            )

    def save_run(self, run_id: str, state: dict[str, Any], *, now: float | None = None) -> None:
        observed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_records(run_id, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (run_id, self._json(state), observed_at, observed_at),
            )

    def load_runs(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, state_json FROM run_records ORDER BY created_at"
            ).fetchall()
        return {str(row["run_id"]): json.loads(row["state_json"]) for row in rows}

    def patch_run(self, run_id, values):
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state_json FROM run_records WHERE run_id=?', (run_id,)).fetchone()
            state = (json.loads(row[0]) if row else {}) | values
            timestamp = time.time()
            db.execute("INSERT INTO run_records VALUES(?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at",
                       (run_id, self._json(state), timestamp, timestamp))
            if {k: values[k] for k in ('state', 'stage', 'message') if k in values}:
                from .task_events import append
                append(db, run_id, state.get('state', 'progress'), data={k: state[k] for k in ('state','stage','message','progress','archive_url') if k in state})
        return state

    def load_run(self, run_id):
        with self._connect() as db:
            row = db.execute('SELECT state_json FROM run_records WHERE run_id=?', (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def enqueue(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        now: float | None = None,
    ) -> int:
        queued_at = time.time() if now is None else float(now)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO run_records(
                        run_id, state_json, created_at, updated_at
                    ) VALUES (?, '{}', ?, ?)
                    """,
                    (run_id, queued_at, queued_at),
                )
                cursor = connection.execute(
                    """
                    INSERT INTO run_jobs(
                        run_id, kind, payload_json, status, queued_at, updated_at
                    ) VALUES (?, ?, ?, 'queued', ?, ?)
                    """,
                    (run_id, kind, self._json(payload), queued_at, queued_at),
                )
                from .task_events import append
                append(connection, run_id, 'queued', data={'kind': kind})
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"run_id already queued: {run_id}") from exc
        from .submission import bind_task
        bind_task(run_id)
        return int(cursor.lastrowid)

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
        now: float | None = None,
    ) -> QueuedRunJob | None:
        claimed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1
                FROM run_jobs
                WHERE status = 'running' AND lease_expires_at > ?
                LIMIT 1
                """,
                (claimed_at,),
            ).fetchone()
            if active is not None:
                connection.rollback()
                return None
            row = connection.execute(
                """
                SELECT sequence, run_id, kind, payload_json, status, attempts
                FROM run_jobs
                WHERE status = 'queued'
                   OR (status = 'running' AND COALESCE(lease_expires_at, 0) <= ?)
                ORDER BY sequence
                LIMIT 1
                """,
                (claimed_at,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            updated = connection.execute(
                """
                UPDATE run_jobs
                SET status = 'running', lease_owner = ?, lease_expires_at = ?,
                    attempts = attempts + 1, updated_at = ?, last_error = NULL
                WHERE sequence = ?
                  AND (
                      status = 'queued'
                      OR (status = 'running' AND COALESCE(lease_expires_at, 0) <= ?)
                  )
                """,
                (
                    worker_id,
                    claimed_at + lease_seconds,
                    claimed_at,
                    int(row["sequence"]),
                    claimed_at,
                ),
            ).rowcount
            if updated != 1:
                connection.rollback()
                return None
            from .task_events import append
            append(connection, str(row['run_id']), 'running', attempt=int(row['attempts'])+1,
                   data={'reclaimed': str(row['status']) == 'running'})
            connection.commit()
        return QueuedRunJob(
            run_id=str(row["run_id"]),
            kind=str(row["kind"]),
            payload=json.loads(row["payload_json"]),
            sequence=int(row["sequence"]),
            attempts=int(row["attempts"]) + 1,
            reclaimed=str(row["status"]) == "running",
        )

    def renew_lease(
        self,
        run_id: str,
        worker_id: str,
        *,
        lease_seconds: float,
        now: float | None = None,
    ) -> bool:
        renewed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE run_jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE run_id = ? AND status = 'running' AND lease_owner = ?
                """,
                (renewed_at + lease_seconds, renewed_at, run_id, worker_id),
            ).rowcount
        return updated == 1

    def finish(
        self,
        run_id: str,
        worker_id: str,
        status: str,
        *,
        error: str | None = None,
        now: float | None = None,
    ) -> bool:
        if status not in {"completed", "partial", "failed"}:
            raise ValueError(f"Unsupported terminal queue status: {status}")
        finished_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE run_jobs
                SET status = ?, updated_at = ?, finished_at = ?, last_error = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE run_id = ? AND status = 'running' AND lease_owner = ?
                """,
                # Queue completion records execution, while run_records retains
                # the partial evidence outcome. Existing databases need no migration.
                ("completed" if status == "partial" else status,
                 finished_at, finished_at, error, run_id, worker_id),
            ).rowcount
            if updated:
                from .task_events import append
                row = connection.execute('SELECT attempts FROM run_jobs WHERE run_id=?', (run_id,)).fetchone()
                saved = connection.execute('SELECT state_json FROM run_records WHERE run_id=?', (run_id,)).fetchone()
                archive = json.loads(saved[0]).get('nas_output') if saved else None
                append(connection, run_id, status, attempt=row['attempts'], data={
                    'result_url': f'/api/runs/{run_id}', 'archive': Path(archive).name if archive else None})
        return updated == 1

    def get_job(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT sequence, run_id, kind, payload_json, status, queued_at,
                       updated_at, lease_owner, lease_expires_at, attempts,
                       finished_at, last_error
                FROM run_jobs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["payload"] = json.loads(payload.pop("payload_json"))
        return payload

    def release_for_shutdown(self, run_id, worker_id):
        """Return only our cancelled lease; keep its sealed payload/checkpoints."""
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            now = time.time()
            changed = db.execute("""UPDATE run_jobs SET status='queued',updated_at=?,
                lease_owner=NULL,lease_expires_at=NULL,attempts=MAX(0,attempts-1)
                WHERE run_id=? AND status='running' AND lease_owner=?""", (now, run_id, worker_id)).rowcount
            if changed:
                row = db.execute('SELECT state_json FROM run_records WHERE run_id=?', (run_id,)).fetchone()
                state = json.loads(row[0]) | {'state': 'queued', 'error': None,
                    'message': '后台停止，已保留处理断点，重启后继续', 'recovered_from_durable_queue': True}
                db.execute('UPDATE run_records SET state_json=?,updated_at=? WHERE run_id=?',
                           (self._json(state), now, run_id))
            return bool(changed)

    def retry_failed(self, run_id: str, *, verified_mllm: dict | None = None, resume_stages: bool = False) -> dict[str, Any]:
        """Requeue the saved job atomically, preserving inputs and attempt history."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT j.*, r.state_json FROM run_jobs j JOIN run_records r "
                "USING (run_id) WHERE j.run_id = ?", (run_id,),
            ).fetchone()
            if row is None or row["status"] not in {"failed", "completed"}:
                raise ValueError("Only a finished failed job can be retried")
            previous = json.loads(row["state_json"])
            if row["status"] == "completed" and previous.get("state") != "partial":
                raise ValueError("A completed formal result cannot be retried")
            if previous.get("state") not in {"failed", "interrupted", "partial"}:
                raise ValueError("Run state does not permit retry")
            payload = json.loads(row["payload_json"])
            settings = payload["settings"]
            previous_credential = (settings.get("mllm") or {}).get("credential_ref")
            if verified_mllm is not None:
                settings["mllm"] = verified_mllm
            project = settings.setdefault("project", {})
            project["resume_stages"] = resume_stages
            project["cache_mode"] = "reuse"
            project["semantic_cache_mode"] = "reuse"
            project["semantic_recovery_attempt"] = f"{run_id}:{int(row['attempts']) + 1}"
            history = list(previous.get("attempt_history") or [])
            history.append({
                "attempt": row["attempts"],
                "finished_at": row["finished_at"],
                "state": previous.get("state"),
                "error": previous.get("error"),
                "retry_requested_at": now,
                "previous_credential_ref": previous_credential,
            })
            state = {**previous, "state": "queued", "progress": 0.0,
                     "error": None, "attempt_history": history,
                     "message": "已保留阶段产出，等待校验恢复点并继续未完成环节" if resume_stages else "已保留原输入与阶段产出，等待复跑并校验可复用缓存"}
            # Move to the tail so a retry cannot jump ahead of waiting users.
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_jobs"
            ).fetchone()[0]
            connection.execute(
                "UPDATE run_jobs SET sequence = ?, payload_json = ?, status = 'queued', "
                "queued_at = ?, updated_at = ?, finished_at = NULL, last_error = NULL, "
                "lease_owner = NULL, lease_expires_at = NULL WHERE run_id = ?",
                (sequence, self._json(payload), now, now, run_id),
            )
            connection.execute(
                "UPDATE run_records SET state_json = ?, updated_at = ? WHERE run_id = ?",
                (self._json(state), now, run_id),
            )
        return state

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM run_jobs GROUP BY status"
            ).fetchall()
        counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return counts
