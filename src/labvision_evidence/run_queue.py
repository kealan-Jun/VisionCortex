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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
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
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"run_id already queued: {run_id}") from exc
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
        if status not in {"completed", "failed"}:
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
                (status, finished_at, finished_at, error, run_id, worker_id),
            ).rowcount
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

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM run_jobs GROUP BY status"
            ).fetchall()
        counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return counts
