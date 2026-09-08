from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class StorageReservationError(RuntimeError):
    def __init__(
        self,
        *,
        required_bytes: int,
        free_bytes: int,
        already_reserved_bytes: int,
    ) -> None:
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        self.already_reserved_bytes = int(already_reserved_bytes)
        self.available_bytes = max(0, self.free_bytes - self.already_reserved_bytes)
        self.missing_bytes = max(0, self.required_bytes - self.available_bytes)
        super().__init__(
            "storage reservation failed: "
            f"required={self.required_bytes}, available={self.available_bytes}, "
            f"missing={self.missing_bytes}"
        )


class UploadSessionStore:
    """Durable upload offsets and storage reservations for large browser uploads."""

    def __init__(self, database: Path):
        self.database = database.resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

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
                CREATE TABLE IF NOT EXISTS upload_sessions (
                    session_id TEXT PRIMARY KEY,
                    archive_name TEXT NOT NULL,
                    archive_root TEXT NOT NULL,
                    storage_key TEXT NOT NULL,
                    experiment_name TEXT NOT NULL,
                    view_specs_json TEXT NOT NULL,
                    retention_mode TEXT NOT NULL,
                    expected_source_bytes INTEGER NOT NULL,
                    processing_headroom_bytes INTEGER NOT NULL,
                    safety_headroom_bytes INTEGER NOT NULL,
                    reserved_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('open', 'finalized', 'released', 'cancelled', 'expired')
                    ),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    run_id TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_upload_session_archive
                    ON upload_sessions(archive_root, archive_name);
                CREATE INDEX IF NOT EXISTS idx_upload_session_reservations
                    ON upload_sessions(storage_key, status, expires_at);

                CREATE TABLE IF NOT EXISTS upload_files (
                    session_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('video', 'timestamp_csv', 'audio')),
                    file_index INTEGER NOT NULL,
                    view_id TEXT NOT NULL,
                    segment_ordinal INTEGER,
                    source_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    expected_bytes INTEGER NOT NULL,
                    uploaded_bytes INTEGER NOT NULL DEFAULT 0,
                    final_path TEXT NOT NULL,
                    partial_path TEXT NOT NULL,
                    sha256 TEXT,
                    content_hash TEXT,
                    content_hash_algorithm TEXT,
                    completed_at REAL,
                    PRIMARY KEY (session_id, file_id),
                    UNIQUE (session_id, kind, file_index),
                    FOREIGN KEY (session_id) REFERENCES upload_sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS upload_chunks (
                    session_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    chunk_offset INTEGER NOT NULL,
                    chunk_length INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, file_id, chunk_offset),
                    FOREIGN KEY (session_id, file_id)
                        REFERENCES upload_files(session_id, file_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_upload_chunks_file
                    ON upload_chunks(session_id, file_id, chunk_offset);
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(upload_files)")
            }
            if "content_hash" not in columns:
                connection.execute("ALTER TABLE upload_files ADD COLUMN content_hash TEXT")
            if "content_hash_algorithm" not in columns:
                connection.execute(
                    "ALTER TABLE upload_files ADD COLUMN content_hash_algorithm TEXT"
                )
            if "segment_ordinal" not in columns:
                connection.execute(
                    "ALTER TABLE upload_files ADD COLUMN segment_ordinal INTEGER"
                )
            connection.execute(
                """
                UPDATE upload_files
                SET content_hash = sha256, content_hash_algorithm = 'sha256'
                WHERE sha256 IS NOT NULL AND content_hash IS NULL
                """
            )

        self._allow_audio_files()

    def _allow_audio_files(self) -> None:
        """Upgrade the kind constraint without losing active uploads or chunks."""
        connection = self._connect()
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("BEGIN IMMEDIATE")
            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='upload_files'"
            ).fetchone()[0]
            if "'audio'" not in schema:
                new_schema = schema.replace("upload_files", "upload_files_audio", 1).replace(
                    "'video', 'timestamp_csv'", "'video', 'timestamp_csv', 'audio'"
                )
                if new_schema == schema or "'audio'" not in new_schema:
                    raise RuntimeError("unsupported upload_files schema")
                connection.execute(new_schema)
                columns = ", ".join('"' + row[1].replace('"', '""') + '"' for row in
                                    connection.execute("PRAGMA table_info(upload_files)"))
                connection.execute(f"INSERT INTO upload_files_audio ({columns}) SELECT {columns} FROM upload_files")
                connection.execute("DROP TABLE upload_files")
                connection.execute("ALTER TABLE upload_files_audio RENAME TO upload_files")
                if connection.execute("PRAGMA foreign_key_check").fetchone():
                    raise RuntimeError("audio upload migration failed foreign key check")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _json(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    def reserve(
        self,
        *,
        session_id: str,
        archive_name: str,
        archive_root: Path,
        storage_key: str,
        experiment_name: str,
        view_specs: list[dict[str, Any]],
        retention_mode: str,
        expected_source_bytes: int,
        processing_headroom_bytes: int,
        safety_headroom_bytes: int,
        free_bytes: int,
        files: list[dict[str, Any]],
        expires_at: float,
        now: float | None = None,
    ) -> dict[str, int]:
        observed_at = time.time() if now is None else float(now)
        required_bytes = (
            int(expected_source_bytes)
            + int(processing_headroom_bytes)
            + int(safety_headroom_bytes)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN s.reserved_bytes > COALESCE(f.uploaded_bytes, 0)
                        THEN s.reserved_bytes - COALESCE(f.uploaded_bytes, 0)
                        ELSE 0
                    END
                ), 0) AS outstanding_bytes
                FROM upload_sessions AS s
                LEFT JOIN (
                    SELECT session_id, SUM(uploaded_bytes) AS uploaded_bytes
                    FROM upload_files
                    GROUP BY session_id
                ) AS f ON f.session_id = s.session_id
                WHERE s.storage_key = ?
                  AND s.status IN ('open', 'finalized')
                """,
                (storage_key,),
            ).fetchone()
            already_reserved = int(row["outstanding_bytes"] if row else 0)
            if required_bytes > max(0, int(free_bytes) - already_reserved):
                connection.rollback()
                raise StorageReservationError(
                    required_bytes=required_bytes,
                    free_bytes=free_bytes,
                    already_reserved_bytes=already_reserved,
                )
            connection.execute(
                """
                INSERT INTO upload_sessions(
                    session_id, archive_name, archive_root, storage_key,
                    experiment_name, view_specs_json, retention_mode,
                    expected_source_bytes, processing_headroom_bytes,
                    safety_headroom_bytes, reserved_bytes, status,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
                """,
                (
                    session_id,
                    archive_name,
                    str(archive_root),
                    storage_key,
                    experiment_name,
                    self._json(view_specs),
                    retention_mode,
                    int(expected_source_bytes),
                    int(processing_headroom_bytes),
                    int(safety_headroom_bytes),
                    required_bytes,
                    observed_at,
                    observed_at,
                    float(expires_at),
                ),
            )
            connection.executemany(
                """
                INSERT INTO upload_files(
                    session_id, file_id, kind, file_index, view_id, segment_ordinal,
                    source_name, stored_name, expected_bytes,
                    final_path, partial_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        item["file_id"],
                        item["kind"],
                        int(item["file_index"]),
                        item["view_id"],
                        item.get("segment_ordinal"),
                        item["source_name"],
                        item["stored_name"],
                        int(item["expected_bytes"]),
                        str(item["final_path"]),
                        str(item["partial_path"]),
                    )
                    for item in files
                ],
            )
            connection.commit()
        return {
            "free_bytes": int(free_bytes),
            "already_reserved_bytes": already_reserved,
            "available_before_bytes": max(0, int(free_bytes) - already_reserved),
            "reserved_bytes": required_bytes,
            "available_after_bytes": max(
                0, int(free_bytes) - already_reserved - required_bytes
            ),
        }

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            session = connection.execute(
                "SELECT * FROM upload_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return None
            files = connection.execute(
                """
                SELECT * FROM upload_files
                WHERE session_id = ?
                ORDER BY CASE kind WHEN 'video' THEN 0 ELSE 1 END, file_index
                """,
                (session_id,),
            ).fetchall()
        payload = dict(session)
        payload["view_specs"] = json.loads(payload.pop("view_specs_json"))
        payload["files"] = [dict(item) for item in files]
        return payload

    def update_progress(
        self,
        session_id: str,
        file_id: str,
        uploaded_bytes: int,
        *,
        expires_at: float,
        now: float | None = None,
    ) -> None:
        observed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE upload_files
                SET uploaded_bytes = ?
                WHERE session_id = ? AND file_id = ?
                  AND expected_bytes >= ?
                """,
                (int(uploaded_bytes), session_id, file_id, int(uploaded_bytes)),
            ).rowcount
            if updated != 1:
                raise ValueError("upload progress exceeds the declared file size")
            connection.execute(
                """
                DELETE FROM upload_chunks
                WHERE session_id = ? AND file_id = ?
                  AND chunk_offset + chunk_length > ?
                """,
                (session_id, file_id, int(uploaded_bytes)),
            )
            connection.execute(
                """
                UPDATE upload_sessions
                SET updated_at = ?, expires_at = ?
                WHERE session_id = ? AND status = 'open'
                """,
                (observed_at, float(expires_at), session_id),
            )

    def record_chunk(
        self,
        session_id: str,
        file_id: str,
        chunk_offset: int,
        chunk_length: int,
        sha256: str,
        *,
        now: float | None = None,
    ) -> None:
        observed_at = time.time() if now is None else float(now)
        if chunk_offset < 0 or chunk_length <= 0:
            raise ValueError("invalid upload chunk range")
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT chunk_length, sha256 FROM upload_chunks
                WHERE session_id = ? AND file_id = ? AND chunk_offset = ?
                """,
                (session_id, file_id, int(chunk_offset)),
            ).fetchone()
            if existing is not None:
                if int(existing["chunk_length"]) != int(chunk_length) or str(
                    existing["sha256"]
                ) != str(sha256):
                    raise ValueError("upload chunk ledger conflicts with retained bytes")
                return
            connection.execute(
                """
                INSERT INTO upload_chunks(
                    session_id, file_id, chunk_offset, chunk_length, sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    file_id,
                    int(chunk_offset),
                    int(chunk_length),
                    str(sha256),
                    observed_at,
                ),
            )

    def chunks(self, session_id: str, file_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_offset, chunk_length, sha256
                FROM upload_chunks
                WHERE session_id = ? AND file_id = ?
                ORDER BY chunk_offset
                """,
                (session_id, file_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def chunk_tree_identity(
        self, session_id: str, file_id: str, expected_bytes: int
    ) -> str | None:
        chunks = self.chunks(session_id, file_id)
        offset = 0
        digest = hashlib.sha256(b"visioncortex-upload-chunk-tree-v1\n")
        for chunk in chunks:
            chunk_offset = int(chunk["chunk_offset"])
            chunk_length = int(chunk["chunk_length"])
            chunk_sha256 = str(chunk["sha256"])
            if chunk_offset != offset or len(chunk_sha256) != 64:
                return None
            digest.update(f"{chunk_offset}:{chunk_length}:{chunk_sha256}\n".encode("ascii"))
            offset += chunk_length
        return digest.hexdigest() if offset == int(expected_bytes) else None

    def complete_file(
        self,
        session_id: str,
        file_id: str,
        sha256: str | None,
        *,
        content_hash: str,
        content_hash_algorithm: str,
        now: float | None = None,
    ) -> None:
        observed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE upload_files
                SET uploaded_bytes = expected_bytes, sha256 = ?, content_hash = ?,
                    content_hash_algorithm = ?, completed_at = ?
                WHERE session_id = ? AND file_id = ?
                  AND uploaded_bytes = expected_bytes
                """,
                (
                    sha256,
                    content_hash,
                    content_hash_algorithm,
                    observed_at,
                    session_id,
                    file_id,
                ),
            ).rowcount
            if updated != 1:
                raise ValueError("file is not fully uploaded")
            connection.execute(
                """
                UPDATE upload_sessions SET updated_at = ? WHERE session_id = ?
                """,
                (observed_at, session_id),
            )

    def assign_run(self, session_id: str, run_id: str, *, now: float | None = None) -> None:
        observed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE upload_sessions
                SET run_id = ?, updated_at = ?
                WHERE session_id = ? AND status = 'open'
                  AND (run_id IS NULL OR run_id = ?)
                """,
                (run_id, observed_at, session_id, run_id),
            ).rowcount
        if updated != 1:
            raise ValueError("upload session cannot be assigned to this run")

    def finalize(self, session_id: str, run_id: str, *, now: float | None = None) -> None:
        observed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            incomplete = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM upload_files
                WHERE session_id = ?
                  AND (uploaded_bytes != expected_bytes OR content_hash IS NULL)
                """,
                (session_id,),
            ).fetchone()
            if incomplete is None or int(incomplete["count"]) != 0:
                connection.rollback()
                raise ValueError("upload session still contains incomplete files")
            updated = connection.execute(
                """
                UPDATE upload_sessions
                SET status = 'finalized', run_id = ?, updated_at = ?
                WHERE session_id = ? AND status = 'open'
                """,
                (run_id, observed_at, session_id),
            ).rowcount
            if updated != 1:
                connection.rollback()
                raise ValueError("upload session is not open")
            connection.commit()

    def release(self, session_id: str, *, now: float | None = None) -> bool:
        observed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE upload_sessions
                SET status = 'released', updated_at = ?
                WHERE session_id = ? AND status IN ('open', 'finalized')
                """,
                (observed_at, session_id),
            ).rowcount
        return updated == 1

    def cancel(self, session_id: str, *, now: float | None = None) -> bool:
        observed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE upload_sessions
                SET status = 'cancelled', updated_at = ?
                WHERE session_id = ? AND status = 'open' AND run_id IS NULL
                """,
                (observed_at, session_id),
            ).rowcount
        return updated == 1

    def expire_stale(self, *, now: float | None = None) -> list[dict[str, str]]:
        observed_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT session_id, archive_root
                FROM upload_sessions
                WHERE status = 'open' AND expires_at <= ?
                """,
                (observed_at,),
            ).fetchall()
            if rows:
                connection.executemany(
                    """
                    UPDATE upload_sessions
                    SET status = 'expired', updated_at = ?
                    WHERE session_id = ? AND status = 'open' AND expires_at <= ?
                    """,
                    [
                        (observed_at, str(row["session_id"]), observed_at)
                        for row in rows
                    ],
                )
            connection.commit()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM upload_sessions GROUP BY status"
            ).fetchall()
        counts = {
            "open": 0,
            "finalized": 0,
            "released": 0,
            "cancelled": 0,
            "expired": 0,
        }
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return counts
