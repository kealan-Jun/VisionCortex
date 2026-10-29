from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


class Database:
    def __init__(self, url: str):
        self.url = url

    @contextmanager
    def transaction(self, *, readonly: bool = False):
        with psycopg.connect(self.url, row_factory=dict_row, connect_timeout=5) as conn:
            with conn.transaction():
                conn.execute("SET LOCAL statement_timeout = '10s'")
                conn.execute("SET LOCAL lock_timeout = '5s'")
                if readonly:
                    conn.execute("SET TRANSACTION READ ONLY")
                yield conn

    def migrate(self, directory: Path) -> list[str]:
        applied = []
        with self.transaction() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(729420261023)")
            conn.execute("CREATE TABLE IF NOT EXISTS vc_lab_migrations(version text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())")
            for path in sorted(directory.glob("[0-9]*.sql")):
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                row = conn.execute("SELECT sha256 FROM vc_lab_migrations WHERE version=%s", (path.name,)).fetchone()
                if row:
                    if row["sha256"] != digest:
                        raise ValueError("Previously applied migration has changed: " + path.name)
                    continue
                conn.execute(raw.decode())
                conn.execute("INSERT INTO vc_lab_migrations(version,sha256) VALUES(%s,%s)", (path.name, digest))
                applied.append(path.name)
        return applied
