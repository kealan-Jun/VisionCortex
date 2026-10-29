#!/usr/bin/env python3
"""Inspect, back up and migrate an explicitly selected local SQLite database."""

import argparse
import json
import os
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from field_recognition.runtime.database import (
    Database,
    MIGRATIONS,
    apply_migrations,
    get_migration_status,
)


def _read_only(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def inspect_database(path):
    """Read-only checks; never create a missing database or its parent directory."""
    path = Path(path)
    result = {
        "database": str(path.resolve()),
        "exists": path.is_file(),
        "integrity_check": [],
        "foreign_key_check": [],
        "counts": {},
        "multiple_member_cameras": 0,
        "unknown_migrations": [],
        "errors": [],
    }
    if not result["exists"]:
        result["errors"].append("database_missing")
        result["status"] = get_migration_status(Database(path))
        return result
    try:
        with closing(_read_only(path)) as conn:
            result["integrity_check"] = [row[0] for row in conn.execute("PRAGMA integrity_check")]
            result["foreign_key_check"] = [
                list(row) for row in conn.execute("PRAGMA foreign_key_check")
            ]
            tables = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            for table in (
                "users",
                "camera_users",
                "camera_active_users",
                "bindings",
                "jobs",
                "experiment_records",
                "archive_outbox",
                "binding_session_audit",
                "binding_change_requests",
            ):
                if table in tables:
                    result["counts"][table] = conn.execute(
                        "SELECT count(*) FROM " + table
                    ).fetchone()[0]
            if "camera_users" in tables:
                result["multiple_member_cameras"] = conn.execute(
                    "SELECT count(*) FROM (SELECT camera_id FROM camera_users GROUP BY camera_id HAVING count(*)>1)"
                ).fetchone()[0]
            if "schema_migrations" in tables:
                known = {migration["version"] for migration in MIGRATIONS}
                result["unknown_migrations"] = sorted(
                    row[0]
                    for row in conn.execute("SELECT version FROM schema_migrations")
                    if row[0] not in known
                )
        if result["integrity_check"] != ["ok"]:
            result["errors"].append("integrity_check_failed")
        if result["foreign_key_check"]:
            result["errors"].append("foreign_key_check_failed")
        if result["unknown_migrations"]:
            result["errors"].append("unknown_schema_version")
        result["status"] = get_migration_status(Database(path))
    except sqlite3.Error as exc:
        result["errors"].append("sqlite_error: " + str(exc))
    return result


def _require_healthy(report):
    if report["errors"]:
        raise ValueError("Database check failed: " + ", ".join(report["errors"]))


def backup_database(source, destination):
    """SQLite backup API includes committed WAL contents; never overwrite a file."""
    source, destination = Path(source), Path(destination)
    _require_healthy(inspect_database(source))
    if source.resolve() == destination.resolve():
        raise ValueError("Backup destination must differ from the source database")
    # O_EXCL prevents accidentally overwriting a previous backup, including symlinks.
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with (
            closing(_read_only(source)) as original,
            closing(sqlite3.connect(destination)) as backup,
        ):
            original.backup(backup)
        _require_healthy(inspect_database(destination))
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def migrate_database(path, backup_path=None):
    """The caller must quiesce all writers; no live service is stopped here."""
    path = Path(path)
    before = inspect_database(path)
    _require_healthy(before)
    if not before["status"]["pending"]:
        return {"before": before, "backup": None, "applied": [], "after": before}
    if backup_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup_path = path.with_name(f"{path.name}.before-{stamp}-{uuid.uuid4().hex[:8]}.sqlite3")
    backup = backup_database(path, backup_path)
    # Print the verified backup location before mutation, so a failed migration is recoverable.
    print(f"Backup verified: {backup.resolve()}", file=sys.stderr)
    applied = apply_migrations(Database(path))
    after = inspect_database(path)
    _require_healthy(after)
    return {"before": before, "backup": str(backup.resolve()), "applied": applied, "after": after}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", help="Explicit SQLite path; read-only commands default to Data/Demo.sqlite3"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run", action="store_true", help="Read-only preflight and pending migrations"
    )
    modes.add_argument("--status", action="store_true", help="Read-only migration status")
    modes.add_argument(
        "--check", action="store_true", help="Read-only integrity, foreign-key and schema checks"
    )
    modes.add_argument(
        "--restore-from", metavar="BACKUP", help="Restore a verified backup to a NEW --db path"
    )
    parser.add_argument("--backup", help="New backup file path (default: unique sibling of --db)")
    args = parser.parse_args(argv)
    read_only = args.status or args.dry_run or args.check
    if not read_only and not args.db:
        parser.error(
            "Writes require an explicit --db path; use --status for default-path inspection"
        )
    if args.backup and (read_only or args.restore_from):
        parser.error("--backup applies only to a migration")
    path = Path(args.db or "Data/Demo.sqlite3")
    try:
        if read_only:
            report = inspect_database(path)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            # Missing-file status remains a read-only way to see pending migrations.
            errors = [
                error
                for error in report["errors"]
                if not (args.status and error == "database_missing")
            ]
            return 1 if errors else 0
        if args.restore_from:
            restored = backup_database(args.restore_from, path)
            print(
                json.dumps(
                    {"restored": str(restored.resolve()), "check": inspect_database(restored)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(migrate_database(path, args.backup), ensure_ascii=False, indent=2))
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Migration/restore refused or failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
