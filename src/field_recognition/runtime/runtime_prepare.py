"""Validate and rehearse a database upgrade before managed writers start."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
from field_recognition.runtime.paths import PROJECT_ROOT, SOURCE_MODULES
import sqlite3
import subprocess
import tempfile

from field_recognition.runtime.migrate_db import backup_database, inspect_database, migrate_database
from field_recognition.runtime.worker_support import process_lock

WRITERS = tuple(
    "field-recognition-" + name + ".service" for name in ("api", "ocr", "cameras", "archive")
)


def check_imports():
    # Resolve the actual readout pipeline before any durable photo is claimed.
    for name in (
        "panel_readout",
        "reading_results",
        "measurement_records",
        "result_reprocessing",
        "saved_photo",
        "remote_ocr",
    ):
        importlib.import_module(SOURCE_MODULES[name])


def ensure_stopped(run=subprocess.run):
    for unit in WRITERS:
        response = run(
            ["systemctl", "--user", "show", unit, "-p", "MainPID", "-p", "ControlPID"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)
        if not values or any(int(value) for value in values.values()):
            raise RuntimeError("Stop all managed writers before migration: " + unit)


def history(path):
    """Hash immutable business documents without returning people or credentials."""
    import hashlib

    result = {}
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("bindings", "jobs", "scans", "scene_visits", "experiment_records"):
            if table not in tables:
                continue
            columns = {r[1] for r in conn.execute("PRAGMA table_info(" + table + ")")}
            if not {"id", "document"} <= columns:
                continue
            digest = hashlib.sha256()
            count = 0
            for row in conn.execute("SELECT id,document FROM " + table + " ORDER BY id"):
                digest.update(json.dumps(row, ensure_ascii=False).encode())
                digest.update(b"\n")
                count += 1
            result[table] = {"count": count, "sha256": digest.hexdigest()}
    return result


def prepare(path, *, assert_stopped=ensure_stopped):
    path = Path(path).resolve()
    before = inspect_database(path)
    if before["errors"]:
        raise ValueError("Database preflight failed: " + ", ".join(before["errors"]))
    check_imports()
    if not before["status"]["pending"]:
        return {"status": "ready", "schema": before["status"]["current_version"], "applied": []}
    assert_stopped()
    runtime = path.parent / "Runtime"
    with process_lock(path.parent, "schema-upgrade"):
        original = history(path)
        with tempfile.TemporaryDirectory(prefix="MigrationRehearsal-", dir=runtime) as temporary:
            trial = Path(temporary) / "Trial.sqlite3"
            backup_database(path, trial)
            rehearsal = migrate_database(trial)
            if history(trial) != original:
                raise ValueError("Rehearsal changed historical business documents")
            restored = Path(temporary) / "Restored.sqlite3"
            backup_database(rehearsal["backup"], restored)
            if history(restored) != original:
                raise ValueError("Rehearsal restore did not preserve history")
        assert_stopped()
        result = migrate_database(path)
        if history(path) != original:
            raise ValueError(
                "History verification failed; preserve the backup and keep services stopped"
            )
        result.update(
            status="ready", rehearsal="passed", restore_rehearsal="passed", history_preserved=True
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        receipt = runtime / ("Migration-" + stamp + ".json")
        with open(receipt, "x", opener=lambda p, f: os.open(p, f, 0o600)) as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        return result | {"receipt": str(receipt)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = Path(os.environ.get("FIELD_DEMO_DATA", PROJECT_ROOT / "Data"))
    path = Path(os.environ.get("FIELD_DATABASE_PATH") or data / "Demo.sqlite3")
    if args.check:
        check_imports()
        report = inspect_database(path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(bool(report["errors"] or report["status"]["pending"]))
    print(json.dumps(prepare(path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
