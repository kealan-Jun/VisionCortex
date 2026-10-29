"""Daily read-only verification of acknowledged receipts and their image objects."""

import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta


class CheckStopped(Exception):
    pass


class IntegrityError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def issue_code(exc):
    return (
        exc.code
        if isinstance(exc, IntegrityError)
        else "missing"
        if isinstance(exc, FileNotFoundError)
        else "invalid_or_unreadable"
    )


def relative_file(root, relative):
    from field_recognition.archive.archive_paths import resolve_file

    return resolve_file(root, relative)


class ArchiveIntegrity:
    def __init__(self, archive):
        self.archive = archive
        self.core = archive.core
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.run_lock = threading.Lock()
        self.worker = None
        self.running = False
        with self.core["db"]() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS archive_integrity_runs(
                id TEXT PRIMARY KEY, document TEXT NOT NULL, published_path TEXT)""")

    def interval(self):
        config = self.core.get("config")
        if config is not None:
            return config.archive_check_seconds
        try:
            return max(3600, int(os.environ.get("FIELD_ARCHIVE_CHECK_SECONDS", "86400")))
        except ValueError:
            return 86400

    def latest(self):
        with self.core["db"]() as conn:
            row = conn.execute(
                "SELECT * FROM archive_integrity_runs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        return (json.loads(row["document"]), row["published_path"]) if row else (None, None)

    def snapshot(self):
        report, path = self.latest()
        retry = (
            900
            if report and report["status"] in {"unavailable", "interrupted"}
            else self.interval()
        )
        due = (
            (datetime.fromisoformat(report["finished_at"]) + timedelta(seconds=retry)).isoformat()
            if report
            else None
        )
        return {
            "status": "running" if self.running else report["status"] if report else "not_checked",
            "interval_seconds": self.interval(),
            "next_check_at": due,
            "last_report": {
                k: report[k]
                for k in (
                    "run_id",
                    "status",
                    "finished_at",
                    "expected_receipts",
                    "verified_receipts",
                    "verified_objects",
                    "issue_count",
                    "bytes_checked",
                )
            }
            if report
            else None,
            "issues": report["issues"][:50] if report else [],
            "report_path": path,
            "report_pending": bool(report and not path),
        }

    def due(self):
        due = self.snapshot()["next_check_at"]
        return due is None or datetime.now(timezone.utc) >= datetime.fromisoformat(due)

    def digest(self, path, *, keep=False):
        # Pin an open descriptor while the single writer can move paths. Hashing
        # a whole historical snapshot must not hold the writer's migration gate.
        with self.lock:
            resolved = path() if callable(path) else path
            source = resolved.open("rb")
            before = os.fstat(source.fileno())
        sha, pieces, count = hashlib.sha256(), [], 0
        with source:
            if keep and before.st_size > 16 * 1024 * 1024:
                raise ValueError("Receipt exceeds read limit")
            while True:
                if self.stop.is_set():
                    raise CheckStopped()
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
                count += len(chunk)
                if keep:
                    pieces.append(chunk)
                # Yield between chunks; this worker never uses the OCR queue.
                if self.stop.wait(0.005):
                    raise CheckStopped()
            after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise IntegrityError("changed_during_check")
        return sha.hexdigest(), count, b"".join(pieces)

    def stable_digest(self, locate, *, keep=False):
        # CIFS can refresh cached metadata during a rename. Reopen the current
        # alias once; callers still compare the bytes against immutable hashes.
        for attempt in range(2):
            try:
                return self.digest(locate, keep=keep)
            except IntegrityError as exc:
                if exc.code != "changed_during_check" or attempt:
                    raise

    def run_once(self):
        if not self.archive.enabled() or not self.run_lock.acquire(blocking=False):
            return None
        self.running = True
        started = time.monotonic()
        report = {
            "schema": "field-recognition-integrity/1",
            "run_id": str(uuid.uuid4()),
            "source_instance": self.archive.instance,
            "started_at": self.core["now"](),
            "status": "completed",
            "expected_receipts": 0,
            "verified_receipts": 0,
            "expected_objects": 0,
            "verified_objects": 0,
            "bytes_checked": 0,
            "issues": [],
            "scope": "Acknowledged receipt snapshot and referenced objects; new writes are checked on the next run.",
            "repair_performed": False,
        }
        try:
            root = self.archive._root(create=False)
            from field_recognition.archive.archive_paths import load, checked, resolve_name

            with self.lock:
                aliases = load(root)["aliases"]

            def locate(relative):
                nonlocal aliases
                path = checked(root, resolve_name(root, relative, aliases))
                if not path.is_file():
                    # A move may have completed since this snapshot loaded its
                    # aliases. Refresh only on a missing old location.
                    aliases = load(root)["aliases"]
                    path = checked(root, resolve_name(root, relative, aliases))
                return path

            with self.core["db"]() as conn:
                rows = conn.execute(
                    "SELECT * FROM archive_outbox WHERE archived_at IS NOT NULL ORDER BY seq"
                ).fetchall()
            report["expected_receipts"] = len(rows)
            report["through_sequence"] = rows[-1]["seq"] if rows else None
            artifacts = {}
            for row in rows:
                relative = row["receipt_path"]
                try:
                    digest, size, raw = self.stable_digest(lambda: locate(relative), keep=True)
                    report["bytes_checked"] += size
                    if digest != Path(relative).stem.split("-", 1)[-1]:
                        raise IntegrityError("sha256_mismatch")
                    receipt = json.loads(raw)
                    if (
                        receipt["schema"] != "field-recognition-receipt/1"
                        or receipt["source_instance"] != self.archive.instance
                        or receipt["sequence"] != row["seq"]
                        or receipt["entity"] != row["entity"]
                        or receipt["entity_id"] != row["entity_id"]
                        or receipt["document"] != json.loads(row["document"])
                    ):
                        raise IntegrityError("snapshot_mismatch")
                    for kind, item in receipt.get("artifacts", {}).items():
                        if not isinstance(item, dict):
                            continue  # Legacy original_status is an explicit string.
                        with self.lock:
                            locate(item["path"])
                        if item["path"] in artifacts and artifacts[item["path"]] != item:
                            raise IntegrityError("conflicting_object_metadata")
                        artifacts[item["path"]] = item
                    report["verified_receipts"] += 1
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.archive._root(
                        create=False
                    )  # A lost mount is an incomplete check, not a missing-file claim.
                    report["issues"].append(
                        {
                            "kind": "receipt",
                            "path": relative,
                            "sequence": row["seq"],
                            "entity": row["entity"],
                            "entity_id": row["entity_id"],
                            "code": issue_code(exc),
                        }
                    )
            report["expected_objects"] = len(artifacts)
            for relative, item in artifacts.items():
                try:
                    digest, size, _ = self.stable_digest(lambda: locate(relative))
                    report["bytes_checked"] += size
                    if digest != item["sha256"]:
                        raise IntegrityError("sha256_mismatch")
                    if size != item["size_bytes"]:
                        raise IntegrityError("size_mismatch")
                    report["verified_objects"] += 1
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.archive._root(create=False)
                    report["issues"].append(
                        {"kind": "object", "path": relative, "code": issue_code(exc)}
                    )
            if report["issues"]:
                report["status"] = "findings"
        except CheckStopped:
            report["status"] = "interrupted"
        except Exception as exc:
            report.update(status="unavailable", error=type(exc).__name__)
        finally:
            report.update(
                finished_at=self.core["now"](),
                elapsed_seconds=round(time.monotonic() - started, 3),
                issue_count=len(report["issues"]),
            )
            try:
                with self.core["db"]() as conn:
                    conn.execute(
                        "INSERT INTO archive_integrity_runs(id,document) VALUES(?,?)",
                        (report["run_id"], json.dumps(report, ensure_ascii=False)),
                    )
            finally:
                self.running = False
                self.run_lock.release()
        return report

    def publish_pending(self):
        # Run on the archive writer so index/report publication has one owner.
        from field_recognition.archive.archive_store import canonical, immutable_write, replace_view

        with self.core["db"]() as conn:
            rows = conn.execute(
                "SELECT * FROM archive_integrity_runs WHERE published_path IS NULL ORDER BY rowid LIMIT 5"
            ).fetchall()
        if not rows:
            return
        root = self.archive._root(create=False)
        for row in rows:
            report = json.loads(row["document"])
            raw = canonical(report)
            relative = f".System/Integrity/Reports/{report['finished_at'][:10]}/{row['id']}-{hashlib.sha256(raw).hexdigest()}.json"
            immutable_write(root / relative, raw)
            replace_view(
                root / ".System/Integrity/Latest.json",
                canonical(report | {"report_path": relative}),
            )
            with self.core["db"]() as conn:
                conn.execute(
                    "UPDATE archive_integrity_runs SET published_path=? WHERE id=?",
                    (relative, row["id"]),
                )

    def start(self):
        def loop():
            while not self.stop.is_set():
                if self.due():
                    self.run_once()
                self.stop.wait(30)

        self.worker = threading.Thread(target=loop, name="archive-integrity", daemon=True)
        self.worker.start()

    def close(self):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=3)
