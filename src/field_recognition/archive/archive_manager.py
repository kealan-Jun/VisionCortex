"""Read-only archive inspection and explicit recovery into a new destination."""

import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from field_recognition.archive.archive_integrity import relative_file


class ArchiveManager:
    def __init__(self, db, archive_root):
        self.db = db
        # Construction never creates a NAS mountpoint or performs network I/O.
        self.archive_root = Path(archive_root)

    def check_capacity(self, required_bytes=0):
        stat = shutil.disk_usage(self.archive_root)
        return {
            "total_bytes": stat.total,
            "used_bytes": stat.used,
            "free_bytes": stat.free,
            "used_percent": stat.used / stat.total * 100 if stat.total else 0,
            "free_gb": stat.free / 1024**3,
            "can_store": stat.free >= max(0, required_bytes),
        }

    def enforce_capacity_limit(self, max_used_percent=90.0, target_percent=80.0):
        usage = self.check_capacity()["used_percent"]
        return {
            "cleanup_needed": usage >= max_used_percent,
            "current_usage": usage,
            "removed_count": 0,
            "action": "pause_writes" if usage >= max_used_percent else "none",
        }

    def _receipts(self, entity, ident):
        with self.db.connection() as conn:
            return conn.execute(
                "SELECT * FROM archive_outbox WHERE entity=? AND entity_id=? AND archived_at IS NOT NULL ORDER BY seq",
                (entity, ident),
            ).fetchall()

    def verify_archive(self, entity_type, entity_id):
        rows = self._receipts(entity_type, entity_id)
        files, errors = {}, []
        for row in rows:
            try:
                relative = row["receipt_path"]
                path = relative_file(self.archive_root, relative)
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != path.stem.split("-", 1)[-1]:
                    raise ValueError("Receipt SHA-256 mismatch")
                doc = json.loads(raw)
                if (doc["sequence"], doc["entity"], doc["entity_id"], doc["document"]) != (
                    row["seq"],
                    row["entity"],
                    row["entity_id"],
                    json.loads(row["document"]),
                ):
                    raise ValueError("Receipt differs from the database snapshot")
                files[relative] = digest
                for item in doc.get("artifacts", {}).values():
                    if not isinstance(item, dict):
                        continue
                    obj = relative_file(self.archive_root, item["path"])
                    data = obj.read_bytes()
                    if (
                        hashlib.sha256(data).hexdigest() != item["sha256"]
                        or len(data) != item["size_bytes"]
                    ):
                        raise ValueError("Evidence checksum mismatch")
                    files[item["path"]] = item["sha256"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({"sequence": row["seq"], "error": str(exc)})
        return {
            "exists": bool(rows),
            "valid": bool(rows) and not errors,
            "errors": errors,
            "files": files,
        }

    def restore_archive(self, entity_type, entity_id, target_dir=None):
        verification = self.verify_archive(entity_type, entity_id)
        if not verification["valid"]:
            raise ValueError("Archive integrity verification failed")
        destination = (
            Path(target_dir)
            if target_dir
            else Path(tempfile.mkdtemp(prefix="FieldRecovery")) / "Evidence"
        )
        if destination.exists():
            raise FileExistsError("Recovery requires a new destination")
        if destination.resolve().is_relative_to(self.archive_root.resolve()):
            raise ValueError("Recovery must be outside the archive")
        destination.mkdir(parents=True)
        for relative, digest in verification["files"].items():
            source = relative_file(self.archive_root, relative)
            target = relative_file(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("Evidence changed during recovery")
        return {"success": True, "restored_to": str(destination)}

    def backup_database(self, backup_path):
        path = Path(backup_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve exclusively; never truncate a prior backup.
        with path.open("xb"):
            pass
        try:
            with self.db.connection() as source:
                target = sqlite3.connect(path)
                try:
                    source.backup(target)
                    if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError("Backup integrity check failed")
                finally:
                    target.close()
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {
            "success": True,
            "backup_path": str(path),
            "size_bytes": path.stat().st_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def restore_database(self, backup_path, *, target_path=None):
        # Never overwrite a running SQLite database or its WAL. Stage a new DB for offline switchover.
        if target_path is None:
            raise ValueError(
                "Specify a new target_path; live database replacement is not supported"
            )
        source = Path(backup_path).resolve()
        target = Path(target_path).resolve()
        if target == Path(self.db.path).resolve() or target == source or target.exists():
            raise ValueError("Recovery target must be a new database")
        conn = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        try:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Backup is invalid")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb"):
                pass
            dest = sqlite3.connect(target)
            try:
                conn.backup(dest)
            finally:
                dest.close()
        finally:
            conn.close()
        return {"success": True, "restored_to": str(target), "live_database_changed": False}

    def update_catalog_incremental(self, entity_type, entity_id):
        result = self.verify_archive(entity_type, entity_id)
        if not result["valid"]:
            raise ValueError("Cannot index unverified evidence")
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO archive_catalog(entity,entity_id,manifest,indexed_at) VALUES(?,?,?,?)",
                (
                    entity_type,
                    entity_id,
                    json.dumps(result),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_archive_stats(self):
        with self.db.connection() as conn:
            total, archived = conn.execute(
                "SELECT count(*),count(archived_at) FROM archive_outbox"
            ).fetchone()
        result = {
            "total_items": total,
            "archived": archived,
            "pending": total - archived,
            "archive_percent": archived / total * 100 if total else 0,
        }
        try:
            result["capacity"] = self.check_capacity()
        except OSError:
            result["capacity"] = None
            result["status"] = "unavailable"
        return result
