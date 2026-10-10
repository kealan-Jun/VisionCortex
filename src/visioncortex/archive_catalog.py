from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .indexing import (
    INDEX_DB_NAME,
    INDEX_MANIFEST_NAME,
    action_search_aliases,
)
from .indexing import _liquid_index_values
from .pathing import archive_contains
from .storage import (
    ARCHIVE_DIRECTORIES,
    archive_promotion_in_progress,
    read_current_release_pointer,
)


CATALOG_SCHEMA_VERSION = "visioncortex-archive-read-catalog/5"
CATALOG_DB_NAME = "archive_read_catalog.sqlite3"
_CATALOG_LOCK = threading.RLock()


def archive_catalog_path(local_runtime_root: Path, archive_root: Path) -> Path:
    identity = hashlib.sha256(str(archive_root.resolve()).encode("utf-8")).hexdigest()[:16]
    return local_runtime_root / "state" / f"{Path(CATALOG_DB_NAME).stem}-{identity}.sqlite3"


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _epoch(value: Any, fallback: float) -> float:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return fallback


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        existing_schema = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        existing_schema = None
    if existing_schema and existing_schema["value"] != CATALOG_SCHEMA_VERSION:
        connection.executescript(
            "DROP TABLE IF EXISTS key_events; DROP TABLE IF EXISTS archives; "
            "DROP TABLE IF EXISTS metadata;"
        )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS archives (
            name TEXT PRIMARY KEY,
            source_revision TEXT NOT NULL,
            modified_epoch_us INTEGER NOT NULL,
            modified_at TEXT NOT NULL,
            published_epoch REAL NOT NULL,
            release_id TEXT,
            experiment_count INTEGER NOT NULL,
            key_event_count INTEGER NOT NULL,
            pipeline_stage TEXT NOT NULL,
            progress REAL,
            has_model_understanding INTEGER NOT NULL,
            has_daily_report INTEGER NOT NULL,
            has_evidence_index INTEGER NOT NULL,
            evidence_level TEXT,
            formal_accuracy_claim_allowed INTEGER,
            integrity_status TEXT NOT NULL,
            catalog_error TEXT
        );
        CREATE INDEX IF NOT EXISTS archives_order_idx
            ON archives(modified_epoch_us DESC, name);
        CREATE TABLE IF NOT EXISTS key_events (
            archive_name TEXT NOT NULL REFERENCES archives(name) ON DELETE CASCADE,
            release_id TEXT,
            published_epoch_us INTEGER NOT NULL,
            event_uid TEXT NOT NULL,
            archive_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            parent_event_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            start_us INTEGER NOT NULL,
            end_us INTEGER NOT NULL,
            peak_timestamp_us INTEGER NOT NULL,
            cross_view_supported INTEGER NOT NULL,
            dual_view_material_ready INTEGER NOT NULL,
            search_text TEXT NOT NULL,
            event_json TEXT NOT NULL,
            artifacts_json TEXT NOT NULL,
            liquid_state_status TEXT,
            liquid_present INTEGER,
            visible_flow INTEGER,
            PRIMARY KEY(archive_name, event_uid)
        );
        CREATE INDEX IF NOT EXISTS catalog_events_timeline_idx
            ON key_events(published_epoch_us DESC, archive_name, peak_timestamp_us, event_uid);
        CREATE INDEX IF NOT EXISTS catalog_events_filter_idx
            ON key_events(action_type, parent_event_id, cross_view_supported);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
        (CATALOG_SCHEMA_VERSION,),
    )
    connection.commit()
    return connection


def lightweight_release_integrity(root: Path, pointer: dict[str, Any]) -> str:
    relative = Path(str(pointer.get("release_manifest") or ""))
    manifest_path = (root / relative).resolve()
    if not archive_contains(manifest_path, root.resolve()) or not manifest_path.is_file():
        return "failed_release_manifest"
    expected = str(pointer.get("release_manifest_file_sha256") or "")
    if not expected or not hmac.compare_digest(_sha256(manifest_path), expected):
        return "failed_release_manifest"
    manifest = _read_json(manifest_path, {})
    if str(manifest.get("release_id") or "") != str(pointer.get("release_id") or ""):
        return "failed_release_identity"
    return "release_manifest_verified"


def _verify_released_index(
    root: Path, pointer: dict[str, Any], database: Path
) -> bool:
    manifest_path = (root / str(pointer.get("release_manifest") or "")).resolve()
    manifest = _read_json(manifest_path, {}) or {}
    receipt = next(
        (
            item
            for item in (manifest.get("manifests") or {}).get(
                "JSON-Config-Files", []
            )
            if str(item.get("path") or "") == INDEX_DB_NAME
        ),
        None,
    )
    return bool(
        receipt
        and int(receipt.get("size_bytes") or -1) == database.stat().st_size
        and str(receipt.get("sha256") or "") == _sha256(database)
    )


def _archive_snapshot(root: Path) -> dict[str, Any] | None:
    if archive_promotion_in_progress(root):
        return None
    json_root = root / "JSON-Config-Files"
    database = json_root / INDEX_DB_NAME
    package = json_root / "evidence_package.json"
    if not database.is_file() and not package.is_file():
        return None
    try:
        modified_epoch = root.stat().st_mtime
    except OSError:
        return None
    pointer = read_current_release_pointer(root) or {}
    manifest = _read_json(json_root / INDEX_MANIFEST_NAME, {}) or {}
    status_path = json_root / "pipeline_status.json"
    if not status_path.is_file():
        status_path = root / "run_status.json"
    status = _read_json(status_path, {}) or {}
    status_signature = (
        f"{status_path.stat().st_size}:{status_path.stat().st_mtime_ns}"
        if status_path.is_file() else "no-status"
    )
    release_id = str(pointer.get("release_id") or "") or None
    manifest_signature = "no-release-manifest"
    if pointer:
        manifest_path = (root / str(pointer.get("release_manifest") or "")).resolve()
        if archive_contains(manifest_path, root.resolve()) and manifest_path.is_file():
            manifest_status = manifest_path.stat()
            manifest_signature = (
                f"{manifest_status.st_size}:{manifest_status.st_mtime_ns}"
            )
    index_signature = (
        f"{database.stat().st_size}:{database.stat().st_mtime_ns}"
        if database.is_file()
        else "no-index"
    )
    source_revision = "|".join(
        (
            release_id or "legacy",
            str(pointer.get("release_manifest_file_sha256") or ""),
            manifest_signature,
            index_signature,
            status_signature,
        )
    )
    experiment_root = root / "Experiment-Clips"
    experiment_count = (
        sum(1 for item in experiment_root.iterdir() if item.is_dir())
        if experiment_root.is_dir()
        else 0
    )
    counts = manifest.get("counts") or {}
    integrity_status = (
        "release_manifest_pending"
        if pointer
        else "legacy_archive_not_release_verified"
    )
    published_at = pointer.get("published_at")
    return {
        "name": root.name,
        "source_revision": source_revision,
        "modified_epoch_us": round(modified_epoch * 1_000_000),
        "modified_at": datetime.fromtimestamp(modified_epoch).astimezone().isoformat(),
        "published_epoch": _epoch(published_at, modified_epoch),
        "release_id": release_id,
        "experiment_count": experiment_count,
        "key_event_count": int(counts.get("key_events") or 0),
        "pipeline_stage": str(status.get("stage") or "archived"),
        "progress": status.get("progress"),
        "has_model_understanding": int(
            (root / "Key-Materials" / "Key-Materials-Model-Understanding.json").is_file()
        ),
        "has_daily_report": int((json_root / "daily_report_manifest.json").is_file()),
        "has_evidence_index": int(database.is_file()),
        "evidence_level": pointer.get("evidence_level"),
        "formal_accuracy_claim_allowed": (
            int(bool(pointer.get("formal_accuracy_claim_allowed"))) if pointer else None
        ),
        "integrity_status": integrity_status,
        "catalog_error": None,
        "database": database,
        "root": root,
        "pointer": pointer,
    }


def _replace_archive(connection: sqlite3.Connection, item: dict[str, Any]) -> None:
    connection.execute("DELETE FROM archives WHERE name = ?", (item["name"],))
    columns = (
        "name", "source_revision", "modified_epoch_us", "modified_at", "published_epoch",
        "release_id", "experiment_count", "key_event_count", "pipeline_stage", "progress",
        "has_model_understanding", "has_daily_report", "has_evidence_index", "evidence_level",
        "formal_accuracy_claim_allowed", "integrity_status", "catalog_error",
    )
    values = [item.get(column) for column in columns]
    connection.execute(
        f"INSERT INTO archives({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        values,
    )
    database = Path(item["database"])
    if not database.is_file():
        return
    source = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        artifacts: dict[str, list[dict[str, Any]]] = {}
        for row in source.execute(
            "SELECT event_uid, artifact_json FROM artifacts "
            "ORDER BY event_uid, artifact_type, view_id"
        ):
            artifacts.setdefault(str(row["event_uid"]), []).append(
                json.loads(row["artifact_json"])
            )
        records = []
        for row in source.execute("SELECT * FROM key_events ORDER BY peak_timestamp_us, event_uid"):
            search_text = " ".join(
                filter(
                    None,
                    (
                        str(row["search_text"] or ""),
                        action_search_aliases(str(row["action_type"] or "")),
                    ),
                )
            )
            event_artifacts = artifacts.get(str(row["event_uid"]), [])
            aligned_types = {
                str(artifact.get("artifact_type") or "")
                for artifact in event_artifacts
                if artifact.get("view_role") == "aligned_first_third"
            }
            liquid = _liquid_index_values(json.loads(row["event_json"]))
            records.append(
                (
                    item["name"], item["release_id"],
                    round(float(item["published_epoch"]) * 1_000_000),
                    row["event_uid"], row["archive_id"], row["event_id"],
                    row["parent_event_id"], row["action_type"], row["start_us"],
                    row["end_us"], row["peak_timestamp_us"], row["cross_view_supported"],
                    int({"key_frame", "key_clip"}.issubset(aligned_types)),
                    search_text, row["event_json"],
                    json.dumps(event_artifacts, ensure_ascii=False),
                    liquid["status"], liquid["liquid_present"], liquid["visible_flow"],
                )
            )
        connection.executemany(
            "INSERT INTO key_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            records,
        )
        if not item["key_event_count"]:
            connection.execute(
                "UPDATE archives SET key_event_count = ? WHERE name = ?",
                (len(records), item["name"]),
            )
    finally:
        source.close()


def ensure_archive_catalog(
    archive_root: Path,
    database: Path,
) -> dict[str, Any]:
    """Synchronize a local, disposable read model from formal archive indexes."""

    archive_root = archive_root.resolve()
    if not archive_root.is_dir():
        return {"database": str(database), "archive_count": 0, "refreshed": 0}
    snapshots: dict[str, dict[str, Any]] = {}
    for root in archive_root.iterdir():
        if not root.is_dir() or root.name.startswith(".VisionCortex-"):
            continue
        try:
            directories = {item.name for item in root.iterdir() if item.is_dir()}
        except OSError:
            continue
        if not directories.intersection(ARCHIVE_DIRECTORIES):
            continue
        try:
            snapshot = _archive_snapshot(root)
        except OSError:
            continue
        if snapshot:
            snapshots[root.name] = snapshot
    refreshed = 0
    with _CATALOG_LOCK:
        connection = _connect(database)
        try:
            existing = {
                str(row["name"]): str(row["source_revision"])
                for row in connection.execute("SELECT name, source_revision FROM archives")
            }
            for missing in set(existing) - set(snapshots):
                connection.execute("DELETE FROM archives WHERE name = ?", (missing,))
            for name, snapshot in snapshots.items():
                if existing.get(name) == snapshot["source_revision"]:
                    continue
                try:
                    if snapshot["pointer"]:
                        snapshot["integrity_status"] = lightweight_release_integrity(
                            Path(snapshot["root"]), snapshot["pointer"]
                        )
                        if snapshot["integrity_status"] != "release_manifest_verified":
                            raise ValueError("release manifest integrity failed")
                        if Path(snapshot["database"]).is_file() and not _verify_released_index(
                            Path(snapshot["root"]),
                            snapshot["pointer"],
                            Path(snapshot["database"]),
                        ):
                            raise ValueError("released evidence index integrity failed")
                        snapshot["integrity_status"] = "release_index_verified"
                    _replace_archive(connection, snapshot)
                except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
                    snapshot = dict(snapshot)
                    snapshot["source_revision"] = (
                        f"error:{snapshot['source_revision']}"
                    )
                    snapshot["integrity_status"] = "index_unavailable"
                    snapshot["catalog_error"] = type(exc).__name__
                    snapshot["has_evidence_index"] = 0
                    snapshot["database"] = Path("__missing__")
                    _replace_archive(connection, snapshot)
                refreshed += 1
            connection.commit()
            count = int(connection.execute("SELECT COUNT(*) FROM archives").fetchone()[0])
        finally:
            connection.close()
    return {"database": str(database), "archive_count": count, "refreshed": refreshed}


def list_catalog_archives(
    database: Path,
    *,
    query: str | None = None,
    after_modified_epoch_us: int | None = None,
    after_name: str | None = None,
    limit: int = 51,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    connection = _connect(database)
    try:
        conditions: list[str] = []
        parameters: list[Any] = []
        if query:
            conditions.append("name LIKE ? ESCAPE '\\'")
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            parameters.append(f"%{escaped}%")
        if after_modified_epoch_us is not None and after_name is not None:
            conditions.append("(modified_epoch_us < ? OR (modified_epoch_us = ? AND name > ?))")
            parameters.extend((after_modified_epoch_us, after_modified_epoch_us, after_name))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        count_conditions = conditions[:-1] if after_modified_epoch_us is not None else conditions
        count_parameters = parameters[:-3] if after_modified_epoch_us is not None else parameters
        count_where = f" WHERE {' AND '.join(count_conditions)}" if count_conditions else ""
        total = int(connection.execute(f"SELECT COUNT(*) FROM archives{count_where}", count_parameters).fetchone()[0])
        aggregates = connection.execute(
            "SELECT COALESCE(SUM(experiment_count), 0) AS experiments, "
            "COALESCE(SUM(key_event_count), 0) AS key_events FROM archives" + count_where,
            count_parameters,
        ).fetchone()
        parameters.append(max(1, int(limit)))
        rows = connection.execute(
            "SELECT * FROM archives" + where + " ORDER BY modified_epoch_us DESC, name LIMIT ?",
            parameters,
        ).fetchall()
        return [dict(row) for row in rows], total, {
            "experiments": int(aggregates["experiments"]),
            "key_events": int(aggregates["key_events"]),
        }
    finally:
        connection.close()


def catalog_archive_release(database: Path, archive_name: str) -> str | None:
    connection = _connect(database)
    try:
        row = connection.execute(
            "SELECT release_id FROM archives WHERE name = ?", (archive_name,)
        ).fetchone()
        return str(row["release_id"]) if row and row["release_id"] is not None else None
    finally:
        connection.close()


def search_catalog_events(
    database: Path,
    *,
    archive_name: str | None = None,
    query: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    cross_view: bool | None = None,
    liquid_state_status: str | None = None,
    liquid_present: bool | None = None,
    visible_flow: bool | None = None,
    material_ready: bool | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    after_position: tuple[int, int, str, int, str] | None = None,
    limit: int = 51,
) -> tuple[list[dict[str, Any]], int]:
    connection = _connect(database)
    try:
        conditions: list[str] = []
        parameters: list[Any] = []
        for field, value in (
            ("archive_name", archive_name),
            ("action_type", action_type),
            ("parent_event_id", parent_event_id),
        ):
            if value:
                conditions.append(f"{field} = ?")
                parameters.append(str(value))
        if cross_view is not None:
            conditions.append("cross_view_supported = ?")
            parameters.append(int(cross_view))
        for field, value in (("liquid_state_status", liquid_state_status), ("liquid_present", liquid_present), ("visible_flow", visible_flow)):
            if value is not None:
                conditions.append(f"{field} = ?")
                parameters.append(value)
        if material_ready is not None:
            conditions.append("dual_view_material_ready = ?")
            parameters.append(int(material_ready))
        if start_us is not None:
            conditions.append("end_us >= ?")
            parameters.append(int(start_us))
        if end_us is not None:
            conditions.append("start_us <= ?")
            parameters.append(int(end_us))
        normalized_query = str(query or "").strip()
        query_terms: list[str] = []
        if normalized_query:
            query_terms = [term for term in normalized_query.split() if term]
            for term in query_terms:
                escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                conditions.append(
                    "(search_text LIKE ? ESCAPE '\\' OR event_id LIKE ? ESCAPE '\\')"
                )
                parameters.extend((f"%{escaped}%", f"%{escaped}%"))
        count_where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        total = int(
            connection.execute(
                "SELECT COUNT(*) FROM key_events" + count_where, parameters
            ).fetchone()[0]
        )
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        if normalized_query:
            score_parts = [
                "CASE WHEN lower(event_id) = lower(?) THEN 100 ELSE 0 END",
                "CASE WHEN lower(action_type) = lower(?) THEN 80 ELSE 0 END",
                "CASE WHEN instr(lower(search_text), lower(?)) > 0 THEN 40 ELSE 0 END",
            ]
            score_parameters: list[Any] = [
                normalized_query,
                normalized_query,
                normalized_query,
            ]
            for term in query_terms:
                score_parts.append(
                    "CASE WHEN instr(lower(search_text), lower(?)) > 0 THEN 5 ELSE 0 END"
                )
                score_parameters.append(term)
            score_sql = " + ".join(score_parts)
        else:
            score_sql = "0"
            score_parameters = []
        ranked_sql = (
            "SELECT key_events.*, " + score_sql
            + " AS relevance_score FROM key_events" + where
        )
        outer_conditions = []
        outer_parameters: list[Any] = []
        if after_position:
            score, published, cursor_archive, peak, event_uid = after_position
            outer_conditions.append(
                "(relevance_score < ? OR (relevance_score = ? AND "
                "(published_epoch_us < ? OR (published_epoch_us = ? AND "
                "(archive_name > ? OR (archive_name = ? AND "
                "(peak_timestamp_us > ? OR (peak_timestamp_us = ? AND event_uid > ?))))))))"
            )
            outer_parameters.extend(
                (
                    score, score, published, published, cursor_archive,
                    cursor_archive, peak, peak, event_uid,
                )
            )
        outer_where = (
            f" WHERE {' AND '.join(outer_conditions)}" if outer_conditions else ""
        )
        sql_parameters = score_parameters + parameters + outer_parameters
        sql_parameters.append(max(1, int(limit)))
        rows = connection.execute(
            "SELECT * FROM (" + ranked_sql + ")" + outer_where
            + " ORDER BY relevance_score DESC, published_epoch_us DESC, "
            "archive_name, peak_timestamp_us, event_uid LIMIT ?",
            sql_parameters,
        ).fetchall()
        results = []
        for row in rows:
            payload = json.loads(row["event_json"])
            payload.update(
                {
                    "archive_name": row["archive_name"],
                    "release_id": row["release_id"],
                    "published_epoch_us": row["published_epoch_us"],
                    "relevance_score": row["relevance_score"],
                    "event_uid": row["event_uid"],
                    "archive_id": row["archive_id"],
                    "artifact_references": json.loads(row["artifacts_json"]),
                }
            )
            results.append(payload)
        return results, total
    finally:
        connection.close()
