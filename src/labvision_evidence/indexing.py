from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

from .physical_changes import physical_change_records
from .pathing import archive_contains, archive_relative_posix
from .schemas import EvidenceEvent, ExperimentGroup, VideoInfo


INDEX_SCHEMA_VERSION = "visioncortex-evidence-index/3"
INDEX_DB_NAME = "evidence_index.sqlite"
INDEX_MANIFEST_NAME = "evidence_index_manifest.json"
ARTIFACT_REGISTRY_NAME = "artifact_registry.jsonl"
EVIDENCE_REGISTRY_NAME = "evidence_registry.jsonl"
DECISION_REGISTRY_NAME = "decision_receipt_registry.jsonl"
PHYSICAL_CHANGE_REGISTRY_NAME = "physical_change_registry.jsonl"


def stable_event_uid(archive_id: str, group_id: str, event_id: str) -> str:
    """Return a deterministic identifier that remains stable across reruns."""

    return f"{archive_id}:{group_id}:{event_id}"


def stable_artifact_uid(event_uid: str, artifact_type: str, view_id: str) -> str:
    return f"{event_uid}:artifact:{artifact_type}:{view_id}"


def stable_evidence_uid(event_uid: str, evidence_id: str) -> str:
    return f"{event_uid}:evidence:{evidence_id}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    content = "".join(f"{_json_dumps(record)}\n" for record in records)
    _atomic_write_text(path, content)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_media(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if not archive_contains(candidate, resolved_root):
        raise ValueError(f"Artifact path escapes archive root: {relative}")
    return candidate


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [item for child in value.values() for item in _flatten_text(child)]
    if isinstance(value, (list, tuple, set)):
        return [item for child in value for item in _flatten_text(child)]
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return [text] if text else []
    return []


def _group_for_event(
    groups: Sequence[ExperimentGroup], event_id: str
) -> ExperimentGroup:
    return next(group for group in groups if event_id in group.key_event_ids)


def _event_uid_from_payload(
    archive_id: str, payload: dict[str, Any], groups: Sequence[ExperimentGroup]
) -> str:
    existing = (
        (payload.get("provenance") or {}).get("index") or {}
    ).get("event_uid")
    if existing:
        return str(existing)
    group_id = str(payload.get("parent_event_id") or "unassigned")
    return stable_event_uid(archive_id, group_id, str(payload["event_id"]))


def _artifact_record(
    root: Path,
    event_uid: str,
    event_id: str,
    parent_event_id: str,
    artifact_type: str,
    item: dict[str, Any],
    allow_incomplete: bool = False,
    cached: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relative = str(item["path"])
    view_id = str(item.get("view_id") or item.get("view_role") or "unknown")
    if "://" in relative:
        if not relative.startswith("dry-run://"):
            raise ValueError(f"Unsupported virtual artifact reference: {relative}")
        return {
            "schema_version": "visioncortex-artifact-registry/1",
            "artifact_uid": str(
                item.get("artifact_uid")
                or stable_artifact_uid(event_uid, artifact_type, view_id)
            ),
            "event_uid": event_uid,
            "event_id": event_id,
            "parent_event_id": parent_event_id,
            "artifact_type": artifact_type,
            "view_id": view_id,
            "view_role": item.get("view_role"),
            "path": relative,
            "sidecar_path": None,
            "mime_type": "application/x-visioncortex-dry-run",
            "size_bytes": None,
            "sha256": None,
            "sidecar_size_bytes": None,
            "sidecar_sha256": None,
            "integrity_status": "virtual_dry_run",
            "hash_reused": False,
            "timestamp_us": item.get("timestamp_us"),
            "start_us": item.get("start_us"),
            "end_us": item.get("end_us"),
        }
    media = _relative_media(root, relative)
    if not media.is_file() or media.stat().st_size <= 0:
        raise FileNotFoundError(f"Indexed key material is missing or empty: {relative}")
    sidecar = media.with_suffix(".json")
    if (
        not allow_incomplete
        and (not sidecar.is_file() or sidecar.stat().st_size <= 0)
    ):
        raise FileNotFoundError(f"Indexed key material sidecar is missing or empty: {sidecar}")
    sidecar_available = sidecar.is_file() and sidecar.stat().st_size > 0
    media_stat = media.stat()
    sidecar_stat = sidecar.stat() if sidecar_available else None
    cache_matches = bool(
        cached
        and cached.get("path") == relative
        and cached.get("size_bytes") == media_stat.st_size
        and cached.get("mtime_ns") == media_stat.st_mtime_ns
        and cached.get("sidecar_size_bytes")
        == (sidecar_stat.st_size if sidecar_stat else None)
        and cached.get("sidecar_mtime_ns")
        == (sidecar_stat.st_mtime_ns if sidecar_stat else None)
        and cached.get("sha256")
        and (cached.get("sidecar_sha256") if sidecar_available else True)
    )
    return {
        "schema_version": "visioncortex-artifact-registry/1",
        "artifact_uid": str(
            item.get("artifact_uid")
            or stable_artifact_uid(event_uid, artifact_type, view_id)
        ),
        "event_uid": event_uid,
        "event_id": event_id,
        "parent_event_id": parent_event_id,
        "artifact_type": artifact_type,
        "view_id": view_id,
        "view_role": item.get("view_role"),
        "path": relative,
        "sidecar_path": archive_relative_posix(sidecar, root),
        "mime_type": mimetypes.guess_type(media.name)[0] or "application/octet-stream",
        "size_bytes": media_stat.st_size,
        "mtime_ns": media_stat.st_mtime_ns,
        "sha256": cached["sha256"] if cache_matches else _sha256(media),
        "sidecar_size_bytes": sidecar_stat.st_size if sidecar_stat else None,
        "sidecar_mtime_ns": sidecar_stat.st_mtime_ns if sidecar_stat else None,
        "sidecar_sha256": (
            cached["sidecar_sha256"]
            if cache_matches and sidecar_available
            else _sha256(sidecar)
            if sidecar_available
            else None
        ),
        "integrity_status": "verified" if sidecar_available else "dry_run_sidecar_absent",
        "hash_reused": cache_matches,
        "timestamp_us": item.get("timestamp_us"),
        "start_us": item.get("start_us"),
        "end_us": item.get("end_us"),
    }


def _source_reference(
    infos: dict[str, VideoInfo], view_id: str, frame_index: int | None, local_ms: float
) -> dict[str, Any] | None:
    info = infos.get(view_id)
    if info is None:
        return None
    selected = None
    if frame_index is not None:
        selected = next(
            (
                segment
                for segment in info.segments
                if segment.frame_start_index
                <= frame_index
                < segment.frame_start_index + segment.frame_count
            ),
            None,
        )
    if selected is None and info.segments:
        selected = next(
            (
                segment
                for segment in info.segments
                if segment.virtual_start_ms <= local_ms <= segment.virtual_end_ms
            ),
            None,
        )
    if selected is None:
        virtual_local_ms = (
            frame_index * 1000.0 / info.fps
            if frame_index is not None and info.fps > 0
            else local_ms
        )
        return {
            "source_video_path": str(info.path),
            "source_segment_path": str(info.path),
            "source_timestamps_csv": None,
            "source_frame_index": frame_index,
            "virtual_local_ms": virtual_local_ms,
        }
    source_frame_index = (
        frame_index - selected.frame_start_index if frame_index is not None else None
    )
    virtual_local_ms = (
        selected.virtual_start_ms + source_frame_index * 1000.0 / selected.fps
        if source_frame_index is not None and selected.fps > 0
        else local_ms
    )
    return {
        "source_video_path": str(info.path),
        "source_segment_path": str(selected.path),
        "source_timestamps_csv": (
            str(selected.timestamps_csv) if selected.timestamps_csv is not None else None
        ),
        "source_frame_index": source_frame_index,
        "virtual_frame_index": frame_index,
        "virtual_local_ms": virtual_local_ms,
        "source_segment_virtual_start_ms": selected.virtual_start_ms,
        "source_segment_virtual_end_ms": selected.virtual_end_ms,
    }


def _evidence_records(
    archive_id: str,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    infos: dict[str, VideoInfo],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    event_positions = {event.event_id: index for index, event in enumerate(events)}
    for event in events:
        try:
            group = _group_for_event(groups, event.event_id)
        except StopIteration:
            continue
        event_uid = stable_event_uid(archive_id, group.group_id, event.event_id)
        event_index = event_positions[event.event_id]
        for candidate_index, candidate in enumerate(event.candidates):
            candidate_evidence_id = candidate.candidate_id
            candidate_uid = stable_evidence_uid(event_uid, candidate_evidence_id)
            candidate_pointer = (
                f"/events/{event_index}/candidates/{candidate_index}"
            )
            records[candidate_uid] = {
                "schema_version": "visioncortex-evidence-registry/1",
                "evidence_uid": candidate_uid,
                "evidence_id": candidate_evidence_id,
                "event_uid": event_uid,
                "event_id": event.event_id,
                "parent_event_id": group.group_id,
                "evidence_kind": "action_candidate",
                "view_id": candidate.view_id,
                "view_role": candidate.role.value,
                "frame_index": None,
                "global_timestamp_us": round(candidate.key_global_ms * 1000.0),
                "local_timestamp_us": round(
                    ((candidate.local_start_ms + candidate.local_end_ms) / 2.0) * 1000.0
                ),
                "confidence": candidate.confidence,
                "json_references": [
                    {
                        "path": "JSON-Config-Files/evidence_package.json",
                        "json_pointer": candidate_pointer,
                    }
                ],
                "source": _source_reference(
                    infos,
                    candidate.view_id,
                    None,
                    (candidate.local_start_ms + candidate.local_end_ms) / 2.0,
                ),
            }
            evidence_count = max(1, len(candidate.evidence))
            duration = max(0.0, candidate.local_end_ms - candidate.local_start_ms)
            for evidence_index, evidence in enumerate(candidate.evidence):
                frame_index = evidence.get("frame_index")
                if frame_index is None:
                    continue
                frame_index = int(frame_index)
                evidence_id = f"{candidate.view_id}:frame-{frame_index}"
                evidence_uid = stable_evidence_uid(event_uid, evidence_id)
                estimated_local_ms = (
                    candidate.local_start_ms + duration * evidence_index / evidence_count
                )
                source = _source_reference(
                    infos, candidate.view_id, frame_index, estimated_local_ms
                )
                local_ms = float(
                    (source or {}).get("virtual_local_ms", estimated_local_ms)
                )
                local_span = candidate.local_end_ms - candidate.local_start_ms
                global_ms = (
                    candidate.global_start_ms
                    + (local_ms - candidate.local_start_ms)
                    / local_span
                    * (candidate.global_end_ms - candidate.global_start_ms)
                    if abs(local_span) > 1e-9
                    else candidate.key_global_ms
                )
                reference = {
                    "path": "JSON-Config-Files/evidence_package.json",
                    "json_pointer": f"{candidate_pointer}/evidence/{evidence_index}",
                    "candidate_id": candidate.candidate_id,
                }
                if evidence_uid in records:
                    records[evidence_uid]["json_references"].append(reference)
                    continue
                records[evidence_uid] = {
                    "schema_version": "visioncortex-evidence-registry/1",
                    "evidence_uid": evidence_uid,
                    "evidence_id": evidence_id,
                    "event_uid": event_uid,
                    "event_id": event.event_id,
                    "parent_event_id": group.group_id,
                    "evidence_kind": "frame_evidence",
                    "view_id": candidate.view_id,
                    "view_role": candidate.role.value,
                    "frame_index": frame_index,
                    "global_timestamp_us": round(global_ms * 1000.0),
                    "local_timestamp_us": round(local_ms * 1000.0),
                    "timestamp_basis": "source_frame_index_plus_candidate_affine_alignment",
                    "confidence": candidate.confidence,
                    "json_references": [reference],
                    "source": source,
                }
    return sorted(records.values(), key=lambda item: item["evidence_uid"])


def _decision_receipt_records(json_root: Path) -> list[dict[str, Any]]:
    """Collect canonical rule decisions without coupling callers to pipeline state."""

    sources = (
        "audit_layer.json",
        "key_material_selection.json",
        "key_material_selection_preview.json",
        "progressive_fine_scan.json",
    )
    records: dict[str, dict[str, Any]] = {}

    def visit(value: Any, source: str, pointer: str = "") -> None:
        if isinstance(value, dict):
            if value.get("receipt_schema_version") and value.get("decision_id"):
                decision_id = str(value["decision_id"])
                reference = {
                    "path": f"JSON-Config-Files/{source}",
                    "json_pointer": pointer or "/",
                }
                if decision_id not in records:
                    records[decision_id] = {
                        **value,
                        "json_references": [reference],
                    }
                elif reference not in records[decision_id]["json_references"]:
                    records[decision_id]["json_references"].append(reference)
            for key, child in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                visit(child, source, f"{pointer}/{escaped}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, source, f"{pointer}/{index}")

    for source in sources:
        path = json_root / source
        if not path.is_file():
            continue
        try:
            visit(json.loads(path.read_text(encoding="utf-8-sig")), source)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records.values(), key=lambda item: str(item["decision_id"]))


def _build_sqlite(
    path: Path,
    archive_id: str,
    normalized_events: Sequence[dict[str, Any]],
    artifact_records: Sequence[dict[str, Any]],
    evidence_records: Sequence[dict[str, Any]],
    decision_records: Sequence[dict[str, Any]],
    change_records: Sequence[dict[str, Any]],
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    fts5_enabled = True
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE key_events (
                event_uid TEXT PRIMARY KEY,
                archive_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                parent_event_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_subtype TEXT,
                start_us INTEGER NOT NULL,
                end_us INTEGER NOT NULL,
                peak_timestamp_us INTEGER NOT NULL,
                decision_status TEXT,
                cross_view_supported INTEGER NOT NULL,
                search_text TEXT NOT NULL,
                event_json TEXT NOT NULL
            );
            CREATE INDEX key_events_timeline_idx
                ON key_events(archive_id, peak_timestamp_us, event_uid);
            CREATE INDEX key_events_action_idx
                ON key_events(action_type, parent_event_id);
            CREATE TABLE artifacts (
                artifact_uid TEXT PRIMARY KEY,
                event_uid TEXT NOT NULL REFERENCES key_events(event_uid),
                artifact_type TEXT NOT NULL,
                view_id TEXT NOT NULL,
                view_role TEXT,
                path TEXT NOT NULL,
                sidecar_path TEXT,
                mime_type TEXT NOT NULL,
                size_bytes INTEGER,
                sha256 TEXT,
                artifact_json TEXT NOT NULL
            );
            CREATE INDEX artifacts_event_idx ON artifacts(event_uid, artifact_type);
            CREATE TABLE evidence (
                evidence_uid TEXT PRIMARY KEY,
                evidence_id TEXT NOT NULL,
                event_uid TEXT NOT NULL REFERENCES key_events(event_uid),
                evidence_kind TEXT NOT NULL,
                view_id TEXT,
                frame_index INTEGER,
                evidence_json TEXT NOT NULL
            );
            CREATE INDEX evidence_event_idx ON evidence(event_uid, evidence_id);
            CREATE TABLE decision_receipts (
                decision_id TEXT PRIMARY KEY,
                decision_type TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                rule_version TEXT NOT NULL,
                verdict TEXT NOT NULL,
                search_text TEXT NOT NULL,
                receipt_json TEXT NOT NULL
            );
            CREATE INDEX decision_receipts_rule_idx
                ON decision_receipts(rule_id, verdict, decision_id);
            CREATE TABLE decision_subjects (
                decision_id TEXT NOT NULL REFERENCES decision_receipts(decision_id),
                subject_id TEXT NOT NULL,
                PRIMARY KEY(decision_id, subject_id)
            );
            CREATE INDEX decision_subjects_subject_idx
                ON decision_subjects(subject_id, decision_id);
            CREATE TABLE physical_changes (
                change_uid TEXT PRIMARY KEY,
                event_uid TEXT NOT NULL REFERENCES key_events(event_uid),
                parent_event_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                object_role TEXT NOT NULL,
                object_id TEXT NOT NULL,
                state_before TEXT NOT NULL,
                state_after TEXT NOT NULL,
                start_us INTEGER NOT NULL,
                end_us INTEGER NOT NULL,
                peak_timestamp_us INTEGER NOT NULL,
                decision_status TEXT NOT NULL,
                change_json TEXT NOT NULL
            );
            CREATE INDEX physical_changes_timeline_idx
                ON physical_changes(peak_timestamp_us, change_uid);
            CREATE INDEX physical_changes_object_idx
                ON physical_changes(object_id, object_role, action_type);
            """
        )
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE key_events_fts USING fts5(
                    event_uid UNINDEXED,
                    search_text,
                    action_type,
                    objects,
                    current_step,
                    next_step,
                    tokenize='trigram'
                )
                """
            )
        except sqlite3.OperationalError:
            fts5_enabled = False
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", INDEX_SCHEMA_VERSION),
                ("archive_id", archive_id),
                ("fts5_enabled", json.dumps(fts5_enabled)),
            ),
        )
        for payload in normalized_events:
            event_uid = _event_uid_from_payload(archive_id, payload, [])
            understanding = (payload.get("provenance") or {}).get("mllm") or {}
            current_step = str(understanding.get("current_step") or "")
            next_step = str(understanding.get("next_step") or "")
            objects_text = " ".join(_flatten_text(payload.get("objects")))
            search_text = " ".join(
                dict.fromkeys(
                    _flatten_text(
                        {
                            "action_type": payload.get("action_type"),
                            "action_subtype": payload.get("action_subtype"),
                            "objects": payload.get("objects"),
                            "decision": payload.get("decision"),
                            "observations": payload.get("observations"),
                            "current_step": current_step,
                            "next_step": next_step,
                            "experiment": (payload.get("provenance") or {}).get(
                                "experiment_name"
                            ),
                        }
                    )
                )
            )
            cross_view = any(
                item.get("both_views_support_action") is True
                for item in payload.get("cross_view_associations", [])
            )
            connection.execute(
                """
                INSERT INTO key_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_uid,
                    archive_id,
                    str(payload["event_id"]),
                    str(payload.get("parent_event_id") or ""),
                    str(payload.get("action_type") or ""),
                    str(payload.get("action_subtype") or ""),
                    int(payload.get("start_us") or 0),
                    int(payload.get("end_us") or 0),
                    int(payload.get("peak_timestamp_us") or 0),
                    str((payload.get("decision") or {}).get("status") or ""),
                    int(cross_view),
                    search_text,
                    _json_dumps(payload),
                ),
            )
            if fts5_enabled:
                connection.execute(
                    "INSERT INTO key_events_fts VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event_uid,
                        search_text,
                        str(payload.get("action_type") or ""),
                        objects_text,
                        current_step,
                        next_step,
                    ),
                )
        connection.executemany(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    item["artifact_uid"],
                    item["event_uid"],
                    item["artifact_type"],
                    item["view_id"],
                    item.get("view_role"),
                    item["path"],
                    item["sidecar_path"],
                    item["mime_type"],
                    item["size_bytes"],
                    item["sha256"],
                    _json_dumps(item),
                )
                for item in artifact_records
            ],
        )
        connection.executemany(
            "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    item["evidence_uid"],
                    item["evidence_id"],
                    item["event_uid"],
                    item["evidence_kind"],
                    item.get("view_id"),
                    item.get("frame_index"),
                    _json_dumps(item),
                )
                for item in evidence_records
            ],
        )
        connection.executemany(
            "INSERT INTO decision_receipts VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(item["decision_id"]),
                    str(item.get("decision_type") or ""),
                    str(item.get("rule_id") or ""),
                    str(item.get("rule_version") or ""),
                    str(item.get("verdict") or ""),
                    " ".join(_flatten_text(item)),
                    _json_dumps(item),
                )
                for item in decision_records
            ],
        )
        connection.executemany(
            "INSERT INTO decision_subjects VALUES (?, ?)",
            [
                (str(item["decision_id"]), str(subject_id))
                for item in decision_records
                for subject_id in item.get("subject_ids", [])
            ],
        )
        connection.executemany(
            "INSERT INTO physical_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(item["change_uid"]),
                    str(item["event_uid"]),
                    str(item.get("parent_event_id") or ""),
                    str(item.get("action_type") or ""),
                    str(item.get("object_role") or ""),
                    str(item.get("object_id") or ""),
                    str(item.get("state_before") or ""),
                    str(item.get("state_after") or ""),
                    int(item.get("start_us") or 0),
                    int(item.get("end_us") or 0),
                    int(item.get("peak_timestamp_us") or 0),
                    str(item.get("decision_status") or ""),
                    _json_dumps(item),
                )
                for item in change_records
            ],
        )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, path)
    return fts5_enabled


def build_archive_index(
    root: Path,
    archive_id: str,
    normalized_events: Sequence[dict[str, Any]],
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    infos: dict[str, VideoInfo],
    *,
    hash_workers: int = 4,
) -> dict[str, Any]:
    """Build a disposable search index and one-hop provenance registries."""

    started = time.perf_counter()
    json_root = root / "JSON-Config-Files"
    json_root.mkdir(parents=True, exist_ok=True)
    artifact_jobs = []
    previous_registry_path = json_root / ARTIFACT_REGISTRY_NAME
    previous_artifacts: dict[str, dict[str, Any]] = {}
    if previous_registry_path.is_file():
        try:
            previous_artifacts = {
                item["artifact_uid"]: item
                for item in (
                    json.loads(line)
                    for line in previous_registry_path.read_text(
                        encoding="utf-8-sig"
                    ).splitlines()
                    if line.strip()
                )
            }
        except (OSError, json.JSONDecodeError, KeyError):
            previous_artifacts = {}
    allow_incomplete = any(
        str(item.get("path") or "").startswith("dry-run://")
        for payload in normalized_events
        for field in ("key_frames", "key_clips")
        for item in payload.get(field, [])
    )
    for payload in normalized_events:
        event_uid = _event_uid_from_payload(archive_id, payload, groups)
        event_id = str(payload["event_id"])
        parent_event_id = str(payload.get("parent_event_id") or "")
        for artifact_type, field in (("key_frame", "key_frames"), ("key_clip", "key_clips")):
            for item in payload.get(field, []):
                artifact_uid = str(
                    item.get("artifact_uid")
                    or stable_artifact_uid(
                        event_uid,
                        artifact_type,
                        str(item.get("view_id") or item.get("view_role") or "unknown"),
                    )
                )
                artifact_jobs.append(
                    (
                        root,
                        event_uid,
                        event_id,
                        parent_event_id,
                        artifact_type,
                        item,
                        allow_incomplete,
                        previous_artifacts.get(artifact_uid),
                    )
                )
    hash_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, min(hash_workers, len(artifact_jobs) or 1))) as pool:
        artifact_records = list(pool.map(lambda args: _artifact_record(*args), artifact_jobs))
    hash_duration_seconds = time.perf_counter() - hash_started
    artifact_records.sort(key=lambda item: item["artifact_uid"])
    evidence_records = _evidence_records(archive_id, events, groups, infos)
    decision_records = _decision_receipt_records(json_root)
    change_records = physical_change_records(archive_id, normalized_events)
    expected_evidence_uids = {
        str(reference["evidence_uid"])
        for payload in normalized_events
        for reference in (
            ((payload.get("provenance") or {}).get("index") or {}).get(
                "evidence_refs", []
            )
        )
    }
    registered_evidence_uids = {
        str(item["evidence_uid"]) for item in evidence_records
    }
    missing_evidence_uids = sorted(expected_evidence_uids - registered_evidence_uids)
    if missing_evidence_uids:
        raise RuntimeError(
            "Evidence registry is incomplete: "
            + ", ".join(missing_evidence_uids[:10])
        )

    artifact_registry = json_root / ARTIFACT_REGISTRY_NAME
    evidence_registry = json_root / EVIDENCE_REGISTRY_NAME
    decision_registry = json_root / DECISION_REGISTRY_NAME
    physical_change_registry = json_root / PHYSICAL_CHANGE_REGISTRY_NAME
    database = json_root / INDEX_DB_NAME
    registry_started = time.perf_counter()
    _write_jsonl(artifact_registry, artifact_records)
    _write_jsonl(evidence_registry, evidence_records)
    _write_jsonl(decision_registry, decision_records)
    _write_jsonl(physical_change_registry, change_records)
    registry_duration_seconds = time.perf_counter() - registry_started
    sqlite_started = time.perf_counter()
    fts5_enabled = _build_sqlite(
        database,
        archive_id,
        normalized_events,
        artifact_records,
        evidence_records,
        decision_records,
        change_records,
    )
    sqlite_duration_seconds = time.perf_counter() - sqlite_started
    category_index = root / "Key-Materials" / "Key-Material-Category-Index.json"
    category_index_relative = "../Key-Materials/Key-Material-Category-Index.json"
    recall_eval = root / "JSON-Config-Files" / "key_material_recall_eval.json"
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "authority": "JSON files remain canonical; SQLite and JSONL registries are rebuildable derivatives",
        "archive_id": archive_id,
        "event_uid_format": "{archive_id}:{parent_event_id}:{event_id}",
        "counts": {
            "key_events": len(normalized_events),
            "artifacts": len(artifact_records),
            "evidence": len(evidence_records),
            "decision_receipts": len(decision_records),
            "physical_changes": len(change_records),
            "artifact_hashes_reused": sum(
                item.get("hash_reused") is True for item in artifact_records
            ),
            "artifact_hashes_computed": sum(
                item.get("hash_reused") is not True
                and item.get("integrity_status") == "verified"
                for item in artifact_records
            ),
        },
        "fts5_enabled": fts5_enabled,
        "performance": {
            "artifact_hash_and_stat_seconds": round(hash_duration_seconds, 6),
            "registry_write_seconds": round(registry_duration_seconds, 6),
            "sqlite_build_seconds": round(sqlite_duration_seconds, 6),
            "total_seconds_before_manifest_write": round(
                time.perf_counter() - started, 6
            ),
        },
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reason": "deterministic local indexing; no model call",
        },
        "validation": {
            "passed": not missing_evidence_uids,
            "expected_evidence_reference_count": len(expected_evidence_uids),
            "registered_evidence_reference_count": len(
                expected_evidence_uids & registered_evidence_uids
            ),
            "missing_evidence_uids": missing_evidence_uids,
            "all_artifacts_have_integrity_receipts": all(
                item.get("integrity_status")
                in {"verified", "virtual_dry_run", "dry_run_sidecar_absent"}
                for item in artifact_records
            ),
        },
        "files": {
            "database": INDEX_DB_NAME,
            "artifact_registry": ARTIFACT_REGISTRY_NAME,
            "evidence_registry": EVIDENCE_REGISTRY_NAME,
            "decision_receipt_registry": DECISION_REGISTRY_NAME,
            "physical_change_registry": PHYSICAL_CHANGE_REGISTRY_NAME,
            **(
                {"key_material_category_index": category_index_relative}
                if category_index.is_file()
                else {}
            ),
            **(
                {"key_material_recall_eval": "key_material_recall_eval.json"}
                if recall_eval.is_file()
                else {}
            ),
        },
        "integrity": {
            INDEX_DB_NAME: {"size_bytes": database.stat().st_size, "sha256": _sha256(database)},
            ARTIFACT_REGISTRY_NAME: {
                "size_bytes": artifact_registry.stat().st_size,
                "sha256": _sha256(artifact_registry),
            },
            EVIDENCE_REGISTRY_NAME: {
                "size_bytes": evidence_registry.stat().st_size,
                "sha256": _sha256(evidence_registry),
            },
            DECISION_REGISTRY_NAME: {
                "size_bytes": decision_registry.stat().st_size,
                "sha256": _sha256(decision_registry),
            },
            PHYSICAL_CHANGE_REGISTRY_NAME: {
                "size_bytes": physical_change_registry.stat().st_size,
                "sha256": _sha256(physical_change_registry),
            },
            **(
                {
                    "key_material_recall_eval.json": {
                        "size_bytes": recall_eval.stat().st_size,
                        "sha256": _sha256(recall_eval),
                    }
                }
                if recall_eval.is_file()
                else {}
            ),
            **(
                {
                    category_index_relative: {
                        "size_bytes": category_index.stat().st_size,
                        "sha256": _sha256(category_index),
                    }
                }
                if category_index.is_file()
                else {}
            ),
        },
    }
    _atomic_write_text(
        json_root / INDEX_MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    return manifest


def _fts_query(value: str) -> str:
    terms = re.findall(r"[^\s\"'():*]+", value, flags=re.UNICODE)
    return " AND ".join(f'"{term.replace(chr(34), "")}"' for term in terms)


def search_archive_index(
    root: Path,
    *,
    query: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    cross_view: bool | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    after_peak_us: int | None = None,
    after_event_uid: str | None = None,
    limit: int = 51,
) -> list[dict[str, Any]]:
    database = root / "JSON-Config-Files" / INDEX_DB_NAME
    if not database.is_file():
        return []
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        conditions: list[str] = []
        parameters: list[Any] = []
        join = ""
        fts = _fts_query(query or "")
        use_fts = bool(fts and len((query or "").strip()) >= 3)
        if use_fts:
            join = " JOIN key_events_fts f ON f.event_uid = k.event_uid"
            conditions.append("key_events_fts MATCH ?")
            parameters.append(fts)
        elif query:
            conditions.append("k.search_text LIKE ?")
            parameters.append(f"%{query}%")
        if action_type:
            conditions.append("k.action_type = ?")
            parameters.append(action_type)
        if parent_event_id:
            conditions.append("k.parent_event_id = ?")
            parameters.append(parent_event_id)
        if cross_view is not None:
            conditions.append("k.cross_view_supported = ?")
            parameters.append(int(cross_view))
        if start_us is not None:
            conditions.append("k.end_us >= ?")
            parameters.append(int(start_us))
        if end_us is not None:
            conditions.append("k.start_us <= ?")
            parameters.append(int(end_us))
        if after_peak_us is not None and after_event_uid is not None:
            conditions.append("(k.peak_timestamp_us > ? OR (k.peak_timestamp_us = ? AND k.event_uid > ?))")
            parameters.extend((after_peak_us, after_peak_us, after_event_uid))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            "SELECT k.* FROM key_events k"
            f"{join}{where} ORDER BY k.peak_timestamp_us, k.event_uid LIMIT ?"
        )
        parameters.append(max(1, int(limit)))
        try:
            rows = connection.execute(sql, parameters).fetchall()
        except sqlite3.OperationalError:
            if not use_fts:
                raise
            conditions = [item for item in conditions if item != "key_events_fts MATCH ?"]
            parameters = parameters[1:-1]
            conditions.append("k.search_text LIKE ?")
            parameters.append(f"%{query}%")
            parameters.append(max(1, int(limit)))
            where = f" WHERE {' AND '.join(conditions)}"
            rows = connection.execute(
                "SELECT k.* FROM key_events k"
                f"{where} ORDER BY k.peak_timestamp_us, k.event_uid LIMIT ?",
                parameters,
            ).fetchall()
        results = []
        for row in rows:
            payload = json.loads(row["event_json"])
            artifact_rows = connection.execute(
                "SELECT artifact_json FROM artifacts WHERE event_uid = ? ORDER BY artifact_type, view_id",
                (row["event_uid"],),
            ).fetchall()
            payload.update(
                {
                    "event_uid": row["event_uid"],
                    "archive_id": row["archive_id"],
                    "artifact_references": [
                        json.loads(item["artifact_json"]) for item in artifact_rows
                    ],
                    "key_material_json_reference": {
                        "path": "Key-Materials/Key-Materials-Model-Understanding.json",
                        "event_id": row["event_id"],
                    },
                }
            )
            results.append(payload)
        return results
    finally:
        connection.close()


def search_physical_changes(
    root: Path,
    *,
    object_id: str | None = None,
    object_role: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    after_peak_us: int | None = None,
    after_change_uid: str | None = None,
    limit: int = 51,
) -> list[dict[str, Any]]:
    """Search explicit physical state transitions derived from canonical event JSON."""

    database = root / "JSON-Config-Files" / INDEX_DB_NAME
    if not database.is_file():
        return []
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        conditions: list[str] = []
        parameters: list[Any] = []
        for field, value in (
            ("object_id", object_id),
            ("object_role", object_role),
            ("action_type", action_type),
            ("parent_event_id", parent_event_id),
        ):
            if value:
                conditions.append(f"{field} = ?")
                parameters.append(str(value))
        if start_us is not None:
            conditions.append("end_us >= ?")
            parameters.append(int(start_us))
        if end_us is not None:
            conditions.append("start_us <= ?")
            parameters.append(int(end_us))
        if after_peak_us is not None and after_change_uid is not None:
            conditions.append(
                "(peak_timestamp_us > ? OR (peak_timestamp_us = ? AND change_uid > ?))"
            )
            parameters.extend((after_peak_us, after_peak_us, after_change_uid))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.append(max(1, min(int(limit), 1000)))
        rows = connection.execute(
            "SELECT change_json FROM physical_changes"
            f"{where} ORDER BY peak_timestamp_us, change_uid LIMIT ?",
            parameters,
        ).fetchall()
        return [json.loads(row["change_json"]) for row in rows]
    finally:
        connection.close()


def search_decision_receipts(
    root: Path,
    *,
    rule_id: str | None = None,
    verdict: str | None = None,
    subject_id: str | None = None,
    query: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Search deterministic quality decisions by rule, outcome, or subject."""

    database = root / "JSON-Config-Files" / INDEX_DB_NAME
    if not database.is_file():
        return []
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        conditions: list[str] = []
        parameters: list[Any] = []
        join = ""
        if subject_id:
            join = (
                " JOIN decision_subjects s"
                " ON s.decision_id = d.decision_id"
            )
            conditions.append("s.subject_id = ?")
            parameters.append(str(subject_id))
        if rule_id:
            conditions.append("d.rule_id = ?")
            parameters.append(str(rule_id))
        if verdict:
            conditions.append("d.verdict = ?")
            parameters.append(str(verdict))
        if query:
            conditions.append("d.search_text LIKE ?")
            parameters.append(f"%{query}%")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.append(max(1, min(int(limit), 1000)))
        rows = connection.execute(
            "SELECT d.receipt_json FROM decision_receipts d"
            f"{join}{where} ORDER BY d.rule_id, d.decision_id LIMIT ?",
            parameters,
        ).fetchall()
        return [json.loads(row["receipt_json"]) for row in rows]
    finally:
        connection.close()


def get_indexed_event(root: Path, event_uid: str) -> dict[str, Any] | None:
    database = root / "JSON-Config-Files" / INDEX_DB_NAME
    if not database.is_file():
        return None
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM key_events WHERE event_uid = ?", (event_uid,)
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["event_json"])
        artifact_rows = connection.execute(
            "SELECT artifact_json FROM artifacts WHERE event_uid = ? ORDER BY artifact_type, view_id",
            (event_uid,),
        ).fetchall()
        payload.update(
            {
                "event_uid": row["event_uid"],
                "archive_id": row["archive_id"],
                "artifact_references": [
                    json.loads(item["artifact_json"]) for item in artifact_rows
                ],
                "key_material_json_reference": {
                    "path": "Key-Materials/Key-Materials-Model-Understanding.json",
                    "event_id": row["event_id"],
                },
            }
        )
        return payload
    finally:
        connection.close()


def get_indexed_evidence(root: Path, evidence_uid: str) -> dict[str, Any] | None:
    database = root / "JSON-Config-Files" / INDEX_DB_NAME
    if not database.is_file():
        return None
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT evidence_json FROM evidence WHERE evidence_uid = ?", (evidence_uid,)
        ).fetchone()
        return json.loads(row["evidence_json"]) if row is not None else None
    finally:
        connection.close()
