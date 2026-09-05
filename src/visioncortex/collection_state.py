from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .provenance import verify_run_provenance
from .schema_contracts import inspect_archive_contracts
from .storage import (
    archive_promotion_in_progress,
    read_current_release_pointer,
    verify_current_release,
)


COLLECTION_STATE_SCHEMA_VERSION = "visioncortex-collection-processing-registry/1"
_LEDGER_LOCK = threading.Lock()


def _accepted_formal_archive(path: Path) -> bool:
    """Return true only for a still-present archive that passed all gates."""

    try:
        if archive_promotion_in_progress(path):
            return False
        quality = json.loads(
            (path / "JSON-Config-Files" / "quality_acceptance.json").read_text(
                encoding="utf-8-sig"
            )
        )
        evidence = json.loads(
            (path / "JSON-Config-Files" / "evidence_package_eval.json").read_text(
                encoding="utf-8-sig"
            )
        )
        report_paths = list(
            (path / "Lab-Daily-Reports").glob("*/Daily-Report-Eval.json")
        )
        legacy_gates_passed = bool(
            path.is_dir()
            and quality.get("passed") is True
            and evidence.get("passed") is True
            and report_paths
            and all(
                json.loads(item.read_text(encoding="utf-8-sig")).get("passed")
                is True
                for item in report_paths
            )
        )
        if not legacy_gates_passed:
            return False
        pointer = read_current_release_pointer(path)
        if pointer is None:
            return True
        return bool(
            inspect_archive_contracts(
                path, require_release_contracts=True
            ).get("passed") is True
            and verify_run_provenance(path).get("passed") is True
            and verify_current_release(path).get("passed") is True
        )
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return False


def _effective_collection_entry(
    config: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    """Keep an accepted formal archive authoritative across failed retries.

    The latest retry remains visible in ``latest_run_*`` while catalog
    disposition uses the still-valid formal package.  This prevents a failed
    replacement attempt from silently re-queuing an already delivered
    experiment.
    """

    if str(entry.get("state") or "") == "archived":
        return entry
    archived_entries = [
        item
        for item in entry.get("history") or []
        if str(item.get("state") or "") == "archived"
    ]
    if not archived_entries:
        return entry
    archive_root = Path(config["storage"]["archive_root"])
    for archived in reversed(archived_entries):
        configured_path = str(
            (archived.get("details") or {}).get("formal_archive") or ""
        ).strip()
        candidates = []
        if configured_path:
            candidate = Path(configured_path)
            candidates.append(
                candidate if candidate.is_absolute() else archive_root / candidate
            )
        archive_name = str(archived.get("archive_name") or "").strip()
        if archive_name:
            candidates.append(archive_root / archive_name)
        accepted_path = next(
            (candidate for candidate in candidates if _accepted_formal_archive(candidate)),
            None,
        )
        if accepted_path is None:
            continue
        return {
            **entry,
            "state": "archived",
            "archive_name": archived.get("archive_name"),
            "run_id": archived.get("run_id"),
            "updated_at": archived.get("updated_at"),
            "details": {
                **dict(archived.get("details") or {}),
                "formal_archive": str(accepted_path),
            },
            "latest_run_state": entry.get("state"),
            "latest_run_id": entry.get("run_id"),
            "latest_run_updated_at": entry.get("updated_at"),
            "latest_run_details": dict(entry.get("details") or {}),
            "effective_state_reason": (
                "accepted_formal_archive_preserved_across_retry"
            ),
        }
    return entry


def collection_state_path(config: dict[str, Any]) -> Path:
    return (
        Path(config["storage"]["archive_root"])
        / ".VisionCortex-System"
        / "Collection-Processing-Registry.json"
    )


def read_collection_states(config: dict[str, Any]) -> dict[str, Any]:
    path = collection_state_path(config)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        payload = {}
    except (OSError, ValueError):
        return {
            "schema_version": COLLECTION_STATE_SCHEMA_VERSION,
            "path": str(path),
            "available": False,
            "collections": {},
        }
    collections = {
        str(experiment_id): _effective_collection_entry(config, dict(entry))
        for experiment_id, entry in (payload.get("collections") or {}).items()
    }
    return {
        "schema_version": COLLECTION_STATE_SCHEMA_VERSION,
        "path": str(path),
        "available": True,
        "collections": collections,
        "updated_at": payload.get("updated_at"),
    }


def record_collection_state(
    config: dict[str, Any],
    source_experiment_id: str,
    *,
    archive_name: str,
    run_id: str,
    state: str,
    details: dict[str, Any] | None = None,
) -> Path:
    if state not in {"queued", "processing", "failed", "archived"}:
        raise ValueError(f"Unsupported collection processing state: {state}")
    path = collection_state_path(config)
    now = datetime.now().astimezone().isoformat()
    with _LEDGER_LOCK:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            payload = {
                "schema_version": COLLECTION_STATE_SCHEMA_VERSION,
                "collections": {},
            }
        if payload.get("schema_version") != COLLECTION_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported collection state registry: {payload.get('schema_version')}"
            )
        collections = payload.setdefault("collections", {})
        previous = collections.get(source_experiment_id) or {}
        history = list(previous.get("history") or [])
        entry = {
            "state": state,
            "archive_name": archive_name,
            "run_id": run_id,
            "updated_at": now,
            "details": details or {},
        }
        history.append(entry)
        collections[source_experiment_id] = {**entry, "history": history[-20:]}
        payload["updated_at"] = now
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    return path
