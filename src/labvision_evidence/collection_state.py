from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


COLLECTION_STATE_SCHEMA_VERSION = "visioncortex-collection-processing-registry/1"
_LEDGER_LOCK = threading.Lock()


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
    return {
        "schema_version": COLLECTION_STATE_SCHEMA_VERSION,
        "path": str(path),
        "available": True,
        "collections": payload.get("collections") or {},
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
