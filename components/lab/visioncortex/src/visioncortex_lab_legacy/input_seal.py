from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import RunManifest


INPUT_SEAL_SCHEMA_VERSION = "visioncortex-input-seal/1"


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def build_input_seal(
    manifest: RunManifest,
    *,
    source_mode: str,
    sources: list[dict[str, Any]],
    role_resolution: dict[str, Any],
    copied_source_bytes: int,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common immutable-input receipt used by browser and NAS ingest."""

    manifest_payload = manifest.model_dump(mode="json")
    identity_algorithms: dict[str, int] = {}
    for source in sources:
        algorithm = str(
            source.get("content_hash_algorithm")
            or source.get("source_fingerprint_algorithm")
            or "path_size_mtime"
        )
        identity_algorithms[algorithm] = identity_algorithms.get(algorithm, 0) + 1
    payload: dict[str, Any] = {
        "schema_version": INPUT_SEAL_SCHEMA_VERSION,
        "sealed_at": datetime.now().astimezone().isoformat(),
        "experiment_id": manifest.experiment_id,
        "source_mode": source_mode,
        "copied_source_bytes": int(copied_source_bytes),
        "manifest": manifest_payload,
        "manifest_sha256": hashlib.sha256(_canonical_json(manifest_payload)).hexdigest(),
        "source_count": len(sources),
        "identity_algorithms": identity_algorithms,
        "sources": sources,
        "role_resolution": role_resolution,
        "prequeue_preflight": preflight,
    }
    payload["seal_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def write_input_seal(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def verify_input_seal(payload: dict[str, Any]) -> bool:
    if payload.get("schema_version") != INPUT_SEAL_SCHEMA_VERSION:
        return False
    expected = str(payload.get("seal_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "seal_sha256"}
    observed = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return len(expected) == 64 and observed == expected
