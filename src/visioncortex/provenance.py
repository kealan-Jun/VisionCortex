from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROVENANCE_SCHEMA_VERSION = "visioncortex-run-provenance/1"

_BOUND_ARTIFACTS = (
    "JSON-Config-Files/run_manifest.json",
    "JSON-Config-Files/evidence_package.json",
    "JSON-Config-Files/evidence_package_eval.json",
    "JSON-Config-Files/quality_acceptance.json",
    "JSON-Config-Files/run_metrics.json",
    "JSON-Config-Files/evidence_index_manifest.json",
    "JSON-Config-Files/daily_report_manifest.json",
    "JSON-Config-Files/professional_report_manifest.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    value = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _redact_config(value: Any, key: str = "") -> Any:
    normalized = key.lower()
    if any(
        marker in normalized
        for marker in ("password", "secret", "token", "api_key", "authorization")
    ):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(name): _redact_config(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_config(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def _repository_state(repository_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "commit_sha": commit,
            "tracked_worktree_clean": not bool(status),
            "status": "resolved",
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit_sha": None,
            "tracked_worktree_clean": None,
            "status": "unavailable",
        }


def write_run_provenance(
    archive_root: Path,
    config: dict[str, Any],
    model_certification: dict[str, Any],
    *,
    repository_root: Path,
) -> Path:
    """Bind one completed package to its code, configuration and receipts."""

    archive_root = archive_root.resolve()
    json_root = archive_root / "JSON-Config-Files"
    json_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    missing = []
    for relative in _BOUND_ARTIFACTS:
        path = archive_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        artifacts[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    input_seal = json_root / "Input-Manifests" / "input_seal.json"
    if input_seal.is_file():
        relative = "JSON-Config-Files/Input-Manifests/input_seal.json"
        artifacts[relative] = {
            "size_bytes": input_seal.stat().st_size,
            "sha256": _sha256(input_seal),
        }
    repository = _repository_state(repository_root)
    certification_required = bool(model_certification.get("required"))
    certification_valid = (
        model_certification.get("status") == "certified"
        if certification_required
        else model_certification.get("status") == "not_required_by_profile"
    )
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_scope": {
            "artifact_paths": "relative_to_archive_root",
            "generation_root": str(archive_root),
            "generation_root_may_be_release_staging": True,
        },
        "repository": repository,
        "configuration": {
            "sha256": _canonical_sha256(_redact_config(config)),
            "secret_values_included": False,
        },
        "model_certification": dict(model_certification),
        "artifacts": artifacts,
        "missing_required_artifacts": missing,
        "verification_policy": {
            "artifact_hash_algorithm": "sha256",
            "formal_model_certification_required": certification_required,
            "formal_checkout_must_be_identifiable": certification_required,
            "formal_checkout_must_be_clean": certification_required,
        },
    }
    payload["passed"] = bool(
        not missing
        and certification_valid
        and (
            not certification_required
            or (
                repository.get("status") == "resolved"
                and repository.get("tracked_worktree_clean") is True
            )
        )
    )
    payload["provenance_sha256"] = _canonical_sha256(payload)
    path = json_root / "run_provenance.json"
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def verify_run_provenance(archive_root: Path) -> dict[str, Any]:
    archive_root = archive_root.resolve()
    path = archive_root / "JSON-Config-Files" / "run_provenance.json"
    if not path.is_file():
        return {"passed": False, "failures": ["run_provenance_missing"]}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError) as exc:
        return {
            "passed": False,
            "failures": [f"run_provenance_invalid:{type(exc).__name__}"],
        }
    failures: list[str] = []
    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        failures.append("run_provenance_schema_mismatch")
    recorded_digest = payload.get("provenance_sha256")
    digest_payload = dict(payload)
    digest_payload.pop("provenance_sha256", None)
    if recorded_digest != _canonical_sha256(digest_payload):
        failures.append("run_provenance_digest_mismatch")
    artifacts = payload.get("artifacts") or {}
    for relative in _BOUND_ARTIFACTS:
        if relative not in artifacts:
            failures.append(f"required_artifact_receipt_missing:{relative}")
    input_seal_relative = "JSON-Config-Files/Input-Manifests/input_seal.json"
    if (archive_root / input_seal_relative).is_file() and input_seal_relative not in artifacts:
        failures.append("input_seal_receipt_missing")
    if payload.get("missing_required_artifacts"):
        failures.append("run_provenance_declares_missing_artifacts")
    for relative, receipt in artifacts.items():
        artifact = (archive_root / str(relative)).resolve()
        try:
            artifact.relative_to(archive_root)
        except ValueError:
            failures.append(f"artifact_outside_archive:{relative}")
            continue
        if not artifact.is_file():
            failures.append(f"artifact_missing:{relative}")
            continue
        if artifact.stat().st_size != int(receipt.get("size_bytes", -1)):
            failures.append(f"artifact_size_mismatch:{relative}")
            continue
        if _sha256(artifact) != str(receipt.get("sha256") or ""):
            failures.append(f"artifact_hash_mismatch:{relative}")
    if payload.get("passed") is not True:
        failures.append("run_provenance_not_passed")
    return {
        "passed": not failures,
        "failures": failures,
        "provenance_sha256": recorded_digest,
        "model_certification": payload.get("model_certification") or {},
        "repository": payload.get("repository") or {},
    }
