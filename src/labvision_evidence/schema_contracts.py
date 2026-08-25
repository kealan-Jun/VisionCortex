from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pathing import archive_contains


CONTRACT_MANIFEST_VERSION = "visioncortex-schema-contract-manifest/1.0.0"


@dataclass(frozen=True)
class JsonContract:
    contract_id: str
    relative_path: str
    required: bool
    required_fields: tuple[str, ...]
    schema_prefix: str | None = None
    root_types: tuple[type, ...] = (dict,)


ARCHIVE_CONTRACTS = (
    JsonContract(
        "evidence-package",
        "JSON-Config-Files/evidence_package.json",
        True,
        ("schema_version",),
    ),
    JsonContract(
        "evidence-package-eval",
        "JSON-Config-Files/evidence_package_eval.json",
        True,
        ("passed",),
    ),
    JsonContract(
        "quality-acceptance",
        "JSON-Config-Files/quality_acceptance.json",
        True,
        ("schema_version", "status", "experiment_boundaries", "key_materials"),
        "visioncortex-quality-acceptance/",
    ),
    JsonContract(
        "run-metrics",
        "JSON-Config-Files/run_metrics.json",
        True,
        ("stage_durations", "tokens"),
    ),
    JsonContract(
        "audit-layer",
        "JSON-Config-Files/audit_layer.json",
        True,
        ("events", "segments", "experiment_groups"),
    ),
    JsonContract(
        "key-material-event-ledger",
        "Key-Materials/Key-Materials-Model-Understanding.json",
        True,
        (),
        root_types=(list,),
    ),
    JsonContract(
        "continuous-action-state",
        "JSON-Config-Files/continuous_action_state_ledger.json",
        False,
        ("schema_version", "policy", "cv_acceptance_mutated", "receipts"),
        "visioncortex-continuous-action-state-ledger/",
    ),
    JsonContract(
        "evidence-index-manifest",
        "JSON-Config-Files/evidence_index_manifest.json",
        False,
        ("schema_version", "counts", "files", "integrity", "validation"),
        "visioncortex-evidence-index/",
    ),
    JsonContract(
        "physical-change-log",
        "JSON-Config-Files/physical_change_log.json",
        False,
        (),
        root_types=(list,),
    ),
)


EVENT_REQUIRED_FIELDS = (
    "event_id",
    "action_type",
    "start_us",
    "end_us",
    "peak_timestamp_us",
    "objects",
    "observations",
    "cross_view_associations",
    "decision",
    "scores",
    "key_frames",
    "key_clips",
    "evidence_ids",
    "provenance",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_event_contract(event: dict[str, Any]) -> list[str]:
    failures = [field for field in EVENT_REQUIRED_FIELDS if field not in event]
    if event.get("start_us") is not None and event.get("end_us") is not None:
        if int(event["end_us"]) < int(event["start_us"]):
            failures.append("end_us_before_start_us")
    return failures


def _validate_material_references(
    archive_root: Path, event: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    for collection_name in ("key_frames", "key_clips"):
        for index, reference in enumerate(event.get(collection_name) or []):
            if not isinstance(reference, dict) or not reference.get("path"):
                failures.append(f"{collection_name}[{index}].path_missing")
                continue
            value = str(reference["path"])
            if value.startswith("dry-run://"):
                continue
            path = (archive_root / value).resolve()
            if not archive_contains(path, archive_root):
                failures.append(f"{collection_name}[{index}].path_outside_archive")
                continue
            if not path.is_file():
                failures.append(f"{collection_name}[{index}].file_missing")
    return failures


def inspect_archive_contracts(archive_root: Path) -> dict[str, Any]:
    archive_root = archive_root.resolve()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for contract in ARCHIVE_CONTRACTS:
        path = archive_root / contract.relative_path
        result = {
            "contract_id": contract.contract_id,
            "path": contract.relative_path,
            "required": contract.required,
            "present": path.is_file(),
            "status": "missing",
            "schema_version": None,
            "missing_fields": [],
        }
        if not path.is_file():
            target = failures if contract.required else warnings
            target.append(
                {
                    "contract_id": contract.contract_id,
                    "reason": "required_file_missing" if contract.required else "optional_file_missing",
                    "path": contract.relative_path,
                }
            )
            results.append(result)
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError, TypeError) as exc:
            result["status"] = "invalid_json"
            failures.append(
                {
                    "contract_id": contract.contract_id,
                    "reason": "invalid_json",
                    "path": contract.relative_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            results.append(result)
            continue
        if not isinstance(payload, contract.root_types):
            result["status"] = "invalid_root_type"
            failures.append(
                {
                    "contract_id": contract.contract_id,
                    "reason": "invalid_root_type",
                    "path": contract.relative_path,
                    "expected_root_types": [item.__name__ for item in contract.root_types],
                    "observed_root_type": type(payload).__name__,
                }
            )
            results.append(result)
            continue
        result["schema_version"] = payload.get("schema_version") if isinstance(payload, dict) else None
        missing = (
            [field for field in contract.required_fields if field not in payload]
            if isinstance(payload, dict)
            else []
        )
        result["missing_fields"] = missing
        schema_ok = (
            contract.schema_prefix is None
            or str(payload.get("schema_version") or "").startswith(contract.schema_prefix)
        )
        if missing or not schema_ok:
            result["status"] = "contract_failed"
            failures.append(
                {
                    "contract_id": contract.contract_id,
                    "reason": "contract_failed",
                    "path": contract.relative_path,
                    "missing_fields": missing,
                    "schema_prefix": contract.schema_prefix,
                    "observed_schema_version": payload.get("schema_version"),
                }
            )
        else:
            result["status"] = "passed"
        if contract.contract_id == "key-material-event-ledger" and isinstance(payload, list):
            event_failures: list[dict[str, Any]] = []
            seen: set[str] = set()
            for index, event in enumerate(payload):
                if not isinstance(event, dict):
                    event_failures.append(
                        {"index": index, "event_id": None, "failures": ["event_must_be_object"]}
                    )
                    continue
                event_id = str(event.get("event_id") or "")
                issues = validate_event_contract(event)
                issues.extend(_validate_material_references(archive_root, event))
                if event_id in seen:
                    issues.append("duplicate_event_id")
                seen.add(event_id)
                if issues:
                    event_failures.append(
                        {"index": index, "event_id": event_id or None, "failures": issues}
                    )
            result["event_count"] = len(payload)
            result["event_failure_count"] = len(event_failures)
            if event_failures:
                result["status"] = "contract_failed"
                failures.append(
                    {
                        "contract_id": contract.contract_id,
                        "reason": "event_contract_failed",
                        "path": contract.relative_path,
                        "event_failures": event_failures,
                    }
                )
        results.append(result)
    return {
        "schema_version": CONTRACT_MANIFEST_VERSION,
        "archive_root": str(archive_root),
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "contract_count": len(results),
        "required_contract_count": sum(contract.required for contract in ARCHIVE_CONTRACTS),
        "failures": failures,
        "warnings": warnings,
        "contracts": results,
        "event_contract": {
            "schema_version": "visioncortex-key-material-event/1.0.0",
            "required_fields": list(EVENT_REQUIRED_FIELDS),
        },
    }


def write_archive_contract_manifest(archive_root: Path) -> Path:
    payload = inspect_archive_contracts(archive_root)
    path = archive_root / "JSON-Config-Files" / "schema_contract_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path
