from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schemas import ViewRole


DEVICE_REGISTRY_SCHEMA_VERSION = "visioncortex-device-registry/1"
ROLE_RECEIPT_SCHEMA_VERSION = "visioncortex-view-role-resolution/1"
DEFAULT_REGISTRY_PATH = (
    Path(__file__).with_name("templates") / "VC-DEVICE-REGISTRY-V1.json"
)

_INDEX_ROLE_MAP = {
    "first": ViewRole.FIRST_PERSON,
    "first_person": ViewRole.FIRST_PERSON,
    "ego": ViewRole.FIRST_PERSON,
    "wearable": ViewRole.FIRST_PERSON,
    "front": ViewRole.THIRD_PERSON,
    "side": ViewRole.THIRD_PERSON,
    "third": ViewRole.THIRD_PERSON,
    "third_person": ViewRole.THIRD_PERSON,
    "fixed": ViewRole.THIRD_PERSON,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_index_role(value: str | None) -> ViewRole | None:
    return _INDEX_ROLE_MAP.get(str(value or "").strip().lower())


def load_device_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not registry_path.is_absolute():
        registry_path = (Path.cwd() / registry_path).resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != DEVICE_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported device registry schema: {payload.get('schema_version')}"
        )
    if not isinstance(payload.get("devices", {}), dict):
        raise ValueError("Device registry devices must be an object")
    if not isinstance(payload.get("experiment_role_overrides", {}), dict):
        raise ValueError("Device registry experiment_role_overrides must be an object")
    payload["provenance"] = {
        "path": str(registry_path),
        "sha256": _sha256(registry_path),
    }
    return payload


def resolve_view_role(
    registry: dict[str, Any],
    experiment_id: str,
    camera_key: str,
    index_camera_view: str | None,
) -> dict[str, Any]:
    index_role = normalized_index_role(index_camera_view)
    device = (registry.get("devices") or {}).get(camera_key) or {}
    expected_raw = device.get("expected_role")
    expected_role = ViewRole(expected_raw) if expected_raw else None
    override = (
        ((registry.get("experiment_role_overrides") or {}).get(experiment_id) or {}).get(
            camera_key
        )
        or {}
    )
    override_raw = override.get("role")
    override_role = ViewRole(override_raw) if override_raw else None

    conflict_reasons: list[str] = []
    if index_role is None:
        conflict_reasons.append("index_role_unrecognized")
    if expected_role is not None and index_role is not None and expected_role != index_role:
        conflict_reasons.append("index_registry_role_mismatch")
    if override_role is not None and index_role is not None and override_role != index_role:
        conflict_reasons.append("experiment_override_differs_from_index")

    approved_override = bool(
        override_role is not None
        and str(override.get("status") or "").lower()
        in {"approved", "policy_override"}
        and str(override.get("reason") or "").strip()
    )
    if override_role is not None and not approved_override:
        conflict_reasons.append("experiment_override_not_approved")

    if override_role is not None and approved_override:
        resolved_role = override_role
        resolution_source = "approved_experiment_override"
    elif index_role is not None:
        resolved_role = index_role
        resolution_source = "experiment_record_index"
    elif expected_role is not None:
        resolved_role = expected_role
        resolution_source = "device_registry_fallback"
    else:
        resolved_role = None
        resolution_source = "unresolved"

    blocking_reasons = [
        reason
        for reason in conflict_reasons
        if reason
        in {
            "index_role_unrecognized",
            "index_registry_role_mismatch",
            "experiment_override_not_approved",
        }
    ]
    if approved_override:
        blocking_reasons = [
            reason for reason in blocking_reasons if reason != "index_registry_role_mismatch"
        ]
    status = (
        "blocking_conflict"
        if blocking_reasons or resolved_role is None
        else "approved_override"
        if approved_override and conflict_reasons
        else "resolved"
    )
    return {
        "schema_version": ROLE_RECEIPT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "camera_key": camera_key,
        "display_name": device.get("display_name") or camera_key,
        "workstation_id": device.get("workstation_id"),
        "index_camera_view": index_camera_view,
        "index_role": index_role.value if index_role else None,
        "registry_expected_role": expected_role.value if expected_role else None,
        "resolved_role": resolved_role.value if resolved_role else None,
        "resolution_source": resolution_source,
        "status": status,
        "conflict_reasons": conflict_reasons,
        "blocking_reasons": blocking_reasons,
        "override": {
            "role": override_role.value if override_role else None,
            "status": override.get("status"),
            "reason": override.get("reason"),
            "source": override.get("source"),
        }
        if override
        else None,
        "registry_provenance": registry.get("provenance"),
    }

