from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema_contracts import inspect_archive_contracts


REPLAY_SNAPSHOT_VERSION = "visioncortex-archive-regression-snapshot/1.0.0"
REPLAY_RESULT_VERSION = "visioncortex-archive-regression-result/1.0.0"


def _read(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive_regression_snapshot(archive_root: Path) -> dict[str, Any]:
    root = archive_root.resolve()
    json_root = root / "JSON-Config-Files"
    quality = _read(json_root / "quality_acceptance.json", {})
    evaluation = _read(json_root / "evidence_package_eval.json", {})
    audit = _read(json_root / "audit_layer.json", {})
    boundary = quality.get("experiment_boundaries") or {}
    materials = quality.get("key_materials") or {}
    selected_ids = {
        str(item.get("event_id"))
        for item in materials.get("events") or []
        if item.get("event_id")
    }
    quarantined_event_ids: set[str] = set()
    for receipt in audit.get("formal_segment_receipts") or []:
        if str(receipt.get("decision") or "").startswith("quarantined"):
            quarantined_event_ids.update(str(value) for value in receipt.get("event_ids") or [])
    matches = [
        {
            "baseline_id": item.get("baseline_id"),
            "predicted_group_id": item.get("predicted_group_id"),
            "start_error_seconds": item.get("start_error_seconds"),
            "end_error_seconds": item.get("end_error_seconds"),
            "boundary_within_tolerance": item.get("boundary_within_tolerance"),
            "continuity_type": item.get("predicted_continuity_type"),
            "atomic_experiment_count": item.get("predicted_atomic_experiment_count"),
        }
        for item in boundary.get("matches") or []
    ]
    hashes = {}
    for name in (
        "quality_acceptance.json",
        "evidence_package_eval.json",
        "audit_layer.json",
        "run_metrics.json",
        "key_material_model_understanding.json",
    ):
        hashes[name] = _sha256(json_root / name)
    return {
        "schema_version": REPLAY_SNAPSHOT_VERSION,
        "archive_root": str(root),
        "quality_status": quality.get("status"),
        "evidence_package_passed": bool(evaluation.get("passed")),
        "experiment_group_count": int(
            boundary.get("predicted_experiment_count")
            or evaluation.get("experiment_group_count")
            or 0
        ),
        "key_event_count": int(materials.get("event_count") or 0),
        "action_counts": dict(materials.get("action_counts") or {}),
        "media_complete_count": int(materials.get("media_complete_count") or 0),
        "dual_view_material_count": int(evaluation.get("dual_view_material_count") or 0),
        "cross_view_supported_count": int(materials.get("cross_view_supported_count") or 0),
        "cross_view_supported_rate": float(materials.get("cross_view_supported_rate") or 0.0),
        "quarantined_event_ids": sorted(quarantined_event_ids),
        "selected_quarantined_event_ids": sorted(selected_ids & quarantined_event_ids),
        "boundary_evaluated": bool(boundary.get("evaluated")),
        "boundary_passed": bool(boundary.get("passed")) if boundary.get("evaluated") else None,
        "boundary_matches": matches,
        "contract_manifest": inspect_archive_contracts(root),
        "source_hashes": hashes,
        "source_policy": {
            "video_files_opened": 0,
            "clock_csv_files_opened": 0,
            "mode": "durable_json_ledger_replay",
        },
    }


def compare_archive_snapshot(
    snapshot: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    gates = baseline.get("gates") or {}
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    expected_groups = gates.get("experiment_group_count")
    if expected_groups is not None:
        check(
            "experiment_group_count",
            snapshot.get("experiment_group_count") == int(expected_groups),
            snapshot.get("experiment_group_count"),
            expected_groups,
        )
    minimum_events = gates.get("minimum_key_event_count")
    if minimum_events is not None:
        check(
            "minimum_key_event_count",
            int(snapshot.get("key_event_count") or 0) >= int(minimum_events),
            snapshot.get("key_event_count"),
            f">={minimum_events}",
        )
    check(
        "evidence_package_passed",
        bool(snapshot.get("evidence_package_passed")),
        snapshot.get("evidence_package_passed"),
        True,
    )
    check(
        "quarantined_event_leakage",
        not snapshot.get("selected_quarantined_event_ids"),
        snapshot.get("selected_quarantined_event_ids"),
        [],
    )
    check(
        "media_complete",
        int(snapshot.get("media_complete_count") or 0)
        >= int(snapshot.get("key_event_count") or 0),
        snapshot.get("media_complete_count"),
        f">={snapshot.get('key_event_count')}",
    )
    check(
        "schema_contracts",
        bool((snapshot.get("contract_manifest") or {}).get("passed")),
        (snapshot.get("contract_manifest") or {}).get("status"),
        "passed",
    )
    expected_structure = gates.get("boundary_structure") or {}
    observed_structure = {
        str(item.get("baseline_id")): {
            "continuity_type": item.get("continuity_type"),
            "atomic_experiment_count": item.get("atomic_experiment_count"),
        }
        for item in snapshot.get("boundary_matches") or []
    }
    for baseline_id, expected in expected_structure.items():
        check(
            f"boundary_structure:{baseline_id}",
            observed_structure.get(baseline_id) == expected,
            observed_structure.get(baseline_id),
            expected,
        )
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": REPLAY_RESULT_VERSION,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "baseline_id": baseline.get("baseline_id"),
        "checks": checks,
        "failures": [item for item in checks if not item["passed"]],
        "snapshot": snapshot,
    }
