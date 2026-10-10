from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .actions import build_experiment_segments
from .grouping import (
    build_experiment_groups,
    normalize_experiment_segments,
    prepare_formal_experiment_segments,
    select_key_events,
)
from .schemas import ActionCandidate, EvidenceEvent, ExperimentSegment, RunManifest
from .schema_contracts import inspect_archive_contracts


REPLAY_SNAPSHOT_VERSION = "visioncortex-archive-regression-snapshot/1.0.0"
REPLAY_RESULT_VERSION = "visioncortex-archive-regression-result/1.0.0"
QUALITY_REPLAY_VERSION = "visioncortex-quality-ledger-replay/1.0.0"
QUALITY_LEDGER_INPUT_VERSION = "visioncortex-quality-ledger-inputs/1.0.0"


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


def _read_yaml(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _input_receipt(
    path: Path,
    root: Path,
    *,
    source_kind: str,
    payload_format: str,
) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "relative_path": path.relative_to(root).as_posix(),
        "format": payload_format,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_quality_ledger_inputs(
    archive_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load bounded JSON/YAML replay inputs without following source-media paths."""

    root = archive_root.resolve()
    json_root = root / "JSON-Config-Files"
    audit_path = json_root / "audit_layer.json"
    audit = _read(audit_path, {})
    if not audit:
        raise ValueError(
            "quality ledger replay input is missing or empty: "
            "JSON-Config-Files/audit_layer.json"
        )

    manifest_candidates = (
        (
            json_root / "run_manifest.json",
            "packaged_run_manifest_json",
            "json",
        ),
        (
            json_root / "Input-Manifests" / "manifest.yaml",
            "early_input_manifest_yaml",
            "yaml",
        ),
    )
    manifest_payload: dict[str, Any] = {}
    selected_manifest: tuple[Path, str, str] | None = None
    for path, source_kind, payload_format in manifest_candidates:
        payload = _read(path, {}) if payload_format == "json" else _read_yaml(path, {})
        if payload:
            if not isinstance(payload, dict):
                raise ValueError(
                    f"quality ledger manifest must contain an object: {path.relative_to(root)}"
                )
            manifest_payload = payload
            selected_manifest = (path, source_kind, payload_format)
            break
    if selected_manifest is None:
        attempted = ", ".join(
            path.relative_to(root).as_posix() for path, _, _ in manifest_candidates
        )
        raise ValueError(
            "quality ledger replay requires one run manifest; attempted bounded paths: "
            f"{attempted}"
        )

    manifest = RunManifest.model_validate(manifest_payload)
    events = audit.get("events") or []
    if not events:
        raise ValueError("audit_layer.json contains no evidence events")
    manifest_path, source_kind, payload_format = selected_manifest
    receipts = {
        "schema_version": QUALITY_LEDGER_INPUT_VERSION,
        "status": "passed",
        "archive_root": str(root),
        "inputs": {
            "audit_layer": _input_receipt(
                audit_path,
                root,
                source_kind="candidate_audit_ledger_json",
                payload_format="json",
            ),
            "run_manifest": _input_receipt(
                manifest_path,
                root,
                source_kind=source_kind,
                payload_format=payload_format,
            ),
        },
        "summary": {
            "experiment_id": manifest.experiment_id,
            "view_count": len(manifest.views),
            "event_count": len(events),
            "boundary_candidate_count": len(audit.get("boundary_candidates") or []),
            "stored_raw_segment_count": len(audit.get("raw_segments") or []),
        },
        "source_policy": {
            "video_files_opened": 0,
            "clock_csv_files_opened": 0,
            "model_calls": 0,
            "token_usage": 0,
            "mode": "bounded_quality_ledger_input_inspection",
        },
    }
    return audit, manifest_payload, receipts


def inspect_quality_ledger_inputs(archive_root: Path) -> dict[str, Any]:
    """Validate both required ledger inputs before any deterministic replay."""

    _, _, receipts = _load_quality_ledger_inputs(archive_root)
    return receipts


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


def replay_quality_decisions_from_ledgers(
    archive_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Re-run deterministic quality rules from JSON without opening source media.

    New archives persist the exact boundary candidates and can replay the raw
    segment builder. Older ledgers fall back to their stored raw segments and
    motion candidates; that mode is useful for QF2 diagnosis but is explicitly
    marked as degraded and cannot prove raw-boundary equivalence.
    """

    root = archive_root.resolve()
    json_root = root / "JSON-Config-Files"
    audit, manifest_payload, ledger_inputs = _load_quality_ledger_inputs(root)

    manifest = RunManifest.model_validate(manifest_payload)
    events = [EvidenceEvent.model_validate(item) for item in audit.get("events") or []]

    boundary_payload = audit.get("boundary_candidates") or []
    exact_raw_replay = bool(boundary_payload)
    if exact_raw_replay:
        boundary_candidates = [
            ActionCandidate.model_validate(item) for item in boundary_payload
        ]
        raw_boundary_receipts: list[dict[str, Any]] = []
        raw_segments = build_experiment_segments(
            events,
            manifest.views,
            config,
            coarse_windows=boundary_candidates,
            decision_receipts=raw_boundary_receipts,
        )
        raw_segment_source = "recomputed_from_persisted_boundary_candidates"
    else:
        motion = _read(json_root / "motion_probe_windows.json", {})
        boundary_candidates = [
            ActionCandidate.model_validate(item)
            for item in motion.get("candidates") or []
        ]
        raw_segments = [
            ExperimentSegment.model_validate(item)
            for item in audit.get("raw_segments") or []
        ]
        if not raw_segments:
            raise ValueError(
                "legacy ledger has neither persisted boundary candidates nor raw segments"
            )
        raw_boundary_receipts = []
        raw_segment_source = "stored_legacy_raw_segments"

    normalization_receipts: list[dict[str, Any]] = []
    normalized = normalize_experiment_segments(
        raw_segments,
        events,
        manifest.views,
        config,
        decision_receipts=normalization_receipts,
    )
    formal_segments, formal_receipts = prepare_formal_experiment_segments(
        normalized,
        events,
        manifest.views,
        boundary_candidates,
        config,
    )
    continuity_receipts: list[dict[str, Any]] = []
    groups = build_experiment_groups(
        formal_segments,
        events,
        manifest.views,
        config,
        decision_receipts=continuity_receipts,
        coarse_windows=boundary_candidates,
    )
    selection_receipts: list[dict[str, Any]] = []
    selected = select_key_events(
        groups,
        formal_segments,
        events,
        config,
        decision_receipts=selection_receipts,
    )

    by_event = {event.event_id: event for event in events}
    pre_roll_ms = float(config["segmentation"]["experiment_pre_roll_seconds"]) * 1000.0
    post_roll_ms = float(config["segmentation"]["experiment_post_roll_seconds"]) * 1000.0
    legacy_boundary_invariants = []
    if not exact_raw_replay:
        for segment in raw_segments:
            member_events = [
                by_event[event_id]
                for event_id in segment.event_ids
                if event_id in by_event and by_event[event_id].accepted
            ]
            if not member_events:
                continue
            accepted_start_ms = min(event.global_start_ms for event in member_events)
            accepted_end_ms = max(event.global_end_ms for event in member_events)
            expected_start_ms = max(0.0, accepted_start_ms - pre_roll_ms)
            expected_end_ms = accepted_end_ms + post_roll_ms
            legacy_boundary_invariants.append(
                {
                    "segment_id": segment.segment_id,
                    "stored_start_ms": segment.global_start_ms,
                    "stored_end_ms": segment.global_end_ms,
                    "accepted_event_start_ms": accepted_start_ms,
                    "accepted_event_end_ms": accepted_end_ms,
                    "start_delta_from_direct_event_boundary_ms": (
                        segment.global_start_ms - expected_start_ms
                    ),
                    "end_delta_from_direct_event_boundary_ms": (
                        segment.global_end_ms - expected_end_ms
                    ),
                    "requires_explicit_qf1_receipt": bool(
                        segment.global_start_ms < expected_start_ms - 1e-6
                        or segment.global_end_ms > expected_end_ms + 1e-6
                    ),
                }
            )

    result: dict[str, Any] = {
        "schema_version": QUALITY_REPLAY_VERSION,
        "archive_root": str(root),
        "experiment_id": manifest.experiment_id,
        "ledger_inputs": ledger_inputs,
        "evidence_grade": "exact" if exact_raw_replay else "degraded_legacy_ledger",
        "raw_segment_source": raw_segment_source,
        "boundary_candidate_source": (
            "audit_layer.boundary_candidates"
            if exact_raw_replay
            else "motion_probe_windows.candidates"
        ),
        "counts": {
            "events": len(events),
            "raw_segments": len(raw_segments),
            "normalized_segments": len(normalized),
            "formal_segments": len(formal_segments),
            "experiment_groups": len(groups),
            "selected_key_events": len(selected),
        },
        "groups": [group.model_dump(mode="json") for group in groups],
        "selected_event_ids": [event.event_id for event in selected],
        "raw_boundary_decision_receipts": raw_boundary_receipts,
        "normalization_decision_receipts": normalization_receipts,
        "formal_segment_receipts": formal_receipts,
        "continuity_decision_receipts": continuity_receipts,
        "selection_decision_receipts": selection_receipts,
        "legacy_boundary_invariants": legacy_boundary_invariants,
        "limitations": (
            []
            if exact_raw_replay
            else [
                "The retained legacy audit did not persist exact boundary candidates.",
                "Stored raw segments were reused, so raw-boundary equivalence is not proven.",
                "Motion candidates are a degraded proxy for QF1/QF2 coarse-boundary context.",
            ]
        ),
        "source_policy": {
            "video_files_opened": 0,
            "clock_csv_files_opened": 0,
            "model_calls": 0,
            "token_usage": 0,
            "mode": "json_quality_ledger_replay",
        },
    }
    encoded = json.dumps(
        result,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    result["result_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result
