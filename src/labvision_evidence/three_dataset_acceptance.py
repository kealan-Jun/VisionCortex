from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "visioncortex-three-dataset-acceptance/1.0.0"
ACTION_TYPES = (
    "hand_object_contact",
    "object_movement",
    "liquid_movement",
    "container_state_change",
    "device_panel_operation",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    expected_source_experiment_id: str
    archive_name: str
    six_view_reviewed_baseline: bool = False
    preprocessing_target_seconds: float | None = None
    expected_group_count: int | None = None
    minimum_key_event_count: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DatasetSpec":
        return cls(
            key=str(payload["key"]),
            expected_source_experiment_id=str(
                payload["expected_source_experiment_id"]
            ),
            archive_name=str(payload["archive_name"]),
            six_view_reviewed_baseline=bool(
                payload.get("six_view_reviewed_baseline", False)
            ),
            preprocessing_target_seconds=(
                float(payload["preprocessing_target_seconds"])
                if payload.get("preprocessing_target_seconds") is not None
                else None
            ),
            expected_group_count=(
                int(payload["expected_group_count"])
                if payload.get("expected_group_count") is not None
                else None
            ),
            minimum_key_event_count=(
                int(payload["minimum_key_event_count"])
                if payload.get("minimum_key_event_count") is not None
                else None
            ),
        )


class JsonLedgerReader:
    """Read JSON evidence while proving that media and clock payloads stay closed."""

    def __init__(self) -> None:
        self.opened_json_files: list[str] = []
        self.failures: list[dict[str, str]] = []

    def read(self, path: Path, default: Any) -> Any:
        if path.suffix.lower() != ".json":
            raise ValueError(f"The read-only acceptance reader only opens JSON: {path}")
        if not path.is_file():
            return default
        self.opened_json_files.append(str(path.resolve()))
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, TypeError) as exc:
            self.failures.append(
                {
                    "path": str(path.resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return default

    def source_policy(self) -> dict[str, Any]:
        return {
            "mode": "durable_json_ledger_only",
            "json_files_opened": len(self.opened_json_files),
            "json_paths": sorted(set(self.opened_json_files)),
            "video_files_opened": 0,
            "clock_csv_files_opened": 0,
            "image_files_opened": 0,
            "pdf_files_opened": 0,
            "model_api_calls": 0,
            "model_tokens_consumed": 0,
            "source_mutations": 0,
            "archive_mutations": 0,
            "parse_failures": list(self.failures),
        }


def load_acceptance_spec(path: Path) -> tuple[str, list[DatasetSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    datasets = [DatasetSpec.from_payload(item) for item in payload.get("datasets") or []]
    if not datasets:
        raise ValueError("Acceptance specification must contain at least one dataset")
    keys = [item.key for item in datasets]
    if len(keys) != len(set(keys)):
        raise ValueError("Acceptance dataset keys must be unique")
    return str(payload.get("task_id") or "unspecified"), datasets


def _parse_timestamp(value: Any, fallback: float) -> float:
    if value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            pass
    return fallback


def _candidate_timestamp(root: Path, reader: JsonLedgerReader) -> float:
    json_root = root / "JSON-Config-Files"
    status = reader.read(json_root / "pipeline_status.json", {})
    metrics = reader.read(json_root / "run_metrics.json", {})
    submission = reader.read(json_root / "run_submission.json", {})
    fallback = root.stat().st_mtime
    return max(
        _parse_timestamp(status.get("updated_at"), fallback),
        _parse_timestamp(metrics.get("run_ended_at"), fallback),
        _parse_timestamp(submission.get("submitted_at"), fallback),
    )


def discover_latest_candidate(
    archive_root: Path,
    spec: DatasetSpec,
    reader: JsonLedgerReader,
    override: Path | None = None,
) -> tuple[Path | None, str | None, list[dict[str, Any]]]:
    if override is not None:
        root = override.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Acceptance candidate does not exist: {root}")
        kind = "formal" if root == (archive_root / spec.archive_name).resolve() else "staging"
        return root, kind, [{"path": str(root), "kind": kind, "selected": True}]

    candidates: list[tuple[float, Path, str]] = []
    formal = archive_root / spec.archive_name
    if formal.is_dir():
        candidates.append((_candidate_timestamp(formal, reader), formal.resolve(), "formal"))
    staging_parent = archive_root / ".VisionCortex-Run-Staging" / spec.archive_name
    if staging_parent.is_dir():
        for child in staging_parent.iterdir():
            if child.is_dir() and (child / "JSON-Config-Files").is_dir():
                candidates.append(
                    (_candidate_timestamp(child, reader), child.resolve(), "staging")
                )
    if not candidates:
        return None, None, []
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    selected = candidates[0]
    discovery = [
        {
            "path": str(path),
            "kind": kind,
            "timestamp_epoch": timestamp,
            "selected": path == selected[1],
        }
        for timestamp, path, kind in candidates
    ]
    return selected[1], selected[2], discovery


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    category: str,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "category": category,
        }
    )


def _token_bucket(payload: Any) -> dict[str, Any]:
    bucket = payload if isinstance(payload, dict) else {}
    return {
        "input_tokens": bucket.get("input_tokens"),
        "output_tokens": bucket.get("output_tokens"),
        "cached_input_tokens": bucket.get("cached_input_tokens"),
        "total_tokens": bucket.get("total_tokens"),
        "call_count": bucket.get("call_count"),
        "executed_call_count": bucket.get("executed_call_count"),
        "reused_call_count": bucket.get("reused_call_count"),
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return round(ordered[index], 6)


def _model_call_summary(calls: Any) -> dict[str, Any]:
    records = [item for item in calls or [] if isinstance(item, dict)]
    latencies = [
        float(item["latency_seconds"])
        for item in records
        if item.get("latency_seconds") is not None
    ]
    return {
        "call_count": len(records),
        "successful_count": sum(
            str(item.get("status") or "completed") in {"completed", "success"}
            for item in records
        ),
        "failed_count": sum(
            str(item.get("status") or "completed") in {"failed", "error"}
            for item in records
        ),
        "retry_count": sum(max(0, int(item.get("attempts") or 1) - 1) for item in records),
        "latency_seconds": {
            "min": round(min(latencies), 6) if latencies else None,
            "median": round(statistics.median(latencies), 6) if latencies else None,
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 6) if latencies else None,
        },
    }


def _metric_value(stage: dict[str, Any], key: str, statistic: str) -> float | None:
    payload = stage.get(key) or {}
    value = payload.get(statistic) if isinstance(payload, dict) else None
    return float(value) if value is not None else None


def _resource_summary(telemetry: dict[str, Any]) -> dict[str, Any]:
    stages = telemetry.get("stage_summaries") or {}
    if isinstance(stages, list):
        merged: dict[str, Any] = {}
        for item in stages:
            if isinstance(item, dict):
                merged.update(item)
        stages = merged
    if not isinstance(stages, dict):
        stages = {}

    def aggregate(metric: str, statistic: str, operation: str) -> float | None:
        values = [
            value
            for payload in stages.values()
            if isinstance(payload, dict)
            for value in [_metric_value(payload, metric, statistic)]
            if value is not None
        ]
        if not values:
            return None
        return round(max(values) if operation == "max" else statistics.mean(values), 6)

    return {
        "sample_count": telemetry.get("sample_count"),
        "gpu_telemetry_backend": telemetry.get("gpu_telemetry_backend"),
        "stage_count": len(stages),
        "cpu_percent_max": aggregate("cpu_percent", "max", "max"),
        "memory_percent_max": aggregate("memory_percent", "max", "max"),
        "gpu_compute_percent_max": aggregate("gpu_compute_percent", "max", "max"),
        "gpu_compute_percent_mean_of_stages": aggregate(
            "gpu_compute_percent", "mean", "mean"
        ),
        "gpu_memory_used_mib_max": aggregate("gpu_memory_used_mib", "max", "max"),
        "gpu_temperature_c_max": aggregate("gpu_temperature_c", "max", "max"),
        "gpu_power_w_max": aggregate("gpu_power_w", "max", "max"),
        "nvdec_percent_max": aggregate("nvdec_percent", "max", "max"),
        "nvenc_percent_max": aggregate("nvenc_percent", "max", "max"),
        "host_network_receive_mib_s_p95_max": aggregate(
            "host_network_receive_mib_s", "p95", "max"
        ),
        "pipeline_read_mib_s_p95_max": aggregate(
            "pipeline_read_mib_s", "p95", "max"
        ),
        "stage_summaries": stages,
    }


def _scan_summary(payload: dict[str, Any]) -> dict[str, Any]:
    diagnosis = payload.get("bottleneck_diagnosis") or {}
    reports = payload.get("role_reports") or []
    if isinstance(reports, dict):
        reports = [reports]
    return {
        "work_unit_count": len(payload.get("work_units") or []),
        "role_report_count": len(reports),
        "inference_frame_count": diagnosis.get("inference_frame_count"),
        "inference_call_count": diagnosis.get("inference_call_count"),
        "actual_batch_size_mean": diagnosis.get("actual_batch_size_mean"),
        "effective_batch_capacity_mean": diagnosis.get(
            "effective_batch_capacity_mean"
        ),
        "batch_fill_ratio": diagnosis.get("batch_fill_ratio"),
        "queue_wait_seconds": diagnosis.get("queue_wait_seconds"),
        "inference_seconds": diagnosis.get("inference_seconds"),
        "tracking_and_ledger_seconds": diagnosis.get(
            "tracking_and_ledger_seconds"
        ),
        "classification": diagnosis.get("classification"),
        "next_action": diagnosis.get("next_action"),
    }


def _concrete_object_values(event: dict[str, Any]) -> list[str]:
    values: list[str] = []
    objects = event.get("objects") or {}
    raw_values: Iterable[Any] = objects.values() if isinstance(objects, dict) else objects
    for value in raw_values:
        normalized = str(value or "").strip()
        lowered = normalized.lower().replace("_", "-")
        if not normalized or lowered in {"unknown", "none", "null"}:
            continue
        if lowered.endswith("-unresolved") or lowered.startswith("unknown-"):
            continue
        values.append(normalized)
    return values


def _material_semantics(events: list[dict[str, Any]]) -> dict[str, Any]:
    object_aware = 0
    semantically_named = 0
    current_step = 0
    next_step = 0
    action_counts = {action: 0 for action in ACTION_TYPES}
    by_group: dict[str, dict[str, int]] = {}
    for event in events:
        action_type = str(event.get("action_type") or "")
        if action_type in action_counts:
            action_counts[action_type] += 1
        group_id = str(
            event.get("parent_event_id")
            or (event.get("provenance") or {}).get("experiment_group_id")
            or "unassigned"
        )
        group_counts = by_group.setdefault(
            group_id, {action: 0 for action in ACTION_TYPES}
        )
        if action_type in group_counts:
            group_counts[action_type] += 1
        concrete = _concrete_object_values(event)
        object_aware += int(bool(concrete))
        provenance = event.get("provenance") or {}
        classification = provenance.get("archive_classification") or {}
        primary = str(classification.get("primary_object") or "")
        semantic_stem = str(classification.get("semantic_file_stem") or "")
        referenced_paths = [
            str(item.get("path") or "")
            for collection in (event.get("key_frames") or [], event.get("key_clips") or [])
            for item in collection
            if isinstance(item, dict)
        ]
        path_has_semantic_object = bool(
            primary
            and primary != "Unknown-Object"
            and semantic_stem
            and any(primary.lower() in path.lower() for path in referenced_paths)
        )
        semantically_named += int(path_has_semantic_object)
        model = provenance.get("mllm") or {}
        current_step += int(bool(str(model.get("current_step") or "").strip()))
        next_step += int(bool(str(model.get("next_step") or "").strip()))
    count = len(events)
    return {
        "event_count": count,
        "action_counts": action_counts,
        "per_experiment_action_counts": by_group,
        "object_aware_event_count": object_aware,
        "object_aware_rate": object_aware / count if count else 0.0,
        "semantic_object_filename_count": semantically_named,
        "semantic_object_filename_rate": semantically_named / count if count else 0.0,
        "current_step_count": current_step,
        "next_step_count": next_step,
    }


def _group_summary(
    package: dict[str, Any], audit: dict[str, Any], view_roles: dict[str, str]
) -> list[dict[str, Any]]:
    groups = package.get("experiment_groups") or audit.get("experiment_groups") or []
    result: list[dict[str, Any]] = []
    for group in groups:
        views = [str(item) for item in group.get("participating_views") or []]
        roles = sorted({view_roles.get(view, "unknown") for view in views})
        atoms = group.get("atomic_experiment_ids") or []
        result.append(
            {
                "group_id": group.get("group_id"),
                "experiment_name": group.get("experiment_name"),
                "start_ms": group.get("global_start_ms", group.get("start_ms")),
                "end_ms": group.get("global_end_ms", group.get("end_ms")),
                "continuity_type": group.get("continuity_type"),
                "atomic_experiment_count": len(atoms),
                "participating_views": views,
                "participating_roles": roles,
                "first_third_supported": {"first_person", "third_person"}.issubset(
                    roles
                ),
            }
        )
    return result


def _pipeline_state(status: dict[str, Any], candidate_kind: str) -> str:
    stage = str(status.get("stage") or "unknown")
    if status.get("failed_stage") or stage in {"failed", "error"}:
        return "failed"
    if stage == "completed":
        return "archived" if candidate_kind == "formal" else "completed_not_promoted"
    if stage == "unknown":
        return "unknown"
    return "running"


def _failure_categories(checks: list[dict[str, Any]], failed_stage: Any) -> list[str]:
    categories = {item["category"] for item in checks if not item["passed"]}
    stage = str(failed_stage or "")
    if stage in {"preflight", "alignment", "motion_probe", "candidate_coarse", "candidate_fine"}:
        categories.add("preprocessing_runtime")
    elif stage in {"candidate_audit"}:
        categories.add("cv_boundary_or_cross_view_quality")
    elif stage in {"experiment_understanding", "mllm"}:
        categories.add("model_understanding_or_api")
    elif stage in {"experiment_clips", "key_materials", "package", "daily_report"}:
        categories.add("materialization_report_or_archive")
    return sorted(categories)


def inspect_dataset_candidate(
    spec: DatasetSpec,
    root: Path,
    candidate_kind: str,
    discovery: list[dict[str, Any]],
    reader: JsonLedgerReader,
) -> dict[str, Any]:
    json_root = root / "JSON-Config-Files"
    status = reader.read(json_root / "pipeline_status.json", {})
    metrics = reader.read(json_root / "run_metrics.json", {})
    quality = reader.read(json_root / "quality_acceptance.json", {})
    boundary = reader.read(json_root / "boundary_precheck.json", {})
    evaluation = reader.read(json_root / "evidence_package_eval.json", {})
    audit = reader.read(json_root / "audit_layer.json", {})
    progressive = reader.read(json_root / "progressive_fine_scan.json", {})
    volume = reader.read(json_root / "input_volume_report.json", {})
    manifest = reader.read(json_root / "run_manifest.json", {})
    package = reader.read(json_root / "evidence_package.json", {})
    promotion = reader.read(json_root / "fixed_archive_promotion.json", {})
    daily = reader.read(json_root / "daily_report_manifest.json", {})
    professional = reader.read(json_root / "professional_report_manifest.json", {})
    index_manifest = reader.read(json_root / "evidence_index_manifest.json", {})
    state_ledger = reader.read(json_root / "continuous_action_state_ledger.json", {})
    telemetry = reader.read(json_root / "resource_telemetry.json", {})
    fine_runtime = reader.read(json_root / "scan_runtime_fine.json", {})
    motion_runtime = reader.read(json_root / "scan_runtime_motion_probe.json", {})
    materials_payload = reader.read(
        root / "Key-Materials" / "Key-Materials-Model-Understanding.json", []
    )
    materials = [item for item in materials_payload if isinstance(item, dict)]

    ingest = metrics.get("nas_index_ingest") or {}
    source_id = str(
        ingest.get("experiment_id")
        or package.get("experiment_id")
        or manifest.get("experiment_id")
        or ""
    )
    views = volume.get("views") or manifest.get("views") or []
    view_roles = {
        str(item.get("view_id")): str(item.get("role") or "unknown")
        for item in views
        if isinstance(item, dict) and item.get("view_id")
    }
    role_counts: dict[str, int] = {}
    for role in view_roles.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    groups = _group_summary(package, audit, view_roles)
    semantic = _material_semantics(materials)

    quality_boundaries = quality.get("experiment_boundaries") or boundary.get(
        "experiment_boundaries"
    ) or {}
    quality_materials = quality.get("key_materials") or {}
    group_count = int(
        quality_boundaries.get("predicted_experiment_count")
        or evaluation.get("experiment_group_count")
        or len(groups)
    )
    key_event_count = int(
        quality_materials.get("event_count")
        or evaluation.get("key_event_count")
        or semantic["event_count"]
    )
    action_counts = dict(quality_materials.get("action_counts") or semantic["action_counts"])
    media_complete_count = int(
        quality_materials.get("media_complete_count")
        or evaluation.get("dual_view_material_count")
        or 0
    )
    model_complete_count = int(
        quality_materials.get("model_understanding_completed_count") or 0
    )
    baseline_selection = quality.get("baseline_selection") or boundary.get(
        "baseline_selection"
    ) or {}
    unresolved = list(progressive.get("unresolved_candidate_ids") or [])
    quarantine_ids: set[str] = set()
    for receipt in audit.get("formal_segment_receipts") or []:
        if str(receipt.get("decision") or "").startswith("quarantined"):
            quarantine_ids.update(str(item) for item in receipt.get("event_ids") or [])
    selected_ids = {str(item.get("event_id")) for item in materials if item.get("event_id")}
    leaked = sorted(quarantine_ids & selected_ids)

    copy_bytes = ingest.get("ingest_details", {}).get(
        "copied_source_bytes", ingest.get("copied_source_bytes", ingest.get("source_copy_bytes"))
    )
    continuous_copies = ingest.get("ingest_details", {}).get(
        "continuous_source_copies_created"
    )
    if copy_bytes is None:
        copy_bytes = ingest.get("source_copy_bytes", 0)
    if continuous_copies is None:
        continuous_copies = 0
    pipeline_state = _pipeline_state(status, candidate_kind)
    promotion_verified = (
        str((promotion.get("verification") or {}).get("status") or "").lower()
        == "verified"
    )
    stages = [item for item in metrics.get("stage_durations") or [] if isinstance(item, dict)]
    stage_durations = {
        str(item.get("stage")): item.get("duration_seconds") for item in stages
    }
    tokens = metrics.get("tokens") or {}
    token_ledger = {
        "experiment_groups": _token_bucket(tokens.get("experiment_groups")),
        "key_materials": _token_bucket(tokens.get("key_materials")),
        "daily_report": _token_bucket(tokens.get("daily_report")),
        "run_total": _token_bucket(tokens.get("run_total")),
    }
    run_total = token_ledger["run_total"]
    token_complete = all(
        run_total.get(field) is not None
        for field in ("input_tokens", "output_tokens", "total_tokens")
    )
    preproc = metrics.get("preprocessing_sla") or {}
    preproc_actual = preproc.get("actual_seconds")
    if preproc_actual is None:
        preproc_actual = sum(
            float(stage_durations.get(name) or 0.0)
            for name in (
                "preflight",
                "alignment",
                "motion_probe",
                "candidate_coarse",
                "candidate_fine",
                "candidate_audit",
            )
        ) or None

    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "source_experiment_identity",
        source_id == spec.expected_source_experiment_id,
        source_id,
        spec.expected_source_experiment_id,
        "shared_source_identity_or_baseline",
    )
    _check(
        checks,
        "zero_copy_ingest",
        int(copy_bytes or 0) == 0 and int(continuous_copies or 0) == 0,
        {"source_copy_bytes": copy_bytes, "continuous_source_copies": continuous_copies},
        {"source_copy_bytes": 0, "continuous_source_copies": 0},
        "storage_or_ingest",
    )
    _check(
        checks,
        "pipeline_completed",
        pipeline_state in {"archived", "completed_not_promoted"},
        {"state": pipeline_state, "stage": status.get("stage"), "failed_stage": status.get("failed_stage")},
        "completed",
        "pipeline_runtime",
    )
    _check(checks, "experiment_groups_present", group_count > 0, group_count, ">0", "cv_boundary_or_cross_view_quality")
    dual_group_count = sum(bool(item["first_third_supported"]) for item in groups)
    _check(
        checks,
        "formal_groups_are_first_third_aligned",
        bool(groups) and dual_group_count == len(groups),
        {"dual_role_group_count": dual_group_count, "group_count": len(groups)},
        "all formal groups",
        "cv_boundary_or_cross_view_quality",
    )
    _check(
        checks,
        "key_material_quality",
        bool(quality_materials.get("passed")),
        quality_materials.get("passed"),
        True,
        "key_material_quality",
    )
    _check(
        checks,
        "key_material_media_complete",
        key_event_count > 0 and media_complete_count == key_event_count,
        {"media_complete_count": media_complete_count, "event_count": key_event_count},
        "equal",
        "materialization_report_or_archive",
    )
    _check(
        checks,
        "key_material_model_understanding_complete",
        key_event_count > 0 and model_complete_count == key_event_count,
        {"completed": model_complete_count, "event_count": key_event_count},
        "equal",
        "model_understanding_or_api",
    )
    _check(
        checks,
        "object_aware_materials",
        semantic["event_count"] > 0
        and semantic["object_aware_event_count"] == semantic["event_count"],
        {
            "object_aware": semantic["object_aware_event_count"],
            "events": semantic["event_count"],
        },
        "all selected events identify an operated object",
        "key_material_quality",
    )
    _check(
        checks,
        "semantic_object_filenames",
        semantic["event_count"] > 0
        and semantic["semantic_object_filename_count"] == semantic["event_count"],
        {
            "semantic_filenames": semantic["semantic_object_filename_count"],
            "events": semantic["event_count"],
        },
        "all selected events",
        "key_material_quality",
    )
    _check(
        checks,
        "current_and_next_step_understanding",
        semantic["event_count"] > 0
        and semantic["current_step_count"] == semantic["event_count"]
        and semantic["next_step_count"] == semantic["event_count"],
        {
            "current_step": semantic["current_step_count"],
            "next_step": semantic["next_step_count"],
            "events": semantic["event_count"],
        },
        "complete for every selected event",
        "model_understanding_or_api",
    )
    _check(
        checks,
        "evidence_package_eval",
        bool(evaluation.get("passed")),
        evaluation.get("passed"),
        True,
        "evidence_package_or_index",
    )
    _check(
        checks,
        "continuous_action_state_receipts",
        bool(state_ledger)
        and not bool(state_ledger.get("cv_acceptance_mutated", False)),
        {
            "present": bool(state_ledger),
            "receipt_count": len(state_ledger.get("receipts") or []),
            "cv_acceptance_mutated": state_ledger.get("cv_acceptance_mutated"),
        },
        "present and non-mutating",
        "evidence_package_or_index",
    )
    _check(
        checks,
        "searchable_evidence_index",
        bool(index_manifest)
        and bool(index_manifest.get("validation", {}).get("passed", index_manifest.get("passed", False))),
        {
            "present": bool(index_manifest),
            "validation": index_manifest.get("validation"),
        },
        "present and passed",
        "evidence_package_or_index",
    )
    _check(
        checks,
        "daily_report",
        bool(daily.get("passed")),
        {"passed": daily.get("passed"), "template": daily.get("daily_template_id")},
        True,
        "materialization_report_or_archive",
    )
    _check(
        checks,
        "professional_pdf",
        str(professional.get("status") or "") == "generated"
        and bool(professional.get("pdf")),
        {"status": professional.get("status"), "pdf": professional.get("pdf")},
        "generated",
        "materialization_report_or_archive",
    )
    _check(
        checks,
        "timing_ledger",
        bool(stages) and metrics.get("total_duration_seconds") is not None,
        {"stage_count": len(stages), "total_seconds": metrics.get("total_duration_seconds")},
        "complete",
        "observability",
    )
    _check(
        checks,
        "token_ledger",
        token_complete,
        run_total,
        "provider-reported input/output/total",
        "observability",
    )
    _check(
        checks,
        "quarantined_event_leakage",
        not leaked,
        leaked,
        [],
        "cv_boundary_or_cross_view_quality",
    )
    _check(
        checks,
        "verified_atomic_promotion",
        candidate_kind == "formal" and promotion_verified,
        {"candidate_kind": candidate_kind, "verification": (promotion.get("verification") or {}).get("status")},
        {"candidate_kind": "formal", "verification": "verified"},
        "materialization_report_or_archive",
    )

    if spec.six_view_reviewed_baseline:
        _check(
            checks,
            "reviewed_baseline_selection",
            baseline_selection.get("current_experiment_id") == spec.expected_source_experiment_id
            and bool(baseline_selection.get("applied"))
            and baseline_selection.get("reason") == "experiment_id_matched",
            baseline_selection,
            {
                "current_experiment_id": spec.expected_source_experiment_id,
                "applied": True,
                "reason": "experiment_id_matched",
            },
            "shared_source_identity_or_baseline",
        )
        _check(
            checks,
            "reviewed_boundaries",
            bool(quality_boundaries.get("evaluated"))
            and bool(quality_boundaries.get("passed"))
            and all(
                bool(item.get("boundary_within_tolerance"))
                for item in quality_boundaries.get("matches") or []
            ),
            {
                "evaluated": quality_boundaries.get("evaluated"),
                "passed": quality_boundaries.get("passed"),
                "matches": quality_boundaries.get("matches"),
            },
            "all starts/ends within reviewed tolerance",
            "cv_boundary_or_cross_view_quality",
        )
        structures = {
            str(item.get("baseline_id")): {
                "continuity_type": item.get("predicted_continuity_type"),
                "atomic_experiment_count": item.get("predicted_atomic_experiment_count"),
            }
            for item in quality_boundaries.get("matches") or []
        }
        _check(
            checks,
            "G2_structure",
            structures.get("reviewed-exp-002")
            == {"continuity_type": "independent", "atomic_experiment_count": 1},
            structures.get("reviewed-exp-002"),
            {"continuity_type": "independent", "atomic_experiment_count": 1},
            "cv_boundary_or_cross_view_quality",
        )
        _check(
            checks,
            "G4_structure",
            structures.get("reviewed-exp-004")
            == {"continuity_type": "continuous", "atomic_experiment_count": 2},
            structures.get("reviewed-exp-004"),
            {"continuity_type": "continuous", "atomic_experiment_count": 2},
            "cv_boundary_or_cross_view_quality",
        )
        _check(
            checks,
            "formal_anchor_clusters_closed",
            not unresolved,
            unresolved,
            [],
            "cv_boundary_or_cross_view_quality",
        )
    else:
        _check(
            checks,
            "six_view_baseline_inapplicable",
            not bool(baseline_selection.get("applied", False)),
            baseline_selection,
            {"applied": False},
            "shared_source_identity_or_baseline",
        )

    if spec.expected_group_count is not None:
        _check(
            checks,
            "expected_group_count",
            group_count == spec.expected_group_count,
            group_count,
            spec.expected_group_count,
            "cv_boundary_or_cross_view_quality",
        )
    if spec.minimum_key_event_count is not None:
        _check(
            checks,
            "minimum_key_event_count",
            key_event_count >= spec.minimum_key_event_count,
            key_event_count,
            f">={spec.minimum_key_event_count}",
            "key_material_quality",
        )
    if spec.preprocessing_target_seconds is not None:
        _check(
            checks,
            "preprocessing_target",
            preproc_actual is not None
            and float(preproc_actual) <= spec.preprocessing_target_seconds,
            preproc_actual,
            f"<={spec.preprocessing_target_seconds}",
            "performance",
        )

    passed = all(item["passed"] for item in checks)
    failures = [item for item in checks if not item["passed"]]
    result_status = (
        "in_progress"
        if pipeline_state == "running"
        else "passed"
        if passed
        else "failed"
    )
    return {
        "dataset_key": spec.key,
        "expected_source_experiment_id": spec.expected_source_experiment_id,
        "archive_name": spec.archive_name,
        "candidate": {
            "path": str(root),
            "kind": candidate_kind,
            "discovery": discovery,
        },
        "run": {
            "state": pipeline_state,
            "stage": status.get("stage"),
            "failed_stage": status.get("failed_stage"),
            "message": status.get("message"),
            "source_experiment_id": source_id,
            "total_duration_seconds": metrics.get("total_duration_seconds"),
            "preprocessing_seconds": preproc_actual,
            "preprocessing_target_seconds": spec.preprocessing_target_seconds,
            "source_copy_bytes": copy_bytes,
            "continuous_source_copies_created": continuous_copies,
        },
        "input": {
            "view_count": len(view_roles),
            "role_counts": role_counts,
            "views": views,
            "video_segment_count": volume.get("unique_video_path_count"),
            "clock_segment_count": volume.get("unique_clock_path_count"),
            "total_video_bytes": volume.get("total_video_bytes"),
        },
        "quality": {
            "group_count": group_count,
            "groups": groups,
            "key_event_count": key_event_count,
            "action_counts": action_counts,
            "material_semantics": semantic,
            "media_complete_count": media_complete_count,
            "model_understanding_completed_count": model_complete_count,
            "dual_view_material_count": evaluation.get("dual_view_material_count"),
            "unresolved_formal_candidate_ids": unresolved,
            "quarantined_event_ids": sorted(quarantine_ids),
            "selected_quarantined_event_ids": leaked,
            "baseline_selection": baseline_selection,
            "boundary": quality_boundaries,
        },
        "performance": {
            "stage_durations": stage_durations,
            "motion_scan": _scan_summary(motion_runtime),
            "fine_scan": _scan_summary(fine_runtime),
            "resources": _resource_summary(telemetry),
        },
        "model": {
            "tokens": token_ledger,
            "calls": _model_call_summary(metrics.get("mllm_calls")),
        },
        "outputs": {
            "evidence_package_eval_passed": evaluation.get("passed"),
            "evidence_index_validation": index_manifest.get("validation"),
            "continuous_action_receipt_count": len(state_ledger.get("receipts") or []),
            "daily_report": daily,
            "professional_report": professional,
            "promotion": promotion,
        },
        "status": result_status,
        "passed": passed,
        "checks": checks,
        "failures": failures,
        "failure_categories": _failure_categories(checks, status.get("failed_stage")),
    }


def analyze_three_datasets(
    task_id: str,
    archive_root: Path,
    specs: list[DatasetSpec],
    candidate_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    archive_root = archive_root.resolve()
    reader = JsonLedgerReader()
    datasets: list[dict[str, Any]] = []
    overrides = candidate_overrides or {}
    for spec in specs:
        root, kind, discovery = discover_latest_candidate(
            archive_root, spec, reader, overrides.get(spec.key)
        )
        if root is None or kind is None:
            datasets.append(
                {
                    "dataset_key": spec.key,
                    "expected_source_experiment_id": spec.expected_source_experiment_id,
                    "archive_name": spec.archive_name,
                    "status": "not_available",
                    "passed": False,
                    "checks": [],
                    "failures": [
                        {
                            "check": "candidate_available",
                            "passed": False,
                            "observed": None,
                            "expected": "formal archive or staging run",
                            "category": "pipeline_runtime",
                        }
                    ],
                    "failure_categories": ["pipeline_runtime"],
                    "candidate": {"path": None, "kind": None, "discovery": discovery},
                }
            )
            continue
        datasets.append(inspect_dataset_candidate(spec, root, kind, discovery, reader))
    passed = bool(datasets) and all(item.get("passed") for item in datasets)
    in_progress = any(
        (item.get("run") or {}).get("state") == "running" for item in datasets
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive_root),
        "status": "in_progress" if in_progress else "passed" if passed else "failed",
        "passed": passed,
        "production_release_ready": passed,
        "datasets": datasets,
        "summary": {
            "dataset_count": len(datasets),
            "passed_count": sum(bool(item.get("passed")) for item in datasets),
            "in_progress_count": sum(
                item.get("status") == "in_progress" for item in datasets
            ),
            "failed_count": sum(
                item.get("status") in {"failed", "not_available"}
                for item in datasets
            ),
            "total_input_tokens": sum(
                int(
                    (((item.get("model") or {}).get("tokens") or {}).get("run_total") or {}).get(
                        "input_tokens"
                    )
                    or 0
                )
                for item in datasets
            ),
            "total_output_tokens": sum(
                int(
                    (((item.get("model") or {}).get("tokens") or {}).get("run_total") or {}).get(
                        "output_tokens"
                    )
                    or 0
                )
                for item in datasets
            ),
            "total_tokens": sum(
                int(
                    (((item.get("model") or {}).get("tokens") or {}).get("run_total") or {}).get(
                        "total_tokens"
                    )
                    or 0
                )
                for item in datasets
            ),
        },
        "source_policy": reader.source_policy(),
    }


def _format_seconds(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}s"


def render_acceptance_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    lines = [
        f"# {result.get('task_id')} Three-Dataset Acceptance",
        "",
        f"**Result: {str(result.get('status')).upper()}** — "
        f"{summary.get('passed_count', 0)}/{summary.get('dataset_count', 0)} datasets passed. "
        f"Production release ready: **{'YES' if result.get('production_release_ready') else 'NO'}**.",
        "",
        "## Dataset Matrix",
        "",
        "| Dataset | Run state | Views / roles | Groups | Key events | Preprocessing | Total | Tokens | Promotion |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result.get("datasets") or []:
        run = item.get("run") or {}
        inputs = item.get("input") or {}
        quality = item.get("quality") or {}
        token_total = (
            (((item.get("model") or {}).get("tokens") or {}).get("run_total") or {}).get(
                "total_tokens"
            )
        )
        role_text = ", ".join(
            f"{name}:{count}" for name, count in (inputs.get("role_counts") or {}).items()
        )
        promotion = ((item.get("outputs") or {}).get("promotion") or {}).get(
            "verification", {}
        ).get("status")
        lines.append(
            f"| {item.get('dataset_key')} | {run.get('state', item.get('status'))} | "
            f"{inputs.get('view_count', 0)} ({role_text or '—'}) | "
            f"{quality.get('group_count', 0)} | {quality.get('key_event_count', 0)} | "
            f"{_format_seconds(run.get('preprocessing_seconds'))} | "
            f"{_format_seconds(run.get('total_duration_seconds'))} | "
            f"{token_total if token_total is not None else '—'} | {promotion or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Five Physical Action Families",
            "",
            "| Dataset | Hand-object contact | Object movement | Liquid movement | Container state | Device panel |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result.get("datasets") or []:
        counts = (item.get("quality") or {}).get("action_counts") or {}
        lines.append(
            f"| {item.get('dataset_key')} | {counts.get('hand_object_contact', 0)} | "
            f"{counts.get('object_movement', 0)} | {counts.get('liquid_movement', 0)} | "
            f"{counts.get('container_state_change', 0)} | "
            f"{counts.get('device_panel_operation', 0)} |"
        )
    lines.extend(["", "## Failures and Next Action", ""])
    any_failures = False
    for item in result.get("datasets") or []:
        failures = item.get("failures") or []
        if not failures:
            continue
        any_failures = True
        lines.append(f"### {item.get('dataset_key')}")
        lines.append("")
        for failure in failures:
            lines.append(
                f"- **{failure.get('category')} / {failure.get('check')}**: "
                f"observed `{json.dumps(failure.get('observed'), ensure_ascii=False)}`; "
                f"expected `{json.dumps(failure.get('expected'), ensure_ascii=False)}`."
            )
        lines.append("")
    if not any_failures:
        lines.extend(["No failed gates.", ""])
    policy = result.get("source_policy") or {}
    lines.extend(
        [
            "## Read-Only Evidence Policy",
            "",
            f"- JSON ledgers opened: {policy.get('json_files_opened', 0)}",
            f"- Video payloads opened: {policy.get('video_files_opened', 0)}",
            f"- Clock CSV payloads opened: {policy.get('clock_csv_files_opened', 0)}",
            f"- Model API calls: {policy.get('model_api_calls', 0)}",
            f"- New model Tokens: {policy.get('model_tokens_consumed', 0)}",
            f"- Source/archive mutations: {policy.get('source_mutations', 0)}/{policy.get('archive_mutations', 0)}",
            "",
        ]
    )
    return "\n".join(lines)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def write_acceptance_reports(
    result: dict[str, Any], output_json: Path, archive_root: Path
) -> tuple[Path, Path]:
    output_json = output_json.resolve()
    archive_root = archive_root.resolve()
    if _is_within(output_json, archive_root):
        raise ValueError("Acceptance reports must be written outside the NAS archive root")
    output_markdown = output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_markdown.write_text(render_acceptance_markdown(result), encoding="utf-8")
    return output_json, output_markdown
