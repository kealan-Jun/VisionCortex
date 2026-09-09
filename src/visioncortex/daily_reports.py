from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .archive import ArchiveLayout, write_json
from .material_naming import ACTION_LABELS_ZH
from .pathing import archive_relative_posix
from .report_presentations import (
    render_daily_html,
    render_daily_markdown,
    render_professional_pdf,
)
from .schemas import RunSummary


ACTION_LABELS = ACTION_LABELS_ZH
TEMPLATE_DIRECTORY = Path(__file__).with_name("templates")
DEFAULT_DAILY_TEMPLATE_ID = "VC-LAB-DAILY-REPORT-V2"
PROFESSIONAL_TEMPLATE_ID = "VC-PROFESSIONAL-EVIDENCE-REPORT-V1"
PROFESSIONAL_TEMPLATE_PATH = TEMPLATE_DIRECTORY / f"{PROFESSIONAL_TEMPLATE_ID}.json"
PROFESSIONAL_RENDERER_PATH = Path(__file__).with_name("report_presentations.py")


def _daily_template_path(template_id: str) -> Path:
    path = TEMPLATE_DIRECTORY / f"{template_id}.json"
    if not path.is_file():
        raise ValueError(f"Unsupported daily-report template: {template_id}")
    return path


def load_daily_report_template(template_id: str = DEFAULT_DAILY_TEMPLATE_ID) -> dict[str, Any]:
    return json.loads(_daily_template_path(template_id).read_text(encoding="utf-8"))


def _select_group_visuals(key_events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select deterministic aligned dual-view images without another model call.

    ``supporting_roles`` describes which roles directly prove the accepted
    action.  It must not be overloaded to describe the media layout: a valid
    aligned first/third-person composite can still be useful when one role is
    direct evidence and the other is synchronized context.  Keep those facts
    separate so reports can show the real image without inflating cross-view
    semantic support.
    """

    eligible = [
        item
        for item in key_events
        if item.get("aligned_key_frame")
        and item.get("aligned_key_clip")
    ]
    eligible.sort(
        key=lambda item: (
            -float(item.get("confidence") or 0),
            float(item.get("peak_global_ms") or 0),
            str(item.get("event_id") or ""),
        )
    )
    if not eligible:
        return None, []

    def visual(item: dict[str, Any]) -> dict[str, Any]:
        direct_roles = list(item.get("supporting_roles") or [])
        direct_dual_role = {"first_person", "third_person"}.issubset(
            set(direct_roles)
        )
        return {
            "event_id": item["event_id"],
            "action_type": item["action_type"],
            "action_label": item["action_label"],
            "peak_global_ms": item["peak_global_ms"],
            "peak_timecode": item["peak_timecode"],
            "objects": item["objects"],
            "confidence": item["confidence"],
            "supporting_views": item["supporting_views"],
            "supporting_roles": direct_roles,
            "visual_roles": ["first_person", "third_person"],
            "support_scope": (
                "dual_role_direct"
                if direct_dual_role
                else "single_role_direct_with_aligned_cross_role_context"
            ),
            "image_path": item["aligned_key_frame"],
            "clip_path": item["aligned_key_clip"],
            "claim_class": (
                "observed_fact"
                if direct_dual_role
                else "supported_model_understanding"
            ),
        }

    representative = visual(eligible[0])
    gallery: list[dict[str, Any]] = []
    for action_type in ACTION_LABELS:
        if action_type == representative["action_type"]:
            continue
        match = next(
            (
                item
                for item in eligible
                if item["action_type"] == action_type
            ),
            None,
        )
        if match is not None:
            gallery.append(visual(match))
    return representative, gallery


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _clock(ms: float | int | None) -> str:
    value = max(0, int(float(ms or 0)))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _duration(seconds: float | int | None) -> str:
    value = max(0.0, float(seconds or 0))
    minutes, remaining = divmod(value, 60)
    if minutes:
        return f"{int(minutes)} 分 {remaining:.3f} 秒"
    return f"{remaining:.3f} 秒"


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_runtime_audit(
    layout: ArchiveLayout,
    run_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Collect a deterministic runtime/quality digest from archived receipts."""

    json_root = layout.json_config
    ingest = _read_optional_json(json_root / "Input-Manifests" / "nas_ingest.json")
    original_ingest = _read_optional_json(
        json_root / "Stage-Receipts" / "original_ingest.json"
    )
    model_preflight = _read_optional_json(json_root / "model_runtime_preflight.json")
    media_preflight = _read_optional_json(json_root / "media_pipeline_preflight.json")
    telemetry = _read_optional_json(json_root / "resource_telemetry.json")
    quality = _read_optional_json(json_root / "quality_acceptance.json")
    recall = _read_optional_json(json_root / "key_material_recall_eval.json")

    stage_summaries = telemetry.get("stage_summaries") or {}

    def peak(metric: str) -> dict[str, Any]:
        candidates = []
        for stage, values in stage_summaries.items():
            metric_values = values.get(metric) or {}
            if metric_values.get("max") is not None:
                candidates.append((float(metric_values["max"]), str(stage)))
        if not candidates:
            return {"value": None, "stage": None}
        value, stage = max(candidates)
        return {"value": round(value, 3), "stage": stage}

    calls = run_metrics.get("mllm_calls") or []
    status_counts = Counter(str(item.get("status") or "unknown") for item in calls)
    model_counts = Counter(str(item.get("model") or "unknown") for item in calls)
    runtime = model_preflight.get("runtime") or {}
    roles = runtime.get("roles") or {}
    source_validation = ingest.get("source_validation") or {}
    recall_threshold = next(
        (
            item
            for item in recall.get("threshold_results") or []
            if abs(float(item.get("temporal_iou_threshold") or 0) - 0.5) < 1e-9
        ),
        {},
    )
    base_url = str(config.get("mllm", {}).get("base_url") or "")
    provider = "Volcengine Ark" if "ark.cn-beijing.volces.com" in base_url else "configured MLLM"
    return {
        "source": {
            "input_mode": ingest.get("input_mode")
            or run_metrics.get("performance_mode", {}).get("input_mode"),
            "source_copy_bytes": original_ingest.get(
                "source_copy_bytes", ingest.get("copied_source_bytes")
            ),
            "continuous_source_copies_created": ingest.get(
                "continuous_source_copies_created"
            ),
            "verified_file_count": source_validation.get("verified_file_count"),
            "fresh_stat_count": source_validation.get("fresh_stat_count"),
            "cache_hit_count": source_validation.get("cache_hit_count"),
        },
        "tensorrt": {
            "version": runtime.get("tensorrt_version"),
            "role_count": len(roles),
            "all_deserialized": bool(roles)
            and all(bool(item.get("deserialized")) for item in roles.values()),
            "build_batches": sorted(
                {
                    int(item["build_batch"])
                    for item in roles.values()
                    if item.get("build_batch") is not None
                }
            ),
        },
        "video_encoder": {
            "status": media_preflight.get("status"),
            "selected": model_preflight.get("video_encoder", {}).get(
                "selected_encoder"
            ),
            "usable": model_preflight.get("video_encoder", {}).get(
                "requested_encoder_usable"
            ),
            "software_fallback_active": model_preflight.get(
                "video_encoder", {}
            ).get("software_fallback_active"),
        },
        "mllm": {
            "provider": provider,
            "call_count": len(calls),
            "completed_count": status_counts.get("completed", 0),
            "failed_count": len(calls) - status_counts.get("completed", 0),
            "cache_reused_call_count": sum(
                bool(item.get("cache_reused")) for item in calls
            ),
            "models": [name for name, _ in model_counts.most_common()],
        },
        "telemetry": {
            "backend": telemetry.get("gpu_telemetry_backend"),
            "sample_count": telemetry.get("sample_count"),
            "sampling_error_count": telemetry.get("monitor_health", {}).get(
                "sampling_error_count"
            ),
            "peaks": {
                "gpu_compute_percent": peak("gpu_compute_percent"),
                "gpu_memory_used_mib": peak("gpu_memory_used_mib"),
                "host_memory_percent": peak("memory_percent"),
                "nvdec_percent": peak("nvdec_percent"),
                "nvenc_percent": peak("nvenc_percent"),
                "gpu_power_w": peak("gpu_power_w"),
                "gpu_temperature_c": peak("gpu_temperature_c"),
            },
        },
        "quality": {
            "status": quality.get("status"),
            "boundary_precision": quality.get("experiment_boundaries", {}).get(
                "precision"
            ),
            "boundary_recall": quality.get("experiment_boundaries", {}).get(
                "recall"
            ),
            "key_material_precision_at_iou_0_5": recall_threshold.get("precision"),
            "key_material_recall_at_iou_0_5": recall_threshold.get("recall"),
            "key_material_f1_at_iou_0_5": recall_threshold.get("f1"),
            "truth_authority": recall.get("authority"),
        },
        "provenance": {
            "source_ingest": "JSON-Config-Files/Input-Manifests/nas_ingest.json",
            "model_preflight": "JSON-Config-Files/model_runtime_preflight.json",
            "media_preflight": "JSON-Config-Files/media_pipeline_preflight.json",
            "resource_telemetry": "JSON-Config-Files/resource_telemetry.json",
            "quality_acceptance": "JSON-Config-Files/quality_acceptance.json",
            "key_material_recall_eval": "JSON-Config-Files/key_material_recall_eval.json",
        },
    }


def _local_report_date(summary: RunSummary, timezone_name: str) -> str:
    try:
        report_timezone = ZoneInfo(timezone_name)
    except Exception:
        report_timezone = timezone(timedelta(hours=8), name="Asia/Shanghai fallback")
    try:
        created = datetime.fromisoformat(summary.created_at.replace("Z", "+00:00"))
        return created.astimezone(report_timezone).date().isoformat()
    except (ValueError, TypeError, KeyError):
        return datetime.now(report_timezone).date().isoformat()


def build_daily_report(
    summary: RunSummary,
    run_metrics: dict[str, Any],
    evidence_eval: dict[str, Any],
    config: dict[str, Any],
    quality_acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a deterministic report from accepted package evidence only."""

    report_config = config.get("daily_report", {})
    configured_template_id = str(
        report_config.get("template_id") or DEFAULT_DAILY_TEMPLATE_ID
    )
    template_path = _daily_template_path(configured_template_id)
    template = load_daily_report_template(configured_template_id)
    timezone_name = str(report_config.get("timezone", "Asia/Shanghai"))
    report_date = _local_report_date(summary, timezone_name)
    event_by_id = {event.event_id: event for event in summary.events}
    accepted_event_ids = {
        event_id
        for group in summary.experiment_groups
        for event_id in group.key_event_ids
    }
    accepted_physical_changes = [
        item
        for item in summary.physical_change_log
        if item.event_id in accepted_event_ids
    ]
    role_by_view = {view.view_id: view.role.value for view in summary.views}
    timeline = []
    action_counts: Counter[str] = Counter()
    uncertainties: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    if evidence_eval.get("observation_status") == "no_accepted_observation":
        uncertainties.append(
            {
                "scope": "run",
                "id": summary.experiment_id,
                "items": [
                    "No accepted experiment segment was observed; this does not prove that no physical action occurred."
                ],
            }
        )

    from .activity_review import assessment
    for group in sorted(summary.experiment_groups, key=lambda item: item.global_start_ms):
        understanding = group.model_understanding or {}
        steps = []
        for raw in understanding.get("steps") or []:
            steps.append(
                {
                    "step_index": raw.get("step_index"),
                    "start_global_ms": raw.get("start_global_ms"),
                    "end_global_ms": raw.get("end_global_ms"),
                    "start_timecode": _clock(raw.get("start_global_ms")),
                    "end_timecode": _clock(raw.get("end_global_ms")),
                    "current_step": raw.get("current_step"),
                    "operation_title": raw.get("operation_title"),
                    "observed_result": raw.get("observed_result"),
                    "time_scope": raw.get("time_scope"),
                    "next_step": raw.get("next_step"),
                    "next_step_status": raw.get("next_step_status") or "unknown",
                    "next_step_evidence": raw.get("next_step_evidence") or {},
                    "source_next_step": raw.get("source_next_step"),
                    "source_operation_records": raw.get("source_operation_records") or [],
                    "supporting_event_ids": raw.get("supporting_event_ids") or [],
                    "objects": raw.get("objects") or [],
                    "physical_change": raw.get("physical_change"),
                    "supporting_views": raw.get("supporting_views") or [],
                    "confidence": raw.get("confidence"),
                    "speech_segment_ids": raw.get("speech_segment_ids") or [],
                    "claim_class": "supported_model_understanding",
                }
            )
        group_uncertainties = list(understanding.get("uncertainties") or [])
        if group_uncertainties:
            uncertainties.append(
                {
                    "scope": "experiment_group",
                    "id": group.group_id,
                    "items": group_uncertainties,
                }
            )
        key_events = []
        for event_id in group.key_event_ids:
            event = event_by_id.get(event_id)
            if event is None:
                continue
            action_counts[event.action_type.value] += 1
            event_understanding = event.model_understanding or {}
            if event.uncertainty or event_understanding.get("uncertainties"):
                uncertainties.append(
                    {
                        "scope": "key_event",
                        "id": event.event_id,
                        "items": [*event.uncertainty, *(event_understanding.get("uncertainties") or [])],
                    }
                )
            cross_view = event_understanding.get("cross_view_consistency")
            if isinstance(cross_view, str) and any(
                word in cross_view.lower() for word in ("矛盾", "冲突", "contradict", "inconsistent")
            ):
                contradictions.append(
                    {"event_id": event.event_id, "cross_view_consistency": cross_view}
                )
            key_events.append(
                {
                    "event_id": event.event_id,
                    "action_type": event.action_type.value,
                    "action_label": ACTION_LABELS[event.action_type.value],
                    "start_global_ms": event.global_start_ms,
                    "end_global_ms": event.global_end_ms,
                    "peak_global_ms": event.key_global_ms,
                    "peak_timecode": _clock(event.key_global_ms),
                    "objects": event.objects,
                    "confidence": event.confidence,
                    "supporting_views": event.supporting_views,
                    "supporting_roles": [role.value for role in event.supporting_roles],
                    "observed_evidence": event.audit_reason,
                    "speech_interpretation": event_understanding.get("speech_interpretation"),
                    "speech_context": event_understanding.get("speech_context"),
                    "current_step": event_understanding.get("current_step"),
                    "next_step": event_understanding.get("next_step"),
                    "next_step_status": (
                        (event_understanding.get("next_step_evidence") or {}).get(
                            "status"
                        )
                        or "unknown"
                    ),
                    "next_step_evidence": event_understanding.get(
                        "next_step_evidence"
                    )
                    or {},
                    "model_status": event_understanding.get("status"),
                    "model_confidence": event_understanding.get("confidence"),
                    "model_usage": event_understanding.get("usage") or {},
                    "physical_change": event_understanding.get("physical_change") or {},
                    "key_frames": event.key_frames,
                    "key_clips": event.key_clips,
                    "aligned_key_frame": event.key_frames.get("aligned_first_third"),
                    "aligned_key_clip": event.key_clips.get("aligned_first_third"),
                }
            )
        group_event_ids = set(group.key_event_ids)
        group_changes = [
            item.model_dump(mode="json")
            for item in accepted_physical_changes
            if item.event_id in group_event_ids
        ]
        group_change_counts = Counter(item["change_type"] for item in group_changes)
        group_change_objects = Counter(
            str(name) for item in group_changes for name in item.get("object_names") or []
        )
        group_action_counts = Counter(
            str(item["action_type"]) for item in key_events
        )
        duration_seconds = max(
            0.0, (group.global_end_ms - group.global_start_ms) / 1000.0
        )
        sparse_threshold = max(3, min(12, round(duration_seconds / 30.0)))
        representative_visual, evidence_gallery = _select_group_visuals(key_events)
        timeline.append(
            {
                "group_id": group.group_id,
                "experiment_name": ((assessment(group.model_dump(mode="json"))["label"] + "记录") if assessment(group.model_dump(mode="json"))["is_auxiliary"] else group.experiment_name),
                "source_experiment_name": group.experiment_name,
                "activity_assessment": assessment(group.model_dump(mode="json")),
                "experiment_name_en": group.experiment_name_en,
                "continuity_type": group.continuity_type,
                "workflow_kind": group.workflow_kind,
                "workflow_units": group.workflow_units,
                "completion_status": group.completion_status,
                "completion_reason": group.completion_reason,
                "boundary_extension_requires_step_review": group.boundary_extension_requires_step_review,
                "continuity_reason": group.continuity_reason,
                "atomic_experiment_ids": group.atomic_experiment_ids,
                "start_global_ms": group.global_start_ms,
                "end_global_ms": group.global_end_ms,
                "start_timecode": _clock(group.global_start_ms),
                "end_timecode": _clock(group.global_end_ms),
                "duration_seconds": round(duration_seconds, 6),
                "participating_views": group.participating_views,
                "speech_interpretation": understanding.get("speech_interpretation"),
                "speech_context": understanding.get("speech_context"),
                "overall_summary": understanding.get("overall_summary"),
                "model_status": understanding.get("status"),
                "model_name": understanding.get("model"),
                "model_confidence": understanding.get("confidence"),
                "model_usage": understanding.get("usage") or {},
                "steps": steps,
                "key_events": key_events,
                "representative_visual": representative_visual,
                "evidence_gallery": evidence_gallery,
                "key_action_summary": [
                    {
                        "action_type": action_type,
                        "action_label": label,
                        "event_count": group_action_counts.get(action_type, 0),
                    }
                    for action_type, label in ACTION_LABELS.items()
                ],
                "key_material_low_recall_warning": duration_seconds >= 60.0
                and len(key_events) < sparse_threshold,
                "key_material_low_recall_threshold": sparse_threshold,
                "physical_change_count": len(group_changes),
                "physical_change_summary": [
                    {"change_type": name, "count": count}
                    for name, count in group_change_counts.most_common()
                ],
                "top_changed_objects": [
                    {"object": name, "count": count}
                    for name, count in group_change_objects.most_common(12)
                ],
                "videos": group.videos,
            }
        )

    alignment = [item.model_dump(mode="json") for item in summary.alignments]
    for item in alignment:
        item["role"] = role_by_view.get(str(item.get("view_id")), "unknown")
    alignment_summary = {
        "view_count": len(alignment),
        "aligned": sum(item.get("state") == "aligned" for item in alignment),
        "uncertain": sum(item.get("state") == "uncertain" for item in alignment),
        "failed": sum(item.get("state") == "failed" for item in alignment),
        "mean_confidence": round(
            sum(float(item.get("confidence") or 0) for item in alignment) / max(1, len(alignment)), 6
        ),
        "views": alignment,
    }
    token_ledger = run_metrics.get("tokens") or {}
    total_tokens = token_ledger.get("run_total") or {}
    daily_tokens = token_ledger.get("daily_report") or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
    }
    quality_acceptance = dict(quality_acceptance or {})
    boundary_quality = quality_acceptance.get("experiment_boundaries") or {}
    key_event_quality = quality_acceptance.get("key_event_recall") or {}
    evidence_level = str(
        quality_acceptance.get("evidence_level")
        or (
            "dataset_measured"
            if boundary_quality.get("evaluated") and key_event_quality.get("evaluated")
            else "structural_only"
        )
    )
    report = {
        "schema_version": "visioncortex-lab-daily-report/2.0",
        "template_id": template["template_id"],
        "template_schema_version": template["schema_version"],
        "template_sha256": _sha256(template_path),
        "presentation_contract": {
            "daily_template_id": template["template_id"],
            "daily_template_sha256": _sha256(template_path),
            "professional_template_id": PROFESSIONAL_TEMPLATE_ID,
            "professional_template_sha256": _sha256(PROFESSIONAL_TEMPLATE_PATH),
            "professional_renderer_sha256": _sha256(PROFESSIONAL_RENDERER_PATH),
            "brand": "VisionCortex",
            "dual_view_visuals_only": True,
            "reader_layers": ["decision", "experiment", "audit"],
        },
        "report_id": f"{summary.experiment_id}-{report_date}",
        "report_date": report_date,
        "timezone": timezone_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": summary.experiment_id,
        "source_policy": {
            "evidence_package_only": True,
            "accepted_key_events_only": True,
            "additional_mllm_calls": False,
            "layout_editable_by_model": False,
            "report_narrative_source": "accepted_existing_model_understanding",
            "additional_model_tokens": daily_tokens,
            "claim_classes": {
                "observed_fact": "directly traceable to accepted CV/cross-view evidence",
                "supported_model_understanding": "model interpretation grounded in archived frames or clips",
                "uncertain": "insufficient or conflicting evidence; automatically quarantined from formal facts",
            },
        },
        "overview": {
            "input_view_count": summary.stats.get("input_view_count", len(summary.views)),
            "first_person_views": sum(view.role.value == "first_person" for view in summary.views),
            "third_person_views": sum(view.role.value == "third_person" for view in summary.views),
            "experiment_group_count": sum(not assessment(g.model_dump(mode="json"))["is_auxiliary"] for g in summary.experiment_groups),
            "activity_record_count": len(summary.experiment_groups),
            "auxiliary_activity_count": sum(assessment(g.model_dump(mode="json"))["is_auxiliary"] for g in summary.experiment_groups),
            "key_event_count": sum(len(item["key_events"]) for item in timeline),
            "physical_change_count": len(accepted_physical_changes),
            "accepted_event_count": summary.stats.get("accepted_event_count"),
            "rejected_event_count": summary.stats.get("rejected_event_count"),
            "candidate_event_count": len(summary.events),
            "evidence_package_eval_passed": bool(evidence_eval.get("passed")),
            "evidence_package_eval_check_count": len(evidence_eval.get("checks") or []),
            "evidence_package_eval_failure_count": len(evidence_eval.get("failures") or []),
            "observation_status": evidence_eval.get("observation_status"),
            "observation_evidence_classification": evidence_eval.get(
                "observation_evidence_classification"
            ),
            "negative_action_claim_supported": evidence_eval.get(
                "negative_action_claim_supported"
            ),
            "quality_acceptance_passed": quality_acceptance.get("passed") is True,
            "quality_acceptance_status": quality_acceptance.get("status")
            or "not_available",
            "evidence_level": evidence_level,
            "per_run_accuracy_measured": evidence_level == "dataset_measured",
            "representative_visual_count": sum(
                item.get("representative_visual") is not None for item in timeline
            ),
            "professional_visual_count": sum(
                len(item.get("evidence_gallery") or [])
                + int(item.get("representative_visual") is not None)
                for item in timeline
            ),
        },
        "alignment_summary": alignment_summary,
        "experiment_timeline": timeline,
        "action_summary": [
            {
                "action_type": action_type,
                "action_label": label,
                "event_count": action_counts.get(action_type, 0),
            }
            for action_type, label in ACTION_LABELS.items()
        ],
        "physical_change_log": [
            item.model_dump(mode="json") for item in accepted_physical_changes
        ],
        "uncertainties": uncertainties,
        "contradictions": contradictions,
        "quality_acceptance": {
            "passed": quality_acceptance.get("passed") is True,
            "status": quality_acceptance.get("status") or "not_available",
            "evidence_level": evidence_level,
            "structural_passed": quality_acceptance.get("structural_passed"),
            "formal_accuracy_claim_allowed": quality_acceptance.get(
                "formal_accuracy_claim_allowed", False
            ),
            "boundary_evaluated": bool(boundary_quality.get("evaluated")),
            "key_event_recall_evaluated": bool(key_event_quality.get("evaluated")),
            "source": "JSON-Config-Files/quality_acceptance.json",
        },
        "performance": {
            "total_duration_seconds": run_metrics.get("total_duration_seconds"),
            "preprocessing_sla": run_metrics.get("preprocessing_sla") or {},
            "full_run_preprocessing": run_metrics.get("full_run_preprocessing") or {},
            "stage_durations": run_metrics.get("stage_durations") or [],
            "tokens": token_ledger,
            "total_input_tokens": total_tokens.get("input_tokens"),
            "total_output_tokens": total_tokens.get("output_tokens"),
            "total_tokens": total_tokens.get("total_tokens"),
            "runtime_audit": run_metrics.get("runtime_audit") or {},
        },
        "algorithmic_acceptance": {
            "status": "accepted",
            "required_for_completion": False,
            "decision_authority": "algorithmic_fail_closed_gates_only",
            "algorithmic_acceptance_source": (
                "JSON-Config-Files/evidence_package_eval.json"
            ),
            "uncertain_evidence_policy": "automatic_quarantine",
        },
        "provenance": {
            "evidence_package": "JSON-Config-Files/evidence_package.json",
            "evidence_package_eval": "JSON-Config-Files/evidence_package_eval.json",
            "run_metrics": "JSON-Config-Files/run_metrics.json",
            "experiment_understanding": "JSON-Config-Files/experiment_group_understanding.json",
            "key_material_selection": "JSON-Config-Files/key_material_selection.json",
            "key_material_understanding": "JSON-Config-Files/key_material_model_understanding.json",
        },
    }
    return report


def evaluate_daily_report(report: dict[str, Any], summary: RunSummary) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    timeline = report.get("experiment_timeline") or []
    report_event_ids = [event["event_id"] for group in timeline for event in group.get("key_events") or []]
    expected_event_ids = [event_id for group in summary.experiment_groups for event_id in group.key_event_ids]
    expected_change_ids = [
        item.change_id
        for item in summary.physical_change_log
        if item.event_id in set(expected_event_ids)
    ]
    report_changes = report.get("physical_change_log") or []
    report_change_ids = [str(item.get("change_id")) for item in report_changes]
    timeline_change_count = sum(
        int(group.get("physical_change_count") or 0) for group in timeline
    )
    template_id = str(report.get("template_id") or DEFAULT_DAILY_TEMPLATE_ID)
    template_path = _daily_template_path(template_id)
    template = load_daily_report_template(template_id)
    required_section_sources = {
        "daily_overview": "overview",
        "experiment_matrix": "experiment_timeline",
        "alignment_audit": "alignment_summary",
        "experiment_evidence": "experiment_timeline",
        "action_summary": "action_summary",
        "quality_and_uncertainty": "uncertainties",
        "performance_and_tokens": "performance",
        "experiment_briefs": "experiment_timeline",
        "attention_and_handoff": "uncertainties",
        "cost_and_timing": "performance",
        "algorithmic_acceptance": "algorithmic_acceptance",
    }
    missing_required_sections = [
        section["id"]
        for section in template.get("sections") or []
        if section.get("required")
        and required_section_sources.get(section["id"]) not in report
    ]
    check(
        "fixed_template_contract",
        report.get("template_id") == template["template_id"]
        and report.get("template_sha256") == _sha256(template_path)
        and report.get("source_policy", {}).get("layout_editable_by_model") is False,
        "The renderer must use a versioned fixed template; the model cannot alter layout or sections.",
    )
    check(
        "required_template_sections_present",
        not missing_required_sections,
        "missing=" + (",".join(missing_required_sections) or "none"),
    )
    check(
        "experiment_group_count_matches",
        len(timeline) == len(summary.experiment_groups),
        f"report={len(timeline)} evidence_package={len(summary.experiment_groups)}",
    )
    check(
        "key_event_ids_match",
        sorted(report_event_ids) == sorted(expected_event_ids),
        f"report={len(report_event_ids)} evidence_package={len(expected_event_ids)}",
    )
    check(
        "accepted_physical_change_ids_match",
        sorted(report_change_ids) == sorted(expected_change_ids),
        f"report={len(report_change_ids)} accepted_evidence={len(expected_change_ids)}",
    )
    check(
        "physical_change_counts_reconcile",
        int(report.get("overview", {}).get("physical_change_count") or 0)
        == len(report_changes)
        == timeline_change_count,
        "overview="
        f"{report.get('overview', {}).get('physical_change_count')}; "
        f"log={len(report_changes)}; timeline={timeline_change_count}",
    )
    check(
        "all_experiments_have_bounded_timeline",
        all(item["start_global_ms"] < item["end_global_ms"] for item in timeline),
        "Every experiment must have a positive bounded interval.",
    )
    check(
        "all_key_events_have_material_links",
        all(
            event.get("key_frames") and event.get("key_clips")
            for group in timeline
            for event in group.get("key_events") or []
        ),
        "Every key event must link to archived key frames and clips.",
    )
    step_failures: list[str] = []
    for group in timeline:
        group_id = str(group.get("group_id") or "")
        group_event_ids = {
            str(item.get("event_id")) for item in group.get("key_events") or []
        }
        steps = list(group.get("steps") or [])
        if not steps:
            step_failures.append(f"{group_id}:steps_empty")
            continue
        previous_start: float | None = None
        for index, step in enumerate(steps):
            prefix = f"{group_id}:step-{index + 1}"
            start = step.get("start_global_ms")
            end = step.get("end_global_ms")
            if not str(step.get("current_step") or "").strip():
                step_failures.append(f"{prefix}:current_step_empty")
            try:
                start_value = float(start)
                end_value = float(end)
            except (TypeError, ValueError):
                step_failures.append(f"{prefix}:time_missing")
                continue
            if not (
                float(group["start_global_ms"])
                <= start_value
                <= end_value
                <= float(group["end_global_ms"])
            ):
                step_failures.append(f"{prefix}:time_outside_group")
            if previous_start is not None and start_value < previous_start:
                step_failures.append(f"{prefix}:step_order_regression")
            previous_start = start_value
            supporting = {
                str(item) for item in step.get("supporting_event_ids") or []
            }
            if not supporting:
                step_failures.append(f"{prefix}:supporting_events_empty")
            elif not supporting.issubset(group_event_ids):
                step_failures.append(f"{prefix}:unknown_supporting_event")
    check(
        "experiment_steps_are_bounded_and_traceable",
        not step_failures,
        "failures=" + (",".join(step_failures[:20]) or "none"),
    )
    check(
        "daily_report_adds_no_model_tokens",
        (report.get("source_policy", {}).get("additional_model_tokens") or {}).get("total_tokens", 0) == 0,
        "Daily report is deterministic and uses existing accepted understanding.",
    )
    selected_visuals = [
        visual
        for group in timeline
        for visual in [
            group.get("representative_visual"),
            *(group.get("evidence_gallery") or []),
        ]
        if visual
    ]
    check(
        "report_visuals_are_dual_view_and_traceable",
        all(
            visual.get("event_id") in report_event_ids
            and visual.get("image_path")
            and visual.get("clip_path")
            and {"first_person", "third_person"}.issubset(
                set(visual.get("visual_roles") or [])
            )
            for visual in selected_visuals
        ),
        f"selected_visuals={len(selected_visuals)}; all must reference accepted dual-view events",
    )
    check(
        "professional_template_contract",
        report.get("presentation_contract", {}).get("professional_template_id")
        == PROFESSIONAL_TEMPLATE_ID
        and report.get("presentation_contract", {}).get(
            "professional_template_sha256"
        )
        == _sha256(PROFESSIONAL_TEMPLATE_PATH),
        "The professional PDF has a separate versioned presentation contract.",
    )
    check(
        "professional_renderer_integrity",
        report.get("presentation_contract", {}).get(
            "professional_renderer_sha256"
        )
        == _sha256(PROFESSIONAL_RENDERER_PATH),
        "The report records the exact deterministic renderer source hash.",
    )
    check(
        "evidence_package_eval_passed",
        bool(report.get("overview", {}).get("evidence_package_eval_passed")),
        "The underlying evidence package must pass before report promotion.",
    )
    check(
        "quality_acceptance_passed",
        report.get("quality_acceptance", {}).get("passed") is True,
        "The automatic structural/quality gate must pass before report promotion.",
    )
    check(
        "algorithmic_completion_has_no_human_dependency",
        report.get("algorithmic_acceptance", {}).get("required_for_completion")
        is False
        and report.get("algorithmic_acceptance", {}).get("decision_authority")
        == "algorithmic_fail_closed_gates_only",
        "Uncertain evidence is quarantined automatically and never waits for a reviewer.",
    )
    return {
        "schema_version": "visioncortex-lab-daily-report-eval/1.0",
        "passed": all(item["passed"] for item in checks),
        "check_count": len(checks),
        "checks": checks,
    }


def generate_daily_report_archive(
    layout: ArchiveLayout,
    summary: RunSummary,
    run_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    evidence_eval_path = layout.json_config / "evidence_package_eval.json"
    evidence_eval = json.loads(evidence_eval_path.read_text(encoding="utf-8-sig"))
    if not evidence_eval.get("passed"):
        raise RuntimeError("Evidence package did not pass; daily report generation refused")
    quality_path = layout.json_config / "quality_acceptance.json"
    if not quality_path.is_file():
        raise RuntimeError("Quality acceptance is missing; daily report generation refused")
    quality_acceptance = json.loads(quality_path.read_text(encoding="utf-8-sig"))
    if quality_acceptance.get("passed") is not True:
        raise RuntimeError("Quality acceptance did not pass; daily report generation refused")
    effective_metrics = dict(run_metrics)
    effective_metrics["runtime_audit"] = collect_runtime_audit(
        layout, effective_metrics, config
    )
    acceptance_path = layout.json_config / "acceptance_report.json"
    if acceptance_path.is_file():
        try:
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8-sig"))
            full_run = acceptance.get("preprocessing_full_run")
            if isinstance(full_run, dict):
                effective_metrics["full_run_preprocessing"] = full_run
        except (json.JSONDecodeError, OSError):
            pass
    report = build_daily_report(
        summary,
        effective_metrics,
        evidence_eval,
        config,
        quality_acceptance,
    )
    report_date = report["report_date"]
    report_dir = layout.daily_reports / report_date
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Lab-Daily-Report-{report_date}"
    json_path = report_dir / f"{stem}.json"
    markdown_path = report_dir / f"{stem}.md"
    html_path = report_dir / f"{stem}.html"
    eval_path = report_dir / "Daily-Report-Eval.json"
    acceptance_path = report_dir / "Automatic-Acceptance.json"
    write_json(json_path, report)
    markdown_path.write_text(render_daily_markdown(report), encoding="utf-8")
    html_path.write_text(render_daily_html(report), encoding="utf-8")
    evaluation = evaluate_daily_report(report, summary)
    write_json(eval_path, evaluation)
    write_json(acceptance_path, report["algorithmic_acceptance"])
    if not evaluation["passed"]:
        raise RuntimeError("Daily report evaluation failed")
    pdf_path = (
        layout.professional_pdfs
        / f"VisionCortex-Professional-Evidence-Report-{report_date}.pdf"
    )
    if config.get("daily_report", {}).get("generate_pdf", True):
        render_professional_pdf(pdf_path, report, layout.root)
    professional_manifest_path = layout.json_config / "professional_report_manifest.json"
    professional_manifest = {
        "schema_version": "visioncortex-professional-report-manifest/1.0",
        "template_id": PROFESSIONAL_TEMPLATE_ID,
        "template_sha256": _sha256(PROFESSIONAL_TEMPLATE_PATH),
        "renderer_sha256": _sha256(PROFESSIONAL_RENDERER_PATH),
        "report_date": report_date,
        "status": "generated" if pdf_path.is_file() else "not_generated",
        "pdf": archive_relative_posix(pdf_path, layout.root)
        if pdf_path.is_file()
        else None,
        "visual_policy": {
            "accepted_key_materials_only": True,
            "dual_view_only": True,
            "selected_visual_count": report["overview"]["professional_visual_count"],
        },
        "narrative_source": "accepted_existing_model_understanding",
        "additional_model_tokens": 0,
        "checksum_sha256": _sha256(pdf_path) if pdf_path.is_file() else None,
    }
    write_json(professional_manifest_path, professional_manifest)
    artifacts = {
        "report_date": report_date,
        "json": archive_relative_posix(json_path, layout.root),
        "markdown": archive_relative_posix(markdown_path, layout.root),
        "html": archive_relative_posix(html_path, layout.root),
        "pdf": archive_relative_posix(pdf_path, layout.root) if pdf_path.is_file() else None,
        "professional_report_manifest": str(
            archive_relative_posix(professional_manifest_path, layout.root)
        ),
        "daily_template_id": report["template_id"],
        "professional_template_id": PROFESSIONAL_TEMPLATE_ID,
        "representative_visual_count": report["overview"][
            "representative_visual_count"
        ],
        "professional_visual_count": report["overview"][
            "professional_visual_count"
        ],
        "evaluation": archive_relative_posix(eval_path, layout.root),
        "algorithmic_acceptance": archive_relative_posix(
            acceptance_path, layout.root
        ),
        "passed": True,
        "additional_model_tokens": 0,
        "generation_duration_seconds": round(time.perf_counter() - started, 6),
        "checksums": {
            path.name: _sha256(path)
            for path in (
                json_path,
                markdown_path,
                html_path,
                eval_path,
                acceptance_path,
                pdf_path,
                professional_manifest_path,
            )
            if path.is_file()
        },
    }
    write_json(layout.json_config / "daily_report_manifest.json", artifacts)
    return artifacts


def generate_daily_report_from_archive(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    layout = ArchiveLayout(root.resolve())
    summary = RunSummary.model_validate_json(
        (layout.json_config / "evidence_package.json").read_text(encoding="utf-8-sig")
    )
    from .speech_refresh import apply
    from .operation_review import apply as apply_operations
    from .schemas import ExperimentGroup
    from .activity_review import apply as apply_activity
    summary.experiment_groups = [ExperimentGroup.model_validate(item) for item in
                                 apply_activity(root, apply_operations(root, apply(root, [group.model_dump(mode="json") for group in summary.experiment_groups])))]
    run_metrics = json.loads((layout.json_config / "run_metrics.json").read_text(encoding="utf-8-sig"))
    return generate_daily_report_archive(layout, summary, run_metrics, config)
