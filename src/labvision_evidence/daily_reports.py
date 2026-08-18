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
from .pathing import archive_relative_posix
from .report_presentations import (
    render_daily_html,
    render_daily_markdown,
    render_professional_pdf,
)
from .schemas import RunSummary


ACTION_LABELS = {
    "hand_object_contact": "手部与物体接触",
    "object_movement": "物体移动",
    "liquid_movement": "液体移动",
    "container_state_change": "容器状态变化",
    "device_panel_operation": "设备面板操作",
}
TEMPLATE_DIRECTORY = Path(__file__).with_name("templates")
DEFAULT_DAILY_TEMPLATE_ID = "VC-LAB-DAILY-REPORT-V2"
PROFESSIONAL_TEMPLATE_ID = "VC-PROFESSIONAL-EVIDENCE-REPORT-V1"
PROFESSIONAL_TEMPLATE_PATH = TEMPLATE_DIRECTORY / f"{PROFESSIONAL_TEMPLATE_ID}.json"


def _daily_template_path(template_id: str) -> Path:
    path = TEMPLATE_DIRECTORY / f"{template_id}.json"
    if not path.is_file():
        raise ValueError(f"Unsupported daily-report template: {template_id}")
    return path


def load_daily_report_template(template_id: str = DEFAULT_DAILY_TEMPLATE_ID) -> dict[str, Any]:
    return json.loads(_daily_template_path(template_id).read_text(encoding="utf-8"))


def _select_group_visuals(key_events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select deterministic dual-view images without another model call."""

    eligible = [
        item
        for item in key_events
        if item.get("aligned_key_frame")
        and item.get("aligned_key_clip")
        and {"first_person", "third_person"}.issubset(set(item.get("supporting_roles") or []))
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
        return {
            "event_id": item["event_id"],
            "action_type": item["action_type"],
            "action_label": item["action_label"],
            "peak_global_ms": item["peak_global_ms"],
            "peak_timecode": item["peak_timecode"],
            "objects": item["objects"],
            "confidence": item["confidence"],
            "supporting_views": item["supporting_views"],
            "supporting_roles": item["supporting_roles"],
            "image_path": item["aligned_key_frame"],
            "clip_path": item["aligned_key_clip"],
            "claim_class": "observed_fact",
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
    role_by_view = {view.view_id: view.role.value for view in summary.views}
    timeline = []
    action_counts: Counter[str] = Counter()
    uncertainties: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []

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
                    "next_step": raw.get("next_step"),
                    "objects": raw.get("objects") or [],
                    "physical_change": raw.get("physical_change"),
                    "supporting_views": raw.get("supporting_views") or [],
                    "confidence": raw.get("confidence"),
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
                    "current_step": event_understanding.get("current_step"),
                    "next_step": event_understanding.get("next_step"),
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
        group_changes = [
            item.model_dump(mode="json")
            for item in summary.physical_change_log
            if group.global_start_ms <= item.global_ms <= group.global_end_ms
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
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
                "continuity_type": group.continuity_type,
                "continuity_reason": group.continuity_reason,
                "atomic_experiment_ids": group.atomic_experiment_ids,
                "start_global_ms": group.global_start_ms,
                "end_global_ms": group.global_end_ms,
                "start_timecode": _clock(group.global_start_ms),
                "end_timecode": _clock(group.global_end_ms),
                "duration_seconds": round(duration_seconds, 6),
                "participating_views": group.participating_views,
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
                "uncertain": "insufficient or conflicting evidence; human review recommended",
            },
        },
        "overview": {
            "input_view_count": summary.stats.get("input_view_count", len(summary.views)),
            "first_person_views": sum(view.role.value == "first_person" for view in summary.views),
            "third_person_views": sum(view.role.value == "third_person" for view in summary.views),
            "experiment_group_count": len(summary.experiment_groups),
            "key_event_count": sum(len(item["key_events"]) for item in timeline),
            "physical_change_count": len(summary.physical_change_log),
            "accepted_event_count": summary.stats.get("accepted_event_count"),
            "rejected_event_count": summary.stats.get("rejected_event_count"),
            "candidate_event_count": len(summary.events),
            "evidence_package_eval_passed": bool(evidence_eval.get("passed")),
            "evidence_package_eval_check_count": len(evidence_eval.get("checks") or []),
            "evidence_package_eval_failure_count": len(evidence_eval.get("failures") or []),
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
        "physical_change_log": [item.model_dump(mode="json") for item in summary.physical_change_log],
        "uncertainties": uncertainties,
        "contradictions": contradictions,
        "performance": {
            "total_duration_seconds": run_metrics.get("total_duration_seconds"),
            "preprocessing_sla": run_metrics.get("preprocessing_sla") or {},
            "full_run_preprocessing": run_metrics.get("full_run_preprocessing") or {},
            "stage_durations": run_metrics.get("stage_durations") or [],
            "tokens": token_ledger,
            "total_input_tokens": total_tokens.get("input_tokens"),
            "total_output_tokens": total_tokens.get("output_tokens"),
            "total_tokens": total_tokens.get("total_tokens"),
        },
        "human_review": {
            "status": "pending",
            "reviewer_id": None,
            "reviewed_at": None,
            "comments": [],
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
        "human_review": "human_review",
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
                set(visual.get("supporting_roles") or [])
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
        "evidence_package_eval_passed",
        bool(report.get("overview", {}).get("evidence_package_eval_passed")),
        "The underlying evidence package must pass before report promotion.",
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
    effective_metrics = dict(run_metrics)
    acceptance_path = layout.json_config / "acceptance_report.json"
    if acceptance_path.is_file():
        try:
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8-sig"))
            full_run = acceptance.get("preprocessing_full_run")
            if isinstance(full_run, dict):
                effective_metrics["full_run_preprocessing"] = full_run
        except (json.JSONDecodeError, OSError):
            pass
    report = build_daily_report(summary, effective_metrics, evidence_eval, config)
    report_date = report["report_date"]
    report_dir = layout.daily_reports / report_date
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Lab-Daily-Report-{report_date}"
    json_path = report_dir / f"{stem}.json"
    markdown_path = report_dir / f"{stem}.md"
    html_path = report_dir / f"{stem}.html"
    eval_path = report_dir / "Daily-Report-Eval.json"
    review_path = report_dir / "Human-Review.json"
    existing_review = None
    if review_path.is_file():
        try:
            existing_review = json.loads(review_path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            existing_review = None
    if isinstance(existing_review, dict) and existing_review.get("status") not in {None, "pending"}:
        report["human_review"] = existing_review
    write_json(json_path, report)
    markdown_path.write_text(render_daily_markdown(report), encoding="utf-8")
    html_path.write_text(render_daily_html(report), encoding="utf-8")
    evaluation = evaluate_daily_report(report, summary)
    write_json(eval_path, evaluation)
    if not review_path.is_file() or not isinstance(existing_review, dict):
        write_json(review_path, report["human_review"])
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
        "human_review": archive_relative_posix(review_path, layout.root),
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
                review_path,
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
    summary.experiment_id = root.resolve().name
    run_metrics = json.loads((layout.json_config / "run_metrics.json").read_text(encoding="utf-8-sig"))
    return generate_daily_report_archive(layout, summary, run_metrics, config)
