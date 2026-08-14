from __future__ import annotations

import hashlib
import html
import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .archive import ArchiveLayout, write_json
from .schemas import RunSummary


ACTION_LABELS = {
    "hand_object_contact": "手部与物体接触",
    "object_movement": "物体移动",
    "liquid_movement": "液体移动",
    "container_state_change": "容器状态变化",
    "device_panel_operation": "设备面板操作",
}
TEMPLATE_PATH = Path(__file__).with_name("templates") / "VC-LAB-DAILY-REPORT-V1.json"


def load_daily_report_template() -> dict[str, Any]:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


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
    template = load_daily_report_template()
    configured_template_id = str(
        report_config.get("template_id") or template["template_id"]
    )
    if configured_template_id != template["template_id"]:
        raise ValueError(
            "Unsupported daily-report template: "
            f"{configured_template_id}; expected {template['template_id']}"
        )
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
        "schema_version": "visioncortex-lab-daily-report/1.0",
        "template_id": template["template_id"],
        "template_schema_version": template["schema_version"],
        "template_sha256": _sha256(TEMPLATE_PATH),
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
    template = load_daily_report_template()
    required_section_sources = {
        "daily_overview": "overview",
        "experiment_matrix": "experiment_timeline",
        "alignment_audit": "alignment_summary",
        "experiment_evidence": "experiment_timeline",
        "action_summary": "action_summary",
        "quality_and_uncertainty": "uncertainties",
        "performance_and_tokens": "performance",
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
        report.get("template_id") == "VC-LAB-DAILY-REPORT-V1"
        and report.get("template_sha256") == _sha256(TEMPLATE_PATH)
        and report.get("source_policy", {}).get("layout_editable_by_model") is False,
        "The renderer must use the immutable V1 template; the model cannot alter layout or sections.",
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


def _markdown(report: dict[str, Any]) -> str:
    overview = report["overview"]
    performance = report["performance"]
    lines = [
        f"# 实验室日报 - {report['report_date']}",
        "",
        f"- 实验档案：`{report['experiment_id']}`",
        f"- 输入视角：{overview['input_view_count']} 路（第一人称 {overview['first_person_views']}，第三人称 {overview['third_person_views']}）",
        f"- 有界实验：{overview['experiment_group_count']} 个",
        f"- 关键事件：{overview['key_event_count']} 个",
        f"- 物理状态变化：{overview['physical_change_count']} 项",
        "- 日报新增模型调用：0；日报新增 Token：0",
        "",
        "> 本日报只汇总已验收证据。当前/下一步骤属于证据支持的模型理解，不等同于直接观察事实。",
        "",
        "## 实验时间线",
        "",
    ]
    for group in report["experiment_timeline"]:
        continuity = "连续实验" if group["continuity_type"] == "continuous" else "独立实验"
        lines.extend(
            [
                f"### {group['experiment_name']}（{continuity}）",
                "",
                f"- 相对时间：`{group['start_timecode']}` - `{group['end_timecode']}`",
                f"- 持续时间：{group['duration_seconds']:.3f} 秒",
                f"- 跨视角：{', '.join(group['participating_views'])}",
                f"- 模型摘要：{group.get('overall_summary') or '无'}",
                "",
                "| 步骤 | 时间 | 当前步骤 | 下一步骤 | 对象 | 置信度 |",
                "|---:|---|---|---|---|---:|",
            ]
        )
        for step in group["steps"]:
            objects = "、".join(str(item) for item in step["objects"])
            lines.append(
                f"| {step.get('step_index') or '-'} | {step['start_timecode']} - {step['end_timecode']} | "
                f"{step.get('current_step') or '未说明'} | {step.get('next_step') or '证据不足'} | "
                f"{objects or '未明确'} | {step.get('confidence') if step.get('confidence') is not None else '-'} |"
            )
        lines.extend(["", "关键事件："])
        for event in group["key_events"]:
            lines.append(
                f"- `{event['peak_timecode']}` [{event['event_id']}] {event['action_label']}；对象："
                f"{'、'.join(event['objects']) or '未明确'}"
            )
        lines.append("")
    lines.extend(["## 五类关键动作", "", "| 动作 | 事件数 |", "|---|---:|"])
    lines.extend(f"| {item['action_label']} | {item['event_count']} |" for item in report["action_summary"])
    lines.extend(
        [
            "",
            "## 质量与不确定性",
            "",
            f"- 时间对齐：{report['alignment_summary']['aligned']}/{report['alignment_summary']['view_count']} 路 aligned",
            f"- 平均对齐置信度：{report['alignment_summary']['mean_confidence']}",
            f"- 不确定性记录：{len(report['uncertainties'])} 组",
            f"- 跨视角矛盾：{len(report['contradictions'])} 项",
            f"- 证据包验收：{'通过' if overview['evidence_package_eval_passed'] else '未通过'}",
            "",
            "## 耗时与 Token",
            "",
            f"- 流水线总耗时：{_duration(performance.get('total_duration_seconds'))}",
            f"- 本次流水线预处理：{_duration((performance.get('preprocessing_sla') or {}).get('actual_seconds'))}（可能复用已验收 CV 账本）",
            f"- 全量六路预处理验收：{_duration((performance.get('full_run_preprocessing') or {}).get('seconds')) if (performance.get('full_run_preprocessing') or {}).get('seconds') is not None else '无独立全量验收记录'}",
            f"- 输入 Token：{performance.get('total_input_tokens') or 0:,}",
            f"- 输出 Token：{performance.get('total_output_tokens') or 0:,}",
            f"- 总 Token：{performance.get('total_tokens') or 0:,}",
            f"- 日报新增 Token：0",
            "",
            "## 人工复核",
            "",
            "- 状态：待复核",
            "- 复核人：__________",
            "- 复核时间：__________",
            "- 备注：________________________________________",
            "",
        ]
    )
    return "\n".join(lines)


def _html(report: dict[str, Any]) -> str:
    def e(value: Any) -> str:
        return html.escape(str(value or ""))

    experiments = []
    for group in report["experiment_timeline"]:
        rows = "".join(
            f"<tr><td>{e(step.get('step_index'))}</td><td>{e(step['start_timecode'])} - {e(step['end_timecode'])}</td>"
            f"<td>{e(step.get('current_step'))}</td><td>{e(step.get('next_step') or '证据不足')}</td>"
            f"<td>{e('、'.join(step.get('objects') or []))}</td></tr>"
            for step in group["steps"]
        )
        experiments.append(
            f"<section><h2>{e(group['experiment_name'])}</h2><p class='meta'>{e(group['start_timecode'])} - "
            f"{e(group['end_timecode'])} · {'连续实验' if group['continuity_type']=='continuous' else '独立实验'}</p>"
            f"<p>{e(group.get('overall_summary'))}</p><table><thead><tr><th>步骤</th><th>时间</th><th>当前步骤</th>"
            f"<th>下一步骤</th><th>对象</th></tr></thead><tbody>{rows}</tbody></table></section>"
        )
    actions = "".join(
        f"<article><strong>{e(item['event_count'])}</strong><span>{e(item['action_label'])}</span></article>"
        for item in report["action_summary"]
    )
    overview = report["overview"]
    perf = report["performance"]
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>实验室日报 {e(report['report_date'])}</title><style>
body{{margin:0;background:#eef4f4;color:#17343a;font:14px/1.65 'Microsoft YaHei',sans-serif}}main{{max-width:1180px;margin:28px auto;padding:0 20px}}header,section{{background:#fff;border:1px solid #d6e1e2;border-radius:16px;padding:24px;margin-bottom:16px}}h1,h2{{margin:0 0 10px}}.meta{{color:#688086}}.notice{{border-left:4px solid #c87c30;padding:12px;background:#fff8ef}}.stats,.actions{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:18px 0}}.stats article,.actions article{{background:#edf7f7;border-radius:12px;padding:14px;display:grid}}article strong{{font-size:22px;color:#126d69}}article span{{color:#688086}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e3ebec;text-align:left;vertical-align:top}}th{{color:#688086}}@media(max-width:800px){{.stats,.actions{{grid-template-columns:1fr 1fr}}table{{font-size:12px}}}}
</style></head><body><main><header><p class='meta'>VISIONCORTEX · EVIDENCE-BACKED DAILY REPORT</p><h1>实验室日报 · {e(report['report_date'])}</h1>
<p>{e(report['experiment_id'])}</p><div class='stats'><article><strong>{overview['input_view_count']}</strong><span>输入视角</span></article><article><strong>{overview['experiment_group_count']}</strong><span>有界实验</span></article><article><strong>{overview['key_event_count']}</strong><span>关键事件</span></article><article><strong>{overview['physical_change_count']}</strong><span>状态变化</span></article><article><strong>{perf.get('total_tokens') or 0:,}</strong><span>总 Token</span></article></div>
<p class='notice'>日报只汇总已验收证据；当前/下一步骤是证据支持的模型理解。日报新增模型调用与 Token 均为 0。</p></header>
{''.join(experiments)}<section><h2>五类关键动作</h2><div class='actions'>{actions}</div></section>
<section><h2>质量、耗时与成本</h2><p>对齐：{report['alignment_summary']['aligned']}/{report['alignment_summary']['view_count']} 路；平均置信度 {report['alignment_summary']['mean_confidence']}；不确定性 {len(report['uncertainties'])} 组；矛盾 {len(report['contradictions'])} 项。</p><p>本次流水线总耗时 {_duration(perf.get('total_duration_seconds'))}；本次流水线预处理 {_duration((perf.get('preprocessing_sla') or {}).get('actual_seconds'))}；全量六路预处理验收 {(_duration((perf.get('full_run_preprocessing') or {}).get('seconds')) if (perf.get('full_run_preprocessing') or {}).get('seconds') is not None else '无独立记录')}；Token {perf.get('total_input_tokens') or 0:,} 输入 + {perf.get('total_output_tokens') or 0:,} 输出 = {perf.get('total_tokens') or 0:,}。</p></section>
<section><h2>人工复核</h2><p>状态：待复核　复核人：__________　复核时间：__________</p><p>备注：____________________________________________________________</p></section></main></body></html>"""


def _register_fonts() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyhbd.ttc")),
        (Path("C:/Windows/Fonts/simsun.ttc"), Path("C:/Windows/Fonts/simhei.ttf")),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("VCLab", str(regular)))
            pdfmetrics.registerFont(TTFont("VCLabBold", str(bold)))
            return "VCLab", "VCLabBold"
    return "Helvetica", "Helvetica-Bold"


def _pdf(path: Path, report: dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    regular, bold = _register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("VCTitle", parent=styles["Title"], fontName=bold, fontSize=18, leading=24, textColor=colors.HexColor("#17343a"), alignment=TA_CENTER)
    h1 = ParagraphStyle("VCH1", parent=styles["Heading1"], fontName=bold, fontSize=12, leading=16, spaceBefore=7, spaceAfter=4, textColor=colors.HexColor("#126d69"))
    h2 = ParagraphStyle("VCH2", parent=h1, fontSize=9.2, leading=13, spaceBefore=4, spaceAfter=3)
    body = ParagraphStyle("VCBody", parent=styles["BodyText"], fontName=regular, fontSize=7.2, leading=10.3, textColor=colors.HexColor("#29464b"))
    small = ParagraphStyle("VCSmall", parent=body, fontSize=5.8, leading=7.8, textColor=colors.HexColor("#526d73"))
    tiny = ParagraphStyle("VCTiny", parent=small, fontSize=5.1, leading=6.7)
    path.parent.mkdir(parents=True, exist_ok=True)

    header_background = colors.HexColor("#e3f0f0")
    accent_background = colors.HexColor("#f5eadf")
    grid_color = colors.HexColor("#d0dddd")

    def p(value: Any, style=small) -> Paragraph:
        return Paragraph(html.escape(str(value if value not in (None, "") else "-")), style)

    def table_style(*, accent: bool = False, last_total: bool = False) -> TableStyle:
        commands = [
            ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTNAME", (0, 1), (-1, -1), regular),
            ("BACKGROUND", (0, 0), (-1, 0), accent_background if accent else header_background),
            ("GRID", (0, 0), (-1, -1), 0.25, grid_color),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 5.8),
            ("TOPPADDING", (0, 0), (-1, -1), 2.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ]
        if last_total:
            commands.extend(
                [
                    ("FONTNAME", (0, -1), (-1, -1), bold),
                    ("BACKGROUND", (0, -1), (-1, -1), accent_background),
                ]
            )
        return TableStyle(commands)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.HexColor("#688086"))
        canvas.drawString(12 * mm, 7 * mm, f"{report['template_id']} · {report['experiment_id']}")
        canvas.drawRightString(285 * mm, 7 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=12 * mm,
        title=f"实验室日报 {report['report_date']}",
        author="VisionCortex",
    )
    story = [
        Paragraph(f"实验室日报 · {report['report_date']}", title),
        Paragraph(f"{html.escape(report['experiment_id'])} · {report['template_id']}", small),
        Spacer(1, 2.5 * mm),
    ]
    overview = report["overview"]
    overview_data = [
        ["输入视角", "有界实验", "候选/接受/拒绝", "关键事件", "状态变化", "证据包检查", "流水线耗时", "总 Token"],
        [
            f"{overview['input_view_count']}（FP {overview['first_person_views']} / TP {overview['third_person_views']}）",
            overview["experiment_group_count"],
            f"{overview.get('candidate_event_count') or 0}/{overview.get('accepted_event_count') or 0}/{overview.get('rejected_event_count') or 0}",
            overview["key_event_count"],
            overview["physical_change_count"],
            f"{overview.get('evidence_package_eval_check_count') or 0} 项 / 失败 {overview.get('evidence_package_eval_failure_count') or 0}",
            _duration(report["performance"].get("total_duration_seconds")),
            f"{report['performance'].get('total_tokens') or 0:,}",
        ],
    ]
    overview_table = Table(overview_data, colWidths=[34 * mm] * 8)
    overview_table.setStyle(table_style())
    overview_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("FONTNAME", (0, 1), (-1, 1), bold), ("FONTSIZE", (0, 0), (-1, -1), 6.6)]))
    story.extend(
        [
            overview_table,
            Spacer(1, 2 * mm),
            Paragraph(
                "证据口径：只汇总已通过验收的证据；当前步骤和下一步骤属于证据支持的模型理解。模板固定、模型不得修改栏目，日报新增模型调用与 Token 均为 0。",
                body,
            ),
            Paragraph("全天实验矩阵", h1),
        ]
    )
    matrix_rows = [["实验", "相对时间", "类型", "时长(s)", "步骤", "关键事件", "状态变化", "参与视角"]]
    for group in report["experiment_timeline"]:
        matrix_rows.append(
            [
                p(group["experiment_name"]),
                Paragraph(f"{group['start_timecode']}<br/>{group['end_timecode']}", small),
                "连续" if group["continuity_type"] == "continuous" else "独立",
                f"{group['duration_seconds']:.3f}",
                len(group["steps"]),
                len(group["key_events"]),
                group["physical_change_count"],
                len(group["participating_views"]),
            ]
        )
    matrix = Table(matrix_rows, colWidths=[62*mm,34*mm,16*mm,21*mm,17*mm,23*mm,23*mm,22*mm], repeatRows=1)
    matrix.setStyle(table_style())
    matrix.setStyle(TableStyle([("ALIGN", (2, 1), (-1, -1), "CENTER")]))
    story.extend([matrix, Paragraph("跨视角对齐审计", h1)])
    alignment_rows = [["视角", "角色", "状态", "置信度", "CSV 匹配数/率", "CSV RMSE(ms)", "视觉校正(ms)", "视觉置信度"]]
    for item in report["alignment_summary"]["views"]:
        alignment_rows.append(
            [
                p(item.get("view_id"), tiny),
                item.get("role"),
                item.get("state"),
                f"{float(item.get('confidence') or 0):.4f}",
                f"{item.get('csv_match_count') or 0} / {float(item.get('csv_match_ratio') or 0):.4f}",
                "-" if item.get("csv_rmse_ms") is None else f"{float(item['csv_rmse_ms']):.3f}",
                f"{float(item.get('visual_correction_ms') or 0):.3f}",
                f"{float(item.get('visual_confidence') or 0):.3f}",
            ]
        )
    alignment_table = Table(alignment_rows, colWidths=[58*mm,28*mm,20*mm,22*mm,40*mm,33*mm,35*mm,32*mm], repeatRows=1)
    alignment_table.setStyle(table_style())
    alignment_table.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "CENTER")]))
    story.append(alignment_table)
    for index, group in enumerate(report["experiment_timeline"], 1):
        story.append(Paragraph(f"{index}. {html.escape(group['experiment_name'])}", h1))
        continuity = "连续实验" if group["continuity_type"] == "continuous" else "独立实验"
        usage = group.get("model_usage") or {}
        story.append(
            Paragraph(
                f"相对时间 {group['start_timecode']} - {group['end_timecode']}　{continuity}　{group['duration_seconds']:.3f} 秒　"
                f"模型 {html.escape(str(group.get('model_name') or '-'))} / {html.escape(str(group.get('model_status') or '-'))} / "
                f"置信度 {group.get('model_confidence') if group.get('model_confidence') is not None else '-'} / Token {usage.get('total_tokens') or 0}",
                small,
            )
        )
        story.append(Paragraph(html.escape(group.get("overall_summary") or "无模型摘要"), body))
        changes = "；".join(f"{item['change_type']}×{item['count']}" for item in group["physical_change_summary"][:8]) or "无"
        changed_objects = "、".join(f"{item['object']}×{item['count']}" for item in group["top_changed_objects"][:10]) or "无"
        story.append(
            Paragraph(
                f"物理变化 {group['physical_change_count']} 项：{html.escape(changes)}。高频变化对象：{html.escape(changed_objects)}。",
                small,
            )
        )
        step_rows = [["步骤", "时间", "当前步骤", "下一步骤", "对象"]]
        for step in group["steps"]:
            step_rows.append(
                [
                    str(step.get("step_index") or "-"),
                    Paragraph(f"{step['start_timecode']}<br/>{step['end_timecode']}", tiny),
                    p(step.get("current_step") or "未说明", small),
                    p(step.get("next_step") or "证据不足", small),
                    p("、".join(step.get("objects") or []) or "未明确", tiny),
                ]
            )
        step_table = Table(step_rows, colWidths=[11*mm,31*mm,80*mm,80*mm,66*mm], repeatRows=1)
        step_table.setStyle(table_style())
        story.extend([Spacer(1, 1*mm), step_table, Paragraph("关键事件证据索引", h2)])
        event_rows = [["事件", "峰值", "动作", "对象", "跨视角支持", "置信度", "当前/下一步骤", "对齐关键片段"]]
        for event in group["key_events"]:
            event_usage = event.get("model_usage") or {}
            event_rows.append(
                [
                    event["event_id"],
                    event["peak_timecode"],
                    event["action_label"],
                    p("、".join(event["objects"]) or "未明确", tiny),
                    p(f"{len(event['supporting_views'])} 路 / {'+'.join(event['supporting_roles'])}", tiny),
                    f"{float(event['confidence']):.3f}",
                    p(
                        f"当前：{event.get('current_step') or '未说明'}；下一步：{event.get('next_step') or '证据不足'}；Token {event_usage.get('total_tokens') or 0}",
                        tiny,
                    ),
                    p(event.get("aligned_key_clip") or "无", tiny),
                ]
            )
        event_table = Table(event_rows, colWidths=[19*mm,24*mm,28*mm,35*mm,28*mm,17*mm,70*mm,47*mm], repeatRows=1)
        event_table.setStyle(table_style(accent=True))
        story.extend([event_table, Spacer(1, 1.5*mm)])
    story.append(Paragraph("全局关键动作、质量、耗时与 Token", h1))
    action_rows = [["动作类型", "事件数"]] + [[item["action_label"], item["event_count"]] for item in report["action_summary"]]
    action_table = Table(action_rows, colWidths=[100*mm,35*mm], repeatRows=1)
    action_table.setStyle(table_style())
    action_table.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
    perf = report["performance"]
    full_run_preprocessing = perf.get("full_run_preprocessing") or {}
    story.extend([
        action_table,
        Spacer(1, 2 * mm),
        Paragraph(
            f"对齐 {report['alignment_summary']['aligned']}/{report['alignment_summary']['view_count']} 路 aligned，平均置信度 {report['alignment_summary']['mean_confidence']}；不确定性 {len(report['uncertainties'])} 组，跨视角矛盾 {len(report['contradictions'])} 项。完整不确定性和 107 项物理变化见日报 JSON。",
            body,
        ),
        Paragraph(
            f"耗时口径：本次流水线 {_duration(perf.get('total_duration_seconds'))}；本次流水线预处理 "
            f"{_duration((perf.get('preprocessing_sla') or {}).get('actual_seconds'))}（可能复用已验收 CV 账本）；"
            f"全量六路预处理验收 "
            f"{_duration(full_run_preprocessing.get('seconds')) if full_run_preprocessing.get('seconds') is not None else '无独立记录'}。",
            body,
        ),
        Paragraph("阶段耗时与阶段 Token", h2),
    ])
    token_by_stage = {
        "experiment_understanding": perf.get("tokens", {}).get("experiment_groups", {}),
        "mllm": perf.get("tokens", {}).get("key_materials", {}),
        "daily_report": perf.get("tokens", {}).get("daily_report", {}),
    }
    stage_rows = [["阶段", "耗时(s)", "输入 Token", "输出 Token", "总 Token", "调用"]]
    for stage in perf.get("stage_durations") or []:
        usage = token_by_stage.get(stage.get("stage"), {})
        stage_rows.append(
            [
                stage.get("stage"),
                f"{float(stage.get('duration_seconds') or 0):.6f}",
                usage.get("input_tokens") or 0,
                usage.get("output_tokens") or 0,
                usage.get("total_tokens") or 0,
                usage.get("call_count") or 0,
            ]
        )
    stage_rows.append(
        [
            "TOTAL",
            f"{float(perf.get('total_duration_seconds') or 0):.6f}",
            perf.get("total_input_tokens") or 0,
            perf.get("total_output_tokens") or 0,
            perf.get("total_tokens") or 0,
            sum(int(item.get("call_count") or 0) for item in token_by_stage.values()),
        ]
    )
    stage_table = Table(stage_rows, colWidths=[72*mm,38*mm,42*mm,42*mm,42*mm,30*mm], repeatRows=1)
    stage_table.setStyle(table_style(last_total=True))
    stage_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 1.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
            ]
        )
    )
    uncertainty_ids = "、".join(str(item.get("id")) for item in report["uncertainties"][:20]) or "无"
    story.extend(
        [
            stage_table,
            Spacer(1, 2 * mm),
            Paragraph(f"需人工关注的不确定性对象（前 20）：{html.escape(uncertainty_ids)}。完整清单见事实源 JSON。", small),
            Paragraph("人工复核", h1),
            Paragraph(
                "状态：待复核　　复核人：________________　　复核时间：________________　　结论：通过 / 需修订 / 驳回"
                "<br/>备注：______________________________________________________________________________________________________________",
                body,
            ),
        ]
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


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
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    html_path.write_text(_html(report), encoding="utf-8")
    evaluation = evaluate_daily_report(report, summary)
    write_json(eval_path, evaluation)
    if not review_path.is_file() or not isinstance(existing_review, dict):
        write_json(review_path, report["human_review"])
    if not evaluation["passed"]:
        raise RuntimeError("Daily report evaluation failed")
    pdf_path = layout.professional_pdfs / f"{stem}.pdf"
    if config.get("daily_report", {}).get("generate_pdf", True):
        _pdf(pdf_path, report)
    artifacts = {
        "report_date": report_date,
        "json": str(json_path.relative_to(layout.root).as_posix()),
        "markdown": str(markdown_path.relative_to(layout.root).as_posix()),
        "html": str(html_path.relative_to(layout.root).as_posix()),
        "pdf": str(pdf_path.relative_to(layout.root).as_posix()) if pdf_path.is_file() else None,
        "evaluation": str(eval_path.relative_to(layout.root).as_posix()),
        "human_review": str(review_path.relative_to(layout.root).as_posix()),
        "passed": True,
        "additional_model_tokens": 0,
        "generation_duration_seconds": round(time.perf_counter() - started, 6),
        "checksums": {
            path.name: _sha256(path)
            for path in (json_path, markdown_path, html_path, eval_path, review_path, pdf_path)
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
