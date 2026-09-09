from __future__ import annotations

import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .archive import write_json
from .run_queue import DurableRunQueue


def partial_result_available(root: Path) -> bool:
    """Require the terminal status and delivery receipt to agree."""
    def read(name):
        path = root / name
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}

    status = read("JSON-Config-Files/pipeline_status.json") or read("run_status.json")
    if status.get("stage") != "partial":
        return False
    quality = read("JSON-Config-Files/quality_acceptance.json")
    receipt = read("JSON-Config-Files/partial_delivery.json")
    report = root / "Partial-Results/Partial-Evidence-Report.html"
    if (quality.get("passed") is not False
            or receipt.get("formal_archive_promotion_allowed") is not False
            or receipt.get("evidence_classification") != "PARTIAL_EVIDENCE"
            or not report.is_file()
            or hashlib.sha256(report.read_bytes()).hexdigest() != receipt.get("report_sha256")):
        raise RuntimeError("Incomplete quality-attention delivery receipt")
    return True


def quality_gap_summary(quality: dict[str, Any]) -> list[dict[str, Any]]:
    """Explain failed gates without inventing evidence or diagnosing CV accuracy."""
    gaps = []
    for item in quality.get("segmentation_integrity", {}).get("incomplete_boundaries", []):
        gaps.append({"code": "experiment_boundary_incomplete", "group_id": item.get("group_id"),
                     "message": "实验起止尚未确认完整，已保留操作片段；需要继续核对切点前后的视频。",
                     "automatic_retry": False})
    for group in quality.get("segmentation_integrity", {}).get("groups", []):
        if group.get("passed") is not False:
            continue
        resolved = group.get("all_declared_key_events_resolved") is not False
        gaps.append({
            "code": "canonical_pair_support_missing" if resolved else "key_event_reference_missing",
            "group_id": group.get("group_id"),
            "key_event_count": group.get("key_event_count", 0),
            "jointly_supported_count": len(group.get("jointly_supported_key_event_ids") or []),
            "message": (
                f"{group.get('group_id')} 缺少第一与第三人称共同支持的动作证据"
                + (f"；已保留 {group['key_event_count']} 个动作。" if "key_event_count" in group else "。")
                if resolved else f"{group.get('group_id')}：部分动作引用不完整。"
            ),
            "automatic_retry": False,
            "retry_condition": "new_input_or_verified_pipeline_revision",
        })
    labels = {"step_action_consistency": "步骤文字与动作证据不一致", "key_event_recall": "事件质量未达到配置要求"}
    for key, label in labels.items():
        if quality.get(key, {}).get("passed") is False:
            gaps.append({"code": key, "message": label, "automatic_retry": False})
    if quality.get("passed") is False and not gaps:
        gaps.append({"code": "quality_acceptance", "message": "结果完整性或质量检查未通过，详细检查项已保留。", "automatic_retry": False})
    return gaps


def prioritize_retained_candidates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prioritize browsing by recorded support, without accepting or deleting events."""
    reviewed = []
    for item in items:
        reasons = []
        start, end = item.get("start_ms"), item.get("end_ms")
        valid = all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (start, end)) and end > start
        duration = end - start if valid else None
        if duration is None:
            reasons.append("missing_time_bounds")
        elif duration < 1000:
            reasons.append("short_candidate")
        if not {"first_person", "third_person"}.issubset(item.get("source_roles", [])):
            reasons.append("missing_cross_role_sources")
        if not item.get("frame_url") or not item.get("clip_url"):
            reasons.append("incomplete_preview_media")
        if item.get("disposition") != "machine_quarantined_semantic_unavailable":
            reasons.append("semantic_review_not_accepted")
        reviewed.append({**item, "preview_review": {
            "duration_ms": duration, "priority": not reasons,
            "reason_codes": reasons, "related_event_id": None,
        }})

    representatives = []
    for item in sorted(reviewed, key=lambda value: (
        not value["preview_review"]["priority"],
        -len(set(value.get("source_views", []))),
        -(value["preview_review"]["duration_ms"] or 0), str(value.get("event_id")),
    )):
        review = item["preview_review"]
        objects = set(item.get("cv_objects", [])) - {"hand", "gloved_hand"}
        if objects and item.get("group_id") and review["duration_ms"]:
            for other in representatives:
                if (other.get("group_id"), other.get("cv_action_type")) != (
                    item.get("group_id"), item.get("cv_action_type")
                ):
                    continue
                if set(other.get("cv_objects", [])) - {"hand", "gloved_hand"} != objects:
                    continue
                overlap = min(item["end_ms"], other["end_ms"]) - max(item["start_ms"], other["start_ms"])
                if overlap / min(review["duration_ms"], other["preview_review"]["duration_ms"]) >= .8:
                    review["reason_codes"].append("overlapping_similar_candidate")
                    review["related_event_id"] = other.get("event_id")
                    review["priority"] = False
                    break
        if review["priority"]:
            representatives.append(item)
    reviewed.sort(key=lambda item: (item.get("timestamp_ms") or 0, str(item.get("event_id"))))
    return reviewed, {
        "schema_version": "visioncortex-retained-candidate-preview/1",
        "purpose": "browsing_priority_only_not_action_confirmation",
        "minimum_duration_ms": 1000, "similar_interval_overlap_ratio": .8,
        "total_count": len(reviewed),
        "priority_count": sum(item["preview_review"]["priority"] for item in reviewed),
        "reason_counts": {reason: sum(reason in item["preview_review"]["reason_codes"] for item in reviewed)
                          for reason in sorted({code for item in reviewed for code in item["preview_review"]["reason_codes"]})},
        "deleted_count": 0, "formal_accuracy_claim_allowed": False,
    }


def registered_partial_root(config: dict[str, Any], record: dict[str, Any]) -> Path | None:
    """Resolve an explicitly registered local result without expanding NAS roots."""
    if record.get("origin") != "registered_local_partial":
        return None
    runtime = config.get("storage", {}).get("local_runtime_root")
    value = record.get("observability_root")
    if not runtime or not value:
        return None
    root = Path(value).resolve()
    runtime_root = Path(runtime).resolve()
    if root == runtime_root or not root.is_relative_to(runtime_root):
        return None
    try:
        manifest = root / "JSON-Config-Files/run_manifest.json"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != record.get("manifest_sha256"):
            return None
        partial = json.loads((root / "JSON-Config-Files/partial_delivery.json").read_text())
        if partial.get("evidence_classification") != "PARTIAL_EVIDENCE":
            return None
        if partial.get("formal_archive_promotion_allowed") is not False:
            return None
    except (OSError, ValueError, AttributeError):
        return None
    return root


def register_partial_result(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Register a stopped local result for Web preview; never enqueue or copy it."""
    root = root.resolve()
    runtime = Path(config["storage"]["local_runtime_root"]).resolve()
    if root == runtime or not root.is_relative_to(runtime):
        raise ValueError("Local result must remain inside the configured local runtime root")
    manifest_bytes = (root / "JSON-Config-Files/run_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    status_path = root / "JSON-Config-Files/pipeline_status.json"
    if not status_path.is_file():
        status_path = root / "run_status.json"
    status = json.loads(status_path.read_text())
    if status.get("stage") not in {"failed", "interrupted"}:
        raise ValueError("Only stopped partial results may be registered")
    run_id = "local-partial-" + hashlib.sha256(str(root).encode()).hexdigest()[:20]
    record = {
        "origin": "registered_local_partial",
        "state": status["stage"],
        "experiment_id": manifest["experiment_id"],
        "progress": status.get("progress", 1.0),
        "updated_at": status.get("updated_at"),
        "message": "本地运行阶段产出已接入，尚未正式归档",
        "observability_root": str(root),
        "nas_staging": str(root),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "read_only": True,
        "retry_available": False,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    if registered_partial_root(config, record) is None:
        raise ValueError("Local result has no valid partial-delivery receipt")
    store = DurableRunQueue(runtime / "state/web_run_queue.sqlite3")
    if store.get_job(run_id) is not None:
        raise ValueError("Registration would overwrite an executable job")
    store.save_run(run_id, record)
    return {"run_id": run_id, "experiment_id": manifest["experiment_id"],
            "state": record["state"], "enqueued": False, "media_copied_bytes": 0,
            "preview_route": f"#/stage/{run_id}/experiments"}


def component_results(root: Path) -> list[dict[str, Any]]:
    """Describe independent outputs using control records, never decode media."""
    def read(name):
        path = root / "JSON-Config-Files" / name
        try:
            if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
                return {}
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    speech = read("speech.json")
    recording = read("speech_understanding.json")
    try:
        speech_hash = hashlib.sha256((root / "JSON-Config-Files/speech.json").read_bytes()).hexdigest() if speech else None
    except OSError:
        speech_hash = None
    if recording and (not speech or recording.get("index_sha256") != speech_hash):
        recording = {}
    chunks = [chunk for source in speech.get("sources", []) for chunk in source.get("chunks", [])]
    lines = sum(chunk.get("segment_count", 0) for chunk in chunks)
    groups = read("experiment_group_understanding.json").get("groups", [])
    understandings = [group.get("model_understanding") or {} for group in groups]
    understood = sum(item.get("status") == "completed" for item in understandings)
    quality = read("quality_acceptance.json")
    reports = read("daily_report_manifest.json")
    reports_ready = reports.get("passed") is True and quality.get("passed") is True and all(
        isinstance(reports.get(key), str) and (root / reports[key]).resolve().is_relative_to(root.resolve())
        and (root / reports[key]).is_file() for key in ("json", "html"))
    speech_state = speech.get("status", "not_available")
    if speech_state == "completed" and not chunks:
        speech_state = "not_available"
    understanding_state = (
        "completed" if understandings and understood == len(understandings)
        else "partial" if understood else "failed" if understandings
        else recording.get("status", "not_available")
    )
    return [
        {"key": "speech", "label": "录音转写", "state": speech_state,
         "detail": f"{lines} 条文字 · {len(chunks)} 段录音" if chunks else "未产出可用录音", "tab": "speech"},
        {"key": "understanding", "label": "模型理解", "state": understanding_state,
         "detail": f"{understood} 个片段的步骤理解" if groups else "录音与抽样画面的关联说明" if recording.get("parts") else "尚无可用理解", "tab": "experiments" if groups else "speech"},
        {"key": "visual", "label": "视觉证据", "state": "completed" if quality.get("passed") is True else "insufficient" if quality.get("passed") is False else "pending",
         "detail": "通过自动质量检查，准确率以复核为准" if quality.get("passed") is True else "未通过完整质量检查" if quality.get("passed") is False else "等待质量检查", "tab": "materials"},
        {"key": "reports", "label": "正式报告", "state": "completed" if reports_ready else "not_generated",
         "detail": "已生成" if reports_ready else "通过质量检查后生成", "tab": "reports"},
    ]


def write_partial_delivery(root: Path, metrics: dict[str, Any], *, analysis_finished: bool = False) -> dict[str, Any]:
    """Describe retained stage outputs without issuing a release/quality receipt.

    Bounded control files and step-linked derived stills are read and hashed.
    Original media, detection ledgers and derived video bodies are not reread.
    """
    from .run_insights import with_operation_refreshes
    metrics = with_operation_refreshes(root, metrics)
    json_root = root / "JSON-Config-Files"

    def read(relative: str) -> dict[str, Any]:
        path = root / relative
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}

    pending = []
    materialization = read("JSON-Config-Files/key_material_materialization_runtime.json")
    pending_materials = materialization.get("incomplete_artifacts") or []
    for stage in ("experiment_group", "key_material"):
        failure = read(f"JSON-Config-Files/{stage}_semantic_failures.json")
        pending.extend(
            {"stage": stage, "subject_id": item["subject_id"], "status": item["status"]}
            for item in failure.get("incomplete", [])
        )
    status = read("JSON-Config-Files/pipeline_status.json") or read("run_status.json")
    volume = read("JSON-Config-Files/input_volume_report.json")
    quarantine = read("Key-Materials/Machine-Quarantine/Machine-Quarantine-Index.json")
    review = read("Key-Materials/Review-Candidates/Candidate-Index.json")
    retained_ids = {
        str(item["event_id"])
        for index in (quarantine, review)
        for item in index.get("candidates", []) if item.get("event_id")
    }
    quality = read("JSON-Config-Files/quality_acceptance.json")
    groups = read("JSON-Config-Files/experiment_group_understanding.json").get("groups", [])
    from .speech_refresh import apply
    groups = apply(root, groups)
    from .operation_review import apply as apply_operations
    groups = apply_operations(root, groups)
    from .activity_review import apply as apply_activity, counts as activity_counts, assessment
    groups = apply_activity(root, groups)
    from .result_review import inspect as inspect_result
    try:
        result_check = inspect_result(root, save=True)
    except (OSError, ValueError, KeyError, TypeError):
        result_check = {"available":False}
    controls = [
        "JSON-Config-Files/result_check.json",
        "JSON-Config-Files/operation_review.json",
        "JSON-Config-Files/input_manifest.yaml",
        "JSON-Config-Files/run_manifest.json",
        "JSON-Config-Files/Input-Manifests/input_seal.json",
        "JSON-Config-Files/cache_identity.json",
        "JSON-Config-Files/input_volume_report.json",
        "JSON-Config-Files/experiment_group_understanding.json",
        "JSON-Config-Files/key_material_model_understanding.json",
        "JSON-Config-Files/experiment_boundary_review.json",
        "JSON-Config-Files/video_probe.json",
        "JSON-Config-Files/time_alignment.json",
        "JSON-Config-Files/quality_acceptance.json",
        "JSON-Config-Files/evidence_package_eval.json",
        "JSON-Config-Files/run_metrics.json",
        "JSON-Config-Files/speech.json",
        "JSON-Config-Files/speech_understanding.json",
        "JSON-Config-Files/speech_group_understanding.json",
        "JSON-Config-Files/capture_quality.json",
        "JSON-Config-Files/speech_search.json",
        "JSON-Config-Files/experiment_group_semantic_failures.json",
        "JSON-Config-Files/key_material_semantic_failures.json",
        "JSON-Config-Files/key_material_materialization_runtime.json",
        "JSON-Config-Files/semantic_key_material_curation.json",
        "Key-Materials/Machine-Quarantine/Machine-Quarantine-Index.json",
        "Key-Materials/Review-Candidates/Candidate-Index.json",
    ]
    receipts = sorted((json_root / "Stage-Receipts").glob("*.json"))
    receipts.extend(sorted((json_root / "Stage-Refreshes").glob("*.json")))
    receipts.extend(sorted((json_root / "Activity-Reviews").glob("*.json")))
    controls.extend(path.relative_to(root).as_posix() for path in receipts)
    artifacts = []
    for relative in controls:
        path = root / relative
        if path.is_file():
            data = path.read_bytes()
            artifacts.append({"path": relative, "size_bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()})
    report = {
        "schema_version": "visioncortex-partial-delivery/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_classification": "PARTIAL_EVIDENCE",
        "formal_archive_promotion_allowed": False,
        "status": "quality_attention" if quality.get("passed") is False else (
            "awaiting_semantic_recovery" if pending else "incomplete"
        ),
        "failed_stage": status.get("failed_stage") or status.get("stage"),
        "completed_stages": status.get("completed_stages", []),
        "pending_semantic_results": pending,
        "pending_material_outputs": pending_materials,
        "material_retry_event_ids": materialization.get("retry_event_ids") or [],
        "quarantined_event_count": len(retained_ids),
        "input_volume": volume,
        "run_metrics": metrics,
        "components": component_results(root),
        "quality_gaps": quality_gap_summary(quality),
        "analysis_finished": analysis_finished or status.get("stage") == "partial",
        "artifact_references": artifacts,
        "original_media_read_bytes": 0,
        "report": "Partial-Results/Partial-Evidence-Report.html",
        "retry_policy": {
            "original_input_retained": True,
            "verified_cache_reusable": True,
            "failed_semantic_results_reusable": False,
            "revalidate_inputs_models_config_and_code": True,
            "manual_evidence_approval_required": False,
        },
        "missing_gates": ["complete_chain_quality_acceptance", "atomic_archive_publication"],
    }
    explanation = (
        "本次结果未通过自动质量检查，具体缺口记录在 quality_acceptance.json。"
        "已完成的本地分析、片段与阶段素材均已保存，无需人工批准候选。"
        if quality.get("passed") is False else
        "智能理解尚有待补全结果。已完成的本地分析、片段与隔离素材仍然保留；"
        "请根据任务记录中的具体原因处理后复跑。"
        if pending else "本次分析尚未通过全部验收，已完成的阶段产出仍然保留。"
    )
    # A portable stage export contains the saved understanding and its limits,
    # without issuing an accepted daily report or rereading media bodies.
    export = {
        "schema_version": "visioncortex-partial-analysis/1",
        "experiment_id": read("JSON-Config-Files/run_manifest.json").get("experiment_id") or root.parent.name,
        "evidence_classification": "PARTIAL_EVIDENCE",
        "formal_archive_promotion_allowed": False,
        "experiment_groups": groups,
        "activity_counts": activity_counts(groups),
        "result_review": result_check,
        "recording_understanding": read("JSON-Config-Files/speech_understanding.json"),
        "capture_quality": read("JSON-Config-Files/capture_quality.json"),
        "quality_acceptance": quality,
        "quality_gaps": report["quality_gaps"],
        "pending_semantic_results": pending,
        "pending_material_outputs": pending_materials,
        "retained_candidate_ids": sorted(retained_ids),
        "run_metrics": metrics,
        "artifact_references": [dict(item) for item in artifacts],
        "traceability": {
            "step_event_reference": "experiment_groups[].model_understanding.steps[].supporting_event_ids",
            "event_source": "JSON-Config-Files/key_material_model_understanding.json",
            "source_manifest": "JSON-Config-Files/input_manifest.yaml",
            "source_video_bodies_revalidated": False,
        },
    }
    from .partial_reports import retained_report_visuals
    export["representative_visuals"] = retained_report_visuals(
        root, groups, read("JSON-Config-Files/key_material_model_understanding.json").get("events", []))
    report["presentation_derived_frame_count"] = len(export["representative_visuals"])
    export_path = root / "Partial-Results/Analysis-Result.json"
    write_json(export_path, export)
    report["export"] = export_path.relative_to(root).as_posix()
    artifacts.append({"path": report["export"], "size_bytes": export_path.stat().st_size,
                      "sha256": hashlib.sha256(export_path.read_bytes()).hexdigest()})
    rows = "".join(
        f"<tr><td>{html.escape(item['path'])}</td><td>{item['size_bytes']}</td>"
        f"<td><code>{item['sha256']}</code></td></tr>" for item in artifacts
    )
    # Summarize saved control records only; never turn partial evidence into a
    # formal daily report or change the original quality decision.
    def text(value: Any) -> str:
        return html.escape(str(value if value is not None else "未记录"))

    def seconds(value: Any) -> str:
        return f"{value:.1f} 秒" if isinstance(value, (int, float)) else "未记录"

    stage_labels = {
        "speech": "录音转写",
        "preflight": "启动检查", "alignment": "多视角对齐", "motion_probe": "活动筛选",
        "candidate_coarse": "粗扫", "candidate_fine": "精扫", "candidate_audit": "候选检查",
        "experiment_understanding": "片段理解", "experiment_clips": "生成实验片段",
        "key_materials": "生成关键素材", "mllm": "云端动作理解",
        "material_refinement": "素材与步骤复核", "package": "最终质量检查",
        "daily_report": "生成日报",
    }
    stage_rows = "".join(
        f"<tr><td>{text(stage_labels.get(item.get('stage'), item.get('stage')))}</td>"
        f"<td>{seconds(item.get('duration_seconds'))}</td>"
        f"<td>{'完成' if item.get('status') == 'completed' else '未完成'}</td></tr>"
        for item in metrics.get("stage_durations", [])
    )
    timeline = "".join(
        f"<li><strong>{text(assessment(group)['label'])} {index + 1}</strong> · "
        f"{seconds(group.get('global_start_ms') / 1000 if group.get('global_start_ms') is not None else None)}"
        f" → {seconds(group.get('global_end_ms') / 1000 if group.get('global_end_ms') is not None else None)}"
        f"<p>{len(group.get('model_understanding', {}).get('steps', []))} 个模型步骤；"
        f"{len(group.get('model_understanding', {}).get('uncertainties', []))} 项证据限制。</p>"
        f"<details><summary>原始模型说明与证据限制</summary>"
        f"<p>{text(group.get('model_understanding', {}).get('overall_summary'))}</p>"
        + "".join(f"<p>{text(item)}</p>" for item in group.get("model_understanding", {}).get("uncertainties", []))
        + "</details><details><summary>查看已保存步骤及证据编号</summary><ol>"
        + "".join(
            f"<li><strong>{text(step.get('operation_title') or '步骤说明')}</strong>"
            f"<p>{text(step.get('current_step'))}</p>"
            f"<p>证据编号：{text('、'.join(str(value) for value in step.get('supporting_event_ids', [])))}</p></li>"
            for step in group.get("model_understanding", {}).get("steps", [])
        )
        + "</ol></details></li>"
        for index, group in enumerate(groups)
    )
    gaps = [item["message"] for item in report["quality_gaps"]]
    calls = [item for item in metrics.get("mllm_calls", []) if not item.get("cache_reused")]
    unknown = sum(
        int(item.get("usage", {}).get("unknown_attempt_count") or (
            item.get("attempts", 0) if item.get("usage", {}).get("total_tokens") is None else 0
        )) for item in calls
    )
    total_tokens = metrics.get("tokens", {}).get("run_total", {}).get("total_tokens")
    latest_check = ""
    if result_check.get("available"):
        latest_check = (
            "<h2>最新结果检查</h2>"
            f"<p>结果版本：{text(result_check['revision'][:8])}；检查时间：{text(result_check['checked_at'])}。</p>"
            f"<p>步骤引用与文字检查：{'通过' if result_check['step_consistency_passed'] else '仍需核对'}；"
            f"另有 {len(result_check['findings'])} 项完整性提示。无记录可能是等待、遮挡或漏识别，需对照视频核验。</p>"
            "<p>这项检查不替代完整实验验收，不修改原质量决定，也不能证明全部操作均已识别。</p>"
        )
        for gap in result_check.get("gap_observations", []):
            latest_check += (
                f"<h3>补充观察 · {text(gap['group_id'])} · {seconds(gap['start_ms']/1000)} → {seconds(gap['end_ms']/1000)}</h3>"
                f"<p>{text(gap['sample_count'])} 张派生视频采样画面；尚未纳入已审核操作。</p>"
                + "".join(f"<p><strong>{text(item['title'])}</strong>：{text(item['description'])}</p>"
                          for item in gap.get("observations", []))
                + f"<p>{text(gap.get('limitation') or '采样有限，完整操作仍需视频核验。')}</p>"
            )
    readable_record = (
        "<h2>已完成成果与待补全环节</h2><ul>"
        + "".join(f"<li>{text(item['label'])}：{text({'completed': '已完成', 'partial': '部分完成', 'insufficient': '证据不足', 'not_generated': '未生成', 'not_available': '暂无产出', 'failed': '未完成', 'running': '处理中', 'pending': '待检查', 'disabled': '未启用'}.get(item['state'], item['state']))} · {text(item['detail'])}</li>" for item in report["components"])
        + "</ul>"
        +
        "<h2>本次运行记录</h2>"
        f"<p>分析开始（UTC）：{text(metrics.get('run_started_at'))}；"
        f"结束（UTC）：{text(metrics.get('run_ended_at'))}。</p>"
        f"<p>本轮分析耗时：{seconds(metrics.get('total_duration_seconds'))}；"
        f"已知 Token 用量：{text(total_tokens)}；另有 {unknown} 次请求尝试用量未知。</p>"
        "<p>未知用量不计为零；上次运行的缓存响应不计入本轮实际调用。</p>"
        "<h2>完整报告未生成的原因</h2>"
        + ("<ul>" + "".join(f"<li>{text(gap)}</li>" for gap in gaps) + "</ul>" if gaps else "<p>前序分析或质量检查尚未完成，详见任务记录。</p>")
        + "<p>最终质量检查通过后才生成完整日报与正式 PDF / JSON，并执行归档发布。"
        "本记录不能替代这些步骤。</p><h2>已保留片段时间线</h2>"
        + (f"<ol>{timeline}</ol>" if timeline else "<p>尚无已记录片段。</p>")
        + "<h2>录音理解与引用</h2><p>机器转写，声源未确认；录音提及不证明动作完成。</p>"
        + "".join(f"<p>{text(part.get('speech_interpretation', {}).get('summary'))}</p>"
                  f"<p>录音引用：{text('、'.join(part.get('speech_interpretation', {}).get('referenced_segment_ids', [])))}</p>"
                  for part in [*export["recording_understanding"].get("parts", []),
                               *(group.get("model_understanding", {}) for group in groups)])
        + "<h2>分阶段耗时</h2><table><thead><tr><th>环节</th><th>用时</th>"
        f"<th>状态</th></tr></thead><tbody>{stage_rows}</tbody></table>"
    )
    document = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>VisionCortex 阶段结果报告</title>'
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;'
        'font:16px/1.6 sans-serif;color:#172b35}td,th{padding:8px;border-bottom:'
        '1px solid #ccd6dc;text-align:left;overflow-wrap:anywhere}table{width:100%;'
        'table-layout:fixed}code{font-size:12px}aside{padding:16px;background:#fff3d6}</style>'
        '<h1>VisionCortex 阶段结果报告</h1><aside><strong>PARTIAL_EVIDENCE · '
        '尚未正式发布</strong><p>' + explanation + '</p></aside>'
        '<p>本报告记录已保存产出，不确认候选动作，不替代实验室日报、'
        '质量验收或正式归档回执。</p>'
        f"<p>待补全语义结果：{len(pending)}；待补全素材：{len(pending_materials)}；隔离事件：{report['quarantined_event_count']}。</p>"
        + ("<p>部分素材提取未完成，其他已保存画面和片段可继续查看。"
           "请修复任务记录中的素材读取或编码问题后复跑对应事件。</p>" if pending_materials else "")
        +
        '<p>复跑仅复用身份与完整性校验通过的缓存。代码、配置、模型或输入变化'
        '可能导致重新计算；未知 Token 用量不记作零。</p>'
        + latest_check + readable_record +
        '<h2>溯源引用</h2><p>以下哈希绑定控制文件；不代表已重新校验全部视频正文。</p>'
        '<table><thead><tr><th>文件</th><th>字节数</th><th>SHA-256</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></html>'
    )
    destination = root / report["report"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(destination)
    report["report_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    from .partial_reports import render_stage_reports
    from reportlab.platypus import LayoutError
    try:
        report["readable_reports"] = render_stage_reports(root, export, report)
    except (OSError, ValueError, RuntimeError, ImportError, LayoutError) as exc:
        # Preserve the existing stage export if optional presentation fails.
        report["readable_report_error"] = f"{type(exc).__name__}: {exc}"
    write_json(json_root / "partial_delivery.json", report)
    return report
