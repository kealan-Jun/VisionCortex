"""Refresh retained derived results without rerunning source processing."""
from __future__ import annotations

import time
from pathlib import Path

from . import speech_worker
from .partial_delivery import write_partial_delivery


def refresh(root: Path, scope: str, config: dict, *, target: str | None = None,
            revision: str | None = None) -> dict:
    from .daily_reports import generate_daily_report_from_archive
    from .speech_semantics import refresh_recording_understanding
    from .storage import read_current_release_pointer

    if read_current_release_pointer(root):
        raise ValueError("正式发布的实验不可原地刷新")
    json_root = root / "JSON-Config-Files"
    quality_path = json_root / "quality_acceptance.json"
    quality_hash = speech_worker.sha256(quality_path) if quality_path.is_file() else None
    started = time.perf_counter()
    calls = []
    source_decodes = 0
    dependencies = {}
    if scope == "capture_quality":
        from .capture_quality import inspect_archive
        capture = inspect_archive(root, {**config, "capture_quality": {**(config.get("capture_quality") or {}), "enabled": True}})
        source_decodes = sum(len(row["video_samples"])+len(row["audio_samples"]) for row in capture.get("records", []))
    elif scope == "search":
        from .speech_search import build
        dependencies = build(root)
    elif scope == "timeline":
        from .speech_timeline import build
        timeline = build(root)
        source_decodes = sum(not video["preview"].get("cache_reused") for video in timeline["videos"])
    elif scope == "understanding":
        from .speech_refresh import refresh_group, invalidate_reports
        groups_path = json_root / "experiment_group_understanding.json"
        is_group = target is not None and target.startswith("group:")
        current_path = groups_path if is_group else json_root / "speech_understanding.json"
        if revision is not None and speech_worker.sha256(current_path) != revision:
            raise ValueError("所选理解版本已改变，请刷新后重算")
        if is_group:
            result = refresh_group(root, config, target)
            dependencies = {key: result[key] for key in ("affected_group_ids", "updated_step_reference_count")}
        else:
            if groups_path.is_file() and speech_worker.read_json(groups_path).get("groups"):
                raise ValueError("请选择一个实验片段重算")
            result = refresh_recording_understanding(root, config, target)
            invalidate_reports(root)
        selected_parts = result["parts"] if is_group or target is None else [result["parts"][int(target.split(":")[1])]]
        calls = [{key: part.get(key) for key in
                  ("status", "cache_reused", "usage", "attempts", "speech_input_transport")}
                 for part in selected_parts]
        quality = speech_worker.read_json(quality_path) if quality_path.is_file() else {}
        if quality.get("passed") is True:
            generate_daily_report_from_archive(root, config)
            dependencies["reports"] = "regenerated"
        else:
            dependencies["reports"] = "partial_only_quality_gate_not_passed"
    elif scope == "reports":
        quality = speech_worker.read_json(quality_path) if quality_path.is_file() else {}
        if quality.get("passed") is True:
            generate_daily_report_from_archive(root, config)
    else:
        raise ValueError("未知刷新阶段")
    metrics_path = json_root / "run_metrics.json"
    metrics = speech_worker.read_json(metrics_path, 32 * 1024 * 1024) if metrics_path.is_file() else {}
    if (speech_worker.sha256(quality_path) if quality_path.is_file() else None) != quality_hash:
        raise ValueError("刷新阶段意外改变质量门")
    receipt = {"schema_version": "visioncortex-stage-refresh/1", "scope": scope,
               "target": target, "dependencies": dependencies,
               "status": "completed", "wall_seconds": round(time.perf_counter() - started, 3),
               "cv_invocations": 0, "asr_invocations": 0, "source_media_decodes": source_decodes,
               "model_calls": calls, "quality_gate_unchanged": True,
               "formal_archive_promotion_allowed": False}
    executed = [call for call in calls if not call.get("cache_reused")]
    receipt["model_invocations"] = sum(int(call.get("attempts") or 1) for call in executed)
    receipt["additional_usage"] = {
        key: sum(call["usage"][key] for call in executed)
        if all(isinstance((call.get("usage") or {}).get(key), int) for call in executed) else None
        for key in ("input_tokens", "output_tokens", "total_tokens")}
    history = json_root / "Stage-Refreshes" / f"{time.time_ns()}-{scope}.json"
    history.parent.mkdir(parents=True, exist_ok=True)
    speech_worker.atomic_json(history, receipt)
    speech_worker.atomic_json(json_root / f"{scope}_refresh.json", receipt)
    write_partial_delivery(root, metrics)
    return receipt
