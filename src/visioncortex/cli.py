from __future__ import annotations

import gc
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .actions import build_physical_change_log
from .action_state_machine import build_event_state_receipt
from .archive import (
    ArchiveLayout,
    _artifact_json,
    _rerender_curated_participant_annotations,
    curate_semantically_reviewed_key_materials,
    evidence_package_eval,
    materialize_key_materials,
    refresh_key_material_metadata,
    write_json,
    write_timestamp_tables,
)
from .config import load_config, load_manifest
from .collection_state import record_collection_state
from .collection_catalog import discover_collections
from .collection_curation import (
    build_curation_catalog,
    write_curation_catalog,
    write_probe_source_readiness,
)
from .content_probe import (
    adjudicate_full_timeline_sweep_with_cv,
    adjudicate_inconclusive_probe_with_cv,
    run_bounded_content_probe,
    run_full_timeline_content_sweep,
)
from .consensus_labels import build_consensus_box_labels
from .credentials import ensure_ark_api_key
from .daily_reports import generate_daily_report_from_archive
from .detection import validate_models
from .hardware_acceptance import run_hardware_acceptance, tune_hardware_acceptance
from .labpics_evaluation import (
    calibrate_labpics_threshold,
    evaluate_labpics_heldout,
)
from .indexing import build_archive_index
from .model_certification import (
    audit_production_model_certification,
    build_model_certification_readiness,
    build_model_quality_certification,
)
from .model_registry import install_registered_models, validate_model_registry
from .model_promotion import evaluate_yolo_candidate_promotion
from .local_acceptance import run_local_six_view_acceptance
from .local_model_acceptance import run_local_real_model_acceptance
from .partial_delivery import partial_result_available, register_partial_result
from .pipeline import (
    EvidencePipeline,
    _synchronize_final_event_state_receipts,
    _synchronize_segments_with_final_key_events,
    create_dry_run,
    normalize_final_group_action_language,
    validate_final_step_action_consistency,
)
from .public_model_assets import prepare_public_model_assets
from .public_datasets import prepare_public_dataset
from .provenance import write_run_provenance
from .replay_acceptance import (
    inspect_quality_ledger_inputs,
    replay_quality_decisions_from_ledgers,
)
from .recall_evaluation import evaluate_key_event_recall
from .reviewed_artifacts import load_dataset_scoped_json
from .schema_contracts import (
    validate_archive_contracts_or_raise,
    write_archive_contract_manifest,
)
from .schemas import RunManifest, RunSummary, VideoInfo, event_is_formal
from .storage import (
    DERIVED_ARCHIVE_DIRECTORIES,
    ORIGINAL_REFERENCE_NAMES,
    fixed_archive_staging_paths,
    initialize_nas_archive,
    prepare_from_nas_index,
    promote_fixed_archive,
    read_current_release_pointer,
    safe_archive_name,
    validate_formal_archive_release,
)
from .validation import (
    finalize_quality_acceptance_claims,
    validate_experiment_and_material_quality,
)
from .web_access import hash_web_password
from .yolo_evaluation import evaluate_files as evaluate_yolo_files
from .yolo_calibration import (
    audit_yolo_dataset_integrity,
    calibrate_yolo_confidence_thresholds,
    evaluate_yolo_with_calibrated_thresholds,
)
from .yolo_training import (
    build_mapped_public_yolo_union,
    build_public_yolo_training_view,
    build_yolo_training_dataset,
    evaluate_yolo_model_on_human_truth,
    train_yolo_model,
)
from .yolo_candidate_runtime import export_and_benchmark_yolo_candidate
from .yolo_world_calibration import calibrate_yolo_world_prompts


app = typer.Typer(no_args_is_help=True, help="多视角化学实验视频证据流水线")


@app.command("process-device-days")
def process_device_days(
    config: Annotated[Path, typer.Option("--config", exists=True)],
    stage: Annotated[str, typer.Option(help="all / retention / vision / stt / understanding / report")] = "all",
    retry_failed: Annotated[bool, typer.Option(help="显式重试失败阶段；可能再次调用模型")] = False,
    date: Annotated[str | None, typer.Option(help="只处理指定采集日，例如2026-09-07")] = None,
) -> None:
    """按设备/日期处理已关闭采集分片（按实际时长），采集端删除始终关闭。"""
    from .device_day import DeviceDayRunner
    from .ai_settings import apply_active
    settings = apply_active(load_config(config))
    if not (settings.get("device_day") or {}).get("enabled"):
        raise typer.BadParameter("所选配置尚未启用device_day")
    if stage not in {"all", "retention", "vision", "stt", "understanding", "report"}:
        raise typer.BadParameter("未知阶段")
    if date:
        datetime.strptime(date, "%Y-%m-%d")
    runner = DeviceDayRunner(settings)
    result = runner.run_once(stage=stage, retry=retry_failed, date=date)
    from .device_day_capture_files import reconcile_capture_files
    result["capture_files"] = reconcile_capture_files(runner, date=date)
    from .device_day_publication import reconcile_outputs, write_archive_guide
    result["publication"] = reconcile_outputs(runner)
    write_archive_guide(runner)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if any(r.get("status") == "failed" for r in result["results"]):
        raise typer.Exit(1)


@app.command("hash-web-password")
def hash_web_password_command() -> None:
    """Prompt securely and print a PBKDF2 hash for the LAN users file."""

    password = typer.prompt(
        "VisionCortex Web password",
        hide_input=True,
        confirmation_prompt=True,
    )
    try:
        typer.echo(hash_web_password(password))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _progress(stage: str, progress: float, message: str) -> None:
    typer.echo(f"[{progress:6.1%}] {stage}: {message}")


def _configure_index_full_timeline_scan(
    settings: dict[str, object],
    ingest: dict[str, object],
    *,
    requested_negative_audit: bool,
) -> dict[str, object]:
    """Select an auditable all-view fine scan for short indexed media.

    A 0.1 FPS motion sentinel is economical for multi-hour sources, but it is
    not a safe sole recall path for a seconds-long cap/contact operation.  The
    NAS index already records the experiment clock span.  On production
    profiles that opt in, any timeline at or below the configured ceiling is
    therefore scanned end-to-end at the normal fine detector FPS on every
    eligible view.  The explicit negative-audit mode remains separately named
    because content-exclusion receipts depend on that stronger intent.
    """

    performance = settings.setdefault("performance", {})
    assert isinstance(performance, dict)
    threshold_seconds = float(
        performance.get("auto_exhaustive_short_timeline_seconds", 0.0) or 0.0
    )
    recording_hours = ingest.get("recording_hours")
    duration_seconds = (
        float(recording_hours) * 3600.0 if recording_hours is not None else None
    )
    automatic = bool(
        performance.get("auto_exhaustive_short_timeline_enabled", False)
        and threshold_seconds > 0.0
        and duration_seconds is not None
        and 0.0 < duration_seconds <= threshold_seconds
    )
    effective = bool(requested_negative_audit or automatic)
    if effective:
        performance["fine_progressive_cross_view"] = False
        performance["preflight_prefer_clock_metadata"] = False
    if requested_negative_audit:
        performance["exhaustive_negative_audit_full_timeline"] = True
    elif automatic:
        performance["exhaustive_full_timeline_scan"] = True
        performance["exhaustive_full_timeline_reason"] = (
            "automatic_short_indexed_timeline_recall"
        )
    return {
        "schema_version": "visioncortex-index-full-timeline-scan-policy/1",
        "requested_negative_audit": bool(requested_negative_audit),
        "automatic_short_timeline_scan": automatic,
        "effective_exhaustive_full_timeline_scan": effective,
        "recording_duration_seconds": (
            round(duration_seconds, 6) if duration_seconds is not None else None
        ),
        "automatic_ceiling_seconds": threshold_seconds,
        "fine_detection_fps": float(performance.get("detection_fps", 0.0) or 0.0),
        "all_eligible_views": effective,
        "source_copy_bytes": 0,
        "reason": (
            "explicit_exhaustive_negative_audit"
            if requested_negative_audit
            else "automatic_short_indexed_timeline_recall"
            if automatic
            else "progressive_long_timeline_policy"
        ),
    }


def _audit_true_cold_doubao_execution(archive: Path) -> dict[str, object]:
    """Require real, unreused Doubao calls before formal promotion."""

    metrics_path = archive / "JSON-Config-Files" / "run_metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(
            f"Cold Doubao promotion audit is missing run metrics: {metrics_path}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    calls = list(metrics.get("mllm_calls") or [])
    if not calls:
        raise RuntimeError("Cold Doubao promotion audit found no executed MLLM calls")
    invalid_calls = [
        {
            "stage": call.get("stage"),
            "event_id": call.get("event_id"),
            "group_id": call.get("group_id"),
            "status": call.get("status"),
            "model": call.get("model"),
            "cache_reused": call.get("cache_reused"),
        }
        for call in calls
        if (
            call.get("status") != "completed"
            or call.get("cache_reused") is not False
            or not str(call.get("model") or "").startswith("doubao-")
            or int((call.get("usage") or {}).get("total_tokens") or 0) <= 0
            or (call.get("usage") or {}).get("server_reported") is not True
        )
    ]
    if invalid_calls:
        raise RuntimeError(
            "Formal promotion requires completed, server-reported, unreused "
            f"Doubao calls; invalid_call_count={len(invalid_calls)}"
        )
    tokens = metrics.get("tokens") or {}
    experiment_usage = tokens.get("experiment_groups") or {}
    material_usage = tokens.get("key_materials") or {}
    run_usage = tokens.get("run_total") or {}
    reported_call_count = int(experiment_usage.get("call_count") or 0) + int(
        material_usage.get("call_count") or 0
    )
    executed_call_count = int(experiment_usage.get("executed_call_count") or 0) + int(
        material_usage.get("executed_call_count") or 0
    )
    reused_call_count = int(experiment_usage.get("reused_call_count") or 0) + int(
        material_usage.get("reused_call_count") or 0
    )
    if (
        reported_call_count != len(calls)
        or executed_call_count != len(calls)
        or reused_call_count != 0
        or int(run_usage.get("total_tokens") or 0) <= 0
    ):
        raise RuntimeError(
            "Cold Doubao token/call accounting is inconsistent: "
            f"metrics_calls={len(calls)} reported_calls={reported_call_count} "
            f"executed_calls={executed_call_count} reused_calls={reused_call_count}"
        )
    return {
        "schema_version": "visioncortex-cold-doubao-promotion-audit/1",
        "status": "passed",
        "cache_policy": "cold_no_semantic_reuse",
        "model_provider": "volcengine_ark",
        "models": sorted({str(call["model"]) for call in calls}),
        "executed_call_count": len(calls),
        "reused_call_count": 0,
        "input_tokens": int(run_usage.get("input_tokens") or 0),
        "output_tokens": int(run_usage.get("output_tokens") or 0),
        "total_tokens": int(run_usage.get("total_tokens") or 0),
        "provider_cached_input_tokens": sum(
            int((call.get("usage") or {}).get("cached_input_tokens") or 0)
            for call in calls
        ),
        "provider_cache_note": (
            "Provider-side prompt caching is billed/reported by Ark and does "
            "not constitute VisionCortex semantic result reuse."
        ),
    }


def _relocate_unreferenced_key_material_event_directories(
    layout: ArchiveLayout, events: list
) -> list[dict[str, str]]:
    """Move stale event folders out of the user-facing key-material tree."""

    relocated: list[dict[str, str]] = []
    for attribute, media_root, media_kind in (
        ("key_frames", layout.key_frames, "Key-Frames"),
        ("key_clips", layout.key_clips, "Key-Clips"),
    ):
        referenced = {
            (layout.root / relative).parent.resolve(strict=True)
            for event in events
            for relative in dict(getattr(event, attribute)).values()
        }
        if not media_root.is_dir():
            continue
        candidates = sorted(
            path
            for experiment_root in media_root.iterdir()
            if experiment_root.is_dir()
            for action_root in experiment_root.iterdir()
            if action_root.is_dir()
            for path in action_root.iterdir()
            if path.is_dir()
        )
        for candidate in candidates:
            if candidate.is_symlink():
                raise RuntimeError(
                    f"Key-material event directory cannot be a symlink: {candidate}"
                )
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(media_root.resolve(strict=True)):
                raise RuntimeError(
                    f"Key-material event directory escaped media root: {candidate}"
                )
            if resolved in referenced:
                continue
            relative = candidate.relative_to(media_root)
            destination = (
                layout.work
                / "presentation-repair-obsolete-media"
                / media_kind
                / relative
            )
            if destination.exists():
                destination = destination.with_name(
                    f"{destination.name}-retry-{uuid.uuid4().hex[:8]}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(destination))
            relocated.append(
                {
                    "media_kind": media_kind,
                    "from": candidate.relative_to(layout.root).as_posix(),
                    "to": str(destination.resolve()),
                    "operation": "relocated_to_persistent_cache",
                }
            )
    return relocated


def _refresh_repaired_quality_acceptance(
    layout: ArchiveLayout,
    settings: dict,
    experiment_id: str,
    groups: list,
    key_events: list,
) -> dict:
    """Re-evaluate final repaired semantics and annotations before promotion."""

    validation = settings.get("validation", {})
    repository_root = Path(__file__).resolve().parents[2]
    baseline, baseline_selection = load_dataset_scoped_json(
        validation.get("acceptance_baseline"),
        experiment_id,
        repository_root=repository_root,
        artifact_label="验收基线",
    )
    report = validate_experiment_and_material_quality(
        groups,
        key_events,
        baseline,
        boundary_match_iou=float(validation.get("boundary_match_iou", 0.50)),
        max_start_error_seconds=float(validation.get("max_start_error_seconds", 8.0)),
        max_end_error_seconds=float(validation.get("max_end_error_seconds", 8.0)),
        minimum_cross_view_event_rate=float(
            validation.get("minimum_cross_view_event_rate", 0.25)
        ),
        require_participant_only_annotations=bool(
            validation.get("require_participant_only_annotations", False)
        ),
    )
    report["baseline_selection"] = dict(baseline_selection)
    ground_truth, ground_truth_selection = load_dataset_scoped_json(
        validation.get("key_event_ground_truth"),
        experiment_id,
        repository_root=repository_root,
        artifact_label="关键事件真值",
    )
    recall_report = evaluate_key_event_recall(key_events, ground_truth)
    recall_report["ground_truth_selection"] = ground_truth_selection
    recall_gate = {
        "evaluated": bool(recall_report.get("evaluated")),
        "passed": None,
        "temporal_iou_threshold": float(validation.get("key_event_recall_iou", 0.50)),
        "minimum_precision": float(validation.get("minimum_key_event_precision", 0.80)),
        "minimum_recall": float(validation.get("minimum_key_event_recall", 0.80)),
        "precision": None,
        "recall": None,
        "small_sample_warning": recall_report.get("small_sample_warning"),
    }
    if recall_gate["evaluated"]:
        selected_threshold = next(
            (
                item
                for item in recall_report.get("threshold_results") or []
                if abs(
                    float(item["temporal_iou_threshold"])
                    - recall_gate["temporal_iou_threshold"]
                )
                < 1e-9
            ),
            None,
        )
        if selected_threshold is None:
            raise ValueError(
                "Configured key-event recall IoU is absent from evaluation thresholds"
            )
        recall_gate["precision"] = selected_threshold.get("precision")
        recall_gate["recall"] = selected_threshold.get("recall")
        recall_gate["passed"] = bool(
            recall_gate["precision"] is not None
            and recall_gate["recall"] is not None
            and float(recall_gate["precision"]) >= recall_gate["minimum_precision"]
            and float(recall_gate["recall"]) >= recall_gate["minimum_recall"]
        )
        report["passed"] = bool(report.get("passed")) and bool(recall_gate["passed"])
        if not recall_gate["passed"]:
            report["status"] = "failed"
    report["key_event_recall"] = recall_gate
    step_consistency = validate_final_step_action_consistency(groups, key_events)
    report["step_action_consistency"] = step_consistency
    if not step_consistency["passed"]:
        report["passed"] = False
        report["status"] = "failed"
    finalize_quality_acceptance_claims(report, recall_gate, step_consistency)
    write_json(layout.json_config / "quality_acceptance.json", report)
    write_json(layout.json_config / "key_material_recall_eval.json", recall_report)
    return report


@app.command("curate-index")
def curate_index_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
) -> None:
    """Build the test-exclusion and full-timeline production queue."""

    settings = load_config(config)
    catalog = discover_collections(settings, limit=1000)
    payload = build_curation_catalog(catalog, settings)
    artifacts = write_curation_catalog(settings, payload)
    typer.echo(
        json.dumps(
            {"summary": payload["summary"], "artifacts": artifacts},
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("probe-index-content")
def probe_index_content_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    third_view_id: Annotated[
        str | None,
        typer.Option(
            "--third-view-id",
            help="Use one explicitly resolved third-person view for an audited retry.",
        ),
    ] = None,
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
) -> None:
    """Classify one ambiguous indexed recording within the hard probe budget."""

    settings = load_config(config)
    ark_preflight = ensure_ark_api_key(settings, required=True)
    catalog = build_curation_catalog(
        discover_collections(settings, limit=1000), settings
    )

    item = next(
        (
            candidate
            for candidate in catalog.get("experiments") or []
            if candidate.get("experiment_id") == experiment_id
        ),
        None,
    )
    if item is None:
        raise typer.BadParameter(
            f"Experiment is absent from the NAS index: {experiment_id}"
        )
    failed_real_prefix = bool(
        item.get("metadata_classification") == "real_experiment"
        and item.get("processing_state") == "failed"
    )
    if item.get("metadata_classification") != "ambiguous" and not failed_real_prefix:
        raise typer.BadParameter(
            "Content probe is permitted only for metadata-ambiguous recordings "
            "or a real-prefix item whose empty production attempt failed closed"
        )
    if item.get("collection_status") != "ready":
        raise typer.BadParameter(
            f"Content probe source is not ready: {item.get('collection_status')}; "
            f"blocking_issues={item.get('blocking_issues') or []}"
        )
    try:
        manifest, manifest_path, ingest = prepare_from_nas_index(
            settings,
            experiment_id,
            lambda message: typer.echo(f"[NAS] {message}"),
        )
    except FileNotFoundError as exc:
        blocker = write_probe_source_readiness(
            settings,
            experiment_id,
            status="blocked",
            error=str(exc),
        )
        refreshed = build_curation_catalog(
            discover_collections(settings, limit=1000), settings
        )
        write_curation_catalog(settings, refreshed)
        raise typer.BadParameter(
            f"Content probe source is missing; blocker={blocker}"
        ) from exc
    write_probe_source_readiness(settings, experiment_id, status="ready")
    if int(ingest.get("copied_source_bytes") or 0) != 0:
        raise RuntimeError("Content probe attempted to copy original media")
    timeline_seconds = float(
        (item.get("production_run") or {}).get("timeline_duration_seconds") or 0.0
    )
    result = run_bounded_content_probe(
        settings,
        manifest,
        display_name=str(item.get("display_name") or experiment_id),
        expected_timeline_seconds=timeline_seconds,
        third_view_id=third_view_id,
    )
    refreshed = build_curation_catalog(
        discover_collections(settings, limit=1000), settings
    )
    artifacts = write_curation_catalog(settings, refreshed)
    refreshed_item = next(
        candidate
        for candidate in refreshed["experiments"]
        if candidate["experiment_id"] == experiment_id
    )
    typer.echo(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "display_name": item.get("display_name"),
                "manifest": str(manifest_path),
                "source_copy_bytes": 0,
                "decoded_timeline_seconds": result["decoded_timeline_seconds"],
                "timeline_duration_seconds": result["timeline_duration_seconds"],
                "view_count": len(result["views"]),
                "verdict": result["verdict"],
                "model_proposed_verdict": result.get("model_proposed_verdict"),
                "confidence": result.get("confidence"),
                "ark_status": result.get("ark_status"),
                "ark_usage": result.get("ark_usage"),
                "receipt": result["receipt_path"],
                "artifact_root": result["artifact_root"],
                "curation_disposition": refreshed_item["disposition"],
                "production_eligible": refreshed_item["production_eligible"],
                "ark_preflight": ark_preflight,
                "catalog_artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("adjudicate-probe-with-cv")
def adjudicate_probe_with_cv_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    staging: Annotated[Path, typer.Option("--staging", exists=True, file_okay=False)],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
) -> None:
    """Resolve an inconclusive short probe only with independent Ark+CV consensus."""

    settings = load_config(config)
    result = adjudicate_inconclusive_probe_with_cv(settings, experiment_id, staging)
    refreshed = build_curation_catalog(
        discover_collections(settings, limit=1000), settings
    )
    artifacts = write_curation_catalog(settings, refreshed)
    item = next(
        candidate
        for candidate in refreshed["experiments"]
        if candidate["experiment_id"] == experiment_id
    )
    typer.echo(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "verdict": result["verdict"],
                "receipt": result["receipt_path"],
                "adjudication": result["adjudication"],
                "curation_disposition": item["disposition"],
                "catalog_artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("sweep-index-content")
def sweep_index_content_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
) -> None:
    """Review every source-media band of one long ambiguous recording."""

    settings = load_config(config)
    ark_preflight = ensure_ark_api_key(settings, required=True)
    catalog = build_curation_catalog(
        discover_collections(settings, limit=1000), settings
    )
    item = next(
        (
            candidate
            for candidate in catalog.get("experiments") or []
            if candidate.get("experiment_id") == experiment_id
        ),
        None,
    )
    if item is None:
        raise typer.BadParameter(
            f"Experiment is absent from the NAS index: {experiment_id}"
        )
    if item.get("metadata_classification") != "ambiguous":
        raise typer.BadParameter(
            "Full-timeline content sweep is permitted only for ambiguous recordings"
        )
    if item.get("collection_status") != "ready":
        raise typer.BadParameter(
            f"Content sweep source is not ready: {item.get('collection_status')}"
        )
    timeline_seconds = float(
        (item.get("production_run") or {}).get("timeline_duration_seconds") or 0.0
    )
    if timeline_seconds <= float(
        (settings.get("collection_curation") or {}).get(
            "content_probe_max_seconds", 300.0
        )
    ):
        raise typer.BadParameter(
            "Full-timeline sweep requires a timeline over five minutes"
        )
    try:
        manifest, manifest_path, ingest = prepare_from_nas_index(
            settings,
            experiment_id,
            lambda message: typer.echo(f"[NAS] {message}"),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        blocker = write_probe_source_readiness(
            settings, experiment_id, status="blocked", error=str(exc)
        )
        refreshed = build_curation_catalog(
            discover_collections(settings, limit=1000), settings
        )
        write_curation_catalog(settings, refreshed)
        raise typer.BadParameter(
            f"Content sweep source failed readiness; blocker={blocker}"
        ) from exc
    write_probe_source_readiness(settings, experiment_id, status="ready")
    if int(ingest.get("copied_source_bytes") or 0) != 0:
        raise RuntimeError("Content sweep attempted to copy original media")
    result = run_full_timeline_content_sweep(
        settings,
        manifest,
        display_name=str(item.get("display_name") or experiment_id),
        expected_timeline_seconds=timeline_seconds,
    )
    typer.echo(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "manifest": str(manifest_path),
                "source_copy_bytes": 0,
                "recording_timeline_duration_seconds": result[
                    "recording_timeline_duration_seconds"
                ],
                "media_timeline_duration_seconds": result[
                    "media_timeline_duration_seconds"
                ],
                "chunk_count": result["chunk_count"],
                "status": result["status"],
                "ark_usage": result["ark_usage"],
                "receipt": result["receipt_path"],
                "artifact_root": result["artifact_root"],
                "ark_preflight": ark_preflight,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("adjudicate-full-sweep-with-cv")
def adjudicate_full_sweep_with_cv_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    staging: Annotated[Path, typer.Option("--staging", exists=True, file_okay=False)],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
) -> None:
    """Resolve a long negative only with full semantic and exhaustive CV evidence."""

    settings = load_config(config)
    result = adjudicate_full_timeline_sweep_with_cv(settings, experiment_id, staging)
    refreshed = build_curation_catalog(
        discover_collections(settings, limit=1000), settings
    )
    artifacts = write_curation_catalog(settings, refreshed)
    item = next(
        candidate
        for candidate in refreshed["experiments"]
        if candidate["experiment_id"] == experiment_id
    )
    typer.echo(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "verdict": result["verdict"],
                "receipt": result["receipt_path"],
                "full_timeline_adjudication": result["full_timeline_adjudication"],
                "curation_disposition": item["disposition"],
                "catalog_artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _create_staging_only_run_root(
    settings: dict, archive_name: str, run_id: str
) -> Path:
    """Create one isolated run strictly below the configured NAS staging root."""

    configured = Path(str(settings["storage"]["local_staging_root"]))
    archive_root = Path(str(settings["storage"]["archive_root"]))
    if not configured.is_dir():
        raise typer.BadParameter(
            f"Configured staging root must already exist: {configured}"
        )
    expected = archive_root / ".VisionCortex-Run-Staging"
    try:
        staging_root = configured.resolve(strict=True)
        expected_root = expected.resolve(strict=True)
    except OSError as exc:
        raise typer.BadParameter(
            f"Unable to resolve existing NAS staging root: {exc}"
        ) from exc
    if staging_root != expected_root:
        raise typer.BadParameter(
            "Staging-only execution requires the existing archive-local "
            f".VisionCortex-Run-Staging root; configured={staging_root} expected={expected_root}"
        )
    safe_name = safe_archive_name(archive_name)
    campaign_root = staging_root / safe_name
    if campaign_root.exists():
        if not campaign_root.is_dir() or not campaign_root.resolve().is_relative_to(
            staging_root
        ):
            raise typer.BadParameter(
                f"Unsafe existing staging campaign path: {campaign_root}"
            )
    else:
        campaign_root.mkdir()
    target = campaign_root / run_id
    if target.exists():
        raise typer.BadParameter(f"Staging run already exists: {target}")
    target.mkdir()
    resolved_target = target.resolve(strict=True)
    if not resolved_target.is_relative_to(staging_root):
        raise typer.BadParameter(
            f"Staging run escaped configured root: {resolved_target}"
        )
    return resolved_target


def _normalized_cache_policy(
    mode: str, namespace: str | None, run_id: str
) -> tuple[str, str]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"cold", "reuse"}:
        raise typer.BadParameter("--cache-mode must be 'cold' or 'reuse'")
    selected = (namespace or (run_id if normalized_mode == "cold" else "")).strip()
    if not selected:
        raise typer.BadParameter(
            "--cache-namespace is required for a reuse run so it can identify the paired cold run"
        )
    if safe_archive_name(selected) != selected:
        raise typer.BadParameter(
            "--cache-namespace must already be a safe filesystem identifier"
        )
    return normalized_mode, selected


@app.command("generate-daily-report")
def generate_daily_report_command(
    archive: Annotated[
        Path, typer.Option("--archive", "-a", exists=True, file_okay=False)
    ],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Generate or refresh a zero-additional-token report from an accepted archive."""

    result = generate_daily_report_from_archive(archive, load_config(config))
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("register-partial-result")
def register_partial_result_command(
    root: Annotated[Path, typer.Option("--root", exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
) -> None:
    """Register retained local outputs for Web lookup without running analysis."""
    result = register_partial_result(root, load_config(config))
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("run")
def run_command(
    manifest: Annotated[
        Path, typer.Option("--manifest", "-m", exists=True, dir_okay=False)
    ],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    settings = load_config(config)
    # Match the indexed production entrypoints before opening media or starting
    # GPU work. CV-only/local runs must not require or read an Ark credential.
    if (
        settings.get("mllm", {}).get("enabled", False)
        and not settings["project"].get("preprocessing_acceptance_only", False)
    ):
        ensure_ark_api_key(settings, required=True)
    if output:
        settings["project"]["output_root"] = str(output)
    result = EvidencePipeline(settings, _progress).run(load_manifest(manifest))
    if partial_result_available(Path(result)):
        typer.echo("PARTIAL_EVIDENCE：分析结束，阶段成果已保存；详见 Partial-Results/Partial-Evidence-Report.html")
    typer.echo(str(result))


@app.command("dry-run")
def dry_run_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/dry-run"),
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
) -> None:
    result = create_dry_run(output, load_config(config))
    typer.echo(str(result))


@app.command("run-local-acceptance")
def run_local_acceptance_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "/srv/sentinel-data/VisionCortex3090Ti/Runtime/LocalAcceptance"
    ),
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-local.yaml"),
) -> None:
    """Build a six-view synthetic structural package without NAS access."""

    result = run_local_six_view_acceptance(output, load_config(config))
    typer.echo(str(result))


@app.command("benchmark-local-hardware")
def benchmark_local_hardware_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    media: Annotated[list[Path], typer.Option("--media", exists=True, dir_okay=False)],
    duration: Annotated[float, typer.Option("--duration-seconds")] = 60.0,
    workers_per_role: Annotated[int, typer.Option("--workers-per-role")] = 1,
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-local.yaml"),
) -> None:
    """Stress six local NVDEC lanes and both real TensorRT role engines."""

    result = run_hardware_acceptance(
        output,
        load_config(config),
        media,
        duration_seconds=duration,
        workers_per_role=workers_per_role,
    )
    typer.echo(str(result))


@app.command("accept-local-models")
def accept_local_models_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-local.yaml"),
) -> None:
    """Run every production CV model on one bounded local public example."""

    receipt = run_local_real_model_acceptance(dataset, output, load_config(config))
    typer.echo(str(receipt))


@app.command("tune-local-hardware")
def tune_local_hardware_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    media: Annotated[list[Path], typer.Option("--media", exists=True, dir_okay=False)],
    duration: Annotated[float, typer.Option("--duration-seconds")] = 20.0,
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-local.yaml"),
) -> None:
    """Select the fastest stable bounded six-lane TensorRT worker profile."""

    result = tune_hardware_acceptance(
        output,
        load_config(config),
        media,
        duration_seconds=duration,
    )
    typer.echo(str(result))


@app.command("evaluate-liquid-semantic")
def evaluate_liquid_semantic_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    sample_count: Annotated[int, typer.Option("--sample-count")] = 40,
    split: Annotated[str, typer.Option("--split")] = "Test",
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-local.yaml"),
) -> None:
    """Evaluate liquid masks on deterministic public human annotations."""

    result = evaluate_labpics_heldout(
        dataset,
        output,
        load_config(config),
        sample_count=sample_count,
        split=split,
    )
    typer.echo(str(result))


@app.command("calibrate-liquid-semantic")
def calibrate_liquid_semantic_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    thresholds: Annotated[list[float] | None, typer.Option("--threshold")] = None,
    sample_count: Annotated[int, typer.Option("--sample-count")] = 32,
    minimum_precision: Annotated[float, typer.Option("--minimum-precision")] = 0.90,
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-local.yaml"),
) -> None:
    """Calibrate on public Train labels without inspecting Test labels."""

    result = calibrate_labpics_threshold(
        dataset,
        output,
        load_config(config),
        thresholds=thresholds or [0.25, 0.35, 0.45, 0.50],
        sample_count=sample_count,
        minimum_precision=minimum_precision,
    )
    typer.echo(str(result))


@app.command("validate-models")
def validate_models_command(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
) -> None:
    typer.echo(
        json.dumps(validate_models(load_config(config)), ensure_ascii=False, indent=2)
    )


@app.command("validate-closed-set-models")
def validate_closed_set_models_command(
    registry: Annotated[
        Path, typer.Option("--registry", exists=True, dir_okay=False)
    ] = Path("configs/models/closed-set-yolo.json"),
    inspect_ontology: Annotated[
        bool, typer.Option("--inspect-ontology/--hash-only")
    ] = True,
) -> None:
    """Validate the retained trainable YOLO weights against the registry."""

    payload = validate_model_registry(registry, inspect_ontology=inspect_ontology)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("install-closed-set-models")
def install_closed_set_models_command(
    first_person: Annotated[
        Path, typer.Option("--first-person", exists=True, dir_okay=False)
    ],
    third_person: Annotated[
        Path, typer.Option("--third-person", exists=True, dir_okay=False)
    ],
    registry: Annotated[
        Path, typer.Option("--registry", exists=True, dir_okay=False)
    ] = Path("configs/models/closed-set-yolo.json"),
) -> None:
    """Install hash-verified source weights without overwriting mismatches."""

    payload = install_registered_models(
        registry,
        {
            "first_person": first_person,
            "third_person": third_person,
        },
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("prepare-public-models")
def prepare_public_models_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
) -> None:
    """Download and hash-check the profile's pinned public model assets."""

    payload = prepare_public_model_assets(load_config(config))
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("prepare-public-dataset")
def prepare_public_dataset_command(
    dataset_id: Annotated[str, typer.Option("--dataset-id")],
    destination: Annotated[Path, typer.Option("--destination")] = Path(
        "/srv/sentinel-data/VisionCortex3090Ti/Runtime/PublicDatasets"
    ),
    registry: Annotated[
        Path, typer.Option("--registry", exists=True, dir_okay=False)
    ] = Path("configs/public-data-sources.json"),
    extract: Annotated[bool, typer.Option("--extract/--no-extract")] = True,
) -> None:
    payload = prepare_public_dataset(registry, dataset_id, destination, extract=extract)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("build-consensus-labels")
def build_consensus_labels_command(
    source: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    minimum_models: Annotated[int, typer.Option("--minimum-models")] = 2,
    minimum_iou: Annotated[float, typer.Option("--minimum-iou")] = 0.5,
) -> None:
    observations = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(observations, list):
        raise typer.BadParameter("Consensus input must be a JSON list")
    payload = build_consensus_box_labels(
        observations,
        minimum_model_families=minimum_models,
        minimum_iou=minimum_iou,
    )
    write_json(output, payload)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("evaluate-yolo-boxes")
def evaluate_yolo_boxes_command(
    predictions: Annotated[
        Path, typer.Option("--predictions", exists=True, dir_okay=False)
    ],
    ground_truth: Annotated[
        Path, typer.Option("--ground-truth", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    confidence: Annotated[float, typer.Option("--confidence", min=0.0, max=1.0)] = 0.25,
) -> None:
    """Evaluate participant boxes against independent reviewed box truth."""

    payload = evaluate_yolo_files(
        predictions,
        ground_truth,
        output,
        confidence_threshold=confidence,
    )
    typer.echo(
        json.dumps(
            {
                "status": payload["status"],
                "dataset_id": payload.get("dataset_id"),
                "image_count": payload["image_count"],
                "ground_truth_instance_count": payload["ground_truth_instance_count"],
                "micro": payload["micro"],
                "macro": payload["macro"],
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("build-yolo-training-dataset")
def build_yolo_training_dataset_command(
    ground_truth: Annotated[
        Path, typer.Option("--ground-truth", exists=True, dir_okay=False)
    ],
    image_root: Annotated[
        Path, typer.Option("--image-root", exists=True, file_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    link_mode: Annotated[str, typer.Option("--link-mode")] = "symlink",
) -> None:
    """Build a zero-copy YOLO dataset from reviewed box ground truth."""

    payload = build_yolo_training_dataset(
        ground_truth, image_root, output, link_mode=link_mode
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("build-public-yolo-training-view")
def build_public_yolo_training_view_command(
    source: Annotated[Path, typer.Option("--source", exists=True, file_okay=False)],
    dataset_receipt: Annotated[
        Path, typer.Option("--dataset-receipt", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Validate public human labels and build a zero-copy YOLO training view."""

    payload = build_public_yolo_training_view(source, dataset_receipt, output)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("build-mapped-public-yolo-union")
def build_mapped_public_yolo_union_command(
    source: Annotated[
        list[str],
        typer.Option(
            "--source",
            help="Repeat DATASET_ID=/absolute/standardized-view for every mapped source",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    mapping: Annotated[
        Path, typer.Option("--mapping", exists=True, dir_okay=False)
    ] = Path("configs/models/public-yolo-ontology-map.json"),
    target_registry: Annotated[
        Path, typer.Option("--target-registry", exists=True, dir_okay=False)
    ] = Path("configs/models/closed-set-yolo.json"),
) -> None:
    """Map public human boxes into an exact 21-class zero-copy training union."""

    roots: dict[str, Path] = {}
    for value in source:
        dataset_id, separator, raw_path = value.partition("=")
        if not separator or not dataset_id.strip() or not raw_path.strip():
            raise typer.BadParameter("--source must use DATASET_ID=/absolute/path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise typer.BadParameter(f"Mapped public source does not exist: {path}")
        if dataset_id in roots:
            raise typer.BadParameter(f"Duplicate mapped public source: {dataset_id}")
        roots[dataset_id] = path
    payload = build_mapped_public_yolo_union(
        roots,
        mapping,
        target_registry,
        output,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("audit-yolo-dataset-integrity")
def audit_yolo_dataset_integrity_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    focus_classes: Annotated[
        str, typer.Option("--focus-classes", help="Comma-separated ontology classes")
    ] = "hand,pipette",
) -> None:
    """Hash all unique sources and fail closed on train/val/test leakage."""

    focus = [item.strip() for item in focus_classes.split(",") if item.strip()]
    payload = audit_yolo_dataset_integrity(dataset, output, focus_classes=focus)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("calibrate-yolo-confidence")
def calibrate_yolo_confidence_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    model: Annotated[Path, typer.Option("--model", exists=True, dir_okay=False)],
    audit_receipt: Annotated[
        Path, typer.Option("--audit-receipt", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    target_classes: Annotated[
        str, typer.Option("--target-classes", help="Comma-separated ontology classes")
    ] = "hand,pipette",
    thresholds: Annotated[
        str, typer.Option("--thresholds", help="Comma-separated validation sweep")
    ] = "0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9",
    minimum_precision: Annotated[
        float, typer.Option("--minimum-precision", min=0.0, max=1.0)
    ] = 0.85,
    minimum_recall: Annotated[
        float, typer.Option("--minimum-recall", min=0.0, max=1.0)
    ] = 0.80,
    image_size: Annotated[int, typer.Option("--image-size", min=64)] = 640,
    batch: Annotated[int, typer.Option("--batch", min=1)] = 8,
    device: Annotated[str, typer.Option("--device")] = "0",
) -> None:
    """Freeze hand/pipette operating points using validation truth only."""

    targets = [item.strip() for item in target_classes.split(",") if item.strip()]
    try:
        candidates = [
            float(item.strip()) for item in thresholds.split(",") if item.strip()
        ]
    except ValueError as exc:
        raise typer.BadParameter("--thresholds must contain numbers") from exc
    payload = calibrate_yolo_confidence_thresholds(
        dataset,
        model,
        audit_receipt,
        output,
        target_classes=targets,
        thresholds=candidates,
        minimum_precision=minimum_precision,
        minimum_recall=minimum_recall,
        image_size=image_size,
        batch=batch,
        device=device,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("evaluate-yolo-calibrated")
def evaluate_yolo_calibrated_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    model: Annotated[Path, typer.Option("--model", exists=True, dir_okay=False)],
    calibration_receipt: Annotated[
        Path, typer.Option("--calibration-receipt", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    image_size: Annotated[int, typer.Option("--image-size", min=64)] = 640,
    batch: Annotated[int, typer.Option("--batch", min=1)] = 8,
    device: Annotated[str, typer.Option("--device")] = "0",
    test_exposure_status: Annotated[
        str,
        typer.Option(
            "--test-exposure-status",
            help=(
                "Use first_use_independent only before any development has seen "
                "this test split; otherwise use repeat_comparative_benchmark."
            ),
        ),
    ] = "repeat_comparative_benchmark",
) -> None:
    """Apply a validation-frozen operating point with explicit test provenance."""

    payload = evaluate_yolo_with_calibrated_thresholds(
        dataset,
        model,
        calibration_receipt,
        output,
        image_size=image_size,
        batch=batch,
        device=device,
        test_exposure_status=test_exposure_status,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("calibrate-yolo-world-prompts")
def calibrate_yolo_world_prompts_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    model: Annotated[Path, typer.Option("--model", exists=True, dir_okay=False)],
    audit_receipt: Annotated[
        Path, typer.Option("--audit-receipt", exists=True, dir_okay=False)
    ],
    prompt_map: Annotated[
        Path, typer.Option("--prompt-map", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    target_classes: Annotated[
        str, typer.Option("--target-classes", help="Comma-separated ontology classes")
    ] = "hand,pipette",
    minimum_precision: Annotated[
        float, typer.Option("--minimum-precision", min=0.0, max=1.0)
    ] = 0.85,
    minimum_recall: Annotated[
        float, typer.Option("--minimum-recall", min=0.0, max=1.0)
    ] = 0.80,
    image_size: Annotated[int, typer.Option("--image-size", min=64)] = 640,
    batch: Annotated[int, typer.Option("--batch", min=1)] = 16,
    device: Annotated[str, typer.Option("--device")] = "0",
) -> None:
    """Measure YOLO-World prompts on validation truth before integration."""

    raw_prompt_map = json.loads(prompt_map.read_text(encoding="utf-8"))
    if not isinstance(raw_prompt_map, dict):
        raise typer.BadParameter("--prompt-map must contain one JSON object")
    targets = [item.strip() for item in target_classes.split(",") if item.strip()]
    payload = calibrate_yolo_world_prompts(
        dataset,
        model,
        audit_receipt,
        output,
        prompt_map={str(key): str(value) for key, value in raw_prompt_map.items()},
        target_classes=targets,
        minimum_precision=minimum_precision,
        minimum_recall=minimum_recall,
        image_size=image_size,
        batch=batch,
        device=device,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("measure-yolo-candidate-tensorrt")
def benchmark_yolo_candidate_tensorrt_command(
    model: Annotated[Path, typer.Option("--model", exists=True, dir_okay=False)],
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    audit_receipt: Annotated[
        Path, typer.Option("--audit-receipt", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    split: Annotated[str, typer.Option("--split")] = "val",
    image_size: Annotated[int, typer.Option("--image-size", min=64)] = 640,
    export_batch: Annotated[int, typer.Option("--export-batch", min=1)] = 4,
    benchmark_image_limit: Annotated[
        int, typer.Option("--benchmark-image-limit", min=1)
    ] = 256,
    workspace_gib: Annotated[float, typer.Option("--workspace-gib", min=0.1)] = 3.0,
    device: Annotated[str, typer.Option("--device")] = "0",
) -> None:
    """Export and benchmark an isolated candidate without changing production."""

    payload = export_and_benchmark_yolo_candidate(
        model,
        dataset,
        audit_receipt,
        output,
        split=split,
        image_size=image_size,
        export_batch=export_batch,
        benchmark_image_limit=benchmark_image_limit,
        workspace_gib=workspace_gib,
        device=device,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("train-yolo-model")
def train_yolo_model_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    base_model: Annotated[
        Path, typer.Option("--base-model", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    audit_receipt: Annotated[
        Path | None,
        typer.Option("--audit-receipt", exists=True, dir_okay=False),
    ] = None,
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 100,
    image_size: Annotated[int, typer.Option("--image-size", min=64)] = 1280,
    batch: Annotated[int, typer.Option("--batch", min=1)] = 8,
    device: Annotated[str, typer.Option("--device")] = "0",
    patience: Annotated[int, typer.Option("--patience", min=0)] = 20,
    max_hours: Annotated[float, typer.Option("--max-hours", min=0.01, max=24.0)] = 2.0,
    workers: Annotated[int, typer.Option("--workers", min=0)] = 8,
    optimizer: Annotated[str, typer.Option("--optimizer")] = "auto",
    learning_rate: Annotated[
        float, typer.Option("--learning-rate", min=1e-6, max=0.1)
    ] = 0.01,
    final_learning_rate_fraction: Annotated[
        float,
        typer.Option("--final-learning-rate-fraction", min=0.001, max=1.0),
    ] = 0.01,
    cosine_schedule: Annotated[
        bool, typer.Option("--cosine-schedule/--linear-schedule")
    ] = False,
    warmup_epochs: Annotated[
        float, typer.Option("--warmup-epochs", min=0.0, max=10.0)
    ] = 3.0,
    close_mosaic: Annotated[
        int,
        typer.Option(
            "--close-mosaic",
            min=0,
            help="Disable mosaic for the final N epochs; must not exceed --epochs.",
        ),
    ] = 10,
) -> None:
    """Train a real YOLO candidate; deployment remains certification-gated."""

    payload = train_yolo_model(
        dataset,
        base_model,
        output,
        epochs=epochs,
        image_size=image_size,
        batch=batch,
        device=device,
        patience=patience,
        max_hours=max_hours,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        final_learning_rate_fraction=final_learning_rate_fraction,
        cosine_schedule=cosine_schedule,
        warmup_epochs=warmup_epochs,
        close_mosaic=close_mosaic,
        dataset_integrity_audit=audit_receipt,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("evaluate-yolo-model-on-human-truth")
def evaluate_yolo_model_on_human_truth_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    model: Annotated[Path, typer.Option("--model", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    split: Annotated[str, typer.Option("--split")] = "test",
    image_size: Annotated[int, typer.Option("--image-size", min=64)] = 640,
    batch: Annotated[int, typer.Option("--batch", min=1)] = 32,
    device: Annotated[str, typer.Option("--device")] = "0",
    workers: Annotated[int, typer.Option("--workers", min=0)] = 8,
) -> None:
    """Evaluate a candidate on held-out human labels; never promote it."""

    payload = evaluate_yolo_model_on_human_truth(
        dataset,
        model,
        output,
        split=split,
        image_size=image_size,
        batch=batch,
        device=device,
        workers=workers,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("evaluate-yolo-candidate-promotion")
def evaluate_yolo_candidate_promotion_command(
    evaluation_receipt: Annotated[
        Path, typer.Option("--evaluation-receipt", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    gate: Annotated[Path, typer.Option("--gate", exists=True, dir_okay=False)] = Path(
        "configs/models/yolo-candidate-promotion-gate.json"
    ),
    internal_ab_receipt: Annotated[
        Path | None,
        typer.Option("--internal-ab-receipt", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Fail closed unless public metrics and a real internal six-view A/B pass."""

    payload = evaluate_yolo_candidate_promotion(
        evaluation_receipt,
        gate,
        output,
        internal_ab_receipt_path=internal_ab_receipt,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("model-certification-readiness")
def model_certification_readiness_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o"),
    ] = Path(
        "/srv/sentinel-data/VisionCortex3090Ti/Runtime/Model-Quality/production-model-certification-readiness.json"
    ),
) -> None:
    """Report exact truth/model deficits without accessing experiment archives."""

    payload = build_model_certification_readiness(
        load_config(config),
        repository_root=Path(__file__).resolve().parents[2],
    )
    write_json(output, payload)
    typer.echo(
        json.dumps(
            {
                "status": payload["status"],
                "ready_for_certification_run": payload["ready_for_certification_run"],
                "production_certified": payload["production_certified"],
                "event_truth": payload["event_truth"],
                "box_truth": payload["box_truth"],
                "deficits": payload["deficits"],
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("certify-model-quality")
def certify_model_quality_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
) -> None:
    """Build the fail-closed production model certification receipt."""

    settings = load_config(config)
    repository_root = Path(__file__).resolve().parents[2]
    payload = build_model_quality_certification(
        settings, repository_root=repository_root
    )
    configured = (settings.get("validation") or {}).get("model_certification") or {}
    output = Path(str(configured.get("path") or ""))
    if not str(output):
        raise typer.BadParameter("validation.model_certification.path is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    metrics = payload["metrics"]
    typer.echo(
        json.dumps(
            {
                "status": payload["status"],
                "passed": payload["passed"],
                "failures": payload["failures"],
                "event_ground_truth_count": metrics["event_ground_truth_count"],
                "event_precision": metrics["events"]["precision"],
                "event_recall": metrics["events"]["recall"],
                "box_precision": metrics["participant_boxes_iou_0_5"]["precision"],
                "box_recall": metrics["participant_boxes_iou_0_5"]["recall"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("validate-archive-quality")
def validate_archive_quality_command(
    archive: Annotated[
        Path, typer.Option("--archive", "-a", exists=True, file_okay=False)
    ],
    baseline: Annotated[
        Path | None, typer.Option("--baseline", "-b", exists=True, dir_okay=False)
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
    write: Annotated[bool, typer.Option("--write/--no-write")] = True,
) -> None:
    """Re-evaluate packaged quality without decoding video or calling a model."""

    settings = load_config(config)
    package_path = archive / "JSON-Config-Files" / "evidence_package.json"
    package = RunSummary.model_validate_json(
        package_path.read_text(encoding="utf-8-sig")
    )
    repository_root = Path(__file__).resolve().parents[2]
    if baseline is not None:
        baseline_payload = json.loads(baseline.read_text(encoding="utf-8-sig"))
        baseline_selection = {
            "configured": True,
            "applied": True,
            "reason": "explicit_cli_baseline",
            "current_experiment_id": package.experiment_id,
            "artifact_path": str(baseline.resolve()),
            "artifact_id": baseline_payload.get("baseline_id"),
        }
    else:
        baseline_payload, baseline_selection = load_dataset_scoped_json(
            (settings.get("validation") or {}).get("acceptance_baseline"),
            package.experiment_id,
            repository_root=repository_root,
            artifact_label="验收基线",
        )
    key_events = [
        event for event in package.events if event.key_frames or event.key_clips
    ]
    validation = settings.get("validation") or {}
    report = validate_experiment_and_material_quality(
        package.experiment_groups,
        key_events,
        baseline_payload,
        boundary_match_iou=float(validation.get("boundary_match_iou", 0.50)),
        max_start_error_seconds=float(validation.get("max_start_error_seconds", 8.0)),
        max_end_error_seconds=float(validation.get("max_end_error_seconds", 8.0)),
        minimum_cross_view_event_rate=float(
            validation.get("minimum_cross_view_event_rate", 0.25)
        ),
        require_participant_only_annotations=bool(
            validation.get("require_participant_only_annotations", False)
        ),
    )
    step_consistency = validate_final_step_action_consistency(
        package.experiment_groups,
        key_events,
    )
    report["step_action_consistency"] = step_consistency
    report["passed"] = bool(report.get("passed")) and bool(step_consistency["passed"])
    if report.get("baseline", {}).get("available"):
        report["status"] = "passed" if report["passed"] else "failed"
    report["baseline_selection"] = baseline_selection
    ground_truth, ground_truth_selection = load_dataset_scoped_json(
        validation.get("key_event_ground_truth"),
        package.experiment_id,
        repository_root=repository_root,
        artifact_label="关键事件真值",
    )
    recall_report = evaluate_key_event_recall(key_events, ground_truth)
    recall_report["ground_truth_selection"] = ground_truth_selection
    report["ground_truth_selection"] = ground_truth_selection
    recall_iou = float(validation.get("key_event_recall_iou", 0.50))
    minimum_precision = float(validation.get("minimum_key_event_precision", 0.80))
    minimum_recall = float(validation.get("minimum_key_event_recall", 0.80))
    recall_gate = {
        "evaluated": bool(recall_report.get("evaluated")),
        "passed": None,
        "temporal_iou_threshold": recall_iou,
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "precision": None,
        "recall": None,
        "small_sample_warning": recall_report.get("small_sample_warning"),
    }
    if recall_gate["evaluated"]:
        selected_threshold = next(
            (
                item
                for item in recall_report.get("threshold_results") or []
                if abs(float(item["temporal_iou_threshold"]) - recall_iou) < 1e-9
            ),
            None,
        )
        if selected_threshold is None:
            raise ValueError(
                "Configured key-event recall IoU is absent from evaluation thresholds"
            )
        recall_gate["precision"] = selected_threshold.get("precision")
        recall_gate["recall"] = selected_threshold.get("recall")
        recall_gate["passed"] = bool(
            recall_gate["precision"] is not None
            and recall_gate["recall"] is not None
            and float(recall_gate["precision"]) >= minimum_precision
            and float(recall_gate["recall"]) >= minimum_recall
        )
        report["passed"] = bool(report.get("passed")) and bool(recall_gate["passed"])
        report["status"] = "passed" if report["passed"] else "failed"
    report["key_event_recall"] = recall_gate
    report["reevaluation"] = {
        "mode": "packaged_json_only",
        "video_files_opened": 0,
        "clock_csv_files_opened": 0,
        "model_calls": 0,
        "token_usage": 0,
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
    }
    if write:
        quality_path = archive / "JSON-Config-Files" / "quality_acceptance.json"
        previous_path = (
            archive
            / "JSON-Config-Files"
            / "Quality-History"
            / "quality_acceptance.before_packaged_reevaluation.json"
        )
        if quality_path.is_file() and not previous_path.exists():
            previous_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(quality_path, previous_path)
        write_json(quality_path, report)
        write_json(
            archive / "JSON-Config-Files" / "key_material_recall_eval.json",
            recall_report,
        )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("replay-quality-ledger")
def replay_quality_ledger_command(
    archive: Annotated[
        Path, typer.Option("--archive", "-a", exists=True, file_okay=False)
    ],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Replay CV quality decisions from JSON only; never open video or call MLLM."""

    result = replay_quality_decisions_from_ledgers(archive, load_config(config))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(str(output.resolve()))
        return
    typer.echo(rendered)


@app.command("inspect-quality-ledger-inputs")
def inspect_quality_ledger_inputs_command(
    archive: Annotated[
        Path, typer.Option("--archive", "-a", exists=True, file_okay=False)
    ],
) -> None:
    """Validate bounded replay inputs without opening source video or clock CSV."""

    result = inspect_quality_ledger_inputs(archive)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("register-archived-collection")
def register_archived_collection_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    archive: Annotated[
        Path, typer.Option("--archive", "-a", exists=True, file_okay=False)
    ],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Register an existing accepted archive in the central collection ledger."""

    try:
        formal_gate = validate_formal_archive_release(archive)
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(f"Formal archive gate failed: {exc}") from exc
    report_evaluations = list(
        (archive / "Lab-Daily-Reports").glob("*/Daily-Report-Eval.json")
    )
    if not report_evaluations or not all(
        json.loads(path.read_text(encoding="utf-8-sig")).get("passed")
        for path in report_evaluations
    ):
        raise typer.BadParameter("A passing daily-report evaluation is required")
    professional_pdfs = list(
        (archive / "Professional-PDFs").glob(
            "VisionCortex-Professional-Evidence-Report-*.pdf"
        )
    )
    if not professional_pdfs:
        raise typer.BadParameter("A professional evidence PDF is required")
    settings = load_config(config)
    run_id = f"archive-registration-{datetime.now():%Y%m%d-%H%M%S}"
    ledger = record_collection_state(
        settings,
        experiment_id,
        archive_name=archive.name,
        run_id=run_id,
        state="archived",
        details={
            "formal_archive": str(archive.resolve()),
            "registration_only": True,
            "daily_report_evaluations": len(report_evaluations),
            "professional_pdf_count": len(professional_pdfs),
            "formal_gate": formal_gate,
        },
    )
    typer.echo(
        json.dumps(
            {
                "state": "archived",
                "source_experiment_id": experiment_id,
                "archive": str(archive.resolve()),
                "ledger": str(ledger),
                "model_calls": 0,
                "token_usage": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("prepare-engine")
def prepare_engine_command(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
) -> None:
    settings = load_config(config)
    perf = settings["performance"]
    from ultralytics import YOLO

    configured_candidates = perf.get("engine_batch_candidates")
    if configured_candidates is None:
        configured_candidates = [perf.get("engine_batch_size", perf["batch_size"])]
    if not isinstance(configured_candidates, list):
        raise typer.BadParameter("engine_batch_candidates must be a list")
    batch_candidates: list[int] = []
    for value in configured_candidates:
        try:
            batch = int(value)
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter(
                "engine_batch_candidates must contain positive integers"
            ) from exc
        if batch <= 0:
            raise typer.BadParameter(
                "engine_batch_candidates must contain positive integers"
            )
        if batch not in batch_candidates:
            batch_candidates.append(batch)
    dynamic = bool(perf.get("engine_dynamic", True))
    workspace = float(perf.get("engine_workspace_gib", 3.0))
    if workspace <= 0:
        raise typer.BadParameter("engine_workspace_gib must be positive")
    autotune_iterations = int(perf.get("engine_autotune_iterations", 4))
    if autotune_iterations <= 0:
        raise typer.BadParameter("engine_autotune_iterations must be positive")
    autotune_memory_fraction = float(
        perf.get(
            "engine_autotune_max_gpu_memory_fraction",
            perf.get("max_gpu_memory_fraction", 0.9),
        )
    )
    if not 0.0 < autotune_memory_fraction <= 1.0:
        raise typer.BadParameter(
            "engine_autotune_max_gpu_memory_fraction must be in (0, 1]"
        )

    def benchmark_engine(engine: Path, batch: int) -> dict[str, object]:
        import numpy as np
        import torch

        image_size = int(perf["image_size"])
        frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        engine_model = YOLO(str(engine))
        try:
            engine_model.predict(
                [frame] * batch,
                imgsz=image_size,
                device=perf["device"],
                half=bool(perf["half"]),
                verbose=False,
            )
            torch.cuda.synchronize(int(perf["device"]))
            started = time.perf_counter()
            memory_samples: list[float] = []
            for _ in range(autotune_iterations):
                results = engine_model.predict(
                    [frame] * batch,
                    imgsz=image_size,
                    device=perf["device"],
                    half=bool(perf["half"]),
                    verbose=False,
                )
                if len(results) != batch:
                    raise RuntimeError(
                        "TensorRT autotune returned fewer predictions than input frames"
                    )
                torch.cuda.synchronize(int(perf["device"]))
                free_bytes, total_bytes = torch.cuda.mem_get_info(int(perf["device"]))
                memory_samples.append((total_bytes - free_bytes) / total_bytes)
            elapsed = time.perf_counter() - started
            peak_fraction = max(memory_samples)
            if peak_fraction > autotune_memory_fraction:
                raise RuntimeError(
                    "TensorRT autotune exceeded reserved-memory contract: "
                    f"peak={peak_fraction:.4f} limit={autotune_memory_fraction:.4f}"
                )
            return {
                "iterations": autotune_iterations,
                "frames": batch * autotune_iterations,
                "elapsed_seconds": round(elapsed, 6),
                "images_per_second": round(batch * autotune_iterations / elapsed, 6),
                "gpu_memory_fraction_peak": round(peak_fraction, 6),
                "gpu_memory_contract_fraction": autotune_memory_fraction,
            }
        finally:
            del engine_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for role in ("first_person", "third_person"):
        source = Path(settings["models"][role])
        destination = Path(settings["models"][f"{role}_engine"])
        if destination.is_file():
            typer.echo(f"{role}: 已存在 {destination}")
            continue
        typer.echo(f"{role}: 从 {source} 导出 TensorRT；候选 batch={batch_candidates}")
        failures: list[dict[str, object]] = []
        selected_batch: int | None = None
        selected_benchmark: dict[str, object] | None = None
        build_started = time.perf_counter()
        for batch in batch_candidates:
            temporary_root = Path(
                tempfile.mkdtemp(prefix=f"visioncortex-{role}-batch{batch}-")
            )
            model = None
            partial: Path | None = None
            try:
                temporary_source = temporary_root / "model.pt"
                shutil.copy2(source, temporary_source)
                model = YOLO(str(temporary_source))
                exported = Path(
                    model.export(
                        format="engine",
                        imgsz=int(perf["image_size"]),
                        half=bool(perf["half"]),
                        dynamic=dynamic,
                        batch=batch,
                        workspace=workspace,
                        device=perf["device"],
                    )
                )
                if not exported.is_file():
                    raise RuntimeError(
                        f"TensorRT exporter returned a missing artifact: {exported}"
                    )
                del model
                model = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                benchmark = benchmark_engine(exported, batch)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise RuntimeError(
                        f"TensorRT destination appeared during export: {destination}"
                    )
                partial = destination.with_name(
                    f".{destination.name}.partial-{uuid.uuid4().hex}"
                )
                shutil.move(str(exported), str(partial))
                if destination.exists():
                    raise RuntimeError(
                        f"TensorRT destination appeared during export: {destination}"
                    )
                partial.replace(destination)
                selected_batch = batch
                selected_benchmark = benchmark
                break
            except Exception as exc:
                failures.append(
                    {
                        "batch": batch,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[-1000:],
                    }
                )
                typer.echo(
                    f"{role}: batch={batch} 导出失败，尝试更小候选：{type(exc).__name__}",
                    err=True,
                )
            finally:
                if partial is not None and partial.exists():
                    partial.unlink()
                del model
                shutil.rmtree(temporary_root, ignore_errors=True)
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        if selected_batch is None:
            raise RuntimeError(
                f"{role} TensorRT export failed for all batch candidates: "
                f"{[item['batch'] for item in failures]}"
            )
        receipt = {
            "schema_version": "visioncortex-tensorrt-engine-build/1",
            "role": role,
            "source": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "engine": str(destination.resolve()),
            "engine_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "image_size": int(perf["image_size"]),
            "half": bool(perf["half"]),
            "dynamic": dynamic,
            "batch_candidates": batch_candidates,
            "selected_batch": selected_batch,
            "selected_benchmark": selected_benchmark,
            "workspace_gib": workspace,
            "autotune_iterations": autotune_iterations,
            "autotune_max_gpu_memory_fraction": autotune_memory_fraction,
            "failed_candidates": failures,
            "elapsed_seconds": round(time.perf_counter() - build_started, 6),
        }
        write_json(destination.with_suffix(destination.suffix + ".build.json"), receipt)
        typer.echo(f"{role}: {destination}")


@app.command("run-fixed-benchmark")
def run_fixed_benchmark_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx4060-laptop-production.yaml"),
    preprocessing_only: Annotated[
        bool,
        typer.Option(
            "--preprocessing-only/--full-pipeline",
            help="Stop after bounded dual-view CV acceptance; do not call MLLM or promote the fixed archive.",
        ),
    ] = False,
) -> None:
    """Run the registered six-view benchmark and reuse its fixed NAS archive."""

    experiment_id = "exp_20260810_144014_e918b762"
    archive_name = (
        "CustomFlow_standard_correct_12_ABCFA_0001--exp_20260810_144014_e918b762"
    )
    settings = load_config(config)
    if not preprocessing_only:
        ensure_ark_api_key(settings, required=True)
    settings["storage"]["sync_to_nas"] = True
    settings["project"]["preprocessing_acceptance_only"] = preprocessing_only
    run_id = f"cli-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    # The fixed benchmark is a production performance claim. Bind every run
    # to a fresh namespace so neither CV nor Ark evidence can silently become
    # a persisted-cache replay.
    settings["project"]["cache_mode"] = "cold"
    settings["project"]["cache_namespace"] = run_id
    fixed_root, nas_root, history_root = fixed_archive_staging_paths(
        settings, archive_name, run_id
    )
    local_runtime_root = Path(settings["storage"]["local_runtime_root"]).resolve()
    settings["project"]["output_root"] = str(local_runtime_root / "runs" / run_id)
    settings["storage"]["active_archive_path"] = str(nas_root)
    started = time.perf_counter()
    manifest, manifest_path, ingest = prepare_from_nas_index(
        settings, experiment_id, lambda message: typer.echo(f"[NAS] {message}")
    )
    manifest_path.write_text(
        yaml.safe_dump(
            manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    typer.echo(
        f"input_views={len(manifest.views)} input_mode={ingest['input_mode']} "
        f"source_copy_bytes={ingest['copied_source_bytes']} manifest={ingest['manifest']} "
        f"cache_mode=cold cache_namespace={run_id} "
        f"nas_staging={nas_root} fixed_output={fixed_root}"
    )
    result = EvidencePipeline(settings, _progress).run(manifest)
    if partial_result_available(Path(result)):
        typer.echo(f"{result} state=partial evidence=PARTIAL_EVIDENCE fixed_archive_promotion=skipped")
        return
    if preprocessing_only:
        typer.echo(
            f"{result} total_seconds={time.perf_counter() - started:.6f} "
            "run_mode=preprocessing_acceptance_only token_calls=0 "
            "fixed_archive_promotion=skipped"
        )
        return
    receipt = promote_fixed_archive(nas_root, fixed_root, history_root)
    typer.echo(
        f"{fixed_root} total_seconds={time.perf_counter() - started:.6f} "
        f"previous_package_retained={receipt['previous_package_retained']}"
    )


@app.command("run-index-collection")
def run_index_collection_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    archive_name: Annotated[str, typer.Option("--archive-name")],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx4060-laptop-production.yaml"),
    exhaustive_negative_audit: Annotated[
        bool,
        typer.Option(
            "--exhaustive-negative-audit",
            help=(
                "Scan the full physical media timeline of every eligible view "
                "at the configured fine detection FPS before formal promotion."
            ),
        ),
    ] = False,
) -> None:
    """Run one indexed NAS collection through staging and verified promotion."""

    settings = load_config(config)
    model_certification = audit_production_model_certification(settings)
    ark_preflight = ensure_ark_api_key(settings, required=True)
    safe_name = safe_archive_name(archive_name)
    run_id = f"collection-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    fixed_root, staging_root, history_root = fixed_archive_staging_paths(
        settings, safe_name, run_id
    )
    if fixed_root.exists() and not fixed_root.is_dir():
        raise typer.BadParameter(
            f"Formal archive path exists but is not a directory: {fixed_root}"
        )
    formal_archive_preexisting = fixed_root.is_dir()
    settings["storage"]["sync_to_nas"] = True
    settings["storage"]["active_archive_path"] = str(staging_root)
    settings["project"]["output_root"] = str(
        Path(settings["storage"]["local_runtime_root"]).resolve() / "runs" / run_id
    )
    # A formal production collection is a real cold execution. It may populate
    # immutable caches for later audited replay, but it must not silently turn
    # a previous acceptance/cache result into a new formal completion receipt.
    settings["project"]["cache_mode"] = "cold"
    settings["project"]["cache_namespace"] = run_id
    initialize_nas_archive(settings, safe_name)
    started = time.perf_counter()
    record_collection_state(
        settings,
        experiment_id,
        archive_name=safe_name,
        run_id=run_id,
        state="queued",
        details={
            "staging": str(staging_root),
            "source_copy_bytes": 0,
            "formal_archive": str(fixed_root),
            "formal_archive_preexisting": formal_archive_preexisting,
            "promotion_mode": "verified_atomic_replace",
            "cache_mode": "cold",
            "cache_namespace": run_id,
            "exhaustive_negative_audit": exhaustive_negative_audit,
            "ark_preflight": ark_preflight,
            "model_certification": model_certification,
        },
    )
    try:
        record_collection_state(
            settings,
            experiment_id,
            archive_name=safe_name,
            run_id=run_id,
            state="processing",
            details={
                "staging": str(staging_root),
                "formal_archive": str(fixed_root),
                "formal_archive_preexisting": formal_archive_preexisting,
                "promotion_mode": "verified_atomic_replace",
                "cache_mode": "cold",
                "cache_namespace": run_id,
                "exhaustive_negative_audit": exhaustive_negative_audit,
                "ark_preflight": ark_preflight,
                "model_certification": model_certification,
            },
        )
        manifest, manifest_path, ingest = prepare_from_nas_index(
            settings, experiment_id, lambda message: typer.echo(f"[NAS] {message}")
        )
        scan_policy = _configure_index_full_timeline_scan(
            settings,
            ingest,
            requested_negative_audit=exhaustive_negative_audit,
        )
        write_json(
            staging_root / "JSON-Config-Files" / "full_timeline_scan_policy.json",
            scan_policy,
        )
        manifest_path.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        typer.echo(
            f"run_id={run_id} input_views={len(manifest.views)} "
            f"input_mode={ingest['input_mode']} source_copy_bytes="
            f"{ingest['copied_source_bytes']} staging={staging_root}"
        )
        result = EvidencePipeline(settings, _progress).run(manifest)
        if partial_result_available(Path(result)):
            record_collection_state(
                settings, experiment_id, archive_name=safe_name, run_id=run_id,
                state="partial", details={"staging": str(staging_root)},
            )
            typer.echo(f"{result} state=partial evidence=PARTIAL_EVIDENCE fixed_archive_promotion=skipped")
            return
        # Final participant key-frame selection can move two independently
        # proposed events onto the same physical transition.  Re-run the
        # deterministic presentation/semantic deduplication before promotion;
        # this opens only bounded accepted-event frames and makes no model call.
        repair_key_material_presentation_command(staging_root, config)
        cold_doubao_audit = _audit_true_cold_doubao_execution(staging_root)
        receipt = promote_fixed_archive(staging_root, fixed_root, history_root)
        record_collection_state(
            settings,
            experiment_id,
            archive_name=safe_name,
            run_id=run_id,
            state="archived",
            details={
                "formal_archive": str(fixed_root),
                "promotion_verification": receipt.get("verification"),
                "previous_package_retained": receipt.get(
                    "previous_package_retained", False
                ),
                "cold_doubao_audit": cold_doubao_audit,
                "full_timeline_scan_policy": scan_policy,
                "model_certification": model_certification,
            },
        )
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "state": "archived",
                    "source_experiment_id": experiment_id,
                    "archive": str(fixed_root),
                    "staging": str(staging_root),
                    "source_copy_bytes": ingest["copied_source_bytes"],
                    "total_seconds": round(time.perf_counter() - started, 6),
                    "cold_doubao_audit": cold_doubao_audit,
                    "full_timeline_scan_policy": scan_policy,
                    "model_certification": model_certification,
                    "promotion": receipt,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        record_collection_state(
            settings,
            experiment_id,
            archive_name=safe_name,
            run_id=run_id,
            state="failed",
            details={
                "staging": str(staging_root),
                "error": f"{type(exc).__name__}: {exc}",
                "total_seconds": round(time.perf_counter() - started, 6),
                "formal_archive": str(fixed_root),
                "formal_archive_preexisting": formal_archive_preexisting,
                "formal_archive_preserved": formal_archive_preexisting,
            },
        )
        raise


@app.command("run-index-staging")
def run_index_staging_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    archive_name: Annotated[str, typer.Option("--archive-name")],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
    cache_mode: Annotated[
        str,
        typer.Option(
            "--cache-mode",
            help="cold forces new CV/Ark work; reuse is the paired hot replay.",
        ),
    ] = "cold",
    cache_namespace: Annotated[
        str | None,
        typer.Option(
            "--cache-namespace",
            help="Audit namespace shared by one cold run and its hot replay.",
        ),
    ] = None,
    exhaustive_negative_audit: Annotated[
        bool,
        typer.Option(
            "--exhaustive-negative-audit",
            help=(
                "Staging-only audit: scan the full physical media timeline of "
                "every eligible view at the configured fine detection FPS."
            ),
        ),
    ] = False,
) -> None:
    """Run one indexed collection into NAS staging without formal promotion."""

    settings = load_config(config)
    ark_preflight = ensure_ark_api_key(settings, required=True)
    safe_name = safe_archive_name(archive_name)
    run_id = f"staging-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    normalized_mode, selected_namespace = _normalized_cache_policy(
        cache_mode, cache_namespace, run_id
    )
    staging_root = _create_staging_only_run_root(settings, safe_name, run_id)
    settings["storage"]["sync_to_nas"] = True
    settings["storage"]["active_archive_path"] = str(staging_root)
    settings["project"]["output_root"] = str(
        Path(settings["storage"]["local_runtime_root"]).resolve() / "runs" / run_id
    )
    settings["project"]["cache_mode"] = normalized_mode
    settings["project"]["cache_namespace"] = selected_namespace
    initialize_nas_archive(settings, safe_name)
    receipt_path = staging_root / "JSON-Config-Files" / "staging_run_receipt.json"
    started = time.perf_counter()
    receipt = {
        "schema_version": "visioncortex-staging-run/1",
        "run_id": run_id,
        "state": "preparing",
        "source_experiment_id": experiment_id,
        "archive_name": safe_name,
        "staging_only": True,
        "formal_archive_promotion": False,
        "formal_archive_write": False,
        "source_copy_bytes": 0,
        "cache_mode": normalized_mode,
        "cache_namespace": selected_namespace,
        "exhaustive_negative_audit": exhaustive_negative_audit,
        "staging_root": str(staging_root),
        "started_at": datetime.now().astimezone().isoformat(),
        "ark_preflight": ark_preflight,
    }
    write_json(receipt_path, receipt)
    try:
        manifest, manifest_path, ingest = prepare_from_nas_index(
            settings, experiment_id, lambda message: typer.echo(f"[NAS] {message}")
        )
        if int(ingest["copied_source_bytes"]) != 0:
            raise RuntimeError(
                f"Zero-copy invariant failed: copied_source_bytes={ingest['copied_source_bytes']}"
            )
        scan_policy = _configure_index_full_timeline_scan(
            settings,
            ingest,
            requested_negative_audit=exhaustive_negative_audit,
        )
        write_json(
            staging_root / "JSON-Config-Files" / "full_timeline_scan_policy.json",
            scan_policy,
        )
        manifest_path.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        receipt.update(
            {
                "state": "processing",
                "input_view_count": len(manifest.views),
                "input_mode": ingest["input_mode"],
                "manifest": str(manifest_path),
                "source_copy_bytes": int(ingest["copied_source_bytes"]),
                "full_timeline_scan_policy": scan_policy,
            }
        )
        write_json(receipt_path, receipt)
        typer.echo(
            f"run_id={run_id} input_views={len(manifest.views)} "
            f"input_mode={ingest['input_mode']} source_copy_bytes=0 "
            f"cache_mode={normalized_mode} cache_namespace={selected_namespace} "
            f"staging={staging_root}"
        )
        result = EvidencePipeline(settings, _progress).run(manifest)
        receipt.update(
            {
                "state": "partial" if partial_result_available(Path(result)) else "completed",
                "completed_at": datetime.now().astimezone().isoformat(),
                "total_seconds": round(time.perf_counter() - started, 6),
                "pipeline_result": str(result),
            }
        )
        write_json(receipt_path, receipt)
        typer.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
    except Exception as exc:
        receipt.update(
            {
                "state": "failed",
                "failed_at": datetime.now().astimezone().isoformat(),
                "total_seconds": round(time.perf_counter() - started, 6),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(receipt_path, receipt)
        raise


@app.command("serve")
def serve_command(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
) -> None:
    import os

    import uvicorn

    if config:
        os.environ["VISIONCORTEX_CONFIG"] = str(config.resolve())
    # HTTP readers must not prevent the lifespan from draining durable workers.
    uvicorn.run("visioncortex.api:app", host=host, port=port, reload=False,
                timeout_graceful_shutdown=30)


@app.command("refresh-key-json")
def refresh_key_json_command(
    archive: Annotated[
        Path, typer.Option("--archive", "-a", exists=True, file_okay=False)
    ],
) -> None:
    """Rewrite key-material JSON sidecars using the normalized event contract."""
    layout = ArchiveLayout(archive.resolve())
    package = RunSummary.model_validate_json(
        (layout.json_config / "evidence_package.json").read_text(encoding="utf-8")
    )
    transforms = {item.view_id: item for item in package.alignments}
    key_events = [
        event for event in package.events if event.key_frames or event.key_clips
    ]
    refresh_key_material_metadata(
        layout,
        key_events,
        package.experiment_groups,
        transforms,
        archive_id=package.experiment_id,
    )
    normalized = [
        _artifact_json(
            next(
                group
                for group in package.experiment_groups
                if event.event_id in group.key_event_ids
            ),
            event,
            "key_material_event_index",
            "",
            None,
            transforms,
            package.experiment_id,
        )
        for event in key_events
    ]
    write_json(
        layout.key_materials / "Key-Materials-Model-Understanding.json", normalized
    )
    probe_path = layout.json_config / "video_probe.json"
    infos = (
        {
            view_id: VideoInfo.model_validate(payload)
            for view_id, payload in json.loads(
                probe_path.read_text(encoding="utf-8-sig")
            ).items()
        }
        if probe_path.is_file()
        else {}
    )
    index_manifest = build_archive_index(
        layout.root,
        package.experiment_id,
        normalized,
        package.events,
        package.experiment_groups,
        infos,
    )
    typer.echo(
        f"rewritten_events={len(normalized)} indexed_artifacts="
        f"{index_manifest['counts']['artifacts']}"
    )


def _stage_published_archive_for_repair(archive: Path) -> tuple[Path, Path]:
    """Copy one committed release into an isolated repair candidate."""

    run_id = f"repair-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    archive_name = safe_archive_name(archive.name)
    staging_root = (
        archive.parent / ".VisionCortex-Run-Staging" / archive_name / run_id
    )
    history_root = (
        archive.parent / ".VisionCortex-Run-History" / archive_name / run_id
    )
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        source = archive / directory
        if not source.is_dir():
            raise typer.BadParameter(
                f"Published archive directory is missing: {source}"
            )
        shutil.copytree(source, staging_root / directory)
    source_references = archive / "Original-Experiment-Videos"
    staged_references = staging_root / "Original-Experiment-Videos"
    staged_references.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_references.iterdir()):
        if source.is_file() and (
            source.name in ORIGINAL_REFERENCE_NAMES
            or source.suffix.lower() == ".ffconcat"
        ):
            shutil.copy2(source, staged_references / source.name)
    return staging_root, history_root


@app.command("repair-key-material-presentation")
def repair_key_material_presentation_command(
    archive: Annotated[
        Path, typer.Option("--archive", "-a", exists=True, file_okay=False)
    ],
    config: Annotated[
        Path, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = Path("configs/rtx3090ti-ubuntu-production.yaml"),
) -> None:
    """Re-select and re-render participant key frames without model calls.

    This command reuses a completed run's immutable detection ledgers and only
    reads the exact source frames/clips needed for accepted key events. It never
    re-runs the full scan, copies source video, or calls Ark.
    """

    started = time.perf_counter()
    archive = archive.resolve()
    layout = ArchiveLayout(archive)
    lock_path = archive / "run.lock"
    if lock_path.exists():
        raise typer.BadParameter(f"Archive is currently locked: {lock_path}")
    settings = load_config(config)
    if read_current_release_pointer(archive) is not None:
        try:
            validate_formal_archive_release(archive)
        except (OSError, RuntimeError, ValueError) as exc:
            raise typer.BadParameter(
                f"Published archive integrity failed before repair: {exc}"
            ) from exc
        staging_root, history_root = _stage_published_archive_for_repair(archive)
        typer.echo(f"published_archive_repair_staging={staging_root}")
        repair_key_material_presentation_command(staging_root, config)
        promotion = promote_fixed_archive(staging_root, archive, history_root)
        typer.echo(
            json.dumps(
                {
                    "status": "published",
                    "formal_archive": str(archive),
                    "repair_staging": str(staging_root),
                    "previous_release_history": str(history_root),
                    "promotion_verification": promotion.get("verification"),
                    "release_manifest_sha256": promotion.get(
                        "release_manifest_sha256"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    model_certification = audit_production_model_certification(settings)
    package_path = layout.json_config / "evidence_package.json"
    package_before_sha256 = hashlib.sha256(package_path.read_bytes()).hexdigest()
    summary = RunSummary.model_validate_json(
        package_path.read_text(encoding="utf-8-sig")
    )
    manifest = RunManifest.model_validate_json(
        (layout.json_config / "run_manifest.json").read_text(encoding="utf-8-sig")
    )
    infos = {
        view_id: VideoInfo.model_validate(payload)
        for view_id, payload in json.loads(
            (layout.json_config / "video_probe.json").read_text(encoding="utf-8-sig")
        ).items()
    }
    transforms = {item.view_id: item for item in summary.alignments}
    cache_identity = json.loads(
        (layout.json_config / "cache_identity.json").read_text(encoding="utf-8-sig")
    )
    layout.work = (
        Path(settings["storage"]["local_cache_root"]).resolve()
        / summary.experiment_id
        / str(cache_identity["cache_key"])
    )
    key_event_ids = {
        event_id
        for group in summary.experiment_groups
        for event_id in group.key_event_ids
    }
    reviewed_key_events = [
        event for event in summary.events if event.event_id in key_event_ids
    ]
    prior_curation_path = layout.json_config / "semantic_key_material_curation.json"
    prior_curation = (
        json.loads(prior_curation_path.read_text(encoding="utf-8-sig"))
        if prior_curation_path.is_file()
        else {}
    )
    if len(reviewed_key_events) > 1:
        key_events, semantic_curation = curate_semantically_reviewed_key_materials(
            layout,
            reviewed_key_events,
            summary.experiment_groups,
            settings,
            publisher=None,
        )
    else:
        key_events = [
            event for event in reviewed_key_events if event_is_formal(event)
        ]
        semantic_curation = prior_curation
    segment_semantic_repairs = _synchronize_segments_with_final_key_events(
        summary.segments, summary.experiment_groups, key_events
    )
    semantic_state_machine_repairs: list[dict[str, object]] = []
    for event in key_events:
        review = dict(event.semantic_review or {})
        pre_curation_action = str(
            review.get("pre_curation_action_type") or event.action_type.value
        )
        expected_state_action = (
            "liquid_transfer"
            if event.action_type.value == "liquid_movement"
            else event.action_type.value
        )
        current_state_action = str((event.state_machine or {}).get("action_type") or "")
        if (
            pre_curation_action == event.action_type.value
            or current_state_action == expected_state_action
        ):
            continue
        pre_curation_state = dict(
            review.get("pre_curation_state_machine") or event.state_machine or {}
        )
        rebuilt_state = build_event_state_receipt(event, settings)
        rebuilt_state["derivation"] = {
            "source": "semantic_relabel_confirmed_state_proof",
            "pre_curation_action_type": pre_curation_action,
            "final_action_type": event.action_type.value,
            "model_confidence": float(review.get("model_confidence") or 0.0),
            "direct_supporting_views": list(event.supporting_views),
        }
        for transition in rebuilt_state.get("transition_trace") or []:
            transition["source"] = "semantic_relabel_confirmed_state_proof"
        identity = rebuilt_state.get("object_identity") or {}
        identity["track_tokens"] = []
        identity["identity_status"] = "semantic_participant_classes"
        rebuilt_state["object_identity"] = identity
        event.state_machine = rebuilt_state
        review["pre_curation_state_machine"] = pre_curation_state
        review["semantic_state_machine_rebuilt"] = True
        event.semantic_review = review
        semantic_state_machine_repairs.append(
            {
                "event_id": event.event_id,
                "pre_curation_action_type": pre_curation_action,
                "pre_curation_state_action": current_state_action,
                "final_action_type": event.action_type.value,
                "final_state_action": expected_state_action,
            }
        )
    required_view_ids = {
        view_id
        for group in summary.experiment_groups
        for view_id in (group.first_person_view, group.third_person_view)
    }
    detection_paths = {
        view_id: layout.work / "detections-fine" / f"{view_id}.detections.jsonl"
        for view_id in sorted(required_view_ids)
    }
    missing_ledgers = [
        str(path) for path in detection_paths.values() if not path.is_file()
    ]
    if missing_ledgers:
        raise typer.BadParameter(
            "Immutable fine-scan ledger is missing: " + "; ".join(missing_ledgers)
        )
    previous_keys = {event.event_id: event.key_global_ms for event in key_events}
    previous_names = {
        group.group_id: {
            "experiment_name": group.experiment_name,
            "experiment_name_en": group.experiment_name_en,
        }
        for group in summary.experiment_groups
    }
    materialize_key_materials(
        layout,
        key_events,
        summary.experiment_groups,
        manifest.views,
        infos,
        transforms,
        detection_paths,
        settings,
        publisher=None,
        archive_id=summary.experiment_id,
    )
    obsolete_media_relocated = _relocate_unreferenced_key_material_event_directories(
        layout, key_events
    )
    final_state_receipt_repairs = _synchronize_final_event_state_receipts(
        key_events, settings
    )
    for event in key_events:
        review = dict(event.semantic_review or {})
        pre_curation_action = str(
            review.get("pre_curation_action_type") or event.action_type.value
        )
        expected_peak_us = round(event.key_global_ms * 1000.0)
        if (
            pre_curation_action == event.action_type.value
            or int((event.state_machine or {}).get("peak_timestamp_us") or -1)
            == expected_peak_us
        ):
            continue
        rebuilt_state = build_event_state_receipt(event, settings)
        rebuilt_state["derivation"] = {
            "source": "semantic_relabel_confirmed_state_proof",
            "pre_curation_action_type": pre_curation_action,
            "final_action_type": event.action_type.value,
            "model_confidence": float(review.get("model_confidence") or 0.0),
            "direct_supporting_views": list(event.supporting_views),
        }
        for transition in rebuilt_state.get("transition_trace") or []:
            transition["source"] = "semantic_relabel_confirmed_state_proof"
        identity = rebuilt_state.get("object_identity") or {}
        identity["track_tokens"] = []
        identity["identity_status"] = "semantic_participant_classes"
        rebuilt_state["object_identity"] = identity
        event.state_machine = rebuilt_state
        review["semantic_state_machine_rebuilt"] = True
        event.semantic_review = review
        semantic_state_machine_repairs.append(
            {
                "event_id": event.event_id,
                "pre_curation_action_type": pre_curation_action,
                "pre_curation_state_action": str(
                    (review.get("pre_curation_state_machine") or {}).get("action_type")
                    or ""
                ),
                "final_action_type": event.action_type.value,
                "final_state_action": str(rebuilt_state["action_type"]),
                "timing_resynchronized_after_key_frame_selection": True,
            }
        )
    final_annotation = _rerender_curated_participant_annotations(
        layout, key_events, summary.experiment_groups, settings
    )
    language_normalization = normalize_final_group_action_language(
        summary.experiment_groups, key_events
    )
    group_by_id = {group.group_id: group for group in summary.experiment_groups}
    for segment in summary.segments:
        group = group_by_id.get(str(segment.group_id))
        if group is None:
            continue
        segment.experiment_name = group.experiment_name
        segment.experiment_name_en = group.experiment_name_en
        segment.semantic_understanding = group.model_understanding
    refreshed_experiment_clip_sidecars: list[str] = []
    for group in summary.experiment_groups:
        atomic_ids = set(group.atomic_experiment_ids)
        atomic_segments = [
            segment.model_dump(mode="json")
            for segment in summary.segments
            if segment.segment_id in atomic_ids
        ]
        for relative in group.video_json.values():
            sidecar = archive / relative
            if not sidecar.is_file():
                continue
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            payload["group"] = group.model_dump(mode="json")
            payload["atomic_experiments"] = atomic_segments
            write_json(sidecar, payload)
            refreshed_experiment_clip_sidecars.append(str(relative))
    write_json(
        layout.json_config / "experiment_group_understanding.json",
        {
            "schema_version": "visioncortex-experiment-group-understanding/2",
            "refinement_pass": "post_event_semantic_curation",
            "presentation_repair": True,
            "groups": [
                group.model_dump(mode="json") for group in summary.experiment_groups
            ],
        },
    )
    key_understanding_path = (
        layout.json_config / "key_material_model_understanding.json"
    )
    if key_understanding_path.is_file():
        key_understanding = json.loads(
            key_understanding_path.read_text(encoding="utf-8-sig")
        )
        repaired_by_id = {event.event_id: event for event in key_events}
        key_understanding["events"] = [
            repaired_by_id.get(str(payload.get("event_id")), None).model_dump(
                mode="json"
            )
            if repaired_by_id.get(str(payload.get("event_id")), None) is not None
            else payload
            for payload in key_understanding.get("events") or []
        ]
        write_json(key_understanding_path, key_understanding)
    curation_path = layout.json_config / "semantic_key_material_curation.json"
    if curation_path.is_file():
        curation = json.loads(curation_path.read_text(encoding="utf-8-sig"))
        repaired_ids = {
            str(item["event_id"])
            for item in [
                *semantic_state_machine_repairs,
                *final_state_receipt_repairs,
            ]
        }
        for record in curation.get("records") or []:
            if str(record.get("event_id")) in repaired_ids:
                record["semantic_state_machine_rebuilt"] = True
        curation["final_annotation"] = {
            key: value for key, value in final_annotation.items() if key != "records"
        }
        curation["presentation_repair"] = {
            "schema_version": "visioncortex-semantic-presentation-repair/1",
            "full_scan_repeated": False,
            "source_copy_bytes": 0,
            "post_relabel_duplicate_count": semantic_curation.get(
                "post_relabel_duplicate_count", 0
            ),
            "state_receipt_repairs": [
                *semantic_state_machine_repairs,
                *final_state_receipt_repairs,
            ],
            "segment_semantic_repairs": segment_semantic_repairs,
            "obsolete_media_relocated": obsolete_media_relocated,
        }
        write_json(curation_path, curation)
    refresh_key_material_metadata(
        layout,
        key_events,
        summary.experiment_groups,
        transforms,
        archive_id=summary.experiment_id,
    )
    write_timestamp_tables(
        layout, key_events, bool(settings.get("archive", {}).get("create_xlsx", True))
    )
    normalized_events = [
        _artifact_json(
            next(
                group
                for group in summary.experiment_groups
                if event.event_id in group.key_event_ids
            ),
            event,
            "key_material_event_index",
            "",
            None,
            transforms,
            summary.experiment_id,
        )
        for event in key_events
    ]
    write_json(
        layout.key_materials / "Key-Materials-Model-Understanding.json",
        normalized_events,
    )
    summary.physical_change_log = build_physical_change_log(key_events)
    write_json(
        layout.json_config / "physical_change_log.json",
        [item.model_dump(mode="json") for item in summary.physical_change_log],
    )
    write_json(package_path, summary.model_dump(mode="json"))
    quality_report = _refresh_repaired_quality_acceptance(
        layout,
        settings,
        summary.experiment_id,
        summary.experiment_groups,
        key_events,
    )
    evaluation = evidence_package_eval(
        archive,
        summary.experiment_groups,
        summary.segments,
        key_events,
        transforms,
    )
    index_manifest = build_archive_index(
        archive,
        summary.experiment_id,
        normalized_events,
        summary.events,
        summary.experiment_groups,
        infos,
        hash_workers=int(settings.get("performance", {}).get("io_workers", 4)),
    )
    summary.stats["evidence_index"] = {
        "schema_version": index_manifest["schema_version"],
        "manifest": "JSON-Config-Files/evidence_index_manifest.json",
        "counts": index_manifest["counts"],
        "fts5_enabled": index_manifest["fts5_enabled"],
    }
    index_validation = index_manifest.get("validation") or {}
    index_check = {
        "id": summary.experiment_id,
        "check": "evidence_index_and_one_hop_registries",
        "passed": index_validation.get("passed") is True
        and index_validation.get("all_artifacts_have_integrity_receipts") is True,
        "details": index_validation,
    }
    evaluation["checks"].append(index_check)
    if not index_check["passed"]:
        evaluation["failures"].append(
            "Evidence index or one-hop artifact/evidence registry validation failed"
        )
        evaluation["passed"] = False
    write_json(layout.json_config / "evidence_package_eval.json", evaluation)
    write_json(package_path, summary.model_dump(mode="json"))
    report_manifest = generate_daily_report_from_archive(archive, settings)
    write_run_provenance(
        archive,
        settings,
        model_certification,
        repository_root=Path(__file__).resolve().parents[2],
    )
    write_archive_contract_manifest(archive)
    validate_archive_contracts_or_raise(archive)
    receipt = {
        "schema_version": "visioncortex-key-material-presentation-repair/1",
        "status": (
            "completed"
            if evaluation.get("passed") and quality_report.get("passed")
            else "failed"
        ),
        "experiment_id": summary.experiment_id,
        "model_calls": 0,
        "token_usage": 0,
        "full_scan_repeated": False,
        "source_copy_bytes": 0,
        "source_frame_access": "bounded accepted-key-events only",
        "immutable_detection_ledgers": [str(path) for path in detection_paths.values()],
        "key_frame_changes": [
            {
                "event_id": event.event_id,
                "previous_key_global_ms": previous_keys[event.event_id],
                "selected_key_global_ms": event.key_global_ms,
                "selection_offset_ms": round(
                    event.key_global_ms - previous_keys[event.event_id], 3
                ),
            }
            for event in key_events
        ],
        "experiment_name_changes": [
            {
                "group_id": group.group_id,
                "before": previous_names[group.group_id],
                "after": {
                    "experiment_name": group.experiment_name,
                    "experiment_name_en": group.experiment_name_en,
                },
            }
            for group in summary.experiment_groups
            if previous_names[group.group_id]
            != {
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
            }
        ],
        "language_normalization": language_normalization,
        "semantic_state_machine_repairs": semantic_state_machine_repairs,
        "final_state_receipt_repairs": final_state_receipt_repairs,
        "semantic_recuration": {
            "candidate_count": semantic_curation.get("candidate_count", 0),
            "accepted_count": semantic_curation.get("accepted_count", 0),
            "excluded_count": semantic_curation.get("excluded_count", 0),
            "post_relabel_duplicate_count": semantic_curation.get(
                "post_relabel_duplicate_count", 0
            ),
            "post_relabel_deduplication": semantic_curation.get(
                "post_relabel_deduplication", []
            ),
            "segment_semantic_repairs": segment_semantic_repairs,
        },
        "obsolete_media_relocated": obsolete_media_relocated,
        "final_annotation": {
            key: value for key, value in final_annotation.items() if key != "records"
        },
        "refreshed_experiment_clip_sidecars": sorted(
            refreshed_experiment_clip_sidecars
        ),
        "evidence_package_eval_passed": bool(evaluation.get("passed")),
        "quality_acceptance_passed": bool(quality_report.get("passed")),
        "quality_acceptance_status": quality_report.get("status"),
        "interaction_pair_annotation_gate_passed": bool(
            (quality_report.get("key_materials") or {}).get(
                "interaction_pair_annotation_gate_passed"
            )
        ),
        "action_participant_visibility_gate_passed": bool(
            (quality_report.get("key_materials") or {}).get(
                "action_participant_visibility_gate_passed"
            )
        ),
        "daily_report_passed": bool(report_manifest.get("passed")),
        "package_before_sha256": package_before_sha256,
        "package_after_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "duration_seconds": round(time.perf_counter() - started, 6),
    }
    receipt_path = layout.json_config / "key_material_presentation_repair.json"
    write_json(receipt_path, receipt)
    if (
        not evaluation.get("passed")
        or not quality_report.get("passed")
        or not report_manifest.get("passed")
    ):
        raise typer.Exit(code=1)
    typer.echo(json.dumps(receipt, ensure_ascii=False, indent=2))


@app.command("worker")
def worker_command(config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]):
    """Run the common durable worker without starting an HTTP server."""
    import asyncio
    import os
    import signal
    os.environ['VISIONCORTEX_CONFIG'] = str(config.resolve())
    os.environ['VISIONCORTEX_RUNTIME_ROLE'] = 'worker'
    from .api import _lifespan, _storage_maintenance
    if _storage_maintenance():
        raise typer.BadParameter('Storage maintenance is active; worker remains disabled')
    async def main():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, stop.set)
            except NotImplementedError:
                signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))
        async with _lifespan(None):
            await stop.wait()
    asyncio.run(main())


@app.command("cache-impact")
def cache_impact_command(before: Annotated[Path, typer.Option("--before", exists=True)],
                         after: Annotated[Path | None, typer.Option("--after", exists=True)] = None):
    """Plan affected stages without opening models or source media."""
    from .stage_dependencies import compare, source_manifest
    previous = json.loads(before.read_text())
    current = json.loads(after.read_text()) if after else source_manifest()
    typer.echo(json.dumps(compare(previous, current), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
