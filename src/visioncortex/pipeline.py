from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat as stat_module
import subprocess
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .actions import (
    audit_candidates,
    build_experiment_segments,
    build_view_activity_intervals,
    build_physical_change_log,
    generate_candidates,
    generate_coarse_activity_candidates,
    generate_motion_burst_candidates,
    generate_motion_safety_candidates,
    fuse_motion_probe_candidates,
    refine_motion_candidates_with_coarse,
    refine_liquid_events_with_context,
    select_fine_scan_views,
)
from .alignment import alignment_quality_report, build_alignments
from .action_semantics import (
    attach_action_observability,
    build_semantic_review_plan,
)
from .action_state_machine import (
    attach_continuous_action_states,
)
from .archive import (
    ArchiveLayout,
    _rerender_curated_participant_annotations,
    analyze_experiment_groups,
    analyze_key_materials,
    curate_semantically_reviewed_key_materials,
    finalize_archive,
    materialize_experiment_clips,
    materialize_key_materials,
    key_material_action_folder,
    prepare_key_material_category_layout,
    reconcile_visually_reviewed_participants,
    refresh_key_material_metadata,
    write_aligned_csv,
    write_key_material_category_index,
    write_json,
)
from .grouping import (
    build_experiment_groups,
    normalize_experiment_segments,
    prepare_formal_experiment_segments,
    select_key_events,
)
from .analysis.experiments import ExperimentAnalysisOperations, compose_experiments
from .analysis.scan_planning import FineScanPlanner
from .analysis.scan_execution import ProgressiveScanExecutor, ScanAlgorithms, ScanExecutionPorts
from .analysis.final_events import (
    FINAL_STEP_ACTION_PATTERNS as FINAL_STEP_ACTION_PATTERNS,
    _SAFE_POSTURE_ONLY_REWRITES as _SAFE_POSTURE_ONLY_REWRITES,
    _explicit_container_state_direction as _explicit_container_state_direction,
    _semantic_state_participants as _semantic_state_participants,
    _recover_group_storyboard_state_events as _recover_group_storyboard_state_events,
    normalize_final_group_action_language as normalize_final_group_action_language,
    refine_groups_from_final_events as refine_groups_from_final_events,
    _synchronize_final_event_state_receipts as _synchronize_final_event_state_receipts,
    _synchronize_segments_with_final_key_events as _synchronize_segments_with_final_key_events,
    validate_final_step_action_consistency as validate_final_step_action_consistency,
)
from .pathing import archive_relative_posix
from .ordering import candidate_sort_key
from .detection import iter_frame_evidence, scan_videos, validate_models
from .coarse_recall import (
    generate_open_vocabulary_coarse_candidates,
    generate_open_vocabulary_fine_candidates,
)
from .candidate_index import (
    CoarseFrameIndex,
    FineFrameIndex,
    build_coarse_frame_index,
    create_fine_frame_index,
    fine_frame_coverage_report,
    ingest_fine_frame_ledgers,
)
from .daily_reports import generate_daily_report_archive
from .decisions import decision_receipt
from .model_certification import audit_production_model_certification
from .provenance import write_run_provenance
from .partial_delivery import write_partial_delivery
from .schema_contracts import (
    validate_archive_contracts_or_raise,
    write_archive_contract_manifest,
)
from . import speech
from .runtime_control import analysis_execution
from .schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    FrameEvidence,
    PhysicalChange,
    RunManifest,
    VideoInfo,
    ViewInput,
    ViewRole,
)
from .video_io import (
    benchmark_sparse_decode_strategy,
    check_disk_capacity,
    create_grid_video,
    extract_view_clip,
    probe_video,
    probe_views,
    video_encoder_preflight,
    view_source_files,
    view_timestamp_files,
)
from .storage import (
    IncrementalArchivePublisher,
    initialize_nas_archive,
    snapshot_source_paths,
    source_cache_diagnostics,
)
from .telemetry import ResourceMonitor
from .reviewed_artifacts import load_dataset_scoped_json
from .validation import (
    evaluate_key_event_recall,
    finalize_quality_acceptance_claims,
    validate_experiment_and_material_quality,
)


ProgressCallback = Callable[[str, float, str], None]






















def _noop_progress(stage: str, progress: float, message: str) -> None:
    del stage, progress, message


def run_media_pipeline_preflight(
    manifest: RunManifest,
    infos: dict[str, Any],
    work_root: Path,
    preferred_encoder: str,
    duration_seconds: float,
) -> dict[str, Any]:
    """Exercise source decode and paired delivery encoding before long scans."""

    first = next(
        (view for view in manifest.views if view.role == ViewRole.FIRST_PERSON),
        None,
    )
    third = next(
        (view for view in manifest.views if view.role == ViewRole.THIRD_PERSON),
        None,
    )
    if first is None or third is None:
        raise ValueError("media pipeline preflight requires first- and third-person views")
    requested_ms = max(200.0, float(duration_seconds) * 1000.0)
    smoke_root = work_root / "media-pipeline-preflight"
    smoke_root.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[str, Path]] = []
    clip_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for label, view in (("First-Person", first), ("Third-Person", third)):
            info = infos[view.view_id]
            duration_ms = min(requested_ms, float(info.duration_ms))
            if duration_ms <= 0.0:
                raise ValueError(f"view has no probeable duration: {view.view_id}")
            start_ms = min(1000.0, max(0.0, float(info.duration_ms) - duration_ms))
            destination = smoke_root / f"{label}.mp4"
            clip_started = time.perf_counter()
            extract_view_clip(
                view,
                info,
                destination,
                start_ms,
                duration_ms,
                preferred_encoder,
            )
            rendered = probe_video(destination)
            if rendered.duration_ms <= 0.0 or rendered.width <= 0 or rendered.height <= 0:
                raise RuntimeError(f"encoded smoke clip is invalid: {destination}")
            clips.append((label, destination))
            clip_reports.append(
                {
                    "view_id": view.view_id,
                    "role": view.role.value,
                    "source_start_ms": round(start_ms, 3),
                    "requested_duration_ms": round(duration_ms, 3),
                    "encoded_duration_ms": round(rendered.duration_ms, 3),
                    "width": rendered.width,
                    "height": rendered.height,
                    "bytes": destination.stat().st_size,
                    "elapsed_seconds": round(time.perf_counter() - clip_started, 6),
                }
            )
        aligned = smoke_root / "Aligned_First+Third.mp4"
        grid_started = time.perf_counter()
        create_grid_video(clips, aligned, preferred_encoder)
        rendered_grid = probe_video(aligned)
        if (
            rendered_grid.duration_ms <= 0.0
            or rendered_grid.width <= 0
            or rendered_grid.height <= 0
        ):
            raise RuntimeError(f"encoded aligned smoke clip is invalid: {aligned}")
        report = {
            "schema_version": "visioncortex-media-pipeline-preflight/1",
            "status": "passed",
            "source_mode": (
                "segmented_virtual_timeline"
                if all(view.segments for view in manifest.views)
                else "continuous_file"
            ),
            "selected_encoder": preferred_encoder,
            "views": clip_reports,
            "aligned_output": {
                "encoded_duration_ms": round(rendered_grid.duration_ms, 3),
                "width": rendered_grid.width,
                "height": rendered_grid.height,
                "bytes": aligned.stat().st_size,
                "elapsed_seconds": round(time.perf_counter() - grid_started, 6),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "temporary_artifacts_retained": False,
        }
        shutil.rmtree(smoke_root)
        return report
    except Exception:
        # Retain the tiny local smoke directory on failure for incident review.
        raise


def _key_material_selection_report(
    groups: list[ExperimentGroup],
    segments: list[ExperimentSegment],
    events: list[EvidenceEvent],
    decision_receipts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Expose material recall and suspiciously sparse experiment coverage."""

    by_segment = {segment.segment_id: segment for segment in segments}
    by_event = {event.event_id: event for event in events}
    records: list[dict[str, Any]] = []
    for group in groups:
        candidate_ids = {
            event_id
            for segment_id in group.atomic_experiment_ids
            for event_id in by_segment[segment_id].event_ids
            if event_id in by_event and by_event[event_id].accepted
        }
        selected = [
            by_event[event_id]
            for event_id in group.key_event_ids
            if event_id in by_event
        ]
        counts: dict[str, int] = {}
        for event in selected:
            counts[event.action_type.value] = counts.get(event.action_type.value, 0) + 1
        duration_seconds = max(
            0.0, (group.global_end_ms - group.global_start_ms) / 1000.0
        )
        # This is a warning, not a fabricated quota. It makes sparse evidence
        # visible without inventing events or weakening cross-view acceptance.
        sparse_threshold = max(3, min(12, round(duration_seconds / 30.0)))
        records.append(
            {
                "group_id": group.group_id,
                "experiment_name": group.experiment_name,
                "duration_seconds": round(duration_seconds, 3),
                "accepted_physical_events": len(candidate_ids),
                "selected_key_materials": len(selected),
                "selection_rate": round(len(selected) / len(candidate_ids), 4)
                if candidate_ids
                else 0.0,
                "action_type_counts": counts,
                "low_recall_warning": duration_seconds >= 60.0
                and len(selected) < sparse_threshold,
                "low_recall_threshold": sparse_threshold,
            }
        )
    return {
        "schema_version": "visioncortex-key-material-selection/2",
        "selection_rule": (
            "accepted formal physical actions; duplicate only when action/object "
            "identity, positive interval overlap, and peak proximity all agree"
        ),
        "decision_receipt_complete": bool(decision_receipts is not None)
        and len(decision_receipts)
        == sum(record["accepted_physical_events"] for record in records),
        "decision_receipts": list(decision_receipts or []),
        "groups": records,
        "totals": {
            "accepted_physical_events": sum(
                record["accepted_physical_events"] for record in records
            ),
            "selected_key_materials": sum(
                record["selected_key_materials"] for record in records
            ),
            "groups_with_low_recall_warning": sum(
                bool(record["low_recall_warning"]) for record in records
            ),
        },
    }


def _merge_frame_evidence_ledgers(
    existing_path: Path,
    supplement_path: Path,
    output_path: Path,
    *,
    track_id_namespace: int,
) -> dict[str, Any]:
    """Streaming-merge a recall pass without loading full ledgers in RAM."""

    existing_count = 0
    supplement_count = 0
    deduplicated = 0
    merged_count = 0
    offset = max(1, int(track_id_namespace)) * 1_000_000

    def existing_frames():
        nonlocal existing_count
        for frame in iter_frame_evidence(existing_path):
            existing_count += 1
            yield frame

    def supplement_frames():
        nonlocal supplement_count
        for frame in iter_frame_evidence(supplement_path):
            supplement_count += 1
            yield frame.model_copy(
                update={
                    "detections": [
                        detection.model_copy(
                            update={
                                "track_id": (
                                    detection.track_id + offset
                                    if detection.track_id is not None
                                    else None
                                )
                            }
                        )
                        for detection in frame.detections
                    ]
                }
            )

    def identity(frame: FrameEvidence) -> tuple[int, int]:
        return frame.frame_index, round(frame.local_ms * 1000.0)

    def ordering(frame: FrameEvidence) -> tuple[float, int]:
        return frame.local_ms, frame.frame_index

    left_iterator = iter(existing_frames())
    right_iterator = iter(supplement_frames())
    left = next(left_iterator, None)
    right = next(right_iterator, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        while left is not None or right is not None:
            if left is None:
                selected = right
                right = next(right_iterator, None)
            elif right is None:
                selected = left
                left = next(left_iterator, None)
            elif identity(left) == identity(right):
                selected = right
                deduplicated += 1
                left = next(left_iterator, None)
                right = next(right_iterator, None)
            elif ordering(left) < ordering(right):
                selected = left
                left = next(left_iterator, None)
            else:
                selected = right
                right = next(right_iterator, None)
            assert selected is not None
            handle.write(selected.model_dump_json() + "\n")
            merged_count += 1
    temporary.replace(output_path)
    return {
        "existing_frames": existing_count,
        "supplement_frames": supplement_count,
        "merged_frames": merged_count,
        "deduplicated_frames": deduplicated,
        "track_id_namespace": offset,
        "merge_strategy": "streaming_sorted_supplement_overwrites_duplicate",
        "output_path": str(output_path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_cache_identity(config: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    """Bind resumable ledgers to code, config, models and concrete inputs."""

    package_root = Path(__file__).resolve().parent
    # Shared orchestration/schema changes still invalidate conservatively. These
    # modules only affect downstream interpretation, reports, speech or the UI.
    downstream_modules = {
        "mllm.py", "mllm_provider.py", "provider_connection.py", "provider_credentials.py",
        "provider_discovery.py", "ai_settings.py", "speech.py", "speech_worker.py",
        "speech_semantics.py", "daily_reports.py", "report_presentations.py",
        "report_brand.py", "partial_delivery.py", "api.py", "cli.py", "stage_refresh.py", "speech_timeline.py",
        "speech_search.py", "speech_refresh.py", "capture_quality.py",
    }
    from .stage_dependencies import CONTROL_MODULES
    downstream_modules.update(name + ".py" for name in CONTROL_MODULES)
    from .source_identity import SOURCE_RECIPE, source_digest, source_manifest
    code_sha256 = source_digest(source_manifest(package_root, exclude=downstream_modules))

    models = {}
    for role in ("first_person", "third_person"):
        path = Path(config["models"][role]).resolve()
        try:
            model_stat = path.stat()
        except OSError:
            model_stat = None
        if model_stat is not None and not stat_module.S_ISREG(model_stat.st_mode):
            model_stat = None
        models[role] = {
            "path": str(path),
            "size_bytes": model_stat.st_size if model_stat is not None else None,
            "sha256": _sha256_file(path) if model_stat is not None else None,
        }

    def absolute(path: Path) -> Path:
        return Path(os.path.abspath(str(path)))

    files_by_view = {
        view.view_id: {
            "videos": [absolute(path) for path in view_source_files(view)],
            "timestamps_csvs": [absolute(path) for path in view_timestamp_files(view)],
            "audio": [absolute(path) for path in speech.audio_files(view)],
        }
        for view in manifest.views
    }
    all_source_paths = [
        path
        for files in files_by_view.values()
        for file_type in ("videos", "timestamps_csvs", "audio")
        for path in files[file_type]
    ]
    source_snapshots, source_snapshot_report = snapshot_source_paths(
        all_source_paths,
        workers=int(config["performance"].get("source_stat_workers", 24)),
        max_age_seconds=0.0 if config.get("project", {}).get("resume_stages") else float(
            config["performance"].get("source_stat_cache_ttl_seconds", 120.0)
        ),
    )
    inputs = []
    for view in manifest.views:
        source_files = files_by_view[view.view_id]["videos"]
        clock_files = files_by_view[view.view_id]["timestamps_csvs"]
        inputs.append(
            {
                "view_id": view.view_id,
                "role": view.role.value,
                "input_mode": "segmented" if view.segments else "continuous_file",
                "audio": [{"path": str(path), "snapshot": source_snapshots[path]} for path in files_by_view[view.view_id]["audio"]],
                "audio_offsets_ms": [part.audio_offset_ms for part in (view.segments or [view])],
                "videos": [
                    {
                        "path": str(path),
                        "size_bytes": source_snapshots[path]["size_bytes"],
                        "mtime_ns": source_snapshots[path]["mtime_ns"],
                    }
                    for path in source_files
                ],
                "timestamps_csvs": [
                    {
                        "path": str(path),
                        "size_bytes": source_snapshots[path]["size_bytes"],
                        "mtime_ns": source_snapshots[path]["mtime_ns"],
                    }
                    for path in clock_files
                ],
            }
        )

    stable_config = deepcopy(config)
    stable_config.get("project", {}).pop("output_root", None)
    # Execution policy must not fork the deterministic cache identity: a cold
    # run and its explicit hot replay intentionally share the same namespace.
    # ``cache_namespace`` remains in the stable config and is therefore the
    # auditable isolation boundary between independent benchmark campaigns.
    stable_config.get("project", {}).pop("cache_mode", None)
    stable_config.get("project", {}).pop("resume_stages", None)
    # Ark response reuse is an execution policy independent of deterministic
    # CV artifacts.  Excluding it lets a recovery run reuse the exact verified
    # scan while forcing fresh semantic calls.
    stable_config.get("project", {}).pop("semantic_cache_mode", None)
    stable_config.get("project", {}).pop("semantic_recovery_attempt", None)
    # This switch changes only how far a run proceeds. Keeping it out of the
    # CV cache identity lets a later authorized full pipeline reuse the exact
    # accepted cold-start preprocessing ledgers without weakening provenance.
    stable_config.get("project", {}).pop("preprocessing_acceptance_only", None)
    for key in (
        "active_archive_path",
        "archive_root",
        "local_staging_root",
        "local_runtime_root",
        "local_input_root",
        "local_cache_root",
        "sync_to_nas",
    ):
        stable_config.get("storage", {}).pop(key, None)

    downstream_config = {key: stable_config.pop(key, None)
                         for key in ("mllm", "speech_recognition", "daily_report", "capture_quality")}
    audio_inputs = [{"view_id": item["view_id"], "audio": item.pop("audio"),
                     "audio_offsets_ms": item.pop("audio_offsets_ms")} for item in inputs]
    payload = {
        "schema_version": "visioncortex-cache-identity/3",
        "source_recipe": SOURCE_RECIPE,
        "code_sha256": code_sha256,
        "config": stable_config,
        "models": models,
        "inputs": inputs,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    payload["cache_key"] = hashlib.sha256(encoded).hexdigest()[:20]
    payload["downstream_dependencies"] = {
        "config": downstream_config, "audio_inputs": audio_inputs,
        "identity_policy": "ASR worker/runtime/source identity; semantic prompt/provider/evidence identity",
    }
    payload["execution_cache_policy"] = {
        "mode": str(config.get("project", {}).get("cache_mode") or "reuse"),
        "semantic_mode": str(
            config.get("project", {}).get(
                "semantic_cache_mode",
                config.get("project", {}).get("cache_mode") or "reuse",
            )
        ),
        "namespace": config.get("project", {}).get("cache_namespace"),
    }
    payload["source_snapshot_report"] = source_snapshot_report
    return payload


class EvidencePipeline:
    def __init__(self, config: dict[str, Any], progress: ProgressCallback | None = None):
        self.config = config
        mllm = config.get("mllm") or {}
        configured_timeout = float(mllm.get("timeout_seconds", 180))
        # Selected-provider connections are verified with a 180 s budget.
        # Legacy hardware profiles (including retained jobs) must not silently
        # shorten that budget for the larger real-experiment request.
        selected_provider = bool(mllm.get("provider") and mllm.get("quality_mode"))
        effective_timeout = max(180.0, configured_timeout) if selected_provider else configured_timeout
        if effective_timeout != configured_timeout:
            self.config = {**config, "mllm": {**mllm, "timeout_seconds": effective_timeout}}
        self._mllm_timeout_policy = {
            "configured_timeout_seconds": configured_timeout,
            "effective_timeout_seconds": effective_timeout,
            "selected_provider_minimum_seconds": 180 if selected_provider else None,
            "request_content_changed": False,
        }
        self.progress = progress or _noop_progress
        self._run_started_perf = 0.0
        self._run_started_iso = ""
        self._active_stage: str | None = None
        self._active_stage_started = 0.0
        self._active_stage_started_iso = ""
        self._stage_metrics: list[dict[str, Any]] = []
        self._stage_outcomes: dict[str, dict[str, Any]] = {}
        self._startup_metrics: dict[str, Any] = {}
        self._speech_understanding: dict[str, Any] = {}
        self._preprocessing_completed_seconds: float | None = None
        self._input_view_count = 0
        self._input_mode = "unknown"
        self._publisher: IncrementalArchivePublisher | None = None
        self._resource_monitor: ResourceMonitor | None = None
        self._view_runtime: dict[str, dict[str, Any]] = {}
        self._runtime_lock = threading.Lock()
        self._active_layout: ArchiveLayout | None = None
        self._current_experiment_id: str | None = None
        self._acceptance_baseline_selection: dict[str, Any] = {
            "configured": False,
            "applied": False,
            "reason": "not_evaluated",
        }
        self._model_certification_audit: dict[str, Any] = {
            "required": False,
            "status": "not_required_by_profile",
        }

    def _run_speech_stage(self, layout, manifest, infos, transforms):
        started = time.perf_counter()
        result = speech.run_stage(
            self.config, manifest, layout, infos, transforms,
            progress=lambda message: None, publisher=self._publisher,
        )
        if speech.enabled(self.config) and (layout.json_config / "Input-Manifests/input_seal.json").is_file():
            from .speech_timeline import build as build_speech_timeline
            build_speech_timeline(layout.root)
        return result, time.perf_counter() - started

    def _frame_index_path(self, work_dir: Path, filename: str) -> Path:
        # The evidence cache may live on SMB/NAS. SQLite locking and WAL shared
        # memory require a local filesystem; only rebuildable query indexes go
        # here. Authoritative JSONL ledgers remain in their configured cache.
        from .candidate_index import local_frame_index_path
        return local_frame_index_path(self.config, work_dir, filename)

    def _scan_progress(
        self, phase: str, view_id: str, completed_units: int, total_units: int
    ) -> None:
        with self._runtime_lock:
            runtime = self._view_runtime.setdefault(view_id, {})
            runtime.update(
                {
                    "state": f"{phase}_running"
                    if completed_units < total_units
                    else f"{phase}_completed",
                    "phase": phase,
                    "completed_units": completed_units,
                    "total_units": total_units,
                    "unit_progress": completed_units / total_units if total_units else 0.0,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if self._active_layout is None:
                return
            payload = {
                "schema_version": "visioncortex-source-progress/1",
                "phase": phase,
                "updated_at": runtime["updated_at"],
                "views": self._view_runtime,
            }
            path = self._active_layout.json_config / "source_progress.json"
            write_json(path, payload)
            if self._publisher is not None:
                self._publisher.publish_file(path)

    def _status(self, layout: ArchiveLayout, stage: str, progress: float, message: str) -> None:
        if self._resource_monitor is not None:
            self._resource_monitor.set_stage(stage)
        now_perf = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()
        if self._active_stage is not None and stage != self._active_stage:
            self._stage_metrics.append(
                {
                    "stage": self._active_stage,
                    "status": "failed" if stage == "failed" else self._stage_outcomes.get(self._active_stage, {}).get("status", "completed"),
                    "started_at": self._active_stage_started_iso,
                    "ended_at": now_iso,
                    "duration_seconds": round(now_perf - self._active_stage_started, 6),
                }
            )
        if stage != self._active_stage:
            if stage in {"completed", "partial", "failed"}:
                self._active_stage = None
            else:
                self._active_stage = stage
                self._active_stage_started = now_perf
                self._active_stage_started_iso = now_iso
        self.progress(stage, progress, message)
        status_payload = {
            "stage": stage,
            "progress": progress,
            "message": message,
            "updated_at": now_iso,
            "elapsed_seconds": round(now_perf - self._run_started_perf, 6)
            if self._run_started_perf
            else 0.0,
            "input_view_count": self._input_view_count,
            "input_mode": self._input_mode,
            "views": self._view_runtime,
            "stage_outcomes": self._stage_outcomes,
            "completed_stages": [
                item["stage"]
                for item in self._stage_metrics
                if item.get("status", "completed") == "completed"
            ],
            "failed_stage": next(
                (
                    item["stage"]
                    for item in reversed(self._stage_metrics)
                    if item.get("status") == "failed"
                ),
                None,
            ) if stage == "failed" else None,
        }
        write_json(
            layout.root / "run_status.json",
            status_payload,
        )
        if self.config.get("storage", {}).get("run_output_mode") == "nas_direct":
            write_json(layout.json_config / "pipeline_status.json", status_payload)
            stage_note = "处理中" if stage not in {"completed", "partial", "failed"} else (
                "分析结束，阶段成果已保存" if stage == "partial" else
                "分析已结束，等待归档校验" if stage == "completed" else "处理失败，已完成的阶段产出保留"
            )
            note_path = layout.root / "处理状态.txt"
            temporary_note = note_path.with_suffix(".txt.partial")
            temporary_note.write_text(
                f"{stage_note}\n{message}\n更新时间：{now_iso}\n"
                "各阶段产出位于对应文件夹；最终结果以归档校验为准。\n", encoding="utf-8"
            )
            temporary_note.replace(note_path)
        if self._publisher is not None:
            self._publisher.publish_status(status_payload)

    def _finish_quality_attention(self, layout: ArchiveLayout, events, groups) -> Path:
        """Finish an evaluated run without granting formal publication rights."""
        self._status(layout, "quality_attention", .995, "质量检查已结束，正在保存阶段成果与证据缺口")
        metrics = self._metrics(events, groups)
        write_json(layout.json_config / "run_metrics.json", metrics)
        write_partial_delivery(layout.root, metrics, analysis_finished=True)
        if self._publisher is not None:
            self._publisher.publish_directory("Partial-Results")
            self._publisher.publish_directory("JSON-Config-Files")
        self._status(layout, "partial", 1.0, "分析结束，部分证据不足；阶段成果与缺口报告已保存")
        return layout.root

    def _complete_stage(
        self,
        layout: ArchiveLayout,
        stage: str,
        artifacts: list[Path] | tuple[Path, ...] = (),
        *, status: str = "completed", reason: str | None = None,
    ) -> Path:
        """Publish saved artifacts and an explicit outcome for this component."""
        if status not in {"completed", "skipped", "failed", "blocked"}:
            raise ValueError("Invalid stage outcome")

        completed_at = datetime.now(timezone.utc).isoformat()
        elapsed_seconds = round(time.perf_counter() - self._run_started_perf, 6)
        stage_duration = None
        if self._active_stage == stage:
            stage_duration = round(time.perf_counter() - self._active_stage_started, 6)
        else:
            stage_duration = next(
                (
                    item["duration_seconds"]
                    for item in reversed(self._stage_metrics)
                    if item["stage"] == stage
                ),
                None,
            )
        relative_artifacts: list[str] = []
        for artifact in artifacts:
            artifact = Path(artifact)
            if not artifact.exists():
                raise FileNotFoundError(f"Completed stage artifact is missing: {artifact}")
            relative = archive_relative_posix(artifact, layout.root)
            relative_artifacts.append(relative)
            if self._publisher is not None:
                if artifact.is_dir():
                    self._publisher.publish_directory(Path(relative))
                else:
                    self._publisher.publish_file(artifact)
        receipt = {
            "schema_version": "visioncortex-stage-receipt/1",
            "stage": stage,
            "status": status,
            "reason": reason,
            "completed_at": completed_at,
            "run_elapsed_seconds": elapsed_seconds,
            "stage_duration_seconds": stage_duration,
            "archive_mode": self.config.get("storage", {}).get("run_output_mode", "local"),
            "archive_root": str(layout.root),
            "artifacts": relative_artifacts,
            "token_ledger": "JSON-Config-Files/run_metrics.json",
        }
        from .stage_versions import save_version
        version = save_version(layout.root, list(map(Path, artifacts)), receipt)
        if self._publisher is not None:
            self._publisher.publish_directory(version.parent.relative_to(layout.root))
        receipt["version_manifest"] = version.relative_to(layout.root).as_posix()
        receipt["archive_status"] = "pending" if getattr(self._publisher, "pending", None) else "saved"
        receipt_path = layout.json_config / "Stage-Receipts" / f"{stage}.json"
        write_json(receipt_path, receipt)
        if self._publisher is not None:
            self._publisher.publish_file(receipt_path)
        self._stage_outcomes[stage] = {"status": status, "reason": reason, "archive_status": receipt["archive_status"],
                                       "completed_at": completed_at}
        # A visible inventory advances only after the actual files and receipt
        # have been written/published. It is never a formal release pointer.
        completed = []
        for path in sorted(receipt_path.parent.glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("completed_at", "") >= self._run_started_iso:
                completed.append(item)
        index_path = layout.root / "StageInventory.json"
        write_json(index_path, {"schema_version": "visioncortex-stage-outputs/1",
                               "formal_release": False, "updated_at": completed_at,
                               "stages": completed})
        if self._publisher is not None:
            self._publisher.publish_file(index_path)
        return receipt_path

    def _checkpoint_key_material_understanding(
        self,
        layout: ArchiveLayout,
        stage: str,
        events: list[EvidenceEvent],
        groups: list[ExperimentGroup],
        semantic_curation: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Keep each completed understanding pass before later passes mutate it."""

        review_status = {
            "mllm": "pending_material_refinement",
            "material_refinement": "pending_semantic_refinement",
            "semantic_refinement": "pending_quality_acceptance",
        }[stage]
        understanding = {
            "schema_version": "visioncortex-key-material-understanding/1",
            "refinement_stage": stage,
            "review_status": review_status,
            "events": [event.model_dump(mode="json") for event in events],
            "semantic_curation": {
                key: value
                for key, value in (semantic_curation or {}).items()
                if key != "records"
            },
        }
        metrics = self._metrics(events, groups)
        snapshots = layout.json_config / "Stage-Outputs"
        payloads = {
            snapshots / f"{stage}.json": understanding,
            snapshots / f"{stage}-metrics.json": metrics,
            layout.json_config / "key_material_model_understanding.json": understanding,
            layout.json_config / "run_metrics_live.json": metrics,
        }
        for path, payload in payloads.items():
            write_json(path, payload)
        return list(payloads)

    def _acceptance_baseline(self) -> dict[str, Any] | None:
        configured = self.config.get("validation", {}).get("acceptance_baseline")
        payload, selection = load_dataset_scoped_json(
            configured,
            self._current_experiment_id,
            repository_root=Path(__file__).resolve().parents[2],
            artifact_label="验收基线",
        )
        self._acceptance_baseline_selection = selection
        return payload

    def _run_quality_acceptance(
        self,
        layout: ArchiveLayout,
        groups: list[ExperimentGroup],
        key_events: list[EvidenceEvent],
    ) -> dict[str, Any]:
        validation = self.config.get("validation", {})
        baseline = self._acceptance_baseline()
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
        report["baseline_selection"] = dict(
            self._acceptance_baseline_selection
        )
        key_event_ground_truth, ground_truth_selection = (
            load_dataset_scoped_json(
                validation.get("key_event_ground_truth"),
                self._current_experiment_id,
                repository_root=Path(__file__).resolve().parents[2],
                artifact_label="关键事件真值",
            )
        )
        recall_report = evaluate_key_event_recall(
            key_events, key_event_ground_truth
        )
        recall_report["ground_truth_selection"] = ground_truth_selection
        recall_gate = {
            "evaluated": bool(recall_report.get("evaluated")),
            "passed": None,
            "temporal_iou_threshold": float(
                validation.get("key_event_recall_iou", 0.50)
            ),
            "minimum_precision": float(
                validation.get("minimum_key_event_precision", 0.80)
            ),
            "minimum_recall": float(
                validation.get("minimum_key_event_recall", 0.80)
            ),
            "precision": None,
            "recall": None,
            "small_sample_warning": recall_report.get(
                "small_sample_warning"
            ),
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
                and float(recall_gate["precision"])
                >= recall_gate["minimum_precision"]
                and float(recall_gate["recall"])
                >= recall_gate["minimum_recall"]
            )
            report["passed"] = bool(report.get("passed")) and bool(
                recall_gate["passed"]
            )
            if not recall_gate["passed"]:
                report["status"] = "failed"
        report["key_event_recall"] = recall_gate
        step_consistency = validate_final_step_action_consistency(
            groups, key_events
        )
        report["step_action_consistency"] = step_consistency
        if not step_consistency["passed"]:
            report["passed"] = False
            report["status"] = "failed"
        finalize_quality_acceptance_claims(
            report, recall_gate, step_consistency
        )
        write_json(layout.json_config / "quality_acceptance.json", report)
        write_json(
            layout.json_config / "key_material_recall_eval.json",
            recall_report,
        )
        return report

    def _run_boundary_precheck(
        self,
        layout: ArchiveLayout,
        groups: list[ExperimentGroup],
        progressive_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validation = self.config.get("validation", {})
        full_report = validate_experiment_and_material_quality(
            groups,
            [],
            self._acceptance_baseline(),
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
        boundary = full_report["experiment_boundaries"]
        boundary_evaluated = bool(boundary["evaluated"])
        boundary_passed = (
            bool(boundary["passed"]) if boundary_evaluated else True
        )
        progressive_evaluated = progressive_report is not None
        progressive_passed = bool(
            progressive_report.get("quality_complete", True)
            if progressive_report is not None
            else True
        )
        evaluated = boundary_evaluated or progressive_evaluated
        passed = boundary_passed and progressive_passed
        # A reviewed baseline regression invalidates the bounded experiment
        # itself and remains a blocking pre-model failure.  In contrast, an
        # unresolved recall candidate is event-level uncertainty: retain it in
        # the audit ledger, mark the run partial, and continue producing
        # machine-verifiable evidence for the candidates that did close across
        # views.  One uncertain candidate must not discard an otherwise usable
        # analysis run.
        blocking_failure = boundary_evaluated and not boundary_passed
        continuation_status = (
            "passed" if passed else "failed" if blocking_failure else "partial"
        )
        report = {
            "schema_version": "visioncortex-boundary-precheck/1",
            "status": continuation_status,
            "evaluated": evaluated,
            "passed": passed,
            "blocking_failure": blocking_failure,
            "analysis_continuation_allowed": not blocking_failure,
            "evidence_classification": (
                "PROVEN" if passed else "NOT_PROVEN" if blocking_failure else "PARTIAL_EVIDENCE"
            ),
            "baseline": full_report["baseline"],
            "baseline_selection": dict(self._acceptance_baseline_selection),
            "thresholds": full_report["thresholds"],
            "experiment_boundaries": boundary,
            "cross_view_cluster_completeness": {
                "evaluated": progressive_evaluated,
                "passed": progressive_passed,
                "unresolved_candidate_ids": (
                    progressive_report.get("unresolved_candidate_ids", [])
                    if progressive_report is not None
                    else []
                ),
                "group_local_recall": (
                    progressive_report.get("group_local_recall", {})
                    if progressive_report is not None
                    else {}
                ),
            },
            "gate_position": "before_experiment_and_key_material_model_calls",
        }
        path = layout.json_config / "boundary_precheck.json"
        write_json(path, report)
        if (
            blocking_failure
            and bool(validation.get("fail_before_model_on_boundary_regression", True))
        ):
            raise RuntimeError(
                "Bounded experiment quality precheck failed before model calls; "
                f"see {path}"
            )
        return report

    def _metrics(self, events, groups=()) -> dict[str, Any]:
        key_calls = []
        visual_calls_by_fingerprint: dict[str, dict[str, Any]] = {}
        for event in events:
            understanding = event.model_understanding or {}
            if "usage" in understanding or understanding.get("status") in {"completed", "failed"}:
                key_calls.append(
                    {
                        "stage": "key_material_understanding",
                        **{key: understanding.get(key) for key in ("provider", "api_protocol", "request_id", "response_model")},
                        "event_id": event.event_id,
                        "model": understanding.get("model", self.config["mllm"]["model"]),
                        "status": understanding.get("status"),
                        "latency_seconds": understanding.get("latency_seconds"),
                        "attempts": understanding.get("attempts"),
                        "cache_reused": bool(understanding.get("cache_reused")),
                        "attempt_receipts": understanding.get("attempt_receipts", []),
                        "usage": understanding.get("usage", {}),
                    }
                )
            for fingerprint, review in (event.observability.get("participant_visual_review") or {}).items():
                if not review.get("request_attempted"):
                    continue
                call = {
                    "stage": "participant_visual_review",
                    **{key: review.get(key) for key in ("provider", "api_protocol", "request_id", "response_model")},
                    "event_id": event.event_id,
                    "input_fingerprint": fingerprint,
                    "model": review.get("model", self.config["mllm"]["model"]),
                    "status": review.get("status"),
                    "latency_seconds": review.get("latency_seconds"),
                    "attempts": review.get("attempts"),
                    "cache_reused": bool(review.get("cache_reused")),
                    "usage": review.get("usage") or {},
                }
                prior = visual_calls_by_fingerprint.get(fingerprint)
                if prior is None or (prior["cache_reused"] and not call["cache_reused"]):
                    visual_calls_by_fingerprint[fingerprint] = call

        group_calls = []
        seen_boundary_reviews = set()
        for group in groups:
            for review in group.boundary_reviews:
                identity = review.get("input_fingerprint")
                if identity in seen_boundary_reviews:
                    continue
                seen_boundary_reviews.add(identity)
                candidate = review.get("result") or {}
                if "usage" in candidate:
                    group_calls.append({
                        "stage": "experiment_boundary_review", "group_id": group.group_id,
                        **{key: candidate.get(key) for key in ("provider", "model", "api_protocol", "request_id", "response_model", "status", "latency_seconds", "attempts")},
                        "cache_reused": bool(candidate.get("cache_reused")),
                        "attempt_receipts": candidate.get("attempt_receipts", []),
                        "usage": candidate.get("usage", {}),
                    })
            understanding = group.model_understanding or {}
            for candidate in (understanding.get("operation_review") or {}).get("calls", []):
                group_calls.append({"stage": "operation_review", "group_id": group.group_id,
                    **{key: candidate.get(key) for key in ("provider", "model", "request_id", "response_model", "status", "latency_seconds", "attempts", "input_fingerprint")},
                    "cache_reused": bool(candidate.get("cache_reused")),
                    "attempt_receipts": candidate.get("attempt_receipts", []), "usage": candidate.get("usage", {})})
            candidates = [
                (
                    "experiment_group_understanding_pre_curation",
                    understanding.get("pre_curation_understanding") or {},
                ),
                ("experiment_group_understanding", understanding),
            ]
            if (
                understanding.get("refinement_pass") == "deterministic_post_event_semantic_curation"
                and understanding.get("refinement_model_call_count") == 0
                and understanding.get("pre_curation_understanding")
            ):
                # Deterministic curation preserves the original call receipt in
                # both views; it does not make or charge a second model call.
                candidates = candidates[:1]
            for stage_name, candidate in candidates:
                if not (
                    "usage" in candidate
                    or candidate.get("status") in {"completed", "failed"}
                ):
                    continue
                group_calls.append(
                    {
                        "stage": stage_name,
                        **{key: candidate.get(key) for key in ("provider", "api_protocol", "request_id", "response_model")},
                        "group_id": group.group_id,
                        "model": candidate.get(
                            "model", self.config["mllm"]["model"]
                        ),
                        "status": candidate.get("status"),
                        "latency_seconds": candidate.get("latency_seconds"),
                        "attempts": candidate.get("attempts"),
                        "cache_reused": bool(candidate.get("cache_reused")),
                        "attempt_receipts": candidate.get("attempt_receipts", []),
                        "usage": candidate.get("usage", {}),
                    }
                )

        from .stage_recovery import call_identity
        recovered_calls = getattr(getattr(self, "_stage_runner", None), "reused_call_ids", set())
        for call in key_calls + group_calls + list(visual_calls_by_fingerprint.values()):
            if call_identity(call) in recovered_calls:
                call.update(cache_reused=True, stage_checkpoint_reused=True)

        def token_sum(calls, field: str):
            values = [
                call["usage"].get(field)
                for call in calls
                if not call.get("cache_reused")
                and call.get("usage", {}).get(field) is not None
            ]
            return sum(values) if values else None

        key_materials = {
            "input_tokens": token_sum(key_calls, "input_tokens"),
            "output_tokens": token_sum(key_calls, "output_tokens"),
            "total_tokens": token_sum(key_calls, "total_tokens"),
            "cached_input_tokens": token_sum(key_calls, "cached_input_tokens"),
            "call_count": len(key_calls),
            "executed_call_count": sum(not call.get("cache_reused") for call in key_calls),
            "reused_call_count": sum(bool(call.get("cache_reused")) for call in key_calls),
            "server_reported_for_all_calls": bool(key_calls)
            and all(call.get("usage", {}).get("server_reported") for call in key_calls),
        }
        experiment_groups = {
            "input_tokens": token_sum(group_calls, "input_tokens"),
            "output_tokens": token_sum(group_calls, "output_tokens"),
            "total_tokens": token_sum(group_calls, "total_tokens"),
            "cached_input_tokens": token_sum(group_calls, "cached_input_tokens"),
            "call_count": len(group_calls),
            "executed_call_count": sum(not call.get("cache_reused") for call in group_calls),
            "reused_call_count": sum(bool(call.get("cache_reused")) for call in group_calls),
            "server_reported_for_all_calls": bool(group_calls)
            and all(call.get("usage", {}).get("server_reported") for call in group_calls),
        }
        visual_calls = list(visual_calls_by_fingerprint.values())
        visual_review_metrics = {
            "input_tokens": token_sum(visual_calls, "input_tokens"),
            "output_tokens": token_sum(visual_calls, "output_tokens"),
            "total_tokens": token_sum(visual_calls, "total_tokens"),
            "call_count": len(visual_calls),
            "executed_call_count": sum(not call["cache_reused"] for call in visual_calls),
            "reused_call_count": sum(call["cache_reused"] for call in visual_calls),
            "unknown_usage_call_count": sum(
                not call["cache_reused"] and call["usage"].get("total_tokens") is None
                for call in visual_calls
            ),
        }
        speech_calls = [
            {"stage": "recording_speech_understanding", **{
                key: part.get(key) for key in ("provider", "api_protocol", "request_id", "response_model", "status", "latency_seconds", "attempts", "input_fingerprint")
            }, "cache_reused": bool(part.get("cache_reused")), "usage": part.get("usage") or {}}
            for part in self._speech_understanding.get("parts", [])
        ]
        all_calls = group_calls + key_calls + visual_calls + speech_calls
        existing_calls = {call_identity(call) for call in all_calls}
        failed_calls = [call for call in getattr(self, "_retained_failed_calls", [])
                        if call_identity(call) not in existing_calls]
        failed_calls = [{**call, "retained_from_failed_stage": True} for call in failed_calls]
        all_calls.extend(failed_calls)
        return {
            "run_started_at": self._run_started_iso,
            "run_ended_at": datetime.now(timezone.utc).isoformat(),
            "total_duration_seconds": round(time.perf_counter() - self._run_started_perf, 6),
            "preprocessing_sla": {
                "definition": "probe + alignment + coarse scan + fine scan + boundary audit",
                "target_seconds": float(self.config["performance"]["preprocessing_budget_seconds"]),
                "actual_seconds": self._preprocessing_completed_seconds,
                "met": (
                    self._preprocessing_completed_seconds
                    <= float(self.config["performance"]["preprocessing_budget_seconds"])
                    if self._preprocessing_completed_seconds is not None
                    else None
                ),
            },
            "stage_durations": list(self._stage_metrics),
            "startup_durations": dict(self._startup_metrics),
            "tokens": {
                "experiment_groups": experiment_groups,
                "key_materials": key_materials,
                "participant_visual_review": visual_review_metrics,
                "recording_speech_understanding": {
                    "call_count": len(speech_calls),
                    "executed_call_count": sum(not call["cache_reused"] for call in speech_calls),
                    **{field: token_sum(speech_calls, field) for field in ("input_tokens", "output_tokens", "total_tokens")},
                },
                "daily_report": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                    "note": "Deterministic aggregation of already accepted evidence; no additional MLLM call.",
                },
                "run_total": {
                    "input_tokens": token_sum(all_calls, "input_tokens"),
                    "output_tokens": token_sum(all_calls, "output_tokens"),
                    "total_tokens": token_sum(all_calls, "total_tokens"),
                    "note": "MLLM step analysis and participant visual review consume model tokens; CV and FFmpeg consume no tokens.",
                },
                "retained_failed_stage_calls": {
                    "call_count": len(failed_calls),
                    **{field: token_sum(failed_calls, field) for field in ("input_tokens", "output_tokens", "total_tokens")},
                },
            },
            "mllm_calls": all_calls,
            "mllm_request_policy": getattr(self, "_mllm_timeout_policy", None),
            "performance_mode": {
                "concurrent_input_views": self._input_view_count,
                "concurrent_role_scanners": bool(
                    self.config["performance"].get("concurrent_role_scanners", True)
                ),
                "bounded_streaming": True,
                "gpu_batching": True,
                "input_mode": self._input_mode,
                "source_copy_bytes": 0 if self._input_mode == "segmented_virtual_timeline" else None,
                "runtime_evidence": {
                    "resource_telemetry": "JSON-Config-Files/resource_telemetry.json",
                    "coarse_scan": "JSON-Config-Files/scan_runtime_coarse.json",
                    "fine_scan": "JSON-Config-Files/scan_runtime_fine.json",
                },
            },
        }

    def _scan_all_views_concurrently(
        self,
        manifest,
        infos,
        transforms,
        work_dir,
        *,
        windows=None,
        sample_fps=None,
        image_size=None,
        keyframes_only=False,
        phase="fine",
        decode_backends: dict[str, str] | None = None,
    ):
        from .scan_scheduler import scan_views_concurrently
        return scan_views_concurrently(
            self.config, manifest.views, infos, transforms, work_dir,
            windows=windows, sample_fps=sample_fps, image_size=image_size,
            keyframes_only=keyframes_only, phase=phase, decode_backends=decode_backends,
            progress_callback=self._scan_progress, view_runtime=self._view_runtime,
            scanner=scan_videos,
        )

    def _motion_probe_views(self, manifest: RunManifest) -> list[ViewInput]:
        """Choose sentinel views; bounded YOLO scans still use all eligible views."""

        perf = self.config["performance"]
        if perf.get("motion_probe_all_views", False):
            return list(manifest.views)
        first_limit = max(1, int(perf.get("motion_probe_first_person_views", 1)))
        third_limit = max(0, int(perf.get("motion_probe_third_person_views", 1)))
        first = [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON]
        third = [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON]
        return first[:first_limit] + third[:third_limit]

    def _coarse_scan_views(self, manifest: RunManifest) -> list[ViewInput]:
        """Choose boundary sentinels; fine validation still uses required views."""

        perf = self.config["performance"]
        if perf.get("coarse_all_views", False):
            return list(manifest.views)
        first_limit = max(1, int(perf.get("coarse_first_person_views", 1)))
        third_limit = max(1, int(perf.get("coarse_third_person_views", 1)))
        first = [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON]
        third = [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON]
        return first[:first_limit] + third[:third_limit]

    _progressive_fine_view_order = staticmethod(FineScanPlanner.progressive_fine_view_order)

    _progressive_candidate_recall_bounds = staticmethod(FineScanPlanner.progressive_candidate_recall_bounds)

    _quarantine_nonformal_progressive_gaps = staticmethod(FineScanPlanner.quarantine_nonformal_progressive_gaps)

    def _progressive_target_status(
        self, boundary_candidates: list[ActionCandidate], events: list[EvidenceEvent]
    ) -> list[dict[str, Any]]:
        return FineScanPlanner(self.config).progressive_target_status(
            boundary_candidates, events
        )


    _progressive_anchor_windows = staticmethod(FineScanPlanner.progressive_anchor_windows)

    _progressive_scout_anchor_representatives = staticmethod(FineScanPlanner.progressive_scout_anchor_representatives)

    _merge_time_windows = staticmethod(FineScanPlanner.merge_time_windows)

    _window_fully_covered = staticmethod(FineScanPlanner.window_fully_covered)

    _uncovered_time_windows = staticmethod(FineScanPlanner.uncovered_time_windows)

    def _group_local_recall_plan(
        self,
        groups: list[ExperimentGroup],
        segments: list[ExperimentSegment],
        events: list[EvidenceEvent],
        candidates: list[ActionCandidate],
        fine_views: list[ViewInput],
        fine_view_report: dict[str, dict[str, Any]],
        actual_windows: dict[str, list[tuple[float, float]]],
        infos,
        transforms,
        boundary_candidates: list[ActionCandidate] | None = None,
    ) -> dict[str, Any]:
        return FineScanPlanner(self.config).group_local_recall_plan(
            groups,
            segments,
            events,
            candidates,
            fine_views,
            fine_view_report,
            actual_windows,
            infos,
            transforms,
            boundary_candidates,
        )


    def _run_progressive_fine_scan(
        self,
        manifest: RunManifest,
        fine_views: list[ViewInput],
        fine_view_report: dict[str, dict[str, Any]],
        boundary_candidates: list[ActionCandidate],
        fine_windows: dict[str, list[tuple[float, float]]],
        infos,
        transforms,
        work_dir: Path,
    ) -> tuple[dict[str, Path], list[ViewInput], list[ActionCandidate], dict[str, Any]]:
        return ProgressiveScanExecutor(
            self.config,
            planner=FineScanPlanner(self.config),
            ports=ScanExecutionPorts(
                self._scan_all_views_concurrently,
                self._frame_index_path,
                self._fine_windows,
            ),
            operations=ScanAlgorithms(
                merge_frame_evidence_ledgers=_merge_frame_evidence_ledgers,
                attach_action_observability=attach_action_observability,
                audit_candidates=audit_candidates,
                build_experiment_groups=build_experiment_groups,
                build_experiment_segments=build_experiment_segments,
                create_fine_frame_index=create_fine_frame_index,
                fine_frame_coverage_report=fine_frame_coverage_report,
                generate_candidates=generate_candidates,
                ingest_fine_frame_ledgers=ingest_fine_frame_ledgers,
                normalize_experiment_segments=normalize_experiment_segments,
                prepare_formal_experiment_segments=prepare_formal_experiment_segments,
                refine_liquid_events_with_context=refine_liquid_events_with_context,
                select_fine_scan_views=select_fine_scan_views,
                select_key_events=select_key_events,
            ),
        ).run(
            manifest,
            fine_views,
            fine_view_report,
            boundary_candidates,
            fine_windows,
            infos,
            transforms,
            work_dir,
        )


    _window_coverage = staticmethod(FineScanPlanner.window_coverage)

    @staticmethod
    def _input_volume_report(manifest: RunManifest, infos) -> dict[str, Any]:
        listed_video_paths = [
            os.path.abspath(path)
            for view in manifest.views
            for path in view_source_files(view)
        ]
        listed_clock_paths = [
            os.path.abspath(path)
            for view in manifest.views
            for path in view_timestamp_files(view)
        ]
        views = []
        for view in manifest.views:
            info = infos[view.view_id]
            duration_seconds = info.duration_ms / 1000.0
            views.append(
                {
                    "view_id": view.view_id,
                    "role": view.role.value,
                    "segment_count": len(view.segments) if view.segments else 1,
                    "width": info.width,
                    "height": info.height,
                    "fps": info.fps,
                    "duration_seconds": round(duration_seconds, 3),
                    "video_bytes": info.size_bytes,
                    "video_gib": round(info.size_bytes / 1024**3, 3),
                    "estimated_video_mbps": (
                        round(info.size_bytes * 8.0 / duration_seconds / 1_000_000.0, 3)
                        if duration_seconds > 0
                        else 0.0
                    ),
                }
            )
        total_video_bytes = sum(int(item["video_bytes"]) for item in views)
        return {
            "schema_version": "visioncortex-input-volume/1",
            "listed_video_path_count": len(listed_video_paths),
            "unique_video_path_count": len(set(listed_video_paths)),
            "duplicate_video_path_count": len(listed_video_paths) - len(set(listed_video_paths)),
            "listed_clock_path_count": len(listed_clock_paths),
            "unique_clock_path_count": len(set(listed_clock_paths)),
            "total_video_bytes": total_video_bytes,
            "total_video_gib": round(total_video_bytes / 1024**3, 3),
            "total_video_gb_decimal": round(total_video_bytes / 1_000_000_000.0, 3),
            "views": views,
        }

    @staticmethod
    def _archive_scan_runtime(
        layout: ArchiveLayout,
        work_dir: Path,
        phase: str,
        progressive_report: dict[str, Any] | None = None,
    ) -> None:
        role_reports = []
        role_values = {role.value for role in ViewRole}

        def scanner_role(value: str) -> str | None:
            return next(
                (
                    role
                    for role in role_values
                    if value == role or value.startswith(f"{role}_")
                ),
                None,
            )

        for path in sorted(work_dir.rglob(f"runtime_{phase}_*.json")):
            runtime_role = path.stem.removeprefix(f"runtime_{phase}_")
            if scanner_role(runtime_role) is None:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            report["scan_pass"] = str(path.parent.relative_to(work_dir)).replace("\\", "/")
            role_reports.append(report)
        source_activity = []
        for path in sorted(work_dir.rglob(f"source_activity_{phase}_*.jsonl")):
            activity_role = path.stem.removeprefix(f"source_activity_{phase}_")
            if scanner_role(activity_role) is None:
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("phase") not in {None, phase}:
                        continue
                    item["scan_pass"] = str(path.parent.relative_to(work_dir)).replace(
                        "\\", "/"
                    )
                    source_activity.append(item)
        computed_units = sum(
            item.get("event") == "source_unit_completed" for item in source_activity
        )
        reused_units = sum(
            item.get("event") == "source_unit_reused" for item in source_activity
        )
        started_units = sum(
            item.get("event") == "source_unit_started" for item in source_activity
        )
        decoder_receipts = [
            item["decoder_receipt"]
            for item in source_activity
            if item.get("event") == "source_unit_completed"
            and isinstance(item.get("decoder_receipt"), dict)
        ]
        frame_accounting = {
            "session_count": len(decoder_receipts),
            "exact_session_count": sum(
                int(receipt.get("frame_accounting_mismatch") or 0) == 0
                and not bool(receipt.get("frame_accounting_reconciled"))
                for receipt in decoder_receipts
            ),
            "reconciled_terminal_eof_session_count": sum(
                bool(receipt.get("frame_accounting_reconciled"))
                and int(receipt.get("terminal_eof_shortfall_frames") or 0) > 0
                for receipt in decoder_receipts
            ),
            "unreconciled_mismatch_session_count": sum(
                int(receipt.get("frame_accounting_mismatch") or 0) != 0
                and not bool(receipt.get("frame_accounting_reconciled"))
                for receipt in decoder_receipts
            ),
        }
        scheduler_reports = []
        for path in sorted(work_dir.rglob(f"scheduler_{phase}.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("phase") not in {None, phase}:
                continue
            report["scan_pass"] = str(path.parent.relative_to(work_dir)).replace("\\", "/")
            scheduler_reports.append(report)
        if progressive_report is not None:
            scheduler = {
                "schema_version": "visioncortex-role-scheduler/2",
                "phase": phase,
                "mode": "progressive_cross_view",
                "passes": scheduler_reports,
            }
        elif len(scheduler_reports) == 1:
            scheduler = scheduler_reports[0]
        elif scheduler_reports:
            scheduler = {
                "schema_version": "visioncortex-role-scheduler/2",
                "phase": phase,
                "mode": "progressive_cross_view",
                "passes": scheduler_reports,
            }
        else:
            scheduler = None
        inference_calls = sum(int(item.get("inference_call_count", 0)) for item in role_reports)
        inference_frames = sum(int(item.get("inference_frame_count", 0)) for item in role_reports)
        queue_wait_seconds = sum(float(item.get("queue_wait_seconds", 0.0)) for item in role_reports)
        inference_seconds = sum(float(item.get("inference_seconds", 0.0)) for item in role_reports)
        postprocess_seconds = sum(
            float(item.get("tracking_and_ledger_seconds", 0.0)) for item in role_reports
        )
        role_seconds = [float(item.get("role_total_seconds", 0.0)) for item in role_reports]
        observed_seconds = sum(role_seconds)
        actual_batch_mean = inference_frames / inference_calls if inference_calls else 0.0
        effective_batch_slots = sum(
            int(item.get("inference_call_count", 0))
            * int(item.get("final_effective_batch_size", 0))
            for item in role_reports
        )
        batch_fill_ratio = inference_frames / effective_batch_slots if effective_batch_slots else 0.0
        effective_batch_capacity = (
            effective_batch_slots / inference_calls if inference_calls else 0.0
        )
        queue_wait_ratio = queue_wait_seconds / observed_seconds if observed_seconds else 0.0
        inference_ratio = inference_seconds / observed_seconds if observed_seconds else 0.0
        postprocess_ratio = postprocess_seconds / observed_seconds if observed_seconds else 0.0
        inference_frames_per_second = (
            inference_frames / inference_seconds if inference_seconds else 0.0
        )
        inference_milliseconds_per_call = (
            inference_seconds * 1000.0 / inference_calls if inference_calls else 0.0
        )
        if batch_fill_ratio < 0.60 and queue_wait_ratio >= 0.25:
            bottleneck = "decode_or_source_starved"
            next_action = "increase ordered decode supply before adding YOLO contexts"
        elif inference_ratio >= 0.65:
            bottleneck = "gpu_inference_bound"
            next_action = "reduce selected frames/windows or benchmark a faster engine"
        elif postprocess_ratio >= 0.40:
            bottleneck = "cpu_postprocess_bound"
            next_action = "parallelize tracking and ledger serialization"
        else:
            bottleneck = "mixed_or_balanced"
            next_action = "use per-role telemetry before changing concurrency"
        component_reports = [r["component_timings"] for r in role_reports if r.get("component_timings")]
        component_keys = {k for r in component_reports for k, v in r.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)}
        components = {k: round(sum(r.get(k, 0) for r in component_reports), 6) for k in component_keys}
        bottleneck_diagnosis = {
            "component_timings": components,
            "component_profiled_workers": len(component_reports),
            "component_expected_workers": len(role_reports),
            "classification": bottleneck,
            "next_action": next_action,
            "observed_role_seconds": round(observed_seconds, 6),
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "tracking_and_ledger_seconds": round(postprocess_seconds, 6),
            "queue_wait_ratio": round(queue_wait_ratio, 6),
            "inference_ratio": round(inference_ratio, 6),
            "tracking_and_ledger_ratio": round(postprocess_ratio, 6),
            "inference_call_count": inference_calls,
            "inference_frame_count": inference_frames,
            "inference_frames_per_second": round(inference_frames_per_second, 3),
            "inference_milliseconds_per_call": round(
                inference_milliseconds_per_call, 3
            ),
            "actual_batch_size_mean": round(actual_batch_mean, 4),
            "effective_batch_capacity_mean": round(effective_batch_capacity, 4),
            "batch_fill_ratio": round(batch_fill_ratio, 6),
        }
        write_json(
            layout.json_config / f"scan_runtime_{phase}.json",
            {
                "schema_version": "visioncortex-scan-runtime/1",
                "phase": phase,
                "cold_start": reused_units == 0,
                "work_units": {
                    "started": started_units,
                    "computed": computed_units,
                    "reused": reused_units,
                },
                "role_reports": role_reports,
                "scheduler": scheduler,
                "progressive_cross_view": progressive_report,
                "frame_accounting": frame_accounting,
                "bottleneck_diagnosis": bottleneck_diagnosis,
                "source_activity": sorted(
                    source_activity, key=lambda item: float(item.get("timestamp", 0.0))
                ),
            },
        )

    @analysis_execution
    def run(self, manifest: RunManifest) -> Path:
        from .config import require_configured_site
        require_configured_site(self.config)
        from .stage_execution import StageExecutor
        stage_executor = StageExecutor(self.config)
        self._stage_executor = stage_executor
        self._speech_future = None
        self._run_started_perf = time.perf_counter()
        self._run_started_iso = datetime.now(timezone.utc).isoformat()
        self._active_stage = None
        self._stage_metrics = []
        self._stage_outcomes = {}
        self._retained_failed_calls = []
        self._failed_stage_metrics = None
        self._startup_metrics = {}
        self._preprocessing_completed_seconds = None
        self._current_experiment_id = manifest.experiment_id
        self._acceptance_baseline_selection = {
            "configured": False,
            "applied": False,
            "reason": "not_evaluated",
            "current_experiment_id": manifest.experiment_id,
        }
        self._input_view_count = len(manifest.views)
        self._input_mode = (
            "segmented_virtual_timeline"
            if all(view.segments for view in manifest.views)
            else "continuous_file"
        )
        lanes = list(self.config["performance"].get("coarse_decode_lanes") or [])
        if not lanes:
            lanes = [
                "cuda" if self.config["performance"].get("ffmpeg_hwaccel") else "cpu"
            ] * len(manifest.views)
        if len(lanes) < len(manifest.views):
            lanes.extend([lanes[-1]] * (len(manifest.views) - len(lanes)))
        self._view_runtime = {
            view.view_id: {
                "role": view.role.value,
                "segment_count": len(view.segments) if view.segments else 1,
                "decode_backend": lanes[index],
                "state": "registered",
            }
            for index, view in enumerate(manifest.views)
        }
        storage = self.config.get("storage", {})
        if storage.get("run_output_mode") == "nas_direct":
            active_archive = storage.get("active_archive_path")
            if not active_archive:
                raise ValueError("nas_direct output requires storage.active_archive_path")
            # Keep a mapped-drive path mapped. Path.resolve() expands Y: into a
            # longer UNC path on Windows and can push otherwise valid artifact
            # names beyond MAX_PATH when LongPathsEnabled is disabled.
            layout = ArchiveLayout(Path(os.path.abspath(str(active_archive))))
        else:
            output_root = Path(self.config["project"]["output_root"]).resolve()
            layout = ArchiveLayout(output_root / manifest.experiment_id)
        cache_identity_started = time.perf_counter()
        cache_identity = build_cache_identity(self.config, manifest)
        self._startup_metrics["cache_identity_seconds"] = round(
            time.perf_counter() - cache_identity_started,
            6,
        )
        self._startup_metrics["cache_identity_source_snapshot"] = cache_identity.get(
            "source_snapshot_report", {}
        )
        from .cache_paths import model_cache_directory, windows_path
        local_cache = Path(os.path.abspath(self.config["storage"]["local_cache_root"]))
        layout.work = (model_cache_directory(local_cache, "work", cache_identity["cache_key"])
                       if windows_path(local_cache) else
                       local_cache.resolve() / manifest.experiment_id / cache_identity["cache_key"])
        layout.create()
        self._active_layout = layout
        write_json(layout.json_config / "cache_identity.json", cache_identity)
        if storage.get("sync_to_nas") and storage.get("run_output_mode") != "nas_direct":
            from .archive_delivery import DeferredArchivePublisher
            from .storage import safe_archive_name
            try:
                nas_root = initialize_nas_archive(self.config, manifest.experiment_id)
            except OSError:
                nas_root = Path(storage["active_archive_path"]) if storage.get("active_archive_path") else Path(storage["archive_root"]) / safe_archive_name(manifest.experiment_id)
            self._publisher = DeferredArchivePublisher(layout.root, nas_root)
        else:
            self._publisher = None
        lock_path = layout.root / "run.lock"
        self._acquire_lock(lock_path)
        self._resource_monitor = ResourceMonitor(
            layout.json_config / "resource_telemetry.json",
            float(self.config.get("resource_limits", {}).get("telemetry_interval_seconds", 1.0)),
            (
                self._publisher.nas_root
                / "JSON-Config-Files"
                / "resource_telemetry_live.json"
                if self._publisher is not None and not hasattr(self._publisher, "pending")
                else None
            ),
        )
        self._resource_monitor.start()
        try:
            from .stage_recovery import StageRunner
            runner = StageRunner(self, layout, manifest, cache_identity)
            self._stage_runner = runner
            return runner.run()
        except Exception as exc:
            self._status(layout, "failed", 1.0, f"{type(exc).__name__}: {exc}")
            partial_metrics = self._metrics(
                getattr(getattr(getattr(self, "_stage_runner", None), "context", None), "events", []),
                getattr(getattr(getattr(self, "_stage_runner", None), "context", None), "groups", [])
            )
            write_json(
                layout.json_config / "run_metrics.json",
                partial_metrics,
            )
            try:
                write_partial_delivery(layout.root, partial_metrics)
                if self._publisher is not None:
                    self._publisher.publish_directory("Partial-Results")
            except (OSError, ValueError, KeyError, TypeError) as report_exc:
                # Retain the original failure and existing media even if a
                # damaged control file prevents the supplemental report.
                write_json(layout.json_config / "partial_delivery_error.json", {
                    "status": "failed",
                    "exception_class": type(report_exc).__name__,
                })
            if self._publisher is not None:
                self._publisher.publish_directory("JSON-Config-Files")
            raise
        finally:
            stage_executor.close()
            if self._resource_monitor is not None:
                self._resource_monitor.stop()
                if self._publisher is not None:
                    self._publisher.publish_file(layout.json_config / "resource_telemetry.json")
            # A stale run lock blocks all future processing for the same
            # experiment. Cleanup failure is therefore a pipeline failure, not
            # an ignorable housekeeping warning.
            lock_path.unlink(missing_ok=True)

    def _stage_preflight(self, c, layout, manifest):
        self._status(layout, "preflight", 0.02, "检查输入、模型、视频与磁盘")
        preflight_breakdown: dict[str, Any] = {}
        preflight_step_started = time.perf_counter()
        source_paths = [
            path
            for view in manifest.views
            for path in view_source_files(view) + view_timestamp_files(view) + speech.audio_files(view)
        ]
        source_snapshots, source_validation = snapshot_source_paths(
            source_paths,
            workers=int(self.config["performance"].get("source_stat_workers", 24)),
            max_age_seconds=float(
                self.config["performance"].get(
                    "source_stat_cache_ttl_seconds", 120.0
                )
            ),
        )
        missing_sources = [
            str(path)
            for path, snapshot in source_snapshots.items()
            if not snapshot["is_file"]
        ]
        source_validation["cache_diagnostics"] = source_cache_diagnostics()
        write_json(
            layout.json_config / "source_validation.json",
            source_validation,
        )
        if missing_sources:
            raise FileNotFoundError(f"输入源文件不存在: {missing_sources[:4]}")
        preflight_breakdown["source_validation_seconds"] = round(
            time.perf_counter() - preflight_step_started,
            6,
        )
        preflight_breakdown["source_validation"] = source_validation
        preflight_step_started = time.perf_counter()
        model_report = validate_models(self.config)
        self._model_certification_audit = (
            audit_production_model_certification(self.config)
        )
        write_json(
            layout.json_config / "model_certification_audit.json",
            self._model_certification_audit,
        )
        model_report["video_encoder"] = video_encoder_preflight(
            str(self.config["performance"].get("ffmpeg_video_encoder", "h264_nvenc"))
        )
        preflight_breakdown["model_validation_seconds"] = round(
            time.perf_counter() - preflight_step_started,
            6,
        )
        write_json(layout.json_config / "model_runtime_preflight.json", model_report)
        preflight_step_started = time.perf_counter()
        c.infos = probe_views(
            manifest.views,
            workers=int(self.config["performance"].get("preflight_probe_workers", 12)),
            prefer_clock_metadata=bool(
                self.config["performance"].get(
                    "preflight_prefer_clock_metadata", True
                )
            ),
        )
        preflight_breakdown["video_probe_seconds"] = round(
            time.perf_counter() - preflight_step_started,
            6,
        )
        write_json(
            layout.json_config / "input_volume_report.json",
            self._input_volume_report(manifest, c.infos),
        )
        preflight_step_started = time.perf_counter()
        c.disk_report = check_disk_capacity(layout.root, list(c.infos.values()))
        preflight_breakdown["disk_check_seconds"] = round(
            time.perf_counter() - preflight_step_started,
            6,
        )
        media_preflight_path = layout.json_config / "media_pipeline_preflight.json"
        if bool(
            self.config["performance"].get(
                "media_pipeline_preflight_enabled", True
            )
        ):
            preflight_step_started = time.perf_counter()
            smoke_root = layout.work / "media-pipeline-preflight"
            try:
                media_preflight = run_media_pipeline_preflight(
                    manifest,
                    c.infos,
                    layout.work,
                    str(model_report["video_encoder"]["selected_encoder"]),
                    float(
                        self.config["performance"].get(
                            "media_pipeline_preflight_seconds", 1.0
                        )
                    ),
                )
            except Exception as exc:
                write_json(
                    media_preflight_path,
                    {
                        "schema_version": "visioncortex-media-pipeline-preflight/1",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "temporary_artifacts_retained": smoke_root.exists(),
                        "temporary_artifact_path": str(smoke_root),
                        "elapsed_seconds": round(
                            time.perf_counter() - preflight_step_started, 6
                        ),
                    },
                )
                raise RuntimeError(
                    "media pipeline preflight failed before long-running scans"
                ) from exc
            write_json(media_preflight_path, media_preflight)
            preflight_breakdown["media_pipeline_preflight_seconds"] = round(
                time.perf_counter() - preflight_step_started,
                6,
            )
        preflight_breakdown["source_cache_after_probe"] = source_cache_diagnostics()
        write_json(
            layout.json_config / "preflight_runtime.json",
            preflight_breakdown,
        )
        write_json(layout.json_config / "video_probe.json", {key: value.model_dump(mode="json") for key, value in c.infos.items()})
        self._complete_stage(
            layout,
            "preflight",
            [
                layout.json_config / "source_validation.json",
                layout.json_config / "model_runtime_preflight.json",
                layout.json_config / "input_volume_report.json",
                layout.json_config / "preflight_runtime.json",
                layout.json_config / "video_probe.json",
                *([media_preflight_path] if media_preflight_path.exists() else []),
            ],
        )


    def _stage_capture_quality(self, c, layout, manifest):
        from .capture_quality import inspect as inspect_capture
        write_json(layout.json_config / "capture_quality.json", inspect_capture(manifest, self.config))
        self._complete_stage(layout, "capture_quality", [layout.json_config / "capture_quality.json"])

    def _stage_alignment(self, c, layout, manifest):
        self._status(layout, "alignment", 0.08, "最近邻时间戳拟合与视觉锚点校准")
        alignment_runtime: dict[str, Any] = {}
        alignment_step_started = time.perf_counter()
        c.transforms, _ = self._stage_executor.run(
            "alignment", build_alignments, manifest.views, c.infos, self.config
        )
        alignment_runtime["fit_seconds"] = round(
            time.perf_counter() - alignment_step_started,
            6,
        )
        alignment_runtime["view_runtime"] = {
            view_id: transform.runtime
            for view_id, transform in c.transforms.items()
        }
        write_json(
            layout.json_config / "time_alignment.json",
            [transform.model_dump(mode="json") for transform in c.transforms.values()],
        )
        alignment_gate = alignment_quality_report(
            manifest.views,
            c.infos,
            c.transforms,
            self.config,
        )
        write_json(
            layout.json_config / "alignment_quality_gate.json",
            alignment_gate,
        )
        alignment_step_started = time.perf_counter()
        write_aligned_csv(
            layout.json_config / "aligned_timestamps.csv",
            manifest.views,
            c.infos,
            c.transforms,
            float(self.config["alignment"]["aligned_timestamps_fps"]),
        )
        alignment_runtime["aligned_csv_seconds"] = round(
            time.perf_counter() - alignment_step_started,
            6,
        )
        alignment_runtime["source_cache_after_alignment"] = source_cache_diagnostics()
        write_json(
            layout.json_config / "alignment_runtime.json",
            alignment_runtime,
        )
        if (
            bool(self.config["alignment"].get("quality_gate_enabled", True))
            and not bool(alignment_gate["formal_evidence_ready"])
        ):
            raise RuntimeError(
                "alignment quality gate failed before GPU scans: "
                + "; ".join(str(item) for item in alignment_gate["errors"])
            )
        self._complete_stage(
            layout,
            "alignment",
            [
                layout.json_config / "time_alignment.json",
                layout.json_config / "aligned_timestamps.csv",
                layout.json_config / "alignment_runtime.json",
                layout.json_config / "alignment_quality_gate.json",
            ],
        )


    def _stage_speech(self, c, layout, manifest):
        self._status(layout, "speech", 0.09, "整理实验录音与转写")
        speech_status, speech_reason = "completed", None
        try:
            future = getattr(self, "_speech_future", None)
            speech_result, duration = (future.result() if future is not None else
                self._run_speech_stage(layout, manifest, c.infos, c.transforms))
            self._stage_metrics.append({"stage": "speech", "duration_seconds": duration,
                                        "execution": "parallel_with_video" if future else "inline"})
            if not speech.enabled(self.config):
                if speech_result.get("archive_status") == "saved":
                    speech_status, speech_reason = "completed", "原始录音已保存；未启用文字转写，视频分析继续"
                else:
                    speech_status, speech_reason = "skipped", "未启用录音转写，视频分析继续"
            elif not any(source.get("available") for source in speech_result.get("sources", [])):
                speech_status, speech_reason = "skipped", "没有可用录音，视频分析继续"
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if self.config.get("speech_recognition", {}).get("required", False):
                raise
            speech_status, speech_reason = "failed", "录音处理未完成，已保留记录；视频分析继续"
            # Exclude partially written transcripts from model context.
            speech_path = layout.json_config / "speech.json"
            try:
                previous = json.loads(speech_path.read_text(encoding="utf-8")) if speech_path.is_file() else {}
            except (OSError, ValueError):
                previous = {}
            if not isinstance(previous, dict):
                previous = {}
            write_json(speech_path, {**previous, "status": "failed", "optional": True,
                                    "error_type": type(exc).__name__, "message": speech_reason})
        self._complete_stage(layout, "speech", [
            layout.json_config / "speech.json",
            *[layout.json_config / name for name in ("speech_search.json", "speech_search_receipt.json") if (layout.json_config / name).is_file()],
            *([layout.json_config / "speech_timeline.json"] if speech_status == "completed" and (layout.json_config / "speech_timeline.json").is_file() else []),
            *([layout.key_materials / "Experiment-Audio"] if (layout.key_materials / "Experiment-Audio").is_dir() else []),
        ], status=speech_status, reason=speech_reason)


    def _stage_motion_probe(self, c, layout, manifest):
        c.motion_probe_views = self._motion_probe_views(manifest)
        c.preselected_coarse_views = self._coarse_scan_views(manifest)
        c.coarse_full_timeline = bool(
            self.config["performance"].get(
                "coarse_full_timeline_scan", False
            )
        )
        c.shared_coarse_motion_scan = bool(
            self.config["performance"].get(
                "coarse_shared_motion_probe_enabled", False
            )
            and c.coarse_full_timeline
            and {
                view.view_id for view in c.motion_probe_views
            }
            == {view.view_id for view in c.preselected_coarse_views}
            and float(
                self.config["performance"]["coarse_detection_fps"]
            )
            >= float(self.config["performance"]["motion_probe_fps"])
            and int(self.config["performance"]["coarse_image_size"])
            >= int(
                self.config["performance"].get(
                    "motion_probe_max_width", 96
                )
            )
            and not bool(
                self.config["performance"].get(
                    "coarse_keyframes_only", False
                )
            )
        )
        c.coarse_frame_index: CoarseFrameIndex | None = None
        c.coarse_index_report: dict[str, Any] | None = None

        probe_manifest = manifest.model_copy(update={"views": c.motion_probe_views})
        selected_probe_ids = {view.view_id for view in c.motion_probe_views}
        for view in manifest.views:
            self._view_runtime[view.view_id]["state"] = (
                "motion_probe_running"
                if view.view_id in selected_probe_ids
                else "motion_probe_sentinel_not_selected"
            )
        self._status(
            layout,
            "motion_probe",
            0.12,
            (
                "共享全时间轴粗扫解码并生成运动探针，不重复读取原视频"
                if c.shared_coarse_motion_scan
                else "Selecting and running the fastest real-source sparse motion probe"
            ),
        )
        sparse_strategy_path = layout.json_config / "motion_probe_sparse_strategy.json"
        configured_sparse_strategy = str(
            self.config["performance"].get(
                "motion_probe_sparse_strategy", "indexed_seek"
            )
        )
        if configured_sparse_strategy == "auto":
            sentinel = c.motion_probe_views[0]
            from .cuda_decode_admission import CudaDecodeAdmission
            sparse_report = benchmark_sparse_decode_strategy(
                sentinel,
                c.infos[sentinel.view_id],
                sample_fps=float(self.config["performance"]["motion_probe_fps"]),
                max_width=int(
                    self.config["performance"].get("motion_probe_max_width", 96)
                ),
                hwaccel=(
                    str(self.config["performance"].get("ffmpeg_hwaccel"))
                    if self.config["performance"].get("ffmpeg_hwaccel")
                    else None
                ),
                decoder_threads=int(
                    self.config["performance"].get("cpu_decode_threads", 0)
                ),
                cuda_scale=bool(
                    self.config["performance"].get("ffmpeg_cuda_scale", False)
                ),
                benchmark_seconds=float(
                    self.config["performance"].get(
                        "motion_probe_sparse_benchmark_seconds", 60.0
                    )
                ),
                decoder_admission=CudaDecodeAdmission.from_config(self.config),
            )
            selected_sparse_strategy = str(sparse_report["selected_strategy"])
        else:
            if configured_sparse_strategy not in {
                "indexed_seek",
                "sequential_keyframes",
            }:
                raise ValueError(
                    "motion_probe_sparse_strategy must be auto, indexed_seek, "
                    "or sequential_keyframes"
                )
            selected_sparse_strategy = configured_sparse_strategy
            sparse_report = {
                "schema_version": "visioncortex-sparse-decode-benchmark/1",
                "selected_strategy": selected_sparse_strategy,
                "configured_strategy": configured_sparse_strategy,
                "benchmark_skipped": True,
            }
        self.config["performance"][
            "motion_probe_sparse_strategy"
        ] = selected_sparse_strategy
        if configured_sparse_strategy == "auto":
            worker_key = (
                "motion_probe_indexed_segment_workers"
                if selected_sparse_strategy == "indexed_seek"
                else "motion_probe_sequential_segment_workers"
            )
            selected_segment_workers = max(
                1,
                int(
                    self.config["performance"].get(
                        worker_key,
                        self.config["performance"].get(
                            "motion_probe_segment_workers", 1
                        ),
                    )
                ),
            )
        else:
            selected_segment_workers = max(
                1,
                int(
                    self.config["performance"].get(
                        "motion_probe_segment_workers", 1
                    )
                ),
            )
        self.config["performance"][
            "motion_probe_segment_workers"
        ] = selected_segment_workers
        sparse_report["selected_segment_workers"] = selected_segment_workers
        sparse_report["selection_reason"] = (
            "short real-source benchmark"
            if configured_sparse_strategy == "auto"
            else "explicit configuration"
        )
        sparse_report["shared_with_full_timeline_coarse_scan"] = (
            c.shared_coarse_motion_scan
        )
        write_json(sparse_strategy_path, sparse_report)
        selected_probe_ids = {view.view_id for view in c.motion_probe_views}
        for view in manifest.views:
            self._view_runtime[view.view_id]["state"] = (
                "motion_probe_running"
                if view.view_id in selected_probe_ids
                else "motion_probe_sentinel_not_selected"
            )
        self._status(
            layout,
            "motion_probe",
            0.12,
            (
                "全路全时间轴粗扫共享同一次解码，并同步生成0.5 FPS运动探针"
                if c.shared_coarse_motion_scan
                else "哨兵视角低分辨率运动探针；此阶段CUDA计算低占用属于预期"
            ),
        )
        if c.shared_coarse_motion_scan:
            shared_manifest = manifest.model_copy(
                update={"views": c.preselected_coarse_views}
            )
            c.motion_paths = self._scan_all_views_concurrently(
                shared_manifest,
                c.infos,
                c.transforms,
                layout.work / "detections-coarse",
                sample_fps=float(
                    self.config["performance"]["coarse_detection_fps"]
                ),
                image_size=int(
                    self.config["performance"]["coarse_image_size"]
                ),
                keyframes_only=bool(
                    self.config["performance"]["coarse_keyframes_only"]
                ),
                phase="coarse",
            )
            self._archive_scan_runtime(
                layout, layout.work / "detections-coarse", "coarse"
            )
            shared_runtime = json.loads(
                (layout.json_config / "scan_runtime_coarse.json").read_text(
                    encoding="utf-8"
                )
            )
            write_json(
                layout.json_config / "scan_runtime_motion_probe.json",
                {
                    "schema_version": "visioncortex-shared-motion-probe-runtime/1",
                    "phase": "motion_probe",
                    "shared_source_phase": "coarse",
                    "additional_video_decode_bytes": 0,
                    "logical_sample_fps": float(
                        self.config["performance"]["motion_probe_fps"]
                    ),
                    "role_motion_scoring": [
                        {
                            "role": report.get("role"),
                            "motion_scoring": report.get("motion_scoring"),
                        }
                        for report in shared_runtime.get("role_reports", [])
                    ],
                },
            )
            self._ensure_coarse_frame_index(c, layout, c.infos,
                c.motion_paths, c.preselected_coarse_views
            )
        else:
            c.motion_paths = self._scan_all_views_concurrently(
                probe_manifest,
                c.infos,
                c.transforms,
                layout.work / "motion-probe",
                sample_fps=float(
                    self.config["performance"]["motion_probe_fps"]
                ),
                image_size=int(
                    self.config["performance"].get(
                        "motion_probe_max_width", 96
                    )
                ),
                keyframes_only=bool(
                    self.config["performance"].get(
                        "motion_probe_keyframes_only", True
                    )
                ),
                phase="motion_probe",
            )
            self._archive_scan_runtime(
                layout, layout.work / "motion-probe", "motion_probe"
            )
        probe_config = json.loads(json.dumps(self.config))
        probe_config["performance"]["motion_probe_require_objects"] = False
        probe_config["performance"][
            "motion_probe_use_embedded_coarse_scores"
        ] = c.shared_coarse_motion_scan
        probe_config["performance"]["motion_burst_percentile"] = float(
            self.config["performance"].get(
                "motion_probe_percentile",
                self.config["performance"]["motion_burst_percentile"],
            )
        )
        probe_config["performance"]["motion_burst_merge_gap_seconds"] = float(
            self.config["performance"].get(
                "motion_probe_burst_merge_gap_seconds",
                self.config["performance"]["motion_burst_merge_gap_seconds"],
            )
        )
        probe_config["performance"]["motion_burst_min_observations"] = int(
            self.config["performance"].get(
                "motion_probe_min_observations",
                self.config["performance"]["motion_burst_min_observations"],
            )
        )
        raw_motion_candidates = generate_motion_burst_candidates(
            c.motion_probe_views,
            c.motion_paths,
            probe_config,
            c.coarse_frame_index,
        )
        raw_motion_candidates = sorted(
            raw_motion_candidates,
            key=candidate_sort_key,
        )
        c.motion_candidates = fuse_motion_probe_candidates(
            raw_motion_candidates, probe_config
        )
        safety_fallback_used = False
        if not c.motion_candidates:
            safety_fallback_used = True
            c.motion_candidates = generate_motion_safety_candidates(
                c.motion_probe_views,
                c.motion_paths,
                probe_config,
                c.coarse_frame_index,
            )
        c.motion_candidates = sorted(c.motion_candidates, key=candidate_sort_key)
        if not c.motion_candidates:
            raise RuntimeError("输入视频没有产生任何可读运动帧，无法建立实验候选窗口")
        c.motion_windows = self._fine_windows(
            c.motion_candidates,
            c.infos,
            c.transforms,
            padding_seconds=float(
                self.config["performance"].get(
                    "motion_probe_window_padding_seconds", 90.0
                )
            ),
        )
        write_json(
            layout.json_config / "motion_probe_windows.json",
            {
                "schema_version": "visioncortex-motion-probe/1",
                "sentinel_views": [view.view_id for view in c.motion_probe_views],
                "raw_candidate_count": len(raw_motion_candidates),
                "fused_candidate_count": len(c.motion_candidates),
                "safety_fallback_used": safety_fallback_used,
                "coverage": self._window_coverage(c.motion_windows, c.infos),
                "candidates": [
                    item.model_dump(mode="json") for item in c.motion_candidates
                ],
            },
        )
        self._complete_stage(
            layout,
            "motion_probe",
            [
                layout.json_config / "scan_runtime_motion_probe.json",
                layout.json_config / "motion_probe_windows.json",
                sparse_strategy_path,
            ],
        )


    def _stage_candidate_coarse(self, c, layout, manifest):
        reuse_motion_probe = (
            bool(
                self.config["performance"].get(
                    "coarse_reuse_motion_probe", False
                )
            )
            and bool(
                self.config["performance"].get(
                    "motion_probe_run_yolo", False
                )
            )
            and not c.coarse_full_timeline
        )
        coarse_scan_views = (
            c.preselected_coarse_views
            if c.shared_coarse_motion_scan
            else c.motion_probe_views
            if reuse_motion_probe
            else c.preselected_coarse_views
        )
        coarse_scan_ids = {view.view_id for view in coarse_scan_views}
        coarse_manifest = manifest.model_copy(update={"views": coarse_scan_views})
        for view in manifest.views:
            self._view_runtime[view.view_id]["state"] = (
                "coarse_running"
                if view.view_id in coarse_scan_ids
                else "coarse_sentinel_not_selected"
            )
        self._status(
            layout,
            "candidate_coarse",
            0.28,
            (
                "复用运动阶段已经完成的全路全时间轴YOLO粗扫"
                if c.shared_coarse_motion_scan
                else f"全时间轴多视角 {self.config['performance']['coarse_detection_fps']} FPS YOLO粗筛"
                if c.coarse_full_timeline
                else f"候选窗口内 {self.config['performance']['coarse_detection_fps']} FPS YOLO粗筛"
            ),
        )
        if c.shared_coarse_motion_scan:
            coarse_paths = c.motion_paths
            write_json(
                layout.json_config / "coarse_reuse_motion_probe.json",
                {
                    "schema_version": "visioncortex-coarse-reuse/1",
                    "reused": True,
                    "source_phase": "shared_coarse_motion_decode",
                    "source_view_ids": [
                        view.view_id for view in coarse_scan_views
                    ],
                    "sample_fps": float(
                        self.config["performance"]["coarse_detection_fps"]
                    ),
                    "logical_motion_probe_fps": float(
                        self.config["performance"]["motion_probe_fps"]
                    ),
                    "additional_video_decode_bytes": 0,
                },
            )
        elif reuse_motion_probe:
            coarse_paths = c.motion_paths
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = (
                    "coarse_reused_motion_probe"
                    if view.view_id in coarse_scan_ids
                    else "coarse_sentinel_not_selected"
                )
            write_json(
                layout.json_config / "coarse_reuse_motion_probe.json",
                {
                    "schema_version": "visioncortex-coarse-reuse/1",
                    "reused": True,
                    "source_phase": "motion_probe",
                    "source_view_ids": [view.view_id for view in coarse_scan_views],
                    "sample_fps": float(
                        self.config["performance"]["motion_probe_fps"]
                    ),
                    "additional_video_decode_bytes": 0,
                },
            )
        else:
            coarse_paths = self._scan_all_views_concurrently(
                coarse_manifest,
                c.infos,
                c.transforms,
                layout.work / "detections-coarse",
                windows=None if c.coarse_full_timeline else c.motion_windows,
                sample_fps=float(self.config["performance"]["coarse_detection_fps"]),
                image_size=int(self.config["performance"]["coarse_image_size"]),
                keyframes_only=bool(self.config["performance"]["coarse_keyframes_only"]),
                phase="coarse",
            )
            self._archive_scan_runtime(
                layout, layout.work / "detections-coarse", "coarse"
            )
        self._ensure_coarse_frame_index(c, layout, c.infos, coarse_paths, coarse_scan_views)
        coarse_views = [
            view for view in coarse_scan_views if view.role == ViewRole.FIRST_PERSON
        ]
        coarse_config = json.loads(json.dumps(self.config))
        coarse_config["performance"][
            "candidate_alignment_uncertainty_ms_by_view"
        ] = {
            view_id: max(
                float(transform.uncertainty_ms or 0.0),
                float(transform.csv_rmse_ms or 0.0),
            )
            for view_id, transform in c.transforms.items()
        }
        coarse_config["segmentation"]["event_merge_gap_seconds"] = max(
            float(coarse_config["segmentation"]["event_merge_gap_seconds"]),
            1.5
            / float(
                self.config["performance"][
                    "motion_probe_fps"
                    if reuse_motion_probe
                    else "coarse_detection_fps"
                ]
            ),
        )
        coarse_config["segmentation"]["min_event_observations"] = 2
        coarse_candidates = generate_motion_burst_candidates(
            coarse_views,
            coarse_paths,
            coarse_config,
            c.coarse_frame_index,
        )
        comprehensive_coarse = bool(
            self.config["performance"].get(
                "coarse_comprehensive_candidate_union", False
            )
        )
        if comprehensive_coarse:
            coarse_candidates.extend(
                generate_coarse_activity_candidates(
                    coarse_scan_views,
                    coarse_paths,
                    coarse_config,
                    c.coarse_frame_index,
                )
            )
        elif not coarse_candidates:
            fallback_views = [
                view for view in coarse_scan_views if view.role == ViewRole.THIRD_PERSON
            ]
            fallback_paths = {view.view_id: coarse_paths[view.view_id] for view in fallback_views}
            coarse_candidates = generate_coarse_activity_candidates(
                fallback_views,
                fallback_paths,
                coarse_config,
                c.coarse_frame_index,
            )
        open_vocabulary_enabled = bool(
            self.config["performance"].get(
                "coarse_open_vocabulary_recall_enabled", False
            )
        )
        if open_vocabulary_enabled:
            open_vocabulary_candidates, open_vocabulary_report = (
                generate_open_vocabulary_coarse_candidates(
                    coarse_scan_views,
                    c.infos,
                    coarse_paths,
                    self.config,
                    c.coarse_frame_index,
                )
            )
        else:
            open_vocabulary_candidates, open_vocabulary_report = [], None
        coverage_required = bool(
            self.config["performance"].get(
                "coarse_coverage_gate_enabled", False
            )
        )
        coverage_ready = bool(
            not coverage_required
            or (
                c.coarse_index_report is not None
                and c.coarse_index_report.get("formal_evidence_ready")
            )
        )
        open_vocabulary_ready = bool(
            not open_vocabulary_enabled
            or (
                open_vocabulary_report is not None
                and open_vocabulary_report.get("formal_evidence_ready")
            )
        )
        candidate_gate_errors = []
        if not coverage_ready:
            candidate_gate_errors.append("coarse_frame_coverage_incomplete")
        if not open_vocabulary_ready:
            candidate_gate_errors.append(
                "open_vocabulary_recall_unavailable_or_error_rate_exceeded"
            )
        candidate_discovery_gate = {
            "schema_version": "visioncortex-candidate-discovery-quality-gate/1",
            "formal_evidence_ready": not candidate_gate_errors,
            "errors": candidate_gate_errors,
            "coarse_coverage_required": coverage_required,
            "coarse_coverage_ready": coverage_ready,
            "open_vocabulary_enabled": open_vocabulary_enabled,
            "open_vocabulary_ready": open_vocabulary_ready,
            "open_vocabulary_status": (
                open_vocabulary_report.get("status")
                if open_vocabulary_report is not None
                else "disabled"
            ),
        }
        write_json(
            layout.json_config / "candidate_discovery_quality_gate.json",
            candidate_discovery_gate,
        )
        if open_vocabulary_report is not None:
            write_json(
                layout.json_config / "coarse_open_vocabulary_recall.json",
                open_vocabulary_report,
            )
        if (
            self.config["performance"].get(
                "candidate_discovery_quality_gate_enabled", False
            )
            and candidate_gate_errors
        ):
            raise RuntimeError(
                "候选发现质量门禁失败: "
                + "; ".join(candidate_gate_errors)
            )
        coarse_candidates.extend(open_vocabulary_candidates)
        coarse_candidates = sorted(coarse_candidates, key=candidate_sort_key)
        c.boundary_candidates, boundary_report = refine_motion_candidates_with_coarse(
            c.motion_candidates,
            coarse_candidates,
            coarse_config,
        )
        c.boundary_candidates = sorted(
            c.boundary_candidates,
            key=candidate_sort_key,
        )
        boundary_report["coarse_scan_view_ids"] = [
            view.view_id for view in coarse_scan_views
        ]
        if c.coarse_full_timeline or comprehensive_coarse or open_vocabulary_enabled:
            boundary_report["enhancements"] = {
                "full_timeline_scan": c.coarse_full_timeline,
                "comprehensive_candidate_union": comprehensive_coarse,
                "open_vocabulary_candidate_ids": [
                    item.candidate_id for item in open_vocabulary_candidates
                ],
                "preserves_established_candidates": True,
            }
        write_json(
            layout.json_config / "coarse_boundary_refinement.json",
            boundary_report,
        )
        if open_vocabulary_report is not None:
            write_json(
                layout.json_config / "coarse_open_vocabulary_recall.json",
                open_vocabulary_report,
            )
        c.fine_views, c.fine_view_report = select_fine_scan_views(
            manifest.views,
            coarse_paths,
            c.boundary_candidates,
            self.config,
            c.coarse_frame_index,
        )
        short_timeline_ceiling_ms = float(
            self.config["performance"].get(
                "auto_exhaustive_short_timeline_seconds", 0.0
            )
            or 0.0
        ) * 1000.0
        if (
            self.config["performance"].get(
                "auto_exhaustive_short_timeline_enabled", False
            )
            and short_timeline_ceiling_ms > 0.0
            and c.infos
            and max(float(info.duration_ms) for info in c.infos.values())
            <= short_timeline_ceiling_ms
        ):
            self.config["performance"][
                "exhaustive_full_timeline_scan"
            ] = True
            self.config["performance"][
                "exhaustive_full_timeline_reason"
            ] = "automatic_short_probed_timeline_recall"
            self.config["performance"]["fine_progressive_cross_view"] = False
        c.progressive_enabled = bool(
            self.config["performance"].get("fine_progressive_cross_view", False)
        )
        if c.progressive_enabled:
            initial_fine_views, supplemental_fine_views = (
                self._progressive_fine_view_order(
                    c.fine_views,
                    c.fine_view_report,
                    int(
                        self.config["performance"].get(
                            "fine_initial_third_person_views", 1
                        )
                    ),
                    list(
                        self.config["performance"].get(
                            "fine_preferred_third_person_views"
                        )
                        or []
                    ),
                )
            )
            if bool(
                self.config["performance"].get(
                    "fine_dynamic_cross_view_scout", False
                )
            ):
                initial_fine_views = [
                    view
                    for view in c.fine_views
                    if view.role == ViewRole.FIRST_PERSON
                ]
                supplemental_fine_views = [
                    view
                    for view in c.fine_views
                    if view.role == ViewRole.THIRD_PERSON
                ]
            initial_fine_ids = {view.view_id for view in initial_fine_views}
            supplemental_rank = {
                view.view_id: rank
                for rank, view in enumerate(supplemental_fine_views, 1)
            }
            for view in c.fine_views:
                item = c.fine_view_report[view.view_id]
                item["progressive_initial"] = view.view_id in initial_fine_ids
                item["progressive_supplemental_rank"] = supplemental_rank.get(
                    view.view_id
                )
        write_json(layout.json_config / "fine_view_selection.json", c.fine_view_report)
        coarse_artifacts = [
            layout.json_config / "coarse_boundary_refinement.json",
            layout.json_config / "fine_view_selection.json",
        ]
        open_vocabulary_path = (
            layout.json_config / "coarse_open_vocabulary_recall.json"
        )
        if open_vocabulary_path.exists():
            coarse_artifacts.append(open_vocabulary_path)
        coarse_runtime = layout.json_config / "scan_runtime_coarse.json"
        if coarse_runtime.exists():
            coarse_artifacts.append(coarse_runtime)
        reuse_report = layout.json_config / "coarse_reuse_motion_probe.json"
        if reuse_report.exists():
            coarse_artifacts.append(reuse_report)
        coarse_index_manifest = (
            layout.json_config / "coarse_frame_index_manifest.json"
        )
        if coarse_index_manifest.exists():
            coarse_artifacts.append(coarse_index_manifest)
        candidate_quality_gate = (
            layout.json_config / "candidate_discovery_quality_gate.json"
        )
        if candidate_quality_gate.exists():
            coarse_artifacts.append(candidate_quality_gate)
        self._complete_stage(layout, "candidate_coarse", coarse_artifacts)

    def _stage_candidate_fine(self, c, layout, manifest):
        fine_windows = self._fine_windows(
            c.boundary_candidates, c.infos, c.transforms
        )
        fine_windows = {
            view.view_id: fine_windows[view.view_id]
            for view in c.fine_views
        }
        fine_windows, fine_availability_report = (
            self._intersect_fine_windows_with_usable_alignment(
                fine_windows,
                c.infos,
                c.transforms,
                c.fine_views,
            )
        )
        for view in list(c.fine_views):
            if fine_windows.get(view.view_id):
                continue
            c.fine_view_report[view.view_id]["selected"] = False
            c.fine_view_report[view.view_id]["reason"] = (
                "alignment quality gate exposes no usable fine window"
            )
        c.fine_views = [
            view for view in c.fine_views if fine_windows.get(view.view_id)
        ]
        write_json(
            layout.json_config / "fine_view_selection.json",
            c.fine_view_report,
        )
        fine_coverage = self._window_coverage(fine_windows, c.infos)
        fine_sample_fps = float(self.config["performance"]["detection_fps"])
        exhaustive_negative_audit = bool(
            self.config["performance"].get(
                "exhaustive_negative_audit_full_timeline", False
            )
        )
        c.exhaustive_full_timeline = bool(
            exhaustive_negative_audit
            or self.config["performance"].get(
                "exhaustive_full_timeline_scan", False
            )
        )
        fine_window_report = {
            "schema_version": "visioncortex-fine-scan-windows/2",
            "strategy": (
                "exhaustive_all_view_full_timeline_negative_audit"
                if exhaustive_negative_audit
                else "exhaustive_all_view_full_timeline"
                if c.exhaustive_full_timeline
                else "progressive_cross_view"
                if c.progressive_enabled
                else "all_selected_views"
            ),
            "eligible_view_ids": [view.view_id for view in c.fine_views],
            "selected_view_ids": [view.view_id for view in c.fine_views],
            "sample_fps": fine_sample_fps,
            "full_timeline_reason": self.config["performance"].get(
                "exhaustive_full_timeline_reason"
            ),
            "image_size": int(self.config["performance"]["image_size"]),
            "padding_seconds": float(
                self.config["performance"]["fine_window_padding_seconds"]
            ),
            "merge_gap_seconds": float(
                self.config["performance"].get(
                    "fine_window_merge_gap_seconds", 0.0
                )
            ),
            "coverage": fine_coverage,
            "usable_alignment_windows": fine_availability_report,
            "estimated_sampled_frames": int(
                round(
                    sum(float(item["selected_seconds"]) for item in fine_coverage.values())
                    * fine_sample_fps
                )
            ),
        }
        if (
            self.config["performance"].get(
                "fine_risk_window_expansion_enabled", False
            )
            or self.config["performance"].get(
                "fine_low_alignment_extra_padding_enabled", False
            )
        ):
            fine_window_report["risk_window_expansion"] = {
                "mode": "expand_only_preserve_baseline_windows",
                "action_types": list(
                    self.config["performance"].get(
                        "fine_risk_action_types", []
                    )
                ),
                "low_confidence_threshold": float(
                    self.config["performance"].get(
                        "fine_risk_low_confidence_threshold", 0.70
                    )
                ),
                "extra_padding_seconds": float(
                    self.config["performance"].get(
                        "fine_risk_extra_padding_seconds", 30.0
                    )
                ),
                "low_alignment_extra_padding_enabled": bool(
                    self.config["performance"].get(
                        "fine_low_alignment_extra_padding_enabled", False
                    )
                ),
                "low_alignment_confidence_threshold": float(
                    self.config["performance"].get(
                        "fine_low_alignment_confidence_threshold", 0.80
                    )
                ),
                "low_alignment_extra_padding_seconds": float(
                    self.config["performance"].get(
                        "fine_low_alignment_extra_padding_seconds", 30.0
                    )
                ),
            }
        write_json(layout.json_config / "fine_scan_windows.json", fine_window_report)
        if (
            self.config["performance"].get(
                "fine_coverage_gate_enabled", False
            )
            and not fine_availability_report["formal_evidence_ready"]
        ):
            raise RuntimeError(
                "精扫没有同时可用的第一/第三人称对齐窗口；查看 "
                f"{layout.json_config / 'fine_scan_windows.json'}"
            )
        c.fine_windows = fine_windows
        eligible_fine_ids = {view.view_id for view in c.fine_views}
        for view in manifest.views:
            self._view_runtime[view.view_id]["state"] = (
                "fine_pending" if view.view_id in eligible_fine_ids else "fine_not_selected"
            )
        c.progressive_report = None
        self._status(
            layout,
            "candidate_fine",
            0.48,
            (
                "渐进精扫：先核验主双视角，缺失窗口按对齐时间戳补扫"
                if c.progressive_enabled
                else "候选窗精扫并收紧动作边界"
            ),
        )
        fine_work_dir = layout.work / "detections-fine"
        fine_frame_index_report: dict[str, Any] | None = None
        c.fine_runtime_index: FineFrameIndex | None = None
        if c.progressive_enabled:
            c.detection_paths, scanned_fine_views, c.candidates, c.progressive_report = (
                self._run_progressive_fine_scan(
                    manifest,
                    c.fine_views,
                    c.fine_view_report,
                    c.boundary_candidates,
                    fine_windows,
                    c.infos,
                    c.transforms,
                    fine_work_dir,
                )
            )
            write_json(
                layout.json_config / "progressive_fine_scan.json",
                c.progressive_report,
            )
            write_json(
                layout.json_config / "fine_view_selection.json",
                c.fine_view_report,
            )
            fine_window_report.update(
                {
                    "selected_view_ids": c.progressive_report["scanned_view_ids"],
                    "coverage": c.progressive_report["actual_coverage"],
                    "estimated_sampled_frames": c.progressive_report[
                        "total_actual_estimated_frames"
                    ],
                    "full_fine_estimated_frames": c.progressive_report[
                        "actual_estimated_frames"
                    ],
                    "scout_estimated_frames": c.progressive_report[
                        "scout_estimated_frames"
                    ],
                    "full_pool_coverage": fine_coverage,
                    "full_pool_estimated_sampled_frames": c.progressive_report[
                        "all_view_estimated_frames"
                    ],
                    "avoided_selected_seconds": c.progressive_report[
                        "avoided_selected_seconds"
                    ],
                    "avoided_estimated_frames": c.progressive_report[
                        "avoided_estimated_frames"
                    ],
                }
            )
            write_json(
                layout.json_config / "fine_scan_windows.json", fine_window_report
            )
            fine_frame_index_report = c.progressive_report.get(
                "fine_frame_index"
            )
            if fine_frame_index_report is not None:
                c.fine_runtime_index = FineFrameIndex(
                    Path(str(fine_frame_index_report["index_path"]))
                )
        else:
            fine_manifest = manifest.model_copy(update={"views": c.fine_views})
            c.detection_paths = self._scan_all_views_concurrently(
                fine_manifest,
                c.infos,
                c.transforms,
                fine_work_dir,
                windows=fine_windows,
                sample_fps=fine_sample_fps,
                phase="fine",
            )
            scanned_fine_views = c.fine_views
            fine_frame_index: FineFrameIndex | None = None
            if self.config["performance"].get(
                "fine_frame_index_enabled", False
            ):
                fine_frame_index = create_fine_frame_index(
                    self._frame_index_path(fine_work_dir, "fine_frame_index.sqlite3")
                )
                c.fine_runtime_index = fine_frame_index
                ingest_report = ingest_fine_frame_ledgers(
                    fine_frame_index,
                    scanned_fine_views,
                    c.detection_paths,
                    source_pass="single-pass",
                    stitching_enabled=bool(
                        self.config["performance"].get(
                            "fine_track_stitching_enabled", False
                        )
                    ),
                    maximum_stitch_gap_ms=float(
                        self.config["performance"].get(
                            "fine_track_stitch_max_gap_seconds", 2.5
                        )
                    )
                    * 1000.0,
                    maximum_center_distance=float(
                        self.config["performance"].get(
                            "fine_track_stitch_max_center_distance", 0.12
                        )
                    ),
                )
                fine_frame_index_report = fine_frame_coverage_report(
                    fine_frame_index,
                    scanned_fine_views,
                    c.infos,
                    fine_windows,
                    sample_fps=fine_sample_fps,
                    minimum_coverage_ratio=float(
                        self.config["performance"].get(
                            "fine_minimum_coverage_ratio", 0.98
                        )
                    ),
                    maximum_gap_periods=float(
                        self.config["performance"].get(
                            "fine_maximum_gap_periods", 4.0
                        )
                    ),
                    alignment_scales={
                        view.view_id: c.transforms[view.view_id].scale
                        for view in scanned_fine_views
                    },
                )
                fine_frame_index_report["ingest_passes"] = [
                    ingest_report
                ]
                c.detection_paths = fine_frame_index.materialize_ledgers(
                    fine_work_dir / "indexed-detections",
                    scanned_fine_views,
                )
            c.candidates = (
                generate_candidates(
                    scanned_fine_views,
                    c.detection_paths,
                    self.config,
                )
                if fine_frame_index is None
                else generate_candidates(
                    scanned_fine_views,
                    c.detection_paths,
                    self.config,
                    fine_frame_index,
                )
            )
        fine_roi_candidates, fine_roi_report = (
            generate_open_vocabulary_fine_candidates(
                scanned_fine_views,
                c.infos,
                c.detection_paths,
                fine_windows,
                self.config,
                c.fine_runtime_index,
            )
        )
        c.candidates = sorted(
            [*c.candidates, *fine_roi_candidates], key=candidate_sort_key
        )
        write_json(
            layout.json_config / "fine_roi_open_vocabulary_recall.json",
            fine_roi_report,
        )
        if fine_frame_index_report is not None:
            write_json(
                layout.json_config / "fine_frame_index_manifest.json",
                fine_frame_index_report,
            )
        c.scanned_fine_ids = {view.view_id for view in scanned_fine_views}
        for view in c.fine_views:
            if view.view_id not in c.scanned_fine_ids:
                self._view_runtime[view.view_id]["state"] = (
                    "fine_not_scanned_no_remaining_evidence_gap"
                )
        self._archive_scan_runtime(
            layout,
            fine_work_dir,
            "fine",
            progressive_report=c.progressive_report,
        )
        if (
            self.config["performance"].get(
                "fine_coverage_gate_enabled", False
            )
            and (
                fine_frame_index_report is None
                or not fine_frame_index_report.get(
                    "formal_evidence_ready", False
                )
            )
        ):
            raise RuntimeError(
                "精扫实际帧覆盖门禁失败；查看 "
                f"{layout.json_config / 'fine_frame_index_manifest.json'}"
            )
        if (
            self.config["performance"].get(
                "fine_roi_open_vocabulary_recall_enabled", False
            )
            and not fine_roi_report.get("formal_evidence_ready", False)
        ):
            raise RuntimeError(
                "精扫手部 ROI 开放词汇补救门禁失败；查看 "
                f"{layout.json_config / 'fine_roi_open_vocabulary_recall.json'}"
            )
        scout_work_dir = fine_work_dir / "scout"
        if scout_work_dir.is_dir():
            self._archive_scan_runtime(
                layout,
                scout_work_dir,
                "fine_scout",
            )
        from .movement_verification import verify_movement_candidates

        movement_report = verify_movement_candidates(
            c.candidates, scanned_fine_views, c.infos, c.detection_paths, self.config,
            progress=lambda done, total: self._status(
                layout, "candidate_fine", 0.68, f"移动画面核验 {done}/{total}"
            ),
        )
        write_json(layout.json_config / "movement_visual_verification.json", movement_report)
        if self.config["archive"].get("keep_debug_candidates"):
            write_json(layout.json_config / "candidate_layer.json",
                       [candidate.model_dump(mode="json") for candidate in c.candidates])
        fine_artifacts = [
            layout.json_config / "movement_visual_verification.json",
            layout.json_config / "scan_runtime_fine.json",
            layout.json_config / "fine_scan_windows.json",
        ]
        fine_index_manifest_path = (
            layout.json_config / "fine_frame_index_manifest.json"
        )
        if fine_index_manifest_path.exists():
            fine_artifacts.append(fine_index_manifest_path)
        fine_roi_path = (
            layout.json_config / "fine_roi_open_vocabulary_recall.json"
        )
        if fine_roi_path.exists():
            fine_artifacts.append(fine_roi_path)
        progressive_path = layout.json_config / "progressive_fine_scan.json"
        if progressive_path.exists():
            fine_artifacts.append(progressive_path)
        scout_runtime_path = (
            layout.json_config / "scan_runtime_fine_scout.json"
        )
        if scout_runtime_path.exists():
            fine_artifacts.append(scout_runtime_path)
        candidate_layer = layout.json_config / "candidate_layer.json"
        if candidate_layer.exists():
            fine_artifacts.append(candidate_layer)
        self._complete_stage(layout, "candidate_fine", fine_artifacts)


    def _stage_candidate_audit(self, c, layout, manifest):
        self._status(layout, "candidate_audit", 0.68, "持续性、动作密度与跨视角一致性审计")
        c.candidates = sorted(c.candidates, key=candidate_sort_key)
        if c.fine_runtime_index is None:
            c.events, c.rejected = audit_candidates(
                c.candidates,
                c.transforms,
                self.config,
            )
        else:
            c.events, c.rejected = audit_candidates(
                c.candidates,
                c.transforms,
                self.config,
                c.fine_runtime_index,
            )
        if c.fine_runtime_index is None:
            c.rejected.extend(
                refine_liquid_events_with_context(c.events, c.detection_paths)
            )
        else:
            c.rejected.extend(
                refine_liquid_events_with_context(
                    c.events,
                    c.detection_paths,
                    frame_index=c.fine_runtime_index,
                    config=self.config,
                )
            )
        state_machine_ledger = attach_continuous_action_states(c.events, self.config)
        observability_receipts = attach_action_observability(c.events)
        semantic_review_plan = build_semantic_review_plan(c.events, self.config)
        c.observability_path = layout.json_config / "action_observability.json"
        c.semantic_review_plan_path = (
            layout.json_config / "semantic_review_plan.json"
        )
        c.state_machine_path = (
            layout.json_config / "continuous_action_state_ledger.json"
        )
        write_json(c.state_machine_path, state_machine_ledger)
        write_json(
            c.observability_path,
            {
                "schema_version": "visioncortex-action-observability-ledger/1",
                "event_count": len(c.events),
                "receipts": observability_receipts,
            },
        )
        write_json(c.semantic_review_plan_path, semantic_review_plan)
        raw_boundary_receipts: list[dict[str, Any]] = []
        activity_intervals = build_view_activity_intervals(
            c.events, c.boundary_candidates, [v for v in c.fine_views if v.view_id in c.scanned_fine_ids], c.transforms, c.fine_windows, self.config
        )
        segmentation_boundary_candidates = (
            [] if c.exhaustive_full_timeline else c.boundary_candidates
        )
        if c.exhaustive_full_timeline:
            raw_boundary_receipts.append(
                decision_receipt(
                    decision_type="full_timeline_segmentation_basis",
                    rule_id="QF1-EXHAUSTIVE-FINE-EVIDENCE-TIMELINE",
                    verdict="accepted",
                    subject_ids=[manifest.experiment_id],
                    reason_codes=[
                        "sparse_motion_windows_not_used_for_event_routing"
                    ],
                    facts={
                        "full_timeline_fine_scan": True,
                        "fine_view_ids": sorted(c.scanned_fine_ids),
                        "sparse_motion_candidate_ids": [
                            candidate.candidate_id
                            for candidate in c.boundary_candidates
                        ],
                        "segmentation_boundary_candidate_ids": [],
                        "formal_membership_changed": False,
                    },
                    evidence_refs=[
                        candidate.candidate_id
                        for candidate in c.boundary_candidates
                    ],
                )
            )
        analysis = compose_experiments(
            c.events, manifest.views, segmentation_boundary_candidates, self.config,
            operations=ExperimentAnalysisOperations(
                build_experiment_segments, normalize_experiment_segments,
                prepare_formal_experiment_segments, build_experiment_groups, select_key_events,
            ),
            boundary_decisions=raw_boundary_receipts,
        )
        raw_segments, normalized_segments = analysis.raw_segments, analysis.normalized_segments
        c.segments, c.groups = analysis.segments, analysis.groups
        formal_segment_receipts = analysis.formal_decisions
        normalization_receipts = analysis.normalization_decisions
        continuity_receipts = analysis.continuity_decisions
        selection_decisions = analysis.selection_decisions
        precheck_key_events = analysis.selected_key_events
        key_selection_path = (
            layout.json_config / "key_material_selection_preview.json"
        )
        selection_report = _key_material_selection_report(
            c.groups,
            c.segments,
            c.events,
            selection_decisions,
        )
        write_json(key_selection_path, selection_report)
        self._preprocessing_completed_seconds = round(time.perf_counter() - self._run_started_perf, 6)
        write_json(
            layout.json_config / "audit_layer.json",
            {
                "events": [event.model_dump(mode="json") for event in c.events],
                "rejected": c.rejected,
                "activity_intervals_by_view": activity_intervals,
                "boundary_candidates": [
                    candidate.model_dump(mode="json")
                    for candidate in sorted(
                        c.boundary_candidates,
                        key=candidate_sort_key,
                    )
                ],
                "raw_segments": [
                    segment.model_dump(mode="json") for segment in raw_segments
                ],
                "normalized_segments": [
                    segment.model_dump(mode="json")
                    for segment in normalized_segments
                ],
                "segments": [segment.model_dump(mode="json") for segment in c.segments],
                "raw_boundary_decision_receipts": raw_boundary_receipts,
                "formal_segment_receipts": formal_segment_receipts,
                "normalization_decision_receipts": normalization_receipts,
                "continuity_decision_receipts": continuity_receipts,
                "quality_decision_receipts": [
                    *raw_boundary_receipts,
                    *normalization_receipts,
                    *formal_segment_receipts,
                    *continuity_receipts,
                    *selection_decisions,
                ],
                "experiment_groups": [group.model_dump(mode="json") for group in c.groups],
                "action_observability": observability_receipts,
                "continuous_action_state": state_machine_ledger,
                "semantic_review_plan": semantic_review_plan,
            },
        )
        c.boundary_precheck = self._run_boundary_precheck(
            layout, c.groups, progressive_report=c.progressive_report
        )
        self._complete_stage(
            layout,
            "candidate_audit",
            [
                layout.json_config / "audit_layer.json",
                layout.json_config / "boundary_precheck.json",
                key_selection_path,
                c.observability_path,
                c.state_machine_path,
                c.semantic_review_plan_path,
            ],
        )

        if bool(
            self.config.get("project", {}).get(
                "preprocessing_acceptance_only", False
            )
        ):
            self._status(
                layout,
                "preprocessing_acceptance",
                0.99,
                "验证有界双视角实验与关键事件选择，不调用模型或导出媒体",
            )
            c.key_events = precheck_key_events
            baseline = self._acceptance_baseline() or {}
            minimum_key_events = baseline.get("minimum_selected_key_events")
            key_event_gate_passed = (
                minimum_key_events is None
                or len(c.key_events) >= int(minimum_key_events)
            )
            acceptance_passed = bool(c.boundary_precheck["passed"]) and bool(
                key_event_gate_passed
            )
            acceptance_path = (
                layout.json_config / "preprocessing_acceptance.json"
            )
            write_json(
                acceptance_path,
                {
                    "schema_version": (
                        "visioncortex-preprocessing-acceptance/1"
                    ),
                    "status": (
                        "passed" if acceptance_passed else "failed"
                    ),
                    "run_mode": "preprocessing_acceptance_only",
                    "formal_archive_promotion_allowed": False,
                    "model_api_calls": 0,
                    "token_usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "preprocessing_seconds": (
                        self._preprocessing_completed_seconds
                    ),
                    "experiment_group_count": len(c.groups),
                    "selected_key_event_count": len(c.key_events),
                    "minimum_selected_key_events": minimum_key_events,
                    "key_event_gate_passed": key_event_gate_passed,
                    "baseline_selection": dict(
                        self._acceptance_baseline_selection
                    ),
                    "experiment_groups": [
                        group.model_dump(mode="json") for group in c.groups
                    ],
                    "selected_key_event_ids": [
                        event.event_id for event in c.key_events
                    ],
                    "boundary_precheck": c.boundary_precheck,
                    "key_material_selection_preview": selection_report,
                    "limitations": [
                        "No MLLM experiment naming or step understanding was run.",
                        "No experiment clips, key frames, key clips, daily report, or PDF were materialized.",
                        "This staging result must not replace the accepted fixed archive.",
                    ],
                },
            )
            self._status(
                layout,
                "preprocessing_completed",
                1.0,
                "预处理性能与边界质量验收完成；正式归档未提升",
            )
            run_metrics = self._metrics(c.key_events, c.groups)
            run_metrics["run_mode"] = "preprocessing_acceptance_only"
            run_metrics["formal_archive_promotion_allowed"] = False
            write_json(layout.json_config / "run_metrics.json", run_metrics)
            if not acceptance_passed:
                raise RuntimeError(
                    "Preprocessing acceptance failed after evidence selection; "
                    f"see {acceptance_path}"
                )
            self._complete_stage(
                layout,
                "preprocessing_acceptance",
                [
                    acceptance_path,
                    key_selection_path,
                    layout.json_config / "run_metrics.json",
                ],
            )
            c.early_result = layout.root
            return


    def _stage_experiment_understanding(self, c, layout, manifest):
        self._status(layout, "experiment_understanding", 0.72, "用完整有界双视角故事板命名实验并核验连续性")
        if c.groups:
            (layout.json_config / "speech_understanding.json").unlink(missing_ok=True)
        if not c.groups and speech.enabled(self.config):
            from .speech_semantics import analyze_unsegmented_recording

            self._status(layout, "experiment_understanding", 0.72, "未发现可确认实验片段，正在用实际画面核对实验录音")
            try:
                self._speech_understanding = analyze_unsegmented_recording(
                    layout, manifest.views, c.infos, c.transforms, self.config
                )
            finally:
                recording_receipt = layout.json_config / "speech_understanding.json"
                if recording_receipt.is_file():
                    self._speech_understanding = json.loads(recording_receipt.read_text(encoding="utf-8"))
        from .boundary_review import review_experiment_boundaries

        c.groups = review_experiment_boundaries(
            layout, c.groups, c.segments, c.events, manifest.views, c.infos, c.transforms, self.config
        )
        if any(group.boundary_reviews for group in c.groups):
            # Retain the original CV partition and publish the reconciled
            # group identities before names, clips and key materials exist.
            audit_path = layout.json_config / "audit_layer.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
            audit["experiment_groups_before_boundary_review"] = audit.get("experiment_groups", [])
            audit["experiment_groups"] = [g.model_dump(mode="json") for g in c.groups]
            audit["segments"] = [s.model_dump(mode="json") for s in c.segments]
            write_json(audit_path, audit)
            write_json(layout.json_config / "boundary_precheck_before_semantic_review.json", c.boundary_precheck)
            c.boundary_precheck = self._run_boundary_precheck(layout, c.groups, progressive_report=c.progressive_report)
        analyze_experiment_groups(
            layout, c.groups, c.segments, c.events, manifest.views, c.infos, c.transforms, self.config
        )
        group_semantic_recall = _recover_group_storyboard_state_events(
            c.groups,
            c.segments,
            c.events,
            self.config,
        )
        group_semantic_recall_path = (
            layout.json_config / "group_storyboard_semantic_recall.json"
        )
        write_json(
            group_semantic_recall_path,
            {
                "schema_version": (
                    "visioncortex-group-storyboard-semantic-recall/1"
                ),
                "policy": (
                    "explicit dual-view closure transition; recall-only; "
                    "independent event-level Ark proof required"
                ),
                "recovered_event_count": len(group_semantic_recall),
                "events": group_semantic_recall,
            },
        )
        if group_semantic_recall:
            # The first ledgers are written before Ark so preprocessing can
            # fail closed without any model call.  A real group review may
            # add only provisional recall events; refresh the shadow
            # ledgers so their semantic origin and incomplete state remain
            # explicit until event-level adjudication.
            observability_receipts = attach_action_observability(c.events)
            state_machine_ledger = attach_continuous_action_states(
                c.events, self.config
            )
            semantic_review_plan = build_semantic_review_plan(
                c.events, self.config
            )
            write_json(c.observability_path, {
                "schema_version": "visioncortex-action-observability-ledger/1",
                "event_count": len(c.events),
                "receipts": observability_receipts,
            })
            write_json(c.state_machine_path, state_machine_ledger)
            write_json(c.semantic_review_plan_path, semantic_review_plan)
            audit_layer_path = layout.json_config / "audit_layer.json"
            audit_layer = json.loads(
                audit_layer_path.read_text(encoding="utf-8-sig")
            )
            audit_layer.update(
                {
                    "events": [
                        event.model_dump(mode="json") for event in c.events
                    ],
                    "segments": [
                        segment.model_dump(mode="json")
                        for segment in c.segments
                    ],
                    "experiment_groups": [
                        group.model_dump(mode="json") for group in c.groups
                    ],
                    "action_observability": observability_receipts,
                    "continuous_action_state": state_machine_ledger,
                    "semantic_review_plan": semantic_review_plan,
                    "group_storyboard_semantic_recall": {
                        "path": group_semantic_recall_path.relative_to(
                            layout.root
                        ).as_posix(),
                        "events": group_semantic_recall,
                    },
                }
            )
            write_json(audit_layer_path, audit_layer)
        c.group_understanding_path = layout.json_config / "experiment_group_understanding.json"
        write_json(
            c.group_understanding_path,
            {
                "schema_version": "visioncortex-experiment-group-understanding/1",
                "groups": [group.model_dump(mode="json") for group in c.groups],
            },
        )
        write_json(layout.json_config / "run_metrics_live.json", self._metrics(c.events, c.groups))
        final_selection_decisions: list[dict[str, Any]] = []
        c.key_events = select_key_events(
            c.groups,
            c.segments,
            c.events,
            self.config,
            decision_receipts=final_selection_decisions,
        )
        key_selection_path = layout.json_config / "key_material_selection.json"
        write_json(
            key_selection_path,
            _key_material_selection_report(
                c.groups,
                c.segments,
                c.events,
                final_selection_decisions,
            ),
        )
        self._complete_stage(
            layout,
            "experiment_understanding",
            [
                c.group_understanding_path,
                *([layout.json_config / "speech_understanding.json", layout.root / "Key-Materials/Experiment-Audio"] if (layout.json_config / "speech_understanding.json").is_file() else []),
                group_semantic_recall_path,
                key_selection_path,
                layout.json_config / "run_metrics_live.json",
            ],
        )


    def _stage_experiment_clips(self, c, layout, manifest):
        self._status(layout, "experiment_clips", 0.78, "按模型实验名归档第一/第三/并排三份有界视频")
        materialize_experiment_clips(
            layout,
            c.groups,
            c.segments,
            c.events,
            manifest.views,
            c.infos,
            c.transforms,
            self.config,
            publisher=self._publisher,
        )
        # Clip generation assigns folder names and media paths. Publish
        # them now, before later model work, so completed videos can play.
        write_json(c.group_understanding_path, {
            "schema_version": "visioncortex-experiment-group-understanding/1",
            "groups": [group.model_dump(mode="json") for group in c.groups],
        })
        self._complete_stage(
            layout,
            "experiment_clips",
            [
                layout.experiment_clips,
                c.group_understanding_path,
                layout.json_config / "experiment_clip_materialization_runtime.json",
            ],
        )


    def _stage_key_materials(self, c, layout, manifest):
        self._status(layout, "key_materials", 0.84, "按实验组提取去重后的分层动作关键素材")
        materialize_key_materials(
            layout,
            c.key_events,
            c.groups,
            manifest.views,
            c.infos,
            c.transforms,
            c.detection_paths,
            self.config,
            publisher=self._publisher,
            archive_id=manifest.experiment_id,
        )
        self._complete_stage(
            layout,
            "key_materials",
            [
                layout.key_materials,
                layout.json_config / "key_material_materialization_runtime.json",
            ],
        )


    def _stage_mllm(self, c, layout, manifest):
        self._status(layout, "mllm", 0.92, "调用已配置的图像模型理解去重后的关键动作当前/下一步骤")
        analyze_key_materials(
            layout, c.key_events, self.config,
            progress_callback=lambda done, total: self._status(
                layout, "mllm", 0.92 + 0.009 * done / max(1, total),
                f"关键动作模型理解已处理 {done}/{total}，正在核验画面证据",
            ),
        )
        c.reviewed_key_events = list(c.key_events)
        self._complete_stage(
            layout,
            "mllm",
            self._checkpoint_key_material_understanding(
                layout, "mllm", c.reviewed_key_events, c.groups
            ),
        )

    def _stage_material_refinement(self, c, layout, manifest):
        self._status(layout, "material_refinement", 0.93, "复核关键画面与动作参与对象")
        c.key_events, c.semantic_curation = curate_semantically_reviewed_key_materials(
            layout,
            c.reviewed_key_events,
            c.groups,
            self.config,
            publisher=self._publisher,
        )
        segment_semantic_repairs = _synchronize_segments_with_final_key_events(
            c.segments, c.groups, c.key_events
        )
        # Semantic adjudication can replace the action, participants, or
        # strongest supporting role. Re-materialize only events whose final
        # evidence can change the selected frame/boxes instead of repeating
        # every accepted clip after the model pass.
        pre_final_key_timestamps = {
            event.event_id: float(event.key_global_ms) for event in c.key_events
        }
        curation_record_by_event = {
            str(record["event_id"]): record
            for record in c.semantic_curation.get("records") or []
        }
        rematerialize_event_ids: set[str] = set()
        for event in c.key_events:
            record = curation_record_by_event.get(event.event_id) or {}
            selected_source_view = str(
                (
                    (event.observability or {}).get("key_frame_selection")
                    or {}
                ).get("source_view_id")
                or ""
            )
            direct_view_ids = {
                str(item)
                for item in (event.semantic_review or {}).get(
                    "directly_supported_view_ids", []
                )
                if item
            }
            if (
                record.get("semantic_participant_refinement_changed")
                or record.get("cv_action_type")
                != record.get("final_action_type")
                or (
                    direct_view_ids
                    and selected_source_view not in direct_view_ids
                )
            ):
                rematerialize_event_ids.add(event.event_id)
        if rematerialize_event_ids:
            materialize_key_materials(
                layout,
                c.key_events,
                c.groups,
                manifest.views,
                c.infos,
                c.transforms,
                c.detection_paths,
                self.config,
                publisher=self._publisher,
                archive_id=manifest.experiment_id,
                materialize_event_ids=rematerialize_event_ids,
                progress_callback=lambda done, total: self._status(
                    layout,
                    "material_refinement",
                    0.93 + 0.01 * done / max(1, total),
                    f"复核关键画面：{done}/{total}",
                ),
            )
        state_receipt_repairs = _synchronize_final_event_state_receipts(
            c.key_events, self.config
        )
        self._status(layout, "material_refinement", 0.94, "校验参与对象标注与连续性")
        final_annotation = _rerender_curated_participant_annotations(
            layout, c.key_events, c.groups, self.config
        )
        c.key_events, c.semantic_curation = (
            reconcile_visually_reviewed_participants(
                layout,
                c.key_events,
                c.groups,
                c.semantic_curation,
            )
        )
        visual_reconciliation = c.semantic_curation.get(
            "visual_participant_reconciliation"
        ) or {}
        if (
            int(visual_reconciliation.get("pruned_event_count") or 0)
            or int(visual_reconciliation.get("excluded_event_count") or 0)
        ):
            # The first render is the evidence used to adjudicate unstable
            # paper/cap participants.  Rebuild the surviving user tree from
            # the reconciled participant sets.  All cloud requests are
            # content-addressed and reused; removed events stay in the
            # formal Review-Candidates tree.
            state_receipt_repairs.extend(
                _synchronize_final_event_state_receipts(
                    c.key_events, self.config
                )
            )
            segment_semantic_repairs.extend(
                _synchronize_segments_with_final_key_events(
                    c.segments, c.groups, c.key_events
                )
            )
            write_key_material_category_index(
                layout,
                c.groups,
                c.key_events,
                include_empty_categories=bool(
                    self.config.get("archive", {}).get(
                        "include_empty_action_categories", True
                    )
                ),
            )
            final_annotation = _rerender_curated_participant_annotations(
                layout, c.key_events, c.groups, self.config
            )
        c.semantic_curation["final_annotation"] = {
            key: value
            for key, value in final_annotation.items()
            if key != "records"
        }
        c.semantic_curation["post_semantic_participant_key_frame_selection"] = {
            "schema_version": (
                "visioncortex-post-semantic-participant-key-frame-selection/1"
            ),
            "policy": (
                "immutable-fine-ledger; selective refresh for changed final evidence"
            ),
            "full_scan_repeated": False,
            "source_copy_bytes": 0,
            "rematerialized_event_ids": sorted(rematerialize_event_ids),
            "unchanged_media_reused_event_ids": sorted(
                event.event_id
                for event in c.key_events
                if event.event_id not in rematerialize_event_ids
            ),
            "events": [
                {
                    "event_id": event.event_id,
                    "previous_key_global_ms": pre_final_key_timestamps[
                        event.event_id
                    ],
                    "selected_key_global_ms": float(event.key_global_ms),
                    "selection_offset_ms": round(
                        float(event.key_global_ms)
                        - pre_final_key_timestamps[event.event_id],
                        3,
                    ),
                    "media_rematerialized": (
                        event.event_id in rematerialize_event_ids
                    ),
                }
                for event in c.key_events
            ],
            "state_receipt_repairs": state_receipt_repairs,
            "segment_semantic_repairs": segment_semantic_repairs,
        }
        write_json(
            layout.json_config / "semantic_key_material_curation.json",
            c.semantic_curation,
        )
        self._complete_stage(layout, "material_refinement", [layout.key_materials, layout.json_config / "semantic_key_material_curation.json"])

    def _stage_semantic_refinement(self, c, layout, manifest):
        self._status(layout, "semantic_refinement", 0.95, "整理经过画面复核的实验步骤")
        # The initial group pass names the bounded experiment and verifies
        # continuity. Final steps come from the stronger event-level
        # adjudication, so rebuild them deterministically instead of paying
        # for a duplicate group MLLM pass over the same evidence.
        refine_groups_from_final_events(c.groups, c.key_events)
        if self.config.get("mllm", {}).get("enabled") and self.config["mllm"].get("operation_review", {}).get("enabled", False):
            from .operation_review import review
            review(layout.root, c.groups, c.key_events, self.config)
        normalize_final_group_action_language(c.groups, c.key_events)
        for group in c.groups:
            for segment in c.segments:
                if segment.group_id != group.group_id:
                    continue
                segment.experiment_name = group.experiment_name
                segment.experiment_name_en = group.experiment_name_en
                segment.semantic_understanding = group.model_understanding
        write_json(
            c.group_understanding_path,
            {
                "schema_version": "visioncortex-experiment-group-understanding/2",
                "refinement_pass": "post_event_semantic_curation",
                "groups": [group.model_dump(mode="json") for group in c.groups],
            },
        )
        refresh_key_material_metadata(
            layout,
            c.key_events,
            c.groups,
            c.transforms,
            archive_id=manifest.experiment_id,
        )
        self._complete_stage(
            layout,
            "semantic_refinement",
            [
                layout.key_materials,
                c.group_understanding_path,
                *self._checkpoint_key_material_understanding(
                    layout, "semantic_refinement", c.reviewed_key_events,
                    c.groups, c.semantic_curation,
                ),
            ],
        )

    def _stage_package(self, c, layout, manifest):
        # Only semantically curated formal key events may create durable
        # physical-change claims. Accepted CV recall candidates that never
        # reached a formal dual-view segment must not leak into the report
        # as liquid_transferred/container_state_changed facts.
        physical_changes = build_physical_change_log(c.key_events)

        self._status(layout, "package", 0.96, "归档证据包并执行 evidence-package-eval")
        c.summary = finalize_archive(
            layout,
            manifest,
            c.infos,
            c.transforms,
            c.events,
            c.segments,
            c.groups,
            c.key_events,
            physical_changes,
            c.rejected,
            self.config,
            c.disk_report,
        )
        # Validate the archive-level experiment groups. A continuous group
        # may contain multiple atomic segments but must count as one bounded
        # experiment in the user-facing output and boundary evaluation.
        self._run_sidecar_validation(layout, manifest, c.groups)
        quality_acceptance = self._run_quality_acceptance(
            layout, c.groups, c.key_events
        )
        from .result_review import inspect as inspect_result_completeness
        inspect_result_completeness(layout.root, save=True)
        if quality_acceptance.get("passed") is not True:
            c.quality_attention = True
            return
        # A successful retry must not publish the previous attempt's
        # current partial report. The retry endpoint preserves its history.
        (layout.json_config / "partial_delivery.json").unlink(missing_ok=True)
        (layout.root / "Partial-Results/Partial-Evidence-Report.html").unlink(missing_ok=True)
        self._complete_stage(layout, "package", [layout.json_config])

    def _stage_daily_report(self, c, layout, manifest):
        self._status(layout, "daily_report", 0.98, "从已验收证据生成实验室日报并执行一致性校验")
        generate_daily_report_archive(
            layout, c.summary, self._metrics(c.events, c.groups), self.config, defer_pdf=True
        )
        self._complete_stage(
            layout,
            "daily_report",
            [layout.daily_reports, layout.json_config / "daily_report_manifest.json"],
        )

    def _stage_professional_pdf(self, c, layout, manifest):
        from .daily_reports import generate_professional_report_archive
        if not self.config.get("daily_report", {}).get("generate_pdf", True):
            self._complete_stage(layout, "professional_pdf", [], status="skipped", reason="未启用 PDF 导出")
            return
        generate_professional_report_archive(layout)
        self._complete_stage(layout, "professional_pdf", [layout.professional_pdfs, layout.json_config / "professional_report_manifest.json", layout.json_config / "daily_report_manifest.json"])

    def _stage_finalizing(self, c, layout, manifest):
        self._status(
            layout,
            "finalizing",
            0.995,
            "封存最终运行指标、溯源清单和归档契约",
        )
        run_metrics = self._metrics(c.events, c.groups)
        c.summary.stats["run_metrics"] = run_metrics
        c.summary.stats["run_provenance"] = {
            "path": "JSON-Config-Files/run_provenance.json"
        }
        write_json(
            layout.json_config / "run_metrics.json",
            run_metrics,
        )
        write_json(
            layout.json_config / "evidence_package.json",
            c.summary.model_dump(mode="json"),
        )
        write_run_provenance(
            layout.root,
            self.config,
            self._model_certification_audit,
            repository_root=Path(__file__).resolve().parents[2],
        )
        write_archive_contract_manifest(layout.root)
        validate_archive_contracts_or_raise(layout.root)
        self._complete_stage(
            layout,
            "finalizing",
            [
                layout.experiment_clips,
                layout.key_materials,
                layout.json_config,
                layout.daily_reports,
                layout.professional_pdfs,
            ],
        )

    def _ensure_coarse_frame_index(self, c, layout, infos,
        paths: dict[str, Path], views: Sequence[ViewInput]
    ) -> CoarseFrameIndex | None:
        if not self.config["performance"].get(
            "coarse_frame_index_enabled", False
        ):
            return None
        if c.coarse_frame_index is not None:
            return c.coarse_frame_index
        c.coarse_frame_index, c.coarse_index_report = build_coarse_frame_index(
            self._frame_index_path(layout.work, "coarse-frame-index.sqlite3"),
            views,
            c.infos,
            paths,
            sample_fps=float(
                self.config["performance"]["coarse_detection_fps"]
            ),
            minimum_coverage_ratio=float(
                self.config["performance"].get(
                    "coarse_minimum_coverage_ratio", 0.98
                )
            ),
            maximum_gap_periods=float(
                self.config["performance"].get(
                    "coarse_maximum_gap_periods", 4.0
                )
            ),
        )
        manifest_report = dict(c.coarse_index_report)
        manifest_report["index_path"] = str(c.coarse_frame_index.path)
        manifest_report["index_storage"] = "local_runtime_rebuildable"
        write_json(
            layout.json_config / "coarse_frame_index_manifest.json",
            manifest_report,
        )
        if (
            self.config["performance"].get(
                "coarse_coverage_gate_enabled", False
            )
            and not c.coarse_index_report["formal_evidence_ready"]
        ):
            failed = [
                item["view_id"]
                for item in c.coarse_index_report["views"]
                if not item["formal_evidence_ready"]
            ]
            raise RuntimeError(
                "粗扫覆盖门禁失败，以下视角存在采样缺口: "
                + ", ".join(failed)
            )
        return c.coarse_frame_index

    @staticmethod
    def _acquire_lock(lock_path: Path) -> None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = lock_path.read_text(encoding="utf-8", errors="replace") if lock_path.is_file() else "unknown"
            raise RuntimeError(f"该实验已有运行实例: {owner}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))

    def _fine_windows(self, candidates, infos, transforms, padding_seconds=None):
        from .actions import fine_scan_windows
        return fine_scan_windows(candidates, infos, transforms, self.config, padding_seconds)

    def _intersect_fine_windows_with_usable_alignment(
        self,
        windows: dict[str, list[tuple[float, float]]],
        infos: dict[str, VideoInfo],
        transforms: dict[str, AlignmentTransform],
        views: Sequence[ViewInput],
    ) -> tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]:
        """Remove only intervals already declared unavailable by alignment."""

        view_by_id = {view.view_id: view for view in views}
        intersected: dict[str, list[tuple[float, float]]] = {}
        reports: dict[str, dict[str, Any]] = {}
        for view_id, requested in windows.items():
            transform = transforms[view_id]
            info = infos[view_id]
            if transform.state == "failed":
                usable: list[tuple[float, float]] = []
            elif transform.segment_transforms:
                usable = [
                    (
                        max(0.0, float(segment.local_start_ms)),
                        min(float(info.duration_ms), float(segment.local_end_ms)),
                    )
                    for segment in transform.segment_transforms
                    if segment.state != "failed"
                    and segment.local_end_ms > segment.local_start_ms
                ]
            elif (
                transform.local_coverage_start_ms is not None
                and transform.local_coverage_end_ms is not None
            ):
                usable = [
                    (
                        max(0.0, float(transform.local_coverage_start_ms)),
                        min(
                            float(info.duration_ms),
                            float(transform.local_coverage_end_ms),
                        ),
                    )
                ]
            else:
                usable = [(0.0, float(info.duration_ms))]
            selected = self._merge_time_windows(
                [
                    (max(start, usable_start), min(end, usable_end))
                    for start, end in requested
                    for usable_start, usable_end in usable
                    if min(end, usable_end) > max(start, usable_start)
                ]
            )
            requested_seconds = sum(
                max(0.0, end - start) for start, end in requested
            ) / 1000.0
            selected_seconds = sum(
                max(0.0, end - start) for start, end in selected
            ) / 1000.0
            intersected[view_id] = selected
            reports[view_id] = {
                "role": view_by_id[view_id].role.value,
                "alignment_state": transform.state,
                "requested_window_count": len(requested),
                "usable_interval_count": len(usable),
                "selected_window_count": len(selected),
                "requested_seconds": round(requested_seconds, 6),
                "selected_seconds": round(selected_seconds, 6),
                "excluded_unavailable_seconds": round(
                    max(0.0, requested_seconds - selected_seconds), 6
                ),
                "usable_intervals": usable,
            }
        first_ready = any(
            intersected.get(view.view_id)
            for view in views
            if view.role == ViewRole.FIRST_PERSON
        )
        third_ready = any(
            intersected.get(view.view_id)
            for view in views
            if view.role == ViewRole.THIRD_PERSON
        )
        return intersected, {
            "schema_version": "visioncortex-fine-usable-alignment-windows/1",
            "policy": "intersect_only_fail_closed_alignment_intervals",
            "first_person_window_available": first_ready,
            "third_person_window_available": third_ready,
            "formal_evidence_ready": bool(first_ready and third_ready),
            "views": reports,
        }

    def _run_sidecar_validation(self, layout: ArchiveLayout, manifest: RunManifest, groups) -> None:
        if not self.config.get("validation", {}).get("use_sidecar_annotations"):
            return
        from .validation import validate_against_sidecars

        report = validate_against_sidecars(
            manifest.views,
            groups,
            str(self.config["validation"].get("sidecar_name", "video.annotation.json")),
        )
        if report is not None:
            write_json(layout.json_config / "validation_against_annotations.json", report)


def create_dry_run(output: Path, config: dict[str, Any]) -> Path:
    layout = ArchiveLayout(output.resolve())
    layout.create()
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("dry-run-first.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("dry-run-third.mp4")),
    ]
    manifest = RunManifest(experiment_id=output.name or "dry-run", views=views)
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp01",
            confidence=0.98,
            state="aligned",
            csv_match_count=100,
            csv_match_ratio=1.0,
            visual_confidence=0.95,
        )
        for view in views
    }
    candidates = [
        ActionCandidate(
            candidate_id=f"CAND-{view.view_id}-000001",
            action_type=ActionType.HAND_OBJECT_CONTACT,
            view_id=view.view_id,
            role=view.role,
            local_start_ms=10_000.0,
            local_end_ms=12_000.0,
            global_start_ms=10_000.0,
            global_end_ms=12_000.0,
            key_global_ms=11_000.0,
            objects=["gloved_hand", "pipette"],
            confidence=0.91,
        )
        for view in views
    ]
    event = EvidenceEvent(
        event_id="EVT-000001",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=10_000.0,
        global_end_ms=12_000.0,
        key_global_ms=11_000.0,
        objects=["gloved_hand", "pipette"],
        confidence=0.96,
        accepted=True,
        audit_reason="dry-run: 第一/第三人称一致",
        supporting_views=[view.view_id for view in views],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=candidates,
        model_understanding={
            "status": "completed",
            "execution_mode": "dry_run",
            "current_step": "戴手套的手接触移液器",
            "next_step": "未知",
            "confidence": 0.9,
        },
    )
    image = np.full((360, 640, 3), 35, dtype=np.uint8)
    cv2.putText(image, "DRY RUN - ALIGNED KEY FRAME", (45, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    segment = ExperimentSegment(
        segment_id="EXP-0001",
        global_start_ms=8_000.0,
        global_end_ms=15_000.0,
        event_ids=[event.event_id],
        participating_views=[view.view_id for view in views],
        clips={view.view_id: f"dry-run://{view.view_id}/experiment-clip" for view in views},
        micro_segments=[
            {
                "micro_segment_id": "MICRO-0001-0001",
                "start_global_ms": 10_000.0,
                "end_global_ms": 12_000.0,
                "action_type": event.action_type.value,
                "objects": event.objects,
            }
        ],
    )
    group = ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=[segment.segment_id],
        global_start_ms=segment.global_start_ms,
        global_end_ms=segment.global_end_ms,
        participating_views=[view.view_id for view in views],
        first_person_view="fp01",
        third_person_view="tp01",
        continuity_reason="dry-run 独立实验",
        experiment_name="移液操作实验",
        experiment_name_en="Pipetting-Operation-Experiment",
        archive_folder="001_Pipetting-Operation-Experiment",
        key_event_ids=[event.event_id],
        videos={
            "first-person": "dry-run://fp01/experiment-clip",
            "third-person": "dry-run://tp01/experiment-clip",
            "aligned_first_third": "dry-run://aligned/experiment-clip",
        },
        video_json={
            "first-person": "dry-run://fp01/experiment-json",
            "third-person": "dry-run://tp01/experiment-json",
            "aligned_first_third": "dry-run://aligned/experiment-json",
        },
        model_understanding={
            "status": "completed",
            "execution_mode": "dry_run",
            "steps": [
                {
                    "step_index": 1,
                    "start_global_ms": event.global_start_ms,
                    "end_global_ms": event.global_end_ms,
                    "current_step": "戴手套的手接触移液器",
                    "next_step": "未知",
                    "next_step_status": "unknown",
                    "supporting_event_ids": [event.event_id],
                    "objects": event.objects,
                    "supporting_views": event.supporting_views,
                    "confidence": event.confidence,
                }
            ],
        },
    )
    prepare_key_material_category_layout(layout, [group])
    action_folder = key_material_action_folder(event.action_type)
    for view in views:
        path = (
            layout.key_frames
            / str(group.archive_folder)
            / action_folder
            / event.event_id
            / f"{view.view_id}.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        event.key_frames[view.view_id] = archive_relative_posix(path, layout.root)
        event.key_clips[view.view_id] = f"dry-run://{view.view_id}/key-clip"
    aligned_path = (
        layout.key_frames
        / str(group.archive_folder)
        / action_folder
        / event.event_id
        / "Aligned_First+Third.jpg"
    )
    cv2.imwrite(str(aligned_path), image)
    event.key_frames["aligned_first_third"] = archive_relative_posix(
        aligned_path, layout.root
    )
    event.key_clips["aligned_first_third"] = "dry-run://aligned/key-clip"
    write_key_material_category_index(layout, [group], [event])
    physical = [
        PhysicalChange(
            change_id="CHANGE-000001",
            event_id=event.event_id,
            global_ms=11_000.0,
            change_type="contact_started",
            object_names=event.objects,
            supporting_views=event.supporting_views,
            confidence=event.confidence,
        )
    ]
    summary = finalize_archive(
        layout,
        manifest,
        {},
        transforms,
        [event],
        [segment],
        [group],
        [event],
        physical,
        [],
        config,
        {"required_bytes": 0, "free_bytes": 0},
        {
            "total_duration_seconds": 0.0,
            "stage_durations": [],
            "tokens": {
                "key_materials": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "run_total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
            "dry_run": True,
        },
    )
    write_json(
        layout.json_config / "evidence_package_eval.json",
        {
            "passed": True,
            "dry_run": True,
            "checks": [
                {"check": "schema_serialization", "passed": True},
                {"check": "archive_layout", "passed": True},
                {"check": "no_video_or_ffmpeg_required", "passed": True},
            ],
        },
    )
    write_json(
        layout.json_config / "audit_layer.json",
        {
            "events": [event.model_dump(mode="json")],
            "rejected": [],
            "raw_segments": [segment.model_dump(mode="json")],
            "normalized_segments": [segment.model_dump(mode="json")],
            "segments": [segment.model_dump(mode="json")],
            "formal_segment_receipts": [],
            "experiment_groups": [group.model_dump(mode="json")],
            "dry_run": True,
        },
    )
    dry_quality = validate_experiment_and_material_quality(
        [group],
        [event],
        None,
        minimum_cross_view_event_rate=0.0,
    )
    dry_quality["dry_run"] = True
    dry_quality["structural_passed"] = bool(dry_quality.get("passed"))
    dry_quality["evidence_level"] = "synthetic_structural_only"
    dry_quality["formal_accuracy_claim_allowed"] = False
    write_json(layout.json_config / "quality_acceptance.json", dry_quality)
    dry_metrics = {
        "total_duration_seconds": 0.0,
        "preprocessing_sla": {"actual_seconds": 0.0, "target_seconds": 1200.0, "met": True},
        "stage_durations": [],
        "tokens": {
            "experiment_groups": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0},
            "key_materials": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0},
            "daily_report": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0},
            "run_total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
        "dry_run": True,
    }
    write_json(layout.json_config / "run_metrics.json", dry_metrics)
    generate_daily_report_archive(layout, summary, dry_metrics, config)
    write_run_provenance(
        layout.root,
        config,
        {"required": False, "status": "not_required_by_profile"},
        repository_root=Path(__file__).resolve().parents[2],
    )
    write_archive_contract_manifest(layout.root)
    validate_archive_contracts_or_raise(layout.root)
    write_json(layout.root / "run_status.json", {"stage": "completed", "progress": 1.0, "dry_run": True})
    return layout.root
