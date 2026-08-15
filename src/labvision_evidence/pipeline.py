from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat as stat_module
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .actions import (
    audit_candidates,
    build_experiment_segments,
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
from .alignment import build_alignments
from .archive import (
    ArchiveLayout,
    analyze_experiment_groups,
    analyze_key_materials,
    finalize_archive,
    materialize_experiment_clips,
    materialize_key_materials,
    refresh_key_material_metadata,
    write_aligned_csv,
    write_json,
)
from .grouping import (
    build_experiment_groups,
    is_experiment_start_anchor,
    normalize_experiment_segments,
    prepare_formal_experiment_segments,
    select_key_events,
)
from .detection import scan_videos, validate_models
from .daily_reports import generate_daily_report_archive
from .schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    PhysicalChange,
    RunManifest,
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
from .validation import validate_experiment_and_material_quality


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
        "schema_version": "visioncortex-key-material-selection/1",
        "selection_rule": "cross-view accepted physical actions; duplicate only near-identical time/object evidence",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_cache_identity(config: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    """Bind resumable ledgers to code, config, models and concrete inputs."""

    package_root = Path(__file__).resolve().parent
    code_digest = hashlib.sha256()
    for source in sorted(package_root.glob("*.py")):
        code_digest.update(source.name.encode("utf-8"))
        code_digest.update(source.read_bytes())

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
        }
        for view in manifest.views
    }
    all_source_paths = [
        path
        for files in files_by_view.values()
        for file_type in ("videos", "timestamps_csvs")
        for path in files[file_type]
    ]
    source_snapshots, source_snapshot_report = snapshot_source_paths(
        all_source_paths,
        workers=int(config["performance"].get("source_stat_workers", 24)),
        max_age_seconds=float(
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

    payload = {
        "schema_version": "visioncortex-cache-identity/1",
        "code_sha256": code_digest.hexdigest(),
        "config": stable_config,
        "models": models,
        "inputs": inputs,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    payload["cache_key"] = hashlib.sha256(encoded).hexdigest()[:20]
    payload["source_snapshot_report"] = source_snapshot_report
    return payload


class EvidencePipeline:
    def __init__(self, config: dict[str, Any], progress: ProgressCallback | None = None):
        self.config = config
        self.progress = progress or _noop_progress
        self._run_started_perf = 0.0
        self._run_started_iso = ""
        self._active_stage: str | None = None
        self._active_stage_started = 0.0
        self._active_stage_started_iso = ""
        self._stage_metrics: list[dict[str, Any]] = []
        self._startup_metrics: dict[str, Any] = {}
        self._preprocessing_completed_seconds: float | None = None
        self._input_view_count = 0
        self._input_mode = "unknown"
        self._publisher: IncrementalArchivePublisher | None = None
        self._resource_monitor: ResourceMonitor | None = None
        self._view_runtime: dict[str, dict[str, Any]] = {}
        self._runtime_lock = threading.Lock()
        self._active_layout: ArchiveLayout | None = None

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
                    "status": "failed" if stage == "failed" else "completed",
                    "started_at": self._active_stage_started_iso,
                    "ended_at": now_iso,
                    "duration_seconds": round(now_perf - self._active_stage_started, 6),
                }
            )
        if stage != self._active_stage:
            if stage in {"completed", "failed"}:
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
            ),
        }
        write_json(
            layout.root / "run_status.json",
            status_payload,
        )
        if self.config.get("storage", {}).get("run_output_mode") == "nas_direct":
            write_json(layout.json_config / "pipeline_status.json", status_payload)
        if self._publisher is not None:
            self._publisher.publish_status(status_payload)

    def _complete_stage(
        self,
        layout: ArchiveLayout,
        stage: str,
        artifacts: list[Path] | tuple[Path, ...] = (),
    ) -> Path:
        """Publish a durable, atomic NAS receipt only after a stage succeeds."""

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
            relative = artifact.resolve().relative_to(layout.root.resolve())
            relative_artifacts.append(relative.as_posix())
            if self._publisher is not None:
                if artifact.is_dir():
                    self._publisher.publish_directory(relative)
                else:
                    self._publisher.publish_file(artifact)
        receipt = {
            "schema_version": "visioncortex-stage-receipt/1",
            "stage": stage,
            "status": "completed",
            "completed_at": completed_at,
            "run_elapsed_seconds": elapsed_seconds,
            "stage_duration_seconds": stage_duration,
            "archive_mode": self.config.get("storage", {}).get("run_output_mode", "local"),
            "archive_root": str(layout.root),
            "artifacts": relative_artifacts,
            "token_ledger": "JSON-Config-Files/run_metrics.json",
        }
        receipt_path = layout.json_config / "Stage-Receipts" / f"{stage}.json"
        write_json(receipt_path, receipt)
        if self._publisher is not None:
            self._publisher.publish_file(receipt_path)
        return receipt_path

    def _acceptance_baseline(self) -> dict[str, Any] | None:
        configured = self.config.get("validation", {}).get("acceptance_baseline")
        if not configured:
            return None
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.is_file():
            raise FileNotFoundError(f"验收基线不存在: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"验收基线必须是 JSON 对象: {path}")
        return payload

    def _run_quality_acceptance(
        self,
        layout: ArchiveLayout,
        groups: list[ExperimentGroup],
        key_events: list[EvidenceEvent],
    ) -> dict[str, Any]:
        validation = self.config.get("validation", {})
        report = validate_experiment_and_material_quality(
            groups,
            key_events,
            self._acceptance_baseline(),
            boundary_match_iou=float(validation.get("boundary_match_iou", 0.50)),
            max_start_error_seconds=float(validation.get("max_start_error_seconds", 8.0)),
            max_end_error_seconds=float(validation.get("max_end_error_seconds", 8.0)),
            minimum_cross_view_event_rate=float(
                validation.get("minimum_cross_view_event_rate", 0.25)
            ),
        )
        write_json(layout.json_config / "quality_acceptance.json", report)
        return report

    def _run_boundary_precheck(
        self,
        layout: ArchiveLayout,
        groups: list[ExperimentGroup],
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
        )
        boundary = full_report["experiment_boundaries"]
        evaluated = bool(boundary["evaluated"])
        passed = bool(boundary["passed"]) if evaluated else True
        report = {
            "schema_version": "visioncortex-boundary-precheck/1",
            "status": "passed" if passed else "failed",
            "evaluated": evaluated,
            "passed": passed,
            "baseline": full_report["baseline"],
            "thresholds": full_report["thresholds"],
            "experiment_boundaries": boundary,
            "gate_position": "before_experiment_and_key_material_model_calls",
        }
        path = layout.json_config / "boundary_precheck.json"
        write_json(path, report)
        if (
            evaluated
            and not passed
            and bool(validation.get("fail_before_model_on_boundary_regression", True))
        ):
            raise RuntimeError(
                "Bounded experiment quality precheck failed before model calls; "
                f"see {path}"
            )
        return report

    def _metrics(self, events, groups=()) -> dict[str, Any]:
        key_calls = []
        for event in events:
            understanding = event.model_understanding or {}
            if "usage" in understanding or understanding.get("status") in {"completed", "failed"}:
                key_calls.append(
                    {
                        "stage": "key_material_understanding",
                        "event_id": event.event_id,
                        "model": understanding.get("model", self.config["mllm"]["model"]),
                        "status": understanding.get("status"),
                        "latency_seconds": understanding.get("latency_seconds"),
                        "attempts": understanding.get("attempts"),
                        "cache_reused": bool(understanding.get("cache_reused")),
                        "usage": understanding.get("usage", {}),
                    }
                )

        group_calls = []
        for group in groups:
            understanding = group.model_understanding or {}
            if "usage" in understanding or understanding.get("status") in {"completed", "failed"}:
                group_calls.append(
                    {
                        "stage": "experiment_group_understanding",
                        "group_id": group.group_id,
                        "model": understanding.get("model", self.config["mllm"]["model"]),
                        "status": understanding.get("status"),
                        "latency_seconds": understanding.get("latency_seconds"),
                        "attempts": understanding.get("attempts"),
                        "cache_reused": bool(understanding.get("cache_reused")),
                        "usage": understanding.get("usage", {}),
                    }
                )

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
        all_calls = group_calls + key_calls
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
                    "note": "Only MLLM step analysis consumes model tokens; CV and FFmpeg consume no tokens.",
                },
            },
            "mllm_calls": all_calls,
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
        groups = [
            [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON],
            [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON],
        ]
        groups = [group for group in groups if group]
        kwargs = {
            "windows": windows,
            "sample_fps": sample_fps,
            "image_size": image_size,
            "keyframes_only": keyframes_only,
            "phase": phase,
            "progress_callback": lambda view_id, completed, total: self._scan_progress(
                phase, view_id, completed, total
            ),
        }
        perf = self.config["performance"]
        requested_sources = int(perf.get("source_workers", len(manifest.views)))
        if requested_sources < len(manifest.views):
            raise ValueError(
                f"source_workers={requested_sources} cannot keep {len(manifest.views)} views active"
            )
        lanes = list(
            perf.get(
                "coarse_decode_lanes"
                if phase in {"motion_probe", "coarse"}
                else "fine_decode_lanes",
                [],
            )
        )
        if not lanes:
            lanes = ["cuda" if perf.get("ffmpeg_hwaccel") else "cpu"] * len(manifest.views)
        if len(lanes) < len(manifest.views):
            lanes.extend([lanes[-1]] * (len(manifest.views) - len(lanes)))
        concurrent_roles = bool(perf.get("concurrent_role_scanners", True)) and len(groups) > 1
        default_decode_backends = {
            view.view_id: lanes[index] for index, view in enumerate(manifest.views)
        }
        kwargs["decode_backends"] = {
            view.view_id: (decode_backends or default_decode_backends).get(
                view.view_id, default_decode_backends[view.view_id]
            )
            for view in manifest.views
        }
        if (
            phase == "coarse"
            and windows is None
            and perf.get("synchronized_segment_waves")
            and concurrent_roles
            and all(view.segments for view in manifest.views)
        ):
            segment_counts = {view.view_id: len(view.segments) for view in manifest.views}
            if len(set(segment_counts.values())) != 1:
                raise ValueError(
                    f"synchronized segment waves require equal segment counts: {segment_counts}"
                )
            kwargs["wave_barrier"] = threading.Barrier(len(manifest.views))
        for view in manifest.views:
            runtime = self._view_runtime.setdefault(view.view_id, {})
            runtime.update(
                {
                    "role": view.role.value,
                    "decode_backend": kwargs["decode_backends"][view.view_id],
                    "state": f"{phase}_running",
                }
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"scheduler_{phase}.json").write_text(
            json.dumps(
                {
                    "schema_version": "visioncortex-role-scheduler/1",
                    "phase": phase,
                    "mode": "concurrent_roles" if concurrent_roles else "sequential_role_residency",
                    "role_order": [group[0].role.value for group in groups],
                    "configured_decode_lanes": lanes,
                    "active_view_ids": [view.view_id for view in manifest.views],
                    "decode_backends": kwargs["decode_backends"],
                    "reason": (
                        "configured concurrent role scanners"
                        if concurrent_roles
                        else "one TensorRT role model resident at a time to preserve batch capacity"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if not concurrent_roles:
            result = {}
            for group in groups:
                group_kwargs = dict(kwargs)
                group_kwargs["decode_backends"] = {
                    view.view_id: kwargs["decode_backends"][view.view_id]
                    for view in group
                }
                if (
                    phase == "coarse"
                    and windows is None
                    and perf.get("synchronized_segment_waves")
                    and all(view.segments for view in group)
                    and len(group) > 1
                ):
                    group_kwargs["wave_barrier"] = threading.Barrier(len(group))
                for view in group:
                    self._view_runtime[view.view_id]["decode_backend"] = group_kwargs[
                        "decode_backends"
                    ][view.view_id]
                result.update(
                    scan_videos(
                        group,
                        infos,
                        transforms,
                        work_dir,
                        self.config,
                        **group_kwargs,
                    )
                )
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = f"{phase}_completed"
            return result
        result = {}
        with ThreadPoolExecutor(max_workers=len(groups), thread_name_prefix="role-scanner") as executor:
            futures = [
                executor.submit(scan_videos, group, infos, transforms, work_dir, self.config, **kwargs)
                for group in groups
            ]
            for future in futures:
                result.update(future.result())
        for view in manifest.views:
            self._view_runtime[view.view_id]["state"] = f"{phase}_completed"
        return result

    def _motion_probe_views(self, manifest: RunManifest) -> list[ViewInput]:
        """Choose sentinel views; bounded YOLO scans still use all eligible views."""

        perf = self.config["performance"]
        first_limit = max(1, int(perf.get("motion_probe_first_person_views", 1)))
        third_limit = max(0, int(perf.get("motion_probe_third_person_views", 1)))
        first = [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON]
        third = [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON]
        return first[:first_limit] + third[:third_limit]

    def _coarse_scan_views(self, manifest: RunManifest) -> list[ViewInput]:
        """Choose boundary sentinels; fine validation still uses required views."""

        perf = self.config["performance"]
        first_limit = max(1, int(perf.get("coarse_first_person_views", 1)))
        third_limit = max(1, int(perf.get("coarse_third_person_views", 1)))
        first = [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON]
        third = [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON]
        return first[:first_limit] + third[:third_limit]

    @staticmethod
    def _progressive_fine_view_order(
        fine_views: list[ViewInput],
        fine_view_report: dict[str, dict[str, Any]],
        initial_third_person_views: int,
        preferred_third_person_views: list[str] | None = None,
    ) -> tuple[list[ViewInput], list[ViewInput]]:
        """Return the initial views and ordered third-person fallback pool.

        An optional caller-provided preference may rank known deployment views.
        Otherwise each upload is ranked only by its own coarse anchor activity,
        with stable manifest order as the deterministic final tie-breaker.
        """

        positions = {view.view_id: index for index, view in enumerate(fine_views)}
        preferred_positions = {
            view_id: index
            for index, view_id in enumerate(preferred_third_person_views or [])
        }
        first = [view for view in fine_views if view.role == ViewRole.FIRST_PERSON]
        third = sorted(
            (view for view in fine_views if view.role == ViewRole.THIRD_PERSON),
            key=lambda view: (
                0 if view.view_id in preferred_positions else 1,
                preferred_positions.get(view.view_id, len(preferred_positions)),
                -int(fine_view_report.get(view.view_id, {}).get("active_anchor_frames", 0)),
                -int(fine_view_report.get(view.view_id, {}).get("anchor_frames", 0)),
                positions[view.view_id],
            ),
        )
        initial_count = min(len(third), max(0, int(initial_third_person_views)))
        return first + third[:initial_count], third[initial_count:]

    def _progressive_target_status(
        self,
        boundary_candidates: list[ActionCandidate],
        events: list[EvidenceEvent],
    ) -> list[dict[str, Any]]:
        """Classify which recalled windows still need another third-person view.

        A supplemental scan is demanded only by reliable first-person evidence.
        A window is covered only by an accepted, boundary-eligible event with
        both roles. This deliberately prevents a partial liquid hypothesis such
        as DEV-011 EVT-000444 from suppressing a needed supplemental scan.
        """

        audit_margin_ms = (
            float(
                self.config["performance"].get(
                    "fine_progressive_audit_margin_seconds", 5.0
                )
            )
            * 1000.0
        )
        results = []
        for candidate in boundary_candidates:
            # Decode padding maximizes recall and may overlap adjacent experiments.
            # It must not be reused as the evidence-association window, otherwise
            # one event can falsely mark two nearby candidates as cross-view covered.
            window_start = candidate.global_start_ms - audit_margin_ms
            window_end = candidate.global_end_ms + audit_margin_ms
            relevant = []
            for event in events:
                # The candidate/action ledger has already applied temporal and
                # physical-action construction. A first-person anchor rejected
                # only for missing cross-view support is exactly the condition
                # that must trigger a supplemental view.
                if event.action_type == ActionType.LIQUID_MOVEMENT:
                    required = set(
                        self.config["segmentation"].get(
                            "liquid_start_anchor_required_objects", ["pipette"]
                        )
                    )
                    reliable_start_signal = event.accepted and (
                        not required or bool(set(event.objects) & required)
                    )
                else:
                    reliable_start_signal = is_experiment_start_anchor(
                        event, self.config
                    )
                if not reliable_start_signal:
                    continue
                midpoint = (event.global_start_ms + event.global_end_ms) / 2.0
                if window_start <= midpoint <= window_end:
                    relevant.append(event)
            first_signal = [
                event for event in relevant if ViewRole.FIRST_PERSON in event.supporting_roles
            ]
            cross_view = [
                event
                for event in first_signal
                if event.accepted
                and {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}.issubset(
                    set(event.supporting_roles)
                )
            ]
            if cross_view:
                status = "cross_view_covered"
            elif first_signal:
                status = "needs_third_person_supplement"
            else:
                status = "no_reliable_first_person_activity"
            results.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "global_start_ms": candidate.global_start_ms,
                    "global_end_ms": candidate.global_end_ms,
                    "audit_window_start_ms": window_start,
                    "audit_window_end_ms": window_end,
                    "status": status,
                    "first_person_anchor_event_ids": [
                        event.event_id for event in first_signal
                    ],
                    "first_person_anchor_windows": [
                        {
                            "event_id": event.event_id,
                            "global_start_ms": event.global_start_ms,
                            "global_end_ms": event.global_end_ms,
                            "key_global_ms": event.key_global_ms,
                            "confidence": event.confidence,
                            "accepted": event.accepted,
                            "action_type": event.action_type.value,
                            "objects": list(event.objects),
                        }
                        for event in first_signal
                    ],
                    "cross_view_anchor_event_ids": [event.event_id for event in cross_view],
                }
            )
        return results

    @staticmethod
    def _progressive_anchor_windows(
        target_status: list[dict[str, Any]],
        view_ids: list[str],
        infos,
        transforms,
        padding_seconds: float,
        candidate_ids: set[str] | None = None,
        peak_radius_seconds: float | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        """Build aligned narrow windows around reliable first-person events."""

        padding_ms = max(0.0, float(padding_seconds)) * 1000.0
        peak_radius_ms = (
            max(0.0, float(peak_radius_seconds)) * 1000.0
            if peak_radius_seconds is not None
            else None
        )
        grouped: dict[str, list[tuple[float, float]]] = {
            view_id: [] for view_id in view_ids
        }
        for item in target_status:
            if candidate_ids is not None and item["candidate_id"] not in candidate_ids:
                continue
            for anchor in item.get("first_person_anchor_windows") or []:
                if peak_radius_ms is None:
                    global_start = float(anchor["global_start_ms"]) - padding_ms
                    global_end = float(anchor["global_end_ms"]) + padding_ms
                else:
                    peak_ms = float(anchor["key_global_ms"])
                    global_start = peak_ms - peak_radius_ms
                    global_end = peak_ms + peak_radius_ms
                for view_id in view_ids:
                    local_start = max(
                        0.0, transforms[view_id].to_local(global_start)
                    )
                    local_end = min(
                        infos[view_id].duration_ms,
                        transforms[view_id].to_local(global_end),
                    )
                    if local_end > local_start:
                        grouped[view_id].append((local_start, local_end))

        merged: dict[str, list[tuple[float, float]]] = {}
        for view_id, windows in grouped.items():
            result: list[list[float]] = []
            for start, end in sorted(windows):
                if result and start <= result[-1][1]:
                    result[-1][1] = max(result[-1][1], end)
                else:
                    result.append([start, end])
            merged[view_id] = [(item[0], item[1]) for item in result]
        return merged

    @staticmethod
    def _progressive_scout_anchor_representatives(
        target_status: list[dict[str, Any]],
        candidate_ids: set[str],
        *,
        dedup_tolerance_seconds: float,
        cluster_gap_seconds: float,
        representatives_per_cluster: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Collapse repeated candidate references into sparse routing anchors.

        This affects only the low-FPS view-ranking scout. Formal 10 FPS evidence
        continues to use every aligned first-person action span.
        """

        occurrences: list[dict[str, Any]] = []
        for item in target_status:
            candidate_id = str(item["candidate_id"])
            if candidate_id not in candidate_ids:
                continue
            for raw_anchor in item.get("first_person_anchor_windows") or []:
                anchor = dict(raw_anchor)
                anchor["candidate_ids"] = [candidate_id]
                occurrences.append(anchor)

        by_event: dict[str, dict[str, Any]] = {}
        for index, anchor in enumerate(occurrences):
            event_id = str(anchor.get("event_id") or f"anonymous-{index:06d}")
            existing = by_event.get(event_id)
            if existing is None:
                anchor["event_id"] = event_id
                by_event[event_id] = anchor
                continue
            existing["candidate_ids"] = sorted(
                set(existing.get("candidate_ids") or [])
                | set(anchor.get("candidate_ids") or [])
            )

        tolerance_ms = max(0.0, float(dedup_tolerance_seconds)) * 1000.0
        peak_groups: list[list[dict[str, Any]]] = []
        for anchor in sorted(
            by_event.values(), key=lambda item: float(item["key_global_ms"])
        ):
            if (
                peak_groups
                and float(anchor["key_global_ms"])
                - float(peak_groups[-1][-1]["key_global_ms"])
                <= tolerance_ms
            ):
                peak_groups[-1].append(anchor)
            else:
                peak_groups.append([anchor])

        unique_peaks: list[dict[str, Any]] = []
        for group in peak_groups:
            representative = max(
                group,
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    -max(
                        0.0,
                        float(item.get("global_end_ms", 0.0))
                        - float(item.get("global_start_ms", 0.0)),
                    ),
                    str(item.get("event_id", "")),
                ),
            ).copy()
            representative["event_ids"] = sorted(
                str(item["event_id"]) for item in group
            )
            representative["candidate_ids"] = sorted(
                {
                    candidate_id
                    for item in group
                    for candidate_id in item.get("candidate_ids") or []
                }
            )
            unique_peaks.append(representative)

        cluster_gap_ms = max(0.0, float(cluster_gap_seconds)) * 1000.0
        clusters: list[list[dict[str, Any]]] = []
        for anchor in unique_peaks:
            if (
                clusters
                and float(anchor["key_global_ms"])
                - float(clusters[-1][-1]["key_global_ms"])
                <= cluster_gap_ms
            ):
                clusters[-1].append(anchor)
            else:
                clusters.append([anchor])

        limit = max(1, int(representatives_per_cluster))
        selected: list[dict[str, Any]] = []
        cluster_reports: list[dict[str, Any]] = []
        for cluster_index, cluster in enumerate(clusters, 1):
            peak_values = [float(item["key_global_ms"]) for item in cluster]
            median_peak = float(np.median(np.asarray(peak_values, dtype=np.float64)))
            ranked = sorted(
                cluster,
                key=lambda item: (
                    abs(float(item["key_global_ms"]) - median_peak),
                    -float(item.get("confidence", 0.0)),
                    str(item.get("event_id", "")),
                ),
            )
            chosen = sorted(
                ranked[: min(limit, len(ranked))],
                key=lambda item: float(item["key_global_ms"]),
            )
            selected.extend(chosen)
            cluster_reports.append(
                {
                    "cluster_index": cluster_index,
                    "global_start_ms": min(peak_values),
                    "global_end_ms": max(peak_values),
                    "unique_peak_count": len(cluster),
                    "candidate_ids": sorted(
                        {
                            candidate_id
                            for item in cluster
                            for candidate_id in item.get("candidate_ids") or []
                        }
                    ),
                    "representative_event_ids": [
                        str(item["event_id"]) for item in chosen
                    ],
                    "representative_peak_ms": [
                        float(item["key_global_ms"]) for item in chosen
                    ],
                }
            )

        diagnostics = {
            "candidate_count": len(candidate_ids),
            "raw_anchor_occurrence_count": len(occurrences),
            "unique_event_count": len(by_event),
            "unique_peak_count": len(unique_peaks),
            "dedup_tolerance_seconds": float(dedup_tolerance_seconds),
            "cluster_gap_seconds": float(cluster_gap_seconds),
            "cluster_count": len(clusters),
            "representatives_per_cluster": limit,
            "representative_count": len(selected),
            "clusters": cluster_reports,
        }
        return selected, diagnostics

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
        """Resolve dual-view gaps with optional scout-ranked aligned narrow windows."""

        perf = self.config["performance"]
        ordered_initial, ordered_supplemental = self._progressive_fine_view_order(
            fine_views,
            fine_view_report,
            int(perf.get("fine_initial_third_person_views", 1)),
            list(perf.get("fine_preferred_third_person_views") or []),
        )
        if not any(
            view.role == ViewRole.THIRD_PERSON
            for view in ordered_initial + ordered_supplemental
        ):
            raise ValueError("progressive fine scan requires at least one third-person view")
        scout_enabled = bool(perf.get("fine_dynamic_cross_view_scout", False))
        if scout_enabled:
            initial_views = [
                view for view in fine_views if view.role == ViewRole.FIRST_PERSON
            ]
            supplemental_views = [
                view
                for view in ordered_initial + ordered_supplemental
                if view.role == ViewRole.THIRD_PERSON
            ]
        else:
            initial_views = ordered_initial
            supplemental_views = ordered_supplemental

        lanes = list(perf.get("fine_decode_lanes") or [])
        if not lanes:
            lanes = ["cuda" if perf.get("ffmpeg_hwaccel") else "cpu"] * len(
                manifest.views
            )
        if len(lanes) < len(manifest.views):
            lanes.extend([lanes[-1]] * (len(manifest.views) - len(lanes)))
        full_decode_backends = {
            view.view_id: lanes[index] for index, view in enumerate(manifest.views)
        }

        scanned: dict[str, ViewInput] = {}
        detection_paths: dict[str, Path] = {}
        actual_windows: dict[str, list[tuple[float, float]]] = {}
        pass_reports: list[dict[str, Any]] = []
        scout_summary: dict[str, Any] = {"enabled": scout_enabled}

        def execute_pass(
            pass_index: int,
            pass_kind: str,
            pass_views: list[ViewInput],
            pass_candidates: list[ActionCandidate],
            target_snapshot: list[dict[str, Any]] | None = None,
        ) -> list[dict[str, Any]]:
            pass_name = f"pass-{pass_index:02d}-{pass_kind}"
            if pass_kind == "primary":
                pass_windows = {
                    view.view_id: fine_windows[view.view_id] for view in pass_views
                }
                window_strategy = "coarse_candidate_windows"
            elif target_snapshot is not None:
                pass_windows = self._progressive_anchor_windows(
                    target_snapshot,
                    [view.view_id for view in pass_views],
                    infos,
                    transforms,
                    float(
                        perf.get(
                            "fine_progressive_anchor_padding_seconds", 10.0
                        )
                    ),
                    {candidate.candidate_id for candidate in pass_candidates},
                )
                if not any(pass_windows.values()):
                    all_pass_windows = self._fine_windows(
                        pass_candidates, infos, transforms
                    )
                    pass_windows = {
                        view.view_id: all_pass_windows[view.view_id]
                        for view in pass_views
                    }
                    window_strategy = "coarse_candidate_fallback"
                else:
                    window_strategy = "aligned_first_person_anchor_windows"
            else:
                all_pass_windows = self._fine_windows(
                    pass_candidates, infos, transforms
                )
                pass_windows = {
                    view.view_id: all_pass_windows[view.view_id] for view in pass_views
                }
                window_strategy = "coarse_candidate_windows"
            pass_manifest = manifest.model_copy(update={"views": pass_views})
            result = self._scan_all_views_concurrently(
                pass_manifest,
                infos,
                transforms,
                work_dir / pass_name,
                windows=pass_windows,
                sample_fps=float(perf["detection_fps"]),
                phase="fine",
                decode_backends={
                    view.view_id: full_decode_backends[view.view_id] for view in pass_views
                },
            )
            for view in pass_views:
                scanned[view.view_id] = view
                detection_paths[view.view_id] = result[view.view_id]
                actual_windows[view.view_id] = pass_windows[view.view_id]
            scanned_views = [
                view for view in fine_views if view.view_id in scanned
            ]
            current_candidates = generate_candidates(
                scanned_views, detection_paths, self.config
            )
            current_events, _ = audit_candidates(
                current_candidates, transforms, self.config
            )
            liquid_context_rejections = refine_liquid_events_with_context(
                current_events, detection_paths
            )
            target_status = self._progressive_target_status(
                boundary_candidates, current_events
            )
            pass_reports.append(
                {
                    "pass_index": pass_index,
                    "pass_kind": pass_kind,
                    "window_strategy": window_strategy,
                    "view_ids": [view.view_id for view in pass_views],
                    "target_candidate_ids": [
                        candidate.candidate_id for candidate in pass_candidates
                    ],
                    "decode_backends": {
                        view.view_id: full_decode_backends[view.view_id]
                        for view in pass_views
                    },
                    "coverage": self._window_coverage(pass_windows, infos),
                    "candidate_count_after_pass": len(current_candidates),
                    "liquid_context_rejection_count": len(
                        liquid_context_rejections
                    ),
                    "liquid_context_rejected_event_ids": [
                        item["event_id"] for item in liquid_context_rejections
                    ],
                    "target_status_after_pass": target_status,
                }
            )
            return target_status

        target_status = execute_pass(
            0, "primary", initial_views, boundary_candidates
        )
        unresolved_ids = {
            item["candidate_id"]
            for item in target_status
            if item["status"] == "needs_third_person_supplement"
        }
        if scout_enabled and unresolved_ids:
            scout_view_ids = [view.view_id for view in supplemental_views]
            scout_anchors, scout_anchor_diagnostics = (
                self._progressive_scout_anchor_representatives(
                    target_status,
                    unresolved_ids,
                    dedup_tolerance_seconds=float(
                        perf.get(
                            "fine_scout_peak_dedup_tolerance_seconds", 0.25
                        )
                    ),
                    cluster_gap_seconds=float(
                        perf.get("fine_scout_peak_cluster_gap_seconds", 60.0)
                    ),
                    representatives_per_cluster=int(
                        perf.get("fine_scout_representatives_per_cluster", 1)
                    ),
                )
            )
            scout_status = [
                {
                    "candidate_id": "SCOUT-REPRESENTATIVES",
                    "first_person_anchor_windows": scout_anchors,
                }
            ]
            scout_windows = self._progressive_anchor_windows(
                scout_status,
                scout_view_ids,
                infos,
                transforms,
                0.0,
                None,
                peak_radius_seconds=float(
                    perf.get("fine_scout_anchor_radius_seconds", 3.0)
                ),
            )
            scout_manifest = manifest.model_copy(update={"views": supplemental_views})
            scout_paths = self._scan_all_views_concurrently(
                scout_manifest,
                infos,
                transforms,
                work_dir / "scout",
                windows=scout_windows,
                sample_fps=float(perf.get("fine_scout_fps", 1.0)),
                image_size=int(perf.get("fine_scout_image_size", 416)),
                keyframes_only=bool(perf.get("fine_scout_keyframes_only", False)),
                phase="fine_scout",
                decode_backends={
                    view.view_id: full_decode_backends[view.view_id]
                    for view in supplemental_views
                },
            )
            scout_config = deepcopy(self.config)
            _, scout_view_report = select_fine_scan_views(
                supplemental_views,
                scout_paths,
                boundary_candidates,
                scout_config,
            )
            ranked_initial, ranked_tail = self._progressive_fine_view_order(
                initial_views + supplemental_views,
                scout_view_report,
                1,
                list(perf.get("fine_preferred_third_person_views") or []),
            )
            supplemental_views = [
                view
                for view in ranked_initial + ranked_tail
                if view.role == ViewRole.THIRD_PERSON
            ]
            scout_rank = {
                view.view_id: index
                for index, view in enumerate(supplemental_views, 1)
            }
            for view in supplemental_views:
                scout_item = scout_view_report.get(view.view_id, {})
                fine_view_report.setdefault(view.view_id, {}).update(
                    {
                        "progressive_initial": False,
                        "progressive_supplemental_rank": scout_rank[view.view_id],
                        "scout_rank": scout_rank[view.view_id],
                        "scout_anchor_frames": int(
                            scout_item.get("anchor_frames", 0)
                        ),
                        "scout_active_anchor_frames": int(
                            scout_item.get("active_anchor_frames", 0)
                        ),
                        "scout_anchor_classes": list(
                            scout_item.get("anchor_classes") or []
                        ),
                        "scout_motion_threshold": scout_item.get(
                            "motion_threshold"
                        ),
                    }
                )
            scout_coverage = self._window_coverage(scout_windows, infos)
            scout_seconds = sum(
                float(item["selected_seconds"])
                for item in scout_coverage.values()
            )
            scout_fps = float(perf.get("fine_scout_fps", 1.0))
            scout_summary = {
                "enabled": True,
                "view_ids": scout_view_ids,
                "sample_fps": scout_fps,
                "image_size": int(perf.get("fine_scout_image_size", 416)),
                "keyframes_only": bool(
                    perf.get("fine_scout_keyframes_only", False)
                ),
                "anchor_radius_seconds": float(
                    perf.get("fine_scout_anchor_radius_seconds", 3.0)
                ),
                "anchor_selection": scout_anchor_diagnostics,
                "window_strategy": "aligned_first_person_peak_windows",
                "pre_merge_window_count_per_view": len(scout_anchors),
                "post_merge_window_count_by_view": {
                    view_id: len(windows)
                    for view_id, windows in scout_windows.items()
                },
                "total_post_merge_window_count": sum(
                    len(windows) for windows in scout_windows.values()
                ),
                "coverage": scout_coverage,
                "selected_seconds": round(scout_seconds, 3),
                "estimated_frames": int(round(scout_seconds * scout_fps)),
                "ranking": [
                    {
                        "view_id": view.view_id,
                        "rank": scout_rank[view.view_id],
                        "anchor_frames": fine_view_report[view.view_id][
                            "scout_anchor_frames"
                        ],
                        "active_anchor_frames": fine_view_report[view.view_id][
                            "scout_active_anchor_frames"
                        ],
                        "anchor_classes": fine_view_report[view.view_id][
                            "scout_anchor_classes"
                        ],
                    }
                    for view in supplemental_views
                ],
            }
        supplemental_batch_size = max(
            1, int(perf.get("fine_supplemental_view_batch_size", 1))
        )
        supplemental_waves = [
            supplemental_views[index : index + supplemental_batch_size]
            for index in range(0, len(supplemental_views), supplemental_batch_size)
        ]
        for pass_index, pass_views in enumerate(supplemental_waves, 1):
            if not unresolved_ids:
                break
            pass_candidates = [
                candidate
                for candidate in boundary_candidates
                if candidate.candidate_id in unresolved_ids
            ]
            target_status = execute_pass(
                pass_index,
                "supplemental",
                pass_views,
                pass_candidates,
                target_status,
            )
            unresolved_ids = {
                item["candidate_id"]
                for item in target_status
                if item["status"] == "needs_third_person_supplement"
            }

        scanned_views = [view for view in fine_views if view.view_id in scanned]
        final_candidates = generate_candidates(scanned_views, detection_paths, self.config)
        all_coverage = self._window_coverage(fine_windows, infos)
        actual_coverage = self._window_coverage(actual_windows, infos)
        all_seconds = sum(float(item["selected_seconds"]) for item in all_coverage.values())
        actual_seconds = sum(
            float(item["selected_seconds"]) for item in actual_coverage.values()
        )
        sample_fps = float(perf["detection_fps"])
        scout_frames = int(scout_summary.get("estimated_frames", 0))
        full_fine_frames = int(round(actual_seconds * sample_fps))
        all_view_frames = int(round(all_seconds * sample_fps))
        report = {
            "schema_version": "visioncortex-progressive-fine-scan/2",
            "enabled": True,
            "selection_rule": (
                "scan the first-person boundary sensor; rank every third-person view "
                "with a low-FPS aligned scout; then exhaust required third-person views "
                "at full FPS only inside narrow first-person anchor windows"
                if scout_enabled
                else "scan first-person plus the configured initial third-person wave; "
                "scan remaining third-person views in bounded shared-model waves only for recalled windows "
                "with a reliable first-person start anchor but no valid dual-role anchor"
            ),
            "eligible_view_ids": [view.view_id for view in fine_views],
            "initial_view_ids": [view.view_id for view in initial_views],
            "preferred_third_person_views": list(
                perf.get("fine_preferred_third_person_views") or []
            ),
            "supplemental_priority": [view.view_id for view in supplemental_views],
            "supplemental_view_batch_size": supplemental_batch_size,
            "supplemental_waves": [
                [view.view_id for view in wave] for wave in supplemental_waves
            ],
            "dynamic_cross_view_scout": scout_summary,
            "scanned_view_ids": [view.view_id for view in scanned_views],
            "not_scanned_view_ids": [
                view.view_id for view in fine_views if view.view_id not in scanned
            ],
            "passes": pass_reports,
            "final_target_status": target_status,
            "unresolved_candidate_ids": sorted(unresolved_ids),
            "stopping_reason": (
                "all_demanded_windows_have_dual_role_anchor"
                if not unresolved_ids
                else "all_eligible_third_person_views_exhausted"
            ),
            "all_view_selected_seconds": round(all_seconds, 3),
            "actual_selected_seconds": round(actual_seconds, 3),
            "avoided_selected_seconds": round(max(0.0, all_seconds - actual_seconds), 3),
            "all_view_estimated_frames": all_view_frames,
            "actual_estimated_frames": full_fine_frames,
            "scout_estimated_frames": scout_frames,
            "total_actual_estimated_frames": full_fine_frames + scout_frames,
            "avoided_estimated_frames": max(
                0, all_view_frames - full_fine_frames - scout_frames
            ),
            "actual_coverage": actual_coverage,
        }
        return detection_paths, scanned_views, final_candidates, report

    @staticmethod
    def _window_coverage(
        windows: dict[str, list[tuple[float, float]]], infos
    ) -> dict[str, dict[str, float | int]]:
        report: dict[str, dict[str, float | int]] = {}
        for view_id, view_windows in windows.items():
            seconds = sum(max(0.0, end - start) for start, end in view_windows) / 1000.0
            duration_seconds = infos[view_id].duration_ms / 1000.0
            report[view_id] = {
                "window_count": len(view_windows),
                "selected_seconds": round(seconds, 3),
                "source_seconds": round(duration_seconds, 3),
                "coverage_ratio": (
                    round(seconds / duration_seconds, 6) if duration_seconds > 0 else 0.0
                ),
            }
        return report

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
        for path in sorted(work_dir.rglob(f"runtime_{phase}_*.json")):
            runtime_role = path.stem.removeprefix(f"runtime_{phase}_")
            if runtime_role not in {role.value for role in ViewRole}:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            report["scan_pass"] = str(path.parent.relative_to(work_dir)).replace("\\", "/")
            role_reports.append(report)
        source_activity = []
        for path in sorted(work_dir.rglob(f"source_activity_{phase}_*.jsonl")):
            activity_role = path.stem.removeprefix(f"source_activity_{phase}_")
            if activity_role not in {role.value for role in ViewRole}:
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
        bottleneck_diagnosis = {
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
                "bottleneck_diagnosis": bottleneck_diagnosis,
                "source_activity": sorted(
                    source_activity, key=lambda item: float(item.get("timestamp", 0.0))
                ),
            },
        )

    def run(self, manifest: RunManifest) -> Path:
        self._run_started_perf = time.perf_counter()
        self._run_started_iso = datetime.now(timezone.utc).isoformat()
        self._active_stage = None
        self._stage_metrics = []
        self._startup_metrics = {}
        self._preprocessing_completed_seconds = None
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
        layout.work = (
            Path(self.config["storage"]["local_cache_root"]).resolve()
            / manifest.experiment_id
            / cache_identity["cache_key"]
        )
        layout.create()
        self._active_layout = layout
        write_json(layout.json_config / "cache_identity.json", cache_identity)
        if storage.get("sync_to_nas") and storage.get("run_output_mode") != "nas_direct":
            nas_root = initialize_nas_archive(self.config, manifest.experiment_id)
            self._publisher = IncrementalArchivePublisher(layout.root, nas_root)
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
                if self._publisher is not None
                else None
            ),
        )
        self._resource_monitor.start()
        try:
            self._status(layout, "preflight", 0.02, "检查输入、模型、视频与磁盘")
            preflight_breakdown: dict[str, Any] = {}
            preflight_step_started = time.perf_counter()
            source_paths = [
                path
                for view in manifest.views
                for path in view_source_files(view) + view_timestamp_files(view)
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
            model_report["video_encoder"] = video_encoder_preflight(
                str(self.config["performance"].get("ffmpeg_video_encoder", "h264_nvenc"))
            )
            preflight_breakdown["model_validation_seconds"] = round(
                time.perf_counter() - preflight_step_started,
                6,
            )
            write_json(layout.json_config / "model_runtime_preflight.json", model_report)
            preflight_step_started = time.perf_counter()
            infos = probe_views(
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
                self._input_volume_report(manifest, infos),
            )
            preflight_step_started = time.perf_counter()
            disk_report = check_disk_capacity(layout.root, list(infos.values()))
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
                        infos,
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
            write_json(layout.json_config / "video_probe.json", {key: value.model_dump(mode="json") for key, value in infos.items()})
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

            self._status(layout, "alignment", 0.08, "最近邻时间戳拟合与视觉锚点校准")
            alignment_runtime: dict[str, Any] = {}
            alignment_step_started = time.perf_counter()
            transforms, _ = build_alignments(manifest.views, infos, self.config)
            alignment_runtime["fit_seconds"] = round(
                time.perf_counter() - alignment_step_started,
                6,
            )
            write_json(
                layout.json_config / "time_alignment.json",
                [transform.model_dump(mode="json") for transform in transforms.values()],
            )
            alignment_step_started = time.perf_counter()
            write_aligned_csv(
                layout.json_config / "aligned_timestamps.csv",
                manifest.views,
                infos,
                transforms,
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
            self._complete_stage(
                layout,
                "alignment",
                [
                    layout.json_config / "time_alignment.json",
                    layout.json_config / "aligned_timestamps.csv",
                    layout.json_config / "alignment_runtime.json",
                ],
            )

            motion_probe_views = self._motion_probe_views(manifest)
            probe_manifest = manifest.model_copy(update={"views": motion_probe_views})
            selected_probe_ids = {view.view_id for view in motion_probe_views}
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
                "Selecting and running the fastest real-source sparse motion probe",
            )
            sparse_strategy_path = layout.json_config / "motion_probe_sparse_strategy.json"
            configured_sparse_strategy = str(
                self.config["performance"].get(
                    "motion_probe_sparse_strategy", "indexed_seek"
                )
            )
            if configured_sparse_strategy == "auto":
                sentinel = motion_probe_views[0]
                sparse_report = benchmark_sparse_decode_strategy(
                    sentinel,
                    infos[sentinel.view_id],
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
            write_json(sparse_strategy_path, sparse_report)
            selected_probe_ids = {view.view_id for view in motion_probe_views}
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
                "哨兵视角低分辨率运动探针；此阶段CUDA计算低占用属于预期",
            )
            motion_paths = self._scan_all_views_concurrently(
                probe_manifest,
                infos,
                transforms,
                layout.work / "motion-probe",
                sample_fps=float(self.config["performance"]["motion_probe_fps"]),
                image_size=int(self.config["performance"].get("motion_probe_max_width", 96)),
                keyframes_only=bool(
                    self.config["performance"].get("motion_probe_keyframes_only", True)
                ),
                phase="motion_probe",
            )
            self._archive_scan_runtime(layout, layout.work / "motion-probe", "motion_probe")
            probe_config = json.loads(json.dumps(self.config))
            probe_config["performance"]["motion_probe_require_objects"] = False
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
                motion_probe_views, motion_paths, probe_config
            )
            motion_candidates = fuse_motion_probe_candidates(
                raw_motion_candidates, probe_config
            )
            safety_fallback_used = False
            if not motion_candidates:
                safety_fallback_used = True
                motion_candidates = generate_motion_safety_candidates(
                    motion_probe_views, motion_paths, probe_config
                )
            if not motion_candidates:
                raise RuntimeError("输入视频没有产生任何可读运动帧，无法建立实验候选窗口")
            motion_windows = self._fine_windows(
                motion_candidates,
                infos,
                transforms,
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
                    "sentinel_views": [view.view_id for view in motion_probe_views],
                    "raw_candidate_count": len(raw_motion_candidates),
                    "fused_candidate_count": len(motion_candidates),
                    "safety_fallback_used": safety_fallback_used,
                    "coverage": self._window_coverage(motion_windows, infos),
                    "candidates": [
                        item.model_dump(mode="json") for item in motion_candidates
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

            reuse_motion_probe = bool(
                self.config["performance"].get("coarse_reuse_motion_probe", False)
            ) and bool(self.config["performance"].get("motion_probe_run_yolo", False))
            coarse_scan_views = (
                motion_probe_views
                if reuse_motion_probe
                else self._coarse_scan_views(manifest)
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
                f"候选窗口内 {self.config['performance']['coarse_detection_fps']} FPS YOLO粗筛",
            )
            if reuse_motion_probe:
                coarse_paths = motion_paths
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
                    infos,
                    transforms,
                    layout.work / "detections-coarse",
                    windows=motion_windows,
                    sample_fps=float(self.config["performance"]["coarse_detection_fps"]),
                    image_size=int(self.config["performance"]["coarse_image_size"]),
                    keyframes_only=bool(self.config["performance"]["coarse_keyframes_only"]),
                    phase="coarse",
                )
                self._archive_scan_runtime(
                    layout, layout.work / "detections-coarse", "coarse"
                )
            coarse_views = [
                view for view in coarse_scan_views if view.role == ViewRole.FIRST_PERSON
            ]
            coarse_config = json.loads(json.dumps(self.config))
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
                coarse_views, coarse_paths, coarse_config
            )
            if not coarse_candidates:
                fallback_views = [
                    view for view in coarse_scan_views if view.role == ViewRole.THIRD_PERSON
                ]
                fallback_paths = {view.view_id: coarse_paths[view.view_id] for view in fallback_views}
                coarse_candidates = generate_coarse_activity_candidates(
                    fallback_views, fallback_paths, coarse_config
                )
            boundary_candidates, boundary_report = refine_motion_candidates_with_coarse(
                motion_candidates,
                coarse_candidates,
                self.config,
            )
            boundary_report["coarse_scan_view_ids"] = [
                view.view_id for view in coarse_scan_views
            ]
            write_json(
                layout.json_config / "coarse_boundary_refinement.json",
                boundary_report,
            )
            fine_views, fine_view_report = select_fine_scan_views(
                manifest.views, coarse_paths, boundary_candidates, self.config
            )
            progressive_enabled = bool(
                self.config["performance"].get("fine_progressive_cross_view", False)
            )
            if progressive_enabled:
                initial_fine_views, supplemental_fine_views = (
                    self._progressive_fine_view_order(
                        fine_views,
                        fine_view_report,
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
                        for view in fine_views
                        if view.role == ViewRole.FIRST_PERSON
                    ]
                    supplemental_fine_views = [
                        view
                        for view in fine_views
                        if view.role == ViewRole.THIRD_PERSON
                    ]
                initial_fine_ids = {view.view_id for view in initial_fine_views}
                supplemental_rank = {
                    view.view_id: rank
                    for rank, view in enumerate(supplemental_fine_views, 1)
                }
                for view in fine_views:
                    item = fine_view_report[view.view_id]
                    item["progressive_initial"] = view.view_id in initial_fine_ids
                    item["progressive_supplemental_rank"] = supplemental_rank.get(
                        view.view_id
                    )
            write_json(layout.json_config / "fine_view_selection.json", fine_view_report)
            coarse_artifacts = [
                layout.json_config / "coarse_boundary_refinement.json",
                layout.json_config / "fine_view_selection.json",
            ]
            coarse_runtime = layout.json_config / "scan_runtime_coarse.json"
            if coarse_runtime.exists():
                coarse_artifacts.append(coarse_runtime)
            reuse_report = layout.json_config / "coarse_reuse_motion_probe.json"
            if reuse_report.exists():
                coarse_artifacts.append(reuse_report)
            self._complete_stage(layout, "candidate_coarse", coarse_artifacts)
            fine_windows = self._fine_windows(boundary_candidates, infos, transforms)
            fine_windows = {view.view_id: fine_windows[view.view_id] for view in fine_views}
            fine_coverage = self._window_coverage(fine_windows, infos)
            fine_sample_fps = float(self.config["performance"]["detection_fps"])
            fine_window_report = {
                "schema_version": "visioncortex-fine-scan-windows/2",
                "strategy": (
                    "progressive_cross_view" if progressive_enabled else "all_selected_views"
                ),
                "eligible_view_ids": [view.view_id for view in fine_views],
                "selected_view_ids": [view.view_id for view in fine_views],
                "sample_fps": fine_sample_fps,
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
                "estimated_sampled_frames": int(
                    round(
                        sum(float(item["selected_seconds"]) for item in fine_coverage.values())
                        * fine_sample_fps
                    )
                ),
            }
            write_json(layout.json_config / "fine_scan_windows.json", fine_window_report)
            eligible_fine_ids = {view.view_id for view in fine_views}
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = (
                    "fine_pending" if view.view_id in eligible_fine_ids else "fine_not_selected"
                )
            progressive_report = None
            self._status(
                layout,
                "candidate_fine",
                0.48,
                (
                    "渐进精扫：先核验主双视角，缺失窗口按对齐时间戳补扫"
                    if progressive_enabled
                    else "候选窗精扫并收紧动作边界"
                ),
            )
            fine_work_dir = layout.work / "detections-fine"
            if progressive_enabled:
                detection_paths, scanned_fine_views, candidates, progressive_report = (
                    self._run_progressive_fine_scan(
                        manifest,
                        fine_views,
                        fine_view_report,
                        boundary_candidates,
                        fine_windows,
                        infos,
                        transforms,
                        fine_work_dir,
                    )
                )
                write_json(
                    layout.json_config / "progressive_fine_scan.json",
                    progressive_report,
                )
                write_json(
                    layout.json_config / "fine_view_selection.json",
                    fine_view_report,
                )
                fine_window_report.update(
                    {
                        "selected_view_ids": progressive_report["scanned_view_ids"],
                        "coverage": progressive_report["actual_coverage"],
                        "estimated_sampled_frames": progressive_report[
                            "total_actual_estimated_frames"
                        ],
                        "full_fine_estimated_frames": progressive_report[
                            "actual_estimated_frames"
                        ],
                        "scout_estimated_frames": progressive_report[
                            "scout_estimated_frames"
                        ],
                        "full_pool_coverage": fine_coverage,
                        "full_pool_estimated_sampled_frames": progressive_report[
                            "all_view_estimated_frames"
                        ],
                        "avoided_selected_seconds": progressive_report[
                            "avoided_selected_seconds"
                        ],
                        "avoided_estimated_frames": progressive_report[
                            "avoided_estimated_frames"
                        ],
                    }
                )
                write_json(
                    layout.json_config / "fine_scan_windows.json", fine_window_report
                )
            else:
                fine_manifest = manifest.model_copy(update={"views": fine_views})
                detection_paths = self._scan_all_views_concurrently(
                    fine_manifest,
                    infos,
                    transforms,
                    fine_work_dir,
                    windows=fine_windows,
                    sample_fps=fine_sample_fps,
                    phase="fine",
                )
                scanned_fine_views = fine_views
                candidates = generate_candidates(
                    scanned_fine_views, detection_paths, self.config
                )
            scanned_fine_ids = {view.view_id for view in scanned_fine_views}
            for view in fine_views:
                if view.view_id not in scanned_fine_ids:
                    self._view_runtime[view.view_id]["state"] = (
                        "fine_not_scanned_no_remaining_evidence_gap"
                    )
            self._archive_scan_runtime(
                layout,
                fine_work_dir,
                "fine",
                progressive_report=progressive_report,
            )
            scout_work_dir = fine_work_dir / "scout"
            if scout_work_dir.is_dir():
                self._archive_scan_runtime(
                    layout,
                    scout_work_dir,
                    "fine_scout",
                )
            if self.config["archive"].get("keep_debug_candidates"):
                write_json(
                    layout.json_config / "candidate_layer.json",
                    [candidate.model_dump(mode="json") for candidate in candidates],
                )
            fine_artifacts = [
                layout.json_config / "scan_runtime_fine.json",
                layout.json_config / "fine_scan_windows.json",
            ]
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

            self._status(layout, "candidate_audit", 0.68, "持续性、动作密度与跨视角一致性审计")
            events, rejected = audit_candidates(candidates, transforms, self.config)
            rejected.extend(refine_liquid_events_with_context(events, detection_paths))
            raw_segments = build_experiment_segments(
                events, manifest.views, self.config, coarse_windows=boundary_candidates
            )
            normalized_segments = normalize_experiment_segments(
                raw_segments, events, manifest.views, self.config
            )
            segments, formal_segment_receipts = prepare_formal_experiment_segments(
                normalized_segments,
                events,
                manifest.views,
                boundary_candidates,
                self.config,
            )
            groups = build_experiment_groups(segments, events, manifest.views, self.config)
            self._preprocessing_completed_seconds = round(time.perf_counter() - self._run_started_perf, 6)
            write_json(
                layout.json_config / "audit_layer.json",
                {
                    "events": [event.model_dump(mode="json") for event in events],
                    "rejected": rejected,
                    "raw_segments": [
                        segment.model_dump(mode="json") for segment in raw_segments
                    ],
                    "normalized_segments": [
                        segment.model_dump(mode="json")
                        for segment in normalized_segments
                    ],
                    "segments": [segment.model_dump(mode="json") for segment in segments],
                    "formal_segment_receipts": formal_segment_receipts,
                    "experiment_groups": [group.model_dump(mode="json") for group in groups],
                },
            )
            boundary_precheck = self._run_boundary_precheck(layout, groups)
            self._complete_stage(
                layout,
                "candidate_audit",
                [
                    layout.json_config / "audit_layer.json",
                    layout.json_config / "boundary_precheck.json",
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
                key_events = select_key_events(groups, segments, events, self.config)
                key_selection_path = (
                    layout.json_config / "key_material_selection_preview.json"
                )
                selection_report = _key_material_selection_report(
                    groups, segments, events
                )
                write_json(key_selection_path, selection_report)
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
                            "passed" if boundary_precheck["passed"] else "failed"
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
                        "experiment_group_count": len(groups),
                        "selected_key_event_count": len(key_events),
                        "experiment_groups": [
                            group.model_dump(mode="json") for group in groups
                        ],
                        "selected_key_event_ids": [
                            event.event_id for event in key_events
                        ],
                        "boundary_precheck": boundary_precheck,
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
                run_metrics = self._metrics(key_events, groups)
                run_metrics["run_mode"] = "preprocessing_acceptance_only"
                run_metrics["formal_archive_promotion_allowed"] = False
                write_json(layout.json_config / "run_metrics.json", run_metrics)
                self._complete_stage(
                    layout,
                    "preprocessing_acceptance",
                    [
                        acceptance_path,
                        key_selection_path,
                        layout.json_config / "run_metrics.json",
                    ],
                )
                return layout.root

            self._status(layout, "experiment_understanding", 0.72, "用完整有界双视角故事板命名实验并核验连续性")
            analyze_experiment_groups(
                layout, groups, segments, events, manifest.views, infos, transforms, self.config
            )
            group_understanding_path = layout.json_config / "experiment_group_understanding.json"
            write_json(
                group_understanding_path,
                {
                    "schema_version": "visioncortex-experiment-group-understanding/1",
                    "groups": [group.model_dump(mode="json") for group in groups],
                },
            )
            write_json(layout.json_config / "run_metrics_live.json", self._metrics(events, groups))
            key_events = select_key_events(groups, segments, events, self.config)
            key_selection_path = layout.json_config / "key_material_selection.json"
            write_json(
                key_selection_path,
                _key_material_selection_report(groups, segments, events),
            )
            self._complete_stage(
                layout,
                "experiment_understanding",
                [
                    group_understanding_path,
                    key_selection_path,
                    layout.json_config / "run_metrics_live.json",
                ],
            )

            self._status(layout, "experiment_clips", 0.78, "按模型实验名归档第一/第三/并排三份有界视频")
            materialize_experiment_clips(
                layout,
                groups,
                segments,
                events,
                manifest.views,
                infos,
                transforms,
                self.config,
                publisher=self._publisher,
            )
            self._complete_stage(
                layout,
                "experiment_clips",
                [
                    layout.experiment_clips,
                    layout.json_config / "experiment_clip_materialization_runtime.json",
                ],
            )

            self._status(layout, "key_materials", 0.84, "按实验组提取去重后的五类对齐关键素材")
            materialize_key_materials(
                layout,
                key_events,
                groups,
                manifest.views,
                infos,
                transforms,
                detection_paths,
                self.config,
                publisher=self._publisher,
            )
            self._complete_stage(
                layout,
                "key_materials",
                [
                    layout.key_materials,
                    layout.json_config / "key_material_materialization_runtime.json",
                ],
            )

            self._status(layout, "mllm", 0.92, "调用豆包理解去重后的关键动作当前/下一步骤")
            analyze_key_materials(layout, key_events, self.config)
            key_understanding_path = (
                layout.json_config / "key_material_model_understanding.json"
            )
            write_json(
                key_understanding_path,
                {
                    "schema_version": "visioncortex-key-material-understanding/1",
                    "events": [
                        event.model_dump(mode="json") for event in key_events
                    ],
                },
            )
            write_json(
                layout.json_config / "run_metrics_live.json",
                self._metrics(key_events, groups),
            )
            refresh_key_material_metadata(layout, key_events, groups, transforms)
            self._complete_stage(
                layout,
                "mllm",
                [
                    layout.key_materials,
                    key_understanding_path,
                    layout.json_config / "run_metrics_live.json",
                ],
            )
            physical_changes = build_physical_change_log(events)

            self._status(layout, "package", 0.96, "归档证据包并执行 evidence-package-eval")
            summary = finalize_archive(
                layout,
                manifest,
                infos,
                transforms,
                events,
                segments,
                groups,
                key_events,
                physical_changes,
                rejected,
                self.config,
                disk_report,
            )
            # Validate the archive-level experiment groups. A continuous group
            # may contain multiple atomic segments but must count as one bounded
            # experiment in the user-facing output and boundary evaluation.
            self._run_sidecar_validation(layout, manifest, groups)
            self._run_quality_acceptance(layout, groups, key_events)
            self._complete_stage(layout, "package", [layout.json_config])
            self._status(layout, "daily_report", 0.98, "从已验收证据生成实验室日报并执行一致性校验")
            generate_daily_report_archive(layout, summary, self._metrics(events, groups), self.config)
            self._complete_stage(
                layout,
                "daily_report",
                [layout.daily_reports, layout.professional_pdfs],
            )
            if not self.config["archive"].get("keep_debug_candidates") and layout.work.exists():
                shutil.rmtree(layout.work)
            self._status(layout, "completed", 1.0, "处理完成")
            run_metrics = self._metrics(events, groups)
            # Refresh the package after the final stage has been closed so the
            # durable evidence package and the standalone ledger contain the
            # same complete stage durations and provider-reported token usage.
            summary = finalize_archive(
                layout,
                manifest,
                infos,
                transforms,
                events,
                segments,
                groups,
                key_events,
                physical_changes,
                rejected,
                self.config,
                disk_report,
                run_metrics=run_metrics,
            )
            # Refresh the report with the closed daily_report stage duration and
            # final provider-reported token ledger. This remains deterministic.
            generate_daily_report_archive(layout, summary, run_metrics, self.config)
            self._complete_stage(
                layout,
                "completed",
                [
                    layout.experiment_clips,
                    layout.key_materials,
                    layout.json_config,
                    layout.daily_reports,
                    layout.professional_pdfs,
                ],
            )
            return layout.root
        except Exception as exc:
            self._status(layout, "failed", 1.0, f"{type(exc).__name__}: {exc}")
            write_json(
                layout.json_config / "run_metrics.json",
                self._metrics(locals().get("events", []), locals().get("groups", [])),
            )
            if self._publisher is not None:
                self._publisher.publish_directory("JSON-Config-Files")
            raise
        finally:
            if self._resource_monitor is not None:
                self._resource_monitor.stop()
                if self._publisher is not None:
                    self._publisher.publish_file(layout.json_config / "resource_telemetry.json")
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _acquire_lock(lock_path: Path) -> None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = lock_path.read_text(encoding="utf-8", errors="replace") if lock_path.is_file() else "unknown"
            raise RuntimeError(f"该实验已有运行实例: {owner}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))

    def _fine_windows(
        self,
        candidates,
        infos,
        transforms,
        padding_seconds: float | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        padding = float(
            self.config["performance"]["fine_window_padding_seconds"]
            if padding_seconds is None
            else padding_seconds
        ) * 1000.0
        merge_gap = max(
            0.0,
            float(
                self.config["performance"].get(
                    "fine_window_merge_gap_seconds", 0.0
                )
            )
            * 1000.0,
        )
        grouped: dict[str, list[tuple[float, float]]] = {view_id: [] for view_id in infos}
        for candidate in candidates:
            global_start = candidate.global_start_ms - padding
            global_end = candidate.global_end_ms + padding
            for view_id, info in infos.items():
                local_start = max(0.0, transforms[view_id].to_local(global_start))
                local_end = min(info.duration_ms, transforms[view_id].to_local(global_end))
                if local_end > local_start:
                    grouped[view_id].append((local_start, local_end))
        merged: dict[str, list[tuple[float, float]]] = {}
        for view_id, windows in grouped.items():
            result: list[list[float]] = []
            for start, end in sorted(windows):
                if result and start <= result[-1][1] + merge_gap:
                    result[-1][1] = max(result[-1][1], end)
                else:
                    result.append([start, end])
            merged[view_id] = [(item[0], item[1]) for item in result]
        return merged

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
            "status": "dry_run",
            "current_step": "戴手套的手接触移液器",
            "next_step": "未知",
            "confidence": 0.9,
        },
    )
    image = np.full((360, 640, 3), 35, dtype=np.uint8)
    cv2.putText(image, "DRY RUN - ALIGNED KEY FRAME", (45, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    for view in views:
        path = layout.key_frames / event.event_id / f"{view.view_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        event.key_frames[view.view_id] = path.relative_to(layout.root).as_posix()
        event.key_clips[view.view_id] = f"dry-run://{view.view_id}/key-clip"
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
    )
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
    write_json(layout.root / "run_status.json", {"stage": "completed", "progress": 1.0, "dry_run": True})
    return layout.root
