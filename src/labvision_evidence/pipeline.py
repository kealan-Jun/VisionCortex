from __future__ import annotations

import hashlib
import json
import os
import shutil
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
from .grouping import build_experiment_groups, select_key_events
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
from .video_io import check_disk_capacity, probe_view, view_source_files, view_timestamp_files
from .storage import IncrementalArchivePublisher, initialize_nas_archive
from .telemetry import ResourceMonitor
from .validation import validate_experiment_and_material_quality


ProgressCallback = Callable[[str, float, str], None]


def _noop_progress(stage: str, progress: float, message: str) -> None:
    del stage, progress, message


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
        models[role] = {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256_file(path) if path.is_file() else None,
        }

    inputs = []
    for view in manifest.views:
        source_files = [path.resolve() for path in view_source_files(view)]
        clock_files = [path.resolve() for path in view_timestamp_files(view)]
        inputs.append(
            {
                "view_id": view.view_id,
                "role": view.role.value,
                "input_mode": "segmented" if view.segments else "continuous_file",
                "videos": [
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size if path.is_file() else None,
                        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
                    }
                    for path in source_files
                ],
                "timestamps_csvs": [
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size if path.is_file() else None,
                        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
                    }
                    for path in clock_files
                ],
            }
        )

    stable_config = deepcopy(config)
    stable_config.get("project", {}).pop("output_root", None)
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
            "completed_stages": [item["stage"] for item in self._stage_metrics],
        }
        write_json(
            layout.root / "run_status.json",
            status_payload,
        )
        if self.config.get("storage", {}).get("run_output_mode") == "nas_direct":
            write_json(layout.json_config / "pipeline_status.json", status_payload)
        if self._publisher is not None:
            self._publisher.publish_status(status_payload)

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
            perf.get("coarse_decode_lanes" if phase == "coarse" else "fine_decode_lanes", [])
        )
        if not lanes:
            lanes = ["cuda" if perf.get("ffmpeg_hwaccel") else "cpu"] * len(manifest.views)
        if len(lanes) < len(manifest.views):
            lanes.extend([lanes[-1]] * (len(manifest.views) - len(lanes)))
        kwargs["decode_backends"] = {
            view.view_id: lanes[index] for index, view in enumerate(manifest.views)
        }
        if (
            phase == "coarse"
            and perf.get("synchronized_segment_waves")
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
        if not self.config["performance"].get("concurrent_role_scanners", True) or len(groups) == 1:
            result = {}
            for group in groups:
                result.update(scan_videos(group, infos, transforms, work_dir, self.config, **kwargs))
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

    @staticmethod
    def _archive_scan_runtime(layout: ArchiveLayout, work_dir: Path, phase: str) -> None:
        role_reports = []
        for path in sorted(work_dir.glob(f"runtime_{phase}_*.json")):
            role_reports.append(json.loads(path.read_text(encoding="utf-8")))
        source_activity = []
        for path in sorted(work_dir.glob(f"source_activity_{phase}_*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                source_activity.extend(
                    json.loads(line) for line in handle if line.strip()
                )
        write_json(
            layout.json_config / f"scan_runtime_{phase}.json",
            {
                "schema_version": "visioncortex-scan-runtime/1",
                "phase": phase,
                "role_reports": role_reports,
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
            layout = ArchiveLayout(Path(active_archive).resolve())
        else:
            output_root = Path(self.config["project"]["output_root"]).resolve()
            layout = ArchiveLayout(output_root / manifest.experiment_id)
        cache_identity = build_cache_identity(self.config, manifest)
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
            for view in manifest.views:
                for source in view_source_files(view):
                    if not source.is_file():
                        raise FileNotFoundError(f"视频不存在: {source}")
                for clock in view_timestamp_files(view):
                    if not clock.is_file():
                        raise FileNotFoundError(f"时间戳 CSV 不存在: {clock}")
            model_report = validate_models(self.config)
            write_json(layout.json_config / "model_runtime_preflight.json", model_report)
            with ThreadPoolExecutor(max_workers=min(6, len(manifest.views))) as executor:
                probed = list(executor.map(lambda view: (view.view_id, probe_view(view)), manifest.views))
            infos = dict(probed)
            disk_report = check_disk_capacity(layout.root, list(infos.values()))
            write_json(layout.json_config / "video_probe.json", {key: value.model_dump(mode="json") for key, value in infos.items()})
            if self._publisher is not None:
                self._publisher.publish_directory("JSON-Config-Files")

            self._status(layout, "alignment", 0.08, "最近邻时间戳拟合与视觉锚点校准")
            transforms, _ = build_alignments(manifest.views, infos, self.config)
            write_json(
                layout.json_config / "time_alignment.json",
                [transform.model_dump(mode="json") for transform in transforms.values()],
            )
            write_aligned_csv(
                layout.json_config / "aligned_timestamps.csv",
                manifest.views,
                infos,
                transforms,
                float(self.config["alignment"]["aligned_timestamps_fps"]),
            )
            if self._publisher is not None:
                self._publisher.publish_directory("JSON-Config-Files")

            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = "coarse_running"
            self._status(
                layout,
                "candidate_coarse",
                0.16,
                f"{self.config['performance']['coarse_detection_fps']} FPS 全量粗筛：并行解码、YOLO GPU 批推理和 ByteSORT 跟踪",
            )
            coarse_paths = self._scan_all_views_concurrently(
                manifest,
                infos,
                transforms,
                layout.work / "detections-coarse",
                sample_fps=float(self.config["performance"]["coarse_detection_fps"]),
                image_size=int(self.config["performance"]["coarse_image_size"]),
                keyframes_only=bool(self.config["performance"]["coarse_keyframes_only"]),
                phase="coarse",
            )
            self._archive_scan_runtime(layout, layout.work / "detections-coarse", "coarse")
            coarse_views = [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON]
            coarse_config = json.loads(json.dumps(self.config))
            coarse_config["segmentation"]["event_merge_gap_seconds"] = max(
                float(coarse_config["segmentation"]["event_merge_gap_seconds"]),
                1.5 / float(self.config["performance"]["coarse_detection_fps"]),
            )
            coarse_config["segmentation"]["min_event_observations"] = 2
            coarse_candidates = generate_motion_burst_candidates(
                coarse_views, coarse_paths, coarse_config
            )
            if not coarse_candidates:
                fallback_views = [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON]
                fallback_paths = {view.view_id: coarse_paths[view.view_id] for view in fallback_views}
                coarse_candidates = generate_coarse_activity_candidates(
                    fallback_views, fallback_paths, coarse_config
                )
            fine_views, fine_view_report = select_fine_scan_views(
                manifest.views, coarse_paths, coarse_candidates, self.config
            )
            write_json(layout.json_config / "fine_view_selection.json", fine_view_report)
            fine_windows = self._fine_windows(coarse_candidates, infos, transforms)
            fine_windows = {view.view_id: fine_windows[view.view_id] for view in fine_views}
            fine_manifest = manifest.model_copy(update={"views": fine_views})
            selected_fine_ids = {view.view_id for view in fine_views}
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = (
                    "fine_running" if view.view_id in selected_fine_ids else "fine_not_selected"
                )
            self._status(layout, "candidate_fine", 0.48, "候选窗 8 FPS 精扫并收紧动作边界")
            detection_paths = self._scan_all_views_concurrently(
                fine_manifest,
                infos,
                transforms,
                layout.work / "detections-fine",
                windows=fine_windows,
                sample_fps=float(self.config["performance"]["detection_fps"]),
                phase="fine",
            )
            self._archive_scan_runtime(layout, layout.work / "detections-fine", "fine")
            candidates = generate_candidates(fine_views, detection_paths, self.config)
            if self.config["archive"].get("keep_debug_candidates"):
                write_json(
                    layout.json_config / "candidate_layer.json",
                    [candidate.model_dump(mode="json") for candidate in candidates],
                )

            self._status(layout, "candidate_audit", 0.68, "持续性、动作密度与跨视角一致性审计")
            events, rejected = audit_candidates(candidates, transforms, self.config)
            rejected.extend(refine_liquid_events_with_context(events, detection_paths))
            segments = build_experiment_segments(
                events, manifest.views, self.config, coarse_windows=coarse_candidates
            )
            groups = build_experiment_groups(segments, events, manifest.views, self.config)
            self._preprocessing_completed_seconds = round(time.perf_counter() - self._run_started_perf, 6)
            write_json(
                layout.json_config / "audit_layer.json",
                {
                    "events": [event.model_dump(mode="json") for event in events],
                    "rejected": rejected,
                    "segments": [segment.model_dump(mode="json") for segment in segments],
                    "experiment_groups": [group.model_dump(mode="json") for group in groups],
                },
            )
            if self._publisher is not None:
                self._publisher.publish_directory("JSON-Config-Files")

            self._status(layout, "experiment_understanding", 0.72, "用完整有界双视角故事板命名实验并核验连续性")
            analyze_experiment_groups(
                layout, groups, segments, events, manifest.views, infos, transforms, self.config
            )
            write_json(layout.json_config / "run_metrics_live.json", self._metrics(events, groups))
            if self._publisher is not None:
                self._publisher.publish_file(layout.json_config / "run_metrics_live.json")
            key_events = select_key_events(groups, segments, events, self.config)

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
            if self._publisher is not None:
                self._publisher.publish_directory("Experiment-Clips")

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
            if self._publisher is not None:
                self._publisher.publish_directory("Key-Materials")

            self._status(layout, "mllm", 0.92, "调用豆包理解去重后的关键动作当前/下一步骤")
            analyze_key_materials(layout, key_events, self.config)
            write_json(
                layout.json_config / "run_metrics_live.json",
                self._metrics(key_events, groups),
            )
            if self._publisher is not None:
                self._publisher.publish_file(layout.json_config / "run_metrics_live.json")
            refresh_key_material_metadata(layout, key_events, groups, transforms)
            if self._publisher is not None:
                self._publisher.publish_directory("Key-Materials")
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
            self._status(layout, "daily_report", 0.98, "从已验收证据生成实验室日报并执行一致性校验")
            generate_daily_report_archive(layout, summary, self._metrics(events, groups), self.config)
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
            if self._publisher is not None:
                for directory in (
                    "Experiment-Clips",
                    "Key-Materials",
                    "JSON-Config-Files",
                    "Lab-Daily-Reports",
                    "Professional-PDFs",
                ):
                    self._publisher.publish_directory(directory)
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

    def _fine_windows(self, candidates, infos, transforms) -> dict[str, list[tuple[float, float]]]:
        padding = float(self.config["performance"]["fine_window_padding_seconds"]) * 1000.0
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
                if result and start <= result[-1][1] + padding:
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
        archive_folder="001_移液操作实验_Pipetting-Operation-Experiment",
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
