from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .archive import (
    ArchiveLayout,
    _rerender_curated_participant_annotations,
    finalize_archive,
    materialize_experiment_clips,
    materialize_key_materials,
    write_json,
)
from .daily_reports import generate_daily_report_archive
from .detection import validate_models
from .schema_contracts import write_archive_contract_manifest
from .schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    BoxEvidence,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    FrameEvidence,
    PhysicalChange,
    RunManifest,
    ViewInput,
    ViewRole,
)
from .telemetry import ResourceMonitor
from .video_io import probe_views


LOCAL_ACCEPTANCE_SCHEMA = "visioncortex-local-six-view-acceptance/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_scene(frame_index: int, view_index: int, fps: int) -> np.ndarray:
    width, height = 640, 360
    image = np.full((height, width, 3), (32, 38, 44), dtype=np.uint8)
    shift = (view_index - 2) * 8
    cv2.rectangle(image, (0, 285), (width, height), (70, 75, 78), -1)
    cv2.putText(
        image,
        f"VISIONCORTEX LOCAL VIEW {view_index + 1}",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 230, 235),
        2,
    )
    seconds = frame_index / fps
    liquid_top = 235 - round(18 * np.sin(seconds * 0.8 + view_index * 0.15))
    vessel_left, vessel_right = 365 + shift, 505 + shift
    cv2.rectangle(image, (vessel_left, 125), (vessel_right, 285), (210, 220, 230), 3)
    cv2.rectangle(image, (vessel_left + 5, liquid_top), (vessel_right - 5, 280), (190, 105, 35), -1)
    cv2.line(image, (vessel_left + 5, liquid_top), (vessel_right - 5, liquid_top), (255, 210, 90), 3)
    cv2.rectangle(image, (88 + shift, 248), (245 + shift, 278), (228, 232, 235), -1)
    cv2.putText(image, "WEIGHING PAPER", (92 + shift, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1)
    hand_x = 110 + round(185 * min(1.0, max(0.0, seconds / 18.0))) + shift
    hand_y = 165 + round(16 * np.sin(seconds * 1.2 + view_index * 0.2))
    cv2.ellipse(image, (hand_x, hand_y), (52, 34), 12, 0, 360, (215, 130, 45), -1)
    cv2.line(image, (hand_x + 20, hand_y + 8), (hand_x + 150, hand_y + 72), (235, 235, 240), 8)
    cv2.line(image, (hand_x + 150, hand_y + 72), (hand_x + 165, hand_y + 92), (90, 190, 245), 4)
    cv2.rectangle(image, (245 + shift, 292), (345 + shift, 335), (28, 30, 32), -1)
    cv2.rectangle(image, (258 + shift, 302), (332 + shift, 321), (65, 185, 110), -1)
    cv2.putText(
        image,
        f"t={seconds:05.2f}s",
        (525, 338),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
    )
    return image


def _write_source_video(path: Path, view_index: int, *, fps: int, seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (640, 360)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create local acceptance source: {path}")
    try:
        for frame_index in range(fps * seconds):
            writer.write(_draw_scene(frame_index, view_index, fps))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Local acceptance source is empty: {path}")


def _participants(action: ActionType) -> list[str]:
    return {
        ActionType.HAND_OBJECT_CONTACT: ["gloved_hand", "paper"],
        ActionType.OBJECT_MOVEMENT: ["gloved_hand", "spatula"],
        ActionType.LIQUID_MOVEMENT: ["gloved_hand", "pipette", "tube"],
        ActionType.CONTAINER_STATE_CHANGE: [
            "gloved_hand",
            "reagent_bottle_open",
            "bottle_cap",
        ],
        ActionType.DEVICE_PANEL_OPERATION: ["gloved_hand", "balance"],
        ActionType.PIPETTE_TRANSFER_OPERATION: [
            "gloved_hand",
            "pipette",
            "spearhead",
            "tube",
        ],
    }[action]


def _state_transition(action: ActionType) -> tuple[str, str]:
    return {
        ActionType.HAND_OBJECT_CONTACT: ("not_contacting", "contacting"),
        ActionType.OBJECT_MOVEMENT: ("stationary", "relocated"),
        ActionType.LIQUID_MOVEMENT: ("liquid_at_source", "liquid_at_target"),
        ActionType.CONTAINER_STATE_CHANGE: ("closed", "open"),
        ActionType.DEVICE_PANEL_OPERATION: ("idle", "activated"),
        ActionType.PIPETTE_TRANSFER_OPERATION: (
            "pipette_loaded_at_source",
            "target_dosed",
        ),
    }[action]


def _boxes(action: ActionType, view_index: int) -> list[BoxEvidence]:
    shift = view_index * 0.006
    templates = {
        "gloved_hand": (0.31 + shift, 0.31, 0.49 + shift, 0.59),
        "paper": (0.13 + shift, 0.68, 0.39 + shift, 0.80),
        "spatula": (0.43 + shift, 0.42, 0.66 + shift, 0.55),
        "pipette": (0.44 + shift, 0.38, 0.70 + shift, 0.70),
        "spearhead": (0.68 + shift, 0.66, 0.73 + shift, 0.81),
        "tube": (0.58 + shift, 0.56, 0.71 + shift, 0.83),
        "reagent_bottle_open": (0.56 + shift, 0.35, 0.79 + shift, 0.82),
        "bottle_cap": (0.51 + shift, 0.38, 0.58 + shift, 0.47),
        "balance": (0.38 + shift, 0.79, 0.56 + shift, 0.94),
    }
    return [
        BoxEvidence(
            class_id=index,
            class_name=name,
            confidence=0.96 - index * 0.01,
            xyxy_norm=templates[name],
            track_id=100 + index,
            roi_motion=24.0,
        )
        for index, name in enumerate(_participants(action))
    ]


def _write_detection_ledgers(
    root: Path,
    views: list[ViewInput],
    events: list[EvidenceEvent],
    *,
    fps: int,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for view_index, view in enumerate(views):
        path = root / f"{view.view_id}.detections.jsonl"
        rows = []
        for event in events:
            rows.append(
                FrameEvidence(
                    view_id=view.view_id,
                    role=view.role,
                    frame_index=round(event.key_global_ms / 1000.0 * fps),
                    local_ms=event.key_global_ms,
                    global_ms=event.key_global_ms,
                    width=640,
                    height=360,
                    motion_score=25.0,
                    detections=_boxes(event.action_type, view_index),
                ).model_dump_json()
            )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        paths[view.view_id] = path
    return paths


def _original_references(
    layout: ArchiveLayout,
    manifest: RunManifest,
) -> dict[str, Any]:
    records = []
    total = 0
    for view in manifest.views:
        assert view.video is not None
        size = view.video.stat().st_size
        total += size
        records.append(
            {
                "view_id": view.view_id,
                "role": view.role.value,
                "source_path": str(view.video.resolve()),
                "size_bytes": size,
                "sha256": _sha256(view.video),
            }
        )
    payload = {
        "schema_version": "visioncortex-original-video-index/1",
        "experiment_id": manifest.experiment_id,
        "retention_mode": "local_zero_copy_source_references",
        "source_copy_bytes": 0,
        "continuous_video_copies_created": 0,
        "total_video_bytes": total,
        "views": records,
    }
    write_json(layout.original_videos / "Original-Video-Index.json", payload)
    (layout.original_videos / "README.txt").write_text(
        "Local synthetic acceptance source videos are retained outside this archive.\n"
        "Original-Video-Index.json stores immutable absolute references and hashes.\n"
        "No source video bytes are copied into the archive.\n",
        encoding="utf-8",
    )
    return payload


def run_local_six_view_acceptance(
    root: Path,
    config: dict[str, Any],
    *,
    experiment_id: str = "VC-LOCAL-SIX-VIEW-ACCEPTANCE",
) -> Path:
    """Run a real-media, deterministic, explicitly synthetic archive acceptance."""

    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        receipt = root / experiment_id / "JSON-Config-Files" / "local_acceptance.json"
        if receipt.is_file():
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            if payload.get("passed") is True:
                return receipt.parents[1]
        raise RuntimeError(f"Local acceptance root is not empty: {root}")
    source_root = root / "Source-Media" / experiment_id
    archive_root = root / experiment_id
    cache_root = root / "Cache"
    runtime_root = root / "Runtime"
    for path in (source_root, cache_root, runtime_root):
        path.mkdir(parents=True, exist_ok=True)
    fps, seconds = 12, 20
    view_specs = [
        ("local-fp-cam01", ViewRole.FIRST_PERSON),
        ("local-tp-cam01", ViewRole.THIRD_PERSON),
        ("local-tp-cam02", ViewRole.THIRD_PERSON),
        ("local-tp-cam03", ViewRole.THIRD_PERSON),
        ("local-tp-cam04", ViewRole.THIRD_PERSON),
        ("local-tp-cam05", ViewRole.THIRD_PERSON),
    ]
    views = []
    for index, (view_id, role) in enumerate(view_specs):
        video = source_root / view_id / "video.mp4"
        _write_source_video(video, index, fps=fps, seconds=seconds)
        views.append(ViewInput(view_id=view_id, role=role, video=video))
    manifest = RunManifest(experiment_id=experiment_id, views=views)
    local_config = copy.deepcopy(config)
    local_config["mllm"]["enabled"] = False
    local_config["project"]["output_root"] = str(root)
    local_config["storage"].update(
        {
            "archive_root": str(root),
            "local_input_root": str(source_root),
            "local_runtime_root": str(runtime_root),
            "local_cache_root": str(cache_root),
            "local_staging_root": str(root),
            "sync_to_nas": False,
        }
    )
    local_config["models"].setdefault("open_vocabulary_key_frame", {})[
        "enabled"
    ] = False
    local_config["models"].setdefault("temporal_participant_segmentation", {})[
        "enabled"
    ] = False
    local_config["models"].setdefault("liquid_semantic_sidecar", {})[
        "enabled"
    ] = False
    local_config["performance"]["tensor_rt"] = "auto"
    local_config["performance"]["derived_media_cache_enabled"] = True

    layout = ArchiveLayout(archive_root)
    layout.create()
    monitor = ResourceMonitor(
        layout.json_config / "resource_telemetry.json", interval_seconds=0.25
    )
    monitor.start()
    monitor_running = True
    started = time.perf_counter()
    stage_durations: list[dict[str, Any]] = []
    try:
        monitor.set_stage("model_preflight")
        preflight_started = time.perf_counter()
        engine_paths = [
            Path(str(local_config["models"].get("first_person_engine") or "")),
            Path(str(local_config["models"].get("third_person_engine") or "")),
        ]
        model_preflight: dict[str, Any]
        if all(path.is_file() for path in engine_paths):
            local_config["performance"]["tensor_rt"] = "required"
            model_preflight = validate_models(local_config)
            model_preflight["video_encoder"] = {
                "selected_encoder": str(
                    local_config["performance"].get("ffmpeg_video_encoder")
                    or "h264_nvenc"
                ),
                "requested_encoder_usable": True,
                "software_fallback_active": False,
            }
            model_preflight["scope"] = (
                "real artifact/hash/deserialization preflight; synthetic package media"
            )
        else:
            model_preflight = {
                "status": "not_available_in_development_profile",
                "runtime": {"roles": {}},
                "video_encoder": {
                    "selected_encoder": str(
                        local_config["performance"].get("ffmpeg_video_encoder") or ""
                    ),
                    "requested_encoder_usable": None,
                    "software_fallback_active": None,
                },
            }
        write_json(
            layout.json_config / "model_runtime_preflight.json", model_preflight
        )
        stage_durations.append(
            {
                "stage": "model_preflight",
                "duration_seconds": round(time.perf_counter() - preflight_started, 6),
            }
        )
        monitor.set_stage("probe")
        probe_started = time.perf_counter()
        infos = probe_views(views, workers=6, prefer_clock_metadata=False)
        stage_durations.append(
            {
                "stage": "probe",
                "duration_seconds": round(time.perf_counter() - probe_started, 6),
            }
        )
        transforms = {
            view.view_id: AlignmentTransform(
                view_id=view.view_id,
                reference_view_id=views[0].view_id,
                confidence=1.0,
                state="aligned",
                csv_match_count=fps * seconds,
                csv_match_ratio=1.0,
                visual_confidence=1.0,
            )
            for view in views
        }
        actions = list(ActionType)
        centres = (2_500.0, 5_000.0, 7_500.0, 10_000.0, 12_500.0, 15_000.0)
        events = []
        for index, (action, centre) in enumerate(zip(actions, centres, strict=True), 1):
            objects = _participants(action)
            state_before, state_after = _state_transition(action)
            candidates = [
                ActionCandidate(
                    candidate_id=f"CAND-{view.view_id}-{index:06d}",
                    action_type=action,
                    view_id=view.view_id,
                    role=view.role,
                    local_start_ms=centre - 500.0,
                    local_end_ms=centre + 500.0,
                    global_start_ms=centre - 500.0,
                    global_end_ms=centre + 500.0,
                    key_global_ms=centre,
                    objects=objects,
                    confidence=0.96,
                    evidence=[
                        {
                            "frame_index": round(centre / 1000.0 * fps),
                            "track_id": 100 + index,
                            "source": "deterministic_local_fixture",
                        }
                    ],
                )
                for view in views[:2]
            ]
            events.append(
                EvidenceEvent(
                    event_id=f"EVT-{index:06d}",
                    action_type=action,
                    global_start_ms=centre - 500.0,
                    global_end_ms=centre + 500.0,
                    key_global_ms=centre,
                    objects=objects,
                    confidence=0.96,
                    accepted=True,
                    audit_reason="deterministic local six-view media acceptance",
                    supporting_views=[views[0].view_id, views[1].view_id],
                    supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
                    candidates=candidates,
                    uncertainty=["synthetic_contract_acceptance_not_real_experiment"],
                    semantic_review={
                        "final_action_type": action.value,
                        "final_participant_objects": objects,
                        "directly_supported_view_ids": [
                            views[0].view_id,
                            views[1].view_id,
                        ],
                    },
                    model_understanding={
                        "status": "completed",
                        "model": "deterministic-local-contract-fixture",
                        "action_type_confirmed": action.value,
                        "current_step": f"Synthetic observable step: {action.value}",
                        "next_step": "synthetic sequence continuation",
                        "cross_view_consistency": "consistent",
                        "confidence": 1.0,
                        "physical_change": {
                            "before": state_before,
                            "after": state_after,
                            "change_type": f"synthetic_{action.value}",
                            "truth_scope": "deterministic_synthetic_contract",
                        },
                        "uncertainties": [
                            "contract fixture; never valid as production truth"
                        ],
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                        "per_view_observations": [
                            {
                                "view_id": view.view_id,
                                "observation": f"Synthetic {action.value} participants visible",
                            }
                            for view in views[:2]
                        ],
                    },
                )
            )
        segment = ExperimentSegment(
            segment_id="EXP-0001",
            global_start_ms=1_000.0,
            global_end_ms=18_000.0,
            event_ids=[event.event_id for event in events],
            participating_views=[view.view_id for view in views],
            micro_segments=[
                {
                    "micro_segment_id": f"MICRO-0001-{index:04d}",
                    "start_global_ms": event.global_start_ms,
                    "end_global_ms": event.global_end_ms,
                    "action_type": event.action_type.value,
                    "objects": event.objects,
                }
                for index, event in enumerate(events, 1)
            ],
        )
        group = ExperimentGroup(
            group_id="GROUP-0001",
            continuity_type="independent",
            atomic_experiment_ids=[segment.segment_id],
            global_start_ms=segment.global_start_ms,
            global_end_ms=segment.global_end_ms,
            participating_views=[view.view_id for view in views],
            first_person_view=views[0].view_id,
            third_person_view=views[1].view_id,
            continuity_reason="deterministic six-view contract acceptance",
            experiment_name="本地六路全链路结构验收",
            experiment_name_en="Local-Six-View-Full-Chain-Acceptance",
            archive_folder="001_Local-Six-View-Full-Chain-Acceptance",
            key_event_ids=[event.event_id for event in events],
            model_understanding={
                "status": "completed",
                "model": "deterministic-local-contract-fixture",
                "overall_summary": (
                    "本地六路媒体完成探测、对齐、六类动作关键素材、物理变化、"
                    "日报、PDF 与可检索索引的全链路结构验收。"
                ),
                "confidence": 1.0,
                "uncertainties": [
                    "合成结构验收，不代表真实实验或生产域模型质量"
                ],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "server_reported": False,
                },
                "steps": [
                    {
                        "step_index": index,
                        "start_global_ms": event.global_start_ms,
                        "end_global_ms": event.global_end_ms,
                        "current_step": (event.model_understanding or {}).get(
                            "current_step"
                        ),
                        "next_step": (event.model_understanding or {}).get(
                            "next_step"
                        ),
                        "objects": event.objects,
                        "physical_change": {
                            "change_type": f"synthetic_{event.action_type.value}"
                        },
                        "supporting_views": event.supporting_views,
                        "confidence": event.confidence,
                    }
                    for index, event in enumerate(events, 1)
                ],
            },
        )
        detection_paths = _write_detection_ledgers(
            layout.work / "deterministic-detections", views, events, fps=fps
        )
        monitor.set_stage("experiment_clips")
        clips_started = time.perf_counter()
        materialize_experiment_clips(
            layout,
            [group],
            [segment],
            events,
            views,
            infos,
            transforms,
            local_config,
        )
        stage_durations.append(
            {
                "stage": "experiment_clips",
                "duration_seconds": round(time.perf_counter() - clips_started, 6),
            }
        )
        monitor.set_stage("key_materials")
        materials_started = time.perf_counter()
        materialize_key_materials(
            layout,
            events,
            [group],
            views,
            infos,
            transforms,
            detection_paths,
            local_config,
            archive_id=experiment_id,
        )
        _rerender_curated_participant_annotations(
            layout, events, [group], local_config
        )
        stage_durations.append(
            {
                "stage": "key_materials",
                "duration_seconds": round(
                    time.perf_counter() - materials_started, 6
                ),
            }
        )
        original_index = _original_references(layout, manifest)
        write_json(
            layout.json_config / "Stage-Receipts" / "original_ingest.json",
            {
                "schema_version": "visioncortex-local-original-ingest/1",
                "input_mode": "local_zero_copy_source_references",
                "source_copy_bytes": 0,
                "continuous_source_copies_created": 0,
                "verified_file_count": len(views),
                "nas_accessed": False,
            },
        )
        write_json(
            layout.json_config / "Input-Manifests" / "nas_ingest.json",
            {
                "schema_version": "visioncortex-ingest-audit/1",
                "input_mode": "local_zero_copy_source_references",
                "copied_source_bytes": 0,
                "continuous_source_copies_created": 0,
                "nas_accessed": False,
                "source_validation": {
                    "verified_file_count": len(views),
                    "fresh_stat_count": len(views),
                    "cache_hit_count": 0,
                },
            },
        )
        write_json(
            layout.json_config / "media_pipeline_preflight.json",
            {
                "schema_version": "visioncortex-local-media-preflight/1",
                "status": "completed",
                "selected_encoder": str(
                    local_config["performance"].get("ffmpeg_video_encoder")
                    or "h264_nvenc"
                ),
                "source_copy_bytes": 0,
                "nas_accessed": False,
            },
        )
        physical = [
            PhysicalChange(
                change_id=f"CHANGE-{index:06d}",
                event_id=event.event_id,
                global_ms=event.key_global_ms,
                change_type=f"synthetic_{event.action_type.value}",
                object_names=event.objects,
                before_state=_state_transition(event.action_type)[0],
                after_state=_state_transition(event.action_type)[1],
                supporting_views=event.supporting_views,
                confidence=event.confidence,
                uncertainty=["synthetic_contract_acceptance"],
            )
            for index, event in enumerate(events, 1)
        ]
        metrics = {
            "total_duration_seconds": round(time.perf_counter() - started, 6),
            "timing_scope": "updated_through_package_before_report_rendering",
            "stage_durations": list(stage_durations),
            "tokens": {
                "experiment_groups": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                },
                "key_materials": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                },
                "daily_report": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                },
                "run_total": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            },
            "mllm_calls": [],
            "ark_calls": 0,
            "source_copy_bytes": 0,
            "synthetic_local_acceptance": True,
            "performance_mode": {
                "input_mode": "local_zero_copy_source_references",
                "source_copy_bytes": 0,
                "concurrent_input_views": 6,
                "bounded_streaming": True,
            },
        }
        write_json(
            layout.json_config / "quality_acceptance.json",
            {
                "schema_version": "visioncortex-synthetic-quality-contract/1",
                "status": "synthetic_contract_only",
                "experiment_boundaries": {"precision": 1.0, "recall": 1.0},
                "production_domain_claim": False,
            },
        )
        write_json(
            layout.json_config / "key_material_recall_eval.json",
            {
                "schema_version": "visioncortex-synthetic-key-material-eval/1",
                "status": "completed",
                "authority": "deterministic_synthetic_contract_not_production_ground_truth",
                "threshold_results": [
                    {
                        "temporal_iou_threshold": 0.5,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    }
                ],
            },
        )
        write_json(
            layout.json_config / "experiment_group_understanding.json",
            {
                "schema_version": "visioncortex-experiment-group-understanding/2",
                "groups": [group.model_dump(mode="json")],
            },
        )
        write_json(
            layout.json_config / "key_material_model_understanding.json",
            {
                "schema_version": "visioncortex-key-material-understanding/1",
                "events": [event.model_dump(mode="json") for event in events],
            },
        )
        monitor.set_stage("package")
        package_started = time.perf_counter()
        summary = finalize_archive(
            layout,
            manifest,
            infos,
            transforms,
            events,
            [segment],
            [group],
            events,
            physical,
            [],
            local_config,
            {"required_bytes": 0, "free_bytes": 0},
            run_metrics=metrics,
        )
        stage_durations.append(
            {
                "stage": "package",
                "duration_seconds": round(time.perf_counter() - package_started, 6),
            }
        )
        monitor.set_stage("telemetry_close")
        telemetry_close_started = time.perf_counter()
        telemetry = monitor.stop()
        monitor_running = False
        stage_durations.append(
            {
                "stage": "telemetry_close",
                "duration_seconds": round(
                    time.perf_counter() - telemetry_close_started, 6
                ),
            }
        )
        pipeline_elapsed = time.perf_counter() - started
        metrics["total_duration_seconds"] = round(pipeline_elapsed, 6)
        metrics["stage_durations"] = list(stage_durations)
        write_json(layout.json_config / "run_metrics.json", metrics)
        summary.stats["run_metrics"] = metrics
        write_json(
            layout.json_config / "evidence_package.json",
            summary.model_dump(mode="json"),
        )

        reports_started = time.perf_counter()
        generate_daily_report_archive(layout, summary, metrics, local_config)
        reports_duration = time.perf_counter() - reports_started
        contract_started = time.perf_counter()
        write_archive_contract_manifest(layout.root)
        contract_duration = time.perf_counter() - contract_started
        acceptance_elapsed = time.perf_counter() - started
        post_pipeline_stages = [
            {
                "stage": "reports",
                "duration_seconds": round(reports_duration, 6),
            },
            {
                "stage": "contract_manifest",
                "duration_seconds": round(contract_duration, 6),
            },
        ]
        metrics["acceptance_end_to_end_seconds"] = round(acceptance_elapsed, 6)
        metrics["post_pipeline_stage_durations"] = post_pipeline_stages
        write_json(layout.json_config / "run_metrics.json", metrics)
        required = {
            "experiment_clips": any(layout.experiment_clips.rglob("*.mp4")),
            "key_frames": any(layout.key_frames.rglob("*.jpg")),
            "key_clips": any(layout.key_clips.rglob("*.mp4")),
            "daily_report": any(layout.daily_reports.rglob("*.json")),
            "professional_pdf": any(layout.professional_pdfs.rglob("*.pdf")),
            "evidence_json": (
                layout.json_config / "evidence_package.json"
            ).is_file(),
            "sqlite_index": (
                layout.json_config / "evidence_index.sqlite"
            ).is_file(),
            "original_references": (
                layout.original_videos / "Original-Video-Index.json"
            ).is_file(),
        }
        receipt = {
            "schema_version": LOCAL_ACCEPTANCE_SCHEMA,
            "status": "completed",
            "passed": all(required.values()),
            "synthetic": True,
            "production_claim": False,
            "experiment_id": experiment_id,
            "source_view_count": len(views),
            "source_duration_seconds_per_view": seconds,
            "source_copy_bytes": 0,
            "original_reference_count": len(original_index["views"]),
            "event_count": len(events),
            "action_types": [action.value for action in actions],
            "ark_calls": 0,
            "token_usage": 0,
            "pipeline_duration_seconds": round(pipeline_elapsed, 6),
            "total_duration_seconds": round(acceptance_elapsed, 6),
            "stage_durations": [*stage_durations, *post_pipeline_stages],
            "resource_peaks": {
                metric: max(
                    (
                        float(summary_by_stage.get(metric, {}).get("max") or 0.0)
                        for summary_by_stage in (
                            telemetry.get("stage_summaries") or {}
                        ).values()
                    ),
                    default=0.0,
                )
                for metric in (
                    "cpu_percent",
                    "memory_percent",
                    "gpu_compute_percent",
                    "gpu_memory_used_mib",
                    "nvdec_percent",
                    "nvenc_percent",
                    "gpu_power_w",
                    "gpu_temperature_c",
                )
            },
            "required_artifacts": required,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(layout.json_config / "local_acceptance.json", receipt)
        write_json(
            layout.root / "run_status.json",
            {
                "stage": "completed",
                "progress": 1.0,
                "synthetic": True,
                "passed": receipt["passed"],
            },
        )
        if not receipt["passed"]:
            raise RuntimeError("Local six-view archive contract is incomplete")
        return layout.root
    finally:
        if monitor_running:
            monitor.stop()
