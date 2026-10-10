"""Bounded semantic execution, review inputs and resumable failure receipts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

from ..mllm import EVENT_SYSTEM_PROMPT, FINAL_GROUP_SYSTEM_PROMPT, GROUP_SYSTEM_PROMPT
from ..schemas import (
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    VideoInfo,
    ViewInput,
)
from .layout import ArchiveLayout
from .semantic_errors import SemanticAnalysisUnavailable


@dataclass(frozen=True)
class SemanticReviewServices:
    """Named execution ports; supplied explicitly at the composition boundary."""

    ArkStepAnalyzer: Callable[..., Any]
    SpeechContext: Callable[..., Any]
    ViewFrameReader: Callable[..., Any]
    _group_folder_name: Callable[..., Any]
    _raise_for_incomplete_semantic_results: Callable[..., Any]
    _read_semantic_cache: Callable[..., Any]
    _relative: Callable[..., Any]
    _run_bounded_semantic_waves: Callable[..., Any]
    _safe_slug: Callable[..., Any]
    _semantic_cache_path: Callable[..., Any]
    _semantic_cache_reads_enabled: Callable[..., Any]
    _semantic_fingerprint: Callable[..., Any]
    _sha256_file: Callable[..., Any]
    _storyboard_times: Callable[..., Any]
    _with_selected_keyframe_review_images: Callable[..., Any]
    _write_aligned_frame: Callable[..., Any]
    _write_semantic_cache: Callable[..., Any]
    extract_temporal_review_frames: Callable[..., Any]
    key_material_review_images: Callable[..., Any]
    normalize_uncalibrated_hand_identity: Callable[..., Any]
    prompt_with_speech: Callable[..., Any]
    record_semantic_review: Callable[..., Any]
    write_json: Callable[..., Any]


def _raise_for_incomplete_semantic_results(
    layout: "ArchiveLayout",
    *,
    stage: str,
    results: Sequence[tuple[str, dict[str, Any]]],
    fail_run: bool = True,
    services: SemanticReviewServices,
) -> list[str]:
    incomplete = [
        {
            "subject_id": subject_id,
            "status": str(result.get("status") or "unknown"),
            "error": result.get("error"),
            "attempts": result.get("attempts"),
        }
        for subject_id, result in results
        if result.get("status") != "completed"
    ]
    path = layout.json_config / f"{stage}_semantic_failures.json"
    if not incomplete:
        if path.exists():
            services.write_json(
                path,
                {
                    "schema_version": "visioncortex-semantic-stage-failure/1",
                    "stage": stage,
                    "status": "completed",
                    "incomplete": [],
                },
            )
        return []
    services.write_json(
        path,
        {
            "schema_version": "visioncortex-semantic-stage-failure/1",
            "stage": stage,
            "status": "resumable_failure" if fail_run else "partial_evidence",
            "evidence_classification": "NOT_PROVEN" if fail_run else "PARTIAL_EVIDENCE",
            "analysis_continuation_allowed": not fail_run,
            "formal_evidence_mutated": False,
            "completed_results_reusable": True,
            "incomplete": incomplete,
        },
    )
    if fail_run:
        raise SemanticAnalysisUnavailable(
            f"{stage} semantic analysis incomplete for {len(incomplete)} subject(s); "
            f"resume from {path}"
        )
    return [str(item["subject_id"]) for item in incomplete]


def analyze_experiment_groups(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    config: dict[str, Any],
    *,
    final_adjudicated: bool = False,
    services: SemanticReviewServices,
) -> None:
    speech_contexts = services.SpeechContext(layout.root, config)
    analyzer = services.ArkStepAnalyzer(config)
    cache_root = layout.work / "mllm-cache" / "experiment-groups"
    by_view = {view.view_id: view for view in views}
    by_segment = {segment.segment_id: segment for segment in segments}
    base_pairs = max(2, int(config["mllm"].get("storyboard_pairs_per_group", 4)))
    maximum_pairs = max(
        base_pairs,
        int(config["mllm"].get("storyboard_pairs_per_group_max", base_pairs)),
    )
    seconds_per_pair = max(
        1.0,
        float(config["mllm"].get("storyboard_seconds_per_pair", 300.0)),
    )

    def analyze(group: ExperimentGroup) -> tuple[ExperimentGroup, dict[str, Any]]:
        storyboard: list[tuple[str, Path]] = []
        storyboard_dir = layout.work / "group-storyboards" / group.group_id
        atomic = [by_segment[item] for item in group.atomic_experiment_ids]
        event_ids = {event_id for segment in atomic for event_id in segment.event_ids}
        group_events = [event for event in events if event.event_id in event_ids]
        duration_pairs = (
            math.ceil(
                max(1.0, group.global_end_ms - group.global_start_ms)
                / (seconds_per_pair * 1000.0)
            )
            + 1
        )
        action_pairs = (
            len({event.action_type.value for event in group_events if event.accepted})
            + 2
        )
        pair_limit = min(
            maximum_pairs,
            max(base_pairs, duration_pairs, len(atomic) + 2, action_pairs),
        )
        with services.ViewFrameReader(max_open=2) as frame_reader:
            for index, global_ms in enumerate(
                services._storyboard_times(group, group_events, pair_limit, atomic), 1
            ):
                role_paths: dict[str, Path] = {}
                from ..boundary_review import third_person_at

                sampled_third = third_person_at(group, global_ms)
                for role_label, view_id in (
                    ("first_person", group.first_person_view),
                    ("third_person", sampled_third),
                ):
                    if view_id is None:
                        continue
                    local_ms = transforms[view_id].to_local(global_ms)
                    if not 0.0 <= local_ms <= infos[view_id].duration_ms:
                        continue
                    frame = frame_reader.read(
                        by_view[view_id], infos[view_id], local_ms
                    )
                    if frame is None:
                        continue
                    path = (
                        storyboard_dir
                        / f"{index:02d}_{role_label}_{services._safe_slug(view_id)}.jpg"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(
                        str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]
                    ):
                        continue
                    role_paths[role_label] = path
                if {"first_person", "third_person"} <= set(role_paths):
                    aligned = storyboard_dir / f"{index:02d}_aligned_first_third.jpg"
                    services._write_aligned_frame(
                        role_paths["first_person"],
                        role_paths["third_person"],
                        aligned,
                        (
                            f"First-Person {group.first_person_view}",
                            f"Third-Person {sampled_third}",
                        ),
                    )
                    storyboard.append(
                        (
                            f"t={global_ms:.3f}ms; aligned_first_third; "
                            f"first={group.first_person_view}; third={sampled_third}",
                            aligned,
                        )
                    )
                elif "first_person" in role_paths:
                    storyboard.append(
                        (
                            f"t={global_ms:.3f}ms; first={group.first_person_view}; "
                            "third_person_correspondence_unverified",
                            role_paths["first_person"],
                        )
                    )
        if len(storyboard) < 2:
            raise RuntimeError(
                f"{group.group_id} has fewer than two complete aligned storyboard pairs"
            )
        from ..speech_refresh import retain_group_inputs

        retain_group_inputs(layout, group, storyboard, config)
        if group.boundary_reviews:
            retained = []
            directory = layout.json_config / "Workflow-Storyboards" / group.group_id
            directory.mkdir(parents=True, exist_ok=True)
            for label, image_path in storyboard:
                digest = services._sha256_file(image_path)
                target = directory / f"{digest}.jpg"
                if not target.exists():
                    shutil.copyfile(image_path, target)
                retained.append(
                    {
                        "label": label,
                        "image": services._relative(target, layout.root),
                        "sha256": digest,
                    }
                )
            services.write_json(
                directory / "manifest.json",
                {
                    "group_uid": group.group_uid,
                    "samples": retained,
                    "policy": "model understanding inputs; not human ground truth",
                },
            )
        semantic_evidence = {
            "continuity_type": group.continuity_type,
            "continuity_reason": group.continuity_reason,
            "workflow_kind": group.workflow_kind,
            "workflow_units": group.workflow_units,
            "completion_status": group.completion_status,
            "view_timeline": group.view_timeline,
            "global_start_ms": group.global_start_ms,
            "global_end_ms": group.global_end_ms,
            "first_person_view": group.first_person_view,
            "third_person_view": group.third_person_view,
            "atomic_boundaries": [
                segment.model_dump(
                    mode="json",
                    exclude={
                        "semantic_understanding",
                        "clips",
                        "aligned_multiview_clip",
                    },
                )
                for segment in atomic
            ],
            "cv_events": [
                event.model_dump(
                    mode="json",
                    exclude={"model_understanding", "key_frames", "key_clips"},
                )
                for event in group_events
                if event.accepted
            ],
            "storyboard_sampling": {
                "base_pair_count": base_pairs,
                "effective_pair_limit": pair_limit,
                "materialized_pair_count": len(storyboard),
                "maximum_pair_count": maximum_pairs,
                "seconds_per_pair": seconds_per_pair,
                "atomic_experiment_count": len(atomic),
            },
        }
        speech_context = speech_contexts.window(
            group.global_start_ms, group.global_end_ms, group.participating_views
        )
        if speech_context is not None:
            semantic_evidence["speech_context"] = speech_context
        system_prompt = (
            FINAL_GROUP_SYSTEM_PROMPT if final_adjudicated else GROUP_SYSTEM_PROMPT
        )
        fingerprint = services._semantic_fingerprint(
            "experiment-group",
            config,
            services.prompt_with_speech(system_prompt, speech_context),
            semantic_evidence,
            storyboard,
        )
        persistent_cache_path = services._semantic_cache_path(
            config, "experiment-groups", fingerprint
        )
        run_cache_path = cache_root / f"{fingerprint}.json"
        cached = None
        if services._semantic_cache_reads_enabled(config):
            cached = services._read_semantic_cache(persistent_cache_path, fingerprint)
            if cached is None:
                cached = services._read_semantic_cache(run_cache_path, fingerprint)
        if cached is not None:
            return group, cached
        result = analyzer.analyze_group(
            group,
            atomic,
            group_events,
            storyboard,
            system_prompt=system_prompt,
            final_adjudicated=final_adjudicated,
            **(
                {"speech_context": speech_context} if speech_context is not None else {}
            ),
        )
        if result.get("status") == "completed":
            result = services._write_semantic_cache(
                persistent_cache_path, fingerprint, result
            )
            services.write_json(run_cache_path, result)
        return group, result

    workers = max(1, int(config["mllm"].get("group_workers", 2)))
    semantic_results: list[tuple[str, dict[str, Any]]] = []
    try:
        with ThreadPoolExecutor(
            max_workers=min(workers, max(1, len(groups)))
        ) as executor:
            futures = [executor.submit(analyze, group) for group in groups]
            for future in as_completed(futures):
                group, result = future.result()
                semantic_results.append((group.group_id, result))
                if group.completion_status in {
                    "unresolved",
                    "ongoing_at_recording_end",
                }:
                    result = dict(result)
                    result["boundary_assessment"] = {
                        **(result.get("boundary_assessment") or {}),
                        "end_complete": False,
                        "end_reason": group.completion_reason,
                        "localized_rescan_needed": True,
                    }
                group.model_understanding = result
                from ..boundary_review import apply_semantic_units

                apply_semantic_units(group, result)
                if result.get("status") == "completed":
                    group.experiment_name = str(
                        result.get("experiment_name") or group.experiment_name
                    )
                    group.experiment_name_en = services._safe_slug(
                        str(
                            result.get("experiment_name_en") or group.experiment_name_en
                        )
                    )
                    confirmed = result.get("continuity_type_confirmed")
                    # Rule-based continuous chains may be split only by an explicit
                    # model conflict; rule-based independent groups never merge here.
                    if (
                        confirmed == "continuous"
                        and group.continuity_type == "continuous"
                    ):
                        group.continuity_reason += "；模型确认连续"
                    elif confirmed not in {group.continuity_type, "uncertain", None}:
                        result.setdefault("uncertainties", []).append(
                            "模型连续性判断与物理规则冲突，归档采用物理规则并保留冲突"
                        )
                index = int(group.group_id.rsplit("-", 1)[-1])
                group.archive_folder = services._group_folder_name(
                    layout,
                    index,
                    group.experiment_name,
                    group.experiment_name_en,
                )
                for segment_id in group.atomic_experiment_ids:
                    segment = by_segment[segment_id]
                    segment.experiment_name = group.experiment_name
                    segment.experiment_name_en = group.experiment_name_en
                    segment.semantic_understanding = result
    finally:
        analyzer.close()
    services._raise_for_incomplete_semantic_results(
        layout,
        stage="experiment_group",
        results=semantic_results,
        fail_run=bool(config.get("mllm", {}).get("fail_run_on_incomplete", False)),
    )


def _with_selected_keyframe_review_images(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    config: dict[str, Any],
    timeline_images: list[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    """Add the actual retained key frames without truncating temporal evidence.

    Raw-frame sidecars bind the image to its requested and decoder-seek time. Old
    caches without that binding remain usable as timeline evidence only.
    Never silently append one view or evict liquid-cycle context to fit a limit.
    """
    settings = config["mllm"]
    limit = int(
        (settings.get("max_images_per_event_by_action") or {}).get(
            event.action_type.value, settings.get("max_images_per_event", 8)
        )
    )
    views = [view for view in event.key_frames if view != "aligned_first_third"]
    if not views or len(timeline_images) + len(views) > limit:
        return timeline_images
    root = layout.work / "key-material-annotation-inputs" / event.event_id
    selected = {}
    for sidecar in sorted(root.glob("*.json")):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            requested = float(meta["requested_key_global_ms"])
            decoded = float(meta["decoded_key_global_ms"])
        except (OSError, ValueError, TypeError, KeyError):
            continue
        view_id = meta.get("view_id")
        image_path = sidecar.with_suffix(".jpg")
        if (
            meta.get("event_id") != event.event_id
            or view_id not in views
            or not math.isfinite(requested)
            or not math.isfinite(decoded)
            or abs(requested - event.key_global_ms) > 0.001
            or not image_path.is_file()
        ):
            continue
        native_verified = False
        if meta.get("schema_version") == "visioncortex-key-material-annotation-input/2":
            proof = meta.get("source_frame_verification") or {}
            native_verified = bool(
                proof.get("status") == "verified"
                and proof.get("detections_bound_to_pixels") is True
                and proof.get("view_id") == view_id
                and proof.get("decoded_global_ms") == decoded
                and hashlib.sha256(image_path.read_bytes()).hexdigest()
                == meta.get("image_sha256")
            )
            if not native_verified:
                continue
        if view_id in selected:
            # Multiple candidate files cannot establish which raw image is current.
            return timeline_images
        selected[view_id] = (
            f"view_id={view_id}; sample_scope=selected_keyframe; "
            f"requested_global_ms={requested:.3f}; decoded_global_ms={decoded:.3f}; "
            + (
                "time_basis=native_pts_with_alignment_transform; exact_frame_pts=verified; physical_cross_view_sync=unverified"
                if native_verified
                else "time_basis=decoder_seek_target; exact_frame_pts=unverified"
            ),
            image_path,
        )
    if set(selected) != set(views):
        return timeline_images
    return [*timeline_images, *(selected[view] for view in views)]


def key_material_review_images(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    config: dict[str, Any],
    *,
    services: SemanticReviewServices,
) -> list[tuple[str, Path]]:
    """Select bounded temporal evidence without discarding dual-view context.

    Timeline positions never claim an action peak. When image budget permits,
    separately include both retained key frames with their recorded times.
    """

    fallback_images = [
        (view_id, layout.root / relative)
        for view_id, relative in event.key_frames.items()
        if view_id != "aligned_first_third"
    ]
    if not bool(config["mllm"].get("use_key_clip_temporal_samples", True)):
        return fallback_images
    samples_by_action = config["mllm"].get("temporal_samples_per_view_by_action", {})
    samples_per_view = int(
        samples_by_action.get(
            event.action_type.value,
            config["mllm"].get("temporal_samples_per_view", 3),
        )
    )
    compact_actions = {
        str(value)
        for value in config["mllm"].get("compact_aligned_temporal_actions", [])
    }
    measurements = (event.observability or {}).get("measurements") or {}
    repeated_path = measurements.get("repeated_pipette_path") or {}
    if event.action_type == ActionType.LIQUID_MOVEMENT:
        primary_view = str(repeated_path.get("view_id") or "")
        if not primary_view:
            primary_view = next(
                (
                    view_id
                    for view_id, relative in event.key_clips.items()
                    if view_id != "aligned_first_third" and relative
                ),
                "",
            )
        primary_relative = event.key_clips.get(primary_view)
        if primary_relative:
            primary = services.extract_temporal_review_frames(
                layout.root / primary_relative,
                layout.work
                / "mllm-temporal-samples-v2"
                / event.event_id
                / "liquid-dense-primary",
                primary_view,
                samples_per_view=int(config["mllm"].get("liquid_primary_samples", 9)),
            )
            required_primary_count = int(
                config["mllm"].get("liquid_primary_samples", 15)
            )
            if len(primary) == required_primary_count:
                context: list[tuple[str, Path]] = []
                for view_id, relative in event.key_clips.items():
                    if view_id in {primary_view, "aligned_first_third"}:
                        continue
                    context.extend(
                        services.extract_temporal_review_frames(
                            layout.root / relative,
                            layout.work
                            / "mllm-temporal-samples-v2"
                            / event.event_id
                            / "liquid-context",
                            view_id,
                            samples_per_view=int(
                                config["mllm"].get("liquid_context_samples", 3)
                            ),
                        )
                    )
                    break
                return services._with_selected_keyframe_review_images(
                    layout, event, config, primary + context
                )

    if event.action_type.value in compact_actions:
        aligned_relative = event.key_clips.get("aligned_first_third")
        if aligned_relative:
            aligned = services.extract_temporal_review_frames(
                layout.root / aligned_relative,
                layout.work / "mllm-temporal-samples-v2" / event.event_id / "aligned",
                "aligned_first_third",
                samples_per_view=samples_per_view,
            )
            if aligned:
                return services._with_selected_keyframe_review_images(
                    layout, event, config, aligned
                )

    sample_dir = layout.work / "mllm-temporal-samples-v2" / event.event_id
    images: list[tuple[str, Path]] = []
    for view_id, relative in event.key_clips.items():
        if view_id == "aligned_first_third":
            continue
        for label, path in services.extract_temporal_review_frames(
            layout.root / relative,
            sample_dir,
            view_id,
            samples_per_view=samples_per_view,
        ):
            images.append((label, path))

    def sample_order(item: tuple[str, Path]) -> tuple[int, str]:
        label = item[0]
        match = re.search(r"temporal_phase=([^;]+)", label)
        phase = match.group(1) if match else ""
        phases = [
            "clip_early",
            "clip_mid_early",
            "clip_middle",
            "clip_mid_late",
            "clip_late",
        ]
        if phase in phases:
            return phases.index(phase), label
        if phase.startswith("timeline_") and phase[9:].isdigit():
            return int(phase[9:]), label
        return 999, label

    images.sort(key=sample_order)
    return (
        services._with_selected_keyframe_review_images(layout, event, config, images)
        if images
        else fallback_images
    )


def _run_bounded_semantic_waves(
    items: Sequence[Any],
    analyze: Callable[[Any], Any],
    *,
    workers: int,
    failure_threshold: int,
    on_result: Callable[[Any], None] | None = None,
) -> list[Any]:
    """Probe in failure-bounded waves, then keep healthy workers occupied.

    A wholly unavailable provider sees only the original small failure wave.
    After a real successful wave, at most ``workers`` tasks may be in flight;
    a slow response no longer stalls every other worker at a batch barrier.
    """

    wave_size = max(1, min(int(workers), int(failure_threshold)))
    completed: list[Any] = []
    exhausted = object()

    def record(result):
        completed.append(result)
        if on_result:
            on_result(result)

    for wave_start in range(0, len(items), wave_size):
        wave = items[wave_start : wave_start + wave_size]
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = [executor.submit(analyze, item) for item in wave]
            results = [future.result() for future in as_completed(futures)]
        for result in results:
            record(result)
        healthy = all(
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
            and result[1].get("status") == "completed"
            and not result[1].get("cache_reused")
            for result in results
        )
        if healthy:
            remaining = iter(items[wave_start + wave_size :])
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
                pending = set()
                for _ in range(max(1, int(workers))):
                    item = next(remaining, exhausted)
                    if item is not exhausted:
                        pending.add(executor.submit(analyze, item))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        record(future.result())
                        item = next(remaining, exhausted)
                        if item is not exhausted:
                            pending.add(executor.submit(analyze, item))
            break
    return completed


def analyze_key_materials(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    config: dict[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    services: SemanticReviewServices,
) -> None:
    speech_contexts = services.SpeechContext(layout.root, config)
    analyzer = services.ArkStepAnalyzer(config)
    accepted = [event for event in events if event.accepted]
    cache_root = layout.work / "mllm-cache" / "key-materials"
    runtime_started = time.perf_counter()
    runtime_lock = threading.Lock()
    active_calls = 0
    peak_calls = 0
    call_records: list[dict[str, Any]] = []

    def analyze(event: EvidenceEvent) -> tuple[EvidenceEvent, dict[str, Any]]:
        images = services.key_material_review_images(layout, event, config)
        semantic_evidence = event.model_dump(
            mode="json",
            exclude={"model_understanding", "key_frames", "key_clips"},
        )
        speech_context = speech_contexts.window(
            event.global_start_ms, event.global_end_ms, event.supporting_views
        )
        if speech_context is not None:
            semantic_evidence["speech_context"] = speech_context
        fingerprint = services._semantic_fingerprint(
            "key-material",
            config,
            services.prompt_with_speech(EVENT_SYSTEM_PROMPT, speech_context),
            semantic_evidence,
            images,
        )
        persistent_cache_path = services._semantic_cache_path(
            config, "key-materials", fingerprint
        )
        run_cache_path = cache_root / f"{fingerprint}.json"
        cached = None
        if services._semantic_cache_reads_enabled(config):
            cached = services._read_semantic_cache(persistent_cache_path, fingerprint)
            if cached is None:
                cached = services._read_semantic_cache(run_cache_path, fingerprint)
        if cached is not None:
            return event, cached
        nonlocal active_calls, peak_calls
        with runtime_lock:
            active_calls += 1
            peak_calls = max(peak_calls, active_calls)
        try:
            result = analyzer.analyze_event(
                event,
                images,
                **(
                    {"speech_context": speech_context}
                    if speech_context is not None
                    else {}
                ),
            )
        finally:
            with runtime_lock:
                active_calls -= 1
        if result.get("status") == "completed":
            result = services._write_semantic_cache(
                persistent_cache_path, fingerprint, result
            )
            services.write_json(run_cache_path, result)
        return event, result

    workers = max(1, int(config["mllm"].get("workers", 4)))
    failure_threshold = max(
        1, int(config["mllm"].get("failure_circuit_breaker_threshold", 4))
    )
    semantic_results: list[tuple[str, dict[str, Any]]] = []

    def record_progress(value):
        event, result = value
        call_records.append(
            {
                "event_id": event.event_id,
                **{
                    key: result.get(key)
                    for key in (
                        "status",
                        "provider",
                        "model",
                        "request_id",
                        "response_model",
                        "attempts",
                        "latency_seconds",
                        "cache_reused",
                        "usage",
                        "attempt_receipts",
                    )
                },
            }
        )
        # Save bounded progress during long model stages, including incomplete
        # responses and unknown token usage. This is not a quality acceptance.
        if (
            len(call_records) <= min(workers, failure_threshold)
            or len(call_records) % 4 == 0
            or len(call_records) == len(accepted)
        ):
            services.write_json(
                layout.json_config / "key_material_semantic_runtime.json",
                {
                    "schema_version": "visioncortex-semantic-runtime/1",
                    "configured_workers": workers,
                    "initial_failure_wave_size": min(workers, failure_threshold),
                    "peak_analyzer_calls": peak_calls,
                    "concurrency_scope": "analyzer calls including retries; not server-side GPU utilization",
                    "completed_items": len(call_records),
                    "total_items": len(accepted),
                    "duration_seconds": round(time.perf_counter() - runtime_started, 6),
                    "calls": call_records,
                },
            )
            if progress_callback:
                progress_callback(len(call_records), len(accepted))

    try:
        # Start with the failure gate; a successful live wave permits configured
        # concurrency. The shared transport circuit still stops repeated outages.
        completed = services._run_bounded_semantic_waves(
            accepted,
            analyze,
            workers=workers,
            failure_threshold=failure_threshold,
            on_result=record_progress,
        )
        for event, result in completed:
            result = services.normalize_uncalibrated_hand_identity(result)
            semantic_results.append((event.event_id, result))
            event.model_understanding = result
            services.record_semantic_review(event, result)
    finally:
        analyzer.close()
    services._raise_for_incomplete_semantic_results(
        layout,
        stage="key_material",
        results=semantic_results,
        fail_run=bool(config.get("mllm", {}).get("fail_run_on_incomplete", False)),
    )
