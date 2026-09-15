from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .candidate_index import CoarseFrameIndex, FineFrameIndex
from .detection import iter_frame_evidence
from .ordering import candidate_sort_key
from .open_vocabulary_runtime import load_yolo_world_with_local_clip, serialized_open_vocabulary
from .schemas import ActionCandidate, ActionType, FrameEvidence, VideoInfo, ViewInput
from .video_io import read_view_frame_at


_ACTOR_CLASSES = {"hand", "gloved_hand"}
_NON_ACTION_CLASSES = {"person", "face", "background"}


def _normalize_class(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _box_edge_gap_norm(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_box = [float(item) for item in left["xyxy_norm"]]
    right_box = [float(item) for item in right["xyxy_norm"]]
    horizontal = max(0.0, left_box[0] - right_box[2], right_box[0] - left_box[2])
    vertical = max(0.0, left_box[1] - right_box[3], right_box[1] - left_box[3])
    return float(np.hypot(horizontal, vertical))


def select_suspicious_coarse_frames(
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    frame_index: CoarseFrameIndex | None = None,
) -> list[FrameEvidence]:
    """Choose a bounded set of high-motion coarse frames missing an object.

    This selection remains inside the existing coarse stage.  Closed-set
    detections are never replaced; open vocabulary is only asked to explain a
    high-motion frame where the normal detector did not already see both an
    actor and a manipulable object.
    """

    perf = config["performance"]
    percentile = float(perf.get("coarse_open_vocabulary_motion_percentile", 90.0))
    per_hour = max(
        0,
        int(perf.get("coarse_open_vocabulary_frames_per_hour_per_view", 2)),
    )
    adaptive = bool(
        perf.get("coarse_open_vocabulary_adaptive_selection_enabled", False)
    )
    diversity_bucket_ms = max(
        60_000.0,
        float(
            perf.get("coarse_open_vocabulary_diversity_bucket_seconds", 600.0)
        )
        * 1000.0,
    )
    maximum_per_hour = max(
        per_hour,
        int(
            perf.get(
                "coarse_open_vocabulary_max_frames_per_hour_per_view",
                per_hour,
            )
        ),
    )
    selected: list[FrameEvidence] = []
    if per_hour == 0:
        return selected
    for view in views:
        path = detection_paths.get(view.view_id)
        if path is None:
            continue
        frames = list(
            frame_index.iter_frames(view.view_id)
            if frame_index is not None
            else iter_frame_evidence(path)
        )
        if not frames:
            continue
        threshold = float(
            np.percentile(
                np.asarray([frame.motion_score for frame in frames], dtype=np.float64),
                percentile,
            )
        )
        buckets: dict[int, list[FrameEvidence]] = {}
        for frame in frames:
            classes = {_normalize_class(box.class_name) for box in frame.detections}
            has_actor = bool(classes & _ACTOR_CLASSES)
            has_object = bool(classes - _ACTOR_CLASSES - _NON_ACTION_CLASSES)
            uncertainty_trigger = bool(
                has_actor != has_object
                or any(float(box.confidence) < 0.45 for box in frame.detections)
                or any(
                    box.track_id is None
                    and _normalize_class(box.class_name)
                    not in _ACTOR_CLASSES | _NON_ACTION_CLASSES
                    for box in frame.detections
                )
            )
            if frame.motion_score < threshold and not (
                adaptive and uncertainty_trigger
            ):
                continue
            if has_actor and has_object:
                continue
            global_ms = float(frame.global_ms if frame.global_ms is not None else frame.local_ms)
            bucket = max(0, int(global_ms // 3_600_000.0))
            buckets.setdefault(bucket, []).append(frame)
        for bucket_frames in buckets.values():
            legacy = sorted(
                bucket_frames,
                key=lambda frame: (-float(frame.motion_score), frame.local_ms),
            )[:per_hour]
            if not adaptive:
                selected.extend(legacy)
                continue
            diverse: dict[int, list[FrameEvidence]] = {}
            for frame in bucket_frames:
                global_ms = float(
                    frame.global_ms
                    if frame.global_ms is not None
                    else frame.local_ms
                )
                diverse.setdefault(
                    max(0, int(global_ms // diversity_bucket_ms)), []
                ).append(frame)

            def uncertainty_priority(frame: FrameEvidence) -> tuple[Any, ...]:
                classes = {
                    _normalize_class(box.class_name) for box in frame.detections
                }
                has_actor = bool(classes & _ACTOR_CLASSES)
                has_object = bool(classes - _ACTOR_CLASSES - _NON_ACTION_CLASSES)
                low_confidence = any(
                    float(box.confidence) < 0.45 for box in frame.detections
                )
                untracked_object = any(
                    box.track_id is None
                    and _normalize_class(box.class_name)
                    not in _ACTOR_CLASSES | _NON_ACTION_CLASSES
                    for box in frame.detections
                )
                return (
                    int(has_actor and not has_object),
                    int(has_object and not has_actor),
                    int(untracked_object),
                    int(low_confidence),
                    float(frame.motion_score),
                    -float(frame.local_ms),
                )

            adaptive_frames = [
                max(items, key=uncertainty_priority)
                for items in diverse.values()
                if items
            ]
            unique = {
                (frame.view_id, frame.frame_index, frame.local_ms): frame
                for frame in [*legacy, *adaptive_frames]
            }
            selected.extend(
                sorted(
                    unique.values(),
                    key=uncertainty_priority,
                    reverse=True,
                )[:maximum_per_hour]
            )
    return sorted(
        selected,
        key=lambda frame: (
            float(frame.global_ms if frame.global_ms is not None else frame.local_ms),
            frame.view_id,
            frame.frame_index,
        ),
    )


@serialized_open_vocabulary
def _yolo_world_detections(
    frame: np.ndarray,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the already configured pinned YOLO-World model on one coarse frame."""

    prompt_map = {
        str(prompt): _normalize_class(canonical)
        for prompt, canonical in dict(settings.get("prompt_map") or {}).items()
        if str(prompt).strip() and _normalize_class(canonical)
    }
    prompts = list(prompt_map)
    if not prompts:
        raise RuntimeError("Open-vocabulary prompt_map is empty")
    model_path = Path(str(settings.get("model_path") or "")).resolve()
    if not model_path.is_file():
        raise RuntimeError(f"Pinned YOLO-World model is missing: {model_path}")

    # Share the final-key-frame cache so the same configured model is loaded
    # only once during a run.  Importing lazily avoids adding a startup cost.
    from . import archive as archive_module

    cached = archive_module._OPEN_VOCABULARY_MODEL_CACHE.get(str(model_path))
    model_load_seconds = 0.0
    if cached is None:
        started = time.perf_counter()
        cached = {"model": load_yolo_world_with_local_clip(settings), "prompts": None}
        archive_module._OPEN_VOCABULARY_MODEL_CACHE[str(model_path)] = cached
        model_load_seconds = time.perf_counter() - started
    model = cached["model"]
    if cached.get("prompts") != prompts:
        model.to("cpu")
        model.set_classes(prompts)
        cached["prompts"] = list(prompts)
    started = time.perf_counter()
    result = model.predict(
        frame,
        device=int(settings.get("device", 0)),
        imgsz=int(settings.get("image_size", 1280)),
        conf=float(settings.get("confidence", 0.03)),
        iou=float(settings.get("iou", 0.50)),
        verbose=False,
    )[0]
    inference_seconds = time.perf_counter() - started
    height, width = frame.shape[:2]
    admitted: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        minimum_confidence = {
            _normalize_class(class_name): float(value)
            for class_name, value in dict(
                settings.get("minimum_manipulated_object_confidence") or {}
            ).items()
        }
        maximum_area = float(settings.get("coarse_recall_maximum_box_area_norm", 0.50))
        maximum_actor_area = float(
            settings.get("grounded_actor_maximum_box_area_norm", 0.15)
        )
        for class_index, confidence, coordinates in zip(
            boxes.cls, boxes.conf, boxes.xyxy, strict=True
        ):
            prompt = str(result.names[int(class_index)])
            canonical = prompt_map.get(prompt)
            if canonical is None:
                continue
            confidence_value = float(confidence)
            if confidence_value < minimum_confidence.get(canonical, 0.0):
                continue
            x1, y1, x2, y2 = (float(item) for item in coordinates)
            normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
            area = max(0.0, normalized[2] - normalized[0]) * max(
                0.0, normalized[3] - normalized[1]
            )
            class_maximum_area = (
                maximum_actor_area if canonical in _ACTOR_CLASSES else maximum_area
            )
            if area <= 0.0 or area > class_maximum_area:
                continue
            admitted.append(
                {
                    "class_name": canonical,
                    "confidence": confidence_value,
                    "xyxy_norm": normalized,
                    "detector_source": "yolo_world_v2_coarse_recall",
                    "grounding_prompt": prompt,
                }
            )
    return admitted, {
        "status": "executed",
        "model": str(model_path),
        "prompt_count": len(prompts),
        "admitted_count": len(admitted),
        "model_load_seconds": round(model_load_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
    }


def generate_open_vocabulary_coarse_candidates(
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    frame_index: CoarseFrameIndex | None = None,
) -> tuple[list[ActionCandidate], dict[str, Any]]:
    """Add recall-only candidates to the current coarse funnel stage."""

    enabled = bool(
        config["performance"].get("coarse_open_vocabulary_recall_enabled", False)
    )
    report: dict[str, Any] = {
        "schema_version": "visioncortex-coarse-open-vocabulary-recall/1",
        "enabled": enabled,
        "scope": "bounded suspicious frames inside the existing coarse stage",
        "replaces_closed_set_candidates": False,
        "selected_frame_count": 0,
        "candidate_count": 0,
        "frames": [],
        "formal_evidence_ready": not enabled,
        "selection": {
            "motion_percentile": float(
                config["performance"].get(
                    "coarse_open_vocabulary_motion_percentile", 90.0
                )
            ),
            "legacy_frames_per_hour_per_view": int(
                config["performance"].get(
                    "coarse_open_vocabulary_frames_per_hour_per_view", 2
                )
            ),
            "adaptive_time_diversity_enabled": bool(
                config["performance"].get(
                    "coarse_open_vocabulary_adaptive_selection_enabled", False
                )
            ),
        },
    }
    if not enabled:
        report["status"] = "disabled"
        return [], report
    settings = config.get("models", {}).get("open_vocabulary_key_frame") or {}
    if not settings.get("enabled"):
        report["status"] = "model_disabled"
        report["formal_evidence_ready"] = False
        return [], report
    selected = select_suspicious_coarse_frames(
        views, detection_paths, config, frame_index
    )
    report["selected_frame_count"] = len(selected)
    if not selected:
        report["status"] = "no_suspicious_frames"
        report["formal_evidence_ready"] = True
        return [], report

    view_by_id = {view.view_id: view for view in views}
    candidates: list[ActionCandidate] = []
    maximum_gap = float(
        config["performance"].get(
            "coarse_open_vocabulary_max_actor_object_gap_norm", 0.08
        )
    )
    half_window_ms = (
        float(
            config["performance"].get(
                "coarse_open_vocabulary_candidate_window_seconds", 4.0
            )
        )
        * 500.0
    )
    requested_classes = {
        _normalize_class(item)
        for item in dict(settings.get("prompt_map") or {}).values()
        if _normalize_class(item)
    }
    for selected_frame in selected:
        view = view_by_id[selected_frame.view_id]
        info = infos[view.view_id]
        frame_report: dict[str, Any] = {
            "view_id": view.view_id,
            "frame_index": selected_frame.frame_index,
            "local_ms": selected_frame.local_ms,
            "global_ms": selected_frame.global_ms,
            "motion_score": selected_frame.motion_score,
        }
        frame = read_view_frame_at(view, info, selected_frame.local_ms)
        if frame is None:
            frame_report["status"] = "frame_unreadable"
            report["frames"].append(frame_report)
            continue
        try:
            grounded, inference_report = _yolo_world_detections(frame, settings)
            actor_boxes = [
                box for box in grounded if box["class_name"] in _ACTOR_CLASSES
            ]
            object_boxes = [
                box
                for box in grounded
                if box["class_name"] not in _ACTOR_CLASSES | _NON_ACTION_CLASSES
            ]
            if (not actor_boxes or not object_boxes) and requested_classes:
                from .archive import _grounding_dino_key_frame_detections

                fallback_boxes, fallback_report = (
                    _grounding_dino_key_frame_detections(
                        frame, requested_classes, settings
                    )
                )
                grounded.extend(fallback_boxes)
                actor_boxes = [
                    box for box in grounded if box["class_name"] in _ACTOR_CLASSES
                ]
                object_boxes = [
                    box
                    for box in grounded
                    if box["class_name"] not in _ACTOR_CLASSES | _NON_ACTION_CLASSES
                ]
                inference_report["grounding_dino_fallback"] = fallback_report
            pairs = [
                (actor, obj, _box_edge_gap_norm(actor, obj))
                for actor in actor_boxes
                for obj in object_boxes
                if _box_edge_gap_norm(actor, obj) <= maximum_gap
            ]
            if not pairs:
                frame_report.update(
                    {
                        "status": "no_actor_object_contact",
                        "inference": inference_report,
                    }
                )
                report["frames"].append(frame_report)
                continue
            actor, obj, gap = min(
                pairs,
                key=lambda item: (
                    item[2],
                    -min(float(item[0]["confidence"]), float(item[1]["confidence"])),
                ),
            )
            global_ms = float(
                selected_frame.global_ms
                if selected_frame.global_ms is not None
                else selected_frame.local_ms
            )
            local_start = max(0.0, selected_frame.local_ms - half_window_ms)
            local_end = min(float(info.duration_ms), selected_frame.local_ms + half_window_ms)
            candidate = ActionCandidate(
                candidate_id=(
                    f"COARSE-OPEN-VOCAB-{view.view_id}-{len(candidates) + 1:06d}"
                ),
                action_type=ActionType.OBJECT_MOVEMENT,
                view_id=view.view_id,
                role=view.role,
                local_start_ms=local_start,
                local_end_ms=local_end,
                global_start_ms=global_ms - (selected_frame.local_ms - local_start),
                global_end_ms=global_ms + (local_end - selected_frame.local_ms),
                key_global_ms=global_ms,
                objects=[str(obj["class_name"])],
                confidence=min(
                    0.89,
                    max(
                        0.35,
                        min(float(actor["confidence"]), float(obj["confidence"])),
                    ),
                ),
                evidence=[
                    {
                        "frame_index": selected_frame.frame_index,
                        "motion_score": selected_frame.motion_score,
                        "actor": actor,
                        "object": obj,
                        "actor_object_gap_norm": gap,
                        "source_stage": "coarse",
                    }
                ],
                uncertainty=[
                    "开放词汇粗筛只补充精筛召回窗口，不独立确认物理动作"
                ],
            )
            candidates.append(candidate)
            frame_report.update(
                {
                    "status": "candidate_added",
                    "candidate_id": candidate.candidate_id,
                    "inference": inference_report,
                }
            )
        except Exception as exc:  # Keep the established closed-set funnel available.
            frame_report.update(
                {
                    "status": "open_vocabulary_error_closed_set_preserved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        report["frames"].append(frame_report)
    candidates = sorted(candidates, key=candidate_sort_key)
    report["candidate_count"] = len(candidates)
    report["error_count"] = sum(
        str(item.get("status") or "").startswith("open_vocabulary_error")
        for item in report["frames"]
    )
    report["status"] = (
        "completed_with_errors_closed_set_preserved"
        if report["error_count"]
        else "completed"
    )
    maximum_error_rate = float(
        config["performance"].get(
            "coarse_open_vocabulary_maximum_error_rate", 0.10
        )
    )
    selected_count = int(report["selected_frame_count"])
    error_rate = report["error_count"] / max(1, selected_count)
    report["error_rate"] = round(error_rate, 6)
    report["maximum_error_rate"] = maximum_error_rate
    report["formal_evidence_ready"] = bool(
        selected_count == 0 or error_rate <= maximum_error_rate
    )
    return candidates, report


def generate_open_vocabulary_fine_candidates(
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    detection_paths: dict[str, Path],
    windows: dict[str, list[tuple[float, float]]],
    config: dict[str, Any],
    frame_index: FineFrameIndex | None = None,
) -> tuple[list[ActionCandidate], dict[str, Any]]:
    """Add bounded hand-centred recall candidates inside candidate_fine."""

    perf = config["performance"]
    enabled = bool(
        perf.get("fine_roi_open_vocabulary_recall_enabled", False)
    )
    report: dict[str, Any] = {
        "schema_version": "visioncortex-fine-roi-open-vocabulary-recall/1",
        "enabled": enabled,
        "scope": "hand-centred bounded frames inside candidate_fine",
        "replaces_closed_set_candidates": False,
        "selected_frame_count": 0,
        "candidate_count": 0,
        "error_count": 0,
        "frames": [],
        "formal_evidence_ready": not enabled,
    }
    if not enabled:
        report["status"] = "disabled"
        return [], report
    settings = config.get("models", {}).get("open_vocabulary_key_frame") or {}
    if not settings.get("enabled"):
        report["status"] = "model_disabled"
        report["formal_evidence_ready"] = False
        return [], report

    maximum_per_window = max(
        0,
        int(perf.get("fine_roi_open_vocabulary_max_frames_per_window", 2)),
    )
    padding = max(
        0.0,
        float(perf.get("fine_roi_open_vocabulary_hand_padding_norm", 0.20)),
    )
    maximum_gap = float(
        perf.get("coarse_open_vocabulary_max_actor_object_gap_norm", 0.08)
    )
    selected: list[tuple[ViewInput, int, FrameEvidence]] = []

    def admit_suspicious(
        bucket: list[tuple[float, float, float, int, FrameEvidence]],
        frame: FrameEvidence,
    ) -> None:
        hands = [
            box
            for box in frame.detections
            if box.class_name in _ACTOR_CLASSES
        ]
        objects = [
            box
            for box in frame.detections
            if box.class_name not in _ACTOR_CLASSES | _NON_ACTION_CLASSES
        ]
        if not hands or objects:
            return
        bucket.append(
            (
                float(frame.motion_score),
                max(float(box.confidence) for box in hands),
                -float(frame.local_ms),
                -int(frame.frame_index),
                frame,
            )
        )
        bucket.sort(key=lambda item: item[:4], reverse=True)
        del bucket[maximum_per_window:]

    for view in views:
        indexed_windows = list(enumerate(windows.get(view.view_id, [])))
        buckets: dict[
            int, list[tuple[float, float, float, int, FrameEvidence]]
        ] = {window_index: [] for window_index, _ in indexed_windows}
        if frame_index is not None:
            for window_index, (start_ms, end_ms) in indexed_windows:
                for frame in frame_index.iter_frames(
                    view.view_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ):
                    admit_suspicious(buckets[window_index], frame)
        else:
            ordered_windows = sorted(
                indexed_windows, key=lambda item: (item[1][0], item[1][1])
            )
            next_window = 0
            active_windows: list[
                tuple[int, tuple[float, float]]
            ] = []
            for frame in iter_frame_evidence(detection_paths[view.view_id]):
                local_ms = float(frame.local_ms)
                active_windows = [
                    item for item in active_windows if item[1][1] >= local_ms
                ]
                while (
                    next_window < len(ordered_windows)
                    and ordered_windows[next_window][1][0] <= local_ms
                ):
                    active_windows.append(ordered_windows[next_window])
                    next_window += 1
                for window_index, (start_ms, end_ms) in active_windows:
                    if start_ms <= local_ms <= end_ms:
                        admit_suspicious(buckets[window_index], frame)
        for window_index, _ in indexed_windows:
            for _, _, _, _, frame in buckets[window_index]:
                selected.append((view, window_index, frame))
    report["selected_frame_count"] = len(selected)
    candidates: list[ActionCandidate] = []
    for view, window_index, selected_frame in selected:
        frame_report: dict[str, Any] = {
            "view_id": view.view_id,
            "window_index": window_index,
            "frame_index": selected_frame.frame_index,
            "local_ms": selected_frame.local_ms,
            "global_ms": selected_frame.global_ms,
        }
        frame = read_view_frame_at(
            view, infos[view.view_id], selected_frame.local_ms
        )
        if frame is None:
            frame_report["status"] = "frame_unreadable"
            report["frames"].append(frame_report)
            continue
        hand = max(
            (
                box
                for box in selected_frame.detections
                if box.class_name in _ACTOR_CLASSES
            ),
            key=lambda box: box.confidence,
        )
        x1, y1, x2, y2 = hand.xyxy_norm
        crop_norm = (
            max(0.0, x1 - padding),
            max(0.0, y1 - padding),
            min(1.0, x2 + padding),
            min(1.0, y2 + padding),
        )
        height, width = frame.shape[:2]
        left = max(0, int(crop_norm[0] * width))
        top = max(0, int(crop_norm[1] * height))
        right = min(width, int(np.ceil(crop_norm[2] * width)))
        bottom = min(height, int(np.ceil(crop_norm[3] * height)))
        if right - left < 4 or bottom - top < 4:
            frame_report["status"] = "invalid_hand_roi"
            report["frames"].append(frame_report)
            continue
        try:
            grounded, inference = _yolo_world_detections(
                frame[top:bottom, left:right], settings
            )
            crop_width = max(1e-9, crop_norm[2] - crop_norm[0])
            crop_height = max(1e-9, crop_norm[3] - crop_norm[1])
            objects = []
            for box in grounded:
                if box["class_name"] in _ACTOR_CLASSES | _NON_ACTION_CLASSES:
                    continue
                bx1, by1, bx2, by2 = box["xyxy_norm"]
                normalized = {
                    **box,
                    "xyxy_norm": [
                        crop_norm[0] + float(bx1) * crop_width,
                        crop_norm[1] + float(by1) * crop_height,
                        crop_norm[0] + float(bx2) * crop_width,
                        crop_norm[1] + float(by2) * crop_height,
                    ],
                    "detector_source": "yolo_world_v2_fine_hand_roi_recall",
                }
                gap = _box_edge_gap_norm(
                    {
                        "xyxy_norm": list(hand.xyxy_norm),
                    },
                    normalized,
                )
                if gap <= maximum_gap:
                    objects.append((gap, normalized))
            if not objects:
                frame_report.update(
                    {
                        "status": "no_manipulated_object_in_hand_roi",
                        "inference": inference,
                        "crop_norm": crop_norm,
                    }
                )
                report["frames"].append(frame_report)
                continue
            gap, obj = min(
                objects,
                key=lambda item: (
                    item[0],
                    -float(item[1]["confidence"]),
                ),
            )
            global_ms = float(
                selected_frame.global_ms
                if selected_frame.global_ms is not None
                else selected_frame.local_ms
            )
            half_window_ms = 2000.0
            local_start_ms = max(
                0.0, selected_frame.local_ms - half_window_ms
            )
            local_end_ms = min(
                float(infos[view.view_id].duration_ms),
                selected_frame.local_ms + half_window_ms,
            )
            candidates.append(
                ActionCandidate(
                    candidate_id=(
                        f"FINE-OPEN-VOCAB-{view.view_id}-"
                        f"{len(candidates) + 1:06d}"
                    ),
                    action_type=ActionType.HAND_OBJECT_CONTACT,
                    view_id=view.view_id,
                    role=view.role,
                    local_start_ms=local_start_ms,
                    local_end_ms=local_end_ms,
                    global_start_ms=(
                        global_ms
                        - (selected_frame.local_ms - local_start_ms)
                    ),
                    global_end_ms=(
                        global_ms
                        + (local_end_ms - selected_frame.local_ms)
                    ),
                    key_global_ms=global_ms,
                    objects=sorted({hand.class_name, str(obj["class_name"])}),
                    confidence=min(
                        float(hand.confidence), float(obj["confidence"])
                    ),
                    evidence=[
                        {
                            "frame_index": selected_frame.frame_index,
                            "hand_track_id": hand.track_id,
                            "hand_box_norm": list(hand.xyxy_norm),
                            "object": obj,
                            "actor_object_gap_norm": gap,
                            "hand_roi_norm": crop_norm,
                            "source_stage": "candidate_fine",
                            "recall_only": True,
                        }
                    ],
                    uncertainty=[
                        "手部ROI开放词汇结果只补充精扫召回，不能单独确认物理动作"
                    ],
                    provenance={
                        "source_stage": "candidate_fine",
                        "detector_source": (
                            "yolo_world_v2_fine_hand_roi_recall"
                        ),
                        "recall_only": True,
                    },
                )
            )
            frame_report.update(
                {
                    "status": "candidate_added",
                    "candidate_id": candidates[-1].candidate_id,
                    "inference": inference,
                    "crop_norm": crop_norm,
                }
            )
        except Exception as exc:
            frame_report.update(
                {
                    "status": "open_vocabulary_error_closed_set_preserved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        report["frames"].append(frame_report)
    report["candidate_count"] = len(candidates)
    failure_statuses = {"frame_unreadable", "invalid_hand_roi"}
    report["inference_error_count"] = sum(
        str(item.get("status") or "").startswith("open_vocabulary_error")
        for item in report["frames"]
    )
    report["error_count"] = sum(
        str(item.get("status") or "") in failure_statuses
        or str(item.get("status") or "").startswith(
            "open_vocabulary_error"
        )
        for item in report["frames"]
    )
    maximum_error_rate = float(
        perf.get("fine_roi_open_vocabulary_maximum_error_rate", 0.10)
    )
    error_rate = report["error_count"] / max(1, len(selected))
    report["error_rate"] = round(error_rate, 6)
    report["maximum_error_rate"] = maximum_error_rate
    report["formal_evidence_ready"] = bool(
        not selected or error_rate <= maximum_error_rate
    )
    report["status"] = (
        "completed_with_errors_closed_set_preserved"
        if report["error_count"]
        else "completed"
    )
    return sorted(candidates, key=candidate_sort_key), report
