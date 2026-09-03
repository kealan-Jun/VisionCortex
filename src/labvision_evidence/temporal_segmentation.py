from __future__ import annotations

import hashlib
import json
import gc
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


SEGMENTATION_SCHEMA = "visioncortex-sam2-participant-continuity/1"

_MODEL_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
_MODEL_LOCK = threading.RLock()
_VALIDATED_ASSETS: set[tuple[str, str]] = set()


def release_temporal_segmentation_model_cache() -> int:
    """Release cached SAM2 predictors between events on low-memory hosts."""

    with _MODEL_LOCK:
        released = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return released


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(
        (config.get("models") or {}).get("temporal_participant_segmentation")
        or {}
    )


def validate_temporal_segmentation_runtime(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the pinned SAM2 runtime without allocating its GPU model."""

    settings = _settings(config)
    if not settings.get("enabled"):
        return {"enabled": False, "status": "disabled"}
    checkpoint = Path(str(settings.get("checkpoint_path") or "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint is missing: {checkpoint}")
    expected = str(settings.get("checkpoint_sha256") or "").strip().lower()
    if not expected:
        raise RuntimeError("SAM2 checkpoint_sha256 is required")
    validation_key = (str(checkpoint), expected)
    if validation_key not in _VALIDATED_ASSETS:
        actual = _sha256(checkpoint)
        if actual != expected:
            raise RuntimeError(
                f"SAM2 checkpoint hash mismatch: expected={expected} actual={actual}"
            )
        _VALIDATED_ASSETS.add(validation_key)
    try:
        package_version = version("SAM-2")
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError("Pinned SAM2 runtime is not installed") from exc
    device = str(settings.get("device") or "cuda")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("SAM2 requires CUDA but CUDA is unavailable")
    return {
        "enabled": True,
        "status": "validated",
        "backend": "SAM2VideoPredictor",
        "model": str(settings.get("model") or "sam2.1"),
        "model_config": str(settings.get("model_config") or ""),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": expected,
        "source_revision": str(settings.get("source_revision") or ""),
        "package_version": package_version,
        "device": device,
        "scope": "bounded_final_key_clips_only",
    }


def _load_predictor(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    settings = _settings(config)
    validation = validate_temporal_segmentation_runtime(config)
    key = (
        validation["checkpoint"],
        validation["checkpoint_sha256"],
        validation["model_config"],
        validation["device"],
        str(bool(settings.get("vos_optimized", False))),
        str(bool(settings.get("apply_postprocessing", False))),
    )
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached["predictor"], {
            **validation,
            "model_cache_reused": True,
            "model_load_seconds": 0.0,
        }

    from sam2.build_sam import build_sam2_video_predictor

    started = time.perf_counter()
    predictor = build_sam2_video_predictor(
        validation["model_config"],
        validation["checkpoint"],
        device=validation["device"],
        mode="eval",
        vos_optimized=bool(settings.get("vos_optimized", False)),
        apply_postprocessing=bool(settings.get("apply_postprocessing", False)),
    )
    load_seconds = time.perf_counter() - started
    _MODEL_CACHE[key] = {"predictor": predictor}
    return predictor, {
        **validation,
        "model_cache_reused": False,
        "model_load_seconds": round(load_seconds, 6),
    }


def _sample_indices(frame_count: int, maximum_frames: int, seed_index: int) -> list[int]:
    count = min(max(1, maximum_frames), frame_count)
    if count == 1:
        return [seed_index]
    indices = {
        round(index * (frame_count - 1) / (count - 1))
        for index in range(count)
    }
    closest = min(indices, key=lambda item: abs(item - seed_index))
    indices.remove(closest)
    indices.add(seed_index)
    return sorted(indices)


def _sample_clip(
    clip_path: Path,
    seed_frame: np.ndarray,
    output_dir: Path,
    *,
    seed_fraction: float,
    maximum_frames: int,
    jpeg_quality: int,
) -> tuple[list[int], int, tuple[int, int]]:
    capture = cv2.VideoCapture(str(clip_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise RuntimeError(f"SAM2 key clip contains no decodable frames: {clip_path}")
        seed_source_index = min(
            frame_count - 1,
            max(0, round((frame_count - 1) * min(1.0, max(0.0, seed_fraction)))),
        )
        source_indices = _sample_indices(
            frame_count, maximum_frames, seed_source_index
        )
        seed_position = source_indices.index(seed_source_index)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_shape: tuple[int, int] | None = None
        for output_index, source_index in enumerate(source_indices):
            if output_index == seed_position:
                frame = seed_frame.copy()
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"SAM2 frame decode failed: {clip_path} frame={source_index}"
                    )
            if output_shape is None:
                output_shape = (int(frame.shape[0]), int(frame.shape[1]))
            elif frame.shape[:2] != output_shape:
                frame = cv2.resize(
                    frame,
                    (output_shape[1], output_shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            destination = output_dir / f"{output_index:05d}.jpg"
            if not cv2.imwrite(
                str(destination),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise RuntimeError(f"Unable to write SAM2 bounded frame: {destination}")
        if output_shape is None:
            raise RuntimeError(f"SAM2 key clip sampling failed: {clip_path}")
        return source_indices, seed_position, output_shape
    finally:
        capture.release()


def _mask_record(mask_logits: Any, width: int, height: int) -> dict[str, Any]:
    import torch

    binary = (mask_logits > 0.0).squeeze()
    if binary.ndim != 2:
        raise RuntimeError(f"Unexpected SAM2 mask shape: {tuple(binary.shape)}")
    ys, xs = torch.where(binary)
    pixels = int(binary.sum().item())
    if not pixels:
        return {"present": False, "mask_pixels": 0, "mask_area_norm": 0.0}
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return {
        "present": True,
        "mask_pixels": pixels,
        "mask_area_norm": round(pixels / max(1, width * height), 8),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xyxy_norm": [
            round(x1 / width, 8),
            round(y1 / height, 8),
            round(x2 / width, 8),
            round(y2 / height, 8),
        ],
    }


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def _refine_seed_boxes(
    boxes: Sequence[dict[str, Any]],
    seed_masks: dict[int, dict[str, Any]],
    *,
    minimum_seed_iou: float,
    maximum_area_expansion_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refined: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for object_id, original in enumerate(boxes, 1):
        box = dict(original)
        original_xyxy = [float(item) for item in box.get("xyxy_norm") or ()]
        mask = seed_masks.get(object_id) or {}
        mask_xyxy = [float(item) for item in mask.get("bbox_xyxy_norm") or ()]
        original_area = (
            max(0.0, original_xyxy[2] - original_xyxy[0])
            * max(0.0, original_xyxy[3] - original_xyxy[1])
            if len(original_xyxy) == 4
            else 0.0
        )
        mask_area = (
            max(0.0, mask_xyxy[2] - mask_xyxy[0])
            * max(0.0, mask_xyxy[3] - mask_xyxy[1])
            if len(mask_xyxy) == 4
            else 0.0
        )
        iou = (
            _box_iou(original_xyxy, mask_xyxy)
            if len(original_xyxy) == len(mask_xyxy) == 4
            else 0.0
        )
        accepted = bool(
            mask.get("present")
            and iou >= minimum_seed_iou
            and original_area > 0.0
            and mask_area <= original_area * maximum_area_expansion_ratio
        )
        if accepted:
            box["xyxy_norm"] = mask_xyxy
            box["segmentation_refined"] = True
            box["segmentation_source"] = "sam2.1_video_mask_bbox"
        refined.append(box)
        decisions.append(
            {
                "object_id": object_id,
                "class_name": str(box.get("class_name") or ""),
                "accepted": accepted,
                "seed_box_iou": round(iou, 8),
                "original_bbox_xyxy_norm": original_xyxy,
                "mask_bbox_xyxy_norm": mask_xyxy or None,
                "area_expansion_ratio": (
                    round(mask_area / original_area, 8)
                    if original_area > 0.0
                    else None
                ),
            }
        )
    return refined, decisions


def audit_participant_continuity(
    clip_path: Path,
    seed_frame: np.ndarray,
    participant_boxes: Sequence[dict[str, Any]],
    work_root: Path,
    config: dict[str, Any],
    *,
    event_id: str,
    view_id: str,
    action_type: str,
    seed_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Propagate final participant boxes through one bounded key clip.

    The function never reads the original long recording. It samples a fixed
    number of frames from the already-derived key clip, seeds SAM2 with the
    participant-only boxes selected after semantic review, and propagates in
    both temporal directions. The receipt is continuity evidence only; it can
    never promote an unconfirmed action or substitute for liquid-state proof.
    """

    settings = _settings(config)
    enabled_actions = {
        str(item) for item in settings.get("enabled_actions") or []
    }
    if not settings.get("enabled"):
        return list(map(dict, participant_boxes)), {"status": "disabled"}
    if enabled_actions and action_type not in enabled_actions:
        return list(map(dict, participant_boxes)), {
            "status": "not_applicable_action",
            "action_type": action_type,
        }
    if not participant_boxes:
        return [], {
            "schema_version": SEGMENTATION_SCHEMA,
            "status": "not_applicable_no_participant_box",
            "event_id": event_id,
            "view_id": view_id,
        }
    if not clip_path.is_file():
        raise FileNotFoundError(f"SAM2 bounded key clip is missing: {clip_path}")

    fingerprint_payload = {
        "schema_version": SEGMENTATION_SCHEMA,
        "event_id": event_id,
        "view_id": view_id,
        "action_type": action_type,
        "clip_path": str(clip_path),
        "clip_size": clip_path.stat().st_size,
        "clip_mtime_ns": clip_path.stat().st_mtime_ns,
        "seed_fraction": round(seed_fraction, 8),
        "participant_boxes": list(participant_boxes),
        "model": str(settings.get("model") or ""),
        "checkpoint_sha256": str(settings.get("checkpoint_sha256") or ""),
        "maximum_frames_per_clip": int(
            settings.get("maximum_frames_per_clip", 9)
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    run_root = work_root / event_id / view_id / fingerprint
    frames_root = run_root / "frames"
    receipt_path = run_root / "receipt.json"
    if receipt_path.is_file():
        cached = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        if (
            cached.get("schema_version") == SEGMENTATION_SCHEMA
            and cached.get("input_fingerprint") == fingerprint
            and cached.get("status") == "completed"
        ):
            return [
                dict(item) for item in cached.get("refined_participant_boxes") or []
            ], {**cached, "cache_reused": True}

    source_indices, seed_position, (height, width) = _sample_clip(
        clip_path,
        seed_frame,
        frames_root,
        seed_fraction=seed_fraction,
        maximum_frames=int(settings.get("maximum_frames_per_clip", 9)),
        jpeg_quality=int(settings.get("jpeg_quality", 95)),
    )
    predictor, model_receipt = _load_predictor(config)
    device = str(settings.get("device") or "cuda")
    import torch

    frame_masks: dict[int, dict[int, dict[str, Any]]] = {}
    inference_started = time.perf_counter()
    memory_before = (
        int(torch.cuda.memory_allocated()) if device.startswith("cuda") else 0
    )
    with _MODEL_LOCK, torch.inference_mode():
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.autocast("cpu", enabled=False)
        )
        state = None
        try:
            with autocast:
                state = predictor.init_state(
                    str(frames_root),
                    offload_video_to_cpu=bool(
                        settings.get("offload_video_to_cpu", True)
                    ),
                    offload_state_to_cpu=bool(
                        settings.get("offload_state_to_cpu", False)
                    ),
                )
                for object_id, box in enumerate(participant_boxes, 1):
                    normalized = [float(item) for item in box["xyxy_norm"]]
                    absolute = np.asarray(
                        [
                            normalized[0] * width,
                            normalized[1] * height,
                            normalized[2] * width,
                            normalized[3] * height,
                        ],
                        dtype=np.float32,
                    )
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=seed_position,
                        obj_id=object_id,
                        box=absolute,
                    )

                def collect(iterator: Any) -> None:
                    for frame_index, object_ids, mask_logits in iterator:
                        per_object = frame_masks.setdefault(int(frame_index), {})
                        for mask_index, object_id in enumerate(object_ids):
                            per_object[int(object_id)] = _mask_record(
                                mask_logits[mask_index], width, height
                            )

                collect(
                    predictor.propagate_in_video(
                        state,
                        start_frame_idx=seed_position,
                        max_frame_num_to_track=len(source_indices),
                    )
                )
                if seed_position > 0:
                    collect(
                        predictor.propagate_in_video(
                            state,
                            start_frame_idx=seed_position,
                            max_frame_num_to_track=seed_position + 1,
                            reverse=True,
                        )
                    )
        finally:
            if state is not None:
                predictor.reset_state(state)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    minimum_presence_by_action = {
        str(key): float(value)
        for key, value in dict(
            settings.get("minimum_presence_ratio_by_action") or {}
        ).items()
    }
    minimum_presence = float(
        minimum_presence_by_action.get(
            action_type, settings.get("minimum_presence_ratio", 0.34)
        )
    )
    continuity_objects: list[dict[str, Any]] = []
    for object_id, box in enumerate(participant_boxes, 1):
        records = [
            (frame_masks.get(frame_index) or {}).get(object_id) or {}
            for frame_index in range(len(source_indices))
        ]
        presence_count = sum(bool(record.get("present")) for record in records)
        presence_ratio = presence_count / max(1, len(source_indices))
        continuity_objects.append(
            {
                "object_id": object_id,
                "class_name": str(box.get("class_name") or ""),
                "presence_count": presence_count,
                "sampled_frame_count": len(source_indices),
                "presence_ratio": round(presence_ratio, 8),
                "passed": presence_ratio >= minimum_presence,
            }
        )
    refined_boxes, refinement = _refine_seed_boxes(
        participant_boxes,
        frame_masks.get(seed_position) or {},
        minimum_seed_iou=float(settings.get("minimum_seed_box_iou", 0.35)),
        maximum_area_expansion_ratio=float(
            settings.get("maximum_seed_area_expansion_ratio", 1.35)
        ),
    )
    receipt = {
        "schema_version": SEGMENTATION_SCHEMA,
        "status": "completed",
        "passed": all(item["passed"] for item in continuity_objects),
        "policy": (
            "bounded participant continuity and seed-box refinement only; "
            "never action confirmation or liquid-state proof"
        ),
        "event_id": event_id,
        "view_id": view_id,
        "action_type": action_type,
        "input_fingerprint": fingerprint,
        "cache_reused": False,
        "source_scope": "already_derived_key_clip",
        "source_copy_bytes": 0,
        "full_timeline_inference": False,
        "ark_calls": 0,
        "token_usage": 0,
        "sampled_frame_count": len(source_indices),
        "source_frame_indices": source_indices,
        "seed_sample_position": seed_position,
        "minimum_presence_ratio": minimum_presence,
        "objects": continuity_objects,
        "frames": [
            {
                "sample_position": frame_index,
                "source_frame_index": source_indices[frame_index],
                "objects": [
                    {"object_id": object_id, **record}
                    for object_id, record in sorted(
                        (frame_masks.get(frame_index) or {}).items()
                    )
                ],
            }
            for frame_index in range(len(source_indices))
        ],
        "seed_box_refinement": refinement,
        "refined_participant_boxes": refined_boxes,
        "model_runtime": model_receipt,
        "inference_seconds": round(inference_seconds, 6),
        "fps_after_model_load": round(
            len(source_indices) / max(inference_seconds, 1e-9), 6
        ),
        "gpu_memory_allocated_before_mib": round(memory_before / 1024**2, 3),
        "gpu_peak_allocated_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 3)
            if device.startswith("cuda")
            else 0.0
        ),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(receipt_path)
    return refined_boxes, receipt
