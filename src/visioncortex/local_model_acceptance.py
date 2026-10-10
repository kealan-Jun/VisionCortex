from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .archive import _grounding_dino_key_frame_detections
from .detection import FramePacket, RoleScanner, validate_models
from .liquid_semantic import analyze_liquid_semantics
from .schemas import ViewInput, ViewRole
from .telemetry import ResourceMonitor
from .temporal_segmentation import audit_participant_continuity


ACCEPTANCE_SCHEMA = "visioncortex-local-real-model-acceptance/1"


def _select_public_example(dataset_root: Path) -> Path:
    test_root = dataset_root.resolve() / "LabPics Chemistry" / "Test"
    examples = sorted(
        candidate
        for candidate in test_root.iterdir()
        if candidate.is_dir() and (candidate / "Image.jpg").is_file()
    )
    if not examples:
        raise RuntimeError(f"No local LabPics Test examples found: {test_root}")
    # Select the middle item to avoid treating a directory edge case as a
    # representative runtime acceptance input.
    return examples[len(examples) // 2]


def _vessel_seed_box(example: Path, shape: tuple[int, int]) -> list[float]:
    mask_path = example / "SemanticMaps" / "FullImage" / "Vessel.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Public human vessel annotation is missing: {mask_path}")
    height, width = shape
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise RuntimeError(f"Public vessel annotation is empty: {mask_path}")
    return [
        float(xs.min()) / width,
        float(ys.min()) / height,
        float(xs.max() + 1) / width,
        float(ys.max() + 1) / height,
    ]


def _write_bounded_clip(frame: np.ndarray, destination: Path) -> None:
    height, width = frame.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        3.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create bounded SAM2 acceptance clip: {destination}")
    try:
        for offset in (-2, -1, 0, 1, 2, 1, 0, -1, -2):
            transform = np.float32([[1, 0, offset], [0, 1, 0]])
            shifted = cv2.warpAffine(
                frame,
                transform,
                (width, height),
                borderMode=cv2.BORDER_REFLECT,
            )
            writer.write(shifted)
    finally:
        writer.release()
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Bounded SAM2 acceptance clip is empty: {destination}")


def _closed_set_acceptance(
    frame: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = []
    for role in ViewRole:
        view = ViewInput(
            view_id=f"acceptance-{role.value}", role=role, video=Path("public-image.jpg")
        )
        packet = FramePacket(
            view=view,
            frame_index=0,
            local_ms=0.0,
            frame=frame,
            gray=gray,
            previous_gray=None,
            motion_score=0.0,
        )
        started = time.perf_counter()
        scanner = RoleScanner(role, config, image_size=640, batch_size=1)
        try:
            detections = scanner.infer([packet])[0]
            rows.append(
                {
                    "role": role.value,
                    "backend": (
                        "TensorRT"
                        if scanner.model_path.suffix.lower() == ".engine"
                        else "PyTorch"
                    ),
                    "model_path": str(scanner.model_path),
                    "detection_count": len(detections),
                    "detections": [item.model_dump(mode="json") for item in detections],
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "inference_batch_sizes": scanner.last_inference_batch_sizes,
                    "batch_contractions": scanner.batch_contractions,
                }
            )
        finally:
            scanner.close()
    return {"status": "completed", "roles": rows}


def _yolo_world_acceptance(
    frame: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    import torch
    from .open_vocabulary_runtime import load_yolo_world_with_local_clip

    settings = config["models"]["open_vocabulary_key_frame"]
    prompts = [
        "gloved hand",
        "laboratory beaker held by hand",
        "transparent laboratory container held by hand",
        "pipette tip",
    ]
    started = time.perf_counter()
    model = load_yolo_world_with_local_clip(settings)
    model.set_classes(prompts)
    result = model.predict(
        frame,
        device=int(settings.get("device", 0)),
        imgsz=640,
        conf=float(settings.get("confidence", 0.03)),
        iou=float(settings.get("iou", 0.50)),
        verbose=False,
    )[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    boxes = []
    height, width = frame.shape[:2]
    for class_index, confidence, coordinates in zip(
        result.boxes.cls, result.boxes.conf, result.boxes.xyxy, strict=True
    ):
        x1, y1, x2, y2 = (float(item) for item in coordinates)
        boxes.append(
            {
                "prompt": str(result.names[int(class_index)]),
                "confidence": float(confidence),
                "xyxy_norm": [x1 / width, y1 / height, x2 / width, y2 / height],
            }
        )
    return {
        "status": "completed",
        "model": str(settings["model_path"]),
        "prompts": prompts,
        "detection_count": len(boxes),
        "detections": boxes,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def run_local_real_model_acceptance(
    dataset_root: Path,
    output: Path,
    config: dict[str, Any],
) -> Path:
    """Execute every local production CV model on bounded non-NAS inputs."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Local model acceptance output already exists: {output}")
    output.mkdir(parents=True)
    example = _select_public_example(dataset_root)
    image_path = example / "Image.jpg"
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Cannot decode local public acceptance image: {image_path}")

    monitor = ResourceMonitor(output / "resource-telemetry.json", interval_seconds=0.25)
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    stages: dict[str, Any] = {}
    monitor.start()
    try:
        monitor.set_stage("model_runtime_preflight")
        stage_started = time.perf_counter()
        model_validation = validate_models(config)
        stages["model_runtime_preflight"] = {
            "status": "completed",
            "elapsed_seconds": round(time.perf_counter() - stage_started, 6),
            "validation": model_validation,
        }

        monitor.set_stage("closed_set_yolo_tensorrt")
        stages["closed_set_yolo_tensorrt"] = _closed_set_acceptance(frame, config)

        monitor.set_stage("yolo_world_key_frame")
        stages["yolo_world_key_frame"] = _yolo_world_acceptance(frame, config)

        monitor.set_stage("grounding_dino_key_frame")
        grounding_started = time.perf_counter()
        grounding_boxes, grounding_receipt = _grounding_dino_key_frame_detections(
            frame,
            {"beaker", "container", "gloved_hand"},
            config["models"]["open_vocabulary_key_frame"],
        )
        stages["grounding_dino_key_frame"] = {
            **grounding_receipt,
            "admitted_detections": grounding_boxes,
            "elapsed_seconds_total": round(time.perf_counter() - grounding_started, 6),
        }

        monitor.set_stage("labpics_liquid_semantic")
        stages["labpics_liquid_semantic"] = analyze_liquid_semantics(
            frame,
            config,
            output / "labpics-liquid-semantic",
            artifact_stem=example.name,
        )

        monitor.set_stage("sam2_bounded_participant_continuity")
        clip = output / "sam2-bounded-input" / "public-vessel-nine-frames.mp4"
        _write_bounded_clip(frame, clip)
        seed_box = _vessel_seed_box(example, frame.shape[:2])
        refined, sam_receipt = audit_participant_continuity(
            clip,
            frame,
            [
                {
                    "class_name": "vessel",
                    "confidence": 1.0,
                    "xyxy_norm": seed_box,
                    "seed_source": "public_human_vessel_mask",
                }
            ],
            output / "sam2-work",
            config,
            event_id="LOCAL-PUBLIC-MODEL-ACCEPTANCE",
            view_id="labpics-public-test",
            action_type="liquid_movement",
            seed_fraction=0.5,
        )
        stages["sam2_bounded_participant_continuity"] = {
            **sam_receipt,
            "seed_box_xyxy_norm": seed_box,
            "refined_boxes": refined,
        }
    finally:
        telemetry = monitor.stop()

    completed_stages = {
        "model_runtime_preflight": stages.get("model_runtime_preflight", {}).get("status")
        == "completed",
        "closed_set_yolo_tensorrt": stages.get("closed_set_yolo_tensorrt", {}).get("status")
        == "completed",
        "yolo_world_key_frame": stages.get("yolo_world_key_frame", {}).get("status")
        == "completed",
        "grounding_dino_key_frame": stages.get("grounding_dino_key_frame", {}).get("status")
        == "executed",
        "labpics_liquid_semantic": stages.get("labpics_liquid_semantic", {}).get("status")
        == "completed",
        "sam2_bounded_participant_continuity": stages.get(
            "sam2_bounded_participant_continuity", {}
        ).get("status")
        == "completed",
    }
    ended_wall = datetime.now(timezone.utc)
    receipt = output / "local-real-model-acceptance.json"
    created_paths = [
        str(path) for path in sorted(output.rglob("*")) if path.is_file()
    ]
    created_paths.append(str(receipt))
    payload = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "status": "passed" if all(completed_stages.values()) else "failed",
        "passed": all(completed_stages.values()),
        "scope": "real_local_gpu_structural_model_execution_on_bounded_public_input",
        "quality_claim": "structural_execution_only_not_production_domain_certification",
        "production_quality_certified": False,
        "generalization_claim": False,
        "input": {
            "dataset": "LabPics Chemistry V2 Test",
            "example_id": example.name,
            "image_path": str(image_path),
            "image_shape": list(frame.shape),
            "truth_source": "public_human_annotations",
        },
        "completed_stages": completed_stages,
        "stages": stages,
        "started_at": started_wall.isoformat(),
        "ended_at": ended_wall.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "resource_telemetry": str(output / "resource-telemetry.json"),
        "resource_summary": telemetry.get("stage_summaries") or {},
        "telemetry_health": telemetry.get("monitor_health") or {},
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_paths": created_paths,
    }
    temporary = receipt.with_name(f".{receipt.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, receipt)
    return receipt
