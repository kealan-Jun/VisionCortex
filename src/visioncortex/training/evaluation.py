"""Evaluation responsibilities for explicitly authorized training."""
from __future__ import annotations

import json
import numbers
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ..telemetry import ResourceMonitor

from .dataset_common import _sha256, DATASET_SCHEMA, EVALUATION_SCHEMA, PUBLIC_DATASET_SCHEMA, PUBLIC_UNION_SCHEMA, TRAINING_TRUTH_STATUSES

def evaluate_yolo_model_on_human_truth(
    dataset_root: Path,
    model_path: Path,
    output: Path,
    *,
    split: str = "test",
    image_size: int = 640,
    batch: int = 32,
    device: str = "0",
    workers: int = 8,
) -> dict[str, Any]:
    """Evaluate a candidate only against a held-out trusted human-label split."""

    dataset_root = dataset_root.resolve()
    model_path = model_path.resolve()
    output = output.resolve()
    if split not in {"val", "test"}:
        raise ValueError("YOLO human-truth evaluation split must be val or test")
    if image_size < 64 or batch < 1 or workers < 0:
        raise ValueError("YOLO human-truth evaluation limits are invalid")
    if output.exists():
        raise FileExistsError(f"YOLO evaluation output already exists: {output}")
    receipt_path = dataset_root / "dataset-receipt.json"
    dataset_yaml = dataset_root / "dataset.yaml"
    if not receipt_path.is_file() or not dataset_yaml.is_file():
        raise FileNotFoundError("YOLO evaluation dataset receipt or YAML is missing")
    dataset_receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (
        dataset_receipt.get("schema_version")
        not in {DATASET_SCHEMA, PUBLIC_DATASET_SCHEMA, PUBLIC_UNION_SCHEMA}
        or dataset_receipt.get("status") != "completed"
        or dataset_receipt.get("truth_status") not in TRAINING_TRUTH_STATUSES
        or dataset_receipt.get("nas_accessed") is not False
    ):
        raise RuntimeError("YOLO evaluation dataset is not trusted human ground truth")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO evaluation model is missing: {model_path}")

    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    temporary_telemetry = output.parent / (
        f".{output.name}-resource-telemetry-{uuid.uuid4().hex[:8]}.json"
    )
    monitor = ResourceMonitor(temporary_telemetry, interval_seconds=0.25)
    monitor.start()
    monitor.set_stage("public_yolo_human_truth_evaluation")
    try:
        metrics = YOLO(str(model_path)).val(
            data=str(dataset_yaml),
            split=split,
            imgsz=image_size,
            batch=batch,
            device=device,
            workers=workers,
            project=str(output.parent),
            name=output.name,
            exist_ok=False,
            plots=True,
            verbose=True,
        )
    finally:
        telemetry = monitor.stop()
    ended = datetime.now(timezone.utc)
    if not output.is_dir():
        raise RuntimeError(f"YOLO evaluation did not create output: {output}")
    telemetry_path = output / "resource-telemetry.json"
    telemetry_live_path = output / "resource-telemetry_live.json"
    os.replace(temporary_telemetry, telemetry_path)
    os.replace(
        temporary_telemetry.with_name(f"{temporary_telemetry.stem}_live.json"),
        telemetry_live_path,
    )
    results = {
        str(key): float(value)
        for key, value in dict(getattr(metrics, "results_dict", {}) or {}).items()
        if isinstance(value, numbers.Real)
    }
    model_names = {
        int(key): str(value)
        for key, value in dict(getattr(metrics, "names", {}) or {}).items()
    }
    box_metrics = getattr(metrics, "box", None)
    raw_class_maps = getattr(box_metrics, "maps", None)
    class_maps = list(raw_class_maps) if raw_class_maps is not None else []
    raw_class_indexes = getattr(box_metrics, "ap_class_index", None)
    class_indexes = list(raw_class_indexes) if raw_class_indexes is not None else []
    class_positions = {
        int(class_id): position for position, class_id in enumerate(class_indexes)
    }

    def class_metric(name: str, class_id: int) -> float | None:
        raw_values = getattr(box_metrics, name, None)
        values = list(raw_values) if raw_values is not None else []
        if class_positions:
            position = class_positions.get(class_id)
            if position is None:
                return None
        else:
            position = class_id
        if position >= len(values):
            return None
        return round(float(values[position]), 6)

    def class_map_metric(class_id: int) -> float | None:
        if class_positions and class_id not in class_positions:
            return None
        if len(class_maps) == len(model_names):
            position = class_id
        elif class_positions:
            position = class_positions.get(class_id)
            if position is None:
                return None
        else:
            position = class_id
        if position >= len(class_maps):
            return None
        return round(float(class_maps[position]), 6)

    per_class = [
        {
            "class_id": class_id,
            "class_name": model_names.get(class_id, str(class_id)),
            "precision": class_metric("p", class_id),
            "recall": class_metric("r", class_id),
            "f1": class_metric("f1", class_id),
            "map50": class_metric("ap50", class_id),
            "map50_95": class_map_metric(class_id),
        }
        for class_id in sorted(model_names)
    ]
    receipt = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "completed",
        "truth_status": dataset_receipt.get("truth_status"),
        "dataset_receipt": str(receipt_path),
        "dataset_receipt_sha256": _sha256(receipt_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "split": split,
        "image_size": image_size,
        "batch": batch,
        "device": device,
        "workers": workers,
        "metrics": results,
        "per_class": per_class,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "resource_telemetry": str(telemetry_path),
        "resource_summary": telemetry.get("stage_summaries") or {},
        "telemetry_health": telemetry.get("monitor_health") or {},
        "production_certified": False,
        "certification_required_before_deployment": True,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
    }
    (output / "visioncortex-evaluation-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt
