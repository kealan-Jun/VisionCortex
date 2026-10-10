"""Run responsibilities for explicitly authorized training."""
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ..telemetry import ResourceMonitor

from .dataset_common import _sha256, DATASET_SCHEMA, TRAINING_SCHEMA, PUBLIC_DATASET_SCHEMA, PUBLIC_UNION_SCHEMA, DATASET_INTEGRITY_AUDIT_SCHEMA, TRAINING_TRUTH_STATUSES

def train_yolo_model(
    dataset_root: Path,
    base_model: Path,
    output: Path,
    *,
    epochs: int = 100,
    image_size: int = 1280,
    batch: int = 8,
    device: str = "0",
    patience: int = 20,
    max_hours: float = 2.0,
    workers: int = 8,
    optimizer: str = "auto",
    learning_rate: float = 0.01,
    final_learning_rate_fraction: float = 0.01,
    cosine_schedule: bool = False,
    warmup_epochs: float = 3.0,
    close_mosaic: int = 10,
    dataset_integrity_audit: Path | None = None,
) -> dict[str, Any]:
    """Run a real Ultralytics training job from reviewed ground truth only."""

    dataset_root = dataset_root.resolve()
    base_model = base_model.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"YOLO training output already exists: {output}")
    if (
        epochs < 1
        or image_size < 64
        or batch < 1
        or patience < 0
        or max_hours <= 0.0
        or max_hours > 24.0
        or workers < 0
        or optimizer not in {
            "auto",
            "SGD",
            "Adam",
            "AdamW",
            "MuSGD",
            "RMSProp",
            "NAdam",
            "RAdam",
        }
        or not 1e-6 <= learning_rate <= 0.1
        or not 0.001 <= final_learning_rate_fraction <= 1.0
        or not 0.0 <= warmup_epochs <= 10.0
        or close_mosaic < 0
        or close_mosaic > epochs
    ):
        raise ValueError("YOLO training limits are invalid")
    receipt_path = dataset_root / "dataset-receipt.json"
    dataset_yaml = dataset_root / "dataset.yaml"
    if not receipt_path.is_file() or not dataset_yaml.is_file():
        raise FileNotFoundError("YOLO training dataset receipt or dataset.yaml is missing")
    dataset_receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (
        dataset_receipt.get("schema_version")
        not in {DATASET_SCHEMA, PUBLIC_DATASET_SCHEMA, PUBLIC_UNION_SCHEMA}
        or dataset_receipt.get("status") != "completed"
        or dataset_receipt.get("truth_status") not in TRAINING_TRUTH_STATUSES
        or dataset_receipt.get("nas_accessed") is not False
        or int(dataset_receipt.get("source_copy_bytes") or 0) != 0
    ):
        raise RuntimeError("YOLO training dataset is not trusted human ground truth")
    integrity_audit_path = (
        dataset_integrity_audit.resolve()
        if dataset_integrity_audit is not None
        else None
    )
    integrity_audit: dict[str, Any] | None = None
    if (
        dataset_receipt.get("schema_version") == PUBLIC_UNION_SCHEMA
        and integrity_audit_path is None
    ):
        raise RuntimeError(
            "Mapped public YOLO unions require a passing dataset integrity audit"
        )
    if integrity_audit_path is not None:
        if not integrity_audit_path.is_file():
            raise RuntimeError("YOLO dataset integrity audit file is missing")
        integrity_audit = json.loads(
            integrity_audit_path.read_text(encoding="utf-8-sig")
        )
        if (
            integrity_audit.get("schema_version")
            != DATASET_INTEGRITY_AUDIT_SCHEMA
            or integrity_audit.get("passed") is not True
            or integrity_audit.get("dataset_receipt_sha256") != _sha256(receipt_path)
            or int(integrity_audit.get("cross_split_content_hash_count") or 0) != 0
            or int(integrity_audit.get("source_copy_bytes") or 0) != 0
            or integrity_audit.get("nas_accessed") is not False
        ):
            raise RuntimeError(
                "YOLO dataset integrity audit is failed, unsafe, or mismatched"
            )
    if not base_model.is_file():
        raise FileNotFoundError(f"YOLO base model is missing: {base_model}")

    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    model = YOLO(str(base_model))
    if not hasattr(model, "add_callback"):
        raise RuntimeError("Ultralytics model does not support bounded training callbacks")
    deadline_monotonic = time.monotonic() + max_hours * 3600.0

    def enforce_wall_time_bound(trainer: Any) -> None:
        if time.monotonic() >= deadline_monotonic:
            trainer.stop = True

    model.add_callback("on_train_epoch_end", enforce_wall_time_bound)
    temporary_telemetry = output.parent / (
        f".{output.name}-resource-telemetry-{uuid.uuid4().hex[:8]}.json"
    )
    monitor = ResourceMonitor(temporary_telemetry, interval_seconds=0.25)
    monitor.start()
    monitor.set_stage("public_yolo_training")
    try:
        result = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            device=device,
            project=str(output.parent),
            name=output.name,
            exist_ok=False,
            plots=True,
            verbose=True,
            patience=patience,
            workers=workers,
            optimizer=optimizer,
            lr0=learning_rate,
            lrf=final_learning_rate_fraction,
            cos_lr=cosine_schedule,
            warmup_epochs=warmup_epochs,
            close_mosaic=close_mosaic,
        )
    finally:
        telemetry = monitor.stop()
    best = output / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"YOLO training completed without best.pt: {best}")
    telemetry_path = output / "resource-telemetry.json"
    telemetry_live_path = output / "resource-telemetry_live.json"
    os.replace(temporary_telemetry, telemetry_path)
    os.replace(
        temporary_telemetry.with_name(f"{temporary_telemetry.stem}_live.json"),
        telemetry_live_path,
    )
    ended = datetime.now(timezone.utc)
    results_csv = output / "results.csv"
    metric_rows: list[dict[str, str]] = []
    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8-sig") as handle:
            metric_rows = [
                {str(key).strip(): str(value).strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    final_metrics = {
        key: float(value)
        for key, value in (metric_rows[-1] if metric_rows else {}).items()
        if key.startswith("metrics/") and value
    }
    map50_key = "metrics/mAP50(B)"
    map_key = "metrics/mAP50-95(B)"
    best_metric_row_index = (
        max(
            range(len(metric_rows)),
            key=lambda index: (
                0.1 * float(metric_rows[index].get(map50_key) or "-inf")
                + 0.9 * float(metric_rows[index].get(map_key) or "-inf")
            ),
        )
        if metric_rows
        else None
    )
    best_metric_row = (
        metric_rows[best_metric_row_index]
        if best_metric_row_index is not None
        else {}
    )
    best_metrics = {
        key: float(value)
        for key, value in best_metric_row.items()
        if key.startswith("metrics/") and value
    }
    best_csv_epoch = (
        int(float(best_metric_row["epoch"]))
        if best_metric_row.get("epoch") is not None
        else None
    )
    trainer = getattr(model, "trainer", None)
    effective_optimizer_instance = getattr(trainer, "optimizer", None)
    effective_optimizer_name = (
        type(effective_optimizer_instance).__name__
        if effective_optimizer_instance is not None
        else None
    )
    effective_initial_learning_rates = sorted(
        {
            float(group.get("initial_lr", group.get("lr")))
            for group in (
                getattr(effective_optimizer_instance, "param_groups", None) or []
            )
            if group.get("initial_lr", group.get("lr")) is not None
        }
    )
    receipt = {
        "schema_version": TRAINING_SCHEMA,
        "status": "completed",
        "truth_status": dataset_receipt.get("truth_status"),
        "dataset_receipt": str(receipt_path),
        "dataset_receipt_sha256": _sha256(receipt_path),
        "dataset_integrity_audit": (
            str(integrity_audit_path) if integrity_audit_path is not None else None
        ),
        "dataset_integrity_audit_sha256": (
            _sha256(integrity_audit_path)
            if integrity_audit_path is not None
            else None
        ),
        "dataset_integrity_gate_passed": (
            integrity_audit.get("passed") is True
            if integrity_audit is not None
            else None
        ),
        "base_model": str(base_model),
        "base_model_sha256": _sha256(base_model),
        "best_model": str(best),
        "best_model_sha256": _sha256(best),
        "epochs": epochs,
        "image_size": image_size,
        "batch": batch,
        "device": device,
        "patience": patience,
        "max_hours": max_hours,
        "wall_time_bound_enforcement": "epoch_boundary_callback",
        "workers": workers,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "requested_optimizer": optimizer,
        "requested_learning_rate": learning_rate,
        "effective_optimizer": effective_optimizer_name,
        "effective_initial_learning_rates": effective_initial_learning_rates,
        "optimizer_auto_may_override_requested_learning_rate": optimizer == "auto",
        "final_learning_rate_fraction": final_learning_rate_fraction,
        "cosine_schedule": cosine_schedule,
        "warmup_epochs": warmup_epochs,
        "close_mosaic": close_mosaic,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "completed_epoch_count": len(metric_rows),
        "final_validation_metrics": final_metrics,
        "best_validation_metric": (
            "ultralytics_detection_fitness="
            "0.1*metrics/mAP50(B)+0.9*metrics/mAP50-95(B)"
        ),
        "best_validation_csv_epoch": best_csv_epoch,
        "best_validation_completed_epoch_number": (
            best_metric_row_index + 1
            if best_metric_row_index is not None
            else None
        ),
        "best_validation_metrics": best_metrics,
        "results_csv": str(results_csv) if results_csv.is_file() else None,
        "trainer_save_dir": str(getattr(result, "save_dir", output)),
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
    (output / "visioncortex-training-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt
