"""Explicit candidate PyTorch branch contracts; defaults preserve native behavior."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import logging
from pathlib import Path
import threading

import numpy as np

from .schemas import ViewRole


def prediction_arrays(predictions):
    """Transfer one detection batch, retaining the established box port.

    Tensor-backed predictions share a single device-to-host synchronization.
    Optional/test adapters exposing only detach/cpu/numpy keep their old port.
    No coordinates, thresholds or prediction ordering change.
    """
    boxes = [prediction.boxes for prediction in predictions]
    values = [(box.xyxyn, box.conf, box.cls) if box is not None else None for box in boxes]
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and all(value is None or all(isinstance(t, torch.Tensor) for t in value)
                                 for value in values):
        lengths = [len(value[0]) if value is not None else 0 for value in values]
        tensors = [torch.cat((value[0], value[1].reshape(-1, 1), value[2].reshape(-1, 1)), dim=1)
                   for value in values if value is not None and len(value[0])]
        combined = (torch.cat(tensors, dim=0).detach().cpu().numpy() if tensors
                    else np.empty((0, 6), dtype=np.float32))
        offset = 0
        result = []
        for length in lengths:
            rows = combined[offset:offset + length]
            result.append((np.clip(rows[:, :4], 0.0, 1.0), rows[:, 4], rows[:, 5].astype(int)))
            offset += length
        return result
    return [(np.clip(value[0].detach().cpu().numpy(), 0.0, 1.0),
             value[1].detach().cpu().numpy(), value[2].detach().cpu().numpy().astype(int))
            if value is not None else (np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int))
            for value in values]


def predict_complete_batch(model, *, source, options):
    """Never accept Ultralytics' time-limited, silently truncated NMS batch."""
    owner = threading.get_ident()
    logger = logging.getLogger("ultralytics")

    class TimeoutCapture(logging.Handler):
        timed_out = False

        def emit(self, record):
            if record.thread == owner and "NMS time limit" in record.getMessage():
                self.timed_out = True

    capture = TimeoutCapture()
    logger.addHandler(capture)
    try:
        for retry in range(3):
            capture.timed_out = False
            predictions = model.predict(source=source, **options)
            if not capture.timed_out:
                return predictions, retry
        raise RuntimeError("NMS postprocessing repeatedly timed out; refusing incomplete frame evidence")
    finally:
        logger.removeHandler(capture)


def prediction_branches(config: dict) -> dict[ViewRole, str]:
    values = config.get("models", {}).get("prediction_branch_by_role", {})
    if not isinstance(values, Mapping):
        raise ValueError("models.prediction_branch_by_role must be a role mapping")
    result = {}
    for name, branch in values.items():
        try:
            role = ViewRole(name)
        except ValueError as exc:
            raise ValueError(f"Unknown prediction branch role: {name}") from exc
        if branch not in ("one2one", "one2many"):
            raise ValueError("Prediction branch must be one2one or one2many")
        result[role] = branch
    return result


def prediction_contract(branch: str, model: Path, config: dict, image_size: int) -> dict:
    if branch not in ("one2one", "one2many"):
        raise ValueError("Prediction branch must be one2one or one2many")
    if model.suffix.lower() != ".pt":
        raise ValueError("Explicit prediction branches require PyTorch .pt weights; rebuild and verify engines separately")
    settings, perf = config["models"], config["performance"]
    confidence, iou = float(settings["confidence"]), float(settings["iou"])
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (confidence, iou)):
        raise ValueError("Prediction confidence and IoU must be finite values in [0, 1]")
    with model.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(
        schema_version="visioncortex-prediction-contract/1", branch=branch,
        model_path=str(model.resolve()), model_sha256=digest,
        confidence=confidence, iou=iou, image_size=image_size,
        max_detections=int(settings["max_detections"]), half=bool(perf["half"]),
        expected_class_count=int(settings["expected_class_count"]),
    )
