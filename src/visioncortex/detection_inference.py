"""Explicit candidate PyTorch branch contracts; defaults preserve native behavior."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import logging
from pathlib import Path
import threading

from .schemas import ViewRole


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
