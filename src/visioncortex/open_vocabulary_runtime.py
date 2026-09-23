from __future__ import annotations

from functools import wraps
from pathlib import Path
import threading
import time
import traceback
from typing import Any


def load_yolo_world_with_local_clip(settings: dict[str, Any]) -> Any:
    """Bind the text encoder to the configured asset, without a cache download.

    YOLO-World's default set_classes() creates CLIP by its public model name.
    That ignores our configured checkpoint and can download on an offline node.
    Both callers validate the asset hashes before accepting their evidence.
    """
    model_path = Path(str(settings.get("model_path") or "")).resolve()
    clip_path = Path(str(settings.get("clip_model_path") or "")).resolve()
    for path in (model_path, clip_path):
        if not path.is_file():
            raise FileNotFoundError(f"Configured open-vocabulary asset is missing: {path}")

    from ultralytics import YOLOWorld
    from ultralytics.nn.text_model import CLIP

    model = YOLOWorld(str(model_path))
    model.model.clip_model = CLIP(str(clip_path), device="cpu")
    return model


# YOLO-World/CLIP and GroundingDINO cached models have mutable device,
# predictor and prompt state. Protect the whole prepare/predict/interpret
# operation across both coarse recall and key-material entry points.
_OPEN_VOCABULARY_LOCK = threading.RLock()


def yolo_world_prediction_device(model) -> str:
    """Report the predictor that actually ran, not the requested device option."""
    predictor = getattr(model, "predictor", None)
    device = getattr(predictor, "device", None)
    if device is None:
        device = getattr(model, "device", None)
    return str(device) if device is not None else "unknown"


def park_open_vocabulary_model(model):
    """Discard the predictor and move this model to CPU under the shared lock."""
    import gc
    import torch

    with _OPEN_VOCABULARY_LOCK:
        model.to("cpu")
        if hasattr(model, "predictor"):
            model.predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_with_cuda_oom_cpu_fallback(infer_once, *, device, enabled, cleanup):
    """Retry a typed CUDA OOM once on CPU; preserve input/model policy in caller.

    The device change is execution recovery, not proof of CPU/GPU video-quality
    equivalence. The ordinary inference and physical-action gates still apply.
    """
    import torch
    from .runtime_control import ExecutionCancelled, check_cancelled

    requested = str(device)
    cuda = requested.isdigit() or requested.startswith("cuda")
    with _OPEN_VOCABULARY_LOCK:
        check_cancelled()
        try:
            boxes, receipt = infer_once(device)
        except torch.cuda.OutOfMemoryError as exc:
            if not enabled or not cuda:
                raise
            # Drop CUDA tensors held by the failed call before CPU restoration.
            traceback.clear_frames(exc.__traceback__)
        else:
            return boxes, {**receipt, "requested_device": requested,
                           "actual_device": str(receipt.get("actual_device") or receipt.get("device") or "unknown"),
                           "precision": "float32"}
        check_cancelled()
        try:
            cleanup()
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            check_cancelled()
            boxes, receipt = infer_once("cpu")
            actual_device = str(receipt.get("actual_device") or receipt.get("device") or "unknown")
            if actual_device != "cpu":
                raise RuntimeError(f"Open-vocabulary CPU recovery did not confirm CPU execution: {actual_device}")
        except ExecutionCancelled:
            raise
        except Exception as exc:
            # A frame-level retry must not send this failed CPU attempt back
            # to CUDA and repeat the same recovery loop.
            exc.open_vocabulary_recovery_attempted = True
            exc.add_note("Open-vocabulary CUDA OOM recovery on CPU also failed")
            raise
        return boxes, {**receipt, "requested_device": requested,
                       "actual_device": "cpu", "precision": "float32",
                       "device_fallback": "cuda_out_of_memory_to_cpu"}


def serialized_open_vocabulary(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        waiting = time.perf_counter()
        with _OPEN_VOCABULARY_LOCK:
            started = time.perf_counter()
            result = function(*args, **kwargs)
            elapsed = time.perf_counter() - started
        # The callers return (detections, receipt). Keep the decorator useful
        # for other return types and do not mutate a cached receipt in place.
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            result = (result[0], {**result[1],
                "serialization_wait_seconds": round(started - waiting, 6),
                "serialized_service_seconds": round(elapsed, 6),
                "serialized_timing_scope": "current_call_wall_time_nested_calls_overlap_not_gpu_only",
            })
        return result
    return guarded
