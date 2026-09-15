from __future__ import annotations

from functools import wraps
from pathlib import Path
import threading
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


def serialized_open_vocabulary(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with _OPEN_VOCABULARY_LOCK:
            return function(*args, **kwargs)
    return guarded
