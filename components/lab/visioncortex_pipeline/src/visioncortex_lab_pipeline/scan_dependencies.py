"""Share phase selection and bind scan caches to the detector actually used."""
from copy import deepcopy
from pathlib import Path

from .schemas import ViewRole
from .detection_thresholds import prediction_confidence, role_thresholds


def phase_config(config, phase):
    """Apply the same coarse overrides for execution and dependency identity."""
    if phase != "coarse":
        return config
    effective = deepcopy(config)
    models = effective.get("models", {})
    if "confidence_by_role" in models or "coarse_confidence_by_role" in models:
        # Fine-model calibration must not silently alter the retained coarse model.
        models["confidence_by_role"] = role_thresholds(models.get("coarse_confidence_by_role", {}))
    for role in ViewRole:
        engine = effective.get("models", {}).get(f"{role.value}_coarse_engine")
        if engine:
            effective["models"][f"{role.value}_engine"] = engine
    performance = effective["performance"]
    if "coarse_inference_batch_wait_ms" in performance:
        performance["inference_batch_wait_ms"] = performance["coarse_inference_batch_wait_ms"]
    return effective


def scan_model_dependencies(config, role, phase):
    """Keep all unknown settings, excluding only known unused detector paths.

    Model selection is the scanner's implementation, including auto fallback.
    An unused PyTorch checkpoint cannot invalidate an engine-backed scan, but
    losing that engine in auto mode selects the checkpoint and changes identity.
    Global model validation still happens before the device-day scan lookup.
    """
    from .detection import _select_model_path

    role = ViewRole(role)
    if phase not in {"coarse", "fine"}:
        raise ValueError("Device scan cache requires coarse or fine phase")
    effective = phase_config(config, phase)
    selected = _select_model_path(role, effective)
    if not selected.is_file():
        raise FileNotFoundError(f"Selected scan model is unavailable: {selected}")
    detector_paths = {r.value + suffix for r in ViewRole
                      for suffix in ("", "_engine", "_coarse_engine")}
    models = {key: deepcopy(value) for key, value in effective["models"].items()
              if key not in detector_paths | {"confidence_by_role", "coarse_confidence_by_role"}}
    models["confidence"] = prediction_confidence(effective, role)
    files = {key: Path(value) for key, value in models.items()
             if isinstance(value, str) and Path(value).is_file()}
    # A separate field avoids colliding with future producer model settings.
    detector = {"role": role.value, "path": str(selected),
                "backend": "TensorRT" if selected.suffix.lower() == ".engine" else "PyTorch"}
    return effective, models, detector, files
