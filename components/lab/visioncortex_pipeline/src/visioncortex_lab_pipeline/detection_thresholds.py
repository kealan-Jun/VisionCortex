"""Explicit detector confidence selection without binding camera identities."""
import math
from collections.abc import Mapping

from .schemas import ViewRole


def _confidence(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Detector confidence must be a finite number in (0, 1]")
    return float(value)


def role_thresholds(value):
    if not isinstance(value, Mapping):
        raise ValueError("Detector confidence overrides require a role mapping")
    return {ViewRole(role).value: _confidence(threshold) for role, threshold in value.items()}


def prediction_confidence(config, role):
    models = config["models"]
    overrides = role_thresholds(models.get("confidence_by_role", {}))
    role_thresholds(models.get("coarse_confidence_by_role", {}))
    default = _confidence(models["confidence"])
    return overrides.get(ViewRole(role).value, default)
