from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .schemas import RunManifest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_profile(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    visited = set(seen or set())
    if resolved in visited:
        chain = " -> ".join(str(item) for item in [*visited, resolved])
        raise ValueError(f"Configuration inheritance cycle detected: {chain}")
    visited.add(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a mapping: {resolved}")
    extends = payload.pop("extends", None)
    if extends is None:
        return payload
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError(f"Configuration extends must be a non-empty relative path: {resolved}")
    candidate = Path(extends)
    if candidate.is_absolute():
        raise ValueError(f"Configuration extends must be relative: {resolved}")
    base_path = (resolved.parent / candidate).resolve()
    if base_path.parent != resolved.parent:
        raise ValueError(f"Configuration extends must stay in {resolved.parent}: {base_path}")
    return _deep_merge(_load_profile(base_path, visited), payload)


def load_config(path: Path | None = None) -> dict[str, Any]:
    config = _load_profile(DEFAULT_CONFIG)
    if path and path.resolve() != DEFAULT_CONFIG.resolve():
        config = _deep_merge(config, _load_profile(path))
    _apply_environment_overrides(config)
    _validate_mllm_evidence_config(config)
    return config


def _validate_mllm_evidence_config(config: dict[str, Any]) -> None:
    """Reject image budgets that would silently discard required evidence."""

    mllm = config.get("mllm") or {}

    def positive_integer(key: str, default: int) -> int:
        value = int(mllm.get(key, default))
        if value <= 0:
            raise ValueError(f"mllm.{key} must be a positive integer")
        return value

    max_images = positive_integer("max_images_per_event", 8)
    temporal_samples = positive_integer("temporal_samples_per_view", 3)
    if max_images < temporal_samples * 2:
        raise ValueError(
            "mllm.max_images_per_event must preserve both temporal views: "
            f"expected at least {temporal_samples * 2}, got {max_images}"
        )

    liquid_primary = positive_integer("liquid_primary_samples", 9)
    liquid_context = positive_integer("liquid_context_samples", 3)
    action_limits = mllm.get("max_images_per_event_by_action") or {}
    liquid_limit = int(action_limits.get("liquid_movement", max_images))
    required_liquid_images = liquid_primary + liquid_context
    if liquid_limit < required_liquid_images:
        raise ValueError(
            "mllm.max_images_per_event_by_action.liquid_movement must preserve "
            f"{liquid_primary} primary and {liquid_context} context images; "
            f"expected at least {required_liquid_images}, got {liquid_limit}"
        )

    base_storyboard_pairs = positive_integer("storyboard_pairs_per_group", 4)
    maximum_storyboard_pairs = positive_integer(
        "storyboard_pairs_per_group_max", base_storyboard_pairs
    )
    if maximum_storyboard_pairs < base_storyboard_pairs:
        raise ValueError(
            "mllm.storyboard_pairs_per_group_max must be greater than or equal "
            "to mllm.storyboard_pairs_per_group"
        )
    max_group_images = positive_integer(
        "max_images_per_group", maximum_storyboard_pairs
    )
    if max_group_images < maximum_storyboard_pairs:
        raise ValueError(
            "mllm.max_images_per_group must preserve every adaptive aligned "
            f"storyboard pair; expected at least {maximum_storyboard_pairs}, "
            f"got {max_group_images}"
        )


def _apply_environment_overrides(config: dict[str, Any]) -> None:
    """Apply deployment-only settings without baking a machine into the code.

    This deliberately excludes ARK_API_KEY: the MLLM client reads that secret
    directly from the environment and it must never be copied into config dumps.
    """

    perf = config.setdefault("performance", {})
    storage = config.setdefault("storage", {})

    def integer(name: str, key: str) -> None:
        value = os.getenv(name)
        if value is not None:
            perf[key] = int(value)

    def decode_lanes(name: str, key: str) -> None:
        lanes = os.getenv(name)
        if not lanes:
            return
        parsed = [item.strip().lower() for item in lanes.split(",") if item.strip()]
        invalid = [item for item in parsed if item not in {"cuda", "cpu"}]
        if invalid:
            raise ValueError(f"Invalid decode lane(s): {invalid}; expected cuda or cpu")
        perf[key] = parsed

    decode_lanes("VISIONCORTEX_COARSE_DECODE_LANES", "coarse_decode_lanes")
    decode_lanes("VISIONCORTEX_FINE_DECODE_LANES", "fine_decode_lanes")

    integer("VISIONCORTEX_SOURCE_WORKERS", "source_workers")
    integer("VISIONCORTEX_DECODE_QUEUE_DEPTH", "decode_queue_depth")
    integer("VISIONCORTEX_CPU_DECODE_THREADS", "cpu_decode_threads")
    integer("VISIONCORTEX_YOLO_INFERENCE_WORKERS", "yolo_inference_workers")
    integer("VISIONCORTEX_MATERIALIZATION_WORKERS", "materialization_workers")
    integer("VISIONCORTEX_IO_WORKERS", "io_workers")

    tensor_rt = os.getenv("VISIONCORTEX_TENSORRT")
    if tensor_rt:
        perf["tensor_rt"] = tensor_rt.strip().lower()

    path_overrides = {
        "VISIONCORTEX_NAS_INDEX_CSV": "index_csv",
        "VISIONCORTEX_DEVICE_REGISTRY": "device_registry_path",
        "VISIONCORTEX_NAS_ARCHIVE_ROOT": "archive_root",
        "VISIONCORTEX_LOCAL_INPUT_ROOT": "local_input_root",
        "VISIONCORTEX_LOCAL_RUNTIME_ROOT": "local_runtime_root",
        "VISIONCORTEX_LOCAL_CACHE_ROOT": "local_cache_root",
        "VISIONCORTEX_LOCAL_STAGING_ROOT": "local_staging_root",
    }
    for name, key in path_overrides.items():
        value = os.getenv(name)
        if value:
            storage[key] = value

    output_root = os.getenv("VISIONCORTEX_OUTPUT_ROOT")
    if output_root:
        config.setdefault("project", {})["output_root"] = output_root

    models = config.setdefault("models", {})
    model_overrides = {
        "VISIONCORTEX_FIRST_PERSON_MODEL": "first_person",
        "VISIONCORTEX_THIRD_PERSON_MODEL": "third_person",
        "VISIONCORTEX_FIRST_PERSON_ENGINE": "first_person_engine",
        "VISIONCORTEX_THIRD_PERSON_ENGINE": "third_person_engine",
    }
    for name, key in model_overrides.items():
        value = os.getenv(name)
        if value:
            models[key] = value


def load_manifest(path: Path) -> RunManifest:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    manifest = RunManifest.model_validate(raw)
    base = path.parent.resolve()
    for view in manifest.views:
        if view.video is not None and not view.video.is_absolute():
            view.video = (base / view.video).resolve()
        if view.timestamps_csv and not view.timestamps_csv.is_absolute():
            view.timestamps_csv = (base / view.timestamps_csv).resolve()
        for segment in view.segments:
            if not segment.video.is_absolute():
                segment.video = (base / segment.video).resolve()
            if segment.timestamps_csv and not segment.timestamps_csv.is_absolute():
                segment.timestamps_csv = (base / segment.timestamps_csv).resolve()
    return manifest
