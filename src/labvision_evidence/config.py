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


def load_config(path: Path | None = None) -> dict[str, Any]:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if path and path.resolve() != DEFAULT_CONFIG.resolve():
        with path.open("r", encoding="utf-8") as handle:
            config = _deep_merge(config, yaml.safe_load(handle) or {})
    _apply_environment_overrides(config)
    return config


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

    lanes = os.getenv("VISIONCORTEX_COARSE_DECODE_LANES")
    if lanes:
        parsed = [item.strip().lower() for item in lanes.split(",") if item.strip()]
        invalid = [item for item in parsed if item not in {"cuda", "cpu"}]
        if invalid:
            raise ValueError(f"Invalid decode lane(s): {invalid}; expected cuda or cpu")
        perf["coarse_decode_lanes"] = parsed

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
    }
    for name, key in path_overrides.items():
        value = os.getenv(name)
        if value:
            storage[key] = value


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
