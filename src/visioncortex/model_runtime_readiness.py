"""Lightweight local analysis preflight without importing or starting models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
from typing import Any


def local_preflight(settings: dict[str, Any]) -> tuple[bool, str | None]:
    if settings.get("project", {}).get("site_configuration_required"):
        return False, "目标硬件模板尚未配置私有站点"
    if settings.get("project", {}).get("run_purpose") == "demo":
        return False, "当前为本地演示模式，尚未配置真实分析环境"
    models = settings.get("models") or {}
    for role in ("first_person", "third_person"):
        configured = models.get(role)
        if not configured or not Path(str(configured)).is_file():
            return False, "分析模型权重尚未配置或不可用"
    for module in ("torch", "ultralytics"):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            available = False
        if not available:
            return False, "真实模型运行依赖尚未安装"
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return False, "视频分析需要 FFmpeg 与 ffprobe"
    if str(settings.get("performance", {}).get("tensor_rt", "")).lower() in {"required", "true"}:
        for role in ("first_person_engine", "third_person_engine"):
            configured = models.get(role)
            if not configured or not Path(str(configured)).is_file():
                return False, "目标节点 TensorRT 引擎尚未配置或不可用"
        try:
            available = importlib.util.find_spec("tensorrt") is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            available = False
        if not available:
            return False, "TensorRT 运行依赖尚未安装"
    return True, None
