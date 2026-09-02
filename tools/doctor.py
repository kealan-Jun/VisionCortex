from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SCHEMA_VERSION = "visioncortex-environment-doctor/1.0.0"
SUPPORTED_PYTHON = ((3, 11), (3, 13))
CORE_MODULES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "typer": "typer",
    "yaml": "PyYAML",
    "pydantic": "pydantic",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "visioncortex": "visioncortex",
}
OPTIONAL_GPU_MODULES = {
    "torch": "torch",
    "tensorrt": "tensorrt",
    "ultralytics": "ultralytics",
    "transformers": "transformers",
}


def _command_output(command: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_nvidia_smi(output: str | None) -> list[dict[str, object]]:
    gpus: list[dict[str, object]] = []
    if not output:
        return gpus
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory_mb = int(parts[1])
        except ValueError:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_mb": memory_mb,
                "driver_version": parts[2],
            }
        )
    return gpus


def gpu_memory_profile(memory_mb: int) -> str:
    if memory_mb >= 20 * 1024:
        return "gpu-24gb"
    if memory_mb >= 10 * 1024:
        return "gpu-12gb"
    if memory_mb >= 7 * 1024:
        return "gpu-8gb"
    if memory_mb >= 5 * 1024:
        return "gpu-6gb"
    return "gpu-under-6gb"


def _check(
    name: str,
    status: str,
    message: str,
    *,
    required_for_web: bool = False,
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "required_for_web": required_for_web,
    }


def inspect_environment(project_root: Path, *, check_storage: bool = False) -> dict[str, object]:
    root = project_root.resolve()
    checks: list[dict[str, object]] = []
    remediation: list[str] = []
    system_name = platform.system() or "Unknown"
    release = platform.release()
    is_wsl = system_name == "Linux" and "microsoft" in release.lower()

    python_version = sys.version_info[:3]
    python_supported = SUPPORTED_PYTHON[0] <= python_version[:2] < SUPPORTED_PYTHON[1]
    checks.append(
        _check(
            "Python",
            "passed" if python_supported else "failed",
            f"{platform.python_version()} ({sys.executable})"
            if python_supported
            else f"检测到 {platform.python_version()}，项目需要 Python 3.11 或 3.12。",
            required_for_web=True,
        )
    )
    if not python_supported:
        remediation.append("安装 Python 3.11 或 3.12，然后重新创建 .venv。")

    required_files = (root / "pyproject.toml", root / "configs" / "development-local.yaml")
    missing_files = [str(path) for path in required_files if not path.is_file()]
    checks.append(
        _check(
            "项目文件",
            "passed" if not missing_files else "failed",
            f"项目目录：{root}" if not missing_files else f"缺少文件：{', '.join(missing_files)}",
            required_for_web=True,
        )
    )
    if missing_files:
        remediation.append("请在完整的 VisionCortex Git 仓库中运行启动脚本。")

    missing_modules: list[str] = []
    core_versions: dict[str, str] = {}
    for module, distribution in CORE_MODULES.items():
        if importlib.util.find_spec(module) is None:
            missing_modules.append(distribution)
            continue
        core_versions[distribution] = _installed_version(distribution) or "available"
    checks.append(
        _check(
            "Web运行依赖",
            "passed" if not missing_modules else "failed",
            "已安装" if not missing_modules else f"缺少：{', '.join(missing_modules)}",
            required_for_web=True,
        )
    )
    if missing_modules:
        remediation.append(
            f'使用当前 Python 安装基础环境："{sys.executable}" -m pip install -e "{root}"'
        )

    git_sha = _command_output(("git", "rev-parse", "HEAD"), cwd=root)
    checks.append(
        _check(
            "Git版本",
            "passed" if git_sha else "warning",
            git_sha or "无法读取 Git SHA；仍可开发，但运行凭证不完整。",
        )
    )

    ffmpeg_path = shutil.which("ffmpeg")
    checks.append(
        _check(
            "FFmpeg",
            "passed" if ffmpeg_path else "warning",
            ffmpeg_path or "未发现 FFmpeg；网页可启动，但视频任务不能正式运行。",
        )
    )

    nvidia_output = _command_output(
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    gpus = parse_nvidia_smi(nvidia_output)
    if system_name == "Darwin":
        recommended_profile = "mac-development"
        gpu_message = "Mac使用本地开发模式；正式TensorRT任务交给NVIDIA GPU机器。"
    elif gpus:
        largest_gpu = max(gpus, key=lambda item: int(item["memory_mb"]))
        recommended_profile = gpu_memory_profile(int(largest_gpu["memory_mb"]))
        gpu_message = "; ".join(
            f'{gpu["name"]} / {int(gpu["memory_mb"])} MiB / 驱动 {gpu["driver_version"]}'
            for gpu in gpus
        )
    else:
        recommended_profile = "cpu-development"
        gpu_message = "未发现NVIDIA GPU；网页和开发测试可用，正式模型任务不可用。"
    checks.append(_check("GPU", "passed" if gpus else "warning", gpu_message))

    optional_versions: dict[str, str] = {}
    missing_optional: list[str] = []
    for module, distribution in OPTIONAL_GPU_MODULES.items():
        if importlib.util.find_spec(module) is None:
            missing_optional.append(distribution)
            continue
        optional_versions[distribution] = _installed_version(distribution) or "available"
    checks.append(
        _check(
            "模型运行依赖",
            "passed" if not missing_optional else "warning",
            "已安装" if not missing_optional else f"未完整安装：{', '.join(missing_optional)}",
        )
    )

    storage: dict[str, object] = {"checked": check_storage, "paths": {}}
    storage_variables = (
        "VISIONCORTEX_NAS_INDEX_CSV",
        "VISIONCORTEX_NAS_ARCHIVE_ROOT",
        "VISIONCORTEX_NAS_CACHE_ROOT",
    )
    if check_storage:
        for variable in storage_variables:
            value = os.environ.get(variable)
            storage["paths"][variable] = {
                "configured": bool(value),
                "accessible": bool(value and Path(value).exists()),
            }
        checks.append(
            _check("NAS存储", "passed", "已检查已配置的NAS路径；未配置项保持跳过。")
        )
    else:
        checks.append(
            _check("NAS存储", "skipped", "本地启动默认不访问NAS；需要时使用 --check-storage。")
        )

    failed_required = [
        item for item in checks if item["required_for_web"] and item["status"] == "failed"
    ]
    launch_ready = not failed_required
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if launch_ready else "blocked",
        "launch_ready": launch_ready,
        "project_root": str(root),
        "system": {
            "name": system_name,
            "release": release,
            "machine": platform.machine(),
            "is_wsl": is_wsl,
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": python_supported,
        },
        "git_sha": git_sha,
        "gpus": gpus,
        "recommended_profile": recommended_profile,
        "core_versions": core_versions,
        "optional_versions": optional_versions,
        "storage": storage,
        "checks": checks,
        "remediation": remediation,
        "evidence_boundary": {
            "web_start": "PROVEN" if launch_ready else "NOT_PROVEN",
            "model_inference": "NOT_PROVEN",
            "real_video": "NOT_PROVEN",
        },
    }


def render_human(report: dict[str, object]) -> str:
    labels = {
        "passed": "通过",
        "warning": "提醒",
        "failed": "失败",
        "skipped": "跳过",
    }
    lines = ["VisionCortex 环境检查", "=" * 28]
    for item in report["checks"]:
        lines.append(f'[{labels[str(item["status"]) ]}] {item["name"]}：{item["message"]}')
    lines.extend(
        (
            "",
            f'建议机器配置：{report["recommended_profile"]}',
            f'可以启动本地网页：{"是" if report["launch_ready"] else "否"}',
            "正式模型推理：尚未验证",
        )
    )
    remediation = report["remediation"]
    if remediation:
        lines.extend(("", "需要处理："))
        lines.extend(f"  {index}. {message}" for index, message in enumerate(remediation, 1))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="检查VisionCortex本机环境，不安装或修改软件。")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="VisionCortex仓库根目录。",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读JSON。")
    parser.add_argument(
        "--check-storage",
        action="store_true",
        help="检查已配置NAS路径；默认不访问NAS。",
    )
    args = parser.parse_args(argv)
    report = inspect_environment(args.project_root, check_storage=args.check_storage)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_human(report))
    return 0 if report["launch_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
