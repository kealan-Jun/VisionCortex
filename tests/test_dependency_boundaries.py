from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVY_MODEL_PACKAGES = {"clip", "torch", "torchvision", "transformers", "ultralytics"}


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~@\[]", requirement, maxsplit=1)[0].casefold()


def test_core_and_dev_dependencies_do_not_pull_the_gpu_model_stack():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = {_requirement_name(item) for item in project["dependencies"]}
    dev = {
        _requirement_name(item)
        for item in project["optional-dependencies"]["dev"]
    }
    models = {
        _requirement_name(item)
        for item in project["optional-dependencies"]["models"]
    }

    assert not (core | dev) & HEAVY_MODEL_PACKAGES
    assert dev == {"httpx2", "pytest", "pytest-cov"}
    assert {"clip", "transformers", "ultralytics"} <= models


def test_heavy_model_imports_remain_lazy_in_production_modules():
    violations: list[str] = []
    source_root = ROOT / "src" / "labvision_evidence"
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for statement in tree.body:
            imported: set[str] = set()
            if isinstance(statement, ast.Import):
                imported = {alias.name.split(".", 1)[0] for alias in statement.names}
            elif isinstance(statement, ast.ImportFrom) and statement.module:
                imported = {statement.module.split(".", 1)[0]}
            for package in imported & HEAVY_MODEL_PACKAGES:
                violations.append(f"{path.name}:{statement.lineno}:{package}")

    assert violations == []


def test_production_installers_exclude_development_extras():
    rtx4060_installer = (
        ROOT / "deployment" / "rtx4060" / "01-安装与检查.ps1"
    ).read_text(encoding="utf-8")
    rtx4090_installer = (
        ROOT / "deployment" / "rtx4090" / "01-Install-And-Validate.ps1"
    ).read_text(encoding="utf-8")
    assert ".[models,tensorrt]" in rtx4060_installer
    assert "labvision-evidence[tensorrt]" in rtx4090_installer
    assert "[dev" not in rtx4060_installer
    assert "[dev" not in rtx4090_installer

    for relative in (
        "deployment/rtx3090ti-ubuntu/requirements-lock.txt",
        "deployment/rtx4090/requirements-lock.txt",
    ):
        locked = (ROOT / relative).read_text(encoding="utf-8").casefold()
        assert "pytest" not in locked
