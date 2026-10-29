"""Explicit export branch validation for checkpoints trained on one YOLO head."""

from collections.abc import Mapping
from pathlib import Path


def export_branch_options(settings: dict, role: str) -> dict:
    branches = settings.get("models", {}).get("engine_branch_by_role", {})
    if not isinstance(branches, Mapping):
        raise ValueError("models.engine_branch_by_role must be a role mapping")
    for name, branch in branches.items():
        if name not in ("first_person", "third_person"):
            raise ValueError(f"Unknown engine branch role: {name}")
        if branch not in ("one2one", "one2many"):
            raise ValueError("Engine branch must be one2one or one2many")
    branch = branches.get(role)
    return {} if branch is None else {"end2end": branch == "one2one"}


def verify_export_branch(path: Path, options: dict) -> dict:
    from .detection import _tensorrt_plan_and_metadata

    _, metadata, _ = _tensorrt_plan_and_metadata(path)
    if "end2end" in options:
        observed = metadata.get("end2end")
        if type(observed) is not bool or observed != options["end2end"]:
            raise ValueError("TensorRT engine does not match the requested trained branch")
    return metadata
