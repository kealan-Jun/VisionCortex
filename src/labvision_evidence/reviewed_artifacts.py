from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset_scoped_json(
    configured: str | Path | None,
    current_experiment_id: str | None,
    *,
    repository_root: Path,
    artifact_label: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load a reviewed artifact only for an explicitly declared dataset."""

    if not configured:
        return None, {
            "configured": False,
            "applied": False,
            "reason": "not_configured",
            "current_experiment_id": current_experiment_id,
        }
    path = Path(configured)
    if not path.is_absolute():
        path = repository_root / path
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_label}不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_label}必须是 JSON 对象: {path}")

    applicability = payload.get("applicability") or {}
    allowed_ids = {
        str(item)
        for item in (
            applicability.get("experiment_ids")
            or ([payload["experiment_id"]] if payload.get("experiment_id") else [])
        )
    }
    common = {
        "configured": True,
        "current_experiment_id": current_experiment_id,
        "allowed_experiment_ids": sorted(allowed_ids),
        "artifact_path": str(path),
        "artifact_id": payload.get("baseline_id")
        or payload.get("ground_truth_id"),
    }
    if not allowed_ids:
        return None, {
            **common,
            "applied": False,
            "reason": "artifact_missing_dataset_applicability",
        }
    if current_experiment_id not in allowed_ids:
        return None, {
            **common,
            "applied": False,
            "reason": "experiment_id_not_applicable",
        }
    return payload, {
        **common,
        "applied": True,
        "reason": "experiment_id_matched",
    }
