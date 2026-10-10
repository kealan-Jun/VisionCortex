"""Explicit, deployment-owned registration of the optional six-view benchmark."""

from __future__ import annotations

from typing import Any

from .storage import safe_archive_name
from .config import require_configured_site

SUBMISSION_PROTOCOL_VERSION = 1


def require_source_count(benchmark: dict[str, Any], actual_count: int) -> None:
    expected = benchmark["source_count"]
    if actual_count != expected:
        raise ValueError(
            f"The registered benchmark requires {expected} sources; prepared {actual_count}."
        )


def registration(settings: dict[str, Any], *, required: bool = False) -> dict[str, Any]:
    configured = settings.get("fixed_benchmark") or {}
    result = {
        "enabled": False,
        "experiment_id": None,
        "archive_name": None,
        "source_count": 6,
        "submission_protocol_version": SUBMISSION_PROTOCOL_VERSION,
    }
    if not isinstance(configured, dict) or configured.get("enabled") is not True:
        if required:
            raise ValueError("The fixed benchmark requires explicit site registration.")
        return result
    try:
        require_configured_site(settings)
        if settings.get("project", {}).get("portable_desktop"):
            raise ValueError("The fixed benchmark is unavailable in the desktop application.")
        if settings.get("runtime", {}).get("local_only") or settings.get("project", {}).get("run_purpose") == "demo":
            raise ValueError("The fixed benchmark is unavailable in local demo mode.")
        experiment_id = configured.get("experiment_id")
        archive_name = configured.get("archive_name")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("The fixed benchmark requires an experiment identity.")
        if not isinstance(archive_name, str) or not archive_name.strip():
            raise ValueError("The fixed benchmark requires an archive identity.")
        if experiment_id != experiment_id.strip() or archive_name != archive_name.strip():
            raise ValueError("Benchmark identities must not contain surrounding whitespace.")
        if configured.get("source_count", 6) != 6:
            raise ValueError("The registered six-view benchmark must have six sources.")
        if safe_archive_name(archive_name) != archive_name:
            raise ValueError("The fixed benchmark archive identity must be a safe directory name.")
    except ValueError:
        if required:
            raise
        return result
    return result | {"enabled": True, "experiment_id": experiment_id, "archive_name": archive_name}
