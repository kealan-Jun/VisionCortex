from pathlib import Path

import pytest
import typer

from visioncortex.cli import (
    _create_staging_only_run_root,
    _normalized_cache_policy,
)


def _settings(tmp_path: Path) -> dict:
    archive = tmp_path / "archive"
    staging = archive / ".VisionCortex-Run-Staging"
    staging.mkdir(parents=True)
    return {
        "storage": {
            "archive_root": str(archive),
            "local_staging_root": str(staging),
        }
    }


def test_staging_only_root_is_below_existing_archive_staging(tmp_path):
    settings = _settings(tmp_path)
    result = _create_staging_only_run_root(settings, "audit", "run-1")

    expected_staging = (
        Path(settings["storage"]["archive_root"]) / ".VisionCortex-Run-Staging"
    ).resolve()
    assert result == expected_staging / "audit" / "run-1"
    assert result.is_dir()
    assert not (Path(settings["storage"]["archive_root"]) / "audit").exists()


def test_staging_only_root_rejects_replacement_root(tmp_path):
    settings = _settings(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    settings["storage"]["local_staging_root"] = str(replacement)

    with pytest.raises(typer.BadParameter, match="archive-local"):
        _create_staging_only_run_root(settings, "audit", "run-1")


def test_cache_policy_requires_explicit_namespace_for_hot_replay():
    assert _normalized_cache_policy("cold", None, "run-1") == ("cold", "run-1")
    assert _normalized_cache_policy("reuse", "audit-a", "run-2") == (
        "reuse",
        "audit-a",
    )
    with pytest.raises(typer.BadParameter, match="required"):
        _normalized_cache_policy("reuse", None, "run-2")
