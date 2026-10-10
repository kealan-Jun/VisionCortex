from pathlib import Path

import pytest

from visioncortex import cli
from visioncortex.config import load_config
from visioncortex.schemas import RunManifest, ViewInput, ViewRole


def test_fixed_benchmark_preprocessing_only_never_promotes_formal_archive(
    monkeypatch, tmp_path, capsys
):
    fixed_root = tmp_path / "fixed"
    nas_staging = tmp_path / "nas-staging"
    history_root = tmp_path / "history"
    manifest_path = tmp_path / "manifest.yaml"
    manifest = RunManifest(
        experiment_id="fixture-experiment",
        views=[
            ViewInput(view_id=f"view-{index}",
                      role=ViewRole.FIRST_PERSON if index == 0 else ViewRole.THIRD_PERSON,
                      video=Path(f"view-{index}.mp4"))
            for index in range(6)
        ],
    )
    observed = {}
    settings = load_config()
    settings["project"]["run_purpose"] = "analysis"
    settings["runtime"]["local_only"] = False
    settings["fixed_benchmark"] = {"enabled": True, "experiment_id": "fixture-experiment", "archive_name": "Fixture-Benchmark"}
    monkeypatch.setattr(cli, "load_config", lambda *_args: settings)

    monkeypatch.setattr(
        cli,
        "fixed_archive_staging_paths",
        lambda *_args: (fixed_root, nas_staging, history_root),
    )
    monkeypatch.setattr(
        cli,
        "prepare_from_nas_index",
        lambda *_args: (
            manifest,
            manifest_path,
            {
                "input_mode": "segmented_virtual_timeline",
                "copied_source_bytes": 0,
                "manifest": str(manifest_path),
            },
        ),
    )

    class FakePipeline:
        def __init__(self, settings, _progress):
            observed["settings"] = settings

        def run(self, pipeline_manifest):
            observed["experiment_id"] = pipeline_manifest.experiment_id
            return nas_staging

    monkeypatch.setattr(cli, "EvidencePipeline", FakePipeline)

    def forbidden_promotion(*_args):
        raise AssertionError("preprocessing acceptance must not promote")

    monkeypatch.setattr(cli, "promote_fixed_archive", forbidden_promotion)

    cli.run_fixed_benchmark_command(
        Path("configs/rtx4060-laptop-production.yaml"),
        preprocessing_only=True,
    )

    assert observed["settings"]["project"]["preprocessing_acceptance_only"] is True
    assert observed["settings"]["storage"]["active_archive_path"] == str(nas_staging)
    assert observed["experiment_id"] == "fixture-experiment"
    output = capsys.readouterr().out
    assert "run_mode=preprocessing_acceptance_only" in output
    assert "token_calls=0" in output
    assert "fixed_archive_promotion=skipped" in output


@pytest.mark.parametrize("source_count", [2, 5, 7])
def test_fixed_benchmark_rejects_wrong_prepared_source_count(monkeypatch, tmp_path, source_count):
    from types import SimpleNamespace

    settings = load_config()
    settings["project"]["run_purpose"] = "analysis"
    settings["runtime"]["local_only"] = False
    settings["fixed_benchmark"] = {"enabled": True, "experiment_id": "fixture-experiment", "archive_name": "Fixture-Benchmark"}
    monkeypatch.setattr(cli, "load_config", lambda *_: settings)
    monkeypatch.setattr(cli, "fixed_archive_staging_paths",
                        lambda *_: (tmp_path / "fixed", tmp_path / "staging", tmp_path / "history"))
    manifest_path = tmp_path / "manifest.yaml"
    monkeypatch.setattr(cli, "prepare_from_nas_index",
                        lambda *_: (SimpleNamespace(views=[object()] * source_count), manifest_path, {}))
    monkeypatch.setattr(cli, "EvidencePipeline", lambda *_: pytest.fail("wrong source count started models"))
    monkeypatch.setattr(cli, "promote_fixed_archive", lambda *_: pytest.fail("wrong source count was promoted"))
    with pytest.raises(ValueError, match=f"requires 6 sources; prepared {source_count}"):
        cli.run_fixed_benchmark_command(Path("unused-private-site.yaml"), preprocessing_only=True)
    assert not manifest_path.exists()
