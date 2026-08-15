from pathlib import Path

from labvision_evidence import cli
from labvision_evidence.schemas import RunManifest, ViewInput, ViewRole


def test_fixed_benchmark_preprocessing_only_never_promotes_formal_archive(
    monkeypatch, tmp_path, capsys
):
    fixed_root = tmp_path / "fixed"
    nas_staging = tmp_path / "nas-staging"
    history_root = tmp_path / "history"
    manifest_path = tmp_path / "manifest.yaml"
    manifest = RunManifest(
        experiment_id="source",
        views=[
            ViewInput(
                view_id="fp",
                role=ViewRole.FIRST_PERSON,
                video=Path("fp.mp4"),
            ),
            ViewInput(
                view_id="tp",
                role=ViewRole.THIRD_PERSON,
                video=Path("tp.mp4"),
            ),
        ],
    )
    observed = {}

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

        def run(self, _manifest):
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
    output = capsys.readouterr().out
    assert "run_mode=preprocessing_acceptance_only" in output
    assert "token_calls=0" in output
    assert "fixed_archive_promotion=skipped" in output
