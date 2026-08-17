from pathlib import Path

from labvision_evidence import cli


class _Manifest:
    def __init__(self):
        self.experiment_id = "source"
        self.views = [object(), object()]

    def model_dump(self, mode="python"):
        return {"experiment_id": self.experiment_id, "views": []}


def test_index_collection_cli_promotes_only_after_pipeline_success(monkeypatch, tmp_path):
    settings = {
        "project": {},
        "storage": {
            "archive_root": str(tmp_path / "archive"),
            "local_runtime_root": str(tmp_path / "runtime"),
        },
    }
    manifest_path = tmp_path / "input" / "manifest.yaml"
    records = []
    promoted = []

    monkeypatch.setattr(cli, "load_config", lambda *_: settings)
    monkeypatch.setattr(
        cli,
        "initialize_nas_archive",
        lambda config, name: Path(config["storage"]["active_archive_path"]),
    )

    def prepare(*_):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        return _Manifest(), manifest_path, {
            "input_mode": "nas_segmented_virtual_timeline",
            "copied_source_bytes": 0,
        }

    monkeypatch.setattr(cli, "prepare_from_nas_index", prepare)

    class Pipeline:
        def __init__(self, config, progress):
            self.config = config

        def run(self, manifest):
            staging = Path(self.config["storage"]["active_archive_path"])
            staging.mkdir(parents=True, exist_ok=True)
            return staging

    monkeypatch.setattr(cli, "EvidencePipeline", Pipeline)
    monkeypatch.setattr(
        cli,
        "promote_fixed_archive",
        lambda staging, fixed, history: promoted.append((staging, fixed, history))
        or {"verification": {"status": "verified"}},
    )
    monkeypatch.setattr(
        cli,
        "record_collection_state",
        lambda *args, **kwargs: records.append(kwargs["state"]),
    )

    cli.run_index_collection_command(
        experiment_id="exp-source",
        archive_name="Collection 01",
        config=tmp_path / "config.yaml",
    )

    assert manifest_path.is_file()
    assert records == ["queued", "processing", "archived"]
    assert len(promoted) == 1
    assert promoted[0][1].name == "Collection-01"
