import json
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


def test_register_archive_requires_and_records_passing_report_receipts(
    monkeypatch, tmp_path
):
    archive = tmp_path / "Accepted-Archive"
    json_root = archive / "JSON-Config-Files"
    report_root = archive / "Lab-Daily-Reports" / "2026-08-17"
    pdf_root = archive / "Professional-PDFs"
    for root in (json_root, report_root, pdf_root):
        root.mkdir(parents=True, exist_ok=True)
    for name in ("evidence_package_eval.json", "quality_acceptance.json"):
        (json_root / name).write_text(json.dumps({"passed": True}), encoding="utf-8")
    (report_root / "Daily-Report-Eval.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (pdf_root / "VisionCortex-Professional-Evidence-Report-2026-08-17.pdf").write_bytes(
        b"%PDF-test"
    )
    recorded = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda *_: {"storage": {"archive_root": str(tmp_path / "archive-root")}},
    )
    monkeypatch.setattr(
        cli,
        "record_collection_state",
        lambda *args, **kwargs: recorded.append(kwargs) or tmp_path / "ledger.json",
    )

    cli.register_archived_collection_command(
        experiment_id="exp-accepted", archive=archive, config=None
    )

    assert recorded[0]["state"] == "archived"
    assert recorded[0]["archive_name"] == "Accepted-Archive"
    assert recorded[0]["details"]["registration_only"] is True
