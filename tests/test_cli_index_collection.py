import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from labvision_evidence import cli
from labvision_evidence.archive import ArchiveLayout


class _Manifest:
    def __init__(self, experiment_id="source"):
        self.experiment_id = experiment_id
        self.views = [object(), object()]

    def model_dump(self, mode="python"):
        return {"experiment_id": self.experiment_id, "views": []}


def test_short_indexed_timeline_automatically_uses_all_view_fine_scan():
    settings = {
        "performance": {
            "auto_exhaustive_short_timeline_enabled": True,
            "auto_exhaustive_short_timeline_seconds": 300,
            "detection_fps": 10.0,
            "fine_progressive_cross_view": True,
        }
    }

    receipt = cli._configure_index_full_timeline_scan(
        settings,
        {"recording_hours": 27.688 / 3600.0},
        requested_negative_audit=False,
    )

    assert receipt["automatic_short_timeline_scan"] is True
    assert receipt["effective_exhaustive_full_timeline_scan"] is True
    assert receipt["all_eligible_views"] is True
    assert settings["performance"]["exhaustive_full_timeline_scan"] is True
    assert settings["performance"]["fine_progressive_cross_view"] is False
    assert "exhaustive_negative_audit_full_timeline" not in settings["performance"]


def test_long_indexed_timeline_preserves_progressive_policy():
    settings = {
        "performance": {
            "auto_exhaustive_short_timeline_enabled": True,
            "auto_exhaustive_short_timeline_seconds": 300,
            "detection_fps": 10.0,
            "fine_progressive_cross_view": True,
        }
    }

    receipt = cli._configure_index_full_timeline_scan(
        settings,
        {"recording_hours": 3.3},
        requested_negative_audit=False,
    )

    assert receipt["automatic_short_timeline_scan"] is False
    assert receipt["effective_exhaustive_full_timeline_scan"] is False
    assert settings["performance"]["fine_progressive_cross_view"] is True


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
    observed = {}
    fixed_root = tmp_path / "archive" / "Collection-01"
    fixed_root.mkdir(parents=True)
    (fixed_root / "existing.txt").write_text("accepted-old-package", encoding="utf-8")

    monkeypatch.setattr(cli, "load_config", lambda *_: settings)
    monkeypatch.setattr(
        cli,
        "initialize_nas_archive",
        lambda config, name: Path(config["storage"]["active_archive_path"]),
    )

    def prepare(*_):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        return _Manifest("exp-source"), manifest_path, {
            "input_mode": "nas_segmented_virtual_timeline",
            "copied_source_bytes": 0,
        }

    monkeypatch.setattr(cli, "prepare_from_nas_index", prepare)

    class Pipeline:
        def __init__(self, config, progress):
            self.config = config

        def run(self, manifest):
            observed["experiment_id"] = manifest.experiment_id
            staging = Path(self.config["storage"]["active_archive_path"])
            json_root = staging / "JSON-Config-Files"
            json_root.mkdir(parents=True, exist_ok=True)
            (json_root / "run_metrics.json").write_text(
                json.dumps(
                    {
                        "mllm_calls": [
                            {
                                "stage": "experiment_group_understanding",
                                "group_id": "GROUP-1",
                                "model": "doubao-test",
                                "status": "completed",
                                "cache_reused": False,
                                "usage": {
                                    "input_tokens": 100,
                                    "output_tokens": 20,
                                    "total_tokens": 120,
                                    "cached_input_tokens": 0,
                                    "server_reported": True,
                                },
                            }
                        ],
                        "tokens": {
                            "experiment_groups": {
                                "call_count": 1,
                                "executed_call_count": 1,
                                "reused_call_count": 0,
                            },
                            "key_materials": {
                                "call_count": 0,
                                "executed_call_count": 0,
                                "reused_call_count": 0,
                            },
                            "run_total": {
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "total_tokens": 120,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            return staging

    monkeypatch.setattr(cli, "EvidencePipeline", Pipeline)
    repaired = []
    monkeypatch.setattr(
        cli,
        "repair_key_material_presentation_command",
        lambda archive, config: repaired.append((archive, config)),
    )
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
    assert len(repaired) == 1
    assert repaired[0][0] == promoted[0][0]
    assert promoted[0][1].name == "Collection-01"
    assert observed["experiment_id"] == "exp-source"
    assert (fixed_root / "existing.txt").read_text(encoding="utf-8") == "accepted-old-package"


def test_cold_doubao_promotion_audit_rejects_semantic_cache_reuse(tmp_path):
    json_root = tmp_path / "archive" / "JSON-Config-Files"
    json_root.mkdir(parents=True)
    (json_root / "run_metrics.json").write_text(
        json.dumps(
            {
                "mllm_calls": [
                    {
                        "stage": "experiment_group_understanding",
                        "model": "doubao-test",
                        "status": "completed",
                        "cache_reused": True,
                        "usage": {
                            "total_tokens": 120,
                            "server_reported": True,
                        },
                    }
                ],
                "tokens": {
                    "experiment_groups": {
                        "call_count": 1,
                        "executed_call_count": 0,
                        "reused_call_count": 1,
                    },
                    "key_materials": {
                        "call_count": 0,
                        "executed_call_count": 0,
                        "reused_call_count": 0,
                    },
                    "run_total": {"total_tokens": 120},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unreused Doubao calls"):
        cli._audit_true_cold_doubao_execution(tmp_path / "archive")


def test_presentation_repair_relocates_unreferenced_event_directories(tmp_path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache"
    referenced_frame = (
        layout.key_frames / "001-group" / "01-Action" / "Event-current"
    )
    stale_frame = layout.key_frames / "001-group" / "01-Action" / "Event-stale"
    referenced_clip = (
        layout.key_clips / "001-group" / "01-Action" / "Event-current"
    )
    stale_clip = layout.key_clips / "001-group" / "01-Action" / "Event-stale"
    for directory in (
        referenced_frame,
        stale_frame,
        referenced_clip,
        stale_clip,
    ):
        directory.mkdir(parents=True)
        (directory / "artifact.bin").write_bytes(b"test")
    event = SimpleNamespace(
        key_frames={
            "aligned": str(
                (referenced_frame / "artifact.bin").relative_to(layout.root)
            )
        },
        key_clips={
            "aligned": str(
                (referenced_clip / "artifact.bin").relative_to(layout.root)
            )
        },
    )

    relocated = cli._relocate_unreferenced_key_material_event_directories(
        layout, [event]
    )

    assert referenced_frame.is_dir()
    assert referenced_clip.is_dir()
    assert not stale_frame.exists()
    assert not stale_clip.exists()
    assert len(relocated) == 2
    assert all(Path(item["to"]).is_dir() for item in relocated)


def test_index_collection_cli_preserves_existing_archive_when_pipeline_fails(
    monkeypatch, tmp_path
):
    settings = {
        "project": {},
        "storage": {
            "archive_root": str(tmp_path / "archive"),
            "local_runtime_root": str(tmp_path / "runtime"),
        },
    }
    fixed_root = tmp_path / "archive" / "Collection-01"
    fixed_root.mkdir(parents=True)
    marker = fixed_root / "accepted.txt"
    marker.write_text("immutable-old-package", encoding="utf-8")
    manifest_path = tmp_path / "input" / "manifest.yaml"
    records = []

    monkeypatch.setattr(cli, "load_config", lambda *_: settings)
    monkeypatch.setattr(
        cli,
        "initialize_nas_archive",
        lambda config, name: Path(config["storage"]["active_archive_path"]),
    )

    def prepare(*_):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        return _Manifest("exp-source"), manifest_path, {
            "input_mode": "nas_segmented_virtual_timeline",
            "copied_source_bytes": 0,
        }

    monkeypatch.setattr(cli, "prepare_from_nas_index", prepare)

    class FailingPipeline:
        def __init__(self, config, progress):
            self.config = config

        def run(self, manifest):
            staging = Path(self.config["storage"]["active_archive_path"])
            staging.mkdir(parents=True, exist_ok=True)
            raise RuntimeError("quality gate failed")

    monkeypatch.setattr(cli, "EvidencePipeline", FailingPipeline)
    monkeypatch.setattr(
        cli,
        "promote_fixed_archive",
        lambda *_: pytest.fail("promotion must not run after pipeline failure"),
    )
    monkeypatch.setattr(
        cli,
        "record_collection_state",
        lambda *args, **kwargs: records.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="quality gate failed"):
        cli.run_index_collection_command(
            experiment_id="exp-source",
            archive_name="Collection 01",
            config=tmp_path / "config.yaml",
        )

    assert [record["state"] for record in records] == [
        "queued",
        "processing",
        "failed",
    ]
    assert marker.read_text(encoding="utf-8") == "immutable-old-package"
    assert records[-1]["details"]["formal_archive_preserved"] is True


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
