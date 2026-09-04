import hashlib
import json
import time
from pathlib import Path

import pytest

import visioncortex.storage as storage
from visioncortex.archive import (
    ArchiveLayout,
    _group_folder_name,
    _key_material_event_folder_name,
)
from visioncortex.schemas import ActionType, EvidenceEvent
from visioncortex.provenance import write_run_provenance
from visioncortex.schema_contracts import write_archive_contract_manifest
from visioncortex.storage import (
    DERIVED_ARCHIVE_DIRECTORIES,
    promote_fixed_archive,
    verify_current_release,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_release_gate(staging: Path) -> None:
    json_root = staging / "JSON-Config-Files"
    _write_json(json_root / "run_manifest.json", {"experiment_id": "test"})
    _write_json(
        json_root / "evidence_package.json",
        {"schema_version": "2.0.0", "experiment_groups": []},
    )
    _write_json(json_root / "evidence_package_eval.json", {"passed": True})
    _write_json(
        json_root / "quality_acceptance.json",
        {
            "schema_version": "visioncortex-quality-acceptance/1",
            "passed": True,
            "status": "structural_only",
            "evidence_level": "structural_only",
            "formal_accuracy_claim_allowed": False,
            "experiment_boundaries": {},
            "key_materials": {},
        },
    )
    _write_json(json_root / "run_metrics.json", {"stage_durations": [], "tokens": {}})
    _write_json(
        json_root / "audit_layer.json",
        {"events": [], "segments": [], "experiment_groups": []},
    )
    _write_json(json_root / "physical_change_log.json", [])
    _write_json(
        staging / "Key-Materials" / "Key-Materials-Model-Understanding.json", []
    )
    _write_json(
        json_root / "evidence_index_manifest.json",
        {
            "schema_version": "visioncortex-evidence-index/1",
            "counts": {"key_events": 0},
            "files": {},
            "integrity": {},
            "validation": {"passed": True},
        },
    )
    report_dir = staging / "Lab-Daily-Reports" / "2026-08-13"
    report_json = report_dir / "Lab-Daily-Report-2026-08-13.json"
    _write_json(
        report_json,
        {
            "experiment_timeline": [],
            "quality_acceptance": {"passed": True, "status": "structural_only"},
        },
    )
    _write_json(report_dir / "Daily-Report-Eval.json", {"passed": True})
    _write_json(
        json_root / "daily_report_manifest.json",
        {
            "report_date": "2026-08-13",
            "json": report_json.relative_to(staging).as_posix(),
            "html": "Lab-Daily-Reports/2026-08-13/report.html",
            "evaluation": "Lab-Daily-Reports/2026-08-13/Daily-Report-Eval.json",
            "passed": True,
            "checksums": {},
        },
    )
    pdf = (
        staging
        / "Professional-PDFs"
        / "VisionCortex-Professional-Evidence-Report-2026-08-13.pdf"
    )
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-test")
    _write_json(
        json_root / "professional_report_manifest.json",
        {
            "schema_version": "visioncortex-professional-report-manifest/1",
            "template_id": "test",
            "renderer_sha256": "test",
            "status": "generated",
            "pdf": pdf.relative_to(staging).as_posix(),
            "checksum_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        },
    )
    write_run_provenance(
        staging,
        {},
        {"required": False, "status": "not_required_by_profile"},
        repository_root=Path(__file__).resolve().parents[1],
    )
    write_archive_contract_manifest(staging)


def test_fixed_archive_promotion_replaces_derived_outputs_and_retains_previous(tmp_path):
    fixed = tmp_path / "fixed"
    staging = tmp_path / "staging"
    history = tmp_path / "history"
    original = fixed / "Original-Experiment-Videos" / "source.mp4"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original")

    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        old_file = fixed / directory / "version.txt"
        new_file = staging / directory / "version.txt"
        old_file.parent.mkdir(parents=True, exist_ok=True)
        new_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.write_text("old", encoding="utf-8")
        new_file.write_text("new", encoding="utf-8")
    _complete_release_gate(staging)
    staged_original = staging / "Original-Experiment-Videos"
    staged_original.mkdir(parents=True)
    (staged_original / "Original-Video-Index.json").write_text(
        json.dumps({"source_copy_bytes": 0}), encoding="utf-8"
    )
    (staged_original / "view-01.ffconcat").write_text(
        "ffconcat version 1.0\nfile 'Y:/source.mp4'\n", encoding="utf-8"
    )
    (staged_original / "README.txt").write_text("zero-copy", encoding="utf-8")

    receipt = promote_fixed_archive(
        staging,
        fixed,
        history,
        {
            "metric_key": "web_end_to_end",
            "request_received_at": "2026-09-04T12:00:00+08:00",
            "request_started_epoch": time.time() - 0.01,
        },
    )

    assert original.read_bytes() == b"original"
    assert receipt["previous_package_retained"] is True
    assert receipt["original_media_preserved"] is True
    assert len(receipt["promoted_original_references"]) == 3
    assert receipt["verification"]["status"] == "verified"
    assert verify_current_release(fixed)["passed"] is True
    pointer = json.loads(
        (fixed / storage.CURRENT_RELEASE_POINTER_NAME).read_text(encoding="utf-8")
    )
    assert pointer["publication"]["metric_key"] == "web_end_to_end"
    assert pointer["publication"]["completed"] is True
    assert pointer["publication"]["total_duration_seconds"] >= 0.01
    assert (fixed / "Original-Experiment-Videos" / "Original-Video-Index.json").is_file()
    assert (fixed / "Original-Experiment-Videos" / "view-01.ffconcat").is_file()
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        assert (fixed / directory / "version.txt").read_text(encoding="utf-8") == "new"
        assert (history / directory / "version.txt").read_text(encoding="utf-8") == "old"
    (fixed / "Experiment-Clips" / "version.txt").write_text(
        "tampered", encoding="utf-8"
    )
    assert verify_current_release(fixed)["passed"] is False


def test_fixed_archive_promotion_rolls_back_everything_if_pointer_publish_fails(
    tmp_path, monkeypatch
):
    fixed = tmp_path / "fixed"
    staging = tmp_path / "staging" / "release-new"
    history = tmp_path / "history" / "release-new"
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        old_file = fixed / directory / "version.txt"
        new_file = staging / directory / "version.txt"
        old_file.parent.mkdir(parents=True, exist_ok=True)
        new_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.write_text("old", encoding="utf-8")
        new_file.write_text("new", encoding="utf-8")
    old_reference = fixed / "Original-Experiment-Videos" / "README.txt"
    new_reference = staging / "Original-Experiment-Videos" / "README.txt"
    old_reference.parent.mkdir(parents=True, exist_ok=True)
    new_reference.parent.mkdir(parents=True, exist_ok=True)
    old_reference.write_text("old-reference", encoding="utf-8")
    new_reference.write_text("new-reference", encoding="utf-8")
    _complete_release_gate(staging)

    atomic_write = storage._atomic_write_text

    def fail_pointer(path: Path, value: str) -> None:
        if path.name == storage.CURRENT_RELEASE_POINTER_NAME:
            raise OSError("simulated pointer publication failure")
        atomic_write(path, value)

    monkeypatch.setattr(storage, "_atomic_write_text", fail_pointer)

    with pytest.raises(OSError, match="simulated pointer"):
        promote_fixed_archive(staging, fixed, history)

    assert not storage.archive_promotion_in_progress(fixed)
    assert not (fixed / storage.CURRENT_RELEASE_POINTER_NAME).exists()
    assert old_reference.read_text(encoding="utf-8") == "old-reference"
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        assert (fixed / directory / "version.txt").read_text(encoding="utf-8") == "old"
        assert (staging / directory / "version.txt").read_text(encoding="utf-8") == "new"


def test_interrupted_promotion_keeps_recovery_state_and_restores_previous_release(
    tmp_path, monkeypatch
):
    fixed = tmp_path / "fixed"
    staging = tmp_path / "staging" / "release-interrupted"
    history = tmp_path / "history" / "release-interrupted"
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        old_file = fixed / directory / "version.txt"
        new_file = staging / directory / "version.txt"
        old_file.parent.mkdir(parents=True, exist_ok=True)
        new_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.write_text("old", encoding="utf-8")
        new_file.write_text("new", encoding="utf-8")
    _complete_release_gate(staging)
    atomic_write = storage._atomic_write_text

    def interrupt_pointer(path: Path, value: str) -> None:
        if path.name == storage.CURRENT_RELEASE_POINTER_NAME:
            raise KeyboardInterrupt
        atomic_write(path, value)

    monkeypatch.setattr(storage, "_atomic_write_text", interrupt_pointer)
    with pytest.raises(KeyboardInterrupt):
        promote_fixed_archive(staging, fixed, history)

    assert storage.archive_promotion_in_progress(fixed) is True
    state_path = storage._promotion_state_path(fixed)
    assert state_path.is_file()
    orphan_manifest = (
        fixed / storage.RELEASE_MANIFEST_DIRECTORY / f"{staging.name}.json"
    )
    assert orphan_manifest.is_file()

    monkeypatch.setattr(storage, "_atomic_write_text", atomic_write)
    storage._recover_interrupted_promotion(fixed)

    assert not state_path.exists()
    assert not orphan_manifest.exists()
    assert storage.archive_promotion_in_progress(fixed) is False
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        assert (fixed / directory / "version.txt").read_text(encoding="utf-8") == "old"
        assert (staging / directory / "version.txt").read_text(encoding="utf-8") == "new"


def test_model_named_archive_paths_stay_below_classic_windows_limit():
    root = Path(
        "Y:/VisionCortexExperimentArchive/.VisionCortex-Run-Staging/"
        "Six-View-Three-Hour-Experiment-2026-08-13/cli-20260814-131034-f744"
    )
    layout = ArchiveLayout(root)
    folder = _group_folder_name(
        layout,
        1,
        "电子分析天平称量纸放置与试剂瓶准备实验" * 4,
        "Weighing-paper-placement-and-reagent-bottle-preparation-on-electronic-analytical-balance"
        * 3,
    )

    assert folder.startswith("001-")
    assert folder.isascii()
    assert len(str(layout.experiment_clips / folder / "Aligned_First+Third.json")) <= 235
    event = EvidenceEvent(
        event_id="EVT-000001",
        action_type=ActionType.DEVICE_PANEL_OPERATION,
        global_start_ms=10_000,
        global_end_ms=12_000,
        key_global_ms=11_000,
        objects=["panel"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=[],
        supporting_roles=[],
        candidates=[],
    )
    event_folder = _key_material_event_folder_name(layout, folder, event)
    assert event_folder.startswith("Operate-")
    assert (
        len(
            str(
                layout.key_frames
                / folder
                / "05-Device-Panel-Operation"
                / event_folder
                / "Aligned_First+Third.json"
            )
        )
        <= 235
    )
