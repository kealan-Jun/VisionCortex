import json
from pathlib import Path

from labvision_evidence.archive import (
    ArchiveLayout,
    _group_folder_name,
    _key_material_event_folder_name,
)
from labvision_evidence.schemas import ActionType, EvidenceEvent
from labvision_evidence.storage import DERIVED_ARCHIVE_DIRECTORIES, promote_fixed_archive


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
    (staging / "JSON-Config-Files" / "evidence_package_eval.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (staging / "JSON-Config-Files" / "quality_acceptance.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    daily_eval = staging / "Lab-Daily-Reports" / "2026-08-13" / "Daily-Report-Eval.json"
    daily_eval.parent.mkdir(parents=True, exist_ok=True)
    daily_eval.write_text(json.dumps({"passed": True}), encoding="utf-8")
    daily_pdf = (
        staging
        / "Professional-PDFs"
        / "VisionCortex-Professional-Evidence-Report-2026-08-13.pdf"
    )
    daily_pdf.parent.mkdir(parents=True, exist_ok=True)
    daily_pdf.write_bytes(b"%PDF-test")
    staged_original = staging / "Original-Experiment-Videos"
    staged_original.mkdir(parents=True)
    (staged_original / "Original-Video-Index.json").write_text(
        json.dumps({"source_copy_bytes": 0}), encoding="utf-8"
    )
    (staged_original / "view-01.ffconcat").write_text(
        "ffconcat version 1.0\nfile 'Y:/source.mp4'\n", encoding="utf-8"
    )
    (staged_original / "README.txt").write_text("zero-copy", encoding="utf-8")

    receipt = promote_fixed_archive(staging, fixed, history)

    assert original.read_bytes() == b"original"
    assert receipt["previous_package_retained"] is True
    assert receipt["original_media_preserved"] is True
    assert len(receipt["promoted_original_references"]) == 3
    assert receipt["verification"]["status"] == "verified"
    assert (fixed / "Original-Experiment-Videos" / "Original-Video-Index.json").is_file()
    assert (fixed / "Original-Experiment-Videos" / "view-01.ffconcat").is_file()
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        assert (fixed / directory / "version.txt").read_text(encoding="utf-8") == "new"
        assert (history / directory / "version.txt").read_text(encoding="utf-8") == "old"


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
