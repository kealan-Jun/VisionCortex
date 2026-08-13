import json

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
    daily_eval = staging / "Lab-Daily-Reports" / "2026-08-13" / "Daily-Report-Eval.json"
    daily_eval.parent.mkdir(parents=True, exist_ok=True)
    daily_eval.write_text(json.dumps({"passed": True}), encoding="utf-8")
    daily_pdf = staging / "Professional-PDFs" / "Lab-Daily-Report-2026-08-13.pdf"
    daily_pdf.parent.mkdir(parents=True, exist_ok=True)
    daily_pdf.write_bytes(b"%PDF-test")

    receipt = promote_fixed_archive(staging, fixed, history)

    assert original.read_bytes() == b"original"
    assert receipt["previous_package_retained"] is True
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        assert (fixed / directory / "version.txt").read_text(encoding="utf-8") == "new"
        assert (history / directory / "version.txt").read_text(encoding="utf-8") == "old"
