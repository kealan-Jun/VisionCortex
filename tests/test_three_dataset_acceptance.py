from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from labvision_evidence.cli import app
from labvision_evidence.three_dataset_acceptance import (
    DatasetSpec,
    analyze_three_datasets,
    render_acceptance_markdown,
    write_acceptance_reports,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _event(event_id: str, group_id: str, action_type: str, primary: str) -> dict:
    folder = f"{event_id}_{primary}_Evidence"
    return {
        "event_id": event_id,
        "parent_event_id": group_id,
        "action_type": action_type,
        "objects": {"tool": primary, "source": "unknown", "target": "unknown"},
        "key_frames": [{"path": f"Key-Materials/Key-Frames/{folder}/{primary}.jpg"}],
        "key_clips": [{"path": f"Key-Materials/Key-Clips/{folder}/{primary}.mp4"}],
        "provenance": {
            "experiment_group_id": group_id,
            "archive_classification": {
                "primary_object": primary,
                "semantic_file_stem": folder,
            },
            "mllm": {
                "status": "completed",
                "current_step": f"operate {primary}",
                "next_step": f"place {primary}",
            },
        },
    }


def _accepted_candidate(
    root: Path,
    experiment_id: str,
    *,
    updated_at: str = "2026-08-18T12:00:00+08:00",
    baseline_applied: bool = True,
) -> None:
    json_root = root / "JSON-Config-Files"
    events = [
        _event("EVT-001", "GROUP-0002", "hand_object_contact", "Weighing-Paper"),
        _event("EVT-002", "GROUP-0004", "liquid_movement", "Sample-Bottle"),
    ]
    groups = [
        {
            "group_id": "GROUP-0002",
            "experiment_name": "Weighing",
            "continuity_type": "independent",
            "atomic_experiment_ids": ["EXP-2"],
            "participating_views": ["fp", "tp"],
        },
        {
            "group_id": "GROUP-0004",
            "experiment_name": "Transfer",
            "continuity_type": "continuous",
            "atomic_experiment_ids": ["EXP-4A", "EXP-4B"],
            "participating_views": ["fp", "tp"],
        },
    ]
    selection = {
        "current_experiment_id": experiment_id,
        "applied": baseline_applied,
        "reason": "experiment_id_matched" if baseline_applied else "experiment_id_not_matched",
    }
    boundary_matches = [
        {
            "baseline_id": "reviewed-exp-002",
            "predicted_group_id": "GROUP-0002",
            "predicted_continuity_type": "independent",
            "predicted_atomic_experiment_count": 1,
            "boundary_within_tolerance": True,
        },
        {
            "baseline_id": "reviewed-exp-004",
            "predicted_group_id": "GROUP-0004",
            "predicted_continuity_type": "continuous",
            "predicted_atomic_experiment_count": 2,
            "boundary_within_tolerance": True,
        },
    ]
    _write(
        json_root / "pipeline_status.json",
        {"stage": "completed", "failed_stage": None, "updated_at": updated_at},
    )
    _write(
        json_root / "run_metrics.json",
        {
            "run_ended_at": updated_at,
            "total_duration_seconds": 1000.0,
            "preprocessing_sla": {"actual_seconds": 900.0},
            "nas_index_ingest": {
                "experiment_id": experiment_id,
                "ingest_details": {
                    "copied_source_bytes": 0,
                    "continuous_source_copies_created": 0,
                },
            },
            "stage_durations": [
                {"stage": "candidate_fine", "duration_seconds": 600.0},
                {"stage": "mllm", "duration_seconds": 100.0},
            ],
            "tokens": {
                "experiment_groups": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
                "key_materials": {
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "total_tokens": 24,
                },
                "daily_report": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "run_total": {
                    "input_tokens": 30,
                    "output_tokens": 6,
                    "total_tokens": 36,
                },
            },
            "mllm_calls": [
                {"status": "completed", "latency_seconds": 12.0, "attempts": 1}
            ],
        },
    )
    quality = {
        "status": "passed" if baseline_applied else "structural_only",
        "passed": True,
        "baseline_selection": selection,
        "experiment_boundaries": {
            "evaluated": baseline_applied,
            "passed": True if baseline_applied else None,
            "predicted_experiment_count": 2,
            "matches": boundary_matches if baseline_applied else [],
        },
        "key_materials": {
            "passed": True,
            "event_count": 2,
            "action_counts": {
                "hand_object_contact": 1,
                "object_movement": 0,
                "liquid_movement": 1,
                "container_state_change": 0,
                "device_panel_operation": 0,
            },
            "media_complete_count": 2,
            "model_understanding_completed_count": 2,
        },
    }
    _write(json_root / "quality_acceptance.json", quality)
    _write(
        json_root / "boundary_precheck.json",
        {"baseline_selection": selection, "experiment_boundaries": quality["experiment_boundaries"]},
    )
    _write(
        json_root / "evidence_package_eval.json",
        {
            "passed": True,
            "experiment_group_count": 2,
            "key_event_count": 2,
            "dual_view_material_count": 2,
        },
    )
    _write(
        json_root / "audit_layer.json",
        {"events": [], "segments": [], "experiment_groups": groups, "formal_segment_receipts": []},
    )
    _write(json_root / "progressive_fine_scan.json", {"unresolved_candidate_ids": []})
    _write(
        json_root / "input_volume_report.json",
        {
            "unique_video_path_count": 4,
            "unique_clock_path_count": 4,
            "total_video_bytes": 100,
            "views": [
                {"view_id": "fp", "role": "first_person", "segment_count": 2},
                {"view_id": "tp", "role": "third_person", "segment_count": 2},
            ],
        },
    )
    _write(json_root / "run_manifest.json", {"experiment_id": experiment_id, "views": []})
    _write(
        json_root / "evidence_package.json",
        {"experiment_id": experiment_id, "experiment_groups": groups, "events": []},
    )
    _write(
        json_root / "fixed_archive_promotion.json",
        {"verification": {"status": "verified", "directory_file_counts": {"Experiment-Clips": 6}}},
    )
    _write(
        json_root / "daily_report_manifest.json",
        {"passed": True, "daily_template_id": "VC-LAB-DAILY-REPORT-V2"},
    )
    _write(
        json_root / "professional_report_manifest.json",
        {"status": "generated", "pdf": "Professional-PDFs/report.pdf"},
    )
    _write(
        json_root / "evidence_index_manifest.json",
        {"validation": {"passed": True}},
    )
    _write(
        json_root / "continuous_action_state_ledger.json",
        {"cv_acceptance_mutated": False, "receipts": [{"event_id": "EVT-001"}]},
    )
    _write(json_root / "resource_telemetry.json", {"sample_count": 2, "stage_summaries": {}})
    _write(json_root / "scan_runtime_fine.json", {"work_units": [], "role_reports": []})
    _write(json_root / "scan_runtime_motion_probe.json", {"work_units": [], "role_reports": []})
    _write(root / "Key-Materials" / "Key-Materials-Model-Understanding.json", events)


def _six_spec(archive_name: str = "Six") -> DatasetSpec:
    return DatasetSpec(
        key="six_view",
        expected_source_experiment_id="exp-six",
        archive_name=archive_name,
        six_view_reviewed_baseline=True,
        preprocessing_target_seconds=1200,
        expected_group_count=2,
        minimum_key_event_count=2,
    )


def test_read_only_acceptance_passes_without_mutating_archive(tmp_path: Path):
    archive_root = tmp_path / "archive"
    candidate = archive_root / "Six"
    _accepted_candidate(candidate, "exp-six")
    before = _tree_digest(archive_root)

    result = analyze_three_datasets("DEV-TEST", archive_root, [_six_spec()])

    assert result["passed"] is True
    assert result["production_release_ready"] is True
    assert result["source_policy"]["video_files_opened"] == 0
    assert result["source_policy"]["clock_csv_files_opened"] == 0
    assert result["source_policy"]["model_api_calls"] == 0
    assert result["datasets"][0]["quality"]["material_semantics"][
        "semantic_object_filename_rate"
    ] == 1.0
    assert before == _tree_digest(archive_root)
    assert "Production release ready: **YES**" in render_acceptance_markdown(result)


def test_latest_failed_staging_is_not_hidden_by_old_formal_archive(tmp_path: Path):
    archive_root = tmp_path / "archive"
    formal = archive_root / "Six"
    staging = archive_root / ".VisionCortex-Run-Staging" / "Six" / "run-new"
    _accepted_candidate(formal, "exp-six", updated_at="2026-08-18T10:00:00+08:00")
    _accepted_candidate(staging, "exp-six", updated_at="2026-08-18T11:00:00+08:00")
    _write(
        staging / "JSON-Config-Files" / "pipeline_status.json",
        {
            "stage": "failed",
            "failed_stage": "candidate_audit",
            "updated_at": "2026-08-18T11:00:00+08:00",
        },
    )

    result = analyze_three_datasets("DEV-TEST", archive_root, [_six_spec()])

    dataset = result["datasets"][0]
    assert dataset["candidate"]["kind"] == "staging"
    assert dataset["run"]["state"] == "failed"
    assert dataset["passed"] is False
    assert "cv_boundary_or_cross_view_quality" in dataset["failure_categories"]


def test_running_candidate_is_reported_as_in_progress(tmp_path: Path):
    archive_root = tmp_path / "archive"
    candidate = archive_root / ".VisionCortex-Run-Staging" / "Six" / "run-live"
    _accepted_candidate(candidate, "exp-six")
    _write(
        candidate / "JSON-Config-Files" / "pipeline_status.json",
        {
            "stage": "motion_probe",
            "failed_stage": None,
            "updated_at": "2026-08-18T12:30:00+08:00",
        },
    )

    result = analyze_three_datasets("DEV-TEST", archive_root, [_six_spec()])

    assert result["status"] == "in_progress"
    assert result["summary"]["in_progress_count"] == 1
    assert result["summary"]["failed_count"] == 0
    assert result["datasets"][0]["status"] == "in_progress"


def test_natural_dataset_requires_baseline_to_be_inapplicable(tmp_path: Path):
    archive_root = tmp_path / "archive"
    candidate = archive_root / "Natural"
    _accepted_candidate(candidate, "exp-natural", baseline_applied=False)
    spec = DatasetSpec(
        key="a",
        expected_source_experiment_id="exp-natural",
        archive_name="Natural",
    )

    result = analyze_three_datasets("DEV-TEST", archive_root, [spec])

    assert result["passed"] is True
    check = next(
        item
        for item in result["datasets"][0]["checks"]
        if item["check"] == "six_view_baseline_inapplicable"
    )
    assert check["passed"] is True


def test_report_writer_refuses_to_write_inside_archive(tmp_path: Path):
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    result = {"status": "failed", "summary": {}, "datasets": []}

    with pytest.raises(ValueError, match="outside the NAS archive root"):
        write_acceptance_reports(
            result, archive_root / "audit" / "result.json", archive_root
        )


def test_cli_writes_json_and_markdown_outside_archive(tmp_path: Path):
    archive_root = tmp_path / "archive"
    _accepted_candidate(archive_root / "Six", "exp-six")
    spec = tmp_path / "spec.json"
    _write(
        spec,
        {
            "task_id": "DEV-TEST",
            "datasets": [
                {
                    "key": "six_view",
                    "expected_source_experiment_id": "exp-six",
                    "archive_name": "Six",
                    "six_view_reviewed_baseline": True,
                    "preprocessing_target_seconds": 1200,
                    "expected_group_count": 2,
                    "minimum_key_event_count": 2,
                }
            ],
        },
    )
    output = tmp_path / "reports" / "acceptance.json"

    response = CliRunner().invoke(
        app,
        [
            "accept-three-datasets",
            "--archive-root",
            str(archive_root),
            "--spec",
            str(spec),
            "--output",
            str(output),
        ],
    )

    assert response.exit_code == 0, response.output
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_policy"]["video_files_opened"] == 0
    assert "\"production_release_ready\": true" in response.output.lower()
