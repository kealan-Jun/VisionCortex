from __future__ import annotations

import json
from pathlib import Path

from labvision_evidence.replay_acceptance import (
    build_archive_regression_snapshot,
    compare_archive_snapshot,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _archive(root: Path) -> None:
    json_root = root / "JSON-Config-Files"
    _write(json_root / "evidence_package.json", {"schema_version": "2.0.0"})
    _write(
        json_root / "evidence_package_eval.json",
        {"passed": True, "experiment_group_count": 1, "dual_view_material_count": 2},
    )
    _write(
        json_root / "quality_acceptance.json",
        {
            "schema_version": "visioncortex-quality-acceptance/1",
            "status": "passed",
            "experiment_boundaries": {
                "evaluated": True,
                "passed": True,
                "predicted_experiment_count": 1,
                "matches": [
                    {
                        "baseline_id": "reviewed-exp-001",
                        "predicted_group_id": "GROUP-0001",
                        "predicted_continuity_type": "independent",
                        "predicted_atomic_experiment_count": 1,
                        "boundary_within_tolerance": True,
                    }
                ],
            },
            "key_materials": {
                "event_count": 2,
                "events": [{"event_id": "EVT-001"}, {"event_id": "EVT-002"}],
                "action_counts": {"hand_object_contact": 2},
                "media_complete_count": 2,
                "cross_view_supported_count": 2,
                "cross_view_supported_rate": 1.0,
            },
        },
    )
    _write(json_root / "run_metrics.json", {"stage_durations": [], "tokens": {}})
    _write(
        json_root / "audit_layer.json",
        {
            "events": [],
            "segments": [],
            "experiment_groups": [],
            "formal_segment_receipts": [
                {"decision": "quarantined_missing_dual_view", "event_ids": ["EVT-Q"]}
            ],
        },
    )
    _write(json_root / "physical_change_log.json", [])
    _write(root / "Key-Materials" / "Key-Materials-Model-Understanding.json", [])


def test_replay_uses_json_ledgers_and_passes_baseline(tmp_path: Path):
    _archive(tmp_path)
    snapshot = build_archive_regression_snapshot(tmp_path)
    result = compare_archive_snapshot(
        snapshot,
        {
            "baseline_id": "unit-baseline",
            "gates": {
                "experiment_group_count": 1,
                "minimum_key_event_count": 2,
                "boundary_structure": {
                    "reviewed-exp-001": {
                        "continuity_type": "independent",
                        "atomic_experiment_count": 1,
                    }
                },
            },
        },
    )

    assert result["passed"] is True
    assert snapshot["source_policy"] == {
        "video_files_opened": 0,
        "clock_csv_files_opened": 0,
        "mode": "durable_json_ledger_replay",
    }


def test_replay_rejects_quarantined_event_leakage(tmp_path: Path):
    _archive(tmp_path)
    path = tmp_path / "JSON-Config-Files" / "quality_acceptance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["key_materials"]["events"].append({"event_id": "EVT-Q"})
    _write(path, payload)

    snapshot = build_archive_regression_snapshot(tmp_path)
    result = compare_archive_snapshot(
        snapshot,
        {"baseline_id": "unit-baseline", "gates": {}},
    )

    assert result["passed"] is False
    assert any(item["check"] == "quarantined_event_leakage" for item in result["failures"])
