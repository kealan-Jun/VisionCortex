from __future__ import annotations

import json
from pathlib import Path

from labvision_evidence.replay_acceptance import (
    build_archive_regression_snapshot,
    compare_archive_snapshot,
    replay_quality_decisions_from_ledgers,
)
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ExperimentSegment,
    RunManifest,
    ViewInput,
    ViewRole,
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


def _quality_replay_archive(root: Path, *, exact: bool) -> None:
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=Path("tp.mp4")),
    ]
    candidates = [
        ActionCandidate(
            candidate_id=f"CORE-{view.view_id}",
            action_type=ActionType.HAND_OBJECT_CONTACT,
            view_id=view.view_id,
            role=view.role,
            local_start_ms=10_000.0,
            local_end_ms=30_000.0,
            global_start_ms=10_000.0,
            global_end_ms=30_000.0,
            key_global_ms=20_000.0,
            objects=["gloved_hand", "pipette", "tube"],
            confidence=0.9,
        )
        for view in views
    ]
    event = EvidenceEvent(
        event_id="EVT-CORE",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=10_000.0,
        global_end_ms=30_000.0,
        key_global_ms=20_000.0,
        objects=["gloved_hand", "pipette", "tube"],
        confidence=0.95,
        accepted=True,
        audit_reason="dual-view fixture",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=candidates,
    )
    boundary = ActionCandidate(
        candidate_id="MOTION",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=5_000.0,
        local_end_ms=60_000.0,
        global_start_ms=5_000.0,
        global_end_ms=60_000.0,
        key_global_ms=20_000.0,
        objects=[],
        confidence=0.9,
    )
    raw = ExperimentSegment(
        segment_id="EXP-0001",
        global_start_ms=8_000.0,
        global_end_ms=45_000.0,
        event_ids=[event.event_id],
        participating_views=["fp", "tp"],
    )
    json_root = root / "JSON-Config-Files"
    _write(
        json_root / "run_manifest.json",
        RunManifest(experiment_id="quality_replay", views=views).model_dump(mode="json"),
    )
    audit = {
        "events": [event.model_dump(mode="json")],
        "raw_segments": [raw.model_dump(mode="json")],
    }
    if exact:
        audit["boundary_candidates"] = [boundary.model_dump(mode="json")]
    _write(json_root / "audit_layer.json", audit)
    _write(
        json_root / "motion_probe_windows.json",
        {"candidates": [boundary.model_dump(mode="json")]},
    )


def test_quality_ledger_replay_is_exact_when_boundary_candidates_are_persisted(
    tmp_path: Path, default_config
):
    _quality_replay_archive(tmp_path, exact=True)

    result = replay_quality_decisions_from_ledgers(tmp_path, default_config)

    assert result["evidence_grade"] == "exact"
    assert result["counts"]["experiment_groups"] == 1
    assert result["counts"]["selected_key_events"] == 1
    assert result["limitations"] == []
    assert result["source_policy"]["video_files_opened"] == 0
    assert result["source_policy"]["token_usage"] == 0


def test_legacy_quality_replay_marks_implicit_boundary_extension_as_degraded(
    tmp_path: Path, default_config
):
    _quality_replay_archive(tmp_path, exact=False)

    result = replay_quality_decisions_from_ledgers(tmp_path, default_config)

    assert result["evidence_grade"] == "degraded_legacy_ledger"
    assert result["limitations"]
    invariant = result["legacy_boundary_invariants"][0]
    assert invariant["end_delta_from_direct_event_boundary_ms"] == 12_000.0
    assert invariant["requires_explicit_qf1_receipt"] is True
