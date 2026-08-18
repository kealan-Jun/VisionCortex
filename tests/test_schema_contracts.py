from __future__ import annotations

import json
from pathlib import Path

from labvision_evidence.schema_contracts import (
    inspect_archive_contracts,
    validate_event_contract,
    write_archive_contract_manifest,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimum_archive(root: Path) -> None:
    json_root = root / "JSON-Config-Files"
    _write(json_root / "evidence_package.json", {"schema_version": "2.0.0"})
    _write(json_root / "evidence_package_eval.json", {"passed": True})
    _write(
        json_root / "quality_acceptance.json",
        {
            "schema_version": "visioncortex-quality-acceptance/1",
            "status": "passed",
            "experiment_boundaries": {},
            "key_materials": {},
        },
    )
    _write(json_root / "run_metrics.json", {"stage_durations": [], "tokens": {}})
    _write(
        json_root / "audit_layer.json",
        {"events": [], "segments": [], "experiment_groups": []},
    )
    _write(json_root / "physical_change_log.json", [])
    _write(root / "Key-Materials" / "Key-Materials-Model-Understanding.json", [])


def test_archive_contracts_accept_current_package_and_list_change_log(tmp_path: Path):
    _minimum_archive(tmp_path)

    result = inspect_archive_contracts(tmp_path)

    assert result["passed"] is True
    assert not result["failures"]
    assert {item["contract_id"] for item in result["warnings"]} == {
        "continuous-action-state",
        "evidence-index-manifest",
    }


def test_archive_contract_manifest_is_written_atomically(tmp_path: Path):
    _minimum_archive(tmp_path)

    path = write_archive_contract_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "schema_contract_manifest.json"
    assert payload["passed"] is True
    assert not list(path.parent.glob("*.partial-*"))


def test_event_contract_preserves_required_material_references():
    event = {
        "event_id": "event-001",
        "action_type": "liquid_transfer",
        "start_us": 12_000_000,
        "end_us": 20_500_000,
        "peak_timestamp_us": 16_700_000,
        "objects": {},
        "observations": [],
        "cross_view_associations": [],
        "decision": {},
        "scores": {},
        "key_frames": [],
        "key_clips": [],
        "evidence_ids": [],
        "provenance": {},
    }

    assert validate_event_contract(event) == []
    del event["key_clips"]
    event["end_us"] = 1
    assert validate_event_contract(event) == ["key_clips", "end_us_before_start_us"]
