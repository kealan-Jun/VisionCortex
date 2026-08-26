from __future__ import annotations

import json

from labvision_evidence.model_certification import (
    build_model_certification_readiness,
)


def test_readiness_counts_only_eligible_reviewed_events_without_archives(tmp_path):
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-key-event-ground-truth/1.0.0",
                "ground_truth_id": "truth-1",
                "experiment_id": "exp-1",
                "applicability": {"experiment_ids": ["exp-1"]},
                "events": [
                    {
                        "event_id": "e1",
                        "action_type": "hand_object_contact",
                        "start_seconds": 1,
                        "end_seconds": 2,
                        "objects": ["paper"],
                        "decision": {"status": "confirmed"},
                    },
                    {
                        "event_id": "e2",
                        "action_type": "liquid_movement",
                        "start_seconds": 3,
                        "end_seconds": 4,
                        "objects": ["tube"],
                        "decision": {"status": "uncertain"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "models": {},
        "validation": {
            "key_event_ground_truth": {"exp-1": str(truth_path)},
            "model_certification": {
                "path": str(tmp_path / "certification.json"),
                "box_evaluation_reports": [],
                "targets": {
                    "minimum_event_dataset_count": 2,
                    "minimum_event_ground_truth_count": 3,
                    "minimum_ground_truth_per_action": 2,
                    "minimum_box_dataset_count": 1,
                    "minimum_box_ground_truth_instances": 10,
                    "required_action_types": [
                        "hand_object_contact",
                        "liquid_movement",
                    ],
                },
            },
        },
    }

    report = build_model_certification_readiness(
        settings, repository_root=tmp_path
    )

    assert report["nas_accessed"] is False
    assert report["event_truth"]["eligible_event_count"] == 1
    assert report["event_truth"]["per_action"] == {"hand_object_contact": 1}
    assert report["ready_for_certification_run"] is False
    assert report["production_certified"] is False
    assert report["production_certification_audit"] == {"status": "missing"}
    requirements = {item["requirement"] for item in report["deficits"]}
    assert "event_ground_truth_count" in requirements
    assert "action_ground_truth:liquid_movement" in requirements
    assert "box_ground_truth_instance_count" in requirements
