from __future__ import annotations

import hashlib
import json

from labvision_evidence.model_promotion import evaluate_yolo_candidate_promotion


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_metrics_cannot_bypass_internal_real_six_view_gate(tmp_path):
    model = tmp_path / "candidate.pt"
    model.write_bytes(b"candidate")
    classes = ["hand", "pipette", *[f"class-{index}" for index in range(19)]]
    dataset = tmp_path / "dataset-receipt.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-mapped-public-yolo-union/1",
                "class_count": 21,
                "classes": classes,
            }
        ),
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-yolo-human-truth-evaluation/1",
                "status": "completed",
                "truth_status": "public_human_annotations",
                "model": str(model),
                "model_sha256": _sha(model),
                "dataset_receipt": str(dataset),
                "dataset_receipt_sha256": _sha(dataset),
                "metrics": {"precision": 0.96, "recall": 0.97},
                "per_class": [
                    {"class_name": "hand", "precision": 0.96, "recall": 0.97},
                    {"class_name": "pipette", "precision": 0.95, "recall": 0.96},
                ],
                "production_certified": False,
                "nas_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-yolo-promotion-gate/1",
                "ontology": {"required_class_count": 21, "required_classes": classes},
                "public_benchmark": {
                    "overall": {"precision": 0.9, "recall": 0.9},
                    "per_class": {
                        "hand": {"recall": 0.9},
                        "pipette": {"recall": 0.9},
                    },
                },
                "internal_real_six_view_ab": {
                    "required": True,
                    "minimum_precision": 0.95,
                    "minimum_recall": 0.95,
                },
            }
        ),
        encoding="utf-8",
    )

    decision = evaluate_yolo_candidate_promotion(
        evaluation, gate, tmp_path / "decision.json"
    )

    assert decision["ontology_compatible"] is True
    assert decision["public_benchmark_passed"] is True
    assert decision["internal_ab_passed"] is False
    assert decision["promotion_ready"] is False
    assert decision["production_configuration_changed"] is False
    assert decision["nas_accessed"] is False

    internal_ab = tmp_path / "internal-ab.json"
    internal_ab.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-internal-yolo-real-six-view-ab/1",
                "status": "completed",
                "passed": True,
                "real_six_view": True,
                "truth_status": "reviewed_ground_truth",
                "model_sha256": _sha(model),
                "metrics": {"precision": 0.96, "recall": 0.97},
            }
        ),
        encoding="utf-8",
    )

    qualified = evaluate_yolo_candidate_promotion(
        evaluation,
        gate,
        tmp_path / "qualified-decision.json",
        internal_ab_receipt_path=internal_ab,
    )

    assert qualified["public_benchmark_passed"] is True
    assert qualified["internal_ab_passed"] is True
    assert qualified["internal_ab_receipt_sha256"] == _sha(internal_ab)
    assert qualified["promotion_ready"] is True
    assert qualified["status"] == "promotion_ready_requires_explicit_commit"
    assert qualified["production_configuration_changed"] is False
    assert qualified["production_enabled"] is False
