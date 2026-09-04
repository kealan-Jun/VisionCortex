from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMOTION_SCHEMA = "visioncortex-yolo-candidate-promotion-decision/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_checks(
    metrics: dict[str, Any], thresholds: dict[str, Any], prefix: str
) -> list[dict[str, Any]]:
    checks = []
    for name, minimum in thresholds.items():
        value = metrics.get(name)
        passed = value is not None and float(value) >= float(minimum)
        checks.append(
            {
                "metric": f"{prefix}.{name}",
                "value": float(value) if value is not None else None,
                "minimum": float(minimum),
                "passed": passed,
            }
        )
    return checks


def evaluate_yolo_candidate_promotion(
    evaluation_receipt_path: Path,
    gate_path: Path,
    output: Path,
    *,
    internal_ab_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Decide promotion readiness without ever mutating production configuration."""

    evaluation_receipt_path = evaluation_receipt_path.resolve()
    gate_path = gate_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Candidate promotion decision already exists: {output}")
    evaluation = json.loads(evaluation_receipt_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        evaluation.get("schema_version")
        != "visioncortex-yolo-human-truth-evaluation/1"
        or evaluation.get("status") != "completed"
        or evaluation.get("truth_status") != "public_human_annotations"
        or evaluation.get("production_certified") is not False
        or evaluation.get("nas_accessed") is not False
    ):
        raise RuntimeError("Candidate evaluation receipt is not trusted public truth")
    if gate.get("schema_version") != "visioncortex-yolo-promotion-gate/1":
        raise RuntimeError("Unsupported candidate promotion gate schema")
    model_path = Path(str(evaluation.get("model") or "")).resolve()
    if (
        not model_path.is_file()
        or _sha256(model_path) != evaluation.get("model_sha256")
    ):
        raise RuntimeError("Candidate artifact no longer matches its evaluation receipt")
    dataset_receipt_path = Path(str(evaluation.get("dataset_receipt") or "")).resolve()
    dataset_receipt = json.loads(dataset_receipt_path.read_text(encoding="utf-8"))
    if _sha256(dataset_receipt_path) != evaluation.get("dataset_receipt_sha256"):
        raise RuntimeError("Candidate dataset receipt hash mismatch")
    ontology = gate.get("ontology") or {}
    ontology_compatible = bool(
        dataset_receipt.get("schema_version")
        == "visioncortex-mapped-public-yolo-union/1"
        and int(dataset_receipt.get("class_count") or 0)
        == int(ontology.get("required_class_count") or 0)
        and dataset_receipt.get("classes") == ontology.get("required_classes")
    )

    public_gate = gate.get("public_benchmark") or {}
    checks = _metric_checks(
        evaluation.get("metrics") or {}, public_gate.get("overall") or {}, "overall"
    )
    per_class = {
        str(item.get("class_name")): item
        for item in evaluation.get("per_class") or []
        if isinstance(item, dict)
    }
    for class_name, thresholds in (public_gate.get("per_class") or {}).items():
        checks.extend(
            _metric_checks(
                per_class.get(class_name) or {}, thresholds, f"class.{class_name}"
            )
        )
    public_benchmark_passed = bool(checks) and all(item["passed"] for item in checks)

    internal_policy = gate.get("internal_real_six_view_ab") or {}
    internal_ab = None
    internal_ab_sha256 = None
    internal_ab_passed = False
    if internal_ab_receipt_path is not None:
        internal_path = internal_ab_receipt_path.resolve()
        internal_ab_sha256 = _sha256(internal_path)
        internal_ab = json.loads(internal_path.read_text(encoding="utf-8"))
        internal_ab_passed = bool(
            internal_ab.get("schema_version")
            == "visioncortex-internal-yolo-real-six-view-ab/1"
            and internal_ab.get("status") == "completed"
            and internal_ab.get("passed") is True
            and internal_ab.get("real_six_view") is True
            and internal_ab.get("truth_status") == "reviewed_ground_truth"
            and internal_ab.get("model_sha256") == evaluation.get("model_sha256")
            and float((internal_ab.get("metrics") or {}).get("precision") or 0.0)
            >= float(internal_policy.get("minimum_precision") or 1.0)
            and float((internal_ab.get("metrics") or {}).get("recall") or 0.0)
            >= float(internal_policy.get("minimum_recall") or 1.0)
        )
    blockers = []
    if not ontology_compatible:
        blockers.append("candidate dataset does not exactly match the production ontology")
    if not public_benchmark_passed:
        blockers.append("public held-out metric gate did not pass")
    if internal_policy.get("required") is True and not internal_ab_passed:
        blockers.append("independent internal real-six-view A/B receipt is missing or failed")
    promotion_ready = bool(
        ontology_compatible
        and public_benchmark_passed
        and (
            internal_ab_passed
            or internal_policy.get("required") is not True
        )
    )
    payload = {
        "schema_version": PROMOTION_SCHEMA,
        "status": (
            "promotion_ready_requires_explicit_commit"
            if promotion_ready
            else "evaluated_not_promoted"
        ),
        "evaluation_receipt": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": _sha256(evaluation_receipt_path),
        "gate": str(gate_path),
        "gate_sha256": _sha256(gate_path),
        "model": str(model_path),
        "model_sha256": evaluation.get("model_sha256"),
        "ontology_compatible": ontology_compatible,
        "public_metric_checks": checks,
        "public_benchmark_passed": public_benchmark_passed,
        "internal_ab_receipt": (
            str(internal_ab_receipt_path.resolve())
            if internal_ab_receipt_path is not None
            else None
        ),
        "internal_ab_receipt_sha256": internal_ab_sha256,
        "internal_ab_passed": internal_ab_passed,
        "promotion_ready": promotion_ready,
        "production_configuration_changed": False,
        "production_enabled": False,
        "blockers": blockers,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
