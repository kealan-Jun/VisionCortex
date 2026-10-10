"""Recovery responsibilities for explicitly authorized training."""
from __future__ import annotations

import json
from pathlib import Path
from ...project_training_evidence import durable_bytes, verify_training_evidence

from .common import sha256, write_json
from .cohort import validate_export, training_sampling_sources
from .evaluation import project_evaluation
from .trace import optimization_trace_usage


def recover_evaluated_experiment(output: Path) -> dict:
    """Revalidate completed prediction artifacts after a final receipt failure."""
    path = output / "experiment.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    policy = previous.get("branch_loss_policy", "native")
    if policy not in ("native", "one2many") or policy != "native" and not previous.get("trace_branch_loss"):
        raise ValueError("Recovery requires a declared, traced branch objective")
    if previous.get("trace_branch_loss"):
        from ...project_branch_loss import branch_loss_trace_usage

        usage = branch_loss_trace_usage(
            output / "branch-loss-trace.jsonl", output / "candidate/source-sampling-trace.jsonl",
            expected_policy=policy,
        )
        if usage != previous.get("branch_loss_usage"):
            raise ValueError("Recovery requires matching observed branch-loss evidence")
    if previous["status"] == "completed":
        if previous.get("training_evidence_required") and (
            previous.get("training_evidence") != verify_training_evidence(
                output, require_branch_loss=previous.get("trace_branch_loss", False),
            )
        ):
            raise ValueError("Completed receipt differs from durable training evidence")
        return previous
    if previous["status"] != "failed":
        raise ValueError("Only a stopped failed experiment can be recovered")
    if previous.get("training_evidence_required"):
        evidence = verify_training_evidence(
            output, require_branch_loss=previous.get("trace_branch_loss", False),
        )
        if previous.get("training_evidence") != evidence:
            raise ValueError("Recovery requires matching durable training evidence")
    root = Path(previous["configuration"]["data"]).parent.parent
    receipt, rows = validate_export(
        root, previous["dataset_receipt_sha256"], previous["role"],
        supervision=previous.get("supervision", "complete"),
    )
    names = [c["name"] for c in receipt["project"]["classes"]]
    if "nbs" in previous["configuration"]:
        usage = previous.get("optimization_usage", {})
        trace = output.resolve() / "optimization-trace.jsonl"
        if not usage or usage != optimization_trace_usage(trace):
            raise ValueError("Recovery requires matching recorded optimizer steps")
    if previous.get("supervision") == "outside_ignore":
        if previous.get("sampling_policy") == "source_balanced" or "sampling_usage" in previous:
            from ...project_sampling import sampling_usage

            usage = previous.get("sampling_usage", {})
            sources = training_sampling_sources(root, previous["role"], receipt, rows)
            plan = output.resolve() / "candidate/source-sampling-plan.json"
            trace = output.resolve() / "candidate/source-sampling-trace.jsonl"
            if not usage or usage != sampling_usage(plan, trace, sources, previous["sampling_policy"]):
                raise ValueError("Recovery requires matching recorded source sampling usage")
        usage = previous.get("ignore_loss_usage", {})
        if (
            set(usage) not in ({"detection"}, {"one2many", "one2one"})
            or any(r.get("calls", 0) <= 0 for r in usage.values())
            or (any(r["annotation"]["ignore_regions"] for r in rows) and any(
                r.get("ignored_anchor_visits", 0) <= 0 for r in usage.values()
            ))
        ):
            raise ValueError("Partial-supervision recovery requires recorded ignore loss usage")
    validation = [r for r in rows if r["annotation"]["split"] == "val"]
    summaries = {}
    files = {}
    for label, model in [
        ("baseline", Path(previous["base_model"])),
        ("candidate", output / "candidate/weights/best.pt"),
    ]:
        report_path, predictions_path = (
            output / f"{label}-evaluation.json",
            output / f"{label}-predictions.json",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        if report["model_sha256"] != sha256(model):
            raise ValueError("Stored evaluation does not match the model")
        expected_ids = [r["id"] for r in validation]
        if [r["image_id"] for r in predictions] != expected_ids:
            raise ValueError(
                "Stored predictions do not match the frozen validation images"
            )
        recomputed = project_evaluation(predictions, validation, names)
        if (
            recomputed["per_class"] != report["per_class"]
            or recomputed["micro"] != report["micro"]
        ):
            raise ValueError("Stored evaluation metrics failed recomputation")
        summaries[label] = report
        for artifact in [model, report_path, predictions_path]:
            files[str(artifact)] = sha256(artifact)
    if sha256(Path(previous["base_model"])) != previous["base_model_sha256"]:
        raise ValueError("Base model identity changed")
    archive = output / ("failed-experiment-" + sha256(path)[:16] + ".json")
    if not archive.exists():
        durable_bytes(archive, path.read_bytes())
    result = dict(previous)
    result.update(
        status="completed",
        candidate_model_sha256=summaries["candidate"]["model_sha256"],
        candidate_micro=summaries["candidate"]["micro"],
        insufficient_positive_classes=summaries["candidate"][
            "insufficient_positive_classes"
        ],
        recovery=dict(
            previous_failure=str(archive),
            previous_failure_sha256=sha256(archive),
            verified_files=files,
            source_sha256=sha256(Path(__file__)),
            retrained=False,
        ),
    )
    result.pop("error", None)
    write_json(path, result)
    return result
