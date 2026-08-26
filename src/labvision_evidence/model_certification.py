from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .recall_evaluation import evaluate_key_event_recall
from .reviewed_artifacts import load_dataset_scoped_json
from .schemas import ActionType, RunSummary


CERTIFICATION_SCHEMA = "visioncortex-production-model-certification/1"
READINESS_SCHEMA = "visioncortex-production-model-certification-readiness/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(successes: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = rate + z * z / (2.0 * total)
    margin = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    )
    return {
        "method": "wilson_95_percent",
        "lower": max(0.0, (centre - margin) / denominator),
        "upper": min(1.0, (centre + margin) / denominator),
    }


def _metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "precision_confidence_interval": _wilson(tp, tp + fp),
        "recall": recall,
        "recall_confidence_interval": _wilson(tp, tp + fn),
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        ),
    }


def _find_formal_archive(archive_root: Path, experiment_id: str) -> Path | None:
    matches: list[Path] = []
    for candidate in archive_root.iterdir() if archive_root.is_dir() else []:
        if candidate.name.startswith(".") or not candidate.is_dir():
            continue
        package = candidate / "JSON-Config-Files" / "evidence_package.json"
        if not package.is_file():
            continue
        try:
            payload = json.loads(package.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if str(payload.get("experiment_id") or "") == experiment_id:
            matches.append(candidate)
    return max(matches, key=lambda path: path.stat().st_mtime, default=None)


def _artifact_paths(settings: dict[str, Any]) -> list[tuple[str, Path]]:
    models = settings.get("models") or {}
    open_vocabulary = models.get("open_vocabulary_key_frame") or {}
    fallback = open_vocabulary.get("grounding_dino_fallback") or {}
    temporal_segmentation = (
        models.get("temporal_participant_segmentation") or {}
    )
    liquid_semantic = models.get("liquid_semantic_sidecar") or {}
    configured = [
        ("first_person_tensor_rt", models.get("first_person_engine")),
        ("third_person_tensor_rt", models.get("third_person_engine")),
        ("yolo_world", open_vocabulary.get("model_path")),
        ("clip_text_encoder", open_vocabulary.get("clip_model_path")),
        (
            "grounding_dino",
            Path(str(fallback.get("model_path"))) / "model.safetensors"
            if fallback.get("model_path")
            else None,
        ),
        ("sam2_video_segmentation", temporal_segmentation.get("checkpoint_path")),
        ("labpics_liquid_semantic", liquid_semantic.get("checkpoint_path")),
    ]
    return [
        (name, Path(str(path)).resolve())
        for name, path in configured
        if path is not None
    ]


def _certification_input_fingerprint(
    settings: dict[str, Any], repository_root: Path
) -> tuple[str, dict[str, Any]]:
    validation = settings.get("validation") or {}
    certification = validation.get("model_certification") or {}
    configured_truth = validation.get("key_event_ground_truth") or {}
    if not isinstance(configured_truth, dict):
        configured_truth = {}
    truth_inputs = []
    for experiment_id in sorted(
        key for key in configured_truth if str(key) != "default"
    ):
        truth, selection = load_dataset_scoped_json(
            configured_truth,
            str(experiment_id),
            repository_root=repository_root,
            artifact_label="关键事件真值",
        )
        path_value = selection.get("artifact_path")
        path = Path(str(path_value)) if path_value else None
        truth_inputs.append(
            {
                "experiment_id": str(experiment_id),
                "applicable": truth is not None,
                "path": str(path.resolve()) if path is not None else None,
                "sha256": _sha256(path) if path is not None and path.is_file() else None,
            }
        )
    box_inputs = []
    for configured_path in certification.get("box_evaluation_reports") or []:
        path = Path(str(configured_path))
        if not path.is_absolute():
            path = repository_root / path
        box_inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )
    inputs = {
        "targets": certification.get("targets") or {},
        "event_ground_truth": truth_inputs,
        "box_evaluation_reports": box_inputs,
    }
    fingerprint = hashlib.sha256(
        json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fingerprint, inputs


def summarize_certification_metrics(
    event_reports: Sequence[dict[str, Any]],
    box_reports: Sequence[dict[str, Any]],
    targets: dict[str, Any],
) -> dict[str, Any]:
    threshold = float(targets.get("event_temporal_iou", 0.50))
    selected = []
    for report in event_reports:
        item = next(
            (
                candidate
                for candidate in report.get("threshold_results") or []
                if abs(
                    float(candidate.get("temporal_iou_threshold") or 0.0)
                    - threshold
                )
                < 1e-9
            ),
            None,
        )
        if item is not None:
            selected.append(item)
    overall = _metric(
        sum(int(item.get("true_positives") or 0) for item in selected),
        sum(int(item.get("false_positives") or 0) for item in selected),
        sum(int(item.get("false_negatives") or 0) for item in selected),
    )
    action_counts: dict[str, dict[str, int]] = {}
    for item in selected:
        for class_report in item.get("per_class") or []:
            name = str(class_report.get("action_type") or "")
            totals = action_counts.setdefault(name, {"tp": 0, "fp": 0, "fn": 0})
            totals["tp"] += int(class_report.get("true_positives") or 0)
            totals["fp"] += int(class_report.get("false_positives") or 0)
            totals["fn"] += int(class_report.get("false_negatives") or 0)
    per_action = {
        name: {
            **_metric(values["tp"], values["fp"], values["fn"]),
            "ground_truth_event_count": values["tp"] + values["fn"],
        }
        for name, values in sorted(action_counts.items())
    }

    box_tp = box_fp = box_fn = box_images = 0
    evaluated_box_reports = []
    for report in box_reports:
        if report.get("status") != "completed":
            continue
        micro = report.get("micro") or {}
        box_tp += int(micro.get("true_positive") or micro.get("true_positives") or 0)
        box_fp += int(micro.get("false_positive") or micro.get("false_positives") or 0)
        box_fn += int(micro.get("false_negative") or micro.get("false_negatives") or 0)
        box_images += int(report.get("image_count") or 0)
        evaluated_box_reports.append(report)
    boxes = {
        **_metric(box_tp, box_fp, box_fn),
        "evaluated_report_count": len(evaluated_box_reports),
        "image_count": box_images,
        "ground_truth_instance_count": box_tp + box_fn,
    }

    required_actions = [
        str(item)
        for item in targets.get(
            "required_action_types", [item.value for item in ActionType]
        )
    ]
    failures: list[str] = []
    gt_events = overall["true_positives"] + overall["false_negatives"]
    if len(event_reports) < int(targets.get("minimum_event_dataset_count", 1)):
        failures.append("insufficient_event_dataset_count")
    if gt_events < int(targets.get("minimum_event_ground_truth_count", 1)):
        failures.append("insufficient_event_ground_truth_count")
    if overall["precision"] is None or overall["precision"] < float(
        targets.get("minimum_event_precision", 0.0)
    ):
        failures.append("event_precision_below_target")
    if overall["recall"] is None or overall["recall"] < float(
        targets.get("minimum_event_recall", 0.0)
    ):
        failures.append("event_recall_below_target")
    precision_ci = overall.get("precision_confidence_interval") or {}
    recall_ci = overall.get("recall_confidence_interval") or {}
    if float(precision_ci.get("lower") or 0.0) < float(
        targets.get("minimum_event_precision_ci_lower", 0.0)
    ):
        failures.append("event_precision_confidence_lower_below_target")
    if float(recall_ci.get("lower") or 0.0) < float(
        targets.get("minimum_event_recall_ci_lower", 0.0)
    ):
        failures.append("event_recall_confidence_lower_below_target")
    minimum_per_action = int(targets.get("minimum_ground_truth_per_action", 1))
    for action in required_actions:
        result = per_action.get(action)
        if result is None or result["ground_truth_event_count"] < minimum_per_action:
            failures.append(f"insufficient_action_ground_truth:{action}")
            continue
        if result["precision"] is None or result["precision"] < float(
            targets.get("minimum_per_action_precision", 0.0)
        ):
            failures.append(f"action_precision_below_target:{action}")
        if result["recall"] is None or result["recall"] < float(
            targets.get("minimum_per_action_recall", 0.0)
        ):
            failures.append(f"action_recall_below_target:{action}")
    if boxes["evaluated_report_count"] < int(
        targets.get("minimum_box_dataset_count", 1)
    ):
        failures.append("missing_or_insufficient_box_evaluation")
    if boxes["ground_truth_instance_count"] < int(
        targets.get("minimum_box_ground_truth_instances", 1)
    ):
        failures.append("insufficient_box_ground_truth_instances")
    if boxes["precision"] is None or boxes["precision"] < float(
        targets.get("minimum_box_precision", 0.0)
    ):
        failures.append("box_precision_below_target")
    if boxes["recall"] is None or boxes["recall"] < float(
        targets.get("minimum_box_recall", 0.0)
    ):
        failures.append("box_recall_below_target")
    return {
        "event_temporal_iou": threshold,
        "event_ground_truth_count": gt_events,
        "events": overall,
        "per_action": per_action,
        "participant_boxes_iou_0_5": boxes,
        "targets": targets,
        "failures": sorted(set(failures)),
        "passed": not failures,
    }


def build_model_quality_certification(
    settings: dict[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    validation = settings.get("validation") or {}
    certification = validation.get("model_certification") or {}
    targets = dict(certification.get("targets") or {})
    configured_truth = validation.get("key_event_ground_truth") or {}
    if not isinstance(configured_truth, dict):
        configured_truth = {}
    archive_root = Path(settings["storage"]["archive_root"]).resolve()
    event_reports: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    for experiment_id in sorted(
        key for key in configured_truth if str(key) != "default"
    ):
        truth, selection = load_dataset_scoped_json(
            configured_truth,
            str(experiment_id),
            repository_root=repository_root,
            artifact_label="关键事件真值",
        )
        archive = _find_formal_archive(archive_root, str(experiment_id))
        if truth is None or archive is None:
            datasets.append(
                {
                    "experiment_id": experiment_id,
                    "status": "not_evaluated",
                    "archive": str(archive) if archive else None,
                    "ground_truth_selection": selection,
                }
            )
            continue
        package_path = archive / "JSON-Config-Files" / "evidence_package.json"
        package = RunSummary.model_validate_json(
            package_path.read_text(encoding="utf-8-sig")
        )
        predictions = [
            event for event in package.events if event.key_frames or event.key_clips
        ]
        report = evaluate_key_event_recall(predictions, truth)
        report["experiment_id"] = experiment_id
        event_reports.append(report)
        datasets.append(
            {
                "experiment_id": experiment_id,
                "status": report.get("status"),
                "archive": str(archive),
                "archive_package_sha256": _sha256(package_path),
                "ground_truth_path": selection.get("artifact_path"),
                "ground_truth_id": report.get("ground_truth_id"),
                "ground_truth_event_count": report.get("ground_truth_event_count"),
                "annotation_coverage": report.get("annotation_coverage"),
            }
        )

    box_reports: list[dict[str, Any]] = []
    box_inputs: list[dict[str, Any]] = []
    for configured_path in certification.get("box_evaluation_reports") or []:
        path = Path(str(configured_path))
        if not path.is_absolute():
            path = repository_root / path
        if not path.is_file():
            box_inputs.append({"path": str(path), "status": "missing"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        box_reports.append(payload)
        box_inputs.append(
            {
                "path": str(path),
                "status": payload.get("status"),
                "sha256": _sha256(path),
            }
        )

    metrics = summarize_certification_metrics(event_reports, box_reports, targets)
    certification_input_fingerprint, certification_inputs = (
        _certification_input_fingerprint(settings, repository_root)
    )
    artifacts = []
    missing_artifacts = []
    for name, path in _artifact_paths(settings):
        if not path.is_file():
            missing_artifacts.append({"name": name, "path": str(path)})
            continue
        artifacts.append(
            {
                "name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    failures = list(metrics["failures"])
    if missing_artifacts:
        failures.append("required_model_artifact_missing")
    status = "certified" if not failures else "not_certified"
    return {
        "schema_version": CERTIFICATION_SCHEMA,
        "status": status,
        "passed": status == "certified",
        "generated_at": datetime.now().astimezone().isoformat(),
        "policy": (
            "held-out reviewed events and participant-only boxes; production "
            "fails closed when certification is absent, stale, or below target"
        ),
        "datasets": datasets,
        "box_evaluation_inputs": box_inputs,
        "metrics": metrics,
        "model_artifacts": artifacts,
        "missing_model_artifacts": missing_artifacts,
        "certification_input_fingerprint": certification_input_fingerprint,
        "certification_inputs": certification_inputs,
        "failures": sorted(set(failures)),
    }


def build_model_certification_readiness(
    settings: dict[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Audit certification inputs without inspecting production archives.

    This command is safe while NAS storage is unavailable: it reads only the
    repository-scoped truth declarations, explicitly configured local box
    reports, and the local model artifacts named by the profile.
    """

    validation = settings.get("validation") or {}
    certification = validation.get("model_certification") or {}
    targets = dict(certification.get("targets") or {})
    configured_truth = validation.get("key_event_ground_truth") or {}
    if not isinstance(configured_truth, dict):
        configured_truth = {}

    event_datasets = []
    action_counts: dict[str, int] = {}
    eligible_event_count = 0
    for experiment_id in sorted(
        key for key in configured_truth if str(key) != "default"
    ):
        truth, selection = load_dataset_scoped_json(
            configured_truth,
            str(experiment_id),
            repository_root=repository_root,
            artifact_label="关键事件真值",
        )
        if truth is None:
            event_datasets.append(
                {
                    "experiment_id": str(experiment_id),
                    "status": "not_applicable",
                    "selection": selection,
                }
            )
            continue
        empty_prediction_report = evaluate_key_event_recall([], truth)
        count = int(empty_prediction_report.get("ground_truth_event_count") or 0)
        eligible_event_count += count
        per_action = {
            str(item.get("action_type")): int(item.get("false_negatives") or 0)
            for item in (
                (empty_prediction_report.get("threshold_results") or [{}])[0].get(
                    "per_class"
                )
                or []
            )
        }
        for action, action_count in per_action.items():
            action_counts[action] = action_counts.get(action, 0) + action_count
        event_datasets.append(
            {
                "experiment_id": str(experiment_id),
                "status": "truth_available",
                "ground_truth_path": selection.get("artifact_path"),
                "ground_truth_id": empty_prediction_report.get("ground_truth_id"),
                "eligible_event_count": count,
                "per_action": per_action,
                "excluded_uncertain_count": len(
                    empty_prediction_report.get(
                        "excluded_uncertain_ground_truth_event_ids"
                    )
                    or []
                ),
                "excluded_rejected_count": len(
                    empty_prediction_report.get(
                        "excluded_rejected_ground_truth_event_ids"
                    )
                    or []
                ),
            }
        )

    box_inputs = []
    box_report_count = 0
    box_ground_truth_instances = 0
    for configured_path in certification.get("box_evaluation_reports") or []:
        path = Path(str(configured_path))
        if not path.is_absolute():
            path = repository_root / path
        if not path.is_file():
            box_inputs.append({"path": str(path), "status": "missing"})
            continue
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        micro = report.get("micro") or {}
        gt_count = int(micro.get("true_positive") or 0) + int(
            micro.get("false_negative") or 0
        )
        if report.get("status") == "completed":
            box_report_count += 1
            box_ground_truth_instances += gt_count
        box_inputs.append(
            {
                "path": str(path.resolve()),
                "status": report.get("status"),
                "sha256": _sha256(path),
                "ground_truth_instance_count": gt_count,
            }
        )

    model_artifacts = []
    missing_model_artifacts = []
    for name, path in _artifact_paths(settings):
        if not path.is_file():
            missing_model_artifacts.append({"name": name, "path": str(path)})
            continue
        model_artifacts.append(
            {
                "name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    required_actions = [
        str(item)
        for item in targets.get(
            "required_action_types", [item.value for item in ActionType]
        )
    ]
    deficits = []
    minimum_event_datasets = int(targets.get("minimum_event_dataset_count", 1))
    available_event_datasets = sum(
        item["status"] == "truth_available" for item in event_datasets
    )
    if available_event_datasets < minimum_event_datasets:
        deficits.append(
            {
                "requirement": "event_dataset_count",
                "current": available_event_datasets,
                "target": minimum_event_datasets,
                "missing": minimum_event_datasets - available_event_datasets,
            }
        )
    minimum_events = int(targets.get("minimum_event_ground_truth_count", 1))
    if eligible_event_count < minimum_events:
        deficits.append(
            {
                "requirement": "event_ground_truth_count",
                "current": eligible_event_count,
                "target": minimum_events,
                "missing": minimum_events - eligible_event_count,
            }
        )
    minimum_per_action = int(targets.get("minimum_ground_truth_per_action", 1))
    for action in required_actions:
        current = int(action_counts.get(action, 0))
        if current < minimum_per_action:
            deficits.append(
                {
                    "requirement": f"action_ground_truth:{action}",
                    "current": current,
                    "target": minimum_per_action,
                    "missing": minimum_per_action - current,
                }
            )
    minimum_box_datasets = int(targets.get("minimum_box_dataset_count", 1))
    if box_report_count < minimum_box_datasets:
        deficits.append(
            {
                "requirement": "box_evaluation_dataset_count",
                "current": box_report_count,
                "target": minimum_box_datasets,
                "missing": minimum_box_datasets - box_report_count,
            }
        )
    minimum_box_instances = int(
        targets.get("minimum_box_ground_truth_instances", 1)
    )
    if box_ground_truth_instances < minimum_box_instances:
        deficits.append(
            {
                "requirement": "box_ground_truth_instance_count",
                "current": box_ground_truth_instances,
                "target": minimum_box_instances,
                "missing": minimum_box_instances - box_ground_truth_instances,
            }
        )
    if missing_model_artifacts:
        deficits.append(
            {
                "requirement": "required_model_artifacts",
                "current": len(model_artifacts),
                "target": len(model_artifacts) + len(missing_model_artifacts),
                "missing": len(missing_model_artifacts),
            }
        )

    configured_certification_path = str(certification.get("path") or "").strip()
    certification_path = Path(configured_certification_path)
    certification_present = bool(configured_certification_path) and certification_path.is_file()
    certification_audit: dict[str, Any]
    if not certification_present:
        certification_audit = {"status": "missing"}
    else:
        try:
            certification_audit = audit_production_model_certification(settings)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            certification_audit = {
                "status": "invalid_or_stale",
                "error": f"{type(exc).__name__}: {exc}",
            }
    production_certified = certification_audit.get("status") == "certified"
    return {
        "schema_version": READINESS_SCHEMA,
        "status": "ready_for_certification_run" if not deficits else "inputs_incomplete",
        "ready_for_certification_run": not deficits,
        "production_certified": production_certified,
        "production_certification_path": str(certification_path),
        "production_certification_audit": certification_audit,
        "generated_at": datetime.now().astimezone().isoformat(),
        "event_truth": {
            "dataset_count": available_event_datasets,
            "eligible_event_count": eligible_event_count,
            "per_action": dict(sorted(action_counts.items())),
            "datasets": event_datasets,
        },
        "box_truth": {
            "evaluation_report_count": box_report_count,
            "ground_truth_instance_count": box_ground_truth_instances,
            "inputs": box_inputs,
        },
        "model_artifacts": model_artifacts,
        "missing_model_artifacts": missing_model_artifacts,
        "targets": targets,
        "deficits": deficits,
        "nas_accessed": False,
        "ark_calls": 0,
        "token_usage": 0,
        "policy": (
            "Readiness is not certification. Precision and recall are reported "
            "only after held-out predictions are evaluated against independent truth."
        ),
    }


def audit_production_model_certification(settings: dict[str, Any]) -> dict[str, Any]:
    validation = settings.get("validation") or {}
    configured = validation.get("model_certification") or {}
    if not configured.get("required_for_formal_production", False):
        return {"required": False, "status": "not_required_by_profile"}
    path = Path(str(configured.get("path") or ""))
    if not path.is_file():
        raise RuntimeError(f"Production model certification is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != CERTIFICATION_SCHEMA:
        raise RuntimeError("Production model certification schema mismatch")
    if payload.get("status") != "certified" or payload.get("passed") is not True:
        raise RuntimeError(
            "Production model certification has not passed; failures="
            f"{payload.get('failures') or []}"
        )
    current_input_fingerprint, _current_inputs = _certification_input_fingerprint(
        settings, Path(__file__).resolve().parents[2]
    )
    if payload.get("certification_input_fingerprint") != current_input_fingerprint:
        raise RuntimeError(
            "Production model certification inputs are stale: truth, box reports, "
            "or target policy changed"
        )
    certified_hashes = {
        str(item.get("name")): str(item.get("sha256"))
        for item in payload.get("model_artifacts") or []
    }
    stale = []
    for name, artifact in _artifact_paths(settings):
        if not artifact.is_file() or certified_hashes.get(name) != _sha256(artifact):
            stale.append(name)
    if stale:
        raise RuntimeError(f"Production model certification is stale: {stale}")
    metrics = payload.get("metrics") or {}
    return {
        "required": True,
        "status": "certified",
        "path": str(path),
        "sha256": _sha256(path),
        "event_precision": (metrics.get("events") or {}).get("precision"),
        "event_recall": (metrics.get("events") or {}).get("recall"),
        "box_precision": (metrics.get("participant_boxes_iou_0_5") or {}).get(
            "precision"
        ),
        "box_recall": (metrics.get("participant_boxes_iou_0_5") or {}).get(
            "recall"
        ),
    }
