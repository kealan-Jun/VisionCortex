from __future__ import annotations

import json
import copy
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .liquid_semantic import predict_liquid_masks
from .telemetry import ResourceMonitor


LABPICS_EVALUATION_SCHEMA = "visioncortex-labpics-public-annotated-benchmark/1"


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        return np.zeros(shape, dtype=bool)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read LabPics annotation: {path}")
    if mask.shape != shape:
        mask = cv2.resize(
            mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask > 0


def _counts(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    prediction = prediction & valid
    truth = truth & valid
    return {
        "true_positive": int((prediction & truth).sum()),
        "false_positive": int((prediction & ~truth & valid).sum()),
        "false_negative": int((~prediction & truth & valid).sum()),
        "true_negative": int((~prediction & ~truth & valid).sum()),
    }


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    iou = tp / max(1, tp + fp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        **counts,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "iou": round(iou, 6),
        "f1": round(f1, 6),
    }


def _selected_examples(test_root: Path, sample_count: int) -> list[Path]:
    examples = sorted(
        path
        for path in test_root.iterdir()
        if path.is_dir() and (path / "Image.jpg").is_file()
    )
    if not examples:
        raise RuntimeError(f"No LabPics test examples found: {test_root}")
    count = min(sample_count, len(examples))
    indices = np.linspace(0, len(examples) - 1, num=count, dtype=int)
    return [examples[int(index)] for index in sorted(set(indices.tolist()))]


def _truth_masks(
    example: Path, shape: tuple[int, int]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    semantic = example / "SemanticMaps" / "FullImage"
    liquid_family = np.zeros(shape, dtype=bool)
    for name in ("Liquid", "Suspension", "Foam", "Gel"):
        liquid_family |= _read_mask(semantic / f"{name}.png", shape)
    solid_family = np.zeros(shape, dtype=bool)
    for name in ("Solid", "Powder", "Granular"):
        solid_family |= _read_mask(semantic / f"{name}.png", shape)
    truth = {
        "Vessel": _read_mask(semantic / "Vessel.png", shape),
        "Filled": _read_mask(semantic / "Filled.png", shape),
        "Liquid": _read_mask(semantic / "Liquid.png", shape),
        "Liquid-Family": liquid_family,
        "Suspension": _read_mask(semantic / "Suspension.png", shape),
        "Foam": _read_mask(semantic / "Foam.png", shape),
        "Gel": _read_mask(semantic / "Gel.png", shape),
        "Solid-Family": solid_family,
    }
    ignore = _read_mask(example / "Ignore.png", shape)
    return truth, ~ignore


def _prediction_masks(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    liquid_family = np.zeros_like(masks["Vessel"], dtype=bool)
    for name in ("Liquid GENERAL", "Liquid Suspension", "Foam", "Gel"):
        liquid_family |= masks[name]
    solid_family = np.zeros_like(masks["Vessel"], dtype=bool)
    for name in ("Solid GENERAL", "Granular", "Powder", "Solid Bulk"):
        solid_family |= masks[name]
    return {
        "Vessel": masks["Vessel"],
        "Filled": masks["Filled"],
        "Liquid": masks["Liquid GENERAL"],
        "Liquid-Family": liquid_family,
        "Suspension": masks["Liquid Suspension"],
        "Foam": masks["Foam"],
        "Gel": masks["Gel"],
        "Solid-Family": solid_family,
    }


def _audit_panel(
    image: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    destination: Path,
    label: str,
) -> None:
    shape = prediction.shape
    resized = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    pred = resized.copy()
    pred[prediction] = (
        (pred[prediction].astype(np.uint16) + np.array([255, 80, 20])) // 2
    ).astype(np.uint8)
    gt = resized.copy()
    gt[truth] = (
        (gt[truth].astype(np.uint16) + np.array([30, 220, 80])) // 2
    ).astype(np.uint8)
    panel = np.concatenate((pred, gt), axis=1)
    cv2.putText(
        panel,
        f"{label}: prediction (left) | public human annotation (right)",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), panel):
        raise RuntimeError(f"Cannot write LabPics audit panel: {destination}")


def evaluate_labpics_heldout(
    dataset_root: Path,
    output: Path,
    config: dict[str, Any],
    *,
    sample_count: int = 40,
    split: str = "Test",
) -> Path:
    """Evaluate the pinned model against a deterministic annotated split."""

    if sample_count < 1 or sample_count > 2_000:
        raise ValueError("LabPics benchmark sample count must be between 1 and 2000")
    dataset_root = dataset_root.resolve()
    if split not in {"Train", "Test"}:
        raise ValueError("LabPics evaluation split must be Train or Test")
    test_root = dataset_root / "LabPics Chemistry" / split
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "labpics-public-evaluation.json"
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            return receipt_path
        raise RuntimeError(f"Existing LabPics evaluation is incomplete: {receipt_path}")

    examples = _selected_examples(test_root, sample_count)
    totals: dict[str, dict[str, int]] = {}
    rows = []
    monitor = ResourceMonitor(output / "resource-telemetry.json", interval_seconds=0.25)
    monitor.set_stage("labpics_public_annotated_benchmark_inference")
    monitor.start()
    started = time.perf_counter()
    try:
        for index, example in enumerate(examples):
            image = cv2.imread(str(example / "Image.jpg"))
            if image is None:
                raise RuntimeError(f"Cannot read LabPics image: {example / 'Image.jpg'}")
            predicted_raw, inference = predict_liquid_masks(image, config)
            predicted = _prediction_masks(predicted_raw)
            shape = next(iter(predicted.values())).shape
            truth, valid = _truth_masks(example, shape)
            per_class = {}
            for class_name in predicted:
                counts = _counts(predicted[class_name], truth[class_name], valid)
                accumulator = totals.setdefault(
                    class_name,
                    {
                        "true_positive": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "true_negative": 0,
                    },
                )
                for name, value in counts.items():
                    accumulator[name] += value
                per_class[class_name] = _metrics(counts)
            rows.append(
                {
                    "example_id": example.name,
                    "inference_seconds": inference["inference_seconds"],
                    "metrics": per_class,
                }
            )
            if index < 8:
                _audit_panel(
                    image,
                    predicted["Liquid-Family"],
                    truth["Liquid-Family"],
                    output / "audit-panels" / f"{example.name}-liquid-family.jpg",
                    example.name,
                )
    finally:
        monitor.stop()
    elapsed = time.perf_counter() - started
    summary = {name: _metrics(values) for name, values in totals.items()}
    payload = {
        "schema_version": LABPICS_EVALUATION_SCHEMA,
        "status": "completed",
        "truth_status": "public_human_annotations",
        "production_domain_certification": False,
        "generalization_claim": False,
        "checkpoint_split_provenance": (
            "Published training scripts separate Train and Test, but the exact "
            "packaged checkpoint lineage was not independently reproduced."
        ),
        "dataset": {
            "name": f"LabPics Chemistry V2 {split}",
            "source_record": "https://zenodo.org/records/4736111",
            "license": "MIT",
            "split": split,
            "selection": "deterministic evenly spaced across sorted split examples",
            "available_example_count": len(
                [path for path in test_root.iterdir() if path.is_dir()]
            ),
            "evaluated_example_count": len(examples),
        },
        "model": {
            "name": "LabPics PSPNet ResNet101 semantic materials",
            "source_record": "https://zenodo.org/records/3697767",
            "license": "CC-BY-4.0",
            "threshold": float(
                config["models"]["liquid_semantic_sidecar"].get(
                    "mask_threshold", 0.5
                )
            ),
        },
        "elapsed_seconds": round(elapsed, 6),
        "mean_images_per_second": round(len(examples) / max(elapsed, 1e-9), 6),
        "summary": summary,
        "examples": rows,
        "resource_telemetry": str(output / "resource-telemetry.json"),
        "audit_panel_count": min(8, len(examples)),
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt_path


def calibrate_labpics_threshold(
    dataset_root: Path,
    output: Path,
    config: dict[str, Any],
    *,
    thresholds: list[float],
    sample_count: int = 32,
    minimum_precision: float = 0.90,
) -> Path:
    """Select a liquid mask threshold on Train without inspecting Test labels."""

    unique = sorted({round(float(value), 4) for value in thresholds})
    if not unique or any(value <= 0 or value >= 1 for value in unique):
        raise ValueError("Calibration thresholds must be strictly between 0 and 1")
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be in (0, 1]")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "labpics-threshold-calibration.json"
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            return receipt_path
        raise RuntimeError(f"Existing calibration is incomplete: {receipt_path}")

    rows = []
    for threshold in unique:
        candidate = copy.deepcopy(config)
        candidate["models"]["liquid_semantic_sidecar"][
            "mask_threshold"
        ] = threshold
        evaluation_path = evaluate_labpics_heldout(
            dataset_root,
            output / f"threshold-{threshold:.4f}",
            candidate,
            sample_count=sample_count,
            split="Train",
        )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        metrics = evaluation["summary"]["Liquid-Family"]
        rows.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "iou": metrics["iou"],
                "f1": metrics["f1"],
                "evaluation_path": str(evaluation_path),
            }
        )
    eligible = [row for row in rows if row["precision"] >= minimum_precision]
    pool = eligible or rows
    selected = max(
        pool,
        key=lambda row: (
            5 * row["precision"] * row["recall"]
            / max(1e-12, 4 * row["precision"] + row["recall"]),
            row["f1"],
            row["threshold"],
        ),
    )
    payload = {
        "schema_version": "visioncortex-labpics-threshold-calibration/1",
        "status": "completed",
        "truth_status": "public_train_human_annotations",
        "test_split_observed": False,
        "objective": "maximum liquid-family F2 subject to minimum precision",
        "minimum_precision": minimum_precision,
        "sample_count": sample_count,
        "candidates": rows,
        "selected": selected,
        "production_domain_certification": False,
        "ark_calls": 0,
        "token_usage": 0,
        "source_copy_bytes": 0,
        "nas_accessed": False,
    }
    receipt_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt_path
