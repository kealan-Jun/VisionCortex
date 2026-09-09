"""Bounded local detector experiments on explicitly reviewed project annotations.

This entry point preserves the project review status. It does not use or weaken
the independent human-ground-truth training and promotion contracts.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .yolo_evaluation import evaluate_yolo_predictions
from .project_training_evidence import (
    TrainingEvidence, durable_bytes, durable_json, sync_directory, verify_training_evidence,
)

_CODE_AT_IMPORT = {
    str(path): path.read_bytes()
    for path in [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("yolo_evaluation.py"),
        Path(__file__).resolve().with_name("project_ignore_training.py"),
        Path(__file__).resolve().parent / "training_runtime" / "__init__.py",
        Path(__file__).resolve().parent / "training_runtime" / "ignore.py",
        Path(__file__).resolve().with_name("project_sampling.py"),
        Path(__file__).resolve().with_name("project_training_evidence.py"),
        Path(__file__).resolve().with_name("project_augmentation.py"),
        Path(__file__).resolve().with_name("project_branch_loss.py"),
    ]
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    durable_json(path, value)


class OptimizationTrace:
    """Record actual optimizer steps, independently of batches and epochs."""

    def __init__(self, path: Path):
        self.path = path
        self.steps = 0
        self.handle = None
        self.path.touch(exist_ok=False)

    def install(self, trainer: Any) -> None:
        if self.handle is not None:
            raise ValueError("Optimization trace is already installed")

        def record(optimizer, args, kwargs):
            self.steps += 1
            row = dict(
                step=self.steps, epoch=int(trainer.epoch),
                accumulation=int(trainer.accumulate),
                learning_rates=[float(g["lr"]) for g in optimizer.param_groups],
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

        self.handle = trainer.optimizer.register_step_post_hook(record)

    def finish(self) -> dict:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if not self.steps:
            raise ValueError("Training did not record any optimizer steps")
        usage = optimization_trace_usage(self.path)
        if usage["optimizer_steps"] != self.steps:
            raise ValueError("Optimization trace differs from observed steps")
        return usage


def optimization_trace_usage(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("Empty optimization trace")
    previous_epoch = -1
    for index, row in enumerate(rows, 1):
        if (
            row["step"] != index or row["epoch"] < previous_epoch
            or row["accumulation"] < 1 or not row["learning_rates"]
            or any(not math.isfinite(lr) or lr < 0 for lr in row["learning_rates"])
        ):
            raise ValueError("Invalid optimization trace")
        previous_epoch = row["epoch"]
    return dict(
        path=str(path), sha256=sha256(path), optimizer_steps=len(rows),
        measurement="optimizer_step_post_hook_not_epochs_or_image_visits",
    )


def transfer_class_outputs(
    source: dict,
    target: dict,
    old_count: int,
    new_count: int,
    *,
    migrate_legacy_tip_box: bool = False,
) -> list[str]:
    """Preserve only verified shared class output rows when adding new classes."""
    import torch

    if not 0 < old_count < new_count:
        raise ValueError("Class output transfer requires an ontology extension")
    if migrate_legacy_tip_box and (old_count, new_count) != (21, 23):
        raise ValueError(
            "Legacy tip-box migration requires the verified 21-to-23 class ontology"
        )
    copied = []
    with torch.no_grad():
        for key, value in target.items():
            previous = source.get(key)
            if previous is None or not (".cv3." in key or ".one2one_cv3." in key):
                continue
            if (
                value.ndim in {1, 4}
                and previous.ndim == value.ndim
                and previous.shape[0] == old_count
                and value.shape[0] == new_count
                and previous.shape[1:] == value.shape[1:]
            ):
                initial_tip = value[11].clone() if migrate_legacy_tip_box else None
                value[:old_count].copy_(
                    previous.to(device=value.device, dtype=value.dtype)
                )
                if migrate_legacy_tip_box:
                    value[22].copy_(
                        previous[11].to(device=value.device, dtype=value.dtype)
                    )
                    value[11].copy_(initial_tip)
                copied.append(key)
    if (
        not copied
        or not any(k.endswith(".weight") for k in copied)
        or not any(k.endswith(".bias") for k in copied)
    ):
        raise ValueError("No compatible class output weights and biases to preserve")
    return copied


def restrict_to_class_outputs(model: Any, class_count: int) -> list[str]:
    """Freeze a verified YOLO classification probe without changing forward mode."""
    import torch

    names = []
    for name, module in model.named_modules():
        if (
            re.search(r"(?:^|\.)(?:one2one_)?cv3\.\d+\.2$", name)
            and isinstance(module, torch.nn.Conv2d)
            and module.out_channels == class_count
            and module.kernel_size == (1, 1)
            and module.bias is not None
        ):
            names.extend([name + ".weight", name + ".bias"])
    if not names:
        raise ValueError("No verified classification output convolutions found")
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    freeze_normalization_statistics(model)
    return sorted(names)


def freeze_normalization_statistics(model: Any) -> None:
    # The trainer calls model.train() at each epoch. Apply again before every
    # training batch; setting the whole detector to eval would change its output
    # contract and break the training loss.
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def frozen_state_sha256(model: Any, mutable_names: list[str]) -> str:
    import torch

    h = hashlib.sha256()
    mutable = set(mutable_names)
    for name, value in sorted(model.state_dict().items()):
        if name in mutable:
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(f"{name}\0{tensor.dtype}\0{list(tensor.shape)}\0".encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def synchronize_frozen_ema(model: Any, ema: Any, mutable_names: list[str]) -> None:
    """Keep fixed features exact instead of accumulating EMA rounding drift."""
    import torch

    mutable = set(mutable_names)
    state = model.state_dict()
    with torch.no_grad():
        for name, target in ema.state_dict().items():
            if name not in mutable:
                target.copy_(state[name].to(device=target.device, dtype=target.dtype))


def validate_export(
    root: Path, receipt_sha: str, role: str, *, supervision: str = "complete"
) -> tuple[dict, list[dict]]:
    if supervision not in {"complete", "outside_ignore"}:
        raise ValueError("Unknown supervision contract")
    if role not in {"first_person", "third_person"}:
        raise ValueError(
            "A detector experiment requires an explicit first/third person role"
        )
    if sha256(root / "receipt.json") != receipt_sha:
        raise ValueError("Dataset receipt SHA256 mismatch")
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    augmented_partial = (
        supervision == "outside_ignore"
        and receipt.get("schema_version") == "annotation-workbench-partial-training-export/2"
    )
    if (
        receipt.get("schema_version") != (
            "annotation-workbench-yolo-export/1" if supervision == "complete"
            else ("annotation-workbench-partial-training-export/2" if augmented_partial
                  else "annotation-workbench-partial-training-export/1")
        )
        or receipt.get("truth_status") != "project_annotations"
        or receipt.get("independent_ground_truth") is not False
        or receipt.get("check", {}).get("export_ready") is not True
    ):
        raise ValueError("Expected a passing project-annotation export receipt")
    if supervision == "outside_ignore" and (
        receipt.get("supervision") != dict(
            mode="outside_ignore", complete_annotations=False,
            required_trainer_contract="ignore_negative_classification_at_anchor_centers/1",
            known_positive_terms="retained", validation_and_test="complete_annotations_only",
            augmentation="offline_affine_ignore_regions/1" if augmented_partial else "disabled",
            production_ready=False,
        )
        or receipt["check"].get("supervision") != "outside_ignore"
        or ("augmentation" in receipt) != augmented_partial
    ):
        raise ValueError("Partial-supervision receipt requires the exact ignore contract")
    for relative, expected in receipt["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha256(root / path) != expected:
            raise ValueError(f"Dataset file identity mismatch: {relative}")
    for part in ["images", "labels"]:
        expected = {
            name for name in receipt["files"] if name.startswith(f"{role}/{part}/")
        }
        actual = {
            str(path.relative_to(root))
            for path in (root / role / part).rglob("*")
            if path.is_file() and (part == "images" or path.suffix == ".txt")
        }
        if not expected or actual != expected:
            raise ValueError(f"Unregistered or missing {part} in training directory")
    snapshot = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    indexed = {r["id"]: r for r in snapshot["images"]}
    rows = []
    expected_ignores = {}
    seen = set()
    groups: dict[str, set[str]] = {}
    for pin in receipt["records"]:
        iid = pin["image_id"]
        if iid in seen:
            raise ValueError("Duplicate exported image identity")
        seen.add(iid)
        row = indexed[iid]
        a, source = row["annotation"], row["source"]
        complete = (
            a["status"] == "reviewed"
            and a["completeness"] == "all_visible_instances"
            and not a["ignore_regions"]
        )
        partial_train = (
            supervision == "outside_ignore" and a["split"] == "train"
            and a["status"] == "reviewed_partial"
            and a["completeness"] == "all_visible_outside_ignore"
            and bool(a["ignore_regions"])
            and bool(a.get("review_notes", "").strip())
        )
        if (
            pin["revision"] != row["revision"]
            or pin["annotation_sha256"] != canonical_sha(a)
            or pin["source_sha256"] != source["sha256"]
            or sha256(Path(source["path"])) != source["sha256"]
            or not (complete or partial_train)
            or a["truth_status"] != "project_annotations"
            or a["independent_ground_truth"] is not False
            or pin["role"] != a["role"]
            or pin["split"] != a["split"]
            or a["split"] not in {"train", "val", "test"}
        ):
            raise ValueError(
                f"Unreviewed, changed or incompatible project record: {iid}"
            )
        if supervision == "outside_ignore":
            relative = f"{a['role']}/images/{a['split']}/{iid}{Path(source['path']).suffix.lower()}"
            if receipt["files"].get(relative) != source["sha256"]:
                raise ValueError("Missing pinned image for partial-supervision metadata")
            expected_ignores[relative] = dict(
                image_id=iid, source_size=[source["width"], source["height"]],
                annotation_sha256=canonical_sha(a),
                xyxy_px=[r["xyxy_px"] for r in a["ignore_regions"]],
            )
        groups.setdefault(source["source_group"], set()).add(a["split"])
        if a["role"] == role:
            rows.append(row)
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Source group crosses partitions")
    if supervision == "outside_ignore":
        if augmented_partial:
            from .project_augmentation import validate_partial_augmentation

            expected_ignores.update(validate_partial_augmentation(
                root, receipt, {iid: indexed[iid] for iid in seen}, canonical_sha,
            ))
        if {p for p in receipt["files"] if "/images/" in p} != set(expected_ignores):
            raise ValueError("Partial training images differ from registered originals and derivatives")
        if "ignore-regions.json" not in receipt["files"] or json.loads(
            (root / "ignore-regions.json").read_text(encoding="utf-8")
        ) != expected_ignores:
            raise ValueError("Ignore metadata differs from frozen reviewed annotations")
    if not {"train", "val"}.issubset({r["annotation"]["split"] for r in rows}):
        raise ValueError("Role needs both train and validation images")
    return receipt, rows


def training_sampling_sources(root: Path, role: str, receipt: dict, rows: list[dict]) -> dict:
    training = {r["id"]: r for r in rows if r["annotation"]["split"] == "train"}
    # This function consumes the receipt only after validate_export has verified v2.
    derived = {
        d["id"]: d for d in receipt.get("augmentation", {}).get("records", [])
        if receipt.get("schema_version") == "annotation-workbench-partial-training-export/2"
        and d["role"] == role and d["split"] == "train"
    }
    sources = {}
    for relative in receipt["files"]:
        if not relative.startswith(f"{role}/images/train/"):
            continue
        row = training.get(Path(relative).stem)
        if row is None:
            d = derived.get(Path(relative).stem)
            if d is None or relative != d["image_file"] or d["source_image_id"] not in training:
                raise ValueError("Sampling cannot include unregistered or derived training images")
            sources[str((root / relative).absolute())] = dict(
                image_id=d["id"], source_group=d["source_group"], revision=d["source_revision"],
                source_sha256=d["image_sha256"], annotation_sha256=d["derived_annotation_sha256"],
                parent_image_id=d["source_image_id"], parent_source_sha256=d["source_sha256"],
                counts_as_new_source=False,
            )
            continue
        sources[str((root / relative).absolute())] = dict(
            image_id=row["id"], source_group=row["source"]["source_group"],
            revision=row["revision"], source_sha256=row["source"]["sha256"],
            annotation_sha256=canonical_sha(row["annotation"]),
        )
    if (len(sources) != len(training) + len(derived)
            or {r["image_id"] for r in sources.values()} != set(training) | set(derived)):
        raise ValueError("Sampling membership differs from the frozen training images")
    return sources


def project_evaluation(
    predictions: list[dict],
    rows: list[dict],
    names: list[str],
    *,
    confidence: float = 0.25,
) -> dict:
    images, annotations = [], []
    for index, row in enumerate(rows):
        a, source = row["annotation"], row["source"]
        images.append(
            dict(
                image_id=row["id"],
                event_id=source["source_group"],
                role=a["role"],
                frame_index=index,
            )
        )
        for box in a["boxes"]:
            annotations.append(
                dict(
                    image_id=row["id"],
                    annotation_id=row["id"] + ":" + box["id"],
                    class_name=names[box["class_id"]],
                    xyxy=box["xyxy_px"],
                )
            )
    truth = dict(
        schema_version="visioncortex-yolo-project-annotations/1",
        dataset_id="frozen-project-cohort",
        images=images,
        annotations=annotations,
    )
    report = evaluate_yolo_predictions(
        predictions, truth, confidence_threshold=confidence
    )
    report.update(
        schema_version="visioncortex-yolo-project-evaluation/1",
        truth_status="project_annotations",
        independent_ground_truth=False,
        production_ready=False,
        metric_scope="internal_project_comparison_only",
    )
    report["insufficient_positive_classes"] = [
        name
        for name in names
        if report["per_class"].get(name, {}).get("gt_count", 0) < 20
    ]
    report["crop_images"] = [
        r["id"]
        for r in rows
        if r["source"].get("provenance", {}).get("source_kind") == "derived_crop"
        or r["id"].startswith("awcrop-")
    ]
    return report


def run_experiment(
    root: Path,
    receipt_sha: str,
    role: str,
    model_path: Path,
    model_sha: str,
    output: Path,
    *,
    epochs: int = 80,
    image_size: int = 960,
    batch: int = 4,
    max_seconds: int = 3600,
    preserve_class_head: bool = False,
    learning_rate: float = 0.001,
    migrate_legacy_tip_box: bool = False,
    training_scope: str = "detector",
    supervision: str = "complete",
    sampling_policy: str = "image_uniform",
    nominal_batch: int = 64,
    warmup_epochs: float = 3.0,
    warmup_bias_lr: float = 0.1,
    patience: int = 20,
    freeze_layers: int = 10,
    trace_branch_loss: bool = False,
    branch_loss_policy: str = "native",
) -> dict:
    root, model_path, output = root.resolve(), model_path.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(
            "Experiment directory exists; preserve it and select a new run directory"
        )
    if (
        not 1 <= epochs <= 500
        or image_size < 64
        or image_size % 32
        or not 1 <= batch <= 32
        or not 1 <= max_seconds <= 86400
        or not 0.000001 <= learning_rate <= 0.1
        or training_scope not in {"detector", "class_outputs"}
        or sampling_policy not in {"image_uniform", "source_balanced"}
        or type(nominal_batch) is not int or not 1 <= nominal_batch <= 1024
        or not math.isfinite(warmup_epochs) or not 0 <= warmup_epochs <= 500
        or not math.isfinite(warmup_bias_lr) or not 0 <= warmup_bias_lr <= 0.1
        or type(patience) is not int or not 0 <= patience <= 500
        or type(freeze_layers) is not int or not 0 <= freeze_layers <= 10
        or type(trace_branch_loss) is not bool
        or branch_loss_policy not in ("native", "one2many")
    ):
        raise ValueError("Invalid bounded experiment settings")
    if training_scope == "class_outputs" and freeze_layers != 10:
        raise ValueError("Class-output probes use their own fixed training scope")
    if trace_branch_loss and supervision != "outside_ignore":
        raise ValueError("Branch-loss tracing requires explicit outside_ignore supervision")
    if branch_loss_policy != "native" and (
        not trace_branch_loss or training_scope != "detector" or patience != 0
    ):
        raise ValueError("One2many objective requires branch tracing, detector scope and patience=0")
    if sampling_policy == "source_balanced" and supervision != "outside_ignore":
        raise ValueError("Source balancing requires verified outside_ignore supervision")
    receipt, rows = validate_export(root, receipt_sha, role, supervision=supervision)
    sampling_sources = (
        training_sampling_sources(root, role, receipt, rows)
        if supervision == "outside_ignore" else None
    )
    if migrate_legacy_tip_box and not preserve_class_head:
        raise ValueError("Legacy tip-box migration requires class-head preservation")
    if sha256(model_path) != model_sha:
        raise ValueError("Base model SHA256 mismatch")
    output.mkdir(parents=True)
    sync_directory(output.parent)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics-settings")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import YOLO
    import torch

    names = [item["name"] for item in receipt["project"]["classes"]]
    if migrate_legacy_tip_box and (
        len(names) != 23 or names[11] != "spearhead" or names[22] != "pipette_tip_box"
    ):
        raise ValueError("Legacy tip-box migration does not match project classes")
    config = dict(
        data=str(root / role / ("data.yaml" if supervision == "complete" else "partial-training.yaml")),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device="0",
        workers=2,
        optimizer="AdamW",
        lr0=learning_rate,
        lrf=0.05,
        cos_lr=True,
        nbs=nominal_batch,
        warmup_epochs=warmup_epochs,
        warmup_bias_lr=warmup_bias_lr,
        patience=patience,
        seed=20260907,
        deterministic=True,
        freeze=freeze_layers,
        amp=False,
        cache=False,
        plots=False,
        save=True,
        save_period=10,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        erasing=0.0,
        auto_augment=None,
        project=str(output),
        name="candidate",
        exist_ok=False,
    )
    checkout = Path(__file__).resolve().parents[2]
    preflight = dict(
        schema_version="visioncortex-project-detector-experiment/1",
        role=role,
        truth_status="project_annotations",
        independent_ground_truth=False,
        production_ready=False,
        status="started",
        dataset_receipt_sha256=receipt_sha,
        base_model=str(model_path),
        base_model_sha256=model_sha,
        interpreter=sys.executable,
        python=sys.version,
        versions={
            key: importlib.metadata.version(key)
            for key in ["torch", "ultralytics", "numpy", "opencv-python"]
        },
        checkout_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip(),
        source_files={
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in _CODE_AT_IMPORT.items()
        },
        gpu=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,
        configuration=config,
        evaluation=dict(
            confidence=0.25,
            prediction_confidence_floor=0.001,
            requested_iou=0.7,
            max_det=1000,
            image_size=image_size,
        ),
        source_counts={
            split: sum(r["annotation"]["split"] == split for r in rows)
            for split in ["train", "val", "test"]
        },
        preserve_class_head=preserve_class_head,
        migrate_legacy_tip_box=migrate_legacy_tip_box,
        training_scope=training_scope,
        supervision=supervision,
        sampling_policy=sampling_policy,
        trace_branch_loss=trace_branch_loss,
        branch_loss_policy=branch_loss_policy,
        derived_training_images=receipt.get("augmentation", {})
        .get("derived_roles", {})
        .get(role, 0),
    )
    source_dir = output / "source-code"
    source_dir.mkdir()
    for name, raw in _CODE_AT_IMPORT.items():
        durable_bytes(source_dir / Path(name).name, raw)
    preflight["archived_source_files"] = {
        str(source_dir / Path(name).name): value
        for name, value in preflight["source_files"].items()
    }
    write_json(output / "experiment.json", preflight)
    validation = [r for r in rows if r["annotation"]["split"] == "val"]

    def evaluate(weight: Path, label: str) -> dict:
        detector = YOLO(str(weight))
        for index, actual in detector.names.items():
            if index >= len(names) or (
                actual != names[index]
                and not (
                    index == 10 and actual == "tube-cap" and names[index] == "tube_cap"
                )
            ):
                raise ValueError(f"Model ontology mismatch at class {index}: {actual}")
        records = []
        started = time.monotonic()
        for index, row in enumerate(validation):
            result = detector.predict(
                source=row["source"]["path"],
                imgsz=image_size,
                device="0",
                conf=0.001,
                iou=0.7,
                max_det=1000,
                half=False,
                verbose=False,
            )[0]
            detections = [
                dict(
                    class_name=names[int(c)],
                    confidence=float(p),
                    xyxy=[float(v) for v in box],
                )
                for box, c, p in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    strict=True,
                )
            ]
            records.append(
                dict(
                    image_id=row["id"],
                    event_id=row["source"]["source_group"],
                    role=role,
                    frame_index=index,
                    detections=detections,
                )
            )
        report = project_evaluation(records, validation, names)
        report["source_groups"] = {}
        for group in sorted({r["source"]["source_group"] for r in validation}):
            group_records = [r for r in records if r["event_id"] == group]
            group_rows = [r for r in validation if r["source"]["source_group"] == group]
            renumbered = [dict(r, frame_index=i) for i, r in enumerate(group_records)]
            report["source_groups"][group] = project_evaluation(
                renumbered, group_rows, names
            )
        report.update(
            model_sha256=sha256(weight),
            model_class_count=len(detector.names),
            elapsed_seconds=time.monotonic() - started,
            postprocessing=dict(
                requested_iou=0.7,
                end_to_end_direct_predictions=bool(
                    getattr(detector.model, "end2end", False)
                ),
                nms_applied=not bool(getattr(detector.model, "end2end", False)),
            ),
        )
        write_json(output / f"{label}-predictions.json", records)
        write_json(output / f"{label}-evaluation.json", report)
        del detector
        torch.cuda.empty_cache()
        return report

    start = time.monotonic()
    optimization = None
    try:
        baseline = evaluate(model_path, "baseline")
        preflight.update(status="training", baseline_micro=baseline["micro"])
        write_json(output / "experiment.json", preflight)
        validate_export(root, receipt_sha, role, supervision=supervision)
        model = YOLO(str(model_path))
        optimization = OptimizationTrace(output / "optimization-trace.jsonl")
        model.add_callback("on_pretrain_routine_end", optimization.install)
        def observe_usage(trainer):
            if supervision != "outside_ignore":
                return {}
            from .project_ignore_training import loss_usage

            usage = dict(ignore_loss_usage=loss_usage(trainer.model.criterion))
            if trace_branch_loss:
                from .project_branch_loss import branch_loss_trace_usage

                usage["branch_loss_usage"] = branch_loss_trace_usage(
                    trainer.branch_loss_trace_path, trainer.sampling_trace_path,
                    expected_epochs=int(trainer.epoch) + 1,
                    expected_policy=branch_loss_policy,
                )
            return usage

        evidence = TrainingEvidence(output, optimization, observe_usage)
        model.add_callback("on_pretrain_routine_end", evidence.install)
        preflight["training_evidence_required"] = True
        write_json(output / "experiment.json", preflight)
        if preserve_class_head:
            old_count = len(model.names)
            old_state = {
                k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()
            }

            def preserve_outputs(trainer):
                copied = transfer_class_outputs(
                    old_state,
                    trainer.model.state_dict(),
                    old_count,
                    len(names),
                    migrate_legacy_tip_box=migrate_legacy_tip_box,
                )
                if trainer.ema is not None:
                    transfer_class_outputs(
                        old_state,
                        trainer.ema.ema.state_dict(),
                        old_count,
                        len(names),
                        migrate_legacy_tip_box=migrate_legacy_tip_box,
                    )
                preflight["class_head_transfer"] = dict(
                    old_class_count=old_count, new_class_count=len(names), copied=copied
                )
                write_json(output / "experiment.json", preflight)

            model.add_callback("on_pretrain_routine_end", preserve_outputs)
        if training_scope == "class_outputs":
            def restrict_outputs(trainer):
                selected = restrict_to_class_outputs(trainer.model, len(names))
                preflight["classification_probe"] = dict(
                    trainable_parameters=selected,
                    trainable_parameter_count=sum(
                        p.numel() for p in trainer.model.parameters() if p.requires_grad
                    ),
                    frozen_state_sha256=frozen_state_sha256(trainer.model, selected),
                    normalization_statistics_policy="frozen_before_each_train_batch",
                    ema_policy="restore_frozen_state_before_validation_and_save",
                )
                write_json(output / "experiment.json", preflight)

            model.add_callback("on_pretrain_routine_end", restrict_outputs)
            model.add_callback(
                "on_train_batch_start",
                lambda trainer: freeze_normalization_statistics(trainer.model),
            )
            model.add_callback(
                "on_train_epoch_end",
                lambda trainer: synchronize_frozen_ema(
                    trainer.model,
                    trainer.ema.ema,
                    preflight["classification_probe"]["trainable_parameters"],
                ),
            )

        def observe_parameter_scope(trainer):
            # Run after the framework and any class-output restriction. A requested
            # freeze count alone is not evidence of the actual trainable parameters.
            parameters = {
                name: dict(numel=p.numel(), requires_grad=p.requires_grad)
                for name, p in trainer.model.named_parameters()
            }
            preflight["parameter_training_scope"] = dict(
                measurement="on_pretrain_routine_end_after_scope_restrictions",
                parameters=parameters,
                trainable_numel=sum(p["numel"] for p in parameters.values() if p["requires_grad"]),
                frozen_numel=sum(p["numel"] for p in parameters.values() if not p["requires_grad"]),
            )
            write_json(output / "experiment.json", preflight)

        model.add_callback("on_pretrain_routine_end", observe_parameter_scope)
        deadline = time.monotonic() + max_seconds
        model.add_callback(
            "on_train_epoch_end",
            lambda trainer: setattr(trainer, "stop", True)
            if time.monotonic() >= deadline
            else None,
        )
        if supervision == "outside_ignore":
            from functools import partial
            from .project_ignore_training import IgnoreTrainer, loss_usage

            metadata = json.loads((root / "ignore-regions.json").read_text(encoding="utf-8"))
            branch_trace = None
            if trace_branch_loss:
                from .project_branch_loss import BranchLossTrace

                branch_trace = BranchLossTrace(output / "branch-loss-trace.jsonl", epochs, policy=branch_loss_policy)
            model.train(
                trainer=partial(IgnoreTrainer, ignore_metadata={
                    str(root / name): value for name, value in metadata.items()
                }, sampling_sources=sampling_sources, sampling_policy=sampling_policy,
                    branch_loss_trace=branch_trace), **config,
            )
            from .project_sampling import sampling_usage

            preflight["sampling_usage"] = sampling_usage(
                model.trainer.sampling_plan_path, model.trainer.sampling_trace_path,
                sampling_sources, sampling_policy,
            )
            preflight["ignore_loss_usage"] = loss_usage(model.trainer.model.criterion)
            if trace_branch_loss:
                preflight["branch_loss_usage"] = observe_usage(model.trainer)["branch_loss_usage"]
            if any(r["annotation"]["ignore_regions"] for r in rows) and not all(
                r["ignored_anchor_visits"] > 0 for r in preflight["ignore_loss_usage"].values()
            ):
                raise RuntimeError("Partial training did not consume ignored regions in every branch")
        else:
            model.train(**config)
        preflight["optimization_usage"] = optimization.finish()
        preflight["training_evidence"] = verify_training_evidence(
            output, require_branch_loss=trace_branch_loss,
        )
        if preflight["training_evidence"]["optimizer_steps"] != optimization.steps:
            raise RuntimeError("Durable evidence differs from actual optimizer steps")
        if (
            supervision == "outside_ignore"
            and preflight["sampling_usage"]["completed_epochs"]
            != preflight["training_evidence"]["completed_epochs"]
        ):
            raise RuntimeError("Durable evidence differs from actual sampling epochs")
        write_json(output / "experiment.json", preflight)
        if training_scope == "class_outputs":
            probe = preflight["classification_probe"]
            after = frozen_state_sha256(
                model.trainer.model, probe["trainable_parameters"]
            )
            probe["after_training_frozen_state_sha256"] = after
            probe["frozen_state_unchanged"] = after == probe["frozen_state_sha256"]
            if not probe["frozen_state_unchanged"]:
                raise RuntimeError("Classification probe changed frozen model state")
        best = output / "candidate/weights/best.pt"
        if not best.is_file():
            raise RuntimeError("Training produced no best.pt")
        del model
        torch.cuda.empty_cache()
        candidate = evaluate(best, "candidate")
        validate_export(root, receipt_sha, role, supervision=supervision)
        if sha256(model_path) != model_sha:
            raise RuntimeError("Base model changed during experiment")
        preflight.update(
            status="completed",
            elapsed_seconds=time.monotonic() - start,
            candidate_model_sha256=sha256(best),
            candidate_micro=candidate["micro"],
            insufficient_positive_classes=candidate["insufficient_positive_classes"],
        )
    except BaseException as exc:
        if optimization is not None and optimization.handle is not None:
            optimization.handle.remove()
            optimization.handle = None
        preflight.update(
            status="failed" if isinstance(exc, Exception) else "interrupted",
            elapsed_seconds=time.monotonic() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json(output / "experiment.json", preflight)
        raise
    write_json(output / "experiment.json", preflight)
    return preflight


def recover_evaluated_experiment(output: Path) -> dict:
    """Revalidate completed prediction artifacts after a final receipt failure."""
    path = output / "experiment.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    policy = previous.get("branch_loss_policy", "native")
    if policy not in ("native", "one2many") or policy != "native" and not previous.get("trace_branch_loss"):
        raise ValueError("Recovery requires a declared, traced branch objective")
    if previous.get("trace_branch_loss"):
        from .project_branch_loss import branch_loss_trace_usage

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
            from .project_sampling import sampling_usage

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
