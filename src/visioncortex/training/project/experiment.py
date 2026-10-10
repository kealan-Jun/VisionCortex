"""Experiment responsibilities for explicitly authorized training."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from ...project_training_evidence import TrainingEvidence, durable_bytes, sync_directory, verify_training_evidence

from .common import sha256, write_json
from .cohort import validate_export, training_sampling_sources
from .evaluation import project_evaluation
from .trace import OptimizationTrace
from .class_head import transfer_class_outputs, restrict_to_class_outputs, freeze_normalization_statistics, frozen_state_sha256, synchronize_frozen_ema
from .source import _CODE_AT_IMPORT


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
    checkout = Path(__file__).resolve().parents[4]
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
            from ...project_ignore_training import loss_usage

            usage = dict(ignore_loss_usage=loss_usage(trainer.model.criterion))
            if trace_branch_loss:
                from ...project_branch_loss import branch_loss_trace_usage

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
            from ...project_ignore_training import IgnoreTrainer, loss_usage

            metadata = json.loads((root / "ignore-regions.json").read_text(encoding="utf-8"))
            branch_trace = None
            if trace_branch_loss:
                from ...project_branch_loss import BranchLossTrace

                branch_trace = BranchLossTrace(output / "branch-loss-trace.jsonl", epochs, policy=branch_loss_policy)
            model.train(
                trainer=partial(IgnoreTrainer, ignore_metadata={
                    str(root / name): value for name, value in metadata.items()
                }, sampling_sources=sampling_sources, sampling_policy=sampling_policy,
                    branch_loss_trace=branch_trace), **config,
            )
            from ...project_sampling import sampling_usage

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
