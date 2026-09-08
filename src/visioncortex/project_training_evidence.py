"""Durable evidence for bounded local training; never repairs missing history."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sync_directory(path: Path) -> None:
    # Windows does not expose directory fsync through os.open. File fsync and
    # atomic replacement still apply; receipts explicitly record that boundary.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def durable_bytes(path: Path, raw: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    # A failed staging file is retained for inspection, never installed as valid.
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def durable_json(path: Path, value: Any) -> None:
    durable_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode())


def file_identity(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


class TrainingEvidence:
    """Stage native checkpoint writes and snapshot the evidence after each save.

    Only checkpoint output paths are redirected. Native serialization, training,
    best-weight selection and final validation remain the framework's operations.
    A process killed before publication leaves old weights and staging material.
    Publication of multiple files is not one transaction: a missing or mismatched
    final receipt must fail verification, even if a checkpoint itself is readable.
    """

    def __init__(self, output: Path, optimization: Any, usage_observer=None):
        self.output = output.resolve()
        self.root = self.output / "training-evidence"
        self.root.mkdir()
        sync_directory(self.output)
        self.optimization = optimization
        self.usage_observer = usage_observer
        self.installed = False

    def install(self, trainer: Any) -> None:
        if self.installed:
            raise ValueError("Training evidence is already installed")
        self.installed = True
        native_save, native_final = trainer.save_model, trainer.final_eval
        trainer.save_model = lambda: self._stage(trainer, native_save, "epoch")
        trainer.final_eval = lambda: self._stage(trainer, native_final, "finalized")

    def _stage(self, trainer: Any, operation, phase: str):
        name = f"epoch-{int(trainer.epoch):04d}" if phase == "epoch" else "finalized"
        destination = self.root / name
        destination.mkdir()  # Never overwrite a previous epoch or failed attempt.
        sync_directory(self.root)
        original = {key: Path(getattr(trainer, key)) for key in ("wdir", "last", "best")}
        staging = destination / "checkpoint-staging"
        staging.mkdir()
        if phase == "finalized":
            for key in ("last", "best"):
                if original[key].is_file():
                    durable_bytes(staging / original[key].name, original[key].read_bytes())
        trainer.wdir = staging
        trainer.last, trainer.best = staging / original["last"].name, staging / original["best"].name
        try:
            result = operation()
        finally:
            for key, value in original.items():
                setattr(trainer, key, value)
        pending = sorted(staging.iterdir())
        allowed = {"last.pt", "best.pt", f"epoch{int(trainer.epoch)}.pt"}
        if not pending or not (staging / "last.pt").is_file():
            raise ValueError("Native training did not stage a last checkpoint")
        for path in pending:
            if path.name not in allowed or not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Native training staged an invalid checkpoint")
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        for path in pending:
            os.replace(path, original["wdir"] / path.name)
        sync_directory(original["wdir"])
        # Source trace files are closed between batches. Immutable snapshots keep
        # earlier complete epochs readable even if a later append is interrupted.
        sources = [self.optimization.path, Path(trainer.csv), Path(trainer.save_dir) / "args.yaml"]
        for key in ("sampling_plan_path", "sampling_trace_path"):
            path = getattr(trainer, key, None)
            if path is not None:
                sources.append(Path(path))
        artifacts = []
        for path in sources:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            sync_directory(path.parent)
            target = destination / path.name
            durable_bytes(target, path.read_bytes())
            artifacts.append(dict(source_path=str(path), **file_identity(target)))
        checkpoints = [file_identity(original["wdir"] / path.name) for path in pending]
        record = dict(
            schema_version="visioncortex-training-evidence/1", phase=phase,
            epoch=int(trainer.epoch), optimizer_steps=self.optimization.steps,
            artifacts=artifacts, checkpoint_hashes_at_publication=checkpoints,
            checkpoint_paths_mutable=True, independent_ground_truth=False,
            training_complete=False, production_ready=False,
            directory_fsync=os.name != "nt",
            observed_usage=self.usage_observer(trainer) if self.usage_observer else {},
        )
        durable_json(destination / "receipt.json", record)
        durable_json(self.root / "latest.json", file_identity(destination / "receipt.json"))
        staging.rmdir()
        return result


def verify_training_evidence(output: Path) -> dict:
    """Require the final native operation and every saved immutable trace snapshot."""
    root = output.resolve() / "training-evidence"
    latest = json.loads((root / "latest.json").read_text())
    final = root / "finalized/receipt.json"
    if latest != file_identity(final):
        raise ValueError("Training evidence lacks a matching finalized receipt")
    records, steps, epoch = [], -1, -1
    epochs = sorted(root.glob("epoch-*/receipt.json"))
    if not epochs:
        raise ValueError("Training evidence does not contain a saved epoch")
    paths = epochs + [final]
    for path in paths:
        record = json.loads(path.read_text())
        expected_epoch = epoch + 1 if path != final else epoch
        if (
            record["schema_version"] != "visioncortex-training-evidence/1"
            or record["phase"] != ("finalized" if path == final else "epoch")
            or record["epoch"] != expected_epoch or record["optimizer_steps"] < steps
        ):
            raise ValueError("Training evidence has missing epochs or inconsistent optimizer steps")
        if path == final and record["optimizer_steps"] != steps:
            raise ValueError("Final validation unexpectedly changed optimizer steps")
        for item in record["artifacts"]:
            artifact = Path(item["path"])
            if artifact.parent != path.parent or file_identity(artifact) != {
                k: item[k] for k in ("path", "sha256", "bytes")
            }:
                raise ValueError("Training evidence snapshot changed")
            if path == final and artifact.read_bytes() != Path(item["source_path"]).read_bytes():
                raise ValueError("Final training trace differs from its durable snapshot")
        if path == final:
            for item in record["checkpoint_hashes_at_publication"]:
                if file_identity(Path(item["path"])) != item:
                    raise ValueError("Final training checkpoint differs from its receipt")
        records.append(file_identity(path))
        steps, epoch = record["optimizer_steps"], record["epoch"]
    return dict(
        schema_version="visioncortex-training-evidence-verification/1",
        completed_epochs=epoch + 1, optimizer_steps=steps, receipts=records,
        final_receipt_sha256=latest["sha256"], directory_fsync=os.name != "nt",
        checkpoint_retention="Native best/last and periodic checkpoints; historical hashes are not retained weights.",
    )
