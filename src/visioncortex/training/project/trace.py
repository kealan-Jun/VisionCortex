"""Trace responsibilities for explicitly authorized training."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .common import sha256

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
