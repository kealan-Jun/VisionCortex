"""Observed detection losses with an explicit, verifiable branch objective."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _vector(value):
    if (
        not isinstance(value, list) or len(value) != 3
        or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in value)
    ):
        raise ValueError("Branch loss requires three finite nonnegative components")
    return value


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)


def validate_branch_loss_row(row):
    """Check native gains, weighted objective, and the distinct logged items."""
    if (
        row.get("schema_version") != "visioncortex-branch-loss/1"
        or row.get("phase") != "train" or row.get("policy") not in ("native", "one2many")
        or type(row.get("call")) is not int or row["call"] < 1
        or type(row.get("epoch")) is not int or row["epoch"] < 0
        or type(row.get("epochs")) is not int or row["epochs"] <= row["epoch"]
        or type(row.get("batch_size")) is not int or row["batch_size"] < 1
        or not isinstance(row.get("files"), list)
        or len(row["files"]) != row["batch_size"]
        or any(not isinstance(p, str) or not p for p in row["files"])
    ):
        raise ValueError("Invalid training branch-loss identity")
    gains = row.get("gains", {})
    if set(gains) != {"one2many", "one2one"} or any(
        type(g) not in (float, int) or not math.isfinite(g) or not 0 <= g <= 1
        for g in gains.values()
    ):
        raise ValueError("Invalid observed branch-loss gains")
    expected = max(1 - row["epoch"] / max(row["epochs"] - 1, 1), 0) * (0.8 - 0.1) + 0.1
    if row["policy"] == "one2many":
        expected = 1.0
    if not _close(gains["one2many"], expected) or not _close(sum(gains.values()), 1):
        raise ValueError("Observed gains differ from the declared branch objective")
    branches = row.get("branch_vectors", {})
    if set(branches) != {"one2many", "one2one"}:
        raise ValueError("Both native branch losses are required")
    many, one = _vector(branches["one2many"]), _vector(branches["one2one"])
    weighted = _vector(row.get("weighted_vector"))
    logged = _vector(row.get("logged_one2one_items"))
    for a, b, actual, item in zip(many, one, weighted, logged, strict=True):
        if not _close(a * gains["one2many"] + b * gains["one2one"], actual):
            raise ValueError("Observed weighted objective differs from branch components")
        if not _close(b / row["batch_size"], item):
            raise ValueError("Native logged items differ from one2one per-image loss")


class BranchLossTrace:
    """Append only actual training loss calls, with no live tensors or trainer ref."""

    def __init__(self, path: Path, epochs: int, *, policy: str = "native"):
        if type(epochs) is not int or epochs < 1:
            raise ValueError("Branch trace requires a positive training horizon")
        if policy not in ("native", "one2many"):
            raise ValueError("Unknown branch objective")
        self.path = Path(path).resolve()
        self.path.touch(exist_ok=False)
        self.policy = policy
        self.epochs, self.epoch, self.calls = epochs, -1, 0

    def start_epoch(self, epoch: int):
        if type(epoch) is not int or epoch != self.epoch + 1 or epoch >= self.epochs:
            raise ValueError("Branch trace epochs must start at zero and be consecutive")
        self.epoch = epoch

    def record(self, batch, gains, many, one, weighted, logged):
        def values(tensor):
            return tensor.detach().cpu().tolist()

        row = dict(
            schema_version="visioncortex-branch-loss/1", phase="train", policy=self.policy,
            call=self.calls + 1, epoch=self.epoch, epochs=self.epochs,
            files=[str(Path(p).absolute()) for p in batch["im_file"]],
            batch_size=int(batch["img"].shape[0]),
            gains={k: float(v) for k, v in gains.items()},
            branch_vectors=dict(one2many=values(many), one2one=values(one)),
            weighted_vector=values(weighted), logged_one2one_items=values(logged),
        )
        validate_branch_loss_row(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.calls += 1


def branch_loss_trace_usage(path: Path, sampling_trace: Path, *, expected_epochs=None, expected_policy=None):
    """Recompute the trace and bind each loss call to actual preprocessing order."""
    path, sampling_trace = Path(path), Path(sampling_trace)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    samples = [json.loads(line) for line in sampling_trace.read_text(encoding="utf-8").splitlines()]
    if not rows or len(rows) != len(samples):
        raise ValueError("Branch-loss calls differ from actual preprocessing batches")
    epochs, horizon, policy = [], rows[0].get("epochs"), rows[0].get("policy")
    if expected_policy is not None and policy != expected_policy:
        raise ValueError("Branch-loss trace differs from the required objective policy")
    for index, (row, sample) in enumerate(zip(rows, samples, strict=True), 1):
        validate_branch_loss_row(row)
        if (
            row["call"] != index or row["epochs"] != horizon or row["policy"] != policy
            or sample != {"epoch": row["epoch"], "files": row["files"]}
        ):
            raise ValueError("Branch-loss order or inputs differ from preprocessing")
        epoch = row["epoch"]
        if epoch == len(epochs):
            epochs.append(dict(
                epoch=epoch, batches=0, image_visits=0, gains=row["gains"],
                component_sums={k: [0., 0., 0.] for k in ["one2many", "one2one", "weighted"]},
            ))
        if epoch != len(epochs) - 1 or epochs[-1]["gains"] != row["gains"]:
            raise ValueError("Branch-loss epochs or within-epoch gains are inconsistent")
        out = epochs[-1]
        out["batches"] += 1
        out["image_visits"] += row["batch_size"]
        for key in out["component_sums"]:
            vector = row["weighted_vector"] if key == "weighted" else row["branch_vectors"][key]
            out["component_sums"][key] = [a + b for a, b in zip(out["component_sums"][key], vector, strict=True)]
    if expected_epochs is not None and len(epochs) != expected_epochs:
        raise ValueError("Branch-loss trace does not cover the required completed epochs")
    for epoch in epochs:
        epoch["mean_components_per_image"] = {
            k: [x / epoch["image_visits"] for x in v]
            for k, v in epoch.pop("component_sums").items()
        }
    return dict(
        schema_version="visioncortex-branch-loss-usage/1", path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sampling_trace_path=str(sampling_trace),
        sampling_trace_sha256=hashlib.sha256(sampling_trace.read_bytes()).hexdigest(),
        calls=len(rows), image_visits=sum(r["batch_size"] for r in rows),
        completed_epochs=len(epochs), configured_epochs=horizon, per_epoch=epochs,
        measurement="observed_training_forward_losses_and_gains_not_gradient_norms",
        components=["box", "classification", "dfl"],
        batch_scaling="branch_vectors_include_batch_size; means_divide_by_total_image_visits",
        native_csv_branch="one2one", validation_calls_included=False,
        independent_ground_truth=False, production_ready=False,
    )
