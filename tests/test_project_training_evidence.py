import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from visioncortex.project_training_evidence import (
    TrainingEvidence, durable_json, verify_training_evidence,
)


def trainer_fixture(tmp_path):
    weights = tmp_path / "candidate/weights"
    weights.mkdir(parents=True)
    optimization = SimpleNamespace(path=tmp_path / "optimization-trace.jsonl", steps=1)
    optimization.path.write_text('{"step":1}\n')
    candidate = weights.parent
    (candidate / "args.yaml").write_text("epochs: 2\n")
    (candidate / "results.csv").write_text("epoch\n1\n")
    trainer = SimpleNamespace(
        epoch=0, wdir=weights, last=weights / "last.pt", best=weights / "best.pt",
        csv=candidate / "results.csv", save_dir=candidate,
    )

    def save():
        trainer.last.write_bytes(b"checkpoint" + str(trainer.epoch).encode())
        trainer.best.write_bytes(trainer.last.read_bytes())

    def finalize():
        trainer.last.write_bytes(trainer.last.read_bytes() + b" stripped")
        trainer.best.write_bytes(trainer.best.read_bytes() + b" stripped")

    trainer.save_model, trainer.final_eval = save, finalize
    return trainer, optimization


def test_json_fsync_failure_preserves_previous_receipt_and_failed_staging(tmp_path, monkeypatch):
    path = tmp_path / "experiment.json"
    durable_json(path, dict(status="training"))

    def failure(_descriptor):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("visioncortex.project_training_evidence.os.fsync", failure)
    with pytest.raises(OSError, match="disk failure"):
        durable_json(path, dict(status="completed"))
    assert json.loads(path.read_text()) == dict(status="training")
    assert len(list(tmp_path.glob(".experiment.json.*.partial"))) == 1


@pytest.mark.parametrize("phase", ["save", "finalize"])
def test_native_interruption_preserves_published_weights_and_restores_paths(tmp_path, phase):
    trainer, optimization = trainer_fixture(tmp_path)
    trainer.last.write_bytes(b"previous last")
    trainer.best.write_bytes(b"previous best")
    original = (trainer.wdir, trainer.last, trainer.best)

    def interrupt():
        trainer.last.write_bytes(b"")
        raise KeyboardInterrupt()

    if phase == "save":
        trainer.save_model = interrupt
    else:
        trainer.final_eval = interrupt
    evidence = TrainingEvidence(tmp_path, optimization)
    evidence.install(trainer)
    with pytest.raises(KeyboardInterrupt):
        (trainer.save_model if phase == "save" else trainer.final_eval)()
    assert (trainer.wdir, trainer.last, trainer.best) == original
    assert trainer.last.read_bytes() == b"previous last"
    assert trainer.best.read_bytes() == b"previous best"
    assert list(evidence.root.glob("*/checkpoint-staging/last.pt"))
    assert not (evidence.root / "latest.json").exists()


def test_saved_epochs_remain_readable_after_corrupt_tail_and_require_finalization(tmp_path):
    trainer, optimization = trainer_fixture(tmp_path)
    evidence = TrainingEvidence(tmp_path, optimization)
    evidence.install(trainer)
    trainer.save_model()
    first = evidence.root / "epoch-0000/optimization-trace.jsonl"
    optimization.path.write_bytes(b'first epoch damaged\0\0')
    assert first.read_bytes() == b'{"step":1}\n'
    with pytest.raises(FileNotFoundError):
        verify_training_evidence(tmp_path)
    # A new save for the same epoch cannot overwrite its retained evidence.
    with pytest.raises(FileExistsError):
        trainer.save_model()


def test_complete_native_lifecycle_verifies_snapshots_and_rejects_drift(tmp_path):
    from visioncortex.project_annotation_training import recover_evaluated_experiment

    trainer, optimization = trainer_fixture(tmp_path)
    trainer.sampling_plan_path = trainer.save_dir / "source-sampling-plan.json"
    trainer.sampling_trace_path = trainer.save_dir / "source-sampling-trace.jsonl"
    trainer.sampling_plan_path.write_text('{"policy":"source_balanced"}')
    trainer.sampling_trace_path.write_text('{"epoch":0,"files":["a"]}\n')
    evidence = TrainingEvidence(tmp_path, optimization, lambda t: dict(epoch_seen=t.epoch))
    evidence.install(trainer)
    trainer.save_model()
    trainer.epoch = 1
    optimization.steps = 2
    with optimization.path.open("a") as h:
        h.write('{"step":2}\n')
    trainer.save_model()
    trainer.final_eval()
    result = verify_training_evidence(tmp_path)
    assert result["completed_epochs"] == 2 and result["optimizer_steps"] == 2
    assert len(result["receipts"]) == 3
    assert trainer.last.read_bytes() == b"checkpoint1 stripped"
    completed = dict(status="completed", training_evidence_required=True, training_evidence=result)
    durable_json(tmp_path / "experiment.json", completed)
    assert recover_evaluated_experiment(tmp_path) == completed
    # Tampering with a past snapshot is rejected even if the final weights work.
    snapshot = evidence.root / "epoch-0000/source-sampling-trace.jsonl"
    old = snapshot.read_bytes()
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        verify_training_evidence(tmp_path)
    with pytest.raises(ValueError, match="snapshot changed"):
        recover_evaluated_experiment(tmp_path)
    snapshot.write_bytes(old)
    trainer.best.write_bytes(b"a different checkpoint")
    with pytest.raises(ValueError, match="checkpoint differs"):
        verify_training_evidence(tmp_path)


def test_process_exit_during_native_write_cannot_truncate_published_checkpoint(tmp_path):
    import os
    import subprocess
    import sys

    trainer, _ = trainer_fixture(tmp_path)
    trainer.last.write_bytes(b"previous completed checkpoint")
    script = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from visioncortex.project_training_evidence import TrainingEvidence
p=Path(sys.argv[1]); w=p/'candidate/weights'
t=SimpleNamespace(epoch=0, wdir=w, last=w/'last.pt', best=w/'best.pt')
def fail():
    t.last.write_bytes(b'')
    os._exit(77)
t.save_model=fail; t.final_eval=lambda: None
TrainingEvidence(p, SimpleNamespace(path=p/'optimization-trace.jsonl',steps=1)).install(t)
t.save_model()
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env)
    assert result.returncode == 77
    assert trainer.last.read_bytes() == b"previous completed checkpoint"
    assert (tmp_path / "training-evidence/epoch-0000/checkpoint-staging/last.pt").stat().st_size == 0
