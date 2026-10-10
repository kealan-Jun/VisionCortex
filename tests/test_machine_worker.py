import importlib.util
import os
from pathlib import Path

import pytest

from visioncortex.machine_worker import validate_gpu_guard


fcntl = pytest.importorskip("fcntl")
GPU = "GPU-12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def guarded_lock(monkeypatch, tmp_path):
    for prefix in ("VISIONCORTEX", "REALITYLOOP"):
        for name in ("GPU_UUID", "GPU_GUARD_FD", "GPU_LOCK_PATH"):
            monkeypatch.delenv(prefix + "_" + name, raising=False)
    path = tmp_path / (GPU + ".lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield fd, path
    finally:
        os.close(fd)


def set_guard_environment(monkeypatch, prefix, fd, path, *, include_path=True):
    monkeypatch.setenv(prefix + "_GPU_UUID", GPU)
    monkeypatch.setenv(prefix + "_GPU_GUARD_FD", str(fd))
    if include_path:
        monkeypatch.setenv(prefix + "_GPU_LOCK_PATH", str(path))


@pytest.mark.parametrize("prefix", ["VISIONCORTEX", "REALITYLOOP"])
@pytest.mark.parametrize("include_path", [False, True])
def test_accepts_guard_fd_and_retains_exclusive_lock(
    monkeypatch, guarded_lock, prefix, include_path,
):
    fd, path = guarded_lock
    set_guard_environment(monkeypatch, prefix, fd, path, include_path=include_path)
    validate_gpu_guard()
    contender = os.open(path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(contender)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("GPU_UUID", "GPU-unsafe/path"), ("GPU_GUARD_FD", "-1"), ("GPU_LOCK_PATH", "/unrelated.lock")],
)
def test_primary_guard_environment_takes_precedence(
    monkeypatch, guarded_lock, field, invalid,
):
    fd, path = guarded_lock
    set_guard_environment(monkeypatch, "REALITYLOOP", fd, path)
    monkeypatch.setenv("VISIONCORTEX_" + field, invalid)
    with pytest.raises(RuntimeError, match="Start through"):
        validate_gpu_guard()


def test_rejects_non_file_fd(monkeypatch, guarded_lock):
    _fd, path = guarded_lock
    reader, writer = os.pipe()
    try:
        set_guard_environment(monkeypatch, "VISIONCORTEX", reader, path)
        with pytest.raises(RuntimeError, match="Start through"):
            validate_gpu_guard()
    finally:
        os.close(reader)
        os.close(writer)


def test_rejects_mismatched_fd_identity(monkeypatch, guarded_lock, tmp_path):
    _fd, path = guarded_lock
    unrelated = os.open(tmp_path / "unrelated.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        set_guard_environment(monkeypatch, "VISIONCORTEX", unrelated, path)
        with pytest.raises(RuntimeError, match="Start through"):
            validate_gpu_guard()
    finally:
        os.close(unrelated)


def test_rejects_symlink_lock_path(monkeypatch, guarded_lock, tmp_path):
    fd, path = guarded_lock
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    alias = alias_dir / path.name
    alias.symlink_to(path)
    set_guard_environment(monkeypatch, "VISIONCORTEX", fd, alias)
    with pytest.raises(RuntimeError, match="Start through"):
        validate_gpu_guard()


def test_rejects_fd_without_shared_lock_ownership(monkeypatch, guarded_lock):
    _fd, path = guarded_lock
    contender = os.open(path, os.O_RDWR)
    try:
        set_guard_environment(monkeypatch, "VISIONCORTEX", contender, path)
        with pytest.raises(RuntimeError, match="Start through"):
            validate_gpu_guard()
    finally:
        os.close(contender)


def test_guard_exports_both_protocol_names(monkeypatch, tmp_path):
    launcher = Path(__file__).parents[1] / "deployment/inference/gpu_guard.py"
    spec = importlib.util.spec_from_file_location("visioncortex_gpu_guard_test", launcher)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    exported = {}

    class CompletedProcess:
        pid = 999999
        returncode = 0

        def poll(self):
            return self.returncode

    def spawn(_command, *, env, pass_fds, start_new_session):
        assert start_new_session is True
        for name in ("GPU_UUID", "GPU_GUARD_FD", "GPU_LOCK_PATH"):
            assert env["VISIONCORTEX_" + name] == env["REALITYLOOP_" + name]
            exported[name] = env["VISIONCORTEX_" + name]
        assert pass_fds == (int(exported["GPU_GUARD_FD"]),)
        with monkeypatch.context() as child:
            for name, value in exported.items():
                child.setenv("VISIONCORTEX_" + name, value)
            validate_gpu_guard()
        return CompletedProcess()

    def missing_group(_pid, _signal):
        raise ProcessLookupError

    monkeypatch.setattr(guard.subprocess, "Popen", spawn)
    monkeypatch.setattr(guard.os, "killpg", missing_group)
    assert guard.run_guard(["owned-test-command"], GPU, tmp_path, check_gpu=lambda _gpu: None) == 0
    assert exported["GPU_UUID"] == GPU
    assert Path(exported["GPU_LOCK_PATH"]) == tmp_path / (GPU + ".lock")



def test_machine_worker_template_is_rejected_before_gpu_guard(monkeypatch):
    import sys
    from visioncortex import machine_worker

    config = Path(__file__).parents[1] / "deployment/inference/pipeline.example.yaml"
    monkeypatch.setattr(sys, "argv", ["machine_worker", "--config", str(config)])
    monkeypatch.setattr(machine_worker, "validate_gpu_guard", lambda: pytest.fail("must not inspect GPU lock"))
    with pytest.raises(ValueError, match="private site"):
        machine_worker.main()


def test_gpu_launcher_rejects_pipeline_template_before_lock_or_probe(monkeypatch, tmp_path):
    import sys

    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("unprepared_gpu_guard_test", root / "deployment/inference/gpu_guard.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    lock = tmp_path / "locks"
    monkeypatch.setattr(sys, "argv", ["gpu_guard", "--gpu", GPU, "--lock-dir", str(lock),
                                      "--config", str(root / "deployment/inference/pipeline.example.yaml"),
                                      "--", "owned-test-command"])
    monkeypatch.setattr(guard, "run_guard", lambda *_args: pytest.fail("must not inspect GPU or create lock"))
    assert guard.main() == 1
    assert not lock.exists()



def test_public_model_preparation_template_is_rejected_before_download(monkeypatch):
    from visioncortex import cli

    config = Path(__file__).parents[1] / "configs/rtx3050-6gb-ubuntu20-production.yaml"
    monkeypatch.setattr(cli, "prepare_public_model_assets", lambda _settings: pytest.fail("must not download assets"))
    with pytest.raises(ValueError, match="private site"):
        cli.prepare_public_models_command(config)
