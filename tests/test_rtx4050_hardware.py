from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hardware():
    spec = importlib.util.spec_from_file_location("hardware_test", ROOT / "tools/rtx4050_hardware.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(hardware, events, ending="", beginning=""):
    lines = [beginning, "import json, time, os"]
    for event in events:
        lines.append(f"print({hardware.PREFIX!r} + {json.dumps(event)!r}, flush=True)")
    lines.append(ending)
    return [sys.executable, "-u", "-c", "\n".join(lines)]


def result():
    return {"event": "result", "hardware": {"gpu": "RTX 4050", "gpu_uuid": "synthetic-gpu",
            "torch": "2.6.0+cu124", "tensorrt": "10.4.0", "cuda_fp16_invocation": "PROVEN"}}


def successful_events():
    return [{"event": "start", "stage": "torch_import"},
            {"event": "done", "stage": "torch_import"}, result()]


def test_success_continues_immediately_without_waiting_for_deadline(hardware, tmp_path):
    states = []
    started = time.monotonic()
    value = hardware.supervise(tmp_path, command(hardware, successful_events()),
                               lambda *args, **kw: states.append(kw),
                               steps={"torch_import": ("加载 PyTorch", 180)})
    assert time.monotonic() - started < 5
    assert value["gpu"] == "RTX 4050"
    assert states[0]["operation"] == "torch_import"
    receipt = json.loads((tmp_path / "Runtime/Logs/hardware-preflight.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["cuda_fp16_invocation"] == "PROVEN"
    assert receipt["model_invocation"] == "NOT_PROVEN"
    assert receipt["steps"][0]["step"] == "torch_import"


def test_stalled_import_is_killed_and_records_exact_step(hardware, tmp_path):
    began = time.monotonic()
    with pytest.raises(RuntimeError, match="加载 PyTorch"):
        hardware.supervise(tmp_path, command(hardware, [{"event": "start", "stage": "torch_import"}],
                                             "time.sleep(60)"), lambda *_, **kw: None,
                           steps={"torch_import": ("加载 PyTorch", .15)})
    assert time.monotonic() - began < 5
    receipt = json.loads((tmp_path / "Runtime/Logs/hardware-preflight.json").read_text(encoding="utf-8"))
    assert receipt["timed_out_step"] == "torch_import"
    assert receipt["status"] == "failed"
    assert receipt["model_invocation"] == "NOT_PROVEN"
    assert not receipt.get("worker_exit_pending")


@pytest.mark.parametrize("events,ending", [
    (successful_events(), "raise SystemExit(2)"),
    (successful_events(), "time.sleep(60)"),
    (successful_events() + [result()], ""),
    ([{"event": "start", "stage": "torch_import"}] * 2, ""),
    ([{"event": "done", "stage": "torch_import"}], ""),
    ([result()], ""),
])
def test_invalid_order_crash_or_nonexit_cannot_pass(hardware, tmp_path, events, ending):
    with pytest.raises(RuntimeError):
        hardware.supervise(tmp_path, command(hardware, events, ending), lambda *_, **kw: None,
                           steps={"torch_import": ("加载 PyTorch", 1)}, exit_timeout=.15)
    assert json.loads((tmp_path / "Runtime/Logs/hardware-preflight.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_probe_does_not_receive_provider_secrets(hardware, tmp_path, monkeypatch):
    monkeypatch.setenv("MLLM_API_KEY", "test-private-value")
    monkeypatch.setenv("VISIONCORTEX_DESKTOP_CONNECTION", "test-private-settings")
    monkeypatch.setenv("GITHUB_TOKEN", "test-private-token")
    ending = "assert not any(k in os.environ for k in ['MLLM_API_KEY','VISIONCORTEX_DESKTOP_CONNECTION','GITHUB_TOKEN'])"
    hardware.supervise(tmp_path, command(hardware, successful_events(), ending), lambda *_, **kw: None,
                       steps={"torch_import": ("加载 PyTorch", 1)})
    assert "test-private" not in (tmp_path / "Runtime/Logs/hardware-preflight.log").read_text(encoding="utf-8")


def test_worker_that_never_reports_cannot_wait_forever(hardware, tmp_path):
    with pytest.raises(RuntimeError, match="启动检查进程"):
        hardware.supervise(tmp_path, command(hardware, [], "time.sleep(60)"), lambda *_, **kw: None,
                           steps={"torch_import": ("加载 PyTorch", 1)}, startup_timeout=.15)
    receipt = json.loads((tmp_path / "Runtime/Logs/hardware-preflight.json").read_text(encoding="utf-8"))
    assert receipt["timed_out_step"] == "worker_start"


def test_native_probe_checks_driver_before_importing_torch(hardware, monkeypatch):
    events = []
    monkeypatch.setattr(hardware, "emit", lambda kind, **data: events.append((kind, data)))
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware.sys, "version_info", (3, 12))
    monkeypatch.setattr(hardware.platform, "machine", lambda: "AMD64")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=16 * 1024**3)))
    monkeypatch.setattr(hardware.shutil, "which", lambda _: "fake-nvidia-smi")

    def missing_driver(*args, **kwargs):
        assert kwargs["timeout"] == 20
        raise RuntimeError("synthetic driver failure")

    monkeypatch.setattr(hardware.subprocess, "run", missing_driver)
    with pytest.raises(RuntimeError, match="synthetic driver"):
        hardware.run_probe()
    assert events[-1] == ("start", {"stage": "driver_query"})
    assert not any(data.get("stage") == "torch_import" for _, data in events)
