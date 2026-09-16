"""Isolate native GPU initialization from the desktop's responsive supervisor."""
from __future__ import annotations

from contextlib import contextmanager
import faulthandler
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time


PREFIX = "VISIONCORTEX_HARDWARE_PROBE "
STEPS = {
    "platform": ("检查系统与内存", 30),
    "driver_query": ("查询 NVIDIA 驱动", 30),
    "torch_import": ("加载 PyTorch", 180),
    "cuda_available": ("初始化 CUDA 设备查询", 90),
    "cuda_device": ("读取 CUDA 设备属性", 90),
    "tensorrt_import": ("加载 TensorRT", 120),
    "cuda_allocate": ("分配 CUDA 自检张量", 90),
    "cuda_compute": ("执行 CUDA FP16 运算", 90),
    "cuda_sync": ("等待 CUDA 运算完成", 60),
    "cuda_release": ("释放 CUDA 自检资源", 30),
}


def emit(event, **fields):
    print(PREFIX + json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


@contextmanager
def operation(name):
    emit("start", stage=name)
    # A native call can hold the GIL. The parent process owns the deadline.
    # This stack is diagnostic evidence, not a claim about a specific driver bug.
    faulthandler.dump_traceback_later(30, repeat=False)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
    emit("done", stage=name)


def run_probe():
    with operation("platform"):
        if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
            raise RuntimeError("此压缩包只适用于 Windows 10/11 64 位。")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("必须使用包内 Python 3.12。")
        import psutil

        memory = psutil.virtual_memory()
        if memory.available < 8 * 1024**3:
            raise RuntimeError("可用系统内存不足 8 GiB，请关闭其他大型程序。")
    with operation("driver_query"):
        smi = shutil.which("nvidia-smi")
        if not smi:
            candidate = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/nvidia-smi.exe"
            smi = str(candidate) if candidate.is_file() else None
        if not smi:
            raise RuntimeError("未找到 nvidia-smi，无法核验驱动和显卡身份。")
        result = subprocess.run(
            [smi, "--query-gpu=uuid,driver_version,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            check=True, capture_output=True, text=True, timeout=20,
        )
        uuid, driver, vram = [part.strip() for part in result.stdout.strip().split(",")]
        if tuple(map(int, driver.split("."))) < (551, 78):
            raise RuntimeError("NVIDIA 驱动低于 CUDA 12.4 要求，请更新驱动。")
        if float(vram) < 5700:
            raise RuntimeError("显卡总显存不足 6GB 配置要求。")
    with operation("torch_import"):
        import torch

        if torch.__version__ != "2.6.0+cu124":
            raise RuntimeError("PyTorch 版本与离线包不一致。")
    with operation("cuda_available"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请检查 NVIDIA 驱动，然后重新启动。")
    with operation("cuda_device"):
        name = torch.cuda.get_device_name(0)
        if "4050" not in name:
            raise RuntimeError(f"本包固定适配 RTX 4050，当前 GPU：{name}")
    with operation("tensorrt_import"):
        import tensorrt

        if tensorrt.__version__ != "10.4.0":
            raise RuntimeError("TensorRT 版本与离线包不一致。")
    with operation("cuda_allocate"):
        probe = torch.ones((128, 128), device="cuda", dtype=torch.float16)
    with operation("cuda_compute"):
        computed = probe @ probe
        finite = torch.isfinite(computed).all()
    with operation("cuda_sync"):
        torch.cuda.synchronize()
        if not finite.item():
            raise RuntimeError("CUDA FP16 自检失败。")
    with operation("cuda_release"):
        del probe, computed, finite
        torch.cuda.empty_cache()
    emit("result", hardware={
        "gpu": name, "gpu_uuid": uuid, "driver": driver, "vram_mib": float(vram),
        "ram_gib": round(memory.total / 1024**3, 2), "cpu_logical_count": os.cpu_count(),
        "torch": torch.__version__, "tensorrt": tensorrt.__version__,
        "python": platform.python_version(), "windows": platform.version(),
        "cuda_fp16_invocation": "PROVEN", "real_video_quality": "NOT_PROVEN",
    })


def _save(path, value):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def supervise(root, command, notify, *, steps=None, startup_timeout=30, exit_timeout=5):
    steps = STEPS if steps is None else steps
    logs = root / "Runtime/Logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary_path = logs / "hardware-preflight.json"
    started = time.monotonic()
    receipt = {"status": "running", "started_at_unix": time.time(), "steps": [], "active_step": "worker_start",
               "cuda_fp16_invocation": "NOT_PROVEN", "model_invocation": "NOT_PROVEN",
               "real_video_quality": "NOT_PROVEN"}
    _save(summary_path, receipt)
    # This child only checks hardware. It has no use for provider credentials.
    environment = {k: v for k, v in os.environ.items()
                   if not re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", k, re.I)
                   and k != "VISIONCORTEX_DESKTOP_CONNECTION"}
    events = queue.Queue(maxsize=256)
    child = None
    reader = None
    output_bytes = 0
    names = list(steps)
    completed = 0
    active = None
    result = None
    deadline = started + startup_timeout
    stage_started = started
    with (logs / "hardware-preflight.log").open("w", encoding="utf-8") as log:
        try:
            child = subprocess.Popen(
                command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            receipt["pid"] = child.pid
            _save(summary_path, receipt)

            def read_lines():
                try:
                    while line := child.stdout.readline(65536):
                        events.put(line)
                finally:
                    events.put(None)

            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            while True:
                now = time.monotonic()
                if now >= deadline:
                    step = receipt["active_step"]
                    label = steps.get(step, ("启动检查进程" if step == "worker_start" else "结束检查进程", 0))[0]
                    receipt["timed_out_step"] = step
                    raise RuntimeError(f"运行环境检查在「{label}」超过等待上限，已停止本次检查。请查看 hardware-preflight.json 和 hardware-preflight.log。")
                try:
                    line = events.get(timeout=min(0.1, deadline - now))
                except queue.Empty:
                    continue
                if line is None:
                    code = child.wait(timeout=max(0.01, min(exit_timeout, deadline - time.monotonic())))
                    if code != 0 or result is None or completed != len(names):
                        raise RuntimeError("环境检查进程未正常完成，请查看 hardware-preflight.log 中最后一个步骤。")
                    receipt.update(status="passed", active_step=None, cuda_fp16_invocation="PROVEN", hardware=result)
                    return result
                if output_bytes < 1024 * 1024:
                    remaining = 1024 * 1024 - output_bytes
                    fragment = line.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
                    log.write(fragment)
                    log.flush()
                    output_bytes += len(fragment.encode("utf-8"))
                if not line.startswith(PREFIX):
                    continue
                message = json.loads(line[len(PREFIX):])
                kind = message.get("event")
                if kind == "start":
                    name = message.get("stage")
                    if active is not None or completed >= len(names) or name != names[completed] or result is not None:
                        raise RuntimeError("环境检查步骤顺序无效。")
                    active = name
                    stage_started = time.monotonic()
                    label, timeout = steps[name]
                    deadline = stage_started + timeout
                    receipt.update(active_step=name, step_timeout_seconds=timeout)
                    notify("hardware", f"正在{label}…", operation=name, operation_timeout_seconds=timeout)
                elif kind == "done":
                    if active is None or message.get("stage") != active:
                        raise RuntimeError("环境检查完成回执与当前步骤不符。")
                    receipt["steps"].append({"step": active, "seconds": round(time.monotonic() - stage_started, 3)})
                    completed += 1
                    active = None
                    deadline = time.monotonic() + startup_timeout
                    receipt["active_step"] = "worker_exit" if completed == len(names) else "worker_start"
                elif kind == "result":
                    hardware = message.get("hardware")
                    if (result is not None or active is not None or completed != len(names)
                            or not isinstance(hardware, dict) or hardware.get("cuda_fp16_invocation") != "PROVEN"
                            or hardware.get("torch") != "2.6.0+cu124" or hardware.get("tensorrt") != "10.4.0"
                            or "4050" not in str(hardware.get("gpu", "")) or not hardware.get("gpu_uuid")):
                        raise RuntimeError("环境检查结果不完整或与目标运行环境不符。")
                    result = hardware
                    receipt["active_step"] = "worker_exit"
                    deadline = time.monotonic() + exit_timeout
                else:
                    raise RuntimeError("环境检查回执类型无效。")
                _save(summary_path, receipt)
        except Exception as error:
            receipt.update(status="failed", error=str(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    receipt["worker_exit_pending"] = True
            if reader is not None:
                reader.join(timeout=0.2)
                if not reader.is_alive():
                    child.stdout.close()
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            _save(summary_path, receipt)


def hardware_preflight(root, notify):
    return supervise(root, [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--probe"], notify)


if __name__ == "__main__":
    if sys.argv[1:] != ["--probe"]:
        raise SystemExit("Use the VisionCortex launcher to supervise this hardware check.")
    try:
        run_probe()
    except Exception as error:
        print(f"Hardware probe failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
