#!/usr/bin/env python3
"""Host-local GPU lifetime lock shared by OCR and VisionCortex (Linux/WSL2).
No model is loaded here. All services using a GPU must share this lock directory.
"""

import argparse
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def verify_idle_gpu(gpu):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    if gpu not in result.stdout.splitlines():
        raise RuntimeError("Configured GPU UUID is not present")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    if gpu in result.stdout.splitlines():
        raise RuntimeError(
            "GPU has an existing compute process; no process was stopped"
        )


def run_guard(command, gpu, lock_dir, *, check_gpu=verify_idle_gpu):
    if os.name != "posix":
        raise RuntimeError("Use Linux/WSL2 for the GPU service launcher")
    import fcntl

    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu):
        raise ValueError("Use the actual GPU UUID")
    root = Path(lock_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root.resolve() / (gpu + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    proc = None
    previous = {}
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        check_gpu(gpu)
        env = os.environ | {
            "CUDA_VISIBLE_DEVICES": gpu,
            "VISIONCORTEX_GPU_UUID": gpu,
            "VISIONCORTEX_GPU_GUARD_FD": str(fd),
            "VISIONCORTEX_GPU_LOCK_PATH": str(lock_path),
            "REALITYLOOP_GPU_UUID": gpu,
            "REALITYLOOP_GPU_GUARD_FD": str(fd),
            "REALITYLOOP_GPU_LOCK_PATH": str(lock_path),
        }
        proc = subprocess.Popen(
            command, env=env, pass_fds=(fd,), start_new_session=True
        )
        deadline = [None]

        def terminate(*_):
            if deadline[0] is None:
                deadline[0] = time.monotonic() + 30
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, terminate)
        while proc.poll() is None:
            if deadline[0] is not None and time.monotonic() >= deadline[0]:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.1)
        # Do not release the shared lock while orphan descendants in this group live.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        while True:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        # The process group is gone before another cooperating service can acquire.
        return proc.returncode
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--lock-dir", required=True)
    parser.add_argument("--config", type=Path, help="Verify a prepared pipeline before GPU access")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Worker command is required after --")
    try:
        if args.config:
            from visioncortex.config import load_config, require_configured_site
            require_configured_site(load_config(args.config))
        return run_guard(command, args.gpu, args.lock_dir)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        # No underlying exception text: a subprocess error may contain credentials.
        print("GPU launcher failed: " + type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
