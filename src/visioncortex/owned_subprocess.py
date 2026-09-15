"""Cancellation and timeout for subprocesses owned by this invocation only."""
import os
import signal
import subprocess
import time
from .runtime_control import check_cancelled


def run(args, *, timeout, capture_output=False, check=False, **kwargs):
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    kwargs.setdefault('start_new_session', os.name != 'nt')
    child = subprocess.Popen(args, **kwargs)
    deadline = time.monotonic()+timeout
    try:
        while True:
            check_cancelled()
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(args, timeout)
            try:
                out, err = child.communicate(timeout=min(.2, max(.001, deadline-time.monotonic())))
                result = subprocess.CompletedProcess(args, child.returncode, out, err)
                if check:
                    result.check_returncode()
                return result
            except subprocess.TimeoutExpired:
                continue
    finally:
        if child.poll() is None:
            try:
                if os.name == 'nt':
                    child.terminate()
                else:
                    os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # Owned process exited between poll and termination.
            try:
                child.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == 'nt':
                        child.kill()
                    else:
                        os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.communicate()
