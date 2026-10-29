"""Bound both output memory and subprocess lifetime without exposing worker stderr."""
from __future__ import annotations

import os
import selectors
import subprocess
import tempfile
import time


def run_bounded(command, request, *, env, timeout, max_bytes=4 * 1024**2):
    with tempfile.TemporaryFile() as source:
        source.write(request.encode())
        source.seek(0)
        process = subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True)
        output = bytearray()
        total, deadline = 0, time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as poll:
                for pipe in (process.stdout, process.stderr):
                    poll.register(pipe, selectors.EVENT_READ)
                while poll.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Customer worker time budget exceeded")
                    for key, _ in poll.select(min(remaining, .5)):
                        block = os.read(key.fileobj.fileno(), 65536)
                        if not block:
                            poll.unregister(key.fileobj)
                            continue
                        total += len(block)
                        if total > max_bytes:
                            raise ValueError("Customer worker output exceeded the memory bound")
                        if key.fileobj is process.stdout:
                            output.extend(block)
            code = process.wait(timeout=max(.1, deadline - time.monotonic()))
            if code:
                raise RuntimeError("Customer worker exited unsuccessfully; verify its isolated model environment")
            return output.decode("utf-8")
        finally:
            import signal
            # Also terminate owned child model/decoder processes, including a child holding pipes open.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            process.stdout.close()
            process.stderr.close()
