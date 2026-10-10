"""Exercise the real Windows stdin pipe and process ownership boundary."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows process ownership")
TOOLS = Path(__file__).resolve().parents[1] / "tools"
CHILD = """
import importlib.util, os, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location('tested_lifecycle', Path(sys.argv[1]) / 'windows_desktop_lifecycle.py')
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)
lifecycle.supervise_desktop_parent(int(sys.argv[2]))
time.sleep(0.2)  # Let the command reader enter its idle pipe read.
if sys.argv[4] == 'numpy':
    import numpy
Path(sys.argv[3]).write_text('ready')
while True:
    time.sleep(0.1)
"""


def launch(tmp_path, *, probe="none", interpreter=sys.executable):
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [interpreter, "-I", "-B", "-c", CHILD, str(TOOLS), str(os.getpid()), str(ready), probe],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return child, ready


def await_ready(child, ready, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and child.poll() is None:
        if ready.is_file():
            return
        time.sleep(0.05)
    pytest.fail("Desktop child did not become ready while its command pipe was idle")


def cleanup(child):
    if child.poll() is None:
        child.kill()
    child.wait(timeout=5)
    if child.stdin and not child.stdin.closed:
        child.stdin.close()
    child.stdout.close()


@pytest.mark.parametrize("chunks", [(b"stop\n",), (b"st", b"op\r\n"), (b"ignored\n", b"stop\n"), ()])
def test_stop_or_pipe_eof_exits(tmp_path, chunks):
    child, ready = launch(tmp_path)
    try:
        await_ready(child, ready)
        for chunk in chunks:
            child.stdin.write(chunk)
            child.stdin.flush()
            time.sleep(0.1)
        if not chunks:
            child.stdin.close()
        assert child.wait(timeout=5) == 0
    finally:
        cleanup(child)


def test_oversized_command_fails_closed(tmp_path):
    child, ready = launch(tmp_path)
    try:
        await_ready(child, ready)
        child.stdin.write(b"x" * 4097)
        child.stdin.flush()
        assert child.wait(timeout=5) == 1
    finally:
        cleanup(child)


def test_numpy_import_completes_with_idle_command_reader(tmp_path):
    interpreter = os.environ.get("VISIONCORTEX_TEST_PYTHON", sys.executable)
    child, ready = launch(tmp_path, probe="numpy", interpreter=interpreter)
    try:
        await_ready(child, ready)
        child.stdin.write(b"stop\n")
        child.stdin.flush()
        assert child.wait(timeout=5) == 0
    finally:
        cleanup(child)
