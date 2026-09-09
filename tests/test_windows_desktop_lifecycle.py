"""Native Windows regression for an idle command pipe during NumPy imports."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows process ownership")
TOOLS = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = """
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from windows_desktop_lifecycle import supervise_desktop_parent
supervise_desktop_parent(int(sys.argv[2]))
time.sleep(0.2)
import numpy
Path(sys.argv[3]).write_text('ready')
while True:
    time.sleep(0.1)
"""


@pytest.mark.parametrize("chunks", [(b"stop\n",), (b"st", b"op\r\n"), (), (b"x" * 4097,)])
def test_numpy_import_and_owned_pipe_shutdown(tmp_path, chunks):
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", SCRIPT, str(TOOLS), str(os.getpid()), str(ready)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + 20
        while not ready.is_file() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.is_file(), "NumPy import stalled while the command pipe was idle"
        for chunk in chunks:
            child.stdin.write(chunk)
            child.stdin.flush()
            time.sleep(0.1)
        if not chunks:
            child.stdin.close()
        assert child.wait(timeout=5) == (1 if chunks and len(chunks[0]) > 4096 else 0)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        if not child.stdin.closed:
            child.stdin.close()
