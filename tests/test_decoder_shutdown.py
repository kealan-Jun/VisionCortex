"""Process-lifecycle evidence only; no models or production media involved."""
import os
import shutil
import signal
import subprocess
import sys

import pytest

from visioncortex.owned_subprocess import _pidfd_open, decoder_shutdown_guard


pytestmark = pytest.mark.skipif(
    not shutil.which('ffmpeg') or not sys.platform.startswith('linux')
    or not hasattr(signal, 'pidfd_send_signal'), reason='Linux pidfds and ffmpeg required')


@pytest.fixture(autouse=True)
def require_pidfd():
    try:
        fd = _pidfd_open(os.getpid())
    except OSError:
        pytest.skip('native pidfds unavailable')
    else:
        os.close(fd)


def decoder():
    return subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=64x64:rate=1',
         '-f', 'rawvideo', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def cleanup(process):
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=5)


def test_shutdown_unblocks_owned_pipe_and_preserves_other_child():
    raw = decoder()
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        assert raw.stdout.read(1)  # Child has entered its raw-frame producer.
        with decoder_shutdown_guard(grace_seconds=.05, terminate_seconds=.05):
            assert raw.wait(timeout=5) != 0
            assert other.poll() is None
    finally:
        cleanup(raw)
        cleanup(other)


def test_completed_graceful_shutdown_does_not_cancel_decoder():
    raw = decoder()
    try:
        assert raw.stdout.read(1)
        with decoder_shutdown_guard(grace_seconds=10):
            pass
        assert raw.poll() is None
    finally:
        cleanup(raw)


def test_guard_does_not_target_another_process_decoder():
    script = '''import subprocess, sys
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
 'color=size=64x64:rate=1', '-f', 'rawvideo', 'pipe:1'],
 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
try:
 p.stdout.read(1)
 print('ready', flush=True)
 sys.stdin.readline()
 print(p.poll(), flush=True)
finally:
 p.kill()
 p.communicate(timeout=5)
'''
    owner = subprocess.Popen([sys.executable, '-u', '-c', script],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    raw = decoder()
    try:
        assert owner.stdout.readline().strip() == 'ready'
        assert raw.stdout.read(1)
        with decoder_shutdown_guard(grace_seconds=.05, terminate_seconds=.05):
            raw.wait(timeout=5)
        owner.stdin.write('check\n')
        owner.stdin.flush()
        assert owner.stdout.readline().strip() == 'None'
        owner.wait(timeout=5)
    finally:
        cleanup(raw)
        cleanup(owner)
