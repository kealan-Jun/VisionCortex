"""Cancellation and timeout for subprocesses owned by this invocation only."""
import os
import signal
import subprocess
import time
from contextlib import contextmanager
import ctypes
from functools import lru_cache
import logging
from pathlib import Path
import platform
import threading
from .runtime_control import check_cancelled


@lru_cache(maxsize=1)
def _native_pidfd_open():
    # Conda may bundle an older libc while the host kernel supports pidfds.
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'pidfd_open', None)
    if function is None:
        if (platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'aarch64'}
                or ctypes.sizeof(ctypes.c_void_p) != 8):
            raise OSError('native pidfds unavailable')
        # Linux x86-64 and asm-generic AArch64 syscall ABI: __NR_pidfd_open=434.
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        return lambda pid, flags: syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(flags))
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    return function


def _pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    fd = _native_pidfd_open()(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _owned_decoder_handles():
    """Pin direct pipe-only decoders; never target encoders or another service."""
    if not hasattr(signal, 'pidfd_send_signal'):
        return []
    parent = os.getpid()
    handles = []
    seen = set()
    for task in Path('/proc/self/task').glob('*'):
        try:
            children = task.joinpath('children').read_text().split()
        except OSError:
            continue
        for item in children:
            pid = int(item)
            if pid in seen:
                continue
            seen.add(pid)
            fd = None
            try:
                # Hold the identity before inspection, preventing PID-reuse kills.
                fd = _pidfd_open(pid)
                process = Path(f'/proc/{pid}')
                args = process.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
                owner = next(line.split()[1] for line in process.joinpath('status').read_text().splitlines()
                             if line.startswith('PPid:'))
                raw_pipe = any(args[i:i+2] == [b'-f', b'rawvideo'] for i in range(len(args)-1))
                if (int(owner) == parent and Path(os.fsdecode(args[0])).name == 'ffmpeg'
                        and raw_pipe and args[-1] == b'pipe:1'
                        and os.readlink(process / 'fd/1').startswith('pipe:')):
                    handles.append((pid, fd))
                    fd = None
            except (OSError, StopIteration, ValueError):
                pass
            finally:
                if fd is not None:
                    os.close(fd)
    return handles


@contextmanager
def decoder_shutdown_guard(*, grace_seconds=60, terminate_seconds=2):
    """Bound shutdown of owned raw-frame pipes after allowing workers to drain.

    Linux pidfds are required. Other platforms keep their normal shutdown path.
    This guard is activated only during service shutdown, never during analysis.
    """
    finished = threading.Event()

    def reap():
        if finished.wait(grace_seconds):
            return
        while not finished.is_set():
            handles = _owned_decoder_handles()
            try:
                for pid, fd in handles:
                    try:
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                        logging.getLogger(__name__).warning('Shutdown draining owned decoder pid=%s', pid)
                    except ProcessLookupError:
                        pass
                # Complete this owned termination even if the caller just drained.
                finished.wait(terminate_seconds)
                for _, fd in handles:
                    try:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            finally:
                for _, fd in handles:
                    os.close(fd)
            finished.wait(1)

    thread = threading.Thread(target=reap, name='decoder-shutdown', daemon=True)
    thread.start()
    try:
        yield
    finally:
        finished.set()
        thread.join(timeout=terminate_seconds + 2)


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
