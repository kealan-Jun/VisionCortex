"""Bound actual CUDA decoder children across processes sharing one runtime/GPU.

Linux children inherit the locked descriptor. Closing the parent's descriptor
never explicitly unlocks it while a child may still be alive. Other programs
and runtimes are outside this cooperative quota; unsupported OSes use CPU.
"""
from functools import wraps
import inspect
import os
from pathlib import Path
import subprocess
import threading


def _supports_fd_inheritance():
    return os.name == 'posix'


def validate(config):
    value = config.get('performance', {}).get('cuda_decode_process_limit', 6)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64:
        raise ValueError('cuda_decode_process_limit must be an integer in [1,64]')
    device = config.get('performance', {}).get('cuda_decode_device', 0)
    if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device <= 31:
        raise ValueError('cuda_decode_device must be an integer in [0,31]')


class DecoderPermit:
    def __init__(self, fd=None, *, device=0, reason=None):
        self.fd, self.device, self.reason = fd, device, reason
        self.process = None
        self._finished = threading.Event()
        self._watcher = None

    def popen(self, command, **kwargs):
        from .runtime_control import CURRENT, check_cancelled
        stop = CURRENT.get().stop
        check_cancelled(stop)
        if self.fd is not None:
            kwargs['pass_fds'] = (self.fd,)
        self.process = subprocess.Popen(command, **kwargs)
        if stop is not None:
            # A pipe read cannot check a token. Reap this permit's child to
            # unblock it; never signal unrelated decoders or release its fd here.
            def cancel_child():
                while not self._finished.wait(.05):
                    if stop.is_set():
                        self._terminate_child()
                        return
            self._watcher = threading.Thread(target=cancel_child, daemon=True,
                                            name='decoder-cancellation')
            self._watcher.start()
        return self.process

    def _terminate_child(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=2)

    def close(self):
        try:
            self._finished.set()
            if self._watcher is not None:
                self._watcher.join(timeout=5)
            self._terminate_child()
        finally:
            # No LOCK_UN: the inherited open-file description remains locked
            # until the child exits even if termination/wait failed here.
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


class CudaDecodeAdmission:
    def __init__(self, root, capacity=6, device=0):
        validate({'performance': {'cuda_decode_process_limit': capacity, 'cuda_decode_device': device}})
        self.root = Path(root).absolute() / 'state' / 'cuda-decoders' / str(device)
        self.capacity, self.device = capacity, device

    @classmethod
    def from_config(cls, config):
        perf = config.get('performance', {})
        return cls(config['storage']['local_runtime_root'], perf.get('cuda_decode_process_limit', 6),
                   perf.get('cuda_decode_device', 0))

    def acquire(self):
        if not _supports_fd_inheritance():
            return DecoderPermit(reason='cuda_quota_unsupported_platform')
        import fcntl
        self.root.mkdir(parents=True, exist_ok=True)
        control = os.open(self.root / 'capacity.lock', os.O_RDWR | os.O_CREAT, 0o600)
        held = []
        try:
            try:
                fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return DecoderPermit(reason='cuda_quota_coordinator_busy')
            raw = os.read(control, 32)
            previous = int(raw) if raw else self.capacity
            if not 1 <= previous <= 64:
                raise ValueError('Invalid persisted CUDA decoder capacity')
            # Different live capacities must not create disjoint slot ranges.
            for index in range(max(previous, self.capacity)):
                fd = os.open(self.root / f'{index}.lock', os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    if previous != self.capacity:
                        return DecoderPermit(reason='cuda_quota_capacity_conflict')
                    continue
                except BaseException:
                    os.close(fd)
                    raise
                held.append(fd)
                if previous == self.capacity:
                    break
            if previous != self.capacity or not raw:
                os.lseek(control, 0, os.SEEK_SET)
                os.ftruncate(control, 0)
                os.write(control, str(self.capacity).encode())
            if held:
                return DecoderPermit(held.pop(0), device=self.device)
            return DecoderPermit(reason='cuda_decode_process_limit')
        finally:
            for fd in held:
                os.close(fd)
            os.close(control)


def managed_cuda_decoder(function):
    """Reserve before Popen, propagate CPU fallback, release after child exit."""
    signature = inspect.signature(function)
    @wraps(function)
    def iterate(*args, decoder_admission=None, **kwargs):
        from .runtime_control import check_cancelled
        check_cancelled()
        bound = signature.bind(*args, **kwargs)
        requested = bound.arguments.get('hwaccel')
        permit = (decoder_admission.acquire() if requested == 'cuda' and decoder_admission is not None
                  else DecoderPermit(reason='cuda_quota_unconfigured' if requested == 'cuda' else None))
        try:
            actual = requested if requested != 'cuda' or permit.fd is not None else None
            bound.arguments['hwaccel'] = actual
            if actual != 'cuda':
                bound.arguments['cuda_scale'] = False
            bound.arguments['decoder_lease'] = permit
            receipt = bound.arguments.get('receipt')
            if receipt is not None:
                receipt.update(actual_hwaccel=actual, actual_cuda_scale=bool(bound.arguments.get('cuda_scale')),
                               actual_decoder_backend=actual or 'cpu',
                               cuda_admission_reason=permit.reason,
                               cuda_decode_process_limit=decoder_admission.capacity if decoder_admission else None,
                               cuda_decode_device=permit.device if actual == 'cuda' else None,
                               cuda_admission_scope='same_local_runtime_and_cuda_device')
                receipt.setdefault('decoder_attempt_backends', []).append(actual or 'cpu')
            check_cancelled()
            frames = function(*bound.args, **bound.kwargs)
            try:
                for frame in frames:
                    check_cancelled()
                    yield frame
                check_cancelled()
            except Exception:
                # Cancellation can interrupt a blocking read with FFmpeg's
                # nonzero exit; do not mistake that for a GPU fallback request.
                check_cancelled()
                raise
            finally:
                frames.close()
        finally:
            permit.close()
    return iterate


def wait_decoder(process):
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        return process.wait(timeout=2)
