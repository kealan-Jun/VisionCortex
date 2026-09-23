"""CPU-only process/admission tests. No CUDA context or source media is opened."""
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from visioncortex import cuda_decode_admission as admission, video_io
from visioncortex.runtime_control import ExecutionCancelled, execution_context
from visioncortex.schemas import VideoInfo, VideoSegmentInfo, ViewInput, ViewRole


pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX descriptor leases')


def test_concurrent_workers_across_admission_instances_share_total(tmp_path):
    barrier = threading.Barrier(12)

    def acquire(_):
        barrier.wait(timeout=5)
        return admission.CudaDecodeAdmission(tmp_path, 3).acquire()

    with ThreadPoolExecutor(max_workers=12) as executor:
        permits = list(executor.map(acquire, range(12)))
    try:
        granted = sum(p.fd is not None for p in permits)
        assert 1 <= granted <= 3  # Coordinator contention also falls back to CPU.
        policy = admission.CudaDecodeAdmission(tmp_path, 3)
        permits.extend(policy.acquire() for _ in range(3 - granted))
        assert sum(p.fd is not None for p in permits) == 3
        assert policy.acquire().reason == 'cuda_decode_process_limit'
    finally:
        for permit in permits:
            permit.close()
    permit = policy.acquire()
    assert permit.fd is not None
    permit.close()


def test_other_process_and_inherited_child_keep_slot_until_exit(tmp_path):
    policy = admission.CudaDecodeAdmission(tmp_path, 1)
    permit = policy.acquire()
    # This CPU child inherits exactly the descriptor passed to FFmpeg in production.
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                             stdin=subprocess.PIPE, pass_fds=(permit.fd,))
    try:
        permit.close()  # No explicit flock unlock: child remains the owner.
        script = ('import sys; from visioncortex.cuda_decode_admission import CudaDecodeAdmission; '
                  'p=CudaDecodeAdmission(sys.argv[1],1).acquire(); '
                  'print(p.reason); p.close()')
        result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                                capture_output=True, text=True, check=True, timeout=10)
        assert result.stdout.strip() == 'cuda_decode_process_limit'
        assert policy.acquire().fd is None
    finally:
        child.communicate(timeout=5)
        permit.close()
    successor = policy.acquire()
    assert successor.fd is not None
    successor.close()


def test_capacity_change_cannot_bypass_live_slot_and_gpu_has_own_namespace(tmp_path):
    old = admission.CudaDecodeAdmission(tmp_path, 1)
    held = old.acquire()
    expanded = admission.CudaDecodeAdmission(tmp_path, 2)
    try:
        assert expanded.acquire().reason == 'cuda_quota_capacity_conflict'
        other_gpu = admission.CudaDecodeAdmission(tmp_path, 1, device=1).acquire()
        assert other_gpu.fd is not None
        other_gpu.close()
    finally:
        held.close()
    permits = [expanded.acquire(), expanded.acquire()]
    try:
        assert all(p.fd is not None for p in permits)
        assert old.acquire().reason == 'cuda_quota_capacity_conflict'
    finally:
        for permit in permits:
            permit.close()


@pytest.mark.parametrize('value', [0, -1, 65, True, 1.5, '6'])
def test_invalid_limits_fail_closed(tmp_path, value):
    with pytest.raises(ValueError, match='cuda_decode_process_limit'):
        admission.CudaDecodeAdmission(tmp_path, value)


class FakeProcess:
    def __init__(self, data):
        self.stdout = io.BytesIO(data)
        self.stderr = io.BytesIO()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.fixture
def decoder(monkeypatch):
    class Trace:
        enabled = False

        def __init__(self, *args):
            pass

        def filter_prefix(self):
            return ''

        def input_options(self):
            return []

        def output_options(self):
            return []

        def start(self, stream):
            pass

        def identity(self, index, frame):
            return None

        def finish(self, stream):
            return stream.read()

    calls = []

    def popen(command, **kwargs):
        process = FakeProcess(bytes(2 * 2 * 3 * 2))
        calls.append((command, kwargs, process))
        return process

    monkeypatch.setattr(video_io, 'SourceFrameTrace', Trace)
    monkeypatch.setattr(video_io.subprocess, 'Popen', popen)
    monkeypatch.setattr(video_io.shutil, 'which', lambda _: '/test/ffmpeg')
    return calls


def frames(policy, receipt, *, hwaccel='cuda'):
    info = VideoInfo(path=Path('not-opened.mp4'), duration_ms=200, fps=30,
                     width=2, height=2, frame_count=6)
    return video_io.iter_sampled_frames(info.path, info, 0, 200, 10, 2,
                                       hwaccel, False, 2, cuda_scale=True,
                                       decoder_admission=policy, receipt=receipt)


def test_quota_overflow_uses_cpu_and_preserves_sample_times(decoder, tmp_path):
    policy = admission.CudaDecodeAdmission(tmp_path, 1, device=2)
    held = policy.acquire()
    receipt = {}
    try:
        output = list(frames(policy, receipt))
    finally:
        held.close()
    command, options, process = decoder[0]
    assert '-hwaccel' not in command and 'pass_fds' not in options
    assert 'scale_cuda' not in command[command.index('-vf') + 1]
    assert command[command.index('-threads') + 1] == '2'
    assert [item[1] for item in output] == [0, 100]
    assert receipt['actual_hwaccel'] is None
    assert receipt['actual_decoder_backend'] == 'cpu'
    assert receipt['cuda_admission_reason'] == 'cuda_decode_process_limit'
    assert receipt['actual_cuda_scale'] is False
    assert process.poll() == 0


def test_early_close_terminates_and_reaps_before_releasing(decoder, tmp_path):
    policy = admission.CudaDecodeAdmission(tmp_path, 1, device=2)
    receipt = {}
    iterator = frames(policy, receipt)
    next(iterator)
    command, options, process = decoder[0]
    assert command[command.index('-hwaccel_device') + 1] == '2'
    assert len(options['pass_fds']) == 1
    assert policy.acquire().fd is None
    iterator.close()
    assert process.terminated and process.poll() is not None
    assert receipt['actual_hwaccel'] == 'cuda'
    replacement = policy.acquire()
    assert replacement.fd is not None
    replacement.close()


def test_cancellation_closes_child_without_starting_fallback(decoder, tmp_path):
    policy = admission.CudaDecodeAdmission(tmp_path, 1)
    stop = threading.Event()
    with execution_context(stop=stop):
        iterator = frames(policy, {})
        next(iterator)
        stop.set()
        with pytest.raises(ExecutionCancelled):
            next(iterator)
    assert len(decoder) == 1 and decoder[0][2].terminated
    permit = policy.acquire()
    assert permit.fd is not None
    permit.close()


def test_popen_failure_returns_permit(monkeypatch, decoder, tmp_path):
    def fail(*args, **kwargs):
        raise OSError('fixture startup failure')

    monkeypatch.setattr(video_io.subprocess, 'Popen', fail)
    policy = admission.CudaDecodeAdmission(tmp_path, 1)
    with pytest.raises(OSError, match='fixture startup failure'):
        list(frames(policy, {}))
    permit = policy.acquire()
    assert permit.fd is not None
    permit.close()


def test_cancel_after_reserving_never_starts_child(monkeypatch, decoder, tmp_path):
    policy = admission.CudaDecodeAdmission(tmp_path, 1)
    stop = threading.Event()
    acquire = policy.acquire

    def reserve_then_cancel():
        permit = acquire()
        stop.set()
        return permit

    monkeypatch.setattr(policy, 'acquire', reserve_then_cancel)
    with execution_context(stop=stop), pytest.raises(ExecutionCancelled):
        list(frames(policy, {}))
    assert not decoder
    permit = acquire()
    assert permit.fd is not None
    permit.close()


def test_unconfigured_and_unsupported_paths_use_cpu(decoder, tmp_path, monkeypatch):
    for policy in (None, admission.CudaDecodeAdmission(tmp_path, 1)):
        monkeypatch.setattr(admission, '_supports_fd_inheritance', lambda: False)
        receipt = {}
        list(frames(policy, receipt))
        assert receipt['actual_hwaccel'] is None
        assert receipt['cuda_admission_reason'] in {
            'cuda_quota_unconfigured', 'cuda_quota_unsupported_platform'}
        assert '-hwaccel' not in decoder[-1][0]


def test_persistent_session_keeps_actual_cpu_receipt(decoder, tmp_path):
    policy = admission.CudaDecodeAdmission(tmp_path, 1)
    held = policy.acquire()
    segment = VideoSegmentInfo(path=Path('not-opened.mp4'), virtual_start_ms=1000,
                               virtual_end_ms=1200, duration_ms=200, fps=30,
                               width=2, height=2, frame_count=6, frame_start_index=30,
                               size_bytes=1)
    info = VideoInfo(path=segment.path, duration_ms=1200, fps=30, width=2, height=2,
                     frame_count=36, segments=[segment])
    session = video_io.plan_physical_segment_decode_sessions(info, [(1000, 1200)])[0]
    receipt = {}
    try:
        output = list(video_io.iter_physical_segment_session_frames(
            info, session, 10, 2, 'cuda', 2, False, receipt, decoder_admission=policy))
    finally:
        held.close()
    assert [item[1] for item in output] == [1000, 1100]
    assert receipt['actual_hwaccel'] is None
    assert receipt['actual_decoder_backend'] == 'cpu'
    assert receipt['frame_accounting_mismatch'] == 0
    assert '-hwaccel' not in decoder[0][0]


def test_sparse_benchmark_uses_same_quota_and_reports_actual_backend(decoder, tmp_path):
    policy = admission.CudaDecodeAdmission(tmp_path, 1)
    held = policy.acquire()
    info = VideoInfo(path=Path('not-opened.mp4'), duration_ms=200, fps=30,
                     width=2, height=2, frame_count=6)
    view = ViewInput(view_id='fp', role=ViewRole.FIRST_PERSON, video=info.path)
    try:
        report = video_io.benchmark_sparse_decode_strategy(
            view, info, sample_fps=10, max_width=2, hwaccel='cuda',
            decoder_threads=2, cuda_scale=False, benchmark_seconds=10,
            decoder_admission=policy)
    finally:
        held.close()
    assert all(item['decoder_receipt']['actual_decoder_backend'] == 'cpu'
               for item in report['strategies'])
    assert all('-hwaccel' not in command for command, _, _ in decoder)


def test_wait_timeout_kills_and_waits_again():
    process = FakeProcess(b'')
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        if not process.killed:
            raise subprocess.TimeoutExpired('fixture', timeout)
        return process.returncode

    process.wait = wait
    assert admission.wait_decoder(process) == -9
    assert waits == [5, 2] and process.killed


def test_failed_termination_cannot_release_live_inherited_child(tmp_path):
    policy = admission.CudaDecodeAdmission(tmp_path, 1)
    permit = policy.acquire()
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                             stdin=subprocess.PIPE, pass_fds=(permit.fd,))
    # Simulate a process that has not finished even after terminate/kill/wait.
    # Its inherited descriptor must still prevent any replacement CUDA child.
    process = FakeProcess(b'')
    process.terminate = lambda: None
    process.kill = lambda: None

    def wait(timeout=None):
        raise subprocess.TimeoutExpired('fixture', timeout)

    process.wait = wait
    permit.process = process
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            permit.close()
        assert permit.fd is None
        assert child.poll() is None
        assert policy.acquire().fd is None
    finally:
        child.communicate(timeout=5)
        permit.process = None
        permit.close()
        process.stdout.close()
        process.stderr.close()
    successor = policy.acquire()
    assert successor.fd is not None
    successor.close()
