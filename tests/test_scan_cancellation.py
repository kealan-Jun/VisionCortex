"""A failed consumer must release decoders and never commit unfinished chunks."""
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from visioncortex.detection import scan_videos
from visioncortex.runtime_control import ExecutionCancelled, execution_context
from visioncortex.schemas import AlignmentTransform, VideoInfo, ViewInput, ViewRole


@pytest.mark.parametrize('operation', ['get', 'put'])
def test_cancel_wakes_full_or_empty_queue_without_cancelling_siblings(operation):
    from visioncortex.decode_buffers import CancellableQueue
    from visioncortex.runtime_control import CancellationSignal
    parent = threading.Event()
    stop, sibling = CancellationSignal(parent), CancellationSignal(parent)
    output = CancellableQueue(1, stop)
    if operation == 'put':
        output.put('full')
    errors = []

    def blocked():
        try:
            output.put('next') if operation == 'put' else output.get()
        except ExecutionCancelled:
            errors.append('cancelled')

    worker = threading.Thread(target=blocked, daemon=True)
    worker.start()
    stop.set()
    worker.join(1)
    assert not worker.is_alive() and errors == ['cancelled']
    assert not parent.is_set() and not sibling.is_set()
    parent.set()
    assert sibling.wait(.1)


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
def test_inference_failure_closes_blocked_producers(monkeypatch, tmp_path, default_config, phase):
    closed, started, release = [], [], threading.Event()
    output_queues = []
    from visioncortex import detection
    real_producer = detection._producer

    def producer(*args):
        output_queues.append(args[2])
        return real_producer(*args)

    def decode(*args, **kwargs):
        start = args[2]
        started.append(start)
        try:
            for i in range(200):
                if release.is_set():
                    return
                yield i, start + i * 500, np.zeros((8, 8, 3), np.uint8)
        finally:
            closed.append(start)

    class Scanner:
        def __init__(self, *_a, **_k):
            self.model_path = Path('fake.engine')
            self.batch_size = self.engine_build_batch = 1
            self.last_inference_batch_sizes = []

        def infer(self, packets):
            raise RuntimeError('injected inference failure')

        def close(self):
            pass

    monkeypatch.setattr(detection, '_producer', producer)
    monkeypatch.setattr(detection, 'iter_view_sampled_frames', decode)
    monkeypatch.setattr(detection, 'RoleScanner', Scanner)
    config = default_config
    config['performance'].update(decode_queue_depth=1, fine_third_person_decode_workers=2,
                                 fine_decode_prefetch_frames=1, fine_chunk_seconds=1)
    view = ViewInput(view_id='cancel-test', role=ViewRole.THIRD_PERSON, video=Path('fake.mp4'))
    info = VideoInfo(path=view.video, duration_ms=3000, fps=30, width=8, height=8, frame_count=90)
    transform = AlignmentTransform(view_id=view.view_id, reference_view_id=view.view_id,
                                   state='aligned', confidence=1)
    try:
        with pytest.raises(RuntimeError, match='injected inference failure'):
            scan_videos([view], {view.view_id: info}, {view.view_id: transform}, tmp_path,
                        config, phase=phase, windows={view.view_id: [(0, 3000)]})
        assert started and sorted(started) == sorted(closed)
        assert not any(t.name == 'decode-cancel-test' or t.name.startswith('fine-prefetch-cancel-test')
                       for t in threading.enumerate())
        checkpoint = tmp_path / 'cancel-test.checkpoint.json'
        assert not checkpoint.exists() or not json.loads(checkpoint.read_text())['completed_chunks']
    finally:
        # Also clean up the deliberately broken baseline after its assertion fails.
        release.set()
        deadline = time.monotonic() + 2
        while sorted(started) != sorted(closed) and time.monotonic() < deadline:
            for output in output_queues:
                try:
                    output.get(timeout=.02)
                except (queue.Empty, ExecutionCancelled):
                    pass


def test_decoder_context_inherits_job_cancellation(monkeypatch, tmp_path, default_config):
    from visioncortex import detection
    from visioncortex.runtime_control import CURRENT
    stop, entered, exited = threading.Event(), threading.Event(), threading.Event()
    captured = []

    def producer(view, info, output, *_args):
        captured.append(CURRENT.get())
        entered.set()
        try:
            while not stop.wait(.02):
                pass
        finally:
            exited.set()

    monkeypatch.setattr(detection, '_producer', producer)
    view = ViewInput(view_id='empty-cancel', role=ViewRole.THIRD_PERSON, video=Path('fake.mp4'))
    info = VideoInfo(path=view.video, duration_ms=1000, fps=30, width=8, height=8, frame_count=30)
    transform = AlignmentTransform(view_id=view.view_id, reference_view_id=view.view_id,
                                   state='aligned', confidence=1)
    errors = []

    def scan():
        try:
            with execution_context(job_id='job-123', source='device_day', priority=7, stop=stop):
                scan_videos([view], {view.view_id: info}, {view.view_id: transform}, tmp_path,
                            default_config, phase='motion_probe')
        except ExecutionCancelled:
            errors.append('cancelled')

    worker = threading.Thread(target=scan, daemon=True)
    worker.start()
    assert entered.wait(5)
    stop.set()
    worker.join(3)
    assert exited.is_set() and not worker.is_alive()
    assert errors == ['cancelled']
    assert (captured[0].job_id, captured[0].source, captured[0].priority) == ('job-123', 'device_day', 7)


def test_background_yields_only_after_checkpoint_then_resumes_saved_chunks(monkeypatch, tmp_path, default_config):
    from visioncortex import detection
    from visioncortex.runtime_control import ExecutionYielded
    arrival = threading.Event()
    captured = []
    def producer(view, _info, output, completed, *_args):
        from visioncortex.runtime_control import CURRENT
        captured.append(CURRENT.get().yield_signal)
        for index in range(2):
            if index in completed:
                output.put(detection.ChunkEnd(view_id=view.view_id, chunk_index=index, total_chunks=2))
                continue
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            output.put(detection.FramePacket(view=view, frame_index=index, local_ms=index * 500,
                frame=frame, gray=frame[:, :, 0], previous_gray=None, motion_score=0.0))
            if not completed:
                arrival.set()
            output.put(detection.ChunkEnd(view_id=view.view_id, chunk_index=index, total_chunks=2))
        output.put(detection.ProducerEnd(view_id=view.view_id))
    monkeypatch.setattr(detection, '_producer', producer)
    view = ViewInput(view_id='yield-test', role=ViewRole.THIRD_PERSON, video=Path('fake.mp4'))
    info = VideoInfo(path=view.video, duration_ms=1000, fps=2, width=8, height=8, frame_count=2)
    transform = AlignmentTransform(view_id=view.view_id, reference_view_id=view.view_id,
                                   state='aligned', confidence=1)
    with execution_context(source='device_day_backfill', yield_signal=arrival), pytest.raises(ExecutionYielded):
        scan_videos([view], {view.view_id: info}, {view.view_id: transform}, tmp_path,
                    default_config, phase='motion_probe')
    checkpoint = tmp_path/'yield-test.checkpoint.json'
    assert json.loads(checkpoint.read_text())['completed_chunks'] == [0]
    ledger = tmp_path/'yield-test.detections.jsonl'
    committed = ledger.read_bytes()
    assert len(committed.splitlines()) == 1 and captured == [arrival]
    arrival.clear()
    with execution_context(source='device_day_backfill', yield_signal=arrival):
        output = scan_videos([view], {view.view_id: info}, {view.view_id: transform}, tmp_path,
                            default_config, phase='motion_probe')
    assert output[view.view_id].read_bytes().startswith(committed)
    assert len(output[view.view_id].read_bytes().splitlines()) == 2
    assert json.loads(checkpoint.read_text())['completed_chunks'] == [0, 1]
    class ExpiredQuantum:
        reason = 'quantum_elapsed'
        def is_set(self, *, ignore_quantum=False):
            return not ignore_quantum
    with execution_context(source='device_day_backfill', yield_signal=ExpiredQuantum()):
        replay = scan_videos([view], {view.view_id: info}, {view.view_id: transform}, tmp_path,
                             default_config, phase='motion_probe')
    assert replay[view.view_id].read_bytes() == output[view.view_id].read_bytes()
    assert not any(t.name == 'decode-yield-test' for t in threading.enumerate())


def test_yielded_source_probe_cannot_be_mistaken_for_successful_eof(monkeypatch, tmp_path, default_config):
    from visioncortex import detection
    from visioncortex.runtime_control import ExecutionYielded
    def decode(*args, **kwargs):
        raise ExecutionYielded('Synthetic native probe yielded')
        yield  # Generator interface without running FFmpeg or a model.
    monkeypatch.setattr(detection, 'iter_view_sampled_frames', decode)
    view = ViewInput(view_id='probe-yield', role=ViewRole.THIRD_PERSON, video=Path('fake.mp4'))
    info = VideoInfo(path=view.video, duration_ms=1000, fps=2, width=8, height=8, frame_count=2)
    transform = AlignmentTransform(view_id=view.view_id, reference_view_id=view.view_id,
                                   state='aligned', confidence=1)
    with execution_context(source='device_day_backfill', yield_signal=threading.Event()), pytest.raises(ExecutionYielded):
        scan_videos([view], {view.view_id: info}, {view.view_id: transform}, tmp_path,
                    default_config, phase='motion_probe')
    assert not (tmp_path/'probe-yield.checkpoint.json').exists()
    assert not any(t.name == 'decode-probe-yield' for t in threading.enumerate())
