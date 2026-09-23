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
