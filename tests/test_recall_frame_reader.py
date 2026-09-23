"""No media/model I/O: recall seeks keep their source times and close handles."""

import numpy as np
import pytest

from visioncortex import coarse_recall, video_io
from visioncortex.runtime_control import ExecutionCancelled
from visioncortex.schemas import BoxEvidence, FrameEvidence, VideoInfo, ViewInput, ViewRole


@pytest.fixture
def recall_case(tmp_path, default_config, monkeypatch):
    view = ViewInput(view_id='camera', role=ViewRole.FIRST_PERSON, video=tmp_path / 'Source.mp4')
    info = VideoInfo(path=view.video, duration_ms=10000, fps=30, width=32, height=32, frame_count=300)
    frames = [FrameEvidence(
        view_id=view.view_id, role=view.role, frame_index=int(ms * .03), local_ms=ms, global_ms=ms,
        width=32, height=32, motion_score=30-i,
        detections=[BoxEvidence(class_id=0, class_name='hand', confidence=.9,
                                xyxy_norm=(.2, .2, .6, .6), track_id=1)],
    ) for i, ms in enumerate((5000, 1000, 9000))]
    ledger = tmp_path / 'Evidence.jsonl'
    ledger.write_text(''.join(frame.model_dump_json() + '\n' for frame in sorted(frames, key=lambda f: f.local_ms)))
    config = default_config
    config['performance'].update(coarse_open_vocabulary_recall_enabled=True,
        fine_roi_open_vocabulary_recall_enabled=True, fine_roi_open_vocabulary_max_frames_per_window=3,
        coarse_open_vocabulary_maximum_error_rate=0, fine_roi_open_vocabulary_maximum_error_rate=0)
    config['models']['open_vocabulary_key_frame'] = {'enabled': True, 'prompt_map': {}}
    monkeypatch.setattr(coarse_recall, 'select_suspicious_coarse_frames', lambda *args: frames)

    def run(phase):
        args = ([view], {view.view_id: info}, {view.view_id: ledger})
        if phase == 'coarse':
            return coarse_recall.generate_open_vocabulary_coarse_candidates(*args, config)
        return coarse_recall.generate_open_vocabulary_fine_candidates(
            *args, {view.view_id: [(0, 10000)]}, config)

    return view, run


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
def test_recall_reuses_reader_preserves_backward_seek_and_stale_frame_fallback(
    recall_case, monkeypatch, phase,
):
    view, run = recall_case
    captures, seeks, fallbacks, pixels = [], [], [], []

    class Capture:
        def __init__(self, path, *args):
            assert path == str(view.video)
            assert args == (video_io.cv2.CAP_FFMPEG, [
                video_io.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                video_io.cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            self.released = False
            captures.append(self)

        def isOpened(self):
            return not self.released

        def set(self, prop, value):
            self.ms = value
            seeks.append(value)

        def get(self, prop):
            if prop == video_io.cv2.CAP_PROP_POS_MSEC:
                return 0 if self.ms == 1000 else self.ms
            return 30

        def read(self):
            return True, np.full((32, 32, 3), int(self.ms / 1000), dtype=np.uint8)

        def release(self):
            self.released = True

    def fallback(path, ms):
        fallbacks.append((path, ms))
        return np.full((32, 32, 3), 101, dtype=np.uint8)

    def predict(frame, settings):
        pixels.append(int(frame[0, 0, 0]))
        return [], {'status': 'executed'}

    monkeypatch.setattr(video_io.cv2, 'VideoCapture', Capture)
    monkeypatch.setattr(video_io, '_read_frame_at_ffmpeg', fallback)
    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', predict)
    candidates, report = run(phase)
    assert candidates == []
    assert seeks == [5000, 1000, 9000]
    assert fallbacks == [(view.video, 1000)]
    assert pixels == [5, 101, 9]  # Stale OpenCV pixel 1 must never reach the model.
    assert len(captures) == 2  # One handle reused, then replaced after failed seek.
    assert all(capture.released for capture in captures)
    assert [frame['local_ms'] for frame in report['frames']] == seeks
    assert all(frame['source_read_seconds'] >= 0 for frame in report['frames'])
    assert report['formal_evidence_ready']


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
@pytest.mark.parametrize('failure', ['unreadable', 'model', 'read'])
def test_recall_closes_reader_and_keeps_failed_frames_out_of_quality_evidence(
    recall_case, monkeypatch, phase, failure,
):
    _, run = recall_case
    closed = []

    class Reader:
        def __init__(self, max_open):
            assert max_open == 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed.append(True)

        def read(self, *args):
            if failure == 'read':
                raise OSError('source unavailable')
            return None if failure == 'unreadable' else np.zeros((32, 32, 3), dtype=np.uint8)

    def predict(*args):
        raise RuntimeError('model failed')

    monkeypatch.setattr(coarse_recall, 'ViewFrameReader', Reader)
    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', predict)
    candidates, report = run(phase)
    assert candidates == []
    assert report['error_count'] == 3
    assert report['recovered_frame_count'] == 0
    assert not report['formal_evidence_ready']
    assert closed == [True] * (1 if failure == 'model' else 4)


def install_reader(monkeypatch, read):
    """Record physical reader lifetimes; never open source media."""
    readers, reads, closed = [], [], []

    class Reader:
        def __init__(self, max_open):
            assert max_open == 1
            self.identity = len(readers)
            readers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed.append(self.identity)

        def read(self, view, info, local_ms):
            reads.append((self.identity, view.view_id, local_ms))
            return read(self.identity, local_ms)

    monkeypatch.setattr(coarse_recall, 'ViewFrameReader', Reader)
    return readers, reads, closed


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
@pytest.mark.parametrize('read_failure', ['no_frame', 'temporary_os_error'])
def test_recall_retries_transient_read_at_identical_time_with_fresh_handle(
    recall_case, monkeypatch, phase, read_failure,
):
    _, run = recall_case
    pixels = []

    def read(identity, local_ms):
        if identity == 0 and local_ms == 5000:
            if read_failure == 'temporary_os_error':
                raise OSError('temporary source read timeout')
            return None
        return np.full((32, 32, 3), int(local_ms / 1000), dtype=np.uint8)

    readers, reads, closed = install_reader(monkeypatch, read)

    def predict(frame, settings):
        pixels.append(int(frame[0, 0, 0]))
        return [], {'status': 'executed'}

    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', predict)
    candidates, report = run(phase)
    assert candidates == []  # A completed negative prediction is not an execution failure.
    assert [item[2] for item in reads] == [5000, 5000, 1000, 9000]
    assert [item[0] for item in reads] == [0, 1, 0, 0]
    assert len(readers) == 2 and sorted(closed) == [0, 1]
    assert pixels == [5, 1, 9]
    assert [frame['local_ms'] for frame in report['frames']] == [5000, 1000, 9000]
    assert report['selected_frame_count'] == 3
    assert report['recovered_frame_count'] == 1
    assert report['error_count'] == report['read_error_count'] == 0
    assert report['gate_metric'] == 'selected_frame_execution_completion'
    assert report['formal_evidence_ready']


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
def test_recall_retries_only_failed_model_call_using_same_pixels(
    recall_case, monkeypatch, phase,
):
    _, run = recall_case
    pixels, frames = [], []
    _, reads, closed = install_reader(monkeypatch, lambda _, ms:
        np.full((32, 32, 3), int(ms / 1000), dtype=np.uint8))

    def predict(frame, settings):
        pixels.append(int(frame[0, 0, 0]))
        frames.append(frame.copy())
        if len(pixels) == 1:
            raise RuntimeError('temporary CUDA execution failure')
        return [], {'status': 'executed', 'actual_device': '0'}

    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', predict)
    _, report = run(phase)
    assert pixels == [5, 5, 1, 9]
    np.testing.assert_array_equal(frames[0], frames[1])
    assert len(reads) == 3 and closed == [0]
    assert report['error_count'] == 0
    assert report['recovered_frame_count'] == 1
    assert report['formal_evidence_ready']
    attempts = report['frames'][0]['model_attempts']['yolo_world']
    assert [attempt['status'] for attempt in attempts] == ['error', 'completed']


@pytest.mark.parametrize('persistent_failure', [False, True])
def test_grounding_dino_retry_preserves_successful_yolo_work(
    recall_case, default_config, monkeypatch, persistent_failure,
):
    from visioncortex import archive

    _, run = recall_case
    default_config['models']['open_vocabulary_key_frame']['prompt_map'] = {'hand': 'hand', 'beaker': 'beaker'}
    install_reader(monkeypatch, lambda _, ms:
        np.full((32, 32, 3), int(ms / 1000), dtype=np.uint8))
    yolo_calls, dino_calls = [], []

    def yolo(frame, settings):
        yolo_calls.append(int(frame[0, 0, 0]))
        return [{'class_name': 'hand', 'confidence': .8, 'xyxy_norm': [.2, .2, .6, .6]}], {'status': 'executed'}

    def dino(frame, classes, settings):
        dino_calls.append(int(frame[0, 0, 0]))
        if persistent_failure or len(dino_calls) % 2:
            raise RuntimeError('temporary model timeout')
        return [{'class_name': 'beaker', 'confidence': .8, 'xyxy_norm': [.4, .4, .8, .8]}], {'status': 'executed'}

    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', yolo)
    monkeypatch.setattr(archive, '_grounding_dino_key_frame_detections', dino)
    candidates, report = run('coarse')
    assert yolo_calls == [5, 1, 9]
    assert dino_calls == [5, 5, 1, 1, 9, 9]
    assert len(candidates) == (0 if persistent_failure else 3)
    assert report['recovered_frame_count'] == (0 if persistent_failure else 3)
    assert report['error_count'] == (3 if persistent_failure else 0)
    assert report['formal_evidence_ready'] is not persistent_failure
    assert all(candidate.uncertainty for candidate in candidates)
    for frame in report['frames']:
        assert frame['model_receipts']['yolo_world']['status'] == 'executed'
        if persistent_failure:
            assert 'grounding_dino' not in frame['model_receipts']
        else:
            assert frame['model_receipts']['grounding_dino']['status'] == 'executed'
        assert len(frame['model_attempts']['yolo_world']) == 1
        assert len(frame['model_attempts']['grounding_dino']) == 2


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
def test_recovered_read_with_persistent_inference_failure_is_not_a_recovered_frame(
    recall_case, monkeypatch, phase,
):
    _, run = recall_case
    install_reader(monkeypatch, lambda identity, _:
        None if identity == 0 else np.zeros((32, 32, 3), dtype=np.uint8))

    def predict(*args):
        raise RuntimeError('temporary model timeout')

    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', predict)
    _, report = run(phase)
    assert all(frame['source_read_recovered'] for frame in report['frames'])
    assert report['recovered_frame_count'] == 0
    assert report['error_count'] == report['inference_error_count'] == 3
    assert report['read_error_count'] == 0
    assert not report['formal_evidence_ready']


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
def test_persistent_model_failure_counts_selected_slots_not_attempts_and_keeps_ten_percent_gate(
    recall_case, default_config, monkeypatch, phase,
):
    _, run = recall_case
    default_config['performance'][f'{phase}_' + ('roi_' if phase == 'fine' else '')
                                  + 'open_vocabulary_maximum_error_rate'] = .10
    calls = []
    install_reader(monkeypatch, lambda _, ms:
        np.full((32, 32, 3), int(ms / 1000), dtype=np.uint8))

    def predict(frame, settings):
        pixel = int(frame[0, 0, 0])
        calls.append(pixel)
        if pixel == 5:
            raise RuntimeError('temporary execution timeout')
        return [], {'status': 'executed'}

    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', predict)
    _, report = run(phase)
    assert calls == [5, 5, 1, 9]
    assert report['selected_frame_count'] == 3
    assert report['error_count'] == report['inference_error_count'] == 1
    assert report['error_rate'] == .333333
    assert report['maximum_error_rate'] == .10
    assert report['recovered_frame_count'] == 0
    assert not report['formal_evidence_ready']


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
@pytest.mark.parametrize('failure', [FileNotFoundError('model missing'),
                                   ValueError('invalid prompt_map'), RuntimeError('model failed')])
def test_deterministic_model_failure_does_not_repeat_or_fake_success(
    recall_case, monkeypatch, phase, failure,
):
    _, run = recall_case
    calls = []
    install_reader(monkeypatch, lambda *_: np.zeros((32, 32, 3), dtype=np.uint8))

    def predict(*args):
        calls.append(True)
        raise failure

    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', predict)
    _, report = run(phase)
    assert len(calls) == 3
    assert report['error_count'] == 3
    assert not report['formal_evidence_ready']
    assert all(len(frame['model_attempts']['yolo_world']) == 1 for frame in report['frames'])


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
@pytest.mark.parametrize('failure', [FileNotFoundError('source missing'), PermissionError('source denied')])
def test_missing_or_inaccessible_source_is_not_retried_or_accepted(
    recall_case, monkeypatch, phase, failure,
):
    _, run = recall_case

    def read(*args):
        raise failure

    readers, reads, closed = install_reader(monkeypatch, read)
    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', lambda *args: pytest.fail('Unreadable source inferred'))
    _, report = run(phase)
    assert len(readers) == 1 and len(reads) == 3 and closed == [0]
    assert report['error_count'] == report['read_error_count'] == 3
    assert not report['formal_evidence_ready']


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
@pytest.mark.parametrize('cancel_during', ['read', 'model'])
def test_cancellation_escapes_without_retry_or_completed_report(
    recall_case, monkeypatch, phase, cancel_during,
):
    _, run = recall_case
    calls = []

    def read(*args):
        if cancel_during == 'read':
            raise ExecutionCancelled('test cancellation')
        return np.zeros((32, 32, 3), dtype=np.uint8)

    _, reads, closed = install_reader(monkeypatch, read)

    def predict(*args):
        calls.append(True)
        raise ExecutionCancelled('test cancellation')

    monkeypatch.setattr(coarse_recall, '_yolo_world_detections', predict)
    with pytest.raises(ExecutionCancelled, match='test cancellation'):
        run(phase)
    assert len(reads) == 1 and closed == [0]
    assert len(calls) == (1 if cancel_during == 'model' else 0)
