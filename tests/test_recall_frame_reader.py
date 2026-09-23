"""No media/model I/O: recall seeks keep their source times and close handles."""

import numpy as np
import pytest

from visioncortex import coarse_recall, video_io
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
    if failure == 'read':
        with pytest.raises(OSError, match='source unavailable'):
            run(phase)
    else:
        candidates, report = run(phase)
        assert candidates == []
        assert report['error_count'] == 3
        assert not report['formal_evidence_ready']
    assert closed == [True]
