import hashlib
from types import SimpleNamespace

import pytest

from visioncortex import device_day, device_day_models
from visioncortex.device_day_contract import artifact
from visioncortex.schemas import VideoInfo


def test_retention_streams_source_once_and_checks_target(tmp_path, monkeypatch):
    source, target = tmp_path / 'Source.mp4', tmp_path / 'Target.mp4'
    payload = bytes(range(256)) * 8192
    source.write_bytes(payload)
    calls = []
    original = device_day.file_hash
    def hashed(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(device_day, 'file_hash', hashed)
    result = device_day.copy_verified(source, target)
    assert calls and source not in calls
    assert result['sha256'] == hashlib.sha256(payload).hexdigest()
    assert source.read_bytes() == target.read_bytes() == payload
    target.write_bytes(b'x' * len(payload))
    with pytest.raises(ValueError, match='collision'):
        device_day.copy_verified(source, target)


def test_corrupt_copy_never_publishes(tmp_path, monkeypatch):
    source, target = tmp_path / 'Source.mp4', tmp_path / 'Target.mp4'
    source.write_bytes(b'real-byte-equality-test')
    monkeypatch.setattr(device_day, 'file_hash', lambda path: 'incorrect')
    with pytest.raises(ValueError, match='checksum mismatch'):
        device_day.copy_verified(source, target)
    assert not target.exists()
    assert not list(tmp_path.glob('*.partial'))
    assert source.read_bytes() == b'real-byte-equality-test'


@pytest.mark.parametrize('duration', [700, 120000, 900667, 1800333])
def test_actual_closed_file_is_one_scan_unit(default_config, tmp_path, monkeypatch, duration):
    source, clock = tmp_path / 'Video.mp4', tmp_path / 'Frames.csv'
    source.write_bytes(b'fixture-no-model-invocation')
    clock.write_text('')
    info = VideoInfo(path=source, duration_ms=duration, fps=30, frame_count=round(duration*30/1000), width=10, height=10, size_bytes=source.stat().st_size)
    backend = device_day_models.DeviceDayModels(default_config)
    backend._validated = {'test_only': True}
    monkeypatch.setattr(backend, '_probe', lambda path: info)
    monkeypatch.setattr(device_day_models, 'capture_clock', lambda *args: {})
    from visioncortex import source_frames
    monkeypatch.setattr(source_frames, 'SourceFrameTrace', lambda *args: None)
    monkeypatch.setattr(source_frames, 'source_observation_windows', lambda *args: {
        'native_frame_count': info.frame_count, 'available_windows': [[0, duration]],
        'unavailable_intervals': [], 'insufficient_temporal_intervals': []})
    class ScanReached(Exception):
        pass
    def scan(view, info_, transform, retention, phase, ranges, fps, size):
        assert phase == 'coarse' and ranges == [(0, duration)]
        raise ScanReached
    monkeypatch.setattr(backend, '_scan', scan)
    refs = [artifact(tmp_path, p) for p in (source, clock)]
    retention = {'recording': {'camera_key':'cam', 'configured_role':'first_person', 'recording_start_us':0}, 'sources':[{'kind':k, 'retained':r} for k,r in zip(('video','clock'),refs, strict=False)], 'artifacts':refs}
    with pytest.raises(ScanReached):
        backend.vision(SimpleNamespace(root=tmp_path), retention, 'fixture')


def test_changed_source_during_copy_is_not_published(tmp_path, monkeypatch):
    source, target = tmp_path / "Source.mp4", tmp_path / "Target.mp4"
    source.write_bytes(b"original")
    original = device_day.file_hash
    def checked(path):
        result = original(path)
        source.write_bytes(b"changed while copy was checked")
        return result
    monkeypatch.setattr(device_day, "file_hash", checked)
    with pytest.raises(ValueError, match="changed during retention"):
        device_day.copy_verified(source, target)
    assert not target.exists()


def test_recording_reader_is_owned_and_closed_on_failure(default_config, monkeypatch):
    from visioncortex import video_io
    calls = []
    class Reader:
        def __init__(self, max_open):
            assert max_open == 1
        def __enter__(self):
            calls.append('open')
            return self
        def __exit__(self, *args):
            calls.append('close')
    monkeypatch.setattr(video_io, 'ViewFrameReader', Reader)
    backend = device_day_models.DeviceDayModels(default_config)
    def fail(layout, retention, key, reader):
        assert isinstance(reader, Reader)
        raise ValueError('fixture')
    monkeypatch.setattr(backend, '_vision', fail)
    with pytest.raises(ValueError, match='fixture'):
        backend.vision(None, {}, 'fixture')
    assert calls == ['open', 'close']
