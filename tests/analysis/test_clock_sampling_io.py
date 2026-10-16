"""Clock samples stay exact while dense reads amortize transport requests."""
import io
from pathlib import Path

import pytest

from visioncortex.alignment import read_timestamp_csv_bounded


class Transport(io.RawIOBase):
    def __init__(self, data):
        self.data = io.BytesIO(data)
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def readinto(self, target):
        self.requests += 1
        return self.data.readinto(target)

    def seek(self, offset, whence=0):
        return self.data.seek(offset, whence)

    def tell(self):
        return self.data.tell()


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_dense_clock_samples_preserve_sparse_rgb_and_reduce_transport_reads(tmp_path, monkeypatch, newline):
    path = tmp_path / 'Frames.csv'
    rows = ['frame_index,local_timestamp_ms,global_timestamp_ms,rgb_recorded,note']
    rows.extend(f'{i},{i * 100},{1000000 + i * 100},{int(i % 5 == 0)},{"x" * 150}'
                for i in range(40000))
    path.write_text('\ufeff' + newline.join(rows) + newline)
    data = path.read_bytes()
    original = Path.open
    runs = []
    force_small = False

    def open_clock(self, mode='r', *args, **kwargs):
        if self == path and mode == 'rb':
            raw = Transport(data)
            runs.append(raw)
            buffer = kwargs.pop('buffering', io.DEFAULT_BUFFER_SIZE)
            if force_small or buffer < 1:
                buffer = io.DEFAULT_BUFFER_SIZE
            return io.BufferedReader(raw, buffer_size=buffer)
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_clock)
    force_small = True
    expected = read_timestamp_csv_bounded(path, 10, sample_count=4096)
    small_requests = sum(raw.requests for raw in runs)
    runs.clear()
    force_small = False
    actual = read_timestamp_csv_bounded(path, 10, sample_count=4096)
    large_requests = sum(raw.requests for raw in runs)
    assert actual == expected
    assert actual[0].frame_index == 0
    assert actual[-1].frame_index == 39995
    assert all(point.frame_index % 5 == 0 for point in actual)
    assert all(point.source_ms == 1000000 + point.frame_index * 100 for point in actual)
    assert large_requests < small_requests / 20


def test_sparse_clock_requests_keep_small_bounded_reads(tmp_path, monkeypatch):
    path = tmp_path / 'Frames.csv'
    path.write_text('frame_index,local_timestamp_ms,global_timestamp_ms\n'
                    + '\n'.join(f'{i},{i * 100},{1000000 + i * 100}' for i in range(1000)) + '\n')
    original = Path.open
    buffers = []

    def open_clock(self, mode='r', *args, **kwargs):
        if self == path and mode == 'rb':
            buffers.append(kwargs.get('buffering', -1))
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_clock)
    points = read_timestamp_csv_bounded(path, 10, sample_count=5, full_read_limit_bytes=1)
    assert len(points) == 5
    assert all(buffer in (-1, io.DEFAULT_BUFFER_SIZE) for buffer in buffers)
