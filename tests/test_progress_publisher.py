"""A fresh progress read survives loss of its dedicated publisher process."""
import json
import os
from pathlib import Path
import time

from visioncortex.runtime_services import ProgressPublisher


def wait_snapshot(path, *, after=0, pid=None):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.is_file():
            data = json.loads(path.read_text())
            if data['observed_at'] > after and (pid is None or data['snapshot_producer_pid'] == pid):
                return data
        time.sleep(.05)
    raise AssertionError('Isolated publisher did not publish fresh local progress')


def test_progress_publisher_isolated_refresh_and_restart(tmp_path):
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    service = ProgressPublisher(config, interval=.05)
    path = Path(tmp_path) / 'device-day' / 'ProgressSnapshot.json'
    try:
        service.ensure()
        first = wait_snapshot(path)
        assert first['snapshot_producer_pid'] != os.getpid()
        assert first['snapshot_collection_seconds'] >= 0
        second = wait_snapshot(path, after=first['observed_at'])
        assert second['days'] == {}  # No media or fake experiment produced.
        service.process.kill()
        service.process.join(timeout=5)
        service.ensure()
        restored = wait_snapshot(path, after=second['observed_at'], pid=service.process.pid)
        assert restored['snapshot_producer_pid'] != first['snapshot_producer_pid']
    finally:
        service.close()
    service.close()
    service.ensure()
    assert service.process is None
