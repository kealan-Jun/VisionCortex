import json
import time

import pytest

from visioncortex.device_day_backfill_policy import PolicyUnavailable, _monitor
from visioncortex.device_day_live_discovery import publish_status


def test_actual_independent_snapshot_opens_history_and_stop_closes_it(tmp_path):
    current = time.time()
    root = tmp_path / 'device-day'
    config = {'collection_ingest': {'poll_seconds': 5}, 'device_day': {'camera_lanes': True}}
    status = {'status': 'watching', 'updated_at': current, 'poll_seconds': 5,
        'recordings': [], 'errors': [], 'truncated': False, 'pending_live_publication_count': 0,
        'discovery_lanes': [{'camera_key': 'camera-1', 'mode': 'live', 'thread_alive': True,
            'status': 'watching', 'completed_at': current, 'errors': [], 'truncated': False}]}
    publish_status(root, status)
    _monitor(config, tmp_path, current)
    saved = json.loads((tmp_path / 'state/nas-recording-monitor.json').read_text())
    assert saved['discovery_lanes'] == status['discovery_lanes']
    assert saved['monitor_source'] == 'independent_live_discovery'
    publish_status(root, status | {'status': 'stopped'})
    with pytest.raises(PolicyUnavailable, match='monitor_not_watching'):
        _monitor(config, tmp_path, current)


def test_degraded_or_stale_discovery_cannot_authorize_history(tmp_path):
    current = time.time()
    config = {'device_day': {}, 'collection_ingest': {}}
    publish_status(tmp_path / 'device-day', {'status': 'retrying', 'updated_at': current})
    with pytest.raises(PolicyUnavailable, match='monitor_not_watching'):
        _monitor(config, tmp_path, current)
    publish_status(tmp_path / 'device-day', {'status': 'watching', 'updated_at': current - 300})
    with pytest.raises(PolicyUnavailable, match='monitor_stale'):
        _monitor(config, tmp_path, current)
