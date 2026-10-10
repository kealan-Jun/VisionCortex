"""Observer progress stays independent of execution packages and NAS trees."""
import builtins
import json
from pathlib import Path

from visioncortex.device_day_contract import read_json
from visioncortex.device_day_progress import snapshot
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.runtime_services import _publish_progress


def recording(identifier, start=1787792400000000):
    return {'recording_id': identifier, 'camera_key': 'camera',
            'configured_role': 'first_person', 'recording_start_us': start,
            'source_signature': identifier, 'processable': True}


def observer_config(tmp_path):
    return {'storage': {'local_runtime_root': str(tmp_path),
                        'local_cache_root': '/mnt/observer-forbidden-nas/cache'},
            'device_day': {'enabled': True, 'inplace_preprocessing': True}}


def block_execution_and_nas(monkeypatch):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert 'device_day_inplace' not in name and 'device_day_models' not in name
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    original_glob = Path.glob
    def guarded_glob(path, pattern):
        assert not str(path).startswith('/mnt/observer-forbidden-nas')
        return original_glob(path, pattern)
    monkeypatch.setattr(Path, 'glob', guarded_glob)


def test_observer_publishes_actual_snapshot_without_inplace_package_or_nas_scan(tmp_path, monkeypatch):
    queue = DeviceDayQueue(tmp_path / 'device-day/queue-vision.sqlite3')
    queue.enqueue(recording('current'), 'v1')
    block_execution_and_nas(monkeypatch)
    class Control:
        def __init__(self):
            self.closed = False
        def poll(self, timeout=None):
            return timeout is not None
        def close(self):
            self.closed = True
    control = Control()
    _publish_progress(observer_config(tmp_path), control, 5)
    result = read_json(tmp_path / 'device-day/ProgressSnapshot.json')
    assert result['snapshot_producer_pid'] > 0 and result['observed_at'] > 0
    assert next(iter(result['days'].values()))['stages']['vision']['queued'] == 1
    assert result['consumers']['stages']['vision']['scopes']['history']['status'] == 'no_consumer'
    assert control.closed


def test_completed_queue_is_a_hint_and_cannot_claim_verified_publication(tmp_path, monkeypatch):
    queue = DeviceDayQueue(tmp_path / 'device-day/queue-vision.sqlite3')
    queue.enqueue(recording('complete'), 'v1')
    queue.claim('owner')
    queue.finish('owner', 'complete', {'status': 'completed', 'formal_publication': 'completed',
                 'component_timings': {'input_hash_seconds': 1.25}}, 2)
    block_execution_and_nas(monkeypatch)
    result = snapshot(observer_config(tmp_path))
    row = result['input_lifecycle'][0]
    assert row['preprocessing']['vision'] == 'completed'
    assert row['archive_status'] == 'pending'
    assert row['publication_status'] == 'reported_completed'
    assert row['publication_verified'] is False
    assert row['scope'] == 'local_queue_hints_not_receipt_or_artifact_acceptance'
    assert row['timings']['vision']['input_hash_seconds'] == 1.25
    assert row['timings']['vision']['queue_wait_seconds'] >= 0
    assert '正式发布证据尚未核实' in result['latency_html']


def test_absent_lifecycle_is_explicit_and_does_not_hide_discovery_counts(tmp_path, monkeypatch):
    root = tmp_path / 'device-day'
    root.mkdir()
    (root / 'observed-inventory.json').write_text(json.dumps({'recordings': [recording('not-enqueued')]}))
    block_execution_and_nas(monkeypatch)
    result = snapshot(observer_config(tmp_path))
    assert result['input_lifecycle'] == []
    assert result['input_lifecycle_observation']['available'] is False
    assert result['input_lifecycle_observation']['reason'] == 'local_queue_lifecycle_unavailable'
    assert next(iter(result['days'].values()))['total'] == 1


def test_lifecycle_is_bounded_without_truncating_queue_totals(tmp_path):
    queue = DeviceDayQueue(tmp_path / 'device-day/queue-vision.sqlite3')
    for number in range(105):
        queue.enqueue(recording(str(number), start=1787792400000000 + number), 'v1')
    result = snapshot(observer_config(tmp_path))
    assert len(result['input_lifecycle']) == 100
    assert result['input_lifecycle'][0]['recording_id'] == '104'
    assert result['input_lifecycle_observation']['record_count'] == 105
    assert next(iter(result['days'].values()))['stages']['vision']['queued'] == 105
    assert next(iter(result['waiting'].values()))['vision'] == {'no_consumer': 105}
