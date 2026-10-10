"""Queue observations distinguish scheduling ownership from validation work."""
import json

from visioncortex.device_day_consumers import consumer_snapshot
from visioncortex.device_day_progress import snapshot
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retry import EVIDENCE_SCHEMA, apply_retry_plan, plan_retry


NOW = 1_800_000_000


def setup(tmp_path, *, age=20000):
    root = tmp_path / 'device-day'
    record = {'recording_id': 'record', 'configured_role': 'first_person', 'source_signature': 'source',
              'camera_key': 'camera', 'processable': True, 'recording_start_us': (NOW - age) * 1_000_000,
              'recording_end_us': (NOW - age + 5) * 1_000_000}
    queue = DeviceDayQueue(root / 'queue-retention.sqlite3')
    queue.enqueue(record, 'revision')
    config = {'storage': {'local_runtime_root': str(tmp_path)}, 'device_day': {'enabled': True}}
    return config, queue


def reason(config):
    value = snapshot(config)
    return next(iter(value['waiting'].values()))['retention']


def test_missing_consumer_is_not_reported_as_validation_in_progress(tmp_path, monkeypatch):
    config, _queue = setup(tmp_path)
    monkeypatch.setattr('visioncortex.device_day_progress.time.time', lambda: NOW)
    assert reason(config) == {'no_consumer': 1}


def test_verified_owner_exposes_history_capacity_hold_or_dispatch_wait(tmp_path, monkeypatch):
    config, _queue = setup(tmp_path)
    monkeypatch.setattr('visioncortex.device_day_progress.time.time', lambda: NOW)
    consumers = consumer_snapshot(config, now=NOW)
    stage = consumers['stages']['retention']
    stage['scopes']['history']['status'] = 'covered'
    stage['owners'] = [{'owner_state': 'verified', 'backfill_status': 'waiting_for_live'}]
    monkeypatch.setattr('visioncortex.device_day_consumers.consumer_snapshot', lambda *_args, **_kwargs: consumers)
    assert reason(config) == {'history_waiting_for_live': 1}
    stage['owners'][0]['backfill_status'] = 'running'
    assert reason(config) == {'waiting_for_history_dispatch': 1}


def test_exhausted_failure_then_verified_retry_budget_changes_observed_reason(tmp_path, monkeypatch):
    config, queue = setup(tmp_path)
    monkeypatch.setattr('visioncortex.device_day_progress.time.time', lambda: NOW)
    failure = {'status': 'failed', 'error_type': 'PermissionError', 'message': 'denied'}
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?,updated_at=?",
                   (json.dumps(failure), NOW - 100))
    assert reason(config) == {'retry_exhausted': 1}
    repair = {'schema_version': EVIDENCE_SCHEMA, 'repair_kind': 'storage_access', 'recording_id': 'record',
              'revision': 'revision', 'source_signature': 'source', 'repair_revision': 'a' * 64,
              'checks': {key: 'PROVEN' for key in ('current_source_verified', 'successful_requests_preserved',
                                                 'storage_mount_verified', 'source_access_verified')}}
    assert apply_retry_plan(queue, plan_retry(queue.path, 'record', repair))
    # Still failed, but now runnable and held only by absent ownership.
    assert reason(config) == {'no_consumer': 1}
    count = consumer_snapshot(config, now=NOW)['stages']['retention']['scopes']['history']
    assert count['locally_ready'] == 1


def test_recent_ready_work_has_explicit_dispatch_reason(tmp_path, monkeypatch):
    config, _queue = setup(tmp_path, age=10)
    monkeypatch.setattr('visioncortex.device_day_progress.time.time', lambda: NOW)
    consumers = consumer_snapshot(config, now=NOW)
    consumers['stages']['retention']['scopes']['recent']['status'] = 'covered'
    monkeypatch.setattr('visioncortex.device_day_consumers.consumer_snapshot', lambda *_args, **_kwargs: consumers)
    assert reason(config) == {'waiting_for_dispatch': 1}
