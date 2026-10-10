from datetime import datetime, timezone
import sqlite3

from visioncortex.device_day_latency import observe, snapshot, render, published
from visioncortex.device_day_queue import DeviceDayQueue


def test_latencies_preserve_discovery_and_exclude_old_completions(tmp_path, monkeypatch):
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    now = 1_800_000_000
    r = {'recording_id': 'r1', 'camera_key': 'camera1', 'configured_role': 'first_person',
         'recording_start_us': int((now-60)*1e6), 'source_signature': 'original',
         'updated_at': datetime.fromtimestamp(now-20, timezone.utc).isoformat(), 'processable': False}
    observe(config, [r], {'r1': {'cohort': 'live_observation'}}, now=now)
    r['processable'] = True
    observe(config, [r], now=now+5)
    q = DeviceDayQueue(tmp_path/'device-day'/'queue-vision.sqlite3')
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: now+7)
    q.enqueue(r, 'v1')
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: now+10)
    q.claim('worker')
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: now+20)
    q.renew('worker')
    q.finish('worker', 'r1', {'status': 'completed'}, 10)
    assert snapshot(config, now=now+20)['recent'][0]['ready_to_completed_seconds'] is None
    published(config, [r], now=now+20)
    observe(config, [r], now=now+30)
    result = snapshot(config, now=now+30)
    row = result['recent'][0]
    assert row['first_observed_at'] == now
    assert row['ready_at'] == now+5
    assert row['observed_to_ready_seconds'] == 5
    assert row['ready_to_start_seconds'] == 5
    assert row['vision_queue_seconds'] == 3
    assert row['ready_to_completed_seconds'] == 15
    assert row['vision_run_seconds'] == 10
    assert row['discovery_delay_seconds'] is None
    assert row['mtime_to_first_observation_estimate_seconds'] == 20
    assert result['cohorts']['live_observation']['metrics']['ready_to_start_seconds']['p95_seconds'] == 5
    assert '5.0 秒' in render(result)
    # Different source bytes cannot inherit a previous latency or completion.
    observe(config, [r | {'source_signature': 'changed'}], now=now+40)
    row = snapshot(config, now=now+50)['recent'][0]
    assert row['first_observed_at'] == now
    assert row['ready_at'] == now+40
    assert row['ready_to_completed_seconds'] is None
    assert row['waiting_to_start_seconds'] == 10


def test_historical_jobs_never_claim_realtime_and_retries_keep_actual_start(tmp_path, monkeypatch):
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    now = 1_800_000_000
    r = {'recording_id': 'old', 'camera_key': 'c', 'configured_role': 'first_person',
         'recording_start_us': int((now-86400*7)*1e6), 'source_signature': 's', 'processable': True}
    q = DeviceDayQueue(tmp_path/'device-day'/'queue-vision.sqlite3')
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: now-100)
    q.enqueue(r, 'v1')
    q.claim('previous')
    q.finish('previous', 'old', {'status': 'completed'}, 10)
    observe(config, [r], now=now)
    result = snapshot(config, now=now+10)
    assert result['recent'][0]['ready_to_completed_seconds'] is None
    assert result['recent'][0]['waiting_to_start_seconds'] is None
    assert result['cohorts']['live_observation']['observations_24h'] == 0
    assert '已有结果，缺少本次时延样本' in render(result)
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: now+20)
    q.enqueue(r, 'v2')
    q.claim('new')
    q.finish('new', 'old', {'status': 'failed'}, 1)
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: now+40)
    q.claim('retry', retry=True)
    row = snapshot(config, now=now+50)['recent'][0]
    assert row['vision_started_at'] == now+40
    assert row['attempts'] == 2
    assert row['vision_queue_seconds'] == 20


def test_queue_migration_does_not_fabricate_old_start(tmp_path):
    path = tmp_path/'queue.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('''CREATE TABLE recordings(recording_id TEXT PRIMARY KEY, revision TEXT, payload TEXT,
         status TEXT,queued_at REAL,updated_at REAL,lease_owner TEXT,lease_until REAL,attempts INTEGER,
         result TEXT,completed_at REAL,wall_seconds REAL)''')
        db.execute("INSERT INTO recordings(recording_id,status,updated_at) VALUES('old','completed',100)")
    q = DeviceDayQueue(path)
    with q.connect() as db:
        assert db.execute("SELECT started_at FROM recordings WHERE recording_id='old'").fetchone()[0] is None


def test_history_scanner_finding_today_does_not_mark_today_as_backfill(tmp_path):
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    now = 1_800_000_000
    r = {'recording_id': 'today', 'camera_key': 'c', 'recording_start_us': int(now*1e6),
         'source_signature': 's', 'processable': True}
    observe(config, [r], {'today': {'cohort': 'historical_backfill'}}, now=now)
    result = snapshot(config, now=now+1)
    assert result['recent'][0]['cohort'] == 'startup_inventory'
    assert result['cohorts']['historical_backfill']['observations_24h'] == 0
    with sqlite3.connect(tmp_path/'device-day'/'latency.sqlite3') as db:
        db.execute("UPDATE observations SET cohort='historical_backfill'")
    observe(config, [r], {'today': {'cohort': 'historical_backfill'}}, now=now+20)
    row = snapshot(config, now=now+21)['recent'][0]
    assert row['cohort'] == 'startup_inventory'
    assert row['first_observed_at'] == now
