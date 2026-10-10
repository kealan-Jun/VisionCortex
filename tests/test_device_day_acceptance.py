import sqlite3

from visioncortex.device_day_acceptance import observe, render
from visioncortex.device_day_latency import observe as discovered, published, snapshot
from visioncortex.device_day_queue import DeviceDayQueue


NOW = 1_800_000_000


def record(key, *, signature='v1', at=NOW):
    return {'recording_id': key, 'camera_key': 'camera', 'source_signature': signature,
            'configured_role': 'first_person', 'recording_start_us': int(at * 1e6), 'processable': True}


def finish(config, row, monkeypatch, *, start=NOW+2):
    from pathlib import Path
    queue = DeviceDayQueue(Path(config['storage']['local_runtime_root'])/'device-day/queue-vision.sqlite3')
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: start)
    queue.enqueue(row, 'code-v1')
    queue.claim('worker')
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: start+1)
    queue.finish('worker', row['recording_id'], {'status': 'completed'}, 1)


def test_idle_weekend_and_long_observation_are_not_load_acceptance(tmp_path):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    first = observe(cfg, {'observed_at': 1000, 'days': {}})
    final = observe(cfg, {'observed_at': 1000 + 9*3600, 'days': {}})
    assert first['samples'] == 1
    assert final['observed_hours'] == 9
    assert final['max_sample_gap_seconds'] == 9*3600
    assert final['live_inputs'] == 0
    assert final['evidence_status'] == 'PARTIAL_EVIDENCE'
    assert '不能证明生产吞吐能力' in render(final)


def test_exact_live_revision_counted_once_and_old_backfill_excluded(tmp_path, monkeypatch):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    row = record('a')
    observe(cfg, {'observed_at': NOW, 'days': {}})
    discovered(cfg, [row], {'a': {'cohort': 'live_observation'}}, now=NOW+1)
    discovered(cfg, [record('old', at=NOW-86400*2)], now=NOW+1)
    p = {'observed_at': NOW+10, 'days': {}}
    assert observe(cfg, p)['live_inputs'] == 1
    finish(cfg, row, monkeypatch)
    published(cfg, [row], now=NOW+20)
    p['observed_at'] = NOW+30
    final = observe(cfg, p)
    assert final['live_inputs'] == final['live_published'] == 1
    # A later same-source rerun also must not make the original timing invalid.
    with sqlite3.connect(tmp_path/'device-day/queue-vision.sqlite3') as db:
        db.execute('UPDATE recordings SET started_at=?', (NOW+35,))
    assert observe(cfg, p | {'observed_at': NOW+36})['live_published'] == 1
    with sqlite3.connect(tmp_path/'device-day/production-observations.sqlite3') as db:
        assert db.execute('SELECT started,completed FROM live').fetchone() == (NOW+2, NOW+20)
    # A later queue revision cannot erase the exact prior publication evidence.
    finish(cfg, row | {'source_signature': 'v2'}, monkeypatch, start=NOW+40)
    assert observe(cfg, p | {'observed_at': NOW+50})['live_published'] == 1


def test_live_coverage_and_completion_are_not_limited_to_recent_fifty(tmp_path, monkeypatch):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    observe(cfg, {'observed_at': NOW, 'days': {}})
    rows = [record(f'live-{n}') for n in range(65)]
    discovered(cfg, rows, {r['recording_id']: {'cohort': 'live_observation'} for r in rows}, now=NOW+1)
    old = [record(f'old-{n}', at=NOW-86400*2) for n in range(60)]
    discovered(cfg, old, now=NOW+10)
    progress = {'observed_at': NOW+11, 'days': {}, 'latency': snapshot(cfg, now=NOW+11)}
    assert {r['cohort'] for r in progress['latency']['recent']} == {'historical_backfill'}
    result = observe(cfg, progress)
    assert result['live_inputs'] == 65
    assert result['live_published'] == 0
    finish(cfg, rows[0], monkeypatch, start=NOW+20)
    published(cfg, rows[:1], now=NOW+22)
    result = observe(cfg, progress | {'observed_at': NOW+30})
    assert result['live_inputs'] == 65
    assert result['live_published'] == 1


def test_prior_queue_completion_or_other_revision_never_counts_as_new_publication(tmp_path, monkeypatch):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    observe(cfg, {'observed_at': NOW, 'days': {}})
    rows = [record('old-start'), record('other-signature')]
    finish(cfg, rows[0], monkeypatch, start=NOW-10)
    finish(cfg, rows[1] | {'source_signature': 'old'}, monkeypatch, start=NOW+2)
    discovered(cfg, rows, {r['recording_id']: {'cohort': 'live_observation'} for r in rows}, now=NOW+1)
    published(cfg, rows, now=NOW+5)
    result = observe(cfg, {'observed_at': NOW+10, 'days': {}})
    assert result['live_inputs'] == 2
    assert result['live_published'] == 0
    # Queue completed status alone is insufficient, even for the exact source.
    newer = record('unpublished')
    discovered(cfg, [newer], {'unpublished': {'cohort': 'live_observation'}}, now=NOW+11)
    finish(cfg, newer, monkeypatch, start=NOW+12)
    assert observe(cfg, {'observed_at': NOW+20, 'days': {}})['live_published'] == 0


def test_unreadable_latency_preserves_prior_observations_and_reports_coverage_error(tmp_path):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    observe(cfg, {'observed_at': NOW, 'days': {}})
    row = record('a')
    discovered(cfg, [row], {'a': {'cohort': 'live_observation'}}, now=NOW+1)
    assert observe(cfg, {'observed_at': NOW+2, 'days': {}})['live_inputs'] == 1
    with sqlite3.connect(tmp_path/'device-day/latency.sqlite3') as db:
        db.execute('ALTER TABLE observations RENAME TO unavailable')
    result = observe(cfg, {'observed_at': NOW+3, 'days': {}})
    assert result['live_inputs'] == 1
    assert result['samples_with_read_errors'] == 1
    assert result['errors']


def test_observation_window_excludes_preexisting_live_inputs_and_keeps_waiting_counts(tmp_path):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    discovered(cfg, [record('a')], {'a': {'cohort': 'live_observation'}}, now=NOW)
    result = observe(cfg, {'observed_at': NOW+10, 'days': {
        'day': {'stages': {'vision': {'waiting_for_prerequisite': 1}}}}})
    assert result['live_inputs'] == 0
    assert result['vision']['waiting_for_prerequisite'] == 1
