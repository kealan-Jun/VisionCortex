"""Idle admission uses local metadata only; no service, model or NAS access."""
from datetime import datetime, timezone
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from visioncortex.device_day_backfill_policy import (
    BACKFILL_SOURCE, PolicyUnavailable, historical_candidate_ids,
    historical_candidates, live_demand,
)


NOW = 20000.0
STAGES = ('retention', 'vision', 'stt', 'understanding', 'report')


def recording(identity, *, age=1, camera='a', **extra):
    start = int((NOW - age) * 1_000_000)
    return {'recording_id': identity, 'camera_key': camera,
            'configured_role': 'first_person', 'source_signature': identity,
            'recording_start_us': start, 'recording_end_us': start + 1_000_000,
            'processable': True, 'video_path': '/forbidden-nas/capture/video.mp4', **extra}


class LocalState:
    def __init__(self, root):
        self.root = root
        self.base = root / 'device-day'
        self.base.mkdir(parents=True)
        (root / 'state').mkdir()
        self.config = {'storage': {'local_runtime_root': str(root)},
                       'device_day': {'live_priority_seconds': 100, 'failure_retry_limit': 3},
                       'collection_ingest': {'poll_seconds': 5}}
        self.write(self.base / 'observed-inventory.sqlite3',
                   'CREATE TABLE observations(id TEXT PRIMARY KEY,payload TEXT)')
        for stage in STAGES:
            self.write(self.base / f'queue-{stage}.sqlite3',
                       'CREATE TABLE recordings(recording_id TEXT PRIMARY KEY, payload TEXT, '
                       'status TEXT, input_status TEXT, lease_until REAL, attempts INTEGER, updated_at REAL)')
        self.write(root / 'state' / 'resources.sqlite3',
                   'CREATE TABLE leases(resource TEXT,source TEXT,state TEXT,expires REAL)')
        self.monitor()

    @staticmethod
    def write(path, sql, parameters=()):
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(sql, parameters)

    def observe(self, *records):
        for record in records:
            self.write(self.base / 'observed-inventory.sqlite3',
                       'INSERT OR REPLACE INTO observations VALUES(?,?)',
                       (record['recording_id'], json.dumps(record)))

    def queue(self, stage, record, *, status='queued', input_status='ready',
              lease_until=0, attempts=0, updated_at=0):
        self.write(self.base / f'queue-{stage}.sqlite3',
                   'INSERT OR REPLACE INTO recordings VALUES(?,?,?,?,?,?,?)',
                   (record['recording_id'], json.dumps(record), status, input_status,
                    lease_until, attempts, updated_at))

    def complete(self, record, stages=STAGES):
        for stage in stages:
            self.queue(stage, record, status='completed')

    def unavailable(self, record, *, signature=None, state='missing'):
        path = self.base / 'InputAvailability.sqlite3'
        self.write(path, 'CREATE TABLE IF NOT EXISTS inputs(id TEXT PRIMARY KEY,signature TEXT,state TEXT)')
        self.write(path, 'INSERT OR REPLACE INTO inputs VALUES(?,?,?)',
                   (record['recording_id'], signature or record['source_signature'], state))

    def resource(self, resource, *, source='nas', state='running', expires=NOW + 90):
        self.write(self.root / 'state' / 'resources.sqlite3', 'INSERT INTO leases VALUES(?,?,?,?)',
                   (resource, source, state, expires))

    def monitor(self, *, age=0, status='watching', **extra):
        payload = dict(extra)
        payload.setdefault('truncated', False)
        payload.setdefault('pending_publication_count', 0)
        payload['monitor'] = {'status': status, 'poll_seconds': 5,
                             'observed_at': datetime.fromtimestamp(NOW - age, timezone.utc).isoformat()}
        (self.root / 'state' / 'nas-recording-monitor.json').write_text(json.dumps(payload, indent=2))


@pytest.fixture
def local(tmp_path):
    return LocalState(tmp_path / 'runtime')


def test_empty_fresh_catalog_is_idle_and_observation_probe_is_read_only(local, monkeypatch):
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in local.root.rglob('*') if p.is_file()}
    original = Path.open
    def open_local(path, *args, **kwargs):
        if str(path).startswith('/forbidden-nas'):
            raise AssertionError('Policy must not open source media')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_local)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    assert historical_candidates(local.config, 'vision', now=NOW) == []
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in local.root.rglob('*') if p.is_file()}
    assert after == before


@pytest.mark.parametrize('stage', ['vision', 'understanding', 'report'])
def test_not_enqueued_recent_source_reserves_shared_resources(local, stage):
    local.observe(recording('new'))
    value = live_demand(local.config, stage, NOW)
    assert value['status'] == 'waiting_for_live'
    assert value['counts']['eligible'] > 0


def test_live_upstream_blocks_history_even_when_requested_stage_not_yet_ready(local):
    row = recording('new', audio={'status': 'provided', 'source_signature': 'audio'})
    local.observe(row)
    local.queue('vision', row, status='running', lease_until=NOW + 90)
    local.queue('understanding', row, status='waiting_for_prerequisite')
    value = live_demand(local.config, 'understanding', NOW)
    assert value['counts']['running'] == 1
    assert value['status'] == 'waiting_for_live'


@pytest.mark.parametrize('state', ['missing', 'waiting', 'unavailable'])
def test_unavailable_matching_source_does_not_block_idle_forever(local, state):
    row = recording('unavailable')
    local.observe(row)
    local.unavailable(row, state=state)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'


def test_old_unavailable_revision_cannot_hide_new_live_source(local):
    row = recording('new-source')
    local.observe(row)
    local.unavailable(row, signature='old-source')
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_live'


def test_failed_maximum_retries_and_ineligible_children_do_not_block_forever(local):
    row = recording('failed', audio={'status': 'provided', 'source_signature': 'a'})
    local.observe(row)
    local.complete(row, ('retention', 'stt'))
    local.queue('vision', row, status='failed', attempts=3)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.queue('vision', row, status='failed', attempts=2, updated_at=NOW - 120)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_live'
    local.queue('vision', row, status='failed', attempts=2, updated_at=NOW - 10)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'


def test_expired_live_lease_is_recoverable_and_valid_lease_counts_even_when_source_missing(local):
    row = recording('lease')
    local.observe(row)
    local.complete(row)
    local.queue('vision', row, status='running', lease_until=NOW - 1)
    value = live_demand(local.config, 'vision', NOW)
    assert value['counts']['running'] == 0
    assert value['counts']['eligible'] == 1
    local.unavailable(row)
    local.queue('vision', row, status='running', lease_until=NOW + 1)
    assert live_demand(local.config, 'vision', NOW)['counts']['running'] == 1


@pytest.mark.parametrize('state', ['waiting', 'running'])
def test_non_backfill_resource_demand_reserves_capacity(local, state):
    local.resource('nas-io', state=state)
    value = live_demand(local.config, 'vision', NOW)
    assert value['counts']['resource_' + state] == 1
    assert value['status'] == 'waiting_for_live'


def test_self_resource_ignored_but_offline_work_and_live_reserved_lane_protected(local):
    local.resource('vision', source=BACKFILL_SOURCE)
    local.resource('vision', expires=NOW - 1)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.resource('nas-live-vision', source='offline')
    assert live_demand(local.config, 'report', NOW)['counts']['resource_running'] == 1


def test_unrelated_cloud_lease_does_not_hold_historical_vision(local):
    local.resource('cloud', source='offline')
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    assert live_demand(local.config, 'understanding', NOW)['status'] == 'waiting_for_live'


def test_recent_unpublished_result_blocks_even_after_stage_completion(local):
    row = recording('published-later')
    local.observe(row)
    local.complete(row)
    path = local.base / 'publication-journal.sqlite3'
    local.write(path, 'CREATE TABLE dirty(record TEXT)')
    local.write(path, 'INSERT INTO dirty VALUES(?)', (json.dumps(row),))
    value = live_demand(local.config, 'report', NOW)
    assert value['counts']['pending_publication'] == 1
    assert value['status'] == 'waiting_for_live'


@pytest.mark.parametrize('monitor', [
    {'age': 31}, {'age': -10}, {'status': 'retrying'}, {'truncated': True},
    {'pending_publication_count': 1},
    {'pending_publication_count': 78, 'pending_live_publication_count': 1},
    {'discovery_lanes': [{'mode': 'live', 'status': 'failed', 'thread_alive': False}]},
    {'discovery_lanes': [{'mode': 'live', 'status': 'watching', 'thread_alive': True,
                         'completed_at': NOW - 31}]},
])
def test_stale_or_incomplete_monitor_fails_closed(local, monitor):
    local.monitor(**monitor)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'
    assert historical_candidates(local.config, 'vision', now=NOW) == []


def test_missing_and_corrupt_local_files_fail_closed_without_creation(local):
    monitor = local.root / 'state' / 'nas-recording-monitor.json'
    monitor.unlink()
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'
    assert not monitor.exists()
    local.monitor()
    queue = local.base / 'queue-report.sqlite3'
    queue.unlink()
    assert live_demand(local.config, 'vision', NOW)['status'] == 'status_unavailable'
    assert not queue.exists()


def test_large_canonical_monitor_does_not_decode_entire_inventory(local, monkeypatch):
    local.monitor(recordings=[{'opaque': 'x' * 300000}], truncated=False,
                  discovery_lanes=[{'camera_key': 'a', 'mode': 'live', 'status': 'watching', 'thread_alive': True,
                                    'completed_at': NOW}])
    decoder = json.loads
    def bounded(value, *args, **kwargs):
        assert len(value) <= 256 * 1024
        return decoder(value, *args, **kwargs)
    monkeypatch.setattr(json, 'loads', bounded)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'


def successful_lane(camera='a', *, completed_at=NOW - 1, **extra):
    return {'camera_key': camera, 'mode': 'live', 'status': 'watching', 'thread_alive': True,
            'completed_at': completed_at, 'errors': [], 'truncated': False, **extra}


def scanning_lane(camera='a', *, started_at=NOW, **extra):
    # The existing discovery worker intentionally omits its previous completion
    # and error/truncation fields while an ordinary polling scan is in progress.
    return {'camera_key': camera, 'mode': 'live', 'status': 'scanning', 'thread_alive': True,
            'started_at': started_at, **extra}


def test_verified_success_keeps_normal_poll_idle_without_hiding_new_live_work(local):
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'
    # A normal scan never grants an exemption from the current catalog or
    # shared-resource checks which reserve capacity for actual latest arrivals.
    local.observe(recording('new'))
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'waiting_for_live'
    local.monitor(pending_live_publication_count=1, discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 3)['status'] == 'waiting_for_monitor'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 4)['status'] == 'waiting_for_monitor'


def test_first_scan_and_another_runtime_cannot_reuse_success(local, tmp_path):
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    other = LocalState(tmp_path / 'another-runtime')
    other.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(other.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'


def test_large_monitor_success_is_isolated_and_invalidated_by_runtime(local, tmp_path):
    padding = {'recordings': [{'opaque': 'x' * 300000}]}
    local.monitor(**padding, discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(**padding, discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'
    other = LocalState(tmp_path / 'another-runtime')
    other.monitor(**padding, discovery_lanes=[scanning_lane()])
    assert live_demand(other.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'
    # Failure of that separate root must neither clear this evidence nor lend
    # it to another large snapshot which uses the same metadata field names.
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'idle'
    local.monitor(**padding, status='retrying', discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 3)['status'] == 'waiting_for_monitor'
    local.monitor(**padding, discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 4)['status'] == 'waiting_for_monitor'


@pytest.mark.parametrize('change', [
    {'thread_alive': False}, {'thread_alive': None}, {'status': 'retrying'}, {'status': 'failed'},
    {'status': 'unknown'}, {'errors': [{'error_type': 'OSError'}]}, {'errors': None},
    {'errors': ''}, {'truncated': True}, {'truncated': None}, {'started_at': None},
    {'started_at': float('nan')}, {'started_at': True}, {'started_at': NOW - 2},
    {'started_at': NOW + 6}, {'camera_key': None},
])
def test_abnormal_scan_invalidates_success_instead_of_reusing_it_later(local, change):
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane(**change)])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'


def test_completed_success_and_scan_start_both_must_remain_fresh(local):
    local.monitor(discovery_lanes=[successful_lane(completed_at=NOW - 29)])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'waiting_for_monitor'
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'idle'
    local.monitor(age=-32, discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 32)['status'] == 'waiting_for_monitor'


@pytest.mark.parametrize('skew', [.001, 5])
def test_new_scan_can_start_after_callers_clock_sample_within_existing_skew(local, skew):
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane(started_at=NOW + skew)])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'


@pytest.mark.parametrize('missing', ['truncated', 'pending_counts', 'unknown_truncated'])
def test_unknown_global_completeness_cannot_seed_or_inherit_scan_success(local, missing):
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    path = local.root / 'state' / 'nas-recording-monitor.json'
    def remove_completeness():
        payload = json.loads(path.read_text())
        if missing == 'pending_counts':
            payload.pop('pending_publication_count', None)
            payload.pop('pending_live_publication_count', None)
        elif missing == 'unknown_truncated':
            payload['truncated'] = None
        else:
            payload.pop('truncated')
        path.write_text(json.dumps(payload, indent=2))
    local.monitor(discovery_lanes=[scanning_lane()])
    remove_completeness()
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'
    # A legacy watching snapshot still follows its original gate, but cannot
    # restore the successful evidence which that unknown snapshot invalidated.
    local.monitor(discovery_lanes=[successful_lane()])
    remove_completeness()
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'waiting_for_monitor'


@pytest.mark.parametrize('failure', ['stale', 'corrupt', 'utf8', 'missing', 'truncated', 'partial_lane'])
def test_failed_monitor_snapshot_cannot_keep_or_partially_refresh_success(local, failure):
    local.monitor(discovery_lanes=[successful_lane('a'), successful_lane('b')])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    path = local.root / 'state' / 'nas-recording-monitor.json'
    if failure == 'stale':
        local.monitor(age=31, discovery_lanes=[successful_lane()])
    elif failure == 'corrupt':
        path.write_text('{')
    elif failure == 'utf8':
        path.write_bytes(b'\xff')
    elif failure == 'missing':
        path.unlink()
    elif failure == 'truncated':
        local.monitor(truncated=True, discovery_lanes=[successful_lane()])
    else:
        local.monitor(discovery_lanes=[successful_lane('a', completed_at=NOW),
                                      successful_lane('b', errors=[{'error_type': 'OSError'}])])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'
    local.monitor(discovery_lanes=[scanning_lane('a'), scanning_lane('b')])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'


@pytest.mark.parametrize('bad_now', [-1, float('nan'), float('inf')])
def test_invalid_clock_cannot_authorize_a_cached_scan(local, bad_now):
    from visioncortex.device_day_backfill_policy import _monitor
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    with pytest.raises(PolicyUnavailable):
        _monitor(local.config, local.root, bad_now)
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'


@pytest.mark.parametrize('regression', ['snapshot', 'completion'])
def test_time_reversal_invalidates_cached_success(local, regression):
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    if regression == 'snapshot':
        local.monitor(age=1, discovery_lanes=[scanning_lane()])
        now = NOW + 1
    else:
        local.monitor(discovery_lanes=[successful_lane(completed_at=NOW - 2)])
        now = NOW + 1
    assert live_demand(local.config, 'vision', now)['status'] == 'waiting_for_monitor'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'waiting_for_monitor'


def test_real_wall_clock_reversal_invalidates_success(local, monkeypatch):
    from visioncortex import device_day_backfill_policy as policy
    clock = [NOW]
    monkeypatch.setattr(policy.time, 'time', lambda: clock[0])
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    clock[0] = NOW - .5
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'
    clock[0] = NOW + 2
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'waiting_for_monitor'


def test_delayed_concurrent_caller_is_not_a_clock_reversal(local):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from visioncortex.device_day_backfill_policy import _monitor
    local.monitor(discovery_lanes=[successful_lane(completed_at=NOW - 2)])
    _monitor(local.config, local.root, NOW)
    local.monitor(discovery_lanes=[scanning_lane(started_at=NOW - 1)])
    newer_read = Event()
    def delayed_older_read():
        assert newer_read.wait(3)
        return _monitor(local.config, local.root, NOW)
    def newer_read_first():
        result = _monitor(local.config, local.root, NOW + .5)
        newer_read.set()
        return result
    with ThreadPoolExecutor(2) as pool:
        delayed = pool.submit(delayed_older_read)
        newer = pool.submit(newer_read_first)
        assert newer.result(3) >= 0
        assert delayed.result(3) >= 0
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'


def test_older_caller_can_reuse_a_newer_verified_completion_during_normal_scan(local):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from visioncortex.device_day_backfill_policy import _monitor
    newer_read = Event()
    def delayed_older_read():
        assert newer_read.wait(3)
        return _monitor(local.config, local.root, NOW)
    def newer_success_first():
        local.monitor(age=-.5, discovery_lanes=[successful_lane(completed_at=NOW + .2)])
        result = _monitor(local.config, local.root, NOW + .5)
        local.monitor(age=-.5, discovery_lanes=[scanning_lane(started_at=NOW + .3)])
        newer_read.set()
        return result
    with ThreadPoolExecutor(2) as pool:
        delayed = pool.submit(delayed_older_read)
        newer = pool.submit(newer_success_first)
        assert newer.result(3) >= 0
        assert delayed.result(3) >= 0
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'


def test_cached_completion_more_than_five_seconds_ahead_of_caller_fails_closed(local):
    local.monitor(discovery_lanes=[successful_lane(completed_at=NOW + 6)])
    assert live_demand(local.config, 'vision', NOW + 6)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane(started_at=NOW + 6)])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'
    assert live_demand(local.config, 'vision', NOW + 7)['status'] == 'waiting_for_monitor'


def test_success_is_not_reused_for_another_or_removed_camera(local):
    local.monitor(discovery_lanes=[successful_lane('a')])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane('b')])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'
    local.monitor(discovery_lanes=[successful_lane('a'), successful_lane('b')])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'
    local.monitor(discovery_lanes=[successful_lane('b')])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane('a'), scanning_lane('b')])
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'waiting_for_monitor'


def test_legacy_missing_health_fields_cannot_seed_scan_success(local):
    incomplete = successful_lane()
    del incomplete['errors']
    del incomplete['truncated']
    local.monitor(discovery_lanes=[incomplete])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'


def test_success_cache_eviction_requires_a_new_verified_snapshot(local, tmp_path, monkeypatch):
    from visioncortex import device_day_backfill_policy as policy
    monkeypatch.setattr(policy, '_MONITOR_ROOT_LIMIT', 2)
    local.monitor(discovery_lanes=[successful_lane()])
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    for n in range(2):
        other = LocalState(tmp_path / str(n))
        other.monitor(discovery_lanes=[successful_lane()])
        assert live_demand(other.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(discovery_lanes=[scanning_lane()])
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'waiting_for_monitor'


def test_history_selection_is_bounded_newest_first_and_ignores_ineligible_rows(local):
    rows = [recording(str(n), age=200 + n) for n in range(30)]
    local.observe(*rows, recording('latest'), recording('unbound', age=150, configured_role=None))
    local.queue('vision', rows[0], status='completed')
    local.queue('vision', rows[1], status='failed', attempts=3)
    local.unavailable(rows[2])
    local.queue('vision', rows[3], status='running', lease_until=NOW + 90)
    local.queue('vision', rows[4], status='running', lease_until=NOW - 1)
    assert historical_candidate_ids(local.config, 'vision', limit=2, now=NOW) == ['4', '5']
    assert historical_candidate_ids(local.config, 'vision', limit=2, now=NOW, exclude={'4', '5'}) == ['6', '7']


def test_old_publication_backlog_does_not_hide_idle_capacity(local):
    old = recording('history-publication', age=200)
    path = local.base / 'publication-journal.sqlite3'
    local.write(path, 'CREATE TABLE dirty(record TEXT)')
    local.write(path, 'INSERT INTO dirty VALUES(?)', (json.dumps(old),))
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.monitor(pending_publication_count=78, pending_live_publication_count=0)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'


def test_provider_blocked_understanding_does_not_permanently_hold_history(local):
    import hashlib
    row = recording('cloud', audio={'status': 'provided', 'source_signature': 'audio'})
    local.observe(row)
    local.complete(row, ('retention', 'vision', 'stt'))
    local.config['runtime'] = {'provider_circuit_enabled': True}
    local.config['mllm'] = {'provider': 'example', 'base_url': 'https://example.invalid'}
    binding = local.config['mllm']
    values = [binding.get(k) for k in ('provider', 'base_url', 'credential_ref', 'api_key_env')]
    key = hashlib.sha256(json.dumps([*values, 'understanding'], sort_keys=True).encode()).hexdigest()
    path = local.root / 'state' / 'provider-circuits.sqlite3'
    local.write(path, 'CREATE TABLE circuits(key TEXT,reason TEXT,retry_at REAL,probe_until REAL)')
    local.write(path, 'INSERT INTO circuits VALUES(?,?,?,?)', (key, 'transport_unavailable', NOW + 30, 0))
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    local.write(path, 'UPDATE circuits SET retry_at=?', (NOW - 1,))
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_live'


def test_history_uses_current_signature_and_explicit_configured_role(local):
    row = recording('revised', age=200, configured_role=None)
    old = row | {'source_signature': 'old'}
    local.observe(row)
    local.config['collection_ingest']['camera_role_map'] = {'a': 'third_person'}
    local.queue('vision', old, status='completed')
    selected = historical_candidates(local.config, 'vision', now=NOW)
    assert selected[0]['source_signature'] == 'revised'
    assert selected[0]['configured_role'] == 'third_person'


@pytest.mark.parametrize('authorization', ['current', 'old_revision', 'old_limit', 'absent'])
def test_audited_repair_history_is_prioritized_only_for_exact_authorization(local, authorization):
    local.config['device_day']['backfill'] = {'mode': 'fair'}
    older, repaired = recording('older', age=500), recording('repaired', age=200)
    local.observe(older, repaired)
    local.complete(older, ('retention',))
    local.complete(repaired, ('retention',))
    local.queue('vision', older)
    local.queue('vision', repaired, status='failed', attempts=3)
    path = local.base / 'queue-vision.sqlite3'
    local.write(path, "ALTER TABLE recordings ADD COLUMN retry_limit INTEGER DEFAULT 0")
    local.write(path, "ALTER TABLE recordings ADD COLUMN revision TEXT DEFAULT 'current'")
    local.write(path, "CREATE TABLE retry_authorizations(recording_id TEXT,revision TEXT,retry_limit INTEGER)")
    local.write(path, "UPDATE recordings SET retry_limit=4 WHERE recording_id='repaired'")
    if authorization != 'absent':
        local.write(path, 'INSERT INTO retry_authorizations VALUES(?,?,?)',
                    ('repaired', 'old' if authorization == 'old_revision' else 'current',
                     3 if authorization == 'old_limit' else 4))
    expected = ['repaired', 'older'] if authorization == 'current' else ['older', 'repaired']
    assert historical_candidate_ids(local.config, 'vision', now=NOW) == expected
    # Priority cannot bypass exhausted budgets, missing inputs or scope.
    local.write(path, "UPDATE recordings SET attempts=4 WHERE recording_id='repaired'")
    assert historical_candidate_ids(local.config, 'vision', now=NOW) == ['older']


def test_downstream_waits_for_current_transitive_parents(local):
    row = recording('downstream', age=200, audio={'status': 'provided', 'source_signature': 'new-audio'})
    local.observe(row)
    local.complete(row, ('retention', 'vision'))
    assert historical_candidate_ids(local.config, 'understanding', now=NOW) == []
    local.queue('stt', row | {'audio': {'source_signature': 'old-audio'}}, status='completed')
    assert historical_candidate_ids(local.config, 'understanding', now=NOW) == []
    local.queue('stt', row, status='completed')
    assert historical_candidate_ids(local.config, 'understanding', now=NOW) == ['downstream']
    assert historical_candidate_ids(local.config, 'report', now=NOW) == []
    local.queue('understanding', row, status='completed')
    assert historical_candidate_ids(local.config, 'report', now=NOW) == ['downstream']


def test_pause_scope_and_missing_audio_remain_respected(local):
    row = recording('history', age=200)
    local.observe(row)
    assert historical_candidate_ids(local.config, 'stt', now=NOW) == []
    local.config['device_day']['paused_stages'] = ['vision']
    assert historical_candidate_ids(local.config, 'vision', now=NOW) == []
    local.config['device_day']['paused_stages'] = []
    local.config['device_day']['process_since_us'] = int((NOW - 100) * 1_000_000)
    assert historical_candidate_ids(local.config, 'vision', now=NOW) == []


def test_corrupt_inventory_is_reported_instead_of_empty_backlog(local):
    local.write(local.base / 'observed-inventory.sqlite3',
                'INSERT INTO observations VALUES(?,?)', ('corrupt', '{broken'))
    assert live_demand(local.config, 'vision', NOW)['status'] == 'status_unavailable'
    with pytest.raises(PolicyUnavailable, match='historical_catalog_unavailable'):
        historical_candidate_ids(local.config, 'vision', now=NOW)
