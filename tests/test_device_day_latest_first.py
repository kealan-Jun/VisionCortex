"""Latest-first changes admission only, preserving receipts and live leases."""
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import json

import pytest

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_schedule import refresh_queue_priorities, scheduling_record


def record(name, stamp, *, camera='camera', priority=2):
    start = int(datetime.fromisoformat(stamp).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp() * 1e6)
    return {'recording_id': name, 'camera_key': camera, 'configured_role': 'first_person',
            'recording_start_us': start, 'recording_end_us': start + 60_000_000,
            'processing_priority': priority}


@pytest.mark.parametrize('camera_serial', [True, False])
def test_latest_wins_across_midnight_four_hours_and_saved_focus(tmp_path, camera_serial):
    queue = DeviceDayQueue(tmp_path / 'queue-understanding.sqlite3', latest_first=True)
    rows = [record('focus', '2026-09-15T23:00:00', priority=1),
            record('older_yesterday', '2026-09-24T08:00:00'),
            record('late_yesterday', '2026-09-24T23:55:00'),
            record('today', '2026-09-25T00:01:00', priority=-1)]
    for row in rows:
        classified = scheduling_record(row, today='2026-09-25', focus_date='2026-09-15',
                                       now_us=rows[-1]['recording_start_us'], latest_first=True)
        assert classified['processing_priority'] == -1
        # Also tolerate stale pre-switch queue metadata until its refresh.
        queue.enqueue(row, 'same-input-revision')
    for expected in ('today', 'late_yesterday', 'older_yesterday', 'focus'):
        selected = queue.claim('owner', camera_serial=camera_serial)
        assert selected['recording_id'] == expected
        queue.finish('owner', expected, {'status': 'completed'}, 1)


def test_fairness_and_nonpreemption_survive_latest_first(tmp_path):
    queue = DeviceDayQueue(tmp_path / 'queue-understanding.sqlite3', latest_first=True)
    for row in (record('a-old', '2026-09-24T09:00:00', camera='a'),
                record('a-new', '2026-09-24T23:00:00', camera='a'),
                record('b-new', '2026-09-24T22:00:00', camera='b')):
        queue.enqueue(row, 'original')
    assert queue.claim('a-owner', camera_serial=True)['recording_id'] == 'a-new'
    arrival = record('a-arrival', '2026-09-25T00:01:00', camera='a')
    queue.enqueue(arrival, 'new-input')
    # Newer arrival cannot take an occupied camera or interrupt its lease.
    assert queue.claim('b-owner', camera_serial=True)['recording_id'] == 'b-new'
    assert queue.claim('extra', camera_serial=True) is None
    queue.finish('a-owner', 'a-new', {'status': 'completed'}, 1)
    queue.finish('b-owner', 'b-new', {'status': 'completed'}, 1)
    queue.enqueue(record('b-next', '2026-09-24T10:00:00', camera='b'), 'original')
    assert queue.claim('next', camera_serial=True)['recording_id'] == 'a-arrival'
    queue.finish('next', 'a-arrival', {'status': 'completed'}, 1)
    # Fair camera turn still precedes draining all slices from camera a.
    assert queue.claim('fair', camera_serial=True)['recording_id'] == 'b-next'


def test_reclassification_and_enqueue_preserve_completions_leases_and_retry_history(tmp_path):
    queue = DeviceDayQueue(tmp_path / 'queue-understanding.sqlite3', latest_first=True)
    rows = {name: record(name, f'2026-09-24T{hour}:00:00')
            for name, hour in [('done', '23'), ('running', '22'), ('waiting', '21')]}
    for row in rows.values():
        queue.enqueue(row, 'immutable')
    assert queue.claim('done-owner')['recording_id'] == 'done'
    queue.finish('done-owner', 'done', {'status': 'completed', 'proof': 'unchanged'}, 7)
    assert queue.claim('running-owner')['recording_id'] == 'running'
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=2 WHERE recording_id='waiting'")
        before = {r['recording_id']: dict(r) for r in db.execute('SELECT * FROM recordings')}
    refresh_queue_priorities(queue, '2026-09-15', latest_first=True)
    for row in rows.values():
        queue.enqueue(scheduling_record(row, latest_first=True), 'immutable')
    with queue.connect() as db:
        after = {r['recording_id']: dict(r) for r in db.execute('SELECT * FROM recordings')}
    assert after['done'] == before['done']
    assert after['running'] == before['running']
    assert after['waiting'] == before['waiting'] | {'payload': after['waiting']['payload']}
    assert json.loads(after['waiting']['payload'])['processing_priority'] == -1
    assert queue.claim('no-retry') is None
    assert queue.claim('retry', retry=True)['recording_id'] == 'waiting'


def test_retention_expiry_safety_still_precedes_latest(tmp_path):
    queue = DeviceDayQueue(tmp_path / 'queue-retention.sqlite3', latest_first=True)
    queue.enqueue(record('newest', '2026-09-25T00:01:00'), 'same')
    queue.enqueue(record('expiring', '2026-09-03T08:00:00') |
                  {'archive_urgent': True, 'archive_deadline': 123}, 'same')
    assert queue.claim('archive-owner', camera_serial=True)['recording_id'] == 'expiring'


@pytest.mark.parametrize('aged', [True, False])
def test_retention_unknown_deadline_uses_latest_not_backlog_age_or_enqueue_order(tmp_path, aged):
    queue = DeviceDayQueue(tmp_path / 'queue-retention.sqlite3', latest_first=True)
    queue.enqueue(record('old', '2026-09-15T08:00:00') |
                  {'archive_aged': aged, 'archive_deadline': None}, 'original-old')
    queue.enqueue(record('new', '2026-09-24T23:55:00') |
                  {'archive_aged': False, 'archive_deadline': None}, 'original-new')
    assert queue.claim('archive-owner', camera_serial=True)['recording_id'] == 'new'


def test_retention_known_deadlines_and_explicit_cleanup_risk_remain_protected(tmp_path):
    queue = DeviceDayQueue(tmp_path / 'queue-retention.sqlite3', latest_first=True)
    rows = [record('newest', '2026-09-25T00:01:00'),
            record('deadline-later', '2026-09-15T08:00:00') | {'archive_deadline': 200},
            record('deadline-sooner', '2026-09-14T08:00:00') | {'archive_deadline': 100},
            record('cleanup-risk', '2026-09-13T08:00:00') |
            {'archive_urgent': True, 'archive_deadline': None}]
    for row in rows:
        queue.enqueue(row, 'same')
    for expected in ('cleanup-risk', 'deadline-sooner', 'deadline-later', 'newest'):
        assert queue.claim('owner', camera_serial=True)['recording_id'] == expected
        queue.finish('owner', expected, {'status': 'completed'}, 1)


def test_legacy_retention_still_uses_backlog_age_before_latest(tmp_path):
    queue = DeviceDayQueue(tmp_path / 'queue-retention.sqlite3')
    queue.enqueue(record('old', '2026-09-15T08:00:00') | {'archive_aged': True}, 'same')
    queue.enqueue(record('new', '2026-09-24T23:55:00', priority=-1), 'same')
    assert queue.claim('owner', camera_serial=True)['recording_id'] == 'old'


@pytest.fixture
def device_config(default_config, tmp_path):
    from test_device_day import device_config as fixture
    return fixture.__wrapped__(default_config, tmp_path)


@pytest.mark.parametrize('inplace', [False, True])
def test_latest_first_does_not_change_any_stage_key_or_reexecute_completion(device_config, inplace):
    from test_device_day import FakeModels, capture, item_and_layout
    from visioncortex.device_day import DeviceDayRunner, STAGES
    from visioncortex.device_day_contract import read_json
    device_config['device_day']['inplace_preprocessing'] = inplace
    capture(device_config)
    source, layout = item_and_layout(device_config)
    old = DeviceDayRunner(device_config, FakeModels())
    assert old.process(source)['status'] == 'completed'
    receipts = {stage: read_json(old._receipt(layout, source, stage)) for stage in STAGES}
    changed = deepcopy(device_config)
    changed['device_day']['latest_first'] = True
    backend = FakeModels()
    current = DeviceDayRunner(changed, backend)
    assert current._execution_identity == old._execution_identity
    for stage in STAGES:
        assert current._key(stage, source, {}) == old._key(stage, source, {})
    assert current.process(source)['status'] == 'completed'
    assert backend.vision_calls == backend.semantic_calls == 0
    assert {stage: read_json(current._receipt(layout, source, stage)) for stage in STAGES} == receipts


def test_default_queue_policy_retains_legacy_focus_priority(tmp_path):
    queue = DeviceDayQueue(tmp_path / 'queue-understanding.sqlite3')
    queue.enqueue(record('focus', '2026-09-15T12:00:00', priority=1), 'same')
    queue.enqueue(record('newer-history', '2026-09-24T12:00:00', priority=2), 'same')
    assert queue.claim('owner', camera_serial=True)['recording_id'] == 'focus'
