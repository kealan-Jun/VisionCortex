"""A refunded claim retains its exact, still-unused audited history turn."""
import json

import pytest
from test_device_day_backfill_policy import NOW, LocalState, recording

from visioncortex.device_day_backfill_policy import historical_candidate_ids
from visioncortex.device_day_queue import DeviceDayQueue


def audited_history(tmp_path, *, stage='vision', status='failed'):
    state = LocalState(tmp_path / 'runtime')
    state.config['device_day']['backfill'] = {'mode': 'fair'}
    audio = {'status': 'provided', 'source_signature': 'audio-current'}
    older = recording('older', age=500, audio=audio)
    repaired = recording('repaired', age=200, audio=audio)
    state.observe(older, repaired)
    state.complete(older)
    state.complete(repaired)
    path = state.base / f'queue-{stage}.sqlite3'
    path.unlink()  # Replace only this test's legacy fixture with the real queue.
    queue = DeviceDayQueue(path)
    queue.enqueue(older, 'older-revision')
    queue.enqueue(repaired, 'current-revision')
    failure = json.dumps({'status': 'failed', 'message': 'preserved historical failure'})
    state.write(path, "UPDATE recordings SET status=?,attempts=3,retry_limit=4,result=?,updated_at=0 "
                "WHERE recording_id='repaired'", (status, failure))
    state.write(path, 'INSERT INTO retry_authorizations VALUES(?,?,?,?,?,?,?,?,?)',
                ('repaired', 'current-revision', 'verified-fix', NOW - 120, 3, 0, 4,
                 failure, json.dumps({'status': 'PROVEN', 'fixture': True})))
    return state, queue


@pytest.mark.parametrize('result_status', [
    'waiting_for_publication', 'waiting_for_provider', 'running_elsewhere', 'paused_for_live', 'cancelled',
])
def test_real_queue_refund_retains_repair_priority_and_failure_audit(tmp_path, monkeypatch, result_status):
    state, queue = audited_history(tmp_path)
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: NOW)
    with queue.connect() as db:
        before = dict(db.execute('SELECT * FROM retry_authorizations').fetchone())
    receipt = state.root / 'completed-model-response.json'
    receipt.write_bytes(b'{"paid_response":"retained"}')
    receipt_before = receipt.read_bytes(), receipt.stat().st_mtime_ns
    assert queue.claim('owner', retry=True, allowed=['repaired'], max_attempts=3)['recording_id'] == 'repaired'
    queue.finish('owner', 'repaired', {'status': result_status}, 1)
    with queue.connect() as db:
        row = db.execute("SELECT * FROM recordings WHERE recording_id='repaired'").fetchone()
        assert (row['status'], row['attempts'], row['retry_limit']) == ('queued', 3, 4)
        assert dict(db.execute('SELECT * FROM retry_authorizations').fetchone()) == before
    assert historical_candidate_ids(state.config, 'vision', now=NOW, limit=1) == ['repaired']
    assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == receipt_before
    # The same grant does not prioritize an actual failed fourth attempt.
    queue.claim('next-owner', retry=True, allowed=['repaired'], max_attempts=3)
    queue.finish('next-owner', 'repaired', {'status': 'failed'}, 1)
    assert historical_candidate_ids(state.config, 'vision', now=NOW) == ['older']


@pytest.mark.parametrize('status', ['failed', 'queued'])
@pytest.mark.parametrize('authorization', ['current', 'old_revision', 'old_limit', 'other_record', 'absent'])
def test_priority_requires_exact_existing_authorization(tmp_path, status, authorization):
    state, queue = audited_history(tmp_path, status=status)
    changes = {'old_revision': "revision='old-revision'", 'old_limit': 'retry_limit=3',
               'other_record': "recording_id='another-record'"}
    if authorization == 'absent':
        state.write(queue.path, 'DELETE FROM retry_authorizations')
    elif authorization != 'current':
        state.write(queue.path, 'UPDATE retry_authorizations SET ' + changes[authorization])
    expected = ['repaired', 'older'] if authorization == 'current' else ['older', 'repaired']
    assert historical_candidate_ids(state.config, 'vision', now=NOW) == expected


@pytest.mark.parametrize('boundary,expected', [
    ('exhausted', ['older', 'repaired']),
    ('baseline_only', ['older', 'repaired']),
    ('waiting_for_prerequisite', ['older', 'repaired']),
    ('active_lease', ['older']),
    ('expired_lease', ['older', 'repaired']),
    ('completed', ['older']),
    ('queue_input_missing', ['older']),
    ('source_changed', ['older', 'repaired']),
    ('availability_missing', ['older']),
    ('outside_scope', ['older']),
])
def test_queued_priority_preserves_budget_input_scope_and_lease_boundaries(tmp_path, boundary, expected):
    state, queue = audited_history(tmp_path, status='queued')
    changes = {
        'exhausted': 'attempts=4', 'baseline_only': 'retry_limit=3',
        'waiting_for_prerequisite': "status='waiting_for_prerequisite'",
        'active_lease': f"status='running',lease_until={NOW + 90}",
        'expired_lease': f"status='running',lease_until={NOW - 1}",
        'completed': "status='completed'", 'queue_input_missing': "input_status='missing'",
        'source_changed': "payload=json_set(payload,'$.source_signature','old-source')",
    }
    if boundary in changes:
        state.write(queue.path, 'UPDATE recordings SET ' + changes[boundary] + " WHERE recording_id='repaired'")
    elif boundary == 'availability_missing':
        state.unavailable(recording('repaired', age=200))
    else:
        state.config['device_day']['process_since_us'] = int((NOW - 600) * 1_000_000)
        state.observe(recording('repaired', age=700))
    assert historical_candidate_ids(state.config, 'vision', now=NOW) == expected


@pytest.mark.parametrize('boundary', ['audio_changed', 'parent_missing'])
def test_queued_downstream_priority_requires_current_audio_and_all_parents(tmp_path, boundary):
    state, queue = audited_history(tmp_path, stage='understanding', status='queued')
    if boundary == 'audio_changed':
        state.write(queue.path, "UPDATE recordings SET payload=json_set(payload,'$.audio.source_signature','old-audio') "
                    "WHERE recording_id='repaired'")
        expected = ['older', 'repaired']
    else:
        state.write(state.base / 'queue-stt.sqlite3', "UPDATE recordings SET status='queued' WHERE recording_id='repaired'")
        expected = ['older']
    assert historical_candidate_ids(state.config, 'understanding', now=NOW) == expected
