"""A display rollout cannot silently rebill a completed unchanged source."""
import json

import pytest

from visioncortex.device_day_queue import DeviceDayQueue


@pytest.mark.parametrize('stage', ['retention', 'vision', 'stt', 'understanding', 'report'])
def test_preserve_completion_across_recipe_changes_but_accept_a_new_source(tmp_path, stage):
    queue = DeviceDayQueue(tmp_path/f'queue-{stage}.sqlite3', preserve_completed_on_recipe_change=True)
    record = {'recording_id': 'capture', 'configured_role': 'first_person', 'source_signature': 'a'*64}
    queue.enqueue(record, 'old-recipe')
    assert queue.claim('owner')['recording_id'] == 'capture'
    queue.finish('owner', 'capture', {'status': 'completed', 'receipt': 'original'}, 2)
    with queue.connect() as db:
        before = dict(db.execute('SELECT * FROM recordings').fetchone())
    assert queue.enqueue(record | {'processing_priority': -100}, 'new-display-recipe') is False
    with queue.connect() as db:
        assert dict(db.execute('SELECT * FROM recordings').fetchone()) == before
    assert queue.claim('must-not-rebill') is None
    queue.enqueue(record | {'source_signature': 'b'*64}, 'new-source-recipe')
    assert queue.claim('new-source')['source_signature'] == 'b'*64


def test_independent_late_audio_reenters_stt_without_reusing_no_audio_completion(tmp_path):
    queue = DeviceDayQueue(tmp_path/'queue-stt.sqlite3', preserve_completed_on_recipe_change=True)
    old = {'recording_id': 'capture', 'source_signature': 'a'*64, 'audio': {'status': 'not_provided'}}
    queue.enqueue(old, 'no-audio-recipe')
    assert queue.claim('owner')
    queue.finish('owner', 'capture', {'status': 'completed', 'outcome': 'no_audio'}, 0)
    late = old | {'recording_session_id': 'session', 'camera_key': 'camera', 'stt_processable': True,
                  'audio': {'status': 'provided', 'source_signature': 'audio-source', 'start_us': 1, 'end_us': 100}}
    queue.enqueue(late, 'actual-audio-recipe')
    assert queue.claim('actual-audio')['audio']['status'] == 'provided'
    with queue.connect() as db:
        archived = db.execute('SELECT snapshot FROM source_versions').fetchone()[0]
    assert json.loads(archived)['status'] == 'completed'
