"""A repaired publisher may wake its own journal without resetting evidence."""
import sqlite3
import time

from visioncortex.publication_journal import PublicationJournal


def test_repair_wakeup_preserves_failures_and_cannot_wake_another_generation(tmp_path):
    journal = PublicationJournal(tmp_path)
    record = {'recording_id': 'r', 'source_signature': 'original-source'}
    token = journal.begin(record)
    journal.defer('r', token, reason='verification_failure')
    assert journal.pending(ready_only=True) == []
    assert journal.retry_now('r', 'unrelated-token') is False
    assert journal.pending(ready_only=True) == []
    assert journal.retry_now('r', token) is True
    assert journal.pending(ready_only=True) == [(token, record)]
    with sqlite3.connect(journal.path) as db:
        assert db.execute('SELECT attempts,reason,next_attempt_at FROM retries').fetchone() == (
            1, 'verification_failure', 0)
    journal.complete('r', token)
    assert journal.retry_now('r', token) is False


def test_old_token_cannot_reset_backoff_after_source_generation_replacement(tmp_path):
    journal = PublicationJournal(tmp_path)
    first = journal.begin({'recording_id': 'r', 'source_signature': 'first'})
    second = journal.begin({'recording_id': 'r', 'source_signature': 'second'})
    journal.defer('r', second, reason='new-source-unverified', now=time.time())
    assert journal.retry_now('r', first) is False
    assert journal.pending(ready_only=True) == []
