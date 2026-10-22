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


def test_changed_output_wakes_fifo_publication_but_same_output_preserves_backoff(tmp_path):
    journal = PublicationJournal(tmp_path)
    old = {'recording_id': 'old', 'recording_start_us': 1}
    live = {'recording_id': 'live', 'recording_start_us': 1000000}
    first, second = journal.begin(old), journal.begin(live)
    journal.defer('old', first, reason='old-source-failure')
    assert journal.notice_output('old', first, 'new-sealed-video')
    assert journal.pending(1, ready_only=True) == [(first, old)]
    journal.defer('old', first, reason='still-unverified')
    assert not journal.notice_output('old', first, 'new-sealed-video')
    assert journal.pending(1, ready_only=True) == [(second, live)]
    with sqlite3.connect(journal.path) as db:
        assert db.execute("SELECT attempts,reason FROM retries WHERE id='old'").fetchone() == (2, 'still-unverified')
    assert journal.notice_output('old', first, 'second-sealed-video')
    assert journal.pending(1, ready_only=True) == [(first, old)]


def test_output_notice_is_bound_to_existing_source_generation_and_completes_once(tmp_path):
    journal = PublicationJournal(tmp_path)
    token = journal.begin({'recording_id': 'r'})
    assert not journal.notice_output('r', 'other-token', 'new-output')
    assert journal.notice_output('r', token, 'new-output')
    journal.complete('r', token)
    assert not journal.notice_output('r', token, 'new-output')
    with sqlite3.connect(journal.path) as db:
        assert db.execute('SELECT COUNT(*) FROM outputs').fetchone()[0] == 0


def test_latest_first_publication_keeps_capture_order_ahead_of_old_output_notice(tmp_path):
    journal = PublicationJournal(tmp_path)
    old = {'recording_id': 'old', 'recording_start_us': 1}
    new = {'recording_id': 'new', 'recording_start_us': 2}
    first, second = journal.begin(old), journal.begin(new)
    journal.notice_output('old', first, 'old-video-ready')
    assert journal.pending(1, latest_first=True, ready_only=True) == [(second, new)]
