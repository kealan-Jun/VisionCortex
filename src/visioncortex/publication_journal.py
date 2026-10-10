"""Persist unfinished publication before modifying receipts; replay without CV."""
import json
from contextlib import ExitStack
from pathlib import Path
import uuid
import time
from .sqlite_store import connection


class PublicationJournal:
    def __init__(self, root):
        self.path = Path(root) / 'publication-journal.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS dirty(id TEXT PRIMARY KEY, token TEXT NOT NULL, record TEXT NOT NULL);''')

    def begin(self, record):
        token = uuid.uuid4().hex
        with connection(self.path) as db:
            db.execute('INSERT OR REPLACE INTO dirty VALUES(?,?,?)',
                       (record['recording_id'], token, json.dumps(record)))
        return token

    def complete(self, identifier, token):
        with connection(self.path) as db:
            db.execute('DELETE FROM dirty WHERE id=? AND token=?', (identifier, token))

    def pending(self, limit=32, *, since_us=None, prefer_since_us=None):
        with connection(self.path, readonly=True) as db:
            return [(row['token'], json.loads(row['record'])) for row in db.execute(
                "SELECT * FROM dirty WHERE ? IS NULL OR MAX(COALESCE(json_extract(record,'$.recording_start_us'),0),"
                "COALESCE(json_extract(record,'$.recording_end_us'),0))>=? ORDER BY "
                "CASE WHEN ? IS NOT NULL AND COALESCE(json_extract(record,'$.recording_start_us'),0)>=? "
                "THEN 0 ELSE 1 END,rowid LIMIT ?",
                (since_us, since_us, prefer_since_us, prefer_since_us, limit))]

    def defer(self, identifier, token):
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT record FROM dirty WHERE id=? AND token=?', (identifier, token)).fetchone()
            if row:
                db.execute('DELETE FROM dirty WHERE id=? AND token=?', (identifier, token))
                db.execute('INSERT INTO dirty VALUES(?,?,?)', (identifier, token, row['record']))


def reconcile(runner):
    from .device_day import exclusive
    from .device_day_contract import STAGES
    journal = PublicationJournal(runner.runtime_root)
    from .device_day_schedule import processing_cutoff
    completed = 0
    recent = round((time.time() - runner.settings.get('live_priority_seconds', 14400)) * 1e6)
    for token, record in journal.pending(since_us=processing_cutoff(runner.settings), prefer_since_us=recent):
        try:
            # Never acknowledge an in-flight stage before its receipt exists.
            # A crash between receipt and index publication must remain replayable.
            with ExitStack() as locks:
                for stage in ('all', *STAGES):
                    locks.enter_context(exclusive(runner.runtime_root / 'locks' /
                        f"{record['recording_id']}.{stage}.lock"))
                from .device_day_inplace import active, publish
                done = (publish(runner, record) if active(runner, record) else
                        runner.refresh_index(runner.layout(record), recording_id=record['recording_id']))
                if done:
                    journal.complete(record['recording_id'], token)
                    completed += 1
                else:
                    journal.defer(record['recording_id'], token)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # One unavailable archive must not block other recoverable days.
            # The unacknowledged item stays pending for a later bounded round.
            if not isinstance(exc, BlockingIOError):
                import logging
                logging.getLogger(__name__).warning('Publication replay deferred %s: %s',
                                                    record['recording_id'], type(exc).__name__)
            journal.defer(record['recording_id'], token)
    return completed
