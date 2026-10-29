"""Durable changed-day/range notifications, acknowledged by generation."""
from pathlib import Path
import time
import uuid
from .sqlite_store import connection


class TimelineInvalidations:
    def __init__(self, root):
        self.path = Path(root) / 'timeline-invalidations.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS inputs(archive TEXT PRIMARY KEY, digest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pending(day TEXT PRIMARY KEY, token TEXT NOT NULL,
                    start_us INTEGER, end_us INTEGER, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS deferred_audits(day TEXT PRIMARY KEY,
                    token TEXT NOT NULL, updated REAL NOT NULL);''')

    def defer_audit(self, day):
        with connection(self.path) as db:
            db.execute('INSERT INTO deferred_audits VALUES(?,?,?) ON CONFLICT(day) '
                       'DO UPDATE SET token=excluded.token', (day, uuid.uuid4().hex, time.time()))

    def deferred_audits(self):
        with connection(self.path, readonly=True) as db:
            return [dict(row) for row in db.execute('SELECT * FROM deferred_audits ORDER BY updated,day')]

    def complete_audit(self, day, token):
        with connection(self.path) as db:
            db.execute('DELETE FROM deferred_audits WHERE day=? AND token=?', (day, token))

    def changed(self, archive, digest, start=None, end=None):
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT digest FROM inputs WHERE archive=?', (archive,)).fetchone()
            if row and row[0] == digest:
                return False
            db.execute('INSERT OR REPLACE INTO inputs VALUES(?,?)', (archive, digest))
            previous = db.execute('SELECT * FROM pending WHERE day=?', (archive[:10],)).fetchone()
            left = min(start, previous['start_us']) if previous and start is not None and previous['start_us'] is not None else start
            right = max(end, previous['end_us']) if previous and end is not None and previous['end_us'] is not None else end
            # Keep first arrival time for fairness; latest arrivals cannot starve another day.
            db.execute('INSERT OR REPLACE INTO pending VALUES(?,?,?,?,?)',
                       (archive[:10], uuid.uuid4().hex, left, right, previous['updated'] if previous else time.time()))
            return True

    def pending(self):
        with connection(self.path, readonly=True) as db:
            return [dict(row) for row in db.execute('SELECT * FROM pending ORDER BY updated,day')]

    def complete(self, day, token):
        with connection(self.path) as db:
            db.execute('DELETE FROM pending WHERE day=? AND token=?', (day, token))
