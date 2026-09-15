"""Rebuildable, incremental local projection of authoritative stage receipts."""
import json
from pathlib import Path
from .sqlite_store import connection


class ReceiptProjection:
    def __init__(self, runtime_root):
        self.path = Path(runtime_root) / 'receipt-projection.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS days(name TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS receipts(day TEXT, id TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY(day,id));''')

    def records(self, layout, stages, *, recording_id=None):
        from .device_day_contract import read_json
        def read(folder):
            return {s: read_json(folder / f'{s}.json') for s in stages if (folder / f'{s}.json').is_file()}
        # The caller owns the device/day file lock. Do NAS reads before the
        # short SQLite transaction so another day's writer is never blocked by I/O.
        with connection(self.path, readonly=True) as db:
            initialized = db.execute('SELECT 1 FROM days WHERE name=?', (layout.name,)).fetchone()
        full = recording_id is None or not initialized
        if full:
            rows = [(folder.name, read(folder)) for folder in sorted(layout.receipts.glob('*')) if folder.is_dir()]
        else:
            if Path(recording_id).name != recording_id:
                raise ValueError('Invalid recording identity')
            rows = [(recording_id, read(layout.receipts / recording_id))]
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            if full:
                db.execute('DELETE FROM receipts WHERE day=?', (layout.name,))
                db.execute('INSERT OR IGNORE INTO days VALUES(?)', (layout.name,))
            db.executemany('INSERT OR REPLACE INTO receipts VALUES(?,?,?)',
                           [(layout.name, identifier, json.dumps(payload)) for identifier, payload in rows])
            return [(row[0], json.loads(row[1])) for row in db.execute(
                'SELECT id,payload FROM receipts WHERE day=? ORDER BY id', (layout.name,))]
