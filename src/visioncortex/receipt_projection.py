"""Rebuildable, incremental local projection of authoritative stage receipts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
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
        """Compatibility for small callers; publication consumes iter_records."""
        return list(self.iter_records(layout, stages, recording_id=recording_id))

    def iter_records(self, layout, stages, *, recording_id=None):
        # The caller owns the device/day file lock. Do NAS reads before the
        # short SQLite transaction so another day's writer is never blocked by I/O.
        with connection(self.path, readonly=True) as db:
            initialized = db.execute('SELECT 1 FROM days WHERE name=?', (layout.name,)).fetchone()
        full = recording_id is None or not initialized
        if full:
            folders = [folder for folder in sorted(layout.receipts.glob('*')) if folder.is_dir()]
        else:
            if Path(recording_id).name != recording_id:
                raise ValueError('Invalid recording identity')
            folders = [layout.receipts / recording_id]
        # Stage one receipt at a time on the existing local runtime volume.
        # Do not hold a SQLite writer across NAS I/O or retain a whole day's
        # decoded receipts plus their serialized copies. Failure preserves the
        # previous generation; clean exit/exception removes these cache files.
        with TemporaryDirectory(prefix='receipt-projection-', dir=self.path.parent) as temporary:
            staged = []
            for number, folder in enumerate(folders):
                path = Path(temporary) / str(number)
                with path.open('wb') as handle:
                    handle.write(b'{')
                    separator = b''
                    for stage in stages:
                        source = folder / f'{stage}.json'
                        if not source.is_file():
                            continue
                        # Validate the exact text we stage, releasing its parse
                        # immediately. No second read can race a new receipt;
                        # no Python re-serialization of every audit field.
                        raw = source.read_text(encoding='utf-8')
                        json.loads(raw)
                        handle.write(separator + json.dumps(stage).encode('utf-8') + b':')
                        for offset in range(0, len(raw), 256 * 1024):
                            handle.write(raw[offset:offset + 256 * 1024].encode('utf-8'))
                        separator = b','
                        del raw
                    handle.write(b'}')
                staged.append((folder.name, path))
            with connection(self.path) as db:
                db.execute('BEGIN IMMEDIATE')
                if full:
                    db.execute('DELETE FROM receipts WHERE day=?', (layout.name,))
                    db.execute('INSERT OR IGNORE INTO days VALUES(?)', (layout.name,))
                for identifier, path in staged:
                    db.execute('INSERT OR REPLACE INTO receipts VALUES(?,?,?)',
                               (layout.name, identifier, path.read_text(encoding='utf-8')))
        with connection(self.path, readonly=True) as db:
            for row in db.execute(
                'SELECT id,payload FROM receipts WHERE day=? ORDER BY id', (layout.name,)):
                yield row[0], json.loads(row[1])
