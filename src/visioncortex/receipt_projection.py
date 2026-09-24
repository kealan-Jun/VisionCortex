"""Rebuildable, incremental local projection of authoritative stage receipts."""
import json
from decimal import Decimal
import io
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from .sqlite_store import connection


_CHUNK_BYTES = 256 * 1024


class _ChunkReader(io.RawIOBase):
    """Present bounded SQLite BLOB rows as one JSON stream."""

    def __init__(self, rows):
        self.rows = iter(rows)
        self.remaining = memoryview(b'')

    def readable(self):
        return True

    def readinto(self, target):
        if not self.remaining:
            row = next(self.rows, None)
            if row is None:
                return 0
            self.remaining = memoryview(row[0])
        count = min(len(target), len(self.remaining))
        target[:count] = self.remaining[:count]
        self.remaining = self.remaining[count:]
        return count


def _json_events(handle):
    import ijson
    # Default number parsing preserves arbitrary Python integers. Convert only
    # non-integral JSON numbers back to float, matching the historical decoder.
    try:
        for event, value in ijson.basic_parse(handle):
            yield event, float(value) if isinstance(value, Decimal) else value
    except ijson.JSONError as exc:
        raise ValueError('Invalid receipt JSON') from exc


def _read_chunks(rows):
    from ijson.common import ObjectBuilder
    builder = ObjectBuilder()
    with io.BufferedReader(_ChunkReader(rows), buffer_size=_CHUNK_BYTES) as handle:
        for event, value in _json_events(handle):
            builder.event(event, value)
    return builder.value


class ReceiptProjection:
    def __init__(self, runtime_root):
        # Older live workers cannot decode chunked rows. This rebuildable cache
        # uses its own version so a rolling upgrade never changes their schema
        # or removes the previous projection required for a rollback.
        self.path = Path(runtime_root) / 'receipt-projection-v2.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS days(name TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS receipts(day TEXT, id TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY(day,id));
                CREATE TABLE IF NOT EXISTS receipt_chunks(day TEXT, id TEXT, ordinal INTEGER,
                    payload BLOB NOT NULL, PRIMARY KEY(day,id,ordinal),
                    FOREIGN KEY(day,id) REFERENCES receipts(day,id) ON DELETE CASCADE);''')

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
                        handle.write(separator + json.dumps(stage).encode('utf-8') + b':')
                        if source.stat().st_size > _CHUNK_BYTES:
                            with source.open('rb') as reader:
                                shutil.copyfileobj(reader, handle, length=_CHUNK_BYTES)
                        else:
                            raw = source.read_text(encoding='utf-8')
                            json.loads(raw)
                            handle.write(raw.encode('utf-8'))
                            del raw
                        separator = b','
                    handle.write(b'}')
                # Validate the staged bytes, including large sources, without
                # materializing another decoded receipt before the transaction.
                if path.stat().st_size > _CHUNK_BYTES:
                    with path.open('rb') as reader:
                        for _ in _json_events(reader):
                            pass
                staged.append((folder.name, path))
            with connection(self.path) as db:
                db.execute('BEGIN IMMEDIATE')
                if full:
                    db.execute('DELETE FROM receipts WHERE day=?', (layout.name,))
                    db.execute('INSERT OR IGNORE INTO days VALUES(?)', (layout.name,))
                for identifier, path in staged:
                    large = path.stat().st_size > _CHUNK_BYTES
                    db.execute('INSERT OR REPLACE INTO receipts VALUES(?,?,?)',
                               (layout.name, identifier, '' if large else path.read_text(encoding='utf-8')))
                    if large:
                        # SQLite's per-value length and Python's INT_MAX bind
                        # limit apply to each chunk, never the full receipt.
                        with path.open('rb') as reader:
                            for ordinal, chunk in enumerate(iter(lambda: reader.read(_CHUNK_BYTES), b'')):
                                db.execute('INSERT INTO receipt_chunks VALUES(?,?,?,?)',
                                           (layout.name, identifier, ordinal, chunk))
        with connection(self.path, readonly=True) as db:
            for row in db.execute(
                'SELECT id,payload FROM receipts WHERE day=? ORDER BY id', (layout.name,)):
                if row[1]:
                    yield row[0], json.loads(row[1])
                else:
                    chunks = db.execute('SELECT payload FROM receipt_chunks WHERE day=? AND id=? ORDER BY ordinal',
                                        (layout.name, row[0]))
                    yield row[0], _read_chunks(chunks)
