"""Rebuildable, incremental local projection of authoritative stage receipts."""
import json
import hashlib
from decimal import Decimal
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from .sqlite_store import connection


_CHUNK_BYTES = 256 * 1024
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024
_MAX_JSON_EVENTS = 2_000_000
_MAX_JSON_DEPTH = 128


class ProjectionDeferred(BlockingIOError):
    """Keep the old published index and journal pending; never trim evidence."""

    def __init__(self, metric, observed, limit, recording_id=None):
        super().__init__(f'Receipt projection deferred: {metric} {observed} exceeds {limit}')
        self.metric, self.observed, self.limit = metric, observed, limit
        self.recording_id = recording_id


def _check_size(size, recording_id=None):
    if size > _MAX_RECEIPT_BYTES:
        raise ProjectionDeferred('encoded_bytes', size, _MAX_RECEIPT_BYTES, recording_id)


class _ChunkReader(io.RawIOBase):
    """Present bounded SQLite BLOB rows as one JSON stream."""

    def __init__(self, rows):
        self.rows = iter(rows)
        self.remaining = memoryview(b'')
        self.total = 0

    def readable(self):
        return True

    def readinto(self, target):
        if not self.remaining:
            row = next(self.rows, None)
            if row is None:
                return 0
            self.total += len(row[0])
            _check_size(self.total)
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
        events, depth = 0, 0
        for event, value in ijson.basic_parse(handle):
            events += 1
            if events > _MAX_JSON_EVENTS:
                raise ProjectionDeferred('json_events', events, _MAX_JSON_EVENTS)
            if event in {'start_map', 'start_array'}:
                depth += 1
                if depth > _MAX_JSON_DEPTH:
                    raise ProjectionDeferred('json_depth', depth, _MAX_JSON_DEPTH)
            elif event in {'end_map', 'end_array'}:
                depth -= 1
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
        """Bound derived-cache hydration; preserve full authoritative receipts."""
        state = self.path.parent / 'ReceiptProjectionDeferred' / (
            hashlib.sha256(layout.name.encode()).hexdigest()[:24] + '.json')
        from .device_day_contract import atomic_json
        try:
            yield from self._iter_records(layout, stages, recording_id=recording_id)
        except ProjectionDeferred as exc:
            atomic_json(state, {'status': 'deferred', 'archive': layout.name,
                'recording_id': exc.recording_id, 'requested_recording_id': recording_id,
                'reason': 'receipt_projection_memory_limit', 'metric': exc.metric,
                'observed': exc.observed, 'limit': exc.limit, 'updated_at': time.time(),
                'scope': 'derived_index_rebuild', 'authoritative_receipts_modified': False})
            raise
        else:
            if state.exists():
                atomic_json(state, {'status': 'ready', 'archive': layout.name,
                                   'updated_at': time.time(), 'scope': 'derived_index_rebuild'})

    @staticmethod
    def _check_cached(db, day, *, replacing=()):
        # BLOB length uses SQLite's record metadata rather than hydrating its
        # overflow pages into Python. Check in the same read snapshot as decode.
        for row in db.execute('''SELECT id, CASE WHEN payload='' THEN
                COALESCE((SELECT SUM(length(c.payload)) FROM receipt_chunks c
                          WHERE c.day=receipts.day AND c.id=receipts.id),0)
                ELSE length(CAST(payload AS BLOB)) END AS bytes
                FROM receipts WHERE day=?''', (day,)):
            if row['id'] not in replacing:
                _check_size(row['bytes'], row['id'])

    def _iter_records(self, layout, stages, *, recording_id=None):
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
        with connection(self.path, readonly=True) as db:
            self._check_cached(db, layout.name, replacing={folder.name for folder in folders})
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
                        prefix = separator + json.dumps(stage).encode('utf-8') + b':'
                        _check_size(handle.tell() + len(prefix) + source.stat().st_size + 1, folder.name)
                        handle.write(prefix)
                        # Bound every actual read too: a new atomic receipt may
                        # replace the path after stat. Never read it as one string.
                        with source.open('rb') as reader:
                            while chunk := reader.read(_CHUNK_BYTES):
                                _check_size(handle.tell() + len(chunk) + 1, folder.name)
                                handle.write(chunk)
                        separator = b','
                    handle.write(b'}')
                # Validate the staged bytes, including large sources, without
                # materializing another decoded receipt before the transaction.
                if path.stat().st_size > _CHUNK_BYTES:
                    try:
                        with path.open('rb') as reader:
                            for _ in _json_events(reader):
                                pass
                    except ProjectionDeferred as exc:
                        exc.recording_id = folder.name
                        raise
                else:
                    json.loads(path.read_text(encoding='utf-8'))
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
            self._check_cached(db, layout.name)
            for row in db.execute(
                'SELECT id,payload FROM receipts WHERE day=? ORDER BY id', (layout.name,)):
                if row[1]:
                    yield row[0], json.loads(row[1])
                else:
                    chunks = db.execute('SELECT payload FROM receipt_chunks WHERE day=? AND id=? ORDER BY ordinal',
                                        (layout.name, row[0]))
                    try:
                        value = _read_chunks(chunks)
                    except ProjectionDeferred as exc:
                        exc.recording_id = row[0]
                        raise
                    yield row[0], value
                    del value
