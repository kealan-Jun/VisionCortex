"""Rebuildable, incremental local projection of authoritative stage receipts."""
import json
import hashlib
from decimal import Decimal
import io
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from .sqlite_store import connection


_CHUNK_BYTES = 256 * 1024


def configured_limit(name, default, maximum):
    """Explicit bounded site opt-in; default deployments keep their old cap."""
    value = os.environ.get(name)
    try:
        result = default if value is None else int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a bounded positive integer') from exc
    if not 1 <= result <= maximum:
        raise ValueError(f'{name} must be between 1 and {maximum}')
    return result


_MAX_RECEIPT_BYTES = configured_limit('VISIONCORTEX_RECEIPT_PROJECTION_MAX_BYTES', 64 * 1024 * 1024, 256 * 1024 * 1024)
_MAX_JSON_EVENTS = configured_limit('VISIONCORTEX_RECEIPT_PROJECTION_MAX_JSON_EVENTS', 2_000_000, 8_000_000)
_MAX_JSON_DEPTH = 128
_MAX_STREAM_BYTES = 1024 * 1024 * 1024
_MAX_STREAM_EVENTS = 64_000_000
_MAX_SPOOL_BYTES = 2 * 1024 * 1024 * 1024


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

    def __init__(self, rows, *, streaming=False):
        self.rows = iter(rows)
        self.remaining = memoryview(b'')
        self.total = 0
        self.streaming = streaming

    def readable(self):
        return True

    def readinto(self, target):
        if not self.remaining:
            row = next(self.rows, None)
            if row is None:
                return 0
            self.total += len(row[0])
            if self.streaming:
                if self.total > _MAX_STREAM_BYTES:
                    raise ProjectionDeferred('stream_bytes', self.total, _MAX_STREAM_BYTES)
            else:
                _check_size(self.total)
            self.remaining = memoryview(row[0])
        count = min(len(target), len(self.remaining))
        target[:count] = self.remaining[:count]
        self.remaining = self.remaining[count:]
        return count


def _json_events(handle, *, streaming=False):
    import ijson
    # Default number parsing preserves arbitrary Python integers. Convert only
    # non-integral JSON numbers back to float, matching the historical decoder.
    try:
        events, depth = 0, 0
        for event, value in ijson.basic_parse(handle):
            events += 1
            maximum = _MAX_STREAM_EVENTS if streaming else _MAX_JSON_EVENTS
            if events > maximum:
                raise ProjectionDeferred('json_events', events, maximum)
            if event in {'start_map', 'start_array'}:
                depth += 1
                if depth > _MAX_JSON_DEPTH:
                    raise ProjectionDeferred('json_depth', depth, _MAX_JSON_DEPTH)
            elif event in {'end_map', 'end_array'}:
                depth -= 1
            yield event, float(value) if isinstance(value, Decimal) else value
    except ijson.JSONError as exc:
        raise ValueError('Invalid receipt JSON') from exc


def _deduplicated_json(handle):
    """Restore shared historical audits while retaining every JSON value.

    JSON expanded the same batch audit into many segments. Intern only equal
    complete activity_audit objects, after a full canonical digest and equality
    check. Unique retained objects and a single temporary audit are bounded;
    this never substitutes summaries for source evidence or receipt keys.
    """
    from .device_day_contract import digest
    stack, audits, keys, result = [], {}, {}, None
    maximum = configured_limit('VISIONCORTEX_RECEIPT_PROJECTION_RESIDENT_BYTES',
                               _MAX_RECEIPT_BYTES, 1024 * 1024 * 1024)
    memory = 0
    audit_start = None
    for event, value in _json_events(handle, streaming=True):
        if event == 'map_key':
            known_key = keys.get(value)
            if known_key is not None:
                value = known_key
            else:
                memory += sys.getsizeof(value)
                if len(keys) < 4096 and len(value) <= 256:
                    keys[value] = value
            stack[-1]['key'] = value
        elif event in {'start_map', 'start_array'}:
            parent = stack[-1] if stack else None
            key = parent['key'] if parent and isinstance(parent['value'], dict) else None
            item = {} if event == 'start_map' else []
            shared = key in {'activity_audit', 'evidence', 'fine_candidates', 'coarse_candidates', 'detections'}
            audit = shared and audit_start is None
            entry = {'value': item, 'key': None, 'attach': key, 'start': memory, 'audit': audit, 'shared': shared}
            memory += sys.getsizeof(item)
            if audit:
                audit_start = entry['start']
            stack.append(entry)
        elif event in {'end_map', 'end_array'}:
            entry = stack.pop()
            value = entry['value']
            if entry['shared']:
                checksum = digest(value)
                known = audits.get(checksum)
                if known is not None and known == value:
                    memory -= memory - entry['start']
                    value = known
                else:
                    audits[checksum] = value
            if entry['audit']:
                audit_start = None
            if not stack:
                result = value
            else:
                parent = stack[-1]['value']
                before = sys.getsizeof(parent)
                if isinstance(parent, dict):
                    parent[entry['attach']] = value
                else:
                    parent.append(value)
                memory += sys.getsizeof(parent) - before
        else:
            memory += sys.getsizeof(value)
            if not stack:
                result = value
            else:
                parent = stack[-1]
                before = sys.getsizeof(parent['value'])
                if isinstance(parent['value'], dict):
                    parent['value'][parent['key']] = value
                else:
                    parent['value'].append(value)
                memory += sys.getsizeof(parent['value']) - before
        # An incoming duplicate temporarily needs one additional audit. The
        # bound is explicit instead of decoding an arbitrarily large object.
        measured = memory - audit_start if audit_start is not None else memory
        if measured > maximum:
            raise ProjectionDeferred('resident_bytes', measured, maximum)
    return result


def _read_chunks(rows, *, streaming=False):
    from ijson.common import ObjectBuilder
    builder = ObjectBuilder()
    with io.BufferedReader(_ChunkReader(rows, streaming=streaming), buffer_size=_CHUNK_BYTES) as handle:
        if streaming:
            return _deduplicated_json(handle)
        for event, value in _json_events(handle):
            builder.event(event, value)
    return builder.value


class ReceiptProjection:
    def __init__(self, runtime_root):
        option = os.environ.get('VISIONCORTEX_RECEIPT_PROJECTION_SHARED_AUDITS', '0')
        if option not in {'0', '1'}:
            raise ValueError('VISIONCORTEX_RECEIPT_PROJECTION_SHARED_AUDITS must be 0 or 1')
        self.shared_audits = option == '1'
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
            db.execute('CREATE TABLE IF NOT EXISTS source_versions(day TEXT,id TEXT,fingerprint TEXT NOT NULL,'
                       'PRIMARY KEY(day,id))')

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

    def _check_cached(self, db, day, *, replacing=()):
        # BLOB length uses SQLite's record metadata rather than hydrating its
        # overflow pages into Python. Check in the same read snapshot as decode.
        for row in db.execute('''SELECT id, CASE WHEN payload='' THEN
                COALESCE((SELECT SUM(length(c.payload)) FROM receipt_chunks c
                          WHERE c.day=receipts.day AND c.id=receipts.id),0)
                ELSE length(CAST(payload AS BLOB)) END AS bytes
                FROM receipts WHERE day=?''', (day,)):
            if row['id'] not in replacing:
                self._check_encoded(row['bytes'], row['id'])

    def _check_encoded(self, size, identifier):
        maximum = _MAX_STREAM_BYTES if self.shared_audits else _MAX_RECEIPT_BYTES
        if size > maximum:
            raise ProjectionDeferred('stream_bytes' if self.shared_audits else 'encoded_bytes', size, maximum, identifier)

    @staticmethod
    def _source_version(folder, stages):
        rows = []
        for stage in stages:
            source = folder / f'{stage}.json'
            try:
                st = source.stat()
            except FileNotFoundError:
                rows.append((stage, None))
            else:
                rows.append((stage, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        return json.dumps(rows, separators=(',', ':'))

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
            staged, source_versions, spool_bytes = [], {}, 0
            for number, folder in enumerate(folders):
                version = self._source_version(folder, stages)
                source_versions[folder] = version
                with connection(self.path, readonly=True) as db:
                    previous = db.execute('SELECT fingerprint FROM source_versions WHERE day=? AND id=?',
                                          (layout.name, folder.name)).fetchone()
                    present = db.execute('SELECT 1 FROM receipts WHERE day=? AND id=?',
                                         (layout.name, folder.name)).fetchone()
                if previous and present and previous[0] == version:
                    continue  # Previously validated immutable inode generation.
                path = Path(temporary) / str(number)
                with path.open('wb') as handle:
                    handle.write(b'{')
                    separator = b''
                    for stage in stages:
                        source = folder / f'{stage}.json'
                        if not source.is_file():
                            continue
                        prefix = separator + json.dumps(stage).encode('utf-8') + b':'
                        expected = handle.tell() + len(prefix) + source.stat().st_size + 1
                        self._check_encoded(expected, folder.name)
                        if spool_bytes + expected > _MAX_SPOOL_BYTES:
                            raise ProjectionDeferred('spool_bytes', spool_bytes + expected, _MAX_SPOOL_BYTES, folder.name)
                        handle.write(prefix)
                        # Bound every actual read too: a new atomic receipt may
                        # replace the path after stat. Never read it as one string.
                        with source.open('rb') as reader:
                            while chunk := reader.read(_CHUNK_BYTES):
                                self._check_encoded(handle.tell() + len(chunk) + 1, folder.name)
                                if spool_bytes + handle.tell() + len(chunk) + 1 > _MAX_SPOOL_BYTES:
                                    raise ProjectionDeferred('spool_bytes', spool_bytes + handle.tell() + len(chunk) + 1, _MAX_SPOOL_BYTES, folder.name)
                                handle.write(chunk)
                        separator = b','
                    handle.write(b'}')
                # Validate the staged bytes, including large sources, without
                # materializing another decoded receipt before the transaction.
                if path.stat().st_size > _CHUNK_BYTES:
                    try:
                        with path.open('rb') as reader:
                            if self.shared_audits:
                                _deduplicated_json(reader)
                            else:
                                for _ in _json_events(reader):
                                    pass
                    except ProjectionDeferred as exc:
                        exc.recording_id = folder.name
                        raise
                else:
                    json.loads(path.read_text(encoding='utf-8'))
                if self._source_version(folder, stages) != version:
                    raise ValueError('Receipt source changed during projection')
                staged.append((folder.name, path, version))
                spool_bytes += path.stat().st_size
            if any(self._source_version(folder, stages) != version for folder, version in source_versions.items()):
                raise ValueError('Receipt source changed during projection')
            with connection(self.path) as db:
                db.execute('BEGIN IMMEDIATE')
                if full:
                    live = {folder.name for folder in folders}
                    for row in db.execute('SELECT id FROM receipts WHERE day=?', (layout.name,)).fetchall():
                        if row[0] not in live:
                            db.execute('DELETE FROM receipts WHERE day=? AND id=?', (layout.name, row[0]))
                            db.execute('DELETE FROM source_versions WHERE day=? AND id=?', (layout.name, row[0]))
                    db.execute('INSERT OR IGNORE INTO days VALUES(?)', (layout.name,))
                for identifier, path, version in staged:
                    large = path.stat().st_size > _CHUNK_BYTES
                    db.execute('INSERT OR REPLACE INTO receipts VALUES(?,?,?)',
                               (layout.name, identifier, '' if large else path.read_text(encoding='utf-8')))
                    db.execute('INSERT OR REPLACE INTO source_versions VALUES(?,?,?)', (layout.name, identifier, version))
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
                        value = _read_chunks(chunks, streaming=True) if self.shared_audits else _read_chunks(chunks)
                    except ProjectionDeferred as exc:
                        exc.recording_id = row[0]
                        raise
                    yield row[0], value
                    del value
