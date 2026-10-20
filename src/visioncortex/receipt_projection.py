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
_MAX_STREAM_BYTES = configured_limit('VISIONCORTEX_RECEIPT_PROJECTION_STREAM_MAX_BYTES',
                                    1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
_MAX_STREAM_EVENTS = configured_limit('VISIONCORTEX_RECEIPT_PROJECTION_STREAM_MAX_JSON_EVENTS',
                                     64_000_000, 268_435_456)
_MAX_SPOOL_BYTES = 2 * 1024 * 1024 * 1024


def projected_json(path, *, omit=(), max_resident_bytes=128 * 1024**2, opener=None):
    """Project while parsing, before dense omitted values become Python trees.

    Paths are tuples of dictionary keys; None denotes an array item. Unknown
    fields are retained, including literal dots and keys named ``item``.
    """
    from ijson.common import ObjectBuilder
    path = Path(path)
    before = _json_file_generation(path)
    if before[2] > _MAX_STREAM_BYTES:
        raise ProjectionDeferred('stream_bytes', before[2], _MAX_STREAM_BYTES)
    builder, stack, memory = ObjectBuilder(), [], 0
    omitted = set(omit)
    with (opener or path.open)('rb') as handle:
        for count, (event, value) in enumerate(_json_events(handle, streaming=True)):
            if count % 4096 == 0:
                from .runtime_control import check_yield
                check_yield(checkpoint=False)
            parent = stack[-1] if stack else None
            child = (() if parent is None else parent['path'] +
                     ((parent['key'],) if parent['map'] else (None,)))
            if event == 'map_key':
                parent['key'] = value
                keep = not parent['omit'] and parent['path'] + (value,) not in omitted
            elif event in {'start_map', 'start_array'}:
                skip = bool(parent and parent['omit']) or child in omitted
                stack.append({'path': child, 'map': event == 'start_map', 'key': None, 'omit': skip})
                keep = not skip
            elif event in {'end_map', 'end_array'}:
                keep = not stack.pop()['omit']
            else:
                keep = not (parent and parent['omit']) and child not in omitted
            if keep:
                # Account for values, keys, container allocation and entries;
                # no whole-result serialization or second copy is required.
                memory += 2 * (sys.getsizeof(value) + (64 if event in {'map_key', 'start_map', 'start_array'} else 16))
                if memory > max_resident_bytes:
                    raise ProjectionDeferred('projected_resident_bytes', memory, max_resident_bytes)
                builder.event(event, value)
    if _json_file_generation(path) != before:
        raise ValueError('JSON source changed during projection')
    if not isinstance(builder.value, dict):
        raise ValueError('JSON projection requires an object')
    return builder.value


def _json_file_generation(path):
    info = path.stat()
    if path.is_symlink():
        raise ValueError('JSON source must not be a symlink')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def replace_json_field(path, field, value, *, opener=None):
    """Atomically replace one top-level field, preserving all other JSON values.

    Stream both input and output; authoritative diagnostics are never hydrated
    or replaced with their reduced readable projection. The caller owns the
    archive lock and verified storage guard, as for ordinary index writes.
    """
    from .device_day_contract import _atomic_blocks
    path = Path(path)
    before = _json_file_generation(path)
    if before[2] > _MAX_STREAM_BYTES:
        raise ProjectionDeferred('stream_bytes', before[2], _MAX_STREAM_BYTES)
    replacement = json.dumps(field, ensure_ascii=False) + ':' + json.dumps(value, ensure_ascii=False, allow_nan=False)

    def fragments():
        frames, depth, skipping = [], 0, False
        with (opener or path.open)('rb') as handle:
            for count, (event, item) in enumerate(_json_events(handle, streaming=True)):
                if count % 4096 == 0:
                    from .runtime_control import check_yield
                    check_yield(checkpoint=False)
                if event == 'map_key' and depth == 1 and item == field:
                    skipping = True
                    continue
                old_depth = depth
                if event in {'start_map', 'start_array'}:
                    depth += 1
                elif event in {'end_map', 'end_array'}:
                    depth -= 1
                if skipping:
                    if depth == 1 and event != 'map_key':
                        skipping = False
                    continue
                if event == 'map_key':
                    frame = frames[-1]
                    yield (',' if frame[1] else '') + json.dumps(item, ensure_ascii=False) + ':'
                    frame[1] += 1
                elif event in {'end_map', 'end_array'}:
                    frame = frames.pop()
                    if not frames:
                        if event != 'end_map':
                            raise ValueError('JSON field replacement requires an object')
                        yield (',' if frame[1] else '') + replacement
                    yield '}' if event == 'end_map' else ']'
                else:
                    if frames and frames[-1][0] == 'array':
                        yield ',' if frames[-1][1] else ''
                        frames[-1][1] += 1
                    if event in {'start_map', 'start_array'}:
                        if not frames and event != 'start_map':
                            raise ValueError('JSON field replacement requires an object')
                        frames.append(['map' if event == 'start_map' else 'array', 0])
                        yield '{' if event == 'start_map' else '['
                    else:
                        if old_depth == 0:
                            raise ValueError('JSON field replacement requires an object')
                        yield json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if _json_file_generation(path) != before:
            raise ValueError('JSON source changed during field replacement')

    def blocks():
        buffer = bytearray()
        for fragment in fragments():
            buffer.extend(fragment.encode('utf-8'))
            if len(buffer) >= _CHUNK_BYTES:
                yield bytes(buffer)
                buffer.clear()
        if buffer:
            yield bytes(buffer)
        yield b'\n'
    _atomic_blocks(path, blocks())


def artifact_manifest(path, *, max_references=20000, opener=None):
    """Read bounded references without loading frame/model evidence.

    This checks shape and stable generation, not execution acceptance; the
    ordinary receipt owner still owns publication acceptance.
    """
    import ijson
    from ijson.common import ObjectBuilder
    path = Path(path)
    before = path.stat()
    if path.is_symlink() or before.st_size > _MAX_STREAM_BYTES:
        raise ValueError('Unbounded or linked artifact receipt')
    fields, references, builder, collecting, depth, events = {}, [], None, None, 0, 0
    reference_events, reference_bytes = 0, 0

    def parse_events(handle):
        try:
            yield from ijson.parse(handle, buf_size=1024 * 1024)
        except ijson.JSONError as exc:
            raise ValueError('Invalid artifact receipt JSON') from exc

    with (opener or path.open)('rb') as handle:
        for prefix, event, value in parse_events(handle):
            events += 1
            if events % 4096 == 0:
                from .runtime_control import check_yield
                check_yield(checkpoint=False)
            if events > _MAX_STREAM_EVENTS:
                raise ValueError('Artifact receipt event limit exceeded')
            if event in {'start_map', 'start_array'}:
                depth += 1
            elif event in {'end_map', 'end_array'}:
                depth -= 1
            if depth > _MAX_JSON_DEPTH:
                raise ValueError('Artifact receipt depth limit exceeded')
            if prefix in {'status', 'stage', 'recording_id', 'key'} and event == 'string':
                fields[prefix] = value
            if builder is None and prefix in {'artifacts.item', 'audit_artifacts.item'}:
                if event != 'start_map' or len(references) >= max_references:
                    raise ValueError('Invalid or excessive artifact references')
                builder, collecting = ObjectBuilder(), prefix
                reference_events = 0
            if builder is not None:
                reference_events += 1
                if reference_events > 256:
                    raise ValueError('Artifact reference complexity limit exceeded')
                if isinstance(value, str) and len(value) > 8192:
                    raise ValueError('Artifact reference value limit exceeded')
                builder.event(event, value)
                if prefix == collecting and event == 'end_map':
                    reference = builder.value
                    if (not {'path', 'size_bytes', 'sha256'} <= reference.keys()
                            or not isinstance(reference['path'], str)
                            or type(reference['size_bytes']) is not int or reference['size_bytes'] < 0
                            or not isinstance(reference['sha256'], str) or len(reference['sha256']) != 64
                            or any(c not in '0123456789abcdef' for c in reference['sha256'])):
                        raise ValueError('Invalid artifact content identity')
                    references.append((collecting.split('.')[0], reference))
                    reference_bytes += len(json.dumps(reference, default=str).encode())
                    if reference_bytes > 32 * 1024**2:
                        raise ValueError('Artifact reference memory limit exceeded')
                    builder, collecting = None, None
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Artifact receipt changed during verification')
    return fields, references


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
        # Match archive hashing/copy quanta. Tiny parser reads otherwise
        # repeatedly rejoin behind MiB bandwidth reservations under load.
        for event, value in ijson.basic_parse(handle, buf_size=1024 * 1024):
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
