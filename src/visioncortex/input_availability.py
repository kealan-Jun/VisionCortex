"""Input readiness is independent of task execution and historical success."""

import json
import sqlite3
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from .sqlite_store import connection


def configured_record(config, record):
    """Restore the explicit camera binding after metadata-only reinspection.

    No camera-prefix inference: unregistered cameras still need a binding.
    Do not mutate an inventory or an actively leased worker's snapshot.
    """
    from .capture_layout import bind_camera
    record = bind_camera(config.get('collection_ingest', {}), record)
    if record.get('camera_binding_status') == 'needs_directory_binding':
        return record
    role = config.get("collection_ingest", {}).get("camera_role_map", {}).get(
        record.get("camera_key")
    )
    return record | {"configured_role": role} if role else record


def revalidate_input_seals(runner, record):
    with revalidated_input_seals(runner, record) as result:
        return result


@contextmanager
def revalidated_input_seals(runner, record, *, verify_bytes=None):
    """Rebind filesystem identities only after proving every original byte.

    Explicit recovery operation, never admission or automatic retry. Preserve
    content bindings, old seals and model receipts across a verified remount.
    Different content, timestamps, source versions or live owners fail closed.
    """
    from .device_day import exclusive
    from .device_day_contract import STAGES, atomic_bytes, digest, read_json, safe_child
    from .device_day_inputs import binding, durable_json, identity, sync_directory, validate_snapshot
    from .device_day_write_guard import storage_guard
    from .nas_recordings import _inspect

    verify_bytes = verify_bytes or _IntegrityBytes(runner.config, integrity_options(runner.config)['max_hash_bytes'])

    fields = ('recording_id', 'source_signature', 'camera_key', 'configured_role', 'video_path',
              'frames_path', 'size_bytes', 'recording_session_id', 'recording_start_us', 'recording_end_us')
    rid = record['recording_id']
    if not isinstance(rid, str) or not rid or rid in {'.', '..'} or '/' in rid or '\\' in rid:
        raise ValueError('Invalid recording identity')
    root = Path(runner.config['collection_ingest']['source_root'])
    video = Path(record['video_path'])

    def inspect():
        fresh = configured_record(runner.config, _inspect(
            root, video, time.time(), runner.config['collection_ingest'].get('settle_seconds', 120),
            bool(runner.config['collection_ingest'].get('discover_plain_video_csv', False))))
        if (not fresh.get('processable', fresh.get('available'))
                or any(fresh.get(field) != record.get(field) for field in fields)
                or fresh.get('audio', {}) != record.get('audio', {})):
            raise ValueError('Current input does not match the sealed recording')

    with storage_guard(runner.config), ExitStack() as locks:
        for stage in ('all', 'retention', 'vision', 'stt', 'input-vision', 'input-stt', 'publication'):
            locks.enter_context(exclusive(runner.runtime_root / 'locks' / f'{rid}.{stage}.lock'))
        for stage in STAGES:
            queue = runner.runtime_root / f'queue-{stage}.sqlite3'
            if queue.is_file():
                with connection(queue, readonly=True) as db:
                    row = db.execute('SELECT status FROM recordings WHERE recording_id=?', (rid,)).fetchone()
                    if row and row['status'] == 'running':
                        raise ValueError('Input revalidation requires reconciled stage owners')
        if video.is_symlink() or not video.resolve().is_relative_to(root.resolve()):
            raise ValueError('Input revalidation source escaped configured capture root')
        inspect()
        layout, updates = runner.layout(record), []
        for stage in ('vision', 'stt'):
            path = runner._receipt(layout, record, 'input-' + stage)
            if not path.is_file():
                continue
            if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError('Input seal is not a bounded regular receipt')
            original = path.read_bytes()
            saved = json.loads(original)
            expected = (record.get('audio', {}).get('source_signature') if stage == 'stt'
                        else record['source_signature'])
            if (saved.get('stage') != 'input-' + stage or saved.get('status') != 'ready'
                    or saved.get('source_signature') != expected
                    or any(saved.get('recording', {}).get(field) != record.get(field) for field in fields)
                    or not 1 <= len(saved.get('sources', [])) <= 32):
                raise ValueError('Input seal does not match the current recording')
            updated = json.loads(original)
            for source in updated['sources']:
                physical = Path(source['original_path'])
                prefix = video.name[:-len('rgb.mp4')] if video.name.endswith('rgb.mp4') else ''
                allowed = ({('video', str(video)), ('clock', record['frames_path'])}
                           | {('sidecar', str(video.with_name(prefix + suffix)))
                              for suffix in ('meta.json', 'recording_ready.json', 'calibration.json')}
                           | {('audio_' + item['kind'], item['path'])
                              for item in record.get('audio', {}).get('files', [])})
                if (physical.is_symlink() or physical.parent != video.parent
                        or (source['kind'], str(physical)) not in allowed
                        or not physical.resolve().is_relative_to(video.parent.resolve())
                        or source['retained']['path'] != layout.relative(layout.source_path(record, source['kind'], physical))
                        or source['sha256'] != source['retained']['sha256']
                        or source['size_bytes'] != source['retained']['size_bytes']):
                    raise ValueError('Input seal source identity is inconsistent')
                current = identity(physical)
                if (current[2:4] != source['identity'][2:4] or current[2] != source['size_bytes']
                        or not verify_bytes(physical, source)):
                    raise ValueError('Original content differs from its input seal')
                source['identity'], source['resolved_path'] = current, str(physical.resolve())
            validate_snapshot(updated)
            if binding(updated) != binding(saved):
                raise ValueError('Input revalidation changed the content binding')
            updates.append((path, original, saved, updated))
        # Verify the complete set before publishing either independent seal.
        inspect()
        for path, original, _saved, updated in updates:
            if path.read_bytes() != original:
                raise ValueError('Input seal changed during revalidation')
            validate_snapshot(updated)
        changed = []
        for path, original, saved, updated in updates:
            if saved == updated:
                continue
            history = path.parent / 'history' / f'{path.stem}-before-revalidation-{digest(saved)}.json'
            if history.exists():
                if read_json(history) != saved:
                    raise ValueError('Input seal recovery history collision')
            else:
                atomic_bytes(history, original)
                sync_directory(history.parent)
            proof = {'schema_version': 'visioncortex-input-revalidation/1', 'recording_id': rid,
                     'stage': path.stem, 'verified_at': time.time(), 'previous_seal_sha256': digest(saved),
                     'revalidated_seal_sha256': digest(updated), 'content_binding': binding(saved),
                     'complete_stable_bytes_verified': True, 'model_invoked': False, 'capture_modified': False}
            durable_json(safe_child(runner.runtime_root, f'InputRevalidation/{rid}/{path.stem}-{digest(saved)}.json'), proof)
            durable_json(path, updated)
            changed.append(path.stem)
        yield {'status': 'verified', 'recording_id': rid, 'changed_seals': changed,
               'verified_seals': len(updates), 'model_invoked': False, 'capture_modified': False,
               'source_generation': digest([updated for _, _, _, updated in updates])}


class Availability:
    def __init__(self, root):
        self.path = Path(root) / "InputAvailability.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # WAL mode changes can return SQLITE_BUSY immediately while another
        # connection initializes the same new database. Retry only initialization,
        # with a deadline; never hide a different database error.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                with connection(self.path) as db:
                    db.execute('PRAGMA journal_mode=WAL')
                    db.execute("""CREATE TABLE IF NOT EXISTS inputs(id TEXT PRIMARY KEY, signature TEXT,
                        state TEXT, observed REAL, reason TEXT)""")
                break
            except sqlite3.OperationalError as exc:
                if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_BUSY or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def mark(self, record, state, *, observed=None, reason=""):
        self.mark_many([(record, state, observed, reason)])

    def mark_many(self, entries):
        values = []
        for record, state, observed, reason in entries:
            if state not in {"ready", "missing", "waiting", "unavailable"}:
                raise ValueError("Invalid input state")
            values.append(
                (
                    record["recording_id"],
                    record.get("source_signature"),
                    state,
                    time.time() if observed is None else observed,
                    reason,
                )
            )
        if not values:
            return
        with connection(self.path) as db:
            db.executemany(
                """INSERT INTO inputs VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                signature=excluded.signature,state=excluded.state,observed=excluded.observed,reason=excluded.reason
                WHERE excluded.observed>inputs.observed""",
                values,
            )

    def states(self):
        with connection(self.path, readonly=True) as db:
            return {r["id"]: dict(r) for r in db.execute("SELECT * FROM inputs")}

    def migrate_legacy(self, queue):
        # One-time import of old, explicit unavailable observations. No stat of
        # the NAS and no rewriting completed model receipts.
        with queue.connect() as db:
            rows = list(
                db.execute(
                    "SELECT payload,updated_at,result FROM recordings WHERE status!='completed'"
                )
            )
        entries = []
        for row in rows:
            record = json.loads(row["payload"])
            result = json.loads(row["result"] or "{}")
            if (
                result.get("error_type") == "FileNotFoundError"
                or record.get("available") is False
                and record.get("processable") is False
            ):
                entries.append(
                    (
                        record,
                        "missing"
                        if result.get("error_type") == "FileNotFoundError"
                        else "waiting",
                        row["updated_at"],
                        "legacy_unavailable_observation",
                    )
                )
        self.mark_many(entries)


class Reconciler:
    """Bounded metadata checks on a separate lane, never in a progress request."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config["storage"]["local_runtime_root"]) / "device-day"
        self.cursor = ""
        self.migrated = False
        self.last_priority_check = {}

    def tick(self):
        from .device_day_queue import DeviceDayQueue

        path = self.root / "queue-retention.sqlite3"
        if not path.is_file():
            return
        queue = DeviceDayQueue(path)
        availability = Availability(self.root)
        if not self.migrated:
            availability.migrate_legacy(queue)
            self.migrated = True
        states = availability.states()
        from .device_day_schedule import processing_cutoff, in_processing_scope
        cutoff = processing_cutoff(self.config.get('device_day', {}))
        scope_sql = (" AND (json_extract(payload,'$.recording_start_us')>=? "
                     "OR json_extract(payload,'$.recording_end_us')>=?)") if cutoff else ""
        scope_args = (cutoff, cutoff) if cutoff else ()
        with queue.connect() as db:
            rows = list(
                db.execute(
                    "SELECT * FROM recordings WHERE status NOT IN ('completed','running') "
                    "AND recording_id>?" + scope_sql + " ORDER BY recording_id LIMIT 24",
                    (self.cursor, *scope_args),
                )
            )
        if not rows:
            self.cursor = ""
        # Recheck transient I/O failures promptly without starving the bounded
        # historical sweep or repeatedly hitting the same unavailable input.
        current = time.time()
        priority = [key for key, state in states.items()
                    if state['state'] == 'unavailable'
                    and current - self.last_priority_check.get(key, 0) >= 30][:8]
        with queue.connect() as db:
            extra = list(db.execute(
                "SELECT * FROM recordings WHERE status NOT IN ('completed','running') "
                "AND recording_id IN (SELECT value FROM json_each(?))" + scope_sql,
                (json.dumps(priority), *scope_args)))
        for key in priority:
            self.last_priority_check[key] = current
        if rows:
            self.cursor = rows[-1]['recording_id']
        rows = list({row['recording_id']: row for row in rows + extra}.values())
        source = Path(self.config["collection_ingest"]["source_root"])
        try:
            # A mount outage is not proof that every source video was removed.
            if not source.is_dir():
                return
        except OSError:
            return
        for row in rows:
            record = json.loads(row["payload"])
            video = record.get("video_path")
            if not video:
                continue
            inspecting_metadata = False
            try:
                info = Path(video).stat()
                if info.st_size <= 0:
                    availability.mark(record, "waiting", reason="empty_capture")
                    continue
                # A present file is not necessarily a closed, compatible input.
                from .nas_recordings import _inspect

                inspecting_metadata = True
                fresh = _inspect(
                    source,
                    Path(video),
                    time.time(),
                    float(self.config["collection_ingest"].get("settle_seconds", 5)),
                )
                fresh = configured_record(self.config, fresh)
                if fresh.get("processable"):
                    availability.mark(fresh, "ready", reason="capture_reinspected")
                    from .observed_inventory import observe

                    observe(self.root, {"recordings": [fresh]})
                else:
                    availability.mark(record, "waiting", reason="capture_not_closed")
            except FileNotFoundError:
                # Retention recovery separately verifies archive-only originals.
                # Missing capture never authorizes changing a successful receipt.
                availability.mark(
                    record,
                    "waiting" if inspecting_metadata else "missing",
                    reason="capture_metadata_missing" if inspecting_metadata else
                    "capture_missing_pending_archive_verification",
                )
            except (OSError, ValueError) as exc:
                availability.mark(
                    record, "unavailable", reason=f"input_inspection_unavailable:{type(exc).__name__}:errno={getattr(exc, 'errno', None)}"
                )
        # A verified retained original remains usable after capture replacement.
        states = availability.states()
        with queue.connect() as db:
            completed = list(
                db.execute("SELECT payload FROM recordings WHERE status='completed'")
            )
        for row in completed:
            record = json.loads(row["payload"])
            if not in_processing_scope(self.config.get('device_day', {}), record):
                continue
            if states.get(record["recording_id"], {}).get("state", "ready") != "ready":
                availability.mark(
                    record, "ready", reason="verified_retention_completed"
                )


class IntegrityIssue(ValueError):
    """A fixed operational reason, safe for persistent alerts and logs."""


def integrity_options(config):
    value = (config.get('device_day') or {}).get('input_integrity') or {}
    allowed = {'enabled', 'interval_seconds', 'verification_interval_seconds', 'retry_seconds', 'batch_size', 'max_hash_bytes'}
    if not isinstance(value, dict) or value.keys() - allowed:
        raise ValueError('Unknown input integrity settings')
    result = {'enabled': value.get('enabled', False)}
    if type(result['enabled']) is not bool:
        raise ValueError('Input integrity enabled must be boolean')
    for key, default, lower, upper in (
        ('interval_seconds', 5, 1, 300), ('verification_interval_seconds', 86400, 60, 604800),
        ('retry_seconds', 900, 30, 86400), ('batch_size', 16, 1, 128),
        ('max_hash_bytes', 8 * 1024**3, 1024**2, 64 * 1024**3),
    ):
        item = value.get(key, default)
        if type(item) is not int or not lower <= item <= upper:
            raise ValueError('Invalid input integrity bound: ' + key)
        result[key] = item
    return result


class _PacedInput:
    def __init__(self, config, path):
        self.config, self.path = config, Path(path)

    @contextmanager
    def open(self, mode):
        from .device_day_io import pace_copy, slot
        config = self.config
        with self.path.open(mode) as handle:
            class Reader:
                def read(self, size):
                    if size == 0:
                        return b''
                    with slot(config, copy=True):
                        block = handle.read(size)
                    if block:
                        pace_copy(config, len(block))
                    return block
            yield Reader()


class _IntegrityBytes:
    """Fresh full hashes per check, independent of a long-lived model cache."""
    def __init__(self, config, maximum):
        self.config, self.maximum, self.bytes = config, maximum, 0
        self.verified = {}

    def __call__(self, path, reference):
        from .device_day_contract import file_hash
        from .device_day_inputs import identity, remember_content
        path = Path(path)
        before = identity(path)
        if before[2] != reference['size_bytes']:
            return False
        key = (str(path.resolve()), tuple(before))
        if key not in self.verified:
            if self.bytes + before[2] > self.maximum:
                raise IntegrityIssue('verification_budget_exceeded')
            checksum = file_hash(_PacedInput(self.config, path))
            if identity(path) != before:
                raise IntegrityIssue('input_changed_during_verification')
            self.bytes += before[2]
            self.verified[key] = checksum
            remember_content(path, checksum)  # This process just read every byte.
        return before[2] == reference['size_bytes'] and self.verified[key] == reference['sha256']


class IntegrityMonitor:
    """Bounded source recovery and archive audits, with durable quiet alerts.

    One candidate per tick; primary workers alone execute a granted retry.
    Raw recorder data and completed model results are never overwritten.
    """
    LANES = ('retention_failed', 'vision_failed', 'stt_failed', 'retention_completed')

    def __init__(self, config, runner):
        from .local_storage import require_local_path
        self.config, self.runner, self.options = config, runner, integrity_options(config)
        self.root = require_local_path(Path(config['storage']['local_runtime_root'])) / 'device-day'
        self.path = require_local_path(self.root / 'InputIntegrity/State.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript('PRAGMA journal_mode=WAL; PRAGMA max_page_count=32768; '
                'CREATE TABLE IF NOT EXISTS checks(lane TEXT,id TEXT,fingerprint TEXT,next_check REAL,'
                'checked_at REAL,result TEXT,PRIMARY KEY(lane,id)); '
                'CREATE TABLE IF NOT EXISTS cursors(lane TEXT PRIMARY KEY,position INTEGER); '
                'CREATE TABLE IF NOT EXISTS alerts(key TEXT PRIMARY KEY,stage TEXT,id TEXT,reason TEXT,'
                'state TEXT,first_seen REAL,last_seen REAL,occurrences INTEGER); '
                'CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,at REAL,key TEXT,transition TEXT,reason TEXT);')

    @contextmanager
    def _db(self, *, readonly=False):
        with connection(self.path, readonly=readonly) as db:
            if not readonly:
                if db.execute('PRAGMA max_page_count=32768').fetchone()[0] > 32768:
                    raise OSError('Input integrity state exceeds its storage bound')
                db.execute('PRAGMA journal_size_limit=8388608')
            yield db

    def _alert(self, lane, rid, reason):
        import logging
        key, current = lane + ':' + rid, time.time()
        transition = None
        with self._db() as db:
            old = db.execute('SELECT * FROM alerts WHERE key=?', (key,)).fetchone()
            if reason:
                if not old or old['state'] != 'active' or old['reason'] != reason:
                    transition = 'raised' if not old or old['state'] != 'active' else 'changed'
                first = old['first_seen'] if old and old['state'] == 'active' else current
                count = old['occurrences'] + 1 if old else 1
                db.execute('INSERT OR REPLACE INTO alerts VALUES(?,?,?,?,?,?,?,?)',
                           (key, lane.split('_')[0], rid, reason, 'active', first, current, count))
            elif old and old['state'] == 'active':
                transition, reason = 'resolved', old['reason']
                db.execute("UPDATE alerts SET state='resolved',last_seen=? WHERE key=?", (current, key))
            if transition:
                db.execute('INSERT INTO events(at,key,transition,reason) VALUES(?,?,?,?)',
                           (current, key, transition, reason))
            db.execute('DELETE FROM events WHERE seq NOT IN (SELECT seq FROM events ORDER BY seq DESC LIMIT 20000)')
        if transition:
            level = logging.INFO if transition == 'resolved' else logging.WARNING
            logging.getLogger(__name__).log(level, 'Input integrity alert %s: %s %s', transition, key, reason)

    def _select(self):
        from .device_day_contract import digest
        from .device_day_schedule import in_processing_scope
        with self._db(readonly=True) as db:
            cursors = dict(db.execute('SELECT lane,position FROM cursors'))
        start = cursors.get('_lane', 0) % len(self.LANES)
        for offset in range(len(self.LANES)):
            index = (start + offset) % len(self.LANES)
            lane = self.LANES[index]
            stage, kind = lane.split('_')
            queue = self.root / f'queue-{stage}.sqlite3'
            if not queue.is_file():
                continue
            condition = ("status='completed'" if kind == 'completed' else
                         "status='failed' AND input_status='ready' AND blocked_reason IS NULL "
                         "AND json_extract(result,'$.error_type')='InputChanged' "
                         "AND attempts>=MAX(?,retry_limit)")
            parameters = () if kind == 'completed' else ((self.config.get('device_day') or {}).get('failure_retry_limit') or 3,)
            with connection(queue, readonly=True) as db:
                rows = list(db.execute('SELECT rowid AS scan_id,* FROM recordings WHERE rowid>? AND '
                    + condition + ' ORDER BY rowid LIMIT ?',
                    (cursors.get(lane, 0), *parameters, self.options['batch_size'])))
            position = 0
            for candidate in rows:
                row = dict(candidate)
                position = row.pop('scan_id')
                if not in_processing_scope(self.config.get('device_day') or {}, json.loads(row['payload'])):
                    continue
                fingerprint = digest(row)
                with self._db(readonly=True) as db:
                    previous = db.execute('SELECT fingerprint,next_check FROM checks WHERE lane=? AND id=?',
                                          (lane, row['recording_id'])).fetchone()
                if previous and previous['fingerprint'] == fingerprint and previous['next_check'] > time.time():
                    continue
                with self._db() as db:
                    db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)', (lane, position))
                    db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)', ('_lane', (index + 1) % len(self.LANES)))
                return lane, stage, row, fingerprint
            with self._db() as db:
                db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)', (lane, position))
        return None

    def _observe_recovery_outcomes(self):
        """A granted retry is not a successful recovery; observe its owner."""
        for stage in ('retention', 'vision', 'stt'):
            queue = self.root / f'queue-{stage}.sqlite3'
            if not queue.is_file():
                continue
            lane = stage + '_recovery'
            with self._db(readonly=True) as db:
                cursor = db.execute('SELECT position FROM cursors WHERE lane=?', (lane,)).fetchone()
            with connection(queue, readonly=True) as db:
                if not db.execute("SELECT 1 FROM sqlite_master WHERE name='input_recovery_grants'").fetchone():
                    continue
                rows = list(db.execute('SELECT r.rowid AS scan_id,r.recording_id,r.status,r.attempts,r.retry_limit '
                    'FROM recordings r WHERE r.rowid>? AND EXISTS(SELECT 1 FROM input_recovery_grants g '
                    'WHERE g.recording_id=r.recording_id AND g.revision=r.revision) '
                    'ORDER BY r.rowid LIMIT ?', (cursor[0] if cursor else 0, self.options['batch_size'])))
            for row in rows:
                if row['status'] == 'failed' and row['retry_limit'] > 0 and row['attempts'] >= row['retry_limit']:
                    self._alert(lane, row['recording_id'], 'recovery_attempt_failed')
                elif row['status'] == 'completed':
                    self._alert(lane, row['recording_id'], None)
            with self._db() as db:
                db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?)', (lane, rows[-1]['scan_id'] if rows else 0))

    def _archive(self, record, verify):
        from .device_day_contract import VERSION, digest, read_json, safe_child
        from .device_day_inputs import copy_sealed
        layout = self.runner.layout(record)
        path = self.runner._receipt(layout, record, 'retention')
        if not path.is_file():
            raise IntegrityIssue('archive_receipt_missing')
        if path.is_symlink() or path.stat().st_size > 4 * 1024**2:
            raise IntegrityIssue('archive_receipt_invalid')
        saved = read_json(path)
        fields = ('recording_id', 'source_signature', 'camera_key', 'configured_role',
                  'recording_start_us', 'recording_end_us')
        if (saved.get('schema_version') != VERSION or saved.get('status') != 'completed'
                or any(saved.get('recording', {}).get(k) != record.get(k) for k in fields)):
            raise IntegrityIssue('archive_receipt_unverified')
        sources, artifacts = saved.get('sources', []), saved.get('artifacts', [])
        if (not 2 <= len(sources) <= 64 or not artifacts
                or sorted(s.get('kind') for s in sources if s.get('kind') in {'video', 'clock'}) != ['clock', 'video']
                or sorted(digest(s['retained']) for s in sources) != sorted(digest(a) for a in artifacts)):
            raise IntegrityIssue('archive_receipt_invalid')
        missing = []
        for source in sources:
            ref = source['retained']
            original = Path(source['original_path'])
            if (source['sha256'] != ref['sha256'] or source['size_bytes'] != ref['size_bytes']
                    or original.parent != Path(record['video_path']).parent
                    or source['kind'] == 'video' and str(original) != record['video_path']
                    or source['kind'] == 'clock' and str(original) != record['frames_path']
                    or ref['path'] != layout.relative(layout.source_path(record, source['kind'], original))):
                raise IntegrityIssue('archive_source_binding_changed')
            target = safe_child(layout.root, ref['path'])
            if not target.exists():
                missing.append(source)
            elif not verify(target, ref):
                raise IntegrityIssue('archive_content_conflict')
        if missing:
            if layout.recorder_owned_raw:
                raise IntegrityIssue('recorder_input_missing')
            with revalidated_input_seals(self.runner, record, verify_bytes=verify) as proof:
                if not proof['verified_seals']:
                    raise IntegrityIssue('archive_missing_source_unverified')
                seals = [read_json(self.runner._receipt(layout, record, 'input-' + stage))
                         for stage in ('vision', 'stt')
                         if self.runner._receipt(layout, record, 'input-' + stage).is_file()]
                by_path = {s['original_path']: s for seal in seals for s in seal['sources']}
                for source in missing:
                    sealed = by_path.get(source['original_path'])
                    if not sealed or sealed['retained'] != source['retained']:
                        raise IntegrityIssue('archive_missing_source_unverified')
                if read_json(path) != saved:
                    raise BlockingIOError('Retention receipt changed during verification')
                for source in missing:
                    target = safe_child(layout.root, source['retained']['path'])
                    if target == Path(source['original_path']):
                        raise IntegrityIssue('recorder_input_missing')
                    copy_sealed(self.config, by_path[source['original_path']], target)
                    if not verify(target, source['retained']):
                        raise IntegrityIssue('archive_restore_verification_failed')
            from .publication_journal import PublicationJournal
            PublicationJournal(self.root).begin(record)
        return len(artifacts), len(missing)

    def _derived(self, record, verify):
        from .device_day_contract import safe_child
        from .receipt_projection import artifact_manifest
        layout, count = self.runner.layout(record), 0
        for stage in ('vision', 'stt', 'understanding', 'report'):
            with self.runner.queues[stage].connect() as db:
                row = db.execute('SELECT status,payload FROM recordings WHERE recording_id=?',
                                 (record['recording_id'],)).fetchone()
            if not row or row['status'] != 'completed' or json.loads(row['payload']).get('source_signature') != record['source_signature']:
                continue
            path = self.runner._receipt(layout, record, stage)
            if not path.is_file():
                raise IntegrityIssue('completed_receipt_missing')
            fields, refs = artifact_manifest(path, opener=_PacedInput(self.config, path).open)
            if (fields.get('status') != 'completed' or fields.get('recording_id') != record['recording_id']
                    or fields.get('stage', stage) != stage):
                raise IntegrityIssue('completed_receipt_unverified')
            for kind, ref in refs:
                base = layout.root
                if kind == 'audit_artifacts':
                    if ref.get('storage_root') != 'local_cache_root':
                        raise IntegrityIssue('artifact_storage_unverified')
                    base = self.runner.backend_root
                target = safe_child(base, ref['path'])
                if not target.is_file():
                    raise IntegrityIssue('derived_artifact_missing')
                if not verify(target, ref):
                    raise IntegrityIssue('derived_content_conflict')
                count += 1
        publication = self.runner._receipt(layout, record, 'publication')
        if publication.is_file() and publication.stat().st_size <= 4 * 1024**2:
            from .device_day_contract import read_json
            if read_json(publication).get('status') == 'completed' and not layout.index.is_file():
                from .publication_journal import PublicationJournal
                PublicationJournal(self.root).begin(record)
                raise IntegrityIssue('published_index_missing')
        return count

    def _check(self, stage, row):
        from .device_day_contract import digest, read_json
        record = json.loads(row['payload'])
        verify = _IntegrityBytes(self.config, self.options['max_hash_bytes'])
        restored, artifacts, grant = 0, 0, None
        if row['status'] == 'failed':
            from .device_day_inplace import marker
            layout = self.runner.layout(record)
            if not marker(self.runner, record).is_file():
                raise IntegrityIssue('input_seal_unavailable')
            # Independently verify the existing inplace-queue/1 contract. A
            # source repair cannot authorize a different execution recipe.
            inputs = ([record['source_signature'], record.get('audio', {}).get('source_signature')]
                      if stage == 'retention' else record.get('audio', {}).get('source_signature')
                      if stage == 'stt' else record['source_signature'])
            revision = digest(['inplace-queue/1', stage, record['recording_id'], inputs,
                               self.runner._key(stage, record, {})])
            if revision != row['revision']:
                raise IntegrityIssue('execution_recipe_changed')
            retention = self.runner._receipt(layout, record, 'retention')
            if retention.is_file() and retention.stat().st_size <= 4 * 1024**2 and read_json(retention).get('status') == 'completed':
                artifacts, restored = self._archive(record, verify)
            with revalidated_input_seals(self.runner, record, verify_bytes=verify) as proof:
                if not proof['verified_seals']:
                    raise IntegrityIssue('input_seal_unavailable')
                from .device_day_contract import safe_child
                for primary in ('vision', 'stt'):
                    path = self.runner._receipt(layout, record, 'input-' + primary)
                    if not path.is_file():
                        continue
                    for source in read_json(path)['sources']:
                        target = safe_child(layout.root, source['retained']['path'])
                        if target.exists() and not verify(target, source['retained']):
                            raise IntegrityIssue('archive_content_conflict')
                repair_id = digest(['input-integrity-recovery/1', proof['source_generation']])
                proof = proof | {'current_queue_revision': revision}
                grant = self.runner.queues[stage].authorize_input_recovery(row, repair_id, proof)
                if grant == 'repair_already_authorized':
                    raise IntegrityIssue('recovery_retry_exhausted')
                if grant not in {'retry_authorized', 'retry_already_available'}:
                    raise IntegrityIssue('input_recovery_state_changed')
        else:
            artifacts, restored = self._archive(record, verify)
            artifacts += self._derived(record, verify)
        return {'status': 'retry_authorized' if grant else 'restored' if restored else 'verified',
                'recording_id': record['recording_id'], 'artifacts_verified': artifacts,
                'restored_files': restored, 'hash_bytes': verify.bytes, 'model_invoked': False, 'capture_modified': False}

    def tick(self, *, stop=None):
        from .device_day import exclusive
        from .device_day_recovery_worker import verify_storage
        from .device_day_write_guard import storage_guard
        from .runtime_control import check_cancelled
        if not self.options['enabled']:
            return {'status': 'disabled'}
        check_cancelled(stop)
        with exclusive(self.root / 'locks/input-integrity-monitor.lock'):
            # Local queue observations continue even when NAS is unavailable.
            self._observe_recovery_outcomes()
            try:
                verify_storage(self.config)
                with storage_guard(self.config):
                    pass
            except (OSError, ValueError):
                self._alert('system', 'storage', 'storage_identity_unverified')
                return {'status': 'storage_unavailable', 'reason': 'storage_identity_unverified'}
            self._alert('system', 'storage', None)
            selected = self._select()
            if not selected:
                return {'status': 'idle'}
            lane, stage, row, fingerprint = selected
            reason = None
            try:
                verify_storage(self.config)
                with storage_guard(self.config):
                    result = self._check(stage, row)
            except BlockingIOError:
                result = {'status': 'busy', 'recording_id': row['recording_id']}
            except IntegrityIssue as exc:
                reason = str(exc)
                result = {'status': 'attention_required', 'reason': reason, 'recording_id': row['recording_id']}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                if type(exc) is ValueError and str(exc) in {
                        'Input revalidation requires reconciled stage owners',
                        'Artifact receipt changed during verification'}:
                    result = {'status': 'busy', 'recording_id': row['recording_id']}
                else:
                    reason = ('input_revalidation_rejected' if type(exc) is ValueError else 'integrity_verification_unavailable')
                    result = {'status': 'attention_required', 'reason': reason,
                              'recording_id': row['recording_id'], 'error_type': type(exc).__name__}
            interval = (self.options['verification_interval_seconds'] if result['status'] == 'verified'
                        else self.options['retry_seconds'])
            if result['status'] != 'busy':
                self._alert(lane, row['recording_id'], reason)
            if lane == 'retention_completed' and result['status'] in {'verified', 'restored'}:
                signature = json.loads(row['payload'])['source_signature']
                for primary in ('retention', 'vision', 'stt'):
                    with self.runner.queues[primary].connect() as db:
                        completed = db.execute("SELECT 1 FROM recordings WHERE recording_id=? AND status='completed' "
                                               "AND json_extract(payload,'$.source_signature')=?",
                                               (row['recording_id'], signature)).fetchone()
                    if completed:
                        self._alert(primary + '_failed', row['recording_id'], None)
            result.update(queue_revision=row['revision'], queue_updated_at=row['updated_at'])
            with self._db() as db:
                db.execute('INSERT OR REPLACE INTO checks VALUES(?,?,?,?,?,?)',
                           (lane, row['recording_id'], fingerprint, time.time() + interval, time.time(), json.dumps(result)))
                if result['status'] in {'restored', 'retry_authorized'}:
                    db.execute('INSERT INTO events(at,key,transition,reason) VALUES(?,?,?,?)',
                               (time.time(), lane + ':' + row['recording_id'],
                                'authorized' if result['status'] == 'retry_authorized' else 'recovered', result['status']))
            return result


def integrity_status(config, *, now=None):
    """Read local monitor/alert evidence without opening NAS or making stores."""
    from .device_day_consumers import process_start_ticks
    current = time.time() if now is None else now
    root = Path(config['storage']['local_runtime_root']) / 'device-day/InputIntegrity'
    result = {'available': False, 'status': 'not_configured', 'active_alert_count': 0,
              'alerts': [], 'events': [], 'local_metadata_only': True, 'full_archive_sweep_proven': False}
    state = root / 'Service.json'
    try:
        if state.is_file():
            if state.stat().st_size > 64 * 1024:
                raise ValueError('Unbounded integrity service observation')
            value = json.loads(state.read_text())
            if value.get('schema_version') != 'visioncortex-input-integrity-service/1':
                raise ValueError('Unknown integrity service observation')
            stamp = value.get('updated_at', 0)
            age = current - stamp
            live = (type(age) in (int, float) and -5 <= age <= 15
                    and type(value.get('pid')) is int and type(value.get('process_start_ticks')) is int
                    and process_start_ticks(value['pid']) == value['process_start_ticks'])
            result.update(available=live, status=value.get('status') if live else 'stale', service=value)
        elif integrity_options(config)['enabled']:
            result['status'] = 'service_missing'
        database = root / 'State.sqlite3'
        if database.is_file():
            if result['status'] == 'not_configured':
                result['status'] = 'service_missing'
            with connection(database, readonly=True, timeout=.25) as db:
                result['active_alert_count'] = db.execute("SELECT COUNT(*) FROM alerts WHERE state='active'").fetchone()[0]
                result['alerts'] = [dict(row) for row in db.execute(
                    "SELECT stage,id,reason,first_seen,last_seen,occurrences FROM alerts WHERE state='active' "
                    'ORDER BY first_seen LIMIT 100')]
                result['events'] = [dict(row) for row in db.execute(
                    'SELECT at,key,transition,reason FROM events ORDER BY seq DESC LIMIT 20')]
                result['check_counts'] = dict(db.execute("SELECT json_extract(result,'$.status'),COUNT(*) FROM checks "
                                                        "GROUP BY json_extract(result,'$.status')"))
                result['last_check_at'] = db.execute('SELECT MAX(checked_at) FROM checks').fetchone()[0]
                result['overdue_checks'] = db.execute('SELECT COUNT(*) FROM checks WHERE next_check<?', (current,)).fetchone()[0]
                checked = {row['id']: dict(row) for row in db.execute(
                    "SELECT id,next_check,json_extract(result,'$.status') AS status,"
                    "json_extract(result,'$.queue_revision') AS revision,"
                    "json_extract(result,'$.queue_updated_at') AS updated FROM checks WHERE lane='retention_completed'")}
            queue = root.parent / 'queue-retention.sqlite3'
            if queue.is_file():
                with connection(queue, readonly=True, timeout=.25) as db:
                    rows = list(db.execute("SELECT recording_id,revision,updated_at FROM recordings WHERE status='completed'"))
                verified, overdue = 0, 0
                for row in rows:
                    proof = checked.get(row['recording_id'], {})
                    if (proof.get('status') in {'verified', 'restored'} and proof.get('revision') == row['revision']
                            and proof.get('updated') == row['updated_at']):
                        verified += 1
                        overdue += proof['next_check'] < current
                result['coverage'] = {'completed_queue_records': len(rows), 'verified_current': verified,
                                      'unverified_or_changed': len(rows) - verified, 'overdue_current': overdue}
        if result.get('available') and result.get('service', {}).get('phase') == 'checking':
            if current - result['service'].get('tick_started_at', current) > 7200:
                result.update(available=False, status='verification_stalled')
    except (OSError, ValueError, TypeError, sqlite3.Error):
        result.update(available=False, status='evidence_unavailable')
    return result
