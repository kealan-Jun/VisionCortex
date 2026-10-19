"""Input readiness is independent of task execution and historical success."""

import json
import sqlite3
import time
from contextlib import ExitStack
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
    """Rebind filesystem identities only after proving every original byte.

    Explicit recovery operation, never admission or automatic retry. Preserve
    content bindings, old seals and model receipts across a verified remount.
    Different content, timestamps, source versions or live owners fail closed.
    """
    from .device_day import exclusive
    from .device_day_contract import STAGES, atomic_bytes, digest, read_json, safe_child
    from .device_day_inputs import binding, durable_json, identity, sync_directory, validate_snapshot, verify_content
    from .device_day_write_guard import storage_guard
    from .nas_recordings import _inspect

    fields = ('recording_id', 'source_signature', 'camera_key', 'configured_role', 'video_path',
              'frames_path', 'size_bytes', 'recording_session_id', 'recording_start_us', 'recording_end_us')
    rid = record['recording_id']
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
                        or not verify_content(physical, source)):
                    raise ValueError('Original content differs from its input seal')
                source['identity'], source['resolved_path'] = current, str(physical.resolve())
            validate_snapshot(updated, hashes=True)
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
        return {'status': 'verified', 'recording_id': rid, 'changed_seals': changed,
                'verified_seals': len(updates), 'model_invoked': False, 'capture_modified': False}


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
