"""Persist unfinished publication before modifying receipts; replay without CV."""
import json
from copy import deepcopy
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
            # Additive and separate: old workers can still INSERT INTO dirty.
            db.execute('CREATE TABLE IF NOT EXISTS retries(id TEXT PRIMARY KEY,token TEXT NOT NULL,'
                       'attempts INTEGER NOT NULL,next_attempt_at REAL NOT NULL,reason TEXT NOT NULL)')

    def begin(self, record):
        token = uuid.uuid4().hex
        with connection(self.path) as db:
            db.execute('INSERT OR REPLACE INTO dirty VALUES(?,?,?)',
                       (record['recording_id'], token, json.dumps(record)))
            db.execute('DELETE FROM retries WHERE id=?', (record['recording_id'],))
        return token

    def complete(self, identifier, token):
        with connection(self.path) as db:
            db.execute('DELETE FROM dirty WHERE id=? AND token=?', (identifier, token))
            db.execute('DELETE FROM retries WHERE id=? AND token=?', (identifier, token))

    def pending(self, limit=32, *, since_us=None, prefer_since_us=None, latest_first=False, ready_only=False):
        if type(latest_first) is not bool:
            raise ValueError('Publication latest_first must be a boolean')
        time_order = "COALESCE(json_extract(record,'$.recording_start_us'),0) DESC," if latest_first else ''
        with connection(self.path, readonly=True) as db:
            return [(row['token'], json.loads(row['record'])) for row in db.execute(
                "SELECT * FROM dirty WHERE (? IS NULL OR MAX(COALESCE(json_extract(record,'$.recording_start_us'),0),"
                "COALESCE(json_extract(record,'$.recording_end_us'),0))>=?) "
                "AND (?=0 OR NOT EXISTS(SELECT 1 FROM retries r WHERE r.id=dirty.id AND r.token=dirty.token "
                "AND r.next_attempt_at>?)) ORDER BY "
                "CASE WHEN ? IS NOT NULL AND COALESCE(json_extract(record,'$.recording_start_us'),0)>=? "
                "THEN 0 ELSE 1 END," + time_order + "rowid LIMIT ?",
                (since_us, since_us, ready_only, time.time(), prefer_since_us, prefer_since_us, limit))]

    def defer(self, identifier, token, *, reason='not_ready', now=None):
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT record FROM dirty WHERE id=? AND token=?', (identifier, token)).fetchone()
            if row:
                previous = db.execute('SELECT attempts FROM retries WHERE id=? AND token=?',
                                      (identifier, token)).fetchone()
                attempts = previous[0] + 1 if previous else 1
                delay = min(3600, (5 if reason == 'lock_busy' else 30) * 2 ** min(attempts - 1, 8))
                db.execute('INSERT OR REPLACE INTO retries VALUES(?,?,?,?,?)',
                           (identifier, token, attempts, (time.time() if now is None else now) + delay, reason))
                db.execute('DELETE FROM dirty WHERE id=? AND token=?', (identifier, token))
                db.execute('INSERT INTO dirty VALUES(?,?,?)', (identifier, token, row['record']))


def revalidate_remounted_seals(runner, record):
    """Recover only a mount device-number change, with stable verified bytes.

    Called under every recording stage lock by replay. Model receipts and
    content bindings remain untouched; the old input seals stay in history.
    Inode, size, timestamps, resolved path and content must all still match.
    This never authorizes a replacement file or a source-signature change.
    """
    from .device_day_contract import digest, read_json
    from .device_day_inputs import binding, durable_json, identity, input_path, verify_content
    from .device_day_io import slot
    from .device_day_write_guard import storage_guard
    updates = []
    with storage_guard(runner.config), slot(runner.config):
        for stage in ('vision', 'stt'):
            path = input_path(runner, record, stage)
            if not path.is_file():
                continue
            saved = read_json(path)
            expected_signature = (record.get('audio', {}).get('source_signature')
                                  if stage == 'stt' else record['source_signature'])
            if saved.get('source_signature') != expected_signature:
                raise ValueError('Remount recovery source signature differs')
            fields = ('recording_id', 'camera_key', 'configured_role', 'recording_start_us', 'recording_end_us')
            if any(saved.get('recording', {}).get(k) != record.get(k) for k in fields):
                raise ValueError('Remount recovery device/day identity differs')
            updated, changed = deepcopy(saved), False
            snapshots = []
            for source in updated['sources']:
                original = Path(source['original_path'])
                if not original.exists():
                    continue  # Normal publication verifies the surviving archive.
                observed = identity(original)
                if (observed[1:] != source['identity'][1:]
                        or str(original.resolve()) != source.get('resolved_path', str(original.resolve()))):
                    raise ValueError('Remount recovery refuses a changed capture')
                snapshots.append((original, observed))
                if observed != source['identity']:
                    if not verify_content(original, source):
                        raise ValueError('Remount recovery content differs from seal')
                    source['identity'] = observed
                    changed = True
            if changed:
                if binding(updated) != binding(saved):
                    raise ValueError('Remount recovery changed content binding')
                updates.append((path, saved, updated, snapshots))
        # Validate the complete set before publishing any new seal. A crash
        # between independent seals is safe: each has unchanged content binding.
        for path, saved, _updated, snapshots in updates:
            if read_json(path) != saved or any(identity(p) != stamp for p, stamp in snapshots):
                raise ValueError('Capture or input seal changed during remount recovery')
        for path, saved, updated, _snapshots in updates:
            previous = digest(saved)
            durable_json(path.parent / 'history' / f'{path.stem}-{previous}.json', saved)
            updated['mount_revalidation'] = {'previous_seal_digest': previous, 'verified_at': time.time(),
                'content_binding_unchanged': True, 'model_invoked': False}
            durable_json(path, updated)
    return len(updates)


def reconcile(runner, *, stop=None):
    from .device_day import exclusive
    from .device_day_contract import STAGES
    journal = PublicationJournal(runner.runtime_root)
    from .device_day_schedule import processing_cutoff
    completed = 0
    recent = round((time.time() - runner.settings.get('live_priority_seconds', 14400)) * 1e6)
    for token, record in journal.pending(since_us=processing_cutoff(runner.settings), prefer_since_us=recent,
                                        latest_first=runner.settings.get('latest_first', False), ready_only=True):
        if stop is not None and stop.is_set():
            break  # Finish one atomic recording, then preserve the remaining journal.
        try:
            # Never acknowledge an in-flight stage before its receipt exists.
            # A crash between receipt and index publication must remain replayable.
            with ExitStack() as locks:
                for stage in ('all', *STAGES):
                    locks.enter_context(exclusive(runner.runtime_root / 'locks' /
                        f"{record['recording_id']}.{stage}.lock"))
                from .device_day_inplace import active, publish
                if active(runner, record):
                    revalidate_remounted_seals(runner, record)
                    done = publish(runner, record)
                else:
                    done = runner.refresh_index(runner.layout(record), recording_id=record['recording_id'])
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
            journal.defer(record['recording_id'], token,
                          reason='lock_busy' if isinstance(exc, BlockingIOError) else type(exc).__name__)
    return completed
