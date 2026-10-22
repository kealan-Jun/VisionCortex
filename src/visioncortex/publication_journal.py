"""Persist unfinished publication before modifying receipts; replay without CV."""
import json
import sqlite3
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

    def ensure_pending(self, record):
        """Join existing publication work without resetting its token/backoff.

        A file publisher can discover output after a producer crash. It must
        not replace a different source generation or continually invalidate an
        in-progress replay's compare-and-swap token.
        """
        fields = ('recording_id', 'source_signature', 'camera_key', 'recording_start_us', 'recording_end_us')
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT token,record FROM dirty WHERE id=?', (record['recording_id'],)).fetchone()
            if previous:
                saved = json.loads(previous['record'])
                return previous['token'] if all(saved.get(k) == record.get(k) for k in fields) else None
            token = uuid.uuid4().hex
            db.execute('INSERT INTO dirty VALUES(?,?,?)',
                       (record['recording_id'], token, json.dumps(record)))
            return token

    def complete(self, identifier, token):
        with connection(self.path) as db:
            db.execute('DELETE FROM dirty WHERE id=? AND token=?', (identifier, token))
            db.execute('DELETE FROM retries WHERE id=? AND token=?', (identifier, token))

    def retry_now(self, identifier, token):
        """Wake one owned publication after repair; retain its failures and token.

        This only schedules existing file publication. It never changes a
        model queue, execution identity, or a paid request's retry authority.
        The normal replayer must still verify every source and receipt.
        """
        with connection(self.path) as db:
            return bool(db.execute('UPDATE retries SET next_attempt_at=0 WHERE id=? AND token=? '
                'AND EXISTS(SELECT 1 FROM dirty d WHERE d.id=retries.id AND d.token=retries.token)',
                (identifier, token)).rowcount)

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
    """Recover changed mount identities, with stable verified source bytes.

    Called under every recording stage lock by replay. Model receipts and
    content bindings remain untouched; the old input seals stay in history.
    A pinned, verified CIFS mount may also change its server inode numbers.
    Only size/timestamp/path-identical, fully hashed content can be rebound;
    ordinary local files still require their original inode. Never change a
    source signature or reinterpret a model execution key.
    """
    from .device_day_contract import digest, read_json
    from .device_day_inputs import binding, durable_json, identity, input_path, verify_content
    from .device_day_io import slot
    from .device_day_write_guard import storage_guard
    updates = []
    from .runtime_control import resource_slot
    with storage_guard(runner.config) as guard, resource_slot(runner.config, 'cpu'), slot(runner.config):
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
                mount, _ = guard.mount_for(original)
                cifs = mount is not None and mount.snapshot.get('filesystem') == 'cifs'
                if (observed[2:] != source['identity'][2:]
                        or observed[1] != source['identity'][1] and not cifs
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
                'content_binding_unchanged': True, 'model_invoked': False,
                'policy': 'verified_mount_content_identity/2'}
            durable_json(path, updated)
    return len(updates)


def reconcile(runner, *, stop=None, limit=32):
    from .device_day import exclusive
    from .device_day_contract import STAGES
    journal = PublicationJournal(runner.runtime_root)
    from .device_day_schedule import processing_cutoff
    completed = 0
    recent = round((time.time() - runner.settings.get('live_priority_seconds', 14400)) * 1e6)
    for token, record in journal.pending(limit, since_us=processing_cutoff(runner.settings), prefer_since_us=recent,
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
                inventory = runner.runtime_root/'observed-inventory.sqlite3'
                if inventory.is_file():
                    with connection(inventory, readonly=True) as db:
                        observed = db.execute('SELECT payload FROM observations WHERE id=?',
                                              (record['recording_id'],)).fetchone()
                    current = json.loads(observed['payload']) if observed else {}
                    if (not observed or any(current.get(k) != record.get(k) for k in (
                            'recording_id', 'source_signature', 'camera_key', 'recording_start_us', 'recording_end_us'))):
                        raise ValueError('Publication source differs from the current inventory')
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
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            from .receipt_projection import ProjectionDeferred, ProjectionGenerationChanged
            # One unavailable archive must not block other recoverable days.
            # The unacknowledged item stays pending for a later bounded round.
            deferred = isinstance(exc, (ProjectionDeferred, ProjectionGenerationChanged))
            if deferred or not isinstance(exc, BlockingIOError):
                import logging
                logging.getLogger(__name__).warning('Publication replay deferred %s: %s',
                                                    record['recording_id'], type(exc).__name__)
            reason = (json.dumps(exc.details(), sort_keys=True) if isinstance(exc, ProjectionGenerationChanged) else
                json.dumps({'reason': 'readable_projection_limit', 'error_type': type(exc).__name__,
                'metric': exc.metric, 'observed': exc.observed, 'limit': exc.limit}, sort_keys=True)
                if deferred else 'lock_busy' if isinstance(exc, BlockingIOError) else
                json.dumps({'reason': 'publication_verification_failed', 'error_type': type(exc).__name__,
                            'detail': str(exc)[:500]}, sort_keys=True))
            journal.defer(record['recording_id'], token, reason=reason)
    return completed
