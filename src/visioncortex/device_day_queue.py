"""Durable FIFO backlog; discovery limits never remove already enqueued work."""
from __future__ import annotations

import json
from contextlib import contextmanager
import logging
import sqlite3
import threading
import time
from pathlib import Path


class DeviceDayQueue:
    def __init__(self, path: Path, *, latest_first=False, scheduling=None):
        if not isinstance(latest_first, bool):
            raise ValueError('device_day.latest_first must be a boolean')
        self.latest_first = latest_first
        from .device_day_cohort import scheduling_options
        self.scheduling = scheduling_options({'scheduling': scheduling})
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS recordings (
              recording_id TEXT PRIMARY KEY, revision TEXT NOT NULL, payload TEXT NOT NULL,
              status TEXT NOT NULL, queued_at REAL NOT NULL, updated_at REAL NOT NULL,
              lease_owner TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
              result TEXT, completed_at REAL, wall_seconds REAL);
            CREATE TABLE IF NOT EXISTS camera_dispatch (camera_key TEXT PRIMARY KEY, last_claim REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS device_day_fifo ON recordings(status, queued_at);
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {r[1] for r in db.execute("PRAGMA table_info(recordings)")}
            from .task_events import initialize
            initialize(db)
            if "input_status" not in columns:
                db.execute("ALTER TABLE recordings ADD COLUMN input_status TEXT NOT NULL DEFAULT 'ready'")
            if "started_at" not in columns:
                db.execute("ALTER TABLE recordings ADD COLUMN started_at REAL")
            if "retry_limit" not in columns:
                db.execute("ALTER TABLE recordings ADD COLUMN retry_limit INTEGER NOT NULL DEFAULT 0")
            for name, kind in {'processing_cohort': 'TEXT', 'admission_reason': 'TEXT',
                               'admitted_at': 'REAL', 'admission_signature': 'TEXT',
                               'first_ready_at': 'REAL', 'last_dispatch_at': 'REAL', 'blocked_reason': 'TEXT'}.items():
                if name not in columns:
                    db.execute(f'ALTER TABLE recordings ADD COLUMN {name} {kind}')
            db.execute('CREATE TABLE IF NOT EXISTS dispatch_costs ('
                       'kind TEXT NOT NULL, identity TEXT NOT NULL, seconds REAL NOT NULL, '
                       'PRIMARY KEY(kind,identity))')
            db.execute('CREATE TABLE IF NOT EXISTS source_versions ('
                       'recording_id TEXT NOT NULL, source_signature TEXT NOT NULL, revision TEXT NOT NULL, '
                       'snapshot TEXT NOT NULL, superseded_at REAL NOT NULL, '
                       'PRIMARY KEY(recording_id,source_signature,revision))')
            db.execute("""CREATE TABLE IF NOT EXISTS retry_authorizations (
                recording_id TEXT NOT NULL, revision TEXT NOT NULL, repair_id TEXT NOT NULL,
                authorized_at REAL NOT NULL, previous_attempts INTEGER NOT NULL,
                previous_retry_limit INTEGER NOT NULL, retry_limit INTEGER NOT NULL,
                previous_result TEXT NOT NULL, evidence TEXT NOT NULL,
                PRIMARY KEY(recording_id, revision, repair_id))""")

    def connect(self):
        from .sqlite_store import connection
        return connection(self.path)

    def enqueue(self, recording, revision):
        observed = time.time()
        payload = json.dumps(recording, ensure_ascii=False)
        status = "queued" if recording.get("configured_role") in {"first_person", "third_person"} else "needs_camera_role"
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute("SELECT * FROM recordings WHERE recording_id=? AND "
                             "(json_extract(payload,'$.source_signature') IS NOT ? OR revision!=?) AND "
                             "(status!='running' OR COALESCE(lease_until,0)<?)",
                             (recording['recording_id'], recording.get('source_signature'), revision, observed)).fetchone()
            if old:
                signature = json.loads(old['payload']).get('source_signature') or 'unknown'
                db.execute('INSERT OR IGNORE INTO source_versions VALUES(?,?,?,?,?)',
                           (old['recording_id'], signature, old['revision'], json.dumps(dict(old)), observed))
            db.execute("UPDATE recordings SET processing_cohort=NULL,admission_reason=NULL,admitted_at=NULL,"
                       "admission_signature=NULL,first_ready_at=NULL,last_dispatch_at=NULL,blocked_reason=NULL "
                       "WHERE recording_id=? AND json_extract(payload,'$.source_signature') IS NOT ? "
                       "AND (status!='running' OR COALESCE(lease_until,0)<?)",
                       (recording['recording_id'], recording.get('source_signature'), observed))
            changed = db.execute("""INSERT INTO recordings(recording_id,revision,payload,status,queued_at,updated_at)
              VALUES(?,?,?,?,?,?) ON CONFLICT(recording_id) DO UPDATE SET
              revision=excluded.revision,payload=excluded.payload,status=excluded.status,
              queued_at=excluded.queued_at,updated_at=excluded.updated_at,lease_owner=NULL,
              lease_until=NULL,result=NULL,completed_at=NULL,wall_seconds=NULL,started_at=NULL,attempts=0,retry_limit=0,
              input_status=CASE WHEN recordings.revision!=excluded.revision THEN 'ready' ELSE recordings.input_status END
              WHERE (recordings.revision != excluded.revision OR
                (recordings.status='needs_camera_role' AND excluded.status='queued')) AND
                (recordings.status != 'running' OR COALESCE(recordings.lease_until,0) < ?)
            """, (recording["recording_id"], revision, payload, status, observed, observed, observed)).rowcount
            if changed:
                from .task_events import append
                append(db, recording["recording_id"], status, revision=revision, data={"camera": recording.get("camera_key"), "date": recording.get("archive_date")})
            # Visual revisions intentionally exclude scheduling/audio metadata.
            # Keep the claim payload aligned with the inventory used to verify
            # prerequisites, even when the visual work itself is unchanged.
            # Never replace an actively leased worker's input snapshot.
            db.execute("""UPDATE recordings SET payload=?
              WHERE recording_id=? AND revision=? AND payload!=? AND
                (status!='running' OR COALESCE(lease_until,0)<?) AND
                (?=0 OR status!='completed')""",
                       (payload, recording["recording_id"], revision, payload, observed, self.latest_first))

    def pending(self):
        with self.connect() as db:
            return [json.loads(row["payload"]) for row in db.execute(
                "SELECT payload FROM recordings WHERE status != 'completed'")]

    def grant_admission(self, recording, *, cohort='live', reason='recent_ready', at=None):
        """Grant only after readiness checks; old rows remain unadmitted by migration."""
        if cohort not in {'live', 'current_recovery', 'historical_backfill'} or not reason:
            raise ValueError('Invalid processing cohort or admission reason')
        signature = recording.get('source_signature')
        if not isinstance(signature, str) or not signature:
            raise ValueError('Admission requires an exact source signature')
        current = time.time() if at is None else at
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT processing_cohort,admission_signature FROM recordings WHERE recording_id=?',
                                  (recording['recording_id'],)).fetchone()
            if (previous and previous['processing_cohort'] == 'historical_backfill'
                    and previous['admission_signature'] == signature and cohort == 'live'):
                # Seeing an authorized historical job in a recent worker does
                # not promote its paid permission to the live cohort.
                return False
            return bool(db.execute("UPDATE recordings SET processing_cohort=?,admission_reason=?,"
                "admitted_at=CASE WHEN admission_signature=? THEN COALESCE(admitted_at,?) ELSE ? END,"
                "first_ready_at=CASE WHEN admission_signature=? THEN COALESCE(first_ready_at,?) ELSE ? END,"
                "admission_signature=? WHERE recording_id=? AND "
                "json_extract(payload,'$.source_signature')=?",
                (cohort, reason, signature, current, current, signature, current, current,
                 signature, recording['recording_id'], signature)).rowcount)

    def admissions(self, *, include_history=False):
        if type(include_history) is not bool:
            raise ValueError('Historical admission selection must be a boolean')
        with self.connect() as db:
            return {row['recording_id']: row['admission_signature'] for row in db.execute(
                "SELECT recording_id,admission_signature FROM recordings WHERE "
                "(processing_cohort IN ('live','current_recovery') OR (? AND processing_cohort='historical_backfill')) "
                "AND admitted_at IS NOT NULL AND admission_signature=json_extract(payload,'$.source_signature')",
                (include_history,))}

    def release_hold(self, recording_id, *, reason, evidence, expected_revision=None, source_signature=None):
        """An explicit source/review decision releases a hold, never a retry sweep."""
        if not evidence:
            raise ValueError('Hold release requires review evidence')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision,payload FROM recordings WHERE recording_id=? AND blocked_reason=?',
                             (recording_id, reason)).fetchone()
            if row is None:
                return False
            if reason == 'anomalous_inactive_request_plan' and (
                    expected_revision != row['revision'] or source_signature != json.loads(row['payload']).get('source_signature')):
                raise ValueError('Plan hold release requires the exact reviewed revision and source')
            db.execute("UPDATE recordings SET status='queued',blocked_reason=NULL WHERE recording_id=?",
                       (recording_id,))
            from .task_events import append
            append(db, recording_id, 'hold_released', revision=row['revision'],
                   data={'reason': reason, 'evidence': evidence})
            return True

    def has_processing_work(self, max_attempts=0):
        """Keep expensive background audits behind ready stage work."""
        with self.connect() as db:
            return db.execute(
                "SELECT 1 FROM recordings WHERE input_status='ready' AND "
                "(status IN ('queued','running') OR (status='failed' AND attempts<MAX(?,retry_limit))) LIMIT 1",
                (max_attempts,),
            ).fetchone() is not None

    def revise_verified_completion(self, recording, previous_revision, revision):
        """Keep completion only after the caller revalidates the current receipt.

        Compare-and-swap prevents a concurrent claim or source revision from
        being overwritten. This operation does not manufacture a new result.
        """
        with self.connect() as db:
            return db.execute(
                "UPDATE recordings SET revision=?,payload=?,updated_at=? "
                "WHERE recording_id=? AND revision=? AND status='completed'",
                (revision, json.dumps(recording, ensure_ascii=False), time.time(),
                 recording['recording_id'], previous_revision),
            ).rowcount == 1

    def claim(self, owner, *, retry=False, date=None, exclude=(), allowed=None, camera_serial=False,
              camera_limit=1, max_attempts=None, latest_over_camera=False):
        if isinstance(camera_limit, bool) or not isinstance(camera_limit, int) or camera_limit < 1:
            raise ValueError('Camera concurrency must be a positive integer')
        if type(latest_over_camera) is not bool or latest_over_camera and not self.latest_first:
            raise ValueError('Time-first camera admission requires latest_first')
        current = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Filter and order in SQLite before loading one payload. The old
            # implementation decoded every pending JSON for every camera slot.
            # json_each keeps large allowlists below SQLite's parameter limit.
            conditions = ["(r.status='queued' OR (r.status='running' AND COALESCE(r.lease_until,0)<?)"
                          + (" OR (r.status='failed' AND (? IS NULL OR r.attempts<MAX(?,r.retry_limit))))" if retry else ")")]
            conditions.append("r.input_status='ready'")
            conditions.append('r.blocked_reason IS NULL')
            parameters = [current]
            if retry:
                parameters.extend([max_attempts, max_attempts])
            conditions.append("COALESCE(json_extract(r.result,'$.retry_at'),0)<=?")
            parameters.append(current)
            if exclude:
                conditions.append("r.recording_id NOT IN (SELECT value FROM json_each(?))")
                parameters.append(json.dumps(list(exclude)))
            if allowed is not None:
                conditions.append("r.recording_id IN (SELECT value FROM json_each(?))")
                parameters.append(json.dumps(list(allowed)))
            if date is not None:
                conditions.append("json_extract(r.payload,'$.archive_date')=?")
                parameters.append(date)
            if camera_serial:
                # Compute occupied camera slots once, rather than re-reading
                # every running JSON payload for each item in a large backlog.
                conditions.append("""COALESCE(json_extract(r.payload,'$.camera_key'),'') NOT IN (
                  SELECT COALESCE(json_extract(busy.payload,'$.camera_key'),'') FROM recordings busy
                  WHERE busy.status='running' AND busy.lease_until>=?
                  GROUP BY COALESCE(json_extract(busy.payload,'$.camera_key'),'') HAVING COUNT(*)>=?)""")
                parameters.extend((current, camera_limit))
            ordering = "" if self.latest_first else "COALESCE(json_extract(r.payload,'$.processing_priority'),0),"
            if self.path.stem == 'queue-retention':
                if self.latest_first:
                    # A known cleanup risk or deadline still protects source
                    # media. Mere backlog age must not starve newer slices.
                    ordering = ("CASE WHEN json_extract(r.payload,'$.archive_urgent') THEN 0 ELSE 1 END,"
                                "COALESCE(json_extract(r.payload,'$.archive_deadline'),9e99),")
                else:
                    ordering = ("CASE WHEN json_extract(r.payload,'$.archive_urgent') THEN 0 "
                                "WHEN json_extract(r.payload,'$.archive_aged') THEN 1 ELSE 2 END,"
                                "COALESCE(json_extract(r.payload,'$.archive_deadline'),9e99),r.queued_at," + ordering)
            if camera_serial and not latest_over_camera:
                ordering += ("COALESCE((SELECT last_claim FROM camera_dispatch "
                             "WHERE camera_key=json_extract(r.payload,'$.camera_key')),0),")
            if self.latest_first:
                ordering += "COALESCE(json_extract(r.payload,'$.recording_start_us'),0) DESC,"
            elif camera_serial:
                ordering += ("CASE WHEN COALESCE(json_extract(r.payload,'$.processing_priority'),0)<0 "
                             "THEN -COALESCE(json_extract(r.payload,'$.recording_start_us'),0) "
                             "ELSE COALESCE(json_extract(r.payload,'$.recording_start_us'),0) END,")
            if camera_serial and latest_over_camera:
                # Scoped CV explicitly prefers physical time. Camera rotation
                # breaks equal-time ties, while occupied camera slots remain
                # excluded above and every original lease is preserved.
                ordering += ("COALESCE((SELECT last_claim FROM camera_dispatch "
                             "WHERE camera_key=json_extract(r.payload,'$.camera_key')),0),")
            ordering += "r.queued_at,r.recording_id"
            if self.scheduling['enabled'] and self.path.stem != 'queue-retention':
                # Resource seconds, rather than task counts, determine the
                # next class/device. Waiting age is an independent escape.
                shares = json.dumps(self.scheduling['resource_shares'])
                db.execute('CREATE TEMP TABLE IF NOT EXISTS dispatch_shares (identity TEXT PRIMARY KEY, weight REAL)')
                db.execute('DELETE FROM dispatch_shares')
                db.execute('INSERT INTO dispatch_shares SELECT key,value FROM json_each(?)', (shares,))
                age = self.scheduling['max_ready_wait_seconds']
                ready = 'COALESCE(r.last_dispatch_at,r.first_ready_at,r.queued_at)'
                category = "COALESCE(json_extract(r.payload,'$.activity_kind'),r.processing_cohort,'unresolved')"
                ordering = (f'CASE WHEN {ready}<={current-age} THEN 0 ELSE 1 END,'
                            f'CASE WHEN {ready}<={current-age} THEN {ready} ELSE 9e99 END,'
                            f"COALESCE((SELECT seconds FROM dispatch_costs WHERE kind='class' AND identity={category}),0) / "
                            f'COALESCE((SELECT weight FROM dispatch_shares WHERE identity={category}),1),'
                            "COALESCE((SELECT seconds FROM dispatch_costs WHERE kind='camera' AND "
                            "identity=json_extract(r.payload,'$.camera_key')),0)," + ordering)
            row = db.execute("SELECT r.* FROM recordings r WHERE " + " AND ".join(conditions)
                             + " ORDER BY " + ordering + " LIMIT 1", parameters).fetchone()
            if row is None:
                return None
            if camera_serial:
                camera = json.loads(row["payload"]).get("camera_key")
                if camera:
                    db.execute("INSERT INTO camera_dispatch(camera_key,last_claim) VALUES(?,?) "
                               "ON CONFLICT(camera_key) DO UPDATE SET last_claim=excluded.last_claim",
                               (camera, current))
            db.execute("""UPDATE recordings SET status='running',lease_owner=?,lease_until=?,
              attempts=attempts+1,updated_at=?,started_at=?,last_dispatch_at=? WHERE recording_id=?""",
                       (owner, current + 90, current, current, current, row["recording_id"]))
            from .task_events import append
            append(db, row['recording_id'], 'running', revision=row['revision'], attempt=row['attempts']+1)
            return json.loads(row["payload"])

    def renew(self, owner):
        with self.connect() as db:
            db.execute("UPDATE recordings SET lease_until=? WHERE status='running' AND lease_owner=?",
                       (time.time() + 90, owner))

    @contextmanager
    def heartbeat(self, owner, interval=10):
        """Renew while result publication or queue summaries occupy the caller."""
        stopped = threading.Event()
        def renew():
            while not stopped.wait(interval):
                try:
                    self.renew(owner)
                except sqlite3.OperationalError:
                    logging.getLogger(__name__).warning('Queue lease renewal delayed: %s', self.path.name)
        thread = threading.Thread(target=renew, name='device-queue-lease', daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=31)

    def _account_dispatch(self, db, owner, recording_id, seconds):
        row = db.execute('SELECT payload,processing_cohort FROM recordings WHERE recording_id=? AND lease_owner=?',
                         (recording_id, owner)).fetchone()
        if row and self.scheduling['enabled']:
            record = json.loads(row['payload'])
            for kind, identity in (('camera', record.get('camera_key') or 'unknown'),
                                   ('class', record.get('activity_kind') or row['processing_cohort'] or 'unresolved')):
                db.execute('INSERT INTO dispatch_costs VALUES(?,?,?) ON CONFLICT(kind,identity) '
                           'DO UPDATE SET seconds=seconds+excluded.seconds', (kind, identity, max(0, seconds)))

    def authorize_input_recovery(self, expected, repair_id, proof):
        """CAS one bounded retry per verified source generation; keep failures."""
        from .device_day_contract import digest
        if (not isinstance(repair_id, str) or len(repair_id) != 64
                or any(c not in '0123456789abcdef' for c in repair_id)
                or proof.get('status') != 'verified' or not proof.get('verified_seals')
                or proof.get('recording_id') != expected['recording_id']
                or proof.get('current_queue_revision') != expected['revision']
                or proof.get('model_invoked') is not False or proof.get('capture_modified') is not False
                or not proof.get('source_generation')):
            raise ValueError('Input recovery needs verified exact-source evidence')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM recordings WHERE recording_id=?', (expected['recording_id'],)).fetchone()
            if row is None or dict(row) != expected:
                return 'queue_changed'
            row = dict(row)
            failure = json.loads(row['result'] or '{}')
            if (row['status'] != 'failed' or row['input_status'] != 'ready' or row['lease_owner']
                    or row['lease_until'] or row['blocked_reason'] or failure.get('error_type') != 'InputChanged'):
                return 'not_source_failure'
            db.execute('CREATE TABLE IF NOT EXISTS input_recovery_grants('
                       'recording_id TEXT,revision TEXT,repair_id TEXT,proof TEXT,previous_failure TEXT,'
                       'authorized_at REAL,PRIMARY KEY(recording_id,revision,repair_id))')
            if db.execute('SELECT 1 FROM input_recovery_grants WHERE recording_id=? AND revision=? AND repair_id=?',
                          (row['recording_id'], row['revision'], repair_id)).fetchone():
                return 'repair_already_authorized'
            if row['retry_limit'] > row['attempts']:
                return 'retry_already_available'
            db.execute('INSERT INTO input_recovery_grants VALUES(?,?,?,?,?,?)',
                       (row['recording_id'], row['revision'], repair_id, json.dumps(proof), row['result'], time.time()))
            db.execute('UPDATE recordings SET retry_limit=?,updated_at=? WHERE recording_id=?',
                       (row['attempts'] + 1, time.time(), row['recording_id']))
            from .task_events import append
            append(db, row['recording_id'], 'input_recovery_authorized', revision=row['revision'],
                   attempt=row['attempts'], data={'repair_id': repair_id, 'proof_sha256': digest(proof), 'budget': 1})
        return 'retry_authorized'

    def finish(self, owner, recording_id, result, seconds):
        if result.get('status') in {'waiting_for_plan_review', 'submitted_unknown'}:
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                self._account_dispatch(db, owner, recording_id, seconds)
                reason = result.get('reason') or result['status']
                db.execute("UPDATE recordings SET status='blocked',blocked_reason=?,result=?,"
                           "lease_owner=NULL,lease_until=NULL,updated_at=? WHERE recording_id=? AND lease_owner=?",
                           (reason, json.dumps(result, ensure_ascii=False), time.time(), recording_id, owner))
            return
        if result.get('status') == 'waiting_for_prerequisite':
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT revision,attempts FROM recordings WHERE recording_id=? AND lease_owner=?',
                                 (recording_id, owner)).fetchone()
                if row:
                    db.execute("UPDATE recordings SET status='waiting_for_prerequisite',result=?,"
                               "attempts=MAX(0,attempts-1),lease_owner=NULL,lease_until=NULL,updated_at=? "
                               "WHERE recording_id=? AND lease_owner=?",
                               (json.dumps(result), time.time(), recording_id, owner))
                    from .task_events import append
                    append(db, recording_id, 'waiting_for_prerequisite', revision=row['revision'],
                           attempt=row['attempts'], data={'prerequisite_stage': result.get('prerequisite_stage')})
            return
        if result.get("status") in {"running_elsewhere", "waiting_for_publication", "waiting_for_provider", "waiting_for_storage", "cancelled", "paused_for_live", "scheduled_checkpoint"}:
            # Lock contention is not a model failure. Return the slice to
            # scheduling; completed model receipts remain available for reuse.
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                self._account_dispatch(db, owner, recording_id, seconds)
                if result['status'] in {'waiting_for_provider', 'waiting_for_publication', 'waiting_for_storage', 'running_elsewhere'}:
                    previous = db.execute('SELECT result FROM recordings WHERE recording_id=? AND lease_owner=?',
                                          (recording_id, owner)).fetchone()
                    old = json.loads(previous[0] or '{}') if previous else {}
                    count = min(16, old.get('wait_count', 0) + 1) if old.get('status') == result['status'] else 1
                    reason = result.get('reason') or result['status']
                    if result.get('error_type') == 'ReceiptCapacityUnavailable' or str(result.get('message', '')).startswith('Local paid-response receipt recovery capacity exhausted'):
                        reason = 'local_receipt_capacity'
                    base = 5 if result['status'] == 'running_elsewhere' else 30
                    result = result | {'reason': reason, 'wait_count': count,
                                       'retry_at': time.time() + min(900, base * 2 ** min(count - 1, 5))}
                db.execute("""UPDATE recordings SET status='queued',result=?,
                  attempts=MAX(0,attempts-1),lease_owner=NULL,lease_until=NULL,
                  updated_at=? WHERE recording_id=? AND lease_owner=?""",
                           (json.dumps(result, ensure_ascii=False), time.time(), recording_id, owner))
            return
        completed = result.get("status") == "completed"
        status = "completed" if completed else "failed"
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._account_dispatch(db, owner, recording_id, seconds)
            row = db.execute('SELECT revision,attempts,payload FROM recordings WHERE recording_id=? AND lease_owner=?', (recording_id, owner)).fetchone()
            db.execute("""UPDATE recordings SET status=?,result=?,completed_at=?,wall_seconds=?,
              lease_owner=NULL,lease_until=NULL,updated_at=? WHERE recording_id=? AND lease_owner=?""",
                       (status, json.dumps(result, ensure_ascii=False), time.time() if completed else None,
                        seconds, time.time(), recording_id, owner))
            if row:
                from .task_events import append
                append(db, recording_id, status, revision=row['revision'], attempt=row['attempts'], data={
                    key: result[key] for key in ('archive', 'stage', 'error_type', 'component_timings', 'measured_frame_counts') if key in result})
                if self.path.stem == 'queue-retention' and result.get('input_binding_version') != 1 and result.get('error_type') in {'FileNotFoundError', 'PermissionError', 'OSError'}:
                    state = 'missing' if result['error_type'] == 'FileNotFoundError' else 'unavailable'
                    db.execute('UPDATE recordings SET input_status=? WHERE recording_id=?', (state, recording_id))
        if row and self.path.stem == 'queue-retention' and result.get('input_binding_version') != 1 and result.get('error_type') in {'FileNotFoundError', 'PermissionError', 'OSError'}:
            from .input_availability import Availability
            record = json.loads(row['payload'])
            Availability(self.path.parent).mark(record, state, reason=result['error_type'])

    def wait_for_prerequisite(self, recording_id, revision, prerequisite):
        """CAS a queued or precisely identified legacy failure into a wait.

        Only the final historical attempt is known to be a prerequisite error.
        Refund that one attempt, preserving any earlier genuine failure budget.
        """
        from .device_day_prerequisites import legacy_prerequisite_failure
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM recordings WHERE recording_id=? AND revision=?',
                             (recording_id, revision)).fetchone()
            expired = bool(row and row['status'] == 'running' and (row['lease_until'] or 0) < time.time())
            if not row or row['status'] not in {'queued', 'failed'} and not expired:
                return False
            previous = json.loads(row['result'] or '{}')
            failed = row['status'] == 'failed'
            if failed and legacy_prerequisite_failure(previous) != prerequisite:
                return False
            result = {'status': 'waiting_for_prerequisite', 'recording_id': recording_id,
                      'prerequisite_stage': prerequisite}
            if failed:
                result['previous_failure'] = previous
            db.execute("UPDATE recordings SET status='waiting_for_prerequisite',result=?,"
                       "attempts=MAX(0,attempts-?),lease_owner=NULL,lease_until=NULL,updated_at=? "
                       "WHERE recording_id=? AND revision=?",
                       (json.dumps(result), int(failed), time.time(), recording_id, revision))
            from .task_events import append
            append(db, recording_id, 'waiting_for_prerequisite', revision=revision,
                   attempt=row['attempts'], data={'prerequisite_stage': prerequisite,
                   'legacy_failure_refunded': failed, 'refunded_attempts': int(failed),
                   'expired_lease_recovered': expired})
            return True

    def migrate_prerequisite_failures(self):
        from .device_day_prerequisites import legacy_prerequisite_failure
        with self.connect() as db:
            rows = list(db.execute("SELECT recording_id,revision,result FROM recordings WHERE status='failed' "
                                   "AND json_extract(result,'$.error_type')='ValueError'"))
        for row in rows:
            prerequisite = legacy_prerequisite_failure(json.loads(row['result'] or '{}'))
            if prerequisite:
                self.wait_for_prerequisite(row['recording_id'], row['revision'], prerequisite)

    def resume_prerequisite(self, recording_id, revision):
        """Caller has verified every exact prerequisite receipt and artifact."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            changed = db.execute("UPDATE recordings SET status='queued',updated_at=? "
                                 "WHERE recording_id=? AND revision=? AND status='waiting_for_prerequisite'",
                                 (time.time(), recording_id, revision)).rowcount
            if changed:
                from .task_events import append
                append(db, recording_id, 'prerequisite_ready', revision=revision,
                       data={'all_prerequisites_verified': True})
            return bool(changed)

    def sync_availability(self, states):
        with self.connect() as db:
            db.executemany("UPDATE recordings SET input_status=? WHERE recording_id=? AND status NOT IN ('completed','running') AND input_status!=? AND json_extract(payload,'$.source_signature') IS ?",
                           [(s['state'], key, s['state'], s.get('signature')) for key,s in states.items()])

    def snapshot(self):
        with self.connect() as db:
            current = time.time()
            # Expired leases are waiting for recovery, not live concurrency.
            counts = {row["effective_status"]: row["n"] for row in db.execute("""
              SELECT CASE WHEN status NOT IN ('completed','running') AND input_status!='ready' THEN 'input_'||input_status
                WHEN status='running' AND COALESCE(lease_until,0)<?
                THEN 'queued' ELSE status END effective_status, COUNT(*) n
              FROM recordings GROUP BY effective_status""", (current,))}
            expired = db.execute("SELECT COUNT(*) FROM recordings WHERE status='running' AND COALESCE(lease_until,0)<?",
                                 (current,)).fetchone()[0]
            prerequisite_waits = {row['stage'] or 'unknown': row['n'] for row in db.execute(
                "SELECT json_extract(result,'$.prerequisite_stage') AS stage,COUNT(*) AS n "
                "FROM recordings WHERE status='waiting_for_prerequisite' GROUP BY stage")}
            queued = db.execute("""SELECT MIN(queued_at) oldest,
              COALESCE(SUM(COALESCE(json_extract(payload,'$.duration_seconds'),0)),0) capture_seconds,
              COALESCE(SUM(json_extract(payload,'$.media_duration_seconds')),0) media_seconds,
              COALESCE(SUM(json_extract(payload,'$.media_duration_seconds') IS NULL),0) unknown
              FROM recordings WHERE (status='queued' AND input_status='ready') OR (status='running' AND COALESCE(lease_until,0)<?)""",
                                (current,)).fetchone()
            oldest = queued["oldest"]
            blockers = {row['reason']: row['n'] for row in db.execute(
                'SELECT blocked_reason AS reason,count(*) AS n FROM recordings WHERE blocked_reason IS NOT NULL GROUP BY blocked_reason')}
            plan_reviews = db.execute("SELECT MIN(COALESCE(first_ready_at,queued_at)),SUM(?-COALESCE(first_ready_at,queued_at)>900) FROM recordings WHERE blocked_reason='anomalous_inactive_request_plan'", (current,)).fetchone()
            ready_at = db.execute("SELECT MIN(first_ready_at) FROM recordings WHERE status!='completed'").fetchone()[0]
            work = db.execute("SELECT SUM(json_extract(payload,'$.estimated_processing_seconds')),"
                              "SUM(json_extract(payload,'$.estimated_processing_seconds') IS NULL) "
                              "FROM recordings WHERE status='queued' AND input_status='ready' AND blocked_reason IS NULL").fetchone()
            completed = db.execute("SELECT result,wall_seconds FROM recordings WHERE status='completed' ORDER BY completed_at DESC LIMIT 100").fetchall()
        known = [(json.loads(r["result"] or "{}").get("media_duration_seconds"), r["wall_seconds"]) for r in completed]
        measured = [(float(media), float(wall)) for media, wall in known if media is not None and wall and wall > 0]
        return {"counts": counts,
                "prerequisite_wait_count": sum(prerequisite_waits.values()),
                "prerequisite_waits_by_stage": prerequisite_waits,
                "expired_lease_count": expired,
                "queued_capture_window_seconds": queued["capture_seconds"],
                "queued_media_seconds": queued["media_seconds"] if not queued["unknown"] else None,
                "queued_media_duration_unknown_count": queued["unknown"],
                "oldest_wait_seconds": max(0, time.time() - oldest) if oldest else 0,
                "oldest_ready_wait_seconds": max(0, current-ready_at) if ready_at else None,
                "blocked_reasons": blockers,
                "oldest_plan_review_wait_seconds": max(0, current-plan_reviews[0]) if plan_reviews[0] is not None else None,
                "overdue_plan_review_count": plan_reviews[1] or 0,
                "estimated_ready_service_seconds": work[0] if not work[1] else None,
                "overload_status": ('UNKNOWN_COST' if work[1] else 'OVERLOADED' if (work[0] or 0) >
                                    self.scheduling['capacity_horizon_seconds'] else 'WITHIN_HORIZON'),
                "backlog_warning": bool(oldest and time.time() - oldest > 900),
                "completed_sample_count": len(completed), "measured_media_sample_count": len(measured),
                "media_seconds_per_worker_second": sum(m for m, _ in measured) / sum(w for _, w in measured) if measured else None,
                "capacity_claim": "measured_worker_service_rate_not_8_to_12_hour_stability_proof"}
