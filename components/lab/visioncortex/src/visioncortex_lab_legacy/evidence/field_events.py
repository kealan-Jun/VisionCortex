"""Local inbox: immutable producer revisions and cursor commit in one transaction."""

import json
from datetime import datetime, timezone
from pathlib import Path

from ..device_day_contract import digest
from ..local_storage import require_local_path
from ..sqlite_store import connection


KINDS = {"scans": "photo_capture", "jobs": "ocr_observation", "bindings": "instrument_binding",
         "scene_visits": "scene_binding", "photo_measurements": "measurement_review",
         "experiment_records": "measurement_confirmation", "binding_handoffs": "instrument_handoff"}


def instant(value):
    if value is None:
        return None
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError('Event time must have a timezone')
    return moment.astimezone(timezone.utc)


def validate_event(event, instance, camera, tenant):
    if (not isinstance(event, dict) or event.get('schema_version') != 'field-event/1'
            or event.get('source') != 'field_recognition' or event.get('source_instance') != instance
            or event.get('capture_device_id') != camera or event.get('tenant') != tenant):
        raise ValueError('Event source, camera, tenant or schema mismatch')
    sequence = event.get('sequence')
    if type(sequence) is not int or sequence < 1 or event.get('revision') != sequence:
        raise ValueError('Event delivery revision is invalid')
    entity, ident = event.get('entity'), event.get('entity_id')
    if (entity not in KINDS or event.get('type') != KINDS[entity] or not isinstance(ident, str) or not ident
            or event.get('event_id') != f'{instance}:{entity}:{ident}'
            or event.get('delivery_id') != f'{instance}:{sequence}'):
        raise ValueError('Event identity is inconsistent')
    start, end = instant(event.get('occurred_start')), instant(event.get('occurred_end'))
    if end is not None and (start is None or end < start):
        raise ValueError('Event physical interval is invalid')
    for name in ('received_at', 'processed_at', 'recorded_at'):
        instant(event.get(name))
    if not isinstance(event.get('evidence_refs'), list) or not event['evidence_refs']:
        raise ValueError('Event has no receipt reference')
    for ref in event['evidence_refs']:
        if (not isinstance(ref, dict) or ref.get('source_instance') != instance
                or ref.get('outbox_seq') != sequence or ref.get('entity') != entity
                or ref.get('entity_id') != ident):
            raise ValueError('Event receipt identity is inconsistent')
    if not isinstance(event.get('payload'), dict) or not isinstance(event.get('associations'), dict):
        raise ValueError('Event payload or associations are invalid')
    uncertainty = (event.get('clock') or {}).get('uncertainty_us')
    if uncertainty is not None and (type(uncertainty) is not int or uncertainty < 0):
        raise ValueError('Event clock uncertainty is invalid')
    if type(event.get('withdrawn')) is not bool:
        raise ValueError('Event withdrawal state is invalid')
    # Reject nonfinite JSON before obtaining a write transaction.
    digest(event)


class FieldEventInbox:
    def __init__(self, runtime_root):
        self.path = require_local_path(Path(runtime_root) / 'field-events.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS field_delivery(
                    source_key TEXT NOT NULL, source_instance TEXT NOT NULL, sequence INTEGER NOT NULL,
                    camera_id TEXT NOT NULL, event_id TEXT NOT NULL, digest TEXT NOT NULL, document TEXT NOT NULL,
                    PRIMARY KEY(source_key,source_instance,sequence));
                CREATE TABLE IF NOT EXISTS field_current(
                    source_key TEXT NOT NULL, source_instance TEXT NOT NULL, event_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL, document TEXT NOT NULL,
                    PRIMARY KEY(source_key,source_instance,event_id));
                CREATE TABLE IF NOT EXISTS field_cursor(
                    source_key TEXT NOT NULL, camera_id TEXT NOT NULL, source_instance TEXT NOT NULL,
                    cursor INTEGER NOT NULL, PRIMARY KEY(source_key,camera_id));
                CREATE TABLE IF NOT EXISTS device_delivery(
                    archive TEXT NOT NULL, index_revision TEXT NOT NULL, event_id TEXT NOT NULL, document TEXT NOT NULL,
                    PRIMARY KEY(archive,index_revision,event_id));
                CREATE TABLE IF NOT EXISTS device_current(
                    archive TEXT NOT NULL, event_id TEXT NOT NULL, document TEXT NOT NULL,
                    PRIMARY KEY(archive,event_id));
                CREATE TABLE IF NOT EXISTS device_revision(archive TEXT PRIMARY KEY, revision TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS device_revision_history(
                    archive TEXT NOT NULL, revision TEXT NOT NULL, PRIMARY KEY(archive,revision));
                INSERT OR IGNORE INTO device_revision_history SELECT archive,index_revision FROM device_delivery;
                INSERT OR IGNORE INTO device_revision_history SELECT archive,revision FROM device_revision;
                CREATE TABLE IF NOT EXISTS field_publication(
                    id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL,
                    published_generation INTEGER NOT NULL, manifest_digest TEXT);
                INSERT OR IGNORE INTO field_publication VALUES(1,0,0,NULL);''')

    def cursor(self, source_key, camera):
        with connection(self.path, readonly=True) as db:
            row = db.execute('SELECT source_instance,cursor FROM field_cursor WHERE source_key=? AND camera_id=?',
                             (source_key, camera)).fetchone()
            return dict(row) if row else {'source_instance': None, 'cursor': 0}

    def ingest(self, source_key, camera, page, *, tenant=None):
        if (not isinstance(page, dict) or page.get('schema') != 'field-event-feed/1'
                or page.get('source') != 'field_recognition' or page.get('camera_id') != camera
                or page.get('tenant') != tenant):
            raise ValueError('Field event feed scope mismatch')
        instance = page.get('source_instance')
        if not isinstance(instance, str) or not instance:
            raise ValueError('Field event feed has no source instance')
        after, cursor, high = (page.get(key) for key in ('after_seq', 'next_cursor', 'high_watermark'))
        if any(type(value) is not int for value in (after, cursor, high)) or not 0 <= after <= cursor <= high:
            raise ValueError('Field event cursor is invalid')
        items = page.get('items')
        if not isinstance(items, list) or len(items) > 100 or type(page.get('has_more')) is not bool:
            raise ValueError('Field event feed page is unbounded or incomplete')
        previous = after
        for event in items:
            validate_event(event, instance, camera, tenant)
            if not previous < event['sequence'] <= cursor:
                raise ValueError('Field event feed is not in delivery order')
            previous = event['sequence']
        if page['has_more'] and cursor <= after:
            raise ValueError('Field event pagination did not advance')
        inserted = 0
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            state = db.execute('SELECT * FROM field_cursor WHERE source_key=? AND camera_id=?',
                               (source_key, camera)).fetchone()
            current = state['cursor'] if state else 0
            if state and state['source_instance'] != instance:
                raise ValueError('Producer instance changed; previous event epoch is preserved')
            if after > current:
                raise ValueError('Field feed skipped an uncommitted cursor')
            if after < current and cursor > current:
                raise ValueError('Stale field page overlaps new deliveries; fetch the current cursor')
            for event in items:
                key = (source_key, instance, event['sequence'])
                fingerprint = digest(event)
                old = db.execute('SELECT digest FROM field_delivery WHERE source_key=? AND source_instance=? AND sequence=?', key).fetchone()
                if old:
                    if old['digest'] != fingerprint:
                        raise ValueError('Producer reused a delivery identity with different content')
                    continue
                if event['sequence'] <= current:
                    raise ValueError('Replayed cursor contains a previously omitted event')
                document = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
                db.execute('INSERT INTO field_delivery VALUES(?,?,?,?,?,?,?)',
                           (*key, camera, event['event_id'], fingerprint, document))
                db.execute('''INSERT INTO field_current VALUES(?,?,?,?,?)
                    ON CONFLICT(source_key,source_instance,event_id) DO UPDATE SET
                    sequence=excluded.sequence,document=excluded.document
                    WHERE excluded.sequence>field_current.sequence''',
                           (source_key, instance, event['event_id'], event['sequence'], document))
                inserted += 1
            db.execute('''INSERT INTO field_cursor VALUES(?,?,?,?)
                ON CONFLICT(source_key,camera_id) DO UPDATE SET cursor=MAX(field_cursor.cursor,excluded.cursor)''',
                       (source_key, camera, instance, cursor))
            if inserted:
                db.execute('UPDATE field_publication SET generation=generation+1 WHERE id=1')
        return {'inserted': inserted, 'cursor': max(current, cursor), 'source_instance': instance}

    def snapshot(self):
        with connection(self.path, readonly=True) as db:
            state = dict(db.execute('SELECT * FROM field_publication WHERE id=1').fetchone())
            events = [json.loads(row['document']) for row in db.execute(
                'SELECT document FROM field_current ORDER BY source_key,source_instance,event_id')]
            events.extend(json.loads(row['document']) for row in db.execute(
                'SELECT document FROM device_current ORDER BY archive,event_id'))
        return events, state

    def replace_device_index(self, archive, revision, events):
        """Persist index revisions and withdrawals; publishing never invokes vision."""
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            old_revision = db.execute('SELECT revision FROM device_revision WHERE archive=?', (archive,)).fetchone()
            if old_revision and old_revision['revision'] == revision:
                return 0
            if db.execute('SELECT 1 FROM device_revision_history WHERE archive=? AND revision=?',
                          (archive, revision)).fetchone():
                raise ValueError('Historical device index revision cannot replace the current projection')
            current = {row['event_id']: json.loads(row['document']) for row in db.execute(
                'SELECT event_id,document FROM device_current WHERE archive=?', (archive,))}
            replacements = {event['event_id']: event for event in events}
            if len(replacements) != len(events):
                raise ValueError('Device index has duplicate event identities')
            for identifier, old in current.items():
                if identifier not in replacements:
                    replacements[identifier] = old | {'revision': revision, 'withdrawn': True,
                        'withdrawal_basis': 'not_present_in_verified_index_revision'}
            for identifier, event in replacements.items():
                document = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
                archived = db.execute('SELECT document FROM device_delivery WHERE archive=? AND index_revision=? AND event_id=?',
                                      (archive, revision, identifier)).fetchone()
                if archived and archived['document'] != document:
                    raise ValueError('Device index revision was reused with different content')
                db.execute('INSERT OR IGNORE INTO device_delivery VALUES(?,?,?,?)', (archive, revision, identifier, document))
                db.execute('INSERT INTO device_current VALUES(?,?,?) ON CONFLICT(archive,event_id) '
                           'DO UPDATE SET document=excluded.document', (archive, identifier, document))
            db.execute('INSERT INTO device_revision VALUES(?,?) ON CONFLICT(archive) DO UPDATE SET revision=excluded.revision',
                       (archive, revision))
            db.execute('INSERT INTO device_revision_history VALUES(?,?)', (archive, revision))
            db.execute('UPDATE field_publication SET generation=generation+1 WHERE id=1')
        return len(events)

    def events(self):
        return self.snapshot()[0]

    def version_count(self):
        with connection(self.path, readonly=True) as db:
            return db.execute('SELECT count(*) FROM field_delivery').fetchone()[0]

    def publication_state(self):
        state = self.snapshot()[1]
        return state | {'dirty': state['generation'] != state['published_generation']}

    def published(self, generation, fingerprint):
        with connection(self.path) as db:
            db.execute('UPDATE field_publication SET published_generation=?,manifest_digest=? '
                       'WHERE id=1 AND generation=?', (generation, fingerprint, generation))
