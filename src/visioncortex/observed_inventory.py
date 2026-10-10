"""Incremental local discovery catalog; historical JSON is a read fallback.

Only recorder metadata is stored. Opening this database never touches media.
"""
import json
from pathlib import Path
import time

from .sqlite_store import connection


def database(root):
    return Path(root) / 'observed-inventory.sqlite3'


def read_inventory(root):
    path = database(root)
    if not path.is_file():
        legacy = Path(root) / 'observed-inventory.json'
        return json.loads(legacy.read_text()) if legacy.is_file() else {'recordings': [], 'errors': []}
    with connection(path, readonly=True) as db:
        metadata = {row['key']: json.loads(row['value']) for row in db.execute('SELECT * FROM metadata')}
        return metadata | {'source': 'nas_recording_monitor', 'recordings': [json.loads(r[0]) for r in db.execute('SELECT payload FROM observations ORDER BY id')],
                           'errors': [json.loads(r[0]) for r in db.execute('SELECT payload FROM errors ORDER BY path')]}


def revision(root):
    path = database(root)
    if path.is_file():
        with connection(path, readonly=True) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()
            return ('sqlite', int(row[0]) if row else 0)
    legacy = Path(root) / 'observed-inventory.json'
    return ('json', legacy.stat().st_mtime_ns) if legacy.is_file() else None


def observe(root, inventory, *, start_date=None):
    from .device_day_contract import archive_name
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = database(root)
    with connection(path) as db:
        db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS errors(path TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);''')
        db.execute('BEGIN IMMEDIATE')
        if not db.execute("SELECT 1 FROM metadata WHERE key='revision'").fetchone():
            legacy = root / 'observed-inventory.json'
            old = json.loads(legacy.read_text()) if legacy.is_file() else {}
            db.executemany('INSERT OR IGNORE INTO observations VALUES(?,?)',
                           [(r['recording_id'], json.dumps(r, sort_keys=True)) for r in old.get('recordings', [])])
            db.executemany('INSERT OR IGNORE INTO errors VALUES(?,?)',
                           [(r['path'], json.dumps(r, sort_keys=True)) for r in old.get('errors', []) if r.get('path')])
            db.execute("INSERT INTO metadata VALUES('revision','0')")
        changed = False
        for record in inventory.get('recordings', []):
            if record.get('recording_start_us', 0) <= 0:
                continue
            if start_date and archive_name(record['camera_key'], record['recording_start_us'])[:10] < start_date:
                continue
            payload = json.dumps(record, sort_keys=True)
            changed |= bool(db.execute('''INSERT INTO observations VALUES(?,?) ON CONFLICT(id)
                DO UPDATE SET payload=excluded.payload WHERE payload!=excluded.payload''',
                                       (record['recording_id'], payload)).rowcount)
            changed |= bool(db.execute('DELETE FROM errors WHERE path=?', (record.get('relative_path'),)).rowcount)
        for row in inventory.get('errors', []):
            if row.get('path'):
                changed |= bool(db.execute('''INSERT INTO errors VALUES(?,?) ON CONFLICT(path)
                    DO UPDATE SET payload=excluded.payload WHERE payload!=excluded.payload''',
                                           (row['path'], json.dumps(row, sort_keys=True))).rowcount)
        if 'truncated' in inventory:
            changed |= bool(db.execute('''INSERT INTO metadata VALUES('truncated',?) ON CONFLICT(key)
                DO UPDATE SET value=excluded.value WHERE value!=excluded.value''',
                                       (json.dumps(inventory['truncated']),)).rowcount)
        if changed:
            db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
            db.execute("INSERT OR REPLACE INTO metadata VALUES('observed_at',?)", (str(time.time()),))
        return changed
