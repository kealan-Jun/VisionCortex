"""Transactional task outboxes and replayable delivery; no media or secrets."""

import hashlib
import json
from pathlib import Path
import time
from .sqlite_store import connection


def initialize(db):
    db.execute("""CREATE TABLE IF NOT EXISTS task_events(
        seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
        revision TEXT, attempt INTEGER, state TEXT NOT NULL, at REAL NOT NULL, data TEXT NOT NULL)""")


def append(db, task_id, state, *, revision="", attempt=0, data=None):
    # Caller owns the queue transaction: task completion and delivery intent
    # either both commit or both roll back. Never include configuration/secrets.
    initialize(db)
    db.execute(
        "INSERT INTO task_events(task_id,revision,attempt,state,at,data) VALUES(?,?,?,?,?,?)",
        (
            task_id,
            revision,
            attempt,
            state,
            time.time(),
            json.dumps(data or {}, ensure_ascii=False),
        ),
    )


class EventStore:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "state" / "TaskEvents.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript("""PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE, source TEXT, task_id TEXT, revision TEXT,
                attempt INTEGER, state TEXT, at REAL, data TEXT);
              CREATE INDEX IF NOT EXISTS task_history ON events(task_id,seq);
              CREATE TABLE IF NOT EXISTS imports(source TEXT PRIMARY KEY, cursor INTEGER);
              CREATE TABLE IF NOT EXISTS consumers(name TEXT PRIMARY KEY, acknowledged INTEGER NOT NULL,
                delivered INTEGER NOT NULL, updated REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS callbacks(name TEXT PRIMARY KEY, next_at REAL, attempts INTEGER,
                last_error TEXT);""")

    def harvest(self):
        sources = [
            (f"nas:{stage}", self.root / "device-day" / f"queue-{stage}.sqlite3")
            for stage in ("retention", "vision", "stt", "understanding", "report")
        ]
        sources.append(("offline", self.root / "state" / "web_run_queue.sqlite3"))
        count = 0
        for source, path in sources:
            if not path.is_file():
                continue
            with connection(self.path, readonly=True) as db:
                row = db.execute(
                    "SELECT cursor FROM imports WHERE source=?", (source,)
                ).fetchone()
                cursor = row[0] if row else 0
            with connection(path, readonly=True, timeout=0.2) as db:
                if not db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='task_events'"
                ).fetchone():
                    continue
                rows = list(
                    db.execute(
                        "SELECT * FROM task_events WHERE seq>? ORDER BY seq LIMIT 500",
                        (cursor,),
                    )
                )
            with connection(self.path) as db:
                for row in rows:
                    eid = hashlib.sha256(f"{source}:{row['seq']}".encode()).hexdigest()
                    count += db.execute(
                        "INSERT OR IGNORE INTO events(event_id,source,task_id,revision,attempt,state,at,data) VALUES(?,?,?,?,?,?,?,?)",
                        (
                            eid,
                            source,
                            row["task_id"],
                            row["revision"],
                            row["attempt"],
                            row["state"],
                            row["at"],
                            row["data"],
                        ),
                    ).rowcount
                if rows:
                    db.execute(
                        "INSERT INTO imports VALUES(?,?) ON CONFLICT(source) DO UPDATE SET cursor=MAX(cursor,excluded.cursor)",
                        (source, rows[-1]["seq"]),
                    )
        return count

    def read(self, *, after=0, limit=100, task_id=None, consumer=None, task_ids=None):
        limit = max(1, min(500, int(limit)))
        with connection(self.path) as db:
            if consumer:
                db.execute(
                    "INSERT OR IGNORE INTO consumers VALUES(?,0,0,?)",
                    (consumer, time.time()),
                )
                after = db.execute(
                    "SELECT acknowledged FROM consumers WHERE name=?", (consumer,)
                ).fetchone()[0]
            where, params = "seq>?", [max(0, int(after))]
            if task_id:
                where += " AND task_id=?"
                params.append(task_id)
            if task_ids is not None:
                where += " AND task_id IN (SELECT value FROM json_each(?))"
                params.append(json.dumps(task_ids))
            rows = [
                dict(r)
                for r in db.execute(
                    f"SELECT * FROM events WHERE {where} ORDER BY seq LIMIT ?",
                    (*params, limit),
                )
            ]
            cursor = rows[-1]["seq"] if rows else after
            if consumer:
                db.execute(
                    "UPDATE consumers SET delivered=MAX(delivered,?),updated=? WHERE name=?",
                    (cursor, time.time(), consumer),
                )
        for row in rows:
            row["data"] = json.loads(row["data"])
        return {
            "events": rows,
            "next_cursor": cursor,
            "delivery": "at_least_once_until_acknowledged",
        }

    def acknowledge(self, consumer, cursor):
        with connection(self.path) as db:
            row = db.execute(
                "SELECT * FROM consumers WHERE name=?", (consumer,)
            ).fetchone()
            if not row or cursor < 0 or cursor > row["delivered"]:
                raise ValueError("Cannot acknowledge an undelivered cursor")
            db.execute(
                "UPDATE consumers SET acknowledged=MAX(acknowledged,?),updated=? WHERE name=?",
                (cursor, time.time(), consumer),
            )

    def dispatch(self, subscriptions, *, transport=None):
        """Configured recipients only; bounded, signed, no redirects, ACK required."""
        import os
        import hmac
        from urllib.parse import urlsplit
        import httpx

        for sub in subscriptions:
            name = "callback:" + str(sub["name"])
            url = urlsplit(sub["url"])
            if (
                url.scheme != "https"
                or not url.hostname
                or url.username
                or url.password
                or url.fragment
            ):
                raise ValueError(
                    "Callback requires an explicitly configured HTTPS recipient"
                )
            with connection(self.path, readonly=True) as db:
                state = db.execute(
                    "SELECT * FROM callbacks WHERE name=?", (name,)
                ).fetchone()
            if state and state["next_at"] > time.time():
                continue
            payload = self.read(consumer=name)
            if not payload["events"]:
                continue
            secret = os.environ.get(sub["secret_env"])
            if not secret:
                continue
            body = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            stamp = str(int(time.time()))
            signature = hmac.new(
                secret.encode(), stamp.encode() + b"." + body, hashlib.sha256
            ).hexdigest()
            try:
                with httpx.Client(
                    timeout=10, follow_redirects=False, transport=transport
                ) as client:
                    response = client.post(
                        sub["url"],
                        content=body,
                        headers={
                            "Content-Type": "application/json",
                            "X-VisionCortex-Timestamp": stamp,
                            "X-VisionCortex-Signature": signature,
                        },
                    )
                    response.raise_for_status()
                    if (
                        response.json().get("acknowledged_cursor")
                        != payload["next_cursor"]
                    ):
                        raise ValueError("Delivery not acknowledged")
                self.acknowledge(name, payload["next_cursor"])
                attempts, error, delay = 0, None, 0
            except (httpx.HTTPError, ValueError):
                attempts = (state["attempts"] if state else 0) + 1
                error, delay = (
                    "delivery_not_acknowledged",
                    min(3600, 5 * 2 ** min(attempts, 10)),
                )
            with connection(self.path) as db:
                db.execute(
                    "INSERT OR REPLACE INTO callbacks VALUES(?,?,?,?)",
                    (name, time.time() + delay, attempts, error),
                )
