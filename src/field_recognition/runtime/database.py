"""Explicit SQLite migrations and bounded, committing connection scopes."""

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Database:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            raise ValueError("Database path must be a file, not the data directory")

    def get_connection(self):
        """Caller owns this connection. Prefer connection()/transaction()."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    @contextmanager
    def connection(self):
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self, mode="DEFERRED"):
        if mode not in {"DEFERRED", "IMMEDIATE", "EXCLUSIVE"}:
            raise ValueError("Invalid transaction mode")
        with self.connection() as conn:
            conn.execute("BEGIN " + mode)
            yield conn

    def close(self):
        # Each scope closes its own connection, including connections on other threads.
        pass

    def _init_db(self):
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")


def execute_script(conn, script):
    """Unlike sqlite3.executescript, preserve the caller's transaction."""
    statement = ""
    for line in script.splitlines(True):
        # Split on semicolons, but retain complete CREATE TRIGGER statements.
        for part in line.split(";")[:-1]:
            statement += part + ";"
            if sqlite3.complete_statement(statement):
                conn.execute(statement)
                statement = ""
        statement += line.split(";")[-1]
    if statement.strip() and not all(
        not l.strip() or l.strip().startswith("--") for l in statement.splitlines()
    ):
        conn.execute(statement)


MIGRATIONS = [
    {
        "version": 1,
        "name": "add_user_accounts_and_auth",
        "description": "Add user accounts, sessions, and authentication support",
        "up": """
            CREATE TABLE IF NOT EXISTS users(
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'operator', 'reviewer')),
                password_hash TEXT,
                created_at TEXT NOT NULL,
                disabled INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);

            CREATE TABLE IF NOT EXISTS device_credentials(
                device_id TEXT PRIMARY KEY,
                device_type TEXT NOT NULL,
                credential_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                disabled INTEGER NOT NULL DEFAULT 0
            );
        """,
        "down": """
            DROP TABLE IF EXISTS device_credentials;
            DROP TABLE IF EXISTS sessions;
            DROP TABLE IF EXISTS users;
        """,
    },
    {
        "version": 2,
        "name": "add_task_queue_leases",
        "description": "Add worker lease and retry management to jobs table",
        "up": """
            -- Create jobs table if not exists with all required columns
            CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY,
                status TEXT,
                document TEXT NOT NULL,
                lease_holder TEXT,
                lease_expires_at TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(status, lease_expires_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_retry ON jobs(status, retry_count);
        """,
        "down": None,
    },
    {
        "version": 3,
        "name": "add_camera_registry",
        "description": "Support multiple cameras with independent contexts",
        "up": """
            CREATE TABLE IF NOT EXISTS camera_registry(
                camera_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                device_credential_id TEXT,
                receiver_url TEXT,
                nas_photo_root TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                registered_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY(device_credential_id) REFERENCES device_credentials(device_id)
            );
            CREATE INDEX IF NOT EXISTS idx_camera_enabled ON camera_registry(enabled);
        """,
        "down": "DROP TABLE IF EXISTS camera_registry;",
    },
    {
        "version": 4,
        "name": "add_measurement_state_machine",
        "description": "Add explicit measurement lifecycle states",
        "up": """
            CREATE TABLE IF NOT EXISTS photo_measurements(
                id TEXT PRIMARY KEY,
                camera TEXT NOT NULL,
                burst_key TEXT NOT NULL,
                status TEXT NOT NULL,
                document TEXT NOT NULL,
                measurement_state TEXT NOT NULL DEFAULT 'draft',
                confirmed_at TEXT,
                confirmed_by TEXT,
                updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_measurements_state ON photo_measurements(measurement_state);
            CREATE INDEX IF NOT EXISTS idx_measurements_camera ON photo_measurements(camera);

            CREATE TABLE IF NOT EXISTS measurement_confirmations(
                id TEXT PRIMARY KEY,
                measurement_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('confirm', 'reject', 'revise')),
                reason TEXT,
                confirmed_at TEXT NOT NULL,
                fields_json TEXT,
                FOREIGN KEY(measurement_id) REFERENCES photo_measurements(id)
            );
            CREATE INDEX IF NOT EXISTS idx_confirmations_measurement ON measurement_confirmations(measurement_id);

            CREATE TABLE IF NOT EXISTS experiment_records(
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_experiment_records_created ON experiment_records(created_at);
        """,
        "down": None,
    },
    {
        "version": 5,
        "name": "add_instrument_handoff",
        "description": "Add explicit instrument handoff workflow",
        "up": """
            -- Create binding_handoffs table
            CREATE TABLE IF NOT EXISTS binding_handoffs(
                id TEXT PRIMARY KEY,
                binding_id TEXT,
                instrument_id TEXT,
                offered_by TEXT,
                offered_to TEXT,
                offered_at TEXT,
                state TEXT NOT NULL DEFAULT 'pending',
                accepted_at TEXT,
                rejected_at TEXT,
                rejection_reason TEXT,
                status TEXT,
                document TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_handoffs_state ON binding_handoffs(state);
            CREATE INDEX IF NOT EXISTS idx_handoffs_binding ON binding_handoffs(binding_id);
            CREATE INDEX IF NOT EXISTS idx_handoffs_instrument ON binding_handoffs(instrument_id);

            -- Create bindings table if not exists
            CREATE TABLE IF NOT EXISTS bindings(
                id TEXT PRIMARY KEY,
                camera TEXT,
                ended TEXT,
                document TEXT NOT NULL
            );
        """,
        "down": None,
    },
    {
        "version": 6,
        "name": "add_instrument_configuration",
        "description": "Externalize instrument types and measurement definitions",
        "up": """
            CREATE TABLE IF NOT EXISTS instrument_types(
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                measurements_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS instruments(
                id TEXT PRIMARY KEY,
                name TEXT,
                scene TEXT,
                model TEXT,
                device_no TEXT,
                measurement_ranges TEXT NOT NULL DEFAULT '{}',
                type_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_instruments_type ON instruments(type_id);
        """,
        "down": "DROP TABLE IF EXISTS instrument_types;",
    },
    {
        "version": 7,
        "name": "add_audit_trail",
        "description": "Add comprehensive audit logging",
        "up": """
            CREATE TABLE IF NOT EXISTS audit_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                device_id TEXT,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                changes_json TEXT,
                request_id TEXT,
                ip_address TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);
        """,
        "down": "DROP TABLE IF EXISTS audit_log;",
    },
    {
        "version": 8,
        "name": "add_archive_management",
        "description": "Add archive outbox and catalog for capacity control",
        "up": """
            CREATE TABLE IF NOT EXISTS archive_outbox(
                entity TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                archived_at TEXT,
                notes TEXT,
                PRIMARY KEY (entity, entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_archive_outbox_archived ON archive_outbox(archived_at);

            CREATE TABLE IF NOT EXISTS archive_catalog(
                entity TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                manifest TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                PRIMARY KEY (entity, entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_archive_catalog_indexed ON archive_catalog(indexed_at);
        """,
        "down": """
            DROP TABLE IF EXISTS archive_outbox;
            DROP TABLE IF EXISTS archive_catalog;
        """,
    },
]


def _columns(conn, table):
    return {r["name"] for r in conn.execute("PRAGMA table_info(" + table + ")")}


def _compatible_schema(conn):
    # Canonical tables match the deployed receipt store. New columns are additive.
    execute_script(
        conn,
        """
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,status TEXT,document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS photo_measurements(id TEXT PRIMARY KEY,camera TEXT NOT NULL,burst_key TEXT,status TEXT NOT NULL,document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS experiment_records(id TEXT PRIMARY KEY,document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS binding_handoffs(id TEXT PRIMARY KEY,instrument_id TEXT,status TEXT,document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS instruments(id TEXT PRIMARY KEY,name TEXT,scene TEXT,model TEXT);
    """,
    )
    additions = {
        "jobs": {
            "lease_holder": "TEXT",
            "lease_expires_at": "TEXT",
            "retry_count": "INTEGER NOT NULL DEFAULT 0",
            "max_retries": "INTEGER NOT NULL DEFAULT 3",
            "lease_generation": "INTEGER NOT NULL DEFAULT 0",
            "next_attempt_at": "REAL NOT NULL DEFAULT 0",
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
            "camera_id": "TEXT",
            "priority": "INTEGER NOT NULL DEFAULT 10",
        },
        "photo_measurements": {
            "measurement_state": "TEXT NOT NULL DEFAULT 'draft'",
            "confirmed_at": "TEXT",
            "confirmed_by": "TEXT",
            "updated_at": "TEXT",
        },
        "experiment_records": {"created_at": "TEXT NOT NULL DEFAULT ''"},
        "binding_handoffs": {
            "binding_id": "TEXT",
            "offered_by": "TEXT",
            "offered_to": "TEXT",
            "offered_at": "TEXT",
            "state": "TEXT NOT NULL DEFAULT 'pending'",
            "accepted_at": "TEXT",
            "rejected_at": "TEXT",
            "rejection_reason": "TEXT",
        },
        "instruments": {
            "device_no": "TEXT",
            "measurement_ranges": "TEXT NOT NULL DEFAULT '{}'",
            "type_id": "TEXT",
        },
    }
    for table, cols in additions.items():
        existing = _columns(conn, table)
        for name, declaration in cols.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
    old = _columns(conn, "archive_outbox")
    if old and "seq" not in old:
        # Retain the incompatible prototype ledger; never manufacture receipt evidence.
        conn.execute("ALTER TABLE archive_outbox RENAME TO archive_outbox_prototype_backup")
    execute_script(
        conn,
        """
        CREATE TABLE IF NOT EXISTS archive_outbox(
            seq INTEGER PRIMARY KEY AUTOINCREMENT,entity TEXT NOT NULL,entity_id TEXT NOT NULL,
            document TEXT NOT NULL,recorded_at TEXT NOT NULL,archived_at TEXT,receipt_path TEXT,
            retry_after REAL NOT NULL DEFAULT 0,attempts INTEGER NOT NULL DEFAULT 0,last_error TEXT);
        CREATE TABLE IF NOT EXISTS camera_users(
            camera_id TEXT NOT NULL,
            user_id TEXT NOT NULL REFERENCES users(id),
            PRIMARY KEY (camera_id,user_id));
        CREATE INDEX IF NOT EXISTS idx_camera_users_user ON camera_users(user_id);
        CREATE TABLE IF NOT EXISTS scene_visits(id TEXT PRIMARY KEY,camera TEXT,ended TEXT,document TEXT);
        CREATE TABLE IF NOT EXISTS camera_active_users(
            camera_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id));
        CREATE TABLE IF NOT EXISTS runtime_leases(name TEXT PRIMARY KEY,owner TEXT NOT NULL,expires_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS runtime_status(name TEXT PRIMARY KEY,document TEXT NOT NULL,updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS camera_schedule(camera_id TEXT PRIMARY KEY,last_claimed REAL NOT NULL);
    """,
    )
    # Existing duplicates need explicit reconciliation; a migration must never delete them.
    if conn.execute(
        "SELECT 1 FROM photo_measurements WHERE burst_key IS NOT NULL GROUP BY camera,burst_key HAVING count(*)>1"
    ).fetchone():
        raise ValueError("Duplicate burst identities exist; reconcile before migration")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS measurement_burst_identity ON photo_measurements(camera,burst_key)"
    )


MIGRATIONS.append(
    {
        "version": 9,
        "name": "repair_runtime_contracts",
        "description": "Compatible schemas, fencing and camera ownership",
        "up": "SELECT 1;",
        "down": None,
    }
)


MIGRATIONS.append(
    {
        "version": 10,
        "name": "shared_capture_runtime_tables",
        "description": "API-first boot and persisted queue recovery",
        "up": """
    CREATE TABLE IF NOT EXISTS automation_settings(camera TEXT PRIMARY KEY,document TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS queued_camera_jobs ON jobs(camera_id,status,next_attempt_at);
    UPDATE jobs SET camera_id=json_extract(document,'$.camera_id') WHERE camera_id IS NULL;
    UPDATE jobs SET status='queued',document=json_set(document,'$.status','queued','$.phase','restart_recovery')
      WHERE status='interrupted' AND json_extract(document,'$.resume_pending')=1
      AND coalesce(json_extract(document,'$.request_trigger'),'')<>'video_stream';
""",
        "down": None,
    }
)


MIGRATIONS.append(
    {
        "version": 11,
        "name": "register_stirrer_panel_assets",
        "description": "Register the class-0 temperature/speed panel as a stirrer with explicit units",
        "up": """
    INSERT OR IGNORE INTO instrument_types(id,name,measurements_json,created_at,updated_at)
    VALUES('builtin-stirrer-v1','搅拌器',
      '[{"name":"温度","unit":"°C","range_min":null,"range_max":null,"precision":0.1,"display_format":".1f"},
        {"name":"转速","unit":"rpm","range_min":null,"range_max":null,"precision":1,"display_format":".0f"}]',
      datetime('now'),datetime('now'));
    INSERT OR IGNORE INTO instrument_types(id,name,measurements_json,created_at,updated_at)
    VALUES('builtin-balance-v1','质量测量仪',
      '[{"name":"质量","unit":"g","range_min":null,"range_max":null,"precision":0.001,"display_format":".3f"}]',
      datetime('now'),datetime('now'));

    -- Historical installation-specific instrument updates are intentionally excluded.

    -- Historical installation-specific instrument updates are intentionally excluded.
""",
        "down": None,
    }
)


MIGRATIONS.append(
    {
        "version": 12,
        "name": "allow_multiple_camera_users",
        "description": "Keep camera personnel memberships in a composite-key table",
        "up": """
    CREATE TABLE IF NOT EXISTS scene_visits(id TEXT PRIMARY KEY,camera TEXT,ended TEXT,document TEXT);
    CREATE TABLE IF NOT EXISTS camera_users_v12(
        camera_id TEXT NOT NULL,
        user_id TEXT NOT NULL REFERENCES users(id),
        PRIMARY KEY (camera_id,user_id));
    INSERT OR IGNORE INTO camera_users_v12(camera_id,user_id)
        SELECT camera_id,user_id FROM camera_users;
    CREATE TABLE IF NOT EXISTS camera_active_users(
        camera_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id));
    INSERT OR IGNORE INTO camera_active_users(camera_id,user_id)
        SELECT camera_id,min(user_id) FROM camera_users GROUP BY camera_id HAVING count(*)=1;
    DROP TABLE camera_users;
    ALTER TABLE camera_users_v12 RENAME TO camera_users;
    CREATE INDEX IF NOT EXISTS idx_camera_users_user ON camera_users(user_id);
    CREATE TRIGGER IF NOT EXISTS camera_member_default_active
    AFTER INSERT ON camera_users WHEN NOT EXISTS (
        SELECT 1 FROM camera_active_users WHERE camera_id=NEW.camera_id)
    BEGIN
        INSERT INTO camera_active_users(camera_id,user_id) VALUES(NEW.camera_id,NEW.user_id);
    END;
""",
        "down": None,
    }
)


MIGRATIONS.append(
    {
        "version": 13,
        "name": "classify_devices_and_reserve_qr_identities",
        "description": "Add explicit device categories and a durable QR identity registry",
        "up": """
    CREATE TABLE IF NOT EXISTS qr_device_registry(
        qr_hash TEXT PRIMARY KEY,
        device_id TEXT NOT NULL,
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        FOREIGN KEY(device_id) REFERENCES instruments(id)
    );
    CREATE INDEX IF NOT EXISTS idx_qr_device_registry_device ON qr_device_registry(device_id);
    INSERT OR IGNORE INTO qr_device_registry(qr_hash,device_id,first_seen_at,last_seen_at)
        SELECT json_extract(document,'$.qr_hash'), json_extract(document,'$.instrument.id'),
               json_extract(document,'$.started_at'), json_extract(document,'$.started_at')
          FROM bindings
         WHERE json_extract(document,'$.qr_hash') IS NOT NULL
           AND json_extract(document,'$.instrument.id') IS NOT NULL;
""",
        "down": None,
    }
)


# Shared with the isolated, non-managed application bootstrap. No historical
# binding/job documents are rewritten by this migration.
SESSION_SCHEMA = """
    DROP TRIGGER IF EXISTS camera_member_default_active;
    CREATE TABLE IF NOT EXISTS camera_session_state(
        camera_id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS binding_change_requests(
        request_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL, document TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS binding_session_audit(
        id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, occurred_at TEXT NOT NULL,
        relation_type TEXT NOT NULL, previous_id TEXT NOT NULL, next_id TEXT NOT NULL,
        document TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS binding_session_audit_camera ON binding_session_audit(camera_id,occurred_at);
    CREATE TRIGGER IF NOT EXISTS immutable_binding_session_audit_update
    BEFORE UPDATE ON binding_session_audit
    BEGIN SELECT RAISE(ABORT,'Session audit is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_binding_session_audit_delete
    BEFORE DELETE ON binding_session_audit
    BEGIN SELECT RAISE(ABORT,'Session audit is immutable'); END;
"""
MIGRATIONS.append(
    {
        "version": 14,
        "name": "immutable_personnel_sessions",
        "description": "Explicit active user selection, transactional sessions and retry receipts",
        "up": SESSION_SCHEMA,
        "down": None,
    }
)


def get_migration_status(db):
    applied = {}
    if db.path.is_file():
        conn = sqlite3.connect(db.path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='schema_migrations'"
            ).fetchone():
                applied = {r[0]: r for r in conn.execute("SELECT * FROM schema_migrations")}
        finally:
            conn.close()
    return {
        "current_version": max(applied, default=0),
        "latest_version": max(m["version"] for m in MIGRATIONS),
        "applied_count": len(applied),
        "total_count": len(MIGRATIONS),
        "pending": [
            {k: m.get(k) for k in ("version", "name", "description")}
            for m in MIGRATIONS
            if m["version"] not in applied
        ],
    }


def apply_migrations(db, dry_run=False):
    pending = get_migration_status(db)["pending"]
    if dry_run:
        return [m | {"status": "pending"} for m in pending]
    db._init_db()
    results = []
    with db.transaction("IMMEDIATE") as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY,name TEXT NOT NULL,applied_at TEXT NOT NULL,duration_ms INTEGER)"
        )
        applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        if 9 not in applied:
            _compatible_schema(conn)
        if 13 not in applied and "device_category" not in _columns(conn, "instruments"):
            conn.execute(
                "ALTER TABLE instruments ADD COLUMN device_category TEXT NOT NULL DEFAULT 'instrument'"
            )
        for migration in MIGRATIONS:
            if migration["version"] in applied:
                continue
            start = time.monotonic()
            execute_script(conn, migration["up"])
            duration = int((time.monotonic() - start) * 1000)
            conn.execute(
                "INSERT INTO schema_migrations VALUES(?,?,?,?)",
                (
                    migration["version"],
                    migration["name"],
                    datetime.now(timezone.utc).isoformat(),
                    duration,
                ),
            )
            results.append(
                {
                    "version": migration["version"],
                    "name": migration["name"],
                    "status": "applied",
                    "duration_ms": duration,
                }
            )
    return results
