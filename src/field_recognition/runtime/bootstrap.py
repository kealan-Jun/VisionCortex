"""Database checks and the legacy demonstration schema, executed by a factory."""

import json
import uuid
from pathlib import Path

from field_recognition.runtime.database import get_migration_status


def _ensure_instrument_columns(conn):
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(instruments)")}
    for column, declaration in [
        ("device_no", "TEXT"),
        ("measurement_ranges", "TEXT NOT NULL DEFAULT '{}'"),
        ("type_id", "TEXT"),
        ("device_category", "TEXT NOT NULL DEFAULT 'instrument'"),
    ]:
        if column not in columns:
            conn.execute(f"ALTER TABLE instruments ADD COLUMN {column} {declaration}")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Registry contains duplicate JSON keys")
        result[key] = value
    return result


def _invalid_json_number(value):
    raise ValueError("Registry JSON must contain finite numbers")


def _read_registry(path, collection, schema):
    try:
        document = json.loads(
            Path(path).expanduser().read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_invalid_json_number,
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"Invalid {collection} registry file; expected UTF-8 JSON") from None
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise ValueError(f"Invalid {collection} registry schema")
    items = document.get(collection)
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"Registry {collection} must be a list of objects")
    return items


def _registry_id(value):
    try:
        if not isinstance(value, str):
            raise ValueError
        return str(uuid.UUID(value))
    except ValueError:
        raise ValueError("Registry identity must be a UUID string") from None


def _registry_text(item, key, maximum, *, default=None, required=False):
    value = item.get(key)
    if value is None and not required:
        return default
    if not isinstance(value, str) or len(value) > maximum or (required and not value.strip()):
        raise ValueError(f"Invalid registry {key}")
    return value.strip()


def _validate_qr_metadata(item, kind, identifier):
    """Validate declared payload consistency without treating it as decoded evidence."""
    payloads = []
    if "qr_payload" in item:
        payloads.append(item["qr_payload"])
    if "encoded_text" in item:
        try:
            if not isinstance(item["encoded_text"], str):
                raise ValueError
            payloads.append(json.loads(item["encoded_text"], object_pairs_hook=_unique_json_object))
        except (TypeError, ValueError):
            raise ValueError("Invalid registry encoded_text") from None
    for payload in payloads:
        if (
            not isinstance(payload, dict)
            or type(payload.get("v")) is not int
            or payload.get("v") != 1
            or payload.get("type") != kind
            or _registry_id(payload.get("id")) != identifier
        ):
            raise ValueError("Registry QR metadata must match its declared identity and type")


def _instrument_rows(items):
    from field_recognition.api.models import MeasurementRange

    rows, identifiers = [], set()
    for item in items:
        identifier = _registry_id(item.get("instrument_id"))
        if identifier in identifiers:
            raise ValueError("Duplicate instrument identity in registry")
        identifiers.add(identifier)
        _validate_qr_metadata(item, "instrument", identifier)
        ranges = item.get("measurement_ranges")
        if ranges is None:
            ranges = {}
        if not isinstance(ranges, dict):
            raise ValueError("Registry measurement_ranges must be an object")
        for name, spec in ranges.items():
            if (
                not name.strip()
                or len(name) > 80
                or not isinstance(spec, dict)
                or set(spec) - MeasurementRange.model_fields.keys()
            ):
                raise ValueError("Invalid registry measurement_ranges fields")
            try:
                MeasurementRange.model_validate_json(json.dumps(spec, allow_nan=False), strict=True)
            except ValueError:
                raise ValueError("Invalid registry measurement range") from None
        rows.append(
            (
                identifier,
                _registry_text(item, "label", 80, required=True),
                _registry_text(item, "scene", 100, default=""),
                _registry_text(item, "model", 100, default=""),
                _registry_text(item, "device_no", 100),
                json.dumps(ranges, ensure_ascii=False, allow_nan=False),
                _registry_text(item, "type_id", 100),
                _registry_text(item, "device_category", 50, default="instrument", required=False)
                or "instrument",
            )
        )
    return rows


def _scene_rows(items):
    rows, identifiers = [], set()
    for item in items:
        identifier = _registry_id(item.get("scene_id"))
        if identifier in identifiers:
            raise ValueError("Duplicate scene identity in registry")
        identifiers.add(identifier)
        _validate_qr_metadata(item, "scene", identifier)
        name = _registry_text(item, "scene_name", 100)
        if name:
            rows.append((identifier, name))
    return rows


def import_registry(database, instrument_registry=None, scene_registry=None):
    """Validate complete UTF-8 registries, then add new identities in one transaction.

    At least one path is required. Existing identities, labels and evidence are
    never replaced. Registration and declared QR metadata do not create scans,
    decoded matches, bindings or physical identity verification.
    """
    if instrument_registry is None and scene_registry is None:
        raise ValueError("At least one instrument or scene registry path is required")
    instruments = (
        _instrument_rows(
            _read_registry(instrument_registry, "instruments", "FieldRecognitionQrLabels/1")
        )
        if instrument_registry is not None
        else []
    )
    scenes = (
        _scene_rows(_read_registry(scene_registry, "scenes", "FieldRecognitionSceneQrLabels/1"))
        if scene_registry is not None
        else []
    )
    with database.transaction("IMMEDIATE") as conn:
        if instrument_registry is not None:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS instruments(id TEXT PRIMARY KEY, name TEXT, scene TEXT, model TEXT)"
            )
            _ensure_instrument_columns(conn)
        if scene_registry is not None:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scenes(id TEXT PRIMARY KEY, name TEXT NOT NULL)"
            )
        for row in instruments:
            conn.execute(
                """INSERT OR IGNORE INTO instruments
                (id,name,scene,model,device_no,measurement_ranges,type_id,device_category)
                VALUES(?,?,?,?,?,?,?,?)""",
                row,
            )
        for row in scenes:
            conn.execute("INSERT OR IGNORE INTO scenes(id,name) VALUES(?,?)", row)


def import_sample_registry(database, project_root):
    """Explicitly add shipped QR samples without changing their legacy empty scene/model."""
    import_registry(
        database,
        instrument_registry=project_root / "InstrumentRegistry.json",
        scene_registry=project_root / "SceneRegistry.json",
    )


def initialize_database(runtime):
    database = runtime.database
    if runtime["RUNTIME_ENABLED"] or runtime["AUTH_ENABLED"]:
        if get_migration_status(database)["pending"]:
            raise RuntimeError("Pending database migrations; run migrate_db.py before starting")
    if runtime["AUTH_ENABLED"]:
        with database.connection() as conn:
            if not conn.execute("SELECT 1 FROM users WHERE role='admin' AND disabled=0").fetchone():
                raise RuntimeError("Create the initial administrator before starting")

    with database.connection() as conn:
        conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS instruments(id TEXT PRIMARY KEY, name TEXT, scene TEXT, model TEXT);
        CREATE TABLE IF NOT EXISTS scans(id TEXT PRIMARY KEY, document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS bindings(id TEXT PRIMARY KEY, camera TEXT, ended TEXT, document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, status TEXT, document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS camera_resets(camera TEXT PRIMARY KEY, confirmed_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS camera_service_state(camera TEXT PRIMARY KEY, document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fallback_attempts(binding_id TEXT PRIMARY KEY, attempted_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS qr_device_registry(
            qr_hash TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL);
        """)
        if not runtime["RUNTIME_ENABLED"] and not runtime["AUTH_ENABLED"]:
            from field_recognition.runtime.database import SESSION_SCHEMA, execute_script

            execute_script(conn, SESSION_SCHEMA)
        conn.execute("""INSERT OR IGNORE INTO qr_device_registry(qr_hash,device_id,first_seen_at,last_seen_at)
            SELECT json_extract(document,'$.qr_hash'), json_extract(document,'$.instrument.id'),
                   json_extract(document,'$.started_at'), json_extract(document,'$.started_at')
              FROM bindings
             WHERE json_extract(document,'$.qr_hash') IS NOT NULL
               AND json_extract(document,'$.instrument.id') IS NOT NULL""")
        _ensure_instrument_columns(conn)
    if runtime.config.use_sample_registry:
        import_sample_registry(database, runtime.project_root)
