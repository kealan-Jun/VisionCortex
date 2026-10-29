"""Prepare local personnel and cameras; login credentials are opt-in."""

import argparse
import json
import os
import secrets
import uuid
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit
from field_recognition.runtime.database import Database, apply_migrations, get_migration_status
from field_recognition.archive.archive_manager import ArchiveManager
from field_recognition.runtime.auth import AuthService
from field_recognition.bindings.camera_registry import CameraRegistry
from field_recognition.runtime.paths import PROJECT_ROOT


def resolve_runtime_paths(
    data=None, database_path=None, *, project_root=PROJECT_ROOT, environment=None
):
    """Resolve preparation and installation paths without changing the environment."""
    environment = os.environ if environment is None else environment
    root = Path(project_root).resolve()

    def absolute(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()

    configured_data = data if data is not None else environment.get("FIELD_DEMO_DATA")
    configured_database = database_path or environment.get("FIELD_DATABASE_PATH")
    if configured_database:
        database = absolute(configured_database)
        directory = absolute(configured_data) if configured_data else database.parent
    else:
        directory = absolute(configured_data or "Data")
        database = directory / "Demo.sqlite3"
    return directory, database


def configured_login_url(value=None, *, environment=None):
    """Describe the configured listener; never include credentials in its URL."""
    environment = os.environ if environment is None else environment
    value = value or environment.get("FIELD_LOGIN_URL")
    if not value:
        host = environment.get("FIELD_HOST", "127.0.0.1")
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        if ":" in host and not host.startswith("["):
            host = "[" + host + "]"
        port = int(environment.get("FIELD_PORT", "8188"))
        if not 1 <= port <= 65535:
            raise ValueError("FIELD_PORT must be between 1 and 65535")
        value = f"http://{host}:{port}/login"
    parsed = urlsplit(value)
    # Accessing .port rejects malformed or out-of-range explicit URL ports.
    parsed.port
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Login URL must be an http(s) URL without credentials, query or fragment")
    return value


def prepare(
    data,
    username,
    display_name,
    camera_id,
    receiver_url=None,
    photo_root=None,
    *,
    auth_enabled=False,
    database_path=None,
    login_url=None,
    registry_policy="none",
    instrument_registry=None,
    scene_registry=None,
    project_root=PROJECT_ROOT,
):
    if not username.strip() or not display_name.strip():
        raise ValueError("Personnel identifier and display name are required")
    if registry_policy not in {"none", "sample"}:
        raise ValueError("registry_policy must be none or sample")
    if registry_policy == "sample" and (
        instrument_registry is not None or scene_registry is not None
    ):
        raise ValueError("Custom registries cannot be combined with registry_policy=sample")
    root = Path(project_root).expanduser().resolve()
    registries = {}
    for name, value in (
        ("instrument_registry", instrument_registry),
        ("scene_registry", scene_registry),
    ):
        if value is not None:
            path = Path(value).expanduser()
            registries[name] = (path if path.is_absolute() else root / path).resolve()
    access_url = configured_login_url(login_url) if auth_enabled else None
    data, path = resolve_runtime_paths(data, database_path, project_root=root)
    data.mkdir(parents=True, exist_ok=True)
    db = Database(path)
    if db.path.exists() and get_migration_status(db)["pending"]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        ArchiveManager(db, data / "unused").backup_database(
            data / "Backups" / ("BeforeRuntime-" + stamp + ".sqlite3")
        )
    apply_migrations(db)
    if registry_policy == "sample":
        from field_recognition.runtime.bootstrap import import_sample_registry

        import_sample_registry(db, root)
    elif registries:
        from field_recognition.runtime.bootstrap import import_registry

        import_registry(db, **registries)
    credentials = None
    with db.connection() as conn:
        users = conn.execute("SELECT id FROM users WHERE disabled=0 AND role='admin'").fetchall()
    if not auth_enabled:
        # A stable wearer ID is still needed for capture ownership, even when
        # opening the workbench does not require an authenticated account.
        with db.transaction("IMMEDIATE") as conn:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            if row and (row["disabled"] or row["display_name"] != display_name):
                raise ValueError("Existing personnel registration requires explicit correction")
            if not row:
                conn.execute(
                    "INSERT INTO users VALUES(?,?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        username,
                        display_name,
                        "operator",
                        None,
                        datetime.now(timezone.utc).isoformat(),
                        0,
                    ),
                )
                row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            user = SimpleNamespace(**dict(row))
    elif not users:
        if (data / "Access" / "InitialAccess.json").exists():
            raise ValueError(
                "An existing credential delivery file must be resolved before creating an administrator"
            )
        secret = secrets.token_urlsafe(24)
        user = AuthService(db).create_initial_admin(username, display_name, secret)
        credentials = {
            "username": username,
            "password": secret,
            "login_url": access_url,
        }
    else:
        with db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username=? AND disabled=0", (username,)
            ).fetchone()
        if not row:
            raise ValueError(
                "Existing users require an explicit, enabled account for camera assignment"
            )
        user = SimpleNamespace(**dict(row))
    registry = CameraRegistry(db)
    if camera_id and not registry.get_camera(camera_id):
        registry.register_camera(
            camera_id, camera_id, receiver_url=receiver_url, nas_photo_root=photo_root
        )
    if camera_id:
        with db.transaction("IMMEDIATE") as conn:
            from field_recognition.bindings.binding_operator import activate_member

            activate_member(conn, camera_id, user.id, datetime.now(timezone.utc).isoformat())
    if credentials:
        access = data / "Access"
        access.mkdir(mode=0o700, exist_ok=True)
        os.chmod(access, 0o700)
        path = access / "InitialAccess.json"
        with open(path, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as f:
            json.dump(credentials, f, ensure_ascii=False, indent=2)
        print("Initial credentials saved privately to " + str(path))
    return {
        "database": str(db.path),
        "camera_id": camera_id,
        "user_id": user.id,
        "migrations": get_migration_status(db)["current_version"],
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data", help="Local data directory (otherwise FIELD_DEMO_DATA or database parent)"
    )
    p.add_argument(
        "--database-path", help="SQLite path (otherwise FIELD_DATABASE_PATH or data/Demo.sqlite3)"
    )
    p.add_argument(
        "--login-url", help="Public login URL written only to the private initial-access file"
    )
    p.add_argument("--registry-policy", choices=("none", "sample"), default="none")
    p.add_argument(
        "--instrument-registry",
        help="Explicit custom instrument JSON, relative to the project root",
    )
    p.add_argument(
        "--scene-registry", help="Explicit custom scene JSON, relative to the project root"
    )
    p.add_argument(
        "--env-file",
        help="Explicit configuration file; existing environment variables take precedence",
    )
    p.add_argument("--username", required=True)
    p.add_argument("--display-name", required=True)
    p.add_argument("--camera")
    p.add_argument("--receiver-url")
    p.add_argument("--photo-root")
    p.add_argument(
        "--enable-login",
        action="store_true",
        default=os.environ.get("FIELD_AUTH_ENABLED", "0") == "1",
    )
    args = p.parse_args(argv)
    if args.env_file:
        from field_recognition.runtime.environment import load_environment_file

        load_environment_file(args.env_file)
    print(
        json.dumps(
            prepare(
                args.data,
                args.username,
                args.display_name,
                args.camera or os.environ.get("FIELD_CAMERA_ID"),
                args.receiver_url
                if args.receiver_url is not None
                else os.environ.get("FIELD_RECEIVER_URL"),
                args.photo_root
                if args.photo_root is not None
                else os.environ.get("FIELD_SAVED_PHOTO_ROOT"),
                auth_enabled=args.enable_login or os.environ.get("FIELD_AUTH_ENABLED", "0") == "1",
                database_path=args.database_path,
                login_url=args.login_url,
                registry_policy=args.registry_policy,
                instrument_registry=args.instrument_registry,
                scene_registry=args.scene_registry,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
