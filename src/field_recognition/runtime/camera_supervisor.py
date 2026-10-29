"""Supervise one capture process per enabled, assigned camera; share one OCR worker."""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from field_recognition.runtime.database import Database
from field_recognition.runtime.paths import PROJECT_ROOT
from field_recognition.runtime.worker_support import process_lock, heartbeat


def camera_environment(row, base):
    camera = row["camera_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", camera):
        raise ValueError("Invalid camera identifier")
    env = dict(
        base, FIELD_SERVICE_ROLE="capture", FIELD_CAMERA_ID=camera, FIELD_PRODUCTION_ENABLED="1"
    )
    for key, column in [
        ("FIELD_RECEIVER_URL", "receiver_url"),
        ("FIELD_SAVED_PHOTO_ROOT", "nas_photo_root"),
    ]:
        if row[column]:
            env[key] = row[column]
    # Identity comes from the persisted assignment, never another camera's environment.
    env["FIELD_WEARER_ID"] = row["user_id"]
    env["FIELD_OPERATOR_NAME"] = row["display_name"]
    return env


def stop_child(child):
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def capture_stalled(pid, row, progress, now, timeout=120):
    """Only a new heartbeat from this child extends its monotonic deadline."""
    if row is not None:
        document = json.loads(row["document"])
        if document.get("pid") == pid and row["updated_at"] != progress.get("stamp"):
            progress.update(stamp=row["updated_at"], seen=now)
    return now - progress["seen"] > timeout


def release_camera(db, camera, reason):
    from datetime import datetime, timezone
    from field_recognition.bindings.binding_policy import observe_camera_service

    with db.transaction("IMMEDIATE") as conn:
        observe_camera_service(
            conn,
            camera,
            {"online": False, "failure_reason": reason},
            datetime.now(timezone.utc).isoformat(),
        )


def run():
    data = Path(os.environ.get("FIELD_DEMO_DATA", "Data")).resolve()
    db = Database(os.environ.get("FIELD_DATABASE_PATH", str(data / "Demo.sqlite3")))
    stop = threading.Event()
    children = {}
    retry = {}
    progress = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with process_lock(data, "camera-supervisor"):
        try:
            while not stop.is_set():
                with db.connection() as conn:
                    rows = conn.execute("""SELECT c.*,u.id user_id,u.display_name FROM camera_registry c
                                         JOIN camera_active_users a USING(camera_id) JOIN users u ON u.id=a.user_id
                                         WHERE c.enabled=1 AND u.disabled=0""").fetchall()
                    statuses = {
                        r["name"]: dict(r)
                        for r in conn.execute(
                            "SELECT * FROM runtime_status WHERE name LIKE 'camera:%'"
                        )
                    }
                desired = {r["camera_id"]: dict(r) for r in rows}
                for key, (child, fingerprint) in list(children.items()):
                    if key not in desired or fingerprint != json.dumps(
                        desired[key], sort_keys=True
                    ):
                        stop_child(child)
                        del children[key]
                        release_camera(db, key, "capture_service_reconfigured")
                    elif child.poll() is not None:
                        del children[key]
                        retry[key] = time.monotonic() + 10
                        release_camera(db, key, "capture_service_exited")
                    elif capture_stalled(
                        child.pid, statuses.get("camera:" + key), progress[key], time.monotonic()
                    ):
                        stop_child(child)
                        del children[key]
                        retry[key] = time.monotonic() + 10
                        release_camera(db, key, "capture_service_unresponsive")
                for key, row in desired.items():
                    if key in children or time.monotonic() < retry.get(key, 0):
                        continue
                    try:
                        env = camera_environment(row, os.environ)
                        process = subprocess.Popen(
                            [sys.executable, "-m", "field_recognition.runtime.capture_worker"],
                            cwd=PROJECT_ROOT,
                            env=env,
                        )
                        children[key] = (process, json.dumps(row, sort_keys=True))
                        progress[key] = {"seen": time.monotonic()}
                    except (ValueError, OSError):
                        retry[key] = time.monotonic() + 30
                heartbeat(
                    db,
                    "camera-supervisor",
                    {"status": "running", "cameras": list(children), "registered": list(desired)},
                )
                stop.wait(2)
        finally:
            for key, (child, _) in children.items():
                stop_child(child)
                release_camera(db, key, "capture_service_stopped")
            heartbeat(db, "camera-supervisor", {"status": "stopped"})


if __name__ == "__main__":
    run()
