"""Process exclusion and persistent component status."""

import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def process_lock(data, name):
    path = Path(data) / "Runtime" / (name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another " + name + " worker is already running") from None
        yield


def heartbeat(db, name, status):
    with db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runtime_status VALUES(?,?,?)",
            (name, json.dumps(dict(status, pid=os.getpid())), time.time()),
        )
    address = os.environ.get("NOTIFY_SOCKET")
    notify_pid = int(os.environ.get("WATCHDOG_PID", str(os.getpid())))
    if address and notify_pid == os.getpid():
        # Only successful database progress proves that this main loop is alive.
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify:
                notify.connect("\0" + address[1:] if address.startswith("@") else address)
                notify.sendall(b"READY=1\nWATCHDOG=1")
        except OSError:
            pass


@contextmanager
def process_mutex(data, name):
    """Blocking transaction-level file mutex shared by API and capture processes."""
    import hashlib

    path = Path(data) / "Runtime" / ("Mutex-" + hashlib.sha256(name.encode()).hexdigest() + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
