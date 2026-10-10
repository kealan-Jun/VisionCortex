"""Conservative local observations for instance-owned shutdown tools."""

from pathlib import Path
import sqlite3
import time

from .sqlite_store import connection


def snapshot(settings, *, web_runs, gpu_busy=False):
    active = sum(run.get("state") not in {"completed", "partial", "failed", "interrupted"}
                 for run in web_runs)
    active += int(gpu_busy)
    background_enabled = bool((settings.get("device_day") or {}).get("enabled"))
    # A dispatcher may admit work after an empty-queue observation. Its enabled
    # state therefore blocks automatic shutdown even when all queues are empty.
    active += int(background_enabled)
    root = Path(settings["storage"]["local_runtime_root"])
    try:
        for stage in ("retention", "vision", "stt", "understanding", "report"):
            path = root / "device-day" / f"queue-{stage}.sqlite3"
            if path.is_file():
                with connection(path, readonly=True, timeout=.5) as db:
                    active += db.execute("SELECT COUNT(*) FROM recordings WHERE status IN ('running','queued')").fetchone()[0]
        resources = root / "state" / "resources.sqlite3"
        if resources.is_file():
            with connection(resources, readonly=True, timeout=.5) as db:
                active += db.execute("SELECT COUNT(*) FROM leases WHERE expires>? AND state IN ('running','waiting')",
                                     (time.time(),)).fetchone()[0]
    except (OSError, sqlite3.Error, ValueError):
        return {"provable": False, "active_tasks": None, "reason": "local_queue_unavailable"}
    return {"provable": True, "active_tasks": active,
            "background_admission_enabled": background_enabled}
