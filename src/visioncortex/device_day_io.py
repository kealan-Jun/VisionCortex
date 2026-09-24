"""Cooperative cross-process NAS admission; no media access on import."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .device_day_activity import phase
from .runtime_control import CURRENT, ResourceCoordinator


@contextmanager
def slot(config, *, copy=False, urgent=False, whole_copy=False):
    settings = config.get('device_day', {})
    coordinator = ResourceCoordinator(Path(config['storage']['local_runtime_root']) / 'state' / 'resources.sqlite3')
    capacity = settings.get('archive_copy_workers', 1) if whole_copy else settings.get('nas_io_slots', 3)
    context = replace(CURRENT.get(), priority=(-10 if urgent else 10) if copy else 0)
    # ResourceCoordinator ages waiting jobs every 30 seconds. A waiting copy
    # eventually precedes new decoders, even under continuous live intake.
    with phase('io_queue_seconds'):
        manager = coordinator.acquire('nas-copy' if whole_copy else 'nas-io', capacity=capacity,
                                      timeout=settings.get('nas_io_timeout_seconds', 3600), context=context)
        manager.__enter__()
    try:
        yield
    finally:
        import sys
        manager.__exit__(*sys.exc_info())


def pace_copy(config, size):
    """Aggregate copy bandwidth reservation across all archive workers.

    Wait outside the shared I/O slot so a throttled copy cannot occupy a
    decoder's slot. This is a configurable cap, not a measured NAS SLA.
    """
    import time
    from .runtime_control import check_cancelled
    from .sqlite_store import connection
    rate = config.get('device_day', {}).get('archive_copy_bytes_per_second', 8 * 1024 * 1024)
    root = Path(config['storage']['local_runtime_root']) / 'state' / 'resources.sqlite3'
    with connection(root) as db:
        db.execute('CREATE TABLE IF NOT EXISTS archive_bandwidth(id INTEGER PRIMARY KEY, next_at REAL)')
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT next_at FROM archive_bandwidth WHERE id=1').fetchone()
        now = time.time()
        # A crashed worker can leave only its bounded block's reservation.
        at = max(now, min(row['next_at'], now + 10)) if row else now
        db.execute('INSERT OR REPLACE INTO archive_bandwidth VALUES(1,?)', (at + size / rate,))
    with phase('io_queue_seconds'):
        while (delay := at - time.time()) > 0:
            check_cancelled()
            time.sleep(min(delay, .1))
