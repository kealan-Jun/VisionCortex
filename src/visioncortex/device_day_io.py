"""Cooperative cross-process NAS admission; no media access on import."""
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
import time

from .device_day_activity import current_stage, phase
from .runtime_control import CURRENT, ResourceCoordinator


_COPY_QUEUED_AT = ContextVar('device_day_copy_queued_at', default=None)


@contextmanager
def slot(config, *, copy=False, urgent=False, whole_copy=False):
    settings = config.get('device_day', {})
    coordinator = ResourceCoordinator(Path(config['storage']['local_runtime_root']) / 'state' / 'resources.sqlite3')
    capacity = settings.get('archive_copy_workers', 1) if whole_copy else settings.get('nas_io_slots', 3)
    context = replace(CURRENT.get(), priority=-10 if copy and urgent else 0)
    # Keep a copy's age across bounded blocks. Rejoining behind every long
    # decoder after each 8 MiB block can otherwise starve archival for hours.
    queued_at = (_COPY_QUEUED_AT.get() or time.time()) if copy else None
    with ExitStack() as stack:
        with phase('io_queue_seconds'):
            if not copy and not whole_copy and capacity > 1 and current_stage() != 'stt':
                # Keep one shared slot available for bounded archive work. A
                # long vision operation must not occupy every NAS I/O slot.
                stack.enter_context(coordinator.acquire(
                    'nas-io-read', capacity=capacity - 1,
                    timeout=settings.get('nas_io_timeout_seconds', 3600), context=context))
            # STT has its own bounded execution capacity and must not queue
            # behind whole-video decoding. It still shares the total I/O cap
            # with archival and every other reader below.
            stack.enter_context(coordinator.acquire(
                'nas-copy' if whole_copy else 'nas-io', capacity=capacity,
                timeout=settings.get('nas_io_timeout_seconds', 3600), context=context, queued_at=queued_at))
        token = _COPY_QUEUED_AT.set(queued_at) if whole_copy else None
        try:
            yield
        finally:
            if token is not None:
                _COPY_QUEUED_AT.reset(token)


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
