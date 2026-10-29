"""Cooperative cross-process NAS admission; no media access on import."""
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import replace
import os
from pathlib import Path
import threading
import time

from .device_day_activity import current_stage, phase
from .runtime_control import CURRENT, ResourceCoordinator


_COPY_QUEUED_AT = ContextVar('device_day_copy_queued_at', default=None)
_LIVE_VISION = ContextVar('device_day_live_vision_lane', default=False)
_OWNED_IO = ContextVar('device_day_owned_io', default=frozenset())


@contextmanager
def _io_resource(coordinator, path, resource, *, capacity, context, timeout, queued_at=None):
    # Publication composes independently guarded source/hash/export services.
    # A nested reader must use the outer job's unit, including its reader
    # reservation, instead of waiting forever for a second unit of that pool.
    # Keep ordinary unscoped CLI reservations independent, as resource_slot does.
    import asyncio
    from .runtime_control import check_cancelled, check_yield
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    key = (os.getpid(), threading.get_ident(), id(task), str(path.resolve()), context.job_id, resource)
    owned = _OWNED_IO.get()
    check_cancelled(context.stop)
    check_yield(context.yield_signal, checkpoint=False)
    if key in owned:
        yield
        return
    with coordinator.acquire(resource, capacity=capacity, timeout=timeout, context=context, queued_at=queued_at):
        token = _OWNED_IO.set(owned | {key} if context.source == 'device_day_backfill'
                              or context.yield_signal is not None else owned)
        try:
            yield
        finally:
            _OWNED_IO.reset(token)


@contextmanager
def live_vision_lane():
    """Bounded live decoders alongside the existing shared I/O budget.

    Only the independent continuous worker opts in. All processes share its
    configured lease cap; archive/STT work retains the ordinary total budget.
    """
    token = _LIVE_VISION.set(True)
    try:
        yield
    finally:
        _LIVE_VISION.reset(token)


def live_vision_capacity(config):
    settings = config.get('device_day') or {}
    requested = settings.get('live_vision_jobs', 1)
    if isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= 4:
        raise ValueError('device_day.live_vision_jobs must be an integer in [1,4]')
    capacity = ((config.get('runtime') or {}).get('resource_limits') or {}).get('vision', 12)
    limit = settings.get('stage_worker_limits', {}).get('vision', requested)
    return min(requested, capacity, limit)


@contextmanager
def slot(config, *, copy=False, urgent=False, whole_copy=False):
    settings = config.get('device_day', {})
    path = Path(config['storage']['local_runtime_root']) / 'state' / 'resources.sqlite3'
    coordinator = ResourceCoordinator(path)
    capacity = settings.get('archive_copy_workers', 1) if whole_copy else settings.get('nas_io_slots', 3)
    context = replace(CURRENT.get(), priority=-10 if copy and urgent else 0)
    # Keep a copy's age across bounded blocks. Rejoining behind every long
    # decoder after each 8 MiB block can otherwise starve archival for hours.
    queued_at = (_COPY_QUEUED_AT.get() or time.time()) if copy else None
    live = _LIVE_VISION.get() and current_stage() == 'vision' and not copy and not whole_copy
    with ExitStack() as stack:
        with phase('io_queue_seconds'):
            if not live and not copy and not whole_copy and capacity > 1 and current_stage() != 'stt':
                # Keep one shared slot available for bounded archive work. A
                # long vision operation must not occupy every NAS I/O slot.
                stack.enter_context(_io_resource(coordinator, path,
                    'nas-io-read', capacity=capacity - 1,
                    timeout=settings.get('nas_io_timeout_seconds', 3600), context=context))
            # STT has its own bounded execution capacity and must not queue
            # behind whole-video decoding. It still shares the total I/O cap
            # with archival and every other reader below.
            stack.enter_context(_io_resource(coordinator, path,
                'nas-live-vision' if live else 'nas-copy' if whole_copy else 'nas-io',
                capacity=live_vision_capacity(config) if live else capacity,
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
