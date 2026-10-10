"""Shared, local-only execution controls for every analysis entry point."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
import os
import sqlite3
import threading
import time
import uuid

from .sqlite_store import connection


class ExecutionCancelled(RuntimeError):
    pass


class ExecutionYielded(ExecutionCancelled):
    """Background work released a resumable unit for live processing."""
    def __init__(self, message, *, reason='live_demand'):
        super().__init__(message)
        self.reason = reason


class CancellationSignal:
    """Local cancellation also observes its parent, without cancelling siblings."""
    def __init__(self, parent=None):
        self.parent = parent
        self.local = threading.Event()

    def is_set(self):
        return self.local.is_set() or (self.parent is not None and self.parent.is_set())

    def set(self):
        self.local.set()

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = .05 if deadline is None else min(.05, deadline - time.monotonic())
            if remaining <= 0:
                break
            self.local.wait(remaining)
        return self.is_set()


@dataclass(frozen=True)
class ExecutionContext:
    job_id: str = 'interactive'
    source: str = 'offline'
    priority: int = 1
    stop: threading.Event | CancellationSignal | None = None
    yield_signal: object | None = None
    background_fair: bool = False


CURRENT = ContextVar('visioncortex_execution', default=ExecutionContext())
_OWNED_RESOURCES = ContextVar('visioncortex_owned_resources', default=frozenset())


def check_cancelled(stop=None):
    stop = stop or CURRENT.get().stop
    if stop is not None and stop.is_set():
        raise ExecutionCancelled('Execution cancelled at a resumable boundary')


def check_yield(signal=None, *, persist=None, checkpoint=True):
    """Check only before work or after a complete, reusable unit is saved.

    Paid responses and their usage receipts must be persisted before callers
    check this signal. Quantum expiry only applies after a newly saved unit;
    non-incremental work still observes live demand and shutdown. Ordinary
    cancellation remains independently controlled.
    """
    signal = CURRENT.get().yield_signal if signal is None else signal
    if signal is None:
        return
    if callable(signal):
        requested = signal()
    elif checkpoint:
        requested = signal.is_set()
    else:
        try:
            requested = signal.is_set(ignore_quantum=True)
        except TypeError:
            # Ordinary Event/cancellation objects have no quantum protocol.
            requested = signal.is_set()
    if requested:
        if persist is not None:
            persist()
        raise ExecutionYielded('Background work paused for live processing',
                               reason=getattr(signal, 'reason', None) or 'live_demand')


def sync_files(*paths):
    """Seal existing unit files and their directory entries before yielding."""
    directories = set()
    for path in paths:
        path = Path(path)
        with path.open('rb') as handle:
            os.fsync(handle.fileno())
        directories.add(path.parent)
    if os.name != 'nt':
        for directory in directories:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


@contextmanager
def defer_yield():
    """Finish an already admitted paid unit before observing live arrivals."""
    token = CURRENT.set(replace(CURRENT.get(), yield_signal=None))
    try:
        yield
    finally:
        CURRENT.reset(token)


@contextmanager
def execution_context(**kwargs):
    kwargs.setdefault('yield_signal', CURRENT.get().yield_signal)
    kwargs.setdefault('background_fair', CURRENT.get().background_fair)
    token = CURRENT.set(ExecutionContext(**kwargs))
    try:
        check_cancelled()
        yield CURRENT.get()
    finally:
        CURRENT.reset(token)


def analysis_execution(function):
    @wraps(function)
    def run(self, manifest, *args, **kwargs):
        parent = CURRENT.get()
        with execution_context(job_id=manifest.experiment_id, source=parent.source,
                               priority=parent.priority, stop=parent.stop,
                               yield_signal=parent.yield_signal):
            return function(self, manifest, *args, **kwargs)
    return run


class ResourceCoordinator:
    """Cross-process leases, FIFO with priority aging, and bounded admission.

    No model or media access. All participants must share the local runtime root.
    Leases are renewed while an operation runs and expire after process death.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS resources(name TEXT PRIMARY KEY, capacity INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS leases(
                    id TEXT PRIMARY KEY, resource TEXT NOT NULL, job TEXT NOT NULL,
                    source TEXT NOT NULL, units INTEGER NOT NULL, priority INTEGER NOT NULL,
                    created REAL NOT NULL, expires REAL NOT NULL, state TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS resource_waiters ON leases(resource,state,created);''')

    def try_claim(self, identifier, resource, units, capacity, context, *, now=None, queued_at=None):
        now = time.time() if now is None else now
        if context.source == 'device_day_backfill' and not context.background_fair:
            return self._try_claim_idle(identifier, resource, units, capacity, context, now)
        # A live waiting request need not obtain the SQLite writer lock merely
        # to learn that all capacity is still occupied. Renew it only near its
        # expiry; admission itself always rechecks under BEGIN IMMEDIATE below.
        with connection(self.path, readonly=True) as db:
            own = db.execute('SELECT expires,state FROM leases WHERE id=?', (identifier,)).fetchone()
            configured = db.execute('SELECT capacity FROM resources WHERE name=?', (resource,)).fetchone()
            if own and own['state'] == 'waiting' and own['expires'] > now + 30 and configured and configured[0] == capacity:
                used = db.execute("SELECT COALESCE(SUM(units),0) FROM leases WHERE resource=? AND state='running' AND expires>?",
                                  (resource, now)).fetchone()[0]
                first = db.execute("""SELECT id FROM leases WHERE resource=? AND state='waiting' AND expires>?
                    ORDER BY priority-CAST((?-created)/30 AS INTEGER),created,id LIMIT 1""", (resource, now, now)).fetchone()
                if used + units > capacity or (first and first[0] != identifier):
                    return False
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM leases WHERE expires<=?', (now,))
            db.execute('INSERT OR IGNORE INTO resources VALUES(?,?)', (resource, capacity))
            previous = db.execute('SELECT capacity FROM resources WHERE name=?', (resource,)).fetchone()[0]
            if previous != capacity:
                if db.execute('SELECT 1 FROM leases WHERE resource=? LIMIT 1', (resource,)).fetchone():
                    raise ValueError(f'Conflicting live resource capacity: {resource}')
                db.execute('UPDATE resources SET capacity=? WHERE name=?', (capacity, resource))
            db.execute('INSERT OR IGNORE INTO leases VALUES(?,?,?,?,?,?,?,?,?)',
                       (identifier, resource, context.job_id, context.source, units,
                        context.priority, min(now, queued_at) if queued_at is not None else now, now+90, 'waiting'))
            db.execute('UPDATE leases SET expires=? WHERE id=?', (now+90, identifier))
            first = db.execute('''SELECT id FROM leases WHERE resource=? AND state='waiting'
                ORDER BY priority-CAST((?-created)/30 AS INTEGER),created,id LIMIT 1''', (resource, now)).fetchone()
            used = db.execute("SELECT COALESCE(SUM(units),0) FROM leases WHERE resource=? AND state='running'", (resource,)).fetchone()[0]
            if first and first[0] == identifier and used+units <= capacity:
                db.execute("UPDATE leases SET state='running' WHERE id=?", (identifier,))
                return True
            return False

    def _try_claim_idle(self, identifier, resource, units, capacity, context, now):
        """A backfill lease never joins or ages ahead of ordinary waiters."""
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM leases WHERE expires<=?', (now,))
            db.execute('INSERT OR IGNORE INTO resources VALUES(?,?)', (resource, capacity))
            previous = db.execute('SELECT capacity FROM resources WHERE name=?', (resource,)).fetchone()[0]
            if previous != capacity:
                if db.execute('SELECT 1 FROM leases WHERE resource=? LIMIT 1', (resource,)).fetchone():
                    raise ValueError(f'Conflicting live resource capacity: {resource}')
                db.execute('UPDATE resources SET capacity=? WHERE name=?', (capacity, resource))
            own = db.execute('SELECT state FROM leases WHERE id=?', (identifier,)).fetchone()
            if own and own['state'] == 'running':
                db.execute('UPDATE leases SET expires=? WHERE id=?', (now+90, identifier))
                return True
            if db.execute("SELECT 1 FROM leases WHERE resource=? AND state='waiting' LIMIT 1",
                          (resource,)).fetchone():
                return False
            used = db.execute("SELECT COALESCE(SUM(units),0) FROM leases WHERE resource=? AND state='running'",
                              (resource,)).fetchone()[0]
            available = capacity - 1 if capacity > 1 else capacity
            if used + units > available:
                return False
            db.execute('INSERT INTO leases VALUES(?,?,?,?,?,?,?,?,?)',
                       (identifier, resource, context.job_id, context.source, units,
                        context.priority, now, now+90, 'running'))
            return True

    def release(self, identifier):
        with connection(self.path) as db:
            db.execute('DELETE FROM leases WHERE id=?', (identifier,))

    @contextmanager
    def acquire(self, resource, *, capacity, units=1, timeout=300, context=None, queued_at=None):
        if not 1 <= units <= capacity:
            raise ValueError('Resource request must fit its capacity')
        context = context or CURRENT.get()
        identifier, deadline = uuid.uuid4().hex, time.monotonic()+timeout
        finished, lost = threading.Event(), threading.Event()
        renewer = None
        def renew():
            while not finished.wait(15):
                try:
                    with connection(self.path) as db:
                        updated = db.execute('UPDATE leases SET expires=? WHERE id=?',
                                             (time.time()+90, identifier)).rowcount
                    if not updated:
                        lost.set()
                        return
                except Exception:
                    lost.set()
                    return
        try:
            while True:
                check_cancelled(context.stop)
                check_yield(context.yield_signal, checkpoint=False)
                try:
                    admitted = self.try_claim(identifier, resource, units, capacity, context, queued_at=queued_at)
                except sqlite3.OperationalError as exc:
                    if getattr(exc, 'sqlite_errorcode', None) not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                        raise
                    admitted = False
                if admitted:
                    break
                if context.source == 'device_day_backfill' and not context.background_fair:
                    raise ExecutionYielded(f'Background resource is busy: {resource}', reason='resource_busy')
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Resource admission timed out: {resource}')
                if context.stop is not None:
                    context.stop.wait(.25)
                else:
                    time.sleep(.25)
            renewer = threading.Thread(target=renew, daemon=True, name='resource-lease')
            renewer.start()
            yield
            if lost.is_set():
                raise RuntimeError('Resource lease was lost; output cannot be marked complete')
        finally:
            finished.set()
            if renewer:
                renewer.join(timeout=2)
            self.release(identifier)

    def snapshot(self):
        with connection(self.path, readonly=True) as db:
            return [dict(row) for row in db.execute(
                'SELECT resource,job,source,units,state,created FROM leases WHERE expires>? ORDER BY created', (time.time(),))]


@contextmanager
def resource_slot(config, resource, *, units=1):
    check_cancelled()
    check_yield(checkpoint=False)
    settings = config.get('runtime') or {}
    limits = settings.get('resource_limits') or {}
    capacity = limits.get(resource)
    if capacity is None:
        yield
        return
    root = Path(config['storage']['local_runtime_root'])
    if not root.is_absolute() or str(root).startswith(('//', '\\\\')):
        raise ValueError('Resource coordinator requires an absolute local runtime root')
    owned = _OWNED_RESOURCES.get()
    key = (threading.get_ident(), str(root.resolve()), resource)
    if key in owned:
        # Only same-thread nested provider calls reuse an admitted unit. A
        # copied thread context still requires its own cross-process lease.
        yield
        return
    coordinator = ResourceCoordinator(root/'state'/'resources.sqlite3')
    with coordinator.acquire(resource, capacity=capacity, units=min(units, capacity),
                             timeout=settings.get('admission_timeout_seconds', 300)):
        token = _OWNED_RESOURCES.set(owned | {key} if CURRENT.get().source == 'device_day_backfill'
                                    or CURRENT.get().yield_signal is not None else owned)
        try:
            yield
        finally:
            _OWNED_RESOURCES.reset(token)
