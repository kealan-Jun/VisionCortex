"""Main-writer progress, never a timer that masks a blocked NAS operation."""

from contextlib import contextmanager
from contextvars import ContextVar
import time

_observer = ContextVar("archive_progress", default=None)
_checkpoint = ContextVar("archive_checkpoint", default=None)


@contextmanager
def checkpoint(callback, interval=1):
    """Yield maintenance at safe file-operation boundaries to new receipts."""
    token = _checkpoint.set({"callback": callback, "interval": interval, "last": time.monotonic()})
    try:
        yield
    finally:
        _checkpoint.reset(token)


@contextmanager
def observe(callback, interval=5):
    state = {"callback": callback, "interval": interval, "last": time.monotonic(), "details": {}}
    token = _observer.set(state)
    try:
        yield
    finally:
        _observer.reset(token)


def advance(phase=None, completed=None, total=None):
    guard = _checkpoint.get()
    if guard and time.monotonic() - guard["last"] >= guard["interval"]:
        guard["last"] = time.monotonic()
        guard["callback"]()
    state = _observer.get()
    if state is None:
        return
    if phase is not None:
        state["details"] = {
            "work_phase": phase,
            "progress_current": completed,
            "progress_total": total,
        }
    now = time.monotonic()
    if now - state["last"] >= state["interval"]:
        state["callback"](dict(state["details"]))
        state["last"] = now
