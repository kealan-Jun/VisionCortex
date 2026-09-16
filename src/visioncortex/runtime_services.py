"""Independent operational workers. No model startup or media reads."""

from contextlib import contextmanager
import logging
import multiprocessing
import os
from pathlib import Path
import threading
import time

from .device_day_contract import atomic_json


def _publish_progress(config, control, interval):
    """Only local read models; never share the inference process's Python locks."""
    from .device_day_progress import snapshot
    path = Path(config['storage']['local_runtime_root']) / 'device-day' / 'ProgressSnapshot.json'
    try:
        while not control.poll():
            try:
                started = time.monotonic()
                value = snapshot(config)
                value.update(snapshot_producer_pid=os.getpid(),
                             snapshot_collection_seconds=time.monotonic()-started)
                atomic_json(path, value)
            except Exception as exc:
                logging.getLogger(__name__).warning('Progress publication unavailable: %s', type(exc).__name__)
            if control.poll(interval):
                break
    finally:
        control.close()


class ProgressPublisher:
    """Supervised spawn process: no fork of a loaded CUDA runtime."""
    def __init__(self, config, *, interval=5):
        self.config = config
        self.interval = interval
        self.context = multiprocessing.get_context('spawn')
        # A killed child can strand a multiprocessing.Event's semaphore. Give
        # each generation a fresh pipe, with no shared synchronization locks.
        self._lock = threading.Lock()
        self._closed = False
        self._control = None
        self.process = None

    def ensure(self):
        with self._lock:
            if self._closed or self.process is not None and self.process.is_alive():
                return
            if self.process is not None:
                self.process.join(timeout=0)
                self.process.close()
                self.process = None
            if self._control is not None:
                self._control.close()
            reader, self._control = self.context.Pipe(duplex=False)
            process = self.context.Process(target=_publish_progress,
                args=(self.config, reader, self.interval), name='progress-publisher', daemon=True)
            try:
                process.start()
                self.process = process
            except BaseException:
                self._control.close()
                self._control = None
                process.close()
                raise
            finally:
                reader.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._control is not None:
                try:
                    self._control.send_bytes(b'\x00')
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    self._control.close()
                    self._control = None
            if self.process is None:
                return
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if not self.process.is_alive():
                self.process.close()
                self.process = None


@contextmanager
def services(config):
    stop = threading.Event()
    root = Path(config["storage"]["local_runtime_root"])

    def loop(name, interval, action):
        while not stop.is_set():
            try:
                action()
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "%s unavailable: %s", name, type(exc).__name__
                )
            stop.wait(interval)

    from .task_events import EventStore
    from .knowledge import Knowledge
    from .input_availability import Reconciler

    events = EventStore(root)
    knowledge = Knowledge(config) if config["storage"].get("archive_root") else None
    availability = Reconciler(config)

    progress = ProgressPublisher(config)

    tasks = [
        ("progress-supervisor", 5, progress.ensure),
        ("task-outbox", 2, events.harvest),
        (
            "knowledge-index",
            5,
            lambda: knowledge.tick(stop=stop) if knowledge else None,
        ),
        ("input-availability", 5, availability.tick),
    ]
    subscriptions = config.get("runtime", {}).get("result_callbacks", [])
    if subscriptions:
        tasks.append(("result-callbacks", 5, lambda: events.dispatch(subscriptions)))
    threads = [
        threading.Thread(target=loop, args=task, name=task[0], daemon=True)
        for task in tasks
    ]
    for thread in threads:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        deadline = time.monotonic() + 5
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        progress.close()
