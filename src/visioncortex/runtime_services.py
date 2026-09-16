"""Independent operational workers. No model startup or media reads."""

from contextlib import contextmanager
import logging
from pathlib import Path
import threading
import time

from .device_day_contract import atomic_json


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
    from .device_day_progress import snapshot
    from .knowledge import Knowledge
    from .input_availability import Reconciler

    events = EventStore(root)
    knowledge = Knowledge(config) if config["storage"].get("archive_root") else None
    availability = Reconciler(config)

    def publish_progress():
        atomic_json(root / "device-day" / "ProgressSnapshot.json", snapshot(config))

    tasks = [
        ("progress-publisher", 5, publish_progress),
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
