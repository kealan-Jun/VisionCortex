"""Dedicated NAS worker; archive outages cannot block the API or GPU worker."""

import argparse
import os
import signal
import threading

from field_recognition.runtime.worker_support import process_lock, heartbeat
from field_recognition.archive.archive_progress import observe


class ArchiveWorker:
    def __init__(self, worker_id, config, core=None):
        self.worker_id = worker_id
        self.config = config
        self.stop = threading.Event()
        self.closed = False
        if core is None:
            from field_recognition.runtime.factory import create_runtime

            core = create_runtime(config, service_role="archive")
        self.core = core
        self.db = core["database"]
        self.archive_store = core["archive_store"]

    def _process_batch(self):
        before = self.archive_store.snapshot()["pending_receipts"]
        self.archive_store.step()
        return max(0, before - self.archive_store.snapshot()["pending_receipts"])

    def run(self, persistent=True):
        try:
            with process_lock(self.core["DATA"], "archive"):
                try:
                    heartbeat(
                        self.db, "archive", self.archive_store.snapshot() | {"status": "starting"}
                    )
                    self.archive_store.integrity.start()
                    while not self.stop.is_set():
                        delay = 3
                        try:
                            with observe(
                                lambda progress: heartbeat(
                                    self.db, "archive", self.archive_store.snapshot() | progress
                                )
                            ):
                                self._process_batch()
                            status = self.archive_store.snapshot()
                            delay = self.archive_store.next_poll_seconds()
                        except Exception as exc:
                            status = {"status": "retrying", "last_error": type(exc).__name__}
                        heartbeat(self.db, "archive", status)
                        if not persistent:
                            break
                        self.stop.wait(delay)
                finally:
                    self.close()
                    heartbeat(self.db, "archive", {"status": "stopped"})
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        runtime = self.core.get("runtime")
        if runtime is not None:
            runtime.close(wait=True)
        else:
            self.archive_store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.environ["FIELD_SERVICE_ROLE"] = "archive"
    from field_recognition.runtime.production_config import ProductionConfig

    worker = ArchiveWorker(
        args.worker_id or f"archive-{os.getpid()}", ProductionConfig.from_environment()
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.stop.set())
    worker.run(persistent=not args.once)


if __name__ == "__main__":
    main()
