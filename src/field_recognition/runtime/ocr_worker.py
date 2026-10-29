"""Resident GPU worker executing the same panel pipeline as HTTP requests."""

import argparse
import os
import signal
import threading
import time

from field_recognition.runtime.task_queue import TaskQueue, LeaseLost
from field_recognition.runtime.worker_support import process_lock, heartbeat


class OCRWorker:
    def __init__(self, worker_id, config, core=None):
        self.worker_id = worker_id
        self.config = config
        self.stop = threading.Event()
        self.closed = False
        if core is None:
            from field_recognition.runtime.factory import create_runtime

            core = create_runtime(config, service_role="ocr")
        self.core = core
        self.db = core["database"]
        self.queue = TaskQueue(self.db, worker_id)
        core["task_queue"] = self.queue

    def _process_ocr_task(self, task):
        done = threading.Event()

        def renew():
            while not done.wait(self.queue.heartbeat_interval):
                if not self.queue.renew_lease(task["job_id"], task["lease_holder"]):
                    return

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        try:
            self.core["run_ocr"](task)
        except LeaseLost:
            pass  # Superseded attempts may not publish any terminal state.
        except ImportError as exc:
            # A broken runtime dependency must not consume the photo's retries.
            # Preserve only traceback locations, never exception messages/locals
            # which could include configuration credentials.
            import traceback

            task["runtime_error"] = {
                "type": type(exc).__name__,
                "frames": [
                    {
                        "file": os.path.basename(frame.filename),
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback.extract_tb(exc.__traceback__)[-6:]
                ],
            }
            try:
                self.queue.defer_task(task["job_id"], document=task, error=type(exc).__name__)
            except LeaseLost:
                pass
        except Exception as exc:
            try:
                self.queue.fail_task(task["job_id"], type(exc).__name__, document=task)
            except LeaseLost:
                pass
        finally:
            done.set()
            thread.join(timeout=1)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        runtime = self.core.get("runtime")
        if runtime is not None:
            runtime.close(wait=True)
            return
        # Retain support for the historical explicit mapping adapter.
        self.core["stopping"].set()
        self.core["ocr_pool"].shutdown(wait=True, cancel_futures=True)
        self.core["readout_pool"].shutdown(wait=True, cancel_futures=True)
        if self.core.get("vision_pool") is not None:
            self.core["vision_pool"].shutdown(wait=True, cancel_futures=True)
        model = self.core.get("ocr_model")
        close_model = getattr(model, "close", None) if model is not None else None
        if callable(close_model):
            close_model()
        self.core["panel_detector"].close()

    def run(self, persistent=True):
        from field_recognition.runtime.runtime_rpc import RpcServer

        server = None
        try:
            with process_lock(self.core["DATA"], "ocr"):
                try:
                    self.queue.recover_interrupted()
                    server = RpcServer(
                        "ocr",
                        {
                            "predict_panel": self.core["predict_panel"],
                            "predict_readout": self.core["predict_readout"],
                            "status": lambda: dict(self.core["ocr_state"]),
                            "detector.snapshot": self.core["panel_detector"].snapshot,
                            "warmup": lambda: bool(self.core["queue_ocr_warmup"]()),
                        },
                        data_dir=self.core["DATA"],
                    )
                    server.start()
                    self.core["queue_ocr_warmup"]()
                    self._run_tasks(persistent)
                finally:
                    try:
                        if server is not None:
                            server.close()
                    finally:
                        self.close()
                        heartbeat(self.db, "ocr", {"status": "stopped"})
        finally:
            self.close()

    def _run_tasks(self, persistent):
        pending = set()
        health_at = 0
        while not self.stop.is_set():
            self.core["queue_ocr_warmup"]()
            model = self.core.get("ocr_model")
            if hasattr(model, "health") and time.monotonic() >= health_at:
                self.core["ocr_state"].update(model.health())
                model.cleanup()
                health_at = time.monotonic() + 2
            heartbeat(
                self.db,
                "ocr",
                {
                    "status": "running",
                    "ocr": self.core["ocr_state"],
                    "active_tasks": len(pending),
                },
            )
            finished = {f for f in pending if f.done()}
            for future in finished:
                future.result()
            pending -= finished
            available = not hasattr(model, "health") or self.core["ocr_state"].get("resident")
            if len(pending) < 4 and available:
                task = self.queue.claim_task("ocr")
                if task:
                    pending.add(self.core["readout_pool"].submit(self._process_ocr_task, task))
                    continue
            if not persistent and not pending:
                break
            self.stop.wait(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.environ["FIELD_SERVICE_ROLE"] = "ocr"
    from field_recognition.runtime.production_config import ProductionConfig

    worker = OCRWorker(args.worker_id or f"ocr-{os.getpid()}", ProductionConfig.from_environment())
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.stop.set())
    worker.run(persistent=not args.once)


if __name__ == "__main__":
    main()
