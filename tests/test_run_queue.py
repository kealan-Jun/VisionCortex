from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient

from visioncortex import api
from visioncortex.run_queue import DurableRunQueue, QueuedRunJob


def test_durable_queue_preserves_fifo_jobs_and_run_state_across_reopen(tmp_path: Path):
    database = tmp_path / "state" / "queue.sqlite3"
    first_store = DurableRunQueue(database)
    first_store.save_run("run-one", {"state": "queued", "progress": 0.0})
    first_store.enqueue("run-one", "run", {"value": 1}, now=10.0)
    first_store.save_run("run-two", {"state": "queued", "progress": 0.0})
    first_store.enqueue("run-two", "run", {"value": 2}, now=11.0)

    reopened = DurableRunQueue(database)

    assert reopened.load_runs()["run-one"]["state"] == "queued"
    first = reopened.claim_next("worker-a", lease_seconds=60.0, now=20.0)
    assert first is not None
    assert first.run_id == "run-one"
    assert first.payload == {"value": 1}
    assert reopened.claim_next("worker-b", lease_seconds=60.0, now=21.0) is None
    assert reopened.finish("run-one", "worker-a", "completed", now=22.0) is True

    second = reopened.claim_next("worker-b", lease_seconds=60.0, now=23.0)
    assert second is not None
    assert second.run_id == "run-two"


def test_durable_queue_reclaims_expired_job_after_service_restart(tmp_path: Path):
    database = tmp_path / "queue.sqlite3"
    store = DurableRunQueue(database)
    store.save_run("run-restart", {"state": "running"})
    store.enqueue("run-restart", "run", {"checkpoint": "same-run"}, now=10.0)
    original = store.claim_next("old-service", lease_seconds=30.0, now=20.0)

    assert original is not None
    assert store.claim_next("new-service", lease_seconds=30.0, now=49.9) is None

    reopened = DurableRunQueue(database)
    reclaimed = reopened.claim_next("new-service", lease_seconds=30.0, now=50.0)

    assert reclaimed is not None
    assert reclaimed.run_id == "run-restart"
    assert reclaimed.reclaimed is True
    assert reclaimed.attempts == 2
    assert reclaimed.payload["checkpoint"] == "same-run"


def test_api_schedule_writes_sqlite_before_returning(monkeypatch, tmp_path: Path):
    store = DurableRunQueue(tmp_path / "queue.sqlite3")
    scheduled_in_memory = []

    class TaskCollector:
        def add_task(self, function, *args):
            scheduled_in_memory.append((function, args))

    monkeypatch.setattr(api, "_runs", {})
    monkeypatch.setattr(api, "_persistent_queue", store)
    api._update("run-durable", state="queued", progress=0.0)

    persistence = api._schedule_job(
        TaskCollector(),
        run_id="run-durable",
        kind="run",
        payload={"manifest": {"experiment_id": "exp"}},
        fallback=lambda: None,
        fallback_args=(),
    )

    assert persistence == "sqlite"
    assert scheduled_in_memory == []
    assert DurableRunQueue(store.database).get_job("run-durable")["status"] == "queued"
    assert DurableRunQueue(store.database).load_runs()["run-durable"]["state"] == "queued"


def test_persisted_job_dispatch_restores_paths_and_payload(monkeypatch, tmp_path: Path):
    observed = {}
    monkeypatch.setattr(api, "_queue_stop", threading.Event())

    def fake_execute(run_id, settings, nas_root, timing):
        observed.update(
            run_id=run_id,
            settings=settings,
            nas_root=nas_root,
            timing=timing,
        )

    monkeypatch.setattr(api, "_execute_fixed_benchmark_now", fake_execute)
    job = QueuedRunJob(
        run_id="benchmark-one",
        kind="fixed_benchmark",
        payload={
            "settings": {"storage": {"archive_root": "archive"}},
            "nas_root": str(tmp_path / "staging"),
            "timing": {"request_received_at": "2026-09-02T12:00:00+08:00"},
        },
        sequence=1,
        attempts=1,
        reclaimed=False,
    )

    api._dispatch_persisted_job(job)

    assert observed["run_id"] == "benchmark-one"
    assert observed["nas_root"] == tmp_path / "staging"
    assert observed["timing"]["request_received_at"].startswith("2026-09-02")


def test_queue_worker_completes_restored_waiting_job(monkeypatch, tmp_path: Path):
    store = DurableRunQueue(tmp_path / "queue.sqlite3")
    store.save_run("run-restored", {"state": "queued", "progress": 0.0})
    store.enqueue("run-restored", "fixed_benchmark", {"settings": {}})
    completed = threading.Event()

    def fake_dispatch(job):
        api._update(job.run_id, state="completed", progress=1.0)
        completed.set()

    monkeypatch.setattr(api, "_persistent_queue", store)
    monkeypatch.setattr(api, "_runs", store.load_runs())
    monkeypatch.setattr(api, "_dispatch_persisted_job", fake_dispatch)
    monkeypatch.setattr(api, "_QUEUE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(api, "_QUEUE_LEASE_RENEW_SECONDS", 0.01)
    monkeypatch.setattr(api, "_queue_stop", threading.Event())
    monkeypatch.setattr(api, "_queue_wakeup", threading.Event())
    worker = threading.Thread(target=api._queue_worker_loop, args=(store,), daemon=True)
    worker.start()

    assert completed.wait(2.0)
    deadline = time.monotonic() + 2.0
    while store.get_job("run-restored")["status"] != "completed":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    api._queue_stop.set()
    api._queue_wakeup.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert store.load_runs()["run-restored"]["state"] == "completed"


def test_web_lifespan_initializes_local_sqlite_queue(monkeypatch, tmp_path: Path):
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    settings = {
        "mllm": {"api_key_env": "TEST_ARK_KEY", "enabled": False, "model": "test"},
        "storage": {
            "archive_root": str(archive_root),
            "index_csv": str(tmp_path / "index.csv"),
            "local_runtime_root": str(tmp_path / "runtime"),
            "local_cache_root": str(tmp_path / "cache"),
            "sync_to_nas": False,
        },
        "collection_ingest": {"enabled": False},
    }
    monkeypatch.setattr(api, "_settings", lambda: settings)

    with TestClient(api.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    queue = response.json()["execution_queue"]
    assert queue["persistence"] == "sqlite"
    assert queue["survives_web_service_restart"] is True
    assert queue["counts"] == {
        "queued": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
    }
    assert (tmp_path / "runtime" / "state" / "web_run_queue.sqlite3").is_file()
