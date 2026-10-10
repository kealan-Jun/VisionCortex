import threading

import pytest

from visioncortex import api
from visioncortex.machine_worker import drain_requested
from visioncortex.run_queue import DurableRunQueue


def test_planned_drain_finishes_current_job_and_leaves_next_queued(monkeypatch, tmp_path):
    drain = tmp_path / "vision.drain"
    monkeypatch.delenv("VISIONCORTEX_DRAIN_FILE", raising=False)
    monkeypatch.setenv("REALITYLOOP_DRAIN_FILE", str(drain))
    assert not drain_requested()
    store = DurableRunQueue(tmp_path / "queue.sqlite")
    for name in ("one", "two"):
        store.save_run(name, {"state": "queued"})
        store.enqueue(name, "run", {})
    monkeypatch.setattr(api, "_persistent_queue", store)
    monkeypatch.setattr(api, "_runs", store.load_runs())
    monkeypatch.setattr(api, "_queue_stop", threading.Event())
    monkeypatch.setattr(api, "_queue_wakeup", threading.Event())
    def execute(job):
        drain.touch()
        api._update(job.run_id, state="completed")
    monkeypatch.setattr(api, "_dispatch_persisted_job", execute)
    api._queue_worker_loop(store)
    assert drain_requested()
    assert store.get_job("one")["status"] == "completed"
    assert store.get_job("two")["status"] == "queued"


@pytest.mark.parametrize("primary_exists", [False, True])
def test_visioncortex_drain_path_takes_precedence_over_legacy(
    monkeypatch, tmp_path, primary_exists,
):
    primary = tmp_path / "visioncortex.drain"
    legacy = tmp_path / "legacy.drain"
    legacy.touch()
    if primary_exists:
        primary.touch()
    monkeypatch.setenv("VISIONCORTEX_DRAIN_FILE", str(primary))
    monkeypatch.setenv("REALITYLOOP_DRAIN_FILE", str(legacy))
    assert drain_requested() is primary_exists


def test_empty_primary_drain_disables_legacy_path(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy.drain"
    legacy.touch()
    monkeypatch.setenv("VISIONCORTEX_DRAIN_FILE", "")
    monkeypatch.setenv("REALITYLOOP_DRAIN_FILE", str(legacy))
    assert not drain_requested()
