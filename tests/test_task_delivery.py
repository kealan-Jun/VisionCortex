import hashlib
import hmac
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.run_queue import DurableRunQueue
from visioncortex.task_events import EventStore, append
from visioncortex.submission import Submissions, handle, bind_task


def recording():
    return {
        "recording_id": "one",
        "camera_key": "cam",
        "configured_role": "first_person",
        "source_signature": "s1",
        "recording_start_us": 1787792400000000,
    }


def test_transactional_outbox_replay_ack_and_stale_owner(tmp_path):
    q = DeviceDayQueue(tmp_path / "device-day/queue-vision.sqlite3")
    q.enqueue(recording(), "r1")
    q.enqueue(recording(), "r1")
    q.claim("worker")
    q.finish("wrong", "one", {"status": "completed"}, 1)
    q.finish("worker", "one", {"status": "completed"}, 1)
    store = EventStore(tmp_path)
    assert store.harvest() == 3
    assert store.harvest() == 0
    rows = store.read(consumer="agent")
    assert [r["state"] for r in rows["events"]] == ["queued", "running", "completed"]
    assert store.read(consumer="agent") == rows
    with pytest.raises(ValueError):
        store.acknowledge("agent", rows["next_cursor"] + 1)
    store.acknowledge("agent", rows["next_cursor"])
    assert EventStore(tmp_path).read(consumer="agent")["events"] == []
    with pytest.raises(RuntimeError), q.connect() as db:
        append(db, "rollback", "completed")
        raise RuntimeError("transaction aborted")
    assert store.harvest() == 0


def test_offline_events_share_delivery(tmp_path):
    queue = DurableRunQueue(tmp_path / "state/web_run_queue.sqlite3")
    queue.enqueue("run", "run", {})
    queue.claim_next("owner", lease_seconds=60)
    queue.finish("run", "owner", "completed")
    store = EventStore(tmp_path)
    assert store.harvest() == 3
    assert all(r["source"] == "offline" for r in store.read()["events"])


def test_signed_callback_requires_ack_and_retries_without_rerun(tmp_path, monkeypatch):
    q = DeviceDayQueue(tmp_path / "device-day/queue-vision.sqlite3")
    q.enqueue(recording(), "r1")
    store = EventStore(tmp_path)
    store.harvest()
    monkeypatch.setenv("TEST_RESULT_SIGNING", "test-only")
    attempts = []

    def recipient(req):
        stamp = req.headers["X-VisionCortex-Timestamp"]
        expected = hmac.new(
            b"test-only", stamp.encode() + b"." + req.content, hashlib.sha256
        ).hexdigest()
        assert req.headers["X-VisionCortex-Signature"] == expected
        payload = json.loads(req.content)
        attempts.append(payload)
        return httpx.Response(200, json={"acknowledged_cursor": payload["next_cursor"]})

    subscription = [
        {
            "name": "local-test",
            "url": "https://callback.example/result",
            "secret_env": "TEST_RESULT_SIGNING",
        }
    ]
    store.dispatch(subscription, transport=httpx.MockTransport(recipient))
    store.dispatch(subscription, transport=httpx.MockTransport(recipient))
    assert len(attempts) == 1
    assert q.snapshot()["counts"]["queued"] == 1


def test_concurrent_submission_claim_is_unique(tmp_path):
    store = Submissions(tmp_path)
    with ThreadPoolExecutor(8) as pool:
        result = list(pool.map(lambda _: store.claim("same", "body"), range(20)))
    assert sum(won for _, won in result) == 1
    with pytest.raises(ValueError):
        store.claim("same", "different")


def submission_app(tmp_path):
    app = FastAPI()
    calls = []

    @app.middleware("http")
    async def middleware(request, call_next):
        return await handle(
            request, call_next, {"storage": {"local_runtime_root": str(tmp_path)}}
        )

    @app.post("/api/runs/from-paths")
    async def json_submit(request: Request):
        body = await request.json()
        calls.append(body)
        bind_task("run-one")
        return {"run_id": "run-one"}

    @app.post("/api/runs")
    async def file_submit(request: Request):
        async with request.form() as form:
            file = form["video"]
            calls.append(await file.read())
        return {"run_id": "upload-one"}

    return app, calls


def test_http_idempotency_json_and_multipart_boundaries(tmp_path):
    app, calls = submission_app(tmp_path)
    with TestClient(app) as client:
        assert (
            client.post("/api/runs/from-paths", json={"a": 1, "b": 2}).status_code
            == 200
        )
        replay = client.post("/api/runs/from-paths", json={"b": 2, "a": 1})
        assert replay.headers["Idempotency-Replayed"] == "true"
        assert len(calls) == 1
        first = client.post(
            "/api/runs", files={"video": ("test.mp4", b"not-real-media")}
        )
        second = client.post(
            "/api/runs", files={"video": ("test.mp4", b"not-real-media")}
        )
        assert first.status_code == second.status_code == 200
        assert second.headers["Idempotency-Replayed"] == "true"
        assert calls[-1] == b"not-real-media"
        assert len(calls) == 2
        headers = {"Idempotency-Key": "explicit"}
        client.post("/api/runs/from-paths", json={"a": 1}, headers=headers)
        assert (
            client.post(
                "/api/runs/from-paths", json={"a": 2}, headers=headers
            ).status_code
            == 409
        )


def test_progress_last_good_survives_slow_refresh(tmp_path):
    import asyncio
    from threading import Event
    from visioncortex.device_day_progress import ProgressPoller

    release = Event()
    calls = []

    def reader(config):
        calls.append(1)
        if len(calls) > 1:
            assert release.wait(2)
        return {"observed_at": time.time(), "days": {}}

    async def run(pool):
        poller = ProgressPoller(
            lambda: {"storage": {"local_runtime_root": str(tmp_path)}},
            executor=pool,
            reader=reader,
        )
        first = await poller.read()
        poller.started -= 2
        second = await asyncio.wait_for(poller.read(), 0.2)
        assert second["observed_at"] == first["observed_at"]
        assert (tmp_path / "device-day/ProgressSnapshot.json").exists()
        release.set()

    with ThreadPoolExecutor(1) as pool:
        asyncio.run(run(pool))
