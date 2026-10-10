import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from visioncortex.machine_api import (
    MachineConfig,
    Submissions,
    create_app,
    validate_paths,
)
from visioncortex.run_queue import DurableRunQueue
from visioncortex.schemas import RunManifest
from visioncortex.storage import safe_archive_name


def wait_until(check):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError("Preparation did not finish")


@pytest.fixture
def machine(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    inputs = root / "inputs"
    inputs.mkdir()
    for name in ("one.mp4", "two.mp4", "frames.csv"):
        (inputs / name).write_text("owned synthetic test input")
    settings = {
        "storage": {
            "local_runtime_root": str(root / "runtime"),
            "archive_root": str(root / "archives"),
        }
    }
    native = SimpleNamespace(
        settings=lambda: settings,
        maintenance_active=lambda: False,
        queue_database_path=lambda cfg: root / "runtime/state/queue.sqlite3",
    )

    def initialize(_):
        native.queue = DurableRunQueue(
            native.queue_database_path(settings)
        )

    native.initialize_queue = initialize
    calls = []

    def prepare(body, tasks, *, reserved_run_id, reserved_archive, runtime_settings):
        calls.append(reserved_run_id)
        native.queue.save_run(
            reserved_run_id, {"state": "queued", "experiment_id": reserved_archive[0]}
        )
        native.queue.enqueue(reserved_run_id, "run", {"manifest": body})

    native.submit_paths = prepare
    native.get_run = lambda run: (
        native.queue.load_run(run) | {"run_id": run}
    )
    native.archive_detail = lambda name, **kwargs: {"name": name}
    native.search_key_events = lambda **kwargs: {
        "items": [],
        "archive": kwargs["archive"],
    }
    native.indexed_key_event = lambda uid, **kwargs: {"uid": uid}
    native.indexed_evidence = lambda uid, **kwargs: {"uid": uid}

    def reserve(cfg, name):
        path = Path(cfg["storage"]["archive_root"]) / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr("visioncortex.storage.initialize_nas_archive", reserve)
    config = MachineConfig(
        clients=[
            {
                "client_id": f"gateway-{i}",
                "token_sha256": hashlib.sha256(f"test-{i}".encode()).hexdigest(),
                "default_scope": "default",
                "scopes": {"default": [inputs], "other": [inputs]},
            }
            for i in range(5)
        ]
    )
    body = {
        "experiment_id": "agentdesk-batch-job",
        "views": [
            {
                "view_id": "view_one",
                "role": "first_person",
                "segments": [
                    {
                        "video": str(inputs / "one.mp4"),
                        "timestamps_csv": str(inputs / "frames.csv"),
                    }
                ],
            },
            {
                "view_id": "view_two",
                "role": "third_person",
                "segments": [
                    {
                        "video": str(inputs / "two.mp4"),
                        "timestamps_csv": str(inputs / "frames.csv"),
                    }
                ],
            },
        ],
    }
    return config, native, body, calls


def headers(index=0, key="request-1"):
    return {"Authorization": "Bearer test-" + str(index), "Idempotency-Key": key}


def test_concurrent_idempotency_isolation_and_native_queue(machine):
    config, native, body, calls = machine
    with TestClient(create_app(config, native)) as client:
        with ThreadPoolExecutor(max_workers=5) as pool:
            replies = list(
                pool.map(
                    lambda _: client.post(
                        "/api/runs/from-paths", json=body, headers=headers()
                    ),
                    range(5),
                )
            )
        assert all(r.status_code == 202 for r in replies)
        ids = {r.json()["run_id"] for r in replies}
        assert len(ids) == 1
        run = ids.pop()
        wait_until(lambda: native.queue.get_job(run) is not None)
        assert calls == [run]
        assert (
            client.get("/api/runs/" + run, headers=headers()).json()["experiment_id"]
            == body["experiment_id"]
        )
        assert client.get("/api/runs/" + run, headers=headers(1)).status_code == 404
        assert (
            client.get(
                "/api/runs/" + run, headers=headers() | {"X-Execution-Scope": "other"}
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/api/archives/" + body["experiment_id"], headers=headers()
            ).status_code
            == 409
        )
        changed = dict(body, experiment_id="different")
        assert (
            client.post(
                "/api/runs/from-paths", json=changed, headers=headers()
            ).status_code
            == 409
        )
        native.queue.patch_run(run, {"state": "completed", "progress": 1})
        assert (
            client.get(
                "/api/archives/" + body["experiment_id"], headers=headers()
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/api/key-events",
                params={"archive": body["experiment_id"]},
                headers=headers(1),
            ).status_code
            == 404
        )
        assert client.get("/api/key-events", headers=headers()).status_code == 422
        assert client.get("/api/runtime", headers=headers()).status_code == 404
        assert client.get("/api/runs/" + run).status_code == 401


def test_five_clients_and_reopen_recovery(machine):
    config, native, body, calls = machine
    with TestClient(create_app(config, native)) as client:
        with ThreadPoolExecutor(max_workers=5) as pool:
            replies = list(
                pool.map(
                    lambda i: client.post(
                        "/api/runs/from-paths", json=body, headers=headers(i)
                    ),
                    range(5),
                )
            )
        assert all(r.status_code == 202 for r in replies)
        assert len({r.json()["run_id"] for r in replies}) == 5
        assert len({r.json()["experiment_id"] for r in replies}) == 5
        wait_until(lambda: len(calls) == 5)
    with TestClient(create_app(config, native)) as client:
        assert (
            client.get("/api/run-submissions/request-1", headers=headers()).json()[
                "run_id"
            ]
            == replies[0].json()["run_id"]
        )
        assert len(calls) == 5


def test_crash_after_native_enqueue_does_not_duplicate(machine):
    config, native, body, calls = machine
    initialize = native.initialize_queue
    initialize(native.settings())
    submissions = Submissions(native.queue_database_path(native.settings()))
    row = submissions.claim(
        "gateway-0",
        "default",
        "request-1",
        body,
        Path(native.settings()["storage"]["archive_root"]),
        50,
        "prior-process-config",
    )
    native.queue.save_run(
        row["run_id"], {"state": "queued", "experiment_id": row["archive"]}
    )
    native.queue.enqueue(row["run_id"], "run", {"manifest": body})
    # Simulates process exit before marking machine preparation as queued.
    with TestClient(create_app(config, native)) as client:
        wait_until(
            lambda: (
                submissions.find("gateway-0", "default", key="request-1")["state"]
                == "queued"
            )
        )
        assert (
            client.get("/api/run-submissions/request-1", headers=headers()).json()[
                "run_id"
            ]
            == row["run_id"]
        )
        assert calls == []


def test_paths_and_body_bound(machine, tmp_path):
    config, native, body, _ = machine
    root = config.clients[0].scopes["default"][0]
    outside = tmp_path / "outside.mp4"
    outside.write_text("outside")
    changed = json.loads(json.dumps(body))
    changed["views"][0]["segments"][0]["video"] = str(outside)
    with TestClient(create_app(config, native)) as client:
        assert (
            client.post(
                "/api/runs/from-paths", json=changed, headers=headers()
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/runs/from-paths", content=b"x" * 65537, headers=headers()
            ).status_code
            == 413
        )
    link = root / "link.mp4"
    link.symlink_to(root / "one.mp4")
    changed["views"][0]["segments"][0]["video"] = str(link)
    with pytest.raises(Exception, match="SOURCE_LINK_UNSUPPORTED"):
        validate_paths(RunManifest.model_validate(changed), [root])


@pytest.mark.parametrize("source_kind", ["view", "segment"])
@pytest.mark.parametrize(
    ("audio_kind", "expected_status", "expected_code"),
    [
        ("outside", 403, "SOURCE_DENIED"),
        ("relative", 422, "INPUT_PATH_INVALID"),
        ("symlink", 422, "SOURCE_LINK_UNSUPPORTED"),
        ("missing", 503, "STORAGE_UNAVAILABLE"),
    ],
)
def test_audio_inputs_obey_source_scope(
    machine, tmp_path, source_kind, audio_kind, expected_status, expected_code
):
    config, native, body, calls = machine
    root = config.clients[0].scopes["default"][0]
    allowed = root / "allowed.wav"
    allowed.write_text("owned audio fixture")
    outside = tmp_path / "outside.wav"
    outside.write_text("outside scope")
    link = root / "linked.wav"
    link.symlink_to(allowed)
    audio = {
        "outside": outside,
        "relative": Path("allowed.wav"),
        "symlink": link,
        "missing": root / "missing.wav",
    }[audio_kind]
    changed = json.loads(json.dumps(body))
    view = changed["views"][0]
    source = view["segments"][0]
    if source_kind == "view":
        view.update(view.pop("segments")[0])
        source = view
    source["audio"] = str(audio)
    with TestClient(create_app(config, native)) as client:
        response = client.post("/api/runs/from-paths", json=changed, headers=headers())
        assert response.status_code == expected_status
        assert response.json() == {"code": expected_code}
        assert calls == []
        assert (
            client.get("/api/run-submissions/request-1", headers=headers()).status_code
            == 404
        )


@pytest.mark.parametrize("source_kind", ["view", "segment"])
def test_scoped_audio_is_retained_in_native_manifest(machine, source_kind):
    config, native, body, _ = machine
    root = config.clients[0].scopes["default"][0]
    audio = root / "allowed.wav"
    audio.write_text("owned audio fixture")
    changed = json.loads(json.dumps(body))
    view = changed["views"][0]
    source = view["segments"][0]
    if source_kind == "view":
        view.update(view.pop("segments")[0])
        source = view
    source["audio"] = str(audio)
    with TestClient(create_app(config, native)) as client:
        response = client.post("/api/runs/from-paths", json=changed, headers=headers())
        assert response.status_code == 202
        run = response.json()["run_id"]
        wait_until(lambda: native.queue.get_job(run) is not None)
        queued = native.queue.get_job(run)["payload"]["manifest"]["views"][0]
        assert (queued if source_kind == "view" else queued["segments"][0])["audio"] == str(audio)


@pytest.mark.parametrize("experiment_id", [".Experiment-", ".", "Processing", "x" * 140])
def test_machine_archive_reservation_uses_native_names(machine, experiment_id):
    config, native, body, _ = machine
    body = dict(body, experiment_id=experiment_id)
    with TestClient(create_app(config, native)) as client:
        response = client.post("/api/runs/from-paths", json=body, headers=headers())
        assert response.status_code == 202
        result = response.json()
        assert result["experiment_id"] == safe_archive_name(experiment_id)
        wait_until(lambda: native.queue.get_job(result["run_id"]) is not None)
        assert (
            native.queue.get_job(result["run_id"])["payload"]["manifest"]["experiment_id"]
            == result["experiment_id"]
        )


@pytest.mark.parametrize("experiment_id", [".Shared-", "x" * 140])
def test_normalized_archive_collisions_preserve_suffix(machine, experiment_id):
    config, native, body, _ = machine
    canonical = safe_archive_name(experiment_id)
    root = Path(native.settings()["storage"]["archive_root"])
    (root / canonical).mkdir(parents=True)
    with TestClient(create_app(config, native)) as client:
        results = []
        for key, name in (("first", experiment_id), ("second", canonical)):
            response = client.post(
                "/api/runs/from-paths",
                json=dict(body, experiment_id=name),
                headers=headers(key=key),
            )
            assert response.status_code == 202
            results.append(response.json())
        names = [result["experiment_id"] for result in results]
        assert len({canonical, *names}) == 3
        for result in results:
            name = result["experiment_id"]
            assert name == safe_archive_name(name)
            assert len(name) <= 120
            wait_until(lambda: native.queue.get_job(result["run_id"]) is not None)


def test_completed_archive_delegates_to_current_native_api(machine, monkeypatch):
    from visioncortex import api

    config, native, body, _ = machine
    resolved = []
    root = Path(native.settings()["storage"]["archive_root"]) / body["experiment_id"]

    def resolve(name):
        resolved.append(name)
        return root

    monkeypatch.setattr(api, "_resolve_archive", resolve)
    monkeypatch.setattr(
        api, "_archive_detail_from_root", lambda path, name: {"name": name, "root": str(path)}
    )
    native.archive_detail = api.archive_detail
    with TestClient(create_app(config, native)) as client:
        response = client.post("/api/runs/from-paths", json=body, headers=headers())
        run = response.json()["run_id"]
        wait_until(lambda: native.queue.get_job(run) is not None)
        url = "/api/archives/" + body["experiment_id"]
        assert client.get(url, headers=headers()).status_code == 409
        assert resolved == []
        native.queue.patch_run(run, {"state": "completed"})
        response = client.get(url, headers=headers())
        assert response.status_code == 200
        assert response.json() == {"name": body["experiment_id"], "root": str(root)}
        assert resolved == [body["experiment_id"]]


def test_queue_capacity_replay_and_configuration_conflict(machine):
    _config, native, body, _ = machine
    native.initialize_queue(native.settings())
    store = Submissions(native.queue_database_path(native.settings()))
    root = Path(native.settings()["storage"]["archive_root"])
    rows = [
        store.claim("gateway-0", "default", str(i), body, root, 5, "revision-a")
        for i in range(5)
    ]
    assert (
        store.claim("gateway-0", "default", "0", body, root, 5, "revision-a")["run_id"]
        == rows[0]["run_id"]
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as conflict:
        store.claim("gateway-0", "default", "0", body, root, 5, "revision-b")
    assert conflict.value.status_code == 409
    with pytest.raises(HTTPException) as full:
        store.claim("gateway-0", "default", "next", body, root, 5, "revision-a")
    assert full.value.status_code == 429


def test_pending_config_change_fails_without_reinterpreting_input(machine):
    config, native, body, calls = machine
    native.initialize_queue(native.settings())
    store = Submissions(native.queue_database_path(native.settings()))
    row = store.claim(
        "gateway-0",
        "default",
        "request-1",
        body,
        Path(native.settings()["storage"]["archive_root"]),
        50,
        "old-build",
    )
    with TestClient(create_app(config, native)) as client:
        wait_until(
            lambda: (
                store.find("gateway-0", "default", key="request-1")["state"] == "failed"
            )
        )
        result = client.get("/api/runs/" + row["run_id"], headers=headers()).json()
        assert result["error_code"] == "CONFIGURATION_CHANGED"
        assert not calls
