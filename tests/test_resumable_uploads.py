from __future__ import annotations

import hashlib
import json
from collections import namedtuple
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from visioncortex import api
from visioncortex.run_queue import DurableRunQueue
from visioncortex.upload_sessions import (
    StorageReservationError,
    UploadSessionStore,
)


def _settings(tmp_path: Path) -> dict:
    return {
        "project": {"output_root": str(tmp_path / "runs")},
        "mllm": {"api_key_env": "TEST_ARK_KEY", "enabled": False, "model": "test"},
        "storage": {
            "archive_root": str(tmp_path / "archive"),
            "index_csv": str(tmp_path / "index.csv"),
            "local_input_root": str(tmp_path / "input"),
            "local_runtime_root": str(tmp_path / "runtime"),
            "local_cache_root": str(tmp_path / "cache"),
            "local_staging_root": str(tmp_path / "staging"),
            "sync_to_nas": False,
            "run_output_mode": "local",
            "web_upload_retention_mode": "local_only",
        },
        "web_upload": {
            "chunk_size_mib": 1,
            "parallel_files": 2,
            "chunk_sha256_required": True,
            "full_file_sha256_max_gib": 4,
            "prequeue_media_preflight_enabled": False,
            "session_ttl_hours": 24,
            "processing_headroom_ratio": 0,
            "safety_headroom_ratio": 0,
            "minimum_safety_headroom_gib": 0,
        },
        "collection_ingest": {"enabled": False},
    }


def _payload(first: bytes, third: bytes, clock: bytes | None = None) -> dict:
    files = [
        {
            "file_id": "video-0",
            "kind": "video",
            "file_index": 0,
            "name": "first.mp4",
            "size": len(first),
            "last_modified": 1,
        },
        {
            "file_id": "video-1",
            "kind": "video",
            "file_index": 1,
            "name": "third.mp4",
            "size": len(third),
            "last_modified": 2,
        },
    ]
    specs = [
        {
            "view_id": "first-view",
            "role": "first_person",
            "video_index": 0,
            "calibration_hint_ms": 0,
        },
        {
            "view_id": "third-view",
            "role": "third_person",
            "video_index": 1,
            "calibration_hint_ms": 0,
        },
    ]
    if clock is not None:
        files.append(
            {
                "file_id": "timestamp-csv-0",
                "kind": "timestamp_csv",
                "file_index": 0,
                "name": "first.csv",
                "size": len(clock),
                "last_modified": 3,
            }
        )
        specs[0]["csv_index"] = 0
    return {"experiment_name": "large-upload", "view_specs": specs, "files": files}


def _reset_api_stores(monkeypatch, tmp_path: Path) -> dict:
    settings = _settings(tmp_path)
    monkeypatch.setattr(api, "_settings", lambda: settings)
    monkeypatch.setattr(api, "_upload_sessions", None)
    monkeypatch.setattr(api, "_persistent_queue", None)
    monkeypatch.setattr(api, "_schedule_job", lambda *_args, **_kwargs: "sqlite")
    api._runs.clear()
    return settings


def _upload_all(client: TestClient, session_id: str, files: dict[str, bytes]) -> None:
    for file_id, content in files.items():
        response = client.patch(
            f"/api/upload-sessions/{session_id}/files/{file_id}",
            content=content,
            headers={
                "Upload-Offset": "0",
                "X-Chunk-SHA256": hashlib.sha256(content).hexdigest(),
            },
        )
        assert response.status_code == 200


def test_storage_reservations_subtract_other_sessions(tmp_path):
    store = UploadSessionStore(tmp_path / "state.sqlite3")
    common = {
        "archive_root": tmp_path / "archive" / "one",
        "storage_key": str(tmp_path / "archive"),
        "experiment_name": "one",
        "view_specs": [],
        "retention_mode": "nas_only",
        "processing_headroom_bytes": 200,
        "safety_headroom_bytes": 100,
        "files": [
            {
                "file_id": "video-0",
                "kind": "video",
                "file_index": 0,
                "view_id": "first",
                "source_name": "first.mp4",
                "stored_name": "first.mp4",
                "expected_bytes": 400,
                "final_path": tmp_path / "archive" / "one" / "first.mp4",
                "partial_path": tmp_path / "archive" / "one" / ".first.partial",
            }
        ],
        "expires_at": 1000,
        "now": 1,
    }
    capacity = store.reserve(
        session_id="one",
        archive_name="one",
        expected_source_bytes=400,
        free_bytes=1000,
        **common,
    )
    assert capacity["reserved_bytes"] == 700
    with pytest.raises(StorageReservationError) as caught:
        store.reserve(
            session_id="two",
            archive_name="two",
            expected_source_bytes=400,
            free_bytes=1000,
            **{**common, "archive_root": tmp_path / "archive" / "two"},
        )
    assert caught.value.available_bytes == 300
    assert caught.value.missing_bytes == 400


def test_resumable_upload_tracks_offsets_and_finalizes_once(monkeypatch, tmp_path):
    _reset_api_stores(monkeypatch, tmp_path)
    first = b"first-video-payload"
    third = b"third-video-payload"
    clock = b"frame_index,timestamp_ms\n0,0\n"
    client = TestClient(api.app)

    created_response = client.post("/api/upload-sessions", json=_payload(first, third, clock))
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["expected_source_bytes"] == len(first) + len(third) + len(clock)
    assert created["capacity"]["reserved_bytes"] == created["expected_source_bytes"]
    session_id = created["session_id"]

    split = 7
    first_chunk = client.patch(
        f"/api/upload-sessions/{session_id}/files/video-0",
        content=first[:split],
        headers={
            "Upload-Offset": "0",
            "X-Chunk-SHA256": hashlib.sha256(first[:split]).hexdigest(),
        },
    )
    assert first_chunk.status_code == 200
    assert first_chunk.json()["uploaded_bytes"] == split
    replayed = client.patch(
        f"/api/upload-sessions/{session_id}/files/video-0",
        content=first[:split],
        headers={"Upload-Offset": "0"},
    )
    assert replayed.status_code == 200
    assert replayed.json()["replayed"] is True
    mismatched = client.patch(
        f"/api/upload-sessions/{session_id}/files/video-0",
        content=b"X" * split,
        headers={"Upload-Offset": "0"},
    )
    assert mismatched.status_code == 409
    future_offset = client.patch(
        f"/api/upload-sessions/{session_id}/files/video-0",
        content=first[split:],
        headers={"Upload-Offset": str(split + 1)},
    )
    assert future_offset.status_code == 409
    assert future_offset.headers["upload-offset"] == str(split)

    for file_id, content, offset in (
        ("video-0", first[split:], split),
        ("video-1", third, 0),
        ("timestamp-csv-0", clock, 0),
    ):
        response = client.patch(
            f"/api/upload-sessions/{session_id}/files/{file_id}",
            content=content,
            headers={
                "Upload-Offset": str(offset),
                "X-Chunk-SHA256": hashlib.sha256(content).hexdigest(),
            },
        )
        assert response.status_code == 200
        assert response.json()["completed"] is True

    status = client.get(f"/api/upload-sessions/{session_id}").json()
    assert status["uploaded_bytes"] == status["expected_source_bytes"]
    finalized = client.post(f"/api/upload-sessions/{session_id}/finalize")
    assert finalized.status_code == 202
    run_id = finalized.json()["run_id"]
    assert run_id == f"upload-{session_id[:12]}"
    repeated = client.post(f"/api/upload-sessions/{session_id}/finalize")
    assert repeated.status_code == 202
    assert repeated.json()["run_id"] == run_id

    session = api._upload_sessions.get(session_id)
    assert session is not None
    assert session["status"] == "finalized"
    for item in session["files"]:
        final_path = Path(item["final_path"])
        assert final_path.is_file()
        assert item["sha256"] == hashlib.sha256(final_path.read_bytes()).hexdigest()
    manifest = Path(session["archive_root"]) / "JSON-Config-Files" / "input_manifest.yaml"
    receipt = (
        Path(session["archive_root"])
        / "JSON-Config-Files"
        / "Stage-Receipts"
        / "original_ingest.json"
    )
    assert manifest.is_file()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["upload_protocol"] == "resumable_chunks_v2"


def test_dynamic_capacity_refuses_only_when_current_space_is_insufficient(
    monkeypatch, tmp_path
):
    _reset_api_stores(monkeypatch, tmp_path)
    DiskUsage = namedtuple("DiskUsage", "total used free")
    monkeypatch.setattr(api.shutil, "disk_usage", lambda _path: DiskUsage(100, 90, 10))
    client = TestClient(api.app)

    response = client.post("/api/upload-sessions", json=_payload(b"123456", b"abcdef"))

    assert response.status_code == 507
    detail = response.json()["detail"]
    assert detail["required_bytes"] == 12
    assert detail["available_bytes"] == 10
    assert detail["missing_bytes"] == 2
    assert not (tmp_path / "archive" / "large-upload").exists()


def test_metadata_plan_above_old_fixed_limits_is_accepted_when_space_is_available(
    monkeypatch, tmp_path
):
    _reset_api_stores(monkeypatch, tmp_path)
    DiskUsage = namedtuple("DiskUsage", "total used free")
    monkeypatch.setattr(
        api.shutil,
        "disk_usage",
        lambda _path: DiskUsage(2_000_000_000_000, 500_000_000_000, 1_500_000_000_000),
    )
    payload = _payload(b"1", b"2")
    payload["files"][0]["size"] = 350_000_000_000
    payload["files"][1]["size"] = 350_000_000_000

    response = TestClient(api.app).post("/api/upload-sessions", json=payload)

    assert response.status_code == 201
    assert response.json()["expected_source_bytes"] == 700_000_000_000
    assert response.json()["capacity"]["available_after_bytes"] == 800_000_000_000


def test_web_client_uses_resumable_sessions_instead_of_one_shot_multipart():
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "visioncortex"
        / "web"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'api("/api/upload-sessions"' in app_js
    assert "uploadPlanFiles(plan" in app_js
    assert "Upload-Offset" in app_js
    assert "new FormData()" not in app_js
    assert "xhrUpload" not in app_js


def test_gpu_job_terminal_state_releases_upload_storage_reservation(
    monkeypatch, tmp_path
):
    released: list[str] = []

    class FakeUploadStore:
        def release(self, session_id: str) -> None:
            released.append(session_id)

    class FakePipeline:
        def __init__(self, _settings, _progress):
            pass

        def run(self, _manifest):
            return tmp_path / "output"

    monkeypatch.setattr(api, "_upload_sessions", FakeUploadStore())
    monkeypatch.setattr(api, "EvidencePipeline", FakePipeline)
    monkeypatch.setattr(api, "_append_web_end_to_end_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "_update", lambda *_args, **_kwargs: None)

    api._execute_now(
        "run-1",
        object(),
        {},
        tmp_path / "archive",
        {"upload_session_id": "session-1"},
    )

    assert released == ["session-1"]


def test_stale_open_session_is_expired_and_scoped_archive_is_removed(
    monkeypatch, tmp_path
):
    settings = _reset_api_stores(monkeypatch, tmp_path)
    client = TestClient(api.app)
    created = client.post(
        "/api/upload-sessions", json=_payload(b"123456", b"abcdef")
    ).json()
    session = api._upload_sessions.get(created["session_id"])
    assert session is not None
    archive_path = Path(session["archive_root"])
    assert archive_path.is_dir()
    with api._upload_sessions._connect() as connection:
        connection.execute(
            "UPDATE upload_sessions SET expires_at = 0 WHERE session_id = ?",
            (created["session_id"],),
        )

    api._expire_stale_upload_sessions(settings)

    expired = api._upload_sessions.get(created["session_id"])
    assert expired is not None
    assert expired["status"] == "expired"
    assert not archive_path.exists()


def test_segmented_browser_views_keep_one_manifest_and_input_seal(
    monkeypatch, tmp_path
):
    _reset_api_stores(monkeypatch, tmp_path)
    client = TestClient(api.app)
    payload = {
        "experiment_name": "segmented-upload",
        "files": [
            {"file_id": "video-0", "kind": "video", "file_index": 0, "name": "first-001.mp4", "size": 3},
            {"file_id": "video-1", "kind": "video", "file_index": 1, "name": "first-002.mp4", "size": 3},
            {"file_id": "video-2", "kind": "video", "file_index": 2, "name": "third.mp4", "size": 3},
        ],
        "view_specs": [
            {
                "view_id": "first-view",
                "role": "first_person",
                "segments": [{"video_index": 0}, {"video_index": 1}],
            },
            {
                "view_id": "third-view",
                "role": "third_person",
                "video_index": 2,
            },
        ],
    }
    created = client.post("/api/upload-sessions", json=payload)
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    _upload_all(client, session_id, {"video-0": b"aaa", "video-1": b"bbb", "video-2": b"ccc"})

    finalized = client.post(f"/api/upload-sessions/{session_id}/finalize")

    assert finalized.status_code == 202
    session = api._upload_sessions.get(session_id)
    manifest_path = Path(session["archive_root"]) / "JSON-Config-Files" / "input_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["views"][0]["segments"]) == 2
    assert manifest["views"][1]["video"].endswith("third.mp4")
    seal_path = Path(session["archive_root"]) / "JSON-Config-Files" / "Input-Manifests" / "input_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["source_mode"] == "browser_resumable_upload"
    assert seal["source_count"] == 3


def test_browser_role_conflict_is_blocked_by_device_registry(monkeypatch, tmp_path):
    settings = _reset_api_stores(monkeypatch, tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-device-registry/1",
                "devices": {"first-view": {"expected_role": "third_person"}},
                "experiment_role_overrides": {},
            }
        ),
        encoding="utf-8",
    )
    settings["storage"]["device_registry_path"] = str(registry)

    response = TestClient(api.app).post(
        "/api/upload-sessions", json=_payload(b"first", b"third")
    )

    assert response.status_code == 400
    assert "index_registry_role_mismatch" in response.text


def test_cancel_open_upload_releases_reservation_and_scoped_files(monkeypatch, tmp_path):
    _reset_api_stores(monkeypatch, tmp_path)
    client = TestClient(api.app)
    created = client.post(
        "/api/upload-sessions", json=_payload(b"first", b"third")
    ).json()
    session_id = created["session_id"]
    session = api._upload_sessions.get(session_id)
    archive_path = Path(session["archive_root"])

    cancelled = client.delete(f"/api/upload-sessions/{session_id}")

    assert cancelled.status_code == 200
    assert cancelled.json()["released_bytes"] == created["reserved_bytes"]
    assert api._upload_sessions.get(session_id)["status"] == "cancelled"
    assert not archive_path.exists()
    assert client.post(f"/api/upload-sessions/{session_id}/finalize").status_code == 409


def test_new_chunk_requires_sha256_header(monkeypatch, tmp_path):
    _reset_api_stores(monkeypatch, tmp_path)
    client = TestClient(api.app)
    session_id = client.post(
        "/api/upload-sessions", json=_payload(b"first", b"third")
    ).json()["session_id"]

    response = client.patch(
        f"/api/upload-sessions/{session_id}/files/video-0",
        content=b"first",
        headers={"Upload-Offset": "0"},
    )

    assert response.status_code == 428


def test_large_file_mode_finalizes_from_persistent_chunk_tree_without_reread(
    monkeypatch, tmp_path
):
    settings = _reset_api_stores(monkeypatch, tmp_path)
    settings["web_upload"]["full_file_sha256_max_gib"] = 0
    monkeypatch.setattr(
        api,
        "_sha256_path",
        lambda _path: (_ for _ in ()).throw(AssertionError("unexpected full reread")),
    )
    client = TestClient(api.app)
    created = client.post(
        "/api/upload-sessions", json=_payload(b"first", b"third")
    ).json()
    _upload_all(
        client,
        created["session_id"],
        {"video-0": b"first", "video-1": b"third"},
    )

    response = client.post(
        f"/api/upload-sessions/{created['session_id']}/finalize"
    )

    assert response.status_code == 202
    session = api._upload_sessions.get(created["session_id"])
    assert all(item["sha256"] is None for item in session["files"])
    assert all(
        item["content_hash_algorithm"] == "visioncortex-upload-chunk-tree-v1"
        for item in session["files"]
    )


def test_archive_receipt_rebuilds_queue_after_local_sqlite_loss(monkeypatch, tmp_path):
    settings = _reset_api_stores(monkeypatch, tmp_path)
    client = TestClient(api.app)
    created = client.post(
        "/api/upload-sessions", json=_payload(b"first", b"third")
    ).json()
    _upload_all(
        client,
        created["session_id"],
        {"video-0": b"first", "video-1": b"third"},
    )
    finalized = client.post(
        f"/api/upload-sessions/{created['session_id']}/finalize"
    ).json()
    run_id = finalized["run_id"]
    recovered_queue = DurableRunQueue(tmp_path / "recovered" / "queue.sqlite3")
    monkeypatch.setattr(api, "_persistent_queue", recovered_queue)
    api._runs.clear()

    api._recover_jobs_from_archive_receipts(settings)

    recovered = recovered_queue.get_job(run_id)
    assert recovered is not None
    assert recovered["status"] == "queued"
    assert api._runs[run_id]["recovered_from_archive_receipt"] is True

    seal_path = (
        Path(finalized["nas_staging"])
        / "JSON-Config-Files"
        / "Input-Manifests"
        / "input_seal.json"
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["source_count"] += 1
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    rejected_queue = DurableRunQueue(tmp_path / "tampered" / "queue.sqlite3")
    monkeypatch.setattr(api, "_persistent_queue", rejected_queue)
    api._runs.clear()

    api._recover_jobs_from_archive_receipts(settings)

    assert rejected_queue.get_job(run_id) is None
