from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from visioncortex import api


@pytest.fixture
def desktop_config(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "tools"))
    import rtx4050_portable
    shutil.copytree(repo / "configs", tmp_path / "configs")
    (tmp_path / "SHA256SUMS.json").write_text("{}")
    monkeypatch.setattr(rtx4050_portable.os, "environ", dict(rtx4050_portable.os.environ))
    rtx4050_portable.configure_environment(tmp_path)
    _, config = rtx4050_portable.effective_config(tmp_path, {"gpu_uuid": "test-only"})
    monkeypatch.setattr(api, "_settings", lambda: config)
    monkeypatch.setattr(api, "_persistent_queue", None)
    monkeypatch.setattr(api, "_upload_sessions", None)
    monkeypatch.setattr(api, "_runs", {})
    return config


def payload(count, *, all_third=False):
    return {
        "experiment_name": f"dynamic-{count}-view-test",
        "files": [{"file_id": f"video-{i}", "kind": "video", "file_index": i,
                   "name": f"camera-{i}.mp4", "size": 100, "last_modified": 1} for i in range(count)],
        "view_specs": [{"view_id": f"camera-{i}", "role": "first_person" if i == 0 and not all_third else "third_person",
                        "video_index": i, "calibration_hint_ms": 0} for i in range(count)],
    }


@pytest.mark.parametrize("count", [2, 3, 8, 12])
@pytest.mark.parametrize("network", [False, True])
def test_portable_accepts_user_view_count_in_local_and_network_modes(desktop_config, count, network):
    desktop_config["storage"]["sync_to_nas"] = network
    client = TestClient(api.app)
    collections = client.get("/api/collections")
    assert collections.status_code == 200, collections.text
    assert collections.json()["collections"] == []
    response = client.post("/api/upload-sessions", json=payload(count))
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]
    saved = api._upload_sessions.get(session_id)
    assert len(saved["view_specs"]) == count
    assert len(saved["files"]) == count
    registry = json.loads(Path(desktop_config["storage"]["device_registry_path"]).read_text())
    assert registry["devices"] == {}
    health = client.get("/api/health").json()
    assert health["view_count_policy"] == "dynamic"
    assert health["fixed_benchmark"]["enabled"] is False
    assert client.post("/api/benchmarks/six-view-three-hour/runs").status_code == 404
    assert client.delete(f"/api/upload-sessions/{session_id}").status_code == 200


@pytest.mark.parametrize("value", [payload(1), payload(3, all_third=True)])
def test_missing_cross_view_pair_is_rejected_before_any_upload(desktop_config, value):
    response = TestClient(api.app).post("/api/upload-sessions", json=value)
    assert response.status_code == 400
    assert not list(Path(desktop_config["storage"]["archive_root"]).rglob("*.mp4"))


@pytest.mark.parametrize("network", [False, True])
def test_desktop_directory_selection_submits_its_own_index(desktop_config, tmp_path, monkeypatch, network):
    from visioncortex.nas_recordings import scan_recordings, selection_path
    from visioncortex.storage import prepare_from_nas_index

    source = tmp_path / "existing recordings"
    source.mkdir()
    for name in ("camera-one", "camera-two"):
        (source / f"{name}.mp4").write_bytes(b"metadata-contract-not-real-video")
        (source / f"{name}_frames.csv").write_text(
            "frame_system_timestamp_us,rgb_video_frame_index,rgb_recorded\n"
            "1788408000000000,0,1\n1788408001000000,1,1\n"
        )
    import os
    for path in source.iterdir():
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    desktop_config["collection_ingest"].update(
        enabled=True, mode="directory_metadata", source_root=str(source),
        discover_plain_video_csv=True, camera_directories=[],
    )
    desktop_config["storage"]["sync_to_nas"] = network
    original_index = Path(desktop_config["storage"]["index_csv"])
    original_index_bytes = original_index.read_bytes()
    queued = []
    monkeypatch.setattr(api, "_schedule_job", lambda *args, **kwargs: queued.append(kwargs) or "test-only")
    client = TestClient(api.app)
    assert client.get("/api/health").json()["analysis_ready"]
    items = scan_recordings(desktop_config)["recordings"]
    selection = client.post("/api/nas-selections", json={
        "experiment_name": "Selected existing sources",
        "recording_complete_confirmed": True,
        "recordings": [{"recording_id": item["recording_id"], "role": "first_person" if i == 0 else "third_person"}
                       for i, item in enumerate(items)],
    })
    assert selection.status_code == 201, selection.text
    collection_id = selection.json()["collection_id"]
    response = client.post(f"/api/collections/{collection_id}/runs", json={})
    assert response.status_code == 202, response.text
    settings = queued[0]["payload"]["settings"]
    assert settings["storage"]["sync_to_nas"] is network
    assert settings["storage"]["run_output_mode"] == "nas_direct"
    assert settings["storage"]["index_csv"] == str(selection_path(settings, collection_id, ".csv"))
    manifest, _, receipt = prepare_from_nas_index(settings, collection_id)
    assert len(manifest.views) == 2
    assert receipt["copied_source_bytes"] == 0
    assert all(v.segments[0].video.parent == source for v in manifest.views)
    assert original_index.read_bytes() == original_index_bytes
    assert not list(Path(response.json()["nas_staging"]).rglob("*.mp4"))


def test_local_legacy_upload_preserves_local_storage_and_io_metrics(desktop_config, monkeypatch):
    queued = []
    monkeypatch.setattr(api, "_schedule_job", lambda *args, **kwargs: queued.append(kwargs) or "test-only")
    specs = payload(2)["view_specs"]
    response = TestClient(api.app).post("/api/runs", data={
        "experiment_name": "Legacy local upload", "view_specs_json": json.dumps(specs),
    }, files=[("videos", (f"camera-{i}.mp4", b"contract-not-real-video", "video/mp4")) for i in range(2)])
    assert response.status_code == 202, response.text
    settings = queued[0]["payload"]["settings"]
    ingest = queued[0]["payload"]["ingest"]
    assert settings["storage"]["sync_to_nas"] is False
    assert ingest["nas_write_bytes"] == 0
    assert ingest["local_write_bytes"] == ingest["total_bytes"]
    assert len(list(Path(response.json()["nas_output"]).rglob("*.mp4"))) == 2
