import json
import os
import time
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from visioncortex import api
from visioncortex.collection_catalog import discover_collections
from visioncortex.nas_recordings import (
    create_selection, scan_recordings, selection_path, validate_selection,
)


@pytest.fixture
def nas_config(default_config, tmp_path):
    root = tmp_path / "nas"
    root.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": "visioncortex-device-registry/1", "devices": {}}))
    default_config["collection_ingest"].update(enabled=True, mode="directory_metadata", source_root=str(root), settle_seconds=120)
    default_config["storage"].update(
        index_csv=str(root / "producer-index-does-not-exist.csv"),
        archive_root=str(root / "archive"), local_cache_root=str(root / "cache"),
        device_registry_path=str(registry),
    )
    return default_config


def recording(config, camera, *, offset=0, complete=True, finalized=True):
    root = Path(config["collection_ingest"]["source_root"])
    folder = root / camera / "2026-09-03" / "120000"
    folder.mkdir(parents=True)
    start = 1788408000000000 + offset
    (folder / "rgb.mp4").write_bytes(b"catalog-fixture-no-decode")
    (folder / "frames.csv").write_text("global_timestamp_us,rgb_video_frame_index\n1788408000000000,0\n")
    meta = {"rgb_file": "rgb.mp4", "frames_file": "frames.csv", "closed": True,
            "frames_publish_state": "finalized", "recording_session_id": start,
            "recording_window_start_global_us": start, "recording_window_end_global_us": start + 10000000}
    (folder / "meta.json").write_text(json.dumps(meta))
    if finalized:
        (folder / "recording_ready.json").write_text(json.dumps(meta | {"ready": True, "recording_complete": complete, "recording_quality_status": "complete" if complete else "partial"}))
    for path in folder.iterdir():
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    return folder


def selection(config):
    items = scan_recordings(config)["recordings"]
    return {"experiment_name": "实验一", "recordings": [
        {"recording_id": item["recording_id"], "role": "first_person" if item["camera_key"] == "a_cam01" else "third_person"}
        for item in items
    ]}


def test_indexless_selection_preserves_sources_and_requires_user_roles(nas_config):
    a = recording(nas_config, "a_cam01")
    b = recording(nas_config, "b_cam01", offset=1000000)
    before = {str(p): p.read_bytes() for folder in (a, b) for p in folder.iterdir()}
    assert discover_collections(nas_config)["collections"] == []
    with pytest.raises(ValueError, match="拍摄视角"):
        payload = selection(nas_config)
        payload["recordings"][0]["role"] = ""
        create_selection(nas_config, payload)
    receipt = create_selection(nas_config, selection(nas_config))
    assert receipt["membership_source"] == "user_selection"
    assert receipt["source_copy_bytes"] == 0
    assert len(receipt["rows"]) == 2
    assert selection_path(nas_config, receipt["collection_id"], ".csv").is_file()
    assert validate_selection(nas_config, receipt["collection_id"])["collection_id"] == receipt["collection_id"]
    assert discover_collections(nas_config)["collections"][0]["ready_to_analyze"]
    assert not Path(nas_config["storage"]["index_csv"]).exists()
    assert before == {str(p): p.read_bytes() for folder in (a, b) for p in folder.iterdir()}
    (a / "rgb.mp4").write_bytes(b"changed by producer")
    with pytest.raises(ValueError, match="素材已变化"):
        validate_selection(nas_config, receipt["collection_id"])


def test_incomplete_and_unsealed_recordings_remain_visible_but_unavailable(nas_config):
    recording(nas_config, "a_cam01", complete=False)
    recording(nas_config, "b_cam01", finalized=False)
    items = scan_recordings(nas_config)["recordings"]
    assert len(items) == 2
    assert not any(item["available"] for item in items)
    assert any("等待采集完成标记" in item["issues"] for item in items)
    assert any("采集程序报告数据不完整" in item["issues"] for item in items)
    with pytest.raises(ValueError, match="尚未完成"):
        create_selection(nas_config, selection(nas_config))


def test_unrelated_recordings_cannot_be_merged(nas_config):
    recording(nas_config, "a_cam01")
    recording(nas_config, "b_cam01", offset=20000000)
    with pytest.raises(ValueError, match="时间不重叠"):
        create_selection(nas_config, selection(nas_config))


def test_path_escape_and_recent_writes_are_not_accepted(nas_config):
    folder = recording(nas_config, "a_cam01")
    meta = json.loads((folder / "recording_ready.json").read_text())
    meta["frames_file"] = "../../outside.csv"
    (folder / "recording_ready.json").write_text(json.dumps(meta))
    snapshot = scan_recordings(nas_config)
    assert snapshot["errors"] and not snapshot["recordings"]
    meta["frames_file"] = "frames.csv"
    (folder / "recording_ready.json").write_text(json.dumps(meta))
    snapshot = scan_recordings(nas_config)
    assert not snapshot["recordings"][0]["available"]
    assert "等待文件写入稳定" in snapshot["recordings"][0]["issues"]


def test_missing_nas_does_not_create_a_local_fallback(nas_config):
    root = Path(nas_config["collection_ingest"]["source_root"])
    root.rmdir()
    with pytest.raises(OSError, match="NAS 未连接"):
        scan_recordings(nas_config)
    assert not root.exists()


def test_plain_video_csv_requires_user_completion_and_keeps_sources_unchanged(nas_config):
    nas_config["collection_ingest"]["discover_plain_video_csv"] = True
    root = Path(nas_config["collection_ingest"]["source_root"])
    folder = root / "采集批次"
    folder.mkdir()
    for name in ("正面_RGB", "侧面_RGB"):
        (folder / f"{name}.mp4").write_bytes(b"catalog-fixture-no-decode")
        (folder / f"{name[:-4]}_帧时间戳.csv").write_text(
            "frame_system_timestamp_us,rgb_video_frame_index,rgb_recorded\n"
            "1788408000000000,0,1\n1788408001000000,1,1\n1788408002000000,2,1\n"
        )
    for path in folder.iterdir():
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    items = scan_recordings(nas_config)["recordings"]
    assert len(items) == 2 and all(item["available"] for item in items)
    assert all(item["requires_completion_confirmation"] for item in items)
    assert all(item["duration_seconds"] == 2 for item in items)
    payload = {"experiment_name": "普通采集", "recordings": [
        {"recording_id": item["recording_id"], "role": "first_person" if i == 0 else "third_person"}
        for i, item in enumerate(items)
    ]}
    with pytest.raises(ValueError, match="不再写入"):
        create_selection(nas_config, payload)
    receipt = create_selection(nas_config, payload | {"recording_complete_confirmed": True})
    validate_selection(nas_config, receipt["collection_id"])
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    assert discover_collections(nas_config)["collections"][0]["ready_to_analyze"]
    (folder / "正面_RGB.mp4").write_bytes(b"recorder resumed")
    with pytest.raises(ValueError, match="已变化"):
        validate_selection(nas_config, receipt["collection_id"])


def test_plain_discovery_cannot_override_explicit_incomplete_marker(nas_config):
    nas_config["collection_ingest"]["discover_plain_video_csv"] = True
    recording(nas_config, "a_cam01", complete=False)
    item = scan_recordings(nas_config)["recordings"][0]
    assert not item["available"]
    assert not item["requires_completion_confirmation"]


def test_capture_filename_groups_camera_segments_without_assigning_roles(nas_config):
    nas_config["collection_ingest"]["discover_plain_video_csv"] = True
    root = Path(nas_config["collection_ingest"]["source_root"])
    folder = root / "experiment"
    folder.mkdir()
    for camera in ("正面_树莓派01", "侧面_RK3588"):
        for index, stamp in enumerate(("120000", "120010")):
            stem = f"分段{stamp}_{camera}"
            (folder / f"{stem}_RGB.mp4").write_bytes(b"catalog-fixture-no-decode")
            start = 1788408000000000 + index * 10000000
            (folder / f"{stem}_帧时间戳.csv").write_text(
                "global_timestamp_us,rgb_video_frame_index\n"
                f"{start},0\n{start + 9000000},270\n"
            )
    for path in folder.iterdir():
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    items = scan_recordings(nas_config)["recordings"]
    assert len({item["camera_key"] for item in items}) == 2
    assert all(item["camera_identity_source"] == "capture_filename" for item in items)
    payload = {"experiment_name": "连续分段", "recording_complete_confirmed": True,
               "recordings": [{"recording_id": item["recording_id"],
                               "role": "first_person" if "正面" in item["relative_path"] else "third_person"}
                              for item in items]}
    receipt = create_selection(nas_config, payload)
    assert len(receipt["rows"]) == 2
    assert all(row["segment_count"] == 2 for row in receipt["rows"])


def test_configured_camera_pair_becomes_one_click_batch(nas_config, monkeypatch):
    nas_config["collection_ingest"].update(
        camera_directories=["a_cam01", "b_cam01"],
        camera_role_map={
            "a_cam01": "first_person",
            "b_cam01": "third_person",
        },
        discover_plain_video_csv=False,
    )
    recording(nas_config, "a_cam01")
    recording(nas_config, "b_cam01")
    recording(nas_config, "unmonitored_cam01")

    inventory = scan_recordings(nas_config)
    assert {item["camera_key"] for item in inventory["recordings"]} == {
        "a_cam01",
        "b_cam01",
    }
    assert len(inventory["batches"]) == 1
    batch = inventory["batches"][0]
    assert batch["available"] is True
    assert {item["role"] for item in batch["recordings"]} == {
        "first_person",
        "third_person",
    }

    monkeypatch.setattr(api, "_settings", lambda: nas_config)
    monkeypatch.setattr(
        api,
        "create_collection_run",
        lambda collection_id, background_tasks, payload: {
            "run_id": "collection-fixture",
            "archive_name": payload["experiment_name"],
            "source_collection_id": collection_id,
        },
    )
    result = api.create_nas_batch_run(
        batch["batch_id"], BackgroundTasks(), {}
    )
    receipt = json.loads(
        selection_path(nas_config, result["collection_id"]).read_text()
    )
    assert result["run_id"] == "collection-fixture"
    assert receipt["capture_batch_id"] == batch["batch_id"]
    assert receipt["membership_source"] == "user_selected_capture_batch"
    assert receipt["role_source"] == "configured_camera_role_map"


def test_recorder_scan_ignores_depth_media_and_uses_finalized_quality_window(
    nas_config,
):
    folder = recording(nas_config, "a_cam01")
    (folder / "depth.mkv").write_bytes(b"depth-is-not-an-rgb-view")
    for name in ("meta.json", "recording_ready.json"):
        path = folder / name
        payload = json.loads(path.read_text())
        payload.update(
            recording_window_end_global_us=0,
            recording_quality_window_start_global_us=1788408001000000,
            recording_quality_window_end_global_us=1788408009000000,
        )
        path.write_text(json.dumps(payload))
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    os.utime(folder / "depth.mkv", (time.time() - 1000, time.time() - 1000))

    inventory = scan_recordings(nas_config)

    assert inventory["recording_count"] == 1
    assert inventory["recordings"][0]["relative_path"].endswith("/rgb.mp4")
    assert inventory["recordings"][0]["duration_seconds"] == 8.0
    assert inventory["recordings"][0]["available"] is True


def test_service_continuously_publishes_camera_monitor_receipt(
    nas_config, monkeypatch, tmp_path
):
    nas_config["collection_ingest"].update(
        camera_directories=["a_cam01", "b_cam01"],
        camera_role_map={
            "a_cam01": "first_person",
            "b_cam01": "third_person",
        },
        discover_plain_video_csv=False,
        poll_seconds=5,
    )
    nas_config["storage"]["local_runtime_root"] = str(tmp_path / "runtime")
    recording(nas_config, "a_cam01")
    recording(nas_config, "b_cam01")
    monkeypatch.setattr(api, "_settings", lambda: nas_config)
    monkeypatch.setattr(api, "_persistent_queue", None)
    monkeypatch.setattr(api, "_upload_sessions", None)

    with TestClient(api.app) as client:
        deadline = time.time() + 3
        payload = client.get("/api/nas-recordings").json()
        while payload.get("monitor", {}).get("status") != "watching":
            assert time.time() < deadline
            time.sleep(0.05)
            payload = client.get("/api/nas-recordings").json()
        health = client.get("/api/health").json()

    assert payload["batches"][0]["available"] is True
    assert health["collection_ingest"]["monitor_status"] == "watching"
    assert (
        tmp_path / "runtime" / "state" / "nas-recording-monitor.json"
    ).is_file()


def test_configured_nas_alias_is_allowed_but_nested_symlinks_are_not(nas_config):
    nas_config["collection_ingest"]["discover_plain_video_csv"] = True
    folder = recording(nas_config, "a_cam01")
    root = Path(nas_config["collection_ingest"]["source_root"])
    alias = root.parent / "nas-alias"
    alias.symlink_to(root, target_is_directory=True)
    (root / "nested-alias").symlink_to(folder.parent, target_is_directory=True)
    nas_config["collection_ingest"]["source_root"] = str(alias)
    items = scan_recordings(nas_config)["recordings"]
    assert len(items) == 1
    assert items[0]["camera_key"] == "a_cam01"


@pytest.mark.parametrize("endpoint", [
    "/api/runs", "/api/runs/from-paths", "/api/collections/nas-unknown/runs",
    "/api/nas-batches/nas-batch-000000000000000000000000/runs",
    "/api/benchmarks/six-view-three-hour/runs", "/api/upload-sessions",
    "/api/upload-sessions/unknown/finalize",
])
def test_formal_web_analysis_requires_certification(nas_config, monkeypatch, endpoint):
    nas_config["storage"]["sync_to_nas"] = True
    nas_config["validation"]["model_certification"] = {"required_for_formal_production": True, "path": "/nonexistent/certification.json"}
    monkeypatch.setattr(api, "_settings", lambda: nas_config)
    client = TestClient(api.app)
    response = client.post(endpoint, json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_certification_required"
    assert client.get("/api/nas-recordings").status_code == 200


def test_web_starts_without_nas_and_recovers_readiness(nas_config, monkeypatch, tmp_path):
    nas_config["storage"].update(
        sync_to_nas=True, local_runtime_root=str(tmp_path / "service-state")
    )
    nas_config["validation"]["model_certification"] = {
        "required_for_formal_production": False
    }
    monkeypatch.setattr(api, "_settings", lambda: nas_config)
    monkeypatch.setattr(api, "_persistent_queue", None)
    monkeypatch.setattr(api, "_upload_sessions", None)
    archive = Path(nas_config["storage"]["archive_root"])
    cache = Path(nas_config["storage"]["local_cache_root"])
    with TestClient(api.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["analysis_ready"] is False
        response = client.post("/api/upload-sessions", json={})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "nas_unavailable"
        assert not archive.exists() and not cache.exists()
        archive.mkdir()
        cache.mkdir()
        assert client.get("/api/health").json()["analysis_ready"] is True
        cache.rmdir()
        assert client.get("/api/health").json()["analysis_ready"] is False
        assert client.get("/").status_code == 200
