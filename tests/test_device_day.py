import os
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from visioncortex.device_day import DeviceDayRunner, append_comment, delete_capture_video
from visioncortex.device_day_contract import (
    DIRECTORIES, VERSION, DeviceDayLayout, archive_name, artifact, atomic_json,
    digest, read_json, safe_child, validate_config,
)
from visioncortex.device_day_models import check_coverage, validate_understanding, windows
from visioncortex.device_day_service import DeviceDayService, install_routes
from visioncortex.nas_recordings import scan_recordings
from visioncortex.schemas import FrameEvidence, ViewRole


@pytest.fixture
def device_config(default_config, tmp_path):
    config = deepcopy(default_config)
    source = tmp_path / "capture"
    source.mkdir()
    config["collection_ingest"].update(enabled=True, source_root=str(source), mode="directory_metadata",
                                        camera_role_map={"a_cam01": "first_person", "b_cam02": "third_person"})
    config["storage"].update(archive_root=str(tmp_path / "archives"), local_cache_root=str(tmp_path / "cache"),
                              local_runtime_root=str(tmp_path / "runtime"))
    config["device_day"].update(enabled=True, vision_workers=2, understanding_workers=2)
    return config


def capture(config, camera="a_cam01", *, start=1789005600000000, duration=900, label="100000"):
    folder = Path(config["collection_ingest"]["source_root"]) / camera / "2026-09-10" / label
    folder.mkdir(parents=True)
    (folder / "rgb.mp4").write_bytes(b"synthetic-fixture-not-real-media")
    (folder / "frames.csv").write_text(f"global_timestamp_us,rgb_video_frame_index\n{start},0\n")
    metadata = {"rgb_file": "rgb.mp4", "frames_file": "frames.csv", "closed": True,
                "frames_publish_state": "finalized", "recording_session_id": "fixture-1",
                "recording_window_start_global_us": start, "recording_window_end_global_us": start + duration * 1000000}
    atomic_json(folder / "meta.json", metadata)
    atomic_json(folder / "recording_ready.json", metadata | {"ready": True, "recording_complete": True,
                                                             "recording_quality_status": "complete"})
    for path in folder.iterdir():
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    return folder


class FakeModels:
    """Deterministic stage boundary fixture, never a real-inference receipt."""
    def __init__(self):
        self.vision_calls = 0
        self.semantic_calls = 0
        self.fail = False
        self.barrier = None

    def transcribe(self, layout, retention, key):
        return {"outcome": "no_audio", "comments": [], "artifacts": [], "model_invocation": "NOT_PROVEN"}

    def vision(self, layout, retention, key):
        self.vision_calls += 1
        if self.barrier:
            self.barrier.wait(timeout=5)
        if self.fail:
            raise ValueError("synthetic scan failure")
        recording = retention["recording"]
        folder = layout.processed / "Clips" / f"10-00-00_10-15-00_NoExperimentActivity_{key[:8]}"
        folder.mkdir(parents=True, exist_ok=True)
        frame = folder / "frame.jpg"
        frame.write_bytes(b"synthetic-frame")
        reference = artifact(layout.root, frame)
        source = next(s["retained"] for s in retention["sources"] if s["kind"] == "video")
        record = {"schema_version": VERSION, "segment_id": key[:16], "recording_id": recording["recording_id"],
                  "activity": "inactive", "activity_label": "无实验活动", "start_ms": 0, "end_ms": 900000,
                  "start_us": recording["recording_start_us"], "end_us": recording["recording_end_us"],
                  "source_ref": source, "key_frames": [], "scene_frames": [reference | {"frame_id": "f1", "local_ms": 10, "frame_kind": "scene_sample"}],
                  "evidence_status": "PARTIAL_EVIDENCE", "physical_action_confirmed": False,
                  "json_path": layout.relative(folder / "NoExperimentActivity.json")}
        atomic_json(folder / "NoExperimentActivity.json", record)
        return {"segments": [record], "artifacts": [reference, artifact(layout.root, folder / "NoExperimentActivity.json")]}

    def understand(self, layout, recording, vision, context, key):
        self.semantic_calls += 1
        path = layout.understanding / "ClipUnderstanding" / f"{key}.json"
        item = {"segment_id": vision["segments"][0]["segment_id"], "windows": [{
            "summary": "抽样画面未发现实验操作；确定性测试，非真实模型证据",
            "activity_observed": "inactive", "steps": [], "uncertainties": [],
            "frame_observations": [{"frame_id": "f1", "text": "测试关键帧文字"}]}]}
        atomic_json(path, item)
        return {"vision_key": vision["key"], "context_digest": digest(context),
                "understandings": [item], "artifacts": [artifact(layout.root, path)]}


def item_and_layout(config):
    item = scan_recordings(config)["recordings"][0]
    layout = DeviceDayLayout(Path(config["storage"]["archive_root"]), item["camera_key"], item["recording_start_us"], Path(config["storage"]["local_cache_root"]))
    return item, layout


def test_frozen_layout_and_deletion_suspension(device_config):
    assert DIRECTORIES == ("MetaVideo", "ProcessedClips", "MultimodalUnderstanding", "LaboratoryDailyReport", "Comment")
    schema = read_json(Path(__file__).parents[1] / "docs/contracts/device-day-v1.schema.json")
    assert schema["properties"]["schema_version"]["const"] == VERSION
    assert schema["properties"]["capture_deletion"]["const"] == "disabled_by_user"
    assert archive_name("lubancat-52d2ef0c_cam01", 1789005600000000) == "2026-09-10_lubancat-52d2ef0c_cam01"
    for change in ({"chunk_seconds": 1800}, {"schema_version": "other"}, {"delete_capture_sources": True}):
        modified = deepcopy(device_config)
        modified["device_day"].update(change)
        with pytest.raises(ValueError):
            validate_config(modified)


def test_engine_rollover_keeps_exact_completed_receipt_and_recomputes_damage(device_config, tmp_path):
    from visioncortex.device_day import visual_input
    from visioncortex.device_day_contract import file_hash
    capture(device_config)
    record, layout = item_and_layout(device_config)
    old = DeviceDayRunner(device_config, FakeModels())
    assert old.process(record)["status"] == "completed"
    path = layout.receipts / record["recording_id"] / "vision.json"
    receipt = read_json(path)
    retained = read_json(path.with_name("retention.json"))
    changed = deepcopy(device_config)
    changed["performance"]["batch_size"] = 32
    current_key = DeviceDayRunner(changed, FakeModels())._key("vision", record, visual_input(retained))
    assert current_key != receipt["key"]
    manifest = tmp_path / "CompletedVisionReceipts.json"
    atomic_json(manifest, {"schema_version": "visioncortex-engine-rollover/1", "entries": [{
        "expected_key": current_key, "receipt_key": receipt["key"], "receipt_digest": digest(receipt),
        "recording_id": record["recording_id"]}]})
    changed["device_day"]["completed_vision_receipts"] = {"path": str(manifest), "sha256": file_hash(manifest)}
    backend = FakeModels()
    current = DeviceDayRunner(changed, backend)
    assert current.process(record, stage="vision")["status"] == "completed"
    assert backend.vision_calls == 0
    assert read_json(path) == receipt  # Preserve old engine provenance, not a new-model claim.
    assert read_json(layout.index)["segments"]
    assert not current._accepts_receipt(receipt | {"wall_seconds": -1}, current_key)
    assert not current._accepts_receipt(receipt | {"stage": "stt"}, current_key)
    assert not current._accepts_receipt(receipt, "0" * 64)
    changed_again = deepcopy(changed)
    changed_again["performance"]["batch_size"] = 8
    next_runner = DeviceDayRunner(changed_again, FakeModels())
    assert not next_runner._accepts_receipt(receipt, next_runner._key("vision", record, visual_input(retained)))
    safe_child(layout.root, receipt["artifacts"][0]["path"]).write_bytes(b"damaged")
    assert current.process(record, stage="vision")["status"] == "completed"
    assert backend.vision_calls == 1
    assert read_json(path)["key"] == current_key
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        DeviceDayRunner(changed, FakeModels())


def test_dense_scanner_resolves_legacy_sparse_auto_without_mutating_config(device_config):
    from visioncortex.device_day_models import DeviceDayModels
    device_config["performance"]["motion_probe_sparse_strategy"] = "auto"
    backend = DeviceDayModels(device_config)
    assert backend.config["performance"]["motion_probe_sparse_strategy"] == "indexed_seek"
    assert device_config["performance"]["motion_probe_sparse_strategy"] == "auto"


def test_resume_checkpoints_preserve_old_stage_provenance_and_queue_revision(device_config, tmp_path):
    from visioncortex.device_day import load_context, visual_input
    from visioncortex.device_day_contract import STAGES, file_hash
    from visioncortex.device_day_engine_rollover import CompletedStageReceipts
    capture(device_config)
    record, layout = item_and_layout(device_config)
    old = DeviceDayRunner(device_config, FakeModels())
    assert old.process(record)['status'] == 'completed'
    receipts = {stage: read_json(old._receipt(layout, record, stage)) for stage in STAGES}
    for stage in STAGES:
        old.queues[stage].enqueue(record, '1' * 64)
        old.queues[stage].claim('fixture')
        old.queues[stage].finish('fixture', record['recording_id'], {'status': 'completed'}, 1)
    backend = FakeModels()
    current = DeviceDayRunner(device_config, backend)
    current._execution_identity = 'a' * 64
    inputs = {'retention': record, 'vision': visual_input(receipts['retention']),
              'stt': receipts['retention'], 'report': receipts['understanding'],
              'understanding': {'vision': receipts['vision'], 'stt': receipts['stt'],
                                'context': load_context(layout, record)}}
    entries = [{'stage': stage, 'recording_id': record['recording_id'],
                'expected_key': current._key(stage, record, inputs[stage]),
                'receipt_key': receipts[stage]['key'], 'receipt_digest': digest(receipts[stage]),
                'queue_revision': '1' * 64} for stage in STAGES]
    manifest = tmp_path / 'Checkpoints.json'
    atomic_json(manifest, {'schema_version': 'visioncortex-resume-checkpoints/1', 'entries': entries})
    current._completed_stage_receipts = CompletedStageReceipts({'completed_stage_receipts': {
        'path': str(manifest), 'sha256': file_hash(manifest)}})
    for stage in STAGES:
        current._build_stage_inventory({'recordings': [record]}, stage, None)
        with current.queues[stage].connect() as db:
            row = db.execute('SELECT status,revision FROM recordings').fetchone()
            assert (row['status'], row['revision']) == ('completed', '1' * 64)
    assert current.process(record)['status'] == 'completed'
    assert backend.vision_calls == backend.semantic_calls == 0
    assert {stage: read_json(current._receipt(layout, record, stage)) for stage in STAGES} == receipts
    assert read_json(layout.index)['understandings']
    assert all(current._accepts_receipt(receipts[e['stage']], e['expected_key']) for e in entries)
    for entry in entries:
        receipt = receipts[entry['stage']]
        assert not current._completed_stage_receipts.accepts(receipt, entry['expected_key'], queue_revision='2' * 64)
        assert not current._accepts_receipt(receipt | {'wall_seconds': -1}, entry['expected_key'])
    changed = record | {'source_signature': 'changed'}
    assert not current._accepts_receipt(receipts['retention'], current._key('retention', changed, changed))
    frame = safe_child(layout.root, receipts['vision']['artifacts'][0]['path'])
    frame.write_bytes(b'damaged')
    assert current._load(current._receipt(layout, record, 'vision'), entries[1]['expected_key'], layout) is None


def test_closed_partial_capture_is_processed_without_claiming_complete(device_config):
    folder = capture(device_config)
    ready = read_json(folder / "recording_ready.json")
    atomic_json(folder / "recording_ready.json", ready | {"recording_complete": False, "recording_quality_status": "partial"})
    os.utime(folder / "recording_ready.json", (time.time() - 1000, time.time() - 1000))
    item, layout = item_and_layout(device_config)
    assert not item["available"] and item["processable"] and not item["capture_complete"]
    backend = FakeModels()
    assert DeviceDayRunner(device_config, backend).process(item)["status"] == "completed"
    assert backend.vision_calls == backend.semantic_calls == 1
    retained = read_json(layout.receipts / item["recording_id"] / "retention.json")
    assert retained["recording"]["issues"] == ["采集程序报告数据不完整"]


def test_nas_batch_button_routes_partial_single_device_to_new_workflow(device_config, monkeypatch):
    from visioncortex import api
    folder = capture(device_config)
    ready = read_json(folder / "recording_ready.json")
    atomic_json(folder / "recording_ready.json", ready | {"recording_complete": False, "recording_quality_status": "partial"})
    os.utime(folder / "recording_ready.json", (time.time() - 1000, time.time() - 1000))
    inventory = scan_recordings(device_config)
    batch = inventory["batches"][0]
    assert batch["analysis_ready"] and not batch["available"]
    monkeypatch.setattr(api, "_settings", lambda: device_config)
    called = []
    monkeypatch.setattr(api._device_day_service, "submit", lambda settings, recordings:
                        called.append(recordings) or {"workflow": "device_day", "status": "queued"})
    response = TestClient(api.app).post(f"/api/nas-batches/{batch['batch_id']}/runs", json={})
    assert response.status_code == 202
    assert response.json()["workflow"] == "device_day"
    assert called[0][0]["capture_complete"] is False


def test_retained_clock_interpolation_preserves_retimed_video(tmp_path):
    from visioncortex.device_day_models import capture_clock, capture_us
    path = tmp_path / "frames.csv"
    path.write_text("global_timestamp_us,rgb_video_frame_index\n1789005600000000,0\n1789005602000000,30\n")
    mapping = capture_clock(path, 30, 1789005600000000)
    assert mapping["basis"] == "recorder_csv_interpolation"
    assert capture_us(mapping, 500) == 1789005601000000


def test_audio_stt_comments_feed_understanding_without_repeating_yolo(device_config):
    folder = capture(device_config)
    item, layout = item_and_layout(device_config)

    class AudioModels(FakeModels):
        def transcribe(self, layout, retention, key):
            if retention["audio"]["status"] != "provided":
                return super().transcribe(layout, retention, key)
            path = layout.comments / "Stt" / "fixture.json"
            atomic_json(path, {"synthetic_transcript": "这只是测试转写"})
            return {"outcome": "transcribed", "model_invocation": "NOT_PROVEN", "comments": [{
                "comment_id": "stt-fixture", "start_us": item["recording_start_us"], "end_us": item["recording_end_us"],
                "text": "这只是测试转写", "source": "machine_transcribed_speech", "human_reviewed": False,
                "transcript_path": layout.relative(path)}], "artifacts": [artifact(layout.root, path)]}

        def understand(self, layout, recording, vision, context, key):
            self.last_comments = context["comments"]
            return super().understand(layout, recording, vision, context, key)

    backend = AudioModels()
    runner = DeviceDayRunner(device_config, backend)
    assert runner.process(item)["status"] == "completed"
    (folder / "audio.opus").write_bytes(b"fixture-audio-not-real-speech")
    atomic_json(folder / "audio_meta.json", {"audio_valid": True, "first_audio_global_us": item["recording_start_us"]})
    atomic_json(folder / "audio_ready.json", {"ready": True, "audio_valid": True})
    for path in folder.glob("audio*"):
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    updated, _ = item_and_layout(device_config)
    assert runner.process(updated)["status"] == "completed"
    assert backend.vision_calls == 1
    assert backend.semantic_calls == 2
    assert backend.last_comments[0]["source"] == "machine_transcribed_speech"
    assert (layout.raw / "Audio" / "10-00-00_10-15-00.opus").is_file()
    index = read_json(layout.index)
    assert index["recordings"][0]["audio"]["status"] == "provided"
    assert "这只是测试转写" in (layout.reports / "LaboratoryDailyReport.html").read_text()
    assert all(p.name.isascii() for p in layout.root.rglob("*") if p.is_dir())


def test_all_stages_publish_one_report_resume_and_keep_source(device_config):
    original = capture(device_config)
    before = {p.name: p.read_bytes() for p in original.iterdir()}
    item, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    assert runner.process(item)["status"] == "completed"
    assert sorted(p.name for p in layout.root.iterdir()) == sorted(DIRECTORIES)
    assert sorted(p.name for p in layout.processed.iterdir()) == ["Clips", "Index.json"]
    assert (layout.raw / "10-00-00_10-15-00.mp4").is_file()
    index = read_json(layout.index)
    assert index["segments"][0]["activity"] == "inactive"
    assert index["segments"][0]["scene_frames"][0]["understanding_text"] == "测试关键帧文字"
    assert len(index["understandings"]) == 1
    assert read_json(layout.reports / "LaboratoryDailyReport.json")["logical_report_count"] == 1
    assert "测试关键帧文字" in (layout.reports / "LaboratoryDailyReport.html").read_text()
    assert runner.process(item)["status"] == "completed"
    assert backend.vision_calls == backend.semantic_calls == 1
    assert before == {p.name: p.read_bytes() for p in original.iterdir()}


def test_missing_prerequisite_and_failed_scan_never_become_inactive(device_config):
    original = capture(device_config)
    item, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    waiting = runner.process(item, stage="understanding")
    assert waiting['status'] == 'waiting_for_prerequisite'
    assert waiting['prerequisite_stage'] == 'retention'
    assert runner.process(item, stage="retention")["status"] == "completed"
    backend.fail = True
    assert runner.process(item)["status"] == "failed"
    assert read_json(layout.index)["segments"] == []
    assert runner.process(item)["status"] == "failed"
    assert backend.vision_calls == 1
    backend.fail = False
    assert runner.process(item, retry=True)["status"] == "completed"
    assert original.joinpath("rgb.mp4").is_file()


def test_comments_refresh_understanding_without_repeating_vision(device_config):
    capture(device_config)
    item, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    runner.process(item)
    append_comment(runner.archive_root, layout.name, {"text": "实际备注", "start_us": item["recording_start_us"],
                   "end_us": item["recording_end_us"]}, Path(device_config["storage"]["local_runtime_root"]))
    runner.process(item)
    assert backend.vision_calls == 1
    assert backend.semantic_calls == 2


def test_corrupt_material_invalidates_reuse_and_preserves_old_receipt(device_config):
    capture(device_config)
    item, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    runner.process(item)
    index = read_json(layout.index)
    safe_child(layout.root, index["segments"][0]["scene_frames"][0]["path"]).write_bytes(b"corrupted")
    runner.process(item)
    assert backend.vision_calls == 2
    assert list((layout.receipts / item["recording_id"] / "history").glob("vision-*.json"))


def test_multiple_devices_reach_yolo_concurrently(device_config):
    capture(device_config)
    capture(device_config, "b_cam02")
    backend = FakeModels()
    backend.barrier = threading.Barrier(2)
    result = DeviceDayRunner(device_config, backend).run_once()
    assert all(r["status"] == "completed" for r in result["results"])
    assert backend.vision_calls == 2


def test_durable_fifo_keeps_twelve_hours_of_ten_cameras(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    path = tmp_path / "queue.sqlite3"
    queue = DeviceDayQueue(path)
    for ordinal in range(480):
        queue.enqueue({"recording_id": f"r{ordinal:04d}", "configured_role": "first_person",
                       "duration_seconds": 900, "archive_date": "2026-09-10"}, "v1")
    first = queue.claim("crashed-worker")
    with queue.connect() as db:
        db.execute("UPDATE recordings SET lease_until=0 WHERE recording_id=?", (first["recording_id"],))
    recovered = DeviceDayQueue(path)
    assert recovered.claim("recovered")["recording_id"] == first["recording_id"]
    recovered.finish("recovered", first["recording_id"], {"status": "completed"}, 20)
    assert recovered.snapshot()["counts"] == {"completed": 1, "queued": 479}
    assert recovered.snapshot()["queued_capture_window_seconds"] == 479 * 900
    assert recovered.snapshot()["queued_media_seconds"] is None
    assert recovered.claim("next")["recording_id"] == "r0001"


def test_slow_understanding_does_not_block_vision_for_later_slices(device_config):
    capture(device_config)
    capture(device_config, "b_cam02")
    device_config["device_day"].update(vision_workers=1, understanding_workers=1)
    backend = FakeModels()
    second_vision = threading.Event()
    original_vision, original_understand = backend.vision, backend.understand

    def vision(*args):
        result = original_vision(*args)
        if backend.vision_calls == 2:
            second_vision.set()
        return result

    def understand(*args):
        assert second_vision.wait(timeout=5), "Semantic work occupied the only vision worker"
        return original_understand(*args)

    backend.vision, backend.understand = vision, understand
    runner = DeviceDayRunner(device_config, backend)
    result = runner.run_once()
    assert all(r["status"] == "completed" for r in result["results"])
    assert backend.vision_calls == backend.semantic_calls == 2
    assert all(q["counts"] == {"completed": 2} for q in result["queue"].values())


def test_source_change_refuses_overwriting_retained_original(device_config):
    original = capture(device_config)
    item, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(item, stage="retention")
    (original / "rgb.mp4").write_bytes(b"different-media")
    os.utime(original / "rgb.mp4", (time.time() - 1000, time.time() - 1000))
    changed = scan_recordings(device_config)["recordings"][0]
    assert runner.process(changed)["status"] == "failed"
    assert (layout.raw / "10-00-00_10-15-00.mp4").read_bytes() == b"synthetic-fixture-not-real-media"


def test_cleanup_is_disabled_and_requires_matching_durable_receipts(device_config):
    original = capture(device_config)
    item, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(item, stage="retention")
    runner.process(item, stage="vision")
    retention = read_json(runner._receipt(layout, item, "retention"))
    vision = read_json(runner._receipt(layout, item, "vision"))
    assert delete_capture_video(layout.root, retention, vision)["deleted"] is False
    assert (original / "rgb.mp4").exists()
    with pytest.raises(ValueError, match="bind"):
        delete_capture_video(layout.root, retention, vision | {"retention_digest": "wrong"}, enabled=True)
    # Positive cleanup test acts exclusively on this test's owned temporary source.
    assert delete_capture_video(layout.root, retention, vision, enabled=True)["deleted"]
    assert not (original / "rgb.mp4").exists()
    assert (original / "frames.csv").exists()
    assert (layout.raw / "10-00-00_10-15-00.mp4").exists()


def test_unknown_role_and_incomplete_capture_are_not_inferred(device_config):
    capture(device_config, "unmapped_cam01")
    item, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    assert runner.process(item)["status"] == "needs_camera_role"
    assert runner.process(item | {"available": False, "processable": False})["status"] == "waiting_for_capture"


def test_windows_and_coverage_do_not_turn_decode_gaps_into_inactivity():
    assert windows(1801000, 900) == [(0, 900000), (900000, 1800000), (1800000, 1801000)]
    frames = [FrameEvidence(view_id="a", role=ViewRole.FIRST_PERSON, frame_index=i, local_ms=i * 1000,
                            width=2, height=2) for i in range(10)]
    assert check_coverage(frames, 0, 10000, 1)["coverage_ratio"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        check_coverage(frames[:1], 0, 10000, 1)


def test_model_rejects_invented_frames_out_of_window_steps_and_comments():
    images = [{"frame_id": "f1", "local_ms": 500}]
    payload = {"summary": "可见场景", "activity_observed": "inactive", "steps": [],
               "frame_observations": [{"frame_id": "f1", "text": "场景"}], "uncertainties": []}
    assert validate_understanding(payload, images, [], 0, 1000, 1)["steps"] == []
    bad = deepcopy(payload)
    bad["frame_observations"][0]["frame_id"] = "invented"
    with pytest.raises(ValueError, match="supplied"):
        validate_understanding(bad, images, [], 0, 1000, 1)
    for step in (
        {"start_ms": 0, "end_ms": 2000, "frame_ids": ["f1"]},
        {"start_ms": 0, "end_ms": 1000, "frame_ids": ["f1"], "comment_ids": ["invented"]},
    ):
        bad = payload | {"steps": [step | {"description": "测试", "basis": "observed"}]}
        with pytest.raises(ValueError):
            validate_understanding(bad, images, [], 0, 1000, 1)


def test_browse_links_and_path_confinement(device_config, tmp_path):
    capture(device_config)
    item, layout = item_and_layout(device_config)
    DeviceDayRunner(device_config, FakeModels()).process(item)
    app = FastAPI()
    service = DeviceDayService(lambda: device_config, threading.Lock())
    install_routes(app, lambda: device_config, service)
    client = TestClient(app)
    assert client.get("/api/device-days").json()["archives"][0]["archive"] == layout.name
    assert client.get("/device-days").status_code == 200
    browser = client.get(f"/device-days/{layout.name}")
    assert browser.status_code == 200
    assert all(name in browser.text for name in DIRECTORIES)
    assert '场景采样帧' in browser.text and '动作关键帧：0' in browser.text
    report = client.get(f"/api/device-days/{layout.name}/files/{DIRECTORIES[3]}/LaboratoryDailyReport.html")
    assert report.status_code == 200 and "测试关键帧文字" in report.text
    frame = read_json(layout.index)["segments"][0]["scene_frames"][0]["path"]
    assert client.get(f"/api/device-days/{layout.name}/files/{frame}").content == b"synthetic-frame"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"private")
    (layout.raw / "escape.jpg").symlink_to(outside)
    assert client.get(f"/api/device-days/{layout.name}/files/{DIRECTORIES[0]}/escape.jpg").status_code == 404


def test_live_inventory_cannot_clear_unreadable_historical_recording(device_config):
    capture(device_config)
    inventory = scan_recordings(device_config)
    service = DeviceDayService(lambda: device_config, threading.Lock())
    missing = "a_cam01/2026-09-09/182433/rgb.mp4"
    failure = {"path": missing, "message": "NAS metadata unavailable"}
    service.observe(device_config, {"recordings": [], "errors": [failure]})
    service.observe(device_config, inventory | {"errors": []})
    app = FastAPI()
    install_routes(app, lambda: device_config, service)
    client = TestClient(app)
    assert client.get("/api/device-days").json()["discovery_errors"] == [failure]
    readable = inventory["recordings"][0] | {"relative_path": missing, "recording_id": "recovered"}
    service.observe(device_config, {"recordings": [readable], "errors": []})
    assert client.get("/api/device-days").json()["discovery_errors"] == []


def test_queue_eligibility_and_actual_media_duration(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    queue = DeviceDayQueue(tmp_path / "queue.sqlite3")
    for name in ("blocked", "ready"):
        queue.enqueue({"recording_id": name, "configured_role": "first_person", "duration_seconds": 1000}, "v1")
    assert queue.claim("worker", allowed={"ready"})["recording_id"] == "ready"
    queue.finish("worker", "ready", {"status": "completed", "media_duration_seconds": 100}, 10)
    assert queue.snapshot()["media_seconds_per_worker_second"] == 10
    assert queue.snapshot()["queued_media_duration_unknown_count"] == 1
    assert queue.pending()[0]["recording_id"] == "blocked"


@pytest.mark.parametrize("frame_count,start,end", [(1, 0, 1000 / 30), (27006, 900000, 900200)])
def test_short_tail_frame_bounds(device_config, tmp_path, monkeypatch, frame_count, start, end):
    from types import SimpleNamespace
    import numpy as np
    from visioncortex.device_day_models import DeviceDayModels
    model = DeviceDayModels(device_config)
    source = tmp_path / "rgb.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(model, "_probe", lambda _: SimpleNamespace(fps=30, frame_count=frame_count))
    requested = []
    def read_frame(_reader, _source, timestamp):
        assert 0 <= timestamp <= (frame_count - 1) * 1000 / 30
        requested.append(timestamp)
        return np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr("visioncortex.video_io.ViewFrameReader.read_path", read_frame)
    layout = DeviceDayLayout(tmp_path, "a_cam01", 1789005600000000)
    source = layout.raw / "rgb.mp4"
    result = model._frames(layout, source, start, end, 8, layout.processed / "KeyFrames")
    assert len(result) == len(set(requested))
    assert len({r["frame_id"] for r in result}) == len(result)
    if frame_count == 1:
        assert requested == [0]


def test_existing_audit_rejects_weak_office_contact(device_config, tmp_path, monkeypatch):
    from visioncortex.device_day_models import DeviceDayModels
    from visioncortex.schemas import ActionCandidate, ActionType, ViewInput, VideoInfo
    capture(device_config)
    record, layout = item_and_layout(device_config)
    source = Path(record['video_path'])
    with Path(record['frames_path']).open('a') as clock:
        clock.write(f"{record['recording_start_us'] + 900000000},27000\n")
    view = ViewInput(view_id=record['camera_key'], role=ViewRole.FIRST_PERSON,
                     video=source, timestamps_csv=Path(record['frames_path']))
    info = VideoInfo(path=source, duration_ms=900200, fps=30, frame_count=27006,
                     width=1280, height=800, size_bytes=source.stat().st_size)
    fine = tmp_path / 'fine.jsonl'
    fine.write_text('')
    candidate = ActionCandidate(candidate_id='weak-office-contact', action_type=ActionType.HAND_OBJECT_CONTACT,
                                view_id=view.view_id, role=view.role, local_start_ms=395150, local_end_ms=397450,
                                global_start_ms=395150, global_end_ms=397450, key_global_ms=396000,
                                objects=['gloved_hand', 'sample_bottle'], confidence=.47868,
                                uncertainty=['No complete approach was observed'])
    monkeypatch.setattr('visioncortex.coarse_recall.generate_open_vocabulary_fine_candidates',
                        lambda *a: ([], {'formal_evidence_ready': True}))
    monkeypatch.setattr('visioncortex.movement_verification.verify_movement_candidates', lambda *a: {})
    intervals, audit = DeviceDayModels(device_config)._audit_activity(
        view, info, {view.view_id: fine}, [candidate], [], [(391000, 399000)])
    assert intervals == []
    assert audit['rejected']
    assert audit['coarse_only_can_label_activity'] is False
    assert audit['formal_experiment_segments'] == []


def test_device_activity_does_not_require_or_promote_formal_action(device_config):
    from types import SimpleNamespace
    from visioncortex.actions import activity_seed_intervals as device_activity_intervals
    from visioncortex.schemas import ActionType
    evidence = [{"frame_index": n, "interaction_state": {"approach_confirmed": n == 0}}
                for n in range(8)]
    event = SimpleNamespace(event_id="fine-contact", action_type=ActionType.HAND_OBJECT_CONTACT,
                            accepted=False, confidence=.70, global_start_ms=3000, global_end_ms=4500,
                            candidates=[SimpleNamespace(evidence=evidence, objects=["hand", "beaker"])])
    coarse = [SimpleNamespace(candidate_id="coarse", local_start_ms=2000, local_end_ms=5000)]
    intervals, decisions = device_activity_intervals([event], coarse, [(1000, 7000)], device_config)
    assert intervals == [(1000, 7000)]
    assert decisions[0]["physical_action_confirmed"] is False
    assert event.accepted is False
    event.confidence = .48
    assert device_activity_intervals([event], coarse, [(1000, 7000)], device_config)[0] == []
    event.confidence = .9
    evidence[0]["interaction_state"]["approach_confirmed"] = False
    assert device_activity_intervals([event], coarse, [(1000, 7000)], device_config)[0] == []
    evidence[0]["interaction_state"]["approach_confirmed"] = True
    assert device_activity_intervals([event], [], [(1000, 7000)], device_config)[0] == []


def test_first_person_motion_recall_scans_short_source_without_hand_detections(device_config, tmp_path):
    from visioncortex.device_day_models import device_scan_plan
    from visioncortex.schemas import BoxEvidence, ViewInput, VideoInfo
    ledger = tmp_path / "coarse.jsonl"
    frames = [FrameEvidence(view_id="fp", role=ViewRole.FIRST_PERSON, frame_index=n * 15,
                            local_ms=n * 500, global_ms=n * 500, width=640, height=480,
                            motion_score=20 if n in {20, 21} else 0,
                            raw_motion_score=20 if n in {20, 21} else 0,
                            detections=[BoxEvidence(class_id=0, class_name="beaker", confidence=.9,
                                                    xyxy_norm=[.2, .2, .4, .4], track_id=1)])
              for n in range(60)]
    ledger.write_text("\n".join(frame.model_dump_json() for frame in frames))
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=tmp_path / "fixture.mp4")
    info = VideoInfo(path=view.video, duration_ms=30000, fps=30, frame_count=900, width=640, height=480, size_bytes=1)
    device_config["performance"].update(auto_exhaustive_short_timeline_enabled=True,
                                        auto_exhaustive_short_timeline_seconds=600)
    candidates, fine_windows, plan = device_scan_plan(view, {"fp": ledger}, device_config, info, 0, 30000, 2)
    assert any(c.candidate_id.startswith("BURST-") for c in candidates)
    assert fine_windows == [(0, 30000)] and plan["first_person_motion_candidates"] > 0
    for frame in frames:
        frame.motion_score = frame.raw_motion_score = 0
    ledger.write_text("\n".join(frame.model_dump_json() for frame in frames))
    assert device_scan_plan(view, {"fp": ledger}, device_config, info, 0, 30000, 2)[1] == []


def test_continuity_preserves_supported_pause_but_never_opens_on_coarse_motion(device_config):
    from types import SimpleNamespace as NS
    from visioncortex.actions import build_activity_intervals as continuous_device_activity, merge_activity_intervals as merge_intervals
    from visioncortex.schemas import ActionType
    def event(name, start, end, accepted=False):
        return NS(event_id=name, action_type=ActionType.DEVICE_PANEL_OPERATION, confidence=.8 if accepted else .5,
                  global_start_ms=start, global_end_ms=end, accepted=accepted,
                  candidates=[NS(objects=["balance", "gloved_hand"],
                                 evidence=[{"frame_index": n} for n in range(4)])])
    anchor, following, remote = event("anchor", 35000, 36000, True), event("following", 86000, 87000), event("remote", 155000, 156000)
    motion = NS(candidate_id="motion", local_start_ms=65000, local_end_ms=77000,
                objects=["balance"], evidence=[{"coarse_motion": 20}])
    result, decisions = continuous_device_activity([anchor, following, remote], [motion], [(0, 163000)], device_config)
    assert merge_intervals(result, 0, 163000) == [(33000, 90000)]
    assert decisions[-1]["fine_event_ids"] == ["anchor", "following"]
    assert anchor.accepted is True and following.accepted is False
    assert decisions[-1]["physical_action_confirmed"] is False
    assert continuous_device_activity([following], [motion], [(0, 163000)], device_config)[0] == []
    # A terminal motion burst cannot extend a clip without a later fine interaction.
    assert merge_intervals(continuous_device_activity([anchor], [motion], [(0, 163000)], device_config)[0], 0, 163000) == [(33000, 39000)]


def test_shared_activity_projection_preserves_camera_and_clock_identity(device_config, tmp_path):
    from visioncortex.actions import build_view_activity_intervals
    from visioncortex.schemas import ActionCandidate, ActionType, AlignmentTransform, EvidenceEvent, ViewInput
    views = [ViewInput(view_id=name, role=ViewRole.FIRST_PERSON, video=tmp_path / f"{name}.mp4") for name in ("a", "b")]
    transforms = {v.view_id: AlignmentTransform(view_id=v.view_id, reference_view_id="a", scale=1.1, offset_ms=10000) for v in views}
    candidate = ActionCandidate(candidate_id="a-only", view_id="a", role=ViewRole.FIRST_PERSON,
        action_type=ActionType.DEVICE_PANEL_OPERATION, local_start_ms=3000, local_end_ms=4000,
        global_start_ms=13300, global_end_ms=14400, key_global_ms=13500,
        objects=["balance", "gloved_hand"], confidence=.9, evidence=[{"frame_index": n} for n in range(4)])
    event = EvidenceEvent(event_id="event-a", action_type=candidate.action_type, global_start_ms=13300,
        global_end_ms=14400, key_global_ms=13500, objects=candidate.objects, confidence=.9,
        accepted=True, audit_reason="fixture-only", supporting_views=["a"],
        supporting_roles=[ViewRole.FIRST_PERSON], candidates=[candidate])
    original = event.model_dump()
    result = build_view_activity_intervals([event], [candidate], views, transforms, {"a": [(0, 20000)], "b": [(0, 20000)]}, device_config)
    assert result["a"]["intervals"][0] == pytest.approx((transforms["a"].to_local(11300), transforms["a"].to_local(17400)))
    assert result["b"]["intervals"] == []
    assert event.model_dump() == original


def test_fine_index_uses_shared_local_storage_and_publishes_verified_snapshot(device_config, tmp_path):
    from visioncortex.candidate_index import local_frame_index_path
    from visioncortex.device_day_models import DeviceDayModels
    from visioncortex.schemas import ViewInput, VideoInfo
    view = ViewInput(view_id="a", role=ViewRole.FIRST_PERSON, video=tmp_path / "fixture.mp4")
    info = VideoInfo(path=view.video, duration_ms=1000, fps=30, frame_count=30, width=640, height=480, size_bytes=1)
    ledger = tmp_path / "Fine.jsonl"
    ledger.write_text("\n".join(FrameEvidence(view_id="a", role=view.role, frame_index=i*15,
        local_ms=i*500, global_ms=i*500, width=640, height=480).model_dump_json() for i in range(2)))
    device_config["performance"].update(fine_frame_index_enabled=True, fine_coverage_gate_enabled=True, detection_fps=2)
    layout = DeviceDayLayout(Path(device_config["storage"]["archive_root"]), "a_cam01", 1789005600000000,
                             Path(device_config["storage"]["local_cache_root"]))
    paths, report, artifacts = DeviceDayModels(device_config)._index_fine(layout, {"recording_id": "fixture"}, "key", 0,
                                                                        view, info, {"a": ledger}, [(0, 1000)])
    snapshot = layout.backend_root / next(r["path"] for r in artifacts if r["path"].endswith(".sqlite3"))
    live_index = local_frame_index_path(device_config, snapshot.parent, snapshot.name)
    assert live_index.is_relative_to(Path(device_config["storage"]["local_runtime_root"]).resolve())
    assert snapshot.read_bytes() == live_index.read_bytes()
    assert report["formal_evidence_ready"] is True and paths["a"].is_file()


def test_scan_cache_reuses_only_verified_complete_evidence(device_config, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from visioncortex.device_day_models import DeviceDayModels
    model = DeviceDayModels(device_config)
    calls = []
    monkeypatch.setattr(model, 'scan_identity', lambda *a: {'windows': [(0, 1000)], 'source': 'fixture'})
    def scan(views, infos, transforms, directory, *args, **kwargs):
        calls.append(1)
        directory.mkdir(parents=True, exist_ok=True)
        ledger = directory / 'test.detections.jsonl'
        ledger.write_text('fixture')
        return {'test': ledger}
    monkeypatch.setattr('visioncortex.detection.scan_videos', scan)
    args = (SimpleNamespace(view_id='test', role=ViewRole.FIRST_PERSON, segments=[]), None, None, {}, 'coarse', [(0, 1000)], 2, 640)
    _, directory, first = model._scan(*args)
    _, _, second = model._scan(*args)
    assert not first['reused'] and second['reused'] and len(calls) == 1
    (directory / 'test.detections.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='cache was modified'):
        model._scan(*args)


def test_publication_keeps_current_clips_and_relocates_old_versions(device_config):
    import shutil
    from visioncortex.device_day import exclusive
    from visioncortex.device_day_publication import reconcile_outputs, write_archive_guide
    capture(device_config)
    item, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(item)
    current = safe_child(layout.root, read_json(layout.index)['segments'][0]['json_path']).parent
    old = current.with_name('09-00-00_09-01-00_experiment_activity_old')
    shutil.copytree(current, old)
    source = Path(item['video_path'])
    before = source.read_bytes()
    with exclusive(runner.runtime_root / 'locks' / f"{item['recording_id']}.vision.lock"):
        assert reconcile_outputs(runner)['relocated'] == []
    result = reconcile_outputs(runner)
    assert len(result['relocated']) == 1
    assert current.is_dir() and not old.exists()
    moved = safe_child(runner.backend_root, result['relocated'][0]['to'])
    assert moved.is_dir() and source.read_bytes() == before
    assert [a['sha256'] for a in result['relocated'][0]['artifacts_before']] == [a['sha256'] for a in result['relocated'][0]['artifacts_after']]
    write_archive_guide(runner)
    assert 'NoExperimentActivity' in (runner.archive_root / 'Readme.html').read_text()


def test_inactive_scene_samples_cannot_be_action_keyframes(device_config):
    from visioncortex.device_day_contract import validate_segment
    capture(device_config)
    item, layout = item_and_layout(device_config)
    DeviceDayRunner(device_config, FakeModels()).process(item)
    segment = read_json(layout.index)['segments'][0]
    assert segment['key_frames'] == []
    assert segment['scene_frames'][0]['frame_kind'] == 'scene_sample'
    validate_segment(segment)
    segment['key_frames'] = segment['scene_frames']
    with pytest.raises(ValueError, match='Inactive intervals'):
        validate_segment(segment)
    segment['activity'] = 'active'
    with pytest.raises(ValueError, match='Action keyframes'):
        validate_segment(segment)


def test_layout_migration_moves_archive_outputs_without_touching_capture(device_config):
    from visioncortex.device_day_migration import migrate_layout
    capture(device_config)
    item, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(item, stage="retention")
    receipt_dir = layout.receipts / item["recording_id"]
    retained = read_json(receipt_dir / "retention.json")
    before = {s["original_path"]: Path(s["original_path"]).read_bytes() for s in retained["sources"]}
    for source in retained["sources"]:
        path = safe_child(layout.root, source["retained"]["path"])
        old_path = layout.raw / item["recording_id"] / path.name
        old_path.parent.mkdir(parents=True, exist_ok=True)
        path.rename(old_path)
        source["retained"] = artifact(layout.root, old_path)
    atomic_json(receipt_dir / "retention.json", retained)
    old_receipts = layout.processed / "receipts" / item["recording_id"]
    old_receipts.parent.mkdir()
    receipt_dir.rename(old_receipts)
    result = migrate_layout(runner, item)
    assert result["status"] == "completed" and not result["capture_sources_changed"]
    assert not (layout.processed / "receipts").exists()
    assert not (layout.raw / item["recording_id"]).exists()
    assert (layout.raw / "10-00-00_10-15-00.mp4").read_bytes() == before[item["video_path"]]
    assert all(Path(path).read_bytes() == value for path, value in before.items())
    assert migrate_layout(runner, item) == result


def test_backend_evidence_is_verified_before_reuse(device_config):
    capture(device_config)
    item, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(item, stage="retention")
    runner.process(item, stage="vision")
    path = runner._receipt(layout, item, "vision")
    receipt = read_json(path)
    audit = layout.receipts / item["recording_id"] / "YOLO" / "audit.json"
    atomic_json(audit, {"synthetic": True})
    receipt["audit_artifacts"] = [layout.backend_artifact(audit)]
    atomic_json(path, receipt)
    assert runner._load(path, receipt["key"], layout)
    audit.write_text("tampered")
    assert runner._load(path, receipt["key"], layout) is None


def test_existing_monitor_automatically_processes_new_slices_during_slow_understanding(device_config, monkeypatch):
    from visioncortex import device_day_service as module
    backend = FakeModels()
    original_understand = backend.understand
    semantic_started, release = threading.Event(), threading.Event()
    def slow_understand(*args):
        semantic_started.set()
        assert release.wait(8)
        return original_understand(*args)
    backend.understand = slow_understand
    runner = DeviceDayRunner(device_config, backend)
    monkeypatch.setattr(module, "DeviceDayRunner", lambda settings: runner)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 10**12})())
    monkeypatch.setattr("visioncortex.nas_recordings.scan_recordings", scan_recordings)
    capture(device_config)
    inventory = scan_recordings(device_config)
    service = DeviceDayService(lambda: device_config, threading.Lock())
    service.observe(device_config, inventory)
    service.start()
    try:
        assert semantic_started.wait(8), service.last_result
        # The second camera arrives after the first is already at the slow model.
        capture(device_config, "b_cam02")
        updated = scan_recordings(device_config)
        service.observe(device_config, updated)
        deadline = time.monotonic() + 8
        while backend.vision_calls != 2:
            assert time.monotonic() < deadline, "Slow semantics blocked new YOLO work"
            time.sleep(.05)
        release.set()
        while service.monitor_status(updated)["batches"][0]["processing"]["status"] != "completed":
            assert time.monotonic() < deadline
            time.sleep(.05)
        service.observe(device_config, updated)
        time.sleep(.1)
        assert backend.vision_calls == 2
        from visioncortex.observed_inventory import read_inventory
        saved = read_inventory(runner.runtime_root)
        assert saved["source"] == "nas_recording_monitor" and len(saved["recordings"]) == 2
    finally:
        release.set()
        service.stop()


def test_storage_wait_preserves_source_and_blocks_only_new_retention(device_config, monkeypatch):
    from visioncortex import device_day_service as module
    original = capture(device_config)
    item, layout = item_and_layout(device_config)
    layout.create()
    runner = DeviceDayRunner(device_config, FakeModels())
    service = DeviceDayService(lambda: device_config, threading.Lock())
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 100})())
    capacity = service._storage_available(runner, {"recordings": [item]})
    assert not capacity["ready"] and capacity["required_bytes"] > capacity["available_bytes"]
    assert (original / "rgb.mp4").is_file()
    assert not list(layout.raw.glob("*.mp4"))


def test_new_short_slice_uses_idle_slot_while_previous_slice_is_still_running(device_config, monkeypatch):
    from visioncortex import device_day_service as module
    backend = FakeModels()
    original = backend.vision
    started, second, third, release = [threading.Event() for _ in range(4)]
    intervals = []
    def bounded_vision(layout, retention, key):
        record = retention["recording"]
        intervals.append(record["duration_seconds"])
        if record["duration_seconds"] == 300:
            started.set()
            assert release.wait(20)
        elif record["duration_seconds"] == 120:
            second.set()
        else:
            third.set()
        return original(layout, retention, key)
    backend.vision = bounded_vision
    runner = DeviceDayRunner(device_config, backend)
    monkeypatch.setattr(module, "DeviceDayRunner", lambda settings: runner)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 10**12})())
    service = DeviceDayService(lambda: device_config, threading.Lock())
    capture(device_config, duration=300)
    service.observe(device_config, scan_recordings(device_config))
    service.start()
    try:
        assert started.wait(8)
        capture(device_config, start=1789005900000000, duration=120, label="100500")
        service.observe(device_config, scan_recordings(device_config))
        assert second.wait(8), "An idle GPU slot waited for the preceding slice"
        capture(device_config, start=1789006020000000, duration=180, label="100700")
        service.observe(device_config, scan_recordings(device_config))
        assert third.wait(8), "A freed GPU slot was not replenished independently"
        assert not release.is_set()
        assert sorted(intervals) == [120, 180, 300]
    finally:
        release.set()
        service.stop()


def test_live_uploads_take_precedence_over_historical_backfill(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    queue = DeviceDayQueue(tmp_path / "queue.db")
    queue.enqueue({"recording_id": "old", "configured_role": "first_person", "processing_priority": 1}, "v1")
    queue.enqueue({"recording_id": "live", "configured_role": "first_person", "processing_priority": 0}, "v1")
    assert queue.claim("worker")["recording_id"] == "live"
    assert queue.claim("worker")["recording_id"] == "old"


def test_final_naming_migration_preserves_media_and_old_evidence(device_config):
    from visioncortex.device_day_migration import migrate_names
    capture(device_config)
    item, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(item)
    old_names = ("1. meta video", "2. processed clips", "3. multimodal understanding",
                 "4. laboratory daily report", "5. comment")
    for new, old in zip(DIRECTORIES, old_names, strict=True):
        (layout.root / new).rename(layout.root / old)
    path = runner._receipt(layout, item, "retention")
    saved = read_json(path)
    for source in saved["sources"]:
        source["retained"]["path"] = source["retained"]["path"].replace("MetaVideo/", "1. meta video/")
    atomic_json(path, saved)
    migrated = migrate_names(runner, [item])
    assert migrated["status"] == "completed" and not migrated["capture_sources_changed"]
    from visioncortex.device_day_contract import verify_artifact
    assert all(verify_artifact(layout.root, move["artifact"] | {"path": move["to"]})
               for move in migrated["moves"] if "artifact" in move)
    assert set(p.name for p in layout.root.iterdir()) == set(DIRECTORIES)
    assert migrate_names(runner, [item]) == migrated
    assert runner.process(item)["status"] == "completed"


def test_request_completion_uses_the_same_priority_payload(device_config):
    capture(device_config)
    item, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    service = DeviceDayService(lambda: device_config, threading.Lock())
    service._runner = runner
    request = {"kind": "historical_backfill", "recordings": [item]}
    record = item | {"processing_priority": 1, "archive_date": "2026-09-10"}
    runner.process(record)
    assert service._request_complete(request)


@pytest.mark.parametrize("stage", ["retention", "vision", "stt", "understanding", "report"])
@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_pending_request_does_not_read_nas_on_dispatch_thread(device_config, monkeypatch, stage, status):
    runner = DeviceDayRunner(device_config, FakeModels())
    service = DeviceDayService(lambda: device_config, threading.Lock())
    service._runner = runner
    records = [{"recording_id": name, "configured_role": "first_person"} for name in ("older", "pending")]
    queue = runner.queues[stage]
    queue.enqueue(records[1], "revision")
    if status != "queued":
        queue.claim("worker")
        if status == "failed":
            queue.finish("worker", "pending", {"status": "failed"}, 1)
    monkeypatch.setattr(runner, "layout", lambda _: pytest.fail("Pending request must not inspect NAS receipts"))
    assert not service._request_complete({"recordings": records})


def test_completed_request_queues_do_not_replace_receipt_verification(device_config):
    capture(device_config)
    item, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    service = DeviceDayService(lambda: device_config, threading.Lock())
    service._runner = runner
    for queue in runner.queues.values():
        queue.enqueue(item, "revision")
        assert queue.claim("worker")
        queue.finish("worker", item["recording_id"], {"status": "completed"}, 1)
    assert not service._request_complete({"recordings": [item]})


def test_case_only_migration_uses_unambiguous_directory_entries(tmp_path):
    from visioncortex.device_day_migration import _canonical_case
    root = tmp_path / "Archive"
    source = root / "Metavideo" / "Recordingready.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"preserved-source-metadata")
    journal, path = {}, tmp_path / "Migration.json"
    _canonical_case(root, "MetaVideo/RecordingReady.json", journal, path)
    assert [p.name for p in root.iterdir()] == ["MetaVideo"]
    assert (root / "MetaVideo/RecordingReady.json").read_bytes() == b"preserved-source-metadata"
    assert all(row["completed"] for row in journal["case_moves"])
    (root / "Metavideo").mkdir()
    with pytest.raises(ValueError, match="colliding"):
        _canonical_case(root, "MetaVideo/RecordingReady.json", journal, path)


def test_fixed_report_templates_preserve_existing_columns_and_do_not_call_model(device_config):
    from visioncortex.device_day_reports import render_day
    capture(device_config)
    item, layout = item_and_layout(device_config)
    backend = FakeModels()
    DeviceDayRunner(device_config, backend).process(item)
    calls = backend.semantic_calls
    render_day(layout, read_json(layout.index))
    assert backend.semantic_calls == calls
    report = read_json(layout.reports / "LaboratoryDailyReport.json")
    assert report["template_id"] == "VC-DEVICE-DAY-REPORT-V1"
    assert report["base_template_id"] == "VC-LAB-DAILY-REPORT-V2"
    assert report["additional_model_tokens"] == 0 and not report["additional_mllm_calls"]
    base = read_json(Path(__file__).parents[1] / "src/visioncortex/templates/VC-LAB-DAILY-REPORT-V2.json")
    assert report["sections"] == base["sections"]
    semantic = read_json(layout.understanding / "Understanding.json")
    assert semantic["template_id"] == "VC-DEVICE-UNDERSTANDING-REPORT-V1"
    assert '按时间的步骤与场景理解' in (layout.understanding / "UnderstandingReport.html").read_text()
    assert '关注事项与交接' in (layout.reports / "LaboratoryDailyReport.html").read_text()


def test_scene_response_cache_binds_images_times_and_model(device_config):
    from visioncortex import device_day_semantic_cache as cache
    metadata = {'source_ref': {'sha256': 'a' * 64}, 'activity': 'inactive', 'start_ms': 0, 'end_ms': 1000,
                'frames': [{'frame_id': 'frame-1', 'local_ms': 500, 'sha256': 'b' * 64}],
                'comments': [], 'protocol': None}
    key = cache.identity(device_config, 'prompt', metadata)
    response = {'status': 'completed', 'summary': 'deterministic fixture'}
    cache.save(device_config, key, response, provenance={'actual_call': False})
    assert cache.load(device_config, key) == response
    metadata['frames'][0]['local_ms'] = 700
    assert cache.load(device_config, cache.identity(device_config, 'prompt', metadata)) is None
    changed = deepcopy(device_config)
    changed['mllm']['model'] = 'different-model'
    assert cache.load(changed, cache.identity(changed, 'prompt', metadata)) is None
    (cache.location(device_config, key) / 'model-result.json').write_text('{}')
    with pytest.raises(ValueError, match='modified'):
        cache.load(device_config, key)


def test_comments_and_audio_are_not_borrowed_across_dates(device_config):
    from visioncortex.device_day import load_context
    capture(device_config)
    item, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(item)
    later = DeviceDayLayout(runner.archive_root, item['camera_key'], item['recording_start_us'] + 86400000000, runner.backend_root)
    later.create()
    # Even a wrongly timestamped comment in another day is outside this archive.
    (later.comments / 'Comment.jsonl').write_text(
        '{"comment_id":"other-day","start_us":1789005600000000,"end_us":1789006500000000,"text":"different day"}\n')
    assert load_context(layout, item)['comments'] == []
    assert read_json(layout.index)['recordings'][0]['transcription']['outcome'] == 'no_audio'


def test_capture_catalog_keeps_csv_metadata_logs_and_nested_files_but_not_depth(device_config):
    from visioncortex.device_day_capture_files import reconcile_capture_files
    from visioncortex.device_day_browser import render_archive
    from visioncortex.device_day_contract import file_hash
    source = capture(device_config)
    (source / 'ffmpeg.log').write_text('capture log\n')
    (source / 'depth.mkv').write_bytes(b'excluded-depth-fixture')
    (source / 'sensor_data').mkdir()
    (source / 'sensor_data' / 'temperature_data.csv').write_text('timestamp,value\n1,20\n')
    for path in source.rglob('*'):
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    before = {str(p): file_hash(p) for p in source.rglob('*') if p.is_file()}
    item, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    runner.process(item)
    model_calls = (backend.vision_calls, backend.semantic_calls)
    results = reconcile_capture_files(runner, date='2026-09-10')
    assert len(results) == 1 and 'manifest' in results[0]
    index = read_json(layout.index)
    ref = index['recordings'][0]['capture_manifest']
    catalog = read_json(safe_child(layout.root, ref['path']))
    assert catalog['status'] == 'completed'
    assert {f['original_name'] for f in catalog['files']} == {
        'rgb.mp4', 'frames.csv', 'meta.json', 'recording_ready.json',
        'ffmpeg.log', 'sensor_data/temperature_data.csv'}
    assert catalog['excluded'] == [{'original_name': 'depth.mkv', 'reason': 'depth_video_excluded_by_user'}]
    assert not list(layout.root.rglob('*.mkv'))
    for f in catalog['files']:
        assert file_hash(safe_child(layout.root, f['retained']['path'])) == before[f['original_path']]
    assert any(f['retained']['path'].endswith('/Capture/SensorData/TemperatureData.csv') for f in catalog['files'])
    page = render_archive(layout.root, index)
    assert 'Frames.csv' in page and 'Ffmpeg.log' in page and 'CaptureFiles.json' in page
    manifest_mtime = safe_child(layout.root, ref['path']).stat().st_mtime_ns
    reconcile_capture_files(runner, date='2026-09-10')
    assert safe_child(layout.root, ref['path']).stat().st_mtime_ns == manifest_mtime
    assert (backend.vision_calls, backend.semantic_calls) == model_calls
    assert {str(p): file_hash(p) for p in source.rglob('*') if p.is_file()} == before
    # Later auxiliary uploads are discovered without invoking vision again.
    (source / 'new_sensor.csv').write_text('timestamp,pressure\n2,100\n')
    assert reconcile_capture_files(runner, date='2026-09-10')
    assert read_json(safe_child(layout.root, ref['path']))['status'] == 'waiting'
    os.utime(source / 'new_sensor.csv', (time.time() - 1000, time.time() - 1000))
    reconcile_capture_files(runner, date='2026-09-10')
    assert read_json(safe_child(layout.root, ref['path']))['status'] == 'completed'
    assert (backend.vision_calls, backend.semantic_calls) == model_calls
def test_api_readiness_ignores_old_completed_receipts_without_current_outputs():
    from copy import deepcopy
    from visioncortex.device_day_service import published_recording_status

    stages = {name: {"status": "completed"} for name in ("retention", "vision", "stt", "understanding", "report")}
    data = {"recordings": [{"recording_id": rid, "stages": stages} for rid in ("current", "old", "partial")],
            "segments": [{"recording_id": "current", "segment_id": "a"},
                         {"recording_id": "partial", "segment_id": "b"},
                         {"recording_id": "partial", "segment_id": "c"}],
            "understandings": [{"segment_id": "a"}, {"segment_id": "b"}]}
    original = deepcopy(data)
    result = published_recording_status(data)
    assert [r["current_results_ready"] for r in result["recordings"]] == [True, False, False]
    assert data == original


def test_queue_refreshes_same_revision_payload_without_resetting_work(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    queue = DeviceDayQueue(tmp_path / 'queue.sqlite3')
    original = {'recording_id': 'record', 'configured_role': 'first_person'}
    current = original | {'processing_priority': 0}
    queue.enqueue(original, 'same-visual-revision')
    queue.enqueue(current, 'same-visual-revision')
    assert queue.claim('worker') == current
    queue.enqueue(current | {'processing_priority': 1}, 'same-visual-revision')
    with queue.connect() as db:
        import json
        row = db.execute('SELECT * FROM recordings').fetchone()
        assert json.loads(row['payload']) == current
        assert row['status'] == 'running'
    queue.finish('worker', 'record', {'status': 'completed'}, 1)
    queue.enqueue(current | {'processing_priority': 1}, 'same-visual-revision')
    assert queue.claim('next-worker') is None
    with queue.connect() as db:
        row = db.execute('SELECT * FROM recordings').fetchone()
        assert json.loads(row['payload'])['processing_priority'] == 1
        assert row['status'] == 'completed'
        assert row['attempts'] == 1


@pytest.mark.parametrize("camera_count", [2, 9, 12])
def test_camera_lanes_claim_heads_and_do_not_wait_for_other_cameras(tmp_path, camera_count):
    from visioncortex.device_day_queue import DeviceDayQueue
    queue = DeviceDayQueue(tmp_path / 'lanes.sqlite3')
    # Deliberately enqueue later slices first: capture order wins per camera.
    for ordinal in (2, 1):
        for camera in range(camera_count):
            queue.enqueue({'recording_id': f'{camera}-{ordinal}', 'camera_key': f'camera{camera}',
                           'recording_start_us': ordinal * 1000, 'configured_role': 'first_person'}, 'v1')
    first = [queue.claim(f'worker{i}', camera_serial=True) for i in range(camera_count)]
    assert {r['recording_id'] for r in first} == {f'{i}-1' for i in range(camera_count)}
    assert queue.claim('extra', camera_serial=True) is None
    queue.finish('worker0', first[0]['recording_id'], {'status': 'completed'}, 1)
    next_slice = queue.claim('next', camera_serial=True)
    assert next_slice['camera_key'] == first[0]['camera_key']
    assert next_slice['recording_start_us'] == 2000


@pytest.mark.parametrize("camera_count,vision_limit", [(1, 2), (2, 4), (2, 2)])
def test_vision_camera_slots_keep_per_camera_and_shared_resource_limits(
    device_config, monkeypatch, tmp_path, camera_count, vision_limit
):
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from visioncortex.runtime_control import ResourceCoordinator
    from visioncortex.scan_scheduler import scan_views_concurrently
    from visioncortex.schemas import ViewInput

    device_config['device_day'].update(camera_lanes=True, vision_jobs_per_camera=2)
    device_config.setdefault('runtime', {})['resource_limits'] = {'vision': vision_limit}
    runner = DeviceDayRunner(device_config, backend=object())
    coordinator = ResourceCoordinator(runner.runtime_root.parent / 'state' / 'resources.sqlite3')
    records = [{'recording_id': f'camera{camera}-slice{slice_index}', 'camera_key': f'camera{camera}',
                'recording_start_us': slice_index * 1000, 'configured_role': 'first_person'}
               for camera in range(camera_count) for slice_index in range(3)]
    for record in records:
        runner.queue.enqueue(record, 'v1')
    monkeypatch.setattr(runner, '_prepare_stage', lambda *args: {r['recording_id'] for r in records})
    jobs = camera_count * 2
    started = threading.Barrier(jobs + 1)
    saturated, release, stop = threading.Event(), threading.Event(), threading.Event()
    lock = threading.Lock()
    active, maximum = Counter(), Counter()
    total_maximum = 0

    def process(record, **kwargs):
        started.wait(timeout=10)
        view = ViewInput(view_id=record['recording_id'], role='first_person', video=tmp_path / 'unread.mp4')

        def scan(*args, **kwargs):
            nonlocal total_maximum
            camera = record['camera_key']
            with lock:
                active[camera] += 1
                maximum[camera] = max(maximum[camera], active[camera])
                total_maximum = max(total_maximum, sum(active.values()))
                if sum(active.values()) == min(jobs, vision_limit):
                    saturated.set()
            try:
                assert release.wait(10)
                return {}
            finally:
                with lock:
                    active[camera] -= 1

        scan_views_concurrently(runner.config, [view], {}, {}, tmp_path / view.view_id,
                                phase='fine', scanner=scan)
        return {'recording_id': record['recording_id'], 'status': 'completed'}

    monkeypatch.setattr(runner, 'process', process)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner.run_once, {'recordings': records}, stage='vision', stop_event=stop)
        try:
            started.wait(timeout=10)
            assert saturated.wait(10)
            # All available camera slots are leased, while a third slice from
            # every camera remains queued. No extra caller may bypass the cap.
            assert runner.queue.claim('extra', camera_serial=True, camera_limit=2) is None
            admitted = [r for r in coordinator.snapshot() if r['state'] == 'running']
            assert sum(r['units'] for r in admitted) == min(jobs, vision_limit)
        finally:
            stop.set()
            release.set()
            started.abort()
        result = future.result(timeout=10)
    assert len(result['results']) == jobs
    assert all(r['status'] == 'completed' for r in result['results'])
    assert total_maximum == min(jobs, vision_limit)
    assert all(count <= 2 for count in maximum.values())
    assert runner.queue.snapshot()['counts'] == {'completed': jobs, 'queued': camera_count}


def test_stage_preparation_shared_until_inventory_or_parent_changes(device_config, monkeypatch):
    runner = DeviceDayRunner(device_config, backend=object())
    calls = []
    monkeypatch.setattr(runner, '_build_stage_inventory', lambda inventory, stage, date: calls.append(stage) or {'r'})
    inventory = {'recordings': [{'recording_id': 'r'}]}
    for _ in range(9):
        assert runner._prepare_stage(inventory, 'vision', None) == {'r'}
    assert calls == ['vision']
    runner._prerequisite_generation['vision'] += 1
    runner._prepare_stage(inventory, 'vision', None)
    runner._prepare_stage({'recordings': [{'recording_id': 'new'}]}, 'vision', None)
    assert len(calls) == 3


def test_failed_retries_are_bounded_per_revision(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    queue = DeviceDayQueue(tmp_path / 'retry.sqlite3')
    record = {'recording_id': 'r', 'configured_role': 'first_person'}
    queue.enqueue(record, 'v1')
    for _ in range(3):
        assert queue.claim('worker', retry=True, max_attempts=3) == record
        queue.finish('worker', 'r', {'status': 'failed'}, 1)
    assert queue.claim('worker', retry=True, max_attempts=3) is None
    queue.enqueue(record, 'v2')
    assert queue.claim('worker', retry=True, max_attempts=3) == record


def test_preprocessing_only_keeps_full_request_resumable(device_config):
    from visioncortex.device_day_service import DeviceDayService
    import threading
    device_config['device_day']['preprocessing_only'] = True
    service = DeviceDayService(lambda: device_config, threading.Lock())
    service._runner = DeviceDayRunner(device_config, backend=object())
    assert not service._request_complete({'recordings': []})


def test_provider_circuit_isolates_cloud_and_recovers_without_queue_reset(device_config, monkeypatch):
    from visioncortex.device_day_provider_gate import ProviderGate
    from visioncortex import ai_settings
    device_config.setdefault('mllm', {})['provider'] = 'aliyun'
    device_config["speech_recognition"]["provider"] = "aliyun_qwen"
    gate = ProviderGate(device_config)
    assert gate.record_failure({'error': '400: Arrearage'})
    assert all(not gate.blocks(stage) for stage in ('retention', 'vision', 'report'))
    assert all(gate.blocks(stage) for stage in ('stt', 'understanding'))
    calls = []
    monkeypatch.setattr(ai_settings, 'verify_and_activate', lambda *args: calls.append(1) or {'activated': True})
    gate.probe_if_due()
    assert not calls
    atomic_json(gate.path, gate.state() | {'next_probe_at': 0})
    gate.probe_if_due()
    assert calls == [1]
    assert not gate.blocks('understanding')


def test_streamed_provider_error_retains_code_and_redacts_key():
    import httpx
    from visioncortex.mllm import _http_error_message
    response = httpx.Response(400, json={'error': {'code': 'Arrearage', 'message': 'secret-placeholder'}})
    message = _http_error_message(response, 'secret-placeholder')
    assert 'Arrearage' in message
    assert 'secret-placeholder' not in message


def test_dynamic_monitor_adds_cameras_and_does_not_wait_for_blocked_scan(tmp_path, monkeypatch):
    from visioncortex import device_day_monitor as module
    stop, entered = threading.Event(), threading.Event()
    cameras = [tmp_path / 'camera1', tmp_path / 'camera2']
    monkeypatch.setattr(module, '_root', lambda config: tmp_path)
    monkeypatch.setattr(module, '_camera_directories', lambda *args: list(cameras))
    monkeypatch.setattr(module, '_recording_batches', lambda *args: [])
    def scan(config, on_record):
        camera = config['collection_ingest']['camera_directories'][0]
        if camera == 'camera1':
            entered.set()
            stop.wait(5)
        else:
            on_record({'recording_id': camera, 'camera_key': camera, 'updated_at': '2026-09-11'})
        return {'errors': []}
    monkeypatch.setattr(module, 'scan_recordings', scan)
    observed = []
    monitor = module.CameraMonitor({'collection_ingest': {}, 'device_day': {}}, stop,
                                   lambda config, inventory: observed.extend(inventory.get('recordings', [])))
    try:
        monitor.poll()
        assert entered.wait(1)
        deadline = time.monotonic() + 2
        while not observed and time.monotonic() < deadline:
            time.sleep(.01)
        assert any(row['camera_key'] == 'camera2' for row in observed)
        cameras.append(tmp_path / 'camera3')
        result = monitor.poll()
        assert result['camera_directory_count'] == 3
        assert len(monitor.threads) == 6
    finally:
        stop.set()
        for thread in monitor.threads.values():
            thread.join(2)


def test_camera_scheduler_skips_remote_receipts_until_parent_completed(device_config, monkeypatch):
    device_config['device_day']['camera_lanes'] = True
    runner = DeviceDayRunner(device_config, backend=object())
    inventory = {'recordings': [{'recording_id': str(i), 'recording_start_us': i + 1,
                                'camera_key': f'camera{i % 12}', 'available': True}
                               for i in range(2500)]}
    monkeypatch.setattr(runner, 'layout', lambda record: pytest.fail('Unready work must not touch NAS receipts'))
    assert runner._build_stage_inventory(inventory, 'vision', None) == set()


def test_expired_lease_is_queued_in_snapshot_and_cannot_finish_new_owner(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    queue = DeviceDayQueue(tmp_path / 'expired.sqlite3')
    queue.enqueue({'recording_id': 'r', 'configured_role': 'first_person',
                   'duration_seconds': 7, 'camera_key': 'camera'}, 'v1')
    queue.claim('old', camera_serial=True)
    with queue.connect() as db:
        db.execute('UPDATE recordings SET lease_until=0')
    snapshot = queue.snapshot()
    assert snapshot['counts'] == {'queued': 1}
    assert snapshot['expired_lease_count'] == 1
    assert snapshot['queued_capture_window_seconds'] == 7
    assert queue.claim('new', camera_serial=True)['recording_id'] == 'r'
    queue.finish('old', 'r', {'status': 'completed'}, 1)
    assert queue.snapshot()['counts'] == {'running': 1}
    queue.finish('new', 'r', {'status': 'completed'}, 1)
    assert queue.snapshot()['counts'] == {'completed': 1}


def test_large_claim_allowlist_and_priority_remain_bounded(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    queue = DeviceDayQueue(tmp_path / 'allowed.sqlite3')
    for name, priority, date in [('history', 1, '2026-09-01'), ('live', 0, '2026-09-11')]:
        queue.enqueue({'recording_id': name, 'configured_role': 'first_person',
                       'processing_priority': priority, 'archive_date': date}, 'v1')
    allowed = {f'absent{i}' for i in range(2500)} | {'live', 'history'}
    assert queue.claim('one', allowed=allowed, date='2026-09-01')['recording_id'] == 'history'
    assert queue.claim('two', allowed=allowed)['recording_id'] == 'live'
    assert queue.claim('three', allowed=set()) is None


def test_scheduling_changes_do_not_invalidate_sealed_stage_identity(device_config):
    runner = DeviceDayRunner(device_config, backend=FakeModels())
    record = {'recording_id': 'r', 'source_signature': 'source', 'processing_priority': 0,
              'updated_at': 'old', 'archive_date': '2026-09-09'}
    keys = {stage: runner._key(stage, record, record if stage == 'retention' else {})
            for stage in ('retention', 'vision', 'understanding')}
    runner.settings.update(camera_lanes=True, vision_workers=17, retention_workers=9,
                           preprocessing_only=True, failure_retry_limit=1)
    revised = record | {'processing_priority': 1, 'updated_at': 'new'}
    assert runner._key('retention', revised, revised) == keys['retention']
    assert runner._key('vision', revised, {}) == keys['vision']
    assert runner._key('understanding', revised, {}) == keys['understanding']
    runner.settings['inactive_frames'] += 1
    assert runner._key('vision', revised, {}) != keys['vision']
    assert runner._key('retention', revised, revised) == keys['retention']


def test_readiness_cache_skips_unchanged_rows_but_updates_changed_input(device_config, monkeypatch):
    capture(device_config)
    inventory = scan_recordings(device_config)
    runner = DeviceDayRunner(device_config, backend=FakeModels())
    calls = []
    original = runner.queues['retention'].enqueue
    monkeypatch.setattr(runner.queues['retention'], 'enqueue', lambda *args: (calls.append(args), original(*args))[1])
    first = runner._build_stage_inventory(inventory, 'retention', None)
    assert len(first) == 1
    assert runner._build_stage_inventory(inventory, 'retention', None) == first
    assert len(calls) == 1
    rid=inventory['recordings'][0]['recording_id']
    cached=runner._record_readiness['retention'][rid]
    runner._record_readiness['retention'][rid]=(cached[0], -10**9)
    assert runner._build_stage_inventory(inventory, 'retention', None) == first
    assert len(calls) == 1
    inventory['recordings'][0]['source_signature'] = 'changed-source'
    runner._build_stage_inventory(inventory, 'retention', None)
    assert len(calls) == 2


def test_scan_hash_shared_across_lanes_and_invalidated_by_file_change(device_config, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from visioncortex import device_day_models as module
    backend = module.DeviceDayModels(device_config)
    source = tmp_path / 'model.pt'
    source.write_bytes(b'first')
    calls = []
    original = module.file_hash
    monkeypatch.setattr(module, 'file_hash', lambda path: (calls.append(path), original(path))[1])
    with ThreadPoolExecutor(max_workers=9) as pool:
        hashes = list(pool.map(backend._identity_hash, [source] * 9))
    assert len(calls) == 1
    assert len(set(hashes)) == 1
    source.write_bytes(b'changed')
    assert backend._identity_hash(source) != hashes[0]
    assert len(calls) == 2


def test_single_stage_refreshes_day_once_after_its_output(device_config, monkeypatch):
    capture(device_config)
    item, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    assert runner.process(item, stage="retention")["status"] == "completed"
    calls = []
    original = runner.refresh_index
    def refresh(value, **kwargs):
        calls.append(value.name)
        return original(value, **kwargs)
    monkeypatch.setattr(runner, "refresh_index", refresh)
    assert runner.process(item, stage="vision")["status"] == "completed"
    assert calls == [layout.name]
    assert read_json(layout.index)["segments"]


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("objects", [["hand", "paper"], ["gloved_hand", "lab_coat"],
                                     ["hand", "PPE_Storage"], ["hand", "unknown"], ["hand"]])
def test_non_lab_context_cannot_seed_activity(device_config, accepted, objects):
    from types import SimpleNamespace as NS
    from visioncortex.actions import build_activity_intervals
    from visioncortex.schemas import ActionType
    event = NS(event_id="non-lab", action_type=ActionType.HAND_OBJECT_CONTACT,
               accepted=accepted, confidence=.9, global_start_ms=3000, global_end_ms=4500,
               candidates=[NS(objects=objects, evidence=[
                   {"frame_index": n, "interaction_state": {"approach_confirmed": True}}
                   for n in range(8)])])
    coarse = NS(candidate_id="motion", local_start_ms=1000, local_end_ms=6000,
                objects=["beaker"], evidence=[{"coarse_motion": 20}])
    intervals, decisions = build_activity_intervals([event], [coarse], [(0, 8000)], device_config)
    assert intervals == []
    assert decisions[0]["basis"] == "non_laboratory_context_cannot_seed_activity"
    assert decisions[0]["activity_seed_eligible"] is False
    assert event.accepted is accepted
    # Paper or PPE in the scene must not mask a separately observed lab object.
    event.candidates[0].objects = [*objects, "balance"]
    assert build_activity_intervals([event], [coarse], [(0, 8000)], device_config)[0] == [(1000, 7500)]


def test_runner_identity_excludes_scheduler_but_tracks_execution():
    import visioncortex.device_day as module
    from visioncortex.device_day_cache_identity import execution_identity, BASELINE_FILE
    source = Path(module.__file__).read_text()
    assert execution_identity(source) != BASELINE_FILE
    changed = source.replace('cached[1] < 5:', 'cached[1] < 7:')
    assert changed != source
    assert execution_identity(changed) == execution_identity(source)
    changed = source.replace('"capture_deletion": "disabled_by_user"}', '"capture_deletion": "changed"}')
    assert execution_identity(changed) != execution_identity(source)


def test_compatible_retention_still_verifies_artifacts_and_source_identity(device_config, monkeypatch):
    import visioncortex.device_day as module
    capture(device_config)
    record = scan_recordings(device_config)['recordings'][0]
    runner = DeviceDayRunner(device_config, backend=FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    key = runner._key('retention', record, record)
    layout = runner.layout(record)
    path = runner._receipt(layout, record, 'retention')
    receipt = read_json(path)
    # Exercise the historical reader explicitly; current execution must not
    # acquire a compatibility alias merely because its orchestration changed.
    from visioncortex.device_day_cache_identity import BASELINE_FILE
    assert key not in runner._key_aliases
    runner._execution_identity = BASELINE_FILE
    legacy_expected = runner._key('retention', record, record)
    receipt['key'] = next(iter(runner._key_aliases[legacy_expected]))
    key = legacy_expected
    atomic_json(path, receipt)
    monkeypatch.setattr(module, 'retained_recording', lambda *a: pytest.fail('compatible source must not be copied again'))
    assert runner.process(record, stage='vision')['status'] == 'completed'
    different = record | {'source_signature': 'different'}
    other_key = runner._key('retention', different, different)
    assert not runner._matches_key(receipt['key'], other_key)
    target = layout.root / receipt['artifacts'][0]['path']
    target.write_bytes(b'corrupt')
    assert runner._load(path, key, layout) is None
    assert not runner._matches_key(receipt['key'], runner._key('vision', record, {}))


def test_camera_round_robin_feeds_later_cameras_before_older_camera_backlog(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    queue = DeviceDayQueue(tmp_path / 'fair.sqlite3')
    for camera in range(9):
        for part in range(3):
            queue.enqueue({'recording_id': f'{camera}-{part}', 'camera_key': f'camera{camera}',
                           'configured_role': 'first_person', 'processing_priority': 0,
                           'recording_start_us': camera * 100000 + part}, 'v1')
    first = queue.claim('a', camera_serial=True)
    second = queue.claim('b', camera_serial=True)
    assert first['recording_id'] == '0-0' and second['recording_id'] == '1-0'
    queue.finish('a', first['recording_id'], {'status': 'completed'}, 1)
    seen = {first['camera_key'], second['camera_key']}
    for _ in range(7):
        row = queue.claim('a', camera_serial=True)
        assert row['camera_key'] not in seen
        assert row['recording_id'].endswith('-0')
        seen.add(row['camera_key'])
        queue.finish('a', row['recording_id'], {'status': 'completed'}, 1)
    # Dispatch history survives process restart; same-camera chronological FIFO remains.
    reopened = DeviceDayQueue(tmp_path / 'fair.sqlite3')
    assert reopened.claim('a', camera_serial=True)['recording_id'] == '0-1'


def test_retention_reference_matches_independent_hash_without_extra_artifact_read(device_config, monkeypatch):
    from visioncortex.device_day import retained_recording
    import visioncortex.device_day_contract as contract
    capture(device_config)
    item, layout = item_and_layout(device_config)
    original = contract.file_hash
    reads=[]
    def counted(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(contract,'file_hash',counted)
    receipt=retained_recording(layout,item,device_config)
    assert reads == []  # copy_verified independently hashes target through its own binding
    for ref in receipt['artifacts']:
        assert original(layout.root/ref['path']) == ref['sha256']


def test_cleanup_hook_is_per_video_and_precedes_semantics(device_config,monkeypatch):
    capture(device_config)
    capture(device_config,camera='b_cam02')
    model=FakeModels()
    runner=DeviceDayRunner(device_config,backend=model)
    observed=[]
    def hook(owner,layout,record):
        receipt=read_json(owner._receipt(layout,record,'vision'))
        assert receipt['status']=='completed'
        assert layout.index.is_file()
        assert model.semantic_calls==0
        observed.append(record['recording_id'])
    monkeypatch.setattr('visioncortex.capture_link_cleanup.submit_after_preprocessing',hook)
    inventory=scan_recordings(device_config)
    record=inventory['recordings'][0]
    runner.process(record,stage='retention')
    runner.process(record,stage='vision')
    assert observed==[record['recording_id']]
    assert model.semantic_calls==0


@pytest.mark.parametrize('invalidate', [None, 'artifact', 'source'])
def test_queue_recipe_migration_preserves_only_verified_completion(device_config, invalidate):
    capture(device_config)
    inventory = scan_recordings(device_config)
    record = inventory['recordings'][0]
    runner = DeviceDayRunner(device_config, FakeModels())
    receipt = runner.process(record, stage='retention')
    assert receipt['status'] == 'completed'
    queue = runner.queues['retention']
    queue.enqueue(record, 'historical-queue-recipe')
    queue.claim('worker')
    queue.finish('worker', record['recording_id'], receipt, 2.5)
    with queue.connect() as db:
        before = dict(db.execute('SELECT * FROM recordings').fetchone())
    if invalidate == 'artifact':
        saved = read_json(runner._receipt(runner.layout(record), record, 'retention'))
        safe_child(runner.layout(record).root, saved['sources'][0]['retained']['path']).write_bytes(b'corrupt')
    elif invalidate == 'source':
        record['source_signature'] = 'changed'
    runner._build_stage_inventory(inventory, 'retention', None)
    with queue.connect() as db:
        after = dict(db.execute('SELECT * FROM recordings').fetchone())
    if invalidate is None:
        assert after['status'] == 'completed'
        assert after['revision'] != before['revision']
        assert after['completed_at'] == before['completed_at']
        assert after['wall_seconds'] == before['wall_seconds']
        assert queue.claim('again') is None
    else:
        assert after['status'] == 'queued'
        assert after['completed_at'] is None


def test_other_camera_consumes_admitted_jobs_during_slow_preparation(device_config, monkeypatch):
    runner = DeviceDayRunner(device_config, FakeModels())
    started, release = threading.Event(), threading.Event()
    def prepare(*args):
        runner._admitted['vision'].add('validated-camera-a')
        started.set()
        assert release.wait(3)
        return {'validated-camera-a', 'validated-camera-b'}
    monkeypatch.setattr(runner, '_build_stage_inventory', prepare)
    owner = threading.Thread(target=lambda: runner._prepare_stage({'recordings': []}, 'vision', None))
    owner.start()
    try:
        assert started.wait(2)
        assert runner._prepare_stage({'recordings': []}, 'vision', None) == {'validated-camera-a'}
    finally:
        release.set()
        owner.join(3)
    assert not owner.is_alive()


def test_process_preserves_measured_components_without_changing_outcome(device_config, monkeypatch):
    from visioncortex.device_day_activity import phase, counted, observations
    runner = DeviceDayRunner(device_config, FakeModels())
    def execute(*args, **kwargs):
        with phase('coarse_scan_seconds'):
            counted('coarse', 12)
        return {'status': 'completed'}
    monkeypatch.setattr(runner, '_process', execute)
    result = runner.process({'recording_id': 'measured', 'available': True,
                             'configured_role': 'first_person', 'camera_key': 'camera',
                             'recording_start_us': 1789459916524544}, stage='vision')
    assert result['status'] == 'completed'
    assert result['component_timings']['coarse_scan_seconds'] >= 0
    assert result['measured_frame_counts'] == {'coarse': 12}
    assert not observations()


def test_duplicate_claim_cannot_clear_active_model_progress(device_config, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from visioncortex.device_day_activity import phase, counted, observations
    runner = DeviceDayRunner(device_config, FakeModels())
    record = {'recording_id': 'duplicate', 'processable': True,
              'configured_role': 'first_person', 'camera_key': 'camera',
              'recording_start_us': 1789459916524544}
    started, release = Event(), Event()
    def execute(*args, **kwargs):
        with phase('coarse_scan_seconds'):
            counted('coarse', 12)
            started.set()
            assert release.wait(3)
        with phase('fine_scan_seconds'):
            counted('fine', 4)
        return {'status': 'completed'}
    monkeypatch.setattr(runner, '_process', execute)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(runner.process, record, stage='vision')
        try:
            assert started.wait(2)
            assert runner.process(record, stage='vision')['status'] == 'running_elsewhere'
            assert observations()[('vision', 'duplicate')]['frame_counts'] == {'coarse': 12}
        finally:
            release.set()
        assert first.result()['measured_frame_counts'] == {'coarse': 12, 'fine': 4}
    assert not observations()


def test_durable_pending_camera_role_repaired_from_configuration(device_config):
    capture(device_config)
    item, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, backend=FakeModels())
    lost_role = {k: v for k, v in item.items() if k != 'configured_role'}
    inventory = {'recordings': [lost_role]}
    eligible = runner._build_stage_inventory(inventory, 'retention', None)
    assert item['recording_id'] in eligible
    assert runner.queues['retention'].claim('test')['configured_role'] == 'first_person'
