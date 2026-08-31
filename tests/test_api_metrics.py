import asyncio
import hashlib
import io
import json
import time
import csv
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from labvision_evidence import api
from labvision_evidence.collection_catalog import clear_collection_catalog_cache


def test_nas_only_upload_does_not_create_persistent_local_copy(tmp_path):
    payload = b"video-payload" * 1024
    upload = UploadFile(file=io.BytesIO(payload), filename="实验视频.mp4")
    local_path = tmp_path / "local" / "video.mp4"
    nas_path = tmp_path / "nas" / "video.mp4"

    result = asyncio.run(
        api._save_upload_to_local_and_nas(
            upload,
            local_path,
            nas_path,
            retain_local_copy=False,
        )
    )

    assert nas_path.read_bytes() == payload
    assert not local_path.exists()
    assert result["analysis_path"] == str(nas_path)
    assert result["local_path"] is None
    assert result["local_write_bytes"] == 0
    assert result["nas_write_bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["retention_mode"] == "nas_only"
    assert result["effective_source_throughput_mib_s"] > 0


def test_safe_file_name_is_ascii_and_preserves_extension():
    assert api._safe_file_name("实验视频.mp4") == "file.mp4"
    assert api._safe_file_name("实验视频.mp4", "video-01") == "video-01.mp4"
    assert api._safe_file_name("cam 01_称量.MP4") == "cam-01_.mp4"


def test_web_end_to_end_metrics_preserve_pipeline_metrics(tmp_path):
    root = tmp_path / "archive"
    metrics_path = root / "JSON-Config-Files" / "run_metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps({"total_duration_seconds": 12.5}), encoding="utf-8")
    ingest = {
        "request_started_perf": time.perf_counter() - 0.01,
        "request_received_at": "2026-08-13T12:00:00+08:00",
        "duration_seconds": 0.005,
        "file_count": 2,
    }

    api._append_web_end_to_end_metrics([root], ingest, completed=True)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["total_duration_seconds"] == 12.5
    assert payload["web_ingest"]["file_count"] == 2
    assert "request_started_perf" not in payload["web_ingest"]
    assert payload["web_end_to_end"]["completed"] is True
    assert payload["web_end_to_end"]["total_duration_seconds"] >= 0.01


def test_health_exposes_fixed_benchmark_and_cache_locations(monkeypatch, tmp_path):
    settings = {
        "mllm": {"api_key_env": "TEST_ARK_KEY", "model": "doubao-test"},
        "storage": {
            "archive_root": str(tmp_path / "archive"),
            "index_csv": str(tmp_path / "index.csv"),
            "local_runtime_root": str(tmp_path / "runtime"),
            "local_cache_root": str(tmp_path / "cache"),
        },
    }
    monkeypatch.setattr(api, "_settings", lambda: settings)
    client = TestClient(api.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    benchmark = response.json()["fixed_benchmark"]
    assert benchmark["experiment_id"] == "exp_20260810_144014_e918b762"
    assert benchmark["archive_name"] == (
        "CustomFlow_standard_correct_12_ABCFA_0001--exp_20260810_144014_e918b762"
    )
    assert benchmark["submission_protocol_version"] == 1
    assert "zero-copy" in benchmark["input_mode"]
    assert benchmark["local_cache_root"].endswith("cache")
    assert response.json()["collection_ingest"]["recursive_nas_scan"] is False


def test_health_reports_local_storage_without_claiming_nas(monkeypatch, tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    settings = {
        "mllm": {
            "api_key_env": "TEST_ARK_KEY",
            "enabled": False,
            "model": "doubao-test",
        },
        "storage": {
            "archive_root": str(archive),
            "index_csv": str(tmp_path / "index.csv"),
            "local_runtime_root": str(tmp_path / "runtime"),
            "local_cache_root": str(tmp_path / "cache"),
            "sync_to_nas": False,
            "web_upload_retention_mode": "local_only",
        },
        "collection_ingest": {"enabled": False},
    }
    monkeypatch.setattr(api, "_settings", lambda: settings)
    client = TestClient(api.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["storage_mode"] == "local_development"
    assert payload["archive_label"] == "本地开发归档"
    assert payload["archive_root"] == str(archive)
    assert payload["archive_available"] is True
    assert payload["nas_available"] is False
    assert payload["mllm_enabled"] is False
    assert payload["web_upload_retention_mode"] == "local_only"
    assert payload["collection_ingest"]["enabled"] is False
    assert payload["fixed_benchmark"]["input_mode"] == (
        "local files / zero-copy source references"
    )


def test_model_candidate_api_exposes_quality_without_claiming_production():
    client = TestClient(api.app)

    response = client.get("/api/model-candidates")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_count"] >= 1
    assert payload["production_configuration_changed"] is False
    assert all(
        candidate["policy"]["production_enabled"] is False
        and candidate["policy"]["production_certified"] is False
        for candidate in payload["candidates"]
    )
    by_id = {item["candidate_id"]: item for item in payload["candidates"]}
    assert by_id["public-apparatus-21class-yolo26m-v4"]["status"] == (
        "invalidated_split_leakage_not_promoted"
    )
    assert by_id["public-apparatus-21class-yolo26m-v4"]["dataset"][
        "cross_split_content_hash_count"
    ] == 35
    assert by_id["public-apparatus-21class-yolo26s-v6-clean"]["dataset"][
        "cross_split_content_hash_count"
    ] == 0
    v7 = by_id["public-apparatus-21class-yolo26m-v7-clean-augmented"]
    assert v7["dataset"]["cross_split_content_hash_count"] == 0
    assert v7["comparative_test_metrics"]["independent_test_claim_allowed"] is False
    assert v7["policy"]["production_enabled"] is False


def test_collection_api_returns_batch_cards_without_opening_video_paths(
    monkeypatch, tmp_path
):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-device-registry/1",
                "devices": {},
                "experiment_role_overrides": {},
            }
        ),
        encoding="utf-8",
    )
    index_csv = tmp_path / "experiment_record_index.csv"
    rows = [
        {
            "experiment_id": "exp-api",
            "camera_key": "fp",
            "camera_view": "first",
            "recording_start_time": "2026-08-10T08:00:00+00:00",
            "recording_end_time": "2026-08-10T09:00:00+00:00",
            "segment_count": "1",
            "rgb_file": "Z:/missing-fp.mp4",
            "frames_file": "Z:/missing-fp.csv",
            "updated_at": "2026-08-10T09:01:00+00:00",
            "sync_error": "",
        },
        {
            "experiment_id": "exp-api",
            "camera_key": "tp",
            "camera_view": "side",
            "recording_start_time": "2026-08-10T08:00:00+00:00",
            "recording_end_time": "2026-08-10T09:00:00+00:00",
            "segment_count": "1",
            "rgb_file": "Z:/missing-tp.mp4",
            "frames_file": "Z:/missing-tp.csv",
            "updated_at": "2026-08-10T09:01:00+00:00",
            "sync_error": "",
        },
    ]
    with index_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = {
        "storage": {
            "index_csv": str(index_csv),
            "device_registry_path": str(registry),
            "local_cache_root": str(tmp_path / "cache"),
        },
        "collection_ingest": {
            "settle_seconds": 0,
            "persist_snapshot": False,
            "max_results": 20,
        },
    }
    monkeypatch.setattr(api, "_settings", lambda: settings)
    clear_collection_catalog_cache()
    client = TestClient(api.app)

    response = client.get("/api/collections")

    assert response.status_code == 200
    payload = response.json()
    assert payload["collection_count"] == 1
    card = payload["collections"][0]
    assert card["collection_id"] == "exp-api"
    assert card["ready_to_analyze"] is True
    assert card["camera_count"] == 2
    assert card["video_segment_count"] == 2
    assert not (tmp_path / "missing-fp.mp4").exists()


def test_collection_run_queues_zero_copy_index_executor(monkeypatch, tmp_path):
    settings = {
        "project": {},
        "storage": {
            "archive_root": str(tmp_path / "archive"),
            "index_csv": str(tmp_path / "index.csv"),
            "local_runtime_root": str(tmp_path / "runtime"),
        }
    }
    fixed_root = tmp_path / "archive" / "Collection-01"
    staging_root = tmp_path / "archive" / ".staging" / "Collection-01"
    history_root = tmp_path / "archive" / ".history" / "Collection-01"
    queued = []
    updates = []

    class TaskCollector:
        def add_task(self, function, *args):
            queued.append((function, args))

    monkeypatch.setattr(api, "_settings", lambda: settings)
    monkeypatch.setattr(
        api,
        "get_collection",
        lambda *_: {
            "ready_to_analyze": True,
            "recording_start_time": "2026-08-10T08:00:00+00:00",
            "fingerprint_sha256": "abc",
        },
    )
    monkeypatch.setattr(
        api,
        "_reserve_collection_archive",
        lambda *_: ("Collection-01", fixed_root, staging_root, history_root),
    )
    monkeypatch.setattr(api, "_update", lambda run_id, **values: updates.append((run_id, values)))
    monkeypatch.setattr(api.uuid, "uuid4", lambda: type("FixedUUID", (), {"hex": "b" * 32})())

    response = api.create_collection_run(
        "exp-source", TaskCollector(), {"experiment_name": "Collection 01"}
    )

    assert response["run_id"] == "collection-bbbbbbbbbb"
    assert response["source_collection_id"] == "exp-source"
    assert response["source_copy_bytes"] == 0
    assert response["archive_name"] == "Collection-01"
    assert len(queued) == 1
    assert queued[0][0] is api._execute_index_collection
    assert queued[0][1][1] == "exp-source"
    assert queued[0][1][4] == staging_root
    assert queued[0][1][5] == fixed_root
    assert response["nas_output"] == str(fixed_root)
    assert response["nas_staging"] == str(staging_root)
    assert updates[0][1]["source_collection_id"] == "exp-source"


def test_fixed_benchmark_submission_is_owned_by_web_service(monkeypatch, tmp_path):
    settings = {
        "storage": {
            "archive_root": str(tmp_path / "archive"),
            "index_csv": str(tmp_path / "index.csv"),
            "local_runtime_root": str(tmp_path / "runtime"),
        }
    }
    staging = tmp_path / "staging" / "run-001"
    staging.mkdir(parents=True)
    queued = []

    class TaskCollector:
        def add_task(self, function, *args):
            queued.append((function, args))

    monkeypatch.setattr(api, "_settings", lambda: settings)
    monkeypatch.setattr(api, "_reserve_fixed_benchmark", lambda *_: staging)
    monkeypatch.setattr(api, "_update", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api.uuid, "uuid4", lambda: type("FixedUUID", (), {"hex": "a" * 32})())

    response = api.create_fixed_benchmark_run(TaskCollector())

    assert response["run_id"] == "benchmark-aaaaaaaaaa"
    assert response["execution_owner"] == "visioncortex_web_service"
    assert response["client_process_independent"] is True
    assert response["submission_protocol_version"] == 1
    assert response["submission_receipt"].endswith("run_submission.json")
    assert len(queued) == 1
    assert queued[0][0] is api._execute_fixed_benchmark
    receipt = json.loads(Path(response["submission_receipt"]).read_text(encoding="utf-8"))
    assert receipt["run_id"] == response["run_id"]
    assert receipt["execution"]["owner"] == "visioncortex_web_service"
    assert receipt["execution"]["client_process_independent"] is True
    assert receipt["execution"]["requires_web_service_alive"] is True
    assert receipt["monitoring"]["status_url"] == f"/api/runs/{response['run_id']}"


def test_archived_quality_fallback_uses_evidence_without_fabricating_accuracy():
    events = [
        {
            "event_id": "event-001",
            "action_type": "liquid_transfer",
            "cross_view_associations": [{"both_views_support_action": True}],
            "aligned_frame_url": "/frame-001.jpg",
            "aligned_clip_url": "/clip-001.mp4",
        },
        {
            "event_id": "event-002",
            "action_type": "hand_object_contact",
            "cross_view_associations": [{"both_views_support_action": False}],
            "aligned_frame_url": "/frame-002.jpg",
            "aligned_clip_url": "/clip-002.mp4",
        },
    ]
    evidence_eval = {
        "passed": True,
        "checks": [
            {"check": "cross_view_or_explicit_uncertainty", "passed": True},
            {"check": "cross_view_or_explicit_uncertainty", "passed": True},
        ],
    }

    quality = api._derive_archived_quality_summary(
        {"experiment_groups": [{"group_id": "group-001"}]},
        events,
        evidence_eval,
    )

    assert quality["status"] == "evidence_package_passed_no_boundary_ground_truth"
    assert quality["experiment_boundaries"]["evaluated"] is False
    assert "precision" not in quality["experiment_boundaries"]
    assert "recall" not in quality["experiment_boundaries"]
    assert quality["key_materials"]["cross_view_supported_count"] == 1
    assert quality["key_materials"]["cross_view_supported_rate"] == 0.5
    assert quality["key_materials"]["dual_view_material_count"] == 2
    assert quality["key_materials"]["dual_view_material_rate"] == 1.0
    assert quality["key_materials"]["missing_dual_view_material_count"] == 0
    assert quality["key_materials"]["cross_view_or_explicit_uncertainty_count"] == 2


def test_key_material_verification_summary_exposes_models_timing_and_uncertainty():
    annotation = {
        "mode": "event_participants_only",
        "event_count": 1,
        "rendered_view_count": 2,
        "selective_verification": {
            "policy": "bounded final frames only",
            "decision_status_counts": {"admitted": 1, "deferred_budget_exhausted": 1},
            "wall_seconds": 3.5,
            "budget": {"admitted_event_count": 1, "deferred_count": 1},
            "source_copy_bytes": 0,
            "ark_calls": 0,
            "token_usage": 0,
        },
        "records": [
            {
                "event_id": "EVENT-1",
                "view_id": "fp",
                "role_label": "First-Person",
                "rendered_classes": ["hand", "pipette"],
                "rendered_detections": [
                    {"class_name": "hand", "confidence": 0.91},
                    {"class_name": "pipette", "confidence": 0.72},
                ],
                "minimum_rendered_confidence": 0.72,
                "selective_verification": {
                    "status": "admitted",
                    "assessment": {"reasons": ["low_closed_set_confidence"]},
                },
                "open_vocabulary_supplement": {
                    "status": "executed",
                    "model_load_seconds": 1.2,
                    "inference_seconds": 0.3,
                    "grounding_dino_fallback": {
                        "status": "executed",
                        "model_load_seconds": 1.5,
                        "inference_seconds": 0.4,
                    },
                },
            },
            {
                "event_id": "EVENT-1",
                "view_id": "tp",
                "role_label": "Third-Person",
                "rendered_classes": ["hand", "pipette"],
                "rendered_detections": [
                    {"class_name": "pipette", "confidence": 0.84}
                ],
                "minimum_rendered_confidence": 0.84,
                "selective_verification": {
                    "status": "deferred_budget_exhausted",
                    "reason": "wall_time_budget_exhausted",
                    "assessment": {"reasons": []},
                },
            },
        ],
    }

    summary, by_event = api._summarize_key_material_verification(annotation)

    event = by_event["EVENT-1"]
    assert event["status"] == "verification_deferred_budget_exhausted"
    assert event["models"] == [
        "closed_set_yolo_tensorrt",
        "grounding_dino_base",
        "yolo_world_v2",
    ]
    assert event["confidence"] == {"minimum": 0.72, "maximum": 0.91}
    assert event["timing"] == {
        "model_load_seconds": 2.7,
        "inference_seconds": 0.7,
    }
    assert event["uncertain"] is True
    assert "wall_time_budget_exhausted" in event["uncertainty_reasons"]
    assert summary["uncertain_event_count"] == 1
    assert summary["model_execution_counts"]["closed_set_yolo_tensorrt"] == 2
    assert summary["source_copy_bytes"] == 0


def test_archive_performance_display_separates_cold_start_from_reuse_run():
    metrics = {
        "total_duration_seconds": 572.15617,
        "preprocessing_sla": {"actual_seconds": 81.237808},
    }
    acceptance = {
        "preprocessing_full_run": {
            "seconds": 694.056768,
            "includes": "preflight + alignment + full coarse + bounded fine + audit",
        },
        "clean_package_run": {
            "seconds": 572.15617,
            "note": "reused validated CV ledgers; regenerated media/package",
        },
    }

    api._attach_archive_performance_display(metrics, acceptance)

    display = metrics["preprocessing_display"]
    assert metrics["display_preprocessing_source"] == "full_cold_start_benchmark"
    assert display["full_cold_start"]["seconds"] == 694.056768
    assert display["current_run"]["preprocessing_seconds"] == 81.237808
    assert display["current_run"]["total_seconds"] == 572.15617
    assert display["current_run"]["reused_validated_cv_ledgers"] is True


def test_run_snapshot_uses_durable_live_observability(tmp_path):
    root = tmp_path / "archive"
    json_root = root / "JSON-Config-Files"
    json_root.mkdir(parents=True)
    (json_root / "pipeline_status.json").write_text(
        json.dumps({"stage": "coarse_scan", "updated_at": "status-time"}),
        encoding="utf-8",
    )
    (json_root / "source_progress.json").write_text(
        json.dumps(
            {
                "views": [
                    {
                        "view_id": "cam01",
                        "decode_backend": "cuda",
                        "state": "running",
                        "completed_work_units": 3,
                        "total_work_units": 12,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (json_root / "resource_telemetry_live.json").write_text(
        json.dumps(
            {
                "updated_at": "telemetry-time",
                "gpu": {"utilization_percent": 91.0, "decoder_percent": 72.0},
            }
        ),
        encoding="utf-8",
    )
    (json_root / "run_metrics_live.json").write_text(
        json.dumps({"tokens": {"total_tokens": 4321}}), encoding="utf-8"
    )
    receipt_root = json_root / "Stage-Receipts"
    receipt_root.mkdir()
    artifact = root / "Experiment-Clips"
    artifact.mkdir()
    (receipt_root / "experiment_clips.json").write_text(
        json.dumps(
            {
                "stage": "experiment_clips",
                "status": "completed",
                "completed_at": "receipt-time",
                "stage_duration_seconds": 12.5,
                "artifacts": ["Experiment-Clips"],
            }
        ),
        encoding="utf-8",
    )

    snapshot = api._run_snapshot_from_root(root)

    assert snapshot["status"]["views"][0]["view_id"] == "cam01"
    assert snapshot["live_telemetry"]["gpu"]["utilization_percent"] == 91.0
    assert snapshot["metrics"]["tokens"]["total_tokens"] == 4321
    assert snapshot["freshness"]["telemetry_updated_at"] == "telemetry-time"
    assert snapshot["stage_receipts"][0]["stage"] == "experiment_clips"
    assert snapshot["stage_receipts"][0]["artifacts"][0]["available"] is True
    assert next(
        item for item in snapshot["archive_areas"] if item["name"] == "Experiment-Clips"
    )["available"] is True


def test_completed_run_hydrates_from_formal_archive_before_staging(tmp_path):
    staging = tmp_path / "staging"
    formal = tmp_path / "formal"
    for root, stage in ((staging, "candidate_fine"), (formal, "completed")):
        json_root = root / "JSON-Config-Files"
        json_root.mkdir(parents=True)
        (json_root / "pipeline_status.json").write_text(
            json.dumps({"stage": stage}), encoding="utf-8"
        )

    hydrated = api._hydrate_run_snapshot(
        {
            "state": "completed",
            "nas_staging": str(staging),
            "nas_output": str(formal),
        }
    )

    assert hydrated["observability"]["status"]["stage"] == "completed"


def test_completed_run_uses_explicit_current_observability_root(tmp_path):
    historical = tmp_path / "historical"
    current = tmp_path / "current"
    for root, duration in ((historical, 572.156), (current, 4318.780)):
        json_root = root / "JSON-Config-Files"
        json_root.mkdir(parents=True)
        (json_root / "run_metrics.json").write_text(
            json.dumps({"total_duration_seconds": duration}), encoding="utf-8"
        )

    hydrated = api._hydrate_run_snapshot(
        {
            "state": "completed",
            "nas_output": str(historical),
            "observability_root": str(current),
        }
    )

    assert hydrated["observability"]["root"] == str(current)
    assert hydrated["observability"]["metrics"]["total_duration_seconds"] == 4318.780


def test_service_restart_marks_orphaned_task_resumable(monkeypatch, tmp_path):
    archive_root = tmp_path / "archive"
    status_path = (
        archive_root
        / ".VisionCortex-Run-Staging"
        / "benchmark"
        / "run-001"
        / "JSON-Config-Files"
        / "pipeline_status.json"
    )
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps({"stage": "fine_scan", "message": "still running"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)

    api._recover_orphaned_tasks()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "interrupted"
    assert payload["recovery"]["resumable"] is True
    assert payload["recovery"]["previous_stage"] == "fine_scan"
    assert payload["recovery"]["heartbeat"]["active"] is False


def test_runtime_activity_receipt_rejects_stale_heartbeat(tmp_path):
    json_root = tmp_path / "JSON-Config-Files"
    json_root.mkdir()
    status_path = json_root / "pipeline_status.json"
    status_path.write_text("{}", encoding="utf-8")
    heartbeat_path = json_root / "resource_telemetry_live.json"
    heartbeat_path.write_text("{}", encoding="utf-8")
    heartbeat_mtime = heartbeat_path.stat().st_mtime

    receipt = api._runtime_activity_receipt(
        status_path,
        now_epoch=heartbeat_mtime + api._RUNTIME_HEARTBEAT_STALE_SECONDS + 0.001,
    )

    assert receipt["active"] is False
    assert receipt["latest_path"] == "resource_telemetry_live.json"
    assert receipt["age_seconds"] > receipt["stale_after_seconds"]


def test_service_restart_preserves_task_with_fresh_pipeline_heartbeat(monkeypatch, tmp_path):
    archive_root = tmp_path / "archive"
    json_root = (
        archive_root
        / ".VisionCortex-Run-Staging"
        / "benchmark"
        / "run-active"
        / "JSON-Config-Files"
    )
    json_root.mkdir(parents=True)
    status_path = json_root / "pipeline_status.json"
    status_path.write_text(
        json.dumps({"stage": "candidate_fine", "message": "fine scanning"}),
        encoding="utf-8",
    )
    (json_root / "resource_telemetry_live.json").write_text(
        json.dumps({"stage": "candidate_fine"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)

    api._recover_orphaned_tasks()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "candidate_fine"
    assert "recovery" not in payload


def test_service_restart_restores_previous_stage_when_heartbeat_resumes(monkeypatch, tmp_path):
    archive_root = tmp_path / "archive"
    json_root = (
        archive_root
        / ".VisionCortex-Run-Staging"
        / "benchmark"
        / "run-recovered"
        / "JSON-Config-Files"
    )
    json_root.mkdir(parents=True)
    status_path = json_root / "pipeline_status.json"
    status_path.write_text(
        json.dumps(
            {
                "stage": "interrupted",
                "message": "service restarted",
                "recovery": {
                    "status": "orphaned_after_service_restart",
                    "resumable": True,
                    "previous_stage": "candidate_fine",
                },
            }
        ),
        encoding="utf-8",
    )
    (json_root / "source_progress.json").write_text(
        json.dumps({"views": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)

    api._recover_orphaned_tasks()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "candidate_fine"
    assert payload["recovery"]["status"] == "active_after_service_restart"
    assert payload["recovery"]["resumable"] is False
    assert payload["recovery"]["heartbeat"]["active"] is True
