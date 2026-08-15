import asyncio
import hashlib
import io
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from labvision_evidence import api


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
    assert benchmark["archive_name"] == "Six-View-Three-Hour-Experiment-2026-08-13"
    assert benchmark["submission_protocol_version"] == 1
    assert "zero-copy" in benchmark["input_mode"]
    assert benchmark["local_cache_root"].endswith("cache")


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
