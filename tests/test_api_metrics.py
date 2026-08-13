import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from labvision_evidence import api


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
    assert "zero-copy" in benchmark["input_mode"]
    assert benchmark["local_cache_root"].endswith("cache")


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

    snapshot = api._run_snapshot_from_root(root)

    assert snapshot["status"]["views"][0]["view_id"] == "cam01"
    assert snapshot["live_telemetry"]["gpu"]["utilization_percent"] == 91.0
    assert snapshot["metrics"]["tokens"]["total_tokens"] == 4321
    assert snapshot["freshness"]["telemetry_updated_at"] == "telemetry-time"


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
