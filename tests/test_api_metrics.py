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
            "local_input_root": str(tmp_path / "input"),
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
    assert benchmark["local_cache_root"].endswith("cache")
