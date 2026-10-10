"""Library summaries retain identity and evidence class without heavy run ledgers."""
import json

from fastapi.testclient import TestClient
from visioncortex import api
from visioncortex.library_projection import project_staging_library


def test_projection_preserves_material_identity_and_cannot_leak_heavy_audit_data():
    data = {"name":"A", "staging_run_id":"R", "observability":{"status":{"stage":"failed"}},
            "key_events":[{"event_id":"E", "parent_event_id":"G", "dual_view_material_ready":True,
                "decision":{"status":"accepted"}, "aligned_frame_url":"/frame", "aligned_clip_url":"/clip",
                "provenance":{"mllm":{"current_step":"未开盖"},"source_frames":["large"]}}],
            "quarantined_materials":[{"event_id":"Q","evidence_classification":"PARTIAL_EVIDENCE","frame_url":"/q"}],
            "links":{"partial_pdf":"/pdf"}, "daily_report":{},"metrics":{"large":True}}
    before = json.dumps(data)
    result = project_staging_library(data, "library-materials")
    assert result["key_events"][0]["provenance"] == {"mllm":{"current_step":"未开盖"}}
    assert result["key_events"][0]["parent_event_id"] == "G"
    assert result["quarantined_materials"][0]["evidence_classification"] == "PARTIAL_EVIDENCE"
    assert "metrics" not in result and "links" not in result
    report = project_staging_library(data, "library-reports")
    assert report["links"] == {"partial_pdf":"/pdf"}
    assert "key_events" not in report
    assert json.dumps(data) == before


def test_library_route_skips_heavy_ledgers_and_rejects_unknown_section(tmp_path, monkeypatch):
    control = tmp_path / "JSON-Config-Files"
    control.mkdir()
    (control / "pipeline_status.json").write_text('{"stage":"failed"}')
    monkeypatch.setattr(api, "_resolve_staging_run", lambda _: tmp_path)
    original = api._read_json
    def guarded(path, default=None):
        assert path.name not in {"evidence_package.json", "final_key_material_annotation.json", "resource_telemetry.json"}
        return original(path, default)
    monkeypatch.setattr(api, "_read_json", guarded)
    monkeypatch.setattr(api, "_latest_result_review", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("library recomputed analysis")))
    client = TestClient(api.app)
    result = client.get('/api/staging-runs/R/archive?section=library-materials')
    assert result.status_code == 200
    assert result.json()["staging_run_id"] == "R"
    assert result.json()["observability"]["status"]["stage"] == "failed"
    assert "metrics" not in result.json()
    assert client.get('/api/staging-runs/R/archive?section=unknown').status_code == 400
