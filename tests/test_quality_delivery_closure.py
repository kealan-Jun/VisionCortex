"""A finished quality evaluation must deliver a partial result without promotion."""
import json

import pytest

from visioncortex import api
from visioncortex.archive import ArchiveLayout, write_json
from visioncortex.partial_delivery import partial_result_available, write_partial_delivery
from visioncortex.pipeline import EvidencePipeline
from visioncortex.run_queue import DurableRunQueue


def partial_layout(tmp_path):
    layout = ArchiveLayout(tmp_path / "staging")
    layout.create()
    write_json(layout.json_config / "quality_acceptance.json", {
        "passed": False, "formal_accuracy_claim_allowed": False,
        "segmentation_integrity": {"groups": [{
            "group_id": "GROUP-2", "passed": False,
            "key_event_count": 3, "all_declared_key_events_resolved": True,
            "jointly_supported_key_event_ids": [],
        }]},
    })
    return layout


def test_quality_closure_is_terminal_and_retains_negative_gate(tmp_path, monkeypatch):
    layout = partial_layout(tmp_path)
    original_quality = (layout.json_config / "quality_acceptance.json").read_bytes()
    pipeline = EvidencePipeline({"storage": {"run_output_mode": "nas_direct"}})
    monkeypatch.setattr(pipeline, "_metrics", lambda events, groups: {"tokens": {"run_total": {"total_tokens": 19}}})
    pipeline._status(layout, "package", .96, "checking")
    assert pipeline._finish_quality_attention(layout, [], []) == layout.root
    assert pipeline._active_stage is None
    assert partial_result_available(layout.root)
    assert (layout.json_config / "quality_acceptance.json").read_bytes() == original_quality
    receipt = json.loads((layout.json_config / "partial_delivery.json").read_text())
    assert receipt["analysis_finished"] is True
    assert receipt["quality_gaps"][0]["code"] == "canonical_pair_support_missing"
    assert receipt["quality_gaps"][0]["automatic_retry"] is False
    assert not (layout.json_config / "daily_report_manifest.json").exists()
    assert not (layout.root / ".VisionCortex-Current-Release.json").exists()


def test_web_executor_finishes_partial_without_publishing(tmp_path, monkeypatch):
    layout = partial_layout(tmp_path)
    write_json(layout.root / "run_status.json", {"stage": "partial", "progress": 1})
    write_partial_delivery(layout.root, {})
    monkeypatch.setattr(api.EvidencePipeline, "run", lambda self, manifest: layout.root)
    monkeypatch.setattr(api, "_runs", {"R": {"state": "running"}})
    monkeypatch.setattr(api, "_persistent_queue", None)
    monkeypatch.setattr(api, "promote_fixed_archive", lambda *args: pytest.fail("partial result was published"))
    api._execute_now("R", None, {"storage": {"formal_promotion_required": True}}, layout.root)
    result = api._runs["R"]
    assert result["state"] == "partial"
    assert result["error"] is None
    assert result["promotion"] is None and result["archive_url"] is None
    assert result["observability_root"] == str(layout.root)


def test_missing_or_tampered_partial_report_is_still_a_failure(tmp_path):
    layout = partial_layout(tmp_path)
    write_json(layout.root / "run_status.json", {"stage": "partial"})
    with pytest.raises(RuntimeError):
        partial_result_available(layout.root)
    write_partial_delivery(layout.root, {})
    (layout.root / "Partial-Results/Partial-Evidence-Report.html").write_text("tampered")
    with pytest.raises(RuntimeError):
        partial_result_available(layout.root)


def test_partial_detail_keeps_final_quality_instead_of_running_preview(tmp_path, monkeypatch):
    layout = partial_layout(tmp_path)
    write_json(layout.root / "run_status.json", {"stage": "partial"})
    write_partial_delivery(layout.root, {})
    monkeypatch.setattr(api, "_settings", lambda: {"storage": {"archive_root": str(tmp_path)}})
    detail = api._archive_detail_from_root(layout.root, "A", staging_run_id="R")
    assert detail["quality_acceptance"]["passed"] is False
    assert detail["quality_acceptance"]["segmentation_integrity"]["groups"][0]["group_id"] == "GROUP-2"
    assert "run_id=R" in detail["links"]["partial_report"]


def test_partial_job_releases_lease_survives_restart_and_retries_same_id(tmp_path):
    store = DurableRunQueue(tmp_path / "queue.sqlite3")
    store.enqueue("R", "collection", {"settings": {"project": {}}})
    job = store.claim_next("worker", lease_seconds=60)
    assert job.run_id == "R"
    store.save_run("R", {"state": "partial", "nas_staging": "/retained"})
    assert store.finish("R", "worker", "partial")
    restored = DurableRunQueue(store.database)
    assert restored.claim_next("restarted", lease_seconds=60) is None
    assert restored.load_runs()["R"]["state"] == "partial"
    retry = restored.retry_failed("R")
    assert retry["attempt_history"][-1]["state"] == "partial"
    assert retry["nas_staging"] == "/retained"
    assert restored.claim_next("restarted", lease_seconds=60).attempts == 2


def test_formal_completed_job_cannot_use_partial_retry(tmp_path):
    store = DurableRunQueue(tmp_path / "queue.sqlite3")
    store.enqueue("R", "collection", {"settings": {"project": {}}})
    store.claim_next("worker", lease_seconds=60)
    store.save_run("R", {"state": "completed"})
    store.finish("R", "worker", "completed")
    with pytest.raises(ValueError):
        store.retry_failed("R")
