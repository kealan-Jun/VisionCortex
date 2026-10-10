"""Failed recall reports stay inspectable without weakening publication gates."""

from copy import deepcopy

import pytest

from visioncortex import device_day_models as models
from visioncortex.device_day_contract import read_json
from visioncortex.schemas import VideoInfo, ViewInput, ViewRole


@pytest.fixture
def plan_case(default_config, tmp_path):
    default_config["storage"]["local_runtime_root"] = str(tmp_path / "Runtime")
    default_config["models"] = {}
    backend = models.DeviceDayModels(default_config)
    view = ViewInput(view_id="camera", role=ViewRole.FIRST_PERSON, video=tmp_path / "Video.mp4")
    info = VideoInfo(path=view.video, width=100, height=100, fps=30,
                     frame_count=300, duration_ms=10000)
    ledger = tmp_path / "Ledger.jsonl"
    ledger.write_text("synthetic ledger, no media read")
    args = (view, {view.view_id: ledger}, tmp_path, info, 0, 10000, 2)
    return backend, args, tmp_path / "Runtime" / "device-day" / "RecallFailures"


def failed_report():
    return {"status": "completed_with_errors_closed_set_preserved",
            "selected_frame_count": 2, "error_count": 2,
            "error_rate": 1.0, "maximum_error_rate": 0.1,
            "formal_evidence_ready": False,
            "frames": [{"local_ms": 1000, "status": "frame_unreadable"},
                       {"local_ms": 5000, "status": "open_vocabulary_error_closed_set_preserved",
                        "error_type": "RuntimeError", "error": "model fixture failure"}]}


def test_failed_plan_reports_survive_retries_and_are_never_reused_as_success(plan_case, monkeypatch):
    backend, args, failures = plan_case
    report = failed_report()
    original = deepcopy(report)
    calls = []

    def failing(*_args):
        calls.append(1)
        raise models.RecallEvidenceGateError("coarse", report)

    monkeypatch.setattr(models, "device_scan_plan", failing)
    for _ in range(2):
        with pytest.raises(models.RecallEvidenceGateError, match="frame_unreadable.*RuntimeError") as error:
            backend._plan(*args)
        assert "diagnostic=" in str(error.value)
    assert len(calls) == 2
    assert not (args[2] / "Plans").exists()
    saved = [read_json(p) for p in failures.glob("*.json")]
    assert len(saved) == 2
    assert all(r["report"] == original and r["formal_evidence_ready"] is False for r in saved)
    assert all(r["context"]["ledger"] and r["context"]["plan_key"] for r in saved)
    assert report == original

    def succeeds(*_args):
        calls.append(1)
        return [], [], {"formal_evidence_ready": True}

    monkeypatch.setattr(models, "device_scan_plan", succeeds)
    assert not backend._plan(*args)[3]["reused"]
    assert backend._plan(*args)[3]["reused"]
    assert len(calls) == 3
    assert len(list(failures.glob("*.json"))) == 2


def test_diagnostic_write_failure_keeps_original_evidence_failure(plan_case, monkeypatch):
    backend, args, _ = plan_case

    def failing(*_args):
        raise models.RecallEvidenceGateError("coarse", failed_report())

    def disk_full(*_args):
        raise OSError("fixture disk full")

    monkeypatch.setattr(models, "device_scan_plan", failing)
    monkeypatch.setattr(models, "atomic_json", disk_full)
    with pytest.raises(models.RecallEvidenceGateError, match="diagnostic_write=OSError") as error:
        backend._plan(*args)
    assert error.value.report["formal_evidence_ready"] is False


def test_fine_failure_is_saved_before_temporary_ledger_cleanup(plan_case, monkeypatch):
    from visioncortex import actions

    backend, args, failures = plan_case
    view, ledgers, _, info, *_ = args
    monkeypatch.setattr(actions, "generate_candidates", lambda *_args: [])

    def failing(*_args):
        raise models.RecallEvidenceGateError("fine", failed_report())

    monkeypatch.setattr(backend, "_audit_activity", failing)
    with pytest.raises(models.RecallEvidenceGateError, match="Existing fine ROI"):
        backend._audit_fine(view, info, ledgers, {"audit_ledger_storage": "local_runtime_root"}, [], [(0, 10000)])
    assert not ledgers[view.view_id].exists()
    [saved] = [read_json(p) for p in failures.glob("*.json")]
    assert saved["gate"] == "fine"
    assert saved["report"]["error_count"] == 2
    assert saved["context"]["windows"] == [[0, 10000]]


def test_coarse_gate_still_rejects_the_same_failed_report(default_config, tmp_path, monkeypatch):
    from visioncortex import actions, coarse_recall

    view = ViewInput(view_id="camera", role=ViewRole.FIRST_PERSON, video=tmp_path / "Video.mp4")
    info = VideoInfo(path=view.video, duration_ms=10000, fps=30, width=100, height=100, frame_count=300)
    default_config["performance"]["candidate_discovery_quality_gate_enabled"] = True
    monkeypatch.setattr(actions, "generate_coarse_activity_candidates", lambda *_args: [])
    monkeypatch.setattr(actions, "generate_motion_burst_candidates", lambda *_args: [])
    monkeypatch.setattr(coarse_recall, "generate_open_vocabulary_coarse_candidates",
                        lambda *_args: ([], failed_report()))
    with pytest.raises(models.RecallEvidenceGateError) as error:
        models.device_scan_plan(view, {}, default_config, info, 0, 10000, 2)
    assert error.value.summary["inference_error_types"] == {"RuntimeError": 1}
    assert error.value.summary["frame_status_counts"]["frame_unreadable"] == 1
