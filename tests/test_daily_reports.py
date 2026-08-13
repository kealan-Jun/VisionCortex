import json
from pathlib import Path

import pytest

from labvision_evidence.daily_reports import build_daily_report, evaluate_daily_report
from labvision_evidence.schemas import RunSummary


def test_daily_report_reconciles_local_validation_package(default_config):
    root = Path(
        "outputs/clean-local-validation/"
        "exp_20260810_144014_e918b762-clean-validation/JSON-Config-Files"
    )
    if not root.is_dir():
        pytest.skip("Optional local validation archive unavailable")
    summary = RunSummary.model_validate_json(
        (root / "evidence_package.json").read_text(encoding="utf-8-sig")
    )
    metrics = json.loads((root / "run_metrics.json").read_text(encoding="utf-8-sig"))
    evidence_eval = json.loads(
        (root / "evidence_package_eval.json").read_text(encoding="utf-8-sig")
    )

    report = build_daily_report(summary, metrics, evidence_eval, default_config)
    evaluation = evaluate_daily_report(report, summary)

    assert evaluation["passed"] is True
    assert report["overview"]["experiment_group_count"] == 5
    assert report["overview"]["key_event_count"] == 32
    assert report["source_policy"]["additional_model_tokens"]["total_tokens"] == 0
    assert report["template_id"] == "VC-LAB-DAILY-REPORT-V1"
    assert report["source_policy"]["layout_editable_by_model"] is False
    assert report["performance"]["total_tokens"] == 219535
