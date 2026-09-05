import json
from pathlib import Path

import pytest

from visioncortex.daily_reports import build_daily_report, evaluate_daily_report
from visioncortex.schemas import RunSummary, ViewRole
from visioncortex.report_presentations import _headline_status


@pytest.mark.parametrize("quality,human,expected", [
    ({"passed": False, "status": "structural_only"}, "pending", "待质量复核"),
    ({}, "pending", "待质量复核"),
    ({"passed": True}, "pending", "自动检查通过，待人工复核"),
    ({"passed": True}, "rejected", "人工复核未通过"),
    ({"passed": True}, "approved", "证据验收通过"),
])
def test_structure_pass_does_not_imply_quality_approval(quality, human, expected):
    report = {"overview": {"evidence_package_eval_passed": True},
              "quality_acceptance": quality, "human_review": {"status": human}}
    assert _headline_status(report)[0] == expected


def _accepted_quality() -> dict:
    return {
        "passed": True,
        "status": "structural_only",
        "structural_passed": True,
        "evidence_level": "structural_only",
        "formal_accuracy_claim_allowed": False,
        "experiment_boundaries": {"evaluated": False},
        "key_event_recall": {"evaluated": False},
        "key_materials": {},
    }


def _summary_with_post_curation_rejection() -> RunSummary:
    base_event = {
        "action_type": "hand_object_contact",
        "global_start_ms": 1000,
        "global_end_ms": 2000,
        "key_global_ms": 1500,
        "objects": ["gloved_hand", "paper"],
        "confidence": 0.9,
        "accepted": True,
        "audit_reason": "dual-view evidence",
        "supporting_views": ["first", "third"],
        "supporting_roles": ["first_person", "third_person"],
        "candidates": [],
        "key_frames": {"aligned_first_third": "Key-Materials/accepted.jpg"},
        "key_clips": {"aligned_first_third": "Key-Materials/accepted.mp4"},
        "model_understanding": {"status": "completed"},
    }
    return RunSummary.model_validate(
        {
            "experiment_id": "post-curation-filter-test",
            "created_at": "2026-08-24T00:00:00+00:00",
            "views": [
                {"view_id": "first", "role": "first_person", "video": "/tmp/first.mp4"},
                {"view_id": "third", "role": "third_person", "video": "/tmp/third.mp4"},
            ],
            "alignments": [
                {
                    "view_id": "first",
                    "reference_view_id": "first",
                    "state": "aligned",
                    "confidence": 1.0,
                },
                {
                    "view_id": "third",
                    "reference_view_id": "first",
                    "state": "aligned",
                    "confidence": 1.0,
                },
            ],
            "events": [
                {**base_event, "event_id": "EVT-ACCEPTED"},
                {
                    **base_event,
                    "event_id": "EVT-POST-CURATION-REJECTED",
                    "global_start_ms": 3000,
                    "global_end_ms": 4000,
                    "key_global_ms": 3500,
                },
            ],
            "segments": [],
            "experiment_groups": [
                {
                    "group_id": "GROUP-001",
                    "continuity_type": "continuous",
                    "atomic_experiment_ids": ["SEG-001"],
                    "global_start_ms": 0,
                    "global_end_ms": 5000,
                    "participating_views": ["first", "third"],
                    "first_person_view": "first",
                    "third_person_view": "third",
                    "continuity_reason": "unit test",
                    "key_event_ids": ["EVT-ACCEPTED"],
                    "model_understanding": {
                        "status": "completed",
                        "steps": [
                            {
                                "step_index": 1,
                                "start_global_ms": 1000,
                                "end_global_ms": 2000,
                                "current_step": "手接触实验对象",
                                "next_step": "未知",
                                "next_step_status": "unknown",
                                "supporting_event_ids": ["EVT-ACCEPTED"],
                                "objects": ["gloved_hand", "paper"],
                                "supporting_views": ["first", "third"],
                                "confidence": 0.9,
                            }
                        ],
                    },
                }
            ],
            "physical_change_log": [
                {
                    "change_id": "CHANGE-ACCEPTED",
                    "event_id": "EVT-ACCEPTED",
                    "global_ms": 1500,
                    "change_type": "contact_started",
                    "object_names": ["gloved_hand", "paper"],
                    "supporting_views": ["first", "third"],
                    "confidence": 0.9,
                },
                {
                    "change_id": "CHANGE-POST-CURATION-REJECTED",
                    "event_id": "EVT-POST-CURATION-REJECTED",
                    "global_ms": 3500,
                    "change_type": "contact_started",
                    "object_names": ["gloved_hand", "paper"],
                    "supporting_views": ["first", "third"],
                    "confidence": 0.9,
                },
            ],
        }
    )


def test_daily_report_excludes_post_curation_physical_changes(default_config):
    summary = _summary_with_post_curation_rejection()

    report = build_daily_report(
        summary,
        {},
        {"passed": True, "checks": []},
        default_config,
        _accepted_quality(),
    )
    evaluation = evaluate_daily_report(report, summary)

    assert evaluation["passed"] is True
    assert report["overview"]["key_event_count"] == 1
    assert report["overview"]["physical_change_count"] == 1
    assert report["experiment_timeline"][0]["physical_change_count"] == 1
    assert [item["event_id"] for item in report["physical_change_log"]] == [
        "EVT-ACCEPTED"
    ]


def test_daily_report_eval_fails_closed_on_rejected_physical_change(default_config):
    summary = _summary_with_post_curation_rejection()
    report = build_daily_report(
        summary,
        {},
        {"passed": True, "checks": []},
        default_config,
        _accepted_quality(),
    )
    report["physical_change_log"].append(
        summary.physical_change_log[1].model_dump(mode="json")
    )
    report["overview"]["physical_change_count"] = 2
    report["experiment_timeline"][0]["physical_change_count"] = 2

    evaluation = evaluate_daily_report(report, summary)

    assert evaluation["passed"] is False
    failed_checks = {
        item["check"] for item in evaluation["checks"] if not item["passed"]
    }
    assert "accepted_physical_change_ids_match" in failed_checks


def test_daily_report_carries_precomputed_runtime_audit(default_config):
    summary = _summary_with_post_curation_rejection()
    runtime_audit = {
        "source": {"source_copy_bytes": 0},
        "mllm": {"call_count": 3, "completed_count": 3},
    }

    report = build_daily_report(
        summary,
        {"runtime_audit": runtime_audit},
        {"passed": True, "checks": []},
        default_config,
        _accepted_quality(),
    )

    assert report["performance"]["runtime_audit"] == runtime_audit


def test_daily_report_shows_aligned_visual_without_inflating_direct_role_support(
    default_config,
):
    summary = _summary_with_post_curation_rejection()
    event = summary.events[0]
    event.supporting_views = ["third"]
    event.supporting_roles = [ViewRole.THIRD_PERSON]

    report = build_daily_report(
        summary,
        {},
        {"passed": True, "checks": []},
        default_config,
        _accepted_quality(),
    )
    evaluation = evaluate_daily_report(report, summary)
    visual = report["experiment_timeline"][0]["representative_visual"]

    assert evaluation["passed"] is True
    assert visual is not None
    assert visual["visual_roles"] == ["first_person", "third_person"]
    assert visual["supporting_roles"] == ["third_person"]
    assert visual["support_scope"] == (
        "single_role_direct_with_aligned_cross_role_context"
    )
    assert visual["claim_class"] == "supported_model_understanding"


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

    quality = json.loads(
        (root / "quality_acceptance.json").read_text(encoding="utf-8-sig")
    )
    report = build_daily_report(
        summary, metrics, evidence_eval, default_config, quality
    )
    evaluation = evaluate_daily_report(report, summary)

    assert evaluation["passed"] is True
    assert report["overview"]["experiment_group_count"] == 5
    assert report["overview"]["key_event_count"] == 32
    assert report["source_policy"]["additional_model_tokens"]["total_tokens"] == 0
    assert report["template_id"] == "VC-LAB-DAILY-REPORT-V2"
    assert report["source_policy"]["layout_editable_by_model"] is False
    assert (
        report["presentation_contract"]["professional_template_id"]
        == "VC-PROFESSIONAL-EVIDENCE-REPORT-V1"
    )
    assert report["source_policy"]["report_narrative_source"] == (
        "accepted_existing_model_understanding"
    )
    assert report["performance"]["total_tokens"] == 219535
