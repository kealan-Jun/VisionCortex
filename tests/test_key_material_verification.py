from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from labvision_evidence import archive
from labvision_evidence.key_material_verification import (
    SelectiveVerificationBudget,
    plan_selective_key_material_verification,
    validate_selective_key_material_verification,
)
from labvision_evidence.schemas import (
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ViewRole,
)


def _event(
    event_id: str = "EVT-1",
    action_type: ActionType = ActionType.HAND_OBJECT_CONTACT,
    objects: list[str] | None = None,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=event_id,
        action_type=action_type,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=objects or ["gloved_hand", "paper"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        semantic_review={"verdict": "confirmed"},
    )


def _receipt(
    *,
    rendered_classes: list[str],
    groups: list[dict[str, Any]] | None = None,
    same_class_suppressed: int = 0,
) -> dict[str, Any]:
    return {
        "mode": "event_participants_only",
        "rendered_classes": rendered_classes,
        "rendered_track_ids": [],
        "participant_class_groups": groups
        or [
            {"name": "actor", "classes": ["hand", "gloved_hand"]},
            {"name": "paper", "classes": ["paper"]},
        ],
        "suppressed_same_class_instance_count": same_class_suppressed,
        "interaction_relation_gate": {"rejected_box_count": 0},
    }


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "enabled": True,
        "mode": "ambiguous_or_high_risk",
        "minimum_closed_set_confidence": 0.45,
        "high_risk_actions": [
            "liquid_movement",
            "container_state_change",
            "pipette_transfer_operation",
        ],
        "max_events_per_run": 120,
        "max_views_per_event": 2,
        "max_wall_seconds_per_run": 900,
    }
    settings.update(overrides)
    return settings


def _budget(settings: dict[str, Any]) -> SelectiveVerificationBudget:
    return SelectiveVerificationBudget.from_settings(settings)


def test_clear_hand_paper_evidence_skips_expensive_verification() -> None:
    event = _event()
    detections = [
        {"class_name": "gloved_hand", "confidence": 0.91},
        {"class_name": "paper", "confidence": 0.86},
        {"class_name": "balance", "confidence": 0.99},
    ]
    settings = _settings()

    decision = plan_selective_key_material_verification(
        event,
        "fp",
        detections,
        _receipt(rendered_classes=["gloved_hand", "paper"]),
        settings,
        _budget(settings),
    )

    assert decision["should_run"] is False
    assert decision["status"] == "skipped_clear_closed_set_evidence"
    assert decision["assessment"]["ambiguous"] is False
    assert decision["source_copy_bytes"] == 0
    assert decision["ark_calls"] == 0


def test_missing_participant_is_admitted_for_bounded_verification() -> None:
    event = _event()
    settings = _settings()
    budget = _budget(settings)

    decision = plan_selective_key_material_verification(
        event,
        "fp",
        [{"class_name": "gloved_hand", "confidence": 0.91}],
        _receipt(rendered_classes=["gloved_hand"]),
        settings,
        budget,
    )

    assert decision["should_run"] is True
    assert decision["status"] == "admitted"
    assert decision["reason"] == "ambiguous_closed_set_evidence"
    assert decision["assessment"]["missing_participant_groups"] == ["paper"]
    assert budget.summary()["admitted_view_count"] == 1


def test_actor_required_action_with_only_object_is_treated_as_ambiguous() -> None:
    event = _event(objects=["paper"])
    settings = _settings()

    decision = plan_selective_key_material_verification(
        event,
        "fp",
        [{"class_name": "paper", "confidence": 0.91}],
        _receipt(
            rendered_classes=["paper"],
            groups=[{"name": "paper", "classes": ["paper"]}],
        ),
        settings,
        _budget(settings),
    )

    assert decision["should_run"] is True
    assert "required_actor_missing" in decision["assessment"]["reasons"]


def test_high_risk_pipette_event_is_verified_even_when_closed_set_is_clear() -> None:
    event = _event(
        action_type=ActionType.PIPETTE_TRANSFER_OPERATION,
        objects=["gloved_hand", "pipette", "beaker"],
    )
    settings = _settings()
    groups = [
        {"name": "actor", "classes": ["hand", "gloved_hand"]},
        {"name": "tool", "classes": ["pipette"]},
        {"name": "vessel", "classes": ["beaker"]},
    ]
    detections = [
        {"class_name": "gloved_hand", "confidence": 0.91},
        {"class_name": "pipette", "confidence": 0.88},
        {"class_name": "beaker", "confidence": 0.92},
    ]

    decision = plan_selective_key_material_verification(
        event,
        "fp",
        detections,
        _receipt(rendered_classes=["gloved_hand", "pipette", "beaker"], groups=groups),
        settings,
        _budget(settings),
    )

    assert decision["should_run"] is True
    assert decision["status"] == "admitted"
    assert decision["reason"] == "high_risk_action"
    assert decision["assessment"]["ambiguous"] is False


def test_event_budget_defers_without_retrying_or_hiding_evidence() -> None:
    settings = _settings(max_events_per_run=1)
    budget = _budget(settings)
    missing = _receipt(rendered_classes=["gloved_hand"])
    detections = [{"class_name": "gloved_hand", "confidence": 0.91}]

    first = plan_selective_key_material_verification(
        _event("EVT-1"), "fp", detections, missing, settings, budget
    )
    second = plan_selective_key_material_verification(
        _event("EVT-2"), "fp", detections, missing, settings, budget
    )

    assert first["status"] == "admitted"
    assert second["should_run"] is False
    assert second["status"] == "deferred_budget_exhausted"
    assert second["reason"] == "event_budget_exhausted"
    assert budget.summary()["deferred_reasons"] == {"event_budget_exhausted": 1}


def test_disabled_policy_preserves_existing_supplement_behavior() -> None:
    settings = _settings(enabled=False)

    decision = plan_selective_key_material_verification(
        _event(),
        "fp",
        [],
        _receipt(rendered_classes=[]),
        settings,
        _budget(settings),
    )

    assert decision["enabled"] is False
    assert decision["should_run"] is True
    assert decision["status"] == "legacy_unbounded_policy"


def test_invalid_policy_fails_closed_before_model_execution() -> None:
    settings = _settings(minimum_closed_set_confidence=1.5)

    with pytest.raises(ValueError, match="between 0 and 1"):
        plan_selective_key_material_verification(
            _event(),
            "fp",
            [],
            _receipt(rendered_classes=[]),
            settings,
            _budget(settings),
        )


def test_policy_preflight_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_events_per_run must be positive"):
        validate_selective_key_material_verification(_settings(max_events_per_run=0))


def test_archive_skips_expensive_model_for_clear_participant_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "work"
    event = _event()
    event.key_frames = {
        "fp": "Key-Materials/fp.jpg",
        "tp": "Key-Materials/tp.jpg",
        "aligned_first_third": "Key-Materials/aligned.jpg",
    }
    group = ExperimentGroup(
        group_id="GROUP-1",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-1"],
        global_start_ms=1000,
        global_end_ms=2000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
        experiment_name="Test",
        experiment_name_en="Test",
        archive_folder="001-Test",
        key_event_ids=[event.event_id],
    )
    input_root = layout.work / "key-material-annotation-inputs" / event.event_id
    input_root.mkdir(parents=True)
    detections = [
        {
            "class_name": "gloved_hand",
            "confidence": 0.91,
            "xyxy_norm": [0.10, 0.10, 0.35, 0.60],
        },
        {
            "class_name": "paper",
            "confidence": 0.86,
            "xyxy_norm": [0.30, 0.45, 0.62, 0.75],
        },
        {
            "class_name": "balance",
            "confidence": 0.99,
            "xyxy_norm": [0.65, 0.10, 0.95, 0.80],
        },
    ]
    for role_label in ("First-Person", "Third-Person"):
        assert cv2.imwrite(
            str(input_root / f"{role_label}.jpg"),
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        (input_root / f"{role_label}.json").write_text(
            json.dumps({"detections": detections}), encoding="utf-8"
        )

    def unexpected_supplement(
        *args: Any, **kwargs: Any
    ) -> tuple[list[Any], dict[str, Any]]:
        raise AssertionError(
            "clear closed-set evidence must not invoke supplementation"
        )

    monkeypatch.setattr(
        archive, "_open_vocabulary_key_frame_supplement", unexpected_supplement
    )
    report = archive._rerender_curated_participant_annotations(
        layout,
        [event],
        [group],
        {
            "key_materials": {"selective_verification": _settings()},
            "models": {
                "open_vocabulary_key_frame": {
                    "manipulated_object_max_actor_gap_norm": 0.08
                }
            },
        },
    )

    verification = report["selective_verification"]
    assert verification["decision_status_counts"] == {
        "skipped_clear_closed_set_evidence": 2
    }
    assert verification["open_vocabulary_executed_count"] == 0
    assert verification["source_copy_bytes"] == 0
    assert verification["budget"]["admitted_view_count"] == 0
    receipts = event.observability["key_material_annotation"]["views"].values()
    assert all(
        item["rendered_classes"] == ["gloved_hand", "paper"] for item in receipts
    )
    assert all(item["minimum_rendered_confidence"] == 0.86 for item in receipts)
    assert all(
        [box["class_name"] for box in item["rendered_detections"]]
        == ["gloved_hand", "paper"]
        for item in receipts
    )
    assert all("balance" in item["suppressed_background_classes"] for item in receipts)
