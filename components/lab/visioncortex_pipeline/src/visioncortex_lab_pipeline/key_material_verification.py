from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .schemas import ActionType, EvidenceEvent


DEFAULT_HIGH_RISK_ACTIONS = frozenset(
    {
        ActionType.LIQUID_MOVEMENT.value,
        ActionType.CONTAINER_STATE_CHANGE.value,
        ActionType.PIPETTE_TRANSFER_OPERATION.value,
    }
)


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _positive_int(settings: dict[str, Any], key: str, default: int) -> int:
    value = int(settings.get(key, default))
    if value <= 0:
        raise ValueError(f"key_materials.selective_verification.{key} must be positive")
    return value


def _positive_float(settings: dict[str, Any], key: str, default: float) -> float:
    value = float(settings.get(key, default))
    if value <= 0.0:
        raise ValueError(f"key_materials.selective_verification.{key} must be positive")
    return value


@dataclass
class SelectiveVerificationBudget:
    """Bound expensive final-frame verification without blocking production.

    The wall clock is a soft *start* budget: an inference already admitted is
    allowed to finish so a model call is never interrupted into a corrupt
    receipt. Event and view limits are deterministic hard admission limits.
    """

    max_events: int
    max_views_per_event: int
    max_wall_seconds: float
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)
    admitted_views_by_event: dict[str, set[str]] = field(default_factory=dict)
    deferred_reasons: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if self.max_views_per_event <= 0:
            raise ValueError("max_views_per_event must be positive")
        if self.max_wall_seconds <= 0.0:
            raise ValueError("max_wall_seconds must be positive")
        self.started_at = self.clock()

    @classmethod
    def from_settings(
        cls,
        settings: dict[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> SelectiveVerificationBudget:
        return cls(
            max_events=_positive_int(settings, "max_events_per_run", 120),
            max_views_per_event=_positive_int(settings, "max_views_per_event", 2),
            max_wall_seconds=_positive_float(
                settings, "max_wall_seconds_per_run", 900.0
            ),
            clock=clock,
        )

    def reserve(self, event_id: str, view_id: str) -> tuple[bool, str | None]:
        elapsed = self.clock() - self.started_at
        if elapsed >= self.max_wall_seconds:
            self._defer("wall_time_budget_exhausted")
            return False, "wall_time_budget_exhausted"
        admitted = self.admitted_views_by_event.get(event_id)
        if admitted is None:
            if len(self.admitted_views_by_event) >= self.max_events:
                self._defer("event_budget_exhausted")
                return False, "event_budget_exhausted"
            admitted = set()
            self.admitted_views_by_event[event_id] = admitted
        if view_id in admitted:
            return True, None
        if len(admitted) >= self.max_views_per_event:
            self._defer("view_budget_exhausted")
            return False, "view_budget_exhausted"
        admitted.add(view_id)
        return True, None

    def _defer(self, reason: str) -> None:
        self.deferred_reasons[reason] = self.deferred_reasons.get(reason, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "max_events_per_run": self.max_events,
            "max_views_per_event": self.max_views_per_event,
            "max_wall_seconds_per_run": self.max_wall_seconds,
            "admitted_event_count": len(self.admitted_views_by_event),
            "admitted_view_count": sum(
                len(view_ids) for view_ids in self.admitted_views_by_event.values()
            ),
            "deferred_count": sum(self.deferred_reasons.values()),
            "deferred_reasons": dict(sorted(self.deferred_reasons.items())),
            "elapsed_seconds": round(self.clock() - self.started_at, 6),
        }


def validate_selective_key_material_verification(
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Validate the policy during preflight, before any long video scan."""

    mode = str(settings.get("mode") or "ambiguous_or_high_risk")
    if mode != "ambiguous_or_high_risk":
        raise ValueError(
            "key_materials.selective_verification.mode must be ambiguous_or_high_risk"
        )
    minimum_confidence = float(settings.get("minimum_closed_set_confidence", 0.45))
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError(
            "key_materials.selective_verification.minimum_closed_set_confidence "
            "must be between 0 and 1"
        )
    high_risk_actions = {
        str(item) for item in settings.get("high_risk_actions") or []
    } or set(DEFAULT_HIGH_RISK_ACTIONS)
    invalid_actions = high_risk_actions - {item.value for item in ActionType}
    if invalid_actions:
        raise ValueError(
            "Unknown selective verification action(s): "
            + ", ".join(sorted(invalid_actions))
        )
    return {
        "schema_version": "visioncortex-selective-key-material-verification-policy/1",
        "enabled": bool(settings.get("enabled", False)),
        "status": "validated",
        "mode": mode,
        "minimum_closed_set_confidence": minimum_confidence,
        "high_risk_actions": sorted(high_risk_actions),
        "max_events_per_run": _positive_int(settings, "max_events_per_run", 120),
        "max_views_per_event": _positive_int(settings, "max_views_per_event", 2),
        "max_wall_seconds_per_run": _positive_float(
            settings, "max_wall_seconds_per_run", 900.0
        ),
        "full_timeline_inference": False,
        "source_copy_bytes": 0,
        "ark_calls": 0,
    }


def _participant_assessment(
    event: EvidenceEvent,
    detections: Sequence[dict[str, Any]],
    participant_receipt: dict[str, Any],
    *,
    minimum_confidence: float,
) -> dict[str, Any]:
    rendered_classes = {
        _normalize(item) for item in participant_receipt.get("rendered_classes") or []
    }
    groups = list(participant_receipt.get("participant_class_groups") or [])
    missing_groups = sorted(
        str(group.get("name") or "participant")
        for group in groups
        if not (
            {_normalize(item) for item in group.get("classes") or []} & rendered_classes
        )
    )
    rendered_track_ids = set(participant_receipt.get("rendered_track_ids") or [])
    rendered_detections = [
        box
        for box in detections
        if _normalize(box.get("class_name")) in rendered_classes
        and (
            not rendered_track_ids
            or box.get("track_id") is None
            or box.get("track_id") in rendered_track_ids
        )
    ]
    low_confidence_classes = sorted(
        {
            _normalize(box.get("class_name"))
            for box in rendered_detections
            if float(box.get("confidence") or 0.0) < minimum_confidence
        }
    )
    semantic_verdict = str(
        (event.semantic_review or {}).get("verdict")
        or (event.model_understanding or {}).get("evidence_verdict")
        or ""
    )
    semantic_uncertain = bool(
        semantic_verdict and semantic_verdict not in {"confirmed", "relabel_suggested"}
    )
    cross_view_consistency = str(
        (event.semantic_review or {}).get("cross_view_consistency")
        or (event.model_understanding or {}).get("cross_view_consistency")
        or ""
    )
    multiple_instances = int(
        participant_receipt.get("suppressed_same_class_instance_count") or 0
    )
    relation_rejections = int(
        (participant_receipt.get("interaction_relation_gate") or {}).get(
            "rejected_box_count"
        )
        or 0
    )
    reasons: list[str] = []
    if missing_groups:
        reasons.append("missing_participant_groups")
    if low_confidence_classes:
        reasons.append("low_closed_set_confidence")
    if multiple_instances:
        reasons.append("multiple_participant_instances")
    if relation_rejections:
        reasons.append("non_interacting_candidates_rejected")
    if semantic_uncertain:
        reasons.append("semantic_verdict_uncertain")
    actor_classes = {"hand", "gloved_hand"}
    actor_required = event.action_type in {
        ActionType.HAND_OBJECT_CONTACT,
        ActionType.CONTAINER_STATE_CHANGE,
        ActionType.DEVICE_PANEL_OPERATION,
        ActionType.PIPETTE_TRANSFER_OPERATION,
    }
    if actor_required and not rendered_classes & actor_classes:
        reasons.append("required_actor_missing")
    if not rendered_classes - actor_classes:
        reasons.append("required_manipulated_object_missing")
    if cross_view_consistency == "conflict":
        reasons.append("cross_view_conflict")
    return {
        "ambiguous": bool(reasons),
        "reasons": reasons,
        "missing_participant_groups": missing_groups,
        "low_confidence_classes": low_confidence_classes,
        "multiple_participant_instance_count": multiple_instances,
        "non_interacting_candidate_count": relation_rejections,
        "semantic_verdict": semantic_verdict or None,
        "cross_view_consistency": cross_view_consistency or None,
        "minimum_closed_set_confidence": minimum_confidence,
    }


def plan_selective_key_material_verification(
    event: EvidenceEvent,
    view_id: str,
    detections: Sequence[dict[str, Any]],
    participant_receipt: dict[str, Any],
    settings: dict[str, Any],
    budget: SelectiveVerificationBudget,
) -> dict[str, Any]:
    """Decide whether one final role frame needs expensive local verification."""

    policy = validate_selective_key_material_verification(settings)
    if not policy["enabled"]:
        return {
            "schema_version": "visioncortex-selective-key-material-verification/1",
            "status": "legacy_unbounded_policy",
            "enabled": False,
            "should_run": True,
            "reason": "feature_disabled_preserve_existing_behavior",
            "event_id": event.event_id,
            "view_id": view_id,
        }
    mode = str(policy["mode"])
    minimum_confidence = float(policy["minimum_closed_set_confidence"])
    high_risk_actions = set(policy["high_risk_actions"])
    assessment = _participant_assessment(
        event,
        detections,
        participant_receipt,
        minimum_confidence=minimum_confidence,
    )
    high_risk = event.action_type.value in high_risk_actions
    requested = bool(high_risk or assessment["ambiguous"])
    decision = {
        "schema_version": "visioncortex-selective-key-material-verification/1",
        "enabled": True,
        "mode": mode,
        "event_id": event.event_id,
        "view_id": view_id,
        "action_type": event.action_type.value,
        "high_risk_action": high_risk,
        "assessment": assessment,
        "full_timeline_inference": False,
        "source_copy_bytes": 0,
        "ark_calls": 0,
    }
    if not requested:
        return {
            **decision,
            "status": "skipped_clear_closed_set_evidence",
            "should_run": False,
            "reason": "closed_set_participants_complete_and_confident",
        }
    admitted, deferred_reason = budget.reserve(event.event_id, view_id)
    if not admitted:
        return {
            **decision,
            "status": "deferred_budget_exhausted",
            "should_run": False,
            "reason": deferred_reason,
        }
    return {
        **decision,
        "status": "admitted",
        "should_run": True,
        "reason": "high_risk_action" if high_risk else "ambiguous_closed_set_evidence",
    }
