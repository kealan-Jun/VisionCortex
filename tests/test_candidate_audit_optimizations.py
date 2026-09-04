from __future__ import annotations

from copy import deepcopy

from visioncortex.action_semantics import (
    attach_action_observability,
    build_semantic_review_plan,
)
from visioncortex.action_state_machine import build_event_state_receipt
from visioncortex.actions import (
    audit_candidates,
    build_physical_change_log,
    refine_liquid_events_with_context,
)
from visioncortex.candidate_index import create_fine_frame_index
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentSegmentTransform,
    AlignmentTransform,
    BoxEvidence,
    EvidenceEvent,
    FrameEvidence,
    ViewRole,
)


def _candidate(
    candidate_id: str,
    *,
    action_type: ActionType = ActionType.HAND_OBJECT_CONTACT,
    view_id: str = "fp",
    role: ViewRole = ViewRole.FIRST_PERSON,
    start_ms: float = 1_000.0,
    end_ms: float = 2_000.0,
    confidence: float = 0.9,
    objects: list[str] | None = None,
    evidence: list[dict] | None = None,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        view_id=view_id,
        role=role,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=objects or ["gloved_hand", "tube"],
        confidence=confidence,
        evidence=evidence or [{"distance_norm": 0.01}],
    )


def _transform(
    view_id: str,
    *,
    role_state: str = "aligned",
    segments: list[AlignmentSegmentTransform] | None = None,
) -> AlignmentTransform:
    return AlignmentTransform(
        view_id=view_id,
        reference_view_id="fp",
        confidence=0.95,
        uncertainty_ms=25.0,
        state=role_state,
        segment_transforms=segments or [],
    )


def _optimized_config(default_config):
    config = deepcopy(default_config)
    config["performance"].update(
        {
            "audit_constrained_clustering_enabled": True,
            "audit_local_alignment_enabled": True,
            "audit_explainable_scoring_enabled": True,
            "audit_formal_admission_status_enabled": True,
            "audit_core_interval_enabled": True,
            "audit_stable_event_ids_enabled": True,
            "audit_sqlite_candidate_index_enabled": True,
            "audit_liquid_spatial_context_enabled": True,
            "audit_liquid_context_hand_object_gap_norm": 0.08,
            "audit_liquid_context_minimum_frames": 2,
            "audit_liquid_sequence_formal_gate_enabled": True,
            "action_state_observed_timestamps_enabled": True,
        }
    )
    return config


def test_constrained_clustering_prevents_transitive_bridge_and_keeps_stable_ids(
    default_config,
):
    config = _optimized_config(default_config)
    candidates = [
        _candidate("A", start_ms=0, end_ms=1_000),
        _candidate("B", start_ms=1_500, end_ms=2_500),
        _candidate("C", start_ms=3_000, end_ms=4_000),
    ]
    transforms = {"fp": _transform("fp")}

    events, _ = audit_candidates(candidates, transforms, config)
    reordered, _ = audit_candidates(list(reversed(candidates)), transforms, config)

    assert len(events) == 2
    assert [event.event_id for event in events] == [
        event.event_id for event in reordered
    ]
    assert all(event.event_id.startswith("EVT-") for event in events)
    assert all(event.event_fingerprint for event in events)


def test_candidate_local_failed_alignment_cannot_enter_formal_evidence(default_config):
    config = _optimized_config(default_config)
    candidate = _candidate("FAILED-LOCAL")
    segment = AlignmentSegmentTransform(
        segment_index=0,
        local_start_ms=0,
        local_end_ms=5_000,
        confidence=0.0,
        uncertainty_ms=1_000,
        state="failed",
    )

    events, rejected = audit_candidates(
        [candidate],
        {"fp": _transform("fp", segments=[segment])},
        config,
    )

    assert events[0].formal_admission_status == "rejected"
    assert events[0].observability["candidate_alignment"]["candidates"][
        "FAILED-LOCAL"
    ]["available"] is False
    assert rejected[0]["event_id"] == events[0].event_id


def test_explainable_score_core_interval_and_sqlite_lookup_are_auditable(
    default_config, tmp_path
):
    config = _optimized_config(default_config)
    candidates = [
        _candidate("FP-WIDE", start_ms=0, end_ms=10_000, confidence=0.1),
        _candidate("FP-CORE", start_ms=4_000, end_ms=6_000, confidence=0.9),
        _candidate(
            "TP-CORE",
            view_id="tp",
            role=ViewRole.THIRD_PERSON,
            start_ms=4_100,
            end_ms=6_100,
            confidence=0.8,
        ),
    ]
    index = create_fine_frame_index(tmp_path / "fine.sqlite")

    events, _ = audit_candidates(
        candidates,
        {"fp": _transform("fp"), "tp": _transform("tp")},
        config,
        index,
    )

    assert len(events) == 1
    event = events[0]
    assert (event.core_global_start_ms, event.core_global_end_ms) == (4_000, 6_000)
    assert (event.evidence_global_start_ms, event.evidence_global_end_ms) == (
        0,
        10_000,
    )
    receipt = event.observability["confidence_fusion"]
    assert receipt["method"] == "best_per_view_alignment_weighted_independent_support"
    assert len(receipt["contributors"]) == 2
    assert event.observability["audit_lookup"]["strategy"] == "sqlite_time_index"
    assert [item.candidate_id for item in index.iter_audit_candidates(
        start_ms=4_500, end_ms=4_600
    )] == ["FP-WIDE", "FP-CORE", "TP-CORE"]


def test_provisional_event_stays_in_review_but_out_of_formal_changes(default_config):
    config = _optimized_config(default_config)
    direct = _candidate(
        "STATE-FP",
        action_type=ActionType.CONTAINER_STATE_CHANGE,
        confidence=0.5,
        objects=["gloved_hand", "bottle_cap"],
    )
    context = _candidate(
        "MOTION-TP",
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id="tp",
        role=ViewRole.THIRD_PERSON,
        confidence=0.6,
        objects=["tube"],
    )

    events, _ = audit_candidates(
        [direct, context],
        {"fp": _transform("fp"), "tp": _transform("tp")},
        config,
    )
    event = next(item for item in events if item.action_type == direct.action_type)
    attach_action_observability(events)
    plan = build_semantic_review_plan(events, config)

    assert event.accepted is True
    assert event.formal_admission_status == "provisional"
    assert event.event_id in {item["event_id"] for item in plan["selected"]}
    assert build_physical_change_log([event]) == []


def test_stratified_review_queue_batches_every_eligible_event(default_config):
    config = _optimized_config(default_config)
    config["mllm"].update(
        {
            "candidate_review_stratified_queue_enabled": True,
            "candidate_review_time_bucket_seconds": 1,
            "candidate_review_max_events": 2,
        }
    )
    events = []
    for index, action_type in enumerate(
        [
            ActionType.LIQUID_MOVEMENT,
            ActionType.LIQUID_MOVEMENT,
            ActionType.CONTAINER_STATE_CHANGE,
            ActionType.CONTAINER_STATE_CHANGE,
        ]
    ):
        candidate = _candidate(
            f"C-{index}",
            action_type=action_type,
            start_ms=index * 2_000,
            end_ms=index * 2_000 + 500,
            confidence=0.8,
            objects=["pipette", "tube"]
            if action_type == ActionType.LIQUID_MOVEMENT
            else ["gloved_hand", "bottle_cap"],
        )
        events.append(
            EvidenceEvent(
                event_id=f"E-{index}",
                action_type=action_type,
                global_start_ms=candidate.global_start_ms,
                global_end_ms=candidate.global_end_ms,
                key_global_ms=candidate.key_global_ms,
                objects=candidate.objects,
                confidence=0.8,
                accepted=True,
                formal_admission_status="provisional",
                audit_reason="review",
                supporting_views=["fp"],
                supporting_roles=[ViewRole.FIRST_PERSON],
                candidates=[candidate],
            )
        )
    attach_action_observability(events)

    plan = build_semantic_review_plan(events, config)

    assert plan["selected_count"] == 2
    assert plan["batch_count"] == 2
    assert plan["all_eligible_scheduled"] is True
    assert sum(batch["event_count"] for batch in plan["scheduled_batches"]) == 4
    assert len({item["action_type"] for item in plan["selected"]}) == 2
    assert not any(
        item.get("reason") == "semantic_budget_overflow"
        for item in plan["skipped"]
    )


def test_state_trace_uses_observed_evidence_timestamps(default_config):
    config = _optimized_config(default_config)
    candidate = _candidate(
        "CONTACT",
        evidence=[
            {
                "distance_norm": 0.01,
                "observation_global_ms": 1_200.0,
                "interaction_state": {"approach_confirmed": True},
            }
        ],
    )
    candidate.provenance = {
        "interaction_state": {
            "release_observed": True,
            "release_observed_at_global_ms": 1_800.0,
        }
    }
    event = EvidenceEvent(
        event_id="CONTACT-EVENT",
        action_type=candidate.action_type,
        global_start_ms=1_000,
        global_end_ms=2_000,
        key_global_ms=1_500,
        objects=candidate.objects,
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp"],
        supporting_roles=[ViewRole.FIRST_PERSON],
        candidates=[candidate],
    )

    receipt = build_event_state_receipt(event, config)
    by_phase = {item["phase"]: item for item in receipt["transition_trace"]}

    assert receipt["transition_trace_policy"] == (
        "observed_timestamps_with_explicit_inference_labels"
    )
    assert by_phase["contact"]["timestamp_us"] == 1_200_000
    assert by_phase["contact"]["observed"] is True
    assert by_phase["release"]["timestamp_us"] == 1_800_000


def test_liquid_context_rejects_unrelated_distant_hand(default_config, tmp_path):
    config = _optimized_config(default_config)
    candidate = _candidate(
        "LIQUID",
        action_type=ActionType.LIQUID_MOVEMENT,
        objects=["pipette", "tube"],
        evidence=[{"tool_track_id": 7, "roi_motion": 8.0}],
    )
    event = EvidenceEvent(
        event_id="LIQUID-EVENT",
        action_type=ActionType.LIQUID_MOVEMENT,
        global_start_ms=1_000,
        global_end_ms=2_000,
        key_global_ms=1_500,
        objects=candidate.objects,
        confidence=0.9,
        accepted=True,
        formal_admission_status="provisional",
        audit_reason="test",
        supporting_views=["fp"],
        supporting_roles=[ViewRole.FIRST_PERSON],
        candidates=[candidate],
    )
    frames = [
        FrameEvidence(
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            frame_index=index,
            local_ms=global_ms,
            global_ms=global_ms,
            width=100,
            height=100,
            detections=[
                BoxEvidence(
                    class_id=0,
                    class_name="gloved_hand",
                    confidence=0.9,
                    xyxy_norm=(0.0, 0.0, 0.1, 0.1),
                    track_id=1,
                ),
                BoxEvidence(
                    class_id=1,
                    class_name="pipette",
                    confidence=0.9,
                    xyxy_norm=(0.8, 0.8, 0.9, 0.9),
                    track_id=7,
                ),
            ],
        )
        for index, global_ms in enumerate((1_100.0, 1_600.0))
    ]
    ledger = tmp_path / "fp.jsonl"
    ledger.write_text(
        "\n".join(frame.model_dump_json() for frame in frames) + "\n",
        encoding="utf-8",
    )

    rejected = refine_liquid_events_with_context(
        [event], {"fp": ledger}, config=config
    )

    assert event.formal_admission_status == "rejected"
    assert rejected[0]["spatial_support_frame_counts"] == {"fp": 0}
    assert event.observability["liquid_context"]["hand_frame_counts"] == {"fp": 2}
