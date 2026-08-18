from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from labvision_evidence.action_semantics import (
    attach_action_observability,
    build_semantic_review_plan,
    record_semantic_review,
)
from labvision_evidence.actions import (
    _Observation,
    _frame_observations,
    _infer_liquid_transfer_sequences,
    audit_candidates,
)
from labvision_evidence.archive import extract_temporal_review_frames
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    BoxEvidence,
    EvidenceEvent,
    FrameEvidence,
    ViewInput,
    ViewRole,
)


def _candidate(
    candidate_id: str,
    action_type: ActionType,
    view_id: str,
    role: ViewRole,
    objects: list[str],
    *,
    confidence: float = 0.9,
    evidence: list[dict] | None = None,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        view_id=view_id,
        role=role,
        local_start_ms=10_000,
        local_end_ms=11_000,
        global_start_ms=10_000,
        global_end_ms=11_000,
        key_global_ms=10_500,
        objects=objects,
        confidence=confidence,
        evidence=evidence or [],
    )


def _event(
    event_id: str,
    action_type: ActionType,
    candidates: list[ActionCandidate],
    *,
    accepted: bool = True,
    confidence: float = 0.9,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=event_id,
        action_type=action_type,
        global_start_ms=10_000,
        global_end_ms=11_000,
        key_global_ms=10_500,
        objects=sorted({item for candidate in candidates for item in candidate.objects}),
        confidence=confidence,
        accepted=accepted,
        audit_reason="test",
        supporting_views=sorted({candidate.view_id for candidate in candidates}),
        supporting_roles=sorted(
            {candidate.role for candidate in candidates}, key=lambda item: item.value
        ),
        candidates=candidates,
    )


def test_observability_separates_direct_movement_from_indirect_liquid(default_config):
    movement = _event(
        "MOVE",
        ActionType.OBJECT_MOVEMENT,
        [
            _candidate(
                "MOVE-FP",
                ActionType.OBJECT_MOVEMENT,
                "fp",
                ViewRole.FIRST_PERSON,
                ["tube"],
                evidence=[
                    {
                        "track_id": 7,
                        "displacement_norm": 0.08,
                        "camera_compensated_displacement_norm": 0.05,
                    }
                ],
            ),
            _candidate(
                "MOVE-TP",
                ActionType.OBJECT_MOVEMENT,
                "tp",
                ViewRole.THIRD_PERSON,
                ["tube"],
                evidence=[{"track_id": 8, "displacement_norm": 0.06}],
            ),
        ],
    )
    liquid = _event(
        "LIQUID",
        ActionType.LIQUID_MOVEMENT,
        [
            _candidate(
                "LIQUID-FP",
                ActionType.LIQUID_MOVEMENT,
                "fp",
                ViewRole.FIRST_PERSON,
                ["pipette", "tube"],
                evidence=[{"distance_norm": 0.01, "roi_motion": 8.0}],
            )
        ],
    )

    receipts = attach_action_observability([movement, liquid])
    plan = build_semantic_review_plan([movement, liquid], default_config)

    assert receipts[0]["cv_ontology_support"] == "direct_cv"
    assert receipts[0]["semantic_review_priority"] == "optional"
    assert receipts[1]["cv_ontology_support"] == "indirect_cv"
    assert "visible_liquid_region" in receipts[1]["unmet_visual_requirements"]
    assert [item["event_id"] for item in plan["selected"]] == ["LIQUID"]
    assert plan["model_may_mutate_cv_acceptance"] is False
    assert liquid.accepted is True


def test_rejected_high_confidence_candidate_enters_bounded_semantic_recall_plan(
    default_config,
):
    candidate = _candidate(
        "CONTACT-FP",
        ActionType.HAND_OBJECT_CONTACT,
        "fp",
        ViewRole.FIRST_PERSON,
        ["gloved_hand", "paper"],
        evidence=[{"distance_norm": 0.0, "object_track_id": 22}],
    )
    event = _event(
        "CONTACT",
        ActionType.HAND_OBJECT_CONTACT,
        [candidate],
        accepted=False,
        confidence=0.88,
    )
    attach_action_observability([event])

    plan = build_semantic_review_plan([event], default_config)

    assert plan["selected"][0]["accepted_by_cv"] is False
    assert event.accepted is False


def test_semantic_review_can_suggest_relabel_without_mutating_cv_acceptance(
    default_config,
):
    candidate = _candidate(
        "CONTACT-FP",
        ActionType.HAND_OBJECT_CONTACT,
        "fp",
        ViewRole.FIRST_PERSON,
        ["gloved_hand", "tube"],
        evidence=[{"distance_norm": 0.0}],
    )
    event = _event("CONTACT", ActionType.HAND_OBJECT_CONTACT, [candidate])
    attach_action_observability([event])

    receipt = record_semantic_review(
        event,
        {
            "status": "completed",
            "model": "test-model",
            "action_type_confirmed": "container_state_change",
            "confidence": 0.8,
            "cross_view_consistency": "partial",
            "usage": {"total_tokens": 10},
        },
    )

    assert receipt["verdict"] == "relabel_suggested"
    assert receipt["model_mutated_cv_acceptance"] is False
    assert event.accepted is True
    assert event.semantic_review == receipt


def test_camera_translation_is_removed_before_object_movement(default_config):
    cfg = default_config["segmentation"]
    previous_tracks: dict[int, tuple[float, float, float]] = {}

    def box(class_name: str, track_id: int, x: float) -> BoxEvidence:
        return BoxEvidence(
            class_id=0,
            class_name=class_name,
            confidence=0.9,
            xyxy_norm=(x, 0.2, x + 0.1, 0.3),
            track_id=track_id,
        )

    first = FrameEvidence(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        frame_index=1,
        local_ms=0,
        global_ms=0,
        width=640,
        height=360,
        detections=[box("beaker", 1, 0.1), box("tube", 2, 0.4), box("balance", 3, 0.7)],
    )
    second = first.model_copy(
        update={
            "frame_index": 2,
            "local_ms": 100,
            "global_ms": 100,
            "detections": [
                box("beaker", 1, 0.2),
                box("tube", 2, 0.5),
                box("balance", 3, 0.8),
            ],
        }
    )

    _frame_observations(first, previous_tracks, cfg)
    observations = _frame_observations(second, previous_tracks, cfg)

    assert not any(item.action_type == ActionType.OBJECT_MOVEMENT for item in observations)


def test_tracked_source_transport_target_sequence_adds_liquid_recall_candidate(
    default_config,
):
    observations = []
    for local_ms in (1_000.0, 1_100.0, 1_200.0):
        observations.append(
            _Observation(
                action_type=ActionType.LIQUID_MOVEMENT,
                local_ms=local_ms,
                global_ms=local_ms,
                objects=("pipette", "sample_bottle"),
                confidence=0.8,
                evidence={
                    "tool_class": "pipette",
                    "tool_track_id": 7,
                    "vessel_class": "sample_bottle",
                    "vessel_track_id": 11,
                },
            )
        )
    for local_ms in (2_000.0, 2_100.0, 2_200.0):
        observations.append(
            _Observation(
                action_type=ActionType.LIQUID_MOVEMENT,
                local_ms=local_ms,
                global_ms=local_ms,
                objects=("pipette", "tube"),
                confidence=0.82,
                evidence={
                    "tool_class": "pipette",
                    "tool_track_id": 7,
                    "vessel_class": "tube",
                    "vessel_track_id": 12,
                },
            )
        )
    view = ViewInput(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        video=Path("fp.mp4"),
    )

    candidates = _infer_liquid_transfer_sequences(
        observations, view, default_config["segmentation"]
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.objects == ["pipette", "sample_bottle", "tube"]
    assert candidate.evidence[0]["transfer_sequence"] == "source_transport_target"
    assert candidate.evidence[0]["source_track_id"] == 11
    assert candidate.evidence[0]["target_track_id"] == 12
    assert "液体本体" in candidate.uncertainty[0]


def test_unrelated_panel_and_container_candidates_do_not_cross_view_merge(default_config):
    transforms = {
        view_id: AlignmentTransform(
            view_id=view_id,
            reference_view_id="fp",
            confidence=0.9,
            state="aligned",
        )
        for view_id in ("fp", "tp")
    }
    candidates = [
        _candidate(
            "PANEL-FP",
            ActionType.DEVICE_PANEL_OPERATION,
            "fp",
            ViewRole.FIRST_PERSON,
            ["hand", "balance"],
        ),
        _candidate(
            "PANEL-TP",
            ActionType.DEVICE_PANEL_OPERATION,
            "tp",
            ViewRole.THIRD_PERSON,
            ["hand", "magnetic_stirrer"],
        ),
        _candidate(
            "STATE-FP",
            ActionType.CONTAINER_STATE_CHANGE,
            "fp",
            ViewRole.FIRST_PERSON,
            ["hand", "tube_cap"],
        ),
        _candidate(
            "STATE-TP",
            ActionType.CONTAINER_STATE_CHANGE,
            "tp",
            ViewRole.THIRD_PERSON,
            ["hand", "bottle_cap"],
        ),
    ]

    events, _ = audit_candidates(candidates, transforms, default_config)

    assert len(events) == 4
    assert all(len(event.candidates) == 1 for event in events)


def test_temporal_review_reuses_short_key_clip(tmp_path: Path):
    clip = tmp_path / "key.avi"
    writer = cv2.VideoWriter(
        str(clip),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
    writer.release()

    samples = extract_temporal_review_frames(clip, tmp_path / "samples", "fp01", 3)

    assert len(samples) == 3
    assert ["temporal_phase=before" in samples[0][0], "temporal_phase=peak" in samples[1][0], "temporal_phase=after" in samples[2][0]] == [True, True, True]
    assert all(path.is_file() and path.stat().st_size > 0 for _, path in samples)
