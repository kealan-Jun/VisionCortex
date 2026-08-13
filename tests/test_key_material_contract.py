from pathlib import Path

from labvision_evidence.archive import _artifact_json
from labvision_evidence.schemas import (
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ViewRole,
)


def test_key_material_json_uses_normalized_event_contract():
    event = EvidenceEvent(
        event_id="EVT-000001",
        action_type=ActionType.LIQUID_MOVEMENT,
        global_start_ms=12_000,
        global_end_ms=20_500,
        key_global_ms=16_700,
        objects=["pipette", "reagent_bottle_open", "tube"],
        confidence=0.88,
        accepted=True,
        audit_reason="cross-view evidence",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        key_frames={"fp": "fp.jpg", "tp": "tp.jpg", "aligned_first_third": "aligned.jpg"},
        key_clips={"fp": "fp.mp4", "tp": "tp.mp4", "aligned_first_third": "aligned.mp4"},
        model_understanding={
            "status": "completed",
            "action_type_confirmed": "liquid_movement",
            "current_step": "移液器从源容器吸取液体",
            "next_step": "移动至目标离心管并释放液体",
            "physical_change": {"before": "移液器在源容器外", "after": "移液器从目标容器撤回"},
            "per_view_observations": [
                {"view_id": "fp", "observation": "移液器接触源容器"},
                {"view_id": "tp", "observation": "手将移液器移向离心管"},
            ],
            "cross_view_consistency": "consistent",
            "confidence": 0.84,
            "uncertainties": [],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        },
    )
    group = ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-0001"],
        global_start_ms=10_000,
        global_end_ms=22_000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="independent",
        key_event_ids=[event.event_id],
    )
    transforms = {
        view_id: AlignmentTransform(view_id=view_id, reference_view_id="fp", confidence=1.0)
        for view_id in ("fp", "tp")
    }
    payload = _artifact_json(group, event, "key_clip", "clip.mp4", "fp", transforms)
    assert list(payload) == [
        "event_id",
        "parent_event_id",
        "actor_id",
        "workstation_id",
        "action_type",
        "action_subtype",
        "start_us",
        "end_us",
        "peak_timestamp_us",
        "time_uncertainty_us",
        "phases",
        "objects",
        "state_before",
        "state_after",
        "observations",
        "cross_view_associations",
        "decision",
        "scores",
        "key_frames",
        "key_clips",
        "evidence_ids",
        "provenance",
    ]
    assert payload["action_type"] == "liquid_transfer"
    assert payload["action_subtype"] == "pipette_transfer"
    assert payload["decision"]["observed_facts"]
    assert payload["decision"]["supported_inferences"]
    assert len(payload["key_frames"]) == 3
    assert len(payload["key_clips"]) == 3
    assert payload["provenance"]["mllm"]["usage"]["total_tokens"] == 120
