from pathlib import Path

import pytest

from visioncortex.actions import _Observation, _StreamingLiquidSequences, _infer_liquid_transfer_sequences, _merge_observations
from visioncortex.action_state_machine import build_event_state_receipt
from visioncortex.schemas import ActionType, EvidenceEvent, ViewInput, ViewRole


def observation(timestamp, vessel):
    return _Observation(
        action_type=ActionType.LIQUID_MOVEMENT, local_ms=timestamp, global_ms=timestamp,
        objects=("pipette", "beaker"), confidence=.8,
        evidence={"tool_class": "pipette", "tool_track_id": 1,
                  "vessel_class": "beaker", "vessel_track_id": vessel},
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_exclusive_contact_runs_reject_simultaneity_flicker_and_duplicate_samples(default_config, streaming):
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("unused.mp4"))
    cfg = default_config["segmentation"]

    def infer(items):
        if not streaming:
            return _infer_liquid_transfer_sequences(items, view, cfg)
        reducer = _StreamingLiquidSequences(view, cfg)
        for timestamp in sorted({item.global_ms for item in items}):
            reducer.expire(timestamp)
            reducer.add_frame([item for item in items if item.global_ms == timestamp])
        return reducer.finish()

    simultaneous = [observation(t, v) for t in range(0, 1001, 50) for v in (10, 11)]
    assert infer(simultaneous) == []
    assert _merge_observations(simultaneous, view, cfg), "Recall observations must remain available"
    flicker = [observation(t, 10 if t < 100 else 11) for t in (0, 50, 100, 150)]
    assert infer(flicker) == []
    duplicates = [observation(0, 10)] * 20 + [observation(100, 11)] * 20
    assert infer(duplicates) == []
    valid = [observation(t, 10) for t in (0, 100, 200)] + [observation(t, 11) for t in (800, 900, 1000)]
    candidates = infer(valid)
    assert len(candidates) == 1
    receipt = candidates[0].evidence[0]
    assert receipt["source_observation_count"] == 3
    assert receipt["source_contact_duration_ms"] == 200
    assert receipt["transport_gap_ms"] == 600
    assert receipt["transport_observed"] is False
    # An ambiguous intermediate contact must not bridge two exclusive runs.
    assert infer(valid + [observation(400, 10), observation(400, 11)]) == []

    candidate = candidates[0]
    event = EvidenceEvent(event_id="test", action_type=candidate.action_type,
        global_start_ms=0, global_end_ms=1000, key_global_ms=500,
        objects=candidate.objects, confidence=.8, accepted=True, audit_reason="CV candidate",
        supporting_views=["fp"], supporting_roles=[ViewRole.FIRST_PERSON], candidates=candidates)
    state = build_event_state_receipt(event, default_config)
    assert state["lifecycle_state"] != "completed"
    assert "transport" in state["missing_phases"]
    assert state["scores"]["phase_completeness"] < .5
    assert state["state_after"]["target"].endswith("contents_unknown")
