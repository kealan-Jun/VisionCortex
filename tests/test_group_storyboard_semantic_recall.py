from labvision_evidence.pipeline import (
    _recover_group_storyboard_state_events,
)
from labvision_evidence.action_state_machine import (
    attach_continuous_action_states,
)
from labvision_evidence.schemas import (
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    ViewRole,
)


def _event() -> EvidenceEvent:
    return EvidenceEvent(
        event_id="EVT-000005",
        action_type=ActionType.OBJECT_MOVEMENT,
        global_start_ms=8000.0,
        global_end_ms=9600.0,
        key_global_ms=8800.0,
        objects=["hand", "tube"],
        confidence=0.8,
        accepted=True,
        audit_reason="test movement candidate",
        supporting_views=["fp", "tp"],
        supporting_roles=[
            ViewRole.FIRST_PERSON,
            ViewRole.THIRD_PERSON,
        ],
        candidates=[],
    )


def _group(step: dict) -> ExperimentGroup:
    return ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-0001"],
        global_start_ms=6000.0,
        global_end_ms=24900.0,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="one bounded experiment",
        model_understanding={
            "status": "completed",
            "confidence": 0.58,
            "steps": [step],
        },
    )


def _segment() -> ExperimentSegment:
    return ExperimentSegment(
        segment_id="EXP-0001",
        group_id="GROUP-0001",
        global_start_ms=6000.0,
        global_end_ms=24900.0,
        event_ids=["EVT-000005"],
        participating_views=["fp", "tp"],
    )


def _uncapping_step() -> dict:
    return {
        "step_index": 2,
        "start_global_ms": 8000.0,
        "end_global_ms": 21900.0,
        "current_step": (
            "戴蓝色手套的实验者双手扶住带蓝色瓶盖的透明"
            "试剂瓶，旋开瓶盖，将瓶盖放置于台面上"
        ),
        "physical_change": (
            "试剂瓶从密封闭合状态变为开口状态，蓝色瓶盖与"
            "瓶身分离"
        ),
        "supporting_views": ["fp", "tp"],
        "confidence": 0.58,
    }


def test_dual_view_explicit_uncapping_storyboard_adds_provisional_state_event():
    events = [_event()]
    segment = _segment()

    receipts = _recover_group_storyboard_state_events(
        [_group(_uncapping_step())],
        [segment],
        events,
        {},
    )

    assert len(receipts) == 1
    recovered = events[-1]
    assert recovered.event_id == "EVT-000006"
    assert recovered.action_type == ActionType.CONTAINER_STATE_CHANGE
    assert recovered.global_start_ms == 8000.0
    assert recovered.global_end_ms == 21900.0
    assert recovered.key_global_ms == 19120.0
    assert recovered.objects == [
        "gloved_hand",
        "reagent_bottle",
        "bottle_cap",
    ]
    assert recovered.supporting_views == ["fp", "tp"]
    assert recovered.observability["semantic_recall_admission"][
        "mandatory_semantic_review"
    ] is True
    assert recovered.state_machine["publication"]["status"] == (
        "provisional_semantic_review"
    )
    ledger = attach_continuous_action_states(events, {})
    assert recovered.state_machine["publication"]["status"] == (
        "provisional_semantic_review"
    )
    assert ledger["publication_counts"] == {
        "primary": 1,
        "provisional_semantic_review": 1,
    }
    assert segment.event_ids == ["EVT-000005", "EVT-000006"]
    assert segment.micro_segments[-1]["source"] == (
        "full_group_storyboard_semantic_recall"
    )


def test_storyboard_without_explicit_state_transition_does_not_recover_event():
    step = _uncapping_step()
    step["current_step"] = "实验者在台面上移动试管"
    step["physical_change"] = "试管在实验台面上发生位置移动"
    events = [_event()]

    receipts = _recover_group_storyboard_state_events(
        [_group(step)], [_segment()], events, {}
    )

    assert receipts == []
    assert len(events) == 1


def test_single_view_state_storyboard_does_not_recover_event():
    step = _uncapping_step()
    step["supporting_views"] = ["fp"]
    events = [_event()]

    receipts = _recover_group_storyboard_state_events(
        [_group(step)], [_segment()], events, {}
    )

    assert receipts == []
    assert len(events) == 1
