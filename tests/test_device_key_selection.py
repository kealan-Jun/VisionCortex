from copy import deepcopy

from visioncortex.actions import build_experiment_segments
from visioncortex.device_day_contract import PHYSICAL_ACTION_TYPES
from visioncortex.grouping import select_device_key_events
from visioncortex.schemas import ActionCandidate, ActionType, EvidenceEvent, ViewInput, ViewRole


def event(identifier="panel", confidence=0.8):
    candidate = ActionCandidate(
        candidate_id=identifier, view_id="camera", role=ViewRole.THIRD_PERSON,
        action_type=ActionType.DEVICE_PANEL_OPERATION,
        local_start_ms=35950, local_end_ms=36250,
        global_start_ms=35950, global_end_ms=36250, key_global_ms=36200,
        objects=["balance", "gloved_hand"], confidence=confidence, evidence=[])
    return EvidenceEvent(
        event_id=identifier, action_type=candidate.action_type,
        global_start_ms=35950, global_end_ms=36250, key_global_ms=36200,
        objects=candidate.objects, confidence=confidence, accepted=True,
        formal_admission_status="formal", audit_reason="synthetic regression fixture",
        supporting_views=["camera"], supporting_roles=[ViewRole.THIRD_PERSON],
        candidates=[candidate])


def select(events, config, intervals=None, receipts=None):
    return select_device_key_events(
        events, [(33000, 89000)] if intervals is None else intervals,
        "camera", set(PHYSICAL_ACTION_TYPES), config, receipts)


def test_short_formally_audited_event_survives_empty_experiment_segmentation(default_config):
    item = event()
    view = ViewInput(view_id="camera", role=ViewRole.THIRD_PERSON, video="unused.mp4")
    assert build_experiment_segments([item], [view], default_config) == []
    before = item.model_dump()
    receipts = []
    assert select([item], default_config, receipts=receipts) == [item]
    assert item.model_dump() == before
    assert receipts[0]["selection_scope"] == "single_device_action_material"
    assert receipts[0]["experiment_admission_changed"] is False
    assert receipts[0]["physical_action_confirmed"] is False


def test_no_promotion_of_provisional_rejected_components_or_other_camera(default_config):
    items = []
    for status in ("provisional", "rejected"):
        item = event(status)
        item.formal_admission_status = status
        item.accepted = status != "rejected"
        items.append(item)
    component = event("component")
    component.state_machine = {"publication": {"status": "component_only"}}
    foreign = event("foreign")
    foreign.supporting_views = ["other_camera"]
    before = deepcopy([item.model_dump() for item in [*items, component, foreign]])
    assert select([*items, component, foreign], default_config) == []
    assert [item.model_dump() for item in [*items, component, foreign]] == before
    assert select([event()], default_config, intervals=[]) == []
    assert select([event()], default_config, intervals=[(0, 36200)]) == []


def test_shared_dedup_keeps_stronger_event_once_across_overlapping_windows(default_config):
    weak, strong = event("weak", 0.7), event("strong", 0.9)
    receipts = []
    assert select([weak, strong], default_config,
                  intervals=[(33000, 89000), (35000, 90000)], receipts=receipts) == [strong]
    assert len(receipts) == 2


def test_device_material_uses_shared_best_frame_without_mutating_event(default_config, tmp_path, monkeypatch):
    from visioncortex import archive
    from visioncortex.device_day_models import DeviceDayModels
    from visioncortex.schemas import FrameEvidence
    item = event()
    audit = {'selected_key_events': [item.model_dump(mode='json')]}
    before = deepcopy(audit)
    view = ViewInput(view_id='camera', role=ViewRole.THIRD_PERSON, video=tmp_path/'Video.mp4')
    frame = FrameEvidence(view_id='camera', role=view.role, frame_index=1086,
                          local_ms=36200, global_ms=36200, width=10, height=10)
    calls = []
    def shared(path, events):
        calls.append((path, events[0].event_id))
        return {item.event_id: (frame, 201, {'participant_groups': ['actor', 'device']})}
    monkeypatch.setattr(archive, '_best_event_frames_many', shared)
    result = DeviceDayModels(default_config)._key_frame_choices(view, {'camera':tmp_path/'Fine.jsonl'}, audit)
    assert result['panel']['frame_index'] == 1086
    assert result['panel']['selector'] == 'visioncortex.archive._best_event_frames_many'
    assert len(calls) == 1 and audit == before
    assert DeviceDayModels(default_config)._key_frame_choices(view, {}, {}) == {}
    assert len(calls) == 1


def test_incomplete_container_cue_is_retained_but_not_a_keyframe(default_config):
    item = event("cap")
    item.action_type = ActionType.CONTAINER_STATE_CHANGE
    item.state_machine = {"missing_phases": ["state_after", "state_hold"]}
    before = item.model_dump()
    receipts = []
    assert select([item], default_config, receipts=receipts) == []
    assert receipts[0]["decision"] == "container_state_transition_not_observed"
    assert item.model_dump() == before
    item.state_machine["missing_phases"] = []
    assert select([item], default_config) == [item]


def test_offline_selector_respects_same_container_state_gate(default_config):
    from visioncortex.grouping import _KeySelectionScope, _KeySelectionWindow, select_key_events
    item = event("cap")
    item.action_type = ActionType.CONTAINER_STATE_CHANGE
    item.state_machine = {"missing_phases": ["state_after"]}
    window = _KeySelectionWindow("window", 33000, 89000, [item.event_id])
    group = _KeySelectionScope("group", [window.segment_id], 33000, 89000)
    receipts = []
    assert select_key_events([group], [window], [item], default_config, receipts) == []
    assert receipts[0]["decision"] == "container_state_transition_not_observed"
