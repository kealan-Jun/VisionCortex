from types import SimpleNamespace

import numpy as np
import pytest

from visioncortex import archive
from visioncortex.schemas import ActionType, AlignmentTransform, EvidenceEvent


@pytest.mark.parametrize("admit_second", [True, False])
def test_bounded_cap_review_keeps_action_and_selects_only_visible_candidate(tmp_path, monkeypatch, admit_second):
    event = EvidenceEvent(event_id="cap-event", action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=0, global_end_ms=1000, key_global_ms=500, confidence=0.9,
        objects=["gloved_hand", "bottle_cap"], accepted=True, audit_reason="fixture",
        candidates=[], supporting_views=["fp","tp"], supporting_roles=[],
        model_understanding={"status":"completed"})
    config = {"mllm":{"enabled":True}, "key_materials":{"participant_visual_review":{
        "enabled":True, "classes":["paper","bottle_cap"],
        "temporal_keyframe_review":{"enabled":True,"max_frames_per_event":4}}}}
    calls = []

    class Reader:
        def __init__(self, **kwargs): pass
        def read(self, view, info, timestamp):
            assert 0 <= timestamp <= 1000
            return np.zeros((48,64,3), np.uint8)
        def close(self): pass

    class Reviewer:
        def __init__(self, *args): pass
        def review(self, item, views, *, participant_class):
            assert participant_class == "bottle_cap" and len(views) == 2
            calls.append(views)
            visible = admit_second and len(calls) == 2
            return {"status":"completed", "input_fingerprint":str(len(calls))}, {
                view["view_id"]:{"boxes":[{"class_name":"bottle_cap"}] if visible else []} for view in views}

    monkeypatch.setattr(archive, "ViewFrameReader", Reader)
    monkeypatch.setattr(archive, "ParticipantVisualReviewer", Reviewer)
    monkeypatch.setattr(archive, "nearest_frame_evidence_many", lambda path, times: {})
    pair = ("fp","tp")
    selected, receipt = archive._review_bounded_cap_keyframe(
        archive.ArchiveLayout(tmp_path), event, pair,
        [SimpleNamespace(view_id=view) for view in pair],
        {view:SimpleNamespace(duration_ms=1000) for view in pair},
        {view:AlignmentTransform(view_id=view, reference_view_id="fp") for view in pair},
        {view:tmp_path/f"{view}.jsonl" for view in pair}, config)
    assert event.accepted and event.key_global_ms == 500
    assert receipt["original_key_global_ms"] == 500
    if admit_second:
        assert selected == 750 and len(calls) == 2
        assert receipt["status"] == "selected"
    else:
        assert selected is None and len(calls) == 4
        assert receipt["status"] == "no_verified_candidate"


def test_presemantic_materialization_does_not_trigger_cap_requests():
    event = SimpleNamespace(objects=["bottle_cap"], model_understanding={})
    config = {"mllm":{"enabled":True}, "key_materials":{"participant_visual_review":{
        "enabled":True, "classes":["bottle_cap"], "temporal_keyframe_review":{"enabled":True}}}}
    result, receipt = archive._review_bounded_cap_keyframe(None, event, (), [], {}, {}, {}, config)
    assert result is None and receipt["status"] == "not_applicable"
