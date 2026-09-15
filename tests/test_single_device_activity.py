import pytest

from visioncortex.device_day_models import DeviceDayModels
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ExperimentSegment,
    ViewInput,
    ViewRole,
    VideoInfo,
)


@pytest.mark.parametrize("role", [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON])
def test_single_device_nonempty_activity_never_requests_dual_view_group(
    default_config, tmp_path, monkeypatch, role
):
    view = ViewInput(view_id="camera", role=role, video=tmp_path / "Video.mp4")
    info = VideoInfo(
        path=view.video,
        duration_ms=10000,
        fps=30,
        frame_count=300,
        width=1280,
        height=800,
        size_bytes=0,
    )
    fine = tmp_path / "Fine.jsonl"
    fine.write_text("")
    candidate = ActionCandidate(
        candidate_id="candidate",
        action_type=ActionType.DEVICE_PANEL_OPERATION,
        view_id=view.view_id,
        role=role,
        local_start_ms=1000,
        local_end_ms=9000,
        global_start_ms=1000,
        global_end_ms=9000,
        key_global_ms=4000,
        objects=["balance", "gloved_hand"],
        confidence=0.95,
    )
    event = EvidenceEvent(
        event_id="event",
        action_type=candidate.action_type,
        global_start_ms=1000,
        global_end_ms=9000,
        key_global_ms=4000,
        objects=candidate.objects,
        confidence=0.95,
        accepted=True,
        formal_admission_status="formal",
        audit_reason="fixture",
        supporting_views=[view.view_id],
        supporting_roles=[role],
        candidates=[candidate],
    )
    segment = ExperimentSegment(
        segment_id="segment",
        global_start_ms=1000,
        global_end_ms=9000,
        event_ids=["event"],
        participating_views=[view.view_id],
    )
    monkeypatch.setattr(
        "visioncortex.coarse_recall.generate_open_vocabulary_fine_candidates",
        lambda *args: ([], {"formal_evidence_ready": True}),
    )
    monkeypatch.setattr(
        "visioncortex.movement_verification.verify_movement_candidates",
        lambda *args: {},
    )
    monkeypatch.setattr(
        "visioncortex.actions.audit_candidates", lambda *args: ([event], [])
    )
    monkeypatch.setattr(
        "visioncortex.actions.build_experiment_segments",
        lambda *args, **kwargs: [segment],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Device preprocessing must not require an FP/TP pair")

    monkeypatch.setattr("visioncortex.grouping.build_experiment_groups", forbidden)
    intervals, audit = DeviceDayModels(default_config)._audit_activity(
        view, info, {"camera": fine}, [candidate], [], [(0, 10000)]
    )
    assert intervals and audit["device_activity_segments"]
    assert audit["selected_key_events"]
    assert audit["formal_experiment_segments"] == []
    assert audit["experiment_grouping"]["status"] == "not_applicable"
    assert not audit["physical_action_confirmed"]
    assert not audit["cross_camera_alignment_verified"]
