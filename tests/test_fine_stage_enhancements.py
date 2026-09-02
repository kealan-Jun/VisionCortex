from copy import deepcopy
from pathlib import Path

import numpy as np

from visioncortex.actions import audit_candidates, generate_candidates
from visioncortex.candidate_index import (
    create_fine_frame_index,
    fine_frame_coverage_report,
    ingest_fine_frame_ledgers,
)
from visioncortex.coarse_recall import (
    generate_open_vocabulary_fine_candidates,
)
from visioncortex.config import load_config
from visioncortex.pipeline import EvidencePipeline
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentSegmentTransform,
    AlignmentTransform,
    BoxEvidence,
    FrameEvidence,
    VideoInfo,
    ViewInput,
    ViewRole,
)


def _view(view_id: str, role: ViewRole = ViewRole.FIRST_PERSON) -> ViewInput:
    return ViewInput(
        view_id=view_id,
        role=role,
        video=Path(f"{view_id}.mp4"),
    )


def _info(view_id: str, duration_ms: float = 1000.0) -> VideoInfo:
    return VideoInfo(
        path=Path(f"{view_id}.mp4"),
        duration_ms=duration_ms,
        fps=30.0,
        width=640,
        height=360,
        frame_count=max(1, int(duration_ms / 1000.0 * 30.0)),
    )


def _box(
    class_name: str,
    track_id: int,
    xyxy_norm: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
    appearance: tuple[float, ...] = (),
) -> BoxEvidence:
    return BoxEvidence(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        xyxy_norm=xyxy_norm,
        track_id=track_id,
        appearance_signature=appearance,
    )


def _frame(
    view: ViewInput,
    frame_index: int,
    local_ms: float,
    detections: list[BoxEvidence],
) -> FrameEvidence:
    return FrameEvidence(
        view_id=view.view_id,
        role=view.role,
        frame_index=frame_index,
        local_ms=local_ms,
        global_ms=local_ms,
        width=640,
        height=360,
        detections=detections,
    )


def _write_ledger(path: Path, frames: list[FrameEvidence]) -> None:
    path.write_text(
        "\n".join(frame.model_dump_json() for frame in frames) + "\n",
        encoding="utf-8",
    )


def test_3090ti_enables_in_place_fine_stage_enhancements():
    config = load_config(Path("configs/rtx3090ti-ubuntu-production.yaml"))
    performance = config["performance"]

    assert performance["fine_frame_index_enabled"] is True
    assert performance["fine_coverage_gate_enabled"] is True
    assert performance["fine_streaming_candidates_enabled"] is True
    assert performance["fine_instance_association_enabled"] is True
    assert performance["fine_track_stitching_enabled"] is True
    assert performance["fine_contact_state_enabled"] is True
    assert performance["fine_affine_object_motion_enabled"] is True
    assert performance["fine_roi_open_vocabulary_recall_enabled"] is True


def test_fine_index_incrementally_stitches_tracks_and_proves_coverage(tmp_path):
    view = _view("fp")
    base = tmp_path / "base.jsonl"
    supplement = tmp_path / "supplement.jsonl"
    base_frames = [
        _frame(
            view,
            index,
            local_ms,
            [_box("tube", 1, (0.20, 0.20, 0.30, 0.40))],
        )
        for index, local_ms in enumerate((0.0, 50.0, 100.0))
    ]
    supplement_frames = [
        _frame(
            view,
            index + 2,
            local_ms,
            [_box("tube", 1, (0.205, 0.20, 0.305, 0.40))],
        )
        for index, local_ms in enumerate((100.0, 150.0, 200.0))
    ]
    _write_ledger(base, base_frames)
    _write_ledger(supplement, supplement_frames)

    index = create_fine_frame_index(tmp_path / "fine.sqlite3")
    ingest_fine_frame_ledgers(
        index,
        [view],
        {view.view_id: base},
        source_pass="primary",
    )
    receipt = ingest_fine_frame_ledgers(
        index,
        [view],
        {view.view_id: supplement},
        source_pass="supplemental",
    )
    report = fine_frame_coverage_report(
        index,
        [view],
        {view.view_id: _info(view.view_id, 250.0)},
        {view.view_id: [(0.0, 250.0)]},
        sample_fps=20.0,
    )

    frames = list(index.iter_frames(view.view_id))
    assert [frame.local_ms for frame in frames] == [
        0.0,
        50.0,
        100.0,
        150.0,
        200.0,
    ]
    assert {box.track_id for frame in frames for box in frame.detections} == {1}
    assert receipt["stitched_tracks"] == 1
    assert report["formal_evidence_ready"] is True
    assert report["views"][0]["coverage_ratio"] == 1.0


def test_fine_index_rejects_internal_sampling_gap(tmp_path):
    view = _view("fp")
    ledger = tmp_path / "gap.jsonl"
    _write_ledger(
        ledger,
        [
            _frame(view, 0, 0.0, []),
            _frame(view, 1, 50.0, []),
            _frame(view, 10, 500.0, []),
        ],
    )
    index = create_fine_frame_index(tmp_path / "fine.sqlite3")
    ingest_fine_frame_ledgers(
        index,
        [view],
        {view.view_id: ledger},
        source_pass="primary",
    )

    report = fine_frame_coverage_report(
        index,
        [view],
        {view.view_id: _info(view.view_id, 550.0)},
        {view.view_id: [(0.0, 550.0)]},
        sample_fps=20.0,
        maximum_gap_periods=4.0,
    )

    assert report["formal_evidence_ready"] is False
    assert report["views"][0]["unexpected_gap_count"] >= 1


def test_streaming_candidates_keep_two_same_class_instances_separate(
    tmp_path, default_config
):
    view = _view("fp")
    ledger = tmp_path / "instances.jsonl"
    frames = []
    for index in range(11):
        frames.append(
            _frame(
                view,
                index,
                float(index * 50),
                [
                    _box("hand", 101, (0.05, 0.10, 0.15, 0.25)),
                    _box("tube", 1, (0.14, 0.10, 0.24, 0.30)),
                    _box("hand", 102, (0.65, 0.10, 0.75, 0.25)),
                    _box("tube", 2, (0.74, 0.10, 0.84, 0.30)),
                ],
            )
        )
    _write_ledger(ledger, frames)
    default_config["performance"].update(
        {
            "fine_streaming_candidates_enabled": True,
            "fine_instance_association_enabled": True,
            "fine_contact_state_enabled": True,
        }
    )

    candidates = generate_candidates(
        [view], {view.view_id: ledger}, default_config
    )
    contacts = [
        item
        for item in candidates
        if item.action_type == ActionType.HAND_OBJECT_CONTACT
    ]

    assert len(contacts) == 2
    assert {
        tuple(item.instance_signature["track_ids"]) for item in contacts
    } == {(1,), (2,)}
    assert all(
        item.provenance["candidate_reducer"] == "streaming"
        for item in contacts
    )


def test_streaming_contact_records_approach_contact_release_state(
    tmp_path, default_config
):
    view = _view("fp")
    ledger = tmp_path / "contact-state.jsonl"
    frames = [
        _frame(
            view,
            0,
            0.0,
            [
                _box("hand", 10, (0.00, 0.10, 0.10, 0.25)),
                _box("beaker", 20, (0.30, 0.10, 0.45, 0.35)),
            ],
        ),
        _frame(
            view,
            1,
            50.0,
            [
                _box("hand", 10, (0.15, 0.10, 0.25, 0.25)),
                _box("beaker", 20, (0.30, 0.10, 0.45, 0.35)),
            ],
        ),
    ]
    for index in range(2, 13):
        frames.append(
            _frame(
                view,
                index,
                float(index * 50),
                [
                    _box("hand", 10, (0.24, 0.10, 0.34, 0.25)),
                    _box("beaker", 20, (0.30, 0.10, 0.45, 0.35)),
                ],
            )
        )
    frames.append(
        _frame(
            view,
            13,
            650.0,
            [
                _box("hand", 10, (0.00, 0.10, 0.10, 0.25)),
                _box("beaker", 20, (0.30, 0.10, 0.45, 0.35)),
            ],
        )
    )
    _write_ledger(ledger, frames)
    default_config["performance"].update(
        {
            "fine_streaming_candidates_enabled": True,
            "fine_instance_association_enabled": True,
            "fine_contact_state_enabled": True,
        }
    )

    candidate = next(
        item
        for item in generate_candidates(
            [view], {view.view_id: ledger}, default_config
        )
        if item.action_type == ActionType.HAND_OBJECT_CONTACT
    )

    state = candidate.provenance["interaction_state"]
    assert state["approach_observed"] is True
    assert state["contact_observation_count"] == 11
    assert state["release_observed"] is True
    assert state["release_observed_at_global_ms"] == 650.0
    assert state["release_inferred_from_next_observation_gap"] is False


def test_streaming_mode_preserves_legacy_single_instance_candidate(
    tmp_path, default_config
):
    view = _view("fp")
    ledger = tmp_path / "compatibility.jsonl"
    frames = [
        _frame(
            view,
            index,
            float(index * 50),
            [
                _box("hand", 10, (0.24, 0.10, 0.34, 0.25)),
                _box("tube", 20, (0.30, 0.10, 0.45, 0.35)),
            ],
        )
        for index in range(13)
    ]
    _write_ledger(ledger, frames)
    legacy_config = deepcopy(default_config)
    streaming_config = deepcopy(default_config)
    streaming_config["performance"].update(
        {
            "fine_streaming_candidates_enabled": True,
            "fine_instance_association_enabled": True,
            "fine_contact_state_enabled": True,
        }
    )

    legacy = generate_candidates(
        [view], {view.view_id: ledger}, legacy_config
    )
    streaming = generate_candidates(
        [view], {view.view_id: ledger}, streaming_config
    )

    legacy_contacts = [
        item
        for item in legacy
        if item.action_type == ActionType.HAND_OBJECT_CONTACT
    ]
    streaming_contacts = [
        item
        for item in streaming
        if item.action_type == ActionType.HAND_OBJECT_CONTACT
    ]
    assert len(legacy_contacts) == len(streaming_contacts) == 1
    assert (
        legacy_contacts[0].global_start_ms,
        legacy_contacts[0].global_end_ms,
        legacy_contacts[0].objects,
    ) == (
        streaming_contacts[0].global_start_ms,
        streaming_contacts[0].global_end_ms,
        streaming_contacts[0].objects,
    )


def test_cross_view_appearance_conflict_does_not_merge_instances(default_config):
    default_config["performance"].update(
        {
            "fine_instance_cross_view_conflict_quarantine_enabled": True,
            "fine_instance_minimum_cosine_similarity": 0.25,
        }
    )
    candidates = [
        ActionCandidate(
            candidate_id="FP",
            action_type=ActionType.HAND_OBJECT_CONTACT,
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            local_start_ms=0.0,
            local_end_ms=1000.0,
            global_start_ms=0.0,
            global_end_ms=1000.0,
            key_global_ms=500.0,
            objects=["hand", "tube"],
            confidence=0.9,
            instance_signature={
                "track_ids": [1],
                "appearance_signature": [1.0, 0.0],
            },
        ),
        ActionCandidate(
            candidate_id="TP",
            action_type=ActionType.HAND_OBJECT_CONTACT,
            view_id="tp",
            role=ViewRole.THIRD_PERSON,
            local_start_ms=0.0,
            local_end_ms=1000.0,
            global_start_ms=0.0,
            global_end_ms=1000.0,
            key_global_ms=500.0,
            objects=["hand", "tube"],
            confidence=0.9,
            instance_signature={
                "track_ids": [2],
                "appearance_signature": [0.0, 1.0],
            },
        ),
    ]
    transforms = {
        view_id: AlignmentTransform(
            view_id=view_id,
            reference_view_id="fp",
            confidence=1.0,
            state="aligned",
        )
        for view_id in ("fp", "tp")
    }

    events, _ = audit_candidates(candidates, transforms, default_config)

    assert len(events) == 2
    assert all(len(event.supporting_views) == 1 for event in events)


def test_fine_roi_open_vocabulary_only_adds_bounded_recall(
    monkeypatch, tmp_path, default_config
):
    view = _view("fp")
    ledger = tmp_path / "fine.jsonl"
    _write_ledger(
        ledger,
        [
            _frame(
                view,
                10,
                500.0,
                [_box("hand", 1, (0.40, 0.30, 0.50, 0.60))],
            )
        ],
    )
    default_config["performance"].update(
        {
            "fine_roi_open_vocabulary_recall_enabled": True,
            "fine_roi_open_vocabulary_max_frames_per_window": 1,
        }
    )
    default_config["models"]["open_vocabulary_key_frame"] = {
        "enabled": True,
        "prompt_map": {"tube": "tube"},
    }
    monkeypatch.setattr(
        "visioncortex.coarse_recall.read_view_frame_at",
        lambda *_args, **_kwargs: np.zeros((360, 640, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "visioncortex.coarse_recall._yolo_world_detections",
        lambda *_args, **_kwargs: (
            [
                {
                    "class_name": "tube",
                    "confidence": 0.8,
                    "xyxy_norm": [0.40, 0.20, 0.65, 0.80],
                }
            ],
            {"status": "executed"},
        ),
    )

    candidates, report = generate_open_vocabulary_fine_candidates(
        [view],
        {view.view_id: _info(view.view_id)},
        {view.view_id: ledger},
        {view.view_id: [(0.0, 1000.0)]},
        default_config,
    )

    assert len(candidates) == 1
    assert candidates[0].candidate_id.startswith("FINE-OPEN-VOCAB-fp-")
    assert candidates[0].provenance["recall_only"] is True
    assert report["formal_evidence_ready"] is True


def test_fine_roi_open_vocabulary_fails_closed_on_unreadable_frame(
    monkeypatch, tmp_path, default_config
):
    view = _view("fp")
    ledger = tmp_path / "fine-unreadable.jsonl"
    _write_ledger(
        ledger,
        [
            _frame(
                view,
                10,
                500.0,
                [_box("hand", 1, (0.40, 0.30, 0.50, 0.60))],
            )
        ],
    )
    default_config["performance"].update(
        {
            "fine_roi_open_vocabulary_recall_enabled": True,
            "fine_roi_open_vocabulary_max_frames_per_window": 1,
        }
    )
    default_config["models"]["open_vocabulary_key_frame"] = {
        "enabled": True,
        "prompt_map": {"tube": "tube"},
    }
    monkeypatch.setattr(
        "visioncortex.coarse_recall.read_view_frame_at",
        lambda *_args, **_kwargs: None,
    )

    candidates, report = generate_open_vocabulary_fine_candidates(
        [view],
        {view.view_id: _info(view.view_id)},
        {view.view_id: ledger},
        {view.view_id: [(0.0, 1000.0)]},
        default_config,
    )

    assert candidates == []
    assert report["error_count"] == 1
    assert report["formal_evidence_ready"] is False


def test_fine_windows_intersect_only_failed_alignment_intervals(default_config):
    views = [
        _view("fp", ViewRole.FIRST_PERSON),
        _view("tp", ViewRole.THIRD_PERSON),
    ]
    infos = {view.view_id: _info(view.view_id, 10_000.0) for view in views}
    transforms = {
        "fp": AlignmentTransform(
            view_id="fp",
            reference_view_id="fp",
            confidence=1.0,
            state="aligned",
            local_coverage_start_ms=0.0,
            local_coverage_end_ms=10_000.0,
        ),
        "tp": AlignmentTransform(
            view_id="tp",
            reference_view_id="fp",
            confidence=0.9,
            state="aligned",
            segment_transforms=[
                AlignmentSegmentTransform(
                    segment_index=0,
                    local_start_ms=0.0,
                    local_end_ms=5000.0,
                    state="failed",
                ),
                AlignmentSegmentTransform(
                    segment_index=1,
                    local_start_ms=5000.0,
                    local_end_ms=10_000.0,
                    state="aligned",
                ),
            ],
        ),
    }

    windows, report = EvidencePipeline(
        default_config
    )._intersect_fine_windows_with_usable_alignment(
        {"fp": [(0.0, 10_000.0)], "tp": [(0.0, 10_000.0)]},
        infos,
        transforms,
        views,
    )

    assert windows["fp"] == [(0.0, 10_000.0)]
    assert windows["tp"] == [(5000.0, 10_000.0)]
    assert report["formal_evidence_ready"] is True
    assert report["views"]["tp"]["excluded_unavailable_seconds"] == 5.0
