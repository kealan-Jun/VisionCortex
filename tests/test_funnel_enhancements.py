from pathlib import Path

import cv2
import numpy as np

from visioncortex.coarse_recall import (
    generate_open_vocabulary_coarse_candidates,
    select_suspicious_coarse_frames,
)
from visioncortex.actions import (
    generate_coarse_activity_candidates,
    generate_motion_burst_candidates,
    refine_motion_candidates_with_coarse,
)
from visioncortex.candidate_index import build_coarse_frame_index
from visioncortex.config import load_config
from visioncortex.detection import _camera_compensated_motion_score
from visioncortex.pipeline import EvidencePipeline
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    BoxEvidence,
    FrameEvidence,
    RunManifest,
    VideoInfo,
    ViewInput,
    ViewRole,
)


def _view(view_id: str, role: ViewRole) -> ViewInput:
    return ViewInput(view_id=view_id, role=role, video=Path(f"{view_id}.mp4"))


def _candidate(action_type: ActionType = ActionType.OBJECT_MOVEMENT) -> ActionCandidate:
    return ActionCandidate(
        candidate_id="legacy-candidate",
        action_type=action_type,
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        local_start_ms=100_000.0,
        local_end_ms=200_000.0,
        global_start_ms=100_000.0,
        global_end_ms=200_000.0,
        key_global_ms=150_000.0,
        objects=["beaker"],
        confidence=0.9,
    )


def test_camera_motion_compensation_reduces_global_translation_without_forcing_it():
    rng = np.random.default_rng(42)
    previous = rng.integers(0, 256, size=(72, 128), dtype=np.uint8)
    translated = cv2.warpAffine(
        previous,
        np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float32),
        (128, 72),
        borderMode=cv2.BORDER_REFLECT,
    )

    raw, compensated, applied, shift_ratio = _camera_compensated_motion_score(
        previous,
        translated,
        enabled=True,
        minimum_response=0.05,
        maximum_shift_ratio=0.20,
    )
    legacy_raw, legacy_score, legacy_applied, _ = _camera_compensated_motion_score(
        previous,
        translated,
        enabled=False,
        minimum_response=0.05,
        maximum_shift_ratio=0.20,
    )

    assert applied is True
    assert 0.0 < shift_ratio < 0.20
    assert compensated < raw * 0.35
    assert legacy_applied is False
    assert legacy_score == legacy_raw == raw


def test_enhanced_view_selection_only_changes_when_flags_are_enabled(default_config):
    views = [
        _view("fp", ViewRole.FIRST_PERSON),
        _view("tp1", ViewRole.THIRD_PERSON),
        _view("tp2", ViewRole.THIRD_PERSON),
    ]
    manifest = RunManifest(experiment_id="enhanced-selection", views=views)
    pipeline = EvidencePipeline(default_config)

    assert [item.view_id for item in pipeline._motion_probe_views(manifest)] == [
        "fp",
        "tp1",
    ]
    assert [item.view_id for item in pipeline._coarse_scan_views(manifest)] == [
        "fp",
        "tp1",
    ]

    default_config["performance"]["motion_probe_all_views"] = True
    default_config["performance"]["coarse_all_views"] = True

    assert [item.view_id for item in pipeline._motion_probe_views(manifest)] == [
        "fp",
        "tp1",
        "tp2",
    ]
    assert [item.view_id for item in pipeline._coarse_scan_views(manifest)] == [
        "fp",
        "tp1",
        "tp2",
    ]


def test_fine_risk_and_alignment_rules_only_expand_legacy_windows(default_config):
    infos = {
        "fp": VideoInfo(
            path=Path("fp.mp4"),
            duration_ms=400_000.0,
            fps=30.0,
            width=640,
            height=360,
            frame_count=12_000,
        ),
        "tp": VideoInfo(
            path=Path("tp.mp4"),
            duration_ms=400_000.0,
            fps=30.0,
            width=640,
            height=360,
            frame_count=12_000,
        ),
    }
    transforms = {
        "fp": AlignmentTransform(
            view_id="fp",
            reference_view_id="fp",
            confidence=1.0,
            state="aligned",
        ),
        "tp": AlignmentTransform(
            view_id="tp",
            reference_view_id="fp",
            confidence=0.5,
            state="uncertain",
        ),
    }
    pipeline = EvidencePipeline(default_config)
    candidate = _candidate(ActionType.LIQUID_MOVEMENT)

    baseline = pipeline._fine_windows([candidate], infos, transforms)
    default_config["performance"].update(
        {
            "fine_risk_window_expansion_enabled": True,
            "fine_risk_extra_padding_seconds": 30.0,
            "fine_risk_action_types": ["liquid_movement"],
            "fine_low_alignment_extra_padding_enabled": True,
            "fine_low_alignment_confidence_threshold": 0.8,
            "fine_low_alignment_extra_padding_seconds": 20.0,
        }
    )
    enhanced = pipeline._fine_windows([candidate], infos, transforms)

    assert baseline["fp"] == [(25_000.0, 275_000.0)]
    assert enhanced["fp"] == [(0.0, 305_000.0)]
    assert baseline["tp"] == [(25_000.0, 275_000.0)]
    assert enhanced["tp"] == [(0.0, 325_000.0)]
    for view_id in baseline:
        assert enhanced[view_id][0][0] <= baseline[view_id][0][0]
        assert enhanced[view_id][0][1] >= baseline[view_id][0][1]


def test_suspicious_open_vocabulary_selection_is_bounded_per_view_hour(
    tmp_path, default_config
):
    view = _view("fp", ViewRole.FIRST_PERSON)
    path = tmp_path / "fp.detections.jsonl"
    frames = [
        FrameEvidence(
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            frame_index=index,
            local_ms=float(index * 1_000),
            global_ms=float(index * 1_000),
            width=640,
            height=360,
            motion_score=float(index),
            detections=(
                [
                    BoxEvidence(
                        class_id=0,
                        class_name="hand",
                        confidence=0.9,
                        xyxy_norm=(0.1, 0.1, 0.2, 0.2),
                    ),
                    BoxEvidence(
                        class_id=1,
                        class_name="beaker",
                        confidence=0.9,
                        xyxy_norm=(0.2, 0.2, 0.3, 0.3),
                    ),
                ]
                if index == 9
                else []
            ),
        )
        for index in range(10)
    ]
    path.write_text(
        "\n".join(frame.model_dump_json() for frame in frames) + "\n",
        encoding="utf-8",
    )
    default_config["performance"].update(
        {
            "coarse_open_vocabulary_motion_percentile": 50.0,
            "coarse_open_vocabulary_frames_per_hour_per_view": 2,
        }
    )

    selected = select_suspicious_coarse_frames(
        [view], {"fp": path}, default_config
    )

    assert [frame.frame_index for frame in selected] == [7, 8]


def test_open_vocabulary_coarse_candidate_only_adds_a_recall_window(
    monkeypatch, tmp_path, default_config
):
    view = _view("fp", ViewRole.FIRST_PERSON)
    info = VideoInfo(
        path=Path("fp.mp4"),
        duration_ms=10_000.0,
        fps=30.0,
        width=640,
        height=360,
        frame_count=300,
    )
    path = tmp_path / "fp.detections.jsonl"
    evidence = FrameEvidence(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        frame_index=1,
        local_ms=5_000.0,
        global_ms=5_000.0,
        width=640,
        height=360,
        motion_score=20.0,
    )
    path.write_text(evidence.model_dump_json() + "\n", encoding="utf-8")
    default_config["performance"].update(
        {
            "coarse_open_vocabulary_recall_enabled": True,
            "coarse_open_vocabulary_motion_percentile": 0.0,
            "coarse_open_vocabulary_frames_per_hour_per_view": 1,
        }
    )
    default_config["models"]["open_vocabulary_key_frame"] = {
        "enabled": True,
        "prompt_map": {"gloved hand": "gloved_hand", "beaker": "beaker"},
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
                    "class_name": "gloved_hand",
                    "confidence": 0.8,
                    "xyxy_norm": [0.1, 0.1, 0.2, 0.2],
                    "detector_source": "fake",
                },
                {
                    "class_name": "beaker",
                    "confidence": 0.7,
                    "xyxy_norm": [0.19, 0.1, 0.3, 0.3],
                    "detector_source": "fake",
                },
            ],
            {"status": "executed"},
        ),
    )

    candidates, report = generate_open_vocabulary_coarse_candidates(
        [view], {"fp": info}, {"fp": path}, default_config
    )

    assert len(candidates) == 1
    assert candidates[0].candidate_id.startswith("COARSE-OPEN-VOCAB-fp-")
    assert candidates[0].objects == ["beaker"]
    assert candidates[0].uncertainty == [
        "开放词汇粗筛只补充精筛召回窗口，不独立确认物理动作"
    ]
    assert report["candidate_count"] == 1
    assert report["replaces_closed_set_candidates"] is False


def test_3090ti_profile_enables_single_funnel_superset_while_default_stays_legacy():
    default = load_config()
    production = load_config(Path("configs/rtx3090ti-ubuntu-production.yaml"))

    assert not default["performance"].get("motion_probe_all_views", False)
    assert not default["performance"].get("coarse_full_timeline_scan", False)
    assert not default["performance"].get(
        "fine_risk_window_expansion_enabled", False
    )
    assert production["performance"]["motion_probe_all_views"] is True
    assert production["performance"]["motion_probe_fps"] == 0.5
    assert production["performance"]["coarse_all_views"] is True
    assert production["performance"]["coarse_full_timeline_scan"] is True
    assert production["performance"]["coarse_shared_motion_probe_enabled"] is True
    assert production["performance"]["coarse_frame_index_enabled"] is True
    assert production["performance"]["coarse_semantic_association_enabled"] is True
    assert production["performance"]["coarse_micro_action_guard_enabled"] is True
    assert production["performance"]["candidate_discovery_quality_gate_enabled"] is True
    assert production["performance"]["coarse_reuse_motion_probe"] is False
    assert production["performance"]["detection_fps"] == 20.0
    assert production["performance"]["fine_initial_third_person_views"] == 999


def test_rolling_motion_threshold_adds_quiet_period_recall_without_removing_legacy(
    tmp_path, default_config
):
    view = _view("fp", ViewRole.FIRST_PERSON)
    path = tmp_path / "motion.jsonl"
    frames = []
    for index in range(20):
        score = 100.0 if index < 10 else (5.0 if index == 18 else 6.0 if index == 19 else 0.0)
        frames.append(
            FrameEvidence(
                view_id="fp",
                role=ViewRole.FIRST_PERSON,
                frame_index=index,
                local_ms=float(index * 1_000),
                global_ms=float(index * 1_000),
                width=64,
                height=36,
                motion_score=score,
                raw_motion_score=score,
            )
        )
    path.write_text(
        "\n".join(frame.model_dump_json() for frame in frames) + "\n",
        encoding="utf-8",
    )
    default_config["performance"].update(
        {
            "motion_probe_rolling_threshold_enabled": True,
            "motion_probe_rolling_window_seconds": 10.0,
            "motion_probe_legacy_raw_union_enabled": False,
            "motion_probe_require_objects": False,
            "motion_burst_percentile": 85.0,
            "motion_burst_merge_gap_seconds": 2.0,
            "motion_burst_min_observations": 2,
        }
    )

    candidates = generate_motion_burst_candidates(
        [view], {"fp": path}, default_config
    )

    assert any(candidate.global_start_ms == 0.0 for candidate in candidates)
    assert any(candidate.global_start_ms == 18_000.0 for candidate in candidates)


def test_single_frame_spatial_relation_creates_only_a_fine_scan_guard(
    tmp_path, default_config
):
    view = _view("fp", ViewRole.FIRST_PERSON)
    path = tmp_path / "coarse.jsonl"
    frame = FrameEvidence(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        frame_index=10,
        local_ms=5_000.0,
        global_ms=5_000.0,
        width=640,
        height=360,
        detections=[
            BoxEvidence(
                class_id=0,
                class_name="hand",
                confidence=0.9,
                xyxy_norm=(0.10, 0.10, 0.20, 0.30),
                track_id=1,
            ),
            BoxEvidence(
                class_id=1,
                class_name="beaker",
                confidence=0.85,
                xyxy_norm=(0.19, 0.12, 0.35, 0.40),
                track_id=2,
            ),
        ],
    )
    path.write_text(frame.model_dump_json() + "\n", encoding="utf-8")
    default_config["performance"].update(
        {
            "coarse_spatial_relation_enabled": True,
            "coarse_micro_action_guard_enabled": True,
            "coarse_micro_action_min_confidence": 0.70,
        }
    )

    candidates = generate_coarse_activity_candidates(
        [view], {"fp": path}, default_config
    )

    assert len(candidates) == 1
    assert candidates[0].candidate_id.startswith("COARSE-MICRO-")
    assert candidates[0].evidence[0]["micro_action_guard"] is True
    assert "不独立确认物理动作" in candidates[0].uncertainty[0]


def test_semantic_boundary_refinement_does_not_merge_unrelated_simultaneous_objects(
    default_config,
):
    motion = _candidate()
    motion.objects = []
    motion.global_start_ms = 100_000.0
    motion.global_end_ms = 140_000.0
    motion.local_start_ms = 100_000.0
    motion.local_end_ms = 140_000.0

    def coarse(candidate_id, view_id, object_name, start_ms, end_ms):
        return ActionCandidate(
            candidate_id=candidate_id,
            action_type=ActionType.HAND_OBJECT_CONTACT,
            view_id=view_id,
            role=(
                ViewRole.FIRST_PERSON
                if view_id == "fp"
                else ViewRole.THIRD_PERSON
            ),
            local_start_ms=start_ms,
            local_end_ms=end_ms,
            global_start_ms=start_ms,
            global_end_ms=end_ms,
            key_global_ms=(start_ms + end_ms) / 2.0,
            objects=["hand", object_name],
            confidence=0.8,
        )

    coarse_candidates = [
        coarse("beaker-fp", "fp", "beaker", 100_000.0, 135_000.0),
        coarse("beaker-tp", "tp", "beaker", 105_000.0, 140_000.0),
        coarse("tube-fp", "fp", "tube", 100_000.0, 135_000.0),
        coarse("tube-tp", "tp", "tube", 105_000.0, 140_000.0),
    ]
    default_config["performance"]["coarse_semantic_association_enabled"] = True

    refined, report = refine_motion_candidates_with_coarse(
        [motion], coarse_candidates, default_config
    )

    primary = next(item for item in refined if item.candidate_id.startswith("REFINED-"))
    assert not ({"beaker", "tube"} <= set(primary.objects))
    decision = next(
        item for item in report["decisions"] if item["decision"] == "refined_by_coarse_yolo"
    )
    assert len(decision["discarded_semantic_match_ids"]) == 2
    assert report["output_candidate_count"] == 3


def test_coarse_frame_index_records_exact_coverage_and_filters_activity(
    tmp_path, default_config
):
    view = _view("fp", ViewRole.FIRST_PERSON)
    info = VideoInfo(
        path=Path("fp.mp4"),
        duration_ms=2_000.0,
        fps=30.0,
        width=640,
        height=360,
        frame_count=60,
    )
    ledger = tmp_path / "coarse.jsonl"
    frames = []
    for index in range(5):
        detections = []
        if index == 2:
            detections = [
                BoxEvidence(
                    class_id=0,
                    class_name="hand",
                    confidence=0.9,
                    xyxy_norm=(0.1, 0.1, 0.2, 0.2),
                ),
                BoxEvidence(
                    class_id=1,
                    class_name="beaker",
                    confidence=0.8,
                    xyxy_norm=(0.2, 0.1, 0.3, 0.3),
                ),
            ]
        frames.append(
            FrameEvidence(
                view_id="fp",
                role=ViewRole.FIRST_PERSON,
                frame_index=index,
                local_ms=float(index * 500),
                global_ms=float(index * 500),
                width=640,
                height=360,
                detections=detections,
            )
        )
    ledger.write_text(
        "\n".join(frame.model_dump_json() for frame in frames) + "\n",
        encoding="utf-8",
    )

    frame_index, report = build_coarse_frame_index(
        tmp_path / "coarse.sqlite3",
        [view],
        {"fp": info},
        {"fp": ledger},
        sample_fps=2.0,
    )

    assert report["formal_evidence_ready"] is True
    assert report["row_count"] == 5
    activity = list(frame_index.iter_frames("fp", activity_only=True))
    assert [item.frame_index for item in activity] == [2]


def test_open_vocabulary_errors_make_the_recall_receipt_not_formal(
    monkeypatch, tmp_path, default_config
):
    view = _view("fp", ViewRole.FIRST_PERSON)
    info = VideoInfo(
        path=Path("fp.mp4"),
        duration_ms=1_000.0,
        fps=30.0,
        width=640,
        height=360,
        frame_count=30,
    )
    ledger = tmp_path / "coarse.jsonl"
    ledger.write_text(
        FrameEvidence(
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            frame_index=0,
            local_ms=0.0,
            global_ms=0.0,
            width=640,
            height=360,
            motion_score=10.0,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    default_config["performance"].update(
        {
            "coarse_open_vocabulary_recall_enabled": True,
            "coarse_open_vocabulary_motion_percentile": 0.0,
            "coarse_open_vocabulary_frames_per_hour_per_view": 1,
            "coarse_open_vocabulary_maximum_error_rate": 0.0,
        }
    )
    default_config["models"]["open_vocabulary_key_frame"] = {
        "enabled": True,
        "prompt_map": {"gloved hand": "gloved_hand", "beaker": "beaker"},
    }
    monkeypatch.setattr(
        "visioncortex.coarse_recall.read_view_frame_at",
        lambda *_args, **_kwargs: np.zeros((360, 640, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "visioncortex.coarse_recall._yolo_world_detections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
    )

    candidates, report = generate_open_vocabulary_coarse_candidates(
        [view], {"fp": info}, {"fp": ledger}, default_config
    )

    assert candidates == []
    assert report["error_count"] == 1
    assert report["formal_evidence_ready"] is False
