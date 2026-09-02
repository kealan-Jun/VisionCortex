from pathlib import Path

import cv2
import numpy as np

from labvision_evidence.coarse_recall import (
    generate_open_vocabulary_coarse_candidates,
    select_suspicious_coarse_frames,
)
from labvision_evidence.config import load_config
from labvision_evidence.detection import _camera_compensated_motion_score
from labvision_evidence.pipeline import EvidencePipeline
from labvision_evidence.schemas import (
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
        "labvision_evidence.coarse_recall.read_view_frame_at",
        lambda *_args, **_kwargs: np.zeros((360, 640, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "labvision_evidence.coarse_recall._yolo_world_detections",
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
    assert production["performance"]["coarse_reuse_motion_probe"] is False
    assert production["performance"]["detection_fps"] == 20.0
    assert production["performance"]["fine_initial_third_person_views"] == 999
