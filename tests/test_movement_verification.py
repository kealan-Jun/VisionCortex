from copy import deepcopy

import cv2
import numpy as np
import pytest

from visioncortex.actions import _frame_observations, audit_candidates
from visioncortex.archive import ArchiveLayout, write_screening_notes
from visioncortex.movement_verification import verify_image_motion, verify_movement_candidates
from visioncortex.schemas import ActionCandidate, ActionType, AlignmentTransform, BoxEvidence, FrameEvidence, ViewRole


def test_smooth_subthreshold_motion_accumulates_over_time(default_config):
    tracks, history = {}, {}
    observed = []
    for index in range(12):
        offset = index * .003
        frame = FrameEvidence(view_id="fp", role=ViewRole.FIRST_PERSON, frame_index=index,
                              local_ms=index * 50, global_ms=index * 50, width=640, height=400,
                              detections=[BoxEvidence(class_id=12, class_name="pipette", confidence=.9,
                                  track_id=1, xyxy_norm=(.2 + offset, .2, .3 + offset, .5))])
        observed.extend(_frame_observations(frame, tracks, default_config["segmentation"],
                                            movement_history=history))
    movement = [item for item in observed if item.action_type == ActionType.OBJECT_MOVEMENT]
    assert movement
    assert any(item.evidence["delta_ms"] == 250 for item in movement)


def scene(object_shift=0, camera_shift=(0, 0)):
    random = np.random.default_rng(5)
    background = cv2.GaussianBlur(random.integers(0, 255, (240, 320), np.uint8), (3, 3), 0)
    patch = random.integers(0, 255, (60, 60), np.uint8)
    first, last = background.copy(), background.copy()
    first[90:150, 120:180] = patch
    last[90:150, 120 + object_shift:180 + object_shift] = patch
    dx, dy = camera_shift
    last = cv2.warpAffine(last, np.float32([[1, 0, dx], [0, 1, dy]]), (320, 240))
    return first, last, (120 / 320, 90 / 240, 180 / 320, 150 / 240), (
        (120 + object_shift + dx) / 320, (90 + dy) / 240,
        (180 + object_shift + dx) / 320, (150 + dy) / 240,
    )


def test_stationary_texture_contradicts_jumping_detection_box():
    first, last, box, _ = scene()
    result = verify_image_motion(first, last, box, (0.4, 0.4, 0.6, 0.7))
    assert result["status"] == "contradicted"
    assert result["reason"] == "stationary_image_features"


def test_camera_translation_does_not_become_object_movement():
    result = verify_image_motion(*scene(camera_shift=(6, 4)))
    assert result["status"] == "contradicted"
    assert result["reason"] == "camera_motion_only"
    assert result["raw_median_displacement_px"] > 6


@pytest.mark.parametrize("camera_shift", [(0, 0), (6, 4)])
def test_real_relative_motion_survives_background_compensation(camera_shift):
    result = verify_image_motion(*scene(object_shift=8, camera_shift=camera_shift))
    assert result["status"] == "supported"
    assert result["compensated_median_displacement_px"] > 6


def test_textureless_and_occluded_images_are_unverified():
    empty = np.zeros((240, 320), np.uint8)
    assert verify_image_motion(empty, empty, (.2, .2, .4, .4), (.3, .2, .5, .4))["status"] == "unverified"
    first, _, box, last_box = scene()
    assert verify_image_motion(first, empty, box, last_box)["status"] == "unverified"


def candidate(name="move", role=ViewRole.FIRST_PERSON):
    return ActionCandidate(candidate_id=name, action_type=ActionType.OBJECT_MOVEMENT,
                           view_id=role.value, role=role, local_start_ms=1000, local_end_ms=2000,
                           global_start_ms=1000, global_end_ms=2000, key_global_ms=1500,
                           objects=["pipette"], confidence=.9)


def test_unverified_candidate_is_retained_and_cannot_supply_cross_view_support(default_config, tmp_path):
    first = candidate()
    second = candidate("other", ViewRole.THIRD_PERSON)
    original = deepcopy(first.model_dump())
    report = verify_movement_candidates([first], [], {}, {}, default_config)
    assert report["deleted_candidates"] == 0
    assert first.confidence == original["confidence"]
    assert first.provenance["movement_visual_verification"]["status"] == "unverified"
    transforms = {c.view_id: AlignmentTransform(view_id=c.view_id, reference_view_id="first_person",
                                              state="aligned", confidence=1) for c in [first, second]}
    events, rejected = audit_candidates([first, second], transforms, default_config)
    assert any(r["candidate_id"] == first.candidate_id and r["reason"] == "movement_not_supported_by_image" for r in rejected)
    assert all(first.view_id not in event.supporting_views for event in events)
    layout = ArchiveLayout(tmp_path)
    layout.create()
    write_screening_notes(layout, [], events, rejected, [])
    notes = (layout.key_materials / "Screening-Notes.txt").read_text()
    assert "movement_not_supported_by_image" in notes


def test_missing_motion_source_does_not_modify_other_action(default_config, tmp_path):
    moving, contact = candidate(), candidate("contact")
    moving.evidence = [{"observation_local_ms": 1500., "delta_ms": 100., "track_id": 1}]
    contact.action_type = ActionType.HAND_OBJECT_CONTACT
    saved = contact.model_dump()
    report = verify_movement_candidates([moving, contact], [], {}, {moving.view_id: tmp_path / "missing.jsonl"}, default_config)
    assert report["counts"] == {"unverified": 1}
    assert contact.model_dump() == saved


def test_parallel_views_preserve_order_global_limit_and_candidate_receipts(default_config, monkeypatch, tmp_path):
    import visioncortex.movement_verification as module

    monkeypatch.setattr(module, "_selected_frames", lambda _path, _targets: {})
    items = [candidate(str(i), ViewRole.FIRST_PERSON if i % 2 else ViewRole.THIRD_PERSON)
             for i in range(7)]
    for item in items:
        item.evidence = [{"observation_local_ms": 1500., "delta_ms": 100., "track_id": 1}]
    paths = {role.value: tmp_path / role.value for role in ViewRole}
    cfg = deepcopy(default_config)
    cfg["segmentation"]["movement_visual_verification"].update(max_candidates=3, workers=1)
    original, parallel = deepcopy(items), deepcopy(items)
    expected = verify_movement_candidates(original, [], {}, paths, cfg)
    cfg["segmentation"]["movement_visual_verification"]["workers"] = 4
    progress = []
    actual = verify_movement_candidates(parallel, [], {}, paths, cfg,
                                        progress=lambda done, total: progress.append((done, total)))
    assert actual["candidates"] == expected["candidates"]
    assert actual["counts"] == expected["counts"]
    assert [c.model_dump() for c in parallel] == [c.model_dump() for c in original]
    assert sum(c["reason"] == "verification_budget_exhausted" for c in actual["candidates"]) == 4
    assert progress[-1] == (7, 7)
    assert actual["model_calls"] == actual["deleted_candidates"] == 0


def test_parallel_views_preserve_missing_ledger_failures(default_config, tmp_path):
    items = [candidate("fp"), candidate("tp", ViewRole.THIRD_PERSON)]
    for item in items:
        item.evidence = [{"observation_local_ms": 1500., "delta_ms": 100., "track_id": 1}]
    cfg = deepcopy(default_config)
    cfg["segmentation"]["movement_visual_verification"]["workers"] = 2
    paths = {role.value: tmp_path / role.value for role in ViewRole}
    result = verify_movement_candidates(items, [], {}, paths, cfg)
    assert result["counts"] == {"unverified": 2}
    assert set(result["ledger_failures"]) == set(paths)
    assert all(c["reason"] == "source_ledger_unavailable" for c in result["candidates"])
