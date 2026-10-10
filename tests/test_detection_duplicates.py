import json
from pathlib import Path

import numpy as np
import pytest

from visioncortex.detection import (
    ChunkEnd,
    FramePacket,
    ProducerEnd,
    _read_checkpoint,
    _write_checkpoint,
    iter_frame_evidence,
    scan_videos,
)
from visioncortex.detection_duplicates import (
    duplicate_suppression_policies,
    suppress_duplicate_boxes,
)
from visioncortex.schemas import (
    AlignmentTransform,
    BoxEvidence,
    DetectionDuplicateSuppression,
    VideoInfo,
    ViewInput,
    ViewRole,
)


def box(confidence=0.9, xyxy=(0.1, 0.1, 0.3, 0.9), class_id=12, name="pipette"):
    return BoxEvidence(
        class_id=class_id, class_name=name, confidence=confidence, xyxy_norm=xyxy
    )


def test_suppression_keeps_adjacent_instruments_and_other_classes_and_raw_evidence():
    raw = [box(0.8), box(0.95), box(0.9, (0.3, 0.1, 0.5, 0.9)),
           box(0.9, class_id=21, name="pipette_rack")]
    kept, audit = suppress_duplicate_boxes(raw, 0.9)
    assert kept == [raw[1], raw[2], raw[3]]
    assert audit.retained_input_indices == [1, 2, 3]
    assert [(r.removed_input_index, r.retained_input_index) for r in audit.removals] == [(0, 1)]
    kept[0].track_id = 99
    assert all(b.track_id is None for b in audit.raw_detections)
    restored = DetectionDuplicateSuppression.model_validate_json(audit.model_dump_json())
    assert restored == audit
    # Each image is independent, even when every prediction has identical coordinates.
    assert len(suppress_duplicate_boxes([box(), box(0.8)], 0.9)[0]) == 1


def test_postclip_geometry_removes_real_edge_duplicates_without_losing_adjacent_guns():
    # Numeric coordinates from goal19's TP native prediction, 1280 x 372 crop.
    # Before rescale/clipping these duplicate pairs had IoU .652/.695 (< .7).
    coordinates = [
        (864.3133544921875, 196.21591186523438, 1007.609375, 372.0),
        (883.32421875, 197.275390625, 1012.4938354492188, 372.0),
        (881.5611572265625, 185.08041381835938, 1048.70947265625, 372.0),
        (897.7081298828125, 188.0850830078125, 1050.6083984375, 372.0),
    ]
    scores = [.5308635830879211, .8560303449630737, .48451942205429077, .847196102142334]
    raw = [box(score, (x1 / 1280, y1 / 372, x2 / 1280, y2 / 372))
           for score, (x1, y1, x2, y2) in zip(scores, coordinates, strict=True)]
    policy = duplicate_suppression_policies({"models": {
        "duplicate_suppression_iou_by_role": {"third_person": .7},
    }})[ViewRole.THIRD_PERSON]
    kept, audit = suppress_duplicate_boxes(raw, policy["iou_threshold"])
    assert kept == [raw[1], raw[3]]  # Two distinct neighboring guns survive.
    assert audit.retained_input_indices == [1, 3]
    assert {r.removed_input_index for r in audit.removals} == {0, 2}
    assert audit.raw_detections == raw  # Suppressed model evidence is retained.


@pytest.mark.parametrize("threshold", [0.5, 0.9])
def test_suppression_ties_and_exact_threshold_are_deterministic(threshold):
    raw = [box(0.8, (0, 0, threshold, 1)), box(0.9, (0, 0, 1, 1)), box(0.9, (0, 0, 1, 1))]
    kept, audit = suppress_duplicate_boxes(raw, threshold)
    assert kept == [raw[1]]
    assert audit.retained_input_indices == [1]
    assert {r.removed_input_index for r in audit.removals} == {0, 2}
    assert next(r.iou for r in audit.removals if r.removed_input_index == 0) == pytest.approx(threshold)


@pytest.mark.parametrize("value", [True, None, "0.9", 0, -0.1, 1.01, float("nan"), float("inf")])
def test_policy_rejects_invalid_thresholds(value):
    with pytest.raises(ValueError, match="IoU"):
        duplicate_suppression_policies({"models": {"duplicate_suppression_iou_by_role": {
            "first_person": value,
        }}})


def test_policy_is_disabled_by_default_and_does_not_leak_across_roles():
    assert duplicate_suppression_policies({"models": {}}) == {}
    policies = duplicate_suppression_policies({"models": {"duplicate_suppression_iou_by_role": {
        "first_person": 0.9,
    }}})
    assert set(policies) == {ViewRole.FIRST_PERSON}
    with pytest.raises(ValueError, match="role"):
        duplicate_suppression_policies({"models": {"duplicate_suppression_iou_by_role": {
            "cam01": 0.9,
        }}})
    with pytest.raises(ValueError, match="mapping"):
        duplicate_suppression_policies({"models": {"duplicate_suppression_iou_by_role": 0.9}})


@pytest.mark.parametrize("detection", [box(float("nan")), box(xyxy=(0, 0, 0, 1)),
                                      box(xyxy=(0, 0, 2, 1))])
def test_invalid_model_geometry_is_not_silently_suppressed(detection):
    with pytest.raises(ValueError, match="finite, valid"):
        suppress_duplicate_boxes([detection], 0.9)


def test_audit_cannot_drop_a_raw_box_or_reference_another_removed_box():
    _, audit = suppress_duplicate_boxes([box(), box(0.8), box(0.7)], 0.9)
    data = audit.model_dump()
    data["removals"].pop()
    with pytest.raises(ValueError, match="every raw detection"):
        DetectionDuplicateSuppression.model_validate(data)
    data = audit.model_dump()
    data["removals"][0]["retained_input_index"] = 2
    with pytest.raises(ValueError, match="retained raw detection"):
        DetectionDuplicateSuppression.model_validate(data)


@pytest.mark.parametrize("original,replacement", [(None, {"iou_threshold": 0.9}),
                                                ({"iou_threshold": 0.9}, None),
                                                ({"iou_threshold": 0.9}, {"iou_threshold": 0.95})])
def test_changed_policy_rejects_resume_without_truncating_existing_output(
    tmp_path, original, replacement
):
    output = tmp_path / "view.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    output.write_bytes(b"completed\n")
    _write_checkpoint(checkpoint, {0}, output, duplicate_policy=original)
    checkpoint_bytes = checkpoint.read_bytes()
    output.write_bytes(b"completed\nuncommitted tail\n")
    with pytest.raises(RuntimeError, match="policy changed"):
        _read_checkpoint(checkpoint, output, duplicate_policy=replacement)
    assert output.read_bytes() == b"completed\nuncommitted tail\n"
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert _read_checkpoint(checkpoint, output, duplicate_policy=original) == {0}
    assert output.read_bytes() == b"completed\n"


def test_scan_suppresses_before_tracking_per_frame_and_role_and_preserves_resume(
    tmp_path, monkeypatch, default_config
):
    views = [ViewInput(view_id=name, role=role, video=Path(name + ".mp4"))
             for name, role in [("fp1", ViewRole.FIRST_PERSON), ("fp2", ViewRole.FIRST_PERSON),
                                ("tp", ViewRole.THIRD_PERSON)]]
    infos = {v.view_id: VideoInfo(path=v.video, duration_ms=1000, fps=2, width=8, height=8,
                                 frame_count=2) for v in views}
    transforms = {v.view_id: AlignmentTransform(view_id=v.view_id, reference_view_id=v.view_id,
                                               state="aligned", confidence=1) for v in views}

    class FakeScanner:
        calls = 0

        def __init__(self, *_args, **_kwargs):
            self.model_path = Path("fake.engine")
            self.batch_size = self.engine_build_batch = 4
            self.last_inference_batch_sizes = []

        def infer(self, packets):
            FakeScanner.calls += 1
            self.last_inference_batch_sizes = [len(packets)]
            return [[box(), box(0.8), box(0.85, (0.4, 0.1, 0.6, 0.9))] for _ in packets]

        def close(self):
            pass

    def producer(view, _info, output, completed, *_args):
        if 0 not in completed:
            for index in range(2):
                frame = np.zeros((8, 8, 3), dtype=np.uint8)
                output.put(FramePacket(view=view, frame_index=index, local_ms=index * 500,
                                       frame=frame, gray=frame[:, :, 0], previous_gray=None,
                                       motion_score=0.0))
            output.put(ChunkEnd(view_id=view.view_id, chunk_index=0, total_chunks=1))
        output.put(ProducerEnd(view_id=view.view_id))

    monkeypatch.setattr("visioncortex.detection.RoleScanner", FakeScanner)
    monkeypatch.setattr("visioncortex.detection._producer", producer)
    default_config["models"]["duplicate_suppression_iou_by_role"] = {"first_person": 0.9}
    paths = scan_videos(views, infos, transforms, tmp_path, default_config, phase="coarse")
    for view in views:
        frames = list(iter_frame_evidence(paths[view.view_id]))
        assert [f.local_ms for f in frames] == [0, 500]
        for frame in frames:
            assert frame.view_id == view.view_id and frame.role == view.role
            if view.role == ViewRole.FIRST_PERSON:
                assert len(frame.detections) == 2
                assert len(frame.duplicate_suppression.raw_detections) == 3
                assert len(frame.duplicate_suppression.removals) == 1
                assert all(b.track_id is None for b in frame.duplicate_suppression.raw_detections)
                assert {b.track_id for b in frame.detections} == {1, 2}
            else:
                assert len(frame.detections) == 3
                assert frame.duplicate_suppression is None
    runtime = json.loads((tmp_path / "runtime_coarse_first_person.json").read_text())
    assert runtime["duplicate_suppression"]["input_detections"] == 12
    assert runtime["duplicate_suppression"]["suppressed_detections"] == 4
    before = {k: p.read_bytes() for k, p in paths.items()}
    calls = FakeScanner.calls
    scan_videos(views, infos, transforms, tmp_path, default_config, phase="coarse")
    assert FakeScanner.calls == calls
    assert {k: p.read_bytes() for k, p in paths.items()} == before
    default_config["models"]["duplicate_suppression_iou_by_role"] = {}
    with pytest.raises(RuntimeError, match="policy changed"):
        scan_videos(views, infos, transforms, tmp_path, default_config, phase="coarse")
    assert {k: p.read_bytes() for k, p in paths.items()} == before
