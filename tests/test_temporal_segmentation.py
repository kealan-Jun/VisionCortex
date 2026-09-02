from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from visioncortex import temporal_segmentation as module


# The production implementation imports PyTorch lazily.  This deterministic
# unit test exercises the SAM2 contract and cache without making the complete
# GPU runtime a mandatory development dependency.
torch = SimpleNamespace(
    zeros=lambda shape: np.zeros(shape, dtype=np.float32),
    full=lambda shape, value: np.full(shape, value, dtype=np.float32),
    stack=lambda values: np.stack(values),
    where=np.where,
    inference_mode=nullcontext,
    autocast=lambda *_args, **_kwargs: nullcontext(),
)


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", torch)


class _FakePredictor:
    def __init__(self) -> None:
        self.boxes: dict[int, np.ndarray] = {}

    def init_state(self, frame_root: str, **_kwargs):
        frames = sorted(Path(frame_root).glob("*.jpg"))
        image = cv2.imread(str(frames[0]))
        return {
            "frame_count": len(frames),
            "height": image.shape[0],
            "width": image.shape[1],
        }

    def add_new_points_or_box(
        self, _state, *, frame_idx, obj_id, box, **_kwargs
    ):
        self.boxes[int(obj_id)] = np.asarray(box, dtype=np.float32)
        return frame_idx, [obj_id], torch.zeros((1, 1, 1, 1))

    def propagate_in_video(
        self, state, *, start_frame_idx, max_frame_num_to_track, reverse=False
    ):
        if reverse:
            order = range(start_frame_idx, -1, -1)
        else:
            order = range(start_frame_idx, state["frame_count"])
        for frame_index in list(order)[:max_frame_num_to_track]:
            masks = []
            for object_id in sorted(self.boxes):
                x1, y1, x2, y2 = self.boxes[object_id]
                shift = frame_index - start_frame_idx
                left = max(0, min(state["width"] - 1, int(x1 + shift)))
                right = max(left + 1, min(state["width"], int(x2 + shift)))
                top = max(0, min(state["height"] - 1, int(y1)))
                bottom = max(top + 1, min(state["height"], int(y2)))
                mask = torch.full(
                    (1, state["height"], state["width"]), -1.0
                )
                mask[:, top:bottom, left:right] = 1.0
                masks.append(mask)
            yield frame_index, sorted(self.boxes), torch.stack(masks)

    def reset_state(self, _state) -> None:
        return None


def _video(path: Path) -> np.ndarray:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 96)
    )
    assert writer.isOpened()
    seed = None
    for index in range(6):
        frame = np.zeros((96, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30 + index, 24), (70 + index, 70), (0, 180, 255), -1)
        writer.write(frame)
        if index == 3:
            seed = frame.copy()
    writer.release()
    assert seed is not None
    return seed


def _config() -> dict:
    return {
        "models": {
            "temporal_participant_segmentation": {
                "enabled": True,
                "enabled_actions": ["liquid_movement"],
                "device": "cpu",
                "maximum_frames_per_clip": 5,
                "jpeg_quality": 95,
                "minimum_presence_ratio": 0.5,
                "minimum_seed_box_iou": 0.35,
                "maximum_seed_area_expansion_ratio": 1.35,
            }
        }
    }


def test_sample_indices_are_bounded_and_keep_seed():
    indices = module._sample_indices(100, 9, 63)

    assert len(indices) == 9
    assert indices == sorted(set(indices))
    assert 63 in indices


def test_bounded_sam2_receipt_refines_boxes_and_reuses_cache(
    tmp_path: Path, monkeypatch
):
    clip = tmp_path / "key.mp4"
    seed = _video(clip)
    predictor = _FakePredictor()
    monkeypatch.setattr(
        module,
        "_load_predictor",
        lambda _config: (
            predictor,
            {
                "model": "fake-sam2",
                "checkpoint_sha256": "test",
                "model_load_seconds": 0.0,
            },
        ),
    )
    boxes = [
        {
            "class_name": "paper",
            "confidence": 0.8,
            "xyxy_norm": [30 / 160, 24 / 96, 70 / 160, 70 / 96],
        }
    ]

    refined, receipt = module.audit_participant_continuity(
        clip,
        seed,
        boxes,
        tmp_path / "work",
        _config(),
        event_id="evt-1",
        view_id="fp",
        action_type="liquid_movement",
        seed_fraction=0.6,
    )

    assert receipt["status"] == "completed"
    assert receipt["passed"] is True
    assert receipt["source_copy_bytes"] == 0
    assert receipt["full_timeline_inference"] is False
    assert receipt["sampled_frame_count"] <= 5
    assert refined[0]["segmentation_refined"] is True

    cached_boxes, cached = module.audit_participant_continuity(
        clip,
        seed,
        boxes,
        tmp_path / "work",
        _config(),
        event_id="evt-1",
        view_id="fp",
        action_type="liquid_movement",
        seed_fraction=0.6,
    )

    assert cached["cache_reused"] is True
    assert cached_boxes == refined


def test_runtime_validation_fails_closed_when_checkpoint_is_missing(tmp_path: Path):
    config = _config()
    settings = config["models"]["temporal_participant_segmentation"]
    settings.update(
        {
            "checkpoint_path": str(tmp_path / "missing.pt"),
            "checkpoint_sha256": "0" * 64,
        }
    )

    with pytest.raises(FileNotFoundError, match="SAM2 checkpoint is missing"):
        module.validate_temporal_segmentation_runtime(config)
