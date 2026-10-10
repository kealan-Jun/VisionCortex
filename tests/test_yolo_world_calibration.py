from __future__ import annotations

import pytest

from visioncortex.yolo_world_calibration import _canonical_nms


def test_canonical_nms_deduplicates_synonymous_prompt_boxes():
    boxes = _canonical_nms(
        [
            {
                "class_name": "pipette",
                "confidence": 0.8,
                "xyxy": [0.0, 0.0, 10.0, 10.0],
                "grounding_prompt": "automatic micropipette",
            },
            {
                "class_name": "pipette",
                "confidence": 0.7,
                "xyxy": [1.0, 1.0, 11.0, 11.0],
                "grounding_prompt": "handheld electronic pipette",
            },
            {
                "class_name": "hand",
                "confidence": 0.6,
                "xyxy": [1.0, 1.0, 11.0, 11.0],
                "grounding_prompt": "bare hand",
            },
        ]
    )

    assert [item["class_name"] for item in boxes] == ["pipette", "hand"]
    assert boxes[0]["confidence"] == 0.8


def test_canonical_nms_rejects_invalid_iou():
    with pytest.raises(ValueError, match="IoU"):
        _canonical_nms([], iou_threshold=1.1)
