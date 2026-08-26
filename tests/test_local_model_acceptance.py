from __future__ import annotations

import cv2
import numpy as np

from labvision_evidence.local_model_acceptance import (
    _select_public_example,
    _vessel_seed_box,
)


def test_select_public_example_is_deterministic_and_seed_uses_human_mask(tmp_path):
    test_root = tmp_path / "LabPics Chemistry" / "Test"
    for name in ("0001Eval", "0002Eval", "0003Eval"):
        example = test_root / name
        semantic = example / "SemanticMaps" / "FullImage"
        semantic.mkdir(parents=True)
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[2:8, 5:15] = 255
        assert cv2.imwrite(str(example / "Image.jpg"), image)
        assert cv2.imwrite(str(semantic / "Vessel.png"), mask)

    selected = _select_public_example(tmp_path)
    seed = _vessel_seed_box(selected, (10, 20))

    assert selected.name == "0002Eval"
    assert seed == [0.25, 0.2, 0.75, 0.8]
