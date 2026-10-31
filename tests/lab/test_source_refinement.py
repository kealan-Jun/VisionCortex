"""Structural alignment contracts, not real OCR quality or synthetic gold."""

import copy

import cv2
import numpy as np
import pytest

from visioncortex.lab.source_refinement import METHOD, refine, verified


def fixture(dx=1.2, dy=-.6):
    rng = np.random.default_rng(7)
    reference = cv2.GaussianBlur(rng.integers(0, 255, (100, 160), np.uint8), (5, 5), 0)
    reference = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX)
    current = cv2.warpAffine(reference, np.float32([[1, 0, dx], [0, 1, dy]]), (160, 100))
    mask = np.zeros(reference.shape, np.uint8)
    mask[8:-8, 8:-8] = 255
    mask[30:80, 50:110] = 0
    # Independent changing content is excluded from structural fitting.
    reference[40:65, 65:95] = 250
    current[40:65, 65:95] = 0
    image = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
    return image, reference, mask


def registration():
    return dict(method="source_panel_descriptor_reacquisition/1",
                matrix_current_to_reference=np.eye(3).tolist())


def test_small_bidirectional_improvement_preserves_source_pixels():
    image, reference, mask = fixture()
    before = image.copy()
    original = registration()
    result = refine(image, original, reference, mask)
    assert result["method"] == METHOD and verified(result)
    assert np.array_equal(image, before) and original == registration()
    translation = result["stable_translation_refinement"]["translation_pixels"]
    assert translation == pytest.approx([1.2, -.6], abs=.2)
    assert result["stable_translation_refinement"]["numeric_pixels_supplied"] is False


@pytest.mark.parametrize("failure", ["large_motion", "flat", "small_mask", "other_scene", "no_gain"])
def test_unreliable_refinement_retains_initial_geometry(failure):
    image, reference, mask = fixture(dx=8 if failure == "large_motion" else 1.2)
    if failure == "flat":
        reference[:] = 100
    elif failure == "small_mask":
        mask[:] = 0
        mask[1:5, 1:5] = 255
    elif failure == "other_scene":
        image = np.random.default_rng(80).integers(0, 255, image.shape, np.uint8)
    elif failure == "no_gain":
        image = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    original = registration()
    result = refine(image, original, reference, mask)
    assert result["matrix_current_to_reference"] == original["matrix_current_to_reference"]
    assert result["method"] == original["method"]
    assert not result["stable_translation_refinement"]["accepted"]


@pytest.mark.parametrize("key,value", [("accepted", False), ("digits_excluded", False),
                                        ("numeric_pixels_supplied", True),
                                        ("translation_pixels", [3, 0]),
                                        ("reverse_closure_pixels", 1),
                                        ("correlation_after", .3)])
def test_refined_registration_requires_all_source_proofs(key, value):
    image, reference, mask = fixture()
    result = refine(image, registration(), reference, mask)
    assert verified(result)
    changed = copy.deepcopy(result)
    changed["stable_translation_refinement"][key] = value
    assert not verified(changed)
