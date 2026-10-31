"""Structural-search software contracts; no synthetic numeric quality claim."""

import copy
import hashlib

import cv2
import numpy as np
import pytest

from visioncortex.lab.source_reacquisition import METHOD, reacquire, verified
from visioncortex.lab.source_tracking import SourcePanelTracker, align_native_roi


def case():
    cv2.setRNGSeed(0)
    gray = np.random.default_rng(57).integers(20, 220, (240, 360), np.uint8)
    reference = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), .5), cv2.COLOR_GRAY2BGR)
    tracker = SourcePanelTracker(
        reference, [.15, .15, .8, .85], [[.35, .35, .65, .65]],
        required_boxes=[[.3, .3, .7, .7]], motion_digit_boxes=[[.35, .35, .65, .65]],
        scale_compatible_reacquisition=True,
    )
    matrix = cv2.getRotationMatrix2D((180, 120), -9, .63)
    image = cv2.warpAffine(reference, matrix, (360, 240))
    reference_points = np.float32([[65, 45], [275, 195]])
    current = cv2.transform(reference_points[None], matrix)[0]
    return tracker, reference, image, current, reference_points


def test_sparse_seed_can_only_authorize_independently_verified_native_structure():
    tracker, reference, image, current, points = case()
    before = reference.copy(), image.copy()
    result, reason = reacquire(tracker, image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), current, points)
    assert reason is None and result is not None
    registration = result["registration"]
    assert registration["method"] == METHOD and verified(registration)
    assert registration["stable_inliers"] >= 8
    assert registration["error_p95_pixels"] <= 1.5
    assert registration["scale_compatible_search"]["verified_patches"] >= 8
    assert registration["scale_compatible_search"]["minimum_patch_correlation"] >= .85
    left, top, right, bottom = result["box"]
    assert np.array_equal(result["image"], image[top:bottom, left:right])
    assert np.array_equal(before[0], reference) and np.array_equal(before[1], image)
    source = dict(upright_frame_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    align_native_roi(result["image"], registration, source)
    invalid = copy.deepcopy(registration)
    invalid["scale_compatible_search"]["verified_patches"] = 7
    with pytest.raises(ValueError, match="Source-bound"):
        align_native_roi(result["image"], invalid, source)


@pytest.mark.parametrize("different", ["blank", "unrelated", "displaced_seed"])
def test_seed_itself_never_accepts_wrong_pixels_or_an_unverified_pose(different):
    tracker, _, image, current, points = case()
    if different == "blank":
        image[:] = 0
    elif different == "unrelated":
        image = np.random.default_rng(71).integers(0, 255, image.shape, np.uint8)
    else:
        current += [250, 100]
    result, reason = reacquire(tracker, image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), current, points)
    assert result is None and reason


@pytest.mark.parametrize("count", [0, 1, 65])
def test_reacquisition_rejects_unbounded_or_insufficient_seed_sets(count):
    tracker, _, image, _, _ = case()
    points = np.zeros((count, 2), np.float32)
    result, reason = reacquire(tracker, image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), points, points)
    assert result is None and reason == "bounded_sparse_structure_required"


def test_scaled_search_is_opt_in_and_requires_registered_native_resolution():
    tracker, reference, _, _, _ = case()
    assert tracker.scale_compatible_reacquisition is True
    args = (reference, [.15, .15, .8, .85], [[.35, .35, .65, .65]])
    assert SourcePanelTracker(*args).scale_compatible_reacquisition is False
    with pytest.raises(ValueError, match="field-motion"):
        SourcePanelTracker(*args, scale_compatible_reacquisition=True)
    with pytest.raises(ValueError, match="field-motion"):
        SourcePanelTracker(*args, scale_compatible_reacquisition="yes")


@pytest.mark.parametrize("enabled,limit", [(False, 1), (True, 1)])
def test_lost_frames_obey_rolling_descriptor_budget(enabled, limit):
    tracker, _, _, _, _ = case()
    tracker.scale_compatible_reacquisition = enabled
    tracker.current_points = None
    counts = []
    for _ in range(33):
        result = tracker.observe(np.zeros(tracker.shape, np.uint8))
        assert result["status"] == "pending" and result["stale_ROI_reused"] is False
        counts.append(int(result["diagnostics"]["descriptor_search"]))
    assert sum(counts[:15]) == limit
    assert all(sum(counts[i:i+15]) <= limit for i in range(len(counts)-14))
    searched = [i for i, count in enumerate(counts) if count]
    assert all(b-a >= 15 for a, b in zip(searched, searched[1:], strict=False))
