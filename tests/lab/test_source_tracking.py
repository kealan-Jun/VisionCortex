"""Geometry contracts; generated structure is not real-video quality evidence."""

import hashlib

import cv2
import numpy as np
import pytest

from visioncortex.lab.source_tracking import SourcePanelTracker, align_native_roi
from visioncortex.lab.emission_geometry import reanchor
from visioncortex.lab.temporal_display import TemporalDisplay


def structure(seed=12):
    rng = np.random.default_rng(seed)
    gray = rng.integers(20, 220, (240, 360), dtype=np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    return image


def tracker(image):
    return SourcePanelTracker(image, [0.2, 0.2, 0.65, 0.8], [[0.3, 0.3, 0.7, 0.6]])


def test_source_translation_keeps_native_pixels_and_reference_geometry():
    image = structure()
    obj = tracker(image)
    first = obj.observe(image)
    assert first["status"] == "tracked"
    moved = cv2.warpAffine(image, np.float32([[1, 0, 30], [0, 1, 8]]), (360, 240))
    result = obj.observe(moved)
    assert result["status"] == "tracked"
    left, top, right, bottom = result["box"]
    assert np.array_equal(result["image"], moved[top:bottom, left:right])
    assert abs(left - first["box"][0] - 30) <= 1
    assert abs(top - first["box"][1] - 8) <= 1
    source = dict(
        upright_frame_pixels_sha256=hashlib.sha256(moved.tobytes()).hexdigest()
    )
    aligned, shape = align_native_roi(result["image"], result["registration"], source)
    assert aligned.shape == shape == obj.reference_shape
    assert obj.buffer_bytes < 16 * 1024**2


def test_lost_panel_never_reuses_stale_roi_and_can_reacquire_structure():
    image = structure()
    obj = tracker(image)
    assert obj.observe(image)["status"] == "tracked"
    blank = np.zeros_like(image)
    result = obj.observe(blank)
    assert result["status"] == "pending"
    assert result["stale_ROI_reused"] is False
    assert "image" not in result
    # Rate-limited descriptor reacquisition can recover the same source layout.
    for _ in range(15):
        result = obj.observe(image)
    assert result["status"] == "tracked"
    assert result["registration"]["equipment_identity_inferred"] is False


def test_different_structure_and_tampered_pixels_do_not_authorize_geometry():
    image = structure()
    obj = tracker(image)
    result = obj.observe(image)
    wrong_source = dict(upright_frame_pixels_sha256="a" * 64)
    with pytest.raises(ValueError, match="Source-bound"):
        align_native_roi(result["image"], result["registration"], wrong_source)
    source = dict(
        upright_frame_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()
    )
    tampered = result["image"].copy()
    tampered[0, 0] ^= 255
    with pytest.raises(ValueError, match="Source-bound"):
        align_native_roi(tampered, result["registration"], source)
    assert obj.observe(structure(19))["status"] == "pending"


def test_dynamic_digits_are_not_registration_features():
    image = structure()
    obj = tracker(image)
    x, y, _, _ = obj.box
    a, b, c, d = 48 + x, 43 + y, 114 + x, 87 + y
    # Use the actual excluded mask rather than a stored glyph or reading.
    assert obj.reference_points is not None
    for px, py in obj.reference_points[:, 0]:
        assert obj.mask[round(py), round(px)] > 0
    changed = image.copy()
    changed[b:d, a:c] = 255 - changed[b:d, a:c]
    assert obj.observe(changed)["status"] == "tracked"


def test_LED_geometry_can_move_without_supplying_digit_or_decimal_contents():
    image = np.zeros((100, 140, 3), np.uint8)
    image[25:45, 40:50, 2] = 220
    image[25:45, 70:80, 2] = 220
    layout = dict(color="red", digits=[[0.2, 0.4, 0.3, 0.65], [0.4, 0.4, 0.5, 0.65]])
    corrected, receipt = reanchor(image, layout)
    assert corrected is not None
    assert receipt["offset_pixels"][1] < 0
    assert len(corrected["digits"]) == 2
    assert not receipt["glyphs_inferred"] and not receipt["decimal_inferred"]
    pending, receipt = reanchor(np.zeros_like(image), layout)
    assert pending is None and receipt["status"] == "pending"


def test_shared_temporal_reader_uses_unresampled_pixels_after_source_motion():
    image = structure()
    image[92:115, 112:128] = (0, 0, 240)
    image[92:115, 152:168] = (0, 0, 240)
    obj = tracker(image)
    layout = dict(
        revision="geometry-fixture/1",
        color="red",
        display_type="seven_segment",
        numeric_anchor=True,
        numeric_native_ocr=True,
        digits=[[0.24, 0.3, 0.36, 0.5], [0.48, 0.3, 0.6, 0.5]],
        dynamic_boxes=[[0.2, 0.25, 0.65, 0.55]],
    )
    identity = dict(
        device_id="source",
        screen_id="panel",
        field_id="digits",
        track_session_id="session",
    )
    engine = TemporalDisplay()
    calls = []

    def ocr(pixels, **kwargs):
        calls.append((pixels.copy(), kwargs["anchor_source"]))
        return dict(raw_text="12")

    for i in range(6):
        moved = cv2.warpAffine(
            image, np.float32([[1, 0, i * 3], [0, 1, i]]), (360, 240)
        )
        tracked = obj.observe(moved)
        assert tracked["status"] == "tracked"
        x, y, _, _ = tracked["box"]
        ref = dict(
            source_sha256="a" * 64,
            pts=i * 33333,
            time_base="1/1000000",
            source_frame_pixels_sha256=hashlib.sha256(moved.tobytes()).hexdigest(),
            upright_frame_pixels_sha256=hashlib.sha256(moved.tobytes()).hexdigest(),
            source_to_roi=[[1, 0, -x], [0, 1, -y], [0, 0, 1]],
        )
        engine.observe(
            tracked["image"],
            identity=identity,
            source_ref=ref,
            layout=layout,
            ocr=ocr,
            source_registration=tracked["registration"],
        )
    assert len(calls) == 1
    pixels, source = calls[0]
    assert source["native_crop"]["resampled"] is False
    assert (
        hashlib.sha256(pixels.tobytes()).hexdigest()
        == source["native_crop"]["native_bgr_sha256"]
    )


def test_brief_fit_failure_keeps_only_hypotheses_and_requires_new_valid_fit(
    monkeypatch,
):
    image = structure()
    obj = tracker(image)
    assert obj.observe(image)["status"] == "tracked"
    fit = obj._fit
    obj.last_reacquisition = obj.step_count
    monkeypatch.setattr(obj, "_fit", lambda *a: (None, "fixture_geometry_rejected"))
    pending = obj.observe(image)
    assert pending["status"] == "pending" and "image" not in pending
    assert obj.current_points is not None
    monkeypatch.setattr(obj, "_fit", fit)
    recovered = obj.observe(image)
    assert recovered["status"] == "tracked"
    assert recovered["registration"]["reacquired_after_loss"] is True
