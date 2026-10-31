"""Stroke/point contracts; rendered fixtures are not real-video acceptance."""

import copy

import cv2
import numpy as np
import pytest

from visioncortex.lab.native_numeric import prepare, review
from visioncortex.lab.native_segments import observe, assess, polygon_mask
from visioncortex.lab.temporal_display import SEGMENTS, SEGMENT_BOXES


def native(text, omitted=()):
    image = np.zeros((110, 200, 3), np.uint8)
    image[:, :, 0] = 50
    digits = []
    for i, glyph in enumerate(text):
        left, top, width, height = 12 + i * 85, 12, 55, 75
        for name in SEGMENTS[glyph]:
            if (i, name) in omitted:
                continue
            a, b, c, d = SEGMENT_BOXES[name]
            cv2.rectangle(
                image,
                (round(left + a * width), round(top + b * height)),
                (round(left + c * width) - 1, round(top + d * height) - 1),
                (25, 40, 240),
                -1,
            )
        digits.append(
            dict(
                box=[left / 200, top / 110, (left + width) / 200, (top + height) / 110],
                segments=copy.deepcopy(SEGMENT_BOXES),
            )
        )
    layout = dict(
        revision="fixture-cells/1",
        display_type="seven_segment",
        color="red",
        digits=digits,
        numeric_native_stroke_veto=True,
    )
    return prepare(image, layout, np.eye(3))


def test_required_missing_stroke_is_unknown_despite_confident_OCR():
    value = native("47", omitted=[(0, "g")])
    result = review(value, dict(raw_text="47", score=1.0))
    assert result["glyph_candidate"] is None
    assert result["numeric_candidate"] is None
    assert result["raw_text"] == "47"
    assert result["native_stroke_assessment"]["required_strokes_unknown"] == ["0:g"]
    assert result["confirmed"] is False


def test_actual_extra_strokes_veto_an_OCR_digit_without_replacing_it():
    value = native("97")
    result = review(value, dict(raw_text="47"))
    assert result["glyph_candidate"] is None
    assert set(
        result["native_stroke_assessment"]["positive_strokes_contradicting_OCR"]
    ) == {"0:a", "0:d"}
    assert result["raw_text"] == "47" and result["confirmed"] is False


def test_visible_required_strokes_are_not_negative_opportunity_or_confirmation():
    value = native("47")
    evidence = observe(value)
    answer = assess(evidence, "47")
    assert answer["supported"] is True
    assert answer["OFF_inferred"] is False and answer["confirmed"] is False
    assert all(
        c["state"] in {"ON", "UNKNOWN"}
        for s in evidence["slots"]
        for c in s["segments"].values()
    )
    result = review(value, dict(raw_text="47"))
    assert result["glyph_candidate"] == "47" and result["numeric_candidate"] is None
    assert result["full_glyph_correctness_proven"] is False


def test_native_prepare_preserves_explicit_registered_segment_geometry():
    value = native("47")
    assert all(
        isinstance(d, dict) and d["segments"] == SEGMENT_BOXES
        for d in value["layout"]["digits"]
    )
    assert observe(value)["geometry"]["method"] == "explicit_registered_segment_cells/1"


def test_exact_pixel_centers_do_not_include_neighbor_due_to_corner_rounding():
    q = np.float32([[1.1, 1.1], [3.1, 1.1], [3.1, 4.1], [1.1, 4.1]])
    mask = polygon_mask((6, 6), q)
    assert mask.sum() == 6
    assert not mask[:, 3].any()
    assert np.array_equal(mask, polygon_mask((6, 6), q[::-1]))


def test_bad_source_pixels_and_non_boolean_activation_are_rejected():
    value = native("47")
    changed = copy.deepcopy(value)
    changed["image"][0, 0] ^= 255
    with pytest.raises(ValueError, match="Source-pinned"):
        observe(changed)
    value["layout"]["numeric_native_stroke_veto"] = "true"
    with pytest.raises(ValueError, match="boolean"):
        review(value, dict(raw_text="47"))


def test_geometry_fit_never_mutates_input_or_uses_an_OCR_hypothesis():
    value = native("47")
    value["layout"]["digits"] = [d["box"] for d in value["layout"]["digits"]]
    before = value["image"].copy()
    evidence = observe(value)
    assert np.array_equal(value["image"], before)
    assert evidence["geometry"]["evaluated_geometries"] <= 512
    assert evidence["geometry"]["OCR_hypothesis_used"] is False
    offsets = evidence["geometry"]["parameters"][4:]
    assert max(offsets) - min(offsets) <= 0.125
    assert evidence["geometry"]["maximum_relative_slot_offset_spread"] == 0.125
    assert evidence["OFF_inferred"] is False
