"""Geometry/absence/budget contracts; fixtures are not real OCR quality gold."""

import copy
import hashlib

import numpy as np
import pytest

from visioncortex.lab.native_numeric import emission, prepare, review, source
from visioncortex.lab.temporal_display import TemporalConfig, TemporalDisplay
from visioncortex.lab.video_numeric import NumericVideoReader


LAYOUT = dict(
    revision="fixture-native/1",
    color="red",
    display_type="seven_segment",
    digits=[[0.1, 0.15, 0.35, 0.85], [0.6, 0.15, 0.85, 0.85]],
    numeric_anchor=True,
    numeric_native_ocr=True,
    numeric_field_box=[0, 0, 1, 1],
)
IDENTITY = dict(
    device_id="fixture", screen_id="screen", field_id="PV", track_session_id="track"
)


def image():
    a = np.zeros((40, 100, 3), np.uint8)
    a[:, :, 0] = 100  # A blue casing is not red-digit emission.
    a[8:32, 14:30, 2] = 220
    a[8:32, 64:80, 2] = 220
    return a


def ref(i=0, im=None):
    return dict(
        source_sha256="a" * 64,
        pts=i * 33333,
        time_base="1/1000000",
        source_frame_pixels_sha256=hashlib.sha256(
            (image() if im is None else im).tobytes()
        ).hexdigest(),
        source_to_roi=np.eye(3).tolist(),
    )


def payload(im, i=0):
    native = prepare(im, LAYOUT, np.eye(3))
    native["source"] = source(native, ref(i, im), ref(i, im))
    return native


def test_native_crop_has_exact_pixels_and_exact_projected_source_transform():
    im = np.random.default_rng(1).integers(0, 255, (60, 100, 3), dtype=np.uint8)
    layout = dict(
        LAYOUT,
        numeric_field_box=[0.2, 0.2, 0.8, 0.8],
        digits=[[0.25, 0.3, 0.4, 0.7], [0.6, 0.3, 0.75, 0.7]],
    )
    H = np.array([[1, 0, -3], [0, 1, -2], [0, 0, 1]], float)
    native = prepare(im, layout, H)
    x, y, xx, yy = native["metadata"]["crop_box"]
    assert np.array_equal(native["image"], im[y:yy, x:xx])
    assert native["metadata"]["resampled"] is False
    raw = ref(im=im)
    aligned = dict(raw, source_to_roi=H.tolist())
    derived = source(native, raw, aligned)
    assert derived["source_to_roi"] == [
        [1.0, 0.0, -float(x)],
        [0.0, 1.0, -float(y)],
        [0.0, 0.0, 1.0],
    ]
    assert derived["registered_source_to_roi"] == H.tolist()


@pytest.mark.parametrize(
    "change", ["clock", "pixels", "missing_pixels", "other_panel", "physical_context"]
)
def test_different_or_unpinned_original_frames_cannot_supply_native_pixels(change):
    native = payload(image())
    raw = ref()
    aligned = copy.deepcopy(raw)
    if change == "clock":
        aligned["pts"] += 1
    elif change == "pixels":
        aligned["source_frame_pixels_sha256"] = "b" * 64
    elif change == "missing_pixels":
        raw.pop("source_frame_pixels_sha256")
        aligned.pop("source_frame_pixels_sha256")
    elif change == "other_panel":
        aligned["source_to_roi"][0][2] = -50
    else:
        aligned["dynamic_physical_key"] = "other-equipment-epoch"
    with pytest.raises(ValueError):
        source(native, raw, aligned)


def test_color_background_and_dark_slot_are_not_completed_using_mock_tokens():
    im = image()
    im[:, 60:90, 2] = 0
    native = payload(im)
    assert emission(native)[0]["visible"] is True
    assert emission(native)[1]["visible"] is False
    for text in ["12", "1", "1.2"]:
        result = review(native, dict(raw_text=text, score=0.9999))
        assert result["glyph_candidate"] is None
        assert result["numeric_candidate"] is None and result["confirmed"] is False
        assert "native_digit_slot_not_emitting" in result["rejection_reasons"]


def test_reading_digits_and_model_dot_remain_candidates_without_point_evidence():
    native = payload(image())
    for text in ["12", "1.2"]:
        result = review(native, dict(raw_text=text))
        assert result["glyph_candidate"] == "12" and result["raw_text"] == text
        assert result["numeric_candidate"] is None and result["value_text"] is None
        assert result["full_glyph_correctness_proven"] is False
    assert review(native, dict(raw_text="1 2"))["glyph_candidate"] is None


def test_native_reader_uses_one_real_anchor_and_counts_both_pixel_buffers():
    reader = NumericVideoReader(TemporalConfig())
    calls = []

    def ocr(pixels, *, anchor_source):
        calls.append((pixels.copy(), anchor_source))
        return dict(raw_text="12", score=0.999)

    for i in range(16):
        im = image()
        native = payload(im, i)
        out = reader.observe(
            im,
            key=tuple(IDENTITY.values()),
            identity=IDENTITY,
            source_ref=ref(i, im),
            layout=LAYOUT,
            ocr=ocr,
            native=native,
        )
        native["image"][:] = 0  # Caller mutation must not alter cached evidence.
    assert len(calls) == 1 and out["glyph_candidate"] == "12"
    assert np.array_equal(calls[0][0], image())
    assert calls[0][1] == out["source_refs"][0]
    assert out["buffer_bytes"] == 16 * image().nbytes * 2
    assert out["numeric_candidate"] is None and out["gates"]["confirmed"] is False
    assert out["preprocessing"]["resampled"] is False
    assert out["fused_digit_pixels"] is False and out["selected_span_seconds"] == 0


def test_native_byte_budget_and_source_reset_do_not_leak_previous_readings():
    im = image()
    native = payload(im)
    tiny = NumericVideoReader(TemporalConfig(max_buffer_bytes=im.nbytes))
    with pytest.raises(ValueError, match="byte budget"):
        tiny.observe(
            im,
            key=tuple(IDENTITY.values()),
            identity=IDENTITY,
            source_ref=ref(),
            layout=LAYOUT,
            ocr=None,
            native=native,
        )
    assert tiny.buffer_bytes == 0
    reader = NumericVideoReader(TemporalConfig())
    for i in range(16):
        result = reader.observe(
            im,
            key=tuple(IDENTITY.values()),
            identity=IDENTITY,
            source_ref=ref(i),
            layout=LAYOUT,
            ocr=lambda *_args, **_kw: dict(raw_text="12"),
            native=payload(im, i),
        )
    newer = payload(im, 16)
    newer["source"]["source_sha256"] = "b" * 64
    result = reader.observe(
        im,
        key=tuple(IDENTITY.values()),
        identity=IDENTITY,
        source_ref=dict(ref(16), source_sha256="b" * 64),
        layout=LAYOUT,
        ocr=None,
        native=newer,
    )
    assert result["input_frame_count"] == 1 and result["candidates"] == []
    assert result["ocr_port_calls"] == 0


def test_short_valid_track_can_produce_a_candidate_but_resets_do_not_refill_calls():
    reader = NumericVideoReader(TemporalConfig())
    layout = dict(LAYOUT, numeric_max_ocr_calls=1)
    calls = []

    def ocr(pixels, **kwargs):
        calls.append(kwargs["anchor_source"])
        return dict(raw_text="12")

    for i in range(30):
        if i % 3 == 0:
            reader.reset(tuple(IDENTITY.values()))
        result = reader.observe(
            image(),
            key=tuple(IDENTITY.values()),
            identity=IDENTITY,
            source_ref=ref(i),
            layout=layout,
            ocr=ocr,
            native=payload(image(), i),
        )
        if i == 2:
            assert result["glyph_candidate"] == "12"
            assert result["selected_span_seconds"] == 0
    assert len(calls) == 1 and result["ocr_port_calls"] == 1


def test_existing_display_entry_carries_native_crop_to_the_existing_ocr_port():
    engine = TemporalDisplay(TemporalConfig())
    calls = []

    def ocr(pixels, **kwargs):
        calls.append((pixels.copy(), kwargs["anchor_source"]))
        return dict(raw_text="12")

    # Identity registration isolates port behavior from unrelated feature tests.
    class Registrar:
        memory_bytes = 0

        def register(self, im):
            return im.copy(), dict(
                accepted=True, matrix_current_to_reference=np.eye(3).tolist()
            )

    for i in range(16):
        output = engine.observe(
            image(), identity=IDENTITY, source_ref=ref(i), layout=LAYOUT, ocr=ocr
        )
        for state in engine.states.values():
            state.registrar = Registrar()
    assert len(calls) == 1 and np.array_equal(calls[0][0], image())
    assert calls[0][1]["native_crop"]["resampled"] is False
    assert output["reading_status"] == "glyphs_only" and output["value_text"] is None
