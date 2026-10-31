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


@pytest.mark.parametrize("kind", ["clock", "missing_type", "without_stroke_proof", "nonboolean"])
def test_projective_points_cannot_be_activated_on_untyped_or_clock_fields(kind):
    native = payload(image())
    native["layout"].update(numeric_native_projective_decimal=True,
                            numeric_native_stroke_veto=True, data_type="number")
    if kind == "clock":
        native["layout"]["data_type"] = "clock"
    elif kind == "missing_type":
        native["layout"].pop("data_type")
    elif kind == "without_stroke_proof":
        native["layout"]["numeric_native_stroke_veto"] = False
    else:
        native["layout"]["numeric_native_projective_decimal"] = "true"
    with pytest.raises(ValueError):
        review(native, dict(raw_text="1.2"))


@pytest.mark.parametrize("mode", ["fallback", "conflict", "one_call_budget", "support_crop"])
def test_same_frame_color_view_cannot_replace_source_or_escape_call_budget(monkeypatch, mode):
    reader = NumericVideoReader(TemporalConfig())
    calls = []
    layout = dict(LAYOUT, data_type="number", numeric_native_stroke_veto=True,
                  numeric_native_color_fallback=True,
                  numeric_max_ocr_calls=1 if mode == "one_call_budget" else 2)
    if mode == "support_crop":
        layout.update(numeric_native_color_fallback=False, numeric_native_support_crop=True)

        def crop_view(native):
            view = native["image"][4:-4, 3:-3].copy()
            return view, dict(source_bgr_sha256=hashlib.sha256(native["image"].tobytes()).hexdigest(),
                              prepared_bgr_sha256=hashlib.sha256(view.tobytes()).hexdigest(),
                              native_to_prepared=[[1, 0, -3], [0, 1, -4], [0, 0, 1]])

        monkeypatch.setattr("visioncortex.lab.native_views.support_crop", crop_view)

    def ocr(pixels, *, anchor_source):
        calls.append((pixels.copy(), anchor_source))
        return dict(raw_text="12" if len(calls) == 1 else "13")

    def assess(native, reply):
        assert np.array_equal(native["image"], image())
        text = reply.get("raw_text", "")
        glyph = text if mode == "conflict" or text == "13" else None
        return dict(raw_text=text, glyph_candidate=glyph, numeric_candidate=None,
                    rejection_reasons=[], confirmed=False, value_text=None)

    monkeypatch.setattr("visioncortex.lab.native_numeric.review", assess)
    for i in range(6):
        native = prepare(image(), layout, np.eye(3))
        native["source"] = source(native, ref(i), ref(i))
        out = reader.observe(image(), key=tuple(IDENTITY.values()), identity=IDENTITY,
                             source_ref=ref(i), layout=layout, ocr=ocr, native=native)
    assert len(calls) == (1 if mode == "one_call_budget" else 2)
    assert out["ocr_port_calls"] == len(calls)
    if len(calls) == 2:
        assert not np.array_equal(calls[0][0], calls[1][0])
        for key in ("pts", "source_sha256", "source_frame_pixels_sha256"):
            assert calls[0][1][key] == calls[1][1][key]
        assert "ocr_view" in calls[1][1]
        if mode == "support_crop":
            assert calls[1][1]["source_to_roi"] == [[1, 0, -3], [0, 1, -4], [0, 0, 1]]
    if mode == "conflict":
        assert out["glyph_candidate"] is None
        assert "supported_same_frame_views_conflict" in out["rejection_reasons"]
    elif mode in {"fallback", "support_crop"}:
        assert out["glyph_candidate"] == "13"
    assert out["numeric_candidate"] is None and not out["gates"]["confirmed"]


@pytest.mark.parametrize("points", ["one", "absent", "ambiguous", "point_with_missing_stroke"])
def test_point_selection_uses_one_actual_frame_and_cannot_borrow_a_dot(monkeypatch, points):
    reader = NumericVideoReader(TemporalConfig())
    layout = dict(LAYOUT, data_type="number", numeric_native_stroke_veto=True,
                  numeric_native_projective_decimal=True, numeric_native_point_selection=True)
    calls = []

    def probe(native):
        found = [dict(after_digit=0, box=[40, 30, 42, 32])] if native["source"]["pts"] == 33333 else []
        if points == "point_with_missing_stroke" and native["source"]["pts"] == 66666:
            found = [dict(after_digit=0, box=[40, 30, 42, 32])]
        if points == "absent":
            found = []
        elif points == "ambiguous":
            found = found * 2
        return dict(points=found, positive_shape_consistent=not (
            points == "point_with_missing_stroke" and native["source"]["pts"] == 33333))

    def ocr(pixels, *, anchor_source):
        calls.append((pixels.copy(), anchor_source))
        return dict(raw_text="12")

    monkeypatch.setattr("visioncortex.lab.native_numeric.point_probe", probe)
    monkeypatch.setattr("visioncortex.lab.native_numeric.review", lambda native, reply: dict(
        raw_text=reply["raw_text"], glyph_candidate="12", numeric_candidate=None,
        rejection_reasons=[], confirmed=False, value_text=None))
    for i in range(3):
        im = image()
        im[im[:, :, 2] > 0, 2] -= i * 10
        native = prepare(im, layout, np.eye(3))
        native["source"] = source(native, ref(i, im), ref(i, im))
        out = reader.observe(im, key=tuple(IDENTITY.values()), identity=IDENTITY,
                             source_ref=ref(i, im), layout=layout, ocr=ocr, native=native)
    selected = 33333 if points == "one" else 66666 if points == "point_with_missing_stroke" else 0
    assert len(calls) == 1 and calls[0][1]["pts"] == selected
    assert out["source_refs"][0]["pts"] == selected
    assert out["numeric_candidate"] is None and out["fused_digit_pixels"] is False
    assert len(out["native_frame_selection"]["probes"]) == 3


def test_later_observation_cannot_keep_choosing_an_old_brighter_value(monkeypatch):
    reader = NumericVideoReader(TemporalConfig())
    layout = dict(LAYOUT, data_type="number", numeric_native_stroke_veto=True,
                  numeric_native_projective_decimal=True, numeric_native_point_selection=True,
                  numeric_max_ocr_calls=2)
    calls = []
    monkeypatch.setattr("visioncortex.lab.native_numeric.point_probe", lambda native: dict(
        points=[dict(after_digit=0, box=[40, 30, 42, 32])], positive_shape_consistent=True))
    monkeypatch.setattr("visioncortex.lab.native_numeric.review", lambda native, reply: dict(
        raw_text=reply["raw_text"], glyph_candidate=reply["raw_text"], numeric_candidate=None,
        rejection_reasons=[], confirmed=False, value_text=None))

    def ocr(pixels, *, anchor_source):
        text = "12" if pixels[10, 20, 2] > 200 else "34"
        calls.append((text, anchor_source))
        return dict(raw_text=text)

    for i in range(40):
        im = image()
        if i >= 10:
            im[im[:, :, 2] > 0, 2] = 180
        native = prepare(im, layout, np.eye(3))
        native["source"] = source(native, ref(i, im), ref(i, im))
        out = reader.observe(im, key=tuple(IDENTITY.values()), identity=IDENTITY,
                             source_ref=ref(i, im), layout=layout, ocr=ocr, native=native)
    assert len(calls) == 2 and [c[0] for c in calls] == ["12", "34"]
    assert out["raw_text"] == "34" and out["source_refs"][0]["pts"] >= 750000
    assert out["native_frame_selection"]["selected_anchor_age_seconds"] <= .25


@pytest.mark.parametrize("mode", ["enabled", "disabled", "duplicate", "unstable", "cooldown", "budget", "reset", "context"])
def test_debounced_shape_change_reads_fresh_pixels_without_borrowing_an_old_point(monkeypatch, mode):
    reader = NumericVideoReader(TemporalConfig())
    layout = dict(LAYOUT, data_type="number", numeric_native_stroke_veto=True,
                  numeric_native_projective_decimal=True, numeric_native_point_selection=True,
                  numeric_native_change_trigger=mode != "disabled",
                  numeric_max_ocr_calls=1 if mode == "budget" else 4)
    calls = []

    def probe(native):
        old = native["image"][10, 20, 2] > 200
        signature = "a" if old else "c" if native["image"][10, 20, 2] < 160 else "b"
        if mode == "unstable" and not old:
            signature = str(native["source"]["pts"])
        return dict(points=[dict(after_digit=0)] if old else [], positive_shape_consistent=True,
                    positive_shape_signature=hashlib.sha256(signature.encode()).hexdigest())

    def ocr(pixels, *, anchor_source):
        text = "12" if pixels[10, 20, 2] > 200 else "34" if pixels[10, 20, 2] > 160 else "56"
        calls.append((text, anchor_source))
        return dict(raw_text=text)

    monkeypatch.setattr("visioncortex.lab.native_numeric.point_probe", probe)
    monkeypatch.setattr("visioncortex.lab.native_numeric.review", lambda native, reply: dict(
        raw_text=reply["raw_text"], glyph_candidate=reply["raw_text"], numeric_candidate=None,
        rejection_reasons=["fixture_decimal_unknown"], confirmed=False, value_text=None))
    for i in range(22):
        im = image()
        if mode != "duplicate":
            im[0, 0, 0] = i
        if i >= 6:
            im[im[:, :, 2] > 0, 2] = 150 if mode == "cooldown" and i >= 12 else 180
        if mode == "reset" and i == 12:
            reader.reset(tuple(IDENTITY.values()))
        raw_ref = ref(i, im)
        if mode == "context":
            raw_ref["dynamic_physical_key"] = "old" if i < 6 else "new"
        native = prepare(im, layout, np.eye(3))
        native["source"] = source(native, raw_ref, raw_ref)
        out = reader.observe(im, key=tuple(IDENTITY.values()), identity=IDENTITY,
                             source_ref=raw_ref, layout=layout, ocr=ocr, native=native)
    extra = mode in {"enabled", "cooldown", "reset", "context"}
    assert [item[0] for item in calls] == (["12", "34"] if extra else ["12"])
    if extra:
        assert calls[1][1]["pts"] >= 6*33333 and calls[1][1]["pts"] < 1_000_000
        if mode != "reset":
            assert out["native_frame_selection"]["trigger"] == (
                "periodic" if mode == "context" else "positive_shape_change")
            assert out["numeric_candidate"] is None and out["gates"]["confirmed"] is False
    if mode == "context":
        assert out["raw_text"] == "34" and out["source_refs"][0]["dynamic_physical_key"] == "new"


def test_change_trigger_requires_explicit_native_point_selection():
    reader = NumericVideoReader(TemporalConfig())
    im = image()
    layout = dict(LAYOUT, numeric_native_change_trigger=True)
    native = prepare(im, layout, np.eye(3))
    native["source"] = source(native, ref(0, im), ref(0, im))
    with pytest.raises(ValueError, match="explicit native point selection"):
        reader.observe(im, key=tuple(IDENTITY.values()), identity=IDENTITY,
                       source_ref=ref(0, im), layout=layout, ocr=None, native=native)


@pytest.mark.parametrize("early_point", [True, False])
def test_optional_initial_wait_chooses_current_point_or_expires_without_spending_budget(monkeypatch, early_point):
    reader = NumericVideoReader(TemporalConfig())
    layout = dict(LAYOUT, data_type="number", numeric_native_stroke_veto=True,
                  numeric_native_projective_decimal=True, numeric_native_point_selection=True,
                  numeric_initial_quality_wait_seconds=".2", numeric_max_ocr_calls=1)
    calls = []
    monkeypatch.setattr("visioncortex.lab.native_numeric.point_probe", lambda native: dict(
        points=[dict(after_digit=0)] if early_point and native["source"]["pts"] >= 99999 else [],
        positive_shape_consistent=True))
    monkeypatch.setattr("visioncortex.lab.native_numeric.review", lambda native, reply: dict(
        raw_text=reply["raw_text"], glyph_candidate="12", numeric_candidate=None,
        rejection_reasons=[], confirmed=False, value_text=None))

    def ocr(pixels, *, anchor_source):
        calls.append((i, anchor_source))
        return dict(raw_text="12")

    for i in range(10):
        im = image()
        im[im[:, :, 2] > 0, 2] -= i*5
        native = prepare(im, layout, np.eye(3))
        native["source"] = source(native, ref(i, im), ref(i, im))
        out = reader.observe(im, key=tuple(IDENTITY.values()), identity=IDENTITY,
                             source_ref=ref(i, im), layout=layout, ocr=ocr, native=native)
        if i == 2:
            assert calls == [] and out["ocr_port_calls"] == 0
            assert out["rejection_reasons"] == ["waiting_for_source_quality"]
    assert len(calls) == 1 and calls[0][0] == (3 if early_point else 7)
    if early_point:
        assert calls[0][1]["pts"] == 99999


@pytest.mark.parametrize("wait", [-1, ".3", True, "nan", "9"*20])
def test_initial_quality_wait_rejects_unbounded_or_invalid_configuration(wait):
    reader = NumericVideoReader(TemporalConfig())
    im = image()
    layout = dict(LAYOUT, numeric_initial_quality_wait_seconds=wait)
    native = prepare(im, layout, np.eye(3))
    native["source"] = source(native, ref(0, im), ref(0, im))
    with pytest.raises(ValueError):
        reader.observe(im, key=tuple(IDENTITY.values()), identity=IDENTITY,
                       source_ref=ref(0, im), layout=layout, ocr=None, native=native)


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
