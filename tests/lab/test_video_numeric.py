"""Safety/contract tests only; mock tokens and texture are not OCR truth."""

import copy

import numpy as np
import pytest

from visioncortex.lab.temporal_display import TemporalConfig
from visioncortex.lab.video_numeric import NumericVideoReader, number, review_anchors


IDENTITY = dict(
    device_id="scope-a",
    screen_id="panel-a",
    field_id="field-a",
    track_session_id="track-a",
)
LAYOUT = dict(
    revision="controlled-test-only/1",
    color=None,
    digits=[[0.05, 0.1, 0.3, 0.9], [0.35, 0.1, 0.6, 0.9], [0.65, 0.1, 0.9, 0.9]],
)


def reference(i, sha="a" * 64):
    return dict(
        pts=i * 33333,
        time_base="1/1000000",
        source_sha256=sha,
        source_to_roi=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )


def frame(i):
    # A texture exercises bounds/port behavior, not a rendered reading.
    return np.random.default_rng(i).integers(0, 255, (40, 120, 3), dtype=np.uint8)


def observe(reader, i, ocr=None, identity=None, sha="a" * 64):
    identity = identity or IDENTITY
    return reader.observe(
        frame(i),
        key=tuple(identity.values()),
        identity=identity,
        source_ref=reference(i, sha),
        layout=LAYOUT,
        ocr=ocr,
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.200", "1.200"),
        ("001", "001"),
        (".12", ".12"),
        ("123.", "123."),
        ("1 20", None),
        ("1.2 3.4", None),
        ("1.2.3", None),
        ("1,200", None),
        ("12:32", None),
        ("设备4", None),
    ],
)
def test_one_token_never_joins_numbers_or_moves_decimal(text, expected):
    assert number(text) == expected


@pytest.mark.parametrize(
    "tokens,reason",
    [
        (["1.20", "1.20"], "numeric_candidate_not_independent_glyph_truth"),
        (["1.20", "3.20"], "same_frame_OCR_views_disagree_on_digits"),
        (["1.20", "12.0"], "decimal_position_conflict"),
    ],
)
def test_ocr_views_are_same_source_candidates_not_fused_or_confirmed(tokens, reason):
    replies = iter(tokens)
    reader = NumericVideoReader(TemporalConfig())
    for i in range(16):
        result = observe(
            reader, i, ocr=lambda _, **kwargs: dict(raw_text=next(replies), score=0.999)
        )
    assert result["value_text"] is None and result["status"] == "uncertain"
    assert reason in result["rejection_reasons"]
    assert result["fused_digit_pixels"] is False
    assert len(result["source_refs"]) == 1 and result["selected_frame_count"] == 1
    assert result["selected_span_seconds"] == 0
    assert all(v["source_ref"] == result["source_refs"][0] for v in result["ocr_views"])
    if tokens[0] != tokens[1]:
        assert result["numeric_candidate"] is None


def test_exact_panel_key_required_and_other_fields_have_independent_buffers():
    reader = NumericVideoReader(TemporalConfig())
    observe(reader, 0)
    other = dict(IDENTITY, field_id="field-b")
    observe(reader, 0, identity=other)
    assert len(reader.states) == 2
    with pytest.raises(ValueError, match="cache key"):
        reader.observe(
            frame(1),
            key=tuple(IDENTITY.values()),
            identity=other,
            source_ref=reference(1),
            layout=LAYOUT,
            ocr=None,
        )


@pytest.mark.parametrize("change", ["source", "gap", "reverse"])
def test_source_change_gap_reverse_clock_cannot_reuse_prior_reading(change):
    reader = NumericVideoReader(TemporalConfig())
    for i in range(16):
        observe(reader, i, ocr=lambda _, **kwargs: dict(raw_text="120", score=0.999))
    result = observe(
        reader,
        0 if change == "reverse" else (40 if change == "gap" else 16),
        sha="b" * 64 if change == "source" else "a" * 64,
    )
    assert result["numeric_candidate"] is None
    assert result["input_frame_count"] == 1
    assert result["ocr_port_calls"] == 0
    assert result["reset_reasons"] == ["source_layout_clock_or_gap_changed"]


def test_bad_source_reference_and_oversized_ring_fail_before_ocr():
    reader = NumericVideoReader(TemporalConfig())
    bad = copy.deepcopy(reference(0))
    bad["source_to_roi"] = [[0, 0, 0]] * 3
    with pytest.raises(ValueError, match="invertible"):
        reader.observe(
            frame(0),
            key=tuple(IDENTITY.values()),
            identity=IDENTITY,
            source_ref=bad,
            layout=LAYOUT,
            ocr=None,
        )
    reader = NumericVideoReader(TemporalConfig(max_buffer_bytes=1024))
    with pytest.raises(ValueError, match="byte budget"):
        observe(reader, 0)
    assert reader.buffer_bytes == 0


def anchor(text, i=0, field="field-a", **extra):
    return dict(
        identity=dict(IDENTITY, field_id=field),
        source_ref=reference(i),
        raw_texts=[text],
        window_id="continuous-source-window",
        layout_revision="reviewed-geometry/1",
        registration_verified=True,
        **extra,
    )


def test_short_high_score_token_is_vetoed_never_replaced_with_later_value():
    rows = review_anchors([anchor("40", i=0), anchor("140", i=9)])
    assert rows[0]["numeric_candidate"] is None
    assert rows[0]["raw_text"] == "40"
    assert rows[0]["source_refs"][0]["pts"] == 0
    assert rows[1]["numeric_candidate"] == "140"
    assert rows[1]["source_refs"][0]["pts"] == 9 * 33333
    assert all(r["gates"]["confirmed"] is False for r in rows)


def test_separate_fields_and_value_changes_do_not_vote_or_splice():
    rows = review_anchors(
        [anchor("2"), anchor("5", i=1), anchor("140", i=2, field="other")]
    )
    assert [r["numeric_candidate"] for r in rows] == ["2", "5", "140"]
    assert all(
        r["selected_span_seconds"] == 0 and r["fused_digit_pixels"] is False
        for r in rows
    )


@pytest.mark.parametrize(
    "text,extra,reason",
    [
        ("00", dict(clipped=True), "source_field_clipped"),
        ("000", dict(decimal_required=True), "decimal_not_observed"),
        (
            "40",
            dict(expected_digit_slots=3),
            "token_does_not_cover_declared_digit_slots",
        ),
    ],
)
def test_counterexamples_cannot_be_business_readings(text, extra, reason):
    result = review_anchors([anchor(text, **extra)])[0]
    assert result["numeric_candidate"] is None and reason in result["rejection_reasons"]


def test_clock_and_disagreeing_same_source_views():
    a = anchor("96")
    a["raw_texts"] = ["96", "46"]
    rows = review_anchors([a, anchor("12:32", field="timer")])
    assert rows[0]["display_text"] is None
    assert rows[1]["display_text"] == "12:32" and rows[1]["numeric_candidate"] is None
    assert rows[1]["display_kind"] == "clock"


def test_windows_are_bounded_and_unregistered_fields_remain_unbound_candidates():
    with pytest.raises(ValueError, match="one second"):
        review_anchors([anchor("1"), anchor("2", i=40)])
    a = anchor("25")
    a["registration_verified"] = False
    assert review_anchors([a])[0]["numeric_candidate"] is None


def test_selected_anchor_receipt_uses_original_pts_not_callback_trigger():
    reader = NumericVideoReader(TemporalConfig())
    seen = []

    def ocr(pixels, *, anchor_source):
        seen.append(anchor_source)
        return dict(raw_text="1.20", score=0.99)

    for i in range(16):
        result = observe(reader, i, ocr=ocr)
    assert seen and all(s == result["source_refs"][0] for s in seen)


def test_existing_temporal_entry_uses_opt_in_reader_and_preserves_anchor_receipt():
    from visioncortex.lab.temporal_display import TemporalDisplay

    engine = TemporalDisplay()
    layout = dict(LAYOUT, display_type="seven_segment", numeric_anchor=True)
    pixels = frame(91)
    seen = []

    def ocr(image, *, anchor_source):
        seen.append(anchor_source)
        return dict(raw_text="1.20", score=0.99)

    for i in range(16):
        result = engine.observe(
            pixels, identity=IDENTITY, source_ref=reference(i), layout=layout, ocr=ocr
        )
    assert seen and result["method"] == "vc-video-numeric-anchor/3"
    assert result["numeric_candidate"] == "1.20"
    assert result["status"] == "uncertain" and result["gates"]["confirmed"] is False
    assert all(s == result["source_refs"][0] for s in seen)
    assert engine.buffer_bytes <= engine.config.max_buffer_bytes
    engine.reset(IDENTITY)
    assert engine.buffer_bytes == 0


def test_existing_event_projection_displays_candidate_without_business_confirmation():
    from visioncortex.lab.temporal_video import TemporalVideo

    original_ref = dict(
        reference(3), media_id="fixture-original", media_offset_seconds="0.1"
    )
    observation = anchor("001.20")
    observation["source_ref"] = original_ref
    result = review_anchors([observation])[0]
    event = TemporalVideo._event(
        dict(
            result=result,
            source_ref=original_ref,
            field=dict(
                field_key="PV",
                role="process_value",
                screen_id="panel-a",
                profile_id="layout/1",
            ),
        ),
        dict(
            media_id="fixture-original",
            sha256="a" * 64,
            start_seconds="0",
            end_seconds="1",
        ),
        dict(revision="fixture/1", world_clock_status="uncalibrated"),
    )
    assert "001.20" in event["text"] and event["instrument_id"] is None
    assert event["fields"][0]["value"] is None
    assert event["fields"][0]["tentative_value"] == "001.20"
    assert event["payload"]["business_confirmation"] is False
    assert event["spans"][0]["start_seconds"] == "0.100000"


def test_decimal_point_requires_native_gap_pixels_not_vertical_stroke_or_model_only():
    from visioncortex.lab.video_numeric import resolve_decimal_tokens

    image = np.zeros((61, 105, 3), np.uint8)
    boxes = [[8, 7, 34, 54], [39, 7, 64, 54], [68, 7, 96, 54]]
    layout = dict(
        color="red",
        digits=[[x / 105, y / 61, xx / 105, yy / 61] for x, y, xx, yy in boxes],
    )
    image[33:44, 30:33] = [60, 40, 240]  # lower glyph stroke, outside the gap
    absent = resolve_decimal_tokens(image, layout, ["0.00", "000"])
    assert absent["numeric_candidate"] is None
    image[46:49, 35:38] = [60, 40, 240]
    result = resolve_decimal_tokens(image, layout, ["0.00", "000"])
    assert result["numeric_candidate"] == "0.00"
    assert result["native_point_evidence"][0]["after_digit"] == 0
    assert result["confirmed"] is False
    conflict = resolve_decimal_tokens(image, layout, ["00.0", "000"])
    assert conflict["numeric_candidate"] is None
    different_digits = resolve_decimal_tokens(image, layout, ["0.00", "010"])
    assert different_digits["numeric_candidate"] is None
