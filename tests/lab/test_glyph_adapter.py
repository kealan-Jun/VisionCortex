"""Contract fixtures only; random textures are not business OCR truth."""

import copy

import numpy as np
import pytest

from visioncortex.lab.glyph_adapter import calibrate, recognize, reconcile
from visioncortex.lab.temporal_display import TemporalConfig
from visioncortex.lab.video_numeric import NumericVideoReader


def ref(i=0):
    return dict(
        source_sha256="a" * 64,
        pts=i * 33333,
        time_base="1/1000000",
        source_to_roi=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )


def texture(seed):
    return np.random.default_rng(seed).integers(0, 255, (24, 16, 3), dtype=np.uint8)


def fixture_adapter():
    examples = [
        dict(
            label=label,
            label_basis="reviewed_original_glyph",
            image=texture(i),
            box=[0, 0, 1, 1],
            source_ref=ref(i),
        )
        for i, label in [(1, "7"), (2, "1")]
    ]
    return calibrate(examples, revision="fixture-random-not-real-OCR/1")


def test_unknown_and_degenerate_pixels_are_not_forced_into_known_classes():
    adapter = fixture_adapter()
    known = recognize(texture(1), dict(digits=[[0, 0, 1, 1]]), adapter)
    assert known["glyph_pattern"] == "7" and known["value_text"] is None
    for image in [texture(90), np.full((24, 16, 3), 100, np.uint8)]:
        result = recognize(image, dict(digits=[[0, 0, 1, 1]]), adapter)
        assert result["glyph_pattern"] == "?" and not result["complete"]
    assert known["confirmed"] is False and known["decimal_status"] == "unknown"


def test_repeated_digits_are_distinct_slots_in_one_source_frame():
    adapter = fixture_adapter()
    image = np.concatenate([texture(1), texture(1)], axis=1)
    result = recognize(image, dict(digits=[[0, 0, 0.5, 1], [0.5, 0, 1, 1]]), adapter)
    assert result["glyph_pattern"] == "77"
    assert result["shared_source_shift"] == [0, 0]
    assert result["value_text"] is None
    with pytest.raises(ValueError, match="overlap"):
        recognize(image, dict(digits=[[0, 0, 0.6, 1], [0.5, 0, 1, 1]]), adapter)


@pytest.mark.parametrize("change", ["label", "gate", "feature", "clock"])
def test_adapter_rejects_whole_readings_weakened_gates_and_bad_provenance(change):
    adapter = fixture_adapter()
    if change == "label":
        adapter["prototypes"][0]["label"] = "1234"
    elif change == "gate":
        adapter["minimum_similarity"] = 0.1
    elif change == "feature":
        adapter["prototypes"][0]["feature_b64"] = ""
    else:
        adapter["prototypes"][0]["source_ref"]["source_sha256"] = "unknown"
    with pytest.raises(ValueError):
        recognize(texture(1), dict(digits=[[0, 0, 1, 1]]), adapter)


def test_existing_numeric_reader_stores_glyphs_and_exact_anchor_without_numeric_fact():
    adapter = fixture_adapter()
    layout = dict(
        revision="fixture/1",
        digits=[[0, 0, 0.5, 1], [0.5, 0, 1, 1]],
        glyph_adapter=adapter,
    )
    image = np.concatenate([texture(1), texture(1)], axis=1)
    identity = dict(
        device_id="unbound-fixture",
        screen_id="panel",
        field_id="PV",
        track_session_id="track",
    )
    reader = NumericVideoReader(TemporalConfig())
    for i in range(16):
        result = reader.observe(
            image,
            identity=identity,
            key=tuple(identity.values()),
            source_ref=ref(i),
            layout=layout,
            ocr=None,
        )
    assert result["glyph_candidate"] == "77"
    assert result["reading_status"] == "glyphs_only"
    assert result["numeric_candidate"] is None and result["value_text"] is None
    assert result["selected_frame_count"] == 1 and result["fused_digit_pixels"] is False
    assert result["ocr_port_calls"] == 0 and result["gates"]["confirmed"] is False
    changed = copy.deepcopy(ref(17))
    changed["source_sha256"] = "b" * 64
    result = reader.observe(
        image,
        identity=identity,
        key=tuple(identity.values()),
        source_ref=changed,
        layout=layout,
        ocr=None,
    )
    assert result["input_frame_count"] == 1 and result["numeric_candidate"] is None


def test_same_frame_suggestion_does_not_fill_unknown_strokes_or_cross_field_time():
    identity = dict(
        device_id="fixture", screen_id="screen", field_id="PV", track_session_id="track"
    )
    source = dict(ref(), source_frame_pixels_sha256="b" * 64)
    result = dict(
        glyph_pattern="12?0", glyphs=[dict(glyph=g) for g in ["1", "2", None, "0"]]
    )
    observation = dict(identity=identity, source_ref=source, result=result)
    other = dict(identity=identity, source_ref=source, raw_text="1970")
    output = reconcile(observation, other)
    assert output["suggestion"] == "1270" and output["glyph_pattern"] == "12?0"
    assert output["unverified_positions"] == [2] and output["numeric_candidate"] is None
    assert output["confirmed"] is False
    short = dict(other, raw_text="083")
    assert reconcile(observation, short)["suggestion"] is None
    for altered in [
        dict(other, identity=dict(identity, field_id="SV")),
        dict(other, source_ref=dict(source, pts=33333)),
    ]:
        with pytest.raises(ValueError, match="same original frame"):
            reconcile(observation, altered)


def test_optional_ocr_reconcile_runs_one_port_call_on_selected_source():
    import hashlib

    image = np.concatenate([texture(1), texture(1)], axis=1)
    layout = dict(
        revision="fixture/2",
        digits=[[0, 0, 0.5, 1], [0.5, 0, 1, 1]],
        glyph_adapter=fixture_adapter(),
        glyph_ocr_reconcile=True,
    )
    identity = dict(
        device_id="fixture", screen_id="panel", field_id="PV", track_session_id="track"
    )
    reader = NumericVideoReader(TemporalConfig())
    calls = []

    def ocr(pixels, *, anchor_source):
        calls.append(anchor_source)
        return dict(raw_text="77", score=0.99)

    for i in range(16):
        source = dict(
            ref(i),
            source_frame_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        )
        result = reader.observe(
            image,
            key=tuple(identity.values()),
            identity=identity,
            source_ref=source,
            layout=layout,
            ocr=ocr,
        )
    assert len(calls) == 1 and calls[0] == result["source_refs"][0]
    assert result["ocr_reconciliation"]["suggestion"] == "77"
    assert result["numeric_candidate"] is None and result["gates"]["confirmed"] is False
