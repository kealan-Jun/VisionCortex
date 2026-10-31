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


@pytest.mark.parametrize("scale,accepted", [(.6, True), (.1, False)])
def test_field_motion_uses_registered_source_resolution_without_weakening_matches(scale, accepted):
    reference = structure()
    boxes = [[.25, .3, .45, .75], [.55, .3, .75, .75]]
    kwargs = dict(required_boxes=[[.2, .2, .8, .8]])
    obj = SourcePanelTracker(reference, [.2, .2, .65, .8], [[.3, .3, .7, .6]],
                             motion_digit_boxes=boxes, **kwargs)
    legacy = SourcePanelTracker(reference, [.2, .2, .65, .8], [[.3, .3, .7, .6]], **kwargs)
    image = cv2.warpAffine(reference, np.float32([[scale, 0, 10], [0, scale, 10]]), (360, 240))
    points = np.float32([[x, y] for y in [60, 100, 180] for x in [80, 140, 220]])
    current = points * scale + [10, 10]
    result, reason = obj._fit(current.astype(np.float32), points, image, "fixture")
    assert (result is not None) is accepted
    if accepted:
        assert result["registration"]["stable_inliers"] == 9
        assert result["registration"]["error_p95_pixels"] <= 1.5
        assert all(p["height"] >= 12 and p["width"] >= 4
                   for p in result["registration"]["native_registered_digit_pixels"])
    else:
        assert reason == "native_registered_digit_resolution"
    assert legacy._fit(current.astype(np.float32), points, image, "fixture")[1] == "panel_scale_gate"


def test_field_motion_cannot_accept_an_out_of_sensor_numeric_region():
    reference = structure()
    obj = SourcePanelTracker(reference, [.2, .2, .65, .8], [[.3, .3, .7, .6]],
                             required_boxes=[[.2, .2, .8, .8]],
                             motion_digit_boxes=[[.25, .3, .45, .75]])
    points = np.float32([[x, y] for y in [60, 100, 180] for x in [80, 140, 220]])
    result, reason = obj._fit(points + [350, 0], points, reference, "fixture")
    assert result is None and "clipped" in reason


def test_field_motion_still_enforces_the_native_roi_pixel_budget():
    gray = np.random.default_rng(41).integers(20, 220, (800, 1280), np.uint8)
    reference = cv2.cvtColor(cv2.GaussianBlur(gray, (5, 5), 1), cv2.COLOR_GRAY2BGR)
    obj = SourcePanelTracker(reference, [.1, .1, .3, .42], [[.3, .3, .7, .6]],
                             required_boxes=[[.2, .2, .8, .8]],
                             motion_digit_boxes=[[.25, .3, .45, .75]])
    image = cv2.warpAffine(reference, np.float32([[2.1, 0, 0], [0, 2.1, 0]]), (1280, 800))
    points = np.float32([[x, y] for y in [100, 200, 300] for x in [150, 230, 350]])
    result, reason = obj._fit(points * 2.1, points, image, "fixture")
    assert result is None and reason == "tracked_native_ROI_exceeds_budget"


@pytest.mark.parametrize("boxes", [[], [[.1, .1, .11, .11]]])
def test_field_motion_requires_resolvable_reference_geometry(boxes):
    with pytest.raises(ValueError):
        SourcePanelTracker(structure(), [.2, .2, .65, .8], [[.3, .3, .7, .6]],
                           motion_digit_boxes=boxes)


def test_registered_motion_flow_cannot_extrapolate_free_projective_distortion():
    reference = structure()
    obj = SourcePanelTracker(reference, [.15, .15, .8, .85], [[.3, .3, .7, .6]],
                             motion_digit_boxes=[[.35, .3, .65, .7]])
    points = np.float32([[x, y] for y in [45, 120, 190] for x in [60, 180, 280]])
    affine = cv2.getRotationMatrix2D((180, 120), -8, .7)
    image = cv2.warpAffine(reference, affine, (360, 240))
    current = cv2.transform(points[None], affine)[0]
    fit, reason = obj._fit(current, points, image, "source_panel_bidirectional_flow/1")
    assert reason is None
    assert fit["registration"]["structural_motion_model"] == "relative_similarity"
    matrix = fit["matrix_full"]
    assert np.allclose(matrix[2], [0, 0, 1])
    assert np.allclose(matrix[:2, :2].T @ matrix[:2, :2], np.eye(2)/.7**2, atol=1e-4)
    perspective = np.array([[.7, -.1, 70], [.1, .7, 20], [.001, 0, 1.]])
    current = cv2.perspectiveTransform(points[None], perspective)[0]
    image = cv2.warpPerspective(reference, perspective, (360, 240))
    assert obj._fit(current, points, image, "source_panel_bidirectional_flow/1")[0] is None
    fit, reason = obj._fit(current, points, image, "source_panel_descriptor_reacquisition/1")
    assert reason is None and fit["registration"]["structural_motion_model"] == "projective"
    # Preserve an independently fitted perspective pose while constraining the
    # next relative motion; the camera need not face a frontoparallel display.
    obj.flow_reference_matrix = fit["matrix_full"]
    increment = cv2.getRotationMatrix2D((180, 120), -2, .97)
    next_pose = np.vstack([increment, [0, 0, 1]]) @ perspective
    current = cv2.perspectiveTransform(points[None], next_pose)[0]
    image = cv2.warpPerspective(reference, next_pose, (360, 240))
    fit, reason = obj._fit(current, points, image, "source_panel_bidirectional_flow/1")
    assert reason is None and fit["registration"]["error_p95_pixels"] < .01


def test_relative_motion_ransac_preserves_reference_pixel_tolerance_after_zoom():
    image = structure()
    obj = SourcePanelTracker(image, [.2, .2, .65, .8], [[.3, .3, .7, .6]],
                             motion_digit_boxes=[[.25, .3, .45, .75]])
    points = np.float32([[x, y]for x in [80, 120, 170, 220]for y in [60, 95, 135, 180]])
    pose = np.array([[.5, 0, 25], [0, .5, 30], [0, 0, 1.]])
    obj.flow_reference_matrix = np.linalg.inv(pose)
    current = cv2.perspectiveTransform(points[None], pose)[0]
    current[:3] += [1.2, .4]  # Native-pixel errors exceed 1.5 reference pixels.
    observed = cv2.warpPerspective(image, pose, (360, 240))
    fit, reason = obj._fit(current, points, observed, "source_panel_bidirectional_flow/1")
    assert reason is None and fit["registration"]["stable_inliers"] >= 13
    assert fit["registration"]["error_p95_pixels"] < .01


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


def test_clipped_faceplate_context_requires_complete_critical_source_field():
    image = structure()
    moved = cv2.warpAffine(image, np.float32([[1, 0, 0], [0, 1, -60]]), (360, 240))
    reference = np.float32(
        [
            [90, 80],
            [130, 80],
            [180, 80],
            [220, 80],
            [90, 140],
            [130, 140],
            [180, 140],
            [220, 140],
        ]
    )
    current = reference - [0, 60]
    args = (image, [0.2, 0.2, 0.65, 0.8], [[0.3, 0.3, 0.7, 0.6]])
    obj = SourcePanelTracker(*args, required_boxes=[[0.3, 0.4, 0.7, 0.6]])
    result, reason = obj._fit(
        current.astype(np.float32),
        reference,
        moved,
        "source_panel_descriptor_reacquisition/1",
    )
    assert reason is None
    assert result["registration"]["outer_context_clipped"] is True
    assert result["registration"]["required_field_pixels_complete"] is True
    assert result["box"][1] == 0
    assert np.array_equal(
        result["image"], moved[: result["box"][3], result["box"][0] : result["box"][2]]
    )
    # The old path stays strict, and losing a required field never falls back.
    default = SourcePanelTracker(*args)
    assert (
        default._fit(current.astype(np.float32), reference, moved, "fixture")[0] is None
    )
    clipped_field = SourcePanelTracker(*args, required_boxes=[[0.3, 0, 0.7, 0.1]])
    rejected, reason = clipped_field._fit(
        current.astype(np.float32), reference, moved, "fixture"
    )
    assert rejected is None and reason == "required_numeric_field_clipped"


def test_geometry_reference_checks_original_pixels_and_never_runs_OCR(monkeypatch):
    from visioncortex.lab import temporal_video as module

    image = structure()
    spec = dict(
        pts=1200000,
        time_base="1/1000000",
        media_offset_seconds="1.2",
        source_frame_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
    )
    meta = dict(pts=1200000, time_base="1/1000000", media_offset="6/5")
    calls = []

    def decode(path, start, end, provider):
        calls.append((start, end, provider["max_input_frames"]))
        yield image, meta

    monkeypatch.setattr(module, "decode_frames", decode)
    pixels, receipt = module.geometry_reference(
        "unused", spec, dict(rotation_degrees_clockwise=180), {}
    )
    assert np.array_equal(pixels, image[::-1, ::-1])
    assert receipt["OCR_invocations"] == 0 and receipt["numerical_evidence"] is False
    assert calls[0][2] == 1
    with pytest.raises(module.TemporalVideoRejected, match="identity changed"):
        module.geometry_reference(
            "unused",
            dict(spec, source_frame_pixels_sha256="a" * 64),
            dict(rotation_degrees_clockwise=0),
            {},
        )


def test_reference_context_cannot_cross_equipment_or_epoch():
    from visioncortex.lab.temporal_video import context_identity

    first = dict(
        instrument_id="equipment-a",
        epoch=2,
        association_status="spatial_identity_supported",
    )
    assert context_identity(first) == context_identity(
        dict(first, irrelevant_display_name="changed")
    )
    assert context_identity(first) != context_identity(
        dict(first, instrument_id="equipment-b")
    )
    assert context_identity(first) != context_identity(dict(first, epoch=3))


def test_reference_configuration_rejects_future_and_cross_source_geometry():
    import copy
    from visioncortex.lab.temporal_video import (
        validate_registration,
        TemporalVideoRejected,
    )

    document = dict(
        schema_version="vc-lab-temporal-video-ocr/1",
        enabled=True,
        revision="fixture/1",
        source_stream_key="arbitrary-source",
        source_sha256="a" * 64,
        media_version=1,
        rotation_degrees_clockwise=0,
        orientation_basis="fixture source",
        world_clock_status="uncalibrated",
        time_basis="original_media_pts",
        source_intervals=[dict(start_seconds="2", end_seconds="3")],
        fields=[
            dict(
                screen_id="panel",
                field_key="digits",
                profile_id="layout",
                role="process_value",
                roi_box=["0.2", "0.2", "0.7", "0.8"],
                source_tracking=True,
                use_source_reference=True,
                layout=dict(
                    revision="fixture/1",
                    display_type="seven_segment",
                    numeric_field_box=[0.1, 0.2, 0.8, 0.6],
                ),
            )
        ],
        source_tracking_reference=dict(
            schema_version="vc-source-panel-reference/1",
            source_sha256="a" * 64,
            media_version=1,
            pts=1000000,
            time_base="1/1000000",
            media_offset_seconds="1",
            source_frame_pixels_sha256="b" * 64,
        ),
    )
    assert validate_registration(document)
    motion = copy.deepcopy(document)
    motion["fields"][0]["source_field_motion"] = True
    motion["fields"][0]["layout"].update(data_type="number", digits=[[.2, .2, .5, .6]])
    assert validate_registration(motion)
    scaled = copy.deepcopy(motion)
    scaled["fields"][0]["source_scale_reacquisition"] = True
    assert validate_registration(scaled)
    scaled["fields"][0]["source_field_motion"] = False
    with pytest.raises(TemporalVideoRejected, match="explicit field motion"):
        validate_registration(scaled)
    for change in ("nonboolean", "untyped", "missing_digits", "no_tracking"):
        invalid = copy.deepcopy(motion)
        field = invalid["fields"][0]
        if change == "nonboolean":
            field["source_field_motion"] = "true"
        elif change == "untyped":
            field["layout"]["data_type"] = "clock"
        elif change == "missing_digits":
            field["layout"].pop("digits")
        else:
            field["source_tracking"] = False
        with pytest.raises(TemporalVideoRejected):
            validate_registration(invalid)
    future = copy.deepcopy(document)
    future["source_tracking_reference"]["media_offset_seconds"] = "3"
    with pytest.raises(TemporalVideoRejected, match="must precede"):
        validate_registration(future)
    other = copy.deepcopy(document)
    other["source_tracking_reference"]["source_sha256"] = "c" * 64
    with pytest.raises(TemporalVideoRejected, match="same-source"):
        validate_registration(other)
