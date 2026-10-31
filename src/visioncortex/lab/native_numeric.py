"""Preserve native source pixels for an opt-in registered LED-field reader.

Registration locates the crop; it does not resample its glyphs. Emission is an
absence veto, not proof of every stroke or correctness of an OCR prediction.
"""

import hashlib
import json
import math
import re

import numpy as np

from .temporal_display import _reference
from .temporal_registration import normalized_box


def same_transform(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (
        a.shape != (3, 3)
        or b.shape != (3, 3)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        return False
    index = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    if abs(a[index]) < 1e-12 or abs(b[index]) < 1e-12:
        return False
    return bool(np.allclose(a, b * (a[index] / b[index]), rtol=1e-7, atol=1e-8))


def prepare(image, layout, current_to_reference, *, reference_shape=None):
    import cv2

    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] * image.shape[1]
        > (1048576 if reference_shape is not None else 262144)
    ):
        raise ValueError("Bounded native uint8 BGR field required")
    if layout.get("color") not in {"red", "green"}:
        raise ValueError("Native emission reader requires an explicit LED color")
    digits = layout.get("digits", [])
    if not 1 <= len(digits) <= 12:
        raise ValueError("One to twelve registered digit slots required")
    shape = tuple(reference_shape) if reference_shape is not None else image.shape
    if len(shape) < 2 or min(shape[:2]) < 8 or shape[0] * shape[1] > 262144:
        raise ValueError("Bounded explicit reference geometry required")
    boxes = [
        normalized_box(d.get("box") if isinstance(d, dict) else d, shape)
        for d in digits
    ]
    matrix = np.asarray(current_to_reference, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(np.linalg.det(matrix)) < 1e-8
    ):
        raise ValueError("Invertible finite registration required")
    inverse = np.linalg.inv(matrix)
    quads = []
    for x, y, xx, yy in boxes:
        q = cv2.perspectiveTransform(
            np.float32([[[x, y], [xx, y], [xx, yy], [x, yy]]]), inverse
        )[0]
        if (
            not np.isfinite(q).all()
            or not cv2.isContourConvex(q)
            or cv2.contourArea(q) < 8
            or (q < 0).any()
            or (q[:, 0] > image.shape[1]).any()
            or (q[:, 1] > image.shape[0]).any()
        ):
            raise ValueError("Native digit polygon clipped or degenerate")
        quads.append(q)
    all_points = np.concatenate(quads)
    pad = max(2, math.ceil(min(b[3] - b[1] for b in boxes) * 0.14))
    if layout.get("numeric_field_box") is not None:
        left, top, right, bottom = normalized_box(layout["numeric_field_box"], shape)
        if any(
            not (left <= x < xx <= right and top <= y < yy <= bottom)
            for x, y, xx, yy in boxes
        ):
            raise ValueError("Digit geometry escapes its declared native field")
        all_points = cv2.perspectiveTransform(
            np.float32([[[left, top], [right, top], [right, bottom], [left, bottom]]]),
            inverse,
        )[0]
        if (
            not np.isfinite(all_points).all()
            or not cv2.isContourConvex(all_points)
            or (all_points < 0).any()
            or (all_points[:, 0] > image.shape[1]).any()
            or (all_points[:, 1] > image.shape[0]).any()
        ):
            raise ValueError("Declared native field clipped by source")
        pad = 2
    x, y = np.maximum(0, np.floor(all_points.min(axis=0) - pad)).astype(int)
    xx, yy = np.minimum(
        [image.shape[1], image.shape[0]],
        np.ceil(all_points.max(axis=0) + pad),
    ).astype(int)
    crop = image[y:yy, x:xx].copy()
    if crop.shape[0] * crop.shape[1] > 262144:
        raise ValueError("Native crop exceeds its retained pixel budget")
    native_quads = [q - [x, y] for q in quads]
    h, w = crop.shape[:2]
    native_boxes = []
    for q, original_digit in zip(native_quads, digits, strict=True):
        left, top = np.maximum(0, np.floor(q.min(axis=0))).astype(int)
        right, bottom = np.minimum([w, h], np.ceil(q.max(axis=0))).astype(int)
        box = [left / w, top / h, right / w, bottom / h]
        native_boxes.append(
            dict(original_digit, box=box) if isinstance(original_digit, dict) else box
        )
    return dict(
        image=crop,
        quads=[q.tolist() for q in native_quads],
        layout=dict(layout, digits=native_boxes),
        metadata=dict(
            method="native_registered_crop/1",
            crop_box=[int(x), int(y), int(xx), int(yy)],
            current_to_crop=[[1, 0, -int(x)], [0, 1, -int(y)], [0, 0, 1]],
            current_to_reference=matrix.tolist(),
            native_bgr_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
            digit_quadrilaterals=[q.tolist() for q in native_quads],
            resampled=False,
            strokes_completed=False,
        ),
    )


def emission(native):
    """Do not match the blue casing/background of an extinguished red digit."""
    import cv2

    image, layout = native["image"], native["layout"]
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] * image.shape[1] > 262144
        or layout.get("color") not in {"red", "green"}
        or not 1 <= len(native["quads"]) <= 12
        or hashlib.sha256(image.tobytes()).hexdigest()
        != native["metadata"]["native_bgr_sha256"]
    ):
        raise ValueError("Invalid bounded native pixel evidence")
    channel = 2 if layout["color"] == "red" else 1
    pixels = image.astype(float)
    signal = pixels[:, :, channel] - np.max(np.delete(pixels, channel, axis=2), axis=2)
    rows = []
    for q in native["quads"]:
        q = np.asarray(q, dtype=float)
        if (
            q.shape != (4, 2)
            or not np.isfinite(q).all()
            or (q < 0).any()
            or (q[:, 0] > image.shape[1]).any()
            or (q[:, 1] > image.shape[0]).any()
        ):
            raise ValueError("Native emission polygon escapes source pixels")
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.rint(q).astype(np.int32), 1)
        values = signal[mask > 0]
        if not values.size:
            raise ValueError("Native slot has no source pixels")
        area = int((values > 20).sum())
        percentile = float(np.percentile(values, 95))
        height = float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)
        rows.append(
            dict(
                visible=height >= 12
                and percentile >= 20
                and area >= max(2, math.ceil(values.size * 0.02)),
                native_digit_height=height,
                positive_color_area=area,
                chroma95=percentile,
                energy=float(np.maximum(values - 20, 0).mean()),
                complete_glyph_proven=False,
            )
        )
    return rows


def review(native, reply):
    from .video_numeric import number, resolve_decimal_tokens, resolve_observed_points

    text = reply.get("raw_text", "")
    token = number(text)
    support = emission(native)
    reasons = []
    if any(not slot["visible"] for slot in support):
        reasons.append("native_digit_slot_not_emitting")
    if token is None or len(token.replace(".", "")) != len(support):
        reasons.append("OCR_does_not_cover_native_digit_slots")
    glyphs = token.replace(".", "") if token and not reasons else None
    stroke_veto = native["layout"].get("numeric_native_stroke_veto", False)
    if type(stroke_veto) is not bool:
        raise ValueError("Explicit native stroke-veto boolean required")
    alternate = native["layout"].get("numeric_native_alternate_geometry", False)
    if type(alternate) is not bool or alternate and not stroke_veto:
        raise ValueError(
            "Alternate geometry requires explicit source-stroke verification"
        )
    projective = native["layout"].get("numeric_native_projective_decimal", False)
    if type(projective) is not bool or projective and (
        not stroke_veto or native["layout"].get("data_type") != "number"
    ):
        raise ValueError("Projective points require a typed number and source-stroke verification")
    stroke_evidence = stroke_assessment = None
    geometry_hypotheses = []
    decimal_layout = native["layout"]
    if glyphs and stroke_veto:
        if native["layout"].get("display_type") != "seven_segment":
            raise ValueError(
                "Seven-segment source layout required for stroke verification"
            )
        from .native_segments import observe, assess

        stroke_evidence = observe(native)
        stroke_assessment = assess(stroke_evidence, glyphs)
        if alternate:
            geometry_hypotheses.append(
                dict(evidence=stroke_evidence, assessment=stroke_assessment)
            )
            if not stroke_assessment["supported"] and not any(
                isinstance(d, dict) and d.get("segments")
                for d in native["layout"]["digits"]
            ):
                alternative = observe(native, expanded=True)
                assessment = assess(alternative, glyphs)
                geometry_hypotheses.append(
                    dict(evidence=alternative, assessment=assessment)
                )
                if assessment["supported"]:
                    stroke_evidence, stroke_assessment = alternative, assessment
        if not stroke_assessment["supported"]:
            glyphs = None
            reasons.append("OCR_required_strokes_not_established_from_source")
        else:
            h, w = native["image"].shape[:2]
            cells = [
                np.asarray(s["native_cell_polygon"]) for s in stroke_evidence["slots"]
            ]
            decimal_layout = dict(
                decimal_layout,
                digits=[
                    [
                        float(q[:, 0].min() / w),
                        float(q[:, 1].min() / h),
                        float(q[:, 0].max() / w),
                        float(q[:, 1].max() / h),
                    ]
                    for q in cells
                ],
            )
    decimal = None
    point_observation = None
    if glyphs:
        try:
            decimal = resolve_decimal_tokens(native["image"], decimal_layout, [text])
        except ValueError:
            if not stroke_veto:
                raise
            # Slanted cells can have overlapping axis-aligned extents. Such
            # geometry cannot establish the strict native gap; keep it unknown.
            decimal = dict(
                numeric_candidate=None,
                decimal_status="unknown",
                confirmed=False,
                reason="fitted_cell_gap_not_established",
            )
        if projective and not decimal.get("native_point_evidence") and len(cells) >= 2:
            from .native_points import observe as observe_points

            point_observation = observe_points(
                native["image"], cells, native["layout"]["color"],
                glyph_margin=2 if stroke_evidence.get("geometry", {}).get("hypothesis")
                == "expanded_source_envelope/1" else 1,
            )
            resolved = resolve_observed_points([text], len(cells), point_observation["points"])
            decimal = dict(resolved, legacy_gap_evidence=decimal,
                           projective_point_evidence=point_observation)
    candidate = decimal.get("numeric_candidate") if decimal else None
    if glyphs and candidate is None:
        reasons.append("decimal_not_established_from_native_pixels")
    return dict(
        raw_text=text,
        glyph_candidate=glyphs,
        numeric_candidate=candidate,
        native_emission=support,
        native_decimal_evidence=decimal,
        **(dict(native_projective_point_observation=point_observation) if projective else {}),
        rejection_reasons=reasons,
        confirmed=False,
        value_text=None,
        model_score_is_accuracy=False,
        full_glyph_correctness_proven=False,
        **(
            dict(
                native_segment_evidence=stroke_evidence,
                native_stroke_assessment=stroke_assessment,
            )
            if stroke_veto
            else {}
        ),
        **(dict(source_geometry_hypotheses=geometry_hypotheses) if alternate else {}),
    )


def point_probe(native):
    """Locate a point for single-frame selection, without reading a value."""
    from .native_segments import observe
    from .native_points import observe as observe_points
    from .temporal_display import SEGMENTS

    layout = native["layout"]
    if (layout.get("data_type") != "number" or layout.get("display_type") != "seven_segment"
            or layout.get("numeric_native_stroke_veto") is not True):
        raise ValueError("Typed source-verified numeric field required for point selection")
    evidence = observe(native)
    lit = [{name for name, cell in slot["segments"].items() if cell["state"] == "ON"}
           for slot in evidence["slots"]]
    shape_consistent = bool(lit) and all(
        any(value == set(pattern) for pattern in SEGMENTS.values()) for value in lit
    )
    signature = hashlib.sha256(json.dumps(
        [sorted(value) for value in lit], separators=(",", ":")
    ).encode()).hexdigest()
    cells = [slot["native_cell_polygon"] for slot in evidence["slots"]]
    if len(cells) < 2:
        return dict(points=[], reason="source_cell_geometry_pending", confirmed=False,
                    positive_shape_consistent=shape_consistent, OFF_inferred=False,
                    positive_shape_signature=signature)
    points = observe_points(native["image"], cells, layout["color"])
    return dict(points, source_bgr_sha256=native["metadata"]["native_bgr_sha256"],
                geometry_method=evidence["geometry"]["method"], numeric_value=None,
                positive_shape_consistent=shape_consistent,
                positive_shape_signature=signature,
                positive_shape_is_digit_correctness=False)


def source(native, native_source_ref, registered_source_ref):
    """Carry the crop transform without changing its original frame or clock."""
    raw, raw_time = _reference(native_source_ref)
    aligned, aligned_time = _reference(registered_source_ref)
    if raw_time != aligned_time or raw["source_sha256"] != aligned["source_sha256"]:
        raise ValueError("Native crop and registration refer to different frames")
    matched = []
    for key in ("source_frame_pixels_sha256", "original_frame_rgb_sha256"):
        if raw.get(key) != aligned.get(key):
            raise ValueError(
                "Native crop and registration have different source pixels"
            )
        if isinstance(raw.get(key), str) and re.fullmatch("[0-9a-f]{64}", raw[key]):
            matched.append(key)
    if not matched:
        raise ValueError("Original source pixel fingerprint required for native OCR")
    expected = np.asarray(native["metadata"]["current_to_reference"]) @ np.asarray(
        raw["source_to_roi"]
    )
    if not same_transform(expected, aligned["source_to_roi"]):
        raise ValueError(
            "Native pixels and registered digit geometry belong to different ROIs"
        )
    if raw.get("dynamic_physical_key") != aligned.get("dynamic_physical_key"):
        raise ValueError(
            "Native pixels and registration have different physical contexts"
        )
    return dict(
        raw,
        source_to_roi=(
            np.asarray(native["metadata"]["current_to_crop"])
            @ np.asarray(raw["source_to_roi"])
        ).tolist(),
        registered_source_to_roi=aligned["source_to_roi"],
        registration=aligned.get("registration", {"accepted": True}),
        native_crop=native["metadata"],
    )
