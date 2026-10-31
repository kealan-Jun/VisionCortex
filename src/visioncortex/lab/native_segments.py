"""Positive native-pixel stroke support for seven-segment OCR candidates.

An OCR digit must not claim lit strokes that were never observed. Darkness is
UNKNOWN, never OFF: passing this veto still cannot confirm a digit or fact.
The topology is common to decimal seven-segment displays, not learned readings
or site-specific glyph prototypes.
"""

import hashlib

import cv2
import numpy as np

from .temporal_display import SEGMENTS, SEGMENT_BOXES


def polygon_mask(shape, quad):
    """Sample actual pixel centers, without rounding a tiny cell into its neighbor."""
    q = np.asarray(quad, dtype=float)
    h, w = shape[:2]
    left, top = np.maximum(0, np.floor(q.min(axis=0))).astype(int)
    right, bottom = np.minimum([w, h], np.ceil(q.max(axis=0))).astype(int)
    mask = np.zeros((h, w), bool)
    if left >= right or top >= bottom:
        return mask
    xx, yy = np.meshgrid(np.arange(left, right) + 0.5, np.arange(top, bottom) + 0.5)
    inside = np.ones(xx.shape, bool)
    orientation = 1 if cv2.contourArea(q.astype(np.float32), oriented=True) >= 0 else -1
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        cross = (b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])
        inside &= cross * orientation >= -1e-6
    mask[top:bottom, left:right] = inside
    return mask


def positive_pixels(image, quad, color):
    """Measure original coarse-slot contrast once, before fitting geometry.

    A slot's color anchor permits bright achromatic appearance evidence too;
    tinted glass and clipping can change its hue. This is not a calibrated
    illumination opportunity. Flat saturated white remains possible glare.
    """
    channel = 2 if color == "red" else 1
    q = np.asarray(quad, np.float32)
    mask = polygon_mask(image.shape, q)
    colors = image.astype(np.int16)
    values = colors[:, :, channel][mask > 0]
    background, upper = np.percentile(values, [25, 98])
    span = float(upper - background)
    minimum = float(background + max(25, 0.55 * span))
    signal = colors[:, :, channel] - np.max(np.delete(colors, channel, axis=2), axis=2)
    core = (mask > 0) & (colors[:, :, channel] >= minimum)
    glare = (colors.min(axis=2) > 235) & (colors.max(axis=2) - colors.min(axis=2) < 20)
    core &= ~glare
    seeds = core & (signal > 20)
    # Protective tinted glass can shift a bright stroke toward white/blue.
    # Its measured same-slot contrast remains useful appearance evidence even
    # when a scan stripe disconnects it from the colored part. This supplies
    # neither a negative observation nor a calibrated illumination opportunity.
    supported = core if int(seeds.sum()) >= 2 else np.zeros(mask.shape, bool)
    return supported, span, minimum


def observe(native):
    image, layout = native["image"], native["layout"]
    if (
        layout.get("color") not in {"red", "green"}
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] * image.shape[1] > 262144
        or hashlib.sha256(image.tobytes()).hexdigest()
        != native["metadata"]["native_bgr_sha256"]
        or native["metadata"].get("resampled") is not False
        or not 1 <= len(native["quads"]) <= 12
        or len(layout.get("digits", [])) != len(native["quads"])
    ):
        raise ValueError("Source-pinned native LED evidence required")
    for quad in native["quads"]:
        q = np.asarray(quad, np.float32)
        h, w = image.shape[:2]
        if (
            q.shape != (4, 2)
            or not np.isfinite(q).all()
            or (q < 0).any()
            or (q[:, 0] > w).any()
            or (q[:, 1] > h).any()
            or not cv2.isContourConvex(q)
            or cv2.contourArea(q) < 8
        ):
            raise ValueError("Coarse native slot escapes source pixels")
    rows = []
    from .segment_lattice import fit as fit_lattice

    source_observations = [
        positive_pixels(image, q, layout["color"]) for q in native["quads"]
    ]
    if any(isinstance(d, dict) and d.get("segments") for d in layout["digits"]):
        quads, fit = (
            native["quads"],
            dict(
                method="explicit_registered_segment_cells/1",
                status="geometry_candidate",
                OCR_hypothesis_used=False,
            ),
        )
    else:
        quads, fit = fit_lattice(native, [v[0] for v in source_observations])
    if quads is None:
        return dict(
            method="native_seven_segment_positive_support/1",
            slots=[],
            geometry=fit,
            OFF_inferred=False,
            glyphs_confirmed=False,
        )
    for index, quad in enumerate(quads):
        q = np.asarray(quad, np.float32)
        h, w = image.shape[:2]
        if (
            q.shape != (4, 2)
            or not np.isfinite(q).all()
            or (q < 0).any()
            or (q[:, 0] > w).any()
            or (q[:, 1] > h).any()
            or not cv2.isContourConvex(q)
            or cv2.contourArea(q) < 8
        ):
            raise ValueError("Native digit quadrilateral escapes its pixels")
        mask = polygon_mask(image.shape, q)
        if not mask.any():
            raise ValueError("Native digit has no source pixels")
        # A geometry hypothesis must not re-threshold its own evidence. Keep
        # the same measured source-slot contrast and pixels used before fitting.
        positive, span, minimum = source_observations[index]
        height = min(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
        transform = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), q
        )
        digit = layout["digits"][index]
        regions = (
            digit.get("segments", SEGMENT_BOXES)
            if isinstance(digit, dict)
            else SEGMENT_BOXES
        )
        if set(regions) != set("abcdefg"):
            raise ValueError("Seven explicitly located segment cells required")
        cells = {}
        for name, box in regions.items():
            x, y, xx, yy = map(float, box)
            if not 0 <= x < xx <= 1 or not 0 <= y < yy <= 1:
                raise ValueError("Segment cell escaped its native digit")
            polygon = cv2.perspectiveTransform(
                np.float32([[[x, y], [xx, y], [xx, yy], [x, yy]]]), transform
            )[0]
            cell = polygon_mask(image.shape, polygon)
            selected = cell & mask
            count = int(selected.sum())
            lit = int((positive & selected).sum())
            yy, xx = np.nonzero(positive & selected)
            extent = 0.0
            if len(xx) >= 2:
                unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
                projected = cv2.perspectiveTransform(
                    np.float32(np.column_stack([xx + 0.5, yy + 0.5]))[None],
                    cv2.getPerspectiveTransform(polygon.astype(np.float32), unit),
                )[0]
                axis = 0 if name in "adg" else 1
                extent = float(
                    np.percentile(projected[:, axis], 95)
                    - np.percentile(projected[:, axis], 5)
                )
            on = (
                height >= 12
                and span >= 35
                and count >= 2
                and lit >= 2
                and lit / max(1, count) >= 0.3
                and extent >= 0.45
            )
            cells[name] = dict(
                state="ON" if on else "UNKNOWN",
                source_pixels=count,
                observed_active_pixels=lit,
                longitudinal_extent=extent,
                native_polygon=polygon.tolist(),
            )
        rows.append(
            dict(
                slot=index,
                segments=cells,
                contrast_span=span,
                minimum_channel_level=minimum,
                native_digit_height=float(height),
                native_cell_polygon=q.tolist(),
            )
        )
    return dict(
        method="native_seven_segment_positive_support/1",
        slots=rows,
        geometry=fit,
        source_bgr_sha256=native["metadata"]["native_bgr_sha256"],
        OFF_inferred=False,
        glyphs_confirmed=False,
    )


def assess(evidence, glyphs):
    if evidence.get("geometry", {}).get("status") == "pending":
        return dict(
            supported=False,
            reason=evidence["geometry"]["reason"],
            required_strokes_unknown=None,
            positive_strokes_contradicting_OCR=None,
            OFF_inferred=False,
            confirmed=False,
        )
    if (
        not isinstance(glyphs, str)
        or len(glyphs) != len(evidence["slots"])
        or not glyphs.isascii()
        or not glyphs.isdigit()
    ):
        raise ValueError("One OCR digit for each registered native slot required")
    required_unknown, contradictory = [], []
    for slot, glyph in zip(evidence["slots"], glyphs, strict=True):
        lit = {name for name, cell in slot["segments"].items() if cell["state"] == "ON"}
        required = set(SEGMENTS[glyph])
        required_unknown.extend(
            f"{slot['slot']}:{name}" for name in sorted(required - lit)
        )
        contradictory.extend(
            f"{slot['slot']}:{name}" for name in sorted(lit - required)
        )
    return dict(
        supported=not required_unknown and not contradictory,
        required_strokes_unknown=required_unknown,
        positive_strokes_contradicting_OCR=contradictory,
        OFF_inferred=False,
        confirmed=False,
        limitation="unlit cells remain unknown; positive support is only a veto",
    )
