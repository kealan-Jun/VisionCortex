"""Bounded same-frame OCR views; original pixels remain the evidence."""

import hashlib
import math

import numpy as np


def color_residual(image):
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or not 1 <= image.shape[0] * image.shape[1] <= 262144):
        raise ValueError("Bounded native BGR view required")
    pixels = image.astype(np.float32)
    residual = np.maximum(pixels - pixels.min(axis=2)[:, :, None], 0)
    scale = max(1, float(np.percentile(residual.max(axis=2), 99)))
    prepared = np.uint8(np.minimum(255, residual * 255 / scale))
    return prepared, dict(
        method="same_frame_min_channel_subtraction_percentile99/1",
        source_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        prepared_bgr_sha256=hashlib.sha256(prepared.tobytes()).hexdigest(),
        percentile99=scale, resampled=False, source_pixels_replaced=False,
        supplies_stroke_or_point_evidence=False,
    )


def support_crop(native):
    """Crop original pixels around actual emission; never draw a glyph.

    The full original native image remains the stroke/decimal proof. A dim
    inactive leading cell or surrounding label need not be an OCR input.
    """
    from .native_segments import observe, positive_pixels, polygon_mask

    evidence = observe(native)
    if len(evidence["slots"]) != len(native["quads"]):
        return None, dict(status="pending", reason="source_cell_geometry_pending")
    image, layout = native["image"], native["layout"]
    support = np.zeros(image.shape[:2], bool)
    heights, cells = [], []
    for coarse, slot in zip(native["quads"], evidence["slots"], strict=True):
        q = np.asarray(slot["native_cell_polygon"], np.float32)
        positive, _, _ = positive_pixels(image, coarse, layout["color"])
        pixels = positive & polygon_mask(image.shape, q)
        height = float(np.linalg.norm(q[3] - q[0]))
        if int(pixels.sum()) < 2 or height < 16:
            return None, dict(status="pending", reason="source_slot_emission_insufficient")
        support |= pixels
        heights.append(height)
        cells.append(q)
    yy, xx = np.nonzero(support)
    left, top = max(0, int(xx.min()) - 2), max(0, int(yy.min()) - 2)
    right = min(image.shape[1], int(xx.max()) + 3)
    bottom = min(image.shape[0], max(
        int(yy.max()) + 3, math.ceil(max(q[:, 1].max() for q in cells) + .25 * min(heights))))
    crop = image[top:bottom, left:right].copy()
    return crop, dict(
        method="same_frame_native_emission_bounds/1", status="prepared_candidate",
        source_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        prepared_bgr_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
        native_crop_box=[left, top, right, bottom],
        native_to_prepared=[[1, 0, -left], [0, 1, -top], [0, 0, 1]],
        resampled=False, source_pixels_replaced=False, strokes_completed=False,
        supplies_stroke_or_point_evidence=False, OCR_text_used_for_geometry=False,
    )
