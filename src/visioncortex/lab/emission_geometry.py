"""First-frame pose correction of registered LED-slot geometry, without OCR.

Only a common translation is estimated from positive emission. The number,
order and dimensions of registered slots stay fixed. No glyph is supplied by
the registration, and missing/ambiguous emission leaves geometry pending.
"""

import copy
import hashlib
import math

import cv2
import numpy as np

from .temporal_registration import normalized_box


def reanchor(image, layout):
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] * image.shape[1] > 262144
    ):
        raise ValueError("Bounded native panel required for LED pose correction")
    if (
        layout.get("color") not in {"red", "green"}
        or not 1 <= len(layout.get("digits", [])) <= 12
    ):
        raise ValueError("Explicit LED color and registered slots required")
    h, w = image.shape[:2]
    boxes = [
        normalized_box(d.get("box") if isinstance(d, dict) else d, image.shape)
        for d in layout["digits"]
    ]
    if min(b[3] - b[1] for b in boxes) < 12:
        return None, dict(status="pending", reason="insufficient_native_slot_height")
    color = 2 if layout["color"] == "red" else 1
    pixels = image.astype(np.int16)
    signal = pixels[:, :, color] - np.max(np.delete(pixels, color, axis=2), axis=2)
    energy = cv2.integral(np.maximum(signal - 35, 0).astype(np.float32))
    positive = cv2.integral((signal > 35).astype(np.uint8))

    def total(integral, b):
        x, y, xx, yy = b
        return float(
            integral[yy, xx] - integral[y, xx] - integral[yy, x] + integral[y, x]
        )

    proposals = []
    for dy in range(-math.floor(h * 0.2), math.floor(h * 0.2) + 1, 2):
        for dx in range(-math.floor(w * 0.2), math.floor(w * 0.2) + 1, 2):
            shifted = [(x + dx, y + dy, xx + dx, yy + dy) for x, y, xx, yy in boxes]
            if any(x < 0 or y < 0 or xx > w or yy > h for x, y, xx, yy in shifted):
                continue
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in shifted]
            if any(
                total(positive, b) < max(2, area * 0.02)
                for b, area in zip(shifted, areas, strict=True)
            ):
                continue
            values = [
                total(energy, b) / area for b, area in zip(shifted, areas, strict=True)
            ]
            score = min(values) + 0.25 * float(np.median(values))
            proposals.append((score, -(dx * dx + dy * dy), dx, dy))
    if not proposals:
        return None, dict(status="pending", reason="all_registered_slots_not_observed")
    proposals.sort(reverse=True)
    best, _, dx, dy = proposals[0]
    height = float(np.median([b[3] - b[1] for b in boxes]))
    distant = [p[0] for p in proposals if abs(p[3] - dy) > height]
    if distant and max(distant) > best * 0.8:
        return None, dict(status="pending", reason="ambiguous_emissive_row")
    result = copy.deepcopy(layout)

    def translated(box):
        x, y, xx, yy = map(float, box)
        value = [x + dx / w, y + dy / h, xx + dx / w, yy + dy / h]
        if not 0 <= value[0] < value[2] <= 1 or not 0 <= value[1] < value[3] <= 1:
            raise ValueError("Translated registered geometry escaped panel")
        return value

    try:
        result["digits"] = [
            dict(d, box=translated(d["box"])) if isinstance(d, dict) else translated(d)
            for d in layout["digits"]
        ]
        if layout.get("numeric_field_box"):
            result["numeric_field_box"] = translated(layout["numeric_field_box"])
    except ValueError:
        return None, dict(status="pending", reason="translated_field_clipped")
    receipt = dict(
        method="positive_LED_geometry_translation/1",
        status="geometry_candidate",
        offset_pixels=[dx, dy],
        original_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        slot_count=len(boxes),
        glyphs_inferred=False,
        decimal_inferred=False,
        physical_identity_inferred=False,
        score=best,
    )
    result["initial_geometry_observation"] = receipt
    return result, receipt
