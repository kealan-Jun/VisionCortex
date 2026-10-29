"""Read separated red LED bars in an authorized integer display crop.

Only exact seven-segment patterns are accepted across multiple thresholds.
This is pixel evidence, not a calibrated probability or a general OCR model.
"""

import cv2
import numpy as np

from field_recognition.recognition.led_digits import PATTERNS, has_dark_interiors

VERSION = "fragmented-red-segments/3"
CENTERS = (
    (0.5, 0.08),
    (0.82, 0.28),
    (0.78, 0.75),
    (0.45, 0.92),
    (0.12, 0.73),
    (0.2, 0.26),
    (0.5, 0.51),
)


def _glyph(mask, box):
    x, y, w, h = box
    glyph = mask[y : y + h, x : x + w]
    if w / h < 0.30:
        # A narrow 1 consists of two vertical bars. Do not mistake a decimal
        # point, minus sign, or a clipped sliver of another glyph for a 1.
        components = cv2.connectedComponentsWithStats(glyph)[2][1:]
        if (
            len(components) == 2
            and all(c[3] >= h * 0.25 and c[3] > c[2] for c in components)
            and abs(float(components[0][1]) - float(components[1][1])) >= h * 0.35
        ):
            return "1", None
        return None, None
    if not 0.35 <= w / h <= 0.85:
        return None, None
    levels = []
    for cx, cy in CENTERS:
        left, right = max(0, round((cx - 0.18) * w)), min(w, round((cx + 0.18) * w) + 1)
        top, bottom = max(0, round((cy - 0.105) * h)), min(h, round((cy + 0.105) * h) + 1)
        levels.append(float(glyph[top:bottom, left:right].mean()))
    if any(0.16 < v < 0.24 for v in levels):
        return None, levels
    pattern = "".join("1" if v >= 0.24 else "0" for v in levels)
    value = PATTERNS.get(pattern)
    return value if value and has_dark_interiors(glyph, value) else None, levels


def _check(image, factors=(0.60, 0.70, 0.75)):
    evidence = {"rule_version": VERSION, "status": "inconclusive", "views": []}
    if image.ndim != 3 or min(image.shape[:2]) < 16:
        return evidence
    blue, green, red = cv2.split(image.astype(np.float32))
    channel = np.maximum(blue, green)  # Either unsaturated channel can retain a faint bar.
    color = (red - channel > 60) & (red > 180)
    if np.count_nonzero(color) < 40:
        return evidence
    threshold, _ = cv2.threshold(
        channel[color].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if threshold < 20 or np.percentile(channel[color], 95) - threshold < 15:
        return evidence
    for factor in factors:
        mask = ((channel > threshold * factor) & color).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        areas = stats[1:, 4]
        if not np.any(areas >= 2):
            continue
        minimum = max(2, float(np.median(areas[areas >= 2])) * 0.15)
        small = []
        for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if area < minimum:
                mask[labels == index] = 0
                if area >= 2:
                    small.append((x, y, w, h))
        columns = np.flatnonzero(mask.sum(axis=0))
        if not len(columns):
            continue
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
        if not 2 <= len(groups) <= 6:
            continue
        boxes = []
        for group in groups:
            left, right = int(group[0]), int(group[-1]) + 1
            rows = np.flatnonzero(mask[:, left:right].sum(axis=1))
            boxes.append((left, int(rows[0]), right - left, int(rows[-1] - rows[0]) + 1))
        height = max(b[3] for b in boxes)
        top = min(b[1] for b in boxes)
        # At very low native resolution the glow fills unlit bars (7 can look
        # like 9). Enlarging pixels cannot supply that missing evidence.
        if (
            any(b[3] < 20 or b[3] < height * 0.65 for b in boxes)
            or max(b[1] + b[3] / 2 for b in boxes) - min(b[1] + b[3] / 2 for b in boxes)
            > height * 0.4
        ):
            continue
        # An isolated small mark at the baseline could be a decimal point.
        if any(
            y + h > top + height * 0.75 and not any(bx <= x < bx + bw for bx, _, bw, _ in boxes)
            for x, y, w, h in small
        ):
            continue
        text, fills = "", []
        for box in boxes:
            digit, levels = _glyph(mask, box)
            if digit is None:
                break
            text += digit
            fills.append(None if levels is None else [round(v, 4) for v in levels])
        if len(text) == len(boxes):
            evidence["views"].append(
                {
                    "threshold": round(threshold * factor, 2),
                    "text": text,
                    "glyph_boxes": boxes,
                    "segment_fill": fills,
                }
            )
    values = {v["text"] for v in evidence["views"]}
    if len(values) == 1 and len(evidence["views"]) >= 2:
        boxes = [b for v in evidence["views"] for b in v["glyph_boxes"]]
        left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right, bottom = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        evidence.update(
            status="consistent", text=values.pop(), crop=[left, top, right - left, bottom - top]
        )
    elif len(values) > 1:
        evidence["status"] = "disagreement"
    return evidence


def check(image, *, native=None):
    """Check original pixels, with one measured, resolution-preserving deskew.

    A retry requires independently separated native glyphs at the original
    resolution. No angle search, upsampling or synthetic stroke joins are used.
    """
    evidence = _check(image)
    if evidence["status"] != "inconclusive":
        return evidence
    from field_recognition.recognition.digit_regions import red_display_geometry
    from field_recognition.recognition.led_digits import native_diagnostics

    native = native if native is not None else native_diagnostics(image)
    if (
        native.get("status") != "measured"
        or native.get("below_resolution_gate")
        or native.get("possible_decimal_point")
        or native.get("merged_glyphs_possible")
    ):
        return evidence
    geometry = red_display_geometry(image)
    if geometry is None or not 0.5 <= abs(geometry["roll_degrees"]) <= 12:
        return evidence
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), geometry["roll_degrees"], 1.0)
    rectified = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    # One additional, higher threshold can separate glow-connected vertical
    # bars. Complete patterns still need two views and any contrary view vetoes.
    retry = _check(rectified, factors=(0.60, 0.70, 0.75, 0.80))
    copy_views = [dict(view) for view in evidence["views"]]
    diagnostics = {
        "method": "measured_source_roll_nearest_v1",
        "roll_degrees": geometry["roll_degrees"],
        "scale": 1.0,
        "interpolation": "nearest",
        "source_size": [width, height],
        "views": copy_views,
        "status": retry["status"],
    }
    retry["native_deskew"] = diagnostics
    observed = {view["text"] for view in copy_views + retry["views"]}
    if len(observed) > 1:
        retry["status"] = diagnostics["status"] = "disagreement"
        retry.pop("text", None)
        retry.pop("crop", None)
        return retry
    if retry["status"] != "consistent":
        return retry
    if len(retry["text"]) != native["group_count"]:
        retry["status"] = diagnostics["status"] = "inconclusive"
        diagnostics["reason"] = "source_group_count_changed"
        retry.pop("text", None)
        retry.pop("crop", None)
        return retry
    left, top, crop_width, crop_height = retry["crop"]
    points = np.asarray(
        [
            [
                [left, top],
                [left + crop_width - 1, top],
                [left + crop_width - 1, top + crop_height - 1],
                [left, top + crop_height - 1],
            ]
        ],
        np.float32,
    )
    polygon = cv2.transform(points, cv2.invertAffineTransform(transform))[0]
    if (
        np.any(polygon < 0)
        or np.any(polygon[:, 0] > width - 1)
        or np.any(polygon[:, 1] > height - 1)
    ):
        retry["status"] = diagnostics["status"] = "inconclusive"
        diagnostics["reason"] = "source_window_clipped"
        retry.pop("text", None)
        retry.pop("crop", None)
        return retry
    retry["polygon"] = polygon.tolist()
    retry["native_window_size"] = [crop_width, crop_height]
    # The compatibility crop is an enclosing source rectangle; polygon retains
    # the actual pixel window and is used for subsequent source mapping.
    lo, hi = np.floor(polygon.min(0)).astype(int), np.ceil(polygon.max(0)).astype(int)
    retry["crop"] = [int(lo[0]), int(lo[1]), int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)]
    return retry


def corroborates(value, texts):
    """Require an OCR view to agree or omit just one independently visible digit."""
    return any(
        text == value
        or (
            len(text) + 1 == len(value)
            and any(value[:i] + value[i + 1 :] == text for i in range(len(value)))
        )
        for text in texts
    )
