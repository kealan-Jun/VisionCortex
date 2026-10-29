"""Bounded small LCD recovery; changed pixels never replace source evidence."""

import copy
import re
import cv2
import numpy as np
from field_recognition.recognition.digit_regions import refine_digits
from field_recognition.measurements.readout_rules import READOUT


_LCD_ALIASES = {
    "O": ("0",),
    "I": ("1",),
    "L": ("1",),
    "B": ("8",),
    "G": ("6",),
    # Seven-segment 5/9 are commonly returned as the letter S.  Keep both
    # possibilities unless an all-numeric view corroborates one of them.
    "S": ("5", "9"),
    "Z": ("2",),
}


def _lcd_numeric_aliases(text):
    """Return conservative digit aliases for a short LCD OCR token."""
    token = str(text).strip().upper()
    if not re.fullmatch(r"[0-9A-Z+-]+", token):
        return []
    options = [""]
    for char in token:
        values = (char,) if char.isdigit() or char in "+-" else _LCD_ALIASES.get(char)
        if not values:
            return []
        options = [prefix + value for prefix in options for value in values]
        if len(options) > 16:
            return []
    return [value for value in options if re.fullmatch(r"[+-]?\d+", value)]


def _wide_fixed_decimal_retry(predict, panel, result, x, y, places):
    """Retry a small LCD with the whole digit row, including a leading glyph.

    Generic OCR often crops a fixed-decimal display at the first visible digit
    and returns the fractional tail (for example ``1698`` instead of
    ``11698``).  The retry is still bounded to the already detector-authorized
    digit region.  A value is promoted only when two overlapping windows return
    the same complete glyph string; this does not infer a missing digit or
    decimal point.
    """
    region = result.get("digit_region") or {}
    if region.get("status") != "located" or type(places) is not int or places < 1:
        return result
    points = np.asarray(region.get("polygon"), dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return result
    local = points - np.asarray([x, y], dtype=float)
    left, top = local.min(axis=0)
    right, bottom = local.max(axis=0)
    width, height = max(1.0, right - left), max(1.0, bottom - top)
    minimum = places + 1
    # The first view starts just left of the current OCR row and includes the
    # lower display line.  Neighboring views provide an independent check for
    # the leading glyph while remaining inside this panel crop.
    # A one-pixel border change matters on the low-contrast LCD row.  These
    # margins include the upper leading glyph and the lower display baseline
    # seen in the authorized panel crop without expanding to the whole photo.
    y0 = max(0, int(np.floor(top - 1.15 * height)))
    y1 = min(panel.shape[0], int(np.ceil(bottom + 0.37 * height)))
    attempts, candidates = [], []
    for left_factor in (0.14, 0.20, 0.26):
        sx = max(0, int(np.floor(left - left_factor * width)))
        ex = min(panel.shape[1], int(np.ceil(right + 0.25 * width)))
        if ex - sx < 24 or y1 - y0 < 12:
            continue
        crop = panel[y0:y1, sx:ex]
        scale = 4
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        attempt = predict(enlarged, 0, 0)
        attempts.append(
            {
                "crop": [sx + x, y0 + y, ex - sx, y1 - y0],
                "scale": scale,
                "left_factor": left_factor,
                "ocr": copy.deepcopy(attempt),
            }
        )
        options = []
        for line in attempt.get("lines", []):
            text = str(line.get("text", "")).strip()
            digits = re.sub(r"\D", "", text)
            aliases = _lcd_numeric_aliases(text)
            glyph_count = len(text.lstrip("+-"))
            if aliases and glyph_count >= minimum and (line.get("confidence") or 0) >= 0.5:
                mapped = copy.deepcopy(line)
                if mapped.get("polygon") is not None:
                    mapped["polygon"] = (
                        np.asarray(mapped["polygon"], dtype=float) / scale + [sx + x, y0 + y]
                    ).tolist()
                options.append((mapped, aliases))
        if options:
            candidates.extend(options)
    result["fixed_decimal_wide_ocr"] = {
        "method": "fixed_decimal_wide_consensus_v1",
        "required_digits": minimum,
        "attempts": attempts,
    }
    by_text = {}
    exact_by_text = {}
    for candidate, aliases in candidates:
        source_text = candidate["text"].strip().upper()
        for alias in aliases:
            by_text.setdefault(alias, []).append(candidate)
        if re.fullmatch(r"[+-]?\d+", source_text):
            exact_by_text.setdefault(source_text, []).append(candidate)
    if not by_text:
        return result
    text, supporting = max(
        by_text.items(),
        key=lambda item: (len(item[1]), max(v.get("confidence", 0) for v in item[1])),
    )
    second = max((len(values) for key, values in by_text.items() if key != text), default=0)
    # An alias such as S→5/9 is accepted only when an all-numeric OCR view
    # corroborates the winning value and the vote is not tied.
    if len(supporting) < 2 or len(supporting) <= second or text not in exact_by_text:
        return result
    chosen = max(exact_by_text[text], key=lambda line: line.get("confidence", 0))
    raw_texts = sorted({line.get("text", "").strip() for line, _ in candidates})
    if chosen.get("text", "").strip() != text:
        chosen["original_ocr_text"] = chosen.get("text")
    chosen["text"] = text
    chosen["reading_source"] = "fixed_decimal_wide_consensus"
    chosen["numeric_candidates"] = [text]
    result["lines"] = [chosen]
    result["digit_consistency"] = {
        "status": "resolved",
        "method": "fixed_decimal_wide_consensus_v1",
        "text": text,
        "supporting_views": len(supporting),
        "candidates": raw_texts,
        "normalized_text": text,
        "exact_numeric_views": len(exact_by_text[text]),
    }
    # Save the actual source window used for the accepted views, not a
    # fabricated super-resolution image.
    anchor = np.asarray([[left, y0], [right, y0], [right, y1 - 1], [left, y1 - 1]])
    result["digit_region"] = {
        "status": "located",
        "method": "fixed_decimal_wide_consensus_v1",
        "polygon": (anchor + [x, y]).tolist(),
        "size": [int((right - left) * 4), int((y1 - y0) * 4)],
        "initial_text": region.get("initial_text"),
        "refined_readout_available": True,
    }
    return result


def refine_lcd(
    predict, panel, initial, x=0, y=0, *, source_image=None, source_offset=(0, 0), instrument=None
):
    if (
        initial.get("status") != "completed"
        or min(panel.shape[:2]) >= 160
        or max(panel.shape[:2]) >= 480
    ):
        return refine_digits(
            predict, panel, initial, x, y, source_image=source_image, source_offset=source_offset
        )
    enlarged = predict(cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 0, 0)
    mapped = copy.deepcopy(enlarged)
    for line in mapped.get("lines", []):
        if line.get("polygon") is not None:
            line["polygon"] = (np.asarray(line["polygon"]) / 2 + [x, y]).tolist()
    result = refine_digits(
        predict, panel, mapped, x, y, source_image=source_image, source_offset=source_offset
    )
    result.update(
        panel_ocr=copy.deepcopy(initial),
        small_panel_ocr=enlarged,
        small_panel_recovery={"method": "authorized_lcd_2x_v2", "scale": 2},
    )
    spec = ((instrument or {}).get("measurement_ranges") or {}).get("质量") or {}
    places = spec.get("decimal_places")
    minimum = places + 1 if spec.get("fixed_decimal_display") is True and type(places) is int else 1
    complete = lambda line: bool(
        READOUT.fullmatch(line.get("text", "").strip())
        and len(re.sub(r"\D", "", line["text"])) >= minimum
        and (line.get("confidence") or 0) >= 0.6
    )
    if any(complete(line) for line in result.get("lines", [])):
        return result
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4)).apply(gray)
    checked = predict(cv2.resize(contrast, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), 0, 0)
    result["lcd_contrast_ocr"] = checked
    # Enhancement alone is never authoritative: an unmodified view must agree.
    raw_texts = {
        line["text"].strip()
        for view in (initial, mapped)
        for line in view.get("lines", [])
        if complete(line)
    }
    matches = [
        line
        for line in checked.get("lines", [])
        if complete(line) and line["text"].strip() in raw_texts
    ]
    if len(matches) == 1 and matches[0].get("polygon") is not None:
        line = copy.deepcopy(matches[0])
        points = np.asarray(line["polygon"], dtype=float) / 3
        line["polygon"] = (points + [x, y]).tolist()
        lo = points.min(0)
        hi = points.max(0)
        pad = (hi - lo) * 0.1
        lo = np.maximum(0, lo - pad)
        hi = np.minimum([panel.shape[1] - 1, panel.shape[0] - 1], hi + pad)
        result.update(
            lines=[line],
            digit_consistency={
                "status": "resolved",
                "method": "source_contrast_agreement_v1",
                "text": line["text"],
            },
            digit_region={
                "status": "located",
                "method": "source_contrast_agreement_v1",
                "polygon": (
                    np.asarray([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
                    + [x, y]
                ).tolist(),
                "size": [max(8, int((hi[0] - lo[0]) * 3)), max(8, int((hi[1] - lo[1]) * 3))],
                "refined_readout_available": True,
            },
        )
    else:
        result["lines"] = []
        result["digit_consistency"] = {
            "status": "unresolved",
            "method": "source_contrast_agreement_v1",
        }
    # A balance registered with four fixed decimal places may have a leading
    # integer glyph that generic OCR clips from the narrow line.  Retry only
    # after the normal source/contrast checks and require two agreeing views.
    if not any(complete(line) for line in result.get("lines", [])):
        result = _wide_fixed_decimal_retry(predict, panel, result, x, y, places)
    if checked.get("ocr_finished_at"):
        result["ocr_finished_at"] = checked["ocr_finished_at"]
    return result
