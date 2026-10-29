"""Bounded pixel check for integer digits on red seven-segment displays.

This is an independent geometry check, not a measured OCR accuracy score.
Separated red bars use a second bounded geometry check. Signed, decimal and
status displays remain with the existing reader.
"""

import copy
import re

import cv2
import numpy as np

VERSION = "red-led-segments/3"
SEGMENTS = (
    (0.2, 0, 0.8, 0.2),
    (0.65, 0.15, 1, 0.48),
    (0.65, 0.52, 1, 0.9),
    (0.2, 0.8, 0.8, 1),
    (0, 0.52, 0.35, 0.9),
    (0, 0.15, 0.35, 0.48),
    (0.2, 0.4, 0.8, 0.6),
)
PATTERNS = {
    "1111110": "0",
    "0110000": "1",
    "1101101": "2",
    "1111001": "3",
    "0110011": "4",
    "1011011": "5",
    "1011111": "6",
    "1110000": "7",
    "1111111": "8",
    "1111011": "9",
}


def has_dark_interiors(glyph, value):
    """A closed LED loop needs background inside, not just seven bright probes."""
    centers = {"0": (0.28, 0.73), "6": (0.73,), "8": (0.28, 0.73), "9": (0.28,)}
    height, width = glyph.shape
    for center in centers.get(value, ()):
        # Search only the interior band, allowing slanted segment fonts. A
        # substantial connected dark patch is required; isolated noisy pixels
        # in an otherwise luminous column do not establish a loop's background.
        left, right = int(width * 0.2), max(int(width * 0.2) + 1, int(np.ceil(width * 0.8)))
        top = max(0, int((center - 0.13) * height))
        bottom = min(height, max(top + 1, int(np.ceil((center + 0.13) * height))))
        interior = glyph[top:bottom, left:right]
        dark = (interior == 0).astype(np.uint8)
        stats = cv2.connectedComponentsWithStats(dark)[2][1:]
        if not len(stats) or max(stats[:, 4]) < interior.size * 0.2:
            return False
    return True


def native_diagnostics(image):
    """Describe visible strokes without decoding or enlarging native pixels."""
    evidence = {
        "basis": "original_panel_pixels",
        "native_size": [int(image.shape[1]), int(image.shape[0])],
        "minimum_glyph_height": 20,
        "status": "inconclusive",
        "views": [],
    }
    if image.ndim != 3 or min(image.shape[:2]) < 8:
        return evidence
    blue, green, red = cv2.split(image.astype(np.float32))
    channel = np.maximum(blue, green)
    color = (red - channel > 60) & (red > 180)
    if np.count_nonzero(color) >= 40:
        threshold, _ = cv2.threshold(
            channel[color].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if threshold < 20 or float(np.percentile(channel[color], 95)) - threshold < 15:
            return evidence
        factors = (0.60, 0.70, 0.75)
        evidence["appearance"] = "red_bars"
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        surround = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 80)
        if surround.mean() < 0.15:
            return evidence
        channel = green
        threshold, _ = cv2.threshold(
            channel.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if threshold < 10 or float(np.percentile(channel, 95)) - threshold < 20:
            return evidence
        color = np.ones(channel.shape, dtype=bool)
        factors = (0.85, 1.0, 1.15)
        evidence["appearance"] = "bright_bars_on_red"
    collapsed = []
    for factor in factors:
        mask = ((channel > threshold * factor) & color).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index, (_, _, _, _, area) in enumerate(stats[1:], 1):
            if area < 2:
                mask[labels == index] = 0
        columns = np.flatnonzero(mask.sum(axis=0))
        if not len(columns):
            continue
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
        if not 1 <= len(groups) <= 6:
            continue
        boxes = []
        for group in groups:
            left, right = int(group[0]), int(group[-1]) + 1
            rows = np.flatnonzero(mask[:, left:right].sum(axis=1))
            boxes.append([left, int(rows[0]), right - left, int(rows[-1] - rows[0]) + 1])
        height = max(box[3] for box in boxes)
        if len(boxes) == 1:
            # One wide projection can hide several glow-connected glyphs.
            # Preserve this negative evidence without claiming a digit count.
            box = boxes[0]
            if height >= image.shape[0] * 0.25 and box[2] / height > 1.5:
                collapsed.append({"threshold": round(threshold * factor, 2), "box": box})
            continue
        tall = [box for box in boxes if box[3] >= height * 0.65]
        possible_dot = len(tall) >= 2 and any(
            box[3] <= height * 0.25
            and box[2] <= height * 0.25
            and box[1] + box[3] >= min(b[1] for b in tall) + height * 0.75
            and min(b[0] for b in tall) < box[0] < max(b[0] + b[2] for b in tall)
            for box in boxes
        )
        if possible_dot:
            boxes = tall
        if (
            height < image.shape[0] * 0.25
            or any(box[3] < height * 0.65 for box in boxes)
            # Gaps inside a single disconnected glyph are not extra digits.
            or any(b[0] - (a[0] + a[2]) < max(2, height * 0.10) for a, b in zip(boxes, boxes[1:]))
            or max(b[1] + b[3] / 2 for b in boxes) - min(b[1] + b[3] / 2 for b in boxes)
            > height * 0.4
        ):
            continue
        evidence["views"].append(
            {
                "threshold": round(threshold * factor, 2),
                "group_count": len(boxes),
                "glyph_boxes": boxes,
                "possible_decimal_point": possible_dot,
            }
        )
    counts = [view["group_count"] for view in evidence["views"]]
    stable = next((count for count in counts if counts.count(count) >= 2), None)
    if stable is not None:
        boxes = [
            box
            for view in evidence["views"]
            if view["group_count"] == stable
            for box in view["glyph_boxes"]
        ]
        evidence.update(
            status="measured",
            group_count=stable,
            glyph_height_range=[min(b[3] for b in boxes), max(b[3] for b in boxes)],
            below_resolution_gate=any(b[3] < 20 for b in boxes),
            merged_glyphs_possible=any(b[2] / b[3] > 0.95 for b in boxes),
            possible_decimal_point=sum(
                bool(view["possible_decimal_point"])
                for view in evidence["views"]
                if view["group_count"] == stable
            )
            >= 2,
        )
    if len(collapsed) >= 2:
        stable_boxes = []
        for i, first in enumerate(collapsed):
            a = first["box"]
            for second in collapsed[i + 1 :]:
                b = second["box"]
                overlap = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
                    0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                )
                if overlap / (a[2] * a[3] + b[2] * b[3] - overlap) >= 0.8:
                    stable_boxes.extend((a, b))
        if stable_boxes:
            evidence.update(
                collapsed_views=collapsed,
                collapsed_row_height_range=[
                    min(box[3] for box in stable_boxes),
                    max(box[3] for box in stable_boxes),
                ],
                low_resolution_collapsed_row=all(box[3] < 20 for box in stable_boxes),
            )
    return evidence


def _local_confusion(value, text):
    """Only bounded OCR character confusions supported by complete pixel digits."""
    return bool(
        re.fullmatch(r"[\dB]{2,6}", text)
        and re.search(r"\d", text)
        and text.count("B") <= 1
        and len(value) == len(text)
        and all(
            a == b or {a, b} == {"1", "7"} or (a == "8" and b == "B") for a, b in zip(value, text)
        )
    )


def segment_check(crop, digits):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 80)
    evidence = {
        "rule_version": VERSION,
        "status": "inconclusive",
        "minimum_glyph_height": 20,
        "views": [],
    }
    if red.mean() < 0.15 or min(crop.shape[:2]) < 12:
        return evidence
    channel = crop[:, :, 1]  # Bright LED segments against their red display surround.
    threshold, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if threshold < 10 or float(np.percentile(channel, 95)) - threshold < 20:
        return evidence
    for factor in (0.85, 1.0, 1.15):
        _, mask = cv2.threshold(channel, threshold * factor, 255, cv2.THRESH_BINARY)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        glyphs = [
            (x, y, w, h)
            for x, y, w, h, area in stats[1:]
            if h >= max(20, crop.shape[0] * 0.35) and 0.30 <= w / h <= 0.85 and area >= w * h * 0.12
        ]
        glyphs.sort()
        if len(glyphs) != digits or any(a[0] + a[2] > b[0] for a, b in zip(glyphs, glyphs[1:])):
            continue
        text, fills = "", []
        for x, y, w, h in glyphs:
            glyph = mask[y : y + h, x : x + w]
            levels = [
                float(
                    np.mean(
                        glyph[
                            int(t * h) : max(int(t * h) + 1, int(b * h)),
                            int(l * w) : max(int(l * w) + 1, int(r * w)),
                        ]
                        > 0
                    )
                )
                for l, t, r, b in SEGMENTS
            ]
            # A marginal segment cannot settle a conflicting reading.
            if any(0.30 < level < 0.50 for level in levels):
                break
            value = PATTERNS.get("".join("1" if level >= 0.50 else "0" for level in levels))
            if value is None or not has_dark_interiors(glyph, value):
                break
            text += value
            fills.append([round(level, 3) for level in levels])
        if len(text) == digits:
            evidence["views"].append(
                {"threshold": round(threshold * factor, 2), "text": text, "segment_fill": fills}
            )
    values = {view["text"] for view in evidence["views"]}
    if len(values) == 1 and len(evidence["views"]) >= 2:
        evidence.update(status="consistent", text=values.pop())
    elif len(values) > 1:
        evidence.update(status="disagreement")
    return evidence


def verify_digits(image, local, x=0, y=0):
    lines = local.get("lines", [])
    region = local.get("digit_region") or {}
    if lines and (
        len(lines) != 1 or not re.fullmatch(r"[\dB]{1,6}", lines[0].get("text", "").strip())
    ):
        return local
    raw_lines = [
        line
        for key in ("panel_ocr", "digit_ocr", "digit_ocr_check")
        for line in local.get(key, {}).get("lines", [])
    ]
    raw_lines.extend(
        line
        for attempt in local.get("red_display_ocr_attempts", [])
        for line in attempt.get("ocr", {}).get("lines", [])
    )
    texts = [line.get("text", "").strip() for line in raw_lines or lines]
    if any(re.search(r"[+\-.,°/]", text) or text.upper() == "OFF" for text in texts):
        return local
    if not any(re.fullmatch(r"[\dB]{1,6}", text) and re.search(r"\d", text) for text in texts):
        return local
    from field_recognition.recognition.red_segments import check as check_fragmented, corroborates

    native = native_diagnostics(image)
    fragmented = check_fragmented(image, native=native)
    fragmented["native_diagnostics"] = native
    quality_only = bool(lines and len(lines[0].get("text", "").strip()) == 1)
    if not quality_only and fragmented["status"] in {"consistent", "disagreement"}:
        output = copy.deepcopy(local)
        output.update(
            digit_segment_check=fragmented,
            pre_segment_lines=copy.deepcopy(lines),
            digit_consistency_before_segments=copy.deepcopy(local.get("digit_consistency")),
        )
        numeric = [text for text in texts if re.fullmatch(r"\d{1,6}", text)]
        resolved = fragmented["status"] == "consistent" and (
            corroborates(fragmented["text"], numeric)
            or any(_local_confusion(fragmented["text"], text) for text in texts)
        )
        output["digit_consistency"] = {
            "status": "resolved" if resolved else "disagreement",
            "candidates": list(dict.fromkeys(texts)),
            "basis": fragmented["rule_version"],
        }
        output["lines"] = []
        if resolved:
            # Preserve the measured original-pixel window and all previous OCR
            # attempts. Never copy a neighbouring photo's reading or confidence.
            left, top, width, height = fragmented["crop"]
            polygon = fragmented.get("polygon") or [
                [left, top],
                [left + width - 1, top],
                [left + width - 1, top + height - 1],
                [left, top + height - 1],
            ]
            polygon = (np.asarray(polygon) + [x, y]).tolist()
            width, height = fragmented.get("native_window_size", [width, height])
            value = fragmented["text"]
            output["digit_region_before_segments"] = copy.deepcopy(region)
            output["digit_region"] = {
                "status": "located",
                "method": fragmented["rule_version"],
                "polygon": polygon,
                "size": [width * 3, height * 3],
                "initial_text": texts[0] if texts else None,
                "refined_readout_available": True,
            }
            output["lines"] = [
                {
                    "text": value,
                    "value": value,
                    "numeric_candidates": [value],
                    "confidence": None,
                    "polygon": polygon,
                    "reading_source": "led_segment_check",
                    "original_ocr_text": texts[0] if texts else None,
                }
            ]
        return output
    if not lines:
        # An upstream OCR conflict can remove candidates from formal lines;
        # retain the independently measured rejection instead of losing it.
        output = copy.deepcopy(local)
        output["digit_segment_check"] = fragmented
        return output
    if region.get("status") != "located" and not quality_only:
        return local
    if len(lines) != 1:
        return local
    original = lines[0].get("text", "").strip()
    if not re.fullmatch(r"[\dB]{1,6}", original):
        return local
    issues = []
    if quality_only and fragmented["status"] == "consistent":
        issues.append("possible_missing_digit")
    if native.get("status") == "measured":
        if native["possible_decimal_point"]:
            issues.append("possible_decimal_point")
        if native["group_count"] > len(original):
            issues.append("possible_missing_digit")
        if native["merged_glyphs_possible"]:
            issues.append("ambiguous_led_glyphs")
        if issues and native["below_resolution_gate"]:
            issues.insert(0, "low_native_resolution")
    issues = list(dict.fromkeys(issues))
    if issues:
        output = copy.deepcopy(local)
        output.update(
            digit_segment_check=fragmented,
            pre_segment_lines=copy.deepcopy(lines),
            digit_consistency_before_segments=copy.deepcopy(local.get("digit_consistency")),
            digit_consistency={
                "status": "quality_issue",
                "basis": VERSION,
                "candidates": list(dict.fromkeys(texts)),
                "quality_issues": issues,
            },
        )
        # Preserve the candidate text, but prevent its model confidence from
        # turning insufficient native evidence into a normalized measurement.
        output["lines"] = [
            lines[0] | {"quality_issue": issues[0], "quality_issues": issues, "confidence": None}
        ]
        return output
    if quality_only:
        output = copy.deepcopy(local)
        fragmented["reason"] = "single_digit_quality_only"
        output["digit_segment_check"] = fragmented
        return output
    if native.get("below_resolution_gate"):
        # Low resolution alone does not reject an otherwise agreeing OCR
        # candidate, but enlarged pixels cannot support a character repair.
        output = copy.deepcopy(local)
        fragmented["reason"] = "insufficient_native_resolution"
        output["digit_segment_check"] = fragmented
        return output
    # The localized window's saved size is often enlarged for OCR. Derive its
    # physical size from the source polygon instead, and never add resolution
    # for a segment correction, even when color diagnostics are inconclusive.
    points = np.asarray(region["polygon"], np.float32) - np.asarray([x, y], np.float32)
    native_size = [
        int(
            np.floor(
                (np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])) / 2
            )
        ),
        int(
            np.floor(
                (np.linalg.norm(points[3] - points[0]) + np.linalg.norm(points[2] - points[1])) / 2
            )
        ),
    ]
    if min(native_size) < 2:
        return local
    width, height = native_size
    target = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    transform = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(
        image,
        transform,
        tuple(native_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )
    check = segment_check(crop, len(original))
    check["native_diagnostics"] = native
    check["native_window_size"] = native_size
    output = copy.deepcopy(local)
    output["digit_segment_check"] = check
    if check["status"] == "consistent" and check["text"] != original:
        # Resolve only bounded 1/7 or B/8 ambiguity when every segment is clear.
        # Other differences stay unresolved for the normal fallback/draft path.
        resolved = _local_confusion(check["text"], original)
        output["digit_consistency"] = {
            "status": "resolved" if resolved else "disagreement",
            "candidates": [original, check["text"]],
            "basis": VERSION,
        }
        output["lines"] = (
            [
                lines[0]
                | {
                    "text": check["text"],
                    "value": check["text"],
                    "numeric_candidates": [check["text"]],
                    "confidence": None,
                    "reading_source": "led_segment_check",
                    "original_ocr_text": original,
                }
            ]
            if resolved
            else []
        )
    elif check["status"] == "disagreement":
        output["lines"] = []
        output["digit_consistency"] = {"status": "disagreement", "basis": VERSION}
    return output
