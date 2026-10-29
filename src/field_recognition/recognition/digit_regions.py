"""Select the dominant numeric line inside an already authorized display ROI."""

import copy
import math
import re
import time

import cv2
import numpy as np

from field_recognition.measurements.readout_rules import READOUT
from field_recognition.recognition.red_geometry import (
    extract_digits,
    red_display_geometry,
    _red_ocr_views,
    _off_ocr_views,
)


def _off_row_mask(mask):
    """Locate a native character row without filling or changing its pixels."""
    rows = np.flatnonzero(mask.sum(axis=1))
    runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if len(rows) else []
    full_rows = [run for run in runs if int(run[-1] - run[0] + 1) >= 8]
    receipt = {
        "source_row_runs": [[int(run[0]), int(run[-1]) + 1] for run in runs],
        "eligible_row_count": len(full_rows),
        "basis": "unmodified_source_mask",
        "excluded_row_pixel_count": 0,
    }
    if len(full_rows) != 1:
        return mask, receipt
    top, bottom = int(full_rows[0][0]), int(full_rows[0][-1]) + 1
    grouped = mask.copy()
    grouped[:top] = 0
    grouped[bottom:] = 0
    receipt.update(
        basis="single_native_foreground_row",
        row_bounds=[top, bottom],
        excluded_row_pixel_count=int(mask.sum() - grouped.sum()),
    )
    return grouped, receipt


def _off_f_bars(glyph):
    """Prove two F arms and their open right side on the original native mask.

    Fixed segment probes can touch a thick top arm or left stem in a narrow F.
    Measure its actual horizontal arm rows instead; their intervening gaps
    must still contain a left stem and no contrary right-hand stroke.
    """
    height, width = glyph.shape
    right_start = math.ceil(width * 0.65)
    right = glyph[:, right_start:]
    arm_rows = np.flatnonzero((glyph.sum(axis=1) >= width * 0.5) & (right.mean(axis=1) >= 0.3))
    arms = np.split(arm_rows, np.flatnonzero(np.diff(arm_rows) > 1) + 1) if len(arm_rows) else []
    proof = {
        "basis": "native_source_f_bar_topology",
        "arm_rows": [[int(arm[0]), int(arm[-1]) + 1] for arm in arms],
        "matches_f": False,
    }
    if len(arms) != 2:
        return dict(proof, reason="f_arm_count")
    upper, middle = arms
    if (
        upper[0] > height * 0.25
        or middle[0] < height * 0.3
        or middle[-1] > height * 0.75
        or middle[0] - upper[-1] <= 1
    ):
        return dict(proof, reason="f_arm_positions")
    left = glyph[:, : math.ceil(width * 0.4)]
    gaps = [(int(upper[-1]) + 1, int(middle[0])), (int(middle[-1]) + 1, height)]
    left_levels = [float(left[top:bottom].mean()) for top, bottom in gaps]
    right_levels = [float(right[top:bottom].mean()) for top, bottom in gaps]
    stem_coverage = float(np.mean(left.sum(axis=1) > 0))
    proof.update(
        gap_rows=[list(gap) for gap in gaps],
        left_stem_levels=[round(level, 4) for level in left_levels],
        right_counterevidence_levels=[round(level, 4) for level in right_levels],
        left_stem_row_coverage=round(stem_coverage, 4),
    )
    if stem_coverage < 0.7 or any(level < 0.3 for level in left_levels):
        return dict(proof, reason="f_left_stem_incomplete")
    if any(level >= 0.3 for level in right_levels):
        return dict(proof, reason="f_right_stroke_counterevidence")
    return dict(proof, matches_f=True, reason="complete_source_f_bars")


def _off_shape(image):
    """Check O/F/F bar direction at native resolution, independently of OCR."""
    evidence = {"status": "inconclusive", "method": "native_red_off_bars_v3", "views": []}
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.size
        or image.dtype != np.uint8
    ):
        return dict(evidence, reason="invalid_source_image")
    blue, green, red = cv2.split(image.astype(np.int16))
    channel = np.maximum(blue, green).astype(np.uint8)
    # Neutral frame/background pixels are not LED bars. They must not join
    # glyph columns or set the contrast of the independently measured row.
    red_source = (red - channel > 18) & (red > 60)
    values = channel[red_source]
    evidence["red_source_pixel_count"] = int(values.size)
    if values.size < 16:
        return dict(evidence, reason="insufficient_red_source_pixels")
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    views = []
    contrast = float(np.percentile(values, 95)) - threshold
    evidence.update(source_threshold=round(threshold, 3), source_contrast=round(contrast, 3))
    if contrast < 15:
        return dict(evidence, reason="insufficient_source_contrast")
    for factor in (0.2, 0.4):
        cut = threshold + contrast * factor
        mask = ((channel > cut) & red_source).astype(np.uint8)
        grouped_mask, row_receipt = _off_row_mask(mask)
        columns = np.flatnonzero(grouped_mask.sum(axis=0))
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1) if len(columns) else []
        view = {
            "threshold_factor": factor,
            "threshold": round(cut, 3),
            "foreground_pixel_count": int(np.count_nonzero(mask)),
            "group_count": len(groups),
            "row_grouping": row_receipt,
            "glyph_boxes": [],
            "patterns": [],
            "glyph_proofs": [],
            "matches_off": False,
        }
        views.append(view)
        if len(groups) != 3:
            view["reason"] = "glyph_group_count"
            continue
        for group in groups:
            left, right = int(group[0]), int(group[-1]) + 1
            rows = np.flatnonzero(grouped_mask[:, left:right].sum(axis=1))
            top, bottom = int(rows[0]), int(rows[-1]) + 1
            # Grouping may exclude short, separate reflection rows. Pattern
            # and contrary-stroke checks always use the untouched source mask.
            glyph = mask[top:bottom, left:right]
            h, w = glyph.shape
            view["glyph_boxes"].append([left, top, w, h])
            if h < 8 or not 0.3 <= w / h <= 0.95:
                view["reason"] = "glyph_geometry"
                break
            if left == 0 or top == 0 or right == image.shape[1] or bottom == image.shape[0]:
                view["reason"] = "source_glyph_clipped"
                break
            levels = []
            for cx, cy in (
                (0.5, 0.08),
                (0.84, 0.25),
                (0.84, 0.75),
                (0.5, 0.92),
                (0.16, 0.75),
                (0.16, 0.25),
                (0.5, 0.5),
            ):
                sx, ex = max(0, int((cx - 0.17) * w)), min(w, math.ceil((cx + 0.17) * w))
                sy, ey = max(0, int((cy - 0.1) * h)), min(h, math.ceil((cy + 0.1) * h))
                levels.append(float(glyph[sy:ey, sx:ex].mean()))
            pattern = "".join("1" if level >= 0.3 else "0" for level in levels)
            view["patterns"].append(pattern)
            expected = "1111110" if len(view["patterns"]) == 1 else "1000111"
            proof = {"basis": "native_seven_segment_probes", "matches": pattern == expected}
            if expected == "1000111":
                proof["source_f_bars"] = _off_f_bars(glyph)
                if not proof["matches"] and proof["source_f_bars"]["matches_f"]:
                    proof["basis"] = "native_source_f_bar_topology"
                proof["matches"] = proof["matches"] or proof["source_f_bars"]["matches_f"]
            view["glyph_proofs"].append(proof)
        if len(view["patterns"]) == 3 and not view.get("reason"):
            boxes = view["glyph_boxes"]
            common_height = min(top + h for _, top, _, h in boxes) - max(
                top for _, top, _, _ in boxes
            )
            if common_height < min(h for _, _, _, h in boxes) * 0.5:
                view["reason"] = "glyph_row_alignment"
        view["matches_off"] = (
            not view.get("reason")
            and len(view["glyph_proofs"]) == 3
            and all(proof["matches"] for proof in view["glyph_proofs"])
        )
        view.setdefault(
            "reason", "complete_off_bars" if view["matches_off"] else "off_pattern_mismatch"
        )
    evidence["views"] = views
    if len(views) == 2 and all(v["matches_off"] for v in views):
        evidence.update(status="consistent", reason="two_thresholds_complete_off_bars")
    else:
        evidence["reason"] = "two_thresholds_not_consistent"
    return evidence


def _native_off_geometry(image, x, y):
    """Measure display direction before authorizing rotated OCR or an OFF state."""
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.size
        or image.dtype != np.uint8
    ):
        return None
    geometry = red_display_geometry(image, x, y)
    if geometry is None:
        return None
    native_region = {
        "polygon": (np.asarray(geometry["polygon"]) - [x, y]).tolist(),
        "size": geometry["native_size"],
    }
    points = np.asarray(native_region["polygon"], np.float32)
    # The OCR region stays unchanged. OFF proof samples only the physical quad;
    # a saved size measured before corner clipping must not enlarge its pixels.
    width = min(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3]))
    height = min(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1]))
    source_size = [
        min(geometry["native_size"][0], math.floor(float(width))),
        min(geometry["native_size"][1], math.floor(float(height))),
    ]
    if min(source_size) < 2:
        off_shapes = {
            degrees: {
                "status": "inconclusive",
                "reason": "insufficient_native_source_size",
                "views": [],
            }
            for degrees in (0, 180)
        }
        return geometry, native_region, off_shapes, []
    target = np.asarray(
        [
            [0, 0],
            [source_size[0] - 1, 0],
            [source_size[0] - 1, source_size[1] - 1],
            [0, source_size[1] - 1],
        ],
        np.float32,
    )
    native_transform = cv2.getPerspectiveTransform(points, target)
    native = cv2.warpPerspective(
        image,
        native_transform,
        tuple(source_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    off_shapes = {
        degrees: _off_shape(native if degrees == 0 else cv2.rotate(native, cv2.ROTATE_180))
        for degrees in (0, 180)
    }
    for shape in off_shapes.values():
        shape["sampling"] = {"method": "source_quad_nearest", "source_size": source_size}
    native_directions = [
        degrees for degrees, shape in off_shapes.items() if shape["status"] == "consistent"
    ]
    return geometry, native_region, off_shapes, native_directions


def _red_recovery(predict, image, previous, x, y, readout_rejected=False, *, source_evidence=None):
    started = time.monotonic()
    source_evidence = source_evidence or _native_off_geometry(image, x, y)
    if source_evidence is None:
        return previous
    geometry, native_region, off_shapes, native_directions = source_evidence

    def supported(text, degrees):
        if text == "OFF":
            return len(native_directions) <= 1 and (
                not native_directions or degrees in native_directions
            )
        return degrees == 0 and not native_directions

    # Keep a successful numeric path unless independent pixels show an OFF row.
    if (
        previous.get("lines")
        and not readout_rejected
        and not any(v["status"] == "consistent" for v in off_shapes.values())
    ):
        return previous
    output = copy.deepcopy(previous)
    output["red_display_geometry"] = geometry
    output["red_display_off_shape"] = off_shapes
    native_off_pair = (
        len(native_directions) == 1 and off_shapes[native_directions[0]]["status"] == "consistent"
    )
    orientations = tuple(native_directions) if native_off_pair else (0,)
    output["red_display_retry_gate"] = {
        "orientation_degrees": list(orientations),
        "native_directions": native_directions,
        "rotated_retry_basis": "native_off_bars_required",
        "view_selection": "native_off_fixed_source_pair"
        if native_off_pair
        else "source_channel_pair",
        "view_selection_basis": "unique_complete_native_off_source_before_retry"
        if native_off_pair
        else "no_unique_complete_native_off_source",
    }
    if previous.get("digit_consistency", {}).get("native_confirmation_required") and not (
        len(native_directions) == 1
    ):
        # Further resized OCR cannot independently settle a real contradiction.
        # The native segment verifier receives both unchanged OCR candidates.
        output["red_display_retry_gate"]["skip_reason"] = "independent_native_confirmation_required"
        return output
    scale = min(8, max(1, 64 / geometry["native_size"][1]))
    size = [
        min(640, round(geometry["native_size"][0] * scale)),
        min(192, round(geometry["native_size"][1] * scale)),
    ]
    prepare_views = _off_ocr_views if native_off_pair else _red_ocr_views
    views, transform, preprocessing = prepare_views(image, dict(native_region, size=size))
    output["red_display_preprocessing"] = preprocessing
    attempts, accepted = [], []
    for degrees in orientations:
        for name, view in views:
            view = view if degrees == 0 else cv2.rotate(view, cv2.ROTATE_180)
            receipt = predict(view, 0, 0)
            attempt = {
                "view": name,
                "orientation_degrees": degrees,
                "size": size,
                "ocr": copy.deepcopy(receipt),
            }
            attempts.append(attempt)
            options = [line for line in receipt.get("lines", []) if _red_candidate(line, size)]
            if len(options) == 1:
                accepted.append((degrees, name, options[0]))
        same = [a for a in accepted if a[0] == degrees]
        if (
            len(same) == 2
            and same[0][2]["text"].strip().upper() == same[1][2]["text"].strip().upper()
        ):
            # A rotated number alone cannot establish the physical direction.
            if supported(same[0][2]["text"].strip().upper(), degrees):
                break
    output["red_display_ocr_attempts"] = attempts
    output["red_display_recovery_seconds"] = round(time.monotonic() - started, 3)
    output["red_display_orientation"] = {"status": "unconfirmed"}
    finished = next(
        (
            a["ocr"]["ocr_finished_at"]
            for a in reversed(attempts)
            if a["ocr"].get("ocr_finished_at")
        ),
        None,
    )
    if finished:
        output["ocr_finished_at"] = finished
    matches = []
    for degrees in orientations:
        same = [a for a in accepted if a[0] == degrees]
        if (
            len(same) == 2
            and same[0][2]["text"].strip().upper() == same[1][2]["text"].strip().upper()
        ):
            text = same[0][2]["text"].strip().upper()
            if not supported(text, degrees):
                continue
            matches.append((degrees, same[0][2]))
    if len(matches) == 1:
        degrees, selected = matches[0]
        source_lines = previous.get("lines", [])
        if previous.get("digit_consistency", {}).get("status") == "disagreement":
            source_lines = previous.get("panel_ocr", {}).get("lines", [])
        original_texts = (
            [
                line.get("text", "").strip()
                for line in source_lines
                if READOUT.fullmatch(line.get("text", ""))
            ]
            if not readout_rejected
            else []
        )
        selected_text = selected["text"].strip().upper()
        if (
            original_texts
            and selected_text not in original_texts
            and not (selected_text == "OFF" and off_shapes[degrees]["status"] == "consistent")
        ):
            output["lines"] = []
            output["digit_consistency"] = {
                "status": "disagreement",
                "candidates": original_texts + [selected["text"]],
                "basis": "red_geometry_retry_conflicts_with_numeric_source",
            }
            return output
        mapped = copy.deepcopy(selected)
        points = np.asarray([mapped["polygon"]], np.float32)
        if degrees:
            points[:, :, 0] = size[0] - 1 - points[:, :, 0]
            points[:, :, 1] = size[1] - 1 - points[:, :, 1]
        mapped["polygon"] = (
            cv2.perspectiveTransform(points, np.linalg.inv(transform))[0] + [x, y]
        ).tolist()
        output["digit_region"] = dict(
            geometry, status="located", size=size, method="source_red_character_geometry_v1"
        )
        if selected["text"].strip().upper() == "OFF":
            output["lines"] = []
            native_confirmed = off_shapes[degrees]["status"] == "consistent"
            output["display_state"] = {
                "text": "OFF",
                "basis": "native_off_bars_and_ocr_views"
                if native_confirmed
                else "ocr_only_with_source_geometry",
                "verification": "native_corroborated" if native_confirmed else "ocr_only",
                "raw_text": selected["text"],
                "source": "local_ocr.red_display_ocr_attempts",
                "confidence": selected["confidence"],
                "polygon": mapped["polygon"],
            }
            if original_texts and native_confirmed:
                if output.get("digit_consistency"):
                    output["digit_consistency_before_red_display"] = copy.deepcopy(
                        output["digit_consistency"]
                    )
                output["digit_consistency"] = {
                    "status": "resolved",
                    "candidates": original_texts + [selected["text"]],
                    "basis": "native_off_bars_and_two_ocr_views",
                }
            output["red_display_orientation"] = {
                "status": "consistent"
                if off_shapes[degrees]["status"] == "consistent"
                else "ocr_only",
                "orientation_degrees": degrees,
                "basis": "native_off_bars_and_two_ocr_views"
                if off_shapes[degrees]["status"] == "consistent"
                else "two_ocr_views_direction_unverified",
            }
        elif not previous.get("digit_consistency", {}).get("status") == "disagreement":
            output["lines"] = [mapped]
    elif native_directions and previous.get("lines"):
        output["lines"] = []
        output["digit_consistency"] = {
            "status": "disagreement",
            "candidates": list(
                dict.fromkeys(
                    [line["text"] for line in previous["lines"]] + [a[2]["text"] for a in accepted]
                )
            ),
            "basis": "native_off_bars_conflict_with_numeric_ocr",
        }
    return output


def _red_candidate(line, size):
    text = line.get("text")
    if not isinstance(text, str) or not (READOUT.fullmatch(text) or text.strip().upper() == "OFF"):
        return False
    try:
        confidence = float(line.get("confidence") or 0)
        polygon = np.asarray(line.get("polygon"), dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(confidence)
        and confidence >= 0.6
        and polygon.shape == (4, 2)
        and np.isfinite(polygon).all()
        and np.all(polygon >= -1)
        and np.all(polygon <= np.asarray(size) + 1)
        and cv2.isContourConvex(polygon.astype(np.float32))
        and cv2.contourArea(polygon.astype(np.float32)) >= 4
    )


def refine_digits(
    predict,
    image,
    initial,
    x=0,
    y=0,
    *,
    source_image=None,
    source_offset=(0, 0),
    recover_small=False,
    recover_red=False,
    red_readout_rejected=False,
    require_native_conflict_confirmation=False,
):
    source_evidence = None
    off_source = initial.get("lines", [])
    if recover_red and initial.get("status") == "completed" and len(off_source) == 1:
        raw_off = off_source[0]
        try:
            confidence = float(raw_off.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0
        if (
            isinstance(raw_off.get("text"), str)
            and raw_off["text"].strip().upper() == "OFF"
            and math.isfinite(confidence)
            and confidence >= 0.6
            and not red_readout_rejected
        ):
            source_evidence = _native_off_geometry(image, x, y)
            directions = source_evidence[3] if source_evidence else []
            if not directions or directions == [0]:
                # This is the actual first-pass OFF receipt, not a synthetic
                # consensus. Native direction may corroborate it or abstain.
                output = copy.deepcopy(initial)
                output.update(
                    panel_ocr=copy.deepcopy(initial),
                    lines=[],
                    digit_region={"status": "not_located", "method": "dominant_numeric_line_v1"},
                    display_state={
                        "text": "OFF",
                        "basis": "recognized_display_text",
                        "verification": "native_corroborated" if directions else "ocr_only",
                        "raw_text": raw_off["text"],
                        "source": "local_ocr.panel_ocr",
                        "confidence": raw_off["confidence"],
                    },
                    red_display_fastpath={"basis": "original_exact_off", "adapter_retries": 0},
                )
                if raw_off.get("polygon") is not None:
                    output["display_state"]["polygon"] = copy.deepcopy(raw_off["polygon"])
                if source_evidence:
                    output["red_display_geometry"] = source_evidence[0]
                    output["red_display_off_shape"] = source_evidence[2]
                if directions:
                    output["red_display_orientation"] = {
                        "status": "consistent",
                        "orientation_degrees": 0,
                        "basis": "native_off_bars_and_original_ocr",
                    }
                return output
    result = _refine_numeric_digits(
        predict,
        image,
        initial,
        x,
        y,
        source_image=source_image,
        source_offset=source_offset,
        recover_small=recover_small,
        require_native_conflict_confirmation=(
            require_native_conflict_confirmation and not red_readout_rejected
        ),
    )
    if recover_red and initial.get("status") == "completed":
        if source_evidence and (180 in source_evidence[3] or len(source_evidence[3]) > 1):
            result["display_state_rejection"] = {
                "text": "OFF",
                "basis": "native_off_direction_conflict",
                "native_directions": source_evidence[3],
            }
        return _red_recovery(
            predict,
            image,
            result,
            x,
            y,
            red_readout_rejected,
            source_evidence=source_evidence,
        )
    return result


def _refine_numeric_digits(
    predict,
    image,
    initial,
    x=0,
    y=0,
    *,
    source_image=None,
    source_offset=(0, 0),
    recover_small=False,
    require_native_conflict_confirmation=False,
):
    """Use measured glyph geometry; never insert a decimal point or a unit."""
    candidates = []
    for line in initial.get("lines", []):
        polygon = np.asarray(line.get("polygon"), dtype=np.float32)
        text = line.get("text", "")
        # A partly misread number can still locate the glyph row. This is only
        # a geometry candidate; ambiguous text is never returned as a reading.
        numeric_like = bool(re.search(r"\d", text) and re.fullmatch(r"[\d\s+\-.,:OoIl|S]+", text))
        if (
            not (READOUT.fullmatch(text) or numeric_like)
            or polygon.shape != (4, 2)
            or not np.isfinite(polygon).all()
            or (line.get("confidence") or 0) < 0.45
        ):
            continue
        width = (
            np.linalg.norm(polygon[1] - polygon[0]) + np.linalg.norm(polygon[2] - polygon[3])
        ) / 2
        height = (
            np.linalg.norm(polygon[3] - polygon[0]) + np.linalg.norm(polygon[2] - polygon[1])
        ) / 2
        if min(width, height) < 3:
            continue
        # Main measurement glyphs dominate small menu/status numbers. No fixed
        # screen coordinates, so a tilted or displaced panel remains supported.
        candidates.append((float(height * math.sqrt(width)), line, polygon, width, height))
    output = copy.deepcopy(initial)
    output["panel_ocr"] = copy.deepcopy(initial)
    if not candidates:
        if (
            recover_small
            and initial.get("status") == "completed"
            and min(image.shape[:2]) < 160
            and max(image.shape[:2]) < 480
        ):
            # One bounded retry on the already authorized LCD crop. No whole
            # image OCR, super-resolution model, or inferred missing glyphs.
            enlarged = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            attempt = predict(enlarged, 0, 0)
            mapped = copy.deepcopy(attempt)
            for line in mapped.get("lines", []):
                if line.get("polygon") is not None:
                    line["polygon"] = (np.asarray(line["polygon"]) / 2 + [x, y]).tolist()
            recovered = refine_digits(
                predict, image, mapped, x, y, source_image=source_image, source_offset=source_offset
            )
            recovered.update(
                panel_ocr=copy.deepcopy(initial),
                small_panel_ocr=attempt,
                small_panel_recovery={"method": "authorized_lcd_2x_v1", "scale": 2},
            )
            return recovered
        output.update(
            lines=[], digit_region={"status": "not_located", "method": "dominant_numeric_line_v1"}
        )
        return output
    _, selected, polygon, width, height = max(candidates, key=lambda item: item[0])
    points = polygon - [x, y]
    center = points.mean(axis=0)
    points = center + (points - center) * 1.15
    points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)
    scale = min(4.0, max(1.0, 64.0 / height))
    size = [max(8, round(width * 1.15 * scale)), max(8, round(height * 1.15 * scale))]
    region = {
        "status": "located",
        "method": "dominant_numeric_line_v1",
        "polygon": (points + [x, y]).tolist(),
        "size": size,
        "initial_text": selected["text"],
    }
    crop, transform = extract_digits(image, dict(region, polygon=points.tolist()))
    started = time.monotonic()
    refined = predict(crop, 0, 0)
    lines = []
    inverse = np.linalg.inv(transform)
    for line in refined.get("lines", []):
        if not READOUT.fullmatch(line.get("text", "")) or (line.get("confidence") or 0) < 0.45:
            continue
        line = copy.deepcopy(line)
        if line.get("polygon") is not None:
            mapped = cv2.perspectiveTransform(np.asarray([line["polygon"]], np.float32), inverse)[0]
            line["polygon"] = (mapped + [x, y]).tolist()
        lines.append(line)
    # A clipped/warped glyph must not replace a contradictory original reading
    # solely because it came from the second pass. Check a wider source crop.
    best = max(lines, key=lambda line: line.get("confidence", 0)) if lines else None
    original_valid = bool(READOUT.fullmatch(selected["text"]))
    conflict = (
        best is not None and original_valid and best["text"].strip() != selected["text"].strip()
    )
    alternative = None
    if (
        source_image is not None
        and (not best or conflict)
        and not (conflict and require_native_conflict_confirmation)
    ):
        offset = np.asarray(source_offset)
        lo, hi = polygon.min(0) - offset, polygon.max(0) - offset
        margin = (hi - lo) * 0.3
        sx, sy = np.maximum(0, np.floor(lo - margin)).astype(int)
        ex, ey = np.minimum(
            [source_image.shape[1], source_image.shape[0]], np.ceil(hi + margin)
        ).astype(int)
        if ex > sx and ey > sy:
            rectangle = cv2.resize(
                source_image[sy:ey, sx:ex], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )
            rectangle = cv2.copyMakeBorder(rectangle, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
            check = predict(rectangle, 0, 0)
            output["digit_ocr_check"] = copy.deepcopy(check)
            options = [
                l
                for l in check.get("lines", [])
                if READOUT.fullmatch(l.get("text", "")) and (l.get("confidence") or 0) >= 0.45
            ]
            if options:
                alternative = copy.deepcopy(max(options, key=lambda l: l.get("confidence", 0)))
                if alternative.get("polygon") is not None:
                    alternative["polygon"] = (
                        (np.asarray(alternative["polygon"]) - 12) / 3 + [sx, sy] + offset
                    ).tolist()
    if conflict:
        conflicting_text = best["text"]
        best = (
            None
            if require_native_conflict_confirmation
            else next(
                (
                    candidate
                    for candidate in (selected, best)
                    if alternative and candidate["text"].strip() == alternative["text"].strip()
                ),
                None,
            )
        )
        output["digit_consistency"] = {
            "status": "resolved" if best else "disagreement",
            "candidates": [selected["text"], conflicting_text],
            "check_text": alternative and alternative["text"],
        }
        if require_native_conflict_confirmation:
            output["digit_consistency"].update(
                basis="independent_native_confirmation_required", native_confirmation_required=True
            )
    elif not best and alternative:
        # A failed perspective pass does not make the wider crop authoritative.
        # Keep conflicting complete readings unresolved (e.g. OFF -> 055/065).
        if original_valid and selected["text"].strip() != alternative["text"].strip():
            conflict = True
            output["digit_consistency"] = {
                "status": "disagreement",
                "candidates": [selected["text"], alternative["text"]],
                "check_text": alternative["text"],
            }
        else:
            best = alternative
    if best and alternative and best["text"].strip() == alternative["text"].strip():
        # Save the wider source window actually used to resolve the reading,
        # rather than the clipped perspective crop which disagreed with it.
        region.update(
            method="dominant_numeric_line_source_check_v1",
            polygon=(
                np.asarray([[sx, sy], [ex - 1, sy], [ex - 1, ey - 1], [sx, ey - 1]]) + offset
            ).tolist(),
            size=[int(ex - sx) * 3, int(ey - sy) * 3],
        )
    # Keep only the strongest complete line in this isolated number window.
    fallback = [copy.deepcopy(selected)] if READOUT.fullmatch(selected["text"]) else []
    output["lines"] = [copy.deepcopy(best)] if best else [] if conflict else fallback
    output.update(
        digit_region=region,
        digit_ocr=refined,
        digit_refinement_seconds=round(time.monotonic() - started, 3),
    )
    output["digit_region"]["refined_readout_available"] = bool(best)
    finished = output.get("digit_ocr_check", {}).get("ocr_finished_at") or refined.get(
        "ocr_finished_at"
    )
    if finished:
        output["ocr_finished_at"] = finished
    return output
