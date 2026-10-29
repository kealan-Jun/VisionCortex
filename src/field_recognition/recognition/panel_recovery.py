"""Bounded exposure/tilt recovery for the registered A/B display detector.

Enhanced images locate boxes only. OCR and evidence always use source pixels.
"""

import math

import cv2
import numpy as np


VERSION = "exposure_tilt_v1"
GAMMA_LUT = np.uint8((np.arange(256) / 255.0) ** 2.2 * 255)


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def source_box(box, matrix, width, height):
    x1, y1, x2, y2 = box
    if matrix is not None:
        corners = np.asarray([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float32)
        points = cv2.transform(corners, cv2.invertAffineTransform(matrix))[0]
        x1, y1, x2, y2 = (
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max(),
        )
    return [max(0, float(x1)), max(0, float(y1)), min(width, float(x2)), min(height, float(y2))]


def has_display_color(hsv, box):
    """A red windows / B blue LCD context; reject recovered sleeve/desk boxes."""
    x1, y1, x2, y2 = box["xyxy"]
    pad = max(2, round(min(x2 - x1, y2 - y1) * 0.15))
    crop = hsv[
        max(0, math.floor(y1) - pad) : min(hsv.shape[0], math.ceil(y2) + pad),
        max(0, math.floor(x1) - pad) : min(hsv.shape[1], math.ceil(x2) + pad),
    ]
    if not crop.size:
        return False
    hue, saturation, value = cv2.split(crop)
    color = ((hue <= 12) | (hue >= 170)) if box["class_id"] == 0 else ((hue >= 90) & (hue <= 135))
    return float(np.mean(color & (saturation >= 30) & (value >= 35))) >= 0.06


def window_pixels(hsv, coords):
    """Independent sparse red glyph/dark-window evidence, not an OCR identity."""
    x1, y1, x2, y2 = [round(v) for v in coords]
    pad = max(2, math.ceil(min(x2 - x1, y2 - y1) * 0.25))
    if x1 - pad < 0 or y1 - pad < 0 or x2 + pad > hsv.shape[1] or y2 + pad > hsv.shape[0]:
        return None
    if min(x2 - x1, y2 - y1) < 8:
        return None
    # Detector boxes can tightly bound illuminated characters. Inspect the
    # surrounding window on original pixels without widening the OCR/evidence crop.
    crop = hsv[y1 - pad : y2 + pad, x1 - pad : x2 + pad]
    hue, saturation, value = cv2.split(crop)
    red = ((hue <= 12) | (hue >= 170)) & (saturation >= 30) & (value >= 35)
    inner_red_fraction = float(red[pad:-pad, pad:-pad].mean())
    # Relative contrast survives exposure changes; a uniform red object cannot
    # supply a distinct dark-window population merely by being bright or dim.
    contrast = float(np.percentile(value, 90) - np.percentile(value, 10))
    threshold, _ = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    red_fraction, dark_fraction = float(red.mean()), float((value <= threshold).mean())
    ys, xs = np.where(red)
    if (
        red_fraction < 0.06
        # Padding must not dilute a solid red object into apparent glyphs.
        or inner_red_fraction >= 0.9
        or dark_fraction < 0.25
        or contrast < 35
        or not xs.size
        or np.ptp(xs) < 0.2 * crop.shape[1]
        or np.ptp(ys) < 0.25 * crop.shape[0]
    ):
        return None
    evidence = {
        "red_fraction": round(red_fraction, 4),
        "inner_red_fraction": round(inner_red_fraction, 4),
        "dark_fraction": round(dark_fraction, 4),
        "value_contrast": round(contrast, 2),
        "dark_threshold": threshold,
        "evidence_bbox": [x1 - pad, y1 - pad, x2 + pad, y2 + pad],
    }
    if red_fraction > 0.5:
        # A dim red surround can share the glyph hue. It remains part of the
        # original solid-red counterevidence above, but does not establish
        # bright source strokes. Split it using the existing dark threshold.
        bright = red & (value > threshold)
        bys, bxs = np.where(bright)
        bright_fraction = float(bright.mean())
        if not bxs.size:
            return None
        x_span, y_span = int(np.ptp(bxs)), int(np.ptp(bys))
        fill = int(bright.sum()) / ((x_span + 1) * (y_span + 1))
        if (
            not 0.06 <= bright_fraction <= 0.5
            or x_span < 0.2 * crop.shape[1]
            or y_span < 0.25 * crop.shape[0]
            # Thresholding a red gradient/label must not manufacture a solid
            # bright block and claim sparse native display strokes.
            or fill >= 0.9
        ):
            return None
        evidence.update(
            pixel_basis="source_bright_red_otsu",
            bright_red_fraction=round(bright_fraction, 4),
            inner_bright_red_fraction=round(float(bright[pad:-pad, pad:-pad].mean()), 4),
            bright_native_span=[x_span, y_span],
            bright_native_fill=round(fill, 4),
            red_hue_also_dark_count=int((red & (value <= threshold)).sum()),
        )
    return evidence


def _pair_geometry_reason(anchor, box):
    """The same source window/layout contract governs both photo recoveries."""
    from field_recognition.recognition.panel_layout import roles_for_boxes

    x1, y1, x2, y2 = anchor["xyxy"]
    bx1, by1, bx2, by2 = box["xyxy"]
    width, height = x2 - x1, y2 - y1
    gap = max(bx1 - x2, x1 - bx2, 0)
    gap_limit = max(width, bx2 - bx1)
    box.update(pair_horizontal_gap=gap, pair_horizontal_gap_limit=gap_limit)
    if not (0.6 <= (bx2 - bx1) / width <= 1.7 and 0.65 <= (by2 - by1) / height <= 1.5):
        return "pair_window_size_mismatch"
    if gap > gap_limit:
        return "pair_window_gap_exceeded"
    if len(roles_for_boxes([anchor, box])) != 2:
        return "pair_layout_unconfirmed"
    return None


def weak_crop_boundary_issue(image, box, crop_bbox):
    """Reject evidenced leading-stroke clipping, without deriving a readout.

    Only the new weak-pair path uses this guard. Its wider native window
    evidence can establish that an OCR crop cuts a connected bright structure;
    it does not authorize expanding the crop into the neighbouring window.
    """
    evidence = []
    for item in box.get("pair_window_pixels", {}).values():
        bounds = item.get("evidence_bbox") if isinstance(item, dict) else None
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            continue
        try:
            if all(math.isfinite(v) for v in bounds):
                evidence.append(bounds)
        except TypeError:
            continue
    if not evidence:
        return None
    bx1, by1, bx2, by2 = box["xyxy"]
    center = (bx1 + bx2) / 2
    bounds = min(evidence, key=lambda coords: abs((coords[0] + coords[2]) / 2 - center))
    x1, y1, x2, y2 = map(int, bounds)
    cx, cy, ex, ey = crop_bbox
    if not (0 <= x1 < cx < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        return None
    hue, saturation, value = cv2.split(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV))
    threshold, _ = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = ((hue <= 12) | (hue >= 170)) & (saturation >= 30) & (value > threshold)
    # Do not combine another row, or rely on red hue in a dim surround. These
    # are unscaled source pixels, with no closing/painted bridges between them.
    bright[: max(0, cy - y1)] = False
    bright[max(0, ey - y1) :] = False
    _, _, components, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), 8)
    minimum_height = max(8, math.ceil((by2 - by1) * 0.5))
    crossing = []
    for px, py, width, height, area in components[1:]:
        sx, sy = x1 + int(px), y1 + int(py)
        fill = int(area) / (int(width) * int(height))
        if sx < cx < sx + width - 1 and height >= minimum_height and fill < 0.9:
            crossing.append(
                {
                    "source_bbox": [sx, sy, sx + int(width), sy + int(height)],
                    "native_height": int(height),
                    "source_pixel_count": int(area),
                    "fill": round(fill, 4),
                }
            )
    if not crossing:
        return None
    return {
        "status": "rejected",
        "reason": "source_crop_truncated",
        "basis": "native_bright_red_component_crosses_left_crop_boundary",
        "source_crop_bbox": list(crop_bbox),
        "evidence_bbox": [x1, y1, x2, y2],
        "source_value_otsu_threshold": threshold,
        "minimum_native_height": minimum_height,
        "crossing_components": crossing,
    }


def photo_pair_context(image, hsv, anchor, infer, imgsz, confidence, existing_boxes=()):
    """Two actual views corroborate a partner, using at most three forwards."""
    diagnostics = {"attempted": False, "passes": [], "excluded_candidates": []}
    result = {"box": None, "speed_ms": {}, "diagnostics": diagnostics}
    anchor_pixels = window_pixels(hsv, anchor["xyxy"])
    if not anchor_pixels or anchor["confidence"] < confidence:
        diagnostics["reason"] = "anchor_window_evidence_missing"
        return result
    x1, y1, x2, y2 = anchor["xyxy"]
    width, height = x2 - x1, y2 - y1
    # Preserve the successful continuous-box context. Its canonical pixel
    # envelope is a bounded alternative only after an eligible set is empty.
    cx1, cy1, cx2, cy2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    context_width, context_height = cx2 - cx1, cy2 - cy1
    diagnostics["context_anchor_bbox"] = [cx1, cy1, cx2, cy2]
    diagnostics["context_coordinate_basis"] = "continuous_source_box"
    diagnostics["attempted"] = True
    candidates = []
    view_signatures = {}
    eligible_by_view = {}

    # A character-only crop removes the chassis context learned by this model.
    # Two fields of view retain the device while exposing distinct model scales.
    def context(name, horizontal, vertical, size, canonical=False):
        ax1, ay1, ax2, ay2 = (cx1, cy1, cx2, cy2) if canonical else (x1, y1, x2, y2)
        aw, ah = (context_width, context_height) if canonical else (width, height)
        return {
            "view": name,
            "context_bbox": [
                max(0, math.floor(ax1 - horizontal * aw)),
                max(0, math.floor(ay1 - vertical * ah)),
                min(image.shape[1], math.ceil(ax2 + horizontal * aw)),
                min(image.shape[0], math.ceil(ay2 + vertical * ah)),
            ],
            "imgsz": size,
            "context_basis": "source_pixel_floor_ceil" if canonical else "continuous_source_box",
        }

    def skip(view, reason, **extra):
        diagnostics.setdefault("skipped_views", []).append(view | {"reason": reason} | extra)

    def run_view(view):
        name, size = view["view"], view["imgsz"]
        left, top, right, bottom = view["context_bbox"]
        signature = (left, top, right, bottom, size)
        previous = next((v for v, s in view_signatures.items() if s == signature), None)
        if previous is not None:
            skip(view, "duplicate_context_views", same_as_view=previous)
            diagnostics["excluded_candidates"].extend(
                b | {"reason": "duplicate_context_views"} for b in eligible_by_view[previous]
            )
            return []
        view_signatures[name] = (left, top, right, bottom, size)
        pixels = cv2.LUT(image[top:bottom, left:right], GAMMA_LUT)
        detection = infer(pixels, size, confidence)
        receipt = view | {"candidate_count": len(detection["boxes"])}
        diagnostics["passes"].append(receipt)
        for key, value in detection.get("speed_ms", {}).items():
            result["speed_ms"][key] = result["speed_ms"].get(key, 0) + value
        eligible = []
        for item in detection["boxes"]:
            box = dict(item, view=name)
            box["xyxy"] = [
                item["xyxy"][0] + left,
                item["xyxy"][1] + top,
                item["xyxy"][2] + left,
                item["xyxy"][3] + top,
            ]
            bx1, by1, bx2, by2 = box["xyxy"]
            geometry_reason = _pair_geometry_reason(anchor, box)
            reason = (
                "wrong_pair_class"
                if box["class_id"] != anchor["class_id"]
                else "pair_context_out_of_bounds"
                if not (left <= bx1 < bx2 <= right and top <= by1 < by2 <= bottom)
                else "below_recovery_confidence"
                if box["confidence"] < confidence
                else "duplicate_region"
                if iou(box["xyxy"], anchor["xyxy"]) >= 0.4
                else "duplicate_existing_region"
                if any(iou(box["xyxy"], b["xyxy"]) >= 0.4 for b in existing_boxes)
                else geometry_reason
                if geometry_reason
                else "pair_window_evidence_missing"
                if not (pixels_evidence := window_pixels(hsv, box["xyxy"]))
                else None
            )
            if reason:
                diagnostics["excluded_candidates"].append(box | {"reason": reason})
            else:
                eligible.append(box | {"window_pixels": pixels_evidence})
        candidates.extend(eligible)
        eligible_by_view[name] = eligible
        receipt["eligible_partner_count"] = len(eligible)
        return eligible

    medium = context("pair_gamma_medium", 6, 8, min(imgsz, 640))
    wide = context("pair_gamma_wide", 10, 12, imgsz)
    canonical_medium = context("pair_gamma_medium_canonical", 6, 8, min(imgsz, 640), True)
    canonical_wide = context("pair_gamma_wide_canonical", 10, 12, imgsz, True)
    if not run_view(medium):
        diagnostics["fallback_trigger"] = "float_medium_no_eligible_partner"
        skip(wide, "no_eligible_partner_first_view")
        if not run_view(canonical_medium):
            diagnostics["reason"] = "no_eligible_partner_first_view"
            skip(canonical_wide, "no_eligible_partner_first_view")
            return result
        run_view(canonical_wide)
    elif not run_view(wide):
        # A failed wide view may omit a partner visible in the medium view.
        # An eligible but contrary wide view is evidence, never outvoted by
        # an extra crop. Only an empty eligible set authorizes this fallback.
        diagnostics["fallback_trigger"] = "float_wide_no_eligible_partner"
        run_view(canonical_wide)
    consensus = []
    for box in sorted(candidates, key=lambda b: b["confidence"], reverse=True):
        if any(iou(box["xyxy"], b["xyxy"]) >= 0.5 for b in consensus):
            continue
        support = sorted({b["view"] for b in candidates if iou(box["xyxy"], b["xyxy"]) >= 0.5})
        if len(support) != 2:
            diagnostics["excluded_candidates"].append(
                box | {"reason": "insufficient_supporting_views"}
            )
            continue
        if len({view_signatures[name] for name in support}) < 2:
            diagnostics["excluded_candidates"].append(box | {"reason": "duplicate_context_views"})
            continue
        consensus.append(box | {"supporting_views": support})
    diagnostics["reason"] = (
        "recovered"
        if len(consensus) == 1
        else "ambiguous_pair_partner"
        if consensus
        else "no_pair_consensus"
    )
    if len(consensus) == 1:
        box = consensus[0]
        result["box"] = {k: v for k, v in box.items() if k not in {"view", "window_pixels"}} | {
            "localization_method": "photo_pair_context_v2",
            "pair_anchor_xyxy": list(anchor["xyxy"]),
            "pair_window_pixels": {"anchor": anchor_pixels, "partner": box["window_pixels"]},
            "pair_layout_basis": "model_horizontal_pair_contract",
        }
    return result


def photo_weak_pair_context(image, hsv, candidates, infer, imgsz, confidence, existing_boxes=()):
    """Weak votes propose one context; two new full-confidence pairs must agree.

    A proposal is never an accepted box or a field role. This photo-only path
    adds at most two forwards and cannot promote a lone local window.
    """
    diagnostics = {
        "attempted": False,
        "passes": [],
        "excluded_candidates": [],
        "maximum_forwards": 2,
        "proposal_only": True,
    }
    result = {"boxes": [], "speed_ms": {}, "diagnostics": diagnostics}
    eligible = [
        b
        for b in candidates
        if b["class_id"] == 0
        and max(0.1, confidence * 0.4) <= b["confidence"] < confidence
        and all(math.isfinite(v) for v in b["xyxy"])
        and window_pixels(hsv, b["xyxy"])
    ]
    proposals = []
    for box in sorted(eligible, key=lambda b: b["confidence"], reverse=True):
        if any(iou(box["xyxy"], b["xyxy"]) >= 0.5 for b in proposals):
            continue
        support = [b for b in eligible if iou(box["xyxy"], b["xyxy"]) >= 0.5]
        if len({b["view"] for b in support}) >= 2:
            proposals.append(box | {"supporting_views": sorted({b["view"] for b in support})})
    if len(proposals) != 1:
        diagnostics["reason"] = "ambiguous_weak_context" if proposals else "no_weak_context"
        diagnostics["proposal_count"] = len(proposals)
        return result
    cue = proposals[0]
    diagnostics.update(
        attempted=True, proposal=cue, proposal_window_pixels=window_pixels(hsv, cue["xyxy"])
    )
    x1, y1, x2, y2 = cue["xyxy"]
    width, height = x2 - x1, y2 - y1
    signatures = set()
    pairs = []

    def context(name, horizontal, vertical, size):
        left, top, right, bottom = (
            max(0, math.floor(x1 - horizontal * width)),
            max(0, math.floor(y1 - vertical * height)),
            min(image.shape[1], math.ceil(x2 + horizontal * width)),
            min(image.shape[0], math.ceil(y2 + vertical * height)),
        )
        return {
            "view": name,
            "context_bbox": [left, top, right, bottom],
            "imgsz": size,
            "confidence_threshold": confidence,
            "context_basis": "continuous_source_box",
            "view_to_source_matrix": [[1, 0, left], [0, 1, top]],
        }

    def skip(view, reason):
        diagnostics.setdefault("skipped_views", []).append(view | {"reason": reason})

    def run_view(view):
        name, size = view["view"], view["imgsz"]
        left, top, right, bottom = view["context_bbox"]
        signature = (left, top, right, bottom, size)
        if signature in signatures:
            skip(view, "duplicate_context_views")
            return {"eligible_window_count": 0, "pair_status": "duplicate_context_views"}
        signatures.add(signature)
        detection = infer(cv2.LUT(image[top:bottom, left:right], GAMMA_LUT), size, confidence)
        receipt = view | {"candidate_count": len(detection["boxes"]), "model_candidates": []}
        diagnostics["passes"].append(receipt)
        for key, value in detection.get("speed_ms", {}).items():
            result["speed_ms"][key] = result["speed_ms"].get(key, 0) + value
        windows = []
        for item in detection["boxes"]:
            box = dict(
                item,
                xyxy=[v + (left if i % 2 == 0 else top) for i, v in enumerate(item["xyxy"])],
                view=name,
            )
            receipt["model_candidates"].append(dict(item, source_xyxy=box["xyxy"]))
            bx1, by1, bx2, by2 = box["xyxy"]
            reason = (
                "wrong_pair_class"
                if box["class_id"] != 0
                else "pair_context_out_of_bounds"
                if not (left <= bx1 < bx2 <= right and top <= by1 < by2 <= bottom)
                else "below_recovery_confidence"
                if not math.isfinite(box["confidence"]) or box["confidence"] < confidence
                else "duplicate_existing_region"
                if any(iou(box["xyxy"], b["xyxy"]) >= 0.4 for b in existing_boxes)
                else "pair_window_evidence_missing"
                if not (evidence := window_pixels(hsv, box["xyxy"]))
                else None
            )
            if reason:
                diagnostics["excluded_candidates"].append(box | {"reason": reason})
            else:
                windows.append(box | {"window_pixels": evidence})
        receipt["eligible_window_count"] = len(windows)
        receipt["eligible_windows"] = windows
        # Do not select a convenient pair among several devices, overlapping
        # windows or adjacent rows. Each actual view must supply one full pair.
        reason = (
            "ambiguous_pair_windows"
            if len(windows) > 2
            else "incomplete_pair"
            if len(windows) < 2
            else None
        )
        if reason is None:
            windows.sort(key=lambda b: b["xyxy"][0])
            reason = _pair_geometry_reason(windows[0], windows[1])
            if reason is None and not any(iou(b["xyxy"], cue["xyxy"]) >= 0.4 for b in windows):
                reason = "proposal_window_not_reconfirmed"
        receipt["pair_status"] = reason or "eligible"
        if reason is None:
            pairs.append((name, windows))
        return receipt

    medium = context("weak_pair_gamma_medium", 6, 8, min(imgsz, 640))
    wide = context("weak_pair_gamma_wide", 10, 12, imgsz)
    first = run_view(medium)
    if first["pair_status"] == "eligible":
        run_view(wide)
    else:
        skip(wide, "no_complete_pair_first_view")
    # Multiple windows or a two-window geometry/cue contradiction cannot
    # authorize another crop that discards the contrary source evidence.
    if len(pairs) != 2 or not all(
        iou(a["xyxy"], b["xyxy"]) >= 0.5 for a, b in zip(pairs[0][1], pairs[1][1])
    ):
        diagnostics["reason"] = "no_complete_pair_consensus"
        return result
    _, windows = max(pairs, key=lambda pair: min(b["confidence"] for b in pair[1]))
    support = [name for name, _ in pairs]
    result["boxes"] = [
        {k: v for k, v in box.items() if k not in {"view", "window_pixels"}}
        | {
            "localization_method": "photo_weak_pair_context_v1",
            "supporting_views": support,
            "pair_proposal_xyxy": list(cue["xyxy"]),
            "pair_window_pixels": {
                "left": windows[0]["window_pixels"],
                "right": windows[1]["window_pixels"],
            },
            "pair_layout_basis": "model_horizontal_pair_contract",
        }
        for box in windows
    ]
    diagnostics["reason"] = "recovered"
    return result


def detect(image, infer, *, imgsz=960, confidence=0.25, photo=False):
    """Four video forwards; photos add at most three local and eight closeup calls."""
    height, width = image.shape[:2]
    primary = infer(image, imgsz, confidence)
    accepted = [dict(b, localization_method="original") for b in primary["boxes"]]
    recovery = {"version": VERSION, "attempted": False, "passes": 1, "recovered_count": 0}
    diagnostics = {
        "raw_candidate_count": len(primary["boxes"]),
        "raw_count_basis": "model_returned_boxes_after_model_threshold_and_nms",
        "passes": [{"view": "original", "candidate_count": len(primary["boxes"])}],
        "excluded_candidates": [],
    }

    def exclude(box, reason):
        diagnostics["excluded_candidates"].append(
            {k: box[k] for k in ("class_id", "xyxy", "confidence", "view") if k in box}
            | {"reason": reason}
        )

    def output():
        diagnostics["quality_filtered_count"] = len(accepted)
        return {
            "boxes": accepted,
            "speed_ms": speed,
            "recovery": recovery,
            "diagnostics": diagnostics,
        }

    speed = dict(primary.get("speed_ms", {}))
    # Existing full detections win over augmented duplicates, preserving crops.
    if sum(b["class_id"] == 0 for b in accepted) >= 2 and any(b["class_id"] == 1 for b in accepted):
        return output()
    recovery["attempted"] = True
    dark = cv2.LUT(image, GAMMA_LUT)
    views = [("gamma_scale", dark, min(imgsz, 800), None)]
    for angle in (6, -6):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        pixels = cv2.warpAffine(dark, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        views.append((f"gamma_tilt_{angle}", pixels, imgsz, matrix))
    candidates = []
    for name, pixels, size, matrix in views:
        result = infer(pixels, size, max(0.1, confidence * 0.4))
        recovery["passes"] += 1
        diagnostics["raw_candidate_count"] += len(result["boxes"])
        diagnostics["passes"].append({"view": name, "candidate_count": len(result["boxes"])})
        for key, value in result.get("speed_ms", {}).items():
            speed[key] = speed.get(key, 0) + value
        for box in result["boxes"]:
            mapped = source_box(box["xyxy"], matrix, width, height)
            if mapped[2] - mapped[0] >= 4 and mapped[3] - mapped[1] >= 4:
                candidates.append(dict(box, xyxy=mapped, view=name))
            else:
                exclude(dict(box, xyxy=mapped, view=name), "region_too_small")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for box in sorted(candidates, key=lambda b: b["confidence"], reverse=True):
        if box["confidence"] < confidence or len(accepted) >= 6:
            exclude(
                box,
                "below_recovery_confidence"
                if box["confidence"] < confidence
                else "candidate_limit",
            )
            continue
        if any(iou(box["xyxy"], b["xyxy"]) >= 0.4 for b in accepted):
            exclude(box, "duplicate_region")
            continue
        supporting = sorted(
            {
                b["view"]
                for b in candidates
                if b["class_id"] == box["class_id"] and iou(box["xyxy"], b["xyxy"]) >= 0.5
            }
        )
        if len(supporting) < 2 or not has_display_color(hsv, box):
            exclude(
                box,
                "insufficient_supporting_views"
                if len(supporting) < 2
                else "display_color_mismatch",
            )
            continue
        accepted.append(
            {k: v for k, v in box.items() if k != "view"}
            | {"localization_method": VERSION, "supporting_views": supporting}
        )
        recovery["recovered_count"] += 1
    anchors = [b for b in accepted if b["class_id"] == 0]
    if photo and not anchors and len(accepted) <= 4:
        pair = photo_weak_pair_context(image, hsv, candidates, infer, imgsz, confidence, accepted)
        pair_diagnostics = pair["diagnostics"]
        diagnostics["photo_weak_pair_context"] = pair_diagnostics
        recovery["passes"] += len(pair_diagnostics["passes"])
        diagnostics["passes"].extend(pair_diagnostics["passes"])
        diagnostics["raw_candidate_count"] += sum(
            p["candidate_count"] for p in pair_diagnostics["passes"]
        )
        diagnostics["excluded_candidates"].extend(pair_diagnostics["excluded_candidates"])
        for key, value in pair["speed_ms"].items():
            speed[key] = speed.get(key, 0) + value
        accepted.extend(pair["boxes"])
        recovery["recovered_count"] += len(pair["boxes"])
    if photo and len(anchors) == 1 and len(accepted) < 6:
        pair = photo_pair_context(image, hsv, anchors[0], infer, imgsz, confidence, accepted)
        pair_diagnostics = pair["diagnostics"]
        diagnostics["photo_pair_context"] = pair_diagnostics
        recovery["passes"] += len(pair_diagnostics["passes"])
        diagnostics["passes"].extend(pair_diagnostics["passes"])
        diagnostics["raw_candidate_count"] += sum(
            p["candidate_count"] for p in pair_diagnostics["passes"]
        )
        diagnostics["excluded_candidates"].extend(pair_diagnostics["excluded_candidates"])
        for key, value in pair["speed_ms"].items():
            speed[key] = speed.get(key, 0) + value
        if pair["box"]:
            if any(iou(pair["box"]["xyxy"], b["xyxy"]) >= 0.4 for b in accepted):
                pair_diagnostics["reason"] = "duplicate_existing_region"
                exclude(pair["box"], "duplicate_existing_region")
            else:
                accepted.append(pair["box"])
                recovery["recovered_count"] += 1
    if photo and sum(b["class_id"] == 0 for b in accepted) < 2:
        # Close photographs can exceed the scale seen during training. Two
        # padded views must agree; this never runs on the live video path.
        closeups = []
        closeup_diagnostics = {"executed_scales": [], "skipped_scales": []}
        diagnostics["photo_closeup"] = closeup_diagnostics
        for scale in (0.35, 0.45):
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
            pixels = cv2.warpAffine(image, matrix, (width, height), borderValue=(114, 114, 114))
            result = detect(pixels, infer, imgsz=imgsz, confidence=confidence)
            closeup_diagnostics["executed_scales"].append(scale)
            recovery["passes"] += result["recovery"]["passes"]
            nested = result["diagnostics"]
            diagnostics["raw_candidate_count"] += nested["raw_candidate_count"]
            diagnostics["passes"].extend(
                dict(p, view=f"closeup_{scale}/{p['view']}") for p in nested["passes"]
            )
            diagnostics["excluded_candidates"].extend(
                dict(
                    b,
                    view=f"closeup_{scale}/{b.get('view', 'original')}",
                    coordinate_space="scaled_detector_input",
                )
                for b in nested["excluded_candidates"]
            )
            for key, value in result.get("speed_ms", {}).items():
                speed[key] = speed.get(key, 0) + value
            for box in result["boxes"]:
                if box["class_id"] == 0:
                    mapped = dict(
                        box, xyxy=source_box(box["xyxy"], matrix, width, height), view=scale
                    )
                    reason = (
                        "candidate_limit"
                        if len(accepted) >= 6
                        else "region_too_small"
                        if min(
                            mapped["xyxy"][2] - mapped["xyxy"][0],
                            mapped["xyxy"][3] - mapped["xyxy"][1],
                        )
                        < 4
                        else "duplicate_region"
                        if any(iou(mapped["xyxy"], b["xyxy"]) >= 0.4 for b in accepted)
                        else "display_color_mismatch"
                        if not has_display_color(hsv, mapped)
                        else None
                    )
                    if reason:
                        exclude(mapped, reason)
                    else:
                        closeups.append(mapped)
            if scale == 0.35 and not closeups:
                closeup_diagnostics["skipped_scales"].append(
                    {"scale": 0.45, "reason": "no_eligible_new_window_first_scale"}
                )
                break
        for box in sorted(closeups, key=lambda b: b["confidence"], reverse=True):
            if len(accepted) >= 6 or any(iou(box["xyxy"], b["xyxy"]) >= 0.4 for b in accepted):
                exclude(box, "candidate_limit" if len(accepted) >= 6 else "duplicate_region")
                continue
            support = {b["view"] for b in closeups if iou(box["xyxy"], b["xyxy"]) >= 0.5}
            if len(support) < 2 or not has_display_color(hsv, box):
                exclude(
                    box,
                    "insufficient_supporting_views"
                    if len(support) < 2
                    else "display_color_mismatch",
                )
                continue
            accepted.append(
                {k: v for k, v in box.items() if k != "view"}
                | {
                    "localization_method": "photo_closeup_scale_v1",
                    "supporting_views": sorted(support),
                }
            )
            recovery["recovered_count"] += 1
    return output()
