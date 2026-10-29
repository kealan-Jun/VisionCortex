"""Bounded photo rotation recovery and explicit source-coordinate transforms.

Numbers never select an orientation. A single eligible horizontal LED pair
needs source window evidence; ambiguous LCD directions need spatially matching
non-numeric lettering from independent source views. No image is rewritten.
"""

import copy
from dataclasses import dataclass
import hashlib
import re
import time

import cv2
import numpy as np

from field_recognition.recognition.panel_layout import resolve_type_boxes, roles_for_boxes
from field_recognition.recognition.panel_recovery import iou, has_display_color
from field_recognition.recognition.red_geometry import red_display_geometry


VERSION = "photo_orthogonal_orientation_v1"


@dataclass(frozen=True)
class Rotation:
    degrees: int
    width: int
    height: int
    x: int = 0
    y: int = 0

    def image(self, pixels):
        return np.ascontiguousarray(np.rot90(pixels, self.degrees // 90))

    def polygon(self, points):
        points = np.asarray(points, dtype=float)
        u, v = points[..., 0], points[..., 1]
        if self.degrees == 90:
            x, y = self.width - 1 - v, u
        elif self.degrees == 180:
            x, y = self.width - 1 - u, self.height - 1 - v
        elif self.degrees == 270:
            x, y = v, self.height - 1 - u
        else:
            x, y = u, v
        return (np.stack([x, y], axis=-1) + [self.x, self.y]).tolist()

    def box(self, bounds, *, offset=True):
        # xyxy uses half-open pixel edges, unlike OCR polygon pixel centers.
        a, b, c, d = bounds
        if self.degrees == 90:
            result = [self.width - d, a, self.width - b, c]
        elif self.degrees == 180:
            result = [self.width - c, self.height - d, self.width - a, self.height - b]
        elif self.degrees == 270:
            result = [b, self.height - c, d, self.height - a]
        else:
            result = list(bounds)
        if offset:
            result = [value + shift for value, shift in zip(result, [self.x, self.y] * 2)]
        return result

    def crop(self, crop):
        x, y, w, h = crop
        a, b, c, d = self.box([x, y, x + w, y + h])
        return [a, b, c - a, d - b]

    def metadata(self):
        return {
            "degrees_ccw": self.degrees,
            "source_size": [self.width, self.height],
            "source_offset": [self.x, self.y],
            "coordinate_space": "source_image",
            "pixel_operation": "lossless_quarter_turn",
        }


def _words(result, offset=(0, 0), scale=1):
    words = []
    for line in result.get("lines", []):
        text = line.get("text", "").strip().upper()
        # Exclude seven-segment pseudo-words and every numeric token. This
        # lettering supports direction only; it never identifies an asset.
        if (
            not re.fullmatch(r"[A-Z]{4,24}", text)
            or not set(text) - set("BDGILOSZ")
            or (line.get("confidence") or 0) < 0.6
            or line.get("polygon") is None
        ):
            continue
        polygon = np.asarray(line["polygon"], dtype=float) / scale + offset
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            continue
        left, top = polygon.min(axis=0)
        right, bottom = polygon.max(axis=0)
        if bottom - top < 5 or right - left < 1.8 * (bottom - top):
            continue
        words.append(
            {
                "text": text,
                "confidence": line["confidence"],
                "bbox": [float(left), float(top), float(right), float(bottom)],
            }
        )
    return words


def _lettering_agrees(a, b):
    if iou(a["bbox"], b["bbox"]) < 0.3 or max(a["confidence"], b["confidence"]) < 0.8:
        return False
    # OCR need not spell arbitrary lettering perfectly. Compare short letter
    # strings at the same source location, without a brand or instrument registry.
    previous = list(range(len(b["text"]) + 1))
    for i, left in enumerate(a["text"], 1):
        current = [i]
        for j, right in enumerate(b["text"], 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] <= max(len(a["text"]), len(b["text"])) // 3


def _context(pixels, boxes):
    bounds = np.asarray([box["xyxy"] for box in boxes])
    left, top = bounds[:, :2].min(axis=0)
    right, bottom = bounds[:, 2:].max(axis=0)
    px, py = max(16, round((right - left) * 0.25)), max(24, round((bottom - top) * 0.75))
    return [
        max(0, int(left - px)),
        max(0, int(top - py)),
        min(pixels.shape[1], int(right + px)),
        min(pixels.shape[0], int(bottom + py)),
    ]


def recover(core, image, allowed):
    started = time.monotonic()
    proof = {
        "version": VERSION,
        "status": "unconfirmed",
        "attempts": [],
        "maximum_detector_requests": 3,
        "maximum_ocr_requests": 15,
        "readings_used_for_orientation": False,
        "ocr_calls": [],
    }
    candidates = []
    get_instrument = core.get("get_instrument", lambda _: {})
    for degrees in (90, 180, 270):
        rotation = Rotation(degrees, image.shape[1], image.shape[0])
        pixels = rotation.image(image)
        detection = core["panel_detector"].predict(pixels, photo=True)
        boxes = resolve_type_boxes(detection.get("boxes", []), allowed, get_instrument)
        authorized = [box for box in boxes if box.get("instrument_id") in allowed]
        attempt = {
            "degrees_ccw": degrees,
            "detection": copy.deepcopy(detection),
            "authorized_box_count": len(authorized),
            "lettering": [],
        }
        proof["attempts"].append(attempt)
        if detection.get("status") != "completed" or not authorized:
            continue
        groups = {}
        for box in authorized:
            groups.setdefault((str(box["class_id"]), box["instrument_id"]), []).append(box)
        roles = roles_for_boxes(authorized)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
        native_rows = {}
        for box in authorized:
            if box["class_id"] == 0:
                left, top, right, bottom = box["xyxy"]
                crop = pixels[top:bottom, left:right]
                geometry = red_display_geometry(crop, left, top)
                if geometry and has_display_color(hsv, box):
                    native_rows[tuple(box["xyxy"])] = geometry
        attempt["native_red_rows"] = list(native_rows.values())
        pair_ids = {
            group[0]["instrument_id"]
            for group in groups.values()
            if len(group) == 2
            and all(box["class_id"] == 0 for box in group)
            and all(
                ((str(box["class_id"]), box["instrument_id"]), tuple(box["xyxy"])) in roles
                for box in group
            )
            and all(tuple(box["xyxy"]) in native_rows for box in group)
        }
        attempt["native_pair_instrument_ids"] = sorted(pair_ids)
        candidates.append((rotation, detection, attempt, pair_ids))

    # A complete pair in just one direction supplies the existing model layout
    # contract plus native red-window evidence. Competing directions abstain.
    pairs = [candidate for candidate in candidates if candidate[3]]
    if len(pairs) > 1:
        proof["status"] = "ambiguous"
    selected = pairs[0] if len(pairs) == 1 and len(candidates) == 1 else None
    basis = "unique_horizontal_pair_and_native_windows" if selected else None
    if selected is None:
        supported_degrees = {
            candidate[0].degrees
            for candidate in candidates
            if any(sum(iid in other[3] for other in candidates) == 1 for iid in candidate[3])
        }
        for candidate in candidates:
            rotation, detection, attempt, _ = candidate
            pixels = rotation.image(image)
            groups = {}
            for box in detection["boxes"]:
                if box.get("instrument_id") in allowed:
                    groups.setdefault(box["instrument_id"], []).append(box)
            whole = None
            for group in list(groups.values())[:2]:
                left, top, right, bottom = _context(pixels, group)
                crop = pixels[top:bottom, left:right]
                if not crop.size:
                    continue
                views = []
                for scale in (1, 2):
                    view = (
                        crop
                        if scale == 1
                        else cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    )
                    result = core["predict_panel"](view, 0, 0)
                    result = dict(
                        result,
                        orientation_view={
                            "degrees_ccw": rotation.degrees,
                            "context_bbox": [left, top, right, bottom],
                            "scale": scale,
                            "source_pixels_sha256": hashlib.sha256(view.tobytes()).hexdigest(),
                        },
                    )
                    proof["ocr_calls"].append(result)
                    words = _words(result, (left, top), scale)
                    views.append({"view": f"context_{scale}x", "words": words})
                context_agrees = any(
                    _lettering_agrees(a, b) for a in views[0]["words"] for b in views[1]["words"]
                )
                if not context_agrees and any(view["words"] for view in views):
                    if whole is None:
                        whole = core["predict_panel"](pixels, 0, 0)
                        whole = dict(
                            whole,
                            orientation_view={
                                "degrees_ccw": rotation.degrees,
                                "whole_source": True,
                                "source_pixels_sha256": hashlib.sha256(
                                    pixels.tobytes()
                                ).hexdigest(),
                            },
                        )
                        proof["ocr_calls"].append(whole)
                    words = [
                        word
                        for word in _words(whole)
                        if left <= word["bbox"][0] < word["bbox"][2] <= right
                        and top <= word["bbox"][1] < word["bbox"][3] <= bottom
                    ]
                    views.append({"view": "whole_source", "words": words})
                agreeing = [
                    {"first_view": a["view"], "second_view": b["view"], "first": wa, "second": wb}
                    for i, a in enumerate(views)
                    for b in views[i + 1 :]
                    for wa in a["words"]
                    for wb in b["words"]
                    if _lettering_agrees(wa, wb)
                ]
                attempt["lettering"].append(
                    {
                        "instrument_id": group[0]["instrument_id"],
                        "context_bbox": [left, top, right, bottom],
                        "views": views,
                        "support": agreeing,
                    }
                )
            if any(item["support"] for item in attempt["lettering"]):
                supported_degrees.add(rotation.degrees)
        if len(supported_degrees) == 1:
            selected = next(c for c in candidates if c[0].degrees in supported_degrees)
            basis = (
                "source_lettering_multi_view"
                if any(item["support"] for item in selected[2]["lettering"])
                else "unique_horizontal_pair_and_native_windows"
            )
        elif len(supported_degrees) > 1:
            proof["status"] = "ambiguous"
    proof["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    if selected:
        rotation, detection, attempt, pair_ids = selected
        native_ids = {iid for iid in pair_ids if sum(iid in c[3] for c in candidates) == 1}
        lettering_ids = {item["instrument_id"] for item in attempt["lettering"] if item["support"]}
        supported_ids = native_ids | lettering_ids
        detection = copy.deepcopy(detection)
        excluded = [
            box
            for box in detection["boxes"]
            if box.get("instrument_id") in allowed and box["instrument_id"] not in supported_ids
        ]
        detection["boxes"] = [box for box in detection["boxes"] if box not in excluded]
        detection.setdefault("diagnostics", {}).setdefault("excluded_candidates", []).extend(
            dict(box, reason="panel_orientation_unconfirmed") for box in excluded
        )
        proof.update(
            status="recovered",
            degrees_ccw=rotation.degrees,
            basis=basis,
            supported_instrument_ids=sorted(supported_ids),
            instrument_direction_basis={
                iid: "source_lettering_multi_view"
                if iid in lettering_ids
                else "unique_horizontal_pair_and_native_windows"
                for iid in supported_ids
            },
            source_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            pixel_hash_basis="decoded_bgr_c_order",
        )
        return rotation, detection, proof
    proof["reason"] = (
        "competing_directions"
        if proof["status"] == "ambiguous"
        else ("source_direction_unconfirmed" if candidates else "no_authorized_rotated_panel")
    )
    return None, None, proof


def restore(result, rotation, proof):
    """Keep executable region coordinates in the original source image space."""
    result = copy.deepcopy(result)
    oriented_detection = copy.deepcopy(result["panel_detection"])

    def lines(values):
        values = copy.deepcopy(values)
        for line in values:
            if line.get("polygon") is not None:
                line["polygon"] = rotation.polygon(line["polygon"])
        return values

    def digits(value):
        value = copy.deepcopy(value)
        if value and value.get("polygon") is not None:
            value["polygon"] = rotation.polygon(value["polygon"])
        return value

    for region in result.get("panel_regions", []):
        region["bbox"] = rotation.box(region["bbox"])
        region["crop"] = rotation.crop(region["crop"])
        region["digit_region"] = digits(region.get("digit_region"))
        region["photo_orientation"] = rotation.metadata() | {
            "basis": proof["instrument_direction_basis"][region["instrument_id"]]
        }
        local = region["local_ocr"]
        local["lines"] = lines(local.get("lines", []))
        if local.get("digit_region"):
            local["digit_region"] = digits(local["digit_region"])
        if local.get("red_display_geometry"):
            local["red_display_geometry"] = digits(local["red_display_geometry"])
        local["coordinate_basis"] = {
            "lines_digit_region_and_red_display_geometry": "source_image",
            "other_diagnostics_and_raw_views": (
                "oriented_photo_branch_with_original_view_scale_and_crop"
            ),
            **rotation.metadata(),
        }
    result["lines"] = lines(result.get("lines", []))
    for box in result["panel_detection"]["boxes"]:
        box["xyxy"] = rotation.box(box["xyxy"], offset=False)
    result["panel_detection"]["photo_orientation"] = copy.deepcopy(proof) | {
        **rotation.metadata(),
        "oriented_detection": oriented_detection,
    }
    result["panel_detection"]["diagnostics"]["coordinate_space"] = "oriented_photo_branch"
    result["panel_detection"]["diagnostics"]["box_coordinate_space"] = "source_input"
    result["photo_orientation"] = copy.deepcopy(proof) | rotation.metadata()
    result["wall_seconds"] = round(result.get("wall_seconds", 0) + proof["duration_ms"] / 1000, 3)
    record_calls(result, proof)
    return result


def record_calls(result, proof):
    """Orientation OCR is a real model invocation, even when direction is unknown."""
    calls = proof["ocr_calls"]
    result["actual_model_invocation"] = bool(
        result.get("actual_model_invocation")
        or any(call.get("actual_model_invocation") for call in calls)
    )
    count = result.get("ocr_model_call_count")
    result["ocr_model_call_count"] = (
        count + sum(call["ocr_model_call_count"] for call in calls)
        if (
            count is not None
            and all(call.get("ocr_model_call_count") is not None for call in calls)
        )
        else None
    )
    for name in ("ocr_model_lock_wait_ms", "ocr_model_prepare_ms", "ocr_model_call_ms"):
        values = [call.get(name) for call in calls]
        existing = result.get(name)
        result[name] = (
            round(existing + sum(values), 3)
            if (existing is not None and all(value is not None for value in values))
            else None
        )
    result["ocr_model_call_scopes"] = sorted(
        set(result.get("ocr_model_call_scopes", []))
        | {call["ocr_model_call_scope"] for call in calls if call.get("ocr_model_call_scope")}
    )
    result["orientation_ocr_call_count"] = (
        sum(call["ocr_model_call_count"] for call in calls)
        if (all(call.get("ocr_model_call_count") is not None for call in calls))
        else None
    )


def source_panel(image, region, *, digit_crop=False):
    """Reconstruct the same upright source pixels for evidence and cloud review."""
    if digit_crop and (region.get("digit_region") or {}).get("status") == "located":
        from field_recognition.recognition.digit_regions import extract_digits

        return extract_digits(image, region["digit_region"])[0]
    x, y, w, h = region["crop"]
    crop = image[y : y + h, x : x + w].copy()
    degrees = (region.get("photo_orientation") or {}).get("degrees_ccw", 0)
    return np.ascontiguousarray(np.rot90(crop, degrees // 90))
