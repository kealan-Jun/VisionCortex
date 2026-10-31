"""Source-frame anchored numeric video candidates, without cross-value splicing.

A rolling window ranks actual complete-looking source frames. Other frames may
rank visibility and corroborate an observed decimal point; their digit pixels
are never assembled into a new number. Recognition remains a candidate, not an
OFF proof, independent truth, equipment attribution or inventory instruction.
"""

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import re
import time

from .temporal_registration import normalized_box
from .temporal_display import _identity, _reference

REVISION = "vc-video-numeric-anchor/3"


def number(text):
    """One digit sequence and its observed decimal; never join separate tokens."""
    if not isinstance(text, str) or len(text) > 4096:
        return None
    stripped = text.strip()
    return (
        stripped
        if re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", stripped)
        else None
    )


def review_anchors(observations):
    """Veto incomplete source-frame tokens; never reconstruct their contents.

    Caller supplies a source-pinned continuous window and an independently
    tracked field identity. Longer text is only a veto on a shorter reading,
    not proof the longer text is right, nor a value to copy to another PTS.
    Two transforms of one source frame are one observation, never two votes.
    """
    if not isinstance(observations, list) or len(observations) > 128:
        raise ValueError("At most 128 source-frame observations required")
    groups = {}
    cleaned = []
    for observation in observations:
        key, identity = _identity(observation["identity"])
        source, pts = _reference(observation["source_ref"])
        for name in ("window_id", "layout_revision"):
            if (
                not isinstance(observation.get(name), str)
                or not 0 < len(observation[name]) <= 256
            ):
                raise ValueError("Pinned window and layout revision required")
        texts = observation.get("raw_texts")
        if (
            not isinstance(texts, list)
            or not 1 <= len(texts) <= 4
            or any(not isinstance(t, str) or len(t) > 256 for t in texts)
        ):
            raise ValueError("One to four bounded same-source OCR views required")
        group = (
            *key,
            source["source_sha256"],
            observation["window_id"],
            observation["layout_revision"],
        )
        row = dict(observation, identity=identity, source_ref=source)
        cleaned.append((group, pts, row))
        groups.setdefault(group, []).append((pts, row))
    for rows in groups.values():
        if max(t for t, _ in rows) - min(t for t, _ in rows) > 1:
            raise ValueError("Continuous anchor review window exceeds one second")
    results = []
    for group, _, row in cleaned:
        texts = list(dict.fromkeys(t.strip() for t in row["raw_texts"]))
        values = list(dict.fromkeys(v for t in texts if (v := number(t)) is not None))
        clocks = [
            t for t in texts if re.fullmatch(r"[0-9]{1,3}:[0-9]{2}(?::[0-9]{2})?", t)
        ]
        reasons = []
        candidate = values[0] if len(values) == 1 and len(texts) == 1 else None
        if row.get("registration_verified") is not True:
            candidate = None
            reasons.append("field_registration_not_verified")
        if row.get("clipped"):
            candidate = None
            reasons.append("source_field_clipped")
        if len(texts) > 1:
            candidate = None
            reasons.append("same_source_OCR_views_disagree")
        max_digits = max(
            (
                len(v.replace(".", ""))
                for _, other in groups[group]
                for t in other["raw_texts"]
                if (v := number(t)) is not None
            ),
            default=0,
        )
        if candidate and len(candidate.replace(".", "")) < max_digits:
            candidate = None
            reasons.append("shorter_token_may_be_dropout_or_real_value_change")
        expected = row.get("expected_digit_slots")
        if expected is not None:
            if type(expected) is not int or not 1 <= expected <= 12:
                raise ValueError("One to twelve declared digit slots required")
            if candidate and len(candidate.replace(".", "")) != expected:
                candidate = None
                reasons.append("token_does_not_cover_declared_digit_slots")
        if clocks:
            candidate = None
            reasons.append("clock_separator_preserved_not_quantity")
        if candidate and "." not in candidate and row.get("decimal_required"):
            candidate = None
            reasons.append("decimal_not_observed")
        display = (
            clocks[0]
            if len(clocks) == 1
            and len(texts) == 1
            and not row.get("clipped")
            and row.get("registration_verified") is True
            else candidate
        )
        if not display and not reasons:
            reasons.append("no_single_numeric_or_clock_token")
        results.append(
            dict(
                schema_version="vc-lab-temporal-display/1",
                method=REVISION,
                identity=row["identity"],
                source_refs=[row["source_ref"]],
                raw_text=" | ".join(texts),
                candidates=values,
                numeric_candidate=candidate,
                display_text=display,
                display_kind="clock" if clocks else "numeric",
                status="uncertain",
                value_text=None,
                gates=dict(confirmed=False),
                reading_status="recognized_candidate" if display else "pending_numeric",
                selected_frame_count=1,
                selected_span_seconds=0.0,
                rejection_reasons=reasons
                + ["OCR_candidate_not_independent_glyph_truth"],
                fused_digit_pixels=False,
                quality_score_calibrated=False,
                window_id=row["window_id"],
                layout_revision=row["layout_revision"],
                same_source_view_count=len(row["raw_texts"]),
                registration=dict(accepted=row.get("registration_verified") is True),
            )
        )
    return results


def digit_boxes(image, layout):
    digits = layout.get("digits", [])
    if not 1 <= len(digits) <= 12:
        raise ValueError(
            "Numeric video requires one to twelve source layout digit regions"
        )
    return [
        normalized_box(d.get("box") if isinstance(d, dict) else d, image.shape)
        for d in digits
    ]


def signal(image, layout):
    import cv2

    color = layout.get("color")
    if color in {"red", "green"}:
        channel = 2 if color == "red" else 1
        return image[:, :, channel].astype(float)
    return 255.0 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)


def visibility(image, layout):
    import numpy as np

    values = signal(image, layout)
    energy, contrast = [], []
    for x, y, xx, yy in digit_boxes(image, layout):
        p = values[y:yy, x:xx]
        lo, hi = np.percentile(p, [20, 98])
        contrast.append(float(hi - lo))
        # Absolute local emission energy retains weak/missing-digit differences;
        # normalizing every dim digit to white would conceal scan dropout.
        energy.append(float(np.maximum(p - lo, 0).mean()))
    return energy, contrast


def views(image, layout):
    """Two recorded transforms of one frame, never independent observations."""
    import cv2
    import numpy as np

    boxes = digit_boxes(image, layout)
    h = max(b[3] - b[1] for b in boxes)
    pad = max(2, math.ceil(h * 0.14))
    x, y = (
        max(0, min(b[0] for b in boxes) - pad),
        max(0, min(b[1] for b in boxes) - pad),
    )
    xx, yy = (
        min(image.shape[1], max(b[2] for b in boxes) + pad),
        min(image.shape[0], max(b[3] for b in boxes) + pad),
    )
    if layout.get("numeric_field_box") is not None:
        x, y, xx, yy = normalized_box(layout["numeric_field_box"], image.shape)
        if any(not (x <= a < c <= xx and y <= b < d <= yy) for a, b, c, d in boxes):
            raise ValueError("Numeric digit geometry escapes its field")
    native = image[y:yy, x:xx].copy()
    points = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
    slope = (
        float(np.polyfit(points[:, 0], points[:, 1], 1)[0]) if len(points) > 1 else 0.0
    )
    angle = math.degrees(math.atan(slope))
    if abs(angle) > 25:
        raise ValueError("Numeric source baseline exceeds supported rectification")
    m = cv2.getRotationMatrix2D((native.shape[1] / 2, native.shape[0] / 2), angle, 1.0)
    gray = signal(image, layout)[y:yy, x:xx]
    lo, hi = np.percentile(gray, [20, 99])
    normalized = np.uint8(np.clip(255 * (1 - (gray - lo) / max(hi - lo, 1)), 0, 255))
    normalized = cv2.warpAffine(
        normalized,
        m,
        (native.shape[1], native.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    normalized = cv2.copyMakeBorder(
        normalized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
    )
    return [
        ("native_crop", native),
        ("channel_contrast_deskew", cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)),
    ], dict(
        crop_box=[x, y, xx, yy],
        baseline_angle_degrees=angle,
        rotation_matrix=m.tolist(),
        normalized_padding=pad,
        contrast_percentiles=[float(lo), float(hi)],
        transforms_are_not_independent_evidence=True,
        strokes_drawn_or_completed=False,
    )


def decimal_peaks(image, layout):
    """Small separated bright islands at trailing glyph baselines; hypotheses."""
    import cv2
    import numpy as np

    if layout.get("color") not in {"red", "green"}:
        return []
    values = signal(image, layout)
    boxes = digit_boxes(image, layout)
    found = []
    for i, (x, y, xx, yy) in enumerate(boxes[:-1]):
        w, h = xx - x, yy - y
        right = min(image.shape[1], max(xx + 2, boxes[i + 1][0]))
        left = max(x, math.floor(xx - w * 0.22))
        top = max(0, math.floor(y + h * 0.55))
        bottom = min(image.shape[0], math.ceil(yy + h * 0.08))
        # Connected components are measured in a larger patch so a fragment
        # still attached to a long digit stroke cannot masquerade as a dot.
        px = max(0, x - 1)
        py = max(0, y - 1)
        patch = values[py:bottom, px:right]
        lo, hi = np.percentile(values[y:yy, x:xx], [20, 99])
        if hi - lo < 12:
            continue
        levels = []
        for fraction in (0.90, 0.95, 0.98):
            binary = np.uint8(patch > lo + fraction * (hi - lo))
            _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            for a, b, c, d, area in stats[1:]:
                a, b, c, d, area = map(int, (a, b, c, d, area))
                bx, by = px + a, py + b
                if (
                    not left <= bx + c / 2 < right
                    or not top <= by + d / 2 < bottom
                    or not 2 <= area <= max(8, math.ceil(w * h * 0.035))
                    or c > h * 0.24
                    or d > h * 0.24
                    or not 0.3 <= c / max(d, 1) <= 3.0
                ):
                    continue
                levels.append(
                    dict(
                        after_digit=i,
                        box=[bx, by, bx + c, by + d],
                        area=area,
                        threshold_fraction=fraction,
                    )
                )
        for a in levels:
            matches = [
                b
                for b in levels
                if b["threshold_fraction"] != a["threshold_fraction"]
                and max(
                    abs(
                        (a["box"][j] + a["box"][j + 2]) / 2
                        - (b["box"][j] + b["box"][j + 2]) / 2
                    )
                    for j in (0, 1)
                )
                <= 1.5
            ]
            if matches:
                found.append(a)
                break
    return found


def decimal_gap_peaks(image, layout):
    """Observe isolated native luminous points between, never inside, glyphs.

    The unsaturated channel suppresses red/green lens bloom. A point must be
    near the digit baseline and survive two local thresholds. This is a source
    observation, not proof that a missing point was OFF or a business value.
    """
    import cv2
    import numpy as np

    color = layout.get("color")
    if color not in {"red", "green"}:
        return []
    boxes = digit_boxes(image, layout)
    channels = image.astype(float)
    primary = 2 if color == "red" else 1
    others = np.delete(channels, primary, axis=2)
    core = np.min(others, axis=2)
    colored = channels[:, :, primary] - np.max(others, axis=2) > 25
    found = []
    for index, (_x, y, xx, yy) in enumerate(boxes[:-1]):
        height = yy - y
        left, right = xx, boxes[index + 1][0]
        top, bottom = (
            max(0, int(y + 0.80 * height)),
            min(image.shape[0], math.ceil(yy + 0.10 * height)),
        )
        if right - left < 2 or bottom - top < 3:
            continue
        patch = core[top:bottom, left:right]
        background, peak = float(np.percentile(patch, 30)), float(patch.max())
        if peak - background < 16:
            continue
        candidates = []
        for fraction in (0.4, 0.55, 0.7):
            binary = np.uint8(
                (patch > background + fraction * (peak - background))
                & colored[top:bottom, left:right]
            )
            _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            for a, b, w, h, area in stats[1:]:
                if not (
                    2 <= area <= max(8, height * height * 0.025)
                    and 0.5 <= w / h <= 2
                    and w <= height * 0.18
                    and h <= height * 0.18
                ):
                    continue
                if a == 0 or a + w == right - left or b == 0 or b + h == bottom - top:
                    continue  # A stroke clipped by the search window is not a dot.
                candidates.append(
                    dict(
                        after_digit=index,
                        box=[
                            int(left + a),
                            int(top + b),
                            int(left + a + w),
                            int(top + b + h),
                        ],
                        threshold_fraction=fraction,
                        area=int(area),
                    )
                )
        for candidate in candidates:
            support = [
                p
                for p in candidates
                if p["threshold_fraction"] != candidate["threshold_fraction"]
                and max(
                    abs(
                        (p["box"][j] + p["box"][j + 2])
                        - (candidate["box"][j] + candidate["box"][j + 2])
                    )
                    / 2
                    for j in (0, 1)
                )
                <= 1.5
            ]
            if support:
                found.append(
                    dict(
                        candidate,
                        method="native_interglyph_core_island/1",
                        local_contrast=peak - background,
                        threshold_support=sorted(
                            {
                                candidate["threshold_fraction"],
                                *[p["threshold_fraction"] for p in support],
                            }
                        ),
                    )
                )
                break
    return found


def resolve_decimal_tokens(image, layout, texts):
    """Reconcile same-frame OCR dot omission using actual gap pixels only."""
    points = decimal_gap_peaks(image, layout)
    return resolve_observed_points(texts, len(digit_boxes(image, layout)), points)


def resolve_observed_points(texts, digit_count, points):
    """Reconcile whole replies with one actually observed interdigit point."""
    tokens = [number(t) for t in texts]
    result = dict(
        numeric_candidate=None,
        decimal_status="unknown",
        native_point_evidence=points,
        value_text=None,
        confirmed=False,
        raw_texts=texts,
    )
    if (
        not tokens
        or any(t is None for t in tokens)
        or len({t.replace(".", "") for t in tokens}) != 1
    ):
        return dict(result, reason="same_source_digits_disagree")
    digits = tokens[0].replace(".", "")
    if len(digits) != digit_count:
        return dict(result, reason="incomplete_digit_slots")
    observed = {p["after_digit"] for p in points}
    model = {t.index(".") - 1 for t in tokens if "." in t}
    if len(points) != 1 or len(observed) != 1 or (model and model != observed):
        return dict(result, reason="decimal_missing_or_conflicting")
    after = next(iter(observed)) + 1
    return dict(
        result,
        numeric_candidate=digits[:after] + "." + digits[after:],
        decimal_status="observed_same_source_pixel_island",
        reason="same_digits_and_native_point",
    )


@dataclass
class _Window:
    signature: str
    touched: float
    frames: list = field(default_factory=list)
    first: object = None
    last: object = None
    last_infer: object = None
    result: dict | None = None
    calls: int = 0
    count: int = 0


class NumericVideoReader:
    def __init__(self, config):
        self.config = config
        self.states = {}
        self.native_costs = {}

    @property
    def buffer_bytes(self):
        return sum(
            f["image"].nbytes + (f["native"]["image"].nbytes if f.get("native") else 0)
            for s in self.states.values()
            for f in s.frames
        )

    def reset(self, key):
        self.states.pop(key, None)

    def observe(self, image, *, key, identity, source_ref, layout, ocr, native=None):
        import numpy as np

        expected_key, identity = _identity(identity)
        if tuple(key) != expected_key:
            raise ValueError(
                "Numeric cache key must match exact equipment/panel/field/track scope"
            )
        source_ref, timestamp = _reference(source_ref)
        native_mode = layout.get("numeric_native_ocr") is True
        if native_mode:
            import copy
            from .native_numeric import emission, same_transform

            if native is None:
                raise ValueError("Explicit unresampled native crop required")
            native_ref, native_time = _reference(native["source"])
            if (
                native_time != timestamp
                or native_ref["source_sha256"] != source_ref["source_sha256"]
            ):
                raise ValueError("Native crop cannot come from another source frame")
            if not same_transform(
                native_ref.get("registered_source_to_roi"), source_ref["source_to_roi"]
            ):
                raise ValueError(
                    "Native crop cannot come from another registered panel"
                )
            if any(
                native_ref.get(k) != source_ref.get(k)
                for k in (
                    "source_frame_pixels_sha256",
                    "original_frame_rgb_sha256",
                    "dynamic_physical_key",
                )
            ):
                raise ValueError(
                    "Native crop cannot come from other pixels or physical context"
                )
            if native["layout"].get("color") != layout.get("color") or len(
                native["quads"]
            ) != len(layout.get("digits", [])):
                raise ValueError("Native evidence differs from registered field layout")
            native = copy.deepcopy(native)
            native_support = emission(native)
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] * image.shape[1] > self.config.max_roi_pixels
        ):
            raise ValueError("Bounded native BGR numeric panel required")
        boxes = digit_boxes(image, layout)
        signature = hashlib.sha256(
            json.dumps(
                [identity, layout, source_ref["source_sha256"], image.shape],
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        now = time.monotonic()
        native_budget = None
        if native_mode:
            for old in [
                k
                for k, v in self.native_costs.items()
                if now - v["touched"] > self.config.session_ttl_seconds
            ]:
                self.native_costs.pop(old)
            budget_key = (
                tuple(key),
                source_ref["source_sha256"],
                hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest(),
            )
            if budget_key not in self.native_costs:
                if len(self.native_costs) >= self.config.max_sessions:
                    raise ValueError("Native OCR call-budget scope limit reached")
                self.native_costs[budget_key] = dict(calls=0, last=None, touched=now)
            native_budget = self.native_costs[budget_key]
            native_budget["touched"] = now
        for old in [
            k
            for k, s in self.states.items()
            if now - s.touched > self.config.session_ttl_seconds
        ]:
            self.states.pop(old)
        s = self.states.get(key)
        reset = []
        if s is not None and (
            signature != s.signature
            or timestamp <= s.last
            or timestamp - s.last > Fraction(str(self.config.max_gap_seconds))
        ):
            self.states.pop(key)
            s = None
            reset = ["source_layout_clock_or_gap_changed"]
        if s is None:
            if len(self.states) >= self.config.max_sessions:
                self.states.pop(min(self.states, key=lambda k: self.states[k].touched))
            s = _Window(signature, now, first=timestamp)
            self.states[key] = s
        s.touched = now
        s.last = timestamp
        s.count += 1
        if native_mode:
            s.calls = native_budget["calls"]
        energy, contrast = visibility(image, layout)
        frame = dict(
            image=image.copy(),
            source=source_ref,
            pts=timestamp,
            energy=energy,
            contrast=contrast,
            digest=hashlib.sha256(image.tobytes()).hexdigest(),
            decimal=(
                decimal_gap_peaks
                if layout.get("strict_decimal_evidence")
                else decimal_peaks
            )(image, layout),
        )
        if native_mode:
            frame.update(native=native, native_support=native_support)
        s.frames.append(frame)
        while len(s.frames) > 30 or timestamp - s.frames[0]["pts"] > Fraction(
            str(self.config.max_window_seconds)
        ):
            s.frames.pop(0)
        while self.buffer_bytes > self.config.max_buffer_bytes and len(s.frames) > 1:
            s.frames.pop(0)
        while self.buffer_bytes > self.config.max_buffer_bytes and len(self.states) > 1:
            oldest = min(
                (k for k in self.states if k != key),
                key=lambda k: self.states[k].touched,
            )
            self.states.pop(oldest)
        if self.buffer_bytes > self.config.max_buffer_bytes:
            self.states.pop(key)
            raise ValueError("Numeric frame ring exceeds byte budget")
        due = timestamp - s.first >= Fraction(".45") and (
            s.last_infer is None or timestamp - s.last_infer >= 1
        )
        limit = layout.get("numeric_max_ocr_calls", 8)
        if type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("Explicit bounded numeric OCR call limit required")
        if native_mode:
            from .native_numeric import review

            fallback = layout.get("numeric_native_color_fallback", False)
            if type(fallback) is not bool or fallback and (
                layout.get("numeric_native_stroke_veto") is not True
                or layout.get("data_type") != "number"
            ):
                raise ValueError("Color fallback requires a typed number and source-stroke verification")
            # Candidate admission needs native emission, not half a second of
            # uninterrupted tracking. Pixel resets must not replenish budget.
            due = s.count >= 3 and (
                native_budget["last"] is None or timestamp - native_budget["last"] >= 1
            )
            if due and s.calls < limit:
                s.last_infer = timestamp
                eligible = [
                    f
                    for f in s.frames
                    if all(v["visible"] for v in f["native_support"])
                ]
                best = frame
                reply = {}
                replies = []
                if eligible:
                    maximum = np.maximum(
                        np.max(
                            [
                                [v["energy"] for v in f["native_support"]]
                                for f in eligible
                            ],
                            axis=0,
                        ),
                        1.0,
                    )

                    def native_rank(f):
                        ratios = (
                            np.asarray([v["energy"] for v in f["native_support"]])
                            / maximum
                        )
                        return float(ratios.min() * 2 + ratios.mean())

                    best = max(eligible, key=native_rank)
                    if ocr is not None:
                        reply = ocr(
                            best["native"]["image"],
                            anchor_source=best["native"]["source"],
                        )
                        s.calls += 1
                        native_budget["calls"] = s.calls
                        native_budget["last"] = timestamp
                assessed = review(best["native"], reply)
                if eligible and ocr is not None:
                    replies.append(dict(view="native", reply=reply, review=assessed))
                    if fallback and assessed["numeric_candidate"] is None and s.calls < limit:
                        from .native_views import color_residual

                        prepared, preparation = color_residual(best["native"]["image"])
                        if preparation["prepared_bgr_sha256"] != preparation["source_bgr_sha256"]:
                            alternate_reply = ocr(prepared, anchor_source=dict(
                                best["native"]["source"], ocr_view=preparation,
                            ))
                            s.calls += 1
                            native_budget["calls"] = s.calls
                            # Source verification always uses the original image.
                            alternative = review(best["native"], alternate_reply)
                            replies.append(dict(view="color_residual", preparation=preparation,
                                                reply=alternate_reply, review=alternative))
                            if alternative["glyph_candidate"]:
                                conflict = assessed["glyph_candidate"] and (
                                    assessed["glyph_candidate"] != alternative["glyph_candidate"]
                                    or ("." in assessed["raw_text"] and "." in alternative["raw_text"]
                                        and number(assessed["raw_text"]) != number(alternative["raw_text"]))
                                )
                                if conflict:
                                    assessed = dict(assessed, glyph_candidate=None, numeric_candidate=None,
                                                    rejection_reasons=[*assessed["rejection_reasons"],
                                                                       "supported_same_frame_views_conflict"])
                                elif alternative["numeric_candidate"] or not assessed["glyph_candidate"]:
                                    assessed, reply = alternative, alternate_reply
                native_source = best["native"]["source"]
                glyph = assessed["glyph_candidate"]
                candidate = assessed["numeric_candidate"]
                s.result = dict(
                    assessed,
                    schema_version="vc-lab-temporal-display/1",
                    method="native_registered_crop/1",
                    identity=identity,
                    status="uncertain",
                    candidates=[candidate or glyph] if glyph else [],
                    reading_status="recognized_candidate"
                    if candidate
                    else "glyphs_only"
                    if glyph
                    else "pending_numeric",
                    display_text=candidate
                    or (
                        (assessed["raw_text"] + "（小数点未确认）")
                        if glyph
                        else "读数未确认"
                    ),
                    source_refs=[native_source],
                    observation_time_start=dict(
                        pts=native_source["pts"], time_base=native_source["time_base"]
                    ),
                    observation_time_end=dict(
                        pts=native_source["pts"], time_base=native_source["time_base"]
                    ),
                    selected_frame_count=1,
                    selected_span_seconds=0.0,
                    gates=dict(confirmed=False),
                    registration=native_source.get("registration", {"accepted": True}),
                    preprocessing=best["native"]["metadata"],
                    ocr_score=reply.get("score"),
                    **(dict(ocr_views=replies) if fallback else {}),
                    best_frame_selection_window_refs=[f["source"] for f in s.frames],
                    fused_digit_pixels=False,
                    reset_reasons=reset,
                )
            if s.result is not None:
                return dict(
                    s.result,
                    input_frame_count=s.count,
                    buffer_bytes=self.buffer_bytes,
                    ocr_port_calls=s.calls,
                )
            return dict(
                schema_version="vc-lab-temporal-display/1",
                method="native_registered_crop/1",
                identity=identity,
                status="uncertain",
                value_text=None,
                raw_text="",
                candidates=[],
                numeric_candidate=None,
                source_refs=[native["source"]],
                gates=dict(confirmed=False),
                input_frame_count=s.count,
                selected_frame_count=0,
                selected_span_seconds=0.0,
                buffer_bytes=self.buffer_bytes,
                ocr_port_calls=s.calls,
                rejection_reasons=["collecting_continuous_native_frames"],
            )
        if due and layout.get("glyph_adapter") is not None:
            from .glyph_adapter import recognize

            s.last_infer = timestamp
            evaluations = [
                (f, recognize(f["image"], layout, layout["glyph_adapter"]))
                for f in s.frames
            ]
            best, decoded = max(
                evaluations,
                key=lambda item: (
                    sum(g["glyph"] is not None for g in item[1]["glyphs"]),
                    min(g["similarity"] for g in item[1]["glyphs"]),
                    sum(g["similarity"] for g in item[1]["glyphs"]),
                ),
            )
            pattern = decoded["glyph_pattern"]
            decimal = (
                resolve_decimal_tokens(best["image"], layout, [pattern])
                if decoded["complete"]
                else None
            )
            proposed = decimal.get("numeric_candidate") if decimal else None
            reconciliation = None
            ocr_geometry = None
            if (
                layout.get("glyph_ocr_reconcile") is True
                and ocr is not None
                and s.calls < limit
            ):
                from .glyph_adapter import reconcile

                native_views, ocr_geometry = views(best["image"], layout)
                native = native_views[0][1]
                reply = ocr(native, anchor_source=best["source"])
                s.calls += 1
                reconciliation = reconcile(
                    dict(identity=identity, source_ref=best["source"], result=decoded),
                    dict(
                        identity=identity,
                        source_ref=best["source"],
                        raw_text=reply.get("raw_text"),
                    ),
                )
            s.result = dict(
                schema_version="vc-lab-temporal-display/1",
                method=decoded["method"],
                identity=identity,
                status="uncertain",
                value_text=None,
                raw_text=pattern,
                candidates=[proposed or pattern] if decoded["complete"] else [],
                numeric_candidate=proposed,
                glyph_candidate=pattern,
                display_text=proposed
                or pattern
                + ("（小数点未确认）" if decoded["complete"] else "（笔画未确认）"),
                reading_status="recognized_candidate"
                if proposed
                else "glyphs_only"
                if decoded["complete"]
                else "pending_numeric",
                source_refs=[best["source"]],
                selected_frame_count=1,
                selected_span_seconds=0.0,
                observation_time_start=dict(
                    pts=best["source"]["pts"], time_base=best["source"]["time_base"]
                ),
                observation_time_end=dict(
                    pts=best["source"]["pts"], time_base=best["source"]["time_base"]
                ),
                rejection_reasons=[
                    "decimal_observed_not_business_confirmation"
                    if proposed
                    else "decimal_not_established",
                    "glyph_adapter_not_independent_acceptance_truth",
                ],
                gates=dict(confirmed=False),
                registration=dict(accepted=True),
                glyph_adapter=decoded,
                native_decimal_evidence=decimal,
                ocr_reconciliation=reconciliation,
                ocr_preprocessing=ocr_geometry,
                fused_digit_pixels=False,
                quality_score_calibrated=False,
                best_frame_selection_window_refs=[f["source"] for f in s.frames],
                input_frame_count=s.count,
                buffer_bytes=self.buffer_bytes,
                ocr_port_calls=s.calls,
            )
            return dict(s.result)
        if due and ocr is not None and s.calls + 2 <= limit:
            s.last_infer = timestamp
            effective_layout = layout
            maximum = np.maximum(
                np.max(np.asarray([f["energy"] for f in s.frames]), axis=0), 1.0
            )

            def rank(f):
                ratios = np.asarray(f["energy"]) / maximum
                return float(ratios.min() * 2 + ratios.mean())

            best = max(s.frames, key=rank)
            ratios = np.asarray(best["energy"]) / maximum
            reasons = []
            if (
                min(best["contrast"]) < 12
                or min(b[3] - b[1] for b in boxes) < self.config.min_native_digit_height
            ):
                reasons.append("native_contrast_or_resolution_insufficient")
            if ratios.min() < 0.72:
                reasons.append("no_single_source_frame_covers_visible_digit_slots")
            replies = []
            numeric = []
            proof = []
            geometry = None
            if not reasons:
                transformed, geometry = views(best["image"], effective_layout)
                for name, pixels in transformed:
                    # The selected frame may precede this callback's trigger.
                    # Its receipt must identify that original frame, not 'now'.
                    reply = ocr(pixels, anchor_source=best["source"])
                    s.calls += 1
                    raw = reply.get("raw_text", "")
                    candidate = number(raw)
                    if candidate and len(candidate.replace(".", "")) != len(boxes):
                        candidate = None
                    replies.append(
                        dict(
                            view=name,
                            raw_text=raw,
                            score=reply.get("score"),
                            recognized_numeric=candidate,
                            input_bgr_sha256=hashlib.sha256(
                                pixels.tobytes()
                            ).hexdigest(),
                            source_ref=best["source"],
                        )
                    )
                    if candidate:
                        numeric.append(candidate)
                for dot in best["decimal"]:
                    matches = [
                        f
                        for f in s.frames
                        if f["digest"] != best["digest"]
                        and abs(f["pts"] - best["pts"]) <= Fraction(".2")
                        and any(
                            d["after_digit"] == dot["after_digit"]
                            and max(
                                abs(
                                    (d["box"][j] + d["box"][j + 2])
                                    - (dot["box"][j] + dot["box"][j + 2])
                                )
                                / 2
                                for j in (0, 1)
                            )
                            <= 2
                            for d in f["decimal"]
                        )
                    ]
                    if matches:
                        proof.append(
                            dict(
                                **dot,
                                source_refs=[best["source"], matches[0]["source"]],
                                status="observed_local_dot_hypothesis_not_format_calibration",
                            )
                        )
            digits = {v.replace(".", "") for v in numeric}
            dots = {p["after_digit"] for p in proof}
            chosen = None
            if len(digits) == 1 and numeric:
                chars = next(iter(digits))
                model_dots = {v.index(".") - 1 for v in numeric if "." in v}
                if len(dots) == 1 and (not model_dots or model_dots == dots):
                    i = next(iter(dots)) + 1
                    chosen = chars[:i] + "." + chars[i:]
                elif len(dots) > 1 or len(model_dots) > 1:
                    reasons.append("decimal_position_conflict")
                elif len(model_dots) == 1:
                    chosen = next(v for v in numeric if "." in v)
                else:
                    chosen = chars
                    reasons.append("decimal_presence_not_independently_verified")
            elif len(digits) > 1:
                reasons.append("same_frame_OCR_views_disagree_on_digits")
            else:
                reasons.append("no_complete_numeric_token")
            if layout.get("strict_decimal_evidence"):
                resolved = (
                    resolve_decimal_tokens(
                        best["image"],
                        effective_layout,
                        [r["raw_text"] for r in replies],
                    )
                    if replies
                    else None
                )
                chosen = resolved.get("numeric_candidate") if resolved else None
                if resolved and chosen is None:
                    reasons.append(resolved["reason"])
                proof = resolved["native_point_evidence"] if resolved else []
            # Even repeated OCR is not whole-glyph/OFF truth. Store traceable
            # numbers for review, never upgrade them to confirmed facts.
            raw = chosen or (
                numeric[0] if numeric else (replies[0]["raw_text"] if replies else "")
            )
            s.result = dict(
                schema_version="vc-lab-temporal-display/1",
                identity=identity,
                status="uncertain",
                value_text=None,
                raw_text=raw,
                candidates=list(dict.fromkeys(([chosen] if chosen else []) + numeric)),
                numeric_candidate=chosen,
                reading_status="recognized_candidate" if chosen else "pending_numeric",
                method=REVISION,
                source_refs=[best["source"]],
                observation_time_start=dict(
                    pts=best["source"]["pts"], time_base=best["source"]["time_base"]
                ),
                observation_time_end=dict(
                    pts=best["source"]["pts"], time_base=best["source"]["time_base"]
                ),
                observation_time_semantics="exact original anchor PTS; surrounding window does not assert value persistence",
                selected_frame_count=1,
                selected_span_seconds=0.0,
                support_map=[],
                unresolved_regions=[],
                conflicting_evidence=[],
                registration=dict(
                    accepted=True, method="caller_registered_ROI_single_original_anchor"
                ),
                reset_reasons=reset,
                rejection_reasons=reasons
                + ["numeric_candidate_not_independent_glyph_truth"],
                gates=dict(confirmed=False),
                quality_score_calibrated=False,
                ocr_views=replies,
                preprocessing=geometry,
                digit_geometry_basis="explicit_versioned_field_layout",
                decimal_pixel_hypotheses=proof,
                visibility_ratios=ratios.tolist(),
                fused_digit_pixels=False,
                best_frame_selection_window_refs=[f["source"] for f in s.frames],
                confirmed_scope="numeric_visual_candidate_only_not_business_fact",
            )
        if s.result is None:
            result = dict(
                schema_version="vc-lab-temporal-display/1",
                identity=identity,
                status="uncertain",
                value_text=None,
                raw_text="",
                candidates=[],
                numeric_candidate=None,
                method=REVISION,
                source_refs=[source_ref],
                selected_frame_count=0,
                selected_span_seconds=0.0,
                support_map=[],
                unresolved_regions=[],
                conflicting_evidence=[],
                registration=dict(accepted=True),
                gates=dict(confirmed=False),
                reset_reasons=reset,
                rejection_reasons=["collecting_continuous_numeric_video"],
                quality_score_calibrated=False,
            )
        else:
            result = dict(s.result)
        result.update(
            input_frame_count=s.count,
            buffer_bytes=self.buffer_bytes,
            ocr_port_calls=s.calls,
        )
        return result
