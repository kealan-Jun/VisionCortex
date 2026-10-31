"""Bounded point observations outside source-fitted LED glyphs.

Only masks are projected. Image pixels are neither warped nor completed, and
this observer receives no OCR text or expected decimal position.
"""

import cv2
import numpy as np

from .native_segments import polygon_mask


def _near(first, second):
    return max(
        abs((first[j] + first[j + 2]) - (second[j] + second[j + 2])) / 2
        for j in (0, 1)
    ) <= 1.5


def observe(image, quads, color, *, glyph_margin=1):
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] * image.shape[1] > 262144
        or min(image.shape[:2]) < 3
    ):
        raise ValueError("Bounded native BGR point evidence required")
    if color not in {"red", "green"} or type(glyph_margin) is not int or glyph_margin not in (1, 2):
        raise ValueError("Explicit LED color and native glyph margin required")
    if not 2 <= len(quads) <= 12:
        raise ValueError("Two to twelve source-fitted glyphs required")
    h, w = image.shape[:2]
    qs = [np.asarray(q, np.float32) for q in quads]
    glyphs = np.zeros((h, w), np.uint8)
    transforms = []
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    for q in qs:
        if (
            q.shape != (4, 2)
            or not np.isfinite(q).all()
            or (q < 0).any()
            or (q[:, 0] > w).any()
            or (q[:, 1] > h).any()
            or not cv2.isContourConvex(q)
            or cv2.contourArea(q, oriented=True) <= 0
        ):
            raise ValueError("Ordered finite source-fitted quadrilaterals required")
        glyphs |= polygon_mask(image.shape, q).astype(np.uint8)
        transforms.append(cv2.getPerspectiveTransform(unit, q))
    excluded = cv2.dilate(
        glyphs, np.ones((2 * glyph_margin + 1,) * 2, np.uint8)
    ) > 0
    channels = image.astype(np.float32)
    primary = 2 if color == "red" else 1
    core = np.min(np.delete(channels, primary, axis=2), axis=2)
    out, diagnostics = [], []
    for i in range(len(qs) - 1):
        left = cv2.perspectiveTransform(
            np.float32([[[1, .8], [1, 1.25]]]), transforms[i]
        )[0]
        right = cv2.perspectiveTransform(
            np.float32([[[0, .8], [0, 1.25]]]), transforms[i + 1]
        )[0]
        gap = np.float32([left[0], right[0], right[1], left[1]])
        if not cv2.isContourConvex(gap) or cv2.contourArea(gap, oriented=True) <= 0:
            continue
        mask = polygon_mask(image.shape, gap) & ~excluded
        if mask.sum() < 9:
            continue
        background_rgb = np.percentile(channels[mask], 30, axis=0)
        residual = channels - background_rgb
        colored = residual[:, :, primary] - np.max(
            np.delete(residual, primary, axis=2), axis=2
        ) > 25
        chromatic_core = core[mask & colored]
        if len(chromatic_core) < 2:
            continue
        background = float(np.percentile(core[mask], 30))
        contrast = float(chromatic_core.max()) - background
        if contrast < 16:
            continue
        height = min(
            np.linalg.norm(qs[i][3] - qs[i][0]),
            np.linalg.norm(qs[i + 1][3] - qs[i + 1][0]),
        )
        candidates, levels = [], set()
        for fraction in np.linspace(.2, .9, 16):
            threshold = int(np.floor(background + float(fraction) * contrast))
            if threshold in levels:
                continue
            levels.add(threshold)
            binary = np.uint8((core > threshold) & colored)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            for k in range(1, n):
                x, y, cw, ch, area = map(int, stats[k])
                if not (
                    2 <= area <= max(8, height * height * .025)
                    and .5 <= cw / ch <= 2
                    and cw <= height * .18 and ch <= height * .18
                ):
                    continue
                yy, xx = np.nonzero(labels[y:y + ch, x:x + cw] == k)
                yy, xx = yy + y, xx + x
                # Inspect the complete source component, never its clipped gap.
                if not mask[yy, xx].all() or x == 0 or y == 0 or x + cw == w or y + ch == h:
                    continue
                candidates.append(dict(
                    after_digit=i, box=[x, y, x + cw, y + ch], area=area,
                    threshold_fraction=float(fraction), threshold_value=threshold,
                ))
        accepted = []
        for c in candidates:
            supporting = [a for a in candidates if
                          abs(a["threshold_value"] - c["threshold_value"]) >= 4
                          and _near(a["box"], c["box"])]
            if supporting and not any(_near(c["box"], a["box"]) for a in accepted):
                accepted.append(dict(
                    c,
                    threshold_values=sorted({c["threshold_value"], *[a["threshold_value"] for a in supporting]}),
                    method="native_projective_component_persistence/1",
                    glyph_margin=glyph_margin, local_contrast=contrast,
                ))
        # Preserve multiple points as ambiguity; do not silently discard them.
        out.extend(accepted)
        diagnostics.append(dict(
            after_digit=i, gap=gap.tolist(), native_gap_pixels=int(mask.sum()),
            independent_component_candidates=len(accepted),
        ))
    return dict(points=out, gaps=diagnostics, confirmed=False, OFF_inferred=False,
                numeric_value=None, resampled=False)
