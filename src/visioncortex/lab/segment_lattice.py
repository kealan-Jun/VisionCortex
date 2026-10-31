"""Fit shared seven-segment cell geometry, without fitting a displayed value.

Only the seven possible stroke locations are optimized. No character label,
font example or expected number enters the fit. Resampling is used for this
small geometry objective only; returned cells address original source pixels.
"""

import cv2
import numpy as np

from .temporal_display import SEGMENT_BOXES, SEGMENTS


def expanded_quad(quad):
    """A bounded vertical search envelope; never synthesize source pixels."""
    q = np.asarray(quad, np.float32)
    return cv2.perspectiveTransform(
        np.float32([[[0, -0.25], [1, -0.25], [1, 1.25], [0, 1.25]]]),
        cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), q),
    )[0]


def fit(native, positive_masks, *, expanded=False):
    image = native["image"]
    if len(positive_masks) != len(native["quads"]):
        raise ValueError("One source-positive mask per native slot required")
    W, H = 32, 48
    target = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    signals = []
    transforms = []
    for value, positive in zip(native["quads"], positive_masks, strict=True):
        q = expanded_quad(value) if expanded else np.asarray(value, np.float32)
        matrix = cv2.getPerspectiveTransform(q, target)
        patch = cv2.warpPerspective(positive.astype(np.float32), matrix, (W, H))
        patch[:, :2] = 0
        patch[:, -2:] = 0
        total = float(patch.sum())
        if total < 4:
            return None, dict(
                status="pending", reason="insufficient_shared_cell_signal"
            )
        signals.append(patch / total)
        transforms.append(
            cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), q)
        )
    signals = np.stack(signals)
    evaluated = {}
    rendered_cache = {}
    signal_norm = np.sqrt((signals * signals).sum(axis=(1, 2)))

    def mask(parameters, lit):
        key = (*parameters, lit)
        if key in rendered_cache:
            return rendered_cache[key]
        left, top, width, height, shear = parameters
        canvas = np.zeros((H, W), np.float32)
        for name in lit:
            x, y, xx, yy = SEGMENT_BOXES[name]
            points = []
            for a, b in [(x, y), (xx, y), (xx, yy), (x, yy)]:
                v = top + height * b
                u = left + width * a + shear * (v - 0.5)
                points.append([u * (W - 1), v * (H - 1)])
            cv2.fillConvexPoly(canvas, np.rint(points).astype(np.int32), 1.0)
        result = cv2.GaussianBlur(canvas, (5, 5), 1.0)
        if len(rendered_cache) >= 512:
            rendered_cache.pop(next(iter(rendered_cache)))
        rendered_cache[key] = result
        return result

    def score(parameters):
        key = tuple(round(float(v), 5) for v in parameters)
        if key not in evaluated:
            if len(evaluated) >= 512:
                return -1.0
            top, width, height, shear, *offsets = key
            correlations, selected = [], []
            for i, left in enumerate(offsets):
                codes = list(SEGMENTS)
                templates = np.stack(
                    [
                        mask((left, top, width, height, shear), SEGMENTS[code])
                        for code in codes
                    ]
                )
                scores = (signals[i] * templates).sum(axis=(1, 2)) / np.maximum(
                    signal_norm[i] * np.sqrt((templates * templates).sum(axis=(1, 2))),
                    1e-8,
                )
                best = int(np.argmax(scores))
                correlations.append(float(scores[best]))
                selected.append(codes[best])
            value = float(min(correlations) + 0.5 * np.mean(correlations))
            evaluated[key] = (value, correlations, "".join(selected))
        return evaluated[key][0]

    count = len(transforms)
    ranges = [(-0.05, 0.3), (0.3, 0.95), (0.6, 1.1), (-0.7, 0.7)] + [
        (-0.15, 0.45)
    ] * count
    starts = [
        [0.1, 0.55, 0.85, -0.25] + [0.0] * count,
        [0.05, 0.65, 0.9, 0] + [0.1] * count,
        [0.15, 0.5, 0.8, 0.25] + [0.25] * count,
    ]
    for start in starts:
        current = np.asarray(start, float)
        steps = np.array([0.1, 0.2, 0.15, 0.25] + [0.15] * count)
        for _ in range(3):
            for axis in list(range(4, 4 + count)) + [1, 0, 2, 3]:
                lo, hi = ranges[axis]
                choices = []
                for step in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    candidate = current.copy()
                    candidate[axis] = np.clip(
                        current[axis] + step * steps[axis], lo, hi
                    )
                    choices.append(candidate)
                current = max(choices, key=score)
            steps *= 0.5
    for parameters, (value, coverage, selected) in sorted(
        evaluated.items(), key=lambda item: item[1][0], reverse=True
    ):
        top, width, height, shear, *offsets = parameters
        if np.ptp(offsets) > 0.125:
            continue
        cells = []
        for left, matrix in zip(offsets, transforms, strict=True):
            points = []
            for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                v = top + height * y
                points.append([left + width * x + shear * (v - 0.5), v])
            cells.append(cv2.perspectiveTransform(np.float32([points]), matrix)[0])
        if any(
            (q < 0).any()
            or (q[:, 0] > image.shape[1]).any()
            or (q[:, 1] > image.shape[0]).any()
            for q in cells
        ):
            continue
        if any(
            cv2.intersectConvexConvex(a, b)[0] > 0.25
            for i, a in enumerate(cells)
            for b in cells[:i]
        ):
            continue
        return cells, dict(
            method="shared_positive_segment_lattice_coupled/2",
            status="geometry_candidate",
            parameters=list(parameters),
            score=value,
            coverage=coverage,
            geometry_component_models=list(selected),
            evaluated_geometries=len(evaluated),
            maximum_relative_slot_offset_spread=0.125,
            digit_labels_used=False,
            OCR_hypothesis_used=False,
            geometry_objective_resampled=True,
            OCR_pixels_resampled=False,
            physical_identity_inferred=False,
            quality_calibrated=False,
            **(
                dict(
                    vertical_search_margin=0.25, hypothesis="expanded_source_envelope/1"
                )
                if expanded
                else {}
            ),
        )
    return None, dict(status="pending", reason="shared_cell_geometry_escapes_source")
