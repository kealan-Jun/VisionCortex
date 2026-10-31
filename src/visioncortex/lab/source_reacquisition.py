"""Bounded scale-compatible structural search; never numerical evidence.

Sparse descriptor pairs and ECC only propose where to search. Acceptance still
requires the tracker's full fit on at least eight independently checked native
structural correspondences. Temporary resampling is confined to geometry.
"""

from itertools import combinations

import cv2
import numpy as np


METHOD = "source_panel_scale_compatible_reacquisition/1"


def _photometric(tracker, gray, reference, matrix):
    """Verify stable patches at the coarser native spatial resolution."""
    inverse = np.linalg.inv(matrix)
    kept, scores = [], []
    dx, dy = np.meshgrid(np.arange(-7, 8), np.arange(-7, 8))
    h, w = gray.shape
    for i, f in enumerate(reference):
        grid = np.float32(np.stack([dx + f[0], dy + f[1]], axis=-1))
        mapped = cv2.perspectiveTransform(grid.reshape(1, -1, 2), inverse)[0].reshape(15, 15, 2)
        rx, ry = grid[:, :, 0], grid[:, :, 1]
        if (rx.min() < 0 or ry.min() < 0 or rx.max() >= w - 1 or ry.max() >= h - 1
                or not np.isfinite(mapped).all() or mapped.min() < 0
                or mapped[:, :, 0].max() >= w - 1 or mapped[:, :, 1].max() >= h - 1):
            continue
        if not (cv2.remap(tracker.mask, rx, ry, cv2.INTER_NEAREST) > 0).all():
            continue
        p = cv2.remap(tracker.reference_gray, rx, ry, cv2.INTER_LINEAR).astype(float)
        q = cv2.remap(gray, mapped[:, :, 0], mapped[:, :, 1], cv2.INTER_LINEAR).astype(float)
        jacobian = np.column_stack([mapped[7, 8] - mapped[7, 7], mapped[8, 7] - mapped[7, 7]])
        scale = float(np.linalg.svd(jacobian, compute_uv=False).min())
        size = max(5, min(15, int(round(15 * scale))))
        p = cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)
        q = cv2.resize(q, (size, size), interpolation=cv2.INTER_AREA)
        if min(p.std(), q.std()) < 8:
            continue
        correlation = float(np.corrcoef(p.ravel(), q.ravel())[0, 1])
        if np.isfinite(correlation) and correlation >= .85:
            kept.append(i)
            scores.append(correlation)
    return kept, scores


def _search(tracker, image, gray, seed):
    x, y, xx, yy = tracker.box
    to_panel = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
    aligned = cv2.warpPerspective(gray, to_panel @ np.linalg.inv(seed), (xx - x, yy - y))
    reference = tracker.reference_gray[y:yy, x:xx]
    mask = tracker.mask[y:yy, x:xx]
    if np.count_nonzero(mask) < 256 or float(reference[mask > 0].std()) < 8:
        return None, "insufficient_stable_search_texture"
    try:
        _, correction = cv2.findTransformECC(
            reference, aligned, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-4),
            inputMask=mask, gaussFiltSize=5,
        )
        seed = seed @ np.linalg.inv(to_panel) @ np.vstack([correction, [0, 0, 1]]) @ to_panel
        if not np.isfinite(seed).all() or abs(np.linalg.det(seed)) < 1e-8:
            return None, "nonfinite_structural_search"
        # Both optical-flow inputs use the coarser measured native resolution.
        # Enlarging a smaller current image alone creates mismatched blur.
        center = np.float32([[[x + (xx-x)/2, y + (yy-y)/2],
                              [x + (xx-x)/2+1, y + (yy-y)/2],
                              [x + (xx-x)/2, y + (yy-y)/2+1]]])
        predicted = cv2.perspectiveTransform(center, seed)[0]
        scale = min(1., float(np.linalg.svd(np.column_stack(
            [predicted[1]-predicted[0], predicted[2]-predicted[0]]), compute_uv=False).min()))
        if not .3 <= scale <= 1:
            return None, "structural_search_resolution_bound"
        h, w = gray.shape
        shape = (round(w * scale), round(h * scale))
        sx, sy = shape[0]/w, shape[1]/h
        reduction = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
        coarse_ref = cv2.resize(tracker.reference_gray, shape, interpolation=cv2.INTER_AREA)
        coarse_current = cv2.warpPerspective(gray, reduction @ np.linalg.inv(seed), shape)
        feature_mask = np.zeros_like(tracker.mask)
        feature_mask[y:yy, x:xx] = mask
        points = cv2.goodFeaturesToTrack(tracker.reference_gray, maxCorners=160,
                                        qualityLevel=.01, minDistance=3, mask=feature_mask)
        if points is None or len(points) < 8:
            return None, "insufficient_reference_search_corners"
        coarse_points = cv2.perspectiveTransform(points, reduction)
        forward, a, _ = cv2.calcOpticalFlowPyrLK(
            coarse_ref, coarse_current, coarse_points, coarse_points.copy(),
            winSize=(21, 21), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if forward is None:
            return None, "no_scale_compatible_flow"
        reverse, b, _ = cv2.calcOpticalFlowPyrLK(
            coarse_current, coarse_ref, forward, coarse_points.copy(),
            winSize=(21, 21), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if reverse is None:
            return None, "no_reverse_scale_compatible_flow"
        unreduce = np.linalg.inv(reduction)
        forward = cv2.perspectiveTransform(forward, unreduce)
        reverse = cv2.perspectiveTransform(reverse, unreduce)
        current = cv2.perspectiveTransform(forward, seed)
        predicted = cv2.perspectiveTransform(points, seed)
        bound = min(32., .25 * min(xx-x, yy-y) * scale)
        good = ((a.ravel() > 0) & (b.ravel() > 0)
                & (np.linalg.norm(reverse[:, 0]-points[:, 0], axis=1) < .7)
                & (np.linalg.norm(current[:, 0]-predicted[:, 0], axis=1) <= bound))
        indices = []
        for i in np.flatnonzero(good):
            if all(np.linalg.norm(current[i, 0]-current[j, 0]) >= 3 for j in indices):
                indices.append(int(i))
        c, f = current[indices, 0], points[indices, 0]
        provisional, reason = tracker._fit(c, f, image, METHOD)
        if provisional is None:
            return None, reason
        kept, _ = _photometric(tracker, gray, f, provisional["matrix_full"])
        final, reason = tracker._fit(c[kept], f[kept], image, METHOD)
        if final is None:
            return None, reason
        # Recheck all final inliers against the final fit, rather than trusting
        # patches from a discarded provisional homography.
        kept, scores = _photometric(tracker, gray, final["reference"][:, 0], final["matrix_full"])
        if len(kept) != len(final["reference"]):
            return None, "final_structural_patch_verification"
        final["registration"]["scale_compatible_search"] = dict(
            accepted=True, sparse_seed_is_search_only=True, numeric_pixels_supplied=False,
            original_structure_fit_retained=True, max_seed_proposals=3,
            max_ECC_iterations=60, max_reference_corners=160,
            verified_patches=len(kept), minimum_patch_correlation=min(scores),
            optical_flow_common_scale=scale, temporary_images_geometry_only=True,
        )
        return final, None
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return None, "scale_compatible_search_failed"


def reacquire(tracker, image, gray, current, reference):
    """Use at most three separated descriptor pairs as search proposals."""
    if not 2 <= len(reference) <= 64 or len(current) != len(reference):
        return None, "bounded_sparse_structure_required"
    x, y, xx, yy = tracker.box
    pairs = sorted(combinations(range(len(reference)), 2),
                   key=lambda p: -float(np.linalg.norm(reference[p[1]]-reference[p[0]])))
    attempts = 0
    reason = "insufficient_separated_seed_structure"
    for i, j in pairs:
        d, e = reference[j]-reference[i], current[j]-current[i]
        if np.linalg.norm(d) < .25 * np.hypot(xx-x, yy-y) or np.linalg.norm(e) < 8:
            continue
        z = complex(*e) / complex(*d)
        if not .3 <= abs(z) <= 3:
            continue
        linear = np.array([[z.real, -z.imag], [z.imag, z.real]])
        seed = np.vstack([np.column_stack([linear, current[i]-linear @ reference[i]]), [0, 0, 1]])
        attempts += 1
        result, reason = _search(tracker, image, gray, seed)
        if result is not None:
            result["registration"]["scale_compatible_search"]["seed_proposals_used"] = attempts
            return result, None
        if attempts == 3:
            break
    return None, reason


def verified(registration):
    proof = registration.get("scale_compatible_search", {})
    return bool(
        proof.get("accepted") is True and proof.get("sparse_seed_is_search_only") is True
        and proof.get("numeric_pixels_supplied") is False
        and proof.get("original_structure_fit_retained") is True
        and proof.get("temporary_images_geometry_only") is True
        and proof.get("max_seed_proposals") == 3 and proof.get("max_ECC_iterations") == 60
        and proof.get("max_reference_corners") == 160
        and 1 <= proof.get("seed_proposals_used", 0) <= 3
        and proof.get("verified_patches", 0) >= 8
        and proof.get("minimum_patch_correlation", -1) >= .85
        and .3 <= proof.get("optical_flow_common_scale", 0) <= 1
    )
