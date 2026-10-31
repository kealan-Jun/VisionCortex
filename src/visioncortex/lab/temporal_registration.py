"""Stable-structure registration for bounded temporal display ROIs.

Digits and other declared dynamic areas never provide registration features.
The matrix maps the current ROI to the first ROI; failed fits return no image.
"""
from dataclasses import dataclass
import math

from .panel_vision import libraries


def normalized_box(box, shape):
    """Convert a checked normalized LTRB rectangle to half-open pixel bounds."""
    h, w = shape[:2]
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError('Finite normalized box required')
    x1, y1, x2, y2 = map(float, box)
    if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
        raise ValueError('Normalized box outside ROI')
    def pixel(value, extent, upper):
        scaled = value * extent
        nearest = round(scaled)
        # Integer pixel coordinates must survive normalization and JSON float
        # roundtrips. Snap only sub-nanopixel representation error; retain
        # conservative floor/ceil coverage for genuinely fractional edges.
        if abs(scaled - nearest) <= 1e-9:
            scaled = nearest
        return math.ceil(scaled) if upper else math.floor(scaled)

    left, right = pixel(x1, w, False), min(w, pixel(x2, w, True))
    top, bottom = pixel(y1, h, False), min(h, pixel(y2, h, True))
    # A valid tiny subpixel box must not collapse after representation snapping.
    if right <= left:
        left, right = math.floor(x1 * w), min(w, math.ceil(x2 * w))
    if bottom <= top:
        top, bottom = math.floor(y1 * h), min(h, math.ceil(y2 * h))
    return left, top, right, bottom


def stable_feature_mask(shape, dynamic_boxes):
    cv, np = libraries()
    if not dynamic_boxes:
        raise ValueError('Explicit dynamic digit/field exclusion is required')
    mask = np.full(shape[:2], 255, np.uint8)
    for box in dynamic_boxes:
        x1, y1, x2, y2 = normalized_box(box, shape)
        mask[max(0, y1-2):min(shape[0], y2+2), max(0, x1-2):min(shape[1], x2+2)] = 0
    if np.count_nonzero(mask) < max(32, mask.size*.08):
        raise ValueError('Insufficient stable structure outside dynamic digits')
    return mask


@dataclass(frozen=True)
class RegistrationConfig:
    mode: str = 'rigid'
    min_points: int = 6
    max_error_pixels: float = 1.5
    min_inlier_ratio: float = .65
    max_displacement_fraction: float = .25
    max_scale_change: float = .08


class TemporalRegistrar:
    def __init__(self, reference, dynamic_boxes, config=None):
        self.config = config or RegistrationConfig()
        if self.config.mode not in {'rigid', 'perspective'}:
            raise ValueError('Registration must be rigid or perspective')
        cv, np = libraries()
        self.gray = cv.cvtColor(reference, cv.COLOR_BGR2GRAY).copy()
        self.mask = stable_feature_mask(reference.shape, dynamic_boxes)
        self.shape = reference.shape
        self.points = cv.goodFeaturesToTrack(self.gray, maxCorners=160, qualityLevel=.015,
                                             minDistance=3, mask=self.mask)
        self.memory_bytes = self.gray.nbytes + self.mask.nbytes + (self.points.nbytes if self.points is not None else 0)

    def register(self, image):
        cv, np = libraries()
        failure = dict(status='rejected', accepted=False, method=self.config.mode,
                       matrix_current_to_reference=None, dynamic_features_excluded=True)
        if image.shape != self.shape:
            return None, dict(failure, reason='roi_shape_changed')
        if self.points is None or len(self.points) < self.config.min_points:
            return None, dict(failure, reason='insufficient_stable_features')
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        current, forward, _ = cv.calcOpticalFlowPyrLK(self.gray, gray, self.points, None,
                                                    winSize=(15, 15), maxLevel=2)
        if current is None:
            return None, dict(failure, reason='stable_tracking_failed')
        backward, reverse, _ = cv.calcOpticalFlowPyrLK(gray, self.gray, current, None,
                                                     winSize=(15, 15), maxLevel=2)
        if backward is None:
            return None, dict(failure, reason='backward_tracking_failed')
        source = current.reshape(-1, 2)
        target = self.points.reshape(-1, 2)
        backward_error = np.linalg.norm(backward.reshape(-1, 2)-target, axis=1)
        h, w = self.shape[:2]
        good = (forward.ravel() > 0) & (reverse.ravel() > 0) & (backward_error < 1.)
        for index, (x, y) in enumerate(source):
            if not (0 <= x < w and 0 <= y < h) or self.mask[min(h-1, max(0, int(y))), min(w-1, max(0, int(x)))] == 0:
                good[index] = False
        source, target = source[good], target[good]
        if len(source) < self.config.min_points:
            return None, dict(failure, reason='insufficient_bidirectional_structure_matches', matches=len(source))
        if self.config.mode == 'perspective':
            matrix, inliers = cv.findHomography(source, target, cv.RANSAC, self.config.max_error_pixels)
        else:
            affine, inliers = cv.estimateAffinePartial2D(source, target, method=cv.RANSAC,
                                                        ransacReprojThreshold=self.config.max_error_pixels)
            matrix = np.vstack([affine, [0., 0., 1.]]) if affine is not None else None
            if matrix is not None:
                scale = float(np.linalg.norm(matrix[:2, 0]))
                if not 1-self.config.max_scale_change <= scale <= 1+self.config.max_scale_change:
                    return None, dict(failure, reason='scale_change_exceeds_rigid_gate')
                # A rigid fit cannot stretch digit strokes. Refit translation
                # after projecting the linear part onto rotation only.
                selected = inliers.ravel().astype(bool)
                matrix[:2, :2] /= scale
                matrix[:2, 2] = target[selected].mean(axis=0)-matrix[:2, :2]@source[selected].mean(axis=0)
        if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
            return None, dict(failure, reason='degenerate_structure_fit')
        selected = inliers.ravel().astype(bool)
        if selected.sum() < self.config.min_points or selected.mean() < self.config.min_inlier_ratio:
            return None, dict(failure, reason='insufficient_structure_inliers')
        projected = cv.perspectiveTransform(source.reshape(1, -1, 2), matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected-target, axis=1)[selected]
        corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        warped = cv.perspectiveTransform(corners.reshape(1, 4, 2), matrix).reshape(4, 2)
        displacement = float(np.linalg.norm(warped-corners, axis=1).max())
        area = float(cv.contourArea(warped, oriented=True))
        if area <= 0 or displacement > self.config.max_displacement_fraction*max(h, w):
            return None, dict(failure, reason='geometry_or_motion_gate')
        if float(np.percentile(errors, 95)) > self.config.max_error_pixels:
            return None, dict(failure, reason='structure_reprojection_error')
        aligned = cv.warpPerspective(image, matrix, (w, h), flags=cv.INTER_LINEAR,
                                     borderMode=cv.BORDER_CONSTANT)
        # Structural appearance must still agree after the geometric fit. A
        # nearby different instrument must not inherit this track's buffers.
        aligned_gray = cv.cvtColor(aligned, cv.COLOR_BGR2GRAY)
        stable = self.mask > 0
        structural_difference = float(np.median(np.abs(aligned_gray[stable].astype(float)-self.gray[stable])))
        if structural_difference > 35:
            return None, dict(failure, reason='stable_structure_appearance_changed')
        return aligned, dict(status='registered', accepted=True, method=self.config.mode,
            matrix_current_to_reference=matrix.tolist(), matches=len(source), inliers=int(selected.sum()),
            inlier_ratio=float(selected.mean()), error_p95_pixels=float(np.percentile(errors, 95)),
            stable_structure_median_difference=structural_difference, dynamic_features_excluded=True,
            quality_calibrated_probability=False)
