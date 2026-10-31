"""Track stable panel structure in the source frame, outside changing digits.

The first source frame supplies an appearance reference, never a stored number
or equipment identity. Lost tracks are pending; their old ROI is not reused as
if it still covered the display. Numeric crops retain original source pixels.
"""

import hashlib
import math

import cv2
import numpy as np

from .temporal_registration import normalized_box


class SourcePanelTracker:
    def __init__(self, image, roi_box, dynamic_boxes):
        self.shape = image.shape
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Source tracker requires uint8 BGR pixels")
        if image.shape[0] * image.shape[1] > 1048576:
            raise ValueError("Source tracker exceeds its one-megapixel bound")
        self.box = normalized_box(roi_box, image.shape)
        x, y, xx, yy = self.box
        if (xx - x) * (yy - y) > 262144 or not dynamic_boxes:
            raise ValueError(
                "Bounded panel and explicit changing-field exclusions required"
            )
        self.reference_shape = (yy - y, xx - x, 3)
        mask = np.zeros(image.shape[:2], np.uint8)
        # Include nearby faceplate lettering and borders: a bare LED window
        # and protruding knobs alone need not supply enough planar structure.
        padx, pady = math.ceil((xx - x) * 0.5), math.ceil((yy - y) * 0.5)
        mask[
            max(0, y - pady) : min(image.shape[0], yy + pady),
            max(0, x - padx) : min(image.shape[1], xx + padx),
        ] = 255
        for b in dynamic_boxes:
            a, t, c, d = normalized_box(b, self.reference_shape)
            mask[
                max(y, y + t - 3) : min(yy, y + d + 3),
                max(x, x + a - 3) : min(xx, x + c + 3),
            ] = 0
        # Nominal field coordinates may lag the camera pose. Exclude observed
        # emissive pixels too, before they can become registration features.
        colors = image.astype(np.int16)
        changing = (
            colors[:, :, 2] - np.maximum(colors[:, :, 0], colors[:, :, 1]) > 35
        ) | (colors[:, :, 1] - np.maximum(colors[:, :, 0], colors[:, :, 2]) > 35)
        changing = cv2.dilate(changing.astype(np.uint8), np.ones((7, 7), np.uint8))
        mask[changing > 0] = 0
        self.reference_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.mask = mask
        self.previous_gray = self.reference_gray
        self.reference_points = cv2.goodFeaturesToTrack(
            self.reference_gray,
            maxCorners=160,
            qualityLevel=0.01,
            minDistance=3,
            mask=mask,
        )
        self.current_points = self.reference_points
        self.feature = cv2.SIFT_create(nfeatures=600, contrastThreshold=0.025)
        self.keypoints, self.descriptors = self.feature.detectAndCompute(
            self.reference_gray, mask
        )
        self.step_count = 0
        self.last_reacquisition = -15
        self.last_failure = None
        self.pending_flow_frames = 0
        self.reference_sha256 = hashlib.sha256(image.tobytes()).hexdigest()

    @property
    def buffer_bytes(self):
        arrays = [
            self.reference_gray,
            self.mask,
            self.previous_gray,
            self.reference_points,
            self.current_points,
            self.descriptors,
        ]
        return sum(
            a.nbytes
            for i, a in enumerate(arrays)
            if a is not None and all(a is not v for v in arrays[:i])
        )

    def _fit(self, current, reference, image, method):
        if len(current) < 8:
            return None, "insufficient_stable_structure"
        matrix, inliers = cv2.findHomography(current, reference, cv2.RANSAC, 1.5)
        if matrix is None or not np.isfinite(matrix).all():
            return None, "degenerate_structure_fit"
        good = inliers.ravel().astype(bool)
        if good.sum() < 8 or good.mean() < 0.65:
            return None, "insufficient_structure_inliers"
        x, y, xx, yy = self.box
        spread = np.ptp(reference[good], axis=0)
        if spread[0] < (xx - x) * 0.25 or spread[1] < (yy - y) * 0.25:
            return None, "structure_matches_do_not_cover_panel"
        corners = np.float32([[[x, y], [xx, y], [xx, yy], [x, yy]]])
        try:
            polygon = cv2.perspectiveTransform(corners, np.linalg.inv(matrix))[0]
        except np.linalg.LinAlgError:
            return None, "singular_structure_fit"
        h, w = image.shape[:2]
        if (
            not np.isfinite(polygon).all()
            or not cv2.isContourConvex(polygon)
            or cv2.contourArea(polygon, oriented=True) <= 0
            or (polygon < 0).any()
            or (polygon[:, 0] >= w).any()
            or (polygon[:, 1] >= h).any()
        ):
            return None, "tracked_panel_clipped_or_mirrored"
        area_ratio = cv2.contourArea(polygon) / ((xx - x) * (yy - y))
        if not 0.5 <= area_ratio <= 2:
            return None, "panel_scale_gate"
        projected = cv2.perspectiveTransform(current[None], matrix)[0]
        error = np.linalg.norm(projected - reference, axis=1)[good]
        if np.percentile(error, 95) > 1.5:
            return None, "structure_reprojection_error"
        stable = self.mask[y:yy, x:xx] > 0
        to_panel = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]]) @ matrix
        aligned = cv2.warpPerspective(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), to_panel, (xx - x, yy - y)
        )
        difference = np.abs(aligned.astype(float) - self.reference_gray[y:yy, x:xx])
        if not stable.any() or np.median(difference[stable]) > 35:
            return None, "stable_panel_appearance_changed"
        left, top = np.floor(polygon.min(axis=0)).astype(int)
        right, bottom = np.ceil(polygon.max(axis=0)).astype(int)
        if (right - left) * (bottom - top) > 262144:
            return None, "tracked_native_ROI_exceeds_budget"
        crop = np.ascontiguousarray(image[top:bottom, left:right])
        to_reference = (
            np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
            @ matrix
            @ np.array([[1, 0, left], [0, 1, top], [0, 0, 1]])
        )
        return dict(
            image=crop,
            matrix_full=matrix,
            box=[int(v) for v in (left, top, right, bottom)],
            registration=dict(
                accepted=True,
                status="registered",
                method=method,
                matrix_current_to_reference=to_reference.tolist(),
                reference_shape=list(self.reference_shape),
                input_bgr_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                upright_frame_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                first_source_frame_bgr_sha256=self.reference_sha256,
                stable_inliers=int(good.sum()),
                error_p95_pixels=float(np.percentile(error, 95)),
                dynamic_features_excluded=True,
                equipment_identity_inferred=False,
                reacquired_after_loss=self.last_failure is not None,
            ),
            current=current[good].reshape(-1, 1, 2),
            reference=reference[good].reshape(-1, 1, 2),
        ), None

    def observe(self, image):
        if image.shape != self.shape or image.dtype != np.uint8:
            raise ValueError("Source dimensions changed; new track scope required")
        self.step_count += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result, reason = None, "insufficient_stable_structure"
        diagnostics = dict(
            flow_points_before=len(self.current_points)
            if self.current_points is not None
            else 0,
            reference_descriptor_count=len(self.keypoints),
            descriptor_search=False,
        )
        pending_points = pending_reference = None
        if self.current_points is not None and len(self.current_points) >= 8:
            points, a, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                self.current_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            if points is not None:
                reverse, b, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    self.previous_gray,
                    points,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                )
                if reverse is not None:
                    good = (
                        (a.ravel() > 0)
                        & (b.ravel() > 0)
                        & (
                            np.linalg.norm(
                                reverse[:, 0] - self.current_points[:, 0], axis=1
                            )
                            < 0.7
                        )
                    )
                    diagnostics["bidirectional_flow_matches"] = int(good.sum())
                    if good.sum() >= 8:
                        pending_points = points[good]
                        pending_reference = self.reference_points[good]
                    result, reason = self._fit(
                        points[good, 0],
                        self.reference_points[good, 0],
                        image,
                        "source_panel_bidirectional_flow/1",
                    )
                    diagnostics["flow_fit_rejection"] = reason
        # Reacquisition has its own rate bound; a failing frame cannot invoke a
        # descriptor search, or any neural OCR, in an unbounded retry loop.
        if (
            result is None
            and self.step_count - self.last_reacquisition >= 15
            and self.descriptors is not None
            and len(self.keypoints) >= 12
        ):
            self.last_reacquisition = self.step_count
            diagnostics["descriptor_search"] = True
            points, descriptors = cv2.SIFT_create(
                nfeatures=6000, contrastThreshold=0.025
            ).detectAndCompute(gray, None)
            if descriptors is not None and len(points) >= 2:
                pairs = cv2.BFMatcher().knnMatch(self.descriptors, descriptors, k=2)
                matches = [
                    a
                    for pair in pairs
                    if len(pair) == 2
                    for a, b in [pair]
                    if a.distance < 0.7 * b.distance
                ]
                unique = {}
                for m in sorted(matches, key=lambda v: v.distance):
                    unique.setdefault(m.trainIdx, m)
                matches = list(unique.values())
                diagnostics["descriptor_matches"] = len(matches)
                current = np.float32([points[m.trainIdx].pt for m in matches]).reshape(
                    -1, 2
                )
                reference = np.float32(
                    [self.keypoints[m.queryIdx].pt for m in matches]
                ).reshape(-1, 2)
                result, reason = self._fit(
                    current, reference, image, "source_panel_descriptor_reacquisition/1"
                )
        self.previous_gray = gray
        if result is None:
            self.pending_flow_frames += 1
            # Keep source-point hypotheses briefly across an unusable fit.
            # They authorize no ROI/OCR until the original geometry gates pass.
            if pending_points is not None and self.pending_flow_frames <= 5:
                self.current_points = pending_points
                self.reference_points = pending_reference
            else:
                self.current_points = None
            self.last_failure = reason
            return dict(
                status="pending",
                reason=reason,
                geometry=None,
                stale_ROI_reused=False,
                equipment_identity_inferred=False,
                diagnostics=diagnostics,
            )
        self.current_points = result.pop("current")
        self.reference_points = result.pop("reference")
        matrix = result.pop("matrix_full")
        # A monotonically shrinking feature set eventually loses even an
        # unchanged panel. New points must match the original stable structure,
        # rather than accumulating coordinates from a drifting chained fit.
        mask = cv2.warpPerspective(
            self.mask,
            np.linalg.inv(matrix),
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        for x, y in self.current_points[:, 0]:
            cv2.circle(mask, (round(float(x)), round(float(y))), 5, 0, -1)
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=160, qualityLevel=0.01, minDistance=3, mask=mask
        )
        if points is not None:
            predicted = cv2.perspectiveTransform(points, matrix)
            matched, a, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self.reference_gray,
                points,
                predicted.copy(),
                winSize=(21, 21),
                maxLevel=3,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )
            if matched is not None:
                back, b, _ = cv2.calcOpticalFlowPyrLK(
                    self.reference_gray,
                    gray,
                    matched,
                    points.copy(),
                    winSize=(21, 21),
                    maxLevel=3,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                )
                if back is not None:
                    good = (
                        (a.ravel() > 0)
                        & (b.ravel() > 0)
                        & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 0.7)
                        & (
                            np.linalg.norm(matched[:, 0] - predicted[:, 0], axis=1)
                            < 1.5
                        )
                    )
                    h, w = self.shape[:2]
                    for i, (x, y) in enumerate(matched[:, 0]):
                        if (
                            not (0 <= x < w and 0 <= y < h)
                            or self.mask[
                                min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))
                            ]
                            == 0
                        ):
                            good[i] = False
                    self.current_points = np.concatenate(
                        [self.current_points, points[good]]
                    )[:160]
                    self.reference_points = np.concatenate(
                        [self.reference_points, matched[good]]
                    )[:160]
        self.last_failure = None
        self.pending_flow_frames = 0
        result["registration"]["tracking_diagnostics"] = diagnostics
        return dict(status="tracked", **result)


def align_native_roi(image, registration, source_ref):
    """Validate the source-bound tracker result before temporal OCR consumes it."""
    if (
        registration.get("accepted") is not True
        or registration.get("dynamic_features_excluded") is not True
        or registration.get("equipment_identity_inferred") is not False
        or registration.get("method")
        not in {
            "source_panel_bidirectional_flow/1",
            "source_panel_descriptor_reacquisition/1",
        }
        or registration.get("input_bgr_sha256")
        != hashlib.sha256(image.tobytes()).hexdigest()
        or registration.get("upright_frame_bgr_sha256")
        != source_ref.get("upright_frame_pixels_sha256")
    ):
        raise ValueError("Source-bound stable panel geometry required")
    shape = registration.get("reference_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[2] != 3
        or any(type(v) is not int or v < 8 for v in shape[:2])
        or math.prod(shape[:2]) > 262144
    ):
        raise ValueError("Bounded tracked reference extent required")
    matrix = np.asarray(registration["matrix_current_to_reference"], dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(np.linalg.det(matrix)) < 1e-8
    ):
        raise ValueError("Finite invertible tracked transform required")
    return cv2.warpPerspective(image, matrix, (shape[1], shape[0])), tuple(shape)
