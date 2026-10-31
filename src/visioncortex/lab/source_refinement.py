"""Optional subpixel registration refinement using stable structure only."""

import cv2
import numpy as np


METHOD = "source_panel_stable_translation_refinement/1"


def refine(image, registration, reference, stable_mask):
    """Return a new registration, or the unchanged fit with a rejection receipt.

    ECC optimizes geometry on a temporary aligned view. Acceptance compares
    two single warps of original pixels, not a twice-interpolated image. Neither
    aligned image can become numerical evidence or an OCR input.
    """
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] * image.shape[1] > 262144
            or reference.dtype != np.uint8 or reference.ndim != 2
            or not 256 <= reference.size <= 262144
            or stable_mask.shape != reference.shape):
        raise ValueError("Bounded original image and stable reference mask required")
    matrix = np.asarray(registration["matrix_current_to_reference"], float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("Finite original structural registration required")
    height, width = reference.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aligned = cv2.warpPerspective(gray, matrix, (width, height))
    visible = cv2.warpPerspective(np.ones(gray.shape, np.uint8), matrix, (width, height))
    mask = np.uint8((stable_mask > 0) & (visible > 0)) * 255
    proof = dict(accepted=False, digits_excluded=True, native_source_pixels_used=True,
                 max_iterations_each_direction=30, numeric_pixels_supplied=False)

    def rejected(reason):
        return dict(registration, stable_translation_refinement=dict(proof, reason=reason))

    if np.count_nonzero(mask) < max(256, reference.size * .1):
        return rejected("insufficient_stable_structure_pixels")
    if float(reference[mask > 0].std()) < 8:
        return rejected("insufficient_stable_structure_texture")
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-4)
    try:
        _, forward = cv2.findTransformECC(reference, aligned, np.eye(2, 3, dtype=np.float32),
                                        cv2.MOTION_TRANSLATION, criteria, inputMask=mask, gaussFiltSize=5)
        _, reverse = cv2.findTransformECC(aligned, reference, np.eye(2, 3, dtype=np.float32),
                                        cv2.MOTION_TRANSLATION, criteria, inputMask=mask, gaussFiltSize=5)
    except cv2.error:
        return rejected("stable_structure_refinement_not_converged")
    if not np.isfinite(forward).all() or not np.isfinite(reverse).all():
        return rejected("nonfinite_stable_structure_refinement")
    shift = forward[:, 2]
    closure = float(np.linalg.norm(shift + reverse[:, 2]))
    proof.update(translation_pixels=shift.tolist(), reverse_closure_pixels=closure)
    if np.abs(shift).max() > 2 or closure > .35:
        return rejected("translation_or_reverse_consistency_bound")
    refined = np.linalg.inv(np.vstack([forward, [0, 0, 1]])) @ matrix
    after = cv2.warpPerspective(gray, refined, (width, height))
    after_visible = cv2.warpPerspective(np.ones(gray.shape, np.uint8), refined, (width, height))
    valid = (mask > 0) & (after_visible > 0)
    if np.count_nonzero(valid) < max(256, reference.size * .1):
        return rejected("refined_stable_structure_clipped")
    a, b, c = [v[valid].astype(float) for v in (reference, aligned, after)]
    if min(a.std(), b.std(), c.std()) < 8:
        return rejected("refined_structure_has_no_texture")
    before_corr, after_corr = float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(a, c)[0, 1])
    before_error, after_error = float(np.mean((a - b) ** 2)), float(np.mean((a - c) ** 2))
    proof.update(correlation_before=before_corr, correlation_after=after_corr,
                 squared_error_before=before_error, squared_error_after=after_error,
                 stable_source_pixels=int(valid.sum()))
    if after_corr < .85 or after_corr - before_corr < .02 or after_error >= .9 * before_error:
        return rejected("stable_source_structure_not_improved")
    return dict(registration, method=METHOD, matrix_current_to_reference=refined.tolist(),
                initial_structure_method=registration["method"],
                stable_inlier_scope="initial_structure_fit_before_translation_refinement",
                stable_translation_refinement=dict(proof, accepted=True))


def verified(registration):
    proof = registration.get("stable_translation_refinement", {})
    shift = np.asarray(proof.get("translation_pixels"), dtype=float)
    return bool(
        registration.get("initial_structure_method") in {
            "source_panel_bidirectional_flow/1", "source_panel_descriptor_reacquisition/1"}
        and proof.get("accepted") is True and proof.get("digits_excluded") is True
        and proof.get("native_source_pixels_used") is True
        and proof.get("numeric_pixels_supplied") is False
        and shift.shape == (2,) and np.isfinite(shift).all() and np.abs(shift).max() <= 2
        and 0 <= proof.get("reverse_closure_pixels", float("inf")) <= .35
        and proof.get("correlation_after", -1) >= .85
        and proof.get("correlation_after", -1) - proof.get("correlation_before", 1) >= .02
        and proof.get("squared_error_after", float("inf")) < .9 * proof.get("squared_error_before", 0)
    )
