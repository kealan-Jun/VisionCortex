"""One optional same-frame photometric view; not a source-pixel proof."""

import hashlib

import numpy as np


def color_residual(image):
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or not 1 <= image.shape[0] * image.shape[1] <= 262144):
        raise ValueError("Bounded native BGR view required")
    pixels = image.astype(np.float32)
    residual = np.maximum(pixels - pixels.min(axis=2)[:, :, None], 0)
    scale = max(1, float(np.percentile(residual.max(axis=2), 99)))
    prepared = np.uint8(np.minimum(255, residual * 255 / scale))
    return prepared, dict(
        method="same_frame_min_channel_subtraction_percentile99/1",
        source_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        prepared_bgr_sha256=hashlib.sha256(prepared.tobytes()).hexdigest(),
        percentile99=scale, resampled=False, source_pixels_replaced=False,
        supplies_stroke_or_point_evidence=False,
    )
