"""Measure authorized source rows and prepare geometry-preserving OCR views.

These helpers neither invoke a model nor infer a reading, identity or OFF state.
Native geometry describes source pixels; resizing does not prove extra detail.
"""

import math

import cv2
import numpy as np


def extract_digits(image, region):
    points = np.asarray(region["polygon"], np.float32)
    width, height = region["size"]
    target = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    transform = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(
        image, transform, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return crop, transform


def red_display_geometry(image, x=0, y=0):
    """Locate a red character row using source pixels, without reading its value."""
    if image.ndim != 3 or min(image.shape[:2]) < 8:
        return None
    blue, green, red = cv2.split(image.astype(np.int16))
    red_mask = ((red - np.maximum(blue, green) > 18) & (red > 60)).astype(np.uint8)
    connected = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 12]
    if not contours:
        return None
    contours.sort(key=cv2.contourArea, reverse=True)
    if len(contours) > 1 and cv2.contourArea(contours[1]) > cv2.contourArea(contours[0]) * 0.6:
        return None  # More than one comparable row inside this authorized panel.
    window = np.zeros(red_mask.shape, np.uint8)
    cv2.drawContours(window, [contours[0]], -1, 1, cv2.FILLED)
    channel = np.maximum(blue, green).astype(np.uint8)
    values = channel[window.astype(bool)]
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.percentile(values, 95) - threshold < 15:
        return None
    foreground = ((channel > threshold) & window.astype(bool)).astype(np.uint8)
    points = cv2.findNonZero(foreground)
    if points is None or len(points) < 16:
        return None
    _, glyph_size, _ = cv2.minAreaRect(points)
    if min(glyph_size) < 7:
        return None
    # Include the measured red window around the luminous row; a bright
    # threshold alone could clip a faint leading digit from the source crop.
    center, (width, height), angle = cv2.minAreaRect(contours[0])
    if height > width:
        width, height, angle = height, width, angle + 90
    angle = (angle + 90) % 180 - 90
    if abs(angle) > 45 or height < 7 or width < height * 1.1:
        return None
    radians = math.radians(angle)
    along = np.array([math.cos(radians), math.sin(radians)])
    down = np.array([-along[1], along[0]])
    center = np.asarray(center)
    # A source-pixel margin includes complete bars, including faint OFF stems.
    width, height = width + 4, height + 4
    quad = np.asarray(
        [
            center - along * width / 2 - down * height / 2,
            center + along * width / 2 - down * height / 2,
            center + along * width / 2 + down * height / 2,
            center - along * width / 2 + down * height / 2,
        ],
        np.float32,
    )
    quad[:, 0] = np.clip(quad[:, 0], 0, image.shape[1] - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, image.shape[0] - 1)
    native_size = [max(8, round(width)), max(8, round(height))]
    return {
        "method": "source_red_character_geometry_v1",
        "polygon": (quad + [x, y]).tolist(),
        "native_size": native_size,
        "roll_degrees": round(angle, 3),
        "threshold": round(threshold, 3),
        "foreground_pixel_count": int(len(points)),
        "foreground_native_span": [round(float(v), 3) for v in glyph_size],
    }


def _red_ocr_views(image, region):
    """Use unsaturated source channels; never draw, join or replace a stroke."""
    _, transform = extract_digits(image, region)
    blue, green, red = cv2.split(image.astype(np.int16))
    inside = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, np.rint(region["polygon"]).astype(np.int32), 1)
    red_source = (inside != 0) & (red - np.maximum(blue, green) > 18) & (red > 60)
    samples = green[inside != 0]
    diagnostics = {
        "method": "source_unsaturated_channels_v2",
        "source_pixel_count": int(samples.size),
        "red_source_pixel_count": int(np.count_nonzero(red_source)),
        "green_range_basis": "complete_source_polygon_min_max",
    }
    if samples.size < 16 or np.count_nonzero(red_source) < 16:
        return [], transform, dict(diagnostics, status="insufficient_source_pixels")
    low, high = float(samples.min()), float(samples.max())
    diagnostics.update(green_low=round(float(low), 3), green_high=round(float(high), 3))
    if high - low < 15:
        return [], transform, dict(diagnostics, status="insufficient_source_contrast")
    # Red LEDs often saturate R. B/G preserve their brightness difference from
    # the red surround. The second view uses a different source channel and
    # complete source contrast range. A red-only percentile would exclude bright
    # bars whose channels are less saturated, then clip them to the same value.
    # Neither operation changes character geometry.
    channels = [
        ("max_bg_negative", (255 - np.maximum(blue, green)).astype(np.uint8)),
        (
            "green_contrast_negative",
            255 - np.rint(np.clip((green - low) * 255 / (high - low), 0, 255)).astype(np.uint8),
        ),
    ]
    views = [
        (
            name,
            cv2.warpPerspective(
                cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR),
                transform,
                tuple(region["size"]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            ),
        )
        for name, channel in channels
    ]
    if np.array_equal(views[0][1], views[1][1]):
        return [], transform, dict(diagnostics, status="duplicate_preprocessed_views")
    return views, transform, dict(diagnostics, status="prepared")


def _off_ocr_views(image, region):
    """Prepare a fixed source pair after one complete native OFF-like pattern.

    Both views use the unchanged source quad. They are correlated transforms,
    not new source detail or independent truth. The caller must first verify
    one complete native OFF-like pattern; two actual exact OFF OCR readings
    are still required. No prior OCR text selects these views.
    """
    diagnostics = {"method": "source_luminance_and_unsaturated_channels_v1"}
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.size
        or image.dtype != np.uint8
    ):
        return [], np.eye(3), dict(diagnostics, status="invalid_source_image")
    original_views, transform, source_channels = _red_ocr_views(image, region)
    diagnostics["source_channels"] = source_channels
    if len(original_views) != 2:
        return (
            [],
            transform,
            dict(diagnostics, status=source_channels.get("status", "not_prepared")),
        )
    max_bg = next(view for name, view in original_views if name == "max_bg_negative")
    luminance_negative = cv2.cvtColor(
        255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
    )
    luminance_view = cv2.warpPerspective(
        luminance_negative,
        transform,
        tuple(region["size"]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if np.array_equal(luminance_view, max_bg):
        return [], transform, dict(diagnostics, status="duplicate_preprocessed_views")
    return (
        [("source_luminance_negative", luminance_view), ("max_bg_negative", max_bg)],
        transform,
        dict(diagnostics, status="prepared"),
    )
