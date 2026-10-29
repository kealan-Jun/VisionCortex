"""Bounded QR decoding. Never infer identity from text or registry similarity."""

import time

import cv2
import numpy as np
import zxingcpp


def decode_qr(frame, *, fast=False):
    started = time.perf_counter()
    values, polygons, methods = [], [], []

    def add(value, polygon, method):
        if value and value not in values:
            values.append(value)
            polygons.append(np.asarray(polygon, dtype=float).reshape(4, 2).tolist())
            methods.append(method)

    if not fast:
        detector = cv2.QRCodeDetector()
        found, decoded, points, _ = detector.detectAndDecodeMulti(frame)
        if found and points is not None:
            for value, polygon in zip(decoded, points):
                add(value, polygon, "opencv_multi")
        if not values:
            value, polygon, _ = detector.detectAndDecode(frame)
            if value and polygon is not None:
                add(value, polygon, "opencv_single")

    # Independent decoder handles finder/threshold cases OpenCV misses.
    def read(image, method, scale=1):
        for result in zxingcpp.read_barcodes(image, formats=zxingcpp.BarcodeFormat.QRCode):
            p = result.position
            corners = [p.top_left, p.top_right, p.bottom_right, p.bottom_left]
            add(result.text, [[p.x / scale, p.y / scale] for p in corners], method)

    read(frame, "zxing_original")
    attempts = ["original"]
    if not values and not fast:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(2.0, 2048 / max(gray.shape))
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for name, image in [
            ("scaled", enlarged),
            ("contrast", cv2.createCLAHE(2, (8, 8)).apply(enlarged)),
        ]:
            attempts.append(name)
            read(image, "zxing_" + name, scale)
            if values:
                break
    diagnostics = {
        "decoder": "zxing-cpp/2.3.0" if fast else "opencv+zxing-cpp/2.3.0",
        "methods": methods,
        "attempts": attempts,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "decoded_count": len(values),
        "identity_inferred": False,
    }
    return values, polygons, diagnostics
