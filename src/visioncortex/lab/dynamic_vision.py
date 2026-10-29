"""QR occurrences and explainable local pixel-change candidates, no hand model.

No output here confirms that a person operated equipment. QR identity and panel
geometry must still be resolved from the customer's explicit registration.
"""
from collections import defaultdict
import hashlib
import math

from .components import decode_qr
from .domain import fingerprint
from .panel_vision import libraries


SCHEMA = 'vc-lab-dynamic-vision/1'


def polygon(value):
    if not isinstance(value, list) or not 3 <= len(value) <= 8:
        raise ValueError('Bounded explicit polygon required')
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('Polygon points need x/y')
        if any(type(v) not in {float, int} or not math.isfinite(v) for v in point):
            raise ValueError('Finite polygon coordinates required')
        result.append([float(point[0]), float(point[1])])
    cv2, np = libraries()
    contour = np.asarray(result, dtype=np.float32)
    if abs(cv2.contourArea(contour)) < 1e-10 or not cv2.isContourConvex(contour):
        raise ValueError('Nondegenerate convex polygon required')
    return result


def matrix(value):
    cv2, np = libraries()
    del cv2
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.isfinite(array).all() or abs(np.linalg.det(array)) < 1e-12:
        raise ValueError('Finite invertible 3x3 registration required')
    return array


def project(points, transform):
    _, np = libraries()
    values = np.c_[np.asarray(polygon(points)), np.ones(len(points))] @ matrix(transform).T
    if np.any(np.abs(values[:, 2]) < 1e-10):
        raise ValueError('Projective horizon intersects panel')
    return (values[:, :2] / values[:, 2:]).tolist()


def contained(inner, outer):
    cv2, np = libraries()
    boundary = np.asarray(polygon(outer), dtype=np.float32)
    return all(cv2.pointPolygonTest(boundary, tuple(p), False) >= 0 for p in polygon(inner))


def frame_hash(image_bgr):
    cv2, np = libraries()
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.shape[0] * image_bgr.shape[1] > 9000000:
        raise ValueError('Bounded native uint8 BGR image required')
    return hashlib.sha256(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()


def qr_occurrences(image_bgr):
    """Reuse the old decoder, then retain spatial duplicates it deduplicates by text."""
    cv2, np = libraries()
    frame_hash(image_bgr)
    values, polygons, diagnostics = decode_qr(image_bgr)
    found = []
    def add(text, corners, method):
        points = polygon(np.asarray(corners, dtype=float).reshape(4, 2).tolist())
        centre = np.asarray(points).mean(axis=0)
        size = max(1., math.sqrt(abs(cv2.contourArea(np.asarray(points, dtype=np.float32)))))
        for old in found:
            if old['value'] == text and np.linalg.norm(centre - np.asarray(old['polygon']).mean(axis=0)) < .15 * size:
                if method not in old['methods']:
                    old['methods'].append(method)
                return
        found.append(dict(value=text, polygon=points, methods=[method]))
    for text, corners in zip(values, polygons):
        add(text, corners, 'existing_components_decode_qr')
    # The frozen decoder's `value not in values` suppresses repeated physical codes.
    import zxingcpp
    for result in zxingcpp.read_barcodes(image_bgr, formats=zxingcpp.BarcodeFormat.QRCode):
        p = result.position
        add(result.text, [[c.x, c.y] for c in [p.top_left, p.top_right, p.bottom_right, p.bottom_left]], 'zxing_spatial_occurrence')
    counts = defaultdict(int)
    for item in found:
        counts[item['value']] += 1
    for item in found:
        item['repeated_code_in_frame'] = counts[item['value']] > 1
    return dict(schema_version=SCHEMA, frame_rgb_sha256=frame_hash(image_bgr),
                occurrences=found, decoder_diagnostics=diagnostics,
                identity_or_operation_confirmed=False)


def interaction_pixels(previous_bgr, current_bgr, *, control_polygons, background_polygons,
                       min_delta=24, min_local_fraction=.015, max_background_fraction=.08,
                       min_component_pixels=6):
    """Actual pixel measurement, not hand/contact/action recognition.

    Call on registered equal-coordinate images. Large background differences
    reject the candidate rather than inventing camera-motion compensation.
    Dynamic digits/QR must not be designated as controls by the registered link.
    """
    cv2, np = libraries()
    old_sha, new_sha = frame_hash(previous_bgr), frame_hash(current_bgr)
    if previous_bgr.shape != current_bgr.shape or not control_polygons or not background_polygons:
        raise ValueError('Aligned equal-size images and explicit control/background regions required')
    if not 1 <= min_delta <= 255 or not 0 < min_local_fraction <= 1 or not 0 <= max_background_fraction < 1 or min_component_pixels < 1:
        raise ValueError('Explicit bounded pixel thresholds required')
    height, width = current_bgr.shape[:2]
    def mask(regions):
        output = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            points = np.asarray(polygon(region), dtype=float)
            if (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
                raise ValueError('Pixel region outside aligned frame')
            cv2.fillPoly(output, [np.rint(points).astype(np.int32)], 1)
        return output
    control, background = mask(control_polygons), mask(background_polygons)
    if np.any(control & background) or not control.any() or not background.any():
        raise ValueError('Control and independent background masks must be nonempty/disjoint')
    a, b = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(a, b)
    changed = difference >= min_delta
    local = (changed & (control > 0)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    control_count, background_count = int(control.sum()), int(background.sum())
    local_count, background_changed = int(local.sum()), int((changed & (background > 0)).sum())
    candidate = (local_count/control_count >= min_local_fraction and
                 background_changed/background_count <= max_background_fraction and largest >= min_component_pixels and old_sha != new_sha)
    body = dict(schema_version='vc-lab-local-control-change/1', previous_rgb_sha256=old_sha, current_rgb_sha256=new_sha,
                control_pixels=control_count, changed_control_pixels=local_count,
                background_pixels=background_count, changed_background_pixels=background_changed,
                largest_component_pixels=largest, control_mask_sha256=hashlib.sha256(control.tobytes()).hexdigest(),
                background_mask_sha256=hashlib.sha256(background.tobytes()).hexdigest(),
                control_polygons=control_polygons, background_polygons=background_polygons,
                thresholds=dict(min_delta=min_delta, min_local_fraction=min_local_fraction,
                                max_background_fraction=max_background_fraction, min_component_pixels=min_component_pixels),
                signal='localized_control_change_candidate' if candidate else 'no_local_interaction_support',
                background_gate_passed=background_changed/background_count <= max_background_fraction,
                observation='pixel_change_only; no_hand_model_no_contact_or_operation_semantics',
                pixel_measurement_sha256=None)
    body['pixel_measurement_sha256'] = fingerprint({k:v for k,v in body.items() if k != 'pixel_measurement_sha256'})
    return body


def interaction_candidate(measurement, current_rgb_sha256):
    """Validate metric arithmetic/hash, never trust `verified` or a supplied signal flag."""
    if not isinstance(measurement, dict) or measurement.get('schema_version') != 'vc-lab-local-control-change/1':
        return False
    try:
        if fingerprint({k:v for k,v in measurement.items() if k != 'pixel_measurement_sha256'}) != measurement.get('pixel_measurement_sha256'):
            return False
    except (ValueError, TypeError):
        return False
    if measurement.get('current_rgb_sha256') != current_rgb_sha256 or measurement.get('previous_rgb_sha256') == current_rgb_sha256:
        return False
    names = ['control_pixels','changed_control_pixels','background_pixels','changed_background_pixels','largest_component_pixels']
    if any(type(measurement.get(n)) is not int or measurement[n] < 0 for n in names):
        return False
    c, changed, b, bg, largest = [measurement[n] for n in names]
    thresholds = measurement.get('thresholds', {})
    if not isinstance(thresholds, dict):
        return False
    local = thresholds.get('min_local_fraction')
    background = thresholds.get('max_background_fraction')
    component = thresholds.get('min_component_pixels')
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in [local, background]) or type(component) is not int:
        return False
    if not 0 < local <= 1 or not 0 <= background < 1 or component < 1:
        return False
    # Conservative software gates cannot be disabled by an external `signal=True`.
    return bool(c and b and changed <= c and bg <= b and largest <= changed and
                changed/c >= max(.005, local) and
                bg/b <= min(.15, background) and
                largest >= max(3, component))
