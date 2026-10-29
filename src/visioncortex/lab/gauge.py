"""Calibrated needle geometry. Does not detect a needle or infer a pressure reference."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math

from .domain import fingerprint


def exact(value):
    if isinstance(value, (bool, float)):
        raise ValueError('Exact decimal strings are required')
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError('Invalid decimal') from exc
    if not number.is_finite() or abs(number) > Decimal('1e12'):
        raise ValueError('Decimal outside finite bounds')
    return number


def point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('A two-coordinate point is required')
    return tuple(float(exact(x)) for x in value)


def angle(center, tip):
    # Image/dial plane has x right, y down; atan2 increases clockwise.
    return math.degrees(math.atan2(tip[1] - center[1], tip[0] - center[0])) % 360


def read_gauge(calibration, observation, *, source_sha256):
    return _read_scale(calibration, observation, source_sha256=source_sha256, accepted_tip='verified_needle_tip')


def read_automatic_gauge(calibration, observation, *, source_sha256):
    if observation.get('schema_version') != 'vc-lab-needle-detection/1' or observation.get('status') != 'candidate' or not observation.get('algorithm_revision'):
        raise ValueError('A source-pinned automatic needle candidate is required')
    if observation.get('source_sha256') != source_sha256:
        raise ValueError('Needle candidate belongs to another photograph')
    center, expected = point(observation['center']), point(calibration['center'])
    reach = math.dist(center, point(observation['tip']))
    if math.dist(center, expected) > reach * .01:
        return dict(value=None, status='requires_geometry_review',
                    issues=['detected_axis_differs_from_calibrated_axis'],
                    field_key=calibration.get('field_key', 'vacuum_pressure'), role='gauge_indication',
                    unit=calibration['unit'], source_sha256=source_sha256)
    result = _read_scale(calibration, observation, source_sha256=source_sha256, accepted_tip='algorithm_candidate')
    return dict(result, geometry_verification='automatic_candidate_not_manual_verification', needle_algorithm=observation['algorithm_revision'])


def _read_scale(calibration, observation, *, source_sha256, accepted_tip):
    """Interpolate verified dial-plane anchors, preserving uncertainty and sign.

    Perspective rectification/needle detection must precede this function. A
    source-specific geometry receipt prevents applying an old tip to a new photo.
    Both coordinate sets must be in the same rectified dial plane.
    """
    if calibration.get('schema_version') != 'vc-lab-gauge-calibration/1' or calibration.get('status') != 'verified':
        raise ValueError('Verified versioned gauge calibration required')
    if not calibration.get('revision') or not calibration.get('instrument_id') or not calibration.get('unit') or not calibration.get('verified_by'):
        raise ValueError('Gauge identity, unit and calibration author required')
    if observation.get('source_sha256') != source_sha256 or len(source_sha256) != 64:
        raise ValueError('Gauge geometry does not belong to this photograph')
    if observation.get('tip_status') != accepted_tip or observation.get('coordinate_frame') != 'rectified_dial_plane' or not observation.get('geometry_revision'):
        raise ValueError('Unverified tip, counterweight or unrectified geometry cannot produce a reading')
    direction = calibration.get('direction')
    if direction not in {'clockwise', 'counterclockwise'}:
        raise ValueError('Explicit scale direction required')
    center, tip = point(calibration['center']), point(observation['tip'])
    radius = math.dist(center, tip)
    lo, hi = (float(exact(v)) for v in calibration['tip_radius_range'])
    if not 0 < lo < hi or not lo <= radius <= hi:
        raise ValueError('Needle tip radius outside calibrated bounds')
    anchors = calibration['anchors']
    if not isinstance(anchors, list) or not 2 <= len(anchors) <= 64:
        raise ValueError('Two or more verified scale anchors required')
    sign = 1 if direction == 'clockwise' else -1
    origin = angle(center, point(anchors[0]['point']))
    def travel(deg):
        value = (sign * (deg - origin)) % 360
        return 0.0 if abs(value - 360) < 1e-8 else value
    positions = [travel(angle(center, point(a['point']))) for a in anchors]
    values = [exact(a['value']) for a in anchors]
    if any(math.dist(center, point(a['point'])) < 1e-8 for a in anchors):
        raise ValueError('Scale anchor cannot be the center')
    if not all(b > a + 1e-8 for a, b in zip(positions, positions[1:])) or positions[-1] >= 359:
        raise ValueError('Anchors must be ordered along an unambiguous scale arc')
    deltas = [b - a for a, b in zip(values, values[1:])]
    if not (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)):
        raise ValueError('Calibrated scale must be strictly monotonic')
    position = travel(angle(center, tip))
    uncertainty = float(exact(observation['angle_uncertainty_degrees']))
    maximum = float(exact(calibration['max_angle_uncertainty_degrees']))
    if not 0 <= uncertainty <= maximum < 90:
        raise ValueError('Needle angle uncertainty exceeds calibration acceptance')
    if position - uncertainty < -1e-8 or position + uncertainty > positions[-1] + 1e-8:
        return dict(status='outside_calibrated_arc', field_key=calibration.get('field_key', 'vacuum_pressure'),
                    role='gauge_indication', channel=calibration.get('channel', ''), value=None,
                    unit=calibration['unit'], source_sha256=source_sha256)
    def interpolate(p):
        p = min(positions[-1], max(0, p))
        for i in range(len(positions) - 1):
            if positions[i] <= p <= positions[i + 1]:
                fraction = Decimal(str((p - positions[i]) / (positions[i + 1] - positions[i])))
                return values[i] + fraction * (values[i + 1] - values[i])
        raise ValueError('Pointer does not intersect calibrated scale')
    quantum = exact(calibration['report_increment'])
    if quantum <= 0:
        raise ValueError('Positive reporting increment required')
    def rounded(value):
        return format((value / quantum).quantize(Decimal(1), rounding=ROUND_HALF_UP) * quantum, 'f')
    bounds = sorted([interpolate(position - uncertainty), interpolate(position + uncertainty)])
    return dict(status='candidate', field_key=calibration.get('field_key', 'vacuum_pressure'),
                role='gauge_indication', channel=calibration.get('channel', ''), value=rounded(interpolate(position)),
                unit=calibration['unit'], interval=[format(v, 'f') for v in bounds],
                calibration_revision=calibration['revision'], calibration_fingerprint=fingerprint(calibration),
                source_sha256=source_sha256, geometry_revision=observation['geometry_revision'],
                reading_stage=observation.get('reading_stage', 'unspecified'),
                pressure_reference=calibration.get('pressure_reference', 'unspecified'),
                stable_pressure_proven=False, accuracy='NOT_PROVEN')
