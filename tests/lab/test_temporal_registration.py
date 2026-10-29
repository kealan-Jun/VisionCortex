import numpy as np
import cv2
import pytest

from visioncortex.lab.temporal_registration import TemporalRegistrar, stable_feature_mask, RegistrationConfig


def structure():
    rng = np.random.default_rng(91)
    image = np.repeat(rng.integers(20, 190, (110, 200, 1), dtype=np.uint8), 3, axis=2)
    image[20:90, 55:145] = 10
    return image


def test_dynamic_pixels_cannot_supply_registration_features():
    image = np.zeros((100, 160, 3), np.uint8)
    cv2.putText(image, '889', (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    registrar = TemporalRegistrar(image, [[.1, .1, .9, .9]])
    assert registrar.points is None
    result, receipt = registrar.register(image)
    assert result is None
    assert receipt['reason'] == 'insufficient_stable_features'


def test_rigid_registration_uses_stable_border_despite_changed_digits():
    reference = structure()
    current = reference.copy()
    current[30:80, 70:130] = (0, 0, 240)
    transform = np.float32([[1, 0, 3], [0, 1, -2]])
    current = cv2.warpAffine(current, transform, (200, 110))
    aligned, receipt = TemporalRegistrar(reference, [[.25, .15, .75, .85]]).register(current)
    assert aligned is not None
    assert receipt['accepted']
    matrix = np.asarray(receipt['matrix_current_to_reference'])
    assert matrix[:2, 2] == pytest.approx([-3, 2], abs=.5)
    assert np.linalg.det(matrix[:2, :2]) == pytest.approx(1., abs=1e-6)


def test_registration_rejects_large_motion_and_shape_change():
    reference = structure()
    registrar = TemporalRegistrar(reference, [[.25, .15, .75, .85]])
    aligned, receipt = registrar.register(reference[:80])
    assert aligned is None and receipt['reason'] == 'roi_shape_changed'
    moved = cv2.warpAffine(reference, np.float32([[1, 0, 65], [0, 1, 0]]), (200, 110))
    aligned, receipt = registrar.register(moved)
    assert aligned is None
    assert not receipt['accepted']


def test_perspective_fit_retains_checked_projective_mapping():
    reference = structure()
    source = np.float32([[0, 0], [199, 0], [199, 109], [0, 109]])
    destination = np.float32([[2, 2], [196, 1], [198, 107], [1, 108]])
    matrix = cv2.getPerspectiveTransform(source, destination)
    current = cv2.warpPerspective(reference, matrix, (200, 110))
    aligned, receipt = TemporalRegistrar(reference, [[.25, .15, .75, .85]],
        RegistrationConfig(mode='perspective')).register(current)
    assert aligned is not None and receipt['accepted']
    recovered = np.asarray(receipt['matrix_current_to_reference'])@matrix
    recovered /= recovered[2, 2]
    assert recovered == pytest.approx(np.eye(3), abs=.5)


def test_missing_or_full_frame_dynamic_exclusion_fails_closed():
    with pytest.raises(ValueError):
        stable_feature_mask((100, 160, 3), [])
    with pytest.raises(ValueError):
        stable_feature_mask((100, 160, 3), [[0, 0, 1, 1]])
