"""Actual local pixels/generated QR; synthetic support, not site acceptance."""
from copy import deepcopy

import cv2
import numpy as np
import pytest
import qrcode

from visioncortex.lab.domain import fingerprint
from visioncortex.lab.dynamic_vision import (
    frame_hash, interaction_candidate, interaction_pixels, polygon,
    project, qr_occurrences,
)


CONTROL = [[[20, 20], [45, 20], [45, 45], [20, 45]]]
BACKGROUND = [[[60, 60], [90, 60], [90, 90], [60, 90]]]


def pixels():
    before = np.zeros((100, 100, 3), np.uint8)
    after = before.copy()
    after[25:40, 25:40] = 120
    return before, after


def test_actual_local_pixels_are_only_a_candidate_and_hashes_pin_inputs():
    before, after = pixels()
    result = interaction_pixels(before, after, control_polygons=CONTROL, background_polygons=BACKGROUND)
    assert result['changed_control_pixels'] == result['largest_component_pixels'] == 225
    assert result['changed_background_pixels'] == 0
    assert result['previous_rgb_sha256'] == frame_hash(before)
    assert interaction_candidate(result, frame_hash(after))
    assert result['observation'] == 'pixel_change_only; no_hand_model_no_contact_or_operation_semantics'


@pytest.mark.parametrize('kind', ['unchanged', 'global_motion', 'outside_controls'])
def test_no_change_camera_motion_and_screen_outside_controls_do_not_support_operation(kind):
    before, after = pixels()
    if kind == 'unchanged': after = before.copy()
    elif kind == 'global_motion': after[:] = 120
    else:
        after = before.copy()
        after[3:15, 3:40] = 120
    result = interaction_pixels(before, after, control_polygons=CONTROL, background_polygons=BACKGROUND)
    assert not interaction_candidate(result, frame_hash(after))


def test_supplied_verified_flag_or_tampered_arithmetic_does_not_pass():
    before, after = pixels()
    metric = interaction_pixels(before, before.copy(), control_polygons=CONTROL, background_polygons=BACKGROUND)
    metric.update(verified=True, signal='operation_confirmed')
    metric['pixel_measurement_sha256'] = fingerprint({k:v for k,v in metric.items() if k != 'pixel_measurement_sha256'})
    assert not interaction_candidate(metric, frame_hash(before))
    metric = interaction_pixels(before, after, control_polygons=CONTROL, background_polygons=BACKGROUND)
    metric['changed_control_pixels'] += 1
    assert not interaction_candidate(metric, frame_hash(after))
    metric = deepcopy(metric)
    metric['thresholds']['min_component_pixels'] = 'bad'
    metric['pixel_measurement_sha256'] = fingerprint({k:v for k,v in metric.items() if k != 'pixel_measurement_sha256'})
    assert not interaction_candidate(metric, frame_hash(after))


def test_two_actual_equal_qr_occurrences_are_preserved_as_ambiguity():
    qr = np.asarray(qrcode.make('synthetic-repeated-equipment-uuid').convert('RGB'))
    side = qr.shape[0]
    image = np.full((side+40, side*2+60, 3), 255, np.uint8)
    image[20:20+side, 20:20+side] = qr
    image[20:20+side, 40+side:40+side*2] = qr
    result = qr_occurrences(image)
    assert len(result['occurrences']) == 2
    assert all(x['repeated_code_in_frame'] for x in result['occurrences'])
    assert result['identity_or_operation_confirmed'] is False


def test_registration_and_masks_fail_closed():
    with pytest.raises(ValueError, match='convex'):
        polygon([[0, 0], [1, 1], [2, 2]])
    with pytest.raises(ValueError, match='invertible'):
        project(CONTROL[0], [[0,0,0],[0,0,0],[0,0,0]])
    before, after = pixels()
    with pytest.raises(ValueError, match='disjoint'):
        interaction_pixels(before, after, control_polygons=CONTROL, background_polygons=CONTROL)
    assert cv2.contourArea(np.asarray(polygon(CONTROL[0]), np.float32)) > 0
