from copy import deepcopy

import pytest

from visioncortex.detection import RoleScanner
from visioncortex.detection_thresholds import prediction_confidence
from visioncortex.scan_dependencies import phase_config
from visioncortex.schemas import ViewRole


def test_calibrated_fine_threshold_does_not_change_coarse_or_other_role(default_config):
    config = deepcopy(default_config)
    config['models']['confidence_by_role'] = {'first_person': .225}
    scanner = object.__new__(RoleScanner)
    scanner.config, scanner.role = config, ViewRole.FIRST_PERSON
    scanner.image_size, scanner.prediction_end2end = 640, None
    assert scanner._prediction_options()['conf'] == .225
    assert prediction_confidence(config, ViewRole.THIRD_PERSON) == config['models']['confidence']
    coarse = phase_config(config, 'coarse')
    assert prediction_confidence(coarse, ViewRole.FIRST_PERSON) == config['models']['confidence']
    assert config['models']['confidence_by_role'] == {'first_person': .225}
    config['models']['coarse_confidence_by_role'] = {'first_person': .31}
    assert prediction_confidence(phase_config(config, 'coarse'), ViewRole.FIRST_PERSON) == .31


@pytest.mark.parametrize('bad', [True, 0, -1, 2, float('nan'), float('inf'), '0.2'])
def test_bad_threshold_cannot_silently_enter_predictor(default_config, bad):
    config = deepcopy(default_config)
    config['models']['confidence_by_role'] = {'first_person': bad}
    with pytest.raises(ValueError):
        prediction_confidence(config, ViewRole.FIRST_PERSON)


def test_role_mapping_rejects_camera_ids(default_config):
    config = deepcopy(default_config)
    config['models']['confidence_by_role'] = {'camera-one': .2}
    with pytest.raises(ValueError):
        prediction_confidence(config, ViewRole.FIRST_PERSON)
