from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

from visioncortex import detection


CLASSES = {
    str(index): name
    for index, name in enumerate(
        [
            "balance",
            "beaker",
            "gloved_hand",
            "lab_coat",
            "paper",
            "reagent_bottle",
            "sample_bottle",
            "sample_bottle_blue",
            "spatula",
            "tube",
            "tube-cap",
            "spearhead",
            "pipette",
            "container",
            "PPE_Storage",
            "hand",
            "reagent_bottle_open",
            "bottle_cap",
            "magnetic_stirrer",
            "tube_rack",
            "magnetic_stir_bar",
        ]
    )
}


class _Logger:
    ERROR = 0

    def __init__(self, _level):
        pass


class _Runtime:
    def __init__(self, _logger):
        pass

    def deserialize_cuda_engine(self, plan):
        return SimpleNamespace(plan=plan)


def _config(tmp_path):
    first_engine = tmp_path / "first.engine"
    third_engine = tmp_path / "third.engine"
    first_engine.write_bytes(b"first-plan")
    third_engine.write_bytes(b"third-plan")
    return {
        "models": {
            "expected_class_count": 21,
            "first_person": str(tmp_path / "missing-first.pt"),
            "third_person": str(tmp_path / "missing-third.pt"),
            "first_person_engine": str(first_engine),
            "third_person_engine": str(third_engine),
            "open_vocabulary_key_frame": {"enabled": False},
            "temporal_participant_segmentation": {"enabled": False},
            "liquid_semantic_sidecar": {"enabled": False},
        },
        "performance": {"tensor_rt": "required"},
    }


def _install_fake_tensorrt(monkeypatch, *, mismatch=False):
    monkeypatch.setitem(
        sys.modules,
        "tensorrt",
        SimpleNamespace(__version__="10.test", Logger=_Logger, Runtime=_Runtime),
    )

    def metadata(path):
        names = dict(CLASSES)
        if mismatch and path.name.startswith("third"):
            names["20"] = "different_class"
        return b"plan", {"batch": 4, "names": names}, "ultralytics"

    monkeypatch.setattr(detection, "_tensorrt_plan_and_metadata", metadata)


def test_required_tensorrt_validates_engine_only_deployment(tmp_path, monkeypatch):
    _install_fake_tensorrt(monkeypatch)

    report = detection.validate_models(_config(tmp_path))

    assert report["consistent"] is True
    assert report["first_person"]["source_model_available"] is False
    assert report["first_person"]["validation_source"] == "tensorrt_engine_metadata"
    assert report["first_person"]["classes"][10] == "tube_cap"
    assert report["runtime"]["roles"]["third_person"]["deserialized"] is True
    assert report["runtime"]["roles"]["third_person"]["build_batch"] == 4
    assert report["runtime"]["roles"]["first_person"]["sha256"] == hashlib.sha256(
        b"first-plan"
    ).hexdigest()


def test_required_tensorrt_fails_closed_on_cross_role_class_mismatch(
    tmp_path, monkeypatch
):
    _install_fake_tensorrt(monkeypatch, mismatch=True)

    with pytest.raises(ValueError, match="规范化类别表不一致"):
        detection.validate_models(_config(tmp_path))


def test_non_tensorrt_mode_still_requires_source_weights(tmp_path):
    config = _config(tmp_path)
    config["performance"]["tensor_rt"] = "auto"

    with pytest.raises(FileNotFoundError, match="first_person 模型不存在"):
        detection.validate_models(config)
