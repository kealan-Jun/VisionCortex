from types import SimpleNamespace
import sys

import numpy as np
import pytest

from visioncortex import archive, liquid_semantic, temporal_segmentation


class Model:
    def __init__(self):
        self.device = "cuda"

    def to(self, device):
        self.device = device
        return self


def test_park_reuses_same_weights_and_evicts_when_host_memory_is_low(monkeypatch):
    models = [Model() for _ in range(4)]
    monkeypatch.setattr(archive, "_OPEN_VOCABULARY_MODEL_CACHE", {"world": {"model": models[0]}})
    monkeypatch.setattr(archive, "_GROUNDING_DINO_MODEL_CACHE", {("dino", "sha", "cuda"): {"model": models[1]}})
    monkeypatch.setattr(temporal_segmentation, "_MODEL_CACHE", {("sam",): {"predictor": models[2]}})
    monkeypatch.setattr(liquid_semantic, "_MODEL_CACHE", {("labpics",): models[3]})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=12 * 1024**3))
    config = {"performance": {"auxiliary_cpu_cache_min_available_gib": 8}}
    receipt = archive._park_auxiliary_model_caches(config)
    assert receipt["retained_on_cpu"]
    assert all(model.device == "cpu" for model in models)
    assert temporal_segmentation._MODEL_CACHE[("sam",)]["predictor"] is models[2]
    assert liquid_semantic._MODEL_CACHE[("labpics",)] is models[3]
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=4 * 1024**3))
    assert not archive._park_auxiliary_model_caches(config)["retained_on_cpu"]
    assert not archive._OPEN_VOCABULARY_MODEL_CACHE
    assert not archive._GROUNDING_DINO_MODEL_CACHE
    assert not temporal_segmentation._MODEL_CACHE
    assert not liquid_semantic._MODEL_CACHE


def test_sam_warm_cache_restores_without_rebuilding(monkeypatch):
    model = Model()
    validation = {"checkpoint": "s", "checkpoint_sha256": "sha", "model_config": "cfg", "device": "cpu"}
    monkeypatch.setattr(temporal_segmentation, "validate_temporal_segmentation_runtime", lambda _: validation)
    monkeypatch.setattr(temporal_segmentation, "_MODEL_CACHE", {
        ("s", "sha", "cfg", "cpu", "False", "False"): {"predictor": model}})
    result, receipt = temporal_segmentation._load_predictor({})
    assert result is model and model.device == "cpu"
    assert receipt["model_cache_reused"] and receipt["model_load_seconds"] == 0


@pytest.mark.parametrize("component", ["dino", "labpics"])
@pytest.mark.parametrize("failure", ["oom", "invalid"])
def test_cuda_retry_uses_identical_input_and_only_retries_oom(monkeypatch, component, failure):
    class OOM(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM)))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    settings = {"device": "cuda", "cuda_oom_fallback_cpu": True, "half": False}
    calls = []

    def once(image, *args):
        assert image is frame
        cfg = args[-1]
        cfg = cfg["grounding_dino_fallback"] if component == "dino" else cfg["models"]["liquid_semantic_sidecar"]
        calls.append(cfg["device"])
        assert cfg["half"] is False
        if cfg["device"] == "cuda":
            raise OOM("allocation") if failure == "oom" else ValueError("invalid input")
        return [], {"status": "completed", "device": "cpu"}

    if component == "dino":
        monkeypatch.setattr(archive, "_grounding_dino_key_frame_detections_once", once)
        monkeypatch.setattr(archive, "_release_auxiliary_model_caches", lambda **_: {})
        def call():
            return archive._grounding_dino_key_frame_detections(frame, {"paper"}, {"grounding_dino_fallback": settings})
    else:
        monkeypatch.setattr(liquid_semantic, "_predict_liquid_masks_once", once)
        monkeypatch.setattr(liquid_semantic, "release_liquid_semantic_model_cache", lambda **_: 0)
        def call():
            return liquid_semantic.predict_liquid_masks(frame, {"models": {"liquid_semantic_sidecar": settings}})
    if failure == "oom":
        _, receipt = call()
        assert calls == ["cuda", "cpu"]
        assert receipt["device_fallback"] == "cuda_out_of_memory_to_cpu"
    else:
        with pytest.raises(ValueError, match="invalid input"):
            call()
        assert calls == ["cuda"]
