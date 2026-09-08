from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

from visioncortex.open_vocabulary_runtime import load_yolo_world_with_local_clip


def test_world_text_encoder_loads_configured_path_without_named_download(tmp_path, monkeypatch):
    world = tmp_path / "world.pt"
    clip = tmp_path / "custom-name.pt"
    world.write_bytes(b"test")
    clip.write_bytes(b"test")
    calls = []
    model = SimpleNamespace(model=SimpleNamespace(clip_model=None))
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLOWorld = lambda path: model
    encoder = ModuleType("ultralytics.nn.text_model")

    def local_encoder(path, device):
        calls.append((path, device))
        return "local-encoder"

    encoder.CLIP = local_encoder
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.text_model", encoder)
    result = load_yolo_world_with_local_clip({"model_path": str(world), "clip_model_path": str(clip)})
    assert calls == [(str(clip), "cpu")]
    assert result.model.clip_model == "local-encoder"


def test_missing_local_clip_fails_before_import_or_download(tmp_path):
    world = tmp_path / "world.pt"
    world.write_bytes(b"test")
    with pytest.raises(FileNotFoundError, match="Configured open-vocabulary asset"):
        load_yolo_world_with_local_clip({"model_path": str(world), "clip_model_path": str(tmp_path / "missing.pt")})
