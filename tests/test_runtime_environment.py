from __future__ import annotations

from visioncortex.runtime_environment import configure_third_party_runtime


def test_ultralytics_config_uses_visioncortex_state_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    monkeypatch.delenv("VISIONCORTEX_ULTRALYTICS_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    destination = configure_third_party_runtime()

    assert destination == tmp_path / "VisionCortex" / "ThirdParty"


def test_explicit_ultralytics_directory_is_respected(monkeypatch, tmp_path):
    destination = tmp_path / "third-party"
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    monkeypatch.setenv("VISIONCORTEX_ULTRALYTICS_CONFIG_DIR", str(destination))

    assert configure_third_party_runtime() == destination
