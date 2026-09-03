from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from labvision_evidence import archive, liquid_semantic, temporal_segmentation
from labvision_evidence.config import load_config
from labvision_evidence import config as config_module
from labvision_evidence.detection import (
    FramePacket,
    RoleScanner,
    _engine_requires_exact_batch,
)
from labvision_evidence.schemas import ViewInput, ViewRole


ROOT = Path(__file__).resolve().parents[1]


def test_rtx3050_profile_preserves_full_chain_with_bounded_memory():
    config = load_config(ROOT / "configs/rtx3050-6gb-ubuntu20-production.yaml")

    performance = config["performance"]
    assert performance["profile"] == "rtx3050-6gb-ubuntu20-full-chain"
    assert performance["tensor_rt"] == "required"
    assert performance["engine_dynamic"] is False
    assert performance["engine_batch_candidates"] == [16, 8, 4, 2, 1]
    assert performance["fine_batch_size"] == 16
    assert performance["engine_autotune_iterations"] == 6
    assert performance["engine_autotune_max_gpu_memory_fraction"] <= 0.78
    assert performance["concurrent_role_scanners"] is False
    assert performance["release_auxiliary_models_after_event"] is True
    assert performance["max_gpu_memory_fraction"] <= 0.86
    assert config["mllm"]["enabled"] is True
    assert config["models"]["open_vocabulary_key_frame"]["enabled"] is True
    assert (
        config["models"]["open_vocabulary_key_frame"]["grounding_dino_fallback"][
            "device"
        ]
        == "cpu"
    )
    assert config["models"]["temporal_participant_segmentation"]["enabled"] is True
    assert config["models"]["liquid_semantic_sidecar"]["enabled"] is True
    assert config["models"]["liquid_semantic_sidecar"]["device"] == "cpu"
    certification_path = config["validation"]["model_certification"]["path"]
    assert certification_path == (
        "/opt/visioncortex-rtx3050/Runtime/Model-Quality/"
        "production_model_certification.json"
    )
    assert "3090" not in certification_path


def test_rtx3050_local_profile_cannot_inherit_nas_paths():
    config = load_config(ROOT / "configs/rtx3050-6gb-ubuntu20-local.yaml")

    assert config["mllm"]["enabled"] is False
    assert config["storage"]["sync_to_nas"] is False
    assert config["storage"]["run_output_mode"] == "local"
    for key in (
        "index_csv",
        "device_registry_path",
        "archive_root",
        "local_input_root",
        "local_runtime_root",
        "local_cache_root",
        "local_staging_root",
    ):
        assert str(config["storage"][key]).startswith(
            "/opt/visioncortex-rtx3050/Runtime/NoNasWeb/"
        )


def test_installed_runtime_can_resolve_sibling_default_config(tmp_path, monkeypatch):
    config_root = tmp_path / "app/configs"
    config_root.mkdir(parents=True)
    default = config_root / "default.yaml"
    profile = config_root / "target.yaml"
    default.write_text("performance:\n  batch_size: 1\n", encoding="utf-8")
    profile.write_text("performance:\n  batch_size: 8\n", encoding="utf-8")
    monkeypatch.setattr(
        config_module, "DEFAULT_CONFIG", tmp_path / "venv/configs/default.yaml"
    )

    loaded = load_config(profile)

    assert loaded["performance"]["batch_size"] == 8


def test_static_tensorrt_tail_is_padded_without_dropping_source_frames(tmp_path):
    metadata = json.dumps({"batch": 4, "dynamic": False}).encode("utf-8")
    engine = tmp_path / "model.engine"
    engine.write_bytes(len(metadata).to_bytes(4, "little") + metadata + b"fake-plan")
    assert _engine_requires_exact_batch(engine) is True

    class FakeModel:
        calls: list[int] = []

        def predict(self, *, source, **_kwargs):
            self.calls.append(len(source))
            return [SimpleNamespace(boxes=None) for _ in source]

    scanner = RoleScanner.__new__(RoleScanner)
    scanner.model_path = engine
    scanner.model = FakeModel()
    scanner.role = ViewRole.FIRST_PERSON
    scanner.config = {
        "models": {"confidence": 0.25, "iou": 0.55, "max_detections": 300},
        "performance": {"device": 0, "half": True},
    }
    scanner.names = {}
    scanner.requested_batch_size = 4
    scanner.engine_build_batch = 4
    scanner.engine_requires_exact_batch = True
    scanner.batch_size = 4
    scanner.initial_batch_size = 4
    scanner.batch_contractions = []
    scanner.last_inference_batch_sizes = []
    scanner.last_engine_batch_sizes = []
    scanner.exact_batch_padding_frames = 0
    scanner.image_size = 640
    view = ViewInput(
        view_id="first", role=ViewRole.FIRST_PERSON, video=Path("first.mp4")
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    gray = np.zeros((8, 8), dtype=np.uint8)
    packets = [
        FramePacket(
            view=view,
            frame_index=index,
            local_ms=float(index),
            frame=frame,
            gray=gray,
            previous_gray=None,
            motion_score=0.0,
        )
        for index in range(3)
    ]

    results = scanner.infer(packets)

    assert len(results) == 3
    assert scanner.model.calls == [4]
    assert scanner.last_inference_batch_sizes == [3]
    assert scanner.last_engine_batch_sizes == [4]
    assert scanner.exact_batch_padding_frames == 1


def test_low_memory_release_drops_all_auxiliary_model_caches():
    class FakeModel:
        def __init__(self):
            self.devices: list[str] = []

        def to(self, device: str):
            self.devices.append(device)
            return self

    yolo = FakeModel()
    dino = FakeModel()
    archive._OPEN_VOCABULARY_MODEL_CACHE["yolo"] = {"model": yolo}
    archive._GROUNDING_DINO_MODEL_CACHE["dino"] = {"model": dino}
    temporal_segmentation._MODEL_CACHE[("sam2",)] = {"predictor": object()}
    liquid_semantic._MODEL_CACHE[("labpics",)] = object()

    released = archive._release_auxiliary_model_caches()

    assert released == {
        "open_vocabulary": 1,
        "grounding_dino": 1,
        "temporal_segmentation": 1,
        "liquid_semantic": 1,
    }
    assert yolo.devices == ["cpu"]
    assert dino.devices == ["cpu"]
    assert not archive._OPEN_VOCABULARY_MODEL_CACHE
    assert not archive._GROUNDING_DINO_MODEL_CACHE
    assert not temporal_segmentation._MODEL_CACHE
    assert not liquid_semantic._MODEL_CACHE


def test_rtx3050_usb_scripts_are_offline_and_fail_closed():
    installer = (
        ROOT / "deployment/rtx3050-ubuntu20/Install-VisionCortex.sh"
    ).read_text(encoding="utf-8")
    preflight = (ROOT / "deployment/rtx3050-ubuntu20/00-Preflight.sh").read_text(
        encoding="utf-8"
    )

    assert "--offline --no-index" in installer
    assert "vendor/ffmpeg/bin/ffmpeg" in installer
    assert "--link-mode hardlink" in installer
    assert 'cache clean --cache-dir "$uv_install_cache"' in installer
    assert "prepare-engine" in installer
    assert "rtx3050-engine-smoke.json" in installer
    assert '"$runtime_root/Model-Quality"' in installer
    assert "libnvinfer_builder_resource_" in installer
    assert "RTX 3050" in preflight
    assert "minimum_free_gib=18" in preflight
    assert "ffmpeg_h264_nvenc=false" in preflight
    assert "nas_index_missing" in preflight


def test_package_manifest_rejects_escape(tmp_path):
    path = ROOT / "tools/build_rtx3050_offline_package.py"
    spec = importlib.util.spec_from_file_location("rtx3050_builder", path)
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    with pytest.raises((RuntimeError, ValueError)):
        builder._safe_manifest_path(tmp_path, "../escape")


def test_package_builder_rejects_case_insensitive_path_collisions(tmp_path):
    path = ROOT / "tools/build_rtx3050_offline_package.py"
    spec = importlib.util.spec_from_file_location("rtx3050_builder_casefold", path)
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    (tmp_path / "E").mkdir()
    try:
        (tmp_path / "e").mkdir()
    except FileExistsError:
        pytest.skip("test filesystem is already case-insensitive")

    with pytest.raises(RuntimeError, match="case-insensitive USB filesystem"):
        builder._assert_case_insensitive_filesystem_compatible(tmp_path)


def test_package_builder_fails_closed_on_offline_dependency_resolution():
    builder = (ROOT / "tools/build_rtx3050_offline_package.py").read_text(
        encoding="utf-8"
    )

    assert '"--dry-run"' in builder
    assert '"--offline"' in builder
    assert '"--no-index"' in builder
    assert "resolved_distribution_count" in builder
    assert "FAT32_unsupported" in builder
    assert 'output / "app" / source_relative' in builder
    assert '"VisionCortex-RTX3050-交付手册.md"' in builder
    assert 'output / "vendor/python/share/terminfo"' in builder
    assert "_assert_case_insensitive_filesystem_compatible(output)" in builder
