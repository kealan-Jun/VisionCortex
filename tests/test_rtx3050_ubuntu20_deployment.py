from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visioncortex import archive, liquid_semantic, temporal_segmentation
from visioncortex.config import load_config
from visioncortex import config as config_module
from visioncortex.detection import (
    FramePacket,
    RoleScanner,
    _engine_requires_exact_batch,
)
from visioncortex.schemas import ViewInput, ViewRole
from test_rtx3090ti_ubuntu_deployment import relocated_web_host as relocated_web_host


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
    assert config["mllm"]["enabled"] is False
    assert config["project"]["site_configuration_required"] is True
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
    assert config["storage"]["archive_root"] == (
        "./outputs/development-runtime/archives"
    )
    assert "FlowTest" not in config["storage"]["archive_root"]
    assert config["archive"]["create_side_by_side_video"] is True
    certification_path = config["validation"]["model_certification"]["path"]
    assert certification_path == (
        "./Runtime/Model-Quality/"
        "production_model_certification.json"
    )
    assert "3090" not in certification_path
    assert config["validation"]["model_certification"].get(
        "required_for_formal_production"
    ) is False


def test_rtx3050_local_profile_cannot_inherit_nas_paths():
    config = load_config(ROOT / "configs/rtx3050-6gb-ubuntu20-local.yaml")

    assert config["mllm"]["enabled"] is False
    assert config["storage"]["sync_to_nas"] is False
    assert config["storage"]["run_output_mode"] == "local"
    for key in ("archive_root", "local_input_root", "local_runtime_root",
                "local_cache_root", "local_staging_root"):
        assert str(config["storage"][key]).startswith("./outputs/rtx3050-6gb-ubuntu20-runtime")
    assert config["storage"]["device_registry_path"] is None


def test_installed_runtime_can_resolve_sibling_default_config(tmp_path, monkeypatch):
    config_root = tmp_path / "app/configs"
    config_root.mkdir(parents=True)
    default = config_root / "default.yaml"
    profile = config_root / "target.yaml"
    default.write_text(
        "performance:\n  batch_size: 1\nmllm:\n  max_images_per_event: 12\n",
        encoding="utf-8",
    )
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
    scanner._prepared = True
    scanner._closed = False
    scanner.prediction_end2end = None
    scanner.nms_timeout_retries = 0
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
    builder = (ROOT / "tools/packaging/build_rtx3050_offline_package.py").read_text(
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


@pytest.fixture
def relocated_3050_host(relocated_web_host):
    host = relocated_web_host
    deployment = host["project"] / "deployment/rtx3050-ubuntu20"
    deployment.mkdir()
    for source in (ROOT / "deployment/rtx3050-ubuntu20").iterdir():
        if source.suffix in {".sh", ".service"}:
            shutil.copy2(source, deployment / source.name)
    host["deployment3050"] = deployment
    host["unit3050"] = host["home"] / ".config/systemd/user/visioncortex-rtx3050.service"
    return host


def test_rtx3050_local_preflight_does_not_probe_gpu_or_nas(relocated_3050_host):
    host = relocated_3050_host
    result = subprocess.run(["bash", str(host["deployment3050"] / "00-Preflight.sh")],
                            env=host["env"], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "local configuration only" in result.stdout
    assert not host["log"].exists()


@pytest.mark.parametrize("entry", ["Start-VisionCortex.sh", "00-Preflight.sh", "Install-Autostart.sh"])
@pytest.mark.parametrize("prepared_path", [False, True])
def test_rtx3050_unprepared_production_stops_before_probes(relocated_3050_host, entry, prepared_path):
    host = relocated_3050_host
    env = dict(host["env"])
    if prepared_path:
        env["VISIONCORTEX_CONFIG"] = str(ROOT / "configs/rtx3050-6gb-ubuntu20-production.yaml")
    else:
        env.pop("VISIONCORTEX_CONFIG")
    result = subprocess.run(["bash", str(host["deployment3050"] / entry), "--production"],
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "config" in result.stderr.lower()
    assert not host["log"].exists()
    assert not host["runtime"].exists()


def test_rtx3050_service_is_portable_and_does_not_replace_another_instance(relocated_3050_host):
    host = relocated_3050_host
    result = subprocess.run(["bash", str(host["deployment3050"] / "Install-Autostart.sh")],
                            env=host["env"], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    unit = host["unit3050"].read_text()
    assert "@RUNNER@" not in unit and "@PROJECT_ROOT@" not in unit
    assert "rtx3050-ubuntu20/Start-VisionCortex.sh" in unit
    assert "VISIONCORTEX_DEPLOYMENT_MODE=local" in unit
    assert "VISIONCORTEX_CONFIG=" in unit
    assert "TimeoutStopSec=infinity" in unit and "SendSIGKILL=no" in unit
    assert "injected" not in {path.name for path in host["project"].iterdir()}
    log = host["log"].read_text()
    assert "FORBIDDEN" not in log
    assert "visioncortex-analysis.service" not in log
    assert "visioncortex-local.service" not in log
    assert not (host["home"] / ".config/autostart").exists()
    desktop = (host["home"] / ".local/share/applications/visioncortex-rtx3050.desktop").read_text()
    assert "rtx3050-ubuntu20/Open-VisionCortex.sh" in desktop
    assert "VISIONCORTEX_DEPLOYMENT_MODE=local" in desktop


def test_rtx3050_existing_service_of_another_checkout_is_preserved(relocated_3050_host):
    host = relocated_3050_host
    unit = host["unit3050"]
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\nWorkingDirectory=/other/customer/checkout\n")
    before = unit.read_bytes()
    result = subprocess.run(["bash", str(host["deployment3050"] / "Install-Autostart.sh")],
                            env=host["env"], text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "another checkout" in result.stderr
    assert unit.read_bytes() == before
    assert "enable --now" not in host["log"].read_text()


def test_rtx3050_installer_refuses_existing_target_before_any_action(tmp_path):
    root = tmp_path / "existing installation"
    root.mkdir()
    marker = root / "identity"
    marker.write_bytes(b"existing prepared identity")
    env = {key: value for key, value in os.environ.items() if not key.startswith("VISIONCORTEX_")}
    result = subprocess.run(["bash", str(ROOT / "deployment/rtx3050-ubuntu20/Install-VisionCortex.sh"),
                             "--install-root", str(root)], env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert marker.read_bytes() == b"existing prepared identity"


def test_rtx3050_model_preparation_requires_explicit_private_site(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("VISIONCORTEX_")}
    target = tmp_path / "new installation"
    result = subprocess.run(["bash", str(ROOT / "deployment/rtx3050-ubuntu20/Install-VisionCortex.sh"),
                             "--install-root", str(target), "--prepare-models"],
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "requires --config" in result.stderr
    assert not target.exists()


def test_badcase_audit_requires_explicit_models_before_output_writes(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools/evaluation/yolo_badcase_audit.py"),
                             "--archive-root", str(tmp_path), "--output-root", str(tmp_path / "output"),
                             "--first-person-model", str(tmp_path / "missing-first.pt"),
                             "--third-person-model", str(tmp_path / "missing-third.pt")],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "existing explicitly selected file" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("installed", [False, True])
def test_rtx3050_local_start_uses_effective_storage_and_active_python(relocated_3050_host, installed):
    import shlex

    host = relocated_3050_host
    wrapper = host["project"].parent / "selected python"
    args_log = host["project"].parent / "serve-arguments.json"
    wrapper.write_text("#!/usr/bin/env bash\n"
                       'if [[ $1 == -m ]]; then printf "%s\n" "$@" > "$TEST_SERVE_LOG"; exit 0; fi\n'
                       + "exec " + shlex.quote(sys.executable) + ' "$@"\n')
    wrapper.chmod(0o755)
    env = host["env"] | {"VISIONCORTEX_PYTHON": str(wrapper), "TEST_SERVE_LOG": str(args_log)}
    entry = host["deployment3050"] / "Start-VisionCortex.sh"
    if installed:
        install = host["project"].parent / "portable install"
        install.mkdir()
        (install / "app").symlink_to(host["project"], target_is_directory=True)
        entry = install / "Start-VisionCortex.sh"
        shutil.copy2(host["deployment3050"] / entry.name, entry)
    result = subprocess.run(["bash", str(entry)], env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert args_log.read_text().splitlines() == ["-m", "visioncortex", "serve", "--host", "127.0.0.1",
                                               "--port", host["port"], "--config", str(host["config"])]
    assert (host["runtime"] / "archives").is_dir()
    assert (host["runtime"] / "tmp").is_dir()
    assert not host["log"].exists()



def test_rtx3050_custom_install_service_uses_its_own_environment(relocated_3050_host):
    host = relocated_3050_host
    install = host["project"].parent / "custom installed instance"
    app = install / "app"
    shutil.copytree(host["project"], app, symlinks=True)
    (install / ".venv/bin").mkdir(parents=True)
    (install / ".venv/bin/python").symlink_to(sys.executable)
    (install / ".package-id").write_text("test-only package identity")
    env = {key: value for key, value in host["env"].items()
           if key not in {"VISIONCORTEX_PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX"}}
    env["TEST_ENVIRONMENT_ADAPTER"] = str(app / "deployment/rtx3050-ubuntu20/_environment.sh")
    result = subprocess.run(["bash", "-c",
                             'script_dir=$(dirname -- "$TEST_ENVIRONMENT_ADAPTER"); '
                             'source "$TEST_ENVIRONMENT_ADAPTER"; printf "%s\n" "$install_root" "$app_root" "$python"'],
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(install), str(app), str(install / ".venv/bin/python")]



def test_desktop_launch_forwards_selected_files_without_key_values(relocated_3050_host):
    host = relocated_3050_host
    target = host["project"].parent / "selected desktop.desktop"
    key_file = host["project"].parent / "selected-provider-key"
    env = host["env"] | {"VISIONCORTEX_MODEL_API_KEY_FILE": str(key_file),
                         "VISIONCORTEX_DEFAULT_CONFIG": str(host["project"] / "configs/default.yaml"),
                         "VISIONCORTEX_MODEL_API_KEY": "synthetic-private-marker"}
    result = subprocess.run([sys.executable, str(ROOT / "deployment/rtx3090ti-ubuntu/render_service.py"),
                             "desktop", str(host["project"]), str(host["config"]),
                             "visioncortex-test.service", host["port"], str(target)],
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    desktop = target.read_text()
    assert "VISIONCORTEX_MODEL_API_KEY_FILE=" in desktop
    assert "selected-provider-key" in desktop
    assert "VISIONCORTEX_DEFAULT_CONFIG=" in desktop
    assert "synthetic-private-marker" not in desktop
