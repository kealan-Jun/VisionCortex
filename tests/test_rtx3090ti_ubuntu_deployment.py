from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from visioncortex.api import _folder_open_command


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "deployment" / "rtx3090ti-ubuntu"


def test_ubuntu_shell_scripts_are_syntactically_valid_and_gpu_scoped():
    if os.name == "nt":
        pytest.skip("Ubuntu shell syntax is validated by the Ubuntu CI job")
    for name in (
        "00-Preflight.sh",
        "01-Install-And-Validate.sh",
        "02-Start-Web.sh",
        "03-Stop-Web.sh",
        "05-Install-Local-Service.sh",
        "06-Run-LAN-Server.sh",
        "07-Install-LAN-Server.sh",
        "08-Server-Status.sh",
    ):
        path = DEPLOYMENT / name
        subprocess.run(["bash", "-n", str(path)], check=True)

    preflight = (DEPLOYMENT / "00-Preflight.sh").read_text(encoding="utf-8")
    installer = (DEPLOYMENT / "01-Install-And-Validate.sh").read_text(encoding="utf-8")
    assert "RTX 3090 Ti" in preflight
    assert "--production" in preflight
    assert "minimum_free_gib=30" in preflight
    assert "nas_free_gib" in preflight
    assert "Python 3.11 or 3.12 required" in preflight
    assert "prepare-engine" in installer
    assert "prepare-public-models" in installer
    assert 'export PATH="$venv/bin:$PATH"' in installer
    assert "/home/x1/anaconda3/envs/gaoqing/bin/python" in installer
    assert "VisionCortexExperimentCache" in installer
    assert "nas_cache_missing" in preflight
    assert 'engine_root=${VISIONCORTEX_ENGINE_ROOT:-"$runtime_base/Engines"}' in installer
    assert "sha256sum" in installer
    assert "ark-" not in installer.lower()


def test_ubuntu_dependency_set_is_linux_pinned():
    requirements = (DEPLOYMENT / "requirements-lock.txt").read_text(encoding="utf-8")

    assert "torch==2.6.0" in requirements
    assert "torchvision==0.21.0" in requirements
    assert "https://mirrors.aliyun.com/pypi/simple/" in requirements
    assert "tensorrt-cu12==10.16.1.11" in requirements
    assert "onnx==1.22.0" in requirements
    assert "onnxruntime-gpu==1.29.0" in requirements
    assert "onnxslim==0.1.96" in requirements
    assert "ml-dtypes==0.5.4" in requirements
    assert "transformers==4.57.6" in requirements
    assert "scipy==1.17.1" in requirements
    assert "hydra-core==1.3.2" in requirements
    assert "iopath==0.1.10" in requirements
    assert "ultralytics/CLIP.git@68dce32140994dfcb645a1320c4ebdc034fc19fd" in requirements
    assert "https://pypi.nvidia.com/" in requirements
    assert "win_amd64" not in requirements

    installer = (DEPLOYMENT / "01-Install-And-Validate.sh").read_text(
        encoding="utf-8"
    )
    assert "SAM2_BUILD_CUDA=0" in installer
    assert "--no-build-isolation --no-deps" in installer
    assert "2b90b9f5ceec907a1c18123530e92e794ad901a4" in installer


def test_web_lifecycle_is_local_and_pid_scoped():
    start = (DEPLOYMENT / "02-Start-Web.sh").read_text(encoding="utf-8")
    stop = (DEPLOYMENT / "03-Stop-Web.sh").read_text(encoding="utf-8")

    assert "--host 127.0.0.1" in start
    assert "rtx3090ti-ubuntu-production.yaml" in start
    assert "VISIONCORTEX_ARK_API_KEY_FILE" in start
    assert "/home/x1/.config/VisionCortex/ark_api_key" in start
    assert "permissions must be 600" in start
    assert "/proc/$recorded_pid/cmdline" in start
    assert "/proc/$pid/cmdline" in stop
    assert "kill -9" not in stop


def test_local_systemd_service_is_reboot_resilient_and_nas_independent():
    service = (DEPLOYMENT / "visioncortex-local.service").read_text(
        encoding="utf-8"
    )
    installer = (DEPLOYMENT / "05-Install-Local-Service.sh").read_text(
        encoding="utf-8"
    )

    assert "Restart=on-failure" in service
    assert "WantedBy=default.target" in service
    assert "rtx3090ti-ubuntu-local.yaml" in service
    assert "/srv/sentinel-data/VisionCortex3090Ti/Runtime/NoNasWeb" in service
    assert "/home/x1/桌面/nas" not in service
    assert "systemctl --user enable --now visioncortex-local.service" in installer


def test_lan_systemd_service_is_authenticated_private_and_production_scoped():
    service = (DEPLOYMENT / "visioncortex-lan.service").read_text(encoding="utf-8")
    runner = (DEPLOYMENT / "06-Run-LAN-Server.sh").read_text(encoding="utf-8")
    installer = (DEPLOYMENT / "07-Install-LAN-Server.sh").read_text(
        encoding="utf-8"
    )
    status = (DEPLOYMENT / "08-Server-Status.sh").read_text(encoding="utf-8")

    assert "rtx3090ti-ubuntu-production.yaml" in service
    assert "VISIONCORTEX_WEB_ACCESS_MODE=lan" in service
    assert "VISIONCORTEX_WEB_PASSWORD_FILE=" in service
    assert "VISIONCORTEX_WEB_ALLOWED_NETWORKS=" in service
    assert "VISIONCORTEX_NAS_ARCHIVE_ROOT=/home/x1/桌面/nas/" in service
    assert "ExecStartPre=" in service and "--production" in service
    assert "Restart=on-failure" in service
    assert "--host 0.0.0.0" in runner
    assert "ARK_API_KEY=$(<\"$ark_key_file\")" in runner
    assert "permissions must be 600" in runner
    assert "read -r -s" in installer
    assert "systemctl --user enable --now visioncortex-lan.service" in installer
    assert "sudo loginctl enable-linger" in installer
    assert "api/health" in status
    assert 'cat "$password_file"' not in status
    assert "url=http://%s:8000/#/home" in status


def test_linux_archive_folder_uses_xdg_open(monkeypatch, tmp_path):
    monkeypatch.setattr("visioncortex.api.shutil.which", lambda name: "/usr/bin/xdg-open")

    command = _folder_open_command(tmp_path, os_name="posix", platform="linux")

    assert command == ["/usr/bin/xdg-open", str(tmp_path)]
