from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools import doctor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_parse_nvidia_smi_and_choose_memory_profiles():
    output = "NVIDIA GeForce RTX 4060, 8188, 591.44\nNVIDIA RTX 3090 Ti, 24564, 591.44"

    assert doctor.parse_nvidia_smi(output) == [
        {"name": "NVIDIA GeForce RTX 4060", "memory_mb": 8188, "driver_version": "591.44"},
        {"name": "NVIDIA RTX 3090 Ti", "memory_mb": 24564, "driver_version": "591.44"},
    ]
    assert doctor.gpu_memory_profile(6144) == "gpu-6gb"
    assert doctor.gpu_memory_profile(8188) == "gpu-8gb"
    assert doctor.gpu_memory_profile(12288) == "gpu-12gb"
    assert doctor.gpu_memory_profile(24564) == "gpu-24gb"


def test_environment_doctor_does_not_probe_storage_by_default(monkeypatch):
    monkeypatch.setattr(doctor, "_command_output", lambda *_args, **_kwargs: None)

    report = doctor.inspect_environment(REPOSITORY_ROOT)

    assert report["storage"] == {"checked": False, "paths": {}}
    storage_check = next(item for item in report["checks"] if item["name"] == "NAS存储")
    assert storage_check["status"] == "skipped"
    assert report["evidence_boundary"]["model_inference"] == "NOT_PROVEN"
    assert report["evidence_boundary"]["real_video"] == "NOT_PROVEN"


def test_default_launcher_profile_is_local_only():
    profile = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "development-local.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert profile["storage"]["sync_to_nas"] is False
    assert profile["storage"]["web_upload_retention_mode"] == "local_only"
    assert profile["collection_ingest"]["enabled"] is False
    assert profile["mllm"]["enabled"] is False


def test_environment_doctor_json_cli_is_machine_readable():
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools" / "doctor.py"),
            "--project-root",
            str(REPOSITORY_ROOT),
            "--json",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == (0 if payload["launch_ready"] else 1)
    assert payload["schema_version"] == doctor.SCHEMA_VERSION
    assert payload["project_root"] == str(REPOSITORY_ROOT)
    assert payload["storage"]["checked"] is False


@pytest.mark.skipif(sys.platform == "win32", reason="Bash syntax check runs on Unix CI.")
def test_unix_launchers_have_valid_bash_syntax():
    for relative_path in ("start-visioncortex.sh", "Start-VisionCortex.command"):
        subprocess.run(
            ["bash", "-n", str(REPOSITORY_ROOT / relative_path)],
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell syntax check runs on Windows CI.")
def test_windows_launcher_has_valid_powershell_syntax():
    script = REPOSITORY_ROOT / "Start-VisionCortex.ps1"
    command = f"[void][scriptblock]::Create((Get-Content -LiteralPath '{script}' -Raw))"
    subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
