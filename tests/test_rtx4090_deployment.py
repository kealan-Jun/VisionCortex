from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "deployment" / "rtx4090"


def test_rtx4090_installer_is_offline_integrity_checked_and_target_built():
    source = (DEPLOYMENT / "01-Install-And-Validate.ps1").read_text(encoding="utf-8")

    assert "Verify-Package.ps1" in source
    assert "--no-index" in source
    assert "vendor\\wheelhouse" in source
    assert "python-3.12.10-amd64.exe" in source
    assert "prepare-engine" in source
    assert "tensorrt" in source
    assert "RTX 4090" in source
    assert "[Parameter(Mandatory = $true)][string]$ConfigPath" in source
    assert "Get-VisionCortexRuntimeConfiguration" in source
    assert "Read-Host" not in source
    assert "ark-" not in source


def test_rtx4090_web_start_does_not_hard_code_camera_count():
    source = (DEPLOYMENT / "02-Start-Web.ps1").read_text(encoding="utf-8")

    assert "VISIONCORTEX_SOURCE_WORKERS" not in source
    assert "cuda,cuda,cuda,cuda" not in source
    assert "02-启动Web.ps1" in source
    assert "@PSBoundParameters" in source
    assert "[Parameter(Mandatory = $true)][string]$ConfigPath" in source


def test_rtx4090_package_verifier_constrains_manifest_paths():
    source = (DEPLOYMENT / "Verify-Package.ps1").read_text(encoding="utf-8")

    assert "IsPathRooted" in source
    assert "escapes the package root" in source
    assert "Get-FileHash" in source
