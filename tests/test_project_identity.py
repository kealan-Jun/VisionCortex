from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from visioncortex.api import app
from visioncortex.identity import CONFIG_ENV, PRODUCT_NAME


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_public_project_identity_is_visioncortex_only() -> None:
    metadata = tomllib.loads(_read("pyproject.toml"))["project"]

    assert PRODUCT_NAME == "VisionCortex"
    assert CONFIG_ENV == "VISIONCORTEX_CONFIG"
    assert metadata["name"] == "visioncortex"
    assert set(metadata["scripts"]) == {"visioncortex"}
    assert app.title == PRODUCT_NAME


def test_user_facing_surfaces_do_not_reintroduce_old_project_names() -> None:
    surfaces = (
        "README.md",
        "pyproject.toml",
        "src/visioncortex/web/index.html",
        "src/visioncortex/archive.py",
        "deployment/rtx3090ti-ubuntu/README.md",
        "docs/daily-report-template-v1.md",
    )
    old_names = ("Lab" + "Vision", "Lab" + "Embodied", "现实" + "回环")

    for relative in surfaces:
        content = _read(relative)
        assert (
            PRODUCT_NAME.casefold() in content.casefold() or "PRODUCT_NAME" in content
        ), relative
        for old_name in old_names:
            assert old_name not in content, f"{relative}: {old_name}"
        assert re.search(r"(?<![\w-])labvision\s", content, re.IGNORECASE) is None


def test_active_launchers_use_the_canonical_config_environment_name() -> None:
    launchers = (
        "Start-VisionCortex.ps1",
        "start-visioncortex.sh",
        "deployment/rtx3090ti-ubuntu/01-Install-And-Validate.sh",
        "deployment/rtx3090ti-ubuntu/02-Start-Web.sh",
        "deployment/rtx3090ti-ubuntu/06-Run-LAN-Server.sh",
        "deployment/rtx3090ti-ubuntu/visioncortex-lan.service",
        "deployment/rtx3090ti-ubuntu/visioncortex-local.service",
        "deployment/rtx4060/02-启动Web.ps1",
        "deployment/rtx4090/02-Start-Web.ps1",
    )
    old_config_env = "LAB" + "VISION_CONFIG"

    for relative in launchers:
        content = _read(relative)
        assert CONFIG_ENV in content, relative
        assert old_config_env not in content, relative


def test_model_data_and_report_contracts_use_visioncortex_identity() -> None:
    model_registry = json.loads(_read("configs/models/closed-set-yolo.json"))
    dataset_registry = json.loads(_read("configs/public-data-sources.json"))
    daily_report = json.loads(
        _read("src/visioncortex/templates/VC-LAB-DAILY-REPORT-V2.json")
    )
    evidence_report = json.loads(
        _read("src/visioncortex/templates/VC-PROFESSIONAL-EVIDENCE-REPORT-V1.json")
    )

    assert model_registry["schema_version"].startswith("visioncortex-")
    assert dataset_registry["schema_version"].startswith("visioncortex-")
    assert daily_report["title"].startswith(PRODUCT_NAME)
    assert daily_report["page"]["theme"] == PRODUCT_NAME
    assert evidence_report["title"].startswith(PRODUCT_NAME)
