"""Public onboarding must not depend on retired site screenshots."""
from pathlib import Path

from repo_paths import ROOT
from tools.packaging.documentation import DOCUMENTATION_ENTRYPOINTS, documentation_resources


GUIDES = (
    "docs/VisionCortex-普通用户10分钟操作指南.md",
    "docs/VisionCortex-端到端视频分析交付手册.md",
    "docs/VisionCortex-交付验收单模板.md",
)


def test_customer_guides_close_resources_without_site_screenshots(monkeypatch):
    monkeypatch.setitem(DOCUMENTATION_ENTRYPOINTS, "customer_guides", GUIDES)
    resources = documentation_resources(ROOT, "customer_guides")
    assert set(GUIDES).issubset(resources)
    assert not any(Path(name).as_posix().startswith("docs/assets/delivery-guide/") for name in resources)
    assert "docs/assets/visioncortex-banner.svg" in documentation_resources(ROOT, "portable_desktop")


def test_first_clone_instructions_distinguish_page_and_real_analysis():
    guide = (ROOT / GUIDES[0]).read_text(encoding="utf-8")
    assert "demo" in guide and "视频处理待配置" in guide
    assert "真实视频质量需要本次分析回执" in guide
