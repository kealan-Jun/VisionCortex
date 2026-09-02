from __future__ import annotations

import base64
import re
from functools import lru_cache
from pathlib import Path


BRAND_COLORS = {
    "brand": "#1e4a52",
    "brand_hover": "#153b42",
    "brand_secondary": "#2d7c82",
    "brand_soft": "#dbecee",
    "accent": "#cf8337",
    "background": "#eaf2f3",
    "surface": "#fbfdfd",
    "text": "#112d32",
    "text_muted": "#526d73",
    "border": "#cbdadc",
    "success": "#208a68",
    "warning": "#b66a24",
    "danger": "#b84d45",
}

_LOGO_SCRIPT = Path(__file__).with_name("web") / "brand-logo.js"
_LOGO_PATTERN = re.compile(
    r'VisionCortexBrandLogoDataUrl\s*=\s*"(data:image/png;base64,[^"]+)"'
)


@lru_cache(maxsize=1)
def brand_logo_data_url() -> str:
    """Return the approved product mark already bundled with the Web client."""

    source = _LOGO_SCRIPT.read_text(encoding="utf-8")
    match = _LOGO_PATTERN.search(source)
    if match is None:
        raise RuntimeError(f"VisionCortex brand logo is missing from {_LOGO_SCRIPT}")
    return match.group(1)


@lru_cache(maxsize=1)
def brand_logo_png() -> bytes:
    _, encoded = brand_logo_data_url().split(",", 1)
    return base64.b64decode(encoded)
