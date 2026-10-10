"""Keep model scratch paths bounded without changing evidence identities."""
from __future__ import annotations

import os
import re
from pathlib import Path


def windows_path(path: Path) -> bool:
    text = str(path)
    return os.name == "nt" or bool(re.match(r"^[A-Za-z]:[\\/]", text)) or text.startswith("\\\\")


def check_path_budget(path: Path) -> None:
    # Leave room for temporary suffixes, also for libraries without longPathAware.
    if windows_path(path) and len(str(path).encode("utf-16-le")) // 2 > 235:
        raise ValueError(
            "Windows 缓存路径过长：请在应用存储设置中选择更短的本地运行目录。"
            "已保存的实验成果不受影响。"
        )


def model_cache_directory(cache_root: Path, namespace: str, fingerprint: str) -> Path:
    length = 64 if namespace == "s2" else 20
    if namespace not in {"s2", "work"} or not re.fullmatch(rf"[a-f0-9]{{{length}}}", fingerprint):
        raise ValueError("Invalid model cache identity")
    # The complete digest includes the source, event, view and model identities.
    # Do not repeat long human-facing media names or truncate the digest.
    result = cache_root / namespace / fingerprint
    check_path_budget(result / "frames" / "00000.jpg.partial")
    return result
