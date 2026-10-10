"""Private local credential revisions; keys never enter job configs or receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from .mllm_provider import connection_identity


def settings_directory() -> Path:
    return Path(
        os.getenv("VISIONCORTEX_AI_SETTINGS_DIR")
        or Path.home() / ".config/VisionCortex/ai-services"
    )


def private_directory(path: Path, *, create: bool = False) -> None:
    if os.name != "posix":
        raise RuntimeError("本机密钥存储需要 Linux 用户权限保护。")
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise RuntimeError("AI 配置目录必须由当前用户拥有，且权限为 700。")


def read_private(path: Path) -> dict[str, Any]:
    private_directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 1024 * 1024
        ):
            raise RuntimeError("AI 配置文件的类型、所有者或权限无效。")
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("AI 配置格式无效。")
        return value
    finally:
        os.close(fd)


def write_private(path: Path, value: dict) -> None:
    private_directory(path.parent, create=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_revision(reference: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", reference):
        raise RuntimeError("AI 凭据版本标识无效。")
    root = settings_directory()
    private_directory(root)
    return read_private(root / "credentials" / f"{reference}.json")


def model_api_key(settings: dict) -> str | None:
    reference = settings.get("credential_ref")
    if not reference:
        return os.getenv(str(settings.get("api_key_env") or "ARK_API_KEY"))
    record = read_revision(reference)
    from .provider_connection import verification_matches

    if connection_identity(record["connection"]) != connection_identity(
        settings
    ) or not verification_matches(settings, record["verification"]):
        raise RuntimeError(
            "任务的厂商配置与已验证凭据不一致，请在 AI 服务设置中重新验证。"
        )
    key = record.get("api_key")
    if not isinstance(key, str) or not key or any(ord(c) < 32 for c in key):
        raise RuntimeError("保存的 AI 密钥格式无效，请重新输入。")
    return key


def key_configured(settings: dict) -> bool:
    try:
        return bool(model_api_key(settings))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return False
