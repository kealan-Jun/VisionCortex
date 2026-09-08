from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from .provider_credentials import model_api_key


DEFAULT_ARK_API_KEY_FILE = Path("/home/x1/.config/VisionCortex/ark_api_key")


def ensure_ark_api_key(
    config: dict[str, Any],
    *,
    required: bool = True,
) -> dict[str, Any]:
    """Load the workstation Ark credential without exposing it in config/logs.

    Production callers use the returned metadata in preflight receipts.  It is
    intentionally limited to presence/source/path/validation facts and never
    contains the credential value.
    """

    mllm = config.get("mllm") or {}
    enabled = bool(mllm.get("enabled", False))
    environment_name = str(mllm.get("api_key_env") or "ARK_API_KEY")
    if mllm.get("credential_ref") or mllm.get("provider") in {"aliyun", "zhipu", "custom"}:
        key = model_api_key(mllm)
        if key and (len(key) > 4096 or any(ord(c) < 32 for c in key)):
            raise RuntimeError("Multimodal credential format is invalid")
        if not key and enabled and required:
            raise RuntimeError("请在 AI 服务设置中配置并验证所选厂商的密钥。")
        if key:
            os.environ[environment_name] = key
        return {"enabled": enabled, "configured": bool(key),
                "source": "private_credential_revision" if mllm.get("credential_ref") else "environment",
                "environment_name": environment_name, "format_validated": bool(key),
                "secret_recorded": False, "provider": mllm.get("provider")}
    existing = os.getenv(environment_name, "").strip()
    if existing:
        if not existing.startswith("ark-"):
            raise RuntimeError(
                f"{environment_name} has an unexpected format; Ark preflight refused"
            )
        return {
            "enabled": enabled,
            "configured": True,
            "source": "environment",
            "environment_name": environment_name,
            "format_validated": True,
            "secret_recorded": False,
        }

    if not enabled:
        return {
            "enabled": False,
            "configured": False,
            "source": "not_required",
            "environment_name": environment_name,
            "format_validated": False,
            "secret_recorded": False,
        }

    configured_path = os.getenv("VISIONCORTEX_ARK_API_KEY_FILE")
    credential_path = Path(configured_path) if configured_path else DEFAULT_ARK_API_KEY_FILE
    try:
        file_status = credential_path.lstat()
    except OSError as exc:
        if not required:
            return {
                "enabled": True,
                "configured": False,
                "source": "missing_file",
                "credential_file": str(credential_path),
                "environment_name": environment_name,
                "format_validated": False,
                "secret_recorded": False,
            }
        raise RuntimeError(
            f"Ark credential file is unavailable: {credential_path}"
        ) from exc

    if stat.S_ISLNK(file_status.st_mode) or not stat.S_ISREG(file_status.st_mode):
        raise RuntimeError("Ark credential file must be a regular non-symbolic-link file")
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise RuntimeError(
            "Ark credential file ownership and 0600 permissions require POSIX; "
            f"use {environment_name} on this platform"
        )
    if file_status.st_uid != os.geteuid():
        raise RuntimeError("Ark credential file owner is invalid")
    if stat.S_IMODE(file_status.st_mode) != 0o600:
        raise RuntimeError("Ark credential file permissions must be 600")

    try:
        lines = credential_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("Ark credential file could not be read") from exc
    if len(lines) != 1 or not lines[0].strip().startswith("ark-"):
        raise RuntimeError("Ark credential file has invalid contents or format")

    # Do not retain the value in config or return it to the caller.  ArkAnalyzer
    # reads it from the process environment and request receipts redact headers.
    os.environ[environment_name] = lines[0].strip()
    return {
        "enabled": True,
        "configured": True,
        "source": "credential_file",
        "credential_file": str(credential_path.resolve(strict=True)),
        "environment_name": environment_name,
        "owner_validated": True,
        "permissions_validated": "0600",
        "format_validated": True,
        "secret_recorded": False,
    }
