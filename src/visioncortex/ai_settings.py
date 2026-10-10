"""Opt-in workstation settings with real vision verification before activation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import hmac
import os
from pathlib import Path
import tempfile
import uuid

from .mllm_provider import connection_identity, normalize_connection
from .provider_connection import verification_matches, verify_connection
from .provider_discovery import DiscoveryError as DiscoveryError, discover_models
from .provider_credentials import (
    private_directory,
    read_private,
    read_revision,
    settings_directory,
    write_private,
)


def enabled() -> bool:
    return os.name == "posix" and os.getenv("VISIONCORTEX_WEB_AI_SETTINGS") == "1"


def catalog() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/mllm-providers.json"
        ).read_text()
    )


def read_index() -> dict:
    root = settings_directory()
    if not root.exists():
        return {"profiles": {}}
    private_directory(root)
    try:
        return read_private(root / "profiles.json")
    except FileNotFoundError:
        return {"profiles": {}}


def apply_active(settings: dict) -> dict:
    if not enabled():
        return settings
    reference = read_index().get("active_ref")
    if not reference:
        return settings
    record = read_revision(reference)
    result = deepcopy(settings)
    result["mllm"].update(record["connection"])
    result["mllm"].update(
        enabled=True,
        credential_ref=reference,
        api_key_env="VISIONCORTEX_MLLM_" + reference.upper(),
        connection_verification=record["verification"],
    )
    return result


def reverified_job_mllm(settings: dict) -> dict:
    """Reuse a newer verification only for the exact saved model and key.

    Updating the application invalidates the transport receipt. An explicit
    verification in the UI can refresh a stopped job without switching it to
    another provider, model, account or credential value.
    """
    mllm = settings["mllm"]
    reference = mllm.get("credential_ref")
    if not reference:
        return mllm
    original = read_revision(reference)
    if verification_matches(mllm, original["verification"]):
        return mllm
    newer_ref = read_index().get("profiles", {}).get(original["connection"]["provider"])
    newer = read_revision(newer_ref) if newer_ref else {}
    if not (
        newer
        and connection_identity(original["connection"]) == connection_identity(mllm)
        and connection_identity(newer["connection"]) == connection_identity(mllm)
        and hmac.compare_digest(original["api_key"].encode(), newer["api_key"].encode())
        and verification_matches(mllm, newer["verification"])
    ):
        raise ValueError("应用接口已更新，请先在 AI 服务设置中用原厂商、模型与已保存密钥重新验证，再复跑。")
    return {**mllm, "credential_ref": newer_ref,
            "api_key_env": "VISIONCORTEX_MLLM_" + newer_ref.upper(),
            "connection_verification": newer["verification"]}


def public_settings(settings: dict) -> dict:
    index = read_index()
    profiles = {}
    for provider, reference in index.get("profiles", {}).items():
        record = read_revision(reference)
        profiles[provider] = {
            "connection": record["connection"],
            "has_saved_key": True,
            "verification": record["verification"],
            "verified": verification_matches(
                record["connection"], record["verification"]
            ),
        }
    reference = index.get("active_ref")
    active = None
    if reference:
        record = read_revision(reference)
        active = {
            "connection": record["connection"],
            "verification": record["verification"],
            "verified": verification_matches(
                record["connection"], record["verification"]
            ),
        }
    return {
        "enabled": True,
        "providers": catalog(),
        "profiles": profiles,
        "active": active,
        "legacy": {
            "provider": settings["mllm"].get("provider", "volcengine"),
            "model": settings["mllm"].get("model"),
            "base_url": settings["mllm"].get("base_url"),
            "has_key": bool(
                os.getenv(str(settings["mllm"].get("api_key_env") or "ARK_API_KEY"))
            ),
        },
        "credential_storage": "private_local_user_directory_0700_files_0600",
        "applies_to": "new_tasks_only",
    }


@contextmanager
def settings_lock():
    import fcntl

    root = settings_directory()
    private_directory(root, create=True)
    fd = os.open(root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def selected_api_key(selected: dict, api_key: str, settings: dict, index: dict) -> str:
    if not isinstance(api_key, str) or len(api_key) > 4096 or any(ord(c) < 32 for c in api_key):
        raise ValueError("请输入有效的 API 密钥。")
    key = api_key.strip()
    reference = index.get("profiles", {}).get(selected["provider"])
    if not key and reference:
        saved = read_revision(reference)
        if all(
            saved["connection"][field] == selected[field]
            for field in ("provider", "base_url")
        ):
            key = saved["api_key"]
    legacy = settings["mllm"]
    if (
        not key
        and selected["provider"] == legacy.get("provider", "volcengine")
        and selected["base_url"] == str(legacy.get("base_url", "")).rstrip("/")
    ):
        key = os.getenv(str(legacy.get("api_key_env") or "ARK_API_KEY"), "")
    if not key:
        raise ValueError("请输入所选厂商的 API 密钥。")
    return key


def discover_available_models(connection: dict, api_key: str, settings: dict) -> dict:
    selected = normalize_connection({**connection, "model": "discovery-only"})
    key = selected_api_key(selected, api_key, settings, read_index())
    return discover_models(selected, key)


def verify_and_activate(connection: dict, api_key: str, settings: dict) -> dict:
    selected = normalize_connection(connection)
    with settings_lock():
        index = read_index()
        key = selected_api_key(selected, api_key, settings, index)
        # Test media stay in the local private directory, never in NAS or archives.
        with tempfile.TemporaryDirectory(
            prefix=".probe-", dir=settings_directory()
        ) as work:
            receipt = verify_connection(selected, key, Path(work))
        if not verification_matches(selected, receipt):
            return {"activated": False, "verification": receipt}
        reference = uuid.uuid4().hex
        root = settings_directory()
        write_private(
            root / "credentials" / f"{reference}.json",
            {"connection": selected, "api_key": key, "verification": receipt},
        )
        # Immutable credential revisions preserve queued/running tasks on rotation.
        index.setdefault("profiles", {})[selected["provider"]] = reference
        index["active_ref"] = reference
        write_private(root / "profiles.json", index)
        return {
            "activated": True,
            "verification": receipt,
            "applies_to": "new_tasks_only",
        }
