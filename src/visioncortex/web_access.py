from __future__ import annotations

import base64
import binascii
import hmac
import ipaddress
import os
import re
import stat
from pathlib import Path


DEFAULT_LAN_NETWORKS = (
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "::1/128",
    "fc00::/7",
)
_USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def web_access_mode() -> str:
    mode = os.getenv("VISIONCORTEX_WEB_ACCESS_MODE", "local").strip().lower()
    if mode not in {"local", "lan"}:
        raise RuntimeError(
            "VISIONCORTEX_WEB_ACCESS_MODE must be either 'local' or 'lan'"
        )
    return mode


def _allowed_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    configured = os.getenv("VISIONCORTEX_WEB_ALLOWED_NETWORKS", "").strip()
    values = [item.strip() for item in configured.split(",") if item.strip()]
    if not values:
        values = list(DEFAULT_LAN_NETWORKS)
    try:
        return tuple(ipaddress.ip_network(item, strict=True) for item in values)
    except ValueError as exc:
        raise RuntimeError("VISIONCORTEX_WEB_ALLOWED_NETWORKS contains invalid CIDR") from exc


def is_allowed_lan_client(host: str | None) -> bool:
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return any(address in network for network in _allowed_networks())


def _password_from_file(path: Path) -> str:
    try:
        file_status = path.lstat()
    except OSError as exc:
        raise RuntimeError("VisionCortex Web password file is unavailable") from exc
    if stat.S_ISLNK(file_status.st_mode) or not stat.S_ISREG(file_status.st_mode):
        raise RuntimeError(
            "VisionCortex Web password file must be a regular non-symbolic-link file"
        )
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise RuntimeError(
            "VisionCortex Web password file ownership checks require POSIX; "
            "use VISIONCORTEX_WEB_PASSWORD on this platform"
        )
    if file_status.st_uid != os.geteuid():
        raise RuntimeError("VisionCortex Web password file owner is invalid")
    if stat.S_IMODE(file_status.st_mode) != 0o600:
        raise RuntimeError("VisionCortex Web password file permissions must be 600")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("VisionCortex Web password file could not be read") from exc
    if len(lines) != 1:
        raise RuntimeError("VisionCortex Web password file must contain exactly one line")
    return lines[0]


def web_credentials() -> tuple[str, str]:
    username = os.getenv("VISIONCORTEX_WEB_USERNAME", "visioncortex").strip()
    if not _USERNAME_PATTERN.fullmatch(username):
        raise RuntimeError("VISIONCORTEX_WEB_USERNAME has an invalid format")

    password = os.getenv("VISIONCORTEX_WEB_PASSWORD", "")
    if not password:
        configured_path = os.getenv("VISIONCORTEX_WEB_PASSWORD_FILE", "").strip()
        if not configured_path:
            raise RuntimeError("VisionCortex LAN Web password is not configured")
        password = _password_from_file(Path(configured_path))
    if len(password) < 12:
        raise RuntimeError("VisionCortex LAN Web password must contain at least 12 characters")
    if "\r" in password or "\n" in password:
        raise RuntimeError("VisionCortex LAN Web password must be a single line")
    return username, password


def validate_web_access_configuration() -> dict[str, object]:
    mode = web_access_mode()
    if mode == "local":
        return {"mode": "local", "authentication_required": False}
    username, _ = web_credentials()
    networks = _allowed_networks()
    return {
        "mode": "lan",
        "authentication_required": True,
        "username": username,
        "allowed_networks": [str(network) for network in networks],
    }


def valid_basic_authorization(value: str | None) -> bool:
    if not value:
        return False
    scheme, separator, encoded = value.partition(" ")
    if not separator or scheme.lower() != "basic" or not encoded:
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    provided_username, separator, provided_password = decoded.partition(":")
    if not separator:
        return False
    expected_username, expected_password = web_credentials()
    return hmac.compare_digest(provided_username, expected_username) and hmac.compare_digest(
        provided_password, expected_password
    )
