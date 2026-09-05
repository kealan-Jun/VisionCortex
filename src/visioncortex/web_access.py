from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import stat
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_LAN_NETWORKS = (
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "::1/128",
    "fc00::/7",
)
_USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_ROLES = {"viewer", "operator", "admin"}
_PASSWORD_HASH_ITERATIONS = 600_000
_AUTH_CACHE_SECRET = secrets.token_bytes(32)
_AUTH_CACHE_LOCK = threading.Lock()
_AUTH_CACHE: dict[str, tuple[float, dict[str, str]]] = {}
_AUTH_CACHE_TTL_SECONDS = 15.0
_DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$600000$visioncortex-dummy-salt$"
    + "0" * 64
)


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


def _secure_file_text(path: Path, label: str) -> str:
    try:
        file_status = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"VisionCortex Web {label} file is unavailable") from exc
    if stat.S_ISLNK(file_status.st_mode) or not stat.S_ISREG(file_status.st_mode):
        raise RuntimeError(
            f"VisionCortex Web {label} file must be a regular non-symbolic-link file"
        )
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise RuntimeError(
            f"VisionCortex Web {label} file ownership checks require POSIX"
        )
    if file_status.st_uid != os.geteuid():
        raise RuntimeError(f"VisionCortex Web {label} file owner is invalid")
    if stat.S_IMODE(file_status.st_mode) != 0o600:
        raise RuntimeError(f"VisionCortex Web {label} file permissions must be 600")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"VisionCortex Web {label} file could not be read") from exc


def _password_from_file(path: Path) -> str:
    lines = _secure_file_text(path, "password").splitlines()
    if len(lines) != 1:
        raise RuntimeError("VisionCortex Web password file must contain exactly one line")
    return lines[0]


def hash_web_password(
    password: str,
    *,
    salt: str | None = None,
    iterations: int = _PASSWORD_HASH_ITERATIONS,
) -> str:
    if len(password) < 12 or "\r" in password or "\n" in password:
        raise ValueError("VisionCortex Web password must be a single line of at least 12 characters")
    selected_salt = salt or secrets.token_hex(16)
    if (
        not selected_salt
        or len(selected_salt) > 128
        or not selected_salt.isascii()
        or "$" in selected_salt
        or not 100_000 <= iterations <= 2_000_000
    ):
        raise ValueError("VisionCortex Web password hash parameters are invalid")
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), selected_salt.encode("ascii"), iterations
    ).hex()
    return f"pbkdf2_sha256${iterations}${selected_salt}${digest}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt, expected = encoded.split("$", 3)
        iterations = int(iterations_text)
    except (TypeError, ValueError):
        return False
    if (
        algorithm != "pbkdf2_sha256"
        or not 100_000 <= iterations <= 2_000_000
        or not salt
        or len(salt) > 128
        or not salt.isascii()
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None
    ):
        return False
    try:
        observed = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("ascii"), iterations
        ).hex()
    except (UnicodeEncodeError, OverflowError):
        return False
    return hmac.compare_digest(observed, expected)


@lru_cache(maxsize=8)
def _load_users_file_cached(
    path_text: str, size_bytes: int, mtime_ns: int
) -> tuple[dict[str, str], ...]:
    del size_bytes, mtime_ns
    payload = json.loads(_secure_file_text(Path(path_text), "users"))
    if not isinstance(payload, dict):
        raise RuntimeError("VisionCortex Web users file must contain a JSON object")
    if payload.get("schema_version") != "visioncortex-web-users/1":
        raise RuntimeError("VisionCortex Web users file schema_version is invalid")
    users = []
    seen = set()
    for item in payload.get("users") or []:
        if not isinstance(item, dict):
            raise RuntimeError("VisionCortex Web users file contains an invalid user")
        username = str(item.get("username") or "")
        role = str(item.get("role") or "viewer")
        password_hash = str(item.get("password_hash") or "")
        if (
            not _USERNAME_PATTERN.fullmatch(username)
            or username in seen
            or role not in _ROLES
            or not password_hash.startswith("pbkdf2_sha256$")
            or "password" in item
        ):
            raise RuntimeError("VisionCortex Web users file contains an invalid user")
        seen.add(username)
        users.append(
            {"username": username, "role": role, "password_hash": password_hash}
        )
    if not users:
        raise RuntimeError("VisionCortex Web users file must contain at least one user")
    return tuple(users)


def web_users() -> tuple[dict[str, str], ...] | None:
    configured = os.getenv("VISIONCORTEX_WEB_USERS_FILE", "").strip()
    if not configured:
        return None
    path = Path(configured).resolve()
    try:
        status = path.stat()
        return _load_users_file_cached(str(path), status.st_size, status.st_mtime_ns)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("VisionCortex Web users file is invalid") from exc


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
    users = web_users()
    username = None
    if users is None:
        username, _ = web_credentials()
    networks = _allowed_networks()
    result: dict[str, object] = {
        "mode": "lan",
        "authentication_required": True,
        "authentication_mode": "users_file" if users is not None else "single_account",
        "user_count": len(users) if users is not None else 1,
        "https_required": web_https_required(),
        "allowed_networks": [str(network) for network in networks],
    }
    if username is not None:
        result["username"] = username
    return result


def web_https_required() -> bool:
    return os.getenv("VISIONCORTEX_WEB_REQUIRE_HTTPS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _decode_basic_authorization(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    scheme, separator, encoded = value.partition(" ")
    if not separator or scheme.lower() != "basic" or not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    username, separator, password = decoded.partition(":")
    return (username, password) if separator else None


def _authentication_source_revision(
    users: tuple[dict[str, str], ...] | None,
    single_credentials: tuple[str, str] | None,
) -> str:
    source = users if users is not None else single_credentials
    serialized = json.dumps(source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hmac.new(_AUTH_CACHE_SECRET, serialized, hashlib.sha256).hexdigest()


def _authorization_cache_key(value: str | None, source_revision: str) -> str | None:
    if not value:
        return None
    message = f"{source_revision}\0{value}".encode("utf-8")
    return hmac.new(_AUTH_CACHE_SECRET, message, hashlib.sha256).hexdigest()


def authenticate_basic_authorization(value: str | None) -> dict[str, str] | None:
    credentials = _decode_basic_authorization(value)
    if credentials is None:
        return None
    provided_username, provided_password = credentials
    users = web_users()
    single_credentials = web_credentials() if users is None else None
    cache_key = _authorization_cache_key(
        value, _authentication_source_revision(users, single_credentials)
    )
    if cache_key:
        with _AUTH_CACHE_LOCK:
            cached = _AUTH_CACHE.get(cache_key)
            if cached and cached[0] >= time.monotonic():
                return dict(cached[1])
    if users is not None:
        matched_user: dict[str, str] | None = None
        for user in users:
            if hmac.compare_digest(
                provided_username, user["username"]
            ):
                matched_user = user
        password_matches = _verify_password(
            provided_password,
            matched_user["password_hash"] if matched_user else _DUMMY_PASSWORD_HASH,
        )
        if matched_user is None or not password_matches:
            return None
        identity = {
            "username": matched_user["username"],
            "role": matched_user["role"],
        }
    else:
        assert single_credentials is not None
        expected_username, expected_password = single_credentials
        if not (
            hmac.compare_digest(provided_username, expected_username)
            and hmac.compare_digest(provided_password, expected_password)
        ):
            return None
        identity = {"username": expected_username, "role": "admin"}
    if cache_key:
        with _AUTH_CACHE_LOCK:
            if len(_AUTH_CACHE) >= 256:
                expired = [
                    key for key, item in _AUTH_CACHE.items() if item[0] < time.monotonic()
                ]
                for key in expired or [next(iter(_AUTH_CACHE))]:
                    _AUTH_CACHE.pop(key, None)
            _AUTH_CACHE[cache_key] = (
                time.monotonic() + _AUTH_CACHE_TTL_SECONDS,
                dict(identity),
            )
    return identity


def valid_basic_authorization(value: str | None) -> bool:
    return authenticate_basic_authorization(value) is not None


def append_access_audit(path: Path, record: dict[str, Any]) -> None:
    """Append one credential-free access receipt to server-local storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
