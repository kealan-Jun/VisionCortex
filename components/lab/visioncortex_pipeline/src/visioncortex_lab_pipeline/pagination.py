from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


_CURSOR_VERSION = 1
_CURSOR_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,4096}\Z")


def filter_fingerprint(filters: Mapping[str, Any]) -> str:
    """Bind an opaque cursor to the exact query that produced it."""

    canonical = json.dumps(
        dict(filters),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def encode_cursor(
    *,
    namespace: str,
    position: Mapping[str, str | int],
    filters: Mapping[str, Any],
) -> str:
    _validate_namespace(namespace)
    normalized_position = _validate_position(position)
    raw = json.dumps(
        {
            "v": _CURSOR_VERSION,
            "namespace": namespace,
            "filters_sha256": filter_fingerprint(filters),
            "position": normalized_position,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(
    value: str,
    *,
    namespace: str,
    filters: Mapping[str, Any],
) -> dict[str, str | int]:
    _validate_namespace(namespace)
    if type(value) is not str or _CURSOR_PATTERN.fullmatch(value) is None:
        raise ValueError("cursor encoding is invalid")
    if len(value) % 4 == 1:
        raise ValueError("cursor encoding is invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
        if len(raw) > 3072:
            raise ValueError("cursor payload is too large")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cursor payload is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "v",
        "namespace",
        "filters_sha256",
        "position",
    }:
        raise ValueError("cursor payload is invalid")
    if type(payload["v"]) is not int or payload["v"] != _CURSOR_VERSION:
        raise ValueError("cursor version is unsupported")
    if payload["namespace"] != namespace:
        raise ValueError("cursor belongs to another result set")
    if payload["filters_sha256"] != filter_fingerprint(filters):
        raise ValueError("cursor does not match the current filters")
    if not isinstance(payload["position"], dict):
        raise ValueError("cursor position is invalid")
    return _validate_position(payload["position"])


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("cursor payload contains duplicate fields")
        result[key] = value
    return result


def _validate_namespace(value: object) -> None:
    if type(value) is not str or not value or len(value) > 64:
        raise ValueError("cursor namespace is invalid")
    if not re.fullmatch(r"[a-z0-9_.-]+", value):
        raise ValueError("cursor namespace is invalid")


def _validate_position(value: Mapping[str, object]) -> dict[str, str | int]:
    if not value or len(value) > 8:
        raise ValueError("cursor position is invalid")
    normalized: dict[str, str | int] = {}
    for key, item in value.items():
        if type(key) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
            raise ValueError("cursor position field is invalid")
        if type(item) is int:
            if item < 0:
                raise ValueError("cursor position integer is invalid")
            normalized[key] = item
        elif type(item) is str and len(item) <= 512:
            normalized[key] = item
        else:
            raise ValueError("cursor position value is invalid")
    return normalized


__all__ = ["decode_cursor", "encode_cursor", "filter_fingerprint"]
