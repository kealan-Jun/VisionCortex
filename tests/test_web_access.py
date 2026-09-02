from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from labvision_evidence import api
from labvision_evidence.web_access import (
    is_allowed_lan_client,
    valid_basic_authorization,
    validate_web_access_configuration,
)


def _basic(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {encoded}"


def test_default_web_access_stays_local_and_open(monkeypatch):
    monkeypatch.delenv("VISIONCORTEX_WEB_ACCESS_MODE", raising=False)
    monkeypatch.delenv("VISIONCORTEX_WEB_PASSWORD", raising=False)

    assert validate_web_access_configuration() == {
        "mode": "local",
        "authentication_required": False,
    }


def test_lan_network_policy_accepts_only_configured_private_ranges(monkeypatch):
    monkeypatch.delenv("VISIONCORTEX_WEB_ALLOWED_NETWORKS", raising=False)

    assert is_allowed_lan_client("127.0.0.1") is True
    assert is_allowed_lan_client("192.168.66.25") is True
    assert is_allowed_lan_client("172.31.4.8") is True
    assert is_allowed_lan_client("10.20.30.40") is True
    assert is_allowed_lan_client("8.8.8.8") is False
    assert is_allowed_lan_client("testclient") is False


def test_basic_authorization_uses_configured_server_login(monkeypatch):
    monkeypatch.setenv("VISIONCORTEX_WEB_USERNAME", "visioncortex")
    monkeypatch.setenv("VISIONCORTEX_WEB_PASSWORD", "correct-horse-battery-staple")

    assert valid_basic_authorization(_basic("visioncortex", "correct-horse-battery-staple"))
    assert not valid_basic_authorization(_basic("visioncortex", "wrong-password"))
    assert not valid_basic_authorization("Bearer not-a-basic-login")
    assert not valid_basic_authorization("Basic invalid-base64")


def test_lan_middleware_requires_login_and_rejects_public_clients(monkeypatch):
    monkeypatch.setenv("VISIONCORTEX_WEB_ACCESS_MODE", "lan")
    monkeypatch.setenv("VISIONCORTEX_WEB_USERNAME", "visioncortex")
    monkeypatch.setenv("VISIONCORTEX_WEB_PASSWORD", "correct-horse-battery-staple")

    with TestClient(api.app, client=("192.168.66.25", 50000)) as client:
        unauthenticated = client.get("/")
        assert unauthenticated.status_code == 401
        assert unauthenticated.headers["www-authenticate"].startswith("Basic ")

        authenticated = client.get(
            "/",
            headers={
                "Authorization": _basic(
                    "visioncortex", "correct-horse-battery-staple"
                )
            },
        )
        assert authenticated.status_code == 200

    with TestClient(api.app, client=("8.8.8.8", 50000)) as client:
        rejected = client.get(
            "/",
            headers={
                "Authorization": _basic(
                    "visioncortex", "correct-horse-battery-staple"
                )
            },
        )
        assert rejected.status_code == 403
