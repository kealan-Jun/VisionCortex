from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from visioncortex import api
from visioncortex.web_access import (
    authenticate_basic_authorization,
    hash_web_password,
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


def test_multi_user_password_hashes_return_identity_and_role(monkeypatch):
    from visioncortex import web_access

    encoded = hash_web_password(
        "correct-horse-battery-staple", salt="fixed-test-salt", iterations=100_000
    )
    monkeypatch.setattr(
        web_access,
        "web_users",
        lambda: (
            {
                "username": "researcher-01",
                "role": "viewer",
                "password_hash": encoded,
            },
        ),
    )

    assert authenticate_basic_authorization(
        _basic("researcher-01", "correct-horse-battery-staple")
    ) == {"username": "researcher-01", "role": "viewer"}
    assert authenticate_basic_authorization(
        _basic("researcher-01", "wrong-password")
    ) is None


def test_malformed_multi_user_password_hash_fails_closed(monkeypatch):
    from visioncortex import web_access

    monkeypatch.setattr(
        web_access,
        "web_users",
        lambda: (
            {
                "username": "researcher-01",
                "role": "viewer",
                "password_hash": "pbkdf2_sha256$100000$非法盐$" + "0" * 64,
            },
        ),
    )

    assert authenticate_basic_authorization(
        _basic("researcher-01", "correct-horse-battery-staple")
    ) is None


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


def test_lan_viewer_is_read_only_and_access_is_audited(monkeypatch):
    from visioncortex import web_access

    encoded = hash_web_password(
        "viewer-password-for-test", salt="viewer-test-salt", iterations=100_000
    )
    monkeypatch.setattr(
        web_access,
        "web_users",
        lambda: (
            {
                "username": "viewer-01",
                "role": "viewer",
                "password_hash": encoded,
            },
        ),
    )
    monkeypatch.setenv("VISIONCORTEX_WEB_ACCESS_MODE", "lan")
    monkeypatch.delenv("VISIONCORTEX_WEB_REQUIRE_HTTPS", raising=False)
    audit_records = []
    monkeypatch.setattr(api, "append_access_audit", lambda path, record: audit_records.append(record))

    with TestClient(api.app, client=("192.168.66.25", 50000)) as client:
        headers = {"Authorization": _basic("viewer-01", "viewer-password-for-test")}
        readable = client.get("/api/health", headers=headers)
        denied = client.post("/api/archives/anything/open", headers=headers)

    assert readable.status_code == 200
    assert denied.status_code == 403
    assert any(
        item["username"] == "viewer-01"
        and item["path"] == "/api/health"
        and item["status_code"] == 200
        for item in audit_records
    )
    assert any(item["status_code"] == 403 for item in audit_records)


def test_lan_https_requirement_fails_closed(monkeypatch):
    monkeypatch.setenv("VISIONCORTEX_WEB_ACCESS_MODE", "lan")
    monkeypatch.setenv("VISIONCORTEX_WEB_USERNAME", "visioncortex")
    monkeypatch.setenv("VISIONCORTEX_WEB_PASSWORD", "correct-horse-battery-staple")
    monkeypatch.setenv("VISIONCORTEX_WEB_REQUIRE_HTTPS", "true")

    with TestClient(api.app, client=("192.168.66.25", 50000)) as client:
        response = client.get(
            "/",
            headers={"Authorization": _basic("visioncortex", "correct-horse-battery-staple")},
        )

    assert response.status_code == 426
