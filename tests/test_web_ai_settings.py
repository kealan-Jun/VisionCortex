from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import stat

from fastapi.testclient import TestClient
import httpx
import pytest

from visioncortex import ai_settings, api
from visioncortex.config import load_config
from visioncortex.credentials import ensure_ark_api_key
from visioncortex.mllm import ArkAnalyzer
from visioncortex.mllm_provider import connection_identity, normalize_connection
from visioncortex.provider_connection import adapter_identity, verify_connection
from visioncortex.provider_credentials import (
    model_api_key,
    read_private,
    read_revision,
    settings_directory,
)


def selection(provider="aliyun", **changes):
    return normalize_connection(
        {"provider": provider, **ai_settings.catalog()[provider], **changes}
    )


@pytest.fixture
def local_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONCORTEX_WEB_AI_SETTINGS", "1")
    monkeypatch.setenv("VISIONCORTEX_AI_SETTINGS_DIR", str(tmp_path / "private-ai"))
    monkeypatch.setenv("VISIONCORTEX_WEB_ACCESS_MODE", "local")
    monkeypatch.setenv("ARK_API_KEY", "ark-local-test-key")
    config = load_config(Path("configs/development-local.yaml"))
    for key in (
        "archive_root",
        "local_runtime_root",
        "local_input_root",
        "local_cache_root",
        "local_staging_root",
    ):
        config["storage"][key] = str(tmp_path / key)
    config["mllm"]["enabled"] = True
    monkeypatch.setattr(api, "load_config", lambda *_: deepcopy(config))
    return config


def verified(selected):
    return {
        "status": "verified",
        "model_invocation": "PROVEN",
        "connection": selected,
        "connection_sha256": connection_identity(selected),
        "adapter_sha256": adapter_identity(),
        "checked_at": "2026-09-07T00:00:00Z",
        "request_id": "synthetic-request",
        "response_model": selected["model"],
        "usage": {"total_tokens": 80},
        "real_video_quality": "NOT_PROVEN",
    }


def client():
    return TestClient(
        api.app, base_url="http://127.0.0.1:8001", client=("127.0.0.1", 50000)
    )


def submit(client, selected, key):
    return client.post(
        "/api/ai-settings/verify",
        headers={"Origin": "http://127.0.0.1:8001"},
        json={"connection": selected, "api_key": key},
    )


def test_local_web_page_catalog_never_returns_legacy_key(local_settings):
    response = client().get("/api/ai-settings")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert set(response.json()["providers"]) == {
        "volcengine",
        "aliyun",
        "zhipu",
        "custom",
        "google",
        "openrouter",
        "siliconflow",
    }
    assert "ark-local-test-key" not in response.text
    assert not settings_directory().exists()


def discovery_transport(monkeypatch, handler):
    from visioncortex import provider_discovery

    original = httpx.Client

    def factory(**kwargs):
        assert kwargs == {"timeout": 20, "follow_redirects": False, "trust_env": False}
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(provider_discovery.httpx, "Client", factory)


def discover(c, selected, key="synthetic-discovery-secret"):
    return c.post("/api/ai-settings/models", headers={"Origin": "http://127.0.0.1:8001"},
                  json={"connection": selected, "api_key": key})


def test_discovery_keeps_unknown_capabilities_without_activating_or_echoing_key(local_settings, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == "https://unlisted.example/v1/models"
        assert request.headers["Authorization"] == "Bearer synthetic-discovery-secret"
        return httpx.Response(200, json={"data": [
            {"id": "text-only", "architecture": {"input_modalities": ["text"]}},
            {"id": "unknown"},
            {"id": "vision", "architecture": {"input_modalities": ["image", "text"]}},
            {"id": "synthetic-discovery-secret"},
            {"id": "unknown", "name": "synthetic-discovery-secret"},
            {"id": "invalid\nmodel"}, None,
        ]})

    discovery_transport(monkeypatch, handler)
    response = discover(client(), {"provider": "new-vendor", "base_url": "https://unlisted.example/v1/",
                                   "api_protocol": "chat_completions"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["models"] == [{"id": "vision", "label": "vision", "vision_declared": True},
                                {"id": "unknown", "label": "unknown", "vision_declared": None}]
    assert result["excluded_non_image"] == 1
    assert result["model_invocation"] == "NOT_PROVEN" and result["activated"] is False
    assert "synthetic-discovery-secret" not in response.text
    assert len(calls) == 1 and not settings_directory().exists()


@pytest.mark.parametrize("status", [301, 302, 307, 401, 403, 404, 405, 429, 500])
def test_discovery_never_follows_redirects_or_exposes_upstream_errors(status, local_settings, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"Location": "https://another.example/models"},
                              text="synthetic-discovery-secret")

    discovery_transport(monkeypatch, handler)
    response = discover(client(), selection())
    assert response.status_code == 400 and "synthetic-discovery-secret" not in response.text
    assert len(calls) == 1 and not settings_directory().exists()


@pytest.mark.parametrize("payload", [b"not-json", b"null", b'{"data":{}}', b"x" * (4 * 1024 * 1024 + 1)])
def test_discovery_rejects_malformed_or_unbounded_response(payload, local_settings, monkeypatch):
    discovery_transport(monkeypatch, lambda _: httpx.Response(200, content=payload))
    assert discover(client(), selection()).status_code == 400
    assert not settings_directory().exists()


def test_discovery_timeout_is_safe_and_does_not_retry(local_settings, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("synthetic-discovery-secret")

    discovery_transport(monkeypatch, handler)
    response = discover(client(), selection())
    assert response.status_code == 400 and "synthetic-discovery-secret" not in response.text
    assert len(calls) == 1


def test_discovery_reuses_only_same_provider_endpoint_and_never_rotates_active_key(local_settings, monkeypatch):
    selected = selection()
    monkeypatch.setattr(ai_settings, "verify_connection", lambda *_: verified(selected))
    assert submit(client(), selected, "synthetic-discovery-secret").json()["activated"]
    before = ai_settings.read_index()
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer synthetic-discovery-secret"
        return httpx.Response(200, json={"data": []})

    discovery_transport(monkeypatch, handler)
    assert discover(client(), selected, "").status_code == 200
    assert discover(client(), selected | {"base_url": "https://different.example/v1"}, "").status_code == 400
    assert discover(client(), selected | {"provider": "different"}, "").status_code == 400
    assert len(calls) == 1 and ai_settings.read_index() == before


@pytest.mark.parametrize("headers,body", [({}, {}), ({"Origin": "https://evil.example"}, {}),
                                        ({"Origin": "http://127.0.0.1:8001"}, {"api_key": "x" * 17000})])
def test_discovery_enforces_local_request_contract_before_network(headers, body, local_settings, monkeypatch):
    monkeypatch.setattr(ai_settings, "discover_available_models", lambda *_: pytest.fail("must not query provider"))
    response = client().post("/api/ai-settings/models", headers=headers, json=body)
    assert response.status_code in {403, 413}


def test_local_only_origin_and_json_gate_before_any_billable_probe(
    local_settings, monkeypatch
):
    monkeypatch.setattr(
        ai_settings,
        "verify_connection",
        lambda *_: pytest.fail("must not call provider"),
    )
    for url, origin, remote in [
        ("http://evil.test:8001", "http://evil.test:8001", "127.0.0.1"),
        ("http://127.0.0.1:8001", "https://evil.test", "127.0.0.1"),
        ("http://127.0.0.1:8001", "http://127.0.0.1:8001", "192.168.1.5"),
    ]:
        c = TestClient(api.app, base_url=url, client=(remote, 50000))
        assert (
            c.post(
                "/api/ai-settings/verify",
                headers={"Origin": origin},
                json={"connection": selection(), "api_key": "synthetic-key"},
            ).status_code
            == 403
        )
    assert (
        client()
        .post(
            "/api/ai-settings/verify",
            json={"connection": selection(), "api_key": "synthetic-key"},
        )
        .status_code
        == 403
    )
    monkeypatch.setenv("VISIONCORTEX_WEB_AI_SETTINGS", "0")
    assert client().get("/api/ai-settings").status_code == 404


def test_invalid_body_never_echoes_key(local_settings):
    response = client().post(
        "/api/ai-settings/verify",
        headers={"Origin": "http://127.0.0.1:8001"},
        json={"api_key": {"secret": "synthetic-sensitive-value"}},
    )
    assert response.status_code == 400
    assert "synthetic-sensitive-value" not in response.text


def test_failed_actual_transport_cannot_change_existing_settings(
    local_settings, monkeypatch
):
    selected = selection()
    calls = []

    def factory(config):
        analyzer = ArkAnalyzer(config)
        analyzer.client.close()

        def handler(request):
            calls.append(request)
            return httpx.Response(
                401, json={"error": {"message": "synthetic-rejected-key"}}
            )

        analyzer.client = httpx.Client(transport=httpx.MockTransport(handler))
        return analyzer

    monkeypatch.setattr(
        ai_settings,
        "verify_connection",
        lambda conn, key, work: verify_connection(
            conn, key, work, analyzer_factory=factory
        ),
    )
    response = submit(client(), selected, "synthetic-rejected-key")
    assert response.status_code == 200 and not response.json()["activated"]
    assert len(calls) == 1
    assert "synthetic-rejected-key" not in response.text
    assert not (settings_directory() / "profiles.json").exists()
    assert api._settings()["mllm"]["model"] == local_settings["mllm"]["model"]


def test_saved_keys_bind_immutable_tasks_and_survive_process_restart(
    local_settings, monkeypatch
):
    calls = []

    def verify(conn, key, work):
        calls.append((conn, key))
        return verified(conn)

    monkeypatch.setattr(ai_settings, "verify_connection", verify)
    c = client()
    aliyun = selection()
    zhipu = selection("zhipu")
    assert submit(c, aliyun, "synthetic-aliyun-one").json()["activated"]
    old = api._settings()
    assert old["mllm"]["provider"] == "aliyun"
    assert submit(c, zhipu, "synthetic-zhipu").json()["activated"]
    assert api._settings()["mllm"]["provider"] == "zhipu"
    assert submit(c, aliyun, "synthetic-aliyun-two").json()["activated"]
    current = api._settings()
    assert current["mllm"]["credential_ref"] != old["mllm"]["credential_ref"]
    assert model_api_key(old["mllm"]) == "synthetic-aliyun-one"
    assert model_api_key(current["mllm"]) == "synthetic-aliyun-two"
    assert "synthetic-aliyun-one" not in json.dumps(old)
    # Simulate a new process restoring an old queued job from its JSON payload.
    monkeypatch.delenv(old["mllm"]["api_key_env"], raising=False)
    restored = json.loads(json.dumps(old))
    analyzer = ArkAnalyzer(restored)
    assert analyzer.api_key == "synthetic-aliyun-one"
    analyzer.close()
    assert ensure_ark_api_key(restored)["configured"]
    public = c.get("/api/ai-settings").json()
    assert set(public["profiles"]) == {"aliyun", "zhipu"}
    assert public["active"]["connection"] == aliyun
    assert "synthetic-aliyun-" not in json.dumps(public)
    assert "synthetic-zhipu" not in json.dumps(public)
    assert c.get("/api/health").json()["mllm_connection"]["provider"] == "aliyun"
    for path in settings_directory().rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


def test_reusing_key_requires_same_provider_and_endpoint(local_settings, monkeypatch):
    used = []

    def verify(conn, key, work):
        used.append(key)
        return verified(conn)

    monkeypatch.setattr(ai_settings, "verify_connection", verify)
    c = client()
    assert submit(c, selection(), "synthetic-aliyun").json()["activated"]
    assert submit(c, selection(model="other-vision"), "").json()["activated"]
    assert used[-1] == "synthetic-aliyun"
    assert (
        submit(c, selection(base_url="https://another.test/v1"), "").status_code == 400
    )
    assert submit(c, selection("zhipu"), "").status_code == 400
    assert len(used) == 2


def test_bad_permissions_and_changed_connection_fail_closed(
    local_settings, monkeypatch
):
    monkeypatch.setattr(
        ai_settings, "verify_connection", lambda conn, *_: verified(conn)
    )
    ai_settings.verify_and_activate(selection(), "synthetic-aliyun", local_settings)
    settings = api._settings()["mllm"]
    changed = dict(settings, model="other-model")
    with pytest.raises(RuntimeError):
        model_api_key(changed)
    root = settings_directory()
    reference = settings["credential_ref"]
    path = root / "credentials" / f"{reference}.json"
    path.chmod(0o644)
    with pytest.raises(RuntimeError):
        read_revision(reference)
    path.chmod(0o600)
    path.rename(path.with_suffix(".backup"))
    path.symlink_to(path.with_suffix(".backup"))
    with pytest.raises(OSError):
        read_revision(reference)
    with pytest.raises(RuntimeError):
        read_revision("../escape")


def test_declined_validation_preserves_prior_active_key(local_settings, monkeypatch):
    monkeypatch.setattr(
        ai_settings, "verify_connection", lambda conn, *_: verified(conn)
    )
    ai_settings.verify_and_activate(selection(), "synthetic-aliyun", local_settings)
    before = read_private(settings_directory() / "profiles.json")
    monkeypatch.setattr(
        ai_settings,
        "verify_connection",
        lambda conn, *_: {"status": "failed", "message": "failed"},
    )
    assert not ai_settings.verify_and_activate(
        selection("zhipu"), "synthetic-invalid", local_settings
    )["activated"]
    assert read_private(settings_directory() / "profiles.json") == before


def test_non_ark_provider_environment_preflight(local_settings, monkeypatch):
    settings = deepcopy(local_settings)
    settings["mllm"].update(selection(), api_key_env="SYNTHETIC_ALI_KEY")
    monkeypatch.setenv("SYNTHETIC_ALI_KEY", "synthetic-provider-key")
    assert ensure_ark_api_key(settings)["configured"]


def test_retry_refreshes_only_same_model_and_credential(local_settings, monkeypatch):
    from visioncortex.provider_credentials import write_private
    selected = selection()
    monkeypatch.setattr(ai_settings, "verify_connection", lambda connection, key, root: verified(connection))
    ai_settings.verify_and_activate(selected, "synthetic-retry-key", local_settings)
    original = ai_settings.apply_active(local_settings)
    old_ref = original["mllm"]["credential_ref"]
    old = read_revision(old_ref)
    old["verification"]["adapter_sha256"] = "old-adapter"
    write_private(settings_directory() / "credentials" / f"{old_ref}.json", old)
    with pytest.raises(ValueError):
        ai_settings.reverified_job_mllm(original)
    ai_settings.verify_and_activate(selected, "synthetic-different-key", local_settings)
    with pytest.raises(ValueError):
        ai_settings.reverified_job_mllm(original)
    ai_settings.verify_and_activate(selected, "synthetic-retry-key", local_settings)
    current = ai_settings.reverified_job_mllm(original)
    assert current["credential_ref"] != old_ref
    assert original["mllm"]["credential_ref"] == old_ref
    assert model_api_key(current) == "synthetic-retry-key"
    assert read_revision(old_ref)["verification"]["adapter_sha256"] == "old-adapter"
