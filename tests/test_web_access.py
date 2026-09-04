from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from visioncortex import api
from visioncortex.web_access import (
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


def test_frontend_keeps_polling_when_one_progress_read_fails():
    app_js = (
        Path(api.__file__).parent / "web" / "app.js"
    ).read_text(encoding="utf-8")

    poller = app_js.split("async function pollRun", 1)[1].split(
        "function renderTasks", 1
    )[0]
    assert "consecutiveReadFailures += 1" in poller
    assert "任务仍在后台运行" in poller
    assert "continue;" in poller
    assert "run.state === \"failed\"" in poller


def test_frontend_refreshes_grouped_nas_batches_and_monitor_state():
    app_js = (
        Path(api.__file__).parent / "web" / "app.js"
    ).read_text(encoding="utf-8")

    assert "state.nasBatches = payload.batches || []" in app_js
    assert "state.nasMonitor = payload.monitor || null" in app_js
    assert 'document.querySelector("#nas-batches")' in app_js
    assert "batchPicker.outerHTML = nasBatchPicker()" in app_js
    assert "持续监控中" in app_js


def test_frontend_uses_distinct_product_libraries_and_shareable_filters():
    app_js = (
        Path(api.__file__).parent / "web" / "app.js"
    ).read_text(encoding="utf-8")

    assert "function renderMaterialsLibrary(" in app_js
    assert "function renderReportsLibrary(" in app_js
    assert "function archiveLibraryHash(" in app_js
    assert 'query.set("status", filters.status)' in app_js
    assert 'query.set("view", filters.view)' in app_js
    assert "data-global-material-filter=\"date\"" in app_js
    assert "data-global-material-filter=\"object\"" in app_js


def test_frontend_supports_grouped_search_focus_mode_and_safe_rerun():
    app_js = (
        Path(api.__file__).parent / "web" / "app.js"
    ).read_text(encoding="utf-8")

    assert "function globalSearchGroups(" in app_js
    assert '[["实验",experimentResults],["步骤",stepResults],["关键素材",materialResults],["报告",reportResults]]' in app_js
    assert "function openMaterialFocus(" in app_js
    assert "data-material-focus" in app_js
    assert "data-focus-nav" in app_js
    assert "function rerunArchive(" in app_js
    assert "original_upload_manifest.json" in app_js
    assert 'api("/api/runs/from-paths"' in app_js


def test_frontend_has_2k_4k_density_without_forcing_1080p_zoom():
    web_root = Path(api.__file__).parent / "web"
    styles = (web_root / "styles.css").read_text(encoding="utf-8")
    index = (web_root / "index.html").read_text(encoding="utf-8")

    large_screen = styles.split("@media (min-width: 2200px) {", 1)[1].split(
        "@media (min-width: 2200px) and", 1
    )[0]
    assert "--font-md: 16px" in large_screen
    assert "width: min(100%,2800px)" in large_screen
    assert ".home-launchpad {" in large_screen
    assert ".library-card-grid { grid-template-columns: repeat(4" in large_screen
    assert "width: min(2100px,calc(100vw - 96px))" in large_screen
    assert "styles.css?v=20260904-product-shell-31" in index
    assert "app.js?v=20260904-product-shell-31" in index

    app_js = (web_root / "app.js").read_text(encoding="utf-8")
    assert "function bindHomeLaunchpad()" in app_js
    assert 'location.hash = "#/new"' in app_js
    assert 'launchpad.addEventListener("drop"' in app_js
    assert "importFromHome(event.dataTransfer.files)" in app_js


def test_normal_product_pages_hide_raw_paths_and_explain_partial_results():
    app_js = (
        Path(api.__file__).parent / "web" / "app.js"
    ).read_text(encoding="utf-8")

    result_header = app_js.split("function resultHeader", 1)[1].split(
        "function experimentExecutiveSummary", 1
    )[0]
    run_card = app_js.split("function runObservabilityCard", 1)[1].split(
        "function renderOperations", 1
    )[0]
    attention_panel = app_js.split("function experimentAttentionPanel", 1)[1].split(
        "function resultHeader", 1
    )[0]
    archive_actions = app_js.split("function bindArchiveActions", 1)[1].split(
        "function updateResultNavDensity", 1
    )[0]

    assert "data.path" not in result_header
    assert "archive-file-details" not in result_header
    assert "archivePath" not in run_card
    assert "run.error" not in run_card
    assert "查看技术信息" not in attention_panel
    assert "result.path" not in archive_actions
    assert "本页仅展示处理停止前已完成的内容" in result_header
    assert "function experimentAttentionPanel(" in app_js
    assert "本次处理未生成实验日报" in app_js
    assert "本次处理未生成专业报告" in app_js
