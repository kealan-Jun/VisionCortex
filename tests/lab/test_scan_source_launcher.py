"""Original HTTP/static resources with owned state, sources off; no real camera."""

import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from visioncortex.lab.domain import Conflict
from visioncortex.lab.scan_source_scanner import app_factory


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    (assets / "static").mkdir(parents=True)
    index = assets / "static/index.html"
    index.write_bytes(b"<html>unchanged fixture asset</html>")
    for filename in ("archive.html", "archive.js", "view-utils.js"):
        (assets / "static" / filename).write_bytes(b"neutral fixture resource")
    document = dict(
        schema_version="vc-lab-scan-launcher/1",
        asset_root=str(assets),
        asset_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        runtime_root=str(tmp_path / "owned-state"),
        max_runtime_bytes=16 * 1024**2,
        source_enabled=False,
        field_config={},
    )
    manifest = tmp_path / "launcher.json"
    manifest.write_text(json.dumps(document))
    manifest.chmod(0o600)
    monkeypatch.setenv("VC_LAB_SCAN_LAUNCHER_MANIFEST", str(manifest))
    return document, manifest


def test_original_assets_and_health_start_with_owned_state_and_all_sources_off(
    launcher, monkeypatch
):
    document, _ = launcher
    monkeypatch.setenv("FIELD_RECEIVER_URL", "http://fixture.invalid/never-connect")
    monkeypatch.setenv("FIELD_VIDEO_OCR_ENABLED", "1")
    monkeypatch.setenv("FIELD_PRODUCTION_ENABLED", "1")
    import urllib.request
    from field_recognition.recognition import service

    def prohibited(*args, **kwargs):
        raise AssertionError("Disabled launcher attempted to construct a live Receiver")

    monkeypatch.setattr(urllib.request, "urlopen", prohibited)
    monkeypatch.setattr(service, "create_ocr_model", prohibited)
    app = app_factory()
    runtime = app.state.runtime
    assert runtime.config.receiver_url is None and runtime.service_role == "api"
    assert runtime["receiver_camera"].target == "UnconfiguredNeckCamera"
    assert runtime["ocr_model"] is None and not runtime["RUNTIME_ENABLED"]
    assert (
        not runtime.config.video_ocr_enabled
        and not runtime.config.saved_photo_watch_enabled
    )
    assert not runtime.config.auto_run_enabled and not runtime.config.archive_enabled
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["status"] == "ready"
        assert (
            client.get("/static/index.html").content
            == b"<html>unchanged fixture asset</html>"
        )
    assert runtime._closed
    from pathlib import Path

    assert (Path(document["runtime_root"]) / "Scanner.sqlite3").is_file()
    assert not (Path(document["asset_root"]) / "Data").exists()


@pytest.mark.parametrize(
    "change", ["unknown", "asset_hash", "source_override", "state_inside_assets"]
)
def test_launcher_requires_exact_private_assets_and_disabled_source_boundary(
    launcher, change
):
    document, manifest = launcher
    if change == "unknown":
        document["unapproved_option"] = True
    elif change == "asset_hash":
        document["asset_index_sha256"] = "0" * 64
    elif change == "source_override":
        document["field_config"] = {"receiver_url": "http://fixture.invalid"}
    else:
        document["runtime_root"] = document["asset_root"] + "/state"
    manifest.write_text(json.dumps(document))
    with pytest.raises((ValueError, Conflict)):
        app_factory()
