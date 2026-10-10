from fastapi.testclient import TestClient

from visioncortex import api


def test_maintenance_boot_does_not_touch_nas_or_start_workers(monkeypatch):
    monkeypatch.setenv("VISIONCORTEX_STORAGE_MAINTENANCE", "1")
    monkeypatch.setattr(api, "validate_web_access_configuration", lambda: None)
    calls = []
    for name in ("_settings", "_initialize_persistent_queue", "_expire_stale_upload_sessions",
                 "_recover_jobs_from_archive_receipts", "_recover_orphaned_tasks",
                 "_start_queue_worker", "_start_nas_monitor"):
        monkeypatch.setattr(api, name, lambda *args: calls.append(args))
    monkeypatch.setattr(api._device_day_service, "start", lambda: calls.append("start"))
    monkeypatch.setattr(api._device_day_service, "stop", lambda: calls.append("stop"))
    with TestClient(api.app) as client:
        response = client.get("/archive-overview")
        assert response.status_code == 503
        assert "NAS 正在维护" in response.text
        assert response.headers["cache-control"] == "no-store"
        response = client.post("/api/experiments")
        assert response.status_code == 503
        assert response.json()["status"] == "storage_maintenance"
    assert not calls


def test_maintenance_requires_explicit_switch(monkeypatch):
    monkeypatch.delenv("VISIONCORTEX_STORAGE_MAINTENANCE", raising=False)
    assert not api._storage_maintenance()
