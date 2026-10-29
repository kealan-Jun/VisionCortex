"""Explicit runtime actions for dashboard."""

from __future__ import annotations

import json
import field_recognition.recognition.aliyun_vision as aliyun_vision
from fastapi import HTTPException
from fastapi.responses import FileResponse
from field_recognition.measurements.activity import recent_activity
from field_recognition.api.security import filter_payload


def index(_runtime):
    return FileResponse(
        _runtime["BASE"] / "static" / "index.html", headers={"Cache-Control": "no-cache"}
    )


def state(_runtime):
    from field_recognition.measurements.history_records import (
        job_rows,
        panel_readings,
        latest_photo_job,
    )
    from field_recognition.bindings.binding_policy import sweep

    sweep(_runtime)
    if _runtime["RUNTIME_ENABLED"] and _runtime["SERVICE_ROLE"] == "api":
        from field_recognition.runtime.runtime_rpc import call

        try:
            _runtime["ocr_state"].update(call("ocr", "status", data_dir=_runtime.data_dir))
        except (OSError, HTTPException):
            _runtime["ocr_state"].update(status="worker_unavailable", resident=False)
    with _runtime["db"]() as conn:
        instruments = [
            _runtime["instrument_record"](row)
            for row in conn.execute("SELECT * FROM instruments ORDER BY device_category,name")
        ]
        bindings = [
            json.loads(row["document"])
            for row in conn.execute(
                "SELECT document FROM bindings WHERE ended IS NULL OR rowid IN (SELECT rowid FROM bindings ORDER BY rowid DESC LIMIT 30) ORDER BY rowid DESC"
            )
        ]
        jobs = [
            json.loads(row["document"]) | {"status": row["status"]}
            for row in conn.execute("SELECT status,document FROM jobs ORDER BY rowid DESC LIMIT 20")
        ]
        last_hit = (
            conn.execute(
                "SELECT document FROM scans WHERE json_extract(document,'$.camera_id')=? "
                "AND json_extract(document,'$.source')='neck_camera_gwhp_main' "
                "AND (json_array_length(document,'$.matches')>0 OR json_array_length(document,'$.scene_matches')>0) "
                "ORDER BY rowid DESC LIMIT 1",
                (_runtime["receiver_camera"].target,),
            ).fetchone()
            if _runtime["receiver_camera"]
            else None
        )
        activity = recent_activity(
            conn,
            _runtime["receiver_camera"].target
            if _runtime["receiver_camera"]
            else _runtime.config.camera_id,
        )
        latest_rows = job_rows(
            conn,
            _runtime["receiver_camera"].target
            if _runtime["receiver_camera"]
            else _runtime.config.camera_id,
            limit=1,
            recent=True,
        )
        latest_panel_job = (
            (json.loads(latest_rows[0]["document"]) | {"status": latest_rows[0]["status"]})
            if latest_rows
            else None
        )
        if latest_panel_job:
            related = panel_readings(latest_panel_job)
            latest_panel_job = latest_panel_job | {"readings": related, "lines": related}
        photo_job = latest_photo_job(
            conn,
            _runtime["receiver_camera"].target
            if _runtime["receiver_camera"]
            else _runtime.config.camera_id,
        )
    from field_recognition.measurements.measurement_records import display_fields

    for job in [*jobs, latest_panel_job, photo_job]:
        if job:
            job["display_fields"] = display_fields(job)
    categories = {}
    for instrument in instruments:
        categories.setdefault(instrument["device_category"], []).append(instrument["id"])
    return filter_payload(
        {
            "scenes": _runtime["scene_records"](),
            "scene_visits": _runtime["scene_visits"](),
            "instruments": instruments,
            "personnel_membership_enabled": _runtime["RUNTIME_ENABLED"],
            "device_categories": categories,
            "bindings": bindings,
            "jobs": jobs,
            "ocr": dict(_runtime["ocr_state"]),
            "last_camera_scan": json.loads(last_hit["document"]) if last_hit else None,
            "activity": activity,
            "latest_panel_job": latest_panel_job,
            "latest_photo_job": photo_job,
            "record_policy": {
                "mode": _runtime["RECORD_MODE"],
                "automatic_write_scope": "test_only",
                "production_photo_submission": "draft_confirmation",
                "drafts_url": "/photo-measurements",
            },
            "archive": _runtime["archive_store"].snapshot(),
            "vision_fallback": aliyun_vision.public_config(_runtime.config.recognition),
            "photo_watch": _runtime["saved_photo_watcher"].snapshot(),
            "automation": _runtime["automatic_runner"].snapshot(),
            "video_ocr": _runtime["video_ocr"].snapshot(),
            "panel_detector": _runtime["panel_detector"].snapshot(),
            "camera": _runtime["receiver_camera"].snapshot()
            if _runtime["receiver_camera"]
            else {
                "configured": bool(_runtime.config.camera_snapshot_url),
                "id": _runtime.config.camera_id or "UnconfiguredNeckCamera",
                "mode": "http_snapshot",
            },
            "product": "FieldRecognition",
        }
    )
