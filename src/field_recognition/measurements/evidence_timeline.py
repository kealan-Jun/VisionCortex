"""Two acquisition streams on a capture-time axis; never manufacture a clock or burst."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import AwareDatetime

from field_recognition.measurements.measurement_records import (
    build_records,
    capture_clock,
    photo_time,
)


def stream_of(document):
    if (
        document.get("request_trigger") == "video_stream"
        or document.get("source") == "neck_camera_video_ocr"
    ):
        return "video"
    if document.get("external_photo") or document.get("request_trigger") in {
        "voice_photo_directory",
        "voice_burst",
    }:
        return "voice"
    return "legacy_photo"


def capture_time(document):
    value, basis = capture_clock(document)
    try:
        instant = datetime.fromisoformat(value)
        if instant.tzinfo is None:
            raise ValueError("Missing timezone")
        instant = instant.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        instant = None
    external = document.get("external_photo") or {}
    basis = external.get("timestamp_basis", basis) if external else basis
    return {
        "captured_at": instant.isoformat() if instant else None,
        "timestamp_ms": photo_time(document)["timestamp_ms"] if instant else None,
        "basis": basis,
        "clock_sync_valid": (document.get("video_observation") or {}).get("clock_sync_valid"),
        "uncertainty_ms": None,
        "quality": (
            "unknown"
            if instant is None
            else "filename_time"
            if basis == "nas_filename_local_time_not_hardware_verified"
            else "synchronized_frame"
            if basis == "synchronized_frame_timestamp"
            else "source_declared"
        ),
    }


def acquisition(document, group=None):
    stream = stream_of(document)
    context = (group or {}).get("context") or document.get("measurement_context") or {}
    burst = context.get("burst_id")
    return {
        "stream": stream,
        "purpose": "continuous_observation"
        if stream == "video"
        else "voice_capture"
        if stream == "voice"
        else "legacy_material",
        "time": capture_time(document),
        "burst_id": burst,
        "expected_photos": context.get("expected_photos") if burst else None,
        "batch_basis": (
            (document.get("external_photo") or {}).get("batch_basis") or "explicit_burst_id"
        )
        if burst
        else "not_applicable"
        if stream != "voice"
        else "unverified",
        "evidence_reason": (document.get("video_observation") or {}).get("evidence_reason"),
    }


def timeline(
    conn,
    camera,
    *,
    start,
    end,
    stream="all",
    instrument_id=None,
    wearer_id=None,
    limit=100,
    offset=0,
):
    """Read a consistent snapshot, group only explicit measurements, retain each photo.

    A group's member outside the requested interval remains visible as evidence.
    No observation is extended until the next observation, and no nearest-time
    video/photo pair is asserted to describe the same physical action.
    """
    from field_recognition.measurements.result_reprocessing import preferred_jobs
    from field_recognition.archive.archive_store import receipt_status
    from field_recognition.api.security import visible

    rows = conn.execute(
        "SELECT status,document FROM jobs WHERE json_extract(document,'$.camera_id')=?", (camera,)
    )
    documents = [json.loads(r["document"]) | {"status": r["status"]} for r in rows]
    documents = preferred_jobs([d for d in documents if visible(d)])
    groups = {
        r["id"]: json.loads(r["document"])
        for r in conn.execute(
            "SELECT id,document FROM photo_measurements WHERE camera=?", (camera,)
        )
    }
    bundles = {}
    unknown = 0
    for doc in documents:
        source = stream_of(doc)
        if source not in {"video", "voice"} or stream not in {"all", source}:
            continue
        snapshots = doc.get("all_binding_snapshots", doc.get("binding_snapshots", []))
        if instrument_id and not any(
            (s.get("instrument") or {}).get("id") == instrument_id for s in snapshots
        ):
            continue
        if wearer_id and wearer_id not in {
            doc.get("wearer_id"),
            *(s.get("wearer_id") for s in snapshots),
        }:
            continue
        mid = doc.get("measurement_id") if source == "voice" else None
        # Never merge separate legacy photos by filename, time proximity or pixel hash.
        key = "measurement:" + mid if mid else "capture:" + (doc.get("capture_id") or doc["job_id"])
        bundle = bundles.setdefault(
            key,
            {
                "event_id": key,
                "stream": source,
                "camera_id": camera,
                "measurement_id": mid,
                "documents": [],
                "group": groups.get(mid),
            },
        )
        bundle["documents"].append(doc)
    events = []
    for bundle in bundles.values():
        group = bundle.pop("group")
        members = bundle.pop("documents")
        clocks = [capture_time(d) for d in members]
        instants = [datetime.fromisoformat(c["captured_at"]) for c in clocks if c["captured_at"]]
        if not instants:
            unknown += 1
            continue
        if not any(start <= moment < end for moment in instants):
            continue
        sources = []
        for doc, clock in sorted(
            zip(members, clocks), key=lambda p: (p[1]["captured_at"] or "", p[0]["job_id"])
        ):
            summary = acquisition(doc, group)
            measurements = build_records(doc)
            instruments = {
                (r.get("instrument") or {}).get("id"): (r.get("instrument") or {})
                for r in doc.get("panel_regions", [])
            }
            measurements = [
                m | {"instrument_name": instruments.get(m["instrument_id"], {}).get("name")}
                for m in measurements
            ]
            if instrument_id:
                measurements = [m for m in measurements if m["instrument_id"] == instrument_id]
            row = conn.execute(
                "SELECT document FROM scans WHERE id=?", (doc.get("capture_id"),)
            ).fetchone()
            capture = json.loads(row["document"]) if row else {}
            sources.append(
                {
                    "job_id": doc["job_id"],
                    "capture_id": doc.get("capture_id"),
                    "status": doc.get("status"),
                    "result_stage": doc.get("result_stage"),
                    "parallel_review": doc.get("parallel_review"),
                    "record_scope": doc.get("record_scope"),
                    "acquisition": summary,
                    "operator": doc.get("operator"),
                    "wearer_id": doc.get("wearer_id"),
                    "image_url": doc.get("image_url"),
                    "image_sha256": doc.get("image_sha256"),
                    "original_url": "/api/captures/" + doc["capture_id"] + "/original"
                    if capture.get("original_blob")
                    else None,
                    "source_ref": (doc.get("external_photo") or {}).get("source_ref"),
                    "source_sha256": capture.get("source_sha256")
                    or (doc.get("external_photo") or {}).get("source_sha256"),
                    "frame_metadata": doc.get("frame_metadata"),
                    "binding_snapshots": doc.get(
                        "all_binding_snapshots", doc.get("binding_snapshots", [])
                    ),
                    "measurements": measurements,
                    "skip_reason": doc.get("skip_reason"),
                    "result_url": "/api/jobs/" + doc["job_id"],
                    "versions_url": "/api/jobs/" + doc["job_id"] + "/versions",
                    "archive": receipt_status(conn, "jobs", doc["job_id"]),
                }
            )
        context = (group or {}).get("context") or members[0].get("measurement_context") or {}
        expected = context.get("expected_photos") if context.get("burst_id") else None
        received = len({d.get("capture_id") or d["job_id"] for d in members})
        bundle.update(
            start_at=min(instants).isoformat(),
            end_at=max(instants).isoformat(),
            sources=sources,
            burst_id=context.get("burst_id"),
            expected_photos=expected,
            received_photos=received,
            batch_status=(
                "not_applicable"
                if bundle["stream"] == "video"
                else "unverified"
                if not expected
                else "complete"
                if received == expected
                else "incomplete"
            ),
            status=(group or {}).get("status", members[-1].get("status")),
            record_scope=(group or {}).get("record_scope", members[-1].get("record_scope")),
            decision_url="/api/photo-measurements/" + bundle["measurement_id"] if group else None,
        )
        events.append(bundle)
    events.sort(key=lambda e: (e["start_at"], e["event_id"]))
    return {
        "schema": "field-evidence-timeline/1",
        "camera_id": camera,
        "order": "capture_time_asc",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval": "[start,end)",
        "items": events[offset : offset + limit],
        "total": len(events),
        "next_offset": offset + limit if offset + limit < len(events) else None,
        "unknown_time_events": unknown,
        "unknown_time_scope": "selected_camera_stream_and_identity_all_history",
        "physical_action_verified": False,
        "cross_stream_matching": "not_inferred",
    }


def install(core):
    @core["app"].get("/trace")
    def page():
        return FileResponse(
            Path(core["BASE"]) / "static" / "trace.html", headers={"Cache-Control": "no-store"}
        )

    @core["app"].get("/api/evidence-timeline")
    def read(
        start: AwareDatetime,
        end: AwareDatetime,
        stream: Literal["all", "video", "voice"] = "all",
        camera_id: str | None = None,
        instrument_id: str | None = None,
        wearer_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        from field_recognition.runtime.runtime_rpc import selected_camera
        from field_recognition.api.security import require_camera

        camera = (
            camera_id
            or selected_camera.get()
            or (
                core["receiver_camera"].target
                if core["receiver_camera"]
                else (
                    core["config"].camera_id
                    if core.get("config")
                    else os.environ.get("FIELD_CAMERA_ID")
                )
            )
        )
        if not camera:
            raise HTTPException(422, "请选择相机")
        require_camera(camera)
        start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        if (
            not 0 < (end - start).total_seconds() <= 31 * 86400
            or not 1 <= limit <= 100
            or offset < 0
        ):
            raise HTTPException(422, "查询区间须大于零且不超过 31 天，每页 1–100 条")
        with core["db"]() as conn:
            conn.execute("BEGIN")  # Consistent read snapshot while writers continue.
            return timeline(
                conn,
                camera,
                start=start,
                end=end,
                stream=stream,
                instrument_id=instrument_id,
                wearer_id=wearer_id,
                limit=limit,
                offset=offset,
            )
