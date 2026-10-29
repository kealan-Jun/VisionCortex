"""Replay business snapshots from the existing transactional archive outbox.

Reading this feed never acknowledges archival, loads models or opens media.
An outbox sequence is a delivery revision, not a physical capture identity.
"""

import json
from datetime import datetime, timezone

from fastapi import HTTPException, Query

from field_recognition.api.security import require_camera, visible
from field_recognition.measurements.measurement_records import capture_clock, display_fields


SCHEMA = "field-event-feed/1"
ENTITIES = {"scans": "photo_capture", "jobs": "ocr_observation",
            "bindings": "instrument_binding", "scene_visits": "scene_binding",
            "photo_measurements": "measurement_review", "experiment_records": "measurement_confirmation",
            "binding_handoffs": "instrument_handoff"}


def instant(value):
    try:
        moment = datetime.fromisoformat(value)
        return moment.astimezone(timezone.utc).isoformat() if moment.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def association(value, refs, *, conflict=False):
    return {"id": value, "state": "conflict" if conflict else "explicit" if value else "unknown",
            "evidence_refs": refs if value or conflict else [], "version": 1}


def normalize(row, source_instance, tenant=None):
    """Project the immutable outbox snapshot; never borrow the current binding."""
    doc = json.loads(row["document"])
    entity, ident, sequence = row["entity"], row["entity_id"], row["seq"]
    camera = doc.get("camera_id") or doc.get("camera")
    event_id = f"{source_instance}:{entity}:{ident}"
    ref = {"source": "field_recognition", "source_instance": source_instance,
           "entity": entity, "entity_id": ident, "outbox_seq": sequence}
    captured, basis = capture_clock(doc)
    start = instant(doc.get("started_at")) if entity in {"bindings", "scene_visits"} else instant(captured)
    end = instant(doc.get("ended_at")) if entity in {"bindings", "scene_visits"} else start
    snapshots = doc.get("all_binding_snapshots", doc.get("binding_snapshots", []))
    instrument = (doc.get("instrument") or {}).get("id")
    wearer, operator = doc.get("wearer_id"), doc.get("operator_id")
    conflict = doc.get("attribution_status") == "needs_review"
    payload = {"status": doc.get("status"), "record_scope": doc.get("record_scope"),
               "operator_label": doc.get("operator"), "binding_id": doc.get("binding_id"),
               "binding_snapshots": snapshots, "recognition_root_job_id": doc.get("recognition_root_job_id"),
               "recognition_revision": doc.get("recognition_revision", 1),
               "review_status": doc.get("status") if entity in {"photo_measurements", "experiment_records"} else "unreviewed",
               "physical_action_verified": False}
    if entity in {"bindings", "scene_visits"}:
        payload.update(valid_from=start, valid_to=end, valid_interval="[valid_from,valid_to)",
                       role=doc.get("role", "wearer" if wearer else "unknown"),
                       binding_id=doc.get("binding_id") or doc.get("visit_id"),
                       scan_id=doc.get("scan_id"), end_reason=doc.get("end_reason"),
                       recorded_at=instant(row["recorded_at"]))
    if entity == "jobs":
        payload.update(raw_lines=doc.get("lines", []), raw_readings=doc.get("readings", []),
                       normalized_values=display_fields(doc), panel_regions=doc.get("panel_regions", []),
                       skip_reason=doc.get("skip_reason"), error=doc.get("error"),
                       recognition_versions=doc.get("recognition_versions"),
                       local_ocr=doc.get("local_ocr"), fallback=doc.get("fallback"))
    if entity in {"photo_measurements", "experiment_records"}:
        # Review corrections retain their own source IDs and human decisions.
        payload.update(fields=doc.get("confirmed_fields", doc.get("fields", [])),
                       sources=doc.get("sources", []), corrections=doc.get("corrections", []),
                       context=doc.get("context", {}), revision=doc.get("revision"),
                       confirmation=doc.get("confirmation"), confirmed_at=doc.get("confirmed_at"))
        moments = [instant((source.get("external_photo") or {}).get("captured_at")
                           or source.get("captured_at")) for source in doc.get("sources", [])]
        moments = [moment for moment in moments if moment]
        if moments:
            start, end = min(moments), max(moments)
    context = doc.get("measurement_context") or doc.get("context") or {}
    associations = {"wearer": association(wearer, [ref], conflict=conflict),
                    "operator": association(operator, [ref], conflict=conflict),
                    "instrument": association(instrument, [ref]),
                    "experiment": association(context.get("experiment_context_ref"), [ref]),
                    "sample": association(context.get("sample_id"), [ref]),
                    "step": association(context.get("step_id"), [ref])}
    external, metadata = doc.get("external_photo") or {}, doc.get("frame_metadata") or {}
    return {"schema_version": "field-event/1", "source": "field_recognition",
            "source_instance": source_instance, "event_id": event_id, "delivery_id": f"{source_instance}:{sequence}",
            "sequence": sequence, "revision": sequence, "tenant": tenant, "type": ENTITIES[entity],
            "entity": entity, "entity_id": ident, "occurred_start": start, "occurred_end": end,
            "received_at": instant(doc.get("received_at") or doc.get("submitted_at")),
            "processed_at": instant(doc.get("finished_at")), "recorded_at": instant(row["recorded_at"]),
            "capture_device_id": camera, "instrument_id": instrument, "wearer_id": wearer,
            "operator_id": operator, "actor_track_id": doc.get("actor_track_id"),
            "capture_id": doc.get("capture_id"), "capture_session_id": metadata.get("session_id") or doc.get("capture_session_id"),
            "clock": {"basis": basis if start else None, "source_pts_us": metadata.get("pts_us"),
                      "mapping_version": metadata.get("clock_mapping_version"),
                      "uncertainty_us": metadata.get("clock_uncertainty_us")},
            "media": {"blob_sha256": external.get("source_sha256") or doc.get("source_sha256") or doc.get("image_sha256"),
                      "image_sha256": doc.get("image_sha256"), "source_ref": external.get("source_ref"),
                      "image_url": doc.get("image_url"), "original_url": "/api/captures/" + doc["capture_id"] + "/original" if doc.get("capture_id") else None,
                      "coordinates": doc.get("crop"), "intervals": []},
            "associations": associations, "evidence_refs": [ref], "payload": payload,
            "withdrawn": doc.get("status") in {"rejected", "cancelled"}}


def feed(conn, camera, *, after_seq=0, limit=100, source_instance=None, tenant=None):
    row = conn.execute("SELECT value FROM archive_meta WHERE key='source_instance'").fetchone()
    if not row:
        raise HTTPException(503, "Transactional event outbox identity is unavailable")
    instance = row[0]
    scope = conn.execute("SELECT value FROM archive_meta WHERE key='event_tenant'").fetchone()
    if scope is not None:
        frozen_tenant = json.loads(scope[0])
        if tenant is not None and tenant != frozen_tenant:
            raise HTTPException(409, "Event tenant differs from the producer's immutable namespace")
        tenant = frozen_tenant
    if source_instance is not None and source_instance != instance:
        raise HTTPException(409, "Event source instance changed; preserve the previous inbox epoch")
    high = conn.execute("SELECT COALESCE(MAX(seq),0) FROM archive_outbox").fetchone()[0]
    if after_seq > high:
        raise HTTPException(409, "Event cursor exceeds the producer ledger; reconciliation is required")
    # The global cursor can skip other cameras, but cannot skip a selected camera's revision.
    entities = ",".join("?" for _ in ENTITIES)
    rows = conn.execute(
        f"SELECT * FROM archive_outbox WHERE seq>? AND seq<=? AND entity IN ({entities}) "
        "AND COALESCE(json_extract(document,'$.camera_id'),json_extract(document,'$.camera'))=? "
        "ORDER BY seq LIMIT ?", (after_seq, high, *ENTITIES, camera, limit + 1)).fetchall()
    selected, more = rows[:limit], len(rows) > limit
    cursor = selected[-1]["seq"] if more else high
    items = [normalize(row, instance, tenant) for row in selected if visible(json.loads(row["document"]))]
    return {"schema": SCHEMA, "source": "field_recognition", "source_instance": instance,
            "tenant": tenant, "camera_id": camera, "after_seq": after_seq, "next_cursor": cursor,
            "high_watermark": high, "has_more": more, "items": items,
            "delivery_semantics": "at_least_once", "archive_acknowledged": False}


def install(core):
    # Namespace is fixed to the producer epoch, rather than restamping old
    # deliveries whenever a service configuration changes.
    tenant = core["config"].event_tenant
    with core["db"]() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO archive_meta(key,value) VALUES('event_tenant',?)",
                     (json.dumps(tenant),))
        scope = json.loads(conn.execute("SELECT value FROM archive_meta WHERE key='event_tenant'").fetchone()[0])
        if scope != tenant:
            raise ValueError("FIELD_EVENT_TENANT changed within an existing producer epoch")

    @core["app"].get("/api/events/feed")
    def read(camera_id: str = Query(min_length=1, max_length=200),
             after_seq: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=100),
             source_instance: str | None = None):
        require_camera(camera_id)
        with core["db"]() as conn:
            conn.execute("BEGIN")
            return feed(conn, camera_id, after_seq=after_seq, limit=limit, source_instance=source_instance,
                        tenant=core["config"].event_tenant)
