"""Freeze photo-time relationships without choosing the first of several instruments."""

import json
from datetime import datetime
from field_recognition.bindings.binding_policy import contains, interruption_at
from field_recognition.bindings.binding_operator import at_time


def binding_contains_photo(binding, capture):
    try:
        from field_recognition.measurements.measurement_records import capture_clock

        timestamp, _ = capture_clock(capture)
        moment = datetime.fromisoformat(timestamp)
        return moment.tzinfo is not None and contains(binding, moment)
    except (TypeError, KeyError, ValueError):
        return False


def at_capture(core, conn, capture, *, linked=None, automatic=False):
    from field_recognition.measurements.measurement_records import capture_clock

    captured_at, time_basis = capture_clock(capture)
    try:
        moment = datetime.fromisoformat(captured_at)
        if moment.tzinfo is None:
            raise ValueError("timezone required")
    except (TypeError, ValueError):
        captured_at = None
        moment = datetime.fromisoformat(capture["received_at"])

    def contained(document):
        return contains(document, moment)

    snapshots = [linked] if linked and (not captured_at or contained(linked)) else []
    interrupted = False
    if automatic and not linked:
        snapshots = [
            json.loads(row["document"])
            for row in conn.execute(
                "SELECT document FROM bindings WHERE camera=? ORDER BY rowid",
                (capture["camera_id"],),
            )
        ]
        interrupted = any(interruption_at(b, moment) for b in snapshots)
        snapshots = [
            b
            for b in snapshots
            if contained(b) and (bool(captured_at) or core["current_readout_binding"](b))
        ]
    # Legacy overlapping ownership has no uniquely provable operator. Preserve the
    # snapshots as a conflict, never pick the first or the newest relationship.
    conflicts = []
    for binding in snapshots:
        others = [
            json.loads(r[0])
            for r in conn.execute(
                "SELECT document FROM bindings WHERE id<>? AND json_extract(document,'$.instrument.id')=?",
                (binding["binding_id"], binding["instrument"]["id"]),
            )
        ]
        if any(contained(other) for other in others):
            conflicts.append(binding["instrument"]["id"])
    snapshots = [at_time(b, moment) for b in snapshots if b["instrument"]["id"] not in conflicts]
    attribution_reason = (
        "capture_time_unverified"
        if not captured_at
        else "ambiguous_binding_history"
        if conflicts
        else "historical_binding_missing"
        if not snapshots
        else None
    )
    if not snapshots and interrupted and captured_at:
        attribution_reason = "camera_connection_interrupted"
    if attribution_reason:
        # Receive-time relations are diagnostic candidates, never evidence of
        # who wore the camera when an undated upload was captured.
        snapshots = [
            b | {"operator": None, "wearer_id": None, "attribution_status": "needs_review"}
            for b in snapshots
        ]
    hits = capture.get("matches", [])
    visible = {hit["id"] for hit in hits}
    selected = [b for b in snapshots if not visible or b["instrument"]["id"] in visible]
    candidates = [
        {
            "instrument": b["instrument"],
            "binding_id": b["binding_id"],
            "basis": "same_image_qr" if visible else "session_binding",
        }
        for b in selected
    ]
    if automatic:
        for hit in hits:
            if not any(c["instrument"]["id"] == hit["id"] for c in candidates):
                candidates.append(
                    {
                        "instrument": {
                            k: hit[k] for k in ("id", "name", "scene", "model") if k in hit
                        },
                        "binding_id": None,
                        "basis": "same_image_qr",
                    }
                )
    visits = [
        json.loads(row["document"])
        for row in conn.execute(
            "SELECT document FROM scene_visits WHERE camera=? ORDER BY rowid",
            (capture["camera_id"],),
        )
    ]
    visits = [at_time(v, moment) for v in visits if contained(v)]
    unique = selected[0] if len(selected) == 1 and len(candidates) == 1 else None
    names = {c["instrument"].get("scene") for c in candidates if c["instrument"].get("scene")}
    if not names:
        names = {v["scene"]["name"] for v in visits}
    workbenches = []
    for name in sorted(names):
        scene = conn.execute("SELECT * FROM scenes WHERE name=?", (name,)).fetchone()
        visit = next((v for v in visits if v["scene"]["name"] == name), None)
        workbenches.append(
            {
                "id": scene["id"] if scene else None,
                "name": name,
                "basis": "decoded_scene_qr" if visit else "instrument_registration",
                "scene_visit_id": visit["visit_id"] if visit else None,
            }
        )
    return {
        "all_binding_snapshots": snapshots,
        "binding_snapshots": selected,
        "binding_ids": [b["binding_id"] for b in selected],
        "instrument_candidates": candidates,
        "scene_snapshots": visits,
        "qr_matches": hits,
        "qr_scene_matches": capture.get("scene_matches", []),
        "workbenches": workbenches,
        "workbench": workbenches[0] if len(workbenches) == 1 else None,
        "association_status": "multiple_candidates"
        if len(candidates) > 1
        else "single_candidate"
        if candidates
        else "unbound",
        "ownership_conflicts": conflicts,
        "attribution_status": "needs_review" if attribution_reason else "resolved",
        "attribution_reason": attribution_reason,
        "binding_snapshot_version": "binding-snapshot/2",
        "binding_time_basis": time_basis if captured_at else "received_time_only",
        "binding_effective_at": captured_at,
        "allowed_instrument_ids": sorted({b["instrument"]["id"] for b in snapshots})
        if not attribution_reason
        else [],
        "allowed_instrument_ids_basis": "capture_time_binding_snapshot",
        "resolved_binding": unique,
    }
