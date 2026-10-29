"""Workbench → instrument → reading index, preserving unassigned evidence."""

import json
from field_recognition.measurements.reading_results import build_readings
from field_recognition.measurements.history_records import panel_readings, job_rows


def summarize(conn, camera, *, limit=200, related_only=False):
    registry = [dict(row) for row in conn.execute("SELECT * FROM instruments ORDER BY name")]
    scenes = {r["name"]: r["id"] for r in conn.execute("SELECT * FROM scenes")}
    groups = {}

    def bench(name):
        name = name or "未确定实验台"
        if name not in groups:
            groups[name] = {
                "scene_id": scenes.get(name),
                "name": name,
                "instruments": {},
                "unassigned_readings": [],
                "photo_count": 0,
                "reading_count": 0,
                "job_ids": set(),
            }
        return groups[name]

    def instrument(group, asset):
        if asset["id"] not in group["instruments"]:
            group["instruments"][asset["id"]] = {
                "id": asset["id"],
                "name": asset["name"],
                "readings": [],
            }
        return group["instruments"][asset["id"]]

    for asset in registry:
        instrument(bench(asset["scene"]), asset)
    where, params = (
        (" WHERE json_extract(document,'$.camera_id')=?", [camera]) if camera else ("", [])
    )
    rows = (
        job_rows(conn, camera, limit=limit + 1)
        if related_only
        else conn.execute(
            "SELECT document,status FROM jobs" + where + " ORDER BY rowid DESC LIMIT ?",
            params + [limit + 1],
        ).fetchall()
    )
    for row in rows[:limit]:
        doc = json.loads(row["document"])
        readings = panel_readings(doc, row["status"]) if related_only else doc.get("readings")
        if readings is None:
            readings = build_readings(doc)
        context = doc.get("workbench") or {}
        default_name = (
            context.get("name")
            or (doc.get("instrument") or {}).get("scene")
            or (doc.get("scene") or {}).get("name")
        )
        if not related_only:
            group = bench(default_name)
            group["job_ids"].add(doc["job_id"])
        for r in readings:
            asset = r.get("instrument")
            target = bench(asset.get("scene") if asset else default_name)
            target["job_ids"].add(doc["job_id"])
            entry = r | {
                "job_id": doc["job_id"],
                "camera_id": doc["camera_id"],
                "operator": doc.get("operator"),
                "captured_at": (doc.get("external_photo") or {}).get("captured_at")
                or (doc.get("video_observation") or {}).get("observed_at")
                or doc["submitted_at"],
                "result_url": "/api/jobs/" + doc["job_id"],
                "image_url": doc.get("image_url"),
                "input_mode": "video" if doc.get("request_trigger") == "video_stream" else "photo",
            }
            if asset:
                instrument(target, asset)["readings"].append(entry)
            else:
                target["unassigned_readings"].append(entry)
            target["reading_count"] += 1
    for g in groups.values():
        g["photo_count"] = len(g.pop("job_ids"))
        g["instruments"] = list(g["instruments"].values())
    return {
        "workbenches": list(groups.values()),
        "scope": f"latest_{limit}_jobs",
        "related_only": related_only,
        "has_older": len(rows) > limit,
        "camera_id": camera,
        "measurement_values_summed": False,
        "activity_inferred": False,
    }
