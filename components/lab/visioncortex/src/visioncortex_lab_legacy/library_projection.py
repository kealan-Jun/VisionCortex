"""Presentation-only summaries. Full evidence stays behind the detail endpoints."""

from __future__ import annotations


def project_staging_library(data: dict, section: str) -> dict:
    result = {
        key: data.get(key)
        for key in (
            "name",
            "path",
            "staging_run_id",
            "release_id",
            "counts",
            "integrity_status",
        )
    }
    result["observability"] = {
        "status": data.get("observability", {}).get("status", {})
    }
    result["library_section"] = section
    if section == "library-reports":
        result.update(
            daily_report=data.get("daily_report", {}), links=data.get("links", {})
        )
        return result
    if section != "library-materials":
        raise ValueError("Unsupported library section")
    fields = {
        "event_id",
        "event_uid",
        "parent_event_id",
        "parent_event_uid",
        "action_type",
        "cv_action_type",
        "objects",
        "cv_objects",
        "start_us",
        "end_us",
        "aligned_frame_url",
        "aligned_clip_url",
        "dual_view_material_ready",
        "frame_url",
        "clip_url",
        "disposition",
        "review_status",
        "evidence_classification",
        "group_name",
        "release_id",
        "decision",
    }
    for key in ("key_events", "preliminary_materials", "quarantined_materials"):
        rows = []
        for event in data.get(key, []):
            row = {k: v for k, v in event.items() if k in fields}
            row["provenance"] = {
                "mllm": {
                    "current_step": (event.get("provenance", {}).get("mllm") or {}).get(
                        "current_step"
                    )
                }
            }
            rows.append(row)
        result[key] = rows
    return result
