from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


PHYSICAL_CHANGE_SCHEMA_VERSION = "visioncortex-physical-change/1"


def physical_change_records(
    archive_id: str,
    normalized_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project explicit before/after assertions without inventing missing state."""

    records: list[dict[str, Any]] = []
    for event in normalized_events:
        event_id = str(event.get("event_id") or "")
        parent_event_id = str(event.get("parent_event_id") or "unassigned")
        event_uid = str(
            (((event.get("provenance") or {}).get("index") or {}).get("event_uid"))
            or f"{archive_id}:{parent_event_id}:{event_id}"
        )
        before = event.get("state_before") or {}
        after = event.get("state_after") or {}
        objects = event.get("objects") or {}
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        decision = event.get("decision") or {}
        for object_role in sorted(set(before) | set(after)):
            state_before = _known_state(before.get(object_role))
            state_after = _known_state(after.get(object_role))
            if state_before is None or state_after is None or state_before == state_after:
                continue
            object_id = (
                str(objects.get(object_role) or "unresolved")
                if isinstance(objects, dict)
                else "unresolved"
            )
            digest = hashlib.sha256(
                json.dumps(
                    [event_uid, object_role, state_before, state_after],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:20]
            records.append(
                {
                    "schema_version": PHYSICAL_CHANGE_SCHEMA_VERSION,
                    "change_uid": f"{event_uid}:change:{digest}",
                    "archive_id": archive_id,
                    "event_uid": event_uid,
                    "event_id": event_id,
                    "parent_event_id": parent_event_id,
                    "action_type": str(event.get("action_type") or "unknown"),
                    "object_role": str(object_role),
                    "object_id": object_id,
                    "state_before": state_before,
                    "state_after": state_after,
                    "start_us": int(event.get("start_us") or 0),
                    "end_us": int(event.get("end_us") or 0),
                    "peak_timestamp_us": int(event.get("peak_timestamp_us") or 0),
                    "decision_status": str(decision.get("status") or "unknown")
                    if isinstance(decision, dict)
                    else "unknown",
                    "evidence_ids": [str(item) for item in event.get("evidence_ids") or []],
                    "provenance": {
                        "source": "normalized_key_material_event",
                        "projection": "explicit_state_difference_only",
                        "new_inference_performed": False,
                    },
                }
            )
    records.sort(
        key=lambda item: (
            item["peak_timestamp_us"],
            item["event_uid"],
            item["object_role"],
        )
    )
    return records


def _known_state(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized.casefold() in {
        "unknown",
        "unobserved",
        "not_observed",
        "uncertain",
        "未知",
        "不确定",
    }:
        return None
    return normalized


__all__ = ["PHYSICAL_CHANGE_SCHEMA_VERSION", "physical_change_records"]
