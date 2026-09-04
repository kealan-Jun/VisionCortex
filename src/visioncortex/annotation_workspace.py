from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


WORKSPACE_SCHEMA_VERSION = "visioncortex-yolo-annotation-workspace/1.0.0"
DECISION_LEDGER_SCHEMA_VERSION = "visioncortex-yolo-annotation-decisions/1.0.0"
GT_EXPORT_SCHEMA_VERSION = "visioncortex-yolo-reviewed-ground-truth/1.0.0"
ALLOWED_DECISIONS = {
    "confirmed_false_negative",
    "false_positive",
    "class_error",
    "correct_detection",
    "not_actionable",
    "needs_review",
}


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_annotation_item_id(item: dict[str, Any]) -> str:
    identity = "|".join(
        str(item.get(key) or "")
        for key in (
            "event_id",
            "role",
            "class_name",
            "issue_type",
            "representative_phase_index",
            "image_path",
        )
    )
    return f"ann-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def workspace_paths(settings: dict[str, Any]) -> tuple[Path | None, Path]:
    storage = settings.get("storage") or {}
    runtime_root = Path(str(storage.get("local_runtime_root") or "./runtime"))
    configured = storage.get("yolo_annotation_queue_path")
    if configured:
        queue_path = Path(str(configured))
    else:
        audit_root = runtime_root / "Model-Audits"
        candidates = list(audit_root.glob("*/YOLO-Annotation-Queue.json")) if audit_root.is_dir() else []
        queue_path = max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None
    ledger_path = runtime_root / "Annotation-Workspace" / "YOLO-Annotation-Decisions.json"
    return queue_path, ledger_path


def _load(settings: dict[str, Any]) -> tuple[Path | None, dict[str, Any], Path, dict[str, Any]]:
    queue_path, ledger_path = workspace_paths(settings)
    queue = _read_json(queue_path, {}) if queue_path else {}
    if not isinstance(queue, dict):
        raise ValueError("YOLO annotation queue root must be an object")
    ledger = _read_json(
        ledger_path,
        {
            "schema_version": DECISION_LEDGER_SCHEMA_VERSION,
            "source_queue": str(queue_path) if queue_path else None,
            "source_queue_sha256": _sha256(queue_path) if queue_path else None,
            "updated_at": None,
            "decisions": {},
            "history": [],
        },
    )
    if not isinstance(ledger, dict):
        raise ValueError("YOLO annotation decision ledger root must be an object")
    ledger.setdefault("decisions", {})
    ledger.setdefault("history", [])
    return queue_path, queue, ledger_path, ledger


def load_annotation_workspace(
    settings: dict[str, Any],
    *,
    priority: str | None = None,
    review_status: str | None = None,
    query: str | None = None,
    offset: int = 0,
    limit: int = 24,
) -> dict[str, Any]:
    queue_path, queue, ledger_path, ledger = _load(settings)
    decisions = ledger.get("decisions") or {}
    items: list[dict[str, Any]] = []
    for source_item in queue.get("items") or []:
        item = dict(source_item)
        item_id = stable_annotation_item_id(item)
        decision = decisions.get(item_id)
        item.update(
            {
                "item_id": item_id,
                "image_available": Path(str(item.get("image_path") or "")).is_file(),
                "decision": decision,
                "effective_review_status": (
                    "reviewed" if decision and decision.get("decision") != "needs_review" else "pending"
                ),
            }
        )
        items.append(item)
    normalized_query = str(query or "").strip().casefold()
    if priority:
        items = [item for item in items if str(item.get("priority")) == priority]
    if review_status:
        items = [item for item in items if item["effective_review_status"] == review_status]
    if normalized_query:
        items = [
            item
            for item in items
            if normalized_query
            in " ".join(
                str(item.get(key) or "")
                for key in ("event_id", "class_name", "action_type", "issue_type", "role")
            ).casefold()
        ]
    total = len(items)
    selected = items[max(0, offset) : max(0, offset) + max(1, min(limit, 200))]
    all_items = queue.get("items") or []
    reviewed = sum(stable_annotation_item_id(item) in decisions for item in all_items)
    decision_counts: dict[str, int] = {}
    for decision in decisions.values():
        name = str(decision.get("decision") or "unknown")
        decision_counts[name] = decision_counts.get(name, 0) + 1
    return {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "available": queue_path is not None and queue_path.is_file(),
        "queue_path": str(queue_path) if queue_path else None,
        "queue_schema_version": queue.get("schema_version"),
        "decision_ledger_path": str(ledger_path),
        "summary": {
            "total": len(all_items),
            "reviewed": reviewed,
            "pending": max(0, len(all_items) - reviewed),
            "priority_counts": queue.get("priority_counts") or {},
            "decision_counts": decision_counts,
        },
        "filters": {
            "priority": priority,
            "review_status": review_status,
            "query": query,
            "offset": max(0, offset),
            "limit": max(1, min(limit, 200)),
            "matched": total,
        },
        "items": selected,
    }


def resolve_annotation_image(settings: dict[str, Any], item_id: str) -> Path:
    _, queue, _, _ = _load(settings)
    match = next(
        (
            item
            for item in queue.get("items") or []
            if stable_annotation_item_id(item) == item_id
        ),
        None,
    )
    if not match:
        raise KeyError(item_id)
    path = Path(str(match.get("image_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def record_annotation_decision(
    settings: dict[str, Any], item_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    queue_path, queue, ledger_path, ledger = _load(settings)
    source_item = next(
        (item for item in queue.get("items") or [] if stable_annotation_item_id(item) == item_id),
        None,
    )
    if not source_item:
        raise KeyError(item_id)
    decision = str(payload.get("decision") or "").strip()
    reviewer = str(payload.get("reviewer") or "").strip()
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported decision: {decision}")
    if not reviewer:
        raise ValueError("reviewer is required")
    xyxy = payload.get("xyxy")
    if xyxy in (None, ""):
        normalized_xyxy = None
    else:
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            raise ValueError("xyxy must contain four numbers")
        normalized_xyxy = [float(value) for value in xyxy]
        if normalized_xyxy[2] <= normalized_xyxy[0] or normalized_xyxy[3] <= normalized_xyxy[1]:
            raise ValueError("xyxy must define a positive-area box")
    now = datetime.now().astimezone().isoformat()
    record = {
        "item_id": item_id,
        "event_id": source_item.get("event_id"),
        "source_class_name": source_item.get("class_name"),
        "source_issue_type": source_item.get("issue_type"),
        "decision": decision,
        "corrected_class": str(payload.get("corrected_class") or "").strip() or None,
        "xyxy": normalized_xyxy,
        "visibility": payload.get("visibility"),
        "occlusion": payload.get("occlusion"),
        "notes": str(payload.get("notes") or "").strip(),
        "reviewer": reviewer,
        "reviewed_at": now,
        "image_path": source_item.get("image_path"),
        "image_sha256": _sha256(Path(str(source_item.get("image_path") or ""))),
    }
    ledger.update(
        {
            "schema_version": DECISION_LEDGER_SCHEMA_VERSION,
            "source_queue": str(queue_path) if queue_path else None,
            "source_queue_sha256": _sha256(queue_path) if queue_path else None,
            "updated_at": now,
        }
    )
    ledger["decisions"][item_id] = record
    ledger["history"].append(record)
    _write_json_atomic(ledger_path, ledger)
    return record


def export_reviewed_ground_truth(settings: dict[str, Any]) -> dict[str, Any]:
    queue_path, _, _, ledger = _load(settings)
    annotations = []
    excluded = []
    for item_id, decision in sorted((ledger.get("decisions") or {}).items()):
        if decision.get("xyxy") and (
            decision.get("corrected_class") or decision.get("source_class_name")
        ):
            annotations.append(
                {
                    "annotation_id": item_id,
                    "event_id": decision.get("event_id"),
                    "image_path": decision.get("image_path"),
                    "image_sha256": decision.get("image_sha256"),
                    "class_name": decision.get("corrected_class")
                    or decision.get("source_class_name"),
                    "xyxy": decision.get("xyxy"),
                    "visibility": decision.get("visibility"),
                    "occlusion": decision.get("occlusion"),
                    "reviewer": decision.get("reviewer"),
                    "reviewed_at": decision.get("reviewed_at"),
                }
            )
        else:
            excluded.append(
                {"item_id": item_id, "reason": "missing_reviewed_box_or_class"}
            )
    return {
        "schema_version": GT_EXPORT_SCHEMA_VERSION,
        "source_queue": str(queue_path) if queue_path else None,
        "source_decision_ledger": str(workspace_paths(settings)[1]),
        "annotation_count": len(annotations),
        "excluded_count": len(excluded),
        "annotations": annotations,
        "excluded": excluded,
        "policy": "Only reviewer-supplied boxes are exported; VisionCortex never fabricates ground-truth boxes.",
    }
