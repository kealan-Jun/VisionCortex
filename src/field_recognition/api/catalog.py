"""Explicit runtime actions for catalog."""

from __future__ import annotations

import json
import uuid
from fastapi import HTTPException
from field_recognition.api.models import InstrumentEdit
from field_recognition.api.security import require_role, visible


def instrument_record(_runtime, row):
    record = dict(row)
    record["measurement_ranges"] = json.loads(record.get("measurement_ranges") or "{}")
    # `device_category` is deliberately independent from the measurement type:
    # a category describes the asset family while type_id describes its panel.
    record["device_category"] = (
        record.get("device_category") or "instrument"
    ).strip() or "instrument"
    type_id = record.get("type_id")
    record["device_type"] = (
        record.get("device_type") or _runtime["DEVICE_TYPE_NAMES"].get(type_id) or type_id or None
    )
    return record


def get_instrument(_runtime, instrument_id):
    with _runtime["db"]() as conn:
        row = conn.execute("SELECT * FROM instruments WHERE id=?", (instrument_id,)).fetchone()
    return _runtime["instrument_record"](row) if row else None


def devices(_runtime, category: str | None = None):
    """List QR-addressable devices without mixing them with capture cameras."""
    with _runtime["db"]() as conn:
        rows = conn.execute("SELECT * FROM instruments ORDER BY device_category,name").fetchall()
        active = {}
        for row in conn.execute("SELECT document FROM bindings WHERE ended IS NULL"):
            document = json.loads(row["document"])
            # The directory itself is global, but an operator must not learn
            # which camera currently holds a device outside their assignment.
            if visible(document):
                active[document["instrument"]["id"]] = document
        qr_rows = {
            row["device_id"]: row["qr_hash"]
            for row in conn.execute("SELECT device_id,qr_hash FROM qr_device_registry")
        }
    items = []
    for row in rows:
        device = _runtime["instrument_record"](row)
        if category and device["device_category"] != category:
            continue
        binding = active.get(device["id"])
        items.append(
            device
            | {
                "device_id": device["id"],
                "qr_registered": device["id"] in qr_rows,
                "qr_hash": qr_rows.get(device["id"]),
                "active_binding_id": binding.get("binding_id") if binding else None,
                "active_camera_id": binding.get("camera_id") if binding else None,
            }
        )
    categories = sorted({item["device_category"] for item in items})
    return {"items": items, "categories": categories}


def edit_instrument(_runtime, instrument_id: uuid.UUID, body: InstrumentEdit):
    require_role("admin")
    if not _runtime["get_instrument"](str(instrument_id)):
        raise HTTPException(404, "仪器不存在")
    if not body.name.strip() or not body.scene.strip():
        raise HTTPException(422, "请填写仪器名称和场景")
    with _runtime["db"]() as conn:
        category = body.device_category.strip() if body.device_category is not None else None
        if body.device_category is not None and not category:
            raise HTTPException(422, "设备分类不能为空")
        if category is None:
            conn.execute(
                "UPDATE instruments SET name=?,scene=?,model=? WHERE id=?",
                (body.name.strip(), body.scene.strip(), body.model.strip(), str(instrument_id)),
            )
        else:
            conn.execute(
                "UPDATE instruments SET name=?,scene=?,model=?,device_category=? WHERE id=?",
                (
                    body.name.strip(),
                    body.scene.strip(),
                    body.model.strip(),
                    category,
                    str(instrument_id),
                ),
            )
        if "device_no" in body.model_fields_set:
            conn.execute(
                "UPDATE instruments SET device_no=? WHERE id=?",
                ((body.device_no or "").strip() or None, str(instrument_id)),
            )
        if "measurement_ranges" in body.model_fields_set:
            conn.execute(
                "UPDATE instruments SET measurement_ranges=? WHERE id=?",
                (
                    json.dumps(
                        {
                            k: v.model_dump(exclude_none=True)
                            for k, v in (body.measurement_ranges or {}).items()
                        }
                    ),
                    str(instrument_id),
                ),
            )
    return _runtime["get_instrument"](str(instrument_id))
