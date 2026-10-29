"""Exclusive instrument occupancy and an explicit request/release/accept handoff."""

import json
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


class HandoffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: uuid.UUID
    binding_id: uuid.UUID
    recipient_scan_id: uuid.UUID
    recipient_operator: str = Field(min_length=1, max_length=80)
    recipient_wearer_id: str | None = Field(default=None, max_length=100)


class HandoffDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    actor: str = Field(min_length=1, max_length=80)
    camera_id: str = Field(min_length=1, max_length=100)
    revision: int = Field(ge=1)


def expire_requests(conn):
    now = datetime.now(timezone.utc)
    for row in conn.execute(
        "SELECT document FROM binding_handoffs WHERE status IN ('requested','released')"
    ):
        item = json.loads(row[0])
        expires = item.get("expires_at")
        if expires and datetime.fromisoformat(expires) <= now:
            item.update(status="expired", expired_at=now.isoformat(), revision=item["revision"] + 1)
            conn.execute(
                "UPDATE binding_handoffs SET status=?,document=? WHERE id=?",
                ("expired", json.dumps(item), item["handoff_id"]),
            )


def check_available(conn, instrument_id, camera_id, *, qr_hash=None, handoff_id=None):
    expire_requests(conn)
    for row in conn.execute(
        "SELECT document FROM bindings WHERE ended IS NULL AND json_extract(document,'$.instrument.id')=?",
        (instrument_id,),
    ):
        active = json.loads(row[0])
        if active["camera_id"] != camera_id:
            raise HTTPException(
                409,
                {
                    "code": "instrument_in_use",
                    "message": "仪器仍被另一台相机占用；该相机关机、采集服务异常或北京时间跨日后自动解除，再扫码即可绑定",
                    "binding_id": active["binding_id"],
                    "camera_id": active["camera_id"],
                    "operator": active["operator"],
                    "instrument_id": instrument_id,
                    "started_at": active["started_at"],
                    "handoff_required": False,
                    "release_policy": "camera-service-session/1",
                },
            )
    if qr_hash:
        for row in conn.execute(
            "SELECT document FROM bindings WHERE ended IS NULL AND json_extract(document,'$.qr_hash')=?",
            (qr_hash,),
        ):
            active = json.loads(row[0])
            if active["camera_id"] != camera_id:
                raise HTTPException(
                    409,
                    {
                        "code": "qr_in_use",
                        "message": "该二维码已绑定到另一台设备，不能重复绑定",
                        "binding_id": active["binding_id"],
                        "camera_id": active["camera_id"],
                        "device_id": active["instrument"]["id"],
                        "qr_hash": qr_hash,
                        "handoff_required": False,
                        "release_policy": "camera-service-session/1",
                    },
                )
    for row in conn.execute(
        "SELECT document FROM binding_handoffs WHERE instrument_id=? AND status='released'",
        (instrument_id,),
    ):
        transfer = json.loads(row[0])
        if transfer["handoff_id"] != handoff_id:
            raise HTTPException(
                409,
                {
                    "code": "handoff_reserved",
                    "message": "仪器已交出，等待指定接收人确认",
                    "handoff_id": transfer["handoff_id"],
                    "recipient_camera_id": transfer["recipient_camera_id"],
                },
            )


def install(core):
    """Compatibility installer; business actions are independent of HTTP adapters."""
    from field_recognition.api.bindings_routes import install as install_routes

    return install_routes(core)
