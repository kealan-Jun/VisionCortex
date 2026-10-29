"""Shared instrument handoff transactions; preserve QR, actor and revision gates."""

import json
import uuid
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
from field_recognition.api.security import current, require_camera, require_role, actor_name
from field_recognition.bindings.instrument_ownership import (
    HandoffRequest,
    HandoffDecision,
    expire_requests,
)


class HandoffActions:
    def __init__(self, core):
        self.core = core

    def fetch(self, conn, hid):
        row = conn.execute(
            "SELECT document FROM binding_handoffs WHERE id=?", (str(hid),)
        ).fetchone()
        if not row:
            raise HTTPException(404, "交接请求不存在")
        return json.loads(row[0])

    def save(self, conn, item):
        item["revision"] += 1
        conn.execute(
            "UPDATE binding_handoffs SET status=?,document=? WHERE id=?",
            (item["status"], json.dumps(item), item["handoff_id"]),
        )
        return item

    def listing(self):
        with self.core["db"]() as conn:
            return {
                "items": [
                    doc
                    for r in conn.execute(
                        "SELECT document FROM binding_handoffs ORDER BY rowid DESC LIMIT 100"
                    )
                    for doc in [json.loads(r[0])]
                    if not current()
                    or current().role in {"admin", "reviewer"}
                    or {doc["source_camera_id"], doc["recipient_camera_id"]} & current().cameras
                ],
                "identity_assurance": "authenticated_account"
                if current()
                else "local_registered_operator_not_authenticated_account",
            }

    def request(self, body: HandoffRequest):
        require_role("admin", "operator")
        from field_recognition.bindings.binding_policy import sweep

        sweep(self.core)
        if current():
            body = body.model_copy(
                update={
                    "recipient_operator": actor_name(),
                    "recipient_wearer_id": current().user_id,
                }
            )
        with self.core["live_scanner"].lock, self.core["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expire_requests(conn)
            existing = conn.execute(
                "SELECT document FROM binding_handoffs WHERE id=?", (str(body.request_id),)
            ).fetchone()
            if existing:
                transfer = json.loads(existing[0])
                if transfer["request"] != body.model_dump(mode="json"):
                    raise HTTPException(409, "同一交接请求编号的内容不能更改")
                return transfer
            old = conn.execute(
                "SELECT document FROM bindings WHERE id=? AND ended IS NULL",
                (str(body.binding_id),),
            ).fetchone()
            scan = conn.execute(
                "SELECT document FROM scans WHERE id=?", (str(body.recipient_scan_id),)
            ).fetchone()
            if not old or not scan:
                raise HTTPException(409, "原绑定已结束或接收人的扫码证据不存在")
            old, scan = (json.loads(old[0]), json.loads(scan[0]))
            self.core["validate_capture_epoch"](conn, scan)
            if scan["camera_id"] == old["camera_id"] or not any(
                (q["id"] == old["instrument"]["id"] for q in scan["matches"])
            ):
                raise HTTPException(409, "接收相机必须重新扫到该仪器码")
            if conn.execute(
                "SELECT 1 FROM binding_handoffs WHERE instrument_id=? AND status IN ('requested','released')",
                (old["instrument"]["id"],),
            ).fetchone():
                raise HTTPException(409, "此仪器已有未完成的交接请求")
            transfer = {
                "handoff_id": str(body.request_id),
                "instrument_id": old["instrument"]["id"],
                "instrument": old["instrument"],
                "binding_id": old["binding_id"],
                "source_camera_id": old["camera_id"],
                "source_operator": old["operator"],
                "recipient_camera_id": scan["camera_id"],
                "camera_id": scan["camera_id"],
                "scan_id": scan["scan_id"],
                "image_url": scan["image_url"],
                "recipient_operator": body.recipient_operator,
                "request": body.model_dump(mode="json"),
                "status": "requested",
                "revision": 1,
                "created_at": self.core["now"](),
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
                "identity_assurance": "authenticated_account"
                if current()
                else "local_registered_operator_not_authenticated_account",
            }
            conn.execute(
                "INSERT INTO binding_handoffs(id,instrument_id,status,document) VALUES(?,?,?,?)",
                (
                    transfer["handoff_id"],
                    transfer["instrument_id"],
                    transfer["status"],
                    json.dumps(transfer),
                ),
            )
            return transfer

    def transition(self, hid: uuid.UUID, action: str, body: HandoffDecision):
        require_role("admin", "operator")
        require_camera(body.camera_id)
        body = body.model_copy(update={"actor": actor_name(body.actor)})
        if action not in {"release", "accept", "cancel", "reject"}:
            raise HTTPException(404, "交接动作不存在")
        with self.core["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            from field_recognition.bindings.binding_policy import expire

            expire(conn, self.core["now"]())
            expire_requests(conn)
        with self.core["live_scanner"].lock, self.core["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expire_requests(conn)
            transfer = self.fetch(conn, hid)
            expected_actor = (
                transfer["recipient_operator"]
                if action in {"accept", "reject"}
                else transfer["source_operator"]
            )
            expected_camera = (
                transfer["recipient_camera_id"]
                if action in {"accept", "reject"}
                else transfer["source_camera_id"]
            )
            if (body.actor, body.camera_id) != (expected_actor, expected_camera):
                raise HTTPException(409, "交出须由原使用人确认，接收须由指定接收人确认")
            receipt = body.model_dump()
            if transfer.get(action + "_decision") == receipt:
                return transfer
            if body.revision != transfer["revision"]:
                raise HTTPException(409, "交接状态已变化，请刷新")
            timestamp = self.core["now"]()
            if action == "release":
                if transfer["status"] != "requested":
                    raise HTTPException(409, "此交接不能再交出")
                row = conn.execute(
                    "SELECT document FROM bindings WHERE id=? AND ended IS NULL",
                    (transfer["binding_id"],),
                ).fetchone()
                if not row:
                    raise HTTPException(409, "原使用关系已结束，请取消交接并重新扫码")
                previous = json.loads(row[0]) | {
                    "ended_at": timestamp,
                    "end_reason": "explicit_handoff",
                    "handoff_id": str(hid),
                }
                conn.execute(
                    "UPDATE bindings SET ended=?,document=? WHERE id=?",
                    (timestamp, json.dumps(previous), transfer["binding_id"]),
                )
                transfer.update(status="released", released_at=timestamp)
            elif action == "accept":
                if transfer["status"] != "released":
                    raise HTTPException(409, "原使用人尚未确认交出")
                request = transfer["request"]
                binding = self.core["persist_binding"](
                    conn,
                    self.core["BindingRequest"](
                        scan_id=request["recipient_scan_id"],
                        instrument_id=transfer["instrument_id"],
                        operator=transfer["recipient_operator"],
                        wearer_id=request.get("recipient_wearer_id"),
                    ),
                    handoff_id=str(hid),
                )
                transfer.update(
                    status="completed", accepted_at=timestamp, new_binding_id=binding["binding_id"]
                )
            else:
                if transfer["status"] not in {"requested", "released"}:
                    raise HTTPException(409, "交接已经结束")
                transfer.update(
                    status="rejected" if action == "reject" else "cancelled", cancelled_at=timestamp
                )
            transfer[action + "_decision"] = receipt
            return self.save(conn, transfer)
