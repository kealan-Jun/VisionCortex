"""Explicit runtime actions for binding actions."""

from __future__ import annotations

import json
import uuid
from fastapi import HTTPException
from field_recognition.api.models import BindingRequest, EndRelations
from field_recognition.api.security import require_camera, actor_name, current


def register_qr_identity(_runtime, conn, qr_hash, device_id, timestamp):
    """Reserve the decoded QR identity for one *active* registered device.

    The decoded instrument UUID remains the canonical identity, while the hash
    keeps the exact label evidence. A conflicting active binding means that a
    physically different device was assigned the same QR label. Once the old
    binding is explicitly ended (including a confirmed device reboot), the
    registry row may be reassigned while the old binding evidence remains
    immutable in ``bindings``.
    """
    if not qr_hash:
        raise HTTPException(409, "扫码记录缺少二维码身份凭证")

    # The registry is durable history, not a permanent lock. The active
    # binding check is authoritative for exclusivity and runs in the same
    # IMMEDIATE transaction as the eventual binding insert.
    active_rows = conn.execute(
        "SELECT document FROM bindings WHERE ended IS NULL "
        "AND json_extract(document,'$.qr_hash')=?",
        (qr_hash,),
    ).fetchall()
    for active_row in active_rows:
        active = json.loads(active_row["document"])
        active_device = (active.get("instrument") or {}).get("id")
        if active_device == device_id:
            continue
        raise HTTPException(
            409,
            {
                "code": "qr_assigned_to_other_device",
                "message": "该二维码已经绑定到另一台设备，不能重复绑定；请先解绑或确认原设备已解除",
                "qr_hash": qr_hash,
                "device_id": active_device,
                "binding_id": active.get("binding_id"),
                "camera_id": active.get("camera_id"),
            },
        )

    row = conn.execute(
        "SELECT device_id FROM qr_device_registry WHERE qr_hash=?", (qr_hash,)
    ).fetchone()
    if row:
        # A prior ended binding (operator release, daily expiry, or confirmed
        # physical reboot) permits reassignment to a replacement device.
        conn.execute(
            "UPDATE qr_device_registry SET device_id=?,last_seen_at=? WHERE qr_hash=?",
            (device_id, timestamp, qr_hash),
        )
    else:
        conn.execute(
            "INSERT INTO qr_device_registry(qr_hash,device_id,first_seen_at,last_seen_at) VALUES(?,?,?,?)",
            (qr_hash, device_id, timestamp, timestamp),
        )


def validate_capture_epoch(_runtime, conn, scan):
    require_camera(scan["camera_id"])
    from field_recognition.bindings.binding_policy import moment, ZONE, expire

    timestamp = _runtime["now"]()
    expire(conn, timestamp)
    observed = scan.get("qr_observed_at") or scan["received_at"]
    if moment(observed).astimezone(ZONE).date() != moment(timestamp).astimezone(ZONE).date():
        raise HTTPException(409, "二维码凭证已跨日，请重新扫描当天的仪器码或场景码")
    service = conn.execute(
        "SELECT document FROM camera_service_state WHERE camera=?", (scan["camera_id"],)
    ).fetchone()
    if service and not json.loads(service["document"])["online"]:
        raise HTTPException(409, "设备采集服务已离线，重新启动后再扫码绑定")
    reset = conn.execute(
        "SELECT confirmed_at FROM camera_resets WHERE camera=?", (scan["camera_id"],)
    ).fetchone()
    if reset and moment(observed) <= moment(reset["confirmed_at"]):
        raise HTTPException(409, "这是设备采集服务上一次会话的照片，请重新扫码")


def bind(_runtime, body: BindingRequest, *, automatic: bool = False):
    with _runtime["live_scanner"].lock:
        result = _runtime["save_binding"](body, automatic=automatic)
    # Enqueue only after the binding transaction commits, without delaying its response.
    _runtime["queue_ocr_warmup"]()
    return result


def save_binding(_runtime, body: BindingRequest, *, automatic: bool = False):
    with _runtime["db"]() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return _runtime["persist_binding"](conn, body, automatic=automatic)


def persist_binding(_runtime, conn, body, *, automatic=False, handoff_id=None):
    row = conn.execute("SELECT document FROM scans WHERE id=?", (str(body.scan_id),)).fetchone()
    if not row:
        raise HTTPException(404, "扫码记录不存在")
    scan = json.loads(row["document"])
    from field_recognition.bindings.binding_operator import (
        request_fingerprint,
        replay_request,
        remember_request,
        check_revision,
        update_camera_relations,
        update_relation,
        activate_member,
    )

    require_camera(scan["camera_id"])
    fingerprint = request_fingerprint(
        {
            "operation": "bind",
            "body": body.model_dump(mode="json"),
            "principal": current().user_id if current() else None,
        }
    )
    replay = replay_request(conn, body.request_id, fingerprint)
    if replay is not None:
        return replay
    if body.action == "refresh" and body.request_id is None:
        raise HTTPException(422, "显式刷新需要 request_id 以保证重试幂等")
    _runtime["validate_capture_epoch"](conn, scan)
    check_revision(conn, scan["camera_id"], body.expected_revision)
    if not any(item["id"] == str(body.instrument_id) for item in scan["matches"]):
        raise HTTPException(409, "该图片没有识别到此仪器码")
    qr_hash = next(
        (item.get("qr_hash") for item in scan["matches"] if item["id"] == str(body.instrument_id)),
        None,
    )
    _runtime["register_qr_identity"](conn, qr_hash, str(body.instrument_id), _runtime["now"]())
    asset = _runtime["get_instrument"](str(body.instrument_id))
    if not asset["scene"]:
        raise HTTPException(409, "先在仪器登记中填写所属场景")
    operator = actor_name(body.operator).strip()
    if not operator:
        raise HTTPException(422, "实验员不能为空")
    owner_id = current().user_id if current() else body.wearer_id
    if _runtime["RUNTIME_ENABLED"]:
        from field_recognition.bindings.binding_operator import resolve_wearer

        resolved = resolve_wearer(conn, scan["camera_id"], operator, owner_id)
        if not resolved:
            raise HTTPException(422, "佩戴人编号与相机登记人员不一致")
        owner_id = resolved
    wearer_id = (owner_id or "").strip() or None
    receipt_id = body.request_id
    if receipt_id is None:
        # Legacy clients have no request UUID. Within the same decoded scan and
        # asset configuration, a handover to this person is one retryable intent.
        fingerprint = request_fingerprint(
            {
                "operation": "legacy_bind",
                "body": body.model_dump(mode="json"),
                "operator": operator,
                "wearer_id": wearer_id,
                "instrument": asset,
            }
        )
        receipt_id = "legacy-bind:" + fingerprint
        replay = replay_request(conn, receipt_id, fingerprint)
        if replay is not None:
            return replay
    visit = _runtime["optional_scene"](conn, scan["camera_id"], asset["scene"])

    timestamp = _runtime["now"]()
    existing_row = conn.execute(
        "SELECT id,document FROM bindings WHERE camera=? AND ended IS NULL "
        "AND json_extract(document,'$.instrument.id')=? LIMIT 1",
        (scan["camera_id"], str(body.instrument_id)),
    ).fetchone()
    if body.expected_binding_id and (
        not existing_row or existing_row["id"] != str(body.expected_binding_id)
    ):
        raise HTTPException(
            409, {"code": "binding_session_conflict", "message": "绑定时段已变化，请刷新"}
        )
    if automatic:
        # A scanner carrying the former operator must not undo a committed handover.
        if _runtime["RUNTIME_ENABLED"]:
            chosen = conn.execute(
                "SELECT user_id FROM camera_active_users WHERE camera_id=?", (scan["camera_id"],)
            ).fetchone()
            if chosen and chosen[0] != wearer_id:
                raise HTTPException(
                    409, {"code": "stale_operator_session", "message": "实际使用人已交接，请刷新"}
                )
        occupants = [
            json.loads(r[0])
            for r in conn.execute(
                "SELECT document FROM bindings WHERE camera=? AND ended IS NULL",
                (scan["camera_id"],),
            )
        ]
        if any((b.get("operator"), b.get("wearer_id")) != (operator, wearer_id) for b in occupants):
            raise HTTPException(
                409, {"code": "stale_operator_session", "message": "扫码人员会话已变化，请刷新"}
            )
    if _runtime["RUNTIME_ENABLED"] and wearer_id:
        activate_member(conn, scan["camera_id"], wearer_id, timestamp)
    else:
        update_camera_relations(conn, scan["camera_id"], operator, wearer_id, timestamp)
    # All of the camera's instrument sessions move together during personnel
    # handover. Instrument addition/refresh remains scoped to the decoded asset.
    visit = _runtime["optional_scene"](conn, scan["camera_id"], asset["scene"])
    current_row = conn.execute(
        "SELECT id,document FROM bindings WHERE camera=? AND ended IS NULL "
        "AND json_extract(document,'$.instrument.id')=? LIMIT 1",
        (scan["camera_id"], str(body.instrument_id)),
    ).fetchone()
    if current_row:
        prior = json.loads(current_row["document"])
        changes = {
            "instrument": asset,
            "qr_hash": qr_hash,
            "scene_visit_id": visit["visit_id"] if visit else None,
            "scene": visit["scene"] if visit else {"id": None, "name": asset["scene"]},
            "scene_qr_verified": visit is not None,
            "scene_basis": "decoded_scene_qr" if visit else "instrument_registration",
        }
        changed_content = any(prior.get(k) != v for k, v in changes.items())
        if changed_content or body.action == "refresh":
            changes.update(scan_id=scan["scan_id"], image_url=scan["image_url"])
            prior = update_relation(
                conn,
                "bindings",
                prior["binding_id"],
                operator,
                wearer_id,
                timestamp,
                reason="explicit_refresh"
                if body.action == "refresh"
                else "binding_content_updated",
                changes=changes,
                force=body.action == "refresh",
            )
        return remember_request(
            conn,
            receipt_id if body.request_id or prior.get("binding_action") == "handover" else None,
            scan["camera_id"],
            fingerprint,
            prior,
        )

    body = body.model_copy(update={"operator": operator, "wearer_id": wearer_id})
    from field_recognition.bindings.instrument_ownership import check_available

    timestamp = _runtime["now"]()
    check_available(
        conn, str(body.instrument_id), scan["camera_id"], qr_hash=qr_hash, handoff_id=handoff_id
    )
    result = {
        "binding_id": str(uuid.uuid4()),
        "operator": body.operator.strip(),
        "wearer_id": (body.wearer_id or "").strip() or None,
        "qr_hash": qr_hash,
        "binding_action": "created",
        "session_revision": 1,
        "session_policy_version": "binding-session/2",
        "camera_id": scan["camera_id"],
        "instrument": asset,
        "started_at": timestamp,
        "ended_at": None,
        "scan_id": scan["scan_id"],
        "image_url": scan["image_url"],
        "handoff_id": handoff_id,
        "identity_basis": "unsigned_qr_and_continuous_scan_opt_in"
        if automatic
        else "unsigned_qr_and_operator_confirmation",
        "activity_confirmed": False,
        "scene_visit_id": visit["visit_id"] if visit else None,
        "scene": visit["scene"] if visit else {"id": None, "name": asset["scene"]},
        "scene_qr_verified": visit is not None,
        "scene_basis": "decoded_scene_qr" if visit else "instrument_registration",
    }
    from field_recognition.bindings.binding_policy import fields

    result.update(fields(timestamp))
    service = conn.execute(
        "SELECT document FROM camera_service_state WHERE camera=?", (scan["camera_id"],)
    ).fetchone()
    result["device_service"] = json.loads(service["document"]) if service else None
    conn.execute(
        "INSERT INTO bindings VALUES(?,?,NULL,?)",
        (result["binding_id"], result["camera_id"], json.dumps(result)),
    )
    return remember_request(conn, body.request_id, scan["camera_id"], fingerprint, result)


def end_binding(_runtime, binding_id: uuid.UUID):
    with _runtime["live_scanner"].lock:
        return _runtime["finish_binding"](binding_id)


def finish_binding(_runtime, binding_id):
    with _runtime["db"]() as conn:
        row = conn.execute("SELECT * FROM bindings WHERE id=?", (str(binding_id),)).fetchone()
        if not row:
            raise HTTPException(404, "绑定不存在")
        result = json.loads(row["document"])
        require_camera(result["camera_id"])
        if not row["ended"]:
            if (
                _runtime["receiver_camera"]
                and result["camera_id"] == _runtime["receiver_camera"].target
            ):
                _runtime["automatic_runner"].pause("binding_ended")
            result["ended_at"] = _runtime["now"]()
            conn.execute(
                "UPDATE bindings SET ended=?,document=? WHERE id=?",
                (result["ended_at"], json.dumps(result), str(binding_id)),
            )
    return result


def end_relations(_runtime, body: EndRelations):
    require_camera(body.camera_id)
    with _runtime["live_scanner"].lock:
        if _runtime["receiver_camera"] and body.camera_id == _runtime["receiver_camera"].target:
            _runtime["automatic_runner"].pause("binding_ended")
        timestamp = _runtime["now"]()
        ended = []
        with _runtime["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for table in ("bindings", "scene_visits"):
                for row in conn.execute(
                    f"SELECT * FROM {table} WHERE camera=? AND ended IS NULL", (body.camera_id,)
                ).fetchall():
                    document = json.loads(row["document"]) | {
                        "ended_at": timestamp,
                        "end_reason": "operator_ended_session",
                    }
                    conn.execute(
                        f"UPDATE {table} SET ended=?,document=? WHERE id=?",
                        (timestamp, json.dumps(document), row["id"]),
                    )
                    ended.append(row["id"])
        return {"camera_id": body.camera_id, "ended_ids": ended, "ended_at": timestamp}
