"""Helpers for auditable operator changes on active relations."""

import json
import hashlib
import uuid
from datetime import datetime
from fastapi import HTTPException


def resolve_wearer(conn, camera_id, operator, wearer_id=None):
    """Resolve a declared camera member only when the identity is unique."""
    rows = conn.execute(
        """SELECT u.id FROM camera_users c JOIN users u ON u.id=c.user_id
                           WHERE c.camera_id=? AND u.display_name=? AND u.disabled=0""",
        (camera_id, operator),
    ).fetchall()
    if wearer_id:
        return wearer_id if any(row["id"] == wearer_id for row in rows) else None
    return rows[0]["id"] if len(rows) == 1 else None


def at_time(document, timestamp):
    """Project recorded personnel changes back to a photo's capture time."""
    moment = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
    result = dict(document)
    history = list(document.get("operator_history") or [])
    while history and datetime.fromisoformat(history[-1]["changed_at"]) > moment:
        change = history.pop()
        result.update(operator=change["previous_operator"], wearer_id=change["previous_wearer_id"])
        if "registered_at" in result:
            result["registered_at"] = change.get("previous_registered_at")
    if document.get("operator_history"):
        result["operator_history"] = history
        result["last_operator_change_at"] = history[-1]["changed_at"] if history else None
        result["last_operator_change_reason"] = history[-1]["reason"] if history else None
    return result


def changed_document(
    document, operator, wearer_id=None, timestamp=None, *, reason="same_day_operator_replacement"
):
    """Keep settings history for late photos; binding rows use update_relation."""
    operator = (operator or "").strip()
    if not operator:
        raise ValueError("operator is required")
    wearer_id = (wearer_id or "").strip() or None
    if document.get("operator") == operator and document.get("wearer_id") == wearer_id:
        return document, False
    history = list(document.get("operator_history") or [])
    history.append(
        {
            "previous_operator": document.get("operator"),
            "previous_wearer_id": document.get("wearer_id"),
            "previous_registered_at": document.get("registered_at"),
            "operator": operator,
            "wearer_id": wearer_id,
            "changed_at": timestamp,
            "reason": reason,
        }
    )
    return document | {
        "operator": operator,
        "wearer_id": wearer_id,
        "operator_history": history,
        "last_operator_change_at": timestamp,
        "last_operator_change_reason": reason,
    }, True


def update_relation(
    conn,
    table,
    row_id,
    operator,
    wearer_id=None,
    timestamp=None,
    *,
    reason="same_day_operator_replacement",
    changes=None,
    force=False,
):
    """Close [start,end) and open a successor; caller owns an IMMEDIATE transaction."""
    if table not in {"bindings", "scene_visits"}:
        raise ValueError("unsupported relation table")
    row = conn.execute(
        f"SELECT document FROM {table} WHERE id=? AND ended IS NULL", (str(row_id),)
    ).fetchone()
    if not row:
        return None
    previous = json.loads(row["document"])
    _, changed = changed_document(previous, operator, wearer_id, timestamp, reason=reason)
    if not changed and not force and all(previous.get(k) == v for k, v in (changes or {}).items()):
        return previous
    from field_recognition.bindings.binding_policy import contains

    if not contains(previous, timestamp):
        raise HTTPException(409, "使用时段已结束，请刷新或重新扫码")
    key = "binding_id" if table == "bindings" else "visit_id"
    next_id = str(uuid.uuid4())
    closed = previous | {"ended_at": timestamp, "end_reason": reason, "superseded_by": next_id}
    conn.execute(
        f"UPDATE {table} SET ended=?,document=? WHERE id=? AND ended IS NULL",
        (timestamp, json.dumps(closed, ensure_ascii=False), str(row_id)),
    )
    document = {
        k: v
        for k, v in previous.items()
        if k
        not in {
            "operator_history",
            "last_operator_change_at",
            "last_operator_change_reason",
            "end_reason",
            "superseded_by",
        }
    }
    document.update(changes or {})
    document.update(
        {
            key: next_id,
            "operator": operator.strip(),
            "wearer_id": wearer_id,
            "started_at": timestamp,
            "ended_at": None,
            "supersedes_binding_id" if table == "bindings" else "supersedes_visit_id": str(row_id),
            "session_revision": previous.get("session_revision", 1) + 1,
            "session_policy_version": "binding-session/2",
            "binding_action": "handover" if changed else "refresh" if force else "updated",
        }
    )
    conn.execute(
        f"INSERT INTO {table}(id,camera,ended,document) VALUES(?,?,NULL,?)",
        (next_id, document["camera_id"], json.dumps(document, ensure_ascii=False)),
    )
    audit = {
        "camera_id": document["camera_id"],
        "occurred_at": timestamp,
        "reason": reason,
        "relation_type": table,
        "previous_id": str(row_id),
        "next_id": next_id,
        "previous_operator": previous.get("operator"),
        "previous_wearer_id": previous.get("wearer_id"),
        "operator": operator.strip(),
        "wearer_id": wearer_id,
        "interval": "[started_at,ended_at)",
        "previous_revision": previous.get("session_revision", 1),
        "next_revision": document["session_revision"],
    }
    conn.execute(
        "INSERT INTO binding_session_audit VALUES(?,?,?,?,?,?,?)",
        (
            str(uuid.uuid4()),
            document["camera_id"],
            timestamp,
            table,
            str(row_id),
            next_id,
            json.dumps(audit, ensure_ascii=False),
        ),
    )
    return document


def update_camera_relations(
    conn,
    camera_id,
    operator,
    wearer_id=None,
    timestamp=None,
    *,
    reason="same_day_operator_replacement",
):
    """Update all active camera relations for camera-level operator settings."""
    from field_recognition.bindings.binding_policy import expire

    expire(conn, timestamp)
    updated = []
    rotated = False
    visit_ids = {}
    for table in ("scene_visits", "bindings"):
        for row in conn.execute(
            f"SELECT id FROM {table} WHERE camera=? AND ended IS NULL", (camera_id,)
        ).fetchall():
            original = json.loads(
                conn.execute(f"SELECT document FROM {table} WHERE id=?", (row["id"],)).fetchone()[0]
            )
            changes = (
                {"scene_visit_id": visit_ids[original["scene_visit_id"]]}
                if original.get("scene_visit_id") in visit_ids
                else None
            )
            document = update_relation(
                conn,
                table,
                row["id"],
                operator,
                wearer_id,
                timestamp,
                reason=reason,
                changes=changes,
            )
            if document:
                rotated = rotated or row["id"] != document.get(
                    "visit_id", document.get("binding_id")
                )
                if table == "scene_visits" and row["id"] != document["visit_id"]:
                    visit_ids[row["id"]] = document["visit_id"]
                updated.append(document)
    if rotated:
        bump_revision(conn, camera_id)
    return updated


def camera_revision(conn, camera_id):
    row = conn.execute(
        "SELECT revision FROM camera_session_state WHERE camera_id=?", (camera_id,)
    ).fetchone()
    return row[0] if row else 0


def check_revision(conn, camera_id, expected):
    if expected is not None and expected != camera_revision(conn, camera_id):
        raise HTTPException(
            409, {"code": "camera_session_conflict", "message": "使用人已被其他请求更改，请刷新"}
        )


def bump_revision(conn, camera_id):
    conn.execute(
        """INSERT INTO camera_session_state(camera_id,revision) VALUES(?,1)
                    ON CONFLICT(camera_id) DO UPDATE SET revision=revision+1""",
        (camera_id,),
    )
    return camera_revision(conn, camera_id)


def request_fingerprint(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def replay_request(conn, request_id, fingerprint):
    if not request_id:
        return None
    row = conn.execute(
        "SELECT fingerprint,document FROM binding_change_requests WHERE request_id=?",
        (str(request_id),),
    ).fetchone()
    if row:
        if row["fingerprint"] != fingerprint:
            raise HTTPException(409, "同一请求编号的内容或操作身份不能更改")
        return json.loads(row["document"])


def remember_request(conn, request_id, camera_id, fingerprint, document):
    if request_id:
        conn.execute(
            "INSERT INTO binding_change_requests VALUES(?,?,?,?)",
            (str(request_id), camera_id, fingerprint, json.dumps(document, ensure_ascii=False)),
        )
    return document


def activate_member(conn, camera_id, user_id, timestamp):
    """Switch background capture and current relations in the same transaction."""
    row = conn.execute(
        "SELECT id,display_name FROM users WHERE id=? AND disabled=0", (user_id,)
    ).fetchone()
    if not row:
        raise ValueError("Personnel registration does not exist or is disabled")
    previous_active = conn.execute(
        "SELECT user_id FROM camera_active_users WHERE camera_id=?", (camera_id,)
    ).fetchone()
    conn.execute(
        "INSERT OR IGNORE INTO camera_users(camera_id,user_id) VALUES(?,?)", (camera_id, user_id)
    )
    conn.execute(
        """INSERT INTO camera_active_users(camera_id,user_id) VALUES(?,?)
                    ON CONFLICT(camera_id) DO UPDATE SET user_id=excluded.user_id""",
        (camera_id, user_id),
    )
    reason = "camera_active_member_changed"
    update_camera_relations(conn, camera_id, row["display_name"], user_id, timestamp, reason=reason)
    if not previous_active or previous_active[0] != user_id:
        bump_revision(conn, camera_id)
    settings = conn.execute(
        "SELECT document FROM automation_settings WHERE camera=?", (camera_id,)
    ).fetchone()
    if settings:
        document, changed = changed_document(
            json.loads(settings[0]), row["display_name"], user_id, timestamp, reason=reason
        )
        if changed:
            document["registered_at"] = timestamp
            conn.execute(
                "UPDATE automation_settings SET document=? WHERE camera=?",
                (json.dumps(document, ensure_ascii=False), camera_id),
            )
