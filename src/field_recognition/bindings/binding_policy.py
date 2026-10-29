"""Daily QR authorization in Beijing time; source-time history stays immutable."""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("Asia/Shanghai")
VERSION = "daily-qr-binding/1"
RECONNECT_GRACE_SECONDS = 30


def _end_camera_session(conn, camera, reason, at, observation):
    """Release ownership and stale receipts in the caller's transaction."""
    for table in ("bindings", "scene_visits"):
        for record in conn.execute(
            f"SELECT id,document FROM {table} WHERE camera=? AND ended IS NULL", (camera,)
        ).fetchall():
            doc = json.loads(record["document"]) | {
                "ended_at": at,
                "end_reason": reason,
                "service_observation": observation,
                "session_lifecycle_policy": "camera-service-session/2",
            }
            doc.pop("reconnect_deadline", None)
            doc["connection_status"] = "ended"
            conn.execute(
                f"UPDATE {table} SET ended=?,document=? WHERE id=?",
                (at, json.dumps(doc), record["id"]),
            )
    conn.execute("INSERT OR REPLACE INTO camera_resets VALUES(?,?)", (camera, at))
    for row in conn.execute(
        "SELECT id,document FROM binding_handoffs WHERE status IN ('requested','released')"
    ).fetchall():
        doc = json.loads(row["document"])
        if camera in (doc["source_camera_id"], doc["recipient_camera_id"]):
            doc.update(
                status="expired", expired_at=at, end_reason=reason, revision=doc["revision"] + 1
            )
            conn.execute(
                "UPDATE binding_handoffs SET status=?,document=? WHERE id=?",
                ("expired", json.dumps(doc), row["id"]),
            )


def _connection_interval(conn, camera, pending, at, *, recovered=False, observation=None):
    for table in ("bindings", "scene_visits"):
        for row in conn.execute(
            f"SELECT id,document FROM {table} WHERE camera=? AND ended IS NULL", (camera,)
        ).fetchall():
            doc = json.loads(row["document"])
            intervals = doc.setdefault("service_interruptions", [])
            if recovered:
                if intervals and not intervals[-1].get("ended_at"):
                    intervals[-1].update(
                        ended_at=at,
                        recovered_media_session_id=observation.get("media_session_id"),
                        continuity_basis="same_camera_within_reconnect_window",
                    )
                doc.pop("reconnect_deadline", None)
            else:
                intervals.append(dict(pending, ended_at=None))
                doc["reconnect_deadline"] = pending["deadline"]
            doc["connection_status"] = "connected" if recovered else "recovering"
            doc["session_lifecycle_policy"] = "camera-service-session/2"
            conn.execute(f"UPDATE {table} SET document=? WHERE id=?", (json.dumps(doc), row["id"]))
    # Invalidate old scans on both edges; resumption retains only existing relations.
    conn.execute("INSERT OR REPLACE INTO camera_resets VALUES(?,?)", (camera, at))


def _expire_reconnections(conn, at):
    # Older registry-only databases may reach daily maintenance before the
    # acquisition tables are initialized. There cannot be pending windows yet.
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='camera_service_state'"
    ).fetchone():
        return
    for row in conn.execute("SELECT camera,document FROM camera_service_state").fetchall():
        state = json.loads(row["document"])
        pending = state.get("reconnect_pending")
        if not pending or moment(pending["deadline"]) > moment(at):
            continue
        _end_camera_session(conn, row["camera"], "camera_reconnect_timeout", at, state)
        state.pop("reconnect_pending")
        state.update(connection_status="ended", reconnect_timed_out_at=at, online=False)
        conn.execute(
            "UPDATE camera_service_state SET document=? WHERE camera=?",
            (json.dumps(state), row["camera"]),
        )


def observe_camera_service(conn, camera, observation, at):
    """Reserve brief transport outages; release expired or definitively ended sessions."""
    expire(conn, at)
    row = conn.execute(
        "SELECT document FROM camera_service_state WHERE camera=?", (camera,)
    ).fetchone()
    previous = json.loads(row[0]) if row else {}
    observed_at = observation.get("observed_at") or at
    if (
        observation.get("observed_at")
        and previous.get("observed_at")
        and moment(observed_at) <= moment(previous["observed_at"])
    ):
        return None  # A delayed snapshot cannot undo a newer service observation.
    online = observation.get("online") is True
    old_id, new_id = previous.get("media_session_id"), observation.get("media_session_id")
    changed = old_id not in (None, 0) and new_id not in (None, 0) and old_id != new_id
    old_boot, new_boot = previous.get("device_boot_id"), observation.get("device_boot_id")
    rebooted = bool(old_boot and new_boot and old_boot != new_boot)
    active = any(
        conn.execute(
            f"SELECT 1 FROM {table} WHERE camera=? AND ended IS NULL LIMIT 1", (camera,)
        ).fetchone()
        for table in ("bindings", "scene_visits")
    )
    pending = previous.get("reconnect_pending")
    recoverable = observation.get("reconnectable") is True
    resumed = False
    reason = None
    if rebooted:
        reason = "device_boot_changed"
    elif pending and (recoverable or online):
        if online and observation.get("frame_verified") is True:
            _connection_interval(conn, camera, pending, at, recovered=True, observation=observation)
            pending = None
            resumed = True
        else:
            online = False  # A heartbeat alone cannot resume acquisition.
    elif recoverable and not online and previous.get("online") and active:
        pending = {
            "started_at": at,
            "deadline": (moment(at) + timedelta(seconds=RECONNECT_GRACE_SECONDS)).isoformat(),
            "reason": observation.get("failure_reason"),
            "previous_media_session_id": old_id,
        }
        _connection_interval(conn, camera, pending, at)
    elif not online and (not previous or previous.get("online") or active):
        reason = observation.get("failure_reason") or "camera_service_offline"
    elif online:
        if changed:
            reason = "camera_session_changed"
        elif previous and not previous.get("online"):
            reason = "camera_service_recovered"
    if reason:
        _end_camera_session(conn, camera, reason, at, observation)
        pending = None
    stored = observation | {
        "online": online,
        "media_session_id": new_id or old_id,
        "observed_at": observed_at,
        "session_lifecycle_policy": "camera-service-session/2",
    }
    if new_boot or old_boot:
        stored["device_boot_id"] = new_boot or old_boot
    if not online:
        stored["offline_since"] = previous.get("offline_since") or at
    if pending:
        stored.update(reconnect_pending=pending, connection_status="recovering")
    elif resumed:
        stored.update(connection_status="connected", reconnected_at=at)
    if changed:
        stored["last_transport_change"] = {
            "observed_at": at,
            "previous_id": old_id,
            "current_id": new_id,
            "binding_action": "waiting_reconnect" if pending else "ended_service_session",
        }
    elif previous.get("last_transport_change"):
        stored["last_transport_change"] = previous["last_transport_change"]
    if resumed and stored.get("last_transport_change"):
        stored["last_transport_change"] = dict(
            stored["last_transport_change"], binding_action="resumed_service_session"
        )
    conn.execute(
        "INSERT OR REPLACE INTO camera_service_state VALUES(?,?)", (camera, json.dumps(stored))
    )
    return reason


def moment(value):
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if result.tzinfo is None:
        raise ValueError("Binding time must include a timezone")
    return result


def next_midnight(started_at):
    local = moment(started_at).astimezone(ZONE)
    return (
        local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    ).astimezone(timezone.utc)


def fields(started_at):
    return {
        "binding_policy_version": VERSION,
        "binding_date": moment(started_at).astimezone(ZONE).date().isoformat(),
        "binding_timezone": "Asia/Shanghai",
        "valid_until": next_midnight(started_at).isoformat(),
    }


def interruption_at(document, at):
    """A reserved binding is not capture-time authorization during transport loss."""
    at = moment(at)
    if any(
        at >= moment(end) for end in (document.get("ended_at"), document.get("valid_until")) if end
    ):
        return None
    for interval in document.get("service_interruptions", []):
        if moment(interval["started_at"]) <= at and (
            not interval.get("ended_at") or at < moment(interval["ended_at"])
        ):
            return interval
    return None


def contains(document, at):
    at = moment(at)
    if moment(document["started_at"]) > at or interruption_at(document, at):
        return False
    # Closed legacy records retain their historical interval. New/current daily
    # relationships carry an explicit expiry even if maintenance has not run yet.
    until = document.get("valid_until")
    if not until and not document.get("ended_at"):
        until = next_midnight(document["started_at"]).isoformat()
    return all(at < moment(end) for end in (document.get("ended_at"), until) if end)


def expire(conn, at):
    at = moment(at)
    expired = []
    for table in ("bindings", "scene_visits"):
        for row in conn.execute(f"SELECT id,document FROM {table} WHERE ended IS NULL").fetchall():
            doc = json.loads(row["document"])
            deadline = moment(
                doc.get("valid_until") or next_midnight(doc["started_at"]).isoformat()
            )
            if deadline > at:
                continue
            doc.update(
                fields(doc["started_at"]),
                ended_at=deadline.isoformat(),
                end_reason="daily_qr_expired",
                expiry_observed_at=at.isoformat(),
            )
            if doc.get("connection_status"):
                doc["connection_status"] = "ended"
                doc.pop("reconnect_deadline", None)
            changed = conn.execute(
                f"UPDATE {table} SET ended=?,document=? WHERE id=? AND ended IS NULL",
                (doc["ended_at"], json.dumps(doc), row["id"]),
            ).rowcount
            if changed:
                expired.append(row["id"])
    _expire_reconnections(conn, at.isoformat())
    # A pending cross-day handoff cannot authorize use from yesterday's scan.
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='binding_handoffs'"
    ).fetchone()
    if exists:
        for row in conn.execute(
            "SELECT id,document FROM binding_handoffs WHERE status IN ('requested','released')"
        ).fetchall():
            doc = json.loads(row["document"])
            source = conn.execute(
                "SELECT document FROM bindings WHERE id=?", (doc["binding_id"],)
            ).fetchone()
            if source and next_midnight(json.loads(source[0])["started_at"]) <= at:
                doc.update(
                    status="expired",
                    expired_at=at.isoformat(),
                    end_reason="daily_qr_expired",
                    revision=doc["revision"] + 1,
                )
                conn.execute(
                    "UPDATE binding_handoffs SET status=?,document=? WHERE id=?",
                    ("expired", json.dumps(doc), row["id"]),
                )
    return expired


def sweep(core):
    with core["db"]() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return expire(conn, core["now"]())
