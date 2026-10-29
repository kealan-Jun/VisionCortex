"""HTTP adapters for evidence-backed bindings actions."""

import json
import uuid
from field_recognition.api.security import visible
from field_recognition.bindings.instrument_ownership import HandoffRequest, HandoffDecision
from field_recognition.bindings.handoff_actions import HandoffActions


def install(core):
    actions = HandoffActions(core)
    with core["db"]() as conn:
        conn.executescript("""CREATE TABLE IF NOT EXISTS binding_handoffs(
            id TEXT PRIMARY KEY, instrument_id TEXT NOT NULL, status TEXT NOT NULL, document TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS pending_instrument_handoff ON binding_handoffs(instrument_id)
                WHERE status IN ('requested','released');""")
        # Triggers also protect competing inserts from other local clients. Existing
        # historical rows are preserved even if an older version allowed overlap.
        for action in ("INSERT", "UPDATE"):
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS exclusive_instrument_{action.lower()}
                BEFORE {action} ON bindings WHEN NEW.ended IS NULL AND EXISTS (
                    SELECT 1 FROM bindings WHERE ended IS NULL AND id<>NEW.id AND
                    json_extract(document,'$.instrument.id')=json_extract(NEW.document,'$.instrument.id'))
                BEGIN SELECT RAISE(ABORT,'Instrument already occupied'); END""")
        for action in ("INSERT", "UPDATE"):
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS exclusive_qr_{action.lower()}
                BEFORE {action} ON bindings WHEN NEW.ended IS NULL AND
                    json_extract(NEW.document,'$.qr_hash') IS NOT NULL AND EXISTS (
                    SELECT 1 FROM bindings WHERE ended IS NULL AND id<>NEW.id AND
                    json_extract(document,'$.qr_hash')=json_extract(NEW.document,'$.qr_hash'))
                BEGIN SELECT RAISE(ABORT,'QR already occupied'); END""")

    @core["app"].get("/api/handoffs")
    def listing():
        return actions.listing()

    @core["app"].get("/api/handoffs/eligible-scans")
    def scans():
        with core["db"]() as conn:
            rows = conn.execute(
                "SELECT document FROM scans WHERE json_array_length(document,'$.matches')>0 ORDER BY rowid DESC LIMIT 50"
            ).fetchall()
        return {
            "items": [
                {k: doc.get(k) for k in ("scan_id", "camera_id", "received_at", "matches")}
                for row in rows
                for doc in [json.loads(row[0])]
                if visible(doc)
            ]
        }

    @core["app"].post("/api/handoffs")
    def request(body: HandoffRequest):
        return actions.request(body)

    @core["app"].post("/api/handoffs/{hid}/{action}")
    def transition(hid: uuid.UUID, action: str, body: HandoffDecision):
        return actions.transition(hid, action, body)

    core["handoff_actions"] = {
        "request": actions.request,
        "transition": actions.transition,
        "listing": actions.listing,
    }
