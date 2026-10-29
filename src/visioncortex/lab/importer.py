"""Explicit source-to-customer mapping and resumable receipt import, never inference."""
from __future__ import annotations

from psycopg.types.json import Jsonb

from .domain import Conflict, fingerprint
from .ingest import Ingest
from .processing import insert_observation


class ReceiptImporter:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings

    def page(self, page, mapping, *, dry_run=True):
        if str(self.core.customer) != mapping.get("customer_id") or page.get("source_instance") != mapping.get("source_instance"):
            raise Conflict("Explicit source-to-customer identity mapping is required")
        if page.get("schema") != "field-event-feed/1" or page.get("camera_id") != mapping.get("partition"):
            raise Conflict("Feed partition/schema mismatch")
        if page.get("tenant") is None and not mapping.get("allow_unscoped", False):
            raise Conflict("Unscoped history is not automatically assigned to this customer")
        if page.get("tenant") != mapping.get("source_tenant"):
            raise Conflict("Feed tenant mismatch")
        after, next_cursor, high = (page.get(k) for k in ("after_seq", "next_cursor", "high_watermark"))
        if any(type(n) is not int or n < 0 for n in (after, next_cursor, high)) or not after <= next_cursor <= high:
            raise Conflict("Source cursor is invalid/reset; reconciliation is required")
        items = page.get("items", [])
        sequences = [i.get("sequence") for i in items]
        if any(type(n) is not int or not after < n <= next_cursor for n in sequences) or sequences != sorted(set(sequences)):
            raise Conflict("Source delivery sequence is invalid")
        if any(i.get("source_instance") != mapping["source_instance"] or not i.get("event_id") or i.get("schema_version") != "field-event/1" for i in items):
            raise Conflict("Event identity/schema mismatch")
        summary = dict(dry_run=dry_run, source_count=len(items), accepted=len(items), rejected=0, conflicts=0, duplicates=0, formal_records_created=0, cursor=next_cursor, source_high_watermark=high, coverage="unknown")
        if dry_run:
            return summary
        with self.core.db.transaction() as conn:
            source = Ingest(self.core, self.settings).source(conn, "field-feed:" + mapping["source_instance"], "field-event/1", "receipt_import")
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (f"feed:{self.core.customer}:{source['id']}:{mapping['partition']}",))
            checkpoint = conn.execute("SELECT * FROM consumer_checkpoint WHERE customer_id=%s AND source_id=%s AND partition=%s FOR UPDATE", (self.core.customer, source["id"], mapping["partition"])).fetchone()
            current = checkpoint["cursor"] if checkpoint else mapping.get("initial_cursor", 0)
            if after > current or high < current:
                raise Conflict("Source feed gap/reset requires reconciliation")
            for event in items:
                revision = str(event.get("revision", event["sequence"]))
                old = conn.execute("SELECT fingerprint FROM inbox WHERE customer_id=%s AND source_id=%s AND event_id=%s AND revision=%s", (self.core.customer, source["id"], event["event_id"], revision)).fetchone()
                if old:
                    if old["fingerprint"] != fingerprint(event):
                        raise Conflict("Immutable source revision changed")
                    summary["duplicates"] += 1
                    continue
                conn.execute("INSERT INTO inbox(customer_id,source_id,event_id,revision,fingerprint,payload) VALUES(%s,%s,%s,%s,%s,%s)", (self.core.customer, source["id"], event["event_id"], revision, fingerprint(event), Jsonb(event)))
                # Import preserves draft/unbound/raw values. It never executes a
                # confirmation command or invents verified local media objects.
                payload = dict(source_event=event, analysis_outcome="imported_observation", evidence_status="unverified", normalized_fields=[], physical_action_verified=False)
                insert_observation(self.core, conn, event_id=str(source["id"]) + ":" + event["event_id"], revision=revision, kind="source_receipt", payload=payload, source_kind=mapping.get("source_kind", "derived_replay"), occurred_at=event.get("occurred_start"))
            cursor = max(current, next_cursor)
            conn.execute("INSERT INTO consumer_checkpoint(customer_id,source_id,partition,cursor,status) VALUES(%s,%s,%s,%s,'consumed') ON CONFLICT(customer_id,source_id,partition) DO UPDATE SET cursor=EXCLUDED.cursor,status=EXCLUDED.status,checked_at=now()", (self.core.customer, source["id"], mapping["partition"], cursor))
            summary["cursor"] = cursor
            summary["accepted"] -= summary["duplicates"]
            return summary
