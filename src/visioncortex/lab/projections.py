from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from .domain import clean, ident


class Projections:
    def __init__(self, core):
        self.core = core

    def tick(self, limit=100):
        with self.core.db.transaction() as conn:
            conn.execute("INSERT INTO index_checkpoint(customer_id,cursor) VALUES(%s,0) ON CONFLICT DO NOTHING", (self.core.customer,))
            checkpoint = conn.execute("SELECT * FROM index_checkpoint WHERE customer_id=%s FOR UPDATE", (self.core.customer,)).fetchone()
            events = conn.execute("SELECT * FROM outbox WHERE customer_id=%s AND seq>%s ORDER BY seq LIMIT %s", (self.core.customer, checkpoint["cursor"], min(limit, 1000))).fetchall()
            for event in events:
                body = event["payload"]
                at = body.get("occurred_at")
                conn.execute("INSERT INTO index_document(customer_id,record_id,revision,kind,occurred_at,text_content,payload,outbox_seq) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(customer_id,record_id) DO UPDATE SET revision=EXCLUDED.revision,kind=EXCLUDED.kind,occurred_at=EXCLUDED.occurred_at,text_content=EXCLUDED.text_content,payload=EXCLUDED.payload,outbox_seq=EXCLUDED.outbox_seq WHERE index_document.outbox_seq<EXCLUDED.outbox_seq", (self.core.customer, event["record_id"], event["revision"], event["kind"], at, str(clean(body)), Jsonb(body), event["seq"]))
                if event["kind"] == "observation" and body.get("job_id"):
                    conn.execute("UPDATE processing_job SET index_status='indexed' WHERE customer_id=%s AND id=%s", (self.core.customer, ident(body["job_id"])))
            cursor = events[-1]["seq"] if events else checkpoint["cursor"]
            conn.execute("UPDATE index_checkpoint SET cursor=%s,checked_at=now() WHERE customer_id=%s", (cursor, self.core.customer))
            return dict(indexed=len(events), cursor=cursor)

    def status(self, conn=None):
        if conn is None:
            with self.core.db.transaction(readonly=True) as c:
                return self.status(c)
        high = conn.execute("SELECT COALESCE(MAX(seq),0) AS n FROM outbox WHERE customer_id=%s", (self.core.customer,)).fetchone()["n"]
        row = conn.execute("SELECT * FROM index_checkpoint WHERE customer_id=%s", (self.core.customer,)).fetchone()
        cursor = row["cursor"] if row else 0
        sources = conn.execute("SELECT id,name,coverage,checked_at FROM source_instance WHERE customer_id=%s", (self.core.customer,)).fetchall()
        reasons = []
        if cursor < high:
            coverage = "partial"
            reasons.append("index_behind_committed_outbox")
        elif not sources or any(s["coverage"] != "complete" or not s["checked_at"] or s["checked_at"] < datetime.now(timezone.utc) - timedelta(minutes=5) for s in sources):
            coverage = "unknown"
            reasons.append("capture_completeness_not_independently_established")
        else:
            coverage = "complete"
        return clean(dict(coverage=coverage, coverage_reasons=reasons, source_high_watermark=high, index_watermark=cursor, index_caught_up=cursor == high, last_checked=row["checked_at"] if row else None, sources=sources))

    def search(self, keyword="", *, kind=None, start=None, end=None, limit=50):
        with self.core.db.transaction(readonly=True) as conn:
            clauses, args = ["customer_id=%s"], [self.core.customer]
            if keyword:
                clauses.append("position(lower(%s) in lower(text_content))>0")
                args.append(keyword)
            if kind:
                clauses.append("kind=%s")
                args.append(kind)
            if start:
                clauses.append("occurred_at>=%s")
                args.append(start)
            if end:
                clauses.append("occurred_at<%s")
                args.append(end)
            args.append(min(max(limit, 1), 100))
            rows = conn.execute("SELECT record_id,revision,kind,occurred_at,payload FROM index_document WHERE " + " AND ".join(clauses) + " ORDER BY occurred_at DESC NULLS LAST,record_id LIMIT %s", args).fetchall()
            return clean(dict(items=rows, **self.status(conn)))

    def lineage(self, material_id, as_of=None):
        with self.core.db.transaction(readonly=True) as conn:
            self.core._entity(conn, material_id, "material_unit")
            # State versions and operation IDs remain explicit. Returning the
            # event graph avoids inventing a recursive depletion calculation.
            rows = conn.execute("WITH RECURSIVE ancestors(mid,path) AS (SELECT %s::uuid,ARRAY[]::uuid[] UNION ALL SELECT inp.material_id,a.path||o.id FROM ancestors a JOIN operation_line outp ON outp.customer_id=%s AND outp.material_id=a.mid AND outp.direction='output' JOIN operation o ON o.customer_id=outp.customer_id AND o.id=outp.operation_id JOIN operation_line inp ON inp.customer_id=o.customer_id AND inp.operation_id=o.id AND inp.direction='input' WHERE NOT o.id=ANY(a.path) AND cardinality(a.path)<64 AND (%s::timestamptz IS NULL OR o.occurred_at<=%s)) SELECT DISTINCT o.id,o.kind,o.occurred_at,o.revision,o.status,o.supersedes_id,o.metadata FROM operation o JOIN operation_line l ON l.customer_id=o.customer_id AND l.operation_id=o.id WHERE o.customer_id=%s AND l.material_id IN (SELECT mid FROM ancestors) AND (%s::timestamptz IS NULL OR o.occurred_at<=%s) ORDER BY o.occurred_at,o.id", (ident(material_id), self.core.customer, as_of, as_of, self.core.customer, as_of, as_of)).fetchall()
            ids = [r["id"] for r in rows]
            lines = conn.execute("SELECT * FROM operation_line WHERE customer_id=%s AND operation_id=ANY(%s)", (self.core.customer, ids)).fetchall()
            return clean(dict(material_id=material_id, operations=rows, lines=lines, as_of=as_of, depth_limit=64))
