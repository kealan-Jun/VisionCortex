from __future__ import annotations

from psycopg.types.json import Jsonb

from .domain import Missing, clean, ident


class Fields:
    def __init__(self, core):
        self.core = core

    def template(self, key, name, version, fields):
        def action(conn):
            if not fields or any(not f.get("field_key") or not f.get("accept_scopes") for f in fields):
                raise ValueError("Template fields must declare accepted fact scopes")
            row = conn.execute("INSERT INTO form_template_version(customer_id,id,name,version,fields) VALUES(%s,%s,%s,%s,%s) RETURNING *", (self.core.customer, ident(), name, version, Jsonb(fields))).fetchone()
            return row
        return self.core.command(key, dict(name=name, version=version, fields=fields), action)

    def snapshot(self, key, template_id, fact_ids):
        def action(conn):
            template = conn.execute("SELECT * FROM form_template_version WHERE customer_id=%s AND id=%s", (self.core.customer, ident(template_id))).fetchone()
            if not template:
                raise Missing("Template outside customer scope")
            ids = [ident(f) for f in fact_ids]
            facts = conn.execute("SELECT f.*,c.field_key,o.evidence_ids FROM fact_revision f JOIN field_candidate c ON c.customer_id=f.customer_id AND c.id=f.candidate_id JOIN observation o ON o.customer_id=c.customer_id AND o.id=c.observation_id WHERE f.customer_id=%s AND f.id=ANY(%s)", (self.core.customer, ids)).fetchall()
            if len(facts) != len(set(ids)):
                raise Missing("Fact outside customer scope")
            fields, missing, conflicts, evidence, inputs = [], [], [], [], {}
            for spec in template["fields"]:
                matches = [f for f in facts if f["field_key"] == spec["field_key"] and f["scope"] in spec["accept_scopes"] and f["status"] == "confirmed"]
                if not matches:
                    missing.append(dict(field_key=spec["field_key"], reason="no_fact_with_accepted_semantics"))
                elif len(matches) > 1:
                    conflicts.append(dict(field_key=spec["field_key"], reason="multiple_facts_require_selection"))
                else:
                    f = matches[0]
                    fields.append(dict(field_key=spec["field_key"], target_field_id=spec.get("target_field_id"), value=f["value"], unit=f["unit"], value_semantics=f["scope"], record_id=f["id"], revision=f["revision"], schema_version="vc-lab-field/1", source_kind="confirmed_observation", evidence_refs=f["evidence_ids"]))
                    evidence.extend(f["evidence_ids"])
                    inputs[str(f["id"])] = f["revision"]
            result = dict(fields=fields, missing_fields=missing, conflicts=conflicts, evidence_refs=list(set(evidence)), template_id=template_id, template_version=template["version"])
            sid = ident()
            conn.execute("INSERT INTO field_snapshot(customer_id,id,template_id,input_revisions,fields) VALUES(%s,%s,%s,%s,%s)", (self.core.customer, sid, ident(template_id), Jsonb(inputs), Jsonb(clean(result))))
            return dict(result, snapshot_id=sid, input_snapshot_revision=inputs, stale=False)
        return self.core.command(key, dict(template=template_id, facts=fact_ids), action)
