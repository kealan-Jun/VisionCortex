from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from .db import Database


class Conflict(ValueError):
    pass


class Missing(LookupError):
    pass


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def canonical(value) -> str:
    return json.dumps(clean(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def ident(value=None) -> UUID:
    return UUID(str(value)) if value is not None else uuid4()


def instant(value) -> datetime:
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError("A timezone is required; receiving time is not event time")
    return moment.astimezone(timezone.utc)


def quantity(value, *, signed=False) -> Decimal:
    if isinstance(value, (float, bool)):
        raise ValueError("Quantity must be an exact decimal string or integer")
    try:
        result = Decimal(value)
        if not result.is_finite() or (not signed and result < 0) or abs(result) >= Decimal("1e19"):
            raise ValueError("Invalid quantity")
        if result.quantize(Decimal("0.000000001")) != result:
            raise ValueError("Quantity exceeds nine decimal places")
        return result
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal quantity") from exc


UNITS = {"g": ("g", Decimal(1)), "mg": ("g", Decimal("0.001")),
         "kg": ("g", Decimal(1000)), "mL": ("mL", Decimal(1)),
         "L": ("mL", Decimal(1000)), "pcs": ("pcs", Decimal(1))}


def amount(value, unit):
    if unit not in UNITS:
        raise ValueError("Unsupported or ambiguous unit")
    base, scale = UNITS[unit]
    result = quantity(quantity(value) * scale)
    if base == "pcs" and result != result.to_integral():
        raise ValueError("Discrete quantities must be integers")
    return result, base


class Core:
    """The only writer of formal customer facts and material quantities."""

    def __init__(self, db: Database, customer_id):
        self.db, self.customer = db, ident(customer_id)
        self._transaction = ContextVar("vc_lab_command_transaction", default=None)

    @staticmethod
    def customer_instance(db, code, *, scope="development"):
        with db.transaction() as conn:
            row = conn.execute("INSERT INTO customer_instance(id,code,data_scope) VALUES(%s,%s,%s) ON CONFLICT(code) DO NOTHING RETURNING *", (ident(), code, scope)).fetchone()
            if row is None:
                row = conn.execute("SELECT * FROM customer_instance WHERE code=%s", (code,)).fetchone()
                if row["data_scope"] != scope:
                    raise Conflict("Customer data scope cannot change implicitly")
            return clean(row)

    def command(self, key, request, action):
        if not isinstance(key, str) or not 1 <= len(key) <= 200:
            raise ValueError("A stable command idempotency key is required")
        digest = fingerprint(request)
        def execute(conn):
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (f"{self.customer}:{key}",))
            old = conn.execute("SELECT * FROM command_receipt WHERE customer_id=%s AND key=%s", (self.customer, key)).fetchone()
            if old:
                if old["fingerprint"] != digest:
                    raise Conflict("Idempotency key reused for a different command")
                return old["result"]
            result = clean(action(conn))
            conn.execute("INSERT INTO command_receipt(customer_id,key,fingerprint,result) VALUES(%s,%s,%s,%s)", (self.customer, key, digest, Jsonb(result)))
            return result
        current = self._transaction.get()
        if current is not None:
            return execute(current)
        with self.db.transaction() as conn:
            token = self._transaction.set(conn)
            try:
                return execute(conn)
            finally:
                self._transaction.reset(token)

    def _entity(self, conn, entity_id, kind=None, *, lock=False):
        row = conn.execute("SELECT * FROM entity WHERE customer_id=%s AND id=%s" + (" FOR UPDATE" if lock else ""), (self.customer, ident(entity_id))).fetchone()
        if not row or (kind is not None and row["kind"] != kind):
            raise Missing("Entity is absent from this customer or has the wrong kind")
        return row

    def _register(self, conn, kind, code, metadata=None, entity_id=None):
        if kind not in {"instrument", "capture_device", "container", "tag", "material_type", "source_lot", "material_unit", "process_batch", "experiment_run", "person"}:
            raise ValueError("Unknown identity kind")
        if not code or not isinstance(code, str) or len(code) > 200:
            raise ValueError("Explicit entity code is required")
        row = conn.execute("INSERT INTO entity(customer_id,id,kind,code,metadata) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(customer_id,kind,code) DO NOTHING RETURNING *", (self.customer, ident(entity_id), kind, code, Jsonb(clean(metadata or {})))).fetchone()
        if row is None:
            row = conn.execute("SELECT * FROM entity WHERE customer_id=%s AND kind=%s AND code=%s", (self.customer, kind, code)).fetchone()
            if entity_id is not None and row['id'] != ident(entity_id):
                raise Conflict('Explicit permanent identity differs from the registered object')
            if row["metadata"] != clean(metadata or {}):
                raise Conflict("Existing identity has different registration metadata")
        return row

    def register(self, key, kind, code, metadata=None):
        def action(conn):
            row = self._register(conn, kind, code, metadata)
            self._emit(conn, "entity", row["id"], row["revision"], row)
            return row
        return self.command(key, dict(kind=kind, code=code, metadata=metadata), action)

    def assign_camera_person(self, *args, **kwargs):
        from .capture_context import assign_camera_person
        return assign_camera_person(self, *args, **kwargs)

    def camera_context(self, camera_code, at):
        from .capture_context import camera_context
        return camera_context(self, camera_code, at)

    def issue_identity(self, key, kind, customer_prefix, *, metadata=None, unit=None, initial=None, actor="registrar"):
        """Issue one real identity atomically; proposal PDFs never call this command."""
        markers = {"instrument": "EQ", "material_unit": "MA", "container": "CT", "tag": "TG"}
        if kind not in markers or not re.fullmatch(r"[A-Z][A-Z0-9]{1,11}", customer_prefix):
            raise ValueError("Explicit supported identity kind and fixed customer prefix required")
        if kind == "material_unit" and unit is None:
            raise ValueError("A material's quantity dimension must be explicitly registered")
        def action(conn):
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", ("identity-prefix:" + customer_prefix,))
            conn.execute("INSERT INTO identity_namespace(customer_id,prefix) VALUES(%s,%s) ON CONFLICT DO NOTHING", (self.customer, customer_prefix))
            namespace = conn.execute("SELECT prefix FROM identity_namespace WHERE customer_id=%s FOR UPDATE", (self.customer,)).fetchone()
            if not namespace or namespace["prefix"] != customer_prefix:
                raise Conflict("Customer prefix is already assigned or differs from the established namespace")
            while True:
                current = conn.execute("SELECT last_number FROM identity_counter WHERE customer_id=%s AND kind=%s FOR UPDATE", (self.customer, kind)).fetchone()
                if current and current["last_number"] >= 999999:
                    raise Conflict("Identity number space exhausted; an explicit versioned extension is required")
                number = conn.execute("INSERT INTO identity_counter(customer_id,kind,last_number) VALUES(%s,%s,1) ON CONFLICT(customer_id,kind) DO UPDATE SET last_number=identity_counter.last_number+1 RETURNING last_number", (self.customer, kind)).fetchone()["last_number"]
                code = f"{customer_prefix}-{markers[kind]}-{number:06d}"
                if not conn.execute("SELECT 1 FROM entity WHERE customer_id=%s AND kind=%s AND code=%s", (self.customer, kind, code)).fetchone():
                    break
            if kind == "material_unit":
                result = self.register_material(key + ":material", code, unit, initial, metadata=metadata, actor=actor)
                entity_id = result["material_id"]
            else:
                result = self.register(key + ":entity", kind, code, dict(metadata or {}, **({"mode": "reusable"} if kind == "tag" else {})))
                entity_id = result["id"]
            if kind != "tag":
                tag = self.register(key + ":permanent-label", "tag", code, {"mode": "permanent"})
                self.assign_tag(key + ":permanent-binding", tag["id"], entity_id, datetime.now(timezone.utc), actor=actor, reason="permanent identity issuance")
            return dict(result, code=code, qr_payload=code, customer_prefix=customer_prefix, identity_kind=kind, label_mode="reusable" if kind == "tag" else "permanent")
        return self.command(key, dict(kind=kind, prefix=customer_prefix, metadata=metadata, unit=unit, initial=initial, actor=actor), action)

    def reserve_equipment_catalogue(self, key, catalogue):
        from .equipment import reserve_catalogue
        return reserve_catalogue(self, key, catalogue)

    def issue_equipment(self, key, **request):
        from .equipment import issue_equipment
        return issue_equipment(self, key, **request)

    def activate_equipment(self, key, device_id, **request):
        from .equipment import activate_equipment
        return activate_equipment(self, key, device_id, **request)

    def retire_equipment(self, key, device_id, **request):
        from .equipment import retire_equipment
        return retire_equipment(self, key, device_id, **request)

    def resolve_qr(self, value, at):
        from .equipment import resolve_qr
        return resolve_qr(self, value, at)

    def _emit(self, conn, kind, record_id, revision, payload):
        # Serialize sequence allocation per customer until commit. Otherwise a
        # later committed sequence could let an index cursor skip an older txn.
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", ("outbox:" + str(self.customer),))
        conn.execute("INSERT INTO outbox(customer_id,event_id,kind,record_id,revision,payload) VALUES(%s,%s,%s,%s,%s,%s)", (self.customer, ident(), kind, ident(record_id), revision, Jsonb(clean(payload))))

    def register_material(self, key, code, unit, initial=None, *, metadata=None, occurred_at=None, actor="registrar"):
        at = instant(occurred_at) if occurred_at else datetime.now(timezone.utc)
        initial_value, base = amount(initial if initial is not None else 0, unit)
        body = dict(code=code, unit=unit, initial=initial, metadata=metadata, occurred_at=occurred_at, actor=actor)

        def action(conn):
            for ref, expected_kind in (("source_lot_id", "source_lot"), ("material_type_id", "material_type")):
                if (metadata or {}).get(ref):
                    self._entity(conn, metadata[ref], expected_kind)
            entity = self._register(conn, "material_unit", code, metadata)
            old = conn.execute("SELECT * FROM material_state WHERE customer_id=%s AND material_id=%s", (self.customer, entity["id"])).fetchone()
            if old:
                raise Conflict("Material is already registered; use an explicit receipt/adjustment")
            conn.execute("INSERT INTO material_state(customer_id,material_id,unit,balance) VALUES(%s,%s,%s,%s)", (self.customer, entity["id"], base, initial_value if initial is not None else None))
            op = ident()
            conn.execute("INSERT INTO operation(customer_id,id,kind,occurred_at,actor,metadata) VALUES(%s,%s,'opening',%s,%s,%s)", (self.customer, op, at, actor, Jsonb({"initial_known": initial is not None})))
            if initial is not None:
                conn.execute("INSERT INTO quantity_ledger_entry(customer_id,id,operation_id,material_id,delta,unit) VALUES(%s,%s,%s,%s,%s,%s)", (self.customer, ident(), op, entity["id"], initial_value, base))
                if initial_value > 0:
                    conn.execute("INSERT INTO operation_line(customer_id,id,operation_id,material_id,direction,quantity,unit,quantity_basis,state_revision) VALUES(%s,%s,%s,%s,'output',%s,%s,'confirmed_opening',1)", (self.customer, ident(), op, entity["id"], initial_value, base))
            result = dict(material_id=entity["id"], operation_id=op, unit=base, balance=initial_value if initial is not None else None, balance_status="known" if initial is not None else "unknown", revision=1)
            self._emit(conn, "material", entity["id"], 1, result)
            return result

        return self.command(key, body, action)

    def assign_tag(self, key, tag_id, target_id, valid_from, *, actor, reason):
        at = instant(valid_from)

        def action(conn):
            tag = self._entity(conn, tag_id, "tag", lock=True)
            target = self._entity(conn, target_id)
            retired = conn.execute("SELECT retired_at FROM equipment_identity WHERE customer_id=%s AND device_id=%s AND state='retired'", (self.customer, target['id'])).fetchone()
            if retired and at >= retired['retired_at']:
                raise Conflict('A retired permanent equipment identity cannot be reassigned or reactivated')
            if target["kind"] not in {"instrument", "container", "material_unit"}:
                raise ValueError("Tag target must be a physical equipment, container or material unit")
            rows = conn.execute("SELECT * FROM tag_assignment WHERE customer_id=%s AND tag_id=%s ORDER BY version", (self.customer, ident(tag_id))).fetchall()
            if tag["metadata"].get("mode") == "permanent" and any(r["target_id"] != ident(target_id) for r in rows):
                raise Conflict("A permanent identity code cannot be rebound")
            if rows:
                old = rows[-1]
                if old["valid_to"] is None:
                    if at <= old["valid_from"]:
                        raise Conflict("Binding interval overlaps existing history")
                    conn.execute("UPDATE tag_assignment SET valid_to=%s WHERE customer_id=%s AND id=%s", (at, self.customer, old["id"]))
                elif at < old["valid_to"]:
                    raise Conflict("Binding interval overlaps existing history")
            row = conn.execute("INSERT INTO tag_assignment(customer_id,id,tag_id,target_id,valid_from,version,actor,reason) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *", (self.customer, ident(), ident(tag_id), ident(target_id), at, len(rows) + 1, actor, reason)).fetchone()
            self._emit(conn, "tag_assignment", row["id"], row["version"], row)
            return row

        return self.command(key, dict(tag=tag_id, target=target_id, at=valid_from, actor=actor, reason=reason), action)

    def resolve_tag(self, tag_id, at, until=None):
        start = instant(at)
        end = instant(until) if until else start
        if end < start:
            raise ValueError("Invalid time uncertainty interval")
        with self.db.transaction(readonly=True) as conn:
            self._entity(conn, tag_id, "tag")
            rows = conn.execute("SELECT * FROM tag_assignment WHERE customer_id=%s AND tag_id=%s AND valid_from<=%s AND (valid_to IS NULL OR valid_to>%s) ORDER BY version", (self.customer, ident(tag_id), end, start)).fetchall()
            certain = len(rows) == 1 and rows[0]["valid_from"] <= start and (rows[0]["valid_to"] is None or end < rows[0]["valid_to"])
            return clean(dict(association_status="explicit" if certain else "needs_review", assignments=rows, target_id=rows[0]["target_id"] if certain else None))

    def bind_container(self, key, container_id, material_id, at, *, actor):
        def action(conn):
            for eid in sorted([ident(container_id), ident(material_id)]):
                self._entity(conn, eid, lock=True)
            self._entity(conn, container_id, "container")
            self._entity(conn, material_id, "material_unit")
            when = instant(at)
            old = conn.execute("SELECT * FROM container_content_binding WHERE customer_id=%s AND (container_id=%s OR material_id=%s) AND valid_to IS NULL FOR UPDATE", (self.customer, ident(container_id), ident(material_id))).fetchall()
            for row in old:
                if when <= row["valid_from"]:
                    raise Conflict("Container binding must preserve temporal history")
                conn.execute("UPDATE container_content_binding SET valid_to=%s WHERE customer_id=%s AND id=%s", (when, self.customer, row["id"]))
            version = conn.execute("SELECT COALESCE(MAX(version),0)+1 AS version FROM container_content_binding WHERE customer_id=%s AND container_id=%s", (self.customer, ident(container_id))).fetchone()["version"]
            row = conn.execute("INSERT INTO container_content_binding(customer_id,id,container_id,material_id,valid_from,version) VALUES(%s,%s,%s,%s,%s,%s) RETURNING *", (self.customer, ident(), ident(container_id), ident(material_id), when, version)).fetchone()
            self._emit(conn, "container_binding", row["id"], version, dict(row, actor=actor))
            return row
        return self.command(key, dict(container=container_id, material=material_id, at=at, actor=actor), action)

    def equipment_use(self, key, instrument_id, batch_id, at, *, actor, material_id=None, position=None, end_id=None):
        def action(conn):
            self._entity(conn, instrument_id, "instrument")
            self._entity(conn, batch_id, "process_batch")
            if material_id:
                self._entity(conn, material_id, "material_unit")
            if end_id:
                row = conn.execute("UPDATE equipment_use SET valid_to=%s WHERE customer_id=%s AND id=%s AND instrument_id=%s AND batch_id=%s AND valid_to IS NULL AND valid_from<=%s RETURNING *", (instant(at), self.customer, ident(end_id), ident(instrument_id), ident(batch_id), instant(at))).fetchone()
                if not row:
                    raise Conflict("Active equipment use/version not found")
            else:
                row = conn.execute("INSERT INTO equipment_use(customer_id,id,instrument_id,batch_id,material_id,position,valid_from,actor) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *", (self.customer, ident(), ident(instrument_id), ident(batch_id), ident(material_id) if material_id else None, position, instant(at), actor)).fetchone()
            self._emit(conn, "equipment_use", row["id"], 1, row)
            return row
        return self.command(key, dict(instrument=instrument_id, batch=batch_id, material=material_id, at=at, actor=actor, position=position, end=end_id), action)

    def _states(self, conn, ids):
        unique = sorted({ident(i) for i in ids})
        rows = conn.execute("SELECT * FROM material_state WHERE customer_id=%s AND material_id=ANY(%s) ORDER BY material_id FOR UPDATE", (self.customer, unique)).fetchall()
        if len(rows) != len(unique):
            raise Missing("Material outside customer scope or not registered")
        return {r["material_id"]: r for r in rows}

    def _evidence(self, conn, ids):
        values = [ident(i) for i in ids]
        count = conn.execute("SELECT count(*) AS n FROM evidence_ref WHERE customer_id=%s AND id=ANY(%s)", (self.customer, values)).fetchone()["n"]
        if count != len(set(values)):
            raise Missing("Evidence outside customer scope")
        return values

    def _lines(self, lines):
        result = []
        seen = set()
        for line in lines:
            q, unit = amount(line["quantity"], line["unit"])
            mid = ident(line["material_id"])
            if not q or mid in seen:
                raise ValueError("Operation lines must be positive and unique per direction")
            seen.add(mid)
            result.append(dict(material_id=mid, quantity=q, unit=unit, quantity_basis=line.get("quantity_basis", "explicit_confirmation"), expected_revision=line.get("expected_revision")))
        return result

    def _change(self, conn, op_id, states, changes):
        balances = {}
        for mid, (delta, used_delta, reverses) in changes.items():
            row = states[mid]
            new = row["balance"] + delta if row["balance"] is not None else None
            if new is not None and new < row["reserved"]:
                raise Conflict("Insufficient available material balance")
            if row["used"] + used_delta < 0:
                raise Conflict("Correction would make cumulative usage negative")
            conn.execute("INSERT INTO quantity_ledger_entry(customer_id,id,operation_id,material_id,delta,used_delta,unit,reverses_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)", (self.customer, ident(), op_id, mid, delta, used_delta, row["unit"], reverses))
            updated = conn.execute("UPDATE material_state SET balance=%s,used=used+%s,revision=revision+1 WHERE customer_id=%s AND material_id=%s RETURNING *", (new, used_delta, self.customer, mid)).fetchone()
            balances[str(mid)] = updated
            conn.execute("UPDATE field_snapshot SET stale=true WHERE customer_id=%s AND input_revisions ? %s", (self.customer, str(mid)))
            self._emit(conn, "material", mid, updated["revision"], updated)
        return balances

    def operate(self, key, kind, inputs, outputs=None, *, occurred_at, actor, operation_id=None, batch_id=None, run_id=None, evidence_ids=None, losses="0", metadata=None):
        allowed = {"consume", "split", "mix", "transfer", "return", "destructive_test", "non_destructive_test", "loss", "scrap", "receive"}
        if kind not in allowed or not actor:
            raise ValueError("Explicit supported operation and actor required")
        ins, outs = self._lines(inputs), self._lines(outputs or [])
        if not ins and kind != "receive":
            raise ValueError("Operation inputs are required")
        if kind == "receive" and (ins or not outs):
            raise ValueError("Receipt has outputs only")
        body = dict(kind=kind, inputs=inputs, outputs=outputs, occurred_at=occurred_at, actor=actor, operation_id=operation_id, batch_id=batch_id, run_id=run_id, evidence_ids=evidence_ids, losses=losses, metadata=metadata)

        def action(conn):
            op_id = ident(operation_id)
            if conn.execute("SELECT 1 FROM operation WHERE customer_id=%s AND id=%s", (self.customer, op_id)).fetchone():
                raise Conflict("Business operation already exists; attach evidence or correct it explicitly")
            states = self._states(conn, [l["material_id"] for l in ins + outs])
            for line in ins + outs:
                state = states[line["material_id"]]
                if state["unit"] != line["unit"]:
                    raise Conflict("Material unit/dimension mismatch")
                if line["expected_revision"] is not None and state["revision"] != line["expected_revision"]:
                    raise Conflict("Material version changed")
            if batch_id:
                self._entity(conn, batch_id, "process_batch")
            if run_id:
                self._entity(conn, run_id, "experiment_run")
            ev = self._evidence(conn, evidence_ids or [])
            in_ids, out_ids = {l["material_id"] for l in ins}, {l["material_id"] for l in outs}
            if kind in {"split", "mix"}:
                if not outs or in_ids & out_ids or any(states[i]["balance"] != 0 for i in out_ids):
                    raise Conflict("Split/mix requires distinct empty output identities")
                units = {l["unit"] for l in ins + outs}
                if len(units) != 1:
                    raise Conflict("Different dimensions cannot be added without an explicit conversion contract")
                lost = quantity(losses)
                if sum(l["quantity"] for l in ins) != sum(l["quantity"] for l in outs) + lost:
                    raise Conflict("Input/output difference requires an explicit measured loss")
                if kind == "split" and (len(ins) != 1 or lost):
                    raise Conflict("Split preserves the input quantity")
            if kind in {"transfer", "non_destructive_test"}:
                if {(l["material_id"], l["quantity"]) for l in ins} != {(l["material_id"], l["quantity"]) for l in outs}:
                    raise Conflict("Whole transfer/non-destructive test preserves material identity and quantity")
            if kind in {"consume", "loss", "scrap"} and outs:
                raise ValueError("This operation has no outputs")
            if kind == "return" and (not ins or not outs):
                raise ValueError("Return must identify both the source and receiving material")
            if kind in {"loss", "scrap"} and not (metadata or {}).get("reason"):
                raise ValueError("Loss/scrap requires an explicit reason")
            if kind == "destructive_test":
                if len({l["unit"] for l in ins + outs}) != 1 or sum(l["quantity"] for l in outs) > sum(l["quantity"] for l in ins):
                    raise Conflict("Destructive test cannot create quantity or mix dimensions")
            for line in ins:
                state = states[line["material_id"]]
                if state["balance"] is not None and line["quantity"] > state["balance"] - state["reserved"]:
                    raise Conflict("Insufficient available material balance")
            conn.execute("INSERT INTO operation(customer_id,id,kind,batch_id,run_id,occurred_at,actor,evidence_ids,metadata) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)", (self.customer, op_id, kind, ident(batch_id) if batch_id else None, ident(run_id) if run_id else None, instant(occurred_at), actor, ev, Jsonb(clean(dict(metadata or {}, measured_loss=str(quantity(losses)))))))
            changes = defaultdict(lambda: [Decimal(0), Decimal(0), None])
            for direction, lines in (("input", ins), ("output", outs)):
                for line in lines:
                    mid = line["material_id"]
                    delta = line["quantity"] * (-1 if direction == "input" else 1)
                    changes[mid][0] += delta
                    if direction == "input" and kind in {"consume", "mix", "destructive_test"}:
                        changes[mid][1] += line["quantity"]
                    if direction == "output" and kind == "destructive_test" and mid in in_ids:
                        changes[mid][1] -= line["quantity"]
                    state_revision = states[mid]["revision"] + (1 if direction == "output" else 0)
                    conn.execute("INSERT INTO operation_line(customer_id,id,operation_id,material_id,direction,quantity,unit,quantity_basis,state_revision) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)", (self.customer, ident(), op_id, mid, direction, line["quantity"], line["unit"], line["quantity_basis"], state_revision))
            balances = self._change(conn, op_id, states, changes)
            result = dict(operation_id=op_id, kind=kind, revision=1, balances=balances, occurred_at=instant(occurred_at), batch_id=batch_id, run_id=run_id, evidence_ids=ev, inputs=ins, outputs=outs)
            self._emit(conn, "operation", op_id, 1, result)
            return result

        return self.command(key, body, action)

    def correct_operation(self, key, operation_id, replacement_inputs, *, expected_revision, actor, reason, evidence_ids=None):
        if not actor or not reason:
            raise ValueError("Correction actor and reason are required")
        replacement = self._lines(replacement_inputs)

        def action(conn):
            old = conn.execute("SELECT * FROM operation WHERE customer_id=%s AND id=%s FOR UPDATE", (self.customer, ident(operation_id))).fetchone()
            if not old or old["status"] != "confirmed" or old["revision"] != expected_revision:
                raise Conflict("Operation revision is not current")
            effective_kind = old["metadata"].get("effective_kind", old["kind"])
            if effective_kind not in {"consume", "mix", "destructive_test"}:
                raise ValueError("Input correction is supported for consumption/mixing/testing")
            lines = conn.execute("SELECT * FROM operation_line WHERE customer_id=%s AND operation_id=%s", (self.customer, old["id"])).fetchall()
            previous = {l["material_id"]: l for l in lines if l["direction"] == "input"}
            if set(previous) != {l["material_id"] for l in replacement}:
                raise Conflict("This correction must retain input identities")
            states = self._states(conn, previous)
            cid = ident()
            outputs = [l for l in lines if l["direction"] == "output"]
            metadata = {"reason": reason, "output_measurements_preserved": True, "effective_kind": effective_kind}
            for name in ('usage_scope','weighing_session_id','context_snapshot'):
                if name in old['metadata']:
                    metadata[name]=old['metadata'][name]
            correction_evidence = self._evidence(conn, sorted(set(old['evidence_ids']) | {ident(e) for e in (evidence_ids or [])}, key=str))
            conn.execute("INSERT INTO operation(customer_id,id,kind,batch_id,run_id,occurred_at,actor,evidence_ids,supersedes_id,metadata) VALUES(%s,%s,'correction',%s,%s,%s,%s,%s,%s,%s)", (self.customer, cid, old["batch_id"], old["run_id"], old["occurred_at"], actor, correction_evidence, old["id"], Jsonb(metadata)))
            balances = {}
            for line in replacement:
                mid = line["material_id"]
                if line["unit"] != previous[mid]["unit"]:
                    raise Conflict("Correction unit mismatch")
                ledger = conn.execute("SELECT * FROM quantity_ledger_entry WHERE customer_id=%s AND operation_id=%s AND material_id=%s AND reverses_id IS NULL ORDER BY created_at DESC LIMIT 1", (self.customer, old["id"], mid)).fetchone()
                if not ledger:
                    raise Conflict("Original ledger entry is missing; correction cannot be synthesized")
                reversed_state = self._change(conn, cid, states, {mid: [-ledger["delta"], -ledger["used_delta"], ledger["id"]]})
                states[mid] = reversed_state[str(mid)]
                returned = sum((out["quantity"] for out in outputs if out["material_id"] == mid), Decimal(0))
                if line["quantity"] < returned:
                    raise Conflict("Corrected input cannot be below its independently measured returned amount")
                net = line["quantity"] - returned
                balances.update(self._change(conn, cid, states, {mid: [-net, net, None]}))
                conn.execute("INSERT INTO operation_line(customer_id,id,operation_id,material_id,direction,quantity,unit,quantity_basis,state_revision) VALUES(%s,%s,%s,%s,'input',%s,%s,'correction',%s)", (self.customer, ident(), cid, mid, line["quantity"], line["unit"], states[mid]["revision"]))
            for line in outputs:
                conn.execute("INSERT INTO operation_line(customer_id,id,operation_id,material_id,direction,quantity,unit,quantity_basis,state_revision) VALUES(%s,%s,%s,%s,'output',%s,%s,%s,%s)", (self.customer, ident(), cid, line["material_id"], line["quantity"], line["unit"], line["quantity_basis"], line["state_revision"]))
            discrepancy = sum(l["quantity"] for l in outputs) - sum(l["quantity"] for l in replacement)
            downstream = conn.execute("SELECT DISTINCT operation_id FROM operation_line WHERE customer_id=%s AND direction='input' AND material_id=ANY(%s) AND operation_id<>%s", (self.customer, [l["material_id"] for l in outputs], old["id"])).fetchall()
            metadata.update(quantity_discrepancy=str(discrepancy), downstream_operations=[str(r["operation_id"]) for r in downstream])
            conn.execute("UPDATE operation SET metadata=%s WHERE customer_id=%s AND id=%s", (Jsonb(metadata), self.customer, cid))
            conn.execute("UPDATE operation SET status='corrected',revision=revision+1 WHERE customer_id=%s AND id=%s", (self.customer, old["id"]))
            self._emit(conn, "operation", old["id"], old["revision"] + 1, dict(operation_id=old["id"], status="corrected", superseded_by=cid))
            linked_facts = conn.execute("SELECT * FROM fact_revision WHERE customer_id=%s AND status='confirmed' AND scope='material_consumption' AND value->>'operation_id'=%s FOR UPDATE", (self.customer, str(old["id"]))).fetchall()
            for fact in linked_facts:
                material_id = ident(fact["value"]["material_id"])
                revised_line = next((line for line in replacement if line["material_id"] == material_id), None)
                if not revised_line:
                    raise Conflict("Linked usage fact has no corresponding corrected input")
                new_fact = ident()
                new_value = dict(fact["value"], operation_id=str(cid), quantity=str(revised_line["quantity"]))
                conn.execute("INSERT INTO fact_revision(customer_id,id,candidate_id,revision,scope,value,unit,actor,supersedes_id,correction_metadata) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (self.customer, new_fact, fact["candidate_id"], fact["revision"] + 1, fact["scope"], Jsonb(new_value), revised_line["unit"], actor, fact["id"], Jsonb(dict(reason=reason, operation_id=str(cid)))))
                conn.execute("UPDATE fact_revision SET status='corrected' WHERE customer_id=%s AND id=%s", (self.customer, fact["id"]))
                conn.execute("INSERT INTO confirmation(customer_id,id,fact_id,scope,actor,evidence_ids) VALUES(%s,%s,%s,%s,%s,%s)", (self.customer, ident(), new_fact, fact["scope"], actor, old["evidence_ids"]))
                conn.execute("UPDATE field_snapshot SET stale=true WHERE customer_id=%s AND input_revisions ? %s", (self.customer, str(fact["id"])))
                self._emit(conn, "fact", fact["id"], fact["revision"] + 1, dict(fact_id=fact["id"], status="corrected", superseded_by=new_fact))
                self._emit(conn, "fact", new_fact, fact["revision"] + 1, dict(fact_id=new_fact, revision=fact["revision"] + 1, scope=fact["scope"], value=new_value, unit=revised_line["unit"], evidence_ids=old["evidence_ids"]))
            conn.execute("UPDATE field_snapshot SET stale=true WHERE customer_id=%s AND input_revisions ? %s", (self.customer, str(old["id"])))
            result = dict(operation_id=cid, supersedes_id=old["id"], revision=1, balances=balances, metadata=metadata, outputs=outputs)
            self._emit(conn, "operation", cid, 1, result)
            return result

        request = dict(operation=operation_id, inputs=replacement_inputs, expected=expected_revision, actor=actor, reason=reason)
        if evidence_ids is not None:
            request['evidence_ids'] = evidence_ids
        return self.command(key, request, action)

    def cancel_consumption(self, key, operation_id, *, expected_revision, actor, reason, evidence_ids=None):
        """Explicit full reversal of an unlinked consumption; never delete history."""
        if not actor or not reason:
            raise ValueError('Cancellation actor and reason required')
        request = dict(operation_id=operation_id, expected_revision=expected_revision, actor=actor,
                       reason=reason, evidence_ids=evidence_ids)
        def action(conn):
            old = conn.execute('SELECT * FROM operation WHERE customer_id=%s AND id=%s FOR UPDATE',
                               (self.customer, ident(operation_id))).fetchone()
            if (not old or old['status'] != 'confirmed' or old['revision'] != expected_revision
                    or old['metadata'].get('effective_kind', old['kind']) != 'consume'):
                raise Conflict('Current consumption operation/version required')
            if conn.execute("SELECT 1 FROM fact_revision WHERE customer_id=%s AND status='confirmed' AND value->>'operation_id'=%s LIMIT 1",
                            (self.customer,str(old['id']))).fetchone():
                raise Conflict('Linked consumption facts require their own explicit correction')
            lines = conn.execute('SELECT * FROM operation_line WHERE customer_id=%s AND operation_id=%s',
                                 (self.customer,old['id'])).fetchall()
            if not lines or any(l['direction'] != 'input' for l in lines):
                raise Conflict('One current input-only consumption required')
            states = self._states(conn,[l['material_id'] for l in lines])
            evidence = self._evidence(conn,sorted(set(old['evidence_ids']) | {ident(e) for e in (evidence_ids or [])},key=str))
            cid=ident()
            metadata=dict(reason=reason,effective_kind='consume',consumption_cancelled=True,
                          cancelled_material_ids=[str(l['material_id']) for l in lines])
            conn.execute("INSERT INTO operation(customer_id,id,kind,batch_id,run_id,occurred_at,actor,evidence_ids,supersedes_id,metadata) VALUES(%s,%s,'correction',%s,%s,%s,%s,%s,%s,%s)",
                (self.customer,cid,old['batch_id'],old['run_id'],old['occurred_at'],actor,evidence,old['id'],Jsonb(metadata)))
            changes={}
            for line in lines:
                ledger=conn.execute('SELECT * FROM quantity_ledger_entry WHERE customer_id=%s AND operation_id=%s AND material_id=%s AND reverses_id IS NULL ORDER BY created_at DESC LIMIT 1',
                                    (self.customer,old['id'],line['material_id'])).fetchone()
                if not ledger:
                    raise Conflict('Original consumption ledger entry is unavailable')
                changes[line['material_id']]=[-ledger['delta'],-ledger['used_delta'],ledger['id']]
            balances=self._change(conn,cid,states,changes)
            conn.execute("UPDATE operation SET status='corrected',revision=revision+1 WHERE customer_id=%s AND id=%s",(self.customer,old['id']))
            result=dict(operation_id=cid,supersedes_id=old['id'],revision=1,balances=balances,
                        metadata=metadata,evidence_ids=evidence)
            self._emit(conn,'operation',old['id'],old['revision']+1,dict(operation_id=old['id'],status='corrected',superseded_by=cid))
            self._emit(conn,'operation',cid,1,result)
            return result
        return self.command(key,request,action)

    def correct_fact(self, key, fact_id, value, *, expected_revision, actor, reason, evidence_ids):
        if not actor or not reason or not evidence_ids:
            raise ValueError("Fact correction requires actor, reason and supporting evidence")
        number = quantity(value, signed=True)
        def action(conn):
            old = conn.execute("SELECT * FROM fact_revision WHERE customer_id=%s AND id=%s FOR UPDATE", (self.customer, ident(fact_id))).fetchone()
            if not old or old["status"] != "confirmed" or old["revision"] != expected_revision:
                raise Conflict("Fact revision is not current")
            if old["scope"] not in {"display_reading", "weighing_result"}:
                raise Conflict("Material usage correction must go through its quantity operation")
            ev = self._evidence(conn, evidence_ids)
            fid = ident()
            conn.execute("INSERT INTO fact_revision(customer_id,id,candidate_id,revision,scope,value,unit,actor,supersedes_id,correction_metadata) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (self.customer, fid, old["candidate_id"], old["revision"] + 1, old["scope"], Jsonb(str(number)), old["unit"], actor, old["id"], Jsonb(dict(reason=reason))))
            conn.execute("UPDATE fact_revision SET status='corrected' WHERE customer_id=%s AND id=%s", (self.customer, old["id"]))
            conn.execute("INSERT INTO confirmation(customer_id,id,fact_id,scope,actor,evidence_ids) VALUES(%s,%s,%s,%s,%s,%s)", (self.customer, ident(), fid, old["scope"], actor, ev))
            conn.execute("UPDATE field_snapshot SET stale=true WHERE customer_id=%s AND input_revisions ? %s", (self.customer, str(old["id"])))
            self._emit(conn, "fact", old["id"], old["revision"] + 1, dict(fact_id=old["id"], status="corrected", superseded_by=fid))
            result = dict(fact_id=fid, supersedes_id=old["id"], revision=old["revision"] + 1, value=str(number), unit=old["unit"], scope=old["scope"], evidence_ids=ev, actor=actor, reason=reason)
            self._emit(conn, "fact", fid, old["revision"] + 1, result)
            return result
        return self.command(key, dict(fact=fact_id, value=value, expected=expected_revision, actor=actor, reason=reason, evidence=evidence_ids), action)

    def reserve(self, key, material_id, value, unit, *, reason, release_id=None):
        q, base = amount(value, unit)

        def action(conn):
            mid = ident(material_id)
            state = self._states(conn, [mid])[mid]
            if base != state["unit"] or not q:
                raise ValueError("Reservation quantity/unit invalid")
            if release_id:
                row = conn.execute("UPDATE reservation SET status='released' WHERE customer_id=%s AND id=%s AND material_id=%s AND status='active' RETURNING *", (self.customer, ident(release_id), mid)).fetchone()
                if not row or row["quantity"] != q:
                    raise Conflict("Reservation is not active or has different quantity")
                delta = -q
            else:
                if state["balance"] is None:
                    raise Conflict("Cannot promise availability with unknown balance")
                if state["balance"] - state["reserved"] < q:
                    raise Conflict("Insufficient unreserved balance")
                row = conn.execute("INSERT INTO reservation(customer_id,id,material_id,quantity,unit,reason) VALUES(%s,%s,%s,%s,%s,%s) RETURNING *", (self.customer, ident(), mid, q, base, reason)).fetchone()
                delta = q
            conn.execute("UPDATE material_state SET reserved=reserved+%s,revision=revision+1 WHERE customer_id=%s AND material_id=%s", (delta, self.customer, mid))
            return row
        return self.command(key, dict(material=material_id, value=value, unit=unit, reason=reason, release=release_id), action)

    def balance(self, material_id):
        with self.db.transaction(readonly=True) as conn:
            row = conn.execute("SELECT * FROM material_state WHERE customer_id=%s AND material_id=%s", (self.customer, ident(material_id))).fetchone()
            if not row:
                raise Missing("Material outside this customer")
            row["balance_status"] = "unknown" if row["balance"] is None else "known"
            row["available"] = row["balance"] - row["reserved"] if row["balance"] is not None else None
            return clean(row)

    def confirm_candidate(self, key, candidate_id, *, expected_revision, scope, actor, policy_version=None):
        def action(conn):
            candidate = conn.execute("SELECT * FROM field_candidate WHERE customer_id=%s AND id=%s FOR UPDATE", (self.customer, ident(candidate_id))).fetchone()
            if not candidate:
                raise Missing("Candidate outside this customer")
            if candidate["revision"] != expected_revision or candidate["status"] != "candidate":
                raise Conflict("Candidate has changed or is already decided")
            semantics = {"display_reading": "display_reading", "weighing_result": "net_mass"}
            if scope not in semantics or candidate["semantics"] != semantics[scope] or candidate["value"] is None or candidate["conflicts"]:
                raise Conflict("Confirmation scope/quantity semantics are not established")
            if not candidate["association"].get("instrument_id"):
                raise Conflict("Instrument attribution requires explicit evidence")
            self._entity(conn, candidate["association"]["instrument_id"], "instrument")
            observation = conn.execute("SELECT * FROM observation WHERE customer_id=%s AND id=%s", (self.customer, candidate["observation_id"])).fetchone()
            if not observation["evidence_ids"]:
                raise Conflict("Formal facts require traceable evidence")
            unverified = conn.execute("SELECT count(*) AS n FROM evidence_ref e JOIN media_asset m ON m.customer_id=e.customer_id AND m.id=e.media_id WHERE e.customer_id=%s AND e.id=ANY(%s) AND (m.evidence_status<>'available' OR m.verified_sha256 IS NULL)", (self.customer, observation["evidence_ids"])).fetchone()["n"]
            if unverified:
                raise Conflict("Evidence has not been independently verified")
            fid = ident()
            conn.execute("INSERT INTO fact_revision(customer_id,id,candidate_id,revision,scope,value,unit,actor,policy_version) VALUES(%s,%s,%s,1,%s,%s,%s,%s,%s)", (self.customer, fid, candidate["id"], scope, Jsonb(candidate["value"]), candidate["unit"], actor, policy_version))
            conn.execute("INSERT INTO confirmation(customer_id,id,fact_id,scope,actor,policy_version,evidence_ids) VALUES(%s,%s,%s,%s,%s,%s,%s)", (self.customer, ident(), fid, scope, actor, policy_version, observation["evidence_ids"]))
            conn.execute("UPDATE field_candidate SET status='confirmed',revision=revision+1 WHERE customer_id=%s AND id=%s", (self.customer, candidate["id"]))
            result = dict(fact_id=fid, candidate_id=candidate["id"], revision=1, scope=scope, value=candidate["value"], unit=candidate["unit"], evidence_ids=observation["evidence_ids"], actor=actor, policy_version=policy_version)
            self._emit(conn, "fact", fid, 1, result)
            return result
        return self.command(key, dict(candidate=candidate_id, revision=expected_revision, scope=scope, actor=actor, policy_version=policy_version), action)

    def confirm_usage(self, key, candidate_id, *, expected_revision, material_id, batch_id, expected_material_revision, actor, operation_id, reason, reuse_existing=False):
        if type(reuse_existing) is not bool:
            raise ValueError('Explicit existing-operation evidence link required')
        def action(conn):
            candidate = conn.execute("SELECT c.*,o.evidence_ids,o.occurred_at FROM field_candidate c JOIN observation o ON o.customer_id=c.customer_id AND o.id=c.observation_id WHERE c.customer_id=%s AND c.id=%s FOR UPDATE OF c", (self.customer, ident(candidate_id))).fetchone()
            if not candidate:
                raise Missing("Usage candidate outside customer scope")
            if candidate["status"] != "candidate" or candidate["revision"] != expected_revision or candidate["semantics"] != "actual_increment" or candidate["conflicts"]:
                raise Conflict("Only an unambiguous actual increment can be confirmed as use")
            if not candidate["evidence_ids"] or not candidate["occurred_at"]:
                raise Conflict("Usage requires traceable evidence and an established event time")
            unavailable=conn.execute("SELECT 1 FROM evidence_ref e JOIN media_asset m ON m.customer_id=e.customer_id AND m.id=e.media_id WHERE e.customer_id=%s AND e.id=ANY(%s) AND (m.evidence_status<>'available' OR m.verified_sha256 IS NULL OR e.media_version<>m.version) LIMIT 1",
                (self.customer,candidate['evidence_ids'])).fetchone()
            if unavailable:
                raise Conflict('Usage source evidence is unavailable or superseded')
            value = candidate["value"]
            if value.get("intent") != "actual_increment":
                raise Conflict("Planned/cumulative/correction speech is not an incremental debit")
            material = self._entity(conn, material_id, "material_unit")
            batch = self._entity(conn, batch_id, "process_batch")
            if value.get("batch_code") and value["batch_code"] != batch["code"]:
                raise Conflict("Confirmed target batch contradicts the spoken batch")
            type_id = material["metadata"].get("material_type_id")
            type_code = self._entity(conn, type_id, "material_type")["code"] if type_id else material["code"]
            if value.get("material_code") and value["material_code"] not in {type_code,material['code']}:
                raise Conflict("Confirmed material contradicts the spoken material code")
            if reuse_existing:
                state=self._states(conn,[material_id])[ident(material_id)]
                if state['revision']!=expected_material_revision:
                    raise Conflict('Material revision changed before existing use link')
                operation=conn.execute("SELECT * FROM operation WHERE customer_id=%s AND id=%s FOR UPDATE",
                                       (self.customer,ident(operation_id))).fetchone()
                q,u=amount(value['quantity'],candidate['unit'])
                lines=conn.execute("SELECT * FROM operation_line WHERE customer_id=%s AND operation_id=%s",
                                   (self.customer,ident(operation_id))).fetchall()
                if (not operation or operation['status']!='confirmed'
                        or operation['metadata'].get('effective_kind',operation['kind'])!='consume'
                        or str(operation['batch_id'])!=str(batch_id) or len(lines)!=1
                        or lines[0]['direction']!='input' or str(lines[0]['material_id'])!=str(material_id)
                        or lines[0]['quantity']!=q or lines[0]['unit']!=u):
                    raise Conflict('Existing operation does not represent the explicitly linked material use')
                result=dict(operation_id=operation['id'],revision=operation['revision'],kind='consume',
                            existing_operation_reused=True,quantity_ledger_written=False,
                            evidence_ids=candidate['evidence_ids'],batch_id=batch_id)
            else:
                usage_scope=candidate['association'].get('usage_scope')
                metadata={"candidate_id":str(candidate_id),"reason":reason}
                if usage_scope:metadata['usage_scope']=usage_scope
                result = self.operate(key + ":operation", "consume", [dict(material_id=material_id, quantity=value["quantity"], unit=candidate["unit"], expected_revision=expected_material_revision)], occurred_at=candidate["occurred_at"], actor=actor, operation_id=operation_id, batch_id=batch_id, run_id=usage_scope.get('run_id') if usage_scope else None, evidence_ids=candidate["evidence_ids"], metadata=metadata)
            fid = ident()
            conn.execute("INSERT INTO fact_revision(customer_id,id,candidate_id,revision,scope,value,unit,actor) VALUES(%s,%s,%s,1,'material_consumption',%s,%s,%s)", (self.customer, fid, candidate["id"], Jsonb(dict(value, material_id=str(material_id), batch_id=str(batch_id), operation_id=str(operation_id))), candidate["unit"], actor))
            conn.execute("INSERT INTO confirmation(customer_id,id,fact_id,scope,actor,evidence_ids) VALUES(%s,%s,%s,'material_consumption',%s,%s)", (self.customer, ident(), fid, actor, candidate["evidence_ids"]))
            conn.execute("UPDATE field_candidate SET status='confirmed',revision=revision+1 WHERE customer_id=%s AND id=%s", (self.customer, candidate["id"]))
            self._emit(conn, "fact", fid, 1, dict(fact_id=fid, scope="material_consumption", operation_id=operation_id, value=value))
            return dict(result, fact_id=str(fid))
        request=dict(candidate=candidate_id, expected=expected_revision, material=material_id, batch=batch_id, material_revision=expected_material_revision, actor=actor, operation=operation_id, reason=reason)
        if reuse_existing:request['reuse_existing']=True
        return self.command(key,request,action)
