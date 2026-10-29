"""Read-only, customer-scoped batch views over confirmed Core business records.

A source lot identifies provenance; a process batch identifies the target of an
operation. Neither recognition candidates nor a reusable QR label create stock.
Gross operation flows, ledger changes and current stock are deliberately separate:
Core's used counter includes mixing and cannot be called physical consumption.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext

from .domain import Conflict, clean


SCHEMA = "vc-lab-material-batch/1"
_ZERO = Decimal(0)
_ACTIVE = """o.status='confirmed' AND NOT EXISTS (
    SELECT 1 FROM operation successor WHERE successor.customer_id=o.customer_id
    AND successor.supersedes_id=o.id AND successor.status IN ('confirmed','corrected'))"""


class MaterialBatch:
    """Current committed business snapshot; no writes, allocation or inferred lots.

    ``source_lot(id)`` and ``process_batch(id)`` return JSON-compatible dicts.
    Limits fail closed rather than returning an apparently complete truncated sum.
    These views do not establish recording or physical inventory completeness.
    """

    def __init__(self, core, *, max_records=10000):
        if isinstance(max_records, bool) or not isinstance(max_records, int) or not 1 <= max_records <= 100000:
            raise ValueError("max_records must be an integer in [1,100000]")
        self.core, self.max_records = core, max_records

    def _rows(self, conn, sql, args):
        rows = conn.execute(sql + " LIMIT %s", (*args, self.max_records + 1)).fetchall()
        if len(rows) > self.max_records:
            raise Conflict("Batch view exceeds its record budget; no partial total was returned")
        return rows

    def source_lot(self, lot_id):
        return self._view(lot_id, "source_lot")

    def process_batch(self, batch_id):
        return self._view(batch_id, "process_batch")

    def _view(self, entity_id, kind):
        with self.core.db.transaction(readonly=True) as conn, localcontext() as decimal_context:
            decimal_context.prec = 64
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            root = self.core._entity(conn, entity_id, kind)
            customer = self.core.customer
            if kind == "source_lot":
                members = self._rows(conn, "SELECT id FROM entity WHERE customer_id=%s AND kind='material_unit' AND metadata->>'source_lot_id'=%s ORDER BY id", (customer, str(root["id"])))
                selected = {row["id"] for row in members}
                # UNION (not UNION ALL) terminates material/operation cycles. This
                # follows identities, never quantities or proportional provenance.
                reachable = self._rows(conn, """WITH RECURSIVE reachable(mid) AS (
                    SELECT unnest(%s::uuid[]) UNION
                    SELECT outp.material_id FROM reachable r
                    JOIN operation_line inp ON inp.customer_id=%s AND inp.material_id=r.mid AND inp.direction='input'
                    JOIN operation o ON o.customer_id=inp.customer_id AND o.id=inp.operation_id
                    JOIN operation_line outp ON outp.customer_id=o.customer_id AND outp.operation_id=o.id AND outp.direction='output'
                    WHERE """ + _ACTIVE + ") SELECT mid FROM reachable ORDER BY mid", (sorted(selected), customer))
                scope = [row["mid"] for row in reachable]
                operations = self._rows(conn, """SELECT o.*, (""" + _ACTIVE + """ ) AS effective
                    FROM operation o WHERE o.customer_id=%s AND (EXISTS (
                    SELECT 1 FROM operation_line l WHERE l.customer_id=o.customer_id AND l.operation_id=o.id AND l.material_id=ANY(%s))
                    OR EXISTS (SELECT 1 FROM quantity_ledger_entry q WHERE q.customer_id=o.customer_id AND q.operation_id=o.id AND q.material_id=ANY(%s)))
                    ORDER BY o.occurred_at,o.id""", (customer, scope, scope))
                # An unknown opening has no line/ledger linking it to the material.
                # The authoritative nullable material_state remains explicit below.
            else:
                selected = None
                operations = self._rows(conn, "SELECT o.*, (" + _ACTIVE + ") AS effective FROM operation o WHERE o.customer_id=%s AND o.batch_id=%s ORDER BY o.occurred_at,o.id", (customer, root["id"]))
                scope = []
            op_ids = [op["id"] for op in operations]
            lines = self._rows(conn, "SELECT * FROM operation_line WHERE customer_id=%s AND operation_id=ANY(%s) ORDER BY operation_id,direction,material_id", (customer, op_ids))
            ledger = self._rows(conn, "SELECT * FROM quantity_ledger_entry WHERE customer_id=%s AND operation_id=ANY(%s) ORDER BY created_at,id", (customer, op_ids))
            material_ids = sorted({line["material_id"] for line in lines} | {entry["material_id"] for entry in ledger} | set(scope) | (selected or set()))
            materials = self._rows(conn, """SELECT e.id,e.code,e.metadata,e.revision,
                s.unit,s.balance,s.used AS core_used_counter,s.reserved,s.revision AS state_revision,s.state
                FROM entity e LEFT JOIN material_state s ON s.customer_id=e.customer_id AND s.material_id=e.id
                WHERE e.customer_id=%s AND e.kind='material_unit' AND e.id=ANY(%s) ORDER BY e.id""", (customer, material_ids))
            lot_ids = [m["metadata"].get("source_lot_id") for m in materials if isinstance(m["metadata"].get("source_lot_id"), str)]
            # Compare metadata as text so malformed supplied metadata cannot cause a
            # uuid cast or a lookup outside this customer's namespace.
            lots = self._rows(conn, "SELECT id,code FROM entity WHERE customer_id=%s AND kind='source_lot' AND id::text=ANY(%s) ORDER BY id", (customer, lot_ids))
            lot_map = {str(lot["id"]): lot for lot in lots}
            for m in materials:
                source = m["metadata"].get("source_lot_id")
                resolved = isinstance(source, str) and source in lot_map
                m["source_lot_id"] = source if resolved else None
                m["source_lot_resolution"] = "explicit_registered_source_lot" if resolved else "unresolved"
                m["balance_status"] = "known" if m["balance"] is not None else "unknown"
            return clean(self._assemble(root, kind, selected, operations, lines, ledger, materials, lots))

    def _assemble(self, root, kind, selected, operations, lines, ledger, materials, lots):
        by_op = {op["id"]: op for op in operations}
        lines_by_op, ledger_by_op = defaultdict(list), defaultdict(list)
        for line in lines:
            lines_by_op[line["operation_id"]].append(line)
        for entry in ledger:
            ledger_by_op[entry["operation_id"]].append(entry)
        relevant = lambda mid: selected is None or mid in selected
        flow = defaultdict(lambda: dict(input_quantity=_ZERO, output_quantity=_ZERO, operation_ids=set()))
        batch_flow = defaultdict(lambda: dict(input_quantity=_ZERO, output_quantity=_ZERO, operation_ids=set()))
        missing_ledger, effective_ids = [], []
        for op in operations:
            if not op["effective"]:
                continue
            if not ledger_by_op[op["id"]]:
                missing_ledger.append(op["id"])
                continue
            effective_ids.append(op["id"])
            effective_kind = op["metadata"].get("effective_kind", op["kind"]) if op["kind"] == "correction" else op["kind"]
            for line in lines_by_op[op["id"]]:
                if relevant(line["material_id"]):
                    item = flow[(effective_kind, line["unit"])]
                    item[line["direction"] + "_quantity"] += line["quantity"]
                    item["operation_ids"].add(op["id"])
                    batch_item = batch_flow[(str(op["batch_id"]) if op["batch_id"] else None, effective_kind, line["unit"])]
                    batch_item[line["direction"] + "_quantity"] += line["quantity"]
                    batch_item["operation_ids"].add(op["id"])
        flows = [dict(kind=k, unit=u, **{key: sorted(value) if isinstance(value, set) else value for key, value in item.items()}) for (k, u), item in sorted(flow.items())]
        batch_flows = [dict(process_batch_id=batch, kind=k, unit=u,
                            target_resolution="explicit_operation_batch_id" if batch else "no_target_batch_recorded",
                            **{key: sorted(value) if isinstance(value, set) else value for key, value in item.items()})
                       for (batch, k, u), item in sorted(batch_flow.items(), key=lambda pair: (pair[0][0] or "", pair[0][1], pair[0][2]))]
        ledger_totals = defaultdict(lambda: dict(delta=_ZERO, opening_delta=_ZERO, business_delta=_ZERO, core_used_delta=_ZERO, entry_count=0, reversal_entry_count=0))
        ledger_kinds = defaultdict(lambda: dict(delta=_ZERO, core_used_delta=_ZERO, entry_count=0, reversal_entry_count=0))
        for entry in ledger:
            if relevant(entry["material_id"]):
                item = ledger_totals[entry["unit"]]
                item["delta"] += entry["delta"]
                ledger_kind = by_op[entry["operation_id"]]["kind"]
                item["opening_delta" if ledger_kind == "opening" else "business_delta"] += entry["delta"]
                item["core_used_delta"] += entry["used_delta"]
                item["entry_count"] += 1
                item["reversal_entry_count"] += entry["reverses_id"] is not None
                typed = ledger_kinds[(ledger_kind, entry["unit"])]
                typed["delta"] += entry["delta"]
                typed["core_used_delta"] += entry["used_delta"]
                typed["entry_count"] += 1
                typed["reversal_entry_count"] += entry["reverses_id"] is not None
        balances = [m for m in materials if relevant(m["id"])]
        balance_totals = defaultdict(lambda: dict(member_count=0, known_balance_count=0, unknown_balance_count=0, known_balance_subtotal=_ZERO, reserved=_ZERO, core_used_counter=_ZERO))
        for member in balances:
            if member["state_revision"] is None:
                continue
            total = balance_totals[member["unit"]]
            total["member_count"] += 1
            total["reserved"] += member["reserved"]
            total["core_used_counter"] += member["core_used_counter"]
            if member["balance"] is None:
                total["unknown_balance_count"] += 1
            else:
                total["known_balance_count"] += 1
                total["known_balance_subtotal"] += member["balance"]
        for total in balance_totals.values():
            total["status"] = ("known" if not total["unknown_balance_count"] else "partial" if total["known_balance_count"] else "unknown")
            total["balance_total"] = total["known_balance_subtotal"] if total["status"] == "known" else None
            total["available_total"] = total["balance_total"] - total["reserved"] if total["balance_total"] is not None else None
        # Graph is bipartite operation<->material, retaining measured line amounts
        # and revisions. No implicit per-source allocation is added to its edges.
        graph = dict(materials=materials, source_lots=lots, operations=operations, lines=lines,
                     ledger_entries=ledger, allocation="none; edges are recorded business lines")
        unallocated = []
        if selected is not None:
            by_material = {m["id"]: m for m in materials}
            for op in operations:
                if op["id"] not in effective_ids:
                    continue
                inputs = [l for l in lines_by_op[op["id"]] if l["direction"] == "input" and l["material_id"] in selected]
                if not inputs:
                    continue
                outputs = [l for l in lines_by_op[op["id"]] if l["direction"] == "output" and l["material_id"] not in selected]
                for out in outputs:
                    unallocated.append(dict(operation_id=op["id"], material_id=out["material_id"],
                                            recorded_output_quantity=out["quantity"], unit=out["unit"],
                                            explicit_output_source_lot_id=by_material[out["material_id"]]["source_lot_id"],
                                            allocated_from_this_lot=None,
                                            reason="downstream_identity_has_no_explicit_membership_in_this_source_lot"))
        missing_states = [m["id"] for m in balances if m["state_revision"] is None]
        unresolved_lots = [m["id"] for m in materials if m["source_lot_resolution"] == "unresolved"]
        corrections = [dict(operation_id=op["id"], supersedes_id=op["supersedes_id"], status=op["status"],
                            effective=op["effective"], effective_kind=op["metadata"].get("effective_kind"),
                            metadata=op["metadata"], ledger_entry_ids=[l["id"] for l in ledger_by_op[op["id"]]])
                       for op in operations if op["kind"] == "correction" or op["status"] == "corrected"]
        returns = [op["id"] for op in operations if op["effective"] and op["kind"] == "return" and ledger_by_op[op["id"]]]
        unknown_balances = [m["id"] for m in balances if m["balance"] is None]
        reasons = []
        if not balances and not operations:
            reasons.append("no_recorded_materials_or_operations")
        if unknown_balances:
            reasons.append("unknown_member_balance")
        if missing_states:
            reasons.append("material_identity_without_registered_quantity_state")
        if missing_ledger:
            reasons.append("effective_operation_without_ledger")
        if unresolved_lots:
            reasons.append("material_source_lot_unresolved")
        if unallocated:
            reasons.append("unallocated_downstream_provenance")
        if returns:
            reasons.append("return_not_automatically_linked_to_prior_consumption")
        return dict(schema=SCHEMA, customer_id=self.core.customer, kind=kind, identity=root,
                    read_only=True, snapshot_isolation="repeatable read", time_scope="current_committed_business_state",
                    flow_scope="explicit_source_lot_members" if selected is not None else "operation.batch_id_exact_match",
                    flow_by_kind=flows, flow_by_process_batch=batch_flows,
                    ledger_by_unit=[dict(unit=unit, **total) for unit, total in sorted(ledger_totals.items())],
                    ledger_by_recorded_kind=[dict(kind=k, unit=u, **total) for (k, u), total in sorted(ledger_kinds.items())],
                    member_balances=balances,
                    balance_by_unit=[dict(unit=unit, **total) for unit, total in sorted(balance_totals.items())],
                    balance_scope="explicit_source_lot_members" if selected is not None else "current_stock_of_involved_materials_not_batch_owned_stock",
                    corrections=corrections, unallocated_downstream=unallocated,
                    net_physical_consumption=dict(status="not_computed", value=None,
                        reason="gross flows and Core.used are not physical consumption; returns lack a validated consumption-reversal contract"),
                    coverage=dict(status="empty" if not balances and not operations else "partial" if reasons else "known_recorded_scope",
                        reasons=reasons, capture_or_physical_inventory_completeness="unknown",
                        explicit_member_count=len(selected) if selected is not None else None,
                        operation_count=len(operations), effective_ledger_operation_count=len(effective_ids),
                        missing_ledger_operation_ids=missing_ledger, missing_state_material_ids=missing_states,
                        unknown_balance_material_ids=unknown_balances, unresolved_source_lot_material_ids=unresolved_lots,
                        unlinked_return_operation_ids=returns),
                    lineage=graph,
                    semantics=dict(quantity="exact Decimal strings in base units; units are never added together",
                        flow="active business-line gross input/output per kind; superseded lines excluded",
                        ledger="all scoped entries including historical corrections and reversals; summed exactly once",
                        core_used="Core counter includes consume, mix and destructive tests; not global physical depletion",
                        split="identity transformation, not additional consumption or proportional source allocation",
                        batch_stock="a process batch target does not own the entire remaining stock of its input materials",
                        evidence="recognition candidates and material-type catalogues do not create business quantities"))
