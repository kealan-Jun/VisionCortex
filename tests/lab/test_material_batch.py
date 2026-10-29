"""New batch-query tests against the explicitly owned Unix PostgreSQL fixture.

All quantities and objects here are synthetic business examples, never inventory
issuance, customer stock or model accuracy evidence.
"""
from contextlib import contextmanager
from decimal import Decimal
import json
from uuid import UUID, uuid4

import pytest

from visioncortex.lab.domain import Conflict, Core, Missing
from visioncortex.lab.material_batch import MaterialBatch


AT = "2026-10-29T09:00:00+08:00"


def entity(core, kind, code):
    return core.register("entity-" + code, kind, code)["id"]


def material(core, code, initial="0", unit="g", lot=None):
    return core.register_material("material-" + code, code, unit, initial,
                                  metadata={"source_lot_id": lot} if lot else {}, occurred_at=AT)["material_id"]


def line(mid, quantity, unit="g"):
    return dict(material_id=mid, quantity=str(quantity), unit=unit)


def operate(core, key, kind, inputs, outputs=None, **kwargs):
    return core.operate(key, kind, inputs, outputs, occurred_at=AT, actor="fixture-authorized", **kwargs)


def flow(view, kind, unit="g"):
    rows = [r for r in view["flow_by_kind"] if r["kind"] == kind and r["unit"] == unit]
    assert len(rows) == 1
    return rows[0]


def d(value):
    return Decimal(value)


def emit_example(name, payload):
    # Examples are assertions in memory; test runs do not publish artifacts.
    json.dumps(payload, default=str)


def test_global_lot_and_target_batch_example(core):
    """100g G001; PB1 5+3g; PB2 12g; retransmission never debits again."""
    lot = entity(core, "source_lot", "LOT-G001")
    pb1, pb2 = [entity(core, "process_batch", code) for code in ("PB1", "PB2")]
    bag = material(core, "G001", "100", lot=lot)
    first = operate(core, "pb1-first", "consume", [line(bag, "5")], batch_id=pb1)
    assert operate(core, "pb1-first", "consume", [line(bag, "5")], batch_id=pb1) == first
    operate(core, "pb2-first", "consume", [line(bag, "12")], batch_id=pb2)
    operate(core, "pb1-second", "consume", [line(bag, "3")], batch_id=pb1)
    views = MaterialBatch(core)
    source, one, two = views.source_lot(lot), views.process_batch(pb1), views.process_batch(pb2)
    assert d(source["balance_by_unit"][0]["balance_total"]) == 80
    assert d(flow(one, "consume")["input_quantity"]) == 8
    assert d(flow(two, "consume")["input_quantity"]) == 12
    assert d(flow(source, "consume")["input_quantity"]) == 20
    assert d(source["ledger_by_unit"][0]["opening_delta"]) == 100
    assert d(source["ledger_by_unit"][0]["business_delta"]) == -20
    by_batch = {r["process_batch_id"]: d(r["input_quantity"]) for r in source["flow_by_process_batch"] if r["kind"] == "consume"}
    assert by_batch == {pb1: Decimal(8), pb2: Decimal(12)}
    assert len(flow(one, "consume")["operation_ids"]) == 2
    assert one["balance_scope"] == "current_stock_of_involved_materials_not_batch_owned_stock"
    assert source["net_physical_consumption"]["status"] == "not_computed"
    emit_example("GlobalLotExample.json", dict(source_lot=source, process_batch_pb1=one, process_batch_pb2=two))


def test_unknown_opening_and_units_never_summed_as_zero(core):
    lot = entity(core, "source_lot", "LOT-MULTI")
    pb = entity(core, "process_batch", "PREP")
    a = material(core, "known", "10.000000001", lot=lot)
    b = material(core, "unknown", None, lot=lot)
    c = material(core, "liquid", "0.1", unit="L", lot=lot)
    operate(core, "known-use", "consume", [line(a, "500.000001", "mg")], batch_id=pb)
    operate(core, "unknown-use", "consume", [line(b, "2")], batch_id=pb)
    operate(core, "liquid-use", "consume", [line(c, "25", "mL")], batch_id=pb)
    result = MaterialBatch(core).source_lot(lot)
    totals = {r["unit"]: r for r in result["balance_by_unit"]}
    assert totals["g"]["status"] == "partial" and totals["g"]["balance_total"] is None
    assert d(totals["g"]["known_balance_subtotal"]) == Decimal("9.5")
    assert totals["g"]["unknown_balance_count"] == 1
    assert totals["mL"]["status"] == "known" and d(totals["mL"]["balance_total"]) == 75
    assert result["coverage"]["unknown_balance_material_ids"] == [b]
    assert d(flow(result, "consume")["input_quantity"]) == Decimal("2.500000001")
    assert d(flow(result, "consume", "mL")["input_quantity"]) == 25
    assert result["coverage"]["status"] == "partial"


def test_all_unknown_balances_remain_unknown(core):
    lot = entity(core, "source_lot", "UNKNOWN")
    material(core, "unweighed", None, lot=lot)
    view = MaterialBatch(core).source_lot(lot)
    assert view["balance_by_unit"][0]["status"] == "unknown"
    assert view["balance_by_unit"][0]["balance_total"] is None
    assert view["ledger_by_unit"] == []


def test_mix_split_downstream_not_counted_twice_as_source_consumption(core):
    lot = entity(core, "source_lot", "RAW")
    pb = entity(core, "process_batch", "PREP")
    a, mix, child = material(core, "raw", "100", lot=lot), material(core, "mixed"), material(core, "child")
    operate(core, "mix", "mix", [line(a, 5)], [line(mix, 5)], batch_id=pb)
    operate(core, "split", "split", [line(mix, 5)], [line(child, 5)], batch_id=pb)
    operate(core, "child-use", "consume", [line(child, 2)], batch_id=pb)
    source, target = MaterialBatch(core).source_lot(lot), MaterialBatch(core).process_batch(pb)
    assert d(source["balance_by_unit"][0]["balance_total"]) == 95
    assert [f["kind"] for f in source["flow_by_kind"]] == ["mix", "opening"]
    assert d(flow(source, "mix")["input_quantity"]) == 5
    assert d(flow(target, "consume")["input_quantity"]) == 2
    assert d(flow(target, "split")["input_quantity"]) == 5
    assert d(target["ledger_by_unit"][0]["delta"]) == -2
    assert d(target["ledger_by_unit"][0]["core_used_delta"]) == 7
    assert source["unallocated_downstream"][0]["material_id"] == mix
    assert source["unallocated_downstream"][0]["allocated_from_this_lot"] is None
    assert child in {m["id"] for m in source["lineage"]["materials"]}
    assert "unallocated_downstream_provenance" in source["coverage"]["reasons"]
    assert target["net_physical_consumption"]["value"] is None
    emit_example("MixSplitExample.json", dict(source_lot=source, process_batch=target))


def test_explicit_split_children_balance_without_double_count(core):
    lot = entity(core, "source_lot", "LOT")
    source = material(core, "parent", "100", lot=lot)
    child = material(core, "child", "0", lot=lot)
    operate(core, "split", "split", [line(source, 10)], [line(child, 10)])
    operate(core, "consume-child", "consume", [line(child, 3)])
    view = MaterialBatch(core).source_lot(lot)
    assert d(view["balance_by_unit"][0]["balance_total"]) == 97
    assert d(flow(view, "consume")["input_quantity"]) == 3
    assert d(flow(view, "split")["input_quantity"]) == d(flow(view, "split")["output_quantity"]) == 10
    assert view["unallocated_downstream"] == []


def test_correction_chain_replaces_flows_but_reconciles_all_ledger_entries(core):
    lot = entity(core, "source_lot", "LOT")
    pb = entity(core, "process_batch", "PREP")
    mid = material(core, "bag", "100", lot=lot)
    original = operate(core, "initial-use", "consume", [line(mid, 10)], batch_id=pb)
    one = core.correct_operation("correct-one", original["operation_id"], [line(mid, 8)], expected_revision=1, actor="fixture-reviewer", reason="confirmed correction")
    two = core.correct_operation("correct-two", one["operation_id"], [line(mid, 7)], expected_revision=1, actor="fixture-reviewer", reason="second confirmed correction")
    view = MaterialBatch(core).process_batch(pb)
    assert d(flow(view, "consume")["input_quantity"]) == 7
    assert flow(view, "consume")["operation_ids"] == [two["operation_id"]]
    total = view["ledger_by_unit"][0]
    assert d(total["delta"]) == -7 and d(total["core_used_delta"]) == 7
    assert total["entry_count"] == 5 and total["reversal_entry_count"] == 2
    by_kind = {r["kind"]: r for r in view["ledger_by_recorded_kind"]}
    assert d(by_kind["consume"]["delta"]) == -10
    assert d(by_kind["correction"]["delta"]) == 3
    assert d(MaterialBatch(core).source_lot(lot)["balance_by_unit"][0]["balance_total"]) == 93
    assert len(view["corrections"]) == 3


def test_corrected_mix_preserves_measured_output_and_discrepancy(core):
    lot = entity(core, "source_lot", "LOT")
    pb = entity(core, "process_batch", "PREP")
    parent, child = material(core, "parent", "100", lot=lot), material(core, "child")
    original = operate(core, "mix", "mix", [line(parent, 10)], [line(child, 10)], batch_id=pb)
    core.correct_operation("correct", original["operation_id"], [line(parent, 8)], expected_revision=1, actor="fixture-reviewer", reason="remeasured input")
    view = MaterialBatch(core).process_batch(pb)
    assert d(flow(view, "mix")["input_quantity"]) == 8
    assert d(flow(view, "mix")["output_quantity"]) == 10
    assert d(view["ledger_by_unit"][0]["delta"]) == 2
    correction = next(c for c in view["corrections"] if c["effective"])
    assert d(correction["metadata"]["quantity_discrepancy"]) == 2


def test_core_correction_rejects_replacing_material_identity(core):
    a, b = material(core, "a", "10"), material(core, "b", "10")
    original = operate(core, "use", "consume", [line(a, 2)])
    with pytest.raises(Conflict, match="retain input identities"):
        core.correct_operation("replace-identity", original["operation_id"], [line(b, 2)], expected_revision=1, actor="fixture-reviewer", reason="not supported by current Core")


def test_ledger_only_reversal_is_included_even_without_matching_corrected_line(core):
    """Defensive legacy-import fixture; Core itself forbids identity replacement."""
    lot_a, lot_b = entity(core, "source_lot", "LOT-A"), entity(core, "source_lot", "LOT-B")
    a, b = material(core, "a", "20", lot=lot_a), material(core, "b", "20", lot=lot_b)
    original = operate(core, "original", "consume", [line(a, 5)])
    correction = uuid4()
    with core.db.transaction() as conn:
        old = conn.execute("SELECT * FROM quantity_ledger_entry WHERE customer_id=%s AND operation_id=%s", (core.customer, UUID(original["operation_id"]))).fetchone()
        conn.execute("INSERT INTO operation(customer_id,id,kind,occurred_at,actor,supersedes_id,metadata) VALUES(%s,%s,'correction',%s,'fixture-legacy-import',%s,'{\"effective_kind\":\"consume\"}')", (core.customer, correction, AT, UUID(original["operation_id"])))
        conn.execute("UPDATE operation SET status='corrected',revision=2 WHERE customer_id=%s AND id=%s", (core.customer, UUID(original["operation_id"])))
        conn.execute("INSERT INTO operation_line(customer_id,id,operation_id,material_id,direction,quantity,unit,quantity_basis,state_revision) VALUES(%s,%s,%s,%s,'input',3,'g','legacy-import',2)", (core.customer, uuid4(), correction, UUID(b)))
        conn.execute("INSERT INTO quantity_ledger_entry(customer_id,id,operation_id,material_id,delta,used_delta,unit,reverses_id) VALUES(%s,%s,%s,%s,5,-5,'g',%s)", (core.customer, uuid4(), correction, UUID(a), old["id"]))
        conn.execute("INSERT INTO quantity_ledger_entry(customer_id,id,operation_id,material_id,delta,used_delta,unit) VALUES(%s,%s,%s,%s,-3,3,'g')", (core.customer, uuid4(), correction, UUID(b)))
        conn.execute("UPDATE material_state SET balance=20,used=0,revision=revision+1 WHERE customer_id=%s AND material_id=%s", (core.customer, UUID(a)))
        conn.execute("UPDATE material_state SET balance=17,used=3,revision=revision+1 WHERE customer_id=%s AND material_id=%s", (core.customer, UUID(b)))
    first, second = MaterialBatch(core).source_lot(lot_a), MaterialBatch(core).source_lot(lot_b)
    assert d(first["ledger_by_unit"][0]["delta"]) == 20
    assert d(first["ledger_by_unit"][0]["core_used_delta"]) == 0
    assert not [r for r in first["flow_by_kind"] if r["kind"] == "consume"]
    assert str(correction) in {r["id"] for r in first["lineage"]["operations"]}
    assert d(second["ledger_by_unit"][0]["delta"]) == 17
    assert d(flow(second, "consume")["input_quantity"]) == 3


def test_return_transfer_loss_testing_are_distinct_and_return_is_not_refund(core):
    lot = entity(core, "source_lot", "LOT")
    pb = entity(core, "process_batch", "PREP")
    a, b = material(core, "source", "100", lot=lot), material(core, "return-receiver", "0", lot=lot)
    operate(core, "use", "consume", [line(a, 5)], batch_id=pb)
    operate(core, "return", "return", [line(a, 2)], [line(b, 2)], batch_id=pb)
    operate(core, "transfer", "transfer", [line(a, 1)], [line(a, 1)], batch_id=pb)
    operate(core, "loss", "loss", [line(a, 1)], batch_id=pb, metadata={"reason": "measured spill"})
    operate(core, "test", "destructive_test", [line(a, 3)], [line(a, 1)], batch_id=pb)
    view = MaterialBatch(core).process_batch(pb)
    assert d(flow(view, "consume")["input_quantity"]) == 5
    assert d(flow(view, "return")["input_quantity"]) == 2
    assert d(flow(view, "loss")["input_quantity"]) == 1
    assert d(flow(view, "transfer")["input_quantity"]) == d(flow(view, "transfer")["output_quantity"]) == 1
    assert d(view["ledger_by_unit"][0]["delta"]) == -8
    assert d(view["ledger_by_unit"][0]["core_used_delta"]) == 7
    assert len(view["coverage"]["unlinked_return_operation_ids"]) == 1
    assert view["net_physical_consumption"]["status"] == "not_computed"


def test_batch_id_exact_match_not_run_or_timestamp(core):
    pb1, pb2 = [entity(core, "process_batch", code) for code in ("PB1", "PB2")]
    run = entity(core, "experiment_run", "RUN")
    mid = material(core, "bag", "20")
    operate(core, "one", "consume", [line(mid, 2)], batch_id=pb1, run_id=run)
    operate(core, "two", "consume", [line(mid, 3)], batch_id=pb2, run_id=run)
    operate(core, "unbound", "consume", [line(mid, 1)], run_id=run)
    view = MaterialBatch(core).process_batch(pb1)
    assert d(flow(view, "consume")["input_quantity"]) == 2
    assert view["coverage"]["operation_count"] == 1


def test_empty_catalogue_does_not_issue_or_create_materials(core):
    lot, pb = entity(core, "source_lot", "EMPTY-LOT"), entity(core, "process_batch", "EMPTY-PREP")
    entity(core, "material_type", "TYPE-ONLY")
    views = MaterialBatch(core)
    for result in (views.source_lot(lot), views.process_batch(pb)):
        assert result["coverage"]["status"] == "empty"
        assert result["member_balances"] == result["flow_by_kind"] == result["ledger_by_unit"] == []
        assert result["balance_by_unit"] == []
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM material_state WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0


@pytest.mark.parametrize("method,wrong_kind", [("source_lot", "process_batch"), ("process_batch", "source_lot")])
def test_customer_and_identity_kind_isolation(core, db, method, wrong_kind):
    wrong = entity(core, wrong_kind, "WRONG")
    with pytest.raises(Missing):
        getattr(MaterialBatch(core), method)(wrong)
    right_kind = "source_lot" if method == "source_lot" else "process_batch"
    private = entity(core, right_kind, "PRIVATE")
    other = Core(db, Core.customer_instance(db, "other-" + str(uuid4()), scope="fixture")["id"])
    with pytest.raises(Missing):
        getattr(MaterialBatch(other), method)(private)
    with pytest.raises(Missing):
        getattr(MaterialBatch(core), method)(str(uuid4()))


def test_unregistered_state_and_foreign_lot_metadata_do_not_resolve(core, db):
    lot = entity(core, "source_lot", "LOT")
    bare = core.register("bare", "material_unit", "IDENTITY-ONLY", {"source_lot_id": lot})["id"]
    result = MaterialBatch(core).source_lot(lot)
    assert result["coverage"]["missing_state_material_ids"] == [bare]
    assert result["balance_by_unit"] == []
    foreign = Core(db, Core.customer_instance(db, "foreign-" + str(uuid4()), scope="fixture")["id"])
    secret = entity(foreign, "source_lot", "SECRET-NAME")
    # Generic register permits arbitrary metadata; it must never make cross-tenant
    # metadata a valid source binding in a read projection.
    mid = core.register("bad-source", "material_unit", "BAD-SOURCE", {"source_lot_id": secret})["id"]
    pb = entity(core, "process_batch", "PREP")
    with core.db.transaction() as conn:
        conn.execute("INSERT INTO material_state(customer_id,material_id,unit,balance) VALUES(%s,%s,'g',10)", (core.customer, UUID(mid)))
    operate(core, "use-unresolved", "consume", [line(mid, 1)], batch_id=pb)
    target = MaterialBatch(core).process_batch(pb)
    assert target["lineage"]["source_lots"] == []
    assert target["member_balances"][0]["source_lot_id"] is None
    assert "SECRET-NAME" not in json.dumps(target)


def test_limits_fail_closed_instead_of_truncating_totals(core):
    lot = entity(core, "source_lot", "LOT")
    material(core, "a", "1", lot=lot)
    material(core, "b", "2", lot=lot)
    with pytest.raises(Conflict, match="no partial total"):
        MaterialBatch(core, max_records=1).source_lot(lot)
    with pytest.raises(ValueError):
        MaterialBatch(core, max_records=True)


def test_queries_are_repeatable_read_and_no_business_writes(core, monkeypatch):
    lot = entity(core, "source_lot", "LOT")
    pb = entity(core, "process_batch", "PREP")
    mid = material(core, "bag", "20", lot=lot)
    operate(core, "one", "consume", [line(mid, 2)], batch_id=pb)
    original = core.db.transaction
    seen, states = [], []
    tables = ("entity", "material_state", "operation", "operation_line", "quantity_ledger_entry", "command_receipt", "outbox")
    def counts():
        with original(readonly=True) as conn:
            return {t: conn.execute("SELECT count(*) AS n FROM " + t + " WHERE customer_id=%s", (core.customer,)).fetchone()["n"] for t in tables}
    before = counts()
    class Connection:
        def __init__(self, conn):
            self.conn = conn
        def execute(self, sql, args=None):
            assert sql.lstrip().startswith(("SELECT ", "SET TRANSACTION ", "WITH RECURSIVE "))
            seen.append(sql)
            rows = self.conn.execute(sql, args)
            if sql.startswith("SELECT * FROM entity WHERE"):
                states.append((self.conn.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"], self.conn.execute("SHOW transaction_isolation").fetchone()["transaction_isolation"]))
            return rows
    @contextmanager
    def readonly(*, readonly=False):
        assert readonly is True
        with original(readonly=True) as conn:
            yield Connection(conn)
    monkeypatch.setattr(core.db, "transaction", readonly)
    first = MaterialBatch(core).source_lot(lot)
    assert MaterialBatch(core).source_lot(lot) == first
    MaterialBatch(core).process_batch(pb)
    assert states == [("on", "repeatable read")] * 3
    assert seen and counts() == before


def test_missing_ledger_operation_not_in_business_flow(core):
    pb = entity(core, "process_batch", "PREP")
    mid = material(core, "bag", "20")
    op = uuid4()
    with core.db.transaction() as conn:
        conn.execute("INSERT INTO operation(customer_id,id,kind,batch_id,occurred_at,actor) VALUES(%s,%s,'consume',%s,%s,'fixture-corrupt-import')", (core.customer, op, UUID(pb), AT))
        conn.execute("INSERT INTO operation_line(customer_id,id,operation_id,material_id,direction,quantity,unit,quantity_basis,state_revision) VALUES(%s,%s,%s,%s,'input',5,'g','unbacked-import',1)", (core.customer, uuid4(), op, UUID(mid)))
    result = MaterialBatch(core).process_batch(pb)
    assert result["flow_by_kind"] == []
    assert result["coverage"]["missing_ledger_operation_ids"] == [str(op)]
    assert result["coverage"]["status"] == "partial"
