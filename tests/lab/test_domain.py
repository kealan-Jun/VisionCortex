from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from visioncortex.lab.domain import Conflict, Core, Missing
from visioncortex.lab.projections import Projections

AT = "2026-10-23T10:00:00+08:00"


def material(core, code, initial):
    return core.register_material("register-" + code, code, "g", initial, occurred_at=AT)["material_id"]


def line(mid, q):
    return dict(material_id=mid, quantity=str(q), unit="g")


def operate(core, key, kind, ins, outs=None, **kwargs):
    return core.operate(key, kind, ins, outs, occurred_at=AT, actor="fixture-operator", **kwargs)


def balance(core, mid):
    return Decimal(core.balance(mid)["balance"])


def test_full_lifecycle_conserves_quantity(core):
    a, b = material(core, "A", "100"), material(core, "B", "50")
    m, work, retained, p = [material(core, n, "0") for n in ("M", "work", "retained", "P")]
    operate(core, "mix-one", "mix", [line(a, 20), line(b, 10)], [line(m, 30)])
    operate(core, "split", "split", [line(m, 30)], [line(work, 25), line(retained, 5)])
    operate(core, "destroy", "destructive_test", [line(work, 8)])
    operate(core, "nondestructive", "non_destructive_test", [line(retained, 1)], [line(retained, 1)])
    operate(core, "mix-two", "mix", [line(work, 10), line(a, 2)], [line(p, 12)])
    assert [balance(core, x) for x in (a, b, m, work, retained, p)] == [78, 40, 0, 7, 5, 12]
    assert sum(balance(core, x) for x in (a, b, m, work, retained, p)) + 8 == 150
    graph = Projections(core).lineage(p)
    assert len(graph["operations"]) >= 5
    assert all("state_revision" in l for l in graph["lines"])


def test_command_and_business_identity_separate(core):
    mid = material(core, "A", "20")
    result = operate(core, "one", "consume", [line(mid, 5)])
    for _ in range(3):
        assert operate(core, "one", "consume", [line(mid, 5)]) == result
    with pytest.raises(Conflict):
        operate(core, "different-message", "consume", [line(mid, 5)], operation_id=result["operation_id"])
    operate(core, "actually-second-five", "consume", [line(mid, 5)])
    assert balance(core, mid) == 10


def test_unknown_initial_does_not_become_zero(core):
    mid = material(core, "unknown", None)
    operate(core, "five", "consume", [line(mid, 5)])
    result = core.balance(mid)
    assert result["balance"] is None and result["balance_status"] == "unknown"
    assert Decimal(result["used"]) == 5


def test_correction_preserves_measured_output_and_marks_difference(core):
    a, b, m, child = [material(core, n, v) for n, v in (("A", "100"), ("B", "50"), ("M", "0"), ("child", "0"))]
    original = operate(core, "mix", "mix", [line(a, 20), line(b, 10)], [line(m, 30)])
    operate(core, "downstream", "split", [line(m, 5)], [line(child, 5)])
    result = core.correct_operation("correct", original["operation_id"], [line(a, 18), line(b, 10)], expected_revision=1, actor="reviewer", reason="verified input correction")
    assert balance(core, a) == 82 and balance(core, m) == 25 and balance(core, child) == 5
    assert result["metadata"]["quantity_discrepancy"] == "2.000000000"
    assert result["metadata"]["downstream_operations"]


def test_concurrent_debits_cannot_overspend(core):
    mid = material(core, "A", "100")
    def spend(i):
        try:
            operate(core, "parallel-" + str(i), "consume", [line(mid, 20)])
            return True
        except Conflict:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(spend, range(12)))
    assert sum(results) == 5 and balance(core, mid) == 0


def test_operation_ledger_outbox_rollback_together(core, monkeypatch):
    mid = material(core, "A", "10")
    emit = core._emit
    def fail(conn, kind, *args):
        if kind == "operation":
            raise RuntimeError("fixture interrupted before commit")
        return emit(conn, kind, *args)
    monkeypatch.setattr(core, "_emit", fail)
    with pytest.raises(RuntimeError):
        operate(core, "broken", "consume", [line(mid, 4)])
    assert balance(core, mid) == 10
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM command_receipt WHERE customer_id=%s AND key='broken'", (core.customer,)).fetchone()["n"] == 0


def test_units_and_reservations(core):
    mid = material(core, "A", "10")
    res = core.reserve("reserve", mid, "8", "g", reason="next batch")
    with pytest.raises(Conflict):
        operate(core, "too-much", "consume", [line(mid, 3)])
    core.reserve("release", mid, "8", "g", reason="cancelled", release_id=res["id"])
    operate(core, "milligrams", "consume", [dict(material_id=mid, quantity="500", unit="mg")])
    assert balance(core, mid) == Decimal("9.5")
    with pytest.raises(Conflict):
        operate(core, "wrong-dimension", "consume", [dict(material_id=mid, quantity="1", unit="mL")])
    with pytest.raises(ValueError):
        operate(core, "float", "consume", [dict(material_id=mid, quantity=0.1, unit="g")])


def test_temporal_tags_late_events_and_permanent_code(core):
    a, b = material(core, "old", "10"), material(core, "new", "20")
    tag = core.register("tag", "tag", "Q-M001", {"mode": "reusable"})["id"]
    core.assign_tag("old-binding", tag, a, "2026-10-23T09:00:00+08:00", actor="registrar", reason="old contents")
    core.assign_tag("new-binding", tag, b, "2026-10-23T11:00:00+08:00", actor="registrar", reason="new contents")
    assert core.resolve_tag(tag, "2026-10-23T10:58:00+08:00")["target_id"] == a
    assert core.resolve_tag(tag, "2026-10-23T11:03:00+08:00")["target_id"] == b
    assert core.resolve_tag(tag, "2026-10-23T10:58:00+08:00", "2026-10-23T11:02:00+08:00")["association_status"] == "needs_review"
    permanent = core.register("permanent", "tag", "material-permanent-code", {"mode": "permanent"})["id"]
    core.assign_tag("permanent-a", permanent, a, AT, actor="registrar", reason="identity")
    with pytest.raises(Conflict):
        core.assign_tag("permanent-b", permanent, b, "2026-10-24T10:00:00+08:00", actor="registrar", reason="replacement")


def test_oven_can_hold_multiple_batches_across_midnight(core):
    oven = core.register("oven", "instrument", "OVEN")
    b1, b2 = [core.register(n, "process_batch", n)["id"] for n in ("PB-01", "PB-02")]
    for n, batch in enumerate((b1, b2)):
        core.equipment_use("use-" + str(n), oven["id"], batch, AT, actor="operator", position="shelf-" + str(n))
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM equipment_use WHERE customer_id=%s AND valid_to IS NULL", (core.customer,)).fetchone()["n"] == 2


def test_customer_cannot_read_other_material(core, db):
    mid = material(core, "private", "10")
    other = Core.customer_instance(db, "other-" + str(core.customer), scope="fixture")
    with pytest.raises(Missing):
        Core(db, other["id"]).balance(mid)
