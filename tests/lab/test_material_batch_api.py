"""Read-only HTTP/CLI integration on the explicitly owned Unix PostgreSQL fixture.

All identities, quantities and credentials below are synthetic fixtures. No private
runtime.env, live intake, model invocation or customer stock is accessed.
"""
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
import json
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from visioncortex.lab import cli
from visioncortex.lab.api import create_app
from visioncortex.lab.domain import Core, Missing
from ._fixtures import db as evidence_db, settings as evidence_settings


@pytest.fixture(scope="session", name="db")
def fixture_db():
    # This wrapper skips unless the exact owned Unix socket is explicitly set.
    return evidence_db.__wrapped__()


@pytest.fixture
def settings(core, tmp_path):
    return evidence_settings.__wrapped__(core, tmp_path)


@pytest.fixture
def sample(core):
    lot = core.register("lot", "source_lot", "FIXTURE-LOT")["id"]
    batch = core.register("batch", "process_batch", "FIXTURE-BATCH")["id"]
    material = core.register_material(
        "material", "FIXTURE-MATERIAL", "g", "20.125",
        metadata={"source_lot_id": lot}, occurred_at="2026-10-29T09:00:00+08:00",
    )["material_id"]
    core.operate(
        "consume", "consume", [dict(material_id=material, quantity="2.005", unit="g")],
        occurred_at="2026-10-29T09:01:00+08:00", actor="synthetic fixture",
        batch_id=batch,
    )
    return dict(lot=lot, batch=batch, material=material)


@pytest.fixture
def client(core, settings):
    with TestClient(create_app(core, settings)) as result:
        yield result


KINDS = [("source-lots", "lot", "source_lot"), ("process-batches", "batch", "process_batch")]


def url(kind, value):
    return f"/internal/materials/{kind}/{value}/summary"


def auth(settings):
    return {"Authorization": "Bearer " + settings.token}


@pytest.mark.parametrize("route,key,kind", KINDS)
def test_summary_requires_existing_bearer_token(client, settings, sample, core, route, key, kind):
    path = url(route, sample[key])
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong-fixture-token"}).status_code == 401
    response = client.get(path, headers=auth(settings))
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "vc-lab-material-batch/1" and body["kind"] == kind
    assert body["identity"]["id"] == sample[key] and body["customer_id"] == str(core.customer)
    assert body["read_only"] is True and body["snapshot_isolation"] == "repeatable read"
    flow = next(item for item in body["flow_by_kind"] if item["kind"] == "consume")
    assert Decimal(flow["input_quantity"]) == Decimal("2.005")
    assert body["net_physical_consumption"]["value"] is None


@pytest.mark.parametrize("route,key,kind", KINDS)
def test_summary_cannot_read_other_customer(client, settings, core, route, key, kind):
    other_id = Core.customer_instance(core.db, "batch-api-other-" + str(uuid4()), scope="fixture")["id"]
    other = Core(core.db, other_id)
    foreign = other.register("foreign", kind, "FOREIGN")["id"]
    response = client.get(url(route, foreign), headers=auth(settings), params={"customer_id": other_id})
    assert response.status_code == 404
    assert "FOREIGN" not in response.text and str(other_id) not in response.text


@pytest.mark.parametrize("route,key,kind", KINDS)
def test_source_and_target_identity_kinds_are_not_interchangeable(client, settings, sample, route, key, kind):
    wrong = sample["batch" if key == "lot" else "lot"]
    assert client.get(url(route, wrong), headers=auth(settings)).status_code == 404
    assert client.get(url(route, sample["material"]), headers=auth(settings)).status_code == 404


@pytest.mark.parametrize("route,key,kind", KINDS)
def test_absent_or_malformed_identity_never_returns_empty_success(client, settings, route, key, kind):
    assert client.get(url(route, uuid4()), headers=auth(settings)).status_code == 404
    assert client.get(url(route, "not-a-uuid"), headers=auth(settings)).status_code == 422


@pytest.mark.parametrize("route,key,kind", KINDS)
def test_registered_identity_without_business_data_remains_unknown(client, settings, core, route, key, kind):
    identity = core.register("empty", kind, "EMPTY-FIXTURE")["id"]
    response = client.get(url(route, identity), headers=auth(settings))
    assert response.status_code == 200
    body = response.json()
    assert body["flow_by_kind"] == body["ledger_by_unit"] == body["balance_by_unit"] == []
    assert body["coverage"]["status"] == "empty"
    assert body["coverage"]["capture_or_physical_inventory_completeness"] == "unknown"
    assert body["net_physical_consumption"] == dict(
        status="not_computed", value=None,
        reason="gross flows and Core.used are not physical consumption; returns lack a validated consumption-reversal contract",
    )


def test_unknown_opening_quantity_survives_actual_get(client, settings, core):
    lot = core.register("lot", "source_lot", "UNKNOWN-LOT")["id"]
    core.register_material("unknown", "UNKNOWN-MATERIAL", "g", None, metadata={"source_lot_id": lot})
    response = client.get(url("source-lots", lot), headers=auth(settings))
    assert response.status_code == 200
    body = response.json()
    assert body["member_balances"][0]["balance"] is None
    assert body["member_balances"][0]["balance_status"] == "unknown"
    assert body["balance_by_unit"][0]["status"] == "unknown"
    assert body["balance_by_unit"][0]["balance_total"] is None
    assert body["balance_by_unit"][0]["available_total"] is None


def snapshot(core):
    tables = ("entity", "material_state", "operation", "operation_line", "quantity_ledger_entry",
              "reservation", "command_receipt", "outbox")
    with core.db.transaction(readonly=True) as conn:
        return {table: conn.execute(
            f"SELECT to_jsonb(t) AS row FROM {table} t WHERE customer_id=%s ORDER BY to_jsonb(t)::text",
            (core.customer,),
        ).fetchall() for table in tables}


def test_gets_use_readonly_transactions_and_leave_business_rows_identical(client, settings, sample, core, monkeypatch):
    before = snapshot(core)
    original = core.db.transaction
    calls = []

    @contextmanager
    def transaction(*, readonly=False):
        calls.append(readonly)
        with original(readonly=readonly) as conn:
            yield conn

    monkeypatch.setattr(core.db, "transaction", transaction)
    for route, key, _ in KINDS:
        for _ in range(2):
            assert client.get(url(route, sample[key]), headers=auth(settings)).status_code == 200
    assert calls == [True] * 4
    assert snapshot(core) == before


@pytest.mark.parametrize("kind,key", [("source-lot", "lot"), ("process-batch", "batch")])
def test_cli_uses_existing_customer_and_same_readonly_snapshot(core, settings, sample, monkeypatch, capsys, kind, key):
    with core.db.transaction(readonly=True) as conn:
        code = conn.execute("SELECT code FROM customer_instance WHERE id=%s", (core.customer,)).fetchone()["code"]
    fixture_settings = replace(settings, customer_code=code)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: fixture_settings))
    monkeypatch.setattr("sys.argv", ["visioncortex.lab", "material-batch-summary", kind, sample[key]])
    before = snapshot(core)
    cli.main()
    body = json.loads(capsys.readouterr().out)
    assert body["identity"]["id"] == sample[key] and body["customer_id"] == str(core.customer)
    assert body["kind"] == kind.replace("-", "_")
    assert body["read_only"] is True
    assert snapshot(core) == before
    # CLI preserves the same wrong-kind contract rather than falling back to a
    # balance lookup or creating a new identity.
    monkeypatch.setattr("sys.argv", ["visioncortex.lab", "material-batch-summary", kind,
                                     sample["batch" if key == "lot" else "lot"]])
    with pytest.raises(Missing):
        cli.main()
    assert snapshot(core) == before
