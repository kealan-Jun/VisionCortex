from decimal import Decimal
from uuid import uuid4

import pytest

from visioncortex.lab.ingest import Ingest
from visioncortex.lab.processing import insert_observation
from visioncortex.lab.speech_candidates import parse_usage


@pytest.mark.parametrize("text,intent", [("PB-01批次，使用这袋G001五克", "actual_increment"), ("PB-01准备使用G001 5g", "planned"), ("PB-01累计用了G001 8g", "cumulative"), ("G001还剩80g", "balance"), ("刚才5克改成3克", "correction")])
def test_speech_intents_remain_separate(text, intent):
    result = parse_usage(text)
    assert result[0]["semantics"] == intent
    assert parse_usage(text, final=False) == []
    if intent == "actual_increment":
        assert result[0]["value"]["batch_code"] == "PB-01"
        assert result[0]["value"]["material_code"] == "G001"
        assert result[0]["value"]["quantity"] == "5"


def test_two_additions_are_not_merged_by_text():
    result = parse_usage("PB-01使用G001，两次各加5g")
    assert "multiple_operations_require_explicit_split" in result[0]["conflicts"]


def test_usage_confirmation_and_ledger_share_one_transaction(core, settings, photo):
    receipt = Ingest(core, settings).accept(photo)
    mt = core.register("type", "material_type", "G001")
    material = core.register_material("material", "material-unit-one", "g", "100", metadata={"material_type_id": mt["id"]})
    batch = core.register("batch", "process_batch", "PB-01")
    with core.db.transaction() as conn:
        insert_observation(core, conn, event_id="speech-one", revision="final/1", kind="asr_segment", source_kind="fixture", occurred_at="2026-10-23T10:58:00+08:00", evidence_ids=[receipt["evidence_id"]], payload=dict(analysis_outcome="observations", normalized_fields=parse_usage("PB-01批次，使用这袋G001五克")))
        candidate = conn.execute("SELECT id FROM field_candidate WHERE customer_id=%s", (core.customer,)).fetchone()["id"]
    op_id = str(uuid4())
    args = dict(expected_revision=1, material_id=material["material_id"], batch_id=batch["id"], expected_material_revision=1, actor="fixture-reviewer", operation_id=op_id, reason="explicitly confirmed")
    result = core.confirm_usage("confirm", candidate, **args)
    assert core.confirm_usage("confirm", candidate, **args) == result
    assert Decimal(core.balance(material["material_id"])["balance"]) == 95
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM fact_revision WHERE customer_id=%s AND scope='material_consumption'", (core.customer,)).fetchone()["n"] == 1
    corrected = core.correct_operation("correct-usage", op_id, [dict(material_id=material["material_id"], quantity="3", unit="g")], expected_revision=1, actor="fixture-reviewer", reason="verified correction")
    assert Decimal(core.balance(material["material_id"])["balance"]) == 97
    with core.db.transaction(readonly=True) as conn:
        active = conn.execute("SELECT * FROM fact_revision WHERE customer_id=%s AND status='confirmed' AND scope='material_consumption'", (core.customer,)).fetchall()
        assert len(active) == 1 and Decimal(active[0]["value"]["quantity"]) == 3
        assert active[0]["value"]["operation_id"] == corrected["operation_id"]
        assert conn.execute("SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s AND reverses_id IS NOT NULL", (core.customer,)).fetchone()["n"] == 1
    core.correct_operation("correct-again", corrected["operation_id"], [dict(material_id=material["material_id"], quantity="4", unit="g")], expected_revision=1, actor="fixture-reviewer", reason="second correction")
    assert Decimal(core.balance(material["material_id"])["balance"]) == 96
