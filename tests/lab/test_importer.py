
import pytest

from visioncortex.lab.domain import Conflict
from visioncortex.lab.importer import ReceiptImporter


def feed(core, source="source-one"):
    mapping = dict(customer_id=str(core.customer), source_instance=source, partition="camera-one", source_tenant=None, allow_unscoped=True, source_kind="derived_replay")
    event = dict(schema_version="field-event/1", source_instance=source, sequence=2, revision=2, event_id="local-job-001", occurred_start="2026-10-01T01:00:00+08:00", payload={"status": "draft", "raw_answer": "1.0827"})
    page = dict(schema="field-event-feed/1", source_instance=source, camera_id="camera-one", tenant=None, after_seq=0, next_cursor=2, high_watermark=2, items=[event])
    return page, mapping


def test_dry_run_resume_duplicate_delivery_and_source_namespace(core, settings):
    importer = ReceiptImporter(core, settings)
    page, mapping = feed(core)
    assert importer.page(page, mapping)["dry_run"]
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM inbox WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0
    assert importer.page(page, mapping, dry_run=False)["accepted"] == 1
    assert importer.page(page, mapping, dry_run=False)["duplicates"] == 1
    second, mapping2 = feed(core, "source-two")
    importer.page(second, mapping2, dry_run=False)
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM observation WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 2
        assert conn.execute("SELECT count(*) AS n FROM fact_revision WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0


def test_unscoped_history_requires_explicit_mapping(core, settings):
    page, mapping = feed(core)
    mapping["allow_unscoped"] = False
    with pytest.raises(Conflict):
        ReceiptImporter(core, settings).page(page, mapping)


def test_import_interruption_rolls_back_inbox_and_checkpoint(core, settings, monkeypatch):
    import visioncortex.lab.importer as module
    page, mapping = feed(core)
    original = module.insert_observation
    def fail(*args, **kwargs):
        raise RuntimeError("fixture interrupted")
    monkeypatch.setattr(module, "insert_observation", fail)
    with pytest.raises(RuntimeError):
        ReceiptImporter(core, settings).page(page, mapping, dry_run=False)
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM inbox WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM consumer_checkpoint WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0
    monkeypatch.setattr(module, "insert_observation", original)
    assert ReceiptImporter(core, settings).page(page, mapping, dry_run=False)["accepted"] == 1
