"""Actual existing delivery entry + isolated fixture PostgreSQL + native writers."""

from dataclasses import replace
import json

from visioncortex.lab.archive_layout import contract_bytes
from visioncortex.lab.delivery import NativeDelivery
from visioncortex.lab.processing import ModelRegistry
from visioncortex.lab.timeline_runner import TimelineRunner
from visioncortex.lab.timeline_publication import TimelinePublication

from .test_person_report import scene, media, event, report
from .test_operator_scan import scan
from .test_native_refresh import prepare

__all__ = ["scene"]


def test_committed_timeline_to_native_delivery_main_refresh_and_late_idempotence(
    scene, monkeypatch, tmp_path
):
    s = scene

    def forbidden(*a, **k):
        raise AssertionError(
            "Native refresh must not call model configuration or runner"
        )

    monkeypatch.setattr(ModelRegistry, "descriptor", forbidden)
    monkeypatch.setattr(TimelineRunner, "run_once", forbidden)
    a = event(s, "camera0", media(s, "camera0", camera=0))
    b = event(s, "camera1", media(s, "camera1", camera=1))
    scan(s, "A", run_id=s["run"])
    first_snapshot = report(s)
    root = s["settings"].incoming
    (root / ".System").mkdir()
    (root / ".System/ArchiveContract.json").write_bytes(contract_bytes())
    settings = replace(
        s["settings"],
        source_root=root,
        unified_archive_required=True,
        timeline_publish=True,
    )
    fixtures = [
        prepare(root, s["core"].customer, "fixture-camera-" + str(i)) for i in range(2)
    ]
    original_indexes = [
        (f["day"] / "ProcessedClips/Index.json").read_bytes() for f in fixtures
    ]
    publication = TimelinePublication(s["runtime"], settings)
    delivery = NativeDelivery(s["core"], settings)
    assert (
        delivery.tick()["native_refresh"]["status"]
        == "awaiting_committed_timeline_publication"
    )
    published = publication.tick()
    assert published["revision"] == first_snapshot["revision"]
    refreshed = delivery.tick()
    assert refreshed["native_refresh"]["refreshed"] == 2 and refreshed["errors"] == 0
    native_paths = [
        f["day"] / "LaboratoryDailyReport/LaboratoryDailyReport.json" for f in fixtures
    ]
    mains = [json.loads(p.read_text()) for p in native_paths]
    assert all(
        m["全天综合理解"]["status"] == "pending"
        and m["关联时间线"]["snapshot_revision"] == first_snapshot["revision"]
        for m in mains
    )
    for i, f in enumerate(fixtures):
        assert (
            json.loads(f["main"].read_text())["关联时间线"][
                "native_entry_refresh_status"
            ]
            == "included_by_native_builder"
        )
        assert (
            f["day"] / "ProcessedClips/Index.json"
        ).read_bytes() == original_indexes[i]
    before = [p.read_bytes() for p in native_paths]
    assert delivery.tick()["native_refresh"]["unchanged"] == 2
    assert [p.read_bytes() for p in native_paths] == before
    scan(s, "B", value=s["devices"][1], when="09:02:00")
    second_snapshot = report(s)
    assert second_snapshot["revision"] > first_snapshot["revision"]
    publication.tick()
    late = delivery.tick()
    assert late["native_refresh"]["refreshed"] == 2
    for i, p in enumerate(native_paths):
        document = json.loads(p.read_text())
        assert document["archive"] == fixtures[i]["archive"]
        assert (
            document["关联时间线"]["snapshot_revision"] == second_snapshot["revision"]
        )
    assert a["id"] != b["id"]
    with s["core"].db.transaction(readonly=True) as conn:
        counts = {
            table: conn.execute(
                f"SELECT count(*) AS n FROM {table} WHERE customer_id=%s",
                (s["core"].customer,),
            ).fetchone()["n"]
            for table in ["quantity_ledger_entry", "fact_revision", "field_candidate"]
        }
    assert counts == dict(quantity_ledger_entry=0, fact_revision=0, field_candidate=0)
    (tmp_path / "native-refresh-entry-receipt.json").write_text(
        json.dumps(
            dict(
                publication=published,
                first=refreshed,
                late=late,
                counts=counts,
                model_calls=0,
            ),
            indent=2,
        )
    )
