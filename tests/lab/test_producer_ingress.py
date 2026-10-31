"""Actual inventory schema and codec fixtures; no live sources or model calls."""

from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import time
from uuid import uuid4

import pytest

from visioncortex.lab import producer_ingress as module
from visioncortex.lab.components import inspect_recorder
from visioncortex.lab.domain import Conflict
from visioncortex.lab.producer_ingress import (
    ProducerIngress,
    load_manifest,
    selected_mapping,
)

from .test_native_delivery import native_fixture


def inventory(tmp_path, records):
    path = tmp_path / "producer-observed.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY,payload TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
        )
        for row in records:
            db.execute(
                "INSERT OR REPLACE INTO observations VALUES(?,?)",
                (row["recording_id"], json.dumps(row)),
            )
        db.execute(
            "INSERT OR REPLACE INTO metadata VALUES('revision',?)", (str(len(records)),)
        )
    return path


def configuration(tmp_path, settings, path, *, mode="new_only", ids=None, **extra):
    root = settings.source_root or settings.runtime_root / "incoming"
    doc = dict(
        schema_version=module.SCHEMA,
        enabled=True,
        revision="generated-fixture/1",
        inventory_path=str(path),
        source_root=str(root),
        inventory_relative_prefix="",
        mode=mode,
        recording_ids=ids or [],
        actor="fixture-source-adapter",
        reason="generated codec and inventory fixture",
        **extra,
    )
    manifest = tmp_path / "private-producer.json"
    manifest.write_text(json.dumps(doc))
    manifest.chmod(0o600)
    return replace(
        settings, processing_profile="timeline-v1", producer_ingress_manifest=manifest
    )


def recording(settings, camera, *, prefix=""):
    ready = native_fixture(settings, camera)
    row = inspect_recorder(
        settings.incoming, ready.with_name("rgb.mp4"), time.time(), 0.5
    )
    assert row["processable"]
    if prefix:
        row = dict(row, relative_path=prefix + "/" + row["relative_path"])
        row["recording_id"] = hashlib.sha256(row["relative_path"].encode()).hexdigest()[
            :24
        ]
    return row


def minimal_settings(tmp_path, *, nas=False):
    root = tmp_path / "owned"
    root.mkdir()
    from visioncortex.lab.config import Settings

    return Settings(
        "postgresql://fixture@localhost/unused",
        root,
        "fixture-only",
        processing_profile="timeline-v1",
        source_root=Path("/mnt/fixture-not-mounted/Archive") if nas else None,
    )


def row(identity="source-one", processable=True):
    return dict(
        recording_id=identity,
        processable=processable,
        recording_start_us=1,
        source_signature="fixture-source-signature",
        relative_path="2026-10-23_camera/MetaData/100000/rgb.mp4",
    )


def test_unconfigured_has_no_state_source_or_database_io(tmp_path, monkeypatch):
    settings = minimal_settings(tmp_path)
    monkeypatch.setattr(
        module,
        "_local",
        lambda *a, **k: pytest.fail("Unconfigured adapter touched storage"),
    )
    assert ProducerIngress(SimpleNamespace(), settings).tick() == dict(enabled=False)
    assert list(settings.runtime_root.iterdir()) == []


def test_first_snapshot_only_baselines_finalized_history_without_NAS_or_PG(
    tmp_path, monkeypatch
):
    settings = minimal_settings(tmp_path, nas=True)
    path = inventory(tmp_path, [row("old-finalized"), row("not-yet-finalized", False)])
    settings = configuration(tmp_path, settings, path)
    monkeypatch.setattr(
        module, "publication_config", lambda *a: pytest.fail("Baseline touched NAS")
    )
    result = ProducerIngress(SimpleNamespace(customer=uuid4()), settings).tick()
    assert result["baseline_finalized"] == 1 and result["admitted"] == 0
    state = json.loads(
        (settings.runtime_root / "producer-ingress/State.json").read_text()
    )
    assert state["baseline_finalized"] == ["old-finalized"]


@pytest.mark.parametrize(
    "change",
    [
        dict(unknown_key=True),
        dict(source_root="/different/root"),
        dict(max_admissions_per_tick=17),
        dict(inventory_relative_prefix="../escape"),
    ],
)
def test_manifest_rejects_unknown_scope_and_unbounded_options(tmp_path, change):
    settings = minimal_settings(tmp_path)
    settings = configuration(tmp_path, settings, inventory(tmp_path, []))
    p = settings.producer_ingress_manifest
    doc = json.loads(p.read_text())
    doc.update(change)
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_manifest(settings)


def test_inventory_capacity_is_refused_before_any_source_access(tmp_path, monkeypatch):
    settings = minimal_settings(tmp_path)
    path = inventory(tmp_path, [row("one"), row("two")])
    settings = configuration(tmp_path, settings, path, max_inventory_rows=1)
    monkeypatch.setattr(
        module,
        "publication_config",
        lambda *a: pytest.fail("Exceeded inventory touched source"),
    )
    with pytest.raises(Conflict, match="bounded snapshot"):
        ProducerIngress(SimpleNamespace(customer=uuid4()), settings).tick()


def test_missing_or_changed_storage_guard_prevents_admission(tmp_path, monkeypatch):
    settings = minimal_settings(tmp_path, nas=True)
    path = inventory(tmp_path, [row()])
    settings = configuration(
        tmp_path, settings, path, mode="explicit_ids", ids=["source-one"]
    )

    @contextmanager
    def rejected(config):
        raise OSError("Pinned mount changed")
        yield

    monkeypatch.setattr(module, "publication_config", lambda s: {"fixture": True})
    monkeypatch.setattr(
        module, "pipeline_module", lambda n: SimpleNamespace(storage_guard=rejected)
    )
    with pytest.raises(OSError, match="mount changed"):
        ProducerIngress(SimpleNamespace(customer=uuid4()), settings).tick()


def test_only_exact_mapping_can_bridge_source_time_not_tolerance_or_other_camera():
    original = "2026-10-23_camera/MetaData/100000/rgb.mp4"
    parent = dict(verified_sha256="a" * 64, duration_seconds=Decimal(1))
    entry = dict(
        source_ref=dict(path="MetaData/100000/rgb.mp4", sha256="a" * 64),
        source_mapping=dict(
            source_start_ms=0, source_end_ms=500, derived_start_ms=0, derived_end_ms=500
        ),
        video=dict(
            path="ProcessedClips/Clips/fixture/Clip.mp4", sha256="b" * 64, size_bytes=1
        ),
    )
    assert selected_mapping(entry, original, original.split("/")[0], parent)[:2] == (
        Decimal(0),
        Decimal(".5"),
    )
    entry["source_mapping"]["derived_end_ms"] = "516.667"
    with pytest.raises(Conflict, match="exact contiguous"):
        selected_mapping(entry, original, original.split("/")[0], parent)
    entry["source_mapping"]["derived_end_ms"] = 500
    entry["source_ref"]["path"] = "2026-10-23_other/MetaData/100000/rgb.mp4"
    with pytest.raises(Conflict, match="path/SHA"):
        selected_mapping(entry, original, original.split("/")[0], parent)


def test_new_native_codec_rows_enter_same_Core_late_completion_and_two_cameras(
    core, settings, tmp_path
):
    old = recording(settings, "old-source")
    path = inventory(tmp_path, [old])
    selected = configuration(tmp_path, settings, path)
    adapter = ProducerIngress(core, selected)
    assert adapter.tick()["admitted"] == 0
    first, second = (
        recording(settings, "new-source-A"),
        recording(settings, "new-source-B"),
    )
    inventory(tmp_path, [first, dict(second, processable=False)])
    assert adapter.tick()["admitted"] == 1
    inventory(tmp_path, [second])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert adapter.tick()["admitted"] == 1
    assert adapter.tick()["admitted"] == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with core.db.transaction(readonly=True) as conn:
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM capture WHERE customer_id=%s",
                (core.customer,),
            ).fetchone()["n"]
            == 2
        )
        cameras = conn.execute(
            "SELECT code FROM entity WHERE customer_id=%s AND kind='capture_device'",
            (core.customer,),
        ).fetchall()
        assert {r["code"] for r in cameras} == {"new-source-A", "new-source-B"}
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM entity WHERE customer_id=%s AND kind='person'",
                (core.customer,),
            ).fetchone()["n"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM fact_revision WHERE customer_id=%s",
                (core.customer,),
            ).fetchone()["n"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s",
                (core.customer,),
            ).fetchone()["n"]
            == 0
        )
    assert not list(settings.incoming.rglob("Understanding.json"))


def add_index(settings, source, *, wrong_hash=False):
    import cv2
    import numpy as np

    actual_relative = source["relative_path"].removeprefix("ProducerArchive/")
    archive = actual_relative.split("/")[0]
    path = settings.incoming / archive / "ProcessedClips/Clips/fixture/Clip.mp4"
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (96, 64))
    writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()
    original = settings.incoming / actual_relative
    entry = dict(
        segment_id="generated-exact-retained-interval",
        source_ref=dict(
            path="/".join(actual_relative.split("/")[1:]),
            sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        ),
        source_mapping=dict(
            source_start_ms=0, source_end_ms=250, derived_start_ms=0, derived_end_ms=250
        ),
        video=dict(
            path=str(path.relative_to(settings.incoming / archive)),
            size_bytes=path.stat().st_size,
            sha256="0" * 64
            if wrong_hash
            else hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )
    index = settings.incoming / archive / "ProcessedClips/Index.json"
    index.write_text(json.dumps(dict(archive=archive, segments=[entry])))
    return path


def test_existing_codec_clip_is_admitted_only_after_late_exact_index_and_keeps_source_ID_alias(
    core, settings, tmp_path, monkeypatch
):
    # This test covers exact retained-clip mappings with an explicit fixture
    # clock. Native self-reported epochs alone no longer establish that clock.
    from visioncortex.lab.ingest import Ingest
    class FixtureClockIngest(Ingest):
        def accept(self, receipt, **kwargs):
            assert receipt['source_kind']=='fixture'
            return super().accept(dict(receipt,time_basis=dict(basis='synthetic_fixture_clock',uncertainty_us=0)),**kwargs)
    monkeypatch.setattr(module,'Ingest',FixtureClockIngest)
    source = recording(settings, "dynamic-source", prefix="ProducerArchive")
    inventory_path = inventory(tmp_path, [source])
    selected = configuration(
        tmp_path,
        settings,
        inventory_path,
        mode="explicit_ids",
        ids=[source["recording_id"]],
    )
    doc = json.loads(selected.producer_ingress_manifest.read_text())
    doc["inventory_relative_prefix"] = "ProducerArchive"
    selected.producer_ingress_manifest.write_text(json.dumps(doc))
    adapter = ProducerIngress(core, selected)
    first = adapter.tick()
    assert first["admitted"] == 1 and first["clips"] == 0
    child = add_index(settings, source)
    state_path = settings.runtime_root / "producer-ingress/State.json"
    state = json.loads(state_path.read_text())
    saved = state["admitted"][source["recording_id"]]
    assert saved["original_inventory_recording_id"] == source["recording_id"]
    saved["clips_next_check"] = 0
    state_path.write_text(json.dumps(state))
    second = adapter.tick()
    assert second["clips"] == 1 and second["duplicates"] == 1, second
    assert child.is_file()
    with core.db.transaction(readonly=True) as conn:
        derivations = conn.execute(
            "SELECT transform FROM media_derivation WHERE customer_id=%s",
            (core.customer,),
        ).fetchall()
        assert len(derivations) == 1
        assert (
            derivations[0]["transform"]["derivation_verification"]
            == "supplied_mapping_not_independent_pixel_proof"
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM hj_timeline_task WHERE customer_id=%s",
                (core.customer,),
            ).fetchone()["n"]
            == 2
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s",
                (core.customer,),
            ).fetchone()["n"]
            == 0
        )
    assert adapter.tick()["clips"] == 0


def test_changed_clip_hash_and_inventory_revision_remain_pending_without_fake_derived_input(
    core, settings, tmp_path
):
    source = recording(settings, "changed-source")
    path = inventory(tmp_path, [source])
    selected = configuration(
        tmp_path, settings, path, mode="explicit_ids", ids=[source["recording_id"]]
    )
    add_index(settings, source, wrong_hash=True)
    adapter = ProducerIngress(core, selected)
    result = adapter.tick()
    assert result["admitted"] == 1 and result["clips"] == 0
    with core.db.transaction(readonly=True) as conn:
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM media_derivation WHERE customer_id=%s",
                (core.customer,),
            ).fetchone()["n"]
            == 0
        )
    inventory(tmp_path, [dict(source, source_signature="changed-after-admission")])
    result = adapter.tick()
    assert result["admitted"] == 0
    assert (
        result["pending"][0]["reason"]
        == "source_revision_changed_explicit_review_required"
    )
