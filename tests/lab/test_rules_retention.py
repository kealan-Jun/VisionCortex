from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from visioncortex.lab.components import ROOT, enable_components
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.processing import Worker
from visioncortex.lab.retention import Retention
from visioncortex.lab.rules import Rules, RuleMismatch
from visioncortex.lab.storage import verify_root


def test_frozen_effective_screening_and_tamper_detection(tmp_path):
    identity = Rules().identity()
    profile = Rules().profile("vision")["settings"]
    assert profile["performance"]["coarse_detection_fps"] == 2
    assert profile["performance"]["detection_fps"] == 20
    assert profile["performance"]["fine_coverage_gate_enabled"] is True
    assert profile["device_day"]["local_activity_recall"]["enabled"] is True
    assert profile["models"]["confidence"] == .25
    current = json.loads((ROOT / 'contracts/lab/rules.lock.json').read_text())
    assert identity['revision'] == current['revision']
    assert current['files'] and current['components_sha256']
    shutil.copytree(ROOT / "contracts/lab", tmp_path / "contracts/lab")
    (tmp_path / "components/lab").mkdir(parents=True)
    shutil.copyfile(ROOT / "components/lab/manifest.json", tmp_path / "components/lab/manifest.json")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    config = tmp_path / "contracts/lab/vision-rules.json"
    config.write_text(config.read_text().replace('"confidence": 0.25', '"confidence": 0.1'))
    with pytest.raises(RuleMismatch, match="profile changed"):
        Rules(tmp_path).identity()


@pytest.mark.parametrize("moving,pan,verified,selected", [(True, False, True, True), (False, False, True, False), (False, True, True, False), (True, False, False, False)])
def test_original_activity_recall_keeps_its_semantics(moving, pan, verified, selected):
    enable_components()
    from visioncortex_lab_legacy.analysis.activity_recall import observations, selected_runs
    def box(name, x):
        return {"class_name": name, "confidence": .8, "xyxy_norm": [x, .3, x + .04, .36]}
    rows = []
    for n in range(10):
        shift = n * .02 if pan else 0
        actors, objects = observations([box("gloved_hand", .4 + shift + (n * .005 if moving else 0)), box("beaker", .44 + shift), box("container", .3 + shift)])
        rows.append(dict(local_ms=n * 1000, source_verified=verified, actors=actors, objects=objects, source_frame={"source_pts": n * 1000}))
    intervals, decisions = selected_runs(rows, period_ms=1000)
    assert bool(intervals) == selected
    assert all(d["physical_action_confirmed"] is False for d in decisions)


def test_original_fixed_decimal_requires_registered_device_rule():
    enable_components()
    from field_recognition.measurements.measurement_records import apply_fixed_decimal, number
    line = {"text": "02181"}
    assert number({"value": "02181"}) is None
    assert apply_fixed_decimal("质量", line, {}) == line
    corrected = apply_fixed_decimal("质量", line, {"id": "fixture", "measurement_ranges": {"质量": {"fixed_decimal_display": True, "decimal_places": 4}}})
    assert corrected["value"] == "0.2181"
    assert corrected["normalization"]["raw_text"] == "02181"


def test_automatic_photo_retention_preserves_originals_and_no_unbound_result(core, settings, photo):
    original = settings.incoming / photo["relative_path"]
    before = original.read_bytes()
    captured = Ingest(core, settings).accept(photo)
    assert all(r["status"] in {"completed", "blocked_model"} for r in [Worker(core, settings).run_once(), Worker(core, settings).run_once()])
    retention = Retention(core, settings)
    results = [retention.tick() for _ in range(4)]
    assert sum(r["published"] for r in results) == 2, results
    evidence = list(settings.incoming.glob("*/Comment/OCR/VoicePhotoReadings/*/Evidence.json"))
    assert len(evidence) == 1
    value = json.loads(evidence[0].read_text())
    assert value["sources"][0]["media_id"] == captured["media_id"]
    assert value["measurement_files"] == []
    reading = json.loads(evidence[0].with_name("Readings.json").read_text())
    assert reading["schema"] == "field-numeric-observations/1" and "readings" in reading
    assert reading["aggregation"] == "per_capture_not_merged"
    assert not list(settings.incoming.rglob("Result*.json"))
    assert original.read_bytes() == before
    assert len(list(settings.incoming.rglob("*.png"))) == 1
    receipts = list(settings.incoming.glob(".System/Receipts/*/*/*/*.json"))
    assert len(receipts) == 2
    for receipt in receipts:
        document = json.loads(receipt.read_text())
        assert document["schema"] == "field-recognition-receipt/1"
        assert set(document) == {"schema", "source_instance", "sequence", "entity", "entity_id", "recorded_at", "document", "artifacts", "physical_action_confirmed"}
        for item in document["artifacts"].values():
            assert hashlib.sha256((settings.incoming / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_publication_restart_replays_frozen_plan(core, settings, photo, monkeypatch):
    Ingest(core, settings).accept(photo)
    retention = Retention(core, settings)
    write = retention._write
    interrupted = []
    def fail_once(relative, raw, **kwargs):
        if relative.startswith(".System/Receipts/") and not interrupted:
            interrupted.append(True)
            raise OSError("fixture interruption after writing views")
        return write(relative, raw, **kwargs)
    monkeypatch.setattr(retention, "_write", fail_once)
    assert retention.tick()["status"] == "needs_attention"
    with core.db.transaction() as conn:
        row = conn.execute("SELECT plan FROM archive_publication WHERE customer_id=%s", (core.customer,)).fetchone()
        assert row["plan"]
        conn.execute("UPDATE archive_publication SET next_attempt_at=now() WHERE customer_id=%s", (core.customer,))
    assert Retention(core, settings).tick()["published"] == 1
    assert len(list(settings.incoming.glob(".System/Receipts/*/*/*/*.json"))) == 1


def test_nas_mismatch_fails_before_marker_read_or_directory_creation(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("visioncortex.lab.storage.subprocess.run", lambda *a, **k: SimpleNamespace(stdout='{"filesystems":[{"target":"/","source":"/dev/local","fstype":"ext4"}]}'))
    with pytest.raises(OSError, match="CIFS mount identity mismatch"):
        verify_root(Path("/mnt/fixture-archive/project"), "0" * 64)


def test_equipment_tag_on_photo_uses_capture_time_and_does_not_invent_wearer(core, settings, photo):
    instrument = core.register("instrument", "instrument", "test-balance", {"field_recognition": {"model": "fixture", "device_no": "0007"}})
    tag = core.register("tag", "tag", "Q-M001", {"mode": "permanent"})
    core.assign_tag("binding", tag["id"], instrument["id"], "2026-10-23T10:00:00+08:00", actor="fixture", reason="fixture")
    rows = Worker(core, settings).photo_bindings(settings.incoming / photo["relative_path"], "2026-10-23T10:58:00+08:00")
    assert rows[0]["instrument"]["id"] == instrument["id"]
    assert rows[0]["instrument"]["device_no"] == "0007" and rows[0]["wearer_id"] is None
    assert Worker(core, settings).photo_bindings(settings.incoming / photo["relative_path"], "2026-10-23T09:58:00+08:00") == []


def test_rule_rollout_can_reschedule_model_free_qr_without_touching_old_results(core, settings, photo):
    Ingest(core, settings).accept(photo)
    with core.db.transaction() as conn:
        conn.execute("UPDATE processing_job SET rule_fingerprint='fixture-previous-release' WHERE customer_id=%s AND worker_kind='qr'", (core.customer,))
    worker = Worker(core, settings)
    assert worker.schedule_models() == 1
    assert worker.schedule_models() == 0
