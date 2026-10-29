from __future__ import annotations

import json
import os
import time
import wave
from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from visioncortex.lab.api import create_app
from visioncortex.lab.domain import Conflict, Core
from visioncortex.lab.fields import Fields
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.media import Media, safe_path
from visioncortex.lab.processing import Worker, insert_observation
from visioncortex.lab.projections import Projections


def test_automatic_photo_qr_receipt_and_model_blocking(core, settings, photo):
    ingest = Ingest(core, settings)
    assert ingest.discover_once()["processed"] == 1
    assert ingest.discover_once()["skipped"] == 1
    worker = Worker(core, settings)
    results = [worker.run_once(), worker.run_once()]
    assert {r["status"] for r in results} == {"completed", "blocked_model"}
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM capture WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 1
        row = conn.execute("SELECT payload FROM observation WHERE customer_id=%s", (core.customer,)).fetchone()
        assert row["payload"]["decoded_values"] == ["Q-M001"]
        assert row["payload"]["diagnostics"]["identity_inferred"] is False
        assert conn.execute("SELECT count(*) AS n FROM operation WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0
    while Projections(core).tick()["indexed"]:
        pass
    results = Projections(core).search("Q-M001")
    assert len(results["items"]) == 1
    assert results["coverage"] == "unknown" and results["index_caught_up"]


def test_capture_identity_is_not_just_a_file_hash(core, settings, photo):
    ingest = Ingest(core, settings)
    first = ingest.accept(photo)
    assert ingest.accept(photo)["duplicate"]
    second = ingest.accept(dict(photo, source_instance="other-camera"))
    assert first["capture_id"] != second["capture_id"]
    assert first["media_id"] != second["media_id"]
    assert len(list(settings.cache.rglob("*.png"))) == 1
    with pytest.raises(Conflict):
        ingest.accept(dict(photo, captured_at="2026-10-23T10:59:00+08:00"))


def test_no_numeric_readout_never_promotes_raw_model_answer(core, settings, photo):
    receipt = Ingest(core, settings).accept(photo)
    with core.db.transaction() as conn:
        insert_observation(core, conn, event_id="ocr:no-readout", revision="1", kind="ocr", source_kind="fixture", evidence_ids=[receipt["evidence_id"]], payload=dict(analysis_outcome="no_numeric_readout", raw_answer="1.0827", readings=[], normalized_fields=[dict(field_key="mass", value="1.1313", valid=True, semantics="unknown", unit="g")]))
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM field_candidate WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM fact_revision WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0


def test_confirmed_display_is_not_material_consumption(core, settings, photo):
    receipt = Ingest(core, settings).accept(photo)
    instrument = core.register("instrument", "instrument", "balance")
    with core.db.transaction() as conn:
        insert_observation(core, conn, event_id="ocr:display", revision="1", kind="ocr", source_kind="fixture", evidence_ids=[receipt["evidence_id"]], payload=dict(analysis_outcome="observations", normalized_fields=[dict(field_key="mass", value="5.00", valid=True, semantics="display_reading", unit="g", association={"instrument_id": instrument["id"]})]))
        candidate = conn.execute("SELECT id FROM field_candidate WHERE customer_id=%s", (core.customer,)).fetchone()["id"]
    with pytest.raises(Conflict):
        core.confirm_candidate("bad-scope", candidate, expected_revision=1, scope="material_consumption", actor="reviewer")
    first = core.confirm_candidate("confirm", candidate, expected_revision=1, scope="display_reading", actor="reviewer")
    assert core.confirm_candidate("confirm", candidate, expected_revision=1, scope="display_reading", actor="reviewer") == first
    fields = Fields(core)
    template = fields.template("template", "fixture-only-template", 1, [{"field_key": "mass", "accept_scopes": ["material_consumption"], "target_field_id": "consumed"}])
    snap = fields.snapshot("snapshot", template["id"], [first["fact_id"]])
    assert snap["fields"] == [] and snap["missing_fields"]
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s", (core.customer,)).fetchone()["n"] == 0


def test_evidence_survives_move_and_range_reads_are_authorized(core, settings, photo):
    receipt = Ingest(core, settings).accept(photo)
    media = Media(core, settings)
    old = media.locate(receipt["media_id"])["path"]
    original_bytes = old.read_bytes()
    target = settings.incoming / "moved.png"
    target.write_bytes(original_bytes)
    media.add_location("move", receipt["media_id"], "incoming", "moved.png")
    old.unlink()
    assert media.locate(receipt["media_id"])["status"] == "available"
    # The producer-owned original is now also a registered location. Keep only
    # the moved fixture to test corrupt/missing bytes at the remaining location.
    (settings.incoming / photo["relative_path"]).unlink()
    app = TestClient(create_app(core, settings))
    assert app.get("/internal/media/" + receipt["media_id"]).status_code == 401
    headers = {"Authorization": "Bearer " + settings.token}
    response = app.get("/internal/media/" + receipt["media_id"], headers=dict(headers, Range="bytes=0-15"))
    assert response.status_code == 206 and response.content == original_bytes[:16]
    assert app.get("/internal/evidence/" + receipt["evidence_id"], headers=headers).json()["id"] == receipt["evidence_id"]
    assert app.get("/internal/media/" + str(uuid4()), headers=headers).status_code == 404
    assert app.get("/internal/commands/not-a-command", headers=headers).status_code == 405
    target.write_bytes(b"corrupted fixture")
    assert media.locate(receipt["media_id"])["status"] == "integrity_failed"
    target.unlink()
    assert media.locate(receipt["media_id"])["status"] == "unavailable"
    with pytest.raises(ValueError):
        safe_path(settings.incoming, "../not-allowed")


def test_index_pause_and_late_revision_preserve_coverage(core):
    with core.db.transaction() as conn:
        insert_observation(core, conn, event_id="same-event", revision="1", kind="asr", payload=dict(analysis_outcome="no_transcript", normalized_fields=[], text="old"), source_kind="fixture", occurred_at="2026-10-01T01:00:00+08:00")
    index = Projections(core)
    assert index.search("old")["coverage"] == "partial"
    index.tick()
    with core.db.transaction() as conn:
        insert_observation(core, conn, event_id="same-event", revision="2", kind="asr", payload=dict(analysis_outcome="observations", normalized_fields=[], text="late-revision"), source_kind="fixture", occurred_at="2026-10-01T01:00:00+08:00")
    assert index.search("late-revision")["coverage"] == "partial"
    index.tick()
    assert len(index.search("late-revision")["items"]) == 1


def test_finalized_recorder_video_uses_reused_completion_checks(core, settings):
    import cv2
    import numpy as np
    at = datetime.fromisoformat("2026-10-23T10:00:00+08:00")
    start = int(at.timestamp() * 1000000)
    folder = settings.incoming / "2026-10-23_camera-one" / "MetaData" / "100000"
    folder.mkdir(parents=True)
    path = folder / "rgb.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (96, 64))
    for i in range(4):
        writer.write(np.full((64, 96, 3), i * 40, dtype=np.uint8))
    writer.release()
    (folder / "frames.csv").write_text("rgb_video_frame_index,global_timestamp_us\n" + "\n".join(f"{i},{start+i*250000}" for i in range(4)))
    meta = dict(recording_session_id="fixture-session", camera_key="camera-one", closed=True, frames_publish_state="finalized", recording_window_start_global_us=start, recording_window_end_global_us=start + 750000)
    (folder / "meta.json").write_text(json.dumps(meta))
    (folder / "recording_ready.json").write_text(json.dumps(dict(meta, ready=True, recording_complete=True, recording_quality_status="complete")))
    for item in folder.iterdir():
        os.utime(item, (time.time() - 2, time.time() - 2))
    result = Ingest(core, settings).discover_once()
    assert result["processed"] == 1, result
    outcome = Worker(core, settings).run_once()
    assert outcome["status"] == "blocked_model"


def test_independent_audio_is_admitted_without_a_video(core, settings):
    folder = settings.incoming / "audio"
    folder.mkdir()
    path = folder / "audio.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 4000)
    receipt = dict(capture_id="audio-only", source_instance="recorder-audio", source_revision="1", kind="audio", relative_path="audio/audio.wav", captured_at="2026-10-23T10:00:00+08:00", time_basis={"basis": "fixture_sample_clock"}, source_kind="synthetic_silence", start_seconds="0", end_seconds="0.25")
    Ingest(core, settings).accept(receipt)
    assert Worker(core, settings).run_once()["status"] == "blocked_model"


def test_customer_scope_rejects_synthetic_input(core, settings, photo, db):
    customer = Core.customer_instance(db, "real-" + str(uuid4()), scope="customer")
    real = Core(db, customer["id"])
    with pytest.raises(Conflict):
        Ingest(real, settings).accept(photo)
