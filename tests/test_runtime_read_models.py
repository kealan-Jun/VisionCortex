import json
import time

import pytest

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.input_availability import Availability
from visioncortex.device_day_activity import job, phase, observations, counted
from visioncortex.device_day_latency import observe, upload_completed, snapshot


def test_unavailable_inputs_do_not_block_other_cameras(tmp_path):
    q = DeviceDayQueue(tmp_path / "queue-vision.sqlite3")
    a = Availability(tmp_path)
    record = {
        "recording_id": "missing",
        "source_signature": "v1",
        "configured_role": "first_person",
        "camera_key": "cam1",
    }
    q.enqueue(record, "revision1")
    q.enqueue(record | {"recording_id": "good", "camera_key": "cam2"}, "revision1")
    a.mark(record, "missing")
    q.sync_availability(a.states())
    assert q.claim("worker")["recording_id"] == "good"
    assert q.snapshot()["counts"]["input_missing"] == 1
    a.mark(record, "ready")
    q.sync_availability(a.states())
    assert q.claim("worker2")["recording_id"] == "missing"


def test_old_missing_version_does_not_block_new_source(tmp_path):
    q = DeviceDayQueue(tmp_path / "queue-vision.sqlite3")
    a = Availability(tmp_path)
    record = {
        "recording_id": "one",
        "source_signature": "v1",
        "configured_role": "first_person",
        "camera_key": "cam",
    }
    q.enqueue(record, "rev1")
    a.mark(record, "missing")
    q.sync_availability(a.states())
    q.enqueue(record | {"source_signature": "v2"}, "rev2")
    q.sync_availability(a.states())
    assert q.claim("worker")["source_signature"] == "v2"


def test_durable_phase_and_counts_cleaned_after_completion(tmp_path):
    import sqlite3

    with job("vision", "one", root=tmp_path):
        with phase("coarse_scan_seconds"):
            counted("coarse", 8)
            with sqlite3.connect(tmp_path / "WorkPhases.sqlite3") as db:
                payload = json.loads(
                    db.execute("SELECT payload FROM phases").fetchone()[0]
                )
                assert payload["phase"] == "coarse_scan_seconds"
            assert "粗扫" in observations(tmp_path)[("vision", "one")]["phase"]
    assert observations(tmp_path) == {}


def test_receiver_receipt_revision_scoped_and_not_reconstructed(tmp_path):
    from visioncortex.observed_inventory import observe as inventory

    now = time.time()
    config = {"storage": {"local_runtime_root": str(tmp_path)}}
    record = {
        "recording_id": "r1",
        "camera_key": "cam",
        "source_signature": "sig",
        "processable": True,
        "recording_start_us": round((now - 600) * 1e6),
        "recording_end_us": round(now * 1e6),
    }
    inventory(tmp_path / "device-day", {"recordings": [record]})
    observe(config, [record], now=now)
    assert snapshot(config)["recent"][0]["upload_completed_at"] is None
    payload = {"recording_id": "r1", "source_signature": "sig", "completed_at": now - 3}
    upload_completed(config, payload)
    upload_completed(config, payload)
    data = snapshot(config)
    assert data["upload_completion_time_available"]
    assert data["recent"][0]["discovery_delay_seconds"] == 3
    with pytest.raises(ValueError):
        upload_completed(config, payload | {"source_signature": "wrong"})
    with pytest.raises(ValueError):
        upload_completed(config, payload | {"completed_at": now - 5})
    with pytest.raises(ValueError):
        upload_completed(config, payload | {"completed_at": now + 50})
