import copy
from types import SimpleNamespace

import pytest

from visioncortex.evidence.field_events import FieldEventInbox
from visioncortex.evidence.person_day import person_days
from visioncortex.application.field_event_consumer import FieldEventConsumer


def event(ident="ocr-1", sequence=1, *, camera="camera-1", when="2026-10-14T01:05:00+00:00",
          wearer=None, kind="ocr_observation", withdrawn=False):
    entity = "bindings" if kind == "instrument_binding" else "jobs"
    ref = {"source": "field_recognition", "source_instance": "instance-1", "entity": entity,
           "entity_id": ident, "outbox_seq": sequence}
    return {"schema_version": "field-event/1", "source": "field_recognition", "source_instance": "instance-1",
            "event_id": f"instance-1:{entity}:{ident}", "delivery_id": f"instance-1:{sequence}",
            "sequence": sequence, "revision": sequence, "tenant": "lab-1", "type": kind,
            "entity": entity, "entity_id": ident, "occurred_start": when, "occurred_end": when,
            "received_at": "2026-10-14T02:00:00+00:00", "processed_at": "2026-10-14T02:01:00+00:00",
            "recorded_at": "2026-10-14T02:01:00+00:00", "capture_device_id": camera,
            "instrument_id": "instrument-1", "wearer_id": wearer, "operator_id": None,
            "actor_track_id": None, "capture_id": ident + "-capture", "capture_session_id": "capture-session",
            "clock": {"basis": "source_capture_time", "source_pts_us": None,
                      "mapping_version": None, "uncertainty_us": 0},
            "media": {"blob_sha256": "a" * 64, "image_url": "/api/images/" + ident},
            "associations": {"wearer": {"id": wearer, "state": "explicit" if wearer else "unknown",
                            "evidence_refs": [ref] if wearer else [], "version": 1}},
            "evidence_refs": [ref], "payload": {"status": "completed", "raw_lines": [{"text": "12.3 mg"}],
                                                   "normalized_values": [], "review_status": "unreviewed"},
            "withdrawn": withdrawn}


def page(events, after=0, cursor=None, camera="camera-1", more=False):
    high = cursor if cursor is not None else max((e["sequence"] for e in events), default=after)
    return {"schema": "field-event-feed/1", "source": "field_recognition", "source_instance": "instance-1",
            "tenant": "lab-1", "camera_id": camera, "after_seq": after, "next_cursor": high,
            "high_watermark": high, "has_more": more, "items": events}


def binding(ident, seq, wearer, start, end=None, camera="camera-1"):
    value = event(ident, seq, kind="instrument_binding", wearer=wearer, when=start, camera=camera)
    value["occurred_end"] = end
    value["payload"] = {"binding_id": ident, "role": "wearer", "valid_from": start,
                        "valid_to": end, "valid_interval": "[valid_from,valid_to)"}
    return value


def test_inbox_replay_revision_withdrawal_and_cursor_commit_together(tmp_path):
    inbox = FieldEventInbox(tmp_path)
    initial = page([event(wearer="a")])
    assert inbox.ingest("local-field", "camera-1", initial, tenant="lab-1")["inserted"] == 1
    assert inbox.ingest("local-field", "camera-1", initial, tenant="lab-1")["inserted"] == 0
    assert len(inbox.events()) == 1 and inbox.cursor("local-field", "camera-1")["cursor"] == 1
    revision = event(sequence=2, wearer="b")
    inbox.ingest("local-field", "camera-1", page([revision], after=1), tenant="lab-1")
    assert inbox.events()[0]["wearer_id"] == "b"
    withdrawn = event(sequence=3, wearer="b", withdrawn=True)
    inbox.ingest("local-field", "camera-1", page([withdrawn], after=2), tenant="lab-1")
    assert inbox.events()[0]["withdrawn"] is True
    assert inbox.version_count() == 3
    bad = page([event("next", 4), event("bad", 5, camera="another-camera")], after=3)
    with pytest.raises(ValueError):
        inbox.ingest("local-field", "camera-1", bad, tenant="lab-1")
    assert inbox.cursor("local-field", "camera-1")["cursor"] == 3
    assert inbox.version_count() == 3


def test_delivery_identity_conflict_and_producer_epoch_change_fail_closed(tmp_path):
    inbox = FieldEventInbox(tmp_path)
    value = page([event()])
    inbox.ingest("field", "camera-1", value, tenant="lab-1")
    conflict = copy.deepcopy(value)
    conflict["items"][0]["payload"]["raw_lines"] = [{"text": "different"}]
    with pytest.raises(ValueError):
        inbox.ingest("field", "camera-1", conflict, tenant="lab-1")
    changed = page([], after=1, cursor=2)
    changed["source_instance"] = "another-instance"
    with pytest.raises(ValueError):
        inbox.ingest("field", "camera-1", changed, tenant="lab-1")


def test_person_day_handover_device_switch_conflict_and_clock_boundary():
    values = [binding("a-1", 1, "a", "2026-10-14T01:00:00Z", "2026-10-14T01:10:00Z"),
              binding("b-1", 2, "b", "2026-10-14T01:10:00Z"),
              binding("a-2", 3, "a", "2026-10-14T01:10:00Z", camera="camera-2"),
              event("before", 4, when="2026-10-14T01:09:59Z"),
              event("boundary", 5, when="2026-10-14T01:10:00Z"),
              event("switch", 6, when="2026-10-14T01:11:00Z", camera="camera-2")]
    reports = person_days(values, timezone="Asia/Shanghai")
    people = {(r["person_id"], r["date"]): r for r in reports["items"]}
    assert {r["event_id"].split(":")[-1] for r in people[("a", "2026-10-14")]["events"]} >= {"before", "switch"}
    assert "boundary" in {r["event_id"].split(":")[-1] for r in people[("b", "2026-10-14")]["events"]}
    assert people[("a", "2026-10-14")]["completeness"]["video_understanding"] == "not_connected"
    uncertain = event("uncertain", 7, when="2026-10-14T01:10:00Z")
    uncertain["clock"]["uncertainty_us"] = 1_000_000
    overlap = binding("overlap", 8, "c", "2026-10-14T01:08:00Z", "2026-10-14T01:12:00Z")
    result = person_days(values + [uncertain, overlap], timezone="Asia/Shanghai")
    unresolved = {e["event_id"].split(":")[-1]: e for e in result["unassociated"]}
    assert unresolved["uncertain"]["attribution"]["state"] == "conflict"
    assert unresolved["before"]["attribution"]["state"] == "conflict"


def test_unbound_fixed_camera_does_not_invent_people_and_late_replay_is_deterministic():
    unknown = event(wearer=None)
    unknown["clock"]["uncertainty_us"] = None
    late = event("late", 2, when="2026-10-14T01:00:00Z", wearer="a")
    output = person_days([unknown, late])
    assert len(output["unassociated"]) == 1
    assert output == person_days([late, unknown])
    assert output["items"][0]["events"][0]["evidence_refs"]


def test_publication_tick_invokes_consumer_and_late_events_publish_without_models(tmp_path, monkeypatch):
    from visioncortex.device_day_publication_worker import PublicationReplay
    import visioncortex.publication_journal as journal
    monkeypatch.setattr(journal, "reconcile", lambda runner: 0)
    replies = [page([event(wearer="a")]), page([event("late", 2, wearer="a", when="2026-10-14T01:00:00Z")], after=1)]
    calls = []
    def fetch(url, timeout):
        from urllib.parse import urlsplit, parse_qs
        parsed = urlsplit(url)
        assert parsed.path == '/api/events/feed'
        assert parse_qs(parsed.query)['camera_id'] == ['camera-1']
        calls.append(url)
        return replies.pop(0)
    runner = SimpleNamespace(runtime_root=tmp_path, config={"device_day": {}},
                             settings={"field_recognition": {"enabled": True, "endpoint": "http://field.test",
                                       "camera_ids": ["camera-1"], "tenant": "lab-1"}})
    replay = PublicationReplay(field_fetch=fetch)
    first = replay.tick(runner)
    assert first["field_events"]["inserted"] == 1
    files = list((tmp_path / "PersonDay").glob("*/Latest.json"))
    assert len(files) == 1
    second = replay.tick(runner)
    assert second["field_events"]["inserted"] == 1 and len(calls) == 2
    import json
    result = json.loads(files[0].read_text())
    assert [e["event_id"].split(":")[-1] for e in result["reports"][0]["events"]] == ["late", "ocr-1"]
    assert result["model_invoked"] is False


def test_consumer_ack_lost_replays_and_failed_publication_stays_dirty(tmp_path, monkeypatch):
    feed = page([event(wearer="a")])
    consumer = FieldEventConsumer(tmp_path, {"enabled": True, "endpoint": "http://field.test",
                                           "camera_ids": ["camera-1"], "tenant": "lab-1"},
                                  fetch_json=lambda *_: feed)
    def fail(*args, **kwargs):
        raise OSError("injected publish failure")
    monkeypatch.setattr(consumer, "publish", fail)
    with pytest.raises(OSError):
        consumer.tick()
    assert consumer.inbox.cursor("http://field.test", "camera-1")["cursor"] == 1
    assert consumer.inbox.publication_state()["dirty"] is True
    monkeypatch.undo()
    assert consumer.tick()["inserted"] == 0
    assert consumer.inbox.publication_state()["dirty"] is False


def test_device_index_port_preserves_video_speech_sources_and_no_input(tmp_path):
    import json
    from visioncortex.application.field_event_consumer import consume_device_index, context_events
    from visioncortex.evidence.device_events import device_index_events
    from datetime import datetime
    start = int(datetime.fromisoformat("2026-10-14T01:00:00+00:00").timestamp() * 1e6)
    archive = "2026-10-14_camera-1"
    config = {"storage": {"local_runtime_root": str(tmp_path)}, "device_day": {
        "field_recognition": {"enabled": True, "endpoint": "http://field.test", "camera_ids": ["camera-1"], "tenant": "lab-1"}}}
    index = {"schema_version": "visioncortex-device-day/1", "archive": archive,
             "recordings": [{"recording_id": "recording-1", "start_us": start, "end_us": start + 600_000_000,
                 "capture_complete": True, "sources": [{"path": "MetaData/source.mp4", "sha256": "a" * 64}],
                 "processing": {"clock_mapping": {"origin_us": start, "basis": "recorder_csv_interpolation", "uncertainty_us": 0}},
                 "audio": {"status": "not_provided"}, "transcription": {"status": "completed", "outcome": "no_input", "comments": []}}],
             "segments": [{"segment_id": "segment-1", "recording_id": "recording-1", "start_us": start + 1_000_000,
                  "end_us": start + 5_000_000, "start_ms": 1000, "end_ms": 5000, "activity": "active",
                  "source_ref": {"path": "MetaData/source.mp4", "sha256": "a" * 64}, "key_frames": [], "scene_frames": []}],
             "understandings": [{"segment_id": "segment-1", "status": "completed", "windows": [{"start_ms": 1000,
                 "end_ms": 5000, "summary": "Observed candidate", "input": "MultimodalUnderstanding/Input.json",
                 "model_receipt": "MultimodalUnderstanding/Result.json", "source_frames": [{"capture_us": start + 2_000_000, "source_pts": 20}]}]}]}
    inbox = FieldEventInbox(tmp_path / "device-day")
    bound = binding("person-binding", 1, "a", "2026-10-14T01:00:00Z", "2026-10-14T01:10:00Z")
    inbox.ingest("http://field.test", "camera-1", page([bound]), tenant="lab-1")
    first = consume_device_index(config, index, archive)
    assert first["model_invoked"] is False and first["events"] == 4
    latest = next((tmp_path / "device-day" / "PersonDay").glob("*/Latest.json"))
    reports = json.loads(latest.read_text())["reports"]
    assert any(e["type"] == "video_observation" and e["operator_id"] is None for e in reports[0]["events"])
    available = [e for e in inbox.events() if e["type"] == "speech_availability"][0]
    assert available["payload"]["outcome"] == "no_input"
    assert not any(e["type"] == "speech_observation" for e in inbox.events())
    assert consume_device_index(config, index | {"updated_at": "later"}, archive)["events"] == 0
    changed = copy.deepcopy(index)
    changed["recordings"][0]["audio"] = {"status": "available", "artifacts": [{"path": "MetaData/audio.wav"}]}
    changed["recordings"][0]["transcription"] = {"status": "completed", "outcome": "transcribed",
         "comments": [{"start_us": start + 2_000_000, "end_us": start + 3_000_000, "text": "Actual saved transcript",
                       "transcript_path": "Comment/Transcript.json"}]}
    consume_device_index(config, changed, archive)
    assert any(e["type"] == "speech_observation" for e in inbox.events())
    _, revision = device_index_events(index, archive, tenant="lab-1")
    assert revision
    protected = context_events(config, {"camera_key": "camera-1", "recording_start_us": start,
                                        "recording_end_us": start + 1_000_000})
    assert protected["events"][0]["type"] == "instrument_binding"
    assert protected["events"][0]["start_us"] == start


def test_context_uses_overlap_not_nearest_time_and_does_not_create_an_inbox(tmp_path):
    from visioncortex.application.field_event_consumer import context_events
    config = {"storage": {"local_runtime_root": str(tmp_path)}, "device_day": {"field_recognition": {
        "enabled": True, "endpoint": "http://field.test", "camera_ids": ["camera-1"], "tenant": "lab-1"}}}
    recording = {"camera_key": "camera-1", "recording_start_us": 10, "recording_end_us": 20}
    assert context_events(config, recording)["status"] == "not_received"
    assert not (tmp_path / "device-day" / "field-events.sqlite3").exists()


def test_half_open_duration_ends_at_handover_and_explicit_operator_has_own_view():
    values = [binding("a-1", 1, "a", "2026-10-14T01:00:00Z", "2026-10-14T01:10:00Z"),
              binding("b-1", 2, "b", "2026-10-14T01:10:00Z")]
    duration = event("duration", 3, when="2026-10-14T01:05:00Z")
    duration["occurred_end"] = "2026-10-14T01:10:00Z"
    operator = event("operator", 4, wearer="a")
    operator["operator_id"] = "actual-operator"
    operator["associations"]["operator"] = {"id": "actual-operator", "state": "human_reviewed",
                                            "evidence_refs": operator["evidence_refs"], "version": 1}
    result = person_days(values + [duration, operator])
    people = {report["person_id"]: report for report in result["items"]}
    assert "duration" in {e["entity_id"] for e in people["a"]["events"]}
    assert "duration" not in {e["entity_id"] for e in people["b"]["events"]}
    received = people["actual-operator"]["events"][0]
    assert received["person_roles"] == ["operator"] and received["wearer_id"] == "a"


def test_person_day_splits_real_interval_at_midnight_without_changing_source():
    value = event(wearer="a", when="2026-10-14T15:59:59Z")
    value["occurred_end"] = "2026-10-14T16:00:01Z"
    result = person_days([value])
    assert [report["date"] for report in result["items"]] == ["2026-10-14", "2026-10-15"]
    assert all(report["events"][0]["occurred_start"] == value["occurred_start"] for report in result["items"])
    value["occurred_end"] = "2026-10-14T16:00:00Z"
    assert len(person_days([value])["items"]) == 1


def test_renderer_policy_change_republishes_without_refetch_or_paid_work(tmp_path, monkeypatch):
    import visioncortex.application.field_event_consumer as module
    consumer = FieldEventConsumer(tmp_path, {"enabled": True, "endpoint": "http://field.test",
                                           "camera_ids": ["camera-1"], "tenant": "lab-1"})
    consumer.inbox.ingest("http://field.test", "camera-1", page([event(wearer="a")]), tenant="lab-1")
    first = consumer.publish()
    before = list((tmp_path / "PersonDay").glob("*/Revision*.json"))
    actual = module.file_hash
    monkeypatch.setattr(module, "file_hash", lambda path: actual(path) + "policy-correction")
    second = consumer.publish()
    assert first["generation"] == second["generation"]
    assert second["published"] is True
    assert len(list((tmp_path / "PersonDay").glob("*/Revision*.json"))) == len(before) + 1
    assert consumer.inbox.version_count() == 1


def test_person_day_capacity_retains_dirty_cursor_history_and_old_files(tmp_path):
    from visioncortex.application.field_event_consumer import PersonDayCapacityExceeded
    consumer = FieldEventConsumer(tmp_path, {"enabled": True, "endpoint": "http://field.test",
                                           "camera_ids": ["camera-1"], "tenant": "lab-1", "person_day_max_bytes": 4096})
    large = event(wearer="a")
    large["payload"]["raw_lines"] = [{"text": "saved evidence " * 500}]
    consumer.inbox.ingest("http://field.test", "camera-1", page([large]), tenant="lab-1")
    with pytest.raises(PersonDayCapacityExceeded):
        consumer.publish()
    assert consumer.inbox.cursor("http://field.test", "camera-1")["cursor"] == 1
    assert consumer.inbox.publication_state()["dirty"] is True
    assert consumer.inbox.version_count() == 1
    assert not (tmp_path / "PersonDay" / "Manifest.json").exists()


def test_inbox_rejects_network_storage_before_creation(tmp_path, monkeypatch):
    import visioncortex.local_storage as storage
    monkeypatch.setattr(storage, "mount_filesystem_type", lambda path: "cifs")
    with pytest.raises(RuntimeError, match="local"):
        FieldEventInbox(tmp_path / "network")
    assert not (tmp_path / "network").exists()


def test_timezone_change_rerenders_person_date_without_new_event_or_model(tmp_path):
    import json
    settings = {"enabled": True, "endpoint": "http://field.test", "camera_ids": ["camera-1"], "tenant": "lab-1"}
    consumer = FieldEventConsumer(tmp_path, settings)
    consumer.inbox.ingest("http://field.test", "camera-1", page([event(wearer="a", when="2026-10-14T18:30:00Z")]), tenant="lab-1")
    consumer.publish()
    latest = next((tmp_path / "PersonDay").glob("*/Latest.json"))
    assert json.loads(latest.read_text())["reports"][0]["date"] == "2026-10-15"
    result = FieldEventConsumer(tmp_path, settings, timezone="UTC").publish()
    assert result["published"] is True
    assert json.loads(latest.read_text())["reports"][0]["date"] == "2026-10-14"
    assert consumer.inbox.version_count() == 1


def test_seen_device_index_revision_cannot_revert_current_or_clear_withdrawals(tmp_path):
    inbox = FieldEventInbox(tmp_path)
    original = event(wearer="a")
    inbox.replace_device_index("archive-1", "revision-1", [original])
    corrected = event(wearer="b")
    inbox.replace_device_index("archive-1", "revision-2", [corrected])
    generation = inbox.publication_state()["generation"]
    with pytest.raises(ValueError, match="Historical"):
        inbox.replace_device_index("archive-1", "revision-1", [original])
    assert inbox.events()[0]["wearer_id"] == "b"
    assert inbox.publication_state()["generation"] == generation
    inbox.replace_device_index("archive-1", "revision-empty", [])
    with pytest.raises(ValueError, match="Historical"):
        inbox.replace_device_index("archive-1", "revision-2", [corrected])
    assert inbox.events()[0]["withdrawn"] is True
    # Empty indexes still have a persisted historical identity.
    inbox.replace_device_index("archive-1", "revision-empty-2", [])
    with pytest.raises(ValueError, match="Historical"):
        inbox.replace_device_index("archive-1", "revision-empty", [])


def test_publication_lock_is_shared_across_processes_and_snapshot_is_taken_after_ownership(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from visioncortex.device_day import exclusive
    settings = {"enabled": True, "endpoint": "http://field.test", "camera_ids": ["camera-1"], "tenant": "lab-1"}
    consumer = FieldEventConsumer(tmp_path, settings)
    consumer.inbox.ingest("http://field.test", "camera-1", page([event(wearer="a")]), tenant="lab-1")
    consumer.publish()
    script = """
import json,sys
from visioncortex.application.field_event_consumer import FieldEventConsumer
consumer=FieldEventConsumer(sys.argv[1],json.loads(sys.argv[2]))
print(json.dumps(consumer.publish()))
"""
    with exclusive(tmp_path / "locks" / "person-day-publication.lock"):
        child = subprocess.run([sys.executable, "-c", script, str(tmp_path), json.dumps(settings)],
                               env=os.environ.copy(), capture_output=True, text=True, timeout=10)
        assert child.returncode == 0, child.stderr
        assert json.loads(child.stdout)["status"] == "running_elsewhere"
        consumer.inbox.ingest("http://field.test", "camera-1", page([event(sequence=2, wearer="b")], after=1), tenant="lab-1")
        manifest = json.loads((tmp_path / "PersonDay" / "Manifest.json").read_text())
        assert manifest["generation"] == 1
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path), json.dumps(settings)],
                           env=os.environ.copy(), capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["generation"] == 2
    latest = next((tmp_path / "PersonDay").glob("*/Latest.json"))
    assert json.loads(latest.read_text())["reports"][0]["person_id"] == "b"
    assert consumer.publish()["published"] is False


def test_one_camera_source_failure_does_not_block_other_committed_personnel_events(tmp_path):
    import json
    from urllib.parse import parse_qs, urlsplit
    calls = []
    def fetch(url, timeout):
        camera = parse_qs(urlsplit(url).query)["camera_id"][0]
        calls.append(camera)
        if camera == "camera-2":
            raise OSError("single-device source failure")
        return page([event(wearer="a")])
    consumer = FieldEventConsumer(tmp_path, {"enabled": True, "endpoint": "http://field.test",
        "camera_ids": ["camera-2", "camera-1"], "tenant": "lab-1"}, fetch_json=fetch)
    result = consumer.tick()
    assert result["status"] == "partial" and result["inserted"] == 1
    assert calls == ["camera-2", "camera-1"]
    assert result["errors"][0]["camera_id"] == "camera-2"
    assert consumer.inbox.cursor("http://field.test", "camera-2")["cursor"] == 0
    assert consumer.inbox.cursor("http://field.test", "camera-1")["cursor"] == 1
    manifest = json.loads((tmp_path / "PersonDay" / "Manifest.json").read_text())
    assert manifest["generation"] == 1
    latest = next((tmp_path / "PersonDay").glob("*/Latest.json"))
    assert json.loads(latest.read_text())["reports"][0]["person_id"] == "a"
