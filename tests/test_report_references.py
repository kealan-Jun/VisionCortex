import json
import sqlite3

from visioncortex.report_references import export_reference_index


def test_portable_report_preserves_event_artifact_and_source_links(tmp_path):
    database = tmp_path / "JSON-Config-Files/evidence_index.sqlite"
    database.parent.mkdir()
    with sqlite3.connect(database) as c:
        c.executescript("CREATE TABLE key_events(event_uid TEXT,event_json TEXT);"
                       "CREATE TABLE artifacts(artifact_uid TEXT,artifact_json TEXT);"
                       "CREATE TABLE evidence(evidence_uid TEXT,evidence_json TEXT);")
        c.execute("INSERT INTO key_events VALUES (?,?)", ("event-uid", json.dumps({"event_id": "E1"})))
        c.execute("INSERT INTO artifacts VALUES (?,?)", ("image-uid", json.dumps({"artifact_uid": "image-uid", "event_uid": "event-uid", "path": "Key-Materials/frame.jpg"})))
        c.execute("INSERT INTO evidence VALUES (?,?)", ("source-uid", json.dumps({"event_uid": "event-uid", "frame_index": 17, "view_id": "fp"})))
    before = database.read_bytes()
    groups = [{"group_id": "G1", "model_understanding": {"steps": [{"supporting_event_ids": ["E1", "missing"]}]}}]
    result = export_reference_index(tmp_path, groups)
    assert result["status"] == "references_incomplete"
    assert result["steps"][0]["event_uids"] == ["event-uid"]
    assert result["unresolved_event_ids"] == ["missing"]
    assert result["artifacts"][0]["event_uid"] == result["events"][0]["event_uid"]
    assert result["evidence"][0]["frame_index"] == 17
    assert result["formal_acceptance_implied"] is False
    assert database.read_bytes() == before


def test_missing_index_is_unavailable_not_an_empty_complete_result(tmp_path):
    assert export_reference_index(tmp_path, [])["status"] == "not_available"
