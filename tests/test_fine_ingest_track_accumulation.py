from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3

import pytest

from visioncortex import candidate_index
from visioncortex.schemas import BoxEvidence, FrameEvidence, ViewInput, ViewRole


def box(track: int | None, center: float, name: str = "tube") -> BoxEvidence:
    return BoxEvidence(class_id=1, class_name=name, confidence=.9,
                       xyxy_norm=(center - .05, .2, center + .05, .4), track_id=track)


def frame(timestamp: float, *detections: BoxEvidence, index: int = 0) -> FrameEvidence:
    return FrameEvidence(view_id="fp", role=ViewRole.FIRST_PERSON, frame_index=index,
                         local_ms=timestamp, global_ms=timestamp + 10,
                         width=640, height=480, detections=list(detections))


@pytest.fixture
def setup_index(tmp_path):
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("unread.mp4"))
    index = candidate_index.create_fine_frame_index(tmp_path / "index.sqlite3")

    def ingest(label, frames, **kwargs):
        path = tmp_path / f"{label}.jsonl"
        path.write_text("".join(f.model_dump_json() + "\n" for f in frames))
        return candidate_index.ingest_fine_frame_ledgers(index, [view], {"fp": path}, source_pass=label, **kwargs)

    return index, ingest


def rows(index, table):
    with sqlite3.connect(index.path) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3").fetchall()


def test_overlap_out_of_order_and_equal_timestamp_keep_exact_endpoint_and_refs(setup_index):
    index, ingest = setup_index
    ingest("base", [frame(100, box(10, .2)), frame(200, box(10, .3))])
    receipt = ingest("supp", [frame(200, box(9, .3)), frame(50, box(9, .1)),
                              frame(200, box(9, .35), index=4), frame(150, box(9, .8))])
    track = rows(index, "fine_tracks")[0]
    assert track[:5] == ("fp", 1, "tube", 50., 200.)
    assert track[5:7] == pytest.approx((.35, .3))
    assert json.loads(track[7]) == ["base:10", "supp:9"]
    assert receipt["views"][0]["replaced_frames"] == 2
    assert receipt["views"][0]["indexed_frames"] == 4
    assert receipt["stitched_tracks"] == 1
    frames = list(index.iter_frames("fp"))
    assert [f.local_ms for f in frames] == [50., 100., 150., 200.]
    assert frames[-1].frame_index == 4
    assert frames[-1].detections[0].xyxy_norm == box(9, .35).xyxy_norm
    assert rows(index, "fine_track_stitches")[-1][4:] == ("overlap_iou", 0., 0.)


def test_unassigned_old_track_remains_eligible_but_assigned_tracks_cannot_be_stolen(setup_index):
    index, ingest = setup_index
    ingest("base", [frame(0, box(1, .25), box(2, .29))])
    receipt = ingest("supp", [frame(50, box(7, .25)), frame(100, box(8, .25)),
                              frame(150, box(9, .25)), frame(200, box(7, .3), box(8, .3), box(9, .3))])
    assert receipt["stitched_tracks"] == 2
    assert receipt["new_tracks"] == 1
    assert [d.track_id for d in list(index.iter_frames("fp"))[-1].detections] == [1, 2, 3]
    stitches = [r for r in rows(index, "fine_track_stitches") if r[0] == "supp"]
    assert [(r[2], r[3], r[4]) for r in stitches] == [(7, 1, "recent_endpoint"),
                                                     (8, 2, "recent_endpoint"), (9, 3, "new_track")]


def test_new_track_keeps_initial_class_and_last_detection_wins_timestamp_ties(setup_index):
    index, ingest = setup_index
    ingest("primary", [frame(50, box(1, .2)), frame(50, box(1, .4, "beaker")),
                       frame(25, box(1, .8))])
    track = rows(index, "fine_tracks")[0]
    assert track[2:5] == ("tube", 25., 50.)
    assert track[5:7] == pytest.approx((.4, .3))
    assert json.loads(track[7]) == ["primary:1"]
    assert list(index.iter_frames("fp"))[-1].detections[0].class_name == "beaker"


def test_disabled_stitching_and_untracked_boxes_preserve_frame_replacement(setup_index):
    index, ingest = setup_index
    ingest("base", [frame(0, box(1, .2))])
    receipt = ingest("supp", [frame(0, box(1, .2), box(None, .7))], stitching_enabled=False)
    assert receipt["new_tracks"] == 1
    assert receipt["stitched_tracks"] == 0
    assert [d.track_id for d in list(index.iter_frames("fp"))[0].detections] == [2, None]
    assert [json.loads(r[-1]) for r in rows(index, "fine_tracks")] == [["base:1"], ["supp:1"]]


def test_overlap_payload_is_not_reparsed_for_already_mapped_source_tracks(setup_index, monkeypatch):
    index, ingest = setup_index
    ingest("base", [frame(t, box(1, .2)) for t in range(20)])
    original = FrameEvidence.model_validate_json
    calls = []

    def validate(payload, *args, **kwargs):
        calls.append(payload)
        return original(payload, *args, **kwargs)

    monkeypatch.setattr(FrameEvidence, "model_validate_json", validate)
    ingest("supp", [frame(t, box(9, .2)) for t in range(20)])
    # Twenty source frames are validated by the ledger reader; only the first
    # overlap needs decoding to establish the source-to-unified track mapping.
    assert len(calls) == 21


def test_track_sql_updates_scale_with_unique_tracks_not_detection_count(setup_index, monkeypatch):
    index, ingest = setup_index
    statements = []
    original_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(candidate_index.sqlite3, "connect", connect)
    ingest("primary", [frame(t, box(1, .2), box(2, .5)) for t in range(300)])
    writes = [sql for sql in statements if sql.startswith(("INSERT INTO fine_tracks ", "UPDATE fine_tracks "))]
    endpoint_reads = [sql for sql in statements if sql.startswith("SELECT first_local_ms, last_local_ms")]
    assert len(writes) == len(endpoint_reads) == 2
    assert len(list(index.iter_frames("fp"))) == 300


def test_optional_read_parse_timing_excludes_sql_work(setup_index, monkeypatch):
    _index, ingest = setup_index
    active, measured, sql_during_read = [], [], []

    class Timings:
        @contextmanager
        def measure(self, key, *, cpu):
            measured.append((key, cpu))
            active.append(key)
            try:
                yield
            finally:
                active.pop()

    original_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)

        def check_read_parse_scope(sql):
            if active:
                sql_during_read.append(sql)

        connection.set_trace_callback(check_read_parse_scope)
        return connection

    monkeypatch.setattr(candidate_index.sqlite3, "connect", connect)
    ingest("timed", [frame(0, box(1, .2)), frame(50, box(1, .3))], timings=Timings())
    assert measured == [("ledger_read_parse_seconds", True)] * 3
    assert not sql_during_read
