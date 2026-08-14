import json
import threading
import time

import numpy as np

from labvision_evidence import archive
from labvision_evidence.schemas import (
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    VideoInfo,
    ViewInput,
    ViewRole,
)


def test_key_material_roles_export_concurrently_and_write_runtime(monkeypatch, tmp_path):
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.create()
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=tmp_path / "fp.mp4"),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=tmp_path / "tp.mp4"),
    ]
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=60_000,
            fps=30,
            width=1280,
            height=720,
            frame_count=1800,
        )
        for view in views
    }
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            confidence=1.0,
            state="aligned",
        )
        for view in views
    }
    event = EvidenceEvent(
        event_id="EVT-000001",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=10_000,
        global_end_ms=12_000,
        key_global_ms=11_000,
        objects=["tube"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    group = ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-0001"],
        global_start_ms=9_000,
        global_end_ms=13_000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
        experiment_name_en="Contact-Test",
        archive_folder="01-Contact-Test",
        key_event_ids=[event.event_id],
    )

    class FakeReader:
        def __init__(self, max_open=1):
            self.max_open = max_open

        def read(self, view, info, local_ms):
            return np.zeros((24, 32, 3), dtype=np.uint8)

        def close(self):
            return None

    concurrency = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def fake_extract(view, info, destination, start_ms, duration_ms, encoder):
        with lock:
            concurrency["active"] += 1
            concurrency["maximum"] = max(concurrency["maximum"], concurrency["active"])
        time.sleep(0.03)
        destination.write_bytes(b"clip")
        with lock:
            concurrency["active"] -= 1

    monkeypatch.setattr(archive, "ViewFrameReader", FakeReader)
    monkeypatch.setattr(
        archive,
        "nearest_frame_evidence_many",
        lambda path, timestamps: {float(timestamp): None for timestamp in timestamps},
    )
    monkeypatch.setattr(
        archive,
        "write_annotated_frame",
        lambda frame, boxes, destination: destination.write_bytes(b"frame"),
    )
    monkeypatch.setattr(archive, "extract_view_clip", fake_extract)
    monkeypatch.setattr(
        archive,
        "_write_aligned_frame",
        lambda first, third, destination, labels: destination.write_bytes(b"aligned-frame"),
    )
    monkeypatch.setattr(
        archive,
        "create_grid_video",
        lambda clips, destination: destination.write_bytes(b"aligned-clip"),
    )

    archive.materialize_key_materials(
        layout,
        [event],
        [group],
        views,
        infos,
        transforms,
        {"fp": tmp_path / "fp.jsonl", "tp": tmp_path / "tp.jsonl"},
        {
            "segmentation": {"key_clip_pre_seconds": 2, "key_clip_post_seconds": 3},
            "performance": {"ffmpeg_video_encoder": "h264_nvenc", "materialization_workers": 2},
        },
    )

    assert concurrency["maximum"] == 2
    assert len(event.key_frames) == 3
    assert len(event.key_clips) == 3
    runtime_path = layout.json_config / "key_material_materialization_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["workers"] == 2
    assert runtime["accepted_event_count"] == 1
    assert len(runtime["records"]) == 3
    assert runtime["total_duration_seconds"] > 0
