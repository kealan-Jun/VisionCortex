import json
import threading
import time
from pathlib import PurePosixPath, PureWindowsPath

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


def test_unobserved_action_categories_stay_in_index_without_json_only_folders(
    tmp_path,
):
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.create()
    event = EvidenceEvent(
        event_id="EVT-OBSERVED",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["gloved_hand", "pipette"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    group = ExperimentGroup(
        group_id="GROUP-1",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-1"],
        global_start_ms=1000,
        global_end_ms=2000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
        experiment_name="Test",
        experiment_name_en="Test",
        archive_folder="001-Test",
        key_event_ids=[event.event_id],
    )

    # Reproduce a category tree created before the final quality pass.
    archive.write_key_material_category_index(
        layout,
        [group],
        [event],
        include_empty_categories=True,
    )
    stale_frame_category = (
        layout.key_frames / "001-Test" / "03-Liquid-Movement"
    )
    stale_clip_category = (
        layout.key_clips / "001-Test" / "03-Liquid-Movement"
    )
    assert (stale_frame_category / "Category.json").is_file()
    assert (stale_clip_category / "Category.json").is_file()

    path = archive.write_key_material_category_index(
        layout,
        [group],
        [event],
        include_empty_categories=False,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {
        item["action_type"]: item
        for item in payload["experiments"][0]["action_categories"]
    }
    assert categories["hand_object_contact"]["materialized"] is True
    assert categories["liquid_movement"]["materialized"] is False
    assert categories["liquid_movement"]["coverage_status"] == "not_observed"
    assert categories["liquid_movement"]["folder"] is None
    assert not stale_frame_category.exists()
    assert not stale_clip_category.exists()


def test_key_material_view_selection_uses_real_same_role_fallback_at_short_tail(
    tmp_path,
):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=tmp_path / "fp.mp4"),
        ViewInput(
            view_id="tp-primary",
            role=ViewRole.THIRD_PERSON,
            video=tmp_path / "tp-primary.mp4",
        ),
        ViewInput(
            view_id="tp-fallback",
            role=ViewRole.THIRD_PERSON,
            video=tmp_path / "tp-fallback.mp4",
        ),
    ]
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=5_000,
            fps=30,
            width=1280,
            height=720,
            frame_count=150,
        )
        for view in views
    }
    transforms = {
        "fp": AlignmentTransform(view_id="fp", reference_view_id="fp"),
        "tp-primary": AlignmentTransform(
            view_id="tp-primary", reference_view_id="fp", offset_ms=1_000
        ),
        "tp-fallback": AlignmentTransform(
            view_id="tp-fallback", reference_view_id="fp", offset_ms=200
        ),
    }
    event = EvidenceEvent(
        event_id="EVT-000001",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=300,
        global_end_ms=700,
        key_global_ms=500,
        objects=["tube"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp-primary", "tp-fallback"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    group = ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-0001"],
        global_start_ms=0,
        global_end_ms=1_000,
        participating_views=[view.view_id for view in views],
        first_person_view="fp",
        third_person_view="tp-primary",
        continuity_reason="test",
        key_event_ids=[event.event_id],
    )

    pair, receipt = archive._select_key_material_view_pair(
        group, event, views, infos, transforms
    )

    assert pair == ("fp", "tp-fallback")
    assert receipt["fallback_applied"] is True
    assert receipt["timestamp_clamped"] is False
    assert receipt["synthetic_cross_view_evidence"] is False
    primary = next(
        item
        for item in receipt["candidates"]["third_person"]
        if item["view_id"] == "tp-primary"
    )
    assert primary["local_key_ms"] == -500
    assert primary["in_physical_bounds"] is False


def test_key_material_view_selection_restores_accepted_peak_for_dual_coverage(
    tmp_path,
):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=tmp_path / "fp.mp4"),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=tmp_path / "tp.mp4"),
    ]
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=5_000,
            fps=30,
            width=1280,
            height=720,
            frame_count=150,
        )
        for view in views
    }
    transforms = {
        "fp": AlignmentTransform(view_id="fp", reference_view_id="fp"),
        "tp": AlignmentTransform(
            view_id="tp", reference_view_id="fp", offset_ms=1_000
        ),
    }
    event = EvidenceEvent(
        event_id="EVT-000001",
        action_type=ActionType.OBJECT_MOVEMENT,
        global_start_ms=200,
        global_end_ms=3_000,
        key_global_ms=400,
        objects=["pipette"],
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
        global_start_ms=0,
        global_end_ms=3_000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
        key_event_ids=[event.event_id],
    )

    pair, receipt = archive._select_key_material_view_pair_with_peak_fallback(
        group, event, views, infos, transforms, accepted_peak_global_ms=1_900
    )

    assert pair == ("fp", "tp")
    assert event.key_global_ms == 1_900
    assert receipt["key_timestamp_fallback"]["applied"] is True
    assert receipt["key_timestamp_fallback"]["timestamp_clamped"] is False


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
        lambda clips, destination, encoder: destination.write_bytes(b"aligned-clip"),
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
    expected_category = "01-Hand-Object-Contact"
    for relative in (*event.key_frames.values(), *event.key_clips.values()):
        assert f"/{group.archive_folder}/{expected_category}/" in relative
        assert "Contact-Hand-With-Tube_EVT-000001" in relative
    for category in (
        "01-Hand-Object-Contact",
        "02-Object-Movement",
        "03-Liquid-Movement",
        "04-Container-State-Change",
        "05-Device-Panel-Operation",
        "06-Pipette-Transfer-Operation",
    ):
        assert (layout.key_frames / group.archive_folder / category).is_dir()
        assert (layout.key_clips / group.archive_folder / category).is_dir()
        assert (
            layout.key_frames / group.archive_folder / category / "Category.json"
        ).is_file()
        assert (
            layout.key_clips / group.archive_folder / category / "Category.json"
        ).is_file()
    category_index_path = layout.key_materials / "Key-Material-Category-Index.json"
    category_index = json.loads(category_index_path.read_text(encoding="utf-8"))
    assert category_index["category_count"] == 6
    experiment = category_index["experiments"][0]
    assert experiment["group_id"] == group.group_id
    assert experiment["key_event_count"] == 1
    categories = {
        item["action_type"]: item for item in experiment["action_categories"]
    }
    assert categories["hand_object_contact"]["event_count"] == 1
    assert categories["hand_object_contact"]["events"][0]["event_id"] == event.event_id
    assert categories["liquid_movement"]["event_count"] == 0
    empty_summary = json.loads(
        (
            layout.key_frames
            / group.archive_folder
            / "03-Liquid-Movement"
            / "Category.json"
        ).read_text(encoding="utf-8")
    )
    assert empty_summary["media_kind"] == "key_frame"
    assert empty_summary["event_count"] == 0
    assert empty_summary["coverage_status"] == "not_observed"
    assert empty_summary["absence_reason"]
    frame_sidecar = json.loads(
        (layout.root / event.key_frames["fp"]).with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    classification = frame_sidecar["provenance"]["archive_classification"]
    assert classification["experiment_folder"] == group.archive_folder
    assert classification["action_category_folder"] == expected_category
    assert classification["primary_object"] == "Tube"
    assert classification["semantic_file_stem"] == (
        "Contact-Hand-With-Tube_EVT-000001"
    )
    assert classification["object_labels"] == ["Tube"]
    runtime_path = layout.json_config / "key_material_materialization_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["workers"] == 2
    assert runtime["accepted_event_count"] == 1
    assert runtime["archive_hierarchy_version"] == "2.0.0"
    assert runtime["category_index"] == "Key-Materials/Key-Material-Category-Index.json"
    assert len(runtime["records"]) == 3
    assert runtime["total_duration_seconds"] > 0


def test_component_budget_applies_classic_path_limit_only_on_windows(monkeypatch):
    parent = PurePosixPath("/tmp") / ("long-parent-" * 20)
    windows_parent = PureWindowsPath("Y:/" + ("long-parent-" * 20))

    monkeypatch.setattr(archive.os, "name", "posix")
    assert archive._component_budget(parent, 26, maximum_chars=40) == 40
    assert archive._component_budget(windows_parent, 26, maximum_chars=40) == 12

    monkeypatch.setattr(archive.os, "name", "nt")
    assert archive._component_budget(parent, 26, maximum_chars=40) == 12
