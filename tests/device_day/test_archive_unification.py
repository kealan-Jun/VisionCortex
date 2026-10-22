"""Historical organization preserves existing bytes and never runs models."""

import pytest

from visioncortex.application.archive_unification import (
    ArchiveUnifier,
    read_source_projection,
)
from visioncortex.device_day_contract import (
    DeviceDayLayout,
    artifact,
    atomic_json,
    verify_artifact,
)
from visioncortex.device_day_content_paths import aliases, relocated


def case(tmp_path):
    root = tmp_path / "archives"
    start = 1792639450000000
    layout = DeviceDayLayout(root, "camera_cam01", start)
    source = layout.root / "MetaData/112410/rgb.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original-must-never-change")
    atomic_json(
        source.parent / "meta.json",
        {
            "recording_window_first_valid_rgb_global_us": start,
            "recording_window_last_valid_rgb_global_us": start + 120000000,
        },
    )
    reference = artifact(layout.root, source)
    record = {
        "recording_id": "record",
        "start_us": start,
        "end_us": start + 120000000,
        "sources": [{"kind": "video", "retained": reference}],
        "stages": {},
    }
    segments = []
    old_refs = []
    for i in range(2):
        folder = (
            layout.processed
            / "Clips"
            / f"11-24-{10 + i:02}_11-24-{11 + i:02}_ExperimentActivity_old{i}"
        )
        folder.mkdir(parents=True)
        video = folder / "ExperimentActivity.mp4"
        video.write_bytes(b"previous-derived-video-" + bytes([i]))
        video_ref = artifact(layout.root, video)
        old_refs.append(video_ref)
        segment = {
            "schema_version": "visioncortex-device-day/1",
            "recording_id": "record",
            "segment_id": f"segment-old{i}",
            "source_ref": reference,
            "start_us": start + i * 1000000,
            "end_us": start + (i + 1) * 1000000,
            "start_ms": i * 1000,
            "end_ms": (i + 1) * 1000,
            "activity": "active",
            "video": video_ref,
            "json_path": layout.relative(folder / "ExperimentActivity.json"),
        }
        atomic_json(folder / "ExperimentActivity.json", segment)
        meaning = (
            layout.understanding
            / f"11-24-{10 + i:02}.000000_11-24-{11 + i:02}.000000"
            / "Understanding.json"
        )
        atomic_json(
            meaning,
            {
                "schema_version": "visioncortex-readable-understanding/1",
                "recording_id": "record",
                "segment_id": segment["segment_id"],
                "source_video": reference,
                "start_us": segment["start_us"],
                "end_us": segment["end_us"],
                "source_start_ms": segment["start_ms"],
                "source_end_ms": segment["end_ms"],
                "activity": "active",
                "status": "completed",
                "understanding": [{"text": f"原有真实返回文本 {i}"}],
                "experiment_steps": [{"description": f"原有步骤 {i}"}],
            },
        )
        segments.append(segment)
    index = {"archive": layout.name, "recordings": [record], "segments": segments}
    atomic_json(layout.index, index)
    config = {
        "storage": {
            "archive_root": str(root),
            "local_runtime_root": str(tmp_path / "runtime"),
        },
        "runtime": {"local_only": True},
        "device_day": {"clip_delivery": "source_recording_v2"},
    }
    return layout, index, config, source, old_refs


def test_consolidation_preserves_video_payloads_meaning_and_originals(
    tmp_path, monkeypatch
):
    from visioncortex.device_day import DeviceDayRunner

    monkeypatch.setattr(
        DeviceDayRunner,
        "process",
        lambda *a, **k: pytest.fail("No model-stage work is allowed"),
    )
    layout, index, config, source, refs = case(tmp_path)
    original = source.read_bytes()
    canonical = layout.index.read_bytes()
    unifier = ArchiveUnifier(config, layout.name, index)
    assert unifier.discover()["originals"] == 1
    progress = unifier.publish(maximum_new_videos=2)
    assert not progress["pending"]
    clips = [
        p
        for p in (layout.processed / "Clips").iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    meanings = [
        p
        for p in layout.understanding.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    assert len(clips) == len(meanings) == 1 and clips[0].name == meanings[0].name
    assert clips[0].name.startswith("11-24-10_11-26-10_NoExperimentActivity_")
    assert len(list(clips[0].glob("*.mp4"))) == 2
    assert all(verify_artifact(layout.root, ref) for ref in refs)
    from visioncortex.device_day_contract import read_json

    docs = [
        read_json(p)
        for p in meanings[0].glob("*.json")
        if p.name != "Understanding.json"
    ]
    assert sorted(d["experiment_steps"][0]["description"] for d in docs) == [
        "原有步骤 0",
        "原有步骤 1",
    ]
    assert all(not d["physical_action_confirmed"] for d in docs)
    assert all(not d["historical_format_migration"]["model_invoked"] for d in docs)
    assert source.read_bytes() == original and layout.index.read_bytes() == canonical
    assert read_json(layout.understanding / "Index.json")["理解进度"]["片段总数"] == 2
    assert all(
        (layout.root / relocated(ref["path"], aliases(layout.root))).is_file()
        for ref in refs
    )


def test_budgeted_migration_can_resume_without_losing_unpublished_video(tmp_path):
    layout, index, config, source, refs = case(tmp_path)
    first = ArchiveUnifier(config, layout.name, index)
    first.discover()
    assert first.publish(maximum_new_videos=1)["pending"]
    second = ArchiveUnifier(config, layout.name, index)
    second.discover()
    assert not second.publish(maximum_new_videos=2)["pending"]
    assert all(verify_artifact(layout.root, ref) for ref in refs)


def test_crash_after_retirement_recovers_the_full_plan_and_saved_meaning(
    tmp_path, monkeypatch
):
    from visioncortex.application import clip_delivery
    from visioncortex.device_day_contract import read_json

    layout, index, config, source, refs = case(tmp_path)
    canonical = layout.index.read_bytes()
    original = source.read_bytes()
    first = ArchiveUnifier(config, layout.name, index)
    first.discover()
    checkpoint = tmp_path / "discovery.json"
    first.checkpoint(checkpoint)
    real = clip_delivery.publish
    monkeypatch.setattr(
        clip_delivery,
        "publish",
        lambda *a, **k: (_ for _ in ()).throw(OSError("interrupted")),
    )
    with pytest.raises(OSError, match="interrupted"):
        first.publish()
    monkeypatch.setattr(clip_delivery, "publish", real)
    recovered = ArchiveUnifier(config, layout.name, index)
    recovered.restore(checkpoint)
    assert recovered.discover()["intervals"] == 2
    assert not recovered.publish()["pending"]
    assert source.read_bytes() == original and layout.index.read_bytes() == canonical
    assert all(verify_artifact(layout.root, ref) for ref in refs)
    docs = [
        read_json(p)
        for p in layout.understanding.glob("*/*.json")
        if p.name != "Understanding.json" and not p.parent.name.startswith(".")
    ]
    assert sorted(d["experiment_steps"][0]["description"] for d in docs) == [
        "原有步骤 0",
        "原有步骤 1",
    ]


def test_native_move_journals_recover_a_legacy_interruption_without_a_checkpoint(
    tmp_path,
):
    from visioncortex.application import clip_delivery

    layout, index, config, source, refs = case(tmp_path)
    for parent in (layout.processed / "Clips", layout.understanding):
        for folder in list(parent.iterdir()):
            if folder.is_dir():
                clip_delivery._retire(layout, folder)
    recovered = ArchiveUnifier(
        config, layout.name, {"archive": layout.name, "recordings": [], "segments": []}
    )
    assert recovered.discover()["intervals"] == 2
    assert not recovered.publish()["pending"]
    assert all(verify_artifact(layout.root, ref) for ref in refs)


def test_separate_no_value_retention_outranks_stale_model_classification(tmp_path):
    from visioncortex.device_day_contract import read_json

    layout, index, config, source, refs = case(tmp_path)
    for ref in refs:
        folder = (layout.root / ref["path"]).parent
        candidate = read_json(folder / "ExperimentActivity.json")
        atomic_json(
            folder / "Retention.json",
            {
                k: candidate[k]
                for k in ("recording_id", "source_ref", "start_us", "end_us")
            }
            | {"no_value_proven": True},
        )
    for path in layout.understanding.glob("*/Understanding.json"):
        atomic_json(path, read_json(path) | {"display_category": "ExperimentActivity"})
    unifier = ArchiveUnifier(config, layout.name, index)
    unifier.discover()
    assert {r["category"] for r in unifier.values} == {"NoValueNoExperimentActivity"}
    unifier.publish()
    assert not list((layout.processed / "Clips").glob("*/*.mp4"))
    assert all(verify_artifact(layout.root, ref) for ref in refs)


def test_migration_rejects_a_changed_historical_video_before_relocation(tmp_path):
    layout, index, config, source, refs = case(tmp_path)
    (layout.root / refs[0]["path"]).write_bytes(b"changed-video")
    with pytest.raises(ValueError, match="checksum"):
        ArchiveUnifier(config, layout.name, index).discover()
    assert (layout.root / refs[0]["path"]).is_file()


def test_source_projection_drops_dense_data_and_keeps_literal_source_paths(tmp_path):
    import ijson

    p = tmp_path / "index.json"
    atomic_json(
        p,
        {
            "archive": "2026-10-22_camera_cam01",
            "recordings": [
                {
                    "recording_id": "r",
                    "start_us": 1,
                    "end_us": 2,
                    "processing": {"batches": ["dense"]},
                }
            ],
            "segments": [
                {
                    "segment_id": "s",
                    "recording_id": "r",
                    "source_ref": {"path": "MetaVideo/a.b.mp4"},
                    "activity_audit": ["dense"],
                    "scene_frames": ["dense"],
                }
            ],
            "understandings": [{"large_response": "dense"}],
        },
    )
    before = p.read_bytes()
    value = read_source_projection(p, events=ijson.basic_parse)
    assert "understandings" not in value and "processing" not in value["recordings"][0]
    assert value["segments"] == [
        {
            "segment_id": "s",
            "recording_id": "r",
            "source_ref": {"path": "MetaVideo/a.b.mp4"},
        }
    ]
    assert p.read_bytes() == before


def test_flat_original_uses_recorded_frame_metadata_in_its_native_metadata_tree(
    tmp_path,
):
    from visioncortex.application.clip_delivery import original_units

    layout = DeviceDayLayout(tmp_path, "camera_cam01", 1792639450000000)
    video = layout.root / "MetaVideo/11-24-09_11-26-30.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original")
    atomic_json(
        video.parent / "Metadata" / video.stem / "Video/Meta.json",
        {
            "camera_key": "camera_cam01",
            "recording_window_first_valid_rgb_global_us": 1792639450000000,
            "recording_window_last_valid_rgb_global_us": 1792639582000000,
        },
    )
    units = original_units(layout, {"recordings": [], "segments": []}, [])
    assert next(iter(units.values()))["name"].startswith("11-24-10_11-26-22_")


def test_prefixed_capture_uses_its_own_metadata_and_preserves_physical_source_name(
    tmp_path,
):
    from visioncortex.application.clip_delivery import original_units

    layout = DeviceDayLayout(tmp_path, "camera_cam01", 1792639450000000)
    video = layout.root / "MetaData/112410/hivebench_0001_rgb.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original")
    atomic_json(
        video.with_name("hivebench_0001_meta.json"),
        {
            "camera_key": "camera_cam01",
            "recording_window_first_valid_rgb_global_us": 1792639450000000,
            "recording_window_last_valid_rgb_global_us": 1792639582000000,
        },
    )
    unit = original_units(layout, {"recordings": [], "segments": []}, [])[
        "MetaData/112410"
    ]
    assert unit["name"].startswith("11-24-10_11-26-22_")
    assert unit["observed_source"] == "MetaData/112410/hivebench_0001_rgb.mp4"
