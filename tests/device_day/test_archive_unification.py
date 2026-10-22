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
    assert not list((layout.processed / "Clips").glob("*/*.mp4"))


def test_initial_public_inventory_recovers_only_its_retired_materializations(tmp_path):
    from visioncortex.application import clip_delivery

    layout, index, config, source, refs = case(tmp_path)
    for parent in (layout.processed / "Clips", layout.understanding):
        for folder in list(parent.iterdir()):
            if folder.is_dir():
                clip_delivery._retire(layout, folder)
    recovered = ArchiveUnifier(
        config, layout.name, index, public_video_inventory=refs[:1]
    )
    recovered.discover()
    assert not recovered.publish()["pending"]
    assert len(list((layout.processed / "Clips").glob("*/*.mp4"))) == 1
    assert all(verify_artifact(layout.root, ref) for ref in refs)


def test_duplicate_public_generation_is_retired_even_if_row_was_recovered_first(
    tmp_path,
):
    from visioncortex.application import clip_delivery
    from visioncortex.device_day_contract import read_json

    layout, index, config, source, refs = case(tmp_path)
    folder = (layout.root / refs[0]["path"]).parent
    original = read_json(folder / "ExperimentActivity.json")
    clip_delivery._retire(layout, folder)
    folder = folder.with_name(folder.name + "_another_generation")
    atomic_json(
        folder / "ExperimentActivity.json",
        original | {"dense_legacy_audit": "x" * (4 * 1024**2)},
    )
    unifier = ArchiveUnifier(config, layout.name, index)
    unifier.discover()
    assert folder in unifier.legacy_folders
    assert not unifier.publish()["pending"]
    assert not folder.exists()
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


@pytest.mark.parametrize("missing_file", [True, False])
def test_missing_derived_alias_index_is_rebuilt_from_moves_and_identical_videos(
    tmp_path,
    missing_file,
):
    layout, index, config, source, refs = case(tmp_path)
    first = ArchiveUnifier(config, layout.name, index)
    first.discover()
    first.publish()
    for directory in (layout.processed, layout.understanding):
        path = directory / ".Evidence/LegacyPaths.json"
        if missing_file:
            path.unlink()
        else:
            atomic_json(
                path,
                {"schema_version": "visioncortex-content-aliases/1", "aliases": {}},
            )
    recovered = ArchiveUnifier(config, layout.name, index)
    assert recovered.discover()["intervals"] == 2
    assert not recovered.publish()["pending"]
    assert all(verify_artifact(layout.root, ref) for ref in refs)


def test_returning_public_group_does_not_alias_back_into_its_video_history(tmp_path):
    import shutil
    from visioncortex.application import clip_delivery
    from visioncortex.device_day_contract import read_json

    layout, index, config, source, refs = case(tmp_path)
    first = ArchiveUnifier(config, layout.name, index)
    first.discover()
    first.publish()
    group = next(p for p in (layout.processed / "Clips").iterdir() if p.is_dir())
    clip_delivery._retire(layout, group)
    journals = layout.processed / ".Evidence/ClipRelocations"
    move = next(
        read_json(p)
        for p in journals.glob("*.json")
        if read_json(p)["from"] == layout.relative(group)
    )
    saved = layout.root / move["to"]
    shutil.copytree(saved, group)
    # Reclaimed old copies can already point at the returning public material.
    mapping = {
        layout.relative(saved / p.name): layout.relative(p) for p in group.glob("*.mp4")
    }
    clip_delivery._mapping(layout, "ProcessedClips", mapping)
    recovered = ArchiveUnifier(config, layout.name, index)
    assert recovered.discover()["intervals"] == 2
    assert not recovered.publish()["pending"]
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


def test_prefixed_flat_archive_uses_full_rgb_extent_before_partial_processing_window(tmp_path):
    from visioncortex.application.clip_delivery import original_units

    layout = DeviceDayLayout(tmp_path, "camera_cam01", 1792639450000000)
    video = layout.root / "MetaVideo/11-24-09_11-26-30.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original")
    reference = artifact(layout.root, video)
    record = {"recording_id": "partial", "start_us": 1792639450000000,
              "end_us": 1792639451000000,
              "sources": [{"kind": "video", "retained": reference}]}
    meta = video.parent / "Metadata" / video.stem / "Video/CodexProbePlainMeta.json"
    atomic_json(meta, {"camera_key": "camera_cam01",
                      "recording_window_first_valid_rgb_global_us": 1792639450000000,
                      "recording_window_last_valid_rgb_global_us": 1792639582000000})
    before = {p: p.read_bytes() for p in (video, meta)}
    unit = next(iter(original_units(layout, {"recordings": [record]}, []).values()))
    assert unit["name"].startswith("11-24-10_11-26-22_")
    assert unit["time_basis"] == "recorded_rgb_frame_extent"
    assert all(p.read_bytes() == content for p, content in before.items())
    atomic_json(meta.with_name("AnotherMeta.json"), {"camera_key": "camera_cam01"})
    with pytest.raises(ValueError, match="ambiguous"):
        original_units(layout, {"recordings": [record]}, [])


def test_prefixed_flat_archive_rejects_metadata_for_another_camera(tmp_path):
    from visioncortex.application.clip_delivery import original_units

    layout = DeviceDayLayout(tmp_path, "camera_cam01", 1792639450000000)
    video = layout.root / "MetaVideo/11-24-09_11-26-30.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original")
    atomic_json(video.parent / "Metadata" / video.stem / "Video/ProbeMeta.json",
                {"camera_key": "other_cam02"})
    with pytest.raises(ValueError, match="another device/camera"):
        original_units(layout, {"recordings": []}, [])


def test_missing_historical_original_keeps_saved_results_without_inventing_a_source_group(tmp_path):
    from visioncortex.device_day_contract import read_json, safe_child

    layout, index, config, source, _ = case(tmp_path)
    missing = {"path": "MetaVideo/missing.mp4", "size_bytes": 10, "sha256": "a" * 64}
    index["recordings"].append({"recording_id": "missing", "sources": [],
                                "stages": {"retention": {"status": "failed"}}})
    folder = layout.processed / "Clips/11-24-20_11-24-21_NoExperimentActivity_12345678"
    segment = dict(index["segments"][0], recording_id="missing", segment_id="missing-segment",
                   source_ref=missing, video=None, start_ms=10000, end_ms=11000,
                   start_us=index["segments"][0]["start_us"] + 10000000,
                   end_us=index["segments"][0]["start_us"] + 11000000)
    atomic_json(folder / "NoExperimentActivity.json", segment)
    meaning = layout.understanding / "11-24-20.000000_11-24-21.000000/Understanding.json"
    atomic_json(meaning, dict(segment, source_video=missing, status="completed",
                             experiment_steps=[{"description": "saved synthetic model response"}]))
    saved = artifact(layout.root, meaning)
    atomic_json(layout.index, index)
    canonical, original = layout.index.read_bytes(), source.read_bytes()
    unifier = ArchiveUnifier(config, layout.name, index)
    found = unifier.discover()
    assert found["originals"] == 1 and found["unassigned_source_intervals"] == 1
    checkpoint = tmp_path / "Checkpoint.json"
    unifier.checkpoint(checkpoint)
    unifier.publish()
    assert verify_artifact(layout.root, saved)
    assert safe_child(layout.root, saved["path"]).read_bytes()
    gaps = read_json(layout.processed / ".Evidence/ArchiveUnification/SourceGaps.json")
    assert gaps["unassigned_history_evidence"] == "PARTIAL_EVIDENCE"
    assert gaps["unassigned_source_history"][0]["source_ref"] == missing
    assert len([p for p in (layout.processed / "Clips").iterdir() if p.is_dir()]) == 1
    assert not (layout.root / missing["path"]).exists()
    resumed = ArchiveUnifier(config, layout.name, read_json(layout.index))
    resumed.restore(checkpoint)
    assert resumed.discover()["unassigned_source_intervals"] == 1
    resumed.publish()
    assert verify_artifact(layout.root, saved)
    assert layout.index.read_bytes() == canonical and source.read_bytes() == original


def test_scene_frame_only_legacy_folder_is_preserved_with_exact_links_across_resume(tmp_path):
    from visioncortex.device_day_contract import read_json

    layout, index, config, _, _ = case(tmp_path)
    frame = layout.processed / "Clips/11-24-10_11-24-11_NoExperimentActivity_12345678/SceneFrames/00-00-01.000.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"synthetic-scene-frame")
    ref = artifact(layout.root, frame)
    unifier = ArchiveUnifier(config, layout.name, index)
    assert unifier.discover()["unbound_scene_frame_folders"] == 1
    checkpoint = tmp_path / "Checkpoint.json"
    unifier.checkpoint(checkpoint)
    unifier.publish()
    assert not frame.exists() and verify_artifact(layout.root, ref)
    resumed = ArchiveUnifier(config, layout.name, read_json(layout.index))
    resumed.restore(checkpoint)
    resumed.discover()
    resumed.publish()
    assert verify_artifact(layout.root, ref)
    assert len([p for p in (layout.processed / "Clips").iterdir() if p.is_dir()]) == 1


def test_unadmitted_retired_video_failure_is_reported_without_publishing_corrupt_bytes(tmp_path):
    from visioncortex.application.clip_delivery import _retire
    from visioncortex.device_day_contract import read_json, safe_child

    layout, index, config, source, refs = case(tmp_path)
    original = source.read_bytes()
    folder = (layout.root / refs[0]["path"]).parent
    _retire(layout, folder)
    retired = safe_child(layout.root, refs[0]["path"])
    retired.write_bytes(b"synthetic-preexisting-corrupt-retired-video")
    before = retired.read_bytes()
    unifier = ArchiveUnifier(config, layout.name, index, public_video_inventory=[])
    assert unifier.discover()["unadmitted_retired_video_gaps"] == 1
    unifier.publish()
    gaps = read_json(layout.processed / ".Evidence/ArchiveUnification/SourceGaps.json")
    assert gaps["unadmitted_retired_video_gaps"][0]["expected_video"] == refs[0]
    assert not gaps["unadmitted_retired_video_gaps"][0]["published_as_current_video"]
    videos = list((layout.processed / "Clips").rglob("*.mp4"))
    assert len(videos) == 1 and videos[0].read_bytes() != before
    assert retired.read_bytes() == before and source.read_bytes() == original
