import csv
import json
from pathlib import Path

import pytest

from visioncortex.storage import (
    describe_index_experiment,
    prepare_from_nas_index,
    safe_archive_name,
)


def test_archive_components_are_ascii_only_and_deterministic():
    assert safe_archive_name("固体称量与移液连续实验") == safe_archive_name(
        "固体称量与移液连续实验"
    )
    assert safe_archive_name("固体称量与移液连续实验").isascii()
    assert safe_archive_name("Wet Lab 实验 01") == "Wet-Lab-01"


def test_index_description_accepts_integral_scientific_notation(tmp_path):
    rows = [
        {
            "experiment_id": "legacy",
            "experiment_prefix": "legacy_0001",
            "camera_key": "fp",
            "camera_view": "first",
            "recording_start_us": "1.78E+15",
            "recording_end_us": "1.7800036E+15",
            "segment_count": "1",
        }
    ]

    description = describe_index_experiment(
        tmp_path / "experiment_record_index.csv", "legacy", rows
    )

    assert description["recording_hours"] == 1.0


def test_nas_ingest_registers_segments_without_copying(default_config, tmp_path):
    nas = tmp_path / "nas"
    nas.mkdir()
    rows = []
    for camera, camera_view in (("fp", "first"), ("tp", "third")):
        videos = []
        clocks = []
        for index in range(2):
            video = nas / f"{camera}-{index}.mp4"
            clock = nas / f"{camera}-{index}.csv"
            video.write_bytes(b"segment")
            clock.write_text("frame_index,timestamp_ms\n0,0\n1,40\n", encoding="utf-8")
            videos.append(str(video))
            clocks.append(str(clock))
        rows.append(
            {
                "experiment_id": "segmented",
                "camera_key": camera,
                "camera_view": camera_view,
                "rgb_file": ";".join(videos),
                "frames_file": ";".join(clocks),
                "segment_count": "2",
            }
        )
    index_csv = nas / "experiment_record_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    default_config["storage"].update(
        {
            "index_csv": str(index_csv),
            "local_runtime_root": str(tmp_path / "runtime"),
            "active_archive_path": str(tmp_path / "archive"),
            "manifest_storage": "nas",
            "require_nas_source_paths": False,
        }
    )
    default_config["performance"]["synchronized_segment_waves"] = True

    manifest, manifest_path, ingest = prepare_from_nas_index(default_config, "segmented")

    assert len(manifest.views) == 2
    assert all(len(view.segments) == 2 for view in manifest.views)
    assert ingest["copied_source_bytes"] == 0
    assert ingest["continuous_source_copies_created"] == 0
    assert ingest["source_validation"]["path_count"] == 8
    assert ingest["source_validation"]["verified_file_count"] == 8
    assert ingest["source_validation"]["missing_count"] == 0
    assert manifest_path.is_file()
    role_receipt = json.loads(
        Path(ingest["view_role_resolution"]["receipt"]).read_text(encoding="utf-8")
    )
    assert role_receipt["status"] == "resolved"
    assert role_receipt["resolved_first_person_views"] == 1
    assert role_receipt["resolved_third_person_views"] == 1
    input_seal = json.loads(
        Path(ingest["original_retention"]["input_seal"]).read_text(encoding="utf-8")
    )
    assert input_seal["source_mode"] == "nas_segmented_virtual_timeline"
    assert input_seal["copied_source_bytes"] == 0
    assert input_seal["source_count"] == 8
    assert input_seal["identity_algorithms"] == {
        "sha256-size-plus-64k-head-tail-v1": 8
    }
    assert not (tmp_path / "runtime" / "Input" / "segmented" / "video.mp4").exists()
    original_root = tmp_path / "archive" / "Original-Experiment-Videos"
    original_index = json.loads(
        (original_root / "Original-Video-Index.json").read_text(encoding="utf-8")
    )
    assert original_index["retention_mode"] == "nas_zero_copy_segment_references"
    assert original_index["source_copy_bytes"] == 0
    assert len(original_index["views"]) == 2
    assert all(item["segment_count"] == 2 for item in original_index["views"])
    assert (original_root / "fp.ffconcat").read_text(encoding="utf-8").count("file '") == 2
    assert (original_root / "tp.ffconcat").read_text(encoding="utf-8").count("file '") == 2
    assert (original_root / "README.txt").is_file()
    assert (
        tmp_path
        / "archive"
        / "JSON-Config-Files"
        / "Stage-Receipts"
        / "original_ingest.json"
    ).is_file()


def test_nas_ingest_omits_same_name_segment_outside_declared_window(
    default_config, tmp_path
):
    nas = tmp_path / "nas"
    nas.mkdir()
    rows = []
    old_start = 1_781_751_313_000_000
    new_start = 1_781_751_413_300_000
    for camera, camera_view in (("fp", "first"), ("tp", "third")):
        videos = []
        clocks = []
        for label, start_us in (("old", old_start), ("new", new_start)):
            video = nas / f"{camera}-{label}.mp4"
            clock = nas / f"{camera}-{label}.csv"
            video.write_bytes(b"segment")
            clock.write_text(
                "local_time_us,rgb_recorded,rgb_video_frame_index\n"
                f"{start_us},1,0\n"
                f"{start_us + 5_000_000},1,1\n",
                encoding="utf-8",
            )
            videos.append(str(video))
            clocks.append(str(clock))
        rows.append(
            {
                "experiment_id": "reused-name",
                "experiment_prefix": "Pipetting_standard_correct_23_0005",
                "camera_key": camera,
                "camera_view": camera_view,
                "recording_start_time": "2026-06-18T02:56:53.191555+00:00",
                "recording_end_time": "2026-06-18T02:56:58.724480+00:00",
                "recording_start_us": "1781751413191555",
                "recording_end_us": "1781751418724480",
                "rgb_file": ";".join(videos),
                "frames_file": ";".join(clocks),
                "segment_count": "2",
            }
        )
    index_csv = nas / "experiment_record_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    default_config["storage"].update(
        {
            "index_csv": str(index_csv),
            "local_runtime_root": str(tmp_path / "runtime"),
            "active_archive_path": str(tmp_path / "archive"),
            "manifest_storage": "nas",
            "require_nas_source_paths": False,
        }
    )
    default_config["performance"]["synchronized_segment_waves"] = True

    manifest, _manifest_path, ingest = prepare_from_nas_index(
        default_config, "reused-name"
    )

    assert all(len(view.segments) == 1 for view in manifest.views)
    assert all(view.segments[0].video.name.endswith("-new.mp4") for view in manifest.views)
    assert ingest["segment_counts"] == {"fp": 1, "tp": 1}
    assert len(ingest["omitted_out_of_window_segments"]) == 2
    assert {
        item["reason"] for item in ingest["omitted_out_of_window_segments"]
    } == {"recorder_clock_does_not_overlap_declared_experiment_window"}
    assert ingest["source_validation"]["path_count"] == 4
    assert ingest["copied_source_bytes"] == 0


def test_segment_wave_count_mismatch_is_rejected(default_config, tmp_path):
    index_csv = tmp_path / "experiment_record_index.csv"
    videos = []
    for index in range(3):
        path = tmp_path / f"v-{index}.mp4"
        path.write_bytes(b"x")
        videos.append(str(path))
    with index_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment_id", "camera_key", "camera_view", "rgb_file"],
        )
        writer.writeheader()
        writer.writerow(
            {"experiment_id": "x", "camera_key": "fp", "camera_view": "first", "rgb_file": videos[0]}
        )
        writer.writerow(
            {"experiment_id": "x", "camera_key": "tp", "camera_view": "third", "rgb_file": ";".join(videos[1:])}
        )
    default_config["storage"].update(
        {"index_csv": str(index_csv), "local_runtime_root": str(tmp_path / "runtime")}
    )
    default_config["performance"]["synchronized_segment_waves"] = True

    with pytest.raises(ValueError, match="equal segment counts"):
        prepare_from_nas_index(default_config, "x")


def test_explicitly_failed_empty_camera_is_omitted_with_receipt(
    default_config, tmp_path
):
    index_csv = tmp_path / "experiment_record_index.csv"
    rows = []
    for camera, camera_view in (("fp", "first"), ("tp", "third")):
        video = tmp_path / f"{camera}.mp4"
        video.write_bytes(b"video")
        rows.append(
            {
                "experiment_id": "degraded",
                "camera_key": camera,
                "camera_view": camera_view,
                "rgb_file": str(video),
                "frames_file": "",
                "segment_count": "1",
                "sync_error": "",
            }
        )
    rows.append(
        {
            "experiment_id": "degraded",
            "camera_key": "tp-failed",
            "camera_view": "third",
            "rgb_file": "",
            "frames_file": "",
            "segment_count": "",
            "sync_error": "matching meta.json was not found on Z: for this experiment prefix",
        }
    )
    with index_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    default_config["storage"].update(
        {
            "index_csv": str(index_csv),
            "local_runtime_root": str(tmp_path / "runtime"),
            "require_nas_source_paths": False,
        }
    )
    default_config["performance"]["synchronized_segment_waves"] = False

    manifest, _manifest_path, ingest = prepare_from_nas_index(
        default_config, "degraded"
    )

    assert [view.view_id for view in manifest.views] == ["fp", "tp"]
    assert ingest["indexed_camera_count"] == 3
    assert ingest["camera_count"] == 2
    assert ingest["omitted_unavailable_cameras"][0]["camera_key"] == "tp-failed"
    role_receipt = json.loads(
        Path(ingest["view_role_resolution"]["receipt"]).read_text(encoding="utf-8")
    )
    assert role_receipt["status"] == "degraded_resolved"
    omitted = next(item for item in role_receipt["views"] if item["camera_key"] == "tp-failed")
    assert omitted["source_included"] is False


def test_indexed_missing_mp4_is_omitted_only_with_zero_frame_meta_proof(
    default_config, tmp_path
):
    fp = tmp_path / "fp_rgb.mp4"
    tp = tmp_path / "tp_rgb.mp4"
    missing = tmp_path / "tp-failed_rgb.mp4"
    fp.write_bytes(b"video")
    tp.write_bytes(b"video")
    tp_clock = tmp_path / "tp_frames.csv"
    tp_clock.write_text("rgb_recorded,rgb_video_frame_index\n1,0\n", encoding="utf-8")
    missing_clock = tmp_path / "tp-failed_frames.csv"
    missing_clock.write_text("rgb_recorded,rgb_video_frame_index\n", encoding="utf-8")
    (tmp_path / "tp-failed_meta.json").write_text(
        json.dumps(
            {
                "closed": True,
                "rgb_file": missing.name,
                "rgb_frames": 0,
                "rgb_actual_fps": 0,
                "rgb_record_fps": 0,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "experiment_id": "zero-frame",
            "camera_key": "fp",
            "camera_view": "first",
            "rgb_file": str(fp),
            "frames_file": "",
            "segment_count": "1",
            "sync_error": "",
        },
        {
            "experiment_id": "zero-frame",
            "camera_key": "tp",
            "camera_view": "third",
            "rgb_file": f"{tp};{missing}",
            "frames_file": f"{tp_clock};{missing_clock}",
            "segment_count": "2",
            "sync_error": "",
        },
    ]
    index_csv = tmp_path / "experiment_record_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    default_config["storage"].update(
        {
            "index_csv": str(index_csv),
            "local_runtime_root": str(tmp_path / "runtime"),
            "require_nas_source_paths": False,
        }
    )
    default_config["performance"]["synchronized_segment_waves"] = False

    manifest, _manifest_path, ingest = prepare_from_nas_index(
        default_config, "zero-frame"
    )

    assert len(manifest.views) == 2
    assert ingest["segment_counts"] == {"fp": 1, "tp": 1}
    assert len(ingest["omitted_zero_frame_segments"]) == 1
    evidence = ingest["omitted_zero_frame_segments"][0]
    assert evidence["video_path"] == str(missing)
    assert evidence["reason"] == "closed_recorder_segment_contains_zero_rgb_frames"
    assert evidence["source_copy_bytes"] == 0
