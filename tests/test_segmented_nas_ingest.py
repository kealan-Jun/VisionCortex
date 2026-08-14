import csv
from pathlib import Path

import pytest

from labvision_evidence.storage import prepare_from_nas_index


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
            "manifest_storage": "local",
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
    assert not (tmp_path / "runtime" / "Input" / "segmented" / "video.mp4").exists()


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
