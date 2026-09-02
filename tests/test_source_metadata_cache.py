from pathlib import Path

import pytest

from visioncortex import storage
from visioncortex.alignment import read_timestamp_csv_endpoints
from visioncortex.video_io import _clock_metadata_video_info


def test_source_stats_are_reused_across_startup_stages(tmp_path, monkeypatch):
    storage.clear_source_metadata_cache()
    source = tmp_path / "segment.mp4"
    source.write_bytes(b"video")
    calls = 0
    original = storage._source_stat

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(storage, "_source_stat", counted)
    first, first_report = storage.snapshot_source_paths([source], workers=8)
    second, second_report = storage.snapshot_source_paths([source], workers=8)

    assert first[source] == second[source]
    assert first_report["fresh_stat_count"] == 1
    assert second_report["cache_hit_count"] == 1
    assert second_report["fresh_stat_count"] == 0
    assert calls == 1


def test_clock_preflight_and_alignment_share_csv_edge_read(tmp_path):
    storage.clear_source_metadata_cache()
    video = tmp_path / "segment.mp4"
    video.write_bytes(b"video")
    clock = tmp_path / "segment.csv"
    clock.write_text(
        "rgb_video_frame_index,rgb_recorded,width,height,global_timestamp_us,rgb_actual_fps\n"
        "0,1,1920,1080,1000000,30\n"
        "299,1,1920,1080,10966667,30\n",
        encoding="utf-8",
    )

    info = _clock_metadata_video_info(video, clock)
    endpoints = read_timestamp_csv_endpoints(clock, 30.0)
    diagnostics = storage.source_cache_diagnostics()

    assert info is not None
    assert info.frame_count == 300
    assert info.fps == 30.0
    assert info.duration_ms == pytest.approx(10_000.0, abs=0.01)
    assert len(endpoints) == 2
    assert diagnostics["edge_cache_misses"] == 1
    assert diagnostics["edge_cache_hits"] == 1


def test_stale_z_index_path_remaps_without_network_probe(tmp_path, monkeypatch):
    def forbidden_exists(_path):
        raise AssertionError("path remapping must not perform an SMB existence check")

    monkeypatch.setattr(Path, "exists", forbidden_exists)
    remapped = storage._resolve_nas_path(
        r"Z:\experiment\camera\segment.mp4",
        Path(r"Y:\experiment_record_index.csv"),
    )

    assert str(remapped) == r"Y:\experiment\camera\segment.mp4"


def test_stale_z_index_path_remaps_to_linux_share_root(tmp_path):
    index = tmp_path / "nas" / "experiment_record_index.csv"
    remapped = storage._resolve_nas_path(
        r"Z:\experiment\camera\segment.mp4",
        index,
    )

    assert remapped == index.parent / "experiment" / "camera" / "segment.mp4"


def test_historical_realityloop_unc_maps_to_same_linux_share(tmp_path):
    index = tmp_path / "nas" / "experiment_record_index.csv"
    remapped = storage._resolve_nas_path(
        r"\\REALITYLOOP\video_database\camera\2026-06-01\segment.mp4",
        index,
    )

    assert remapped == index.parent / "camera" / "2026-06-01" / "segment.mp4"
