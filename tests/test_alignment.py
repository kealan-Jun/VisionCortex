from labvision_evidence.alignment import _nearest_pairs, _robust_affine, _timestamp_point
from labvision_evidence.schemas import TimestampPoint


def test_capture_epoch_local_time_is_not_a_playback_coordinate():
    point = _timestamp_point({
        "rgb_video_frame_index": "30",
        "local_time_us": "1781748777314488",
        "frame_system_timestamp_us": "1781748776039118",
    }, 100, 30.0)
    assert point.frame_index == 30
    assert point.local_ms == 1000.0
    assert point.source_ms == 1781748776039.118


def test_relative_video_pts_remains_unchanged():
    point = _timestamp_point({"frame_index": "30", "pts_us": "1001000"}, 0, 30.0)
    assert point.local_ms == 1001.0
    assert point.source_ms is None


def test_nearest_neighbor_and_robust_affine_recover_offset_and_drift():
    reference = [TimestampPoint(frame_index=i, local_ms=i * 100.0, source_ms=1_700_000_000_000 + i * 100.0) for i in range(100)]
    target = [
        TimestampPoint(
            frame_index=i,
            local_ms=i * 100.0,
            source_ms=1_700_000_000_000 + 240.0 + i * 100.1,
        )
        for i in range(95)
    ]
    pairs = _nearest_pairs(reference, target, tolerance_ms=60.0)
    scale, offset, rmse = _robust_affine(pairs, max_drift_ppm=2500.0)
    assert len(pairs) > 80
    assert 0.997 < scale < 1.003
    assert abs(offset) < 400
    assert rmse < 70
