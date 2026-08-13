from labvision_evidence.alignment import _nearest_pairs, _robust_affine
from labvision_evidence.schemas import TimestampPoint


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

