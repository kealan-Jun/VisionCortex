import pytest

from visioncortex.alignment import read_timestamp_csv, read_timestamp_csv_bounded


def write_clock(path, valid):
    rows = ["frame_index,local_timestamp_ms,global_timestamp_ms,rgb_recorded"]
    rows.extend(f"{i},{i * 100},{1000000 + i * 100},{int(i in valid)}" for i in range(100))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_bounded_reader_recovers_rgb_rows_between_sampling_strides(tmp_path):
    path = tmp_path / "Frames.csv"
    write_clock(path, set(range(1, 20)))
    with pytest.raises(ValueError, match="至少需要两行"):
        read_timestamp_csv(path, 10, max_points=5)
    points = read_timestamp_csv_bounded(path, 10, sample_count=5)
    assert [point.frame_index for point in points] == [1, 5, 10, 14, 19]
    assert all(point.source_ms == 1000000 + point.frame_index * 100 for point in points)


def test_successful_bounded_sample_remains_exact(tmp_path):
    path = tmp_path / "Frames.csv"
    write_clock(path, set(range(100)))
    assert read_timestamp_csv_bounded(path, 10, sample_count=5) == read_timestamp_csv(
        path, 10, max_points=5
    )


@pytest.mark.parametrize("valid", [set(), {3}])
def test_insufficient_rgb_evidence_still_fails(tmp_path, valid):
    path = tmp_path / "Frames.csv"
    write_clock(path, valid)
    with pytest.raises(ValueError, match="至少需要两行"):
        read_timestamp_csv_bounded(path, 10, sample_count=5)
