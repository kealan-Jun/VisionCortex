import cv2
import numpy as np
import pytest

from visioncortex.device_day_understanding import frame_windows


def source(tmp_path, seconds=5):
    path = tmp_path/'source.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip('MJPG test encoder unavailable')
    for i in range(seconds*10):
        writer.write(np.full((48,64,3), i % 255, np.uint8))
    writer.release()
    return path


def test_active_submits_every_frame_in_order_across_bounded_windows(tmp_path):
    groups = list(frame_windows(source(tmp_path), 1250, 3150, True, {'understanding_frames_per_request': 4}))
    frames = [f for _, _, rows in groups for f in rows]
    assert [f[0] for f in frames] == list(range(13,32))
    assert all(len(rows) <= 4 for _, _, rows in groups)
    assert groups[0][0] == 1250 and groups[-1][1] == 3150
    assert all(a[1] == b[0] for a,b in zip(groups,groups[1:], strict=False))


def test_inactive_sampling_covers_start_to_end_without_total_frame_cap(tmp_path):
    groups = list(frame_windows(source(tmp_path, 15), 0, 15000, False,
        {'understanding_frames_per_request': 4, 'inactive_sample_seconds': 1}))
    frames = [f for _, _, rows in groups for f in rows]
    assert len(frames) > 8
    assert frames[0][0] == 0 and frames[-1][0] == 149
    assert max(b[1]-a[1] for a,b in zip(frames,frames[1:], strict=False)) <= 1000.001
    assert len({f[0] for f in frames}) == len(frames)


def test_absent_media_is_not_reported_as_full_coverage(tmp_path):
    with pytest.raises(ValueError, match='opened'):
        list(frame_windows(tmp_path/'missing.mp4', 0, 1000, True, {}))
