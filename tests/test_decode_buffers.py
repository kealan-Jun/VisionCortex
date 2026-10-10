import pytest

from visioncortex.decode_buffers import frame_queue_limit


def test_native_1080p_packet_queue_uses_byte_limit():
    assert frame_queue_limit(64, 1920, 1080) == 6


def test_six_prefetch_queues_share_a_single_budget():
    depth = frame_queue_limit(48, 1920, 1080, queues=6, bytes_per_pixel=3)
    assert depth == 1
    assert depth * 6 * 1920 * 1080 * 3 <= 64 * 1024**2


def test_one_frame_can_progress_when_larger_than_budget():
    assert frame_queue_limit(64, 8192, 8192, max_bytes=1024) == 1


@pytest.mark.parametrize("budget", [0, -1, True, None, 1.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        frame_queue_limit(64, 1920, 1080, max_bytes=budget)
