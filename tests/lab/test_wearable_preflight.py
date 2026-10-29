import pytest

from ._support import research_helpers
import sys

sys.path.insert(0, str(research_helpers()))
from preflight_two_wearables import receiver_readiness


def camera(key, **changes):
    return dict(camera_key=key, online=True, rgb_h264_main_available=True,
                rgb_h264_main_age_ms=5, rgb_h264_main_width=1280,
                rgb_h264_main_height=800, **changes)


def test_online_is_not_fresh_main_video_and_one_camera_does_not_cover_two():
    a = camera('wearable-a')
    b = camera('wearable-b')
    b.update(rgb_h264_main_available=False, rgb_h264_main_age_ms=307080)
    result = receiver_readiness(dict(cameras=[a, b]), ['wearable-a', 'wearable-b'])
    assert not result['both_advertised_ready']
    assert result['cameras'][0]['advertised_ready']
    assert 'main_video_not_advertised_available' in result['cameras'][1]['reasons']
    assert 'main_video_age_unknown_or_stale' in result['cameras'][1]['reasons']
    assert not result['frame_decode_performed'] and not result['qr_decode_performed']


@pytest.mark.parametrize('age', [None, True, -1, 1001, float('nan')])
def test_unknown_or_invalid_age_cannot_become_current_video(age):
    a = camera('wearable-a')
    a['rgb_h264_main_age_ms'] = age
    result = receiver_readiness(dict(cameras=[a, camera('wearable-b')]), ['wearable-a', 'wearable-b'])
    assert not result['both_advertised_ready']


def test_two_advertised_frames_are_not_recording_or_clock_acceptance():
    a = camera('wearable-a', clock_sync_valid=True, clock_report_stale=False)
    result = receiver_readiness(dict(cameras=[a, camera('wearable-b')], recording_state='idle'), ['wearable-a', 'wearable-b'])
    assert result['both_advertised_ready']
    assert result['recording_state'] == 'idle'
    assert not result['cameras'][0]['clock_advertisement']['establishes_frame_world_time']


def test_duplicate_selection_or_duplicate_advertisements_cannot_pass():
    with pytest.raises(ValueError):
        receiver_readiness(dict(cameras=[camera('wearable-a')]), ['wearable-a', 'wearable-a'])
    result = receiver_readiness(dict(cameras=[camera('wearable-a'), camera('wearable-a'), camera('wearable-b')]), ['wearable-a', 'wearable-b'])
    assert not result['both_advertised_ready']
    assert 'camera_missing_or_ambiguous' in result['cameras'][0]['reasons']
