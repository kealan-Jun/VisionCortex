"""Same-source relative offsets remain usable without inventing a wall clock."""
import pytest

from visioncortex.lab.timeline_prior_context import build_prior_context
from visioncortex.lab.timeline_tasks import TimelineTasks
from .test_timeline import hj as base_scene, media, activity

POLICY = dict(enabled=True, max_items=16, max_utf8_bytes=8192, lookback_seconds=1800)


@pytest.fixture
def hj(core, settings):
    value = base_scene.__wrapped__(core, settings)
    value['camera'] = core.register('relative-camera', 'capture_device', 'relative-camera')['id']
    return value


def test_same_unknown_clock_video_can_supply_own_speech_ocr_not_other_media_or_vlm(hj):
    current = media(hj, 'current-unclocked', duration=10, captured=False, camera_id='relative-camera')
    speech = activity(hj, 'own-speech', current, start=1, end=2, source_type='speech')
    ocr = activity(hj, 'own-ocr', current, start=3, end=4, source_type='ocr')
    activity(hj, 'old-video-model-answer', current, start=1, end=2, source_type='video')
    other = media(hj, 'unclocked-other', duration=10, kind='audio', captured=False, camera_id='relative-camera')
    activity(hj, 'other-speech', other, start=1, end=2, source_type='speech')
    claimed = TimelineTasks(hj['timeline']).claim('relative-claim', 'video_understanding', 'fixture/1')
    request, window = claimed['request'], claimed['request']['windows'][0]
    before = build_prior_context(hj['core'], request, window, POLICY)
    assert before['status'] == 'source_relative_only' and before['physical_cutoff'] is None
    assert before['world_clock_status'] == 'uncalibrated' and not before['historical_prior']
    assert {r['activity_id'] for r in before['contemporaneous_evidence']} == {speech['id'], ocr['id']}
    assert all(p['time_semantics'] == 'source_relative_only_no_wall_clock_claim'
               for r in before['contemporaneous_evidence'] for p in r['evidence'])
    activity(hj, 'late-new-own-ocr', current, start=5, end=6, source_type='ocr')
    assert build_prior_context(hj['core'], request, window, POLICY) == before


def test_unknown_source_context_whole_items_obey_small_byte_limit(hj):
    current = media(hj, 'current-unclocked', duration=10, captured=False, camera_id='relative-camera')
    activity(hj, 'oversized-own-ocr', current, start=1, end=2, source_type='ocr',
             payload=dict(raw_text='source-data-' * 1000))
    claimed = TimelineTasks(hj['timeline']).claim('relative-small-claim', 'video_understanding', 'fixture/1')
    request = claimed['request']
    result = build_prior_context(hj['core'], request, request['windows'][0], dict(POLICY, max_utf8_bytes=1024))
    import json
    assert len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode()) <= 1024
    assert not result['contemporaneous_evidence']
