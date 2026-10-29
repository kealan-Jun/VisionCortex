"""Existing-channel dependencies; isolated PG, synthetic receipts, no models."""
import hashlib
import pytest

from visioncortex.lab.timeline_policy import TimelinePolicies
from visioncortex.lab.timeline_tasks import TimelineTasks
from visioncortex.lab.ingest import Ingest
from .test_timeline import hj as base_scene, media, result_for


@pytest.fixture
def hj(core, settings):
    value = base_scene.__wrapped__(core, settings)
    value['timeline'].policies = TimelinePolicies(dict(schema_version='vc-lab-timeline-policy/1',
        multimodal_prior_context=dict(enabled=True, max_items=16, max_utf8_bytes=8192,
                                      lookback_seconds=1800)))
    return value


def test_video_waits_for_existing_overlapping_audio_across_channel_session_then_claims(hj):
    video = media(hj, 'video-packet', duration=10, camera_id='shared-camera', session='video-session')
    audio = media(hj, 'audio-packet', duration=10, kind='audio', camera_id='shared-camera', session='audio-session')
    tasks = TimelineTasks(hj['timeline'])
    waiting = tasks.claim('video-waits', 'video_understanding', 'fixture-model/1')
    assert waiting['task'] is None and waiting['pending_prior_channels']
    with hj['core'].db.transaction(readonly=True) as conn:
        row = conn.execute('SELECT attempts FROM hj_timeline_task WHERE customer_id=%s AND media_id=%s',
                            (hj['core'].customer, video['media_id'])).fetchone()
        assert row['attempts'] == 0
    claimed = tasks.claim('audio-first', 'transcription', 'fixture-asr/1')
    assert claimed['request']['primary_media_id'] == audio['media_id']
    tasks.submit('audio-complete', claimed['task_id'], claimed['lease_token'], 'fixture-asr/1', result_for(claimed))
    assert tasks.claim('video-after', 'video_understanding', 'fixture-model/1')['request']['primary_media_id'] == video['media_id']


def test_other_camera_pending_audio_does_not_block_video(hj):
    video = media(hj, 'video-packet', duration=10, camera_id='camera-A')
    media(hj, 'unrelated-audio', duration=10, kind='audio', camera_id='camera-B')
    assert TimelineTasks(hj['timeline']).claim('no-cross-camera', 'video_understanding', 'fixture-model/1')['request']['primary_media_id'] == video['media_id']


def test_uncalibrated_channel_that_cannot_be_prior_does_not_block_video(hj):
    video = media(hj, 'video-packet', duration=10, camera_id='camera-A')
    path = hj['settings'].incoming / 'unknown-clock.wav'
    path.write_bytes(b'explicit synthetic receipt only, not real audio')
    Ingest(hj['core'], hj['settings']).accept(dict(capture_id='unknown-clock',
        source_instance='other-channel', source_revision='fixture/1', kind='audio',
        session_id='different-channel-session', camera_id='camera-A', relative_path=path.name,
        captured_at='2026-10-27T09:00:00+08:00', source_kind='fixture',
        time_basis=dict(basis='synthetic_fixture_clock', uncertainty_us=0, clock_sync_valid=False),
        start_seconds='0', end_seconds='10', sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert TimelineTasks(hj['timeline']).claim('no-unknown-clock-dependency', 'video_understanding', 'fixture-model/1')['request']['primary_media_id'] == video['media_id']
