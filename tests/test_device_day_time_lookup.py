from copy import deepcopy
import os
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from visioncortex.device_day_contract import atomic_json
from visioncortex.device_day_timeline import build_timeline, day_bounds, install_routes, refresh_timeline
from visioncortex.device_day_time_lookup import attach_photos, photo_path, query_materials

DAY = '2026-09-17'
BASE = day_bounds(DAY)[0] + 12 * 3600 * 1000000
ARCHIVE = DAY + '_camera_cam01'


def index():
    return {'archive': ARCHIVE, 'recordings': [{'recording_id': 'slice', 'start_us': BASE, 'end_us': BASE+10000000,
        'sources': [{'kind': 'video', 'retained': {'path': 'MetaVideo/Original.mp4', 'sha256': 'video'}},
                    {'kind': 'audio_audio', 'retained': {'path': 'MetaVideo/Audio/Original.opus', 'sha256': 'audio'}}],
        'audio': {'start_us': BASE+500000, 'end_us': BASE+11000000, 'time_basis': 'recorder_audio_metadata'},
        'transcription': {'status': 'completed', 'outcome': 'transcribed',
            'transcript_file': {'path': 'Comment/Stt/slice/current/Transcript.txt'},
            'comments': [{'start_us': BASE+3000000, 'end_us': BASE+5000000, 'text': '样本转写',
                          'transcript_path': 'Comment/Stt/slice/current/chunk/Transcript.json'}]}}], 'segments': []}


def test_same_timestamp_finds_audio_and_text_before_video_preprocessing():
    timeline = build_timeline(DAY, [(ARCHIVE, index())])
    assert timeline['entries'] == []
    result = query_materials(timeline, BASE+3500000, 1)
    row = result['devices'][0]['recordings'][0]
    assert row['video_offset_seconds'] == 3.5
    assert row['audio_offset_seconds'] == 3  # Its own clock, not the video origin.
    assert row['comments'][0]['text'] == '样本转写'
    assert row['transcript_url'].endswith('/Comment/Stt/slice/current/Transcript.txt')
    assert row['audio_ref']['sha256'] == 'audio'
    # Audio can extend beyond the recording's video boundary.
    last = query_materials(timeline, BASE+10500000, .1)['devices'][0]['recordings'][0]
    assert last['video_url'] is None and last['audio_offset_seconds'] == 10


def test_audio_with_unknown_clock_is_not_falsely_aligned_to_video():
    value = index()
    value['recordings'][0]['audio']['start_us'] = None
    row = query_materials(build_timeline(DAY, [(ARCHIVE, value)]), BASE, 1)['devices'][0]['recordings'][0]
    assert row['audio_offset_seconds'] is None and not row['audio_overlaps_query']
    assert row['comments'] == []


@pytest.mark.parametrize('at,span', [(BASE-86400000000, 60), (BASE, 0), (BASE, float('nan')), (BASE, 3601)])
def test_query_rejects_wrong_day_and_unbounded_windows(at, span):
    with pytest.raises(ValueError):
        query_materials({'date': DAY}, at, span)


def photo_config(tmp_path):
    return {'collection_ingest': {'source_root': str(tmp_path/'capture'), 'camera_role_map': {'camera_cam01': 'first_person'},
                                  'settle_seconds': 5}, 'device_day': {}}


def create_photo(config):
    path, _ = photo_path(config, DAY, 'camera_cam01', '12-00-04', '20260917_120004_001.jpg')
    path.parent.mkdir(parents=True)
    path.write_bytes(b'owned-image-reference-test-not-real-evidence')
    os.utime(path, (time.time()-60, time.time()-60))
    return path


def test_photo_uses_capture_name_not_upload_time_and_does_not_move_files(tmp_path):
    config = photo_config(tmp_path)
    path = create_photo(config)
    original = path.read_bytes(), path.stat().st_mtime_ns
    result = query_materials(build_timeline(DAY, [(ARCHIVE, index())]), BASE+3000000, 2)
    photo = attach_photos(config, result)['devices'][0]['photos'][0]
    assert photo['capture_us'] == BASE+4000000
    assert photo['recording_ids'] == ['slice']
    assert photo['time_basis'] == 'capture_filename_seconds' and not photo['capture_clock_verified']
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    config['device_day']['process_since_us'] = BASE+5000000
    result = query_materials(build_timeline(DAY, [(ARCHIVE, index())]), BASE, 60)
    assert attach_photos(config, result)['devices'][0]['photos'] == []


def test_photo_paths_reject_device_date_mismatch_and_symlinks(tmp_path):
    config = photo_config(tmp_path)
    path = create_photo(config)
    for camera, name in [('unknown', path.name), ('camera_cam01', '20260916_120004.jpg'), ('camera_cam01', '../secret.jpg')]:
        with pytest.raises(ValueError):
            photo_path(config, DAY, camera, '12-00-04', name)
    path.unlink()
    path.symlink_to(tmp_path/'outside.jpg')
    with pytest.raises(ValueError):
        photo_path(config, DAY, 'camera_cam01', '12-00-04', path.name)


def test_lookup_api_returns_existing_paths_without_mutating_archive(default_config, tmp_path):
    config = deepcopy(default_config)
    config.update(photo_config(tmp_path))
    config['storage'].update(archive_root=str(tmp_path/'archive'), local_runtime_root=str(tmp_path/'runtime'))
    path = Path(config['storage']['archive_root'])/ARCHIVE/'ProcessedClips/Index.json'
    atomic_json(path, index())
    before = path.read_bytes()
    create_photo(config)
    app = FastAPI()
    install_routes(app, lambda: config)
    client = TestClient(app)
    response = client.get(f'/api/day-timeline/{DAY}/at', params={'at_us': BASE+3000000, 'duration_seconds': 2})
    assert response.status_code == 200
    device = response.json()['devices'][0]
    assert device['recordings'][0]['comments'][0]['text'] == '样本转写'
    assert client.get(device['photos'][0]['url']).status_code == 200
    assert path.read_bytes() == before
    assert client.get(f'/api/day-timeline/{DAY}/at', params={'at_us': 1}).status_code == 400


def test_live_timeline_refresh_writes_only_runtime_index(tmp_path, monkeypatch):
    def load(config, day, *, audit):
        assert not audit, 'Live lookup must not backfill old multiview jobs'
        return {'date': day, 'cross_view_links': [], 'media_recordings': []}
    monkeypatch.setattr('visioncortex.device_day_timeline.load_timeline', load)
    config = {'storage': {'local_runtime_root': str(tmp_path)}, 'device_day': {'process_since_us': BASE}}
    assert refresh_timeline(config, DAY)['publication_pending'] == []
    assert (tmp_path/'device-day/DayTimeline'/f'{DAY}.json').is_file()
