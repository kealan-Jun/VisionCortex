from copy import deepcopy
from pathlib import Path

import pytest

from test_device_day_time_lookup import ARCHIVE, BASE, index
from visioncortex.device_day_contract import DIRECTORIES, atomic_json, read_json
from visioncortex.device_day_file_index import build_file_index, publish_file_index, FileIndexPublisher


def all_entries(data):
    streams = data['streams']
    return [*data['recordings'], *(e for values in streams.values() for e in values),
            *(s for audio in streams['audio'] for s in audio['transcription']['sentences'])]


def config(tmp_path):
    return {'storage': {'archive_root': str(tmp_path/'archive'), 'local_runtime_root': str(tmp_path/'runtime')},
            'collection_ingest': {'source_root': str(tmp_path/'capture')},
            'device_day': {'process_since_us': BASE}}


def test_readable_text_and_own_audio_clock_do_not_require_video_processing(tmp_path):
    source = index()
    before = deepcopy(source)
    data = build_file_index(config(tmp_path), source, {})
    assert source == before
    speech = next(e for e in all_entries(data) if e['kind'] == 'speech')
    assert speech['text'] == '样本转写' and speech['start_time'] == '2026-09-17T12:00:03.000000+08:00'
    assert speech['audio']['path'] == 'MetaVideo/Audio/Original.opus'
    assert speech['audio']['path_base'] == 'device_day'
    assert speech['accuracy'] == 'NOT_PROVEN' and not speech['physical_action_confirmed']
    audio = next(e for e in all_entries(data) if e['kind'] == 'audio')
    assert audio['start_us'] == BASE+500000 and audio['audio_start_offset_us'] == 500000


def test_empty_transcript_and_missing_audio_clock_are_explicit(tmp_path):
    source = index()
    row = source['recordings'][0]
    row['transcription'].update(outcome='no_transcript', comments=[], model_invocation='PROVEN')
    row['audio'].update(start_us=None, end_us=None, quality_status='partial', capture_complete=False)
    entries = all_entries(build_file_index(config(tmp_path), source, {}))
    assert not any(e['kind'] == 'speech' for e in entries)
    recording = next(e for e in entries if e['kind'] == 'recording')
    assert recording['transcription_outcome'] == 'no_transcript' and recording['model_invocation'] == 'PROVEN'
    audio = next(e for e in entries if e['kind'] == 'audio')
    assert audio['start_us'] is None and audio['audio_start_offset_us'] is None
    assert audio['capture_complete'] is False


def test_old_records_and_their_frames_are_not_backfilled(tmp_path):
    source = index()
    source['segments'] = [{'recording_id': 'slice', 'segment_id': 's', 'start_us': BASE,
        'end_us': BASE+10000000, 'activity': 'inactive', 'key_frames': [],
        'scene_frames': [{'path': 'ProcessedClips/Clips/S/SceneFrames/1.jpg', 'capture_us': BASE+1,
                          'frame_kind': 'scene_sample'}]}]
    entries = all_entries(build_file_index(config(tmp_path), source, {}))
    assert any(e['kind'] == 'scene_frame' and e['frame_kind'] == 'scene_sample' for e in entries)
    assert not any(e['kind'] == 'key_frame' for e in entries)
    cfg = config(tmp_path)
    cfg['device_day']['process_since_us'] = BASE+20000000
    assert all_entries(build_file_index(cfg, source, {})) == []


def test_photos_remain_source_references_with_explicit_clock_uncertainty(tmp_path):
    photos = {'observed_at': 42, 'devices': [{'camera': ARCHIVE[11:], 'photos': [
        {'capture_us': BASE+2, 'time_basis': 'capture_filename_seconds',
         'source_ref': {'path': 'voice_photos/camera_cam01/2026-09-17/12-00-00/20260917_120000.jpg'}}]}]}
    data = build_file_index(config(tmp_path), index(), photos)
    photo = next(e for e in all_entries(data) if e['kind'] == 'voice_photo')
    assert photo['file']['path_base'] == 'capture_root'
    assert photo['timestamp_resolution_us'] == 1000000 and not photo['capture_clock_verified']
    assert not (tmp_path/'capture').exists()  # Projection never reads/copies source media.


def test_atomic_file_publication_keeps_five_directories_and_original_index(tmp_path):
    cfg, source = config(tmp_path), index()
    root = Path(cfg['storage']['archive_root'])/ARCHIVE
    for name in DIRECTORIES:
        (root/name).mkdir(parents=True)
    value = publish_file_index(cfg, source)
    assert read_json(root/'Comment/TimeIndex.json') == value
    assert set(p.name for p in root.iterdir()) == set(DIRECTORIES)
    assert source['time_index']['path'] == 'Comment/TimeIndex.json'
    assert source['recordings'] == index()['recordings']
    assert not list(root.rglob('*.partial'))


def test_publisher_observes_late_transcript_but_ignores_photo_heartbeat(tmp_path):
    from visioncortex.observed_inventory import observe
    cfg, source = config(tmp_path), index()
    root = Path(cfg['storage']['archive_root'])/ARCHIVE
    runtime = Path(cfg['storage']['local_runtime_root'])/'device-day'
    observe(runtime, {'recordings': [{'recording_id': 'slice', 'camera_key': ARCHIVE[11:],
        'recording_start_us': BASE, 'recording_end_us': BASE+10000000}]})
    atomic_json(root/'ProcessedClips/Index.json', source)
    photos = {'devices': [], 'observed_at': 1}
    photo_path = runtime/'CapturePhotos'/f'{ARCHIVE}.json'
    atomic_json(photo_path, photos)
    publisher = FileIndexPublisher(cfg)
    publisher.tick()
    target = root/'Comment/TimeIndex.json'
    first = target.stat().st_mtime_ns
    atomic_json(photo_path, photos | {'observed_at': 2})
    publisher.tick()
    assert target.stat().st_mtime_ns == first
    source['recordings'][0]['transcription']['comments'][0]['text'] = '新识别结果'
    atomic_json(root/'ProcessedClips/Index.json', source)
    publisher.tick()
    assert any(e.get('text') == '新识别结果' for e in all_entries(read_json(target)))


def test_photo_only_device_publishes_without_waiting_for_video(tmp_path):
    cfg = config(tmp_path)
    runtime = Path(cfg['storage']['local_runtime_root'])/'device-day'
    atomic_json(runtime/'CapturePhotos'/f'{ARCHIVE}.json', {'observed_at': 1, 'devices': [
        {'camera': ARCHIVE[11:], 'photos': [{'capture_us': BASE, 'time_basis': 'capture_filename_seconds',
                                           'source_ref': {'path': 'voice_photos/camera_cam01/Photo.jpg'}}]}]})
    FileIndexPublisher(cfg).tick()
    root = Path(cfg['storage']['archive_root'])/ARCHIVE
    assert {p.name for p in root.iterdir()} == set(DIRECTORIES)
    assert read_json(root/'ProcessedClips/Index.json')['recordings'] == []
    assert [e['kind'] for e in all_entries(read_json(root/'Comment/TimeIndex.json'))] == ['voice_photo']


def test_readable_cache_drops_dense_cv_but_pointer_update_preserves_canonical(tmp_path, monkeypatch):
    from visioncortex import device_day_content
    from visioncortex.observed_inventory import observe
    cfg, source = config(tmp_path), index()
    cfg['device_day'].update(readable_content_enabled=True, paused_stages=['report'])
    source['recordings'][0]['processing'] = {'clock_mapping': {'origin_us': BASE},
        'batches': [{'dense': 'x' * 10000}], 'scan_reports': [{'dense': True}], 'audit_artifacts': [{'dense': True}]}
    source['segments'] = [{'recording_id': 'slice', 'segment_id': 's', 'start_us': BASE,
        'end_us': BASE+10000000, 'start_ms': 0, 'end_ms': 10000, 'activity': 'inactive',
        'source_ref': {'path': 'MetaVideo/Original.mp4'}, 'activity_audit': {'dense': 'x' * 10000}}]
    path = Path(cfg['storage']['archive_root'])/ARCHIVE/'ProcessedClips/Index.json'
    atomic_json(path, source)
    observe(Path(cfg['storage']['local_runtime_root'])/'device-day', {'recordings': [{
        'recording_id': 'slice', 'camera_key': ARCHIVE[11:], 'recording_start_us': BASE}]})
    # No understanding exists; the test only exercises projection and pointer writes.
    monkeypatch.setattr(device_day_content, 'content_revision', lambda *args: ())
    publisher = FileIndexPublisher(cfg)
    assert publisher.tick()['published'] == 1
    cached = publisher.source_indexes[ARCHIVE][1]
    assert cached['recordings'][0]['processing'] == {'clock_mapping': {'origin_us': BASE}}
    assert 'activity_audit' not in cached['segments'][0]
    published = read_json(path)
    pointer = published.pop('time_index')
    assert published == source  # Only the pointer may alter canonical evidence.
    assert pointer['path'] == 'Comment/TimeIndex.json'
    before = path.stat().st_mtime_ns
    assert publisher.tick()['unchanged'] == 1
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize('failure', ['content_revision', 'partial_understandings', 'readable_content', 'time_index'])
def test_bad_day_does_not_block_other_days_or_advance_failed_version(tmp_path, monkeypatch, failure):
    from visioncortex import device_day_content, device_day_file_index
    from visioncortex.observed_inventory import observe
    cfg = config(tmp_path)
    cfg['device_day'].update(readable_content_enabled=True, paused_stages=['report'])
    names = [ARCHIVE, ARCHIVE.replace('camera_cam01', 'camera_cam02')]
    runtime = Path(cfg['storage']['local_runtime_root'])/'device-day'
    observe(runtime, {'recordings': [{'recording_id': str(i), 'camera_key': name[11:],
        'recording_start_us': BASE} for i, name in enumerate(names)]})
    for name in names:
        atomic_json(Path(cfg['storage']['archive_root'])/name/'ProcessedClips/Index.json', index() | {'archive': name})
    module, function = (device_day_file_index, 'publish_file_index') if failure == 'time_index' else (
        device_day_content, {'content_revision': 'content_revision',
                            'partial_understandings': 'with_partial_understandings',
                            'readable_content': 'publish_content'}[failure])
    original = getattr(module, function)
    failing = [True]

    def call(config, value, *args):
        if failing[0] and value['archive'] == names[0]:
            raise TypeError('invalid receipt field')
        return original(config, value, *args)

    monkeypatch.setattr(module, function, call)
    publisher = FileIndexPublisher(cfg)
    result = publisher.tick()
    assert result['status'] == 'partial' and result['published'] == 1
    assert result['errors'] == [{'archive': names[0], 'operation': failure, 'error_type': 'TypeError'}]
    assert names[0] not in publisher.versions and names[1] in publisher.versions
    assert (Path(cfg['storage']['archive_root'])/names[1]/'Comment/TimeIndex.json').is_file()
    failing[0] = False
    result = publisher.tick()
    assert result['status'] == 'completed' and result['published'] == 1 and result['unchanged'] == 1


def test_bad_photo_json_is_visible_failure_without_blocking_other_days(tmp_path):
    from visioncortex.observed_inventory import observe
    cfg = config(tmp_path)
    runtime = Path(cfg['storage']['local_runtime_root'])/'device-day'
    observe(runtime, {'recordings': [{'recording_id': 'slice', 'camera_key': ARCHIVE[11:],
                                    'recording_start_us': BASE}]})
    atomic_json(Path(cfg['storage']['archive_root'])/ARCHIVE/'ProcessedClips/Index.json', index())
    bad = runtime/'CapturePhotos'/'2026-09-18_other_cam01.json'
    bad.parent.mkdir(parents=True)
    bad.write_text('{broken')
    result = FileIndexPublisher(cfg).tick()
    assert result['status'] == 'partial' and result['published'] == 1
    assert result['errors'][0]['operation'] == 'photo_index'


@pytest.mark.parametrize('paused', [[], ['report']])
def test_failed_first_retention_without_clock_does_not_block_time_index(tmp_path, monkeypatch, paused):
    from visioncortex import device_day_reports
    from visioncortex.observed_inventory import observe
    cfg, source = config(tmp_path), index()
    cfg['device_day'].update(readable_content_enabled=True, paused_stages=paused)
    failed = {'recording_id': 'failed', 'start_us': None, 'end_us': None,
              'sources': [], 'stages': {'retention': {'status': 'failed'}}}
    source['recordings'].insert(0, failed)
    path = Path(cfg['storage']['archive_root'])/ARCHIVE/'ProcessedClips/Index.json'
    atomic_json(path, source)
    observe(Path(cfg['storage']['local_runtime_root'])/'device-day', {'recordings': [{
        'recording_id': 'slice', 'camera_key': ARCHIVE[11:], 'recording_start_us': BASE}]})
    reports = []

    def render(layout, value):
        assert value['recordings'][0] == failed | {'processing': {}, 'transcription': None}
        reports.append(layout.name)

    monkeypatch.setattr(device_day_reports, 'render_day', render)
    result = FileIndexPublisher(cfg).tick()
    assert result == {'status': 'completed', 'published': 1, 'unchanged': 0, 'busy': 0, 'errors': []}
    assert reports == ([] if paused else [ARCHIVE])
    assert read_json(path)['recordings'][0] == failed  # Never invent the missing capture clock.
    assert read_json(path.parent.parent/'Comment/TimeIndex.json')['archive'] == ARCHIVE
