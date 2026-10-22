"""Published HTTP views stay bounded without rewriting canonical evidence."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from visioncortex.application.device_day_readmodel import DeviceDayReadModels, ReadModelDeferred
from visioncortex.device_day_service import DeviceDayService, install_routes, published_recording_status


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))
    return path


def canonical(root):
    return {'archive': root.name, 'updated_at': 'actual',
        'recordings': [{'recording_id': 'r', 'start_us': 10, 'end_us': 20,
            'processing': {'batches': [{'dense': list(range(1000))}], 'clock_mapping': {'unit': 'us'}},
            'sources': [{'kind': 'video', 'retained': {'path': 'MetaData/Capture/rgb.mp4'}}]}],
        'segments': [{'recording_id': 'r', 'segment_id': 'candidate', 'activity': 'inactive',
            'key_frames': [], 'scene_frames': [{'frame_kind': 'scene_sample', 'path': 'ProcessedClips/f.jpg'}],
            'activity_audit': {'large': list(range(1000))}}],
        'understandings': [{'segment_id': 'candidate', 'windows': [{'steps': [{'description': '实际观察'}]}]}],
        'literal.item': {'item': 'preserved'}}


def timeline(root):
    return {'schema_version': 'visioncortex-file-time-index/2', 'archive': root.name,
        'source_index_updated_at': 'actual', 'recordings': [{'recording_id': 'r', 'start_us': 10,
            'end_us': 20, 'stages': {'stt': {'status': 'completed'}, 'understanding': {'status': 'partial'}},
            'sources': [{'kind': 'video', 'retained': {'path': 'MetaData/Capture/rgb.mp4', 'path_base': 'device_day'}}],
            'capture_manifest': {'path': 'Comment/.Evidence/CaptureFiles/abc.json', 'path_base': 'device_day'}}],
        'field_events_file': {'path': 'Comment/FieldEvents.json', 'path_base': 'device_day'},
        'readable_content': {'redundant': list(range(1000))},
        'streams': {'images': [{'dense': list(range(1000))}], 'audio': [{'transcription': {'sentences': [{'text': '原句'}]}}],
            'multimodal': [{'segment_id': 'candidate'}],
            'video': [{'kind': 'original_video', 'recording_id': 'r'},
                {'kind': 'video_segment', 'record_scope': 'preprocessing_candidate', 'recording_id': 'r', 'segment_id': 'candidate'},
                *[{'kind': 'video_segment', 'record_scope': 'retention_interval', 'recording_id': 'r',
                    'segment_id': status, 'video_value': status} for status in ('selected', 'pending', 'confirmed_no_value')]]}}


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / '2026-10-20_cam'
    write(root / 'ProcessedClips/Index.json', canonical(root))
    return root


def models(**kwargs):
    return DeviceDayReadModels(published_recording_status, **kwargs)


def test_legacy_overview_url_redirects_to_bounded_current_library(archive, tmp_path):
    client = client_for(archive, tmp_path)
    response = client.get('/archive-overview', follow_redirects=False)
    assert response.status_code == 307
    assert response.headers['location'] == '/#/device-days'
    assert response.headers['cache-control'] == 'no-store'


def test_preferred_time_index_never_hydrates_large_canonical_and_keeps_sources(archive):
    source = archive / 'ProcessedClips/Index.json'
    with source.open('wb') as handle:
        handle.truncate(128 * 1024**2 + 1)
    before = source.stat()
    write(archive / 'Comment/TimeIndex.json', timeline(archive))
    view = models()
    summary = view.summary(archive)
    assert summary['segment_count'] == summary['preprocessing_candidate_count'] == 1
    assert summary['selected_interval_count'] == summary['pending_interval_count'] == 1
    assert summary['confirmed_no_value_interval_count'] == 1
    assert summary['transcript_count'] == 1
    assert summary['recordings'][0]['current_results_ready'] is True
    assert source.stat() == before
    data = view.time_index(archive).value
    assert data['recordings'][0]['sources'] == timeline(archive)['recordings'][0]['sources']
    assert data['recordings'][0]['capture_manifest'] == timeline(archive)['recordings'][0]['capture_manifest']
    assert data['field_events_file']['path'] == 'Comment/FieldEvents.json'
    assert 'images' not in data['streams'] and 'readable_content' not in data
    with pytest.raises(ReadModelDeferred, match='source limit'):
        view.index(archive)


def test_legacy_projection_retains_clocks_frames_steps_and_original_bytes(archive):
    path = archive / 'ProcessedClips/Index.json'
    before = path.read_bytes()
    data = models().index(archive).value
    assert data['recordings'][0]['processing'] == {'clock_mapping': {'unit': 'us'}}
    assert 'activity_audit' not in data['segments'][0]
    assert data['segments'][0]['key_frames'] == []
    assert data['segments'][0]['scene_frames'][0]['frame_kind'] == 'scene_sample'
    assert data['understandings'][0]['windows'][0]['steps'][0]['description'] == '实际观察'
    assert data['literal.item'] == {'item': 'preserved'}
    assert path.read_bytes() == before
    summary = models().summary(archive)
    assert summary['selected_interval_count'] is None
    assert summary['preprocessing_candidate_count'] == 1


def test_cache_reuses_encoded_response_and_invalidates_changed_or_invalid_source(archive, monkeypatch):
    from visioncortex import receipt_projection
    original, calls = receipt_projection.projected_json, []
    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(receipt_projection, 'projected_json', load)
    reader = models()
    first = reader.index(archive)
    assert reader.index(archive) is first and reader.index(archive).body is first.body
    assert len(calls) == 1
    replacement = write(archive / 'ProcessedClips/Replacement.json', canonical(archive) | {'updated_at': 'new'})
    replacement.replace(archive / 'ProcessedClips/Index.json')
    assert reader.index(archive).value['updated_at'] == 'new' and len(calls) == 2
    (archive / 'ProcessedClips/Index.json').write_text('{broken')
    with pytest.raises(ValueError):
        reader.index(archive)
    assert not reader.cache


def test_singleflight_does_not_start_a_second_decode_and_other_files_are_bounded(archive, monkeypatch):
    from visioncortex import receipt_projection
    original, entered, release, calls = receipt_projection.projected_json, threading.Event(), threading.Event(), []
    def load(*args, **kwargs):
        calls.append(args[0])
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(receipt_projection, 'projected_json', load)
    reader = models(readers=1)
    other = archive.parent / '2026-10-19_cam'
    write(other / 'ProcessedClips/Index.json', canonical(other))
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(reader.index, archive)
        assert entered.wait(2)
        try:
            with pytest.raises(ReadModelDeferred, match='being read'):
                reader.index(archive)
            with pytest.raises(ReadModelDeferred, match='readers are busy'):
                reader.index(other)
            assert len(calls) == 1
        finally:
            release.set()
        assert reader.index(archive).body == task.result().body


def test_projection_cannot_cache_a_generation_changed_during_loading(archive, monkeypatch):
    from visioncortex import receipt_projection
    original = receipt_projection.projected_json
    def load(*args, **kwargs):
        value = original(*args, **kwargs)
        write(archive / 'ProcessedClips/Index.json', canonical(archive) | {'updated_at': 'arrived later'})
        return value
    monkeypatch.setattr(receipt_projection, 'projected_json', load)
    reader = models()
    with pytest.raises(ReadModelDeferred, match='changed during reading'):
        reader.index(archive)
    assert not reader.cache and not reader.flights


@pytest.mark.parametrize('limit', ['resident_limit', 'response_limit'])
def test_large_projected_result_has_no_success_cache(archive, limit):
    reader = models(**{limit: 100})
    with pytest.raises((ReadModelDeferred, BlockingIOError)):
        reader.index(archive)
    assert not reader.cache


def test_lru_bounds_entries_and_memory_without_evading_file_identity(archive):
    reader = models(cache_entries=2, cache_limit=200000)
    for ordinal in range(4):
        root = archive.parent / f'2026-10-{ordinal + 1:02d}_cam'
        write(root / 'ProcessedClips/Index.json', canonical(root))
        reader.index(root)
    assert len(reader.cache) == 2 and reader.cache_bytes <= reader.cache_limit
    assert str(archive.parent / '2026-10-01_cam' / 'ProcessedClips/Index.json') not in {key[0] for key in reader.cache}


def test_broken_time_index_does_not_hide_failure_with_a_legacy_success(archive):
    write(archive / 'Comment/TimeIndex.json', {'recordings': [], 'streams': 'invalid'})
    with pytest.raises(ReadModelDeferred, match='Invalid published time'):
        models().summary(archive)


def test_legacy_v2_candidate_whole_video_does_not_prove_interval_selection(archive):
    value = timeline(archive)
    value['streams']['video'] = [row for row in value['streams']['video']
                                 if row.get('record_scope') != 'retention_interval']
    value['streams']['video'][-1]['file'] = {'path': 'ProcessedClips/Clips/LegacyWhole.mp4'}
    write(archive / 'Comment/TimeIndex.json', value)
    summary = models().summary(archive)
    assert summary['selected_interval_count'] is None
    assert summary['pending_interval_count'] is None
    assert summary['confirmed_no_value_interval_count'] is None
    assert summary['unpartitioned_candidate_count'] == 1


def test_linked_json_is_not_followed(archive, tmp_path):
    outside = write(tmp_path / 'outside.json', timeline(archive))
    path = archive / 'Comment/TimeIndex.json'
    path.parent.mkdir()
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        models().time_index(archive)


def client_for(archive, tmp_path):
    config = {'storage': {'archive_root': str(archive.parent), 'local_runtime_root': str(tmp_path / 'runtime')}}
    service = DeviceDayService(lambda: config, threading.Lock())
    app = FastAPI()
    install_routes(app, lambda: config, service)
    @app.get('/api/health')
    def health():
        return {'status': 'responsive'}
    return TestClient(app)


def test_api_keeps_large_day_visible_and_published_report_independent(archive, tmp_path):
    with (archive / 'ProcessedClips/Index.json').open('wb') as handle:
        handle.truncate(128 * 1024**2 + 1)
    write(archive / 'Comment/TimeIndex.json', timeline(archive))
    write(archive / 'LaboratoryDailyReport/LaboratoryDailyReport.json', {'actual': 'partial report'})
    client = client_for(archive, tmp_path)
    listing = client.get('/api/device-days').json()
    row = listing['archives'][0]
    assert row['report_exists'] is True and row['report_path'].endswith('.json')
    assert row['selected_interval_count'] == 1 and row['read_status'] == 'available'
    assert row['recordings'][0]['stages']['understanding']['status'] == 'partial'
    response = client.get(f'/api/device-days/{archive.name}/index')
    assert response.status_code == 503 and response.headers['retry-after'] == '3'
    assert client.get('/api/health').json() == {'status': 'responsive'}
    assert client.get(f'/api/device-days/{archive.name}/time-index').json()['recordings'][0]['sources']


def test_bad_archive_stays_explicit_while_other_archive_and_report_are_readable(archive, tmp_path):
    write(archive / 'Comment/TimeIndex.json', {'recordings': [], 'streams': 'broken'})
    other = archive.parent / '2026-10-19_cam'
    write(other / 'ProcessedClips/Index.json', canonical(other))
    result = client_for(archive, tmp_path).get('/api/device-days').json()
    rows = {row['archive']: row for row in result['archives']}
    assert rows[archive.name]['read_status'] == 'deferred'
    assert rows[archive.name]['selected_interval_count'] is None
    assert rows[other.name]['read_status'] == 'available'
    assert result['archive_read_errors'][0]['archive'] == archive.name


def test_api_projection_never_rewrites_or_substitutes_downloaded_json(archive, tmp_path):
    source = archive / 'ProcessedClips/Index.json'
    original = source.read_bytes()
    client = client_for(archive, tmp_path)
    projected = client.get(f'/api/device-days/{archive.name}/index')
    assert projected.status_code == 200 and 'activity_audit' not in projected.json()['segments'][0]
    original_response = client.get(f'/api/device-days/{archive.name}/files/ProcessedClips/Index.json')
    assert original_response.content == original
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(original).digest()


def test_published_files_and_listing_survive_missing_cv_index(archive, tmp_path):
    write(archive / 'Comment/TimeIndex.json', timeline(archive))
    report = {'actual': 'available partial report'}
    write(archive / 'LaboratoryDailyReport/LaboratoryDailyReport.json', report)
    clip = archive / 'ProcessedClips/Clips/Example/ExperimentActivity.mp4'
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b'fixture media bytes; not real-video quality evidence')
    (archive / 'ProcessedClips/Index.json').unlink()
    client = client_for(archive, tmp_path)
    listing = client.get('/api/device-days').json()['archives']
    assert [row['archive'] for row in listing] == [archive.name]
    assert listing[0]['read_status'] == 'available'
    base = f'/api/device-days/{archive.name}'
    assert client.get(base + '/time-index').status_code == 200
    assert client.get(base + '/files/LaboratoryDailyReport/LaboratoryDailyReport.json').json() == report
    assert client.get(base + '/files/ProcessedClips/Clips/Example/ExperimentActivity.mp4').content == clip.read_bytes()
    assert client.get(base + '/index').status_code == 503
    assert not (archive / 'ProcessedClips/Index.json').exists()


def test_missing_cv_index_never_uses_another_device_days_time_index(archive, tmp_path):
    times = timeline(archive) | {'archive': '2026-10-19_another_cam'}
    write(archive / 'Comment/TimeIndex.json', times)
    write(archive / 'LaboratoryDailyReport/LaboratoryDailyReport.json', {'fixture': True})
    (archive / 'ProcessedClips/Index.json').unlink()
    client = client_for(archive, tmp_path)
    response = client.get(f'/api/device-days/{archive.name}/files/LaboratoryDailyReport/LaboratoryDailyReport.json')
    assert response.status_code == 503
    assert client.get('/api/device-days').json()['archives'][0]['read_status'] == 'deferred'


def test_missing_cv_index_does_not_admit_unpublished_directory(archive, tmp_path):
    write(archive / 'LaboratoryDailyReport/LaboratoryDailyReport.json', {'fixture': True})
    (archive / 'ProcessedClips/Index.json').unlink()
    response = client_for(archive, tmp_path).get(
        f'/api/device-days/{archive.name}/files/LaboratoryDailyReport/LaboratoryDailyReport.json')
    assert response.status_code == 404
