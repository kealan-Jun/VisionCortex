"""Owned local fixtures and fake ports; no NAS or real-model evidence."""
from copy import deepcopy
import os
from pathlib import Path
import threading
import time

import pytest

from repo_paths import ROOT
from test_device_day import FakeModels, capture, device_config  # noqa: F401
from visioncortex.capture_layout import METADATA_LAYOUT
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_capture_files import retain_capture_files
from visioncortex.device_day_contract import (
    DIRECTORIES, DeviceDayLayout, archive_name, atomic_json, read_json, safe_child,
)
from visioncortex.device_day_inputs import input_path, seal_input
from visioncortex.device_day_monitor import CameraMonitor
from visioncortex.nas_recordings import scan_recordings


@pytest.fixture
def config(device_config):  # noqa: F811
    device_config['storage']['archive_root'] = str(Path(device_config['collection_ingest']['source_root']) / 'Archive')
    device_config['device_day']['inplace_preprocessing'] = True
    return device_config


def stable(folder, *, audio_only=False):
    for path in folder.iterdir():
        if path.is_file() and (not audio_only or path.name.startswith('audio')):
            os.utime(path, (time.time() - 1000, time.time() - 1000))


def incoming(config, *, camera='a_cam01', start=1789005600000000, label='100000', flat=False):
    original = capture(config, camera, start=start, label=label)
    day = Path(config['storage']['archive_root']) / archive_name(camera, start)
    folder = day / 'MetaData'
    if not flat:
        folder /= label
    folder.mkdir(parents=True, exist_ok=True)
    for path in original.iterdir():
        path.rename(folder / path.name)
    original.rmdir()
    original.parent.rmdir()
    original.parent.parent.rmdir()
    return folder


def audio(folder, start=1789005600000000, *, ready=True):
    (folder / 'audio.opus').write_bytes(b'owned-fake-audio')
    (folder / 'audio_timing.csv').write_text(f'global_timestamp_us\n{start}\n')
    atomic_json(folder / 'audio_meta.json', {'recording_session_id': 'fixture-1', 'audio_valid': True,
                                            'first_audio_global_us': start, 'last_audio_global_us': start + 10000000})
    atomic_json(folder / 'audio_ready.json', {'recording_session_id': 'fixture-1', 'ready': ready, 'audio_valid': True})
    stable(folder, audio_only=True)


def inventory(config):
    result = scan_recordings(config)
    assert not result['errors'], result
    return result['recordings']


@pytest.mark.parametrize('stage', ['retention', 'stt'])
def test_split_worker_admits_ready_native_input_without_camera_role(config, stage):
    from visioncortex.device_day_retention_worker import RetentionWorker
    config['collection_ingest']['camera_role_map'] = {}
    folder = incoming(config)
    audio(folder)
    stable(folder)
    record, = inventory(config)
    assert not record.get('configured_role')
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    backend = FakeModels()
    runner = DeviceDayRunner(config, backend)
    worker = RetentionWorker(stage=stage)
    assert worker.admit(runner, record) is True
    claimed = runner.queues[stage].claim('independent-worker')
    assert claimed and claimed['recording_id'] == record['recording_id']
    result = worker._execute_claimed(runner, record, 'independent-worker', threading.Event())
    assert result['result']['status'] == 'completed', result
    assert backend.vision_calls == backend.semantic_calls == 0
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    assert RetentionWorker(stage='vision').admit(runner, record) is False


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('inplace', [False, True])
def test_precreated_metadata_processes_without_changing_or_copying_sources(config, monkeypatch, flat, inplace):
    config['device_day']['inplace_preprocessing'] = inplace
    folder = incoming(config, flat=flat)
    audio(folder)
    (folder / 'camera-log.txt').write_bytes(b'keep-native-name-and-bytes')
    stable(folder)
    before = {p.name: (p.read_bytes(), p.stat().st_ino) for p in folder.iterdir()}
    record, = inventory(config)
    assert record['input_layout'] == METADATA_LAYOUT
    assert record['camera_key'] == 'a_cam01'
    assert record['configured_role'] == 'first_person'
    backend = FakeModels()
    runner = DeviceDayRunner(config, backend)
    layout = runner.layout(record)
    original_open = Path.open
    def open_checked(path, mode='r', *args, **kwargs):
        if path.is_relative_to(layout.raw) and any(flag in mode for flag in ('w', 'a', '+', 'x')):
            pytest.fail(f'Pipeline tried to write recorder-owned input: {path}')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_checked)
    result = runner.process(record)
    assert result['status'] == 'completed', result
    receipt = read_json(runner._receipt(layout, record, 'retention'))
    for source in receipt['sources']:
        assert safe_child(layout.root, source['retained']['path']) == Path(source['original_path'])
    catalog = retain_capture_files(runner, record)
    assert catalog['path'].startswith('ProcessedClips/.Evidence/CaptureMetadata/')
    entries = read_json(safe_child(layout.root, catalog['path']))['files']
    assert next(e for e in entries if e['original_name'] == 'camera-log.txt')['retained']['path'].endswith('camera-log.txt')
    assert before == {p.name: (p.read_bytes(), p.stat().st_ino) for p in folder.iterdir()}
    assert set(p.name for p in layout.root.iterdir()) == set(DIRECTORIES)
    assert runner.process(record)['status'] == 'completed'
    assert backend.vision_calls == 1
    schema = read_json(ROOT / 'docs/contracts/device-day-v1.schema.json')
    assert 'MetaData' in schema['$defs']['artifact']['properties']['path']['pattern']


def test_empty_precreated_folder_and_unfinished_video_wait_for_publication(config):
    root = Path(config['storage']['archive_root']) / '2026-09-10_a_cam01' / 'MetaData'
    root.mkdir(parents=True)
    assert inventory(config) == []
    assert list(root.parent.iterdir()) == [root]
    folder = incoming(config)
    (folder / 'recording_ready.json').unlink()
    record, = inventory(config)
    assert not record['processable']
    assert '等待采集完成标记' in record['issues']
    assert not (root.parent / 'ProcessedClips').exists()


@pytest.mark.parametrize('sidecar', ['audio_meta.json', 'audio_ready.json'])
def test_incomplete_audio_json_does_not_block_ready_video(config, sidecar):
    folder = incoming(config)
    audio(folder)
    (folder / sidecar).write_text('{"ready":')
    record, = inventory(config)
    assert record['processable']
    assert record['audio']['status'] == 'pending_publication'
    assert record['audio']['files'] == []
    assert record['audio']['read_errors']


def test_late_audio_is_admitted_without_repeating_vision(config):
    folder = incoming(config)
    record, = inventory(config)
    backend = FakeModels()
    runner = DeviceDayRunner(config, backend)
    assert runner.process(record, stage='vision')['status'] == 'completed'
    assert runner.process(record, stage='retention')['status'] == 'completed'
    audio(folder, ready=False)
    waiting, = inventory(config)
    assert waiting['audio']['status'] == 'pending_publication'
    audio(folder)
    fresh, = inventory(config)
    assert fresh['recording_id'] == record['recording_id']
    assert fresh['source_signature'] == record['source_signature']
    assert fresh['audio']['status'] == 'provided'
    assert runner.process(fresh, stage='vision')['status'] == 'completed'
    assert backend.vision_calls == 1
    assert runner.process(fresh, stage='stt')['status'] == 'completed'
    assert runner.process(fresh, stage='retention')['status'] == 'completed'
    saved = read_json(input_path(runner, fresh, 'stt'))
    assert any(s['kind'] == 'audio_audio' and s['retained']['path'].endswith('/audio.opus') for s in saved['sources'])


def test_new_input_and_legacy_archive_are_discovered_once_without_outputs(config):
    folder = incoming(config)
    legacy = capture(config, camera='b_cam02')
    day = folder.parent.parent
    (day / 'MetaVideo').mkdir()
    (day / 'MetaVideo/rgb.mp4').write_bytes(b'not-an-incoming-recording')
    (day / 'ProcessedClips').mkdir()
    (day / 'ProcessedClips/rgb.mp4').write_bytes(b'not-an-incoming-recording')
    items = inventory(config)
    assert {r['video_path'] for r in items} == {str(folder / 'rgb.mp4'), str(legacy / 'rgb.mp4')}
    restricted = deepcopy(config)
    restricted['collection_ingest']['camera_directories'] = ['a_cam01']
    assert [r['video_path'] for r in inventory(restricted)] == [str(folder / 'rgb.mp4')]
    restricted['collection_ingest']['discover_recorder_metadata'] = False
    assert inventory(restricted) == []


def test_archive_itself_can_be_the_configured_source(config):
    folder = incoming(config)
    config['collection_ingest']['source_root'] = config['storage']['archive_root']
    record, = inventory(config)
    assert record['camera_key'] == 'a_cam01'
    assert record['video_path'] == str(folder / 'rgb.mp4')


def test_metadata_device_day_mismatch_and_symlink_fail_closed(config, tmp_path):
    folder = incoming(config)
    ready = read_json(folder / 'recording_ready.json')
    atomic_json(folder / 'recording_ready.json', ready | {'camera_key': 'b_cam02'})
    stable(folder)
    record, = inventory(config)
    assert not record['processable']
    assert 'MetaData 目录与采集设备或日期不一致' in record['issues']
    atomic_json(folder / 'recording_ready.json', ready)
    (folder / 'rgb.mp4').unlink()
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(b'outside')
    (folder / 'rgb.mp4').symlink_to(outside)
    assert inventory(config) == []


def test_inplace_retention_rejects_replaced_recorder_file(config):
    folder = incoming(config)
    record, = inventory(config)
    runner = DeviceDayRunner(config, FakeModels())
    seal_input(runner, record, 'vision')
    target = folder / 'rgb.mp4'
    content = target.read_bytes()
    target.unlink()
    target.write_bytes(content)
    result = runner.process(record, stage='retention')
    assert result['status'] == 'failed', result
    saved = read_json(runner._receipt(runner.layout(record), record, 'retention'))
    assert saved['status'] == 'failed'
    assert not runner.layout(record).index.exists()


def test_existing_metavideo_keeps_historical_layout(config):
    root = Path(config['storage']['archive_root']) / '2026-09-10_a_cam01'
    (root / 'MetaVideo').mkdir(parents=True)
    layout = DeviceDayLayout(Path(config['storage']['archive_root']), 'a_cam01', 1789005600000000)
    layout.create()
    assert layout.raw == root / 'MetaVideo'
    assert not (root / 'MetaData').exists()


def test_prefixed_native_files_and_explicit_shared_audio_names(config):
    folder = incoming(config, flat=True)
    names = {'task_audio_file': 'audio.opus', 'task_audio_meta_file': 'audio_meta.json',
             'task_audio_ready_file': 'audio_ready.json', 'task_audio_timing_file': 'audio_timing.csv'}
    prefix = 'test_Short_0007_'
    for name in ('meta.json', 'recording_ready.json'):
        atomic_json(folder / name, read_json(folder / name) | names |
                    {'rgb_file': prefix + 'rgb.mp4', 'frames_file': prefix + 'frames.csv'})
    for name in ('rgb.mp4', 'frames.csv', 'meta.json', 'recording_ready.json'):
        (folder / name).rename(folder / (prefix + name))
    audio(folder)
    stable(folder)
    record, = inventory(config)
    assert record['processable'] and record['audio']['status'] == 'provided'
    runner = DeviceDayRunner(config, FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    saved = read_json(runner._receipt(runner.layout(record), record, 'retention'))
    assert next(s for s in saved['sources'] if s['kind'] == 'video')['retained']['path'] == 'MetaData/' + prefix + 'rgb.mp4'
    assert next(s for s in saved['sources'] if s['kind'] == 'audio_audio')['retained']['path'] == 'MetaData/audio.opus'


def test_mixed_day_retains_legacy_paths_and_routes_direct_input_to_metadata(config):
    folder = incoming(config)
    root = folder.parent.parent
    (root / 'MetaVideo').mkdir()
    legacy_layout = DeviceDayLayout(Path(config['storage']['archive_root']), 'a_cam01', 1789005600000000)
    assert legacy_layout.raw.name == 'MetaVideo'
    record, = inventory(config)
    direct_layout = DeviceDayRunner(config, FakeModels()).layout(record)
    assert direct_layout.raw.name == 'MetaData'
    assert direct_layout.source_path(record, 'video', folder / 'rgb.mp4') == folder / 'rgb.mp4'
    legacy_record = {k: v for k, v in record.items() if k != 'input_layout'}
    assert DeviceDayRunner(config, FakeModels()).layout(legacy_record).raw.name == 'MetaVideo'


def test_skipped_flat_metadata_root_is_not_reinspected(config, monkeypatch):
    from visioncortex import nas_recordings
    folder = incoming(config, flat=True)
    monkeypatch.setattr(nas_recordings, '_inspect', lambda *_: pytest.fail('Sealed folder was re-read'))
    result = scan_recordings(config, skip_folders={str(folder)})
    assert result['recordings'] == []
    assert result['camera_directory_count'] == 1


def test_missing_metadata_is_not_recreated_by_processing(config, tmp_path):
    folder = incoming(config, flat=True)
    record, = inventory(config)
    layout = DeviceDayRunner(config, FakeModels()).layout(record)
    folder.rename(tmp_path / 'producer-unavailable')
    with pytest.raises(ValueError, match='Recorder-owned MetaData is unavailable'):
        layout.create()
    assert not folder.exists()
    assert not layout.processed.exists()


def test_metadata_date_scope_is_applied_before_record_reads(config, monkeypatch):
    from visioncortex import nas_recordings
    incoming(config)
    folder = incoming(config, start=1789092000000000, label='next-day')
    config['collection_ingest']['capture_date'] = '2026-09-11'
    original = nas_recordings._inspect
    def inspect(root, video, *args):
        assert video.parent == folder
        return original(root, video, *args)
    monkeypatch.setattr(nas_recordings, '_inspect', inspect)
    record, = inventory(config)
    assert record['video_path'] == str(folder / 'rgb.mp4')


@pytest.mark.parametrize('mode', ['legacy_and_metadata', 'metadata_only'])
def test_direct_media_can_be_read_through_existing_web_endpoints(config, mode):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from visioncortex.device_day_service import DeviceDayService, install_routes
    from visioncortex.device_day_timeline import link
    folder = incoming(config, flat=True)
    config['collection_ingest']['discovery_mode'] = mode
    audio(folder)
    record, = inventory(config)
    runner = DeviceDayRunner(config, FakeModels())
    assert runner.process(record)['status'] == 'completed'
    layout = runner.layout(record)
    legacy = layout.root / 'MetaVideo'
    legacy.mkdir()
    (legacy / 'historical.mp4').write_bytes(b'historical-owned-fixture')
    app = FastAPI()
    install_routes(app, lambda: config, DeviceDayService(lambda: config, threading.Lock()))
    with TestClient(app) as client:
        page = client.get(f'/device-days/{layout.name}/watch/MetaData/rgb.mp4')
        assert page.status_code == 200
        video = client.get(f'/api/device-days/{layout.name}/files/MetaData/rgb.mp4', headers={'Range': 'bytes=0-8'})
        assert video.status_code == 206 and video.content == (folder / 'rgb.mp4').read_bytes()[:9]
        recording = client.get(f'/api/device-days/{layout.name}/files/MetaData/audio.opus')
        assert recording.status_code == 200 and recording.content == (folder / 'audio.opus').read_bytes()
        historical = client.get(f'/api/device-days/{layout.name}/files/MetaVideo/historical.mp4')
        assert historical.status_code == 200 and historical.content == b'historical-owned-fixture'
    assert link(layout.name, 'MetaData/audio.opus') == f'/api/device-days/{layout.name}/files/MetaData/audio.opus'


def test_live_monitor_observes_metadata_only_camera_without_duplicate_lane(config):
    config['collection_ingest']['camera_directories'] = ['a_cam01']
    incoming(config, start=int((time.time() - 10) * 1e6))
    stop, observed = threading.Event(), threading.Event()
    records = []
    def observe(_config, snapshot):
        records.extend(snapshot['recordings'])
        if records:
            observed.set()
    monitor = CameraMonitor(config, stop, observe, live_only=True)
    try:
        snapshot = monitor.poll()
        assert snapshot['camera_directories'] == ['a_cam01']
        assert observed.wait(5), monitor.states
        assert len(records) == 1
        assert records[0]['input_layout'] == METADATA_LAYOUT
    finally:
        stop.set()
        for thread in monitor.threads.values():
            thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in monitor.threads.values())


@pytest.mark.parametrize('plain', [False, True])
@pytest.mark.parametrize('explicit_cameras', [False, True])
def test_metadata_only_never_walks_old_inputs(config, monkeypatch, plain, explicit_cameras):
    folder = incoming(config)
    old = capture(config, camera='b_cam02')
    lab = Path(config['collection_ingest']['source_root']) / 'lab_video'
    lab.mkdir()
    settings = config['collection_ingest']
    settings.update(discovery_mode='metadata_only', discover_plain_video_csv=plain,
                    additional_camera_directories=['lab_video'],
                    camera_directories=['a_cam01', 'b_cam02'] if explicit_cameras else [])
    walk = os.walk
    def walk_checked(path, *args, **kwargs):
        assert Path(path).is_relative_to(folder.parent), f'Old input was traversed: {path}'
        return walk(path, *args, **kwargs)
    monkeypatch.setattr(os, 'walk', walk_checked)
    result = scan_recordings(config)
    assert result['discovery_mode'] == 'metadata_only'
    assert [r['video_path'] for r in result['recordings']] == [str(folder / 'rgb.mp4')]
    assert result['camera_directories'] == ['a_cam01']
    assert (old / 'rgb.mp4').is_file()


def test_metadata_only_waits_for_producer_and_never_probes_external_camera(config, monkeypatch):
    from visioncortex.device_day_live_discovery import discovery_config
    Path(config['storage']['archive_root']).mkdir()
    old_camera = Path(config['collection_ingest']['source_root']) / 'a_cam01'
    old_camera.mkdir()
    config['collection_ingest']['discovery_mode'] = 'metadata_only'
    config = discovery_config(config)
    config['collection_ingest']['camera_directories'] = ['a_cam01']
    real_is_dir, real_iterdir = Path.is_dir, Path.iterdir
    def is_dir_checked(path):
        assert path != old_camera, 'Old camera was probed'
        return real_is_dir(path)
    def iterdir_checked(path):
        assert path != old_camera, 'Old camera was listed'
        return real_iterdir(path)
    stop, observed = threading.Event(), threading.Event()
    records = []
    def observe(_config, snapshot):
        records.extend(snapshot['recordings'])
        if records:
            observed.set()
    monitor = CameraMonitor(config, stop, observe, live_only=True)
    try:
        with monkeypatch.context() as guarded:
            guarded.setattr(Path, 'is_dir', is_dir_checked)
            guarded.setattr(Path, 'iterdir', iterdir_checked)
            assert monitor.poll()['discovery_mode'] == 'metadata_only'
            deadline = time.monotonic() + 5
            while not monitor.states['a_cam01', 'live'].get('awaiting_input') and time.monotonic() < deadline:
                stop.wait(0.01)
            assert monitor.states['a_cam01', 'live']['awaiting_input']
        assert list(Path(config['storage']['archive_root']).iterdir()) == []
        # The fixture's producer makes its own native camera/day hierarchy.
        old_camera.rmdir()
        incoming(config, start=int((time.time() - 10) * 1e6))
        with monkeypatch.context() as guarded:
            guarded.setattr(Path, 'is_dir', is_dir_checked)
            guarded.setattr(Path, 'iterdir', iterdir_checked)
            assert observed.wait(7), monitor.states
        assert len(records) == 1 and records[0]['camera_key'] == 'a_cam01'
        assert set(monitor.threads) == {('a_cam01', 'live')}
    finally:
        stop.set()
        for thread in monitor.threads.values():
            thread.join(5)


def test_metadata_only_discovery_uses_device_identity_not_legacy_lab_root():
    from visioncortex.device_day_live_discovery import discovery_config
    source = {'storage': {}, 'device_day': {}, 'collection_ingest': {
        'discovery_mode': 'metadata_only', 'camera_role_map': {
            'a_cam01': 'first_person', 'lab-video-1': 'first_person', 'unknown': 'unset'},
        'directory_camera_bindings': [{'camera_key': 'lab-video-1', 'directory_glob': 'lab_video/20??-??-??-1/*'}]}}
    assert discovery_config(source)['collection_ingest']['camera_directories'] == ['a_cam01', 'lab-video-1']


def test_live_metadata_discovers_unregistered_cameras_in_bounded_dates(config, monkeypatch):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from visioncortex.device_day_live_discovery import live_config
    today = datetime.now(ZoneInfo('Asia/Shanghai')).date()
    for offset, camera in enumerate(('new_cam01', 'late_cam02', 'old_cam03')):
        day = today - timedelta(days=offset)
        start = int(datetime.combine(day, datetime.min.time(), ZoneInfo('Asia/Shanghai')).timestamp() * 1e6)
        incoming(config, camera=camera, start=start)
    config['collection_ingest'].update(discovery_mode='metadata_only', camera_directories=['a_cam01'])
    before = deepcopy(config)
    old = Path(config['storage']['archive_root']) / archive_name('old_cam03', start) / 'MetaData'
    is_dir = Path.is_dir
    def checked(path):
        assert path != old, 'Older metadata must not be probed'
        return is_dir(path)
    monkeypatch.setattr(Path, 'is_dir', checked)
    current = live_config(config)
    assert set(current['collection_ingest']['camera_directories']) == {
        'a_cam01', 'b_cam02', 'new_cam01', 'late_cam02'}
    assert current['collection_ingest']['camera_role_map'] == before['collection_ingest']['camera_role_map']
    assert config == before


def test_unregistered_metadata_is_persisted_and_alerted_without_model_execution(config):
    from visioncortex.device_day_live_discovery import discovery_status, live_config
    from visioncortex.device_day_service import DeviceDayService
    from visioncortex.observed_inventory import read_inventory
    folder = incoming(config, camera='new_cam01', start=int((time.time() - 10) * 1e6))
    before = {path: (path.read_bytes(), path.stat().st_ino) for path in folder.rglob('*') if path.is_file()}
    config['collection_ingest'].update(discovery_mode='metadata_only', camera_role_map={})
    current = live_config(config)
    stop, observed = threading.Event(), threading.Event()
    service = DeviceDayService(lambda: current, threading.Lock())
    def observe(settings, snapshot):
        service.observe(settings, snapshot)
        if snapshot['recordings']:
            observed.set()
    monitor = CameraMonitor(current, stop, observe, live_only=True)
    try:
        monitor.poll(include_recordings=False)
        assert observed.wait(5)
        status = discovery_status(current, monitor.poll(include_recordings=False))
        assert status['status'] == 'degraded'
        assert status['unconfigured_camera_directories'] == ['new_cam01']
        assert status['alerts'][0]['type'] == 'camera_role_unconfigured'
        root = Path(config['storage']['local_runtime_root']) / 'device-day'
        record, = read_inventory(root)['recordings']
        assert record['camera_key'] == 'new_cam01' and not record.get('configured_role')
        models = FakeModels()
        result = DeviceDayRunner(current, models).process(record)
        assert result['status'] == 'needs_camera_role'
        assert {stage: value['status'] for stage, value in result['independent_stages'].items()} == {
            'retention': 'completed', 'stt': 'completed'}
        assert models.vision_calls == models.semantic_calls == 0
        processed = Path(config['storage']['archive_root']) / archive_name('new_cam01', record['recording_start_us']) / 'ProcessedClips'
        assert not list(processed.rglob('*.mp4'))
        assert all((path.read_bytes(), path.stat().st_ino) == expected for path, expected in before.items())
        # Explicit registration resolves the alert; discovery never guesses it.
        current['collection_ingest']['camera_role_map']['new_cam01'] = 'third_person'
        status = discovery_status(current, monitor.poll(include_recordings=False))
        assert status['unconfigured_camera_directories'] == [] and status['alerts'] == []
    finally:
        stop.set()
        for thread in monitor.threads.values():
            thread.join(5)


@pytest.mark.parametrize('update', [
    {'discovery_mode': 'typo'},
    {'discovery_mode': 'metadata_only', 'discover_recorder_metadata': False},
])
def test_invalid_discovery_mode_fails_closed(config, update):
    from visioncortex.capture_layout import validate
    config['collection_ingest'].update(update)
    with pytest.raises(ValueError):
        validate(config['collection_ingest'])
    with pytest.raises(ValueError):
        scan_recordings(config)


def test_metadata_only_unavailable_root_does_not_fall_back_or_create(config):
    capture(config)
    config['collection_ingest']['discovery_mode'] = 'metadata_only'
    with pytest.raises(OSError, match='archive root is unavailable'):
        scan_recordings(config)
    assert not Path(config['storage']['archive_root']).exists()
    config['storage']['archive_root'] = str(Path(config['storage']['local_runtime_root']) / 'outside')
    with pytest.raises(ValueError, match='within source_root'):
        scan_recordings(config)


def test_two_cameras_keep_native_names_audio_and_outputs_in_their_own_day(config):
    config['collection_ingest']['discovery_mode'] = 'metadata_only'
    folders = [incoming(config, camera=camera, label='native-capture/session')
               for camera in ('a_cam01', 'b_cam02')]
    for folder in folders:
        audio(folder)
    records = inventory(config)
    assert len(records) == 2
    runner = DeviceDayRunner(config, FakeModels())
    for record in records:
        assert runner.process(record)['status'] == 'completed'
        layout = runner.layout(record)
        assert layout.name == '2026-09-10_' + record['camera_key']
        saved = read_json(runner._receipt(layout, record, 'retention'))
        for source in saved['sources']:
            assert source['retained']['path'].startswith('MetaData/native-capture/session/')
            assert safe_child(layout.root, source['retained']['path']) == Path(source['original_path'])
            assert Path(source['original_path']).is_relative_to(layout.raw)


def test_wrong_video_metadata_cannot_be_hidden_by_matching_ready_camera(config):
    folder = incoming(config)
    atomic_json(folder / 'meta.json', read_json(folder / 'meta.json') | {'camera_key': 'b_cam02'})
    atomic_json(folder / 'recording_ready.json', read_json(folder / 'recording_ready.json') | {'camera_key': 'a_cam01'})
    stable(folder)
    record, = inventory(config)
    assert not record['processable']
    assert 'MetaData 目录与采集设备或日期不一致' in record['issues']


def test_one_camera_metadata_cannot_nest_another_device_day(config):
    folder = incoming(config)
    wrong = folder / '2026-09-10_b_cam02' / 'MetaData'
    wrong.mkdir(parents=True)
    for path in list(folder.iterdir()):
        if path.is_file():
            path.rename(wrong / path.name)
    result = scan_recordings(config)
    assert result['recordings'] == []
    assert any('Nested MetaData device/day identities' in e['message'] for e in result['errors'])


def test_switching_discovery_mode_keeps_completed_vision_receipt(config):
    folder = incoming(config)
    record, = inventory(config)
    runner = DeviceDayRunner(config, FakeModels())
    assert runner.process(record, stage='vision')['status'] == 'completed'
    path = runner._receipt(runner.layout(record), record, 'vision')
    before = path.read_bytes()
    config['collection_ingest']['discovery_mode'] = 'metadata_only'
    backend = FakeModels()
    current = DeviceDayRunner(config, backend)
    assert current.process(record, stage='vision')['status'] == 'completed'
    assert backend.vision_calls == 0
    assert path.read_bytes() == before
    assert (folder / 'rgb.mp4').is_file()


def test_switching_discovery_keeps_already_admitted_legacy_work_processable(config):
    folder = capture(config)
    before = {path: (path.read_bytes(), path.stat().st_ino) for path in folder.iterdir() if path.is_file()}
    record, = inventory(config)
    layout = DeviceDayLayout(Path(config['storage']['archive_root']), record['camera_key'],
                            record['recording_start_us'], raw_directory='MetaVideo')
    # The fixture supplies a pre-existing historical raw root. The pipeline
    # cannot create a recorder-owned root merely to switch discovery modes.
    layout.raw.mkdir(parents=True)
    layout.create()
    config['collection_ingest']['discovery_mode'] = 'metadata_only'
    assert inventory(config) == []
    runner = DeviceDayRunner(config, FakeModels())
    assert runner.process(record)['status'] == 'completed'
    assert runner.layout(record).raw.name == 'MetaVideo'
    assert (folder / 'rgb.mp4').is_file()
    assert all((path.read_bytes(), path.stat().st_ino) == expected for path, expected in before.items())


@pytest.mark.parametrize('names', ['a_cam01', ['../outside']])
def test_metadata_only_validates_camera_scope_before_monitoring(config, names):
    config['collection_ingest'].update(discovery_mode='metadata_only', camera_directories=names)
    with pytest.raises(ValueError, match='相机目录配置无效'):
        scan_recordings(config)


@pytest.mark.parametrize('field,value,issue', [
    ('camera_key', 'b_cam02', 'camera_key'),
    ('first_audio_global_us', 1789092000000000, 'capture_date'),
])
def test_other_camera_or_day_audio_is_rejected_without_blocking_video(config, field, value, issue):
    folder = incoming(config)
    audio(folder)
    atomic_json(folder / 'audio_meta.json', read_json(folder / 'audio_meta.json') | {field: value})
    stable(folder)
    record, = inventory(config)
    assert record['processable']
    assert record['audio']['status'] == 'association_mismatch'
    assert issue in record['audio']['association_issues']
    assert record['audio']['files'] == []
