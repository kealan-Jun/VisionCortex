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
    assert catalog['path'].startswith('ProcessedClips/CaptureMetadata/')
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


def test_direct_media_can_be_read_through_existing_web_endpoints(config):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from visioncortex.device_day_service import DeviceDayService, install_routes
    from visioncortex.device_day_timeline import link
    folder = incoming(config, flat=True)
    audio(folder)
    record, = inventory(config)
    runner = DeviceDayRunner(config, FakeModels())
    assert runner.process(record)['status'] == 'completed'
    layout = runner.layout(record)
    app = FastAPI()
    install_routes(app, lambda: config, DeviceDayService(lambda: config, threading.Lock()))
    with TestClient(app) as client:
        page = client.get(f'/device-days/{layout.name}/watch/MetaData/rgb.mp4')
        assert page.status_code == 200
        video = client.get(f'/api/device-days/{layout.name}/files/MetaData/rgb.mp4', headers={'Range': 'bytes=0-8'})
        assert video.status_code == 206 and video.content == (folder / 'rgb.mp4').read_bytes()[:9]
        recording = client.get(f'/api/device-days/{layout.name}/files/MetaData/audio.opus')
        assert recording.status_code == 200 and recording.content == (folder / 'audio.opus').read_bytes()
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
