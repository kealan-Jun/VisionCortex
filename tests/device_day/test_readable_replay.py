"""Readable publication composes the existing owner without model invocation."""
from contextlib import nullcontext
from types import SimpleNamespace
import threading

from visioncortex.device_day_contract import archive_name, atomic_json
from visioncortex.device_day_publication_worker import ReadableReplay
from visioncortex.observed_inventory import observe


def test_readable_mode_restores_cursor_and_calls_only_file_index_owner(tmp_path, monkeypatch):
    root = tmp_path / 'runtime' / 'device-day'
    records = [{'recording_id': str(i), 'camera_key': f'owned{i}_cam01',
                'recording_start_us': 1_800_000_000_000_000} for i in range(2)]
    observe(root, {'recordings': records})
    names = [archive_name(r['camera_key'], r['recording_start_us']) for r in records]
    atomic_json(root / 'ReadablePublication' / 'Service.json', {'checked': {names[0]: 999}})
    calls = []
    class Publisher:
        def __init__(self, config):
            calls.append('created')
        def tick(self, *, archives):
            calls.append(archives)
            return {'status': 'completed', 'published': 1}
    monkeypatch.setattr('visioncortex.device_day_file_index.FileIndexPublisher', Publisher)
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    monkeypatch.setattr('visioncortex.device_day_io.slot', lambda config: nullcontext())
    monkeypatch.setattr('visioncortex.runtime_control.resource_slot', lambda *a, **k: nullcontext())
    config = {'storage': {'local_runtime_root': str(root.parent)}, 'runtime': {'local_only': True}}
    worker = ReadableReplay(threading.Event())
    runner = SimpleNamespace(runtime_root=root, config=config)
    result = worker.tick(runner)
    assert result['archive'] == names[1]
    assert calls == ['created', [names[1]]]
    assert worker.tick(runner)['archive'] == names[0]
    assert calls.count('created') == 1


def test_field_archive_discovery_failure_does_not_block_existing_video_publication(tmp_path, monkeypatch):
    root = tmp_path/'runtime/device-day'
    record = {'recording_id': 'a', 'camera_key': 'owned_cam01', 'recording_start_us': 1_800_000_000_000_000}
    observe(root, {'recordings': [record]})
    calls = []
    class Publisher:
        def __init__(self, config):
            pass
        def tick(self, *, archives):
            calls.extend(archives)
            return {'status': 'completed', 'published': 1}
    monkeypatch.setattr('visioncortex.device_day_file_index.FileIndexPublisher', Publisher)
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    monkeypatch.setattr('visioncortex.device_day_io.slot', lambda config: nullcontext())
    monkeypatch.setattr('visioncortex.runtime_control.resource_slot', lambda *a, **k: nullcontext())
    config = {'storage': {'local_runtime_root': str(root.parent)}, 'runtime': {'local_only': True},
              'device_day': {'field_recognition': {'archive_context_enabled': True,
                                                 'archive_root': str(tmp_path/'unavailable')}}}
    result = ReadableReplay(threading.Event()).tick(SimpleNamespace(runtime_root=root, config=config))
    assert result['status'] == 'published' and len(calls) == 1
    assert result['field_archive_discovery']['status'] == 'unavailable'
    assert result['field_archive_discovery']['other_publication_continues'] is True


def test_readable_mode_stops_before_projection_with_no_media_reads(tmp_path, monkeypatch):
    root = tmp_path / 'runtime' / 'device-day'
    observe(root, {'recordings': []})
    monkeypatch.setattr('visioncortex.device_day_file_index.FileIndexPublisher', lambda config: object())
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    stop = threading.Event()
    stop.set()
    runner = SimpleNamespace(runtime_root=root, config={
        'storage': {'local_runtime_root': str(root.parent)}, 'runtime': {'local_only': True}})
    assert ReadableReplay(stop).tick(runner)['status'] == 'stopping'


def test_readable_service_preserves_round_robin_cursor_on_shutdown(tmp_path, monkeypatch):
    from visioncortex import device_day_publication_worker as module
    from visioncortex.device_day_contract import read_json
    root = tmp_path / 'runtime' / 'device-day'
    record = {'recording_id': 'a', 'camera_key': 'owned_cam01', 'recording_start_us': 1_800_000_000_000_000}
    observe(root, {'recordings': [record]})
    name = archive_name(record['camera_key'], record['recording_start_us'])
    path = root / 'ReadablePublication' / 'Service.json'
    atomic_json(path, {'checked': {name: 123}})
    config = {'storage': {'local_runtime_root': str(root.parent)}, 'runtime': {'local_only': True}}
    stop = threading.Event()
    class Publisher:
        def __init__(self, config):
            pass
        def tick(self, *, archives):
            assert read_json(path)['checked'][name] == 123
            stop.set()
            return {'status': 'completed', 'published': 0}
    monkeypatch.setattr('visioncortex.device_day_file_index.FileIndexPublisher', Publisher)
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    monkeypatch.setattr('visioncortex.device_day_io.slot', lambda config: nullcontext())
    monkeypatch.setattr('visioncortex.runtime_control.resource_slot', lambda *a, **k: nullcontext())
    monkeypatch.setattr('visioncortex.config.load_config', lambda path: config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda value: value)
    monkeypatch.setattr('visioncortex.device_day.DeviceDayRunner', lambda config: SimpleNamespace(runtime_root=root, config=config))
    module.serve('owned-config.yaml', stop, readable=True)
    state = read_json(path)
    assert state['status'] == 'stopped' and state['checked'][name] > 123
    assert state['scope'] == 'readable_file_projection' and state['model_invoked'] is False
