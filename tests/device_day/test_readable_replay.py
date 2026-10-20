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


def test_new_local_completed_hint_overtakes_old_cursor_with_historical_share(tmp_path, monkeypatch):
    from visioncortex.timeline_invalidation import TimelineInvalidations
    from visioncortex.device_day_queue import DeviceDayQueue
    import time
    root = tmp_path/'runtime/device-day'
    now = time.time()
    current = {'recording_id': 'current', 'camera_key': 'current_cam01',
               'recording_start_us': round((now-100)*1e6), 'recording_end_us': round(now*1e6)}
    historical = [{'recording_id': 'old'+str(i), 'camera_key': 'old'+str(i)+'_cam01',
                   'recording_start_us': round((now-86400*(i+1))*1e6)} for i in range(3)]
    observe(root, {'recordings': [current, *historical]})
    recent_name = archive_name(current['camera_key'], current['recording_start_us'])
    old_names = [archive_name(r['camera_key'], r['recording_start_us']) for r in historical]
    # An old compatible checkpoint may be hours newer than every old day.
    atomic_json(root/'ReadablePublication/Service.json', {'checked': {recent_name: now-14400}})
    invalidations = TimelineInvalidations(root)
    for name in old_names:
        invalidations.changed(name, 'historical')
    queue = DeviceDayQueue(root/'queue-stt.sqlite3')
    queue.enqueue(current, 'native-stt')
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='completed',updated_at=?", (now,))
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
    config = {'storage': {'local_runtime_root': str(root.parent)}, 'runtime': {'local_only': True}}
    runner = SimpleNamespace(runtime_root=root, config=config)
    worker = ReadableReplay(threading.Event())
    assert worker.tick(runner)['archive'] == recent_name
    # Continuing arrivals keep the recent lane eligible, but not its third turn.
    for stamp in (now+1, now+2):
        with queue.connect() as db:
            db.execute('UPDATE recordings SET updated_at=?', (stamp,))
        worker.tick(runner)
    assert calls[:2] == [recent_name, recent_name]
    assert calls[2] in old_names
    assert worker.turn == 3 and len(worker.published_hints) == 2


def test_failed_day_never_acknowledges_changed_hint_or_blocks_other_archive(tmp_path, monkeypatch):
    from visioncortex.timeline_invalidation import TimelineInvalidations
    import time
    root = tmp_path/'runtime/device-day'
    start = round(time.time()*1e6)
    records = [{'recording_id': str(i), 'camera_key': f'owned{i}_cam01', 'recording_start_us': start}
               for i in range(2)]
    observe(root, {'recordings': records})
    names = [archive_name(r['camera_key'], r['recording_start_us']) for r in records]
    for name in names:
        TimelineInvalidations(root).changed(name, 'new')
    calls = []
    class Publisher:
        def __init__(self, config):
            pass
        def tick(self, *, archives):
            calls.extend(archives)
            return {'status': 'partial', 'published': 0, 'busy': 1}
    monkeypatch.setattr('visioncortex.device_day_file_index.FileIndexPublisher', Publisher)
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    monkeypatch.setattr('visioncortex.device_day_io.slot', lambda config: nullcontext())
    monkeypatch.setattr('visioncortex.runtime_control.resource_slot', lambda *a, **k: nullcontext())
    runner = SimpleNamespace(runtime_root=root, config={
        'storage': {'local_runtime_root': str(root.parent)}, 'runtime': {'local_only': True}})
    worker = ReadableReplay(threading.Event())
    worker.tick(runner)
    worker.tick(runner)
    assert set(calls) == set(names) and worker.published_hints == {}
    assert worker.tick(runner)['status'] == 'publication_backoff'
    assert all(stamp > time.time() for stamp in worker.retry_at.values())
