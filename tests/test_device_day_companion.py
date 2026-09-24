"""Live handoff tests use local fake processes/services; no NAS, GPU or HTTP."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import pytest

from visioncortex.device_day_companion import AncillaryIndexes, Companion, MonitorBridge, validate_handoff
from visioncortex.device_day_contract import atomic_json


def setup_handoff(tmp_path):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('device_day: {enabled: false}')
    config = {'device_day': {'enabled': False},
              'storage': {'local_runtime_root': str(tmp_path / 'runtime'),
                          'local_cache_root': str(tmp_path / 'cache'),
                          'archive_root': str(tmp_path / 'archive')},
              'collection_ingest': {'source_root': str(tmp_path / 'capture')}}
    receipt = {'schema_version': 'visioncortex-device-day-handoff/1',
               'legacy_dispatch_drained': True, 'legacy_pid': 2147483000,
               'legacy_restart_configuration_verified': True,
               'legacy_start_ticks': 123, 'legacy_dispatch_thread_ids': [2],
               'config_path': str(config_path),
               'runtime_root': config['storage']['local_runtime_root']}
    path = tmp_path / 'handoff.json'
    atomic_json(path, receipt)
    return config, config_path, receipt, path


def test_handoff_refuses_live_legacy_dispatch_thread(tmp_path):
    config, path, receipt, _ = setup_handoff(tmp_path)
    proc = tmp_path / 'proc'
    legacy = proc / str(receipt['legacy_pid'])
    (legacy / 'task' / '2').mkdir(parents=True)
    # Simulate /proc stat with parentheses inside the process name.
    (legacy / 'stat').write_text('1 (python (legacy)) ' + ' '.join(['S'] + ['0']*18 + ['123']))
    with pytest.raises(ValueError, match='has not drained'):
        validate_handoff(config, path, receipt, proc_root=proc)
    (legacy / 'task' / '2').rmdir()
    validate_handoff(config, path, receipt, proc_root=proc)


@pytest.mark.parametrize('change', [
    {'legacy_dispatch_drained': False}, {'legacy_dispatch_thread_ids': []},
    {'legacy_restart_configuration_verified': False},
    {'legacy_start_ticks': True}, {'config_path': '/different/config'},
    {'runtime_root': '/different/runtime'},
])
def test_handoff_rejects_invalid_or_different_scope(tmp_path, change):
    config, path, receipt, _ = setup_handoff(tmp_path)
    with pytest.raises(ValueError):
        validate_handoff(config, path, receipt | change, proc_root=tmp_path / 'proc')


def test_handoff_requires_old_dispatch_disabled(tmp_path):
    config, path, receipt, _ = setup_handoff(tmp_path)
    config['device_day']['enabled'] = True
    with pytest.raises(ValueError, match='must be disabled'):
        validate_handoff(config, path, receipt, proc_root=tmp_path / 'proc')


def test_bridge_forwards_host_snapshot_even_when_host_dispatch_disabled(tmp_path):
    bridge = MonitorBridge(tmp_path)
    calls = []
    class Service:
        def observe(self, settings, inventory):
            calls.append((settings, inventory))
    snapshot = {'monitor': {'status': 'watching'}, 'recordings': [{'recording_id': 'new'}]}
    atomic_json(bridge.path, snapshot)
    settings = {'device_day': {'enabled': True}}
    assert bridge.poll(Service(), settings)
    assert not bridge.poll(Service(), settings)
    assert calls == [(settings, snapshot)]
    atomic_json(bridge.path, snapshot | {'monitor': {'status': 'retrying'}})
    assert not bridge.poll(Service(), settings)
    assert len(calls) == 1


def test_bridge_retries_uncommitted_snapshot(tmp_path):
    bridge = MonitorBridge(tmp_path)
    atomic_json(bridge.path, {'monitor': {'status': 'watching'}, 'recordings': []})
    class Service:
        calls = 0
        def observe(self, *_):
            self.calls += 1
            if self.calls == 1:
                raise OSError('local write unavailable')
    service = Service()
    with pytest.raises(OSError):
        bridge.poll(service, {})
    assert bridge.poll(service, {})
    assert service.calls == 2


class DrainingService:
    def __init__(self, settings_factory, gpu_lock):
        self.settings_factory = settings_factory
        self.wakeup = threading.Event()
        self.stop_event = threading.Event()
        self.entered = threading.Event()
        self.finish_job = threading.Event()
        self.thread = None
        self.last_result = {'status': 'starting'}

    def start(self):
        def work():
            self.entered.set()
            self.last_result = {'status': 'running'}
            while self.settings_factory()['device_day']['enabled']:
                time.sleep(.001)
            self.finish_job.wait(3)  # existing model work drains naturally
            self.last_result = {'status': 'completed'}
        self.thread = threading.Thread(target=work)
        self.thread.start()

    def observe(self, *_):
        pass


def create_companion(tmp_path):
    config, path, _, receipt_path = setup_handoff(tmp_path)
    companion = Companion(path, receipt_path, loader=lambda _: deepcopy(config),
                          service_factory=DrainingService)
    # The original host still owns its ancillary threads in these tests.
    companion.ancillaries.process_identity = lambda _: 123
    return config, companion


def test_dynamic_settings_preserve_camera_changes_and_refuse_root_change(tmp_path):
    config, companion = create_companion(tmp_path)
    config['collection_ingest']['camera_role_map'] = {'camera': 'first_person'}
    assert companion.settings()['collection_ingest']['camera_role_map'] == {'camera': 'first_person'}
    assert config['device_day']['enabled'] is False
    assert companion.settings()['device_day']['enabled'] is True
    config['storage']['archive_root'] = '/different/archive'
    assert companion.settings()['device_day']['enabled'] is False
    assert companion.draining.is_set()


def test_shutdown_finishes_existing_job_and_never_sets_cancellation(tmp_path, monkeypatch):
    _, companion = create_companion(tmp_path)
    closed = []
    monkeypatch.setattr('visioncortex.shared_inference.close_pools', lambda: closed.append(True))
    errors = []
    def run():
        try:
            companion.run(interval=.005)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    assert companion.service.entered.wait(3)
    companion.request_drain()
    time.sleep(.02)
    assert thread.is_alive()
    assert not companion.service.stop_event.is_set()
    assert not closed
    companion.service.finish_job.set()
    thread.join(3)
    assert not thread.is_alive()
    assert not errors
    assert closed == [True]
    status = json.loads((companion.root / 'state' / 'DeviceDayCompanionStatus.json').read_text())
    assert status['status'] == 'stopped'
    assert status['pipeline']['status'] == 'completed'


def test_companion_singleton_does_not_conflict_with_host_worker_lock(tmp_path, monkeypatch):
    from visioncortex.device_day import exclusive
    _, companion = create_companion(tmp_path)
    monkeypatch.setattr('visioncortex.shared_inference.close_pools', lambda: None)
    state = Path(companion.root) / 'state'
    with exclusive(state / 'Worker.lock'), exclusive(state / 'DeviceDayCompanion.lock'):
        with pytest.raises(BlockingIOError):
            companion.run(interval=.001)
        assert companion.service.thread is None


def test_companion_supports_enabled_ui_and_separate_disabled_host_config(tmp_path):
    config, path, receipt, receipt_path = setup_handoff(tmp_path)
    legacy = deepcopy(config)
    config['device_day']['enabled'] = True
    legacy_path = tmp_path / 'legacy.yaml'
    legacy_path.write_text('device_day: {enabled: false}')
    atomic_json(receipt_path, receipt | {'legacy_config_path': str(legacy_path)})
    companion = Companion(path, receipt_path, legacy_config_path=legacy_path,
        loader=lambda selected: deepcopy(legacy if selected == legacy_path else config),
        service_factory=DrainingService)
    assert companion.settings()['device_day']['enabled'] is True
    assert config['device_day']['enabled'] is True
    assert legacy['device_day']['enabled'] is False
    legacy['device_day']['enabled'] = True
    assert companion.settings()['device_day']['enabled'] is False
    assert companion.draining.is_set()


def test_ancillaries_start_only_after_original_host_exits(tmp_path, monkeypatch):
    config, _, receipt, _ = setup_handoff(tmp_path)
    config['device_day']['enabled'] = True
    config['collection_ingest']['camera_role_map'] = {'camera': 'first_person'}
    from visioncortex import device_day_file_index, device_day_time_lookup
    indexed, photographed = threading.Event(), threading.Event()
    class Index:
        def __init__(self, _):
            pass
        def tick(self):
            indexed.set()
    monkeypatch.setattr(device_day_file_index, 'FileIndexPublisher', Index)
    monkeypatch.setattr(device_day_time_lookup, 'refresh_photo_index', lambda *_: photographed.set())
    owner = AncillaryIndexes(receipt, lambda: config, process_identity=lambda _: 123)
    owner.ensure()
    assert not owner.threads
    owner.process_identity = lambda _: None
    try:
        owner.ensure()
        assert indexed.wait(3)
        assert photographed.wait(3)
        original_threads = dict(owner.threads)
        owner.ensure()
        assert owner.threads == original_threads
    finally:
        owner.drain()
