from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import subprocess
import threading
from types import SimpleNamespace

import pytest

from test_device_day_prerequisites import device_config as _device_config, prepared
from visioncortex import device_day_admission as admission
from visioncortex.device_day import stage_worker_capacity


@pytest.fixture
def device_config(default_config, tmp_path):
    return _device_config.__wrapped__(default_config, tmp_path)


def test_caps_bound_many_camera_lanes_without_removing_camera_fairness():
    settings = {'camera_lanes': True, 'vision_jobs_per_camera': 2,
                'stage_worker_limits': {'retention': 2, 'vision': 2, 'stt': 2,
                                        'understanding': 2, 'report': 1}}
    assert {stage: stage_worker_capacity(settings, stage, 12)
            for stage in settings['stage_worker_limits']} == settings['stage_worker_limits']
    assert stage_worker_capacity(settings, 'vision', 1) == 2
    assert stage_worker_capacity({'camera_lanes': True, 'vision_jobs_per_camera': 2}, 'vision', 12) == 24
    assert stage_worker_capacity({'vision_workers': 8, 'stage_worker_limits': {'vision': 2}}, 'vision', 12) == 2


@pytest.mark.parametrize('settings', [
    {'stage_worker_limits': {'unknown': 2}}, {'stage_worker_limits': {'vision': 0}},
    {'stage_worker_limits': {'vision': True}}, {'stage_worker_limits': []},
    {'admission_min_available_gib': -1}, {'admission_min_available_gib': float('nan')},
    {'admission_min_available_gib': float('inf')}, {'admission_min_available_gib': True},
])
def test_invalid_admission_config_rejected(settings):
    with pytest.raises(ValueError):
        admission.validate(settings)


def test_memory_reserve_blocks_only_new_heavy_stages_and_recovers(monkeypatch):
    config = {'device_day': {'admission_min_available_gib': 4}}
    memory = SimpleNamespace(available=3 * 1024**3)
    monkeypatch.setattr(admission.psutil, 'virtual_memory', lambda: memory)
    for stage in ['vision', 'understanding', 'report']:
        assert admission.admission_status(config, stage)['status'] == 'waiting_for_memory'
    for stage in ['stt', 'retention']:
        assert admission.admission_status(config, stage) is None
    memory.available = 4 * 1024**3
    assert admission.admission_status(config, 'vision') is None
    monkeypatch.setattr(admission.psutil, 'virtual_memory', lambda: pytest.fail('disabled guard read memory'))
    assert admission.admission_status({}, 'vision') is None


def test_queue_display_snapshot_single_flight_copy_and_forced_refresh(device_config, monkeypatch):
    runner, *_ = prepared(device_config)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def snapshot():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return {'counts': {'completed': len(calls)}}

    for queue in runner.queues.values():
        monkeypatch.setattr(queue, 'snapshot', snapshot)
    runner._queue_snapshot_until = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(runner.queue_snapshot) for _ in range(8)]
        assert entered.wait(5)
        release.set()
        results = [future.result(timeout=5) for future in futures]
    assert len(calls) == len(runner.queues)
    results[0]['vision']['counts']['completed'] = -1
    assert runner.queue_snapshot()['vision']['counts']['completed'] != -1
    runner.queue_snapshot(force=True)
    assert len(calls) == 2 * len(runner.queues)
    monkeypatch.setattr(runner, '_queue_snapshot_until', 0)
    runner.queue_snapshot()
    assert len(calls) == 3 * len(runner.queues)


def test_queue_completion_invalidates_display_cache_immediately(device_config):
    runner, _, _, inventory, _, _ = prepared(device_config)
    assert runner.queue_snapshot(force=True)['vision']['counts']['queued'] == 1
    result = runner.run_once(inventory, stage='vision')
    assert result['results'][0]['status'] == 'completed'
    assert result['queue']['vision']['counts']['completed'] == 1


def test_hot_pause_stops_new_dispatch_without_cancelling_inflight(device_config, monkeypatch):
    from visioncortex import device_day_service as module
    runner, _, _, inventory, _, _ = prepared(device_config)
    device_config['device_day'].update(paused_stages=['retention', 'stt', 'understanding', 'report'],
                                      stage_worker_limits={'vision': 1})
    started, release, reload_seen = (threading.Event() for _ in range(3))
    calls = []

    def run_once(_inventory, *, stage, stop_event, **kwargs):
        calls.append(stage)
        started.set()
        assert release.wait(5)
        assert not stop_event.is_set()
        return {'results': [{'status': 'completed'}]}

    monkeypatch.setattr(runner, 'run_once', run_once)
    monkeypatch.setattr(module, 'DeviceDayRunner', lambda settings: runner)
    monkeypatch.setattr('visioncortex.device_day_recovery.RetentionRecovery.tick', lambda *a: {'status': 'idle'})
    service = module.DeviceDayService(lambda: deepcopy(device_config), threading.Lock())
    old_wait = service.wakeup.wait

    def wait(timeout=None):
        if service.last_result.get('status') == 'waiting_for_config_reload':
            reload_seen.set()
        return old_wait(min(timeout or 0, .01))

    monkeypatch.setattr(service.wakeup, 'wait', wait)
    service.observe(device_config, inventory)
    service.start()
    try:
        assert started.wait(5)
        device_config['device_day']['paused_stages'].append('vision')
        service.wakeup.set()
        assert reload_seen.wait(5)
        assert calls == ['vision']
        assert service.last_result['admission_stopped']
        assert not service.stop_event.is_set()
    finally:
        release.set()
        # Disabling admission uses the same graceful path as companion drain.
        device_config['device_day']['enabled'] = False
        service.wakeup.set()
        service.thread.join(5)
        assert not service.thread.is_alive()


def test_service_memory_wait_does_not_claim_and_resumes_after_recovery(device_config, monkeypatch):
    from visioncortex import device_day_service as module
    runner, record, _, inventory, _, _ = prepared(device_config)
    device_config['device_day'].update(paused_stages=['retention', 'stt', 'understanding', 'report'],
                                      stage_worker_limits={'vision': 1}, admission_min_available_gib=4)
    memory = SimpleNamespace(available=3 * 1024**3)
    waiting, started = threading.Event(), threading.Event()
    monkeypatch.setattr(admission.psutil, 'virtual_memory', lambda: memory)
    monkeypatch.setattr(module, 'DeviceDayRunner', lambda settings: runner)
    monkeypatch.setattr('visioncortex.device_day_recovery.RetentionRecovery.tick', lambda *a: {'status': 'idle'})

    def run_once(*args, **kwargs):
        started.set()
        return {'results': []}

    monkeypatch.setattr(runner, 'run_once', run_once)
    service = module.DeviceDayService(lambda: deepcopy(device_config), threading.Lock())
    old_wait = service.wakeup.wait

    def wait(timeout=None):
        if service.last_result.get('memory_admission', {}).get('vision'):
            waiting.set()
        return old_wait(min(timeout or 0, .01))

    monkeypatch.setattr(service.wakeup, 'wait', wait)
    service.observe(device_config, inventory)
    service.start()
    try:
        assert waiting.wait(5)
        assert not started.is_set()
        with runner.queues['vision'].connect() as db:
            row = db.execute('SELECT status,attempts FROM recordings WHERE recording_id=?',
                             (record['recording_id'],)).fetchone()
        assert tuple(row) == ('queued', 0)
        memory.available = 5 * 1024**3
        service.wakeup.set()
        assert started.wait(5)
    finally:
        device_config['device_day']['enabled'] = False
        service.wakeup.set()
        service.thread.join(5)
        assert not service.thread.is_alive()


def test_scheduling_edits_preserve_execution_identity():
    from visioncortex import device_day
    from visioncortex.device_day_cache_identity import execution_identity
    before = subprocess.check_output(['git', 'show', 'HEAD:src/visioncortex/device_day.py'], text=True)
    after = __import__('pathlib').Path(device_day.__file__).read_text()
    assert execution_identity(before) == execution_identity(after)


def test_memory_settings_preserve_all_stage_cache_keys(device_config):
    from visioncortex.device_day import DeviceDayRunner
    runner, record, _, _, _, backend = prepared(device_config)
    before = {stage: runner._key(stage, record, {}) for stage in admission.STAGES}
    updated = deepcopy(device_config)
    updated['device_day'].update(
        stage_worker_limits={'retention': 2, 'vision': 2, 'stt': 2, 'understanding': 2, 'report': 1},
        admission_min_available_gib=4)
    replacement = DeviceDayRunner(updated, backend)
    assert before == {stage: replacement._key(stage, record, {}) for stage in admission.STAGES}
