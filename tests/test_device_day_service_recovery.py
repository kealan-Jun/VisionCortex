import threading

import pytest

from test_device_day_prerequisites import device_config as _device_config, prepared


@pytest.fixture
def device_config(default_config, tmp_path):
    return _device_config.__wrapped__(default_config, tmp_path)


@pytest.mark.parametrize('recovery_status,stage,old_result_running', [
    ('prerequisite_restored', 'vision', True), ('retention_repair_queued', 'retention', True),
    ('prerequisite_restored', 'understanding', False),
])
def test_recovery_wakeup_survives_older_empty_stage_result(
        device_config, monkeypatch, recovery_status, stage, old_result_running):
    from visioncortex import device_day_service as module
    device_config['device_day']['paused_stages'] = [s for s in ('stt', 'understanding', 'report') if s != stage]
    runner, record, _, inventory, _, _ = prepared(device_config)
    started, release_old_result, resumed, sent = (threading.Event() for _ in range(4))
    idle_result_seen = threading.Event()
    generations = []
    initial = runner._prerequisite_generation[stage]

    def run_once(_inventory, *, stage: str, **kwargs):
        if stage == target_stage:
            generations.append(runner._prerequisite_generation[stage])
            if len(generations) == 1:
                started.set()
                if old_result_running:
                    assert release_old_result.wait(5)
            else:
                resumed.set()
        return {'results': []}

    target_stage = stage
    monkeypatch.setattr(runner, 'run_once', run_once)
    monkeypatch.setattr(module, 'DeviceDayRunner', lambda settings: runner)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 10**12})())

    def recover(_self, _runner):
        if sent.is_set():
            return {'status': 'idle'}
        assert (started if old_result_running else idle_result_seen).wait(5)
        sent.set()
        return {'status': recovery_status, 'recording_id': record['recording_id']}

    monkeypatch.setattr('visioncortex.device_day_recovery.RetentionRecovery.tick', recover)
    service = module.DeviceDayService(lambda: device_config, threading.Lock())
    original_wait = service.wakeup.wait

    def dispatch_turn(timeout=None):
        if not old_result_running and 'results' in service.last_result.get('stages', {}).get(stage, {}):
            idle_result_seen.set()
        # Release the stale empty result only AFTER the real callback advances
        # its generation. This forces the ordering that used to lose wakeups.
        if runner._prerequisite_generation[stage] > initial:
            release_old_result.set()
        return original_wait(min(timeout or 0, .01))

    monkeypatch.setattr(service.wakeup, 'wait', dispatch_turn)
    service.observe(device_config, inventory)
    service.start()
    try:
        assert resumed.wait(5), service.last_result
        assert generations[0] == initial
        assert generations[1] > initial
    finally:
        release_old_result.set()
        service.stop()
