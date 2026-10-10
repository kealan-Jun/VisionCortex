from pathlib import Path
import threading
import time

import pytest

from test_device_day import device_config as _device_config
from test_device_day_recovery import failed_retention
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_recovery import RetentionRecovery, recover_one
from visioncortex import device_day_recovery_worker as worker
from visioncortex.input_availability import Availability, Reconciler
from visioncortex.runtime_control import ExecutionCancelled, execution_context


@pytest.fixture
def device_config(default_config, tmp_path):
    return _device_config.__wrapped__(default_config, tmp_path)


def test_verified_archive_restores_actual_input_gate_without_capture_write(device_config):
    runner, record, _, row = failed_retention(device_config)
    Availability(runner.runtime_root).mark(record, 'missing')
    assert recover_one(runner, row)['status'] == 'completed'
    availability = Availability(runner.runtime_root).states()[record['recording_id']]
    assert availability['state'] == 'ready'
    assert availability['signature'] == record['source_signature']
    assert availability['reason'] == 'verified_archived_original'
    assert not Path(record['video_path']).exists()
    with runner.queues['retention'].connect() as db:
        assert db.execute('SELECT input_status FROM recordings').fetchone()[0] == 'ready'
    Reconciler(device_config).tick()
    assert Availability(runner.runtime_root).states()[record['recording_id']]['state'] == 'ready'


def test_recovery_cooldown_survives_restart_without_persisted_content_trust(device_config, monkeypatch):
    runner, _, _, _ = failed_retention(device_config)
    calls = []
    monkeypatch.setattr('visioncortex.device_day_recovery.recover_one',
                        lambda *args: calls.append(1) or {'status': 'no_verified_archive'})
    assert RetentionRecovery().tick(runner)['status'] == 'no_verified_archive'
    assert RetentionRecovery().tick(runner)['status'] == 'idle'
    assert calls == [1]
    with runner.queues['retention'].connect() as db:
        db.execute('UPDATE recordings SET updated_at=updated_at+1')
    assert RetentionRecovery().tick(runner)['status'] == 'no_verified_archive'
    assert calls == [1, 1]


def test_out_of_scope_history_cannot_hide_future_candidates_in_bounded_batch(device_config, monkeypatch):
    runner, record, _, _ = failed_retention(device_config)
    runner.settings['process_since_us'] = record['recording_end_us'] + 1
    queue = runner.queues['retention']
    for i in range(1, 5):
        newer = record | {'recording_id': 'next' + str(i),
                          'recording_start_us': record['recording_start_us'] + i * 1000000,
                          'recording_end_us': record['recording_end_us'] + i * 1000000}
        queue.enqueue(newer, 'new')
        with queue.connect() as db:
            db.execute("UPDATE recordings SET input_status='missing' WHERE recording_id=?",
                       (newer['recording_id'],))
    # Set a cutoff which excludes the original and next1/next2.
    runner.settings['process_since_us'] = record['recording_end_us'] + 2500000
    calls = []
    monkeypatch.setattr('visioncortex.device_day_recovery.recover_one',
                        lambda _, row: calls.append(row['recording_id']) or {'status': 'no_verified_archive'})
    recovery = RetentionRecovery(batch_size=2)
    assert recovery.tick(runner)['status'] == 'idle'
    assert recovery.tick(runner)['status'] == 'no_verified_archive'
    assert calls == ['next3']


def test_cancelled_content_check_preserves_queue_receipt_and_input_gate(device_config, monkeypatch):
    runner, record, path, row = failed_retention(device_config)
    Availability(runner.runtime_root).mark(record, 'missing')
    before = read_json(path)
    stop = threading.Event()
    def interrupted(*args):
        stop.set()
        return True
    monkeypatch.setattr('visioncortex.device_day_inputs.verify_content', interrupted)
    with execution_context(stop=stop, yield_signal=stop):
        with pytest.raises(ExecutionCancelled):
            recover_one(runner, row)
    assert read_json(path) == before
    assert Availability(runner.runtime_root).states()[record['recording_id']]['state'] == 'missing'
    with runner.queues['retention'].connect() as db:
        assert db.execute('SELECT status FROM recordings').fetchone()[0] == 'failed'


def test_nonlocal_recovery_requires_mount_and_marker_before_any_io(tmp_path):
    mount = tmp_path / 'mounted'
    mount.mkdir()
    marker = mount / 'Marker.json'
    atomic_json(marker, {'id': 'expected', 'schema_version': 'storage/1', 'share': 'fixture'})
    config = {'runtime': {'local_only': False}, 'device_day': {'recovery': {'storage_checks': [{
        'mount_path': str(mount), 'filesystem': 'cifs', 'source': '//fixture/backup',
        'marker_path': str(marker), 'marker_id': 'expected', 'marker_id_field': 'id',
        'marker_fields': {'share': 'fixture', 'schema_version': 'storage/1'}}]}}}
    mountinfo = tmp_path / 'Mountinfo'
    mountinfo.write_text(f'10 1 0:1 / {mount} rw - cifs //fixture/wrong rw\n')
    with pytest.raises(ValueError, match='mount identity'):
        worker.verify_storage(config, mountinfo=mountinfo)
    mountinfo.write_text(f'10 1 0:1 / {mount} rw - cifs //fixture/backup rw\n')
    worker.verify_storage(config, mountinfo=mountinfo)
    mountinfo.write_text(f'10 1 0:1 / {mount} rw - cifs //fixture/backup rw\n'
                         f'11 1 0:2 / {mount} rw - cifs //fixture/wrong rw\n')
    with pytest.raises(ValueError, match='mount identity'):
        worker.verify_storage(config, mountinfo=mountinfo)
    mountinfo.write_text(f'10 1 0:1 / {mount} rw - cifs //fixture/backup rw\n')
    atomic_json(marker, {'id': 'different'})
    with pytest.raises(ValueError, match='volume identity'):
        worker.verify_storage(config, mountinfo=mountinfo)


def test_local_storage_is_explicit_and_modern_circuit_never_spawns_legacy_probe(monkeypatch, tmp_path):
    worker.verify_storage({'runtime': {'local_only': True}})
    with pytest.raises(ValueError, match='verified storage'):
        worker.verify_storage({'runtime': {'local_only': False}})
    monkeypatch.setattr('visioncortex.device_day_provider_gate.ProviderGate.probe_if_due',
                        lambda *_: pytest.fail('Modern transport owns half-open requests'))
    assert worker.provider_probe({'runtime': {'provider_circuit_enabled': True},
                                  'storage': {'local_runtime_root': str(tmp_path)}}) == {'status': 'transport_managed'}


def test_standalone_serve_actually_consumes_recovery_tick_and_publishes_result(device_config, monkeypatch):
    stop = threading.Event()
    device_config['runtime']['local_only'] = True
    monkeypatch.setattr('visioncortex.config.load_config', lambda *_: device_config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda config: config)
    monkeypatch.setattr(worker, 'provider_probe', lambda config: {'status': 'transport_managed'})
    calls = []
    def consumed(recovery, runner):
        calls.append(runner)
        stop.set()
        return {'status': 'completed', 'recording_id': 'fixture'}
    monkeypatch.setattr(RetentionRecovery, 'tick', consumed)
    worker.serve('fixture.yaml', stop)
    assert len(calls) == 1
    status = read_json(Path(device_config['storage']['local_runtime_root']) /
                       'device-day/RetentionRecovery/Service.json')
    assert status['status'] == 'stopped'
    assert status['completed_tick_count'] == status['tick_count'] == status['recovered_count'] == 1
    assert status['last_result'] == {'status': 'completed', 'recording_id': 'fixture'}
    assert status['last_tick_completed_at'] >= status['tick_started_at']


def test_bad_startup_exits_for_supervisor_without_exposing_config_failure(monkeypatch):
    def bad_config(*_):
        raise ValueError('private-config-value')
    monkeypatch.setattr('visioncortex.config.load_config', bad_config)
    with pytest.raises(RuntimeError, match='Archive recovery startup failed: ValueError') as failure:
        worker.serve('fixture.yaml', threading.Event())
    assert 'private-config-value' not in str(failure.value)


def test_recovery_owner_serializes_against_monolithic_recovery(device_config):
    from visioncortex.device_day import exclusive
    runner, _, _, _ = failed_retention(device_config)
    with exclusive(runner.runtime_root / 'locks' / 'retention-recovery-worker.lock'):
        assert RetentionRecovery().tick(runner) == {'status': 'running_elsewhere'}


def test_heartbeat_remains_fresh_without_inventing_completed_ticks(tmp_path):
    with worker.RecoveryStatus(tmp_path, interval=.01) as status:
        status.publish(phase='checking', tick_count=1)
        before = read_json(status.path)['updated_at']
        deadline = time.monotonic() + 2
        while True:
            current = read_json(status.path)
            assert current['phase'] == 'checking'
            assert current['tick_count'] == 1 and current['completed_tick_count'] == 0
            if current['updated_at'] > before:
                break
            assert time.monotonic() < deadline, 'Recovery heartbeat did not become fresh'
            time.sleep(.01)


def test_inactive_legacy_reason_never_triggers_paid_recovery_probe(device_config, monkeypatch):
    from visioncortex.device_day_provider_gate import ProviderGate
    device_config['runtime']['provider_circuit_enabled'] = False
    atomic_json(ProviderGate(device_config).path,
                {'active': False, 'reason': 'Arrearage', 'next_probe_at': 0})
    monkeypatch.setattr('visioncortex.ai_settings.verify_and_activate',
                        lambda *_: pytest.fail('Inactive legacy state must not trigger an API call'))
    assert worker.provider_probe(device_config) == {'status': 'available', 'next_probe_at': 0}


def test_modern_circuit_with_applicable_active_legacy_gate_keeps_original_recovery_owner(device_config, monkeypatch):
    from visioncortex.device_day_provider_gate import ProviderGate
    device_config['runtime']['provider_circuit_enabled'] = True
    device_config['mllm']['provider'] = 'aliyun'
    gate = ProviderGate(device_config)
    atomic_json(gate.path, {'active': True, 'reason': 'Arrearage', 'next_probe_at': 0})
    calls = []
    original = ProviderGate.probe_if_due
    def existing_helper(owner):
        calls.append('original-helper')
        return original(owner)
    monkeypatch.setattr(ProviderGate, 'probe_if_due', existing_helper)
    monkeypatch.setattr('visioncortex.ai_settings.verify_and_activate',
                        lambda *_: calls.append('connection-validation') or {'activated': True})
    assert worker.provider_probe(device_config)['status'] == 'available'
    assert calls == ['original-helper', 'connection-validation']
    assert not read_json(gate.path)['active']
    # An inactive legacy marker must not invent a modern probe protocol.
    assert worker.provider_probe(device_config) == {'status': 'transport_managed'}
    assert calls == ['original-helper', 'connection-validation']


def test_another_provider_legacy_marker_does_not_trigger_probe_when_modern_enabled(device_config, monkeypatch):
    from visioncortex.device_day_provider_gate import ProviderGate
    device_config['runtime']['provider_circuit_enabled'] = True
    device_config['mllm']['provider'] = 'other-provider'
    atomic_json(ProviderGate(device_config).path, {'active': True, 'reason': 'Arrearage', 'next_probe_at': 0})
    monkeypatch.setattr(ProviderGate, 'probe_if_due', lambda *_: pytest.fail('Another provider marker is inapplicable'))
    assert worker.provider_probe(device_config) == {'status': 'transport_managed'}


def test_public_recovery_unit_template_has_standalone_entry_and_shutdown_policy():
    from repo_paths import ROOT
    text = (ROOT / 'deployment/rtx3090ti-ubuntu/visioncortex-recovery.service').read_text()
    assert '-m visioncortex.device_day_recovery_worker --config' in text
    assert 'KillMode=mixed' in text and 'TimeoutStopSec=180' in text
    assert 'visioncortex.api' not in text


def test_rendered_recovery_unit_preserves_verified_virtualenv_interpreter_prefix(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from repo_paths import ROOT
    source = ROOT / 'deployment/rtx3090ti-ubuntu/render_service.py'
    spec = importlib.util.spec_from_file_location('recovery_service_renderer', source)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    project = tmp_path / 'Immutable Source'
    project.mkdir()
    python = tmp_path / 'verified-venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    config = tmp_path / 'Site.yaml'
    config.write_text('fixture: true\n')
    target = tmp_path / 'Recovery.service'
    monkeypatch.setattr(renderer, 'settings_for', lambda *_: {'device_day': {
        'recovery': {'storage_checks': [{'fixture': True}]}}})
    monkeypatch.setattr(sys, 'argv', [str(source), 'render-recovery', str(project), str(python),
                                    str(config), str(ROOT / 'deployment/rtx3090ti-ubuntu/visioncortex-recovery.service'),
                                    str(target)])
    renderer.main()
    text = target.read_text()
    assert f'ExecStart="{python}" -m visioncortex.device_day_recovery_worker' in text
    assert f'"{python.resolve()}" -m visioncortex.device_day_recovery_worker' not in text
    assert '@' not in text
