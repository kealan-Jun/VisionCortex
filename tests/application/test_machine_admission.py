import json
import os
import threading

import pytest

from visioncortex import machine_admission
from visioncortex.application import runtime_host
from visioncortex.run_queue import DurableRunQueue, QueuedRunJob
from visioncortex.runtime_control import ExecutionContext, ResourceCoordinator
from visioncortex.runtime_options import validate
from repo_paths import ROOT


def configured(tmp_path):
    native = tmp_path / 'native'
    coordinator = ResourceCoordinator(native / 'state/resources.sqlite3')
    with coordinator.acquire('vision', capacity=4):
        pass
    config = {'storage': {'local_runtime_root': str(tmp_path / 'machine')},
              'runtime': {'resource_coordinator_root': str(native), 'resource_limits': {'vision': 4},
                          'machine_admission': {'gpu_uuid': 'GPU-test'}}}
    return config, coordinator


def test_memory_and_native_capacity_gate_fail_closed(tmp_path):
    config, native = configured(tmp_path)
    def sample(_gpu):
        return 24564, 2000
    assert machine_admission.status(config, gpu_sample=sample, available_ram=16000)['ready']
    assert machine_admission.status(config, gpu_sample=lambda _: (24564, 12000))['reason'] == 'WAITING_GPU_MEMORY'
    assert machine_admission.status(config, gpu_sample=sample, available_ram=100)['reason'] == 'WAITING_HOST_MEMORY'
    assert native.try_claim('native', 'vision', 4, 4, ExecutionContext(source='nas'))
    assert machine_admission.status(config, gpu_sample=sample)['reason'] == 'WAITING_NATIVE_VISION'
    native.release('native')
    config['runtime']['resource_limits']['vision'] = 12
    assert machine_admission.status(config, gpu_sample=sample)['reason'] == 'RESOURCE_CAPACITY_MISMATCH'


def test_missing_native_ledger_is_not_created(tmp_path):
    config, _ = configured(tmp_path)
    config['runtime']['resource_coordinator_root'] = str(tmp_path / 'missing')
    assert machine_admission.status(config)['reason'] == 'ADMISSION_UNAVAILABLE'
    assert not (tmp_path / 'missing').exists()


def test_scheduling_caps_preserve_sealed_model_provider_and_input_settings(tmp_path):
    config, _ = configured(tmp_path)
    sealed = {'models': {'first_person': '/models/pinned.pt'}, 'storage': {'archive_root': '/outputs'},
              'mllm': {'provider': 'selected-provider', 'model': 'pinned-model', 'workers': 8},
              'performance': {'batch_size': 16, 'engine_batch_size': 4, 'concurrent_role_scanners': True}}
    result = machine_admission.execution_settings(sealed, config['runtime'])
    assert result['models'] == sealed['models']
    assert result['storage'] == sealed['storage']
    assert result['mllm']['provider'] == 'selected-provider'
    assert result['mllm']['model'] == 'pinned-model'
    assert result['mllm']['workers'] == 1
    assert result['performance']['batch_size'] == 4
    assert result['performance']['engine_batch_size'] == 4
    assert result['performance']['concurrent_role_scanners'] is False
    assert sealed['performance']['batch_size'] == 16


def test_oom_hold_only_pauses_machine_claims(tmp_path):
    config, native = configured(tmp_path)
    machine_admission.hold_after_failure(config, 'RuntimeError: CUDA out of memory')
    assert machine_admission.status(config)['reason'] == 'MACHINE_ADMISSION_HOLD'
    assert native.try_claim('still-running-nas', 'vision', 1, 4, ExecutionContext(source='nas'))
    hold = tmp_path / 'machine/state/MachineAdmissionHold.json'
    assert json.loads(hold.read_text())['reason'] == 'GPU_OUT_OF_MEMORY'


def test_execution_uses_only_explicitly_reverified_same_connection(monkeypatch, tmp_path):
    config, _ = configured(tmp_path)
    from visioncortex import ai_settings
    sealed = {'mllm': {'credential_ref': 'original', 'provider': 'aliyun', 'model': 'pinned'}}
    calls = []
    def refreshed(settings):
        calls.append(settings['mllm'].copy())
        return settings['mllm'] | {'credential_ref': 'same-key-new-verification'}
    monkeypatch.setattr(ai_settings, 'reverified_job_mllm', refreshed)
    result = machine_admission.execution_settings(sealed, config['runtime'])
    assert calls == [sealed['mllm']]
    assert result['mllm']['model'] == 'pinned'
    assert sealed['mllm']['credential_ref'] == 'original'
    def refuse(_settings):
        raise ValueError('connection or key differs')
    monkeypatch.setattr(ai_settings, 'reverified_job_mllm', refuse)
    with pytest.raises(ValueError, match='connection or key differs'):
        machine_admission.execution_settings(sealed, config['runtime'])


def test_admission_preserves_queued_and_historical_failed_tasks(monkeypatch, tmp_path):
    config, _ = configured(tmp_path)
    store = DurableRunQueue(tmp_path / 'queue.sqlite3')
    store.enqueue('historical', 'run', {})
    store.claim_next('old-worker', lease_seconds=60)
    store.finish('historical', 'old-worker', 'failed', error='original failure')
    store.enqueue('new-task', 'run', {})
    stop = threading.Event()
    def refuse(_settings):
        stop.set()
        return False
    monkeypatch.setattr(runtime_host, '_settings', lambda: config)
    monkeypatch.setattr(runtime_host, '_queue_stop', stop)
    monkeypatch.setattr(machine_admission, 'may_claim', refuse)
    runtime_host._queue_worker_loop(store)
    assert store.get_job('new-task')['status'] == 'queued'
    assert store.get_job('new-task')['attempts'] == 0
    assert store.get_job('historical')['last_error'] == 'original failure'


def test_machine_policy_requires_shared_coordinator(tmp_path):
    with pytest.raises(ValueError, match='shared coordinator'):
        validate({'runtime': {'machine_admission': {'gpu_uuid': 'GPU-test'}}})


def test_execution_receipt_keeps_each_attempt(tmp_path):
    config, _ = configured(tmp_path)
    payload = {'settings': {'mllm': {}}, 'manifest': {'experiment_id': 'test'},
               'nas_root': str(tmp_path / 'staging')}
    first = QueuedRunJob('owned-run', 'run', payload, 1, 1, False)
    second = QueuedRunJob('owned-run', 'run', payload, 1, 2, True)
    machine_admission.record_execution(first, config)
    machine_admission.record_execution(second, config)
    directory = tmp_path / 'staging/JSON-Config-Files'
    old = json.loads((directory / 'MachineExecutionAttempt1.json').read_text())
    current = json.loads((directory / 'MachineExecution.json').read_text())
    assert old['attempt'] == 1
    assert current['attempt'] == 2
    assert current['reclaimed'] is True
    assert current['manifest_sha256'] == old['manifest_sha256']
    assert current['run_id'] == old['run_id'] == 'owned-run'


def test_unreadable_published_release_cannot_complete(monkeypatch, tmp_path):
    config, _ = configured(tmp_path)
    root = tmp_path / 'published'
    settings = config | {'storage': config['storage'] | {
        'formal_promotion_required': True, 'formal_fixed_root': str(root),
        'formal_history_root': str(tmp_path / 'history')}}
    class Pipeline:
        def __init__(self, *_args):
            pass
        def run(self, _manifest):
            return tmp_path / 'staging'
    states = []
    monkeypatch.setattr(runtime_host.ports, 'EvidencePipeline', Pipeline)
    monkeypatch.setattr(runtime_host.ports, 'promote_fixed_archive', lambda *_args: {})
    monkeypatch.setattr(runtime_host, '_record_partial_completion', lambda *_args: False)
    monkeypatch.setattr(runtime_host, '_update_queue_recovery_state', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime_host, '_update', lambda _run, **values: states.append(values))
    from visioncortex import storage
    monkeypatch.setattr(storage, 'verify_current_release', lambda _root: {'passed': False, 'failures': ['missing_evidence']})
    runtime_host._execute_now('machine-test', None, settings, tmp_path / 'staging')
    assert states[-1]['state'] == 'failed'
    assert 'missing_evidence' in states[-1]['error']
    assert not any(state.get('state') == 'completed' for state in states)


def test_concurrent_guard_keeps_ocr_shared_lock(monkeypatch, tmp_path):
    import fcntl
    import importlib.util
    from visioncortex.machine_worker import validate_gpu_guard
    gpu = 'GPU-test'
    lock = tmp_path / (gpu + '.lock')
    (tmp_path / (gpu + '.mode')).write_text('concurrent')
    ocr = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(ocr, fcntl.LOCK_SH | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('shared_guard', ROOT / 'deployment/inference/gpu_guard.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    class Completed:
        pid = 999999
        returncode = 0
        def poll(self):
            return 0
    def spawn(_command, *, env, **_kwargs):
        with monkeypatch.context() as child:
            for key, value in env.items():
                if key.startswith('VISIONCORTEX_GPU_'):
                    child.setenv(key, value)
            validate_gpu_guard()
        return Completed()
    def missing_group(*_args):
        raise ProcessLookupError
    monkeypatch.setattr(guard.subprocess, 'Popen', spawn)
    monkeypatch.setattr(guard.os, 'killpg', missing_group)
    try:
        assert guard.run_guard(['owned-test'], gpu, tmp_path,
                               check_gpu=lambda _: pytest.fail('must not demand an idle GPU'),
                               check_shared_gpu=lambda _: None) == 0
        contender = os.open(lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    finally:
        os.close(ocr)
