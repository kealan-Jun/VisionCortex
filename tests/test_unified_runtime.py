"""Common runtime contracts using temporary files and fake functions only."""
from concurrent.futures import ThreadPoolExecutor
import errno
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from visioncortex.runtime_control import (
    ExecutionCancelled, ExecutionContext, ResourceCoordinator, execution_context,
)
from visioncortex.sqlite_store import connection


def test_sqlite_transaction_rolls_back_and_closes(tmp_path):
    path = tmp_path / 'db.sqlite3'
    with connection(path) as db:
        db.execute('CREATE TABLE example(value INTEGER)')
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute('SELECT 1')
    with pytest.raises(ValueError):
        with connection(path) as db:
            db.execute('INSERT INTO example VALUES(1)')
            raise ValueError('abort transaction')
    with connection(path, readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM example').fetchone()[0] == 0


def test_common_stage_allows_video_while_speech_waits_and_propagates_failure():
    from visioncortex.stage_execution import StageExecutor, required_stages
    entered, release = Event(), Event()
    events = []
    executor = StageExecutor({}, events.append)
    def speech():
        entered.set()
        assert release.wait(3)
        raise ValueError('speech source failed')
    future = executor.submit('stt', speech)
    try:
        assert entered.wait(3)
        assert executor.run('vision', lambda: ['existing_detector_result']) == ['existing_detector_result']
        assert not future.done()
        release.set()
        with pytest.raises(ValueError, match='speech source failed'):
            future.result(3)
    finally:
        release.set()
        executor.close()
    assert 'stt' not in required_stages('vision')
    assert set(required_stages('understanding')) == {'retention', 'vision', 'stt', 'understanding'}
    assert any(e['stage'] == 'vision' and e['status'] == 'completed' for e in events)
    assert any(e['stage'] == 'stt' and e['status'] == 'failed' for e in events)


def test_resource_capacity_is_shared_and_old_work_ages_ahead_of_live(tmp_path):
    a = ResourceCoordinator(tmp_path / 'resource.sqlite3')
    b = ResourceCoordinator(a.path)
    offline = ExecutionContext(job_id='offline', priority=1)
    live = ExecutionContext(job_id='camera', source='nas', priority=0)
    assert a.try_claim('holder', 'vision', 1, 1, live, now=0)
    assert not b.try_claim('older', 'vision', 1, 1, offline, now=1)
    a.release('holder')
    assert not a.try_claim('new', 'vision', 1, 1, live, now=70)
    assert b.try_claim('older', 'vision', 1, 1, offline, now=70)
    with pytest.raises(ValueError, match='Conflicting'):
        a.try_claim('wrong_capacity', 'vision', 1, 2, live, now=70)


def test_resource_wait_cancellation_releases_only_its_own_request(tmp_path):
    coordinator = ResourceCoordinator(tmp_path / 'resource.sqlite3')
    stop = Event()
    with coordinator.acquire('vision', capacity=1):
        with ThreadPoolExecutor(1) as pool:
            def waiter():
                with coordinator.acquire('vision', capacity=1, context=ExecutionContext(stop=stop)):
                    pytest.fail('Capacity exceeded')
            future = pool.submit(waiter)
            stop.set()
            with pytest.raises(ExecutionCancelled):
                future.result(3)
        assert len(coordinator.snapshot()) == 1
    assert coordinator.snapshot() == []


def test_cancelled_inference_waiter_does_not_poison_other_requests():
    from visioncortex.shared_inference import InferenceBroker
    started, release, closed = Event(), Event(), Event()
    class Scanner:
        batch_size = 1
        def infer(self, packets):
            if packets == [1]:
                started.set()
                assert release.wait(3)
            return packets
        def close(self):
            closed.set()
    broker = InferenceBroker(Scanner)
    stop = Event()
    try:
        with ThreadPoolExecutor(1) as pool:
            def caller():
                with execution_context(stop=stop):
                    return broker.submit([1])
            future = pool.submit(caller)
            assert started.wait(3)
            stop.set()
            with pytest.raises(ExecutionCancelled):
                future.result(3)
            release.set()
        assert broker.submit([2]) == [2]
    finally:
        release.set()
        assert broker.close()
    assert closed.is_set()


def test_inference_timeout_quarantines_stuck_worker():
    from visioncortex.shared_inference import InferenceBroker
    release = Event()
    class Slow:
        batch_size = 1
        def infer(self, packets):
            release.wait(3)
            return packets
    broker = InferenceBroker(Slow, timeout_seconds=.05)
    try:
        with pytest.raises(TimeoutError):
            broker.submit([1])
        with pytest.raises(RuntimeError, match='unavailable'):
            broker.submit([2])
    finally:
        release.set()
        assert broker.close()


def test_owned_subprocess_timeout_and_output(tmp_path):
    from visioncortex.owned_subprocess import run
    import subprocess
    assert run([sys.executable, '-c', 'print(42)'], timeout=3, capture_output=True, text=True).stdout.strip() == '42'
    with pytest.raises(subprocess.TimeoutExpired):
        run([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=.05, capture_output=True)


def test_catalog_migrates_json_once_and_updates_only_changed_rows(tmp_path):
    from visioncortex.observed_inventory import observe, read_inventory, revision
    legacy = tmp_path / 'observed-inventory.json'
    row = {'recording_id': 'one', 'camera_key': 'a_cam01', 'recording_start_us': 1789005600000000}
    original = json.dumps({'recordings': [row], 'errors': [{'path': 'missing'}]})
    legacy.write_text(original)
    observe(tmp_path, {'recordings': [row | {'recording_id': 'two'}]})
    assert len(read_inventory(tmp_path)['recordings']) == 2
    before = revision(tmp_path)
    assert not observe(tmp_path, {'recordings': [row]})
    assert revision(tmp_path) == before
    assert legacy.read_text() == original
    assert read_inventory(tmp_path)['errors'] == [{'path': 'missing'}]


def test_receipt_projection_updates_one_record_without_listing_other_receipts(tmp_path, monkeypatch):
    from visioncortex.receipt_projection import ReceiptProjection
    root = tmp_path / 'receipts'
    for identifier in ('a', 'b'):
        (root / identifier).mkdir(parents=True)
        (root / identifier / 'retention.json').write_text(json.dumps({'status': 'completed', 'id': identifier}))
    layout = SimpleNamespace(name='day', receipts=root)
    projection = ReceiptProjection(tmp_path / 'runtime')
    assert len(projection.records(layout, ['retention'])) == 2
    (root / 'a' / 'retention.json').write_text(json.dumps({'status': 'new'}))
    monkeypatch.setattr(Path, 'glob', lambda *args: pytest.fail('Unexpected full receipt listing'))
    rows = dict(projection.records(layout, ['retention'], recording_id='a'))
    assert rows['a']['retention']['status'] == 'new'
    assert rows['b']['retention']['id'] == 'b'


def test_publication_ack_cannot_erase_a_newer_write(tmp_path):
    from visioncortex.publication_journal import PublicationJournal
    journal = PublicationJournal(tmp_path)
    old = journal.begin({'recording_id': 'slice', 'version': 1})
    new = journal.begin({'recording_id': 'slice', 'version': 2})
    journal.complete('slice', old)
    assert journal.pending() == [(new, {'recording_id': 'slice', 'version': 2})]
    journal.complete('slice', new)
    assert not journal.pending()


def test_provider_account_block_is_scoped_and_half_open_is_single_owner(tmp_path):
    from visioncortex.provider_control import Circuit, ProviderUnavailable
    binding = {'provider': 'aliyun', 'base_url': 'https://example.invalid', 'credential_ref': 'connection-a'}
    vision = Circuit(tmp_path, binding, 'understanding')
    speech = Circuit(tmp_path, binding, 'stt')
    other = Circuit(tmp_path, binding | {'credential_ref': 'connection-b'}, 'understanding')
    assert vision.failed(RuntimeError('Arrearage'), now=0)
    with pytest.raises(ProviderUnavailable):
        speech.admit(now=1)
    assert not other.admit(now=1)
    probes = vision.admit(now=901)
    with pytest.raises(ProviderUnavailable):
        speech.admit(now=902)
    vision.succeeded(probes)
    assert not speech.admit(now=903)


def test_provider_rate_limit_does_not_block_another_service(tmp_path):
    from visioncortex.provider_control import Circuit, ProviderUnavailable
    error = RuntimeError('rate')
    error.response = SimpleNamespace(status_code=429)
    first = Circuit(tmp_path, {}, 'understanding')
    first.failed(error, now=0)
    with pytest.raises(ProviderUnavailable):
        first.admit(now=1)
    assert not Circuit(tmp_path, {}, 'stt').admit(now=1)
    assert not first.failed(ValueError('invalid model answer'), now=1)


def test_late_provider_probe_success_cannot_clear_a_new_failure(tmp_path):
    from visioncortex.provider_control import Circuit, ProviderUnavailable
    circuit = Circuit(tmp_path, {}, 'understanding')
    circuit.failed(RuntimeError('Arrearage'), now=0)
    probe = circuit.admit(now=901)
    circuit.failed(RuntimeError('Arrearage'), now=902)
    circuit.succeeded(probe)
    with pytest.raises(ProviderUnavailable):
        circuit.admit(now=903)


def test_timeline_dirty_ranges_survive_late_arrivals_and_idle_does_nothing(tmp_path):
    from visioncortex.timeline_invalidation import TimelineInvalidations
    queue = TimelineInvalidations(tmp_path)
    assert queue.changed('2026-09-09_a', 'one', 0, 10)
    old = queue.pending()[0]
    assert not queue.changed('2026-09-09_a', 'one', 0, 10)
    assert queue.changed('2026-09-09_b', 'two', 5, 20)
    queue.complete(old['day'], old['token'])
    latest = queue.pending()[0]
    assert (latest['start_us'], latest['end_us']) == (0, 20)
    queue.complete(latest['day'], latest['token'])
    assert queue.pending() == []


def test_resource_lease_is_visible_to_another_python_process(tmp_path):
    import os
    import subprocess
    database = tmp_path / 'shared.sqlite3'
    coordinator = ResourceCoordinator(database)
    script = ('from pathlib import Path; from visioncortex.runtime_control import ResourceCoordinator,ExecutionContext; '
              'import sys; db=ResourceCoordinator(Path(sys.argv[1])); '
              'print(db.try_claim("child", "vision", 1, 1, ExecutionContext()))')
    with coordinator.acquire('vision', capacity=1):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'))
        child = subprocess.run([sys.executable, '-c', script, str(database)], env=env,
                               capture_output=True, text=True, check=True, timeout=10)
    assert child.stdout.strip() == 'False'
    coordinator.release('child')


def test_typed_config_and_local_capability_boundary(tmp_path):
    from visioncortex.runtime_options import validate
    with pytest.raises(ValueError):
        validate({'runtime': {'resource_limits': {'vision': True}}})
    with pytest.raises(ValueError):
        validate({'runtime': {'role': 'unbounded'}})
    config = {'runtime': {'local_only': True}, 'storage': {'local_runtime_root': str(tmp_path), 'archive_root': str(tmp_path / 'archive')}}
    assert validate(config).local_only
    with pytest.raises(ValueError, match='NAS'):
        validate(config | {'device_day': {'enabled': True}})
    with pytest.raises(ValueError, match='under'):
        validate(config | {'storage': config['storage'] | {'archive_root': str(tmp_path.parent / 'outside')}})
    with pytest.raises(ValueError, match='under'):
        validate(config | {'storage': config['storage'] | {'archive_root': str(tmp_path / '..' / 'outside')}})


def test_web_role_never_starts_workers(tmp_path, monkeypatch):
    from visioncortex import api
    from fastapi.testclient import TestClient
    monkeypatch.delenv('VISIONCORTEX_STORAGE_MAINTENANCE', raising=False)
    monkeypatch.setenv('VISIONCORTEX_RUNTIME_ROLE', 'web')
    monkeypatch.setattr(api, 'validate_web_access_configuration', lambda: None)
    monkeypatch.setattr(api, '_settings', lambda: {'storage': {'local_runtime_root': str(tmp_path)}})
    monkeypatch.setattr(api, '_initialize_persistent_queue', lambda _: None)
    for name in ('_expire_stale_upload_sessions', '_recover_jobs_from_archive_receipts', '_recover_orphaned_tasks',
                 '_start_queue_worker', '_start_nas_monitor'):
        monkeypatch.setattr(api, name, lambda *args: pytest.fail('Web process started a worker'))
    with TestClient(api.app) as client:
        assert client.get('/health/live').status_code == 200


def test_worker_has_exclusive_local_ownership(tmp_path):
    from visioncortex.runtime_process import worker_owner, worker_status
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    with worker_owner(config):
        with pytest.raises(BlockingIOError):
            with worker_owner(config):
                pass
    assert worker_status(config)['status'] == 'stopped'


def test_run_record_patches_preserve_other_process_updates(tmp_path):
    from visioncortex.run_queue import DurableRunQueue
    a = DurableRunQueue(tmp_path / 'runs.sqlite3')
    b = DurableRunQueue(a.database)
    a.patch_run('run', {'state': 'queued'})
    b.patch_run('run', {'stage': 'vision'})
    a.patch_run('run', {'state': 'running'})
    assert b.load_run('run') == {'state': 'running', 'stage': 'vision'}


def test_storage_failures_and_time_mapping_preserve_meaning(tmp_path):
    from visioncortex.artifact_reader import resolve, storage_status
    from visioncortex.media_time import capture_us, media_ms
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path, 'missing.mp4')
    assert storage_status(FileNotFoundError())[0] == 404
    assert storage_status(OSError(errno.EIO, 'I/O error'))[0] == 503
    assert storage_status(PermissionError())[0] == 403
    clock = {'basis': 'recorder_csv_interpolation', 'origin_us': 1_000_000, 'points': [[0, 1_000_000], [1000, 2_010_000]]}
    assert media_ms(clock, capture_us(clock, 500), clock['origin_us'])[0] == 500
    assert media_ms({}, 1_001_000, 1_000_000)[1].endswith('estimate')


def test_cache_impact_excludes_control_and_propagates_source_changes():
    from visioncortex.stage_dependencies import impact
    assert impact(['runtime_control.py'])['affected_stages'] == []
    assert impact(['actions.py'])['affected_stages'] == ['vision', 'understanding', 'report']
    assert impact(['speech_qwen.py'])['affected_stages'] == ['stt', 'understanding', 'report']
    assert 'retention' in impact(['new_unknown_cv_module.py'])['affected_stages']


@pytest.mark.parametrize('commit_field', ['commit_sha', 'development_sha'])
def test_release_seal_switch_rollback_and_corruption(tmp_path, commit_field):
    source = Path(__file__).parents[1] / 'deployment/runtime/Release.py'
    spec = importlib.util.spec_from_file_location('release_tools', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for version in ('one', 'two'):
        root = tmp_path / 'Releases' / version
        root.mkdir(parents=True)
        (root / 'requirements.lock').write_text('pinned-placeholder')
        (root / 'Acceptance.json').write_text(json.dumps({commit_field: 'a'*40, 'release_ready': True}))
        (root / 'Application.txt').write_text(version)
        manifest = module.seal(root, 'a'*40, version)
        with pytest.raises(ValueError, match='Install'):
            module.activate(tmp_path, version)
        environment = tmp_path / 'Environments' / manifest['source_digest']
        python = environment / 'bin/python'
        python.parent.mkdir(parents=True)
        python.write_text('test-only interpreter placeholder')
        (environment / 'Installation.json').write_text(json.dumps({
            'source_digest': manifest['source_digest'], 'commit': manifest['commit'],
            'dependency_lock_sha256': manifest['lock_digest'], 'python': str(python)}))
    assert module.activate(tmp_path, 'one')['version'] == 'one'
    assert module.activate(tmp_path, 'two')['version'] == 'two'
    assert module.activate(tmp_path, rollback=True)['version'] == 'one'
    (tmp_path / 'Releases/two/Application.txt').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        module.activate(tmp_path, 'two')
    assert (tmp_path / 'Current').resolve().name == 'one'
    with module.deployment_lock(tmp_path):
        with pytest.raises(OSError):
            module.activate(tmp_path, 'one')
    assert module.accepted_commit({'commit_sha': 'a'*40, 'development_sha': 'a'*40}) == 'a'*40
    with pytest.raises(ValueError, match='conflicting'):
        module.accepted_commit({'commit_sha': 'a'*40, 'development_sha': 'b'*40})


def test_day_context_snapshot_matches_per_recording_reader(tmp_path):
    from visioncortex.device_day import load_context, load_day_context, recording_context
    from visioncortex.device_day_contract import DeviceDayLayout
    layout = DeviceDayLayout(tmp_path / 'archive', 'camera_cam01', 1789005600000000,
                             backend_root=tmp_path / 'cache')
    layout.comments.mkdir(parents=True)
    (layout.comments / 'Comment.jsonl').write_text('\n'.join(json.dumps(row) for row in [
        {'start_us': 0, 'end_us': 20, 'text': 'first'},
        {'start_us': 20, 'end_us': 30, 'text': 'next'}]))
    (layout.comments / 'Protocol.json').write_text('{"name":"protocol"}')
    record = {'recording_id': 'slice', 'recording_start_us': 10, 'recording_end_us': 20}
    stt = {'status': 'completed', 'comments': [{'text': 'speech'}]}
    receipt = layout.receipts / 'slice/stt.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(stt))
    snapshot = load_day_context(layout)
    result = recording_context(snapshot, record, stt)
    assert result == load_context(layout, record)
    assert [item['text'] for item in result['comments']] == ['first', 'speech']
    assert len(snapshot['comments']) == 2


def test_publication_replay_cannot_acknowledge_active_or_unavailable_record(tmp_path):
    from visioncortex.device_day import exclusive
    from visioncortex.publication_journal import PublicationJournal, reconcile
    journal = PublicationJournal(tmp_path)
    for identifier in ('active', 'unavailable', 'ready'):
        journal.begin({'recording_id': identifier})
    published = []
    def refresh(layout, *, recording_id):
        if recording_id == 'unavailable':
            raise OSError('test-only unavailable storage')
        published.append(recording_id)
        return True
    runner = SimpleNamespace(runtime_root=tmp_path, layout=lambda record: record,
                             refresh_index=refresh)
    with exclusive(tmp_path / 'locks/active.vision.lock'):
        assert reconcile(runner) == 1
    assert published == ['ready']
    assert {r['recording_id'] for _, r in journal.pending()} == {'active', 'unavailable'}
    assert reconcile(runner) == 1
    assert published == ['ready', 'active']
    assert [r['recording_id'] for _, r in journal.pending()] == ['unavailable']


@pytest.mark.parametrize('cancel', [False, True])
def test_worker_reads_cross_process_retry_and_retains_shutdown_checkpoint(tmp_path, monkeypatch, cancel):
    from visioncortex import api
    from visioncortex.run_queue import DurableRunQueue
    store = DurableRunQueue(tmp_path / 'queue.sqlite3')
    store.save_run('retry', {'state': 'queued'})
    store.enqueue('retry', 'run', {'sealed_input': 'unchanged'})
    monkeypatch.setattr(api, '_runs', {'retry': {'state': 'failed'}})
    monkeypatch.setattr(api, '_persistent_queue', store)
    stop = Event()
    monkeypatch.setattr(api, '_queue_stop', stop)
    invoked = []
    def dispatch(job):
        invoked.append(job.run_id)
        stop.set()
        if cancel:
            raise ExecutionCancelled('stopping')
        store.patch_run(job.run_id, {'state': 'completed'})
    monkeypatch.setattr(api, '_dispatch_persisted_job', dispatch)
    api._queue_worker_loop(store)
    assert invoked == ['retry']
    assert store.get_job('retry')['status'] == ('queued' if cancel else 'completed')
    assert store.load_run('retry')['state'] == ('queued' if cancel else 'completed')
    assert store.get_job('retry')['payload'] == {'sealed_input': 'unchanged'}
