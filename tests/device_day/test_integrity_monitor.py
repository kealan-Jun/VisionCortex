"""Owned fixtures exercise continuous recovery without a real model or NAS."""
import json
import os
from pathlib import Path
import time

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_inplace import enqueue
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.input_availability import IntegrityMonitor, integrity_status
from visioncortex.receipt_projection import artifact_manifest
from visioncortex.sqlite_store import connection


@pytest.fixture
def monitor_case(device_config):  # noqa: F811
    config = device_config
    config['runtime']['local_only'] = True
    config['device_day'].update(inplace_preprocessing=True,
                               input_integrity={'enabled': True, 'verification_interval_seconds': 60, 'retry_seconds': 30})
    capture(config)
    record, layout = item_and_layout(config)
    backend = FakeModels()
    runner = DeviceDayRunner(config, backend)
    assert runner.process(record, stage='retention')['status'] == 'completed'
    queue = runner.queues['retention']
    enqueue(runner, record, 'retention')
    claimed = queue.claim('fixture')
    assert claimed
    queue.finish('fixture', record['recording_id'], {'status': 'completed'}, 1)
    return config, record, layout, backend, runner, IntegrityMonitor(config, runner)


def tick_selected(monitor):
    for _ in range(8):
        result = monitor.tick()
        if result['status'] != 'idle':
            return result
    raise AssertionError('No candidate selected')


def due(monitor):
    with connection(monitor.path) as db:
        db.execute('UPDATE checks SET next_check=0')
        db.execute('UPDATE cursors SET position=0')


def test_missing_managed_archive_is_restored_and_not_repeated_after_restart(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    saved = read_json(runner._receipt(layout, record, 'retention'))
    source = saved['sources'][0]
    target = layout.root / source['retained']['path']
    original = Path(source['original_path']).read_bytes()
    target.unlink()  # Owned test artifact; never a production capture.
    result = tick_selected(monitor)
    assert result['status'] == 'restored' and result['restored_files'] == 1
    assert target.read_bytes() == original == Path(source['original_path']).read_bytes()
    assert backend.vision_calls == backend.semantic_calls == 0
    assert runner.queues['retention'].snapshot()['counts']['completed'] == 1
    restarted = IntegrityMonitor(config, runner)
    due(restarted)
    assert tick_selected(restarted)['status'] == 'verified'
    with connection(monitor.path, readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE transition='recovered'").fetchone()[0] == 1


def test_archive_conflict_is_preserved_alerts_deduplicate_and_resolve(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    saved = read_json(runner._receipt(layout, record, 'retention'))
    target = layout.root / saved['artifacts'][0]['path']
    original = target.read_bytes()
    target.write_bytes(b'corrupt owned artifact')
    assert tick_selected(monitor)['reason'] == 'archive_content_conflict'
    assert target.read_bytes() == b'corrupt owned artifact'
    due(monitor)
    assert tick_selected(IntegrityMonitor(config, runner))['reason'] == 'archive_content_conflict'
    with connection(monitor.path, readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE transition='raised'").fetchone()[0] == 1
        assert db.execute('SELECT occurrences FROM alerts').fetchone()[0] == 2
    target.write_bytes(original)
    due(monitor)
    assert tick_selected(monitor)['status'] == 'verified'
    state = integrity_status(config)
    assert state['active_alert_count'] == 0
    assert any(e['transition'] == 'resolved' for e in state['events'])


def test_completed_new_recipe_closes_both_alerts_after_full_artifact_verification(monitor_case):
    config, record, _, backend, runner, monitor = monitor_case
    assert runner.process(record, stage='vision')['status'] == 'completed'
    enqueue(runner, record, 'vision')
    queue = runner.queues['vision']
    assert queue.claim('fixture')
    queue.finish('fixture', record['recording_id'], {'status': 'completed'}, 1)
    monitor._alert('vision_failed', record['recording_id'], 'execution_recipe_changed')
    monitor._alert('vision_recovery', record['recording_id'], 'recovery_attempt_failed')
    calls = backend.vision_calls
    assert monitor.tick()['status'] == 'verified'
    assert integrity_status(config)['active_alert_count'] == 0
    assert backend.vision_calls == calls


def test_completed_queue_does_not_clear_alert_when_owned_archive_is_corrupt(monitor_case):
    config, record, layout, _, runner, monitor = monitor_case
    monitor._alert('retention_recovery', record['recording_id'], 'recovery_attempt_failed')
    saved = read_json(runner._receipt(layout, record, 'retention'))
    (layout.root / saved['artifacts'][0]['path']).write_bytes(b'corrupt owned fixture')
    assert monitor.tick()['reason'] == 'archive_content_conflict'
    assert any(a['reason'] == 'recovery_attempt_failed' for a in integrity_status(config)['alerts'])


def test_remounted_identity_gets_only_one_retry_without_erasing_failure(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    video = Path(record['video_path'])
    old_stat = video.stat()
    replacement = video.with_name('owned-copy.tmp')
    replacement.write_bytes(video.read_bytes())
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    os.replace(replacement, video)
    queue = runner.queues['retention']
    failure = {'status': 'failed', 'error_type': 'InputChanged', 'message': 'Capture input changed before archival copy'}
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?", (json.dumps(failure),))
    assert tick_selected(monitor)['status'] == 'retry_authorized'
    with queue.connect() as db:
        row = db.execute('SELECT * FROM recordings').fetchone()
        assert row['attempts'] == 3 and row['retry_limit'] == 4
        assert json.loads(row['result']) == failure
    assert queue.claim('ordinary-worker', retry=True, max_attempts=3)
    queue.finish('ordinary-worker', record['recording_id'], failure, 1)
    due(monitor)
    assert tick_selected(monitor)['reason'] == 'recovery_retry_exhausted'
    with queue.connect() as db:
        row = db.execute('SELECT * FROM recordings').fetchone()
        assert row['attempts'] == row['retry_limit'] == 4
        assert db.execute('SELECT COUNT(*) FROM input_recovery_grants').fetchone()[0] == 1
    assert backend.vision_calls == backend.semantic_calls == 0


def test_changed_source_bytes_cannot_authorize_a_retry(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    source = Path(record['video_path'])
    old = source.stat()
    source.write_bytes(b'X' * old.st_size)
    os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
    queue = runner.queues['retention']
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?",
                   (json.dumps({'error_type': 'InputChanged'}),))
    assert tick_selected(monitor)['status'] == 'attention_required'
    with queue.connect() as db:
        assert db.execute('SELECT attempts,retry_limit FROM recordings').fetchone()[:] == (3, 0)
    assert backend.vision_calls == 0


def test_granted_retry_failure_alerts_even_with_a_new_error_and_nas_offline(monitor_case, monkeypatch):
    from visioncortex import device_day_recovery_worker
    config, record, layout, backend, runner, monitor = monitor_case
    queue = runner.queues['retention']
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?",
                   (json.dumps({'error_type': 'InputChanged'}),))
    assert tick_selected(monitor)['status'] == 'retry_authorized'
    assert queue.claim('retry-owner', retry=True, max_attempts=3)
    queue.finish('retry-owner', record['recording_id'],
                 {'status': 'failed', 'error_type': 'RuntimeError'}, 0)
    def offline(_):
        raise OSError('owned fixture storage outage')
    verify_storage = device_day_recovery_worker.verify_storage
    monkeypatch.setattr(device_day_recovery_worker, 'verify_storage', offline)
    for _ in range(3):
        assert IntegrityMonitor(config, runner).tick()['status'] == 'storage_unavailable'
    key = 'retention_recovery:' + record['recording_id']
    with connection(monitor.path, readonly=True) as db:
        alert = db.execute('SELECT * FROM alerts WHERE key=?', (key,)).fetchone()
        assert alert['state'] == 'active' and alert['reason'] == 'recovery_attempt_failed'
        assert db.execute("SELECT COUNT(*) FROM events WHERE key=? AND transition='raised'", (key,)).fetchone()[0] == 1
    with queue.connect() as db:
        assert db.execute('SELECT attempts,retry_limit FROM recordings').fetchone()[:] == (4, 4)
        db.execute("UPDATE recordings SET status='completed'")
    for _ in range(2):
        monitor.tick()
    with connection(monitor.path, readonly=True) as db:
        assert db.execute('SELECT state FROM alerts WHERE key=?', (key,)).fetchone()[0] == 'active'
    monkeypatch.setattr(device_day_recovery_worker, 'verify_storage', verify_storage)
    assert tick_selected(monitor)['status'] == 'verified'
    with connection(monitor.path, readonly=True) as db:
        assert db.execute('SELECT state FROM alerts WHERE key=?', (key,)).fetchone()[0] == 'resolved'
    assert backend.vision_calls == backend.semantic_calls == 0


def test_retry_authorization_is_not_rendered_as_a_completed_recovery():
    from visioncortex.device_day_progress import render_integrity
    html = render_integrity({'status': 'running', 'available': True, 'events': [
        {'transition': 'recovered', 'key': 'retention_failed:owned', 'reason': 'retry_authorized'}]})
    assert '恢复重试已授权' in html and '完成自动恢复' not in html


def test_changed_archive_blocks_retry_even_if_original_is_valid(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    path = runner._receipt(layout, record, 'retention')
    saved = read_json(path)
    target = layout.root / saved['artifacts'][0]['path']
    target.write_bytes(b'conflicting archive')
    atomic_json(path, {'status': 'failed', 'error_type': 'InputChanged'})
    with runner.queues['retention'].connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?",
                   (json.dumps({'error_type': 'InputChanged'}),))
    assert tick_selected(monitor)['reason'] == 'archive_content_conflict'
    assert target.read_bytes() == b'conflicting archive'
    with runner.queues['retention'].connect() as db:
        assert db.execute('SELECT retry_limit FROM recordings').fetchone()[0] == 0


def test_missing_recorder_native_input_is_never_recreated(monitor_case, monkeypatch):
    config, record, layout, backend, runner, monitor = monitor_case
    saved = read_json(runner._receipt(layout, record, 'retention'))
    target = layout.root / saved['artifacts'][0]['path']
    target.unlink()
    layout.recorder_owned_raw = True
    monkeypatch.setattr(runner, 'layout', lambda _: layout)
    assert tick_selected(monitor)['reason'] == 'recorder_input_missing'
    assert not target.exists()


def test_nas_outage_is_one_persistent_alert_without_inspecting_sources(monitor_case, monkeypatch):
    from visioncortex import device_day_recovery_worker
    config, record, layout, backend, runner, monitor = monitor_case
    def unavailable(_):
        raise ValueError('owned mount mismatch')
    monkeypatch.setattr(device_day_recovery_worker, 'verify_storage', unavailable)
    for _ in range(3):
        assert monitor.tick()['status'] == 'storage_unavailable'
    with connection(monitor.path, readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM checks').fetchone()[0] == 0
    assert integrity_status(config)['active_alert_count'] == 1


def test_artifact_manifest_streams_evidence_and_checks_reference_bounds(tmp_path):
    path = tmp_path / 'vision.json'
    ref = {'path': 'ProcessedClips/Frame.jpg', 'size_bytes': 1, 'sha256': '0' * 64}
    path.write_text(json.dumps({'status': 'completed', 'stage': 'vision', 'recording_id': 'owned',
                                'artifacts': [ref], 'evidence': [{'unused': 'large evidence'}] * 1000}))
    fields, refs = artifact_manifest(path)
    assert fields['recording_id'] == 'owned' and refs == [('artifacts', ref)]
    with pytest.raises(ValueError, match='excessive'):
        artifact_manifest(path, max_references=0)
    path.write_text('{"artifacts": [')
    with pytest.raises(ValueError):
        artifact_manifest(path)


def test_health_reader_never_creates_state_or_treats_stale_heartbeat_as_live(tmp_path):
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    assert integrity_status(config)['status'] == 'not_configured'
    assert not list(tmp_path.iterdir())
    path = tmp_path / 'device-day/InputIntegrity/Service.json'
    atomic_json(path, {'schema_version': 'visioncortex-input-integrity-service/1', 'pid': os.getpid(),
                       'status': 'running', 'updated_at': time.time() - 100})
    assert integrity_status(config)['status'] == 'stale'


def test_missing_derived_output_is_alerted_without_rerunning_models(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    assert runner.process(record, stage='vision')['status'] == 'completed'
    queue = runner.queues['vision']
    queue.enqueue(record, 'vision-revision')
    assert queue.claim('fixture')
    queue.finish('fixture', record['recording_id'], {'status': 'completed'}, 1)
    saved = read_json(runner._receipt(layout, record, 'vision'))
    target = layout.root / saved['artifacts'][0]['path']
    target.unlink()
    calls = backend.vision_calls
    assert tick_selected(monitor)['reason'] == 'derived_artifact_missing'
    assert backend.vision_calls == calls and not target.exists()


def test_periodic_hashes_do_not_trust_a_previous_process_cache(monitor_case, monkeypatch):
    from visioncortex import device_day_contract
    config, record, layout, backend, runner, monitor = monitor_case
    hashes = []
    original = device_day_contract.file_hash
    def fresh(path):
        hashes.append(str(path))
        return original(path)
    monkeypatch.setattr(device_day_contract, 'file_hash', fresh)
    assert tick_selected(monitor)['status'] == 'verified'
    first = len(hashes)
    assert first > 0
    due(monitor)
    assert tick_selected(monitor)['status'] == 'verified'
    assert len(hashes) == 2 * first


def test_live_stage_lease_defers_recovery_without_a_false_alert(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    with runner.queues['retention'].connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?",
                   (json.dumps({'error_type': 'InputChanged'}),))
    queue = runner.queues['vision']
    queue.enqueue(record, 'v')
    assert queue.claim('active')
    assert tick_selected(monitor)['status'] == 'busy'
    assert integrity_status(config)['active_alert_count'] == 0


def test_source_repair_cannot_grant_a_different_execution_recipe(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    with runner.queues['retention'].connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,revision='different-recipe',result=?",
                   (json.dumps({'error_type': 'InputChanged'}),))
    assert tick_selected(monitor)['reason'] == 'readmission_historical_scope_not_authorized'
    with runner.queues['retention'].connect() as db:
        row = db.execute('SELECT * FROM recordings').fetchone()
        assert row['retry_limit'] == 0 and row['revision'] == 'different-recipe'
        assert row['status'] == 'failed' and row['attempts'] == 3


def test_coverage_does_not_count_an_old_generation_as_current(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    assert tick_selected(monitor)['status'] == 'verified'
    assert integrity_status(config)['coverage'] == {'completed_queue_records': 1, 'verified_current': 1,
                                                   'unverified_or_changed': 0, 'overdue_current': 0}
    with runner.queues['retention'].connect() as db:
        db.execute('UPDATE recordings SET updated_at=updated_at+1')
    assert integrity_status(config)['coverage']['unverified_or_changed'] == 1


def test_recovery_cas_never_changes_a_concurrently_updated_queue(monitor_case):
    from visioncortex.device_day_contract import digest
    config, record, layout, backend, runner, monitor = monitor_case
    queue = runner.queues['retention']
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?", (json.dumps({'error_type': 'InputChanged'}),))
        expected = dict(db.execute('SELECT * FROM recordings').fetchone())
        db.execute('UPDATE recordings SET attempts=4')
    proof = {'status': 'verified', 'verified_seals': 2, 'recording_id': record['recording_id'],
             'current_queue_revision': expected['revision'], 'model_invoked': False,
             'capture_modified': False, 'source_generation': digest('owned sources')}
    assert queue.authorize_input_recovery(expected, digest('owned repair'), proof) == 'queue_changed'
    with queue.connect() as db:
        assert db.execute('SELECT retry_limit FROM recordings').fetchone()[0] == 0


def test_each_writer_keeps_the_local_state_size_bound(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    assert tick_selected(monitor)['status'] == 'verified'
    with monitor._db() as db:
        assert db.execute('PRAGMA max_page_count').fetchone()[0] == 32768
        assert db.execute('PRAGMA journal_size_limit').fetchone()[0] == 8388608


def test_published_index_loss_wakes_model_free_replay(monitor_case):
    from visioncortex.publication_journal import PublicationJournal
    config, record, layout, backend, runner, monitor = monitor_case
    atomic_json(runner._receipt(layout, record, 'publication'), {'status': 'completed'})
    layout.index.unlink(missing_ok=True)
    assert tick_selected(monitor)['reason'] == 'published_index_missing'
    assert PublicationJournal(runner.runtime_root).pending()[0][1]['recording_id'] == record['recording_id']
    assert backend.vision_calls == 0


def test_independent_service_heartbeat_and_shutdown_without_models(device_config, tmp_path):  # noqa: F811
    import subprocess
    import sys
    import yaml
    import visioncortex
    config = device_config
    config['project']['run_purpose'] = 'analysis'
    config['runtime']['local_only'] = True
    config['device_day']['enabled'] = config['collection_ingest']['enabled'] = False
    root = Path(config['storage']['local_runtime_root'])
    config['storage'].update(sync_to_nas=False, archive_root=str(root / 'archives'),
                             local_cache_root=str(root / 'cache'), local_staging_root=str(root / 'staging'),
                             local_input_root=str(root / 'inputs'))
    config['device_day']['input_integrity'] = {'enabled': True, 'interval_seconds': 1}
    path = tmp_path / 'Integrity.yaml'
    path.write_text(yaml.safe_dump(config))
    package = Path(visioncortex.__file__).resolve().parent
    environment = os.environ | {'PYTHONPATH': str(package.parent),
                               'VISIONCORTEX_DEFAULT_CONFIG': str(package.parents[1] / 'configs/default.yaml')}
    environment.pop('VISIONCORTEX_SITE_CONFIG', None)
    process = subprocess.Popen([sys.executable, '-m', 'visioncortex.runtime_services', '--input-integrity',
                                '--config', str(path)], cwd=tmp_path, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and process.poll() is None:
            state = integrity_status(config)
            if state.get('service', {}).get('completed_ticks', 0):
                break
            time.sleep(.05)
        assert process.poll() is None, process.stderr.read().decode() if process.poll() is not None else ''
        assert state['available'] and state['status'] == 'running'
        assert state['service']['model_invoked'] is state['service']['capture_modified'] is False
    finally:
        process.terminate()
        process.communicate(timeout=10)
    assert process.returncode == 0
    assert integrity_status(config)['available'] is False


def test_full_seal_verification_clears_content_conflict_wait(monitor_case):
    from visioncortex.input_availability import Availability
    config, record, layout, backend, runner, monitor = monitor_case
    availability = Availability(runner.runtime_root)
    availability.mark(record, 'waiting', reason='content_identity_conflict')
    with runner.queues['retention'].connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?",
                   (json.dumps({'error_type': 'InputChanged'}),))
    runner.queues['retention'].sync_availability(availability.states())
    with runner.queues['retention'].connect() as db:
        assert db.execute('SELECT input_status FROM recordings').fetchone()[0] == 'waiting'
    assert tick_selected(monitor)['status'] == 'retry_authorized'
    with runner.queues['retention'].connect() as db:
        assert tuple(db.execute('SELECT input_status,attempts,retry_limit FROM recordings').fetchone()) == ('ready', 3, 4)
    assert availability.states()[record['recording_id']]['state'] == 'ready'
    assert backend.vision_calls == backend.semantic_calls == 0


def test_paced_reader_preserves_bytes_and_avoids_per_megabyte_admission(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import hashlib
    from visioncortex import device_day_io
    from visioncortex.device_day_contract import file_hash
    from visioncortex.input_availability import _PacedInput
    payload = bytes(range(251)) * (17 * 1024**2 // 251 + 1)
    path = tmp_path / 'owned-bounded-bytes.bin'
    path.write_bytes(payload)
    admissions, paced = [], []
    @contextmanager
    def slot(*args, **kwargs):
        admissions.append(True)
        yield
    monkeypatch.setattr(device_day_io, 'slot', slot)
    monkeypatch.setattr(device_day_io, 'pace_copy', lambda config, size: paced.append(size))
    assert file_hash(_PacedInput({}, path)) == hashlib.sha256(payload).hexdigest()
    assert sum(paced) == len(payload) and len(admissions) < 8
    admissions.clear()
    with _PacedInput({}, path).open('rb') as stream:
        assert stream.read(0) == b'' and admissions == []
        assert stream.read(17) + stream.read(1024**2) + stream.read() == payload
        assert stream.read(1) == b''


def test_completed_alert_priority_wraps_before_full_archive_sweep(monitor_case):
    config, record, layout, backend, runner, monitor = monitor_case
    monitor._alert('retention_failed', record['recording_id'], 'recovery_retry_exhausted')
    with monitor._db() as db:
        db.execute("INSERT OR REPLACE INTO cursors VALUES('_closure_cursor','zzzz')")
    assert tick_selected(monitor)['status'] == 'verified'
    with monitor._db(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM checks WHERE lane='retention_closure'").fetchone()[0] == 1
        assert db.execute("SELECT state FROM alerts WHERE key=?",
                          ('retention_failed:' + record['recording_id'],)).fetchone()[0] == 'resolved'
    assert backend.vision_calls == backend.semantic_calls == 0


def test_ordinary_input_wait_does_not_trigger_a_seal_repair(monitor_case):
    from visioncortex.input_availability import Availability
    config, record, layout, backend, runner, monitor = monitor_case
    availability = Availability(runner.runtime_root)
    availability.mark(record, 'waiting', reason='capture_not_closed')
    queue = runner.queues['retention']
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?",
                   (json.dumps({'error_type': 'InputChanged'}),))
    queue.sync_availability(availability.states())
    with queue.connect() as db:
        assert db.execute('SELECT input_status FROM recordings').fetchone()[0] == 'waiting'
    assert monitor.tick()['status'] == 'idle'
    assert availability.states()[record['recording_id']]['state'] == 'waiting'
    assert backend.vision_calls == backend.semantic_calls == 0
