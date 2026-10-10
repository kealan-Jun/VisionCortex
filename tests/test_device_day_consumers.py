"""Consumer health uses owned local metadata; no NAS or models are involved."""
import json
from pathlib import Path
import threading

from visioncortex.device_day_consumers import VERSION, WorkerStatus, consumer_snapshot, process_start_ticks
from visioncortex.device_day_contract import atomic_json
from visioncortex.device_day_queue import DeviceDayQueue


NOW = 1_800_000_000
COMMIT = 'a' * 40


def configuration(tmp_path):
    return {'storage': {'local_runtime_root': str(tmp_path / 'runtime')},
            'device_day': {'enabled': True, 'inplace_preprocessing': True,
                           'live_priority_seconds': 14400}}


def queued(config, *, rid='old', stage='vision', age=20000, **extra):
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    queue = DeviceDayQueue(root / f'queue-{stage}.sqlite3')
    record = {'recording_id': rid, 'source_signature': rid, 'configured_role': 'first_person',
              'processable': True, 'camera_key': 'camera-a',
              'recording_start_us': (NOW - age) * 1_000_000,
              'recording_end_us': (NOW - age + 5) * 1_000_000, **extra}
    queue.enqueue(record, rid)
    return queue


def owner(config, proc_root, *, pid=100, ticks=10, at=NOW, scopes=('recent', 'history'),
          status='running', stage='vision', build=None):
    path = proc_root / str(pid) / 'stat'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{pid} (worker (test)) ' + ' '.join(['S'] + ['0'] * 18 + [str(ticks)]))
    root = Path(config['storage']['local_runtime_root']) / 'device-day' / 'Consumers'
    data = {'schema_version': VERSION, 'stage': stage, 'pid': pid, 'process_start_ticks': ticks,
            'supported_scopes': list(scopes), 'status': status, 'at': at,
            'build': {'commit': COMMIT} if build is None else build}
    atomic_json(root / f'{stage}-{pid}-{ticks}.json', data)
    return root / f'{stage}-{pid}-{ticks}.json'


def test_old_running_status_cannot_cover_history_and_live_owner_does_not_cover_it(tmp_path):
    config = configuration(tmp_path)
    queued(config)
    proc = tmp_path / 'proc'
    owner(config, proc, pid=101, at=NOW - 100)
    owner(config, proc, pid=102, scopes=('recent',))
    data = consumer_snapshot(config, now=NOW, proc_root=proc)
    stage = data['stages']['vision']
    assert stage['scopes']['recent']['status'] == 'covered'
    assert stage['scopes']['history']['status'] == 'no_consumer'
    assert stage['scopes']['history']['locally_ready'] == 1
    assert [row['owner_state'] for row in stage['owners']] == ['stale', 'verified']


def test_disabled_observer_reports_separate_verified_consumers_and_keeps_pause_policy(tmp_path):
    config = configuration(tmp_path)
    config['device_day']['enabled'] = False
    queued(config)
    proc = tmp_path / 'proc'
    owner(config, proc, stage='vision')
    scopes = consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']['scopes']
    assert scopes['recent']['status'] == scopes['history']['status'] == 'covered'
    config['device_day']['paused_stages'] = ['vision']
    assert consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']['scopes']['history']['status'] == 'paused_by_user'
    config['device_day']['paused_stages'] = []
    assert consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['stt']['scopes']['history']['status'] == 'disabled'


def test_pid_reuse_absent_process_missing_build_and_future_stamp_are_not_owners(tmp_path):
    config = configuration(tmp_path)
    queued(config)
    proc = tmp_path / 'proc'
    path = owner(config, proc)
    assert process_start_ticks(100, proc) == 10
    (proc / '100' / 'stat').write_text('100 (replacement) ' + ' '.join(['S'] + ['0'] * 18 + ['11']))
    stage = consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']
    assert stage['scopes']['history']['status'] == 'no_consumer'
    assert stage['owners'][0]['owner_state'] == 'owner_absent_or_replaced'
    (proc / '100' / 'stat').unlink()
    assert process_start_ticks(100, proc) is None
    path.unlink()
    owner(config, proc, build={})
    assert consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']['owners'][0]['owner_state'] == 'identity_unverified'
    owner(config, proc, at=NOW + 10)
    assert consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']['owners'][0]['owner_state'] == 'stale'


def test_pauses_disabled_scope_cutoff_and_night_policy_are_explicit(tmp_path):
    config = configuration(tmp_path)
    queued(config)
    proc = tmp_path / 'proc'
    config['device_day']['paused_stages'] = ['vision']
    assert consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']['scopes']['history']['status'] == 'paused_by_user'
    config['device_day']['enabled'] = False
    assert consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']['scopes']['history']['status'] == 'disabled'
    config['device_day'].update(enabled=True, paused_stages=[], process_since_us=NOW * 1_000_000)
    assert consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']['scopes']['history']['pending'] == 0
    # NOW is 16:00 in Shanghai, outside this explicit night window.
    config['device_day']['night_processing'] = {'enabled': True, 'start': '23:00', 'end': '23:01'}
    assert consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['understanding']['scopes']['history']['status'] == 'waiting_for_night_window'


def test_input_holds_failure_limits_and_prerequisites_do_not_create_ready_work(tmp_path):
    config = configuration(tmp_path)
    queue = queued(config)
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,updated_at=?", (NOW - 100,))
    assert consumer_snapshot(config, now=NOW)['stages']['vision']['scopes']['history']['locally_ready'] == 0
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='queued',attempts=0,input_status='missing'")
    assert consumer_snapshot(config, now=NOW)['stages']['vision']['scopes']['history']['blocked'] == 1
    queued(config, stage='understanding')
    assert consumer_snapshot(config, now=NOW)['stages']['understanding']['scopes']['history']['locally_ready'] == 0


def test_snapshot_does_not_create_runtime_or_rewrite_queue_and_bad_evidence_is_visible(tmp_path):
    config = configuration(tmp_path)
    consumer_snapshot(config, now=NOW)
    assert not Path(config['storage']['local_runtime_root']).exists()
    queue = queued(config)
    with queue.connect() as db:
        before = [dict(row) for row in db.execute('SELECT * FROM recordings')]
    data = consumer_snapshot(config, now=NOW)
    with queue.connect() as db:
        assert before == [dict(row) for row in db.execute('SELECT * FROM recordings')]
    assert data['local_metadata_only']
    broken = queue.path.parent / 'queue-stt.sqlite3'
    broken.write_text('not a database')
    data = consumer_snapshot(config, now=NOW)
    assert 'stt_queue_unavailable' in data['errors']
    assert data['stages']['stt']['scopes']['history']['status'] == 'evidence_unavailable'


def test_verified_duplicates_are_visible_and_stopped_records_are_not_coverage(tmp_path):
    config = configuration(tmp_path)
    queued(config)
    proc = tmp_path / 'proc'
    owner(config, proc, pid=100)
    owner(config, proc, pid=101)
    owner(config, proc, pid=102, status='stopped')
    stage = consumer_snapshot(config, now=NOW, proc_root=proc)['stages']['vision']
    assert stage['duplicate_owner_scopes'] == ['recent', 'history']
    assert stage['scopes']['history']['verified_owner_count'] == 2


def test_heartbeat_continues_during_work_and_fullscope_worker_covers_history(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setattr('visioncortex.build_identity.identity', lambda: {'commit': COMMIT})
    heartbeat = threading.Event()
    real_atomic = atomic_json
    calls = []
    def publish(path, data):
        calls.append(data)
        real_atomic(path, data)
        if len(calls) >= 2:
            heartbeat.set()
    monkeypatch.setattr('visioncortex.device_day_consumers.atomic_json', publish)
    with WorkerStatus(config, 'stt', recent_seconds=None, interval=.01) as status:
        assert heartbeat.wait(2)
        status.update(scope='history', last_result={'status': 'processed', 'message': 'secret-value',
                      'result': {'provider': 'secret-value'}}, unrelated='secret-value')
        data = json.loads(status.path.read_text())
        assert data['supported_scopes'] == ['recent', 'history']
        assert 'secret-value' not in status.path.read_text()
        assert consumer_snapshot(config)['stages']['stt']['scopes']['history']['status'] == 'covered'
    assert json.loads(status.path.read_text())['status'] == 'stopped'
