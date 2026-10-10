"""Local free-space admission is bounded and does not interrupt leased work."""
import threading
from types import SimpleNamespace

import pytest

from visioncortex import device_day_admission as admission, local_storage
from visioncortex.device_day_contract import STAGES
from visioncortex.device_day_progress import render_consumers
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker


GIB = 1024**3


def config(path, minimum=50):
    return {'device_day': {'local_storage_reserves': [
        {'path': str(path), 'min_available_gib': minimum}]}}


@pytest.mark.parametrize('stage', STAGES)
def test_reserve_boundary_applies_to_every_stage_and_recovers(tmp_path, monkeypatch, stage):
    free = [50 * GIB - 1]
    checked = []
    monkeypatch.setattr(admission.psutil, 'disk_usage',
                        lambda path: checked.append(path) or SimpleNamespace(free=free[0]))
    monkeypatch.setattr(admission.psutil, 'virtual_memory', lambda: pytest.fail('disabled memory guard read memory'))
    settings = config(tmp_path)
    result = admission.admission_status(settings, stage)
    assert result == {'status': 'waiting_for_storage', 'stage': stage, 'path': str(tmp_path),
                      'available_bytes': 50 * GIB - 1, 'reserve_bytes': 50 * GIB}
    free[0] += 1
    assert admission.admission_status(settings, stage) is None
    assert checked == [str(tmp_path)] * 2


def test_fractional_reserve_ceil_and_both_partitions_are_checked(tmp_path, monkeypatch):
    roots = [tmp_path / 'system', tmp_path / 'data']
    for path in roots:
        path.mkdir()
    minimum = 1 / GIB  # One byte, including the exact boundary.
    values = {str(roots[0]): 1, str(roots[1]): 80 * GIB - 1}
    monkeypatch.setattr(admission.psutil, 'disk_usage', lambda path: SimpleNamespace(free=values[path]))
    settings = {'device_day': {'local_storage_reserves': [
        {'path': str(roots[0]), 'min_available_gib': minimum},
        {'path': str(roots[1]), 'min_available_gib': 80}]}}
    result = admission.admission_status(settings, 'stt')
    assert result['path'] == str(roots[1])
    assert result['reserve_bytes'] == 80 * GIB


def test_all_paths_are_proven_local_before_any_disk_usage(tmp_path, monkeypatch):
    first, second = tmp_path / 'local', tmp_path / 'network'
    monkeypatch.setattr(local_storage, 'mount_filesystem_type',
                        lambda path: 'cifs' if path == second else 'ext4')
    monkeypatch.setattr(admission.psutil, 'disk_usage', lambda _path: pytest.fail('network check must precede usage'))
    settings = {'device_day': {'local_storage_reserves': [
        {'path': str(first), 'min_available_gib': 50},
        {'path': str(second), 'min_available_gib': 80}]}}
    with pytest.raises(RuntimeError, match='local non-NAS filesystem'):
        admission.admission_status(settings, 'retention')
    assert not first.exists() and not second.exists()


def test_unknown_capacity_waits_without_creating_local_fallback(tmp_path, monkeypatch):
    absent = tmp_path / 'absent'
    def unavailable(_path):
        raise FileNotFoundError('missing local destination')
    monkeypatch.setattr(admission.psutil, 'disk_usage', unavailable)
    result = admission.admission_status(config(absent), 'retention')
    assert result['status'] == 'waiting_for_storage'
    assert result['reason'] == 'local_capacity_unavailable'
    assert not absent.exists()


@pytest.mark.parametrize('stage', STAGES)
def test_storage_wait_prevents_new_claim_without_changing_active_lease(tmp_path, monkeypatch, stage):
    queue = DeviceDayQueue(tmp_path / f'queue-{stage}.sqlite3')
    records = [{'recording_id': rid, 'configured_role': 'first_person', 'source_signature': rid}
               for rid in ('busy', 'new')]
    for record in records:
        queue.enqueue(record, record['recording_id'])
    assert queue.claim('existing-worker')['recording_id'] == 'busy'
    with queue.connect() as db:
        before = [tuple(row) for row in db.execute('SELECT * FROM recordings ORDER BY recording_id')]
    monkeypatch.setattr(admission.psutil, 'disk_usage', lambda _path: SimpleNamespace(free=49 * GIB))
    runner = SimpleNamespace(config=config(tmp_path))
    worker = RetentionWorker(stage=stage)
    monkeypatch.setattr(worker, 'candidates', lambda _runner: pytest.fail('storage hold must precede admission'))
    assert worker.tick(runner, threading.Event())['status'] == 'waiting_for_storage'
    with queue.connect() as db:
        assert before == [tuple(row) for row in db.execute('SELECT * FROM recordings ORDER BY recording_id')]
    # The soft guard is not a cancellation flag: the original lease can finish.
    queue.finish('existing-worker', 'busy', {'status': 'completed'}, 1)
    with queue.connect() as db:
        assert db.execute("SELECT status FROM recordings WHERE recording_id='busy'").fetchone()[0] == 'completed'
        assert db.execute("SELECT attempts FROM recordings WHERE recording_id='new'").fetchone()[0] == 0


@pytest.mark.parametrize('reserves', [None, {}, ['invalid'],
    [{'path': 'relative', 'min_available_gib': 50}],
    [{'path': '/tmp', 'min_available_gib': -1}],
    [{'path': '/tmp', 'min_available_gib': True}],
    [{'path': '/tmp', 'min_available_gib': float('nan')}],
    [{'path': '/tmp', 'min_available_gib': float('inf')}],
    [{'path': '/tmp', 'min_available_gib': 1, 'unexpected': True}],
    [{'path': '/tmp', 'min_available_gib': 1}] * 2,
])
def test_invalid_storage_config_rejected(reserves):
    with pytest.raises(ValueError, match='local_storage_reserves'):
        admission.validate({'local_storage_reserves': reserves})


def test_public_default_does_not_probe_disk_or_change_memory_guard(monkeypatch):
    monkeypatch.setattr(admission.psutil, 'disk_usage', lambda _path: pytest.fail('unconfigured disk guard read storage'))
    assert admission.admission_status({}, 'retention') is None
    monkeypatch.setattr(admission.psutil, 'virtual_memory', lambda: SimpleNamespace(available=3 * GIB))
    assert admission.admission_status({'device_day': {'admission_min_available_gib': 4}}, 'vision')['status'] == 'waiting_for_memory'
    assert admission.admission_status({'device_day': {'admission_min_available_gib': 4}}, 'stt') is None


def test_consumer_render_exposes_storage_hold_only_for_verified_owner():
    data = {'stages': {'retention': {'scopes': {
        'recent': {'status': 'covered', 'locally_ready': 1, 'blocked': 0},
        'history': {'status': 'covered', 'locally_ready': 1, 'blocked': 0}}, 'owners': [
            {'owner_state': 'verified', 'last_result': {'status': 'waiting_for_storage'},
             'backfill_status': 'waiting_for_storage'}]}}}
    assert render_consumers(data).count('等待本地存储达到留空要求') == 2
    data['stages']['retention']['owners'][0]['owner_state'] = 'stale'
    assert '等待本地存储达到留空要求' not in render_consumers(data)
