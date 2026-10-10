"""Blocked recovery I/O must never hold stage admission or queue unlimited work."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from test_device_day import device_config as _device_config, capture, FakeModels
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_prerequisites import PrerequisiteChecks
from visioncortex.nas_recordings import scan_recordings


@pytest.fixture
def device_config(default_config, tmp_path):
    config = _device_config.__wrapped__(default_config, tmp_path)
    config['device_day'].update(camera_lanes=True, failure_retry_limit=3)
    return config


def two_cameras(config):
    capture(config, camera='a_cam01')
    capture(config, camera='b_cam02')
    inventory = scan_recordings(config)
    runner = DeviceDayRunner(config, FakeModels())
    assert len(runner.run_once(inventory, stage='retention')['results']) == 2
    runner._prepare_stage(inventory, 'vision', None)
    return runner, inventory


def cold_admission(runner):
    runner._prepared.clear()
    runner._record_readiness['vision'].clear()
    runner._admitted['vision'].clear()


def test_first_admission_performs_no_artifact_io(device_config, monkeypatch):
    runner, inventory = two_cameras(device_config)
    cold_admission(runner)
    def unexpected(*args):
        raise AssertionError('artifact I/O ran under admission lock')
    monkeypatch.setattr('visioncortex.device_day_prerequisites.artifact_metadata_identity', unexpected)
    monkeypatch.setattr(runner, '_load', unexpected)
    assert runner._prepare_stage(inventory, 'vision', None) == {r['recording_id'] for r in inventory['recordings']}


def test_blocked_waiting_artifact_does_not_block_another_camera_admission(device_config, monkeypatch):
    runner, inventory = two_cameras(device_config)
    first, second = inventory['recordings']
    queue = runner.queues['vision']
    with queue.connect() as db:
        revision = db.execute('SELECT revision FROM recordings WHERE recording_id=?',
                              (first['recording_id'],)).fetchone()[0]
    assert queue.wait_for_prerequisite(first['recording_id'], revision, 'retention')
    cold_admission(runner)
    assert runner._prepare_stage(inventory, 'vision', None) == {second['recording_id']}
    entered, release = threading.Event(), threading.Event()
    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return False
    monkeypatch.setattr('visioncortex.device_day_prerequisites.verified_prerequisite', blocked)
    with ThreadPoolExecutor(max_workers=2) as workers:
        verification = workers.submit(runner._prerequisite_checks.verify_pending, runner)
        try:
            assert entered.wait(1)
            cold_admission(runner)
            admitted = workers.submit(runner._prepare_stage, inventory, 'vision', None).result(timeout=1)
            assert admitted == {second['recording_id']}
            assert runner._prerequisite_checks.pending is None
            assert runner._prerequisite_checks.running is not None
        finally:
            release.set()
        assert not verification.result(timeout=1)['prerequisite_artifacts_verified']


def test_bad_first_waiter_cools_down_and_does_not_starve_later_waiter(tmp_path, monkeypatch):
    path = tmp_path/'receipt.json'
    receipt = {'status': 'completed'}
    atomic_json(path, receipt)
    checks = [(path, 'key', None, receipt)]
    mailbox = PrerequisiteChecks()
    monkeypatch.setattr('visioncortex.device_day_prerequisites.verified_prerequisite', lambda *a: False)
    assert not mailbox.ready('first', 'revision', checks)
    assert not mailbox.ready('second', 'revision', checks)
    assert mailbox.verify_pending(None)['recording_id'] == 'first'
    assert not mailbox.ready('first', 'revision', checks)
    assert not mailbox.ready('second', 'revision', checks)
    assert mailbox.verify_pending(None)['recording_id'] == 'second'


def test_grant_is_one_use_and_bound_to_revision_and_all_receipts(tmp_path, monkeypatch):
    path = tmp_path/'receipt.json'
    receipt = {'status': 'completed', 'artifact': 'original'}
    atomic_json(path, receipt)
    checks = [(path, 'key', None, receipt)]
    mailbox = PrerequisiteChecks()
    monkeypatch.setattr('visioncortex.device_day_prerequisites.verified_prerequisite', lambda *a: True)
    assert not mailbox.ready('record', 'revision', checks)
    assert mailbox.verify_pending(None)['prerequisite_artifacts_verified']
    assert not mailbox.ready('record', 'new-revision', checks)
    assert mailbox.ready('record', 'revision', checks)
    assert not mailbox.ready('record', 'revision', checks)
    mailbox.verify_pending(None)
    changed = receipt | {'artifact': 'changed'}
    atomic_json(path, changed)
    assert not mailbox.ready('record', 'new-revision', [(path, 'key', None, read_json(path))])


def test_receipt_change_during_background_verification_never_grants(tmp_path, monkeypatch):
    path = tmp_path/'receipt.json'
    receipt = {'status': 'completed'}
    atomic_json(path, receipt)
    mailbox = PrerequisiteChecks()
    def changed(*args):
        atomic_json(path, receipt | {'changed': True})
        return True
    monkeypatch.setattr('visioncortex.device_day_prerequisites.verified_prerequisite', changed)
    assert not mailbox.ready('record', 'revision', [(path, 'key', None, receipt)])
    assert not mailbox.verify_pending(None)['prerequisite_artifacts_verified']
