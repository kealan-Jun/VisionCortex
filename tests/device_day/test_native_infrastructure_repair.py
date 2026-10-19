"""Automatic infrastructure recovery validates real owned input bytes."""
import json
import os
import threading

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_inplace import enqueue
from visioncortex.device_day_retry_worker import RetryRepair, verify_proof
from visioncortex.runtime_services import NativeInfrastructureVerifier
from visioncortex.observed_inventory import observe


@pytest.fixture
def failed_retention(device_config):  # noqa: F811
    config = device_config
    config['runtime']['local_only'] = True
    config['device_day']['inplace_preprocessing'] = True
    capture(config)
    record, layout = item_and_layout(config)
    backend = FakeModels()
    runner = DeviceDayRunner(config, backend)
    assert runner.process(record, stage='retention')['status'] == 'completed'
    enqueue(runner, record, 'retention')
    queue = runner.queues['retention']
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?",
                   (json.dumps({'status': 'failed', 'stage': 'retention', 'error_type': 'OperationalError',
                                'message': 'database is locked'}),))
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
    observe(runner.runtime_root, {'recordings': [record]})
    return config, runner, row, backend


def test_native_verification_proves_repair_without_mutating_failure_or_models(failed_retention):
    config, runner, row, backend = failed_retention
    port = NativeInfrastructureVerifier(config)
    stop = threading.Event()
    assert port.verify_source_view(config, stop)['status'] == 'PROVEN'
    proof = port.verify(config, 'retention', row, 'database_lock', stop)
    evidence = verify_proof(proof, row, 'retention', 'database_lock', config)
    assert evidence['checks']['current_source_verified'] == 'PROVEN'
    assert port.recheck(config, 'retention', row, proof, stop)['status'] == 'PROVEN'
    assert port._row(runner, 'retention', row['recording_id']) == row
    assert backend.vision_calls == backend.semantic_calls == 0


def test_native_recheck_rejects_source_change_after_complete_byte_proof(failed_retention):
    config, runner, row, _ = failed_retention
    port = NativeInfrastructureVerifier(config)
    proof = port.verify(config, 'retention', row, 'database_lock', threading.Event())
    from pathlib import Path
    video = Path(json.loads(row['payload'])['video_path'])
    video.write_bytes(b'different owned source')
    with pytest.raises(ValueError, match='changed before grant'):
        port.recheck(config, 'retention', row, proof, threading.Event())
    assert port._row(runner, 'retention', row['recording_id']) == row


def test_same_size_and_mtime_cannot_replace_the_input_seal(failed_retention):
    config, _, row, _ = failed_retention
    from pathlib import Path
    video = Path(json.loads(row['payload'])['video_path'])
    stat = video.stat()
    data = bytearray(video.read_bytes())
    data[0] ^= 1
    video.write_bytes(data)
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='differs from its input seal'):
        NativeInfrastructureVerifier(config).verify(config, 'retention', row, 'database_lock', threading.Event())


def test_native_factory_cannot_authorize_model_or_quality_failures(failed_retention):
    config, _, row, _ = failed_retention
    with pytest.raises(ValueError, match='own explicit execution proof'):
        NativeInfrastructureVerifier(config).verify(config, 'vision', row, 'vision_nms', threading.Event())


def test_native_storage_access_probe_leaves_no_files(failed_retention):
    config, runner, row, _ = failed_retention
    NativeInfrastructureVerifier(config).verify(config, 'retention', row, 'storage_access', threading.Event())
    folder = runner._receipt(runner.layout(json.loads(row['payload'])), json.loads(row['payload']), 'retention').parent
    assert not list(folder.glob('.repair-probe-*'))


def test_native_worker_grants_one_budget_without_erasing_the_failure(failed_retention):
    config, runner, row, backend = failed_retention
    config['device_day']['retry_repair'] = {'enabled': True, 'allowed_classes': ['database_lock'], 'budget': 1}
    port = NativeInfrastructureVerifier(config)
    worker = RetryRepair(config, port)
    result = worker.tick(threading.Event())
    assert result['status'] == 'granted'
    current = port._row(runner, 'retention', row['recording_id'])
    assert current['attempts'] == 3 and current['retry_limit'] == 4
    with runner.queues['retention'].connect() as db:
        saved = db.execute('SELECT previous_result FROM retry_authorizations').fetchone()[0]
        assert saved == row['result']
        db.execute("UPDATE recordings SET status='failed',attempts=4")
    assert worker.tick(threading.Event())['status'] in {'waiting', 'verification_rejected'}
    assert port._row(runner, 'retention', row['recording_id'])['retry_limit'] == 4
    assert backend.vision_calls == backend.semantic_calls == 0
