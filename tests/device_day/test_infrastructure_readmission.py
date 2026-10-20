"""Known infrastructure failures re-enter existing owners without model calls."""
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import pytest

from test_native_infrastructure_repair import failed_retention  # noqa: F401
from test_device_day import device_config  # noqa: F401
from visioncortex.device_day_contract import atomic_json
from visioncortex.device_day_retry_worker import RetryRepair
from visioncortex.runtime_services import NativeInfrastructureVerifier


@pytest.fixture
def unadmitted(failed_retention):  # noqa: F811
    config, runner, row, backend = failed_retention
    config['device_day'].update(archive_max_wait_seconds=0.001,
        backfill={'enabled': True, 'mode': 'fair', 'stages': ['retention', 'vision']},
        retry_repair={'enabled': True, 'allowed_classes': ['database_lock'], 'budget': 1})
    with runner.queues['retention'].connect() as db:
        db.execute('UPDATE recordings SET processing_cohort=NULL,admission_signature=NULL,admitted_at=NULL')
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
    current = time.time()
    atomic_json(Path(config['storage']['local_runtime_root']) / 'state/nas-recording-monitor.json', {
        'monitor': {'status': 'watching', 'observed_at': datetime.fromtimestamp(current, timezone.utc).isoformat()},
        'truncated': False, 'pending_publication_count': 0,
        'discovery_lanes': [{'camera_key': json.loads(row['payload'])['camera_key'], 'mode': 'live',
                             'status': 'watching', 'thread_alive': True, 'errors': [],
                             'truncated': False, 'completed_at': current}]})
    return config, runner, row, backend


def test_same_recipe_readmission_preserves_failure_then_grants_one_bounded_retry(unadmitted):
    config, runner, row, backend = unadmitted
    port = NativeInfrastructureVerifier(config)
    worker = RetryRepair(config, port)
    assert worker.tick(threading.Event())['status'] == 'source_readmitted'
    admitted = port._row(runner, 'retention', row['recording_id'])
    for field in ('status', 'revision', 'result', 'attempts', 'retry_limit', 'payload'):
        assert admitted[field] == row[field]
    assert admitted['processing_cohort'] == 'historical_backfill'
    assert admitted['admission_signature'] == json.loads(row['payload'])['source_signature']
    # The existing bounded cursor can finish its sweep before revisiting this
    # same row. Admission changes its proof identity without replenishing it.
    outcomes = [worker.tick(threading.Event())['status'] for _ in range(3)]
    assert outcomes.count('granted') == 1 and set(outcomes) <= {'waiting', 'granted'}
    granted = port._row(runner, 'retention', row['recording_id'])
    assert granted['attempts'] == 3 and granted['retry_limit'] == 4
    assert granted['status'] == 'failed' and granted['result'] == row['result']
    assert worker.tick(threading.Event())['status'] == 'waiting'
    assert backend.vision_calls == backend.semantic_calls == 0


def test_changed_recipe_uses_normal_enqueue_and_preserves_old_failure(unadmitted):
    config, runner, _, backend = unadmitted
    with runner.queues['retention'].connect() as db:
        db.execute('UPDATE recordings SET revision=?', ('f' * 64,))
        previous = dict(db.execute('SELECT * FROM recordings').fetchone())
    port = NativeInfrastructureVerifier(config)
    result = port.readmit(config, 'retention', previous, 'database_lock', threading.Event())
    assert result['status'] == 'recipe_readmitted' and result['completed'] is False
    current = port._row(runner, 'retention', previous['recording_id'])
    assert current['status'] == 'queued' and current['revision'] != previous['revision']
    with runner.queues['retention'].connect() as db:
        preserved = json.loads(db.execute('SELECT snapshot FROM source_versions WHERE revision=?',
                                          (previous['revision'],)).fetchone()[0])
    assert preserved == previous
    assert backend.vision_calls == backend.semantic_calls == 0


@pytest.mark.parametrize('gate', ['historical_disabled', 'paused', 'unknown', 'unsupported_class', 'paid',
                                'monitor_missing', 'not_exhausted'])
def test_closed_readmission_gates_reject_before_source_reads(unadmitted, monkeypatch, gate):
    config, runner, row, _ = unadmitted
    stage, kind = 'retention', 'database_lock'
    if gate == 'historical_disabled':
        config['device_day']['backfill']['enabled'] = False
    elif gate == 'paused':
        config['device_day']['paused_stages'] = ['retention']
    elif gate == 'unknown':
        row['blocked_reason'] = 'submitted_unknown'
    elif gate == 'unsupported_class':
        kind = 'vision_nms'
    elif gate == 'paid':
        stage = 'understanding'
    elif gate == 'monitor_missing':
        (Path(config['storage']['local_runtime_root']) / 'state/nas-recording-monitor.json').unlink()
    else:
        row['attempts'] = 0
    from visioncortex import nas_recordings
    monkeypatch.setattr(nas_recordings, '_inspect', lambda *a, **kw: pytest.fail('Closed gate read source'))
    before = NativeInfrastructureVerifier._row(runner, 'retention', row['recording_id'])
    with pytest.raises(ValueError):
        NativeInfrastructureVerifier(config).readmit(config, stage, row, kind, threading.Event())
    assert NativeInfrastructureVerifier._row(runner, 'retention', row['recording_id']) == before


def test_readmission_does_not_grant_on_original_content_conflict(unadmitted):
    config, runner, row, _ = unadmitted
    import os
    video = Path(json.loads(row['payload'])['video_path'])
    stat, content = video.stat(), bytearray(video.read_bytes())
    content[0] ^= 1
    video.write_bytes(content)
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    port = NativeInfrastructureVerifier(config)
    with pytest.raises(ValueError, match='differs from its input seal'):
        port.readmit(config, 'retention', row, 'database_lock', threading.Event())
    assert port._row(runner, 'retention', row['recording_id']) == row


def test_readmission_rejects_unverified_parent_artifact(unadmitted):
    config, runner, retention, backend = unadmitted
    record = json.loads(retention['payload'])
    queue = runner.queues['vision']
    queue.enqueue(record, 'f' * 64)
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',attempts=3,result=?", (retention['result'],))
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
    from visioncortex.device_day_contract import read_json
    layout = runner.layout(record)
    saved = read_json(runner._receipt(layout, record, 'retention'))
    target = layout.root / next(source['retained']['path'] for source in saved['sources'] if source['kind'] == 'video')
    assert target != Path(record['video_path'])
    target.write_bytes(b'corrupt owned fixture archive')
    port = NativeInfrastructureVerifier(config)
    with pytest.raises(ValueError, match='Current parent artifact is not verified'):
        port.readmit(config, 'vision', row, 'database_lock', threading.Event())
    assert port._row(runner, 'vision', row['recording_id']) == row
    assert backend.vision_calls == backend.semantic_calls == 0


def test_readmission_cas_rejects_concurrent_queue_change(unadmitted, monkeypatch):
    config, runner, row, _ = unadmitted
    port = NativeInfrastructureVerifier(config)
    original, calls = port._recheck, []
    def changed(*args):
        result = original(*args)
        calls.append(1)
        if len(calls) == 2:
            with runner.queues['retention'].connect() as db:
                db.execute('UPDATE recordings SET updated_at=updated_at+1')
        return result
    monkeypatch.setattr(port, '_recheck', changed)
    with pytest.raises(ValueError, match='changed before grant'):
        port.readmit(config, 'retention', row, 'database_lock', threading.Event())
    current = port._row(runner, 'retention', row['recording_id'])
    assert current['admitted_at'] is None and current['result'] == row['result']


@pytest.fixture
def changed_input_recipe(unadmitted):
    config, runner, _, backend = unadmitted
    config['device_day']['input_integrity'] = {'enabled': True}
    with runner.queues['retention'].connect() as db:
        db.execute('UPDATE recordings SET revision=?,result=?',
                   ('e' * 64, json.dumps({'status': 'failed', 'error_type': 'InputChanged'})))
        previous = dict(db.execute('SELECT * FROM recordings').fetchone())
    return config, runner, previous, backend


def test_changed_input_recipe_handoff_is_normal_execution_without_a_repair_budget(changed_input_recipe):
    config, runner, previous, backend = changed_input_recipe
    from visioncortex.input_availability import IntegrityMonitor
    result = IntegrityMonitor(config, runner)._check('retention', previous)
    assert result['status'] == 'normal_recipe_readmitted'
    assert result['retry_budget_granted'] is False and result['completed'] is False
    current = NativeInfrastructureVerifier._row(runner, 'retention', previous['recording_id'])
    assert current['status'] == 'queued' and current['retry_limit'] == 0
    assert current['revision'] != previous['revision']
    with runner.queues['retention'].connect() as db:
        preserved = json.loads(db.execute('SELECT snapshot FROM source_versions WHERE revision=?',
                                          (previous['revision'],)).fetchone()[0])
        assert db.execute('SELECT COUNT(*) FROM retry_authorizations').fetchone()[0] == 0
    assert preserved == previous
    assert backend.vision_calls == backend.semantic_calls == 0


@pytest.mark.parametrize('gate', ['same_recipe', 'unknown', 'source_bytes', 'paid'])
def test_normal_input_recipe_handoff_never_weakens_its_gates(changed_input_recipe, gate):
    config, runner, row, backend = changed_input_recipe
    stage = 'retention'
    if gate == 'same_recipe':
        from visioncortex.device_day_inplace import enqueue
        record = json.loads(row['payload'])
        enqueue(runner, record, stage)
        with runner.queues[stage].connect() as db:
            db.execute('UPDATE recordings SET status=?,attempts=3,result=?',
                       ('failed', json.dumps({'error_type': 'InputChanged'})))
            row = dict(db.execute('SELECT * FROM recordings').fetchone())
    elif gate == 'unknown':
        row['blocked_reason'] = 'submitted_unknown'
    elif gate == 'paid':
        stage = 'stt'
    else:
        import os
        video = Path(json.loads(row['payload'])['video_path'])
        before, content = video.stat(), bytearray(video.read_bytes())
        content[0] ^= 1
        video.write_bytes(content)
        os.utime(video, ns=(before.st_atime_ns, before.st_mtime_ns))
    port = NativeInfrastructureVerifier(config)
    current = port._row(runner, 'retention', row['recording_id'])
    with pytest.raises(ValueError):
        port.normal_readmit(config, stage, row, threading.Event())
    assert port._row(runner, 'retention', row['recording_id']) == current
    assert backend.vision_calls == backend.semantic_calls == 0
