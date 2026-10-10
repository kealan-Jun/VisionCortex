import json
from pathlib import Path

import pytest

from test_device_day import device_config as _device_config, capture, FakeModels, item_and_layout
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, read_json, digest
from visioncortex.device_day_recovery import recover_one, RetentionRecovery


@pytest.fixture
def device_config(default_config, tmp_path):
    return _device_config.__wrapped__(default_config, tmp_path)


def failed_retention(config):
    folder = capture(config)
    record, layout = item_and_layout(config)
    runner = DeviceDayRunner(config, FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    path = runner._receipt(layout, record, 'retention')
    completed = read_json(path)
    atomic_json(path.parent / 'history' / 'retention-previous.json', completed | {'key': 'old-runtime-identity'})
    atomic_json(path, {'status': 'failed', 'error_type': 'FileNotFoundError'})
    folder.joinpath('rgb.mp4').unlink()  # This test owns the synthetic source.
    queue = runner.queues['retention']
    queue.enqueue(record, 'failed-version')
    queue.claim('test')
    queue.finish('test', record['recording_id'], {'status':'failed','error_type':'FileNotFoundError'}, 1)
    with queue.connect() as db:
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
    return runner, record, path, row


def test_restore_only_identical_verified_retention_preserving_failed_attempts(device_config):
    runner, record, path, row = failed_retention(device_config)
    result = recover_one(runner, row)
    assert result['status'] == 'completed'
    receipt = read_json(path)
    assert receipt['key'] == runner._key('retention', record, record)
    assert receipt['recovery']['model_invoked'] is False
    assert receipt['recovery']['capture_source_reinspected'] is False
    assert not Path(record['video_path']).exists()
    assert not path.with_name('vision.json').exists()
    with runner.queues['retention'].connect() as db:
        updated = db.execute('SELECT * FROM recordings').fetchone()
        assert updated['status'] == 'completed'
        assert updated['attempts'] == 1
        assert updated['revision'] == runner._key('retention', record, [record, None])
    assert any(read_json(p).get('status') == 'failed' for p in path.parent.joinpath('history').glob('*.json'))


def test_live_cutoff_pauses_historical_recovery(device_config, monkeypatch):
    runner, record, path, row = failed_retention(device_config)
    runner.settings['process_since_us'] = record['recording_end_us'] + 1
    monkeypatch.setattr('visioncortex.device_day_recovery.recover_one',
                        lambda *args: pytest.fail('Old recovery must remain paused'))
    assert RetentionRecovery().tick(runner) == {'status': 'idle'}
    assert read_json(path)['status'] == 'failed'


@pytest.mark.parametrize('change', ['corrupt', 'missing', 'source_changed', 'audio_omitted', 'snapshot_omitted', 'live_capture'])
def test_recovery_rejects_unverified_or_different_inputs(device_config, change):
    runner, record, path, row = failed_retention(device_config)
    history = path.parent / 'history' / 'retention-previous.json'
    receipt = read_json(history)
    target = runner.layout(record).root / receipt['sources'][0]['retained']['path']
    if change == 'corrupt':
        target.write_bytes(b'corrupted')
    elif change == 'missing':
        target.unlink()
    elif change == 'source_changed':
        receipt['recording']['recording_start_us'] += 1
        atomic_json(history, receipt)
    elif change == 'audio_omitted':
        record['audio']['files'] = [{'kind':'audio','path':str(Path(record['video_path']).with_name('audio.opus'))}]
        receipt['recording'] = record
        atomic_json(history, receipt)
        row['payload'] = json.dumps(record)
        with runner.queues['retention'].connect() as db:
            db.execute('UPDATE recordings SET payload=?',(row['payload'],))
    elif change == 'snapshot_omitted':
        receipt['sources'] = [s for s in receipt['sources'] if s['kind'] != 'sidecar']
        receipt['artifacts'] = [s['retained'] for s in receipt['sources']]
        atomic_json(history, receipt)
    else:
        Path(record['video_path']).write_bytes(b'new capture must not be replaced')
    before = digest(read_json(path))
    assert recover_one(runner, row)['status'] in {'no_verified_archive','capture_present'}
    assert digest(read_json(path)) == before
    with runner.queues['retention'].connect() as db:
        assert db.execute('SELECT status FROM recordings').fetchone()[0] == 'failed'


def test_recovery_compare_and_swap_preserves_new_queue_owner(device_config):
    runner, _, path, row = failed_retention(device_config)
    # Reappearance must first clear the independent input gate.
    with runner.queues['retention'].connect() as db:
        db.execute("UPDATE recordings SET input_status='ready'")
    assert runner.queues['retention'].claim('new-owner', retry=True) is not None
    assert recover_one(runner, row)['status'] == 'queue_changed'
    assert read_json(path)['status'] == 'failed'


def test_recovery_crash_between_receipt_and_queue_is_resumable(device_config):
    runner, _, path, row = failed_retention(device_config)
    assert recover_one(runner, row)['status'] == 'completed'
    with runner.queues['retention'].connect() as db:
        db.execute("UPDATE recordings SET status='failed',result=?",(json.dumps({'error_type':'FileNotFoundError'}),))
        new = dict(db.execute('SELECT * FROM recordings').fetchone())
    assert recover_one(runner, new)['status'] == 'completed'
    assert read_json(path)['recovery']['model_invoked'] is False


def test_background_tick_is_bounded_and_cools_down_unrecoverable_items(device_config, monkeypatch):
    runner, _, _, _ = failed_retention(device_config)
    calls = []
    monkeypatch.setattr('visioncortex.device_day_recovery.recover_one',
                        lambda *args: calls.append(args) or {'status':'no_verified_archive'})
    worker = RetentionRecovery()
    assert worker.tick(runner)['status'] == 'no_verified_archive'
    assert worker.tick(runner)['status'] == 'idle'
    assert len(calls) == 1


def test_capture_root_alias_preserves_original_snapshot_signature(device_config, tmp_path):
    root = Path(device_config['collection_ingest']['source_root'])
    alias = tmp_path / 'desktop-nas'
    alias.symlink_to(root, target_is_directory=True)
    device_config['collection_ingest']['source_root'] = str(alias)
    runner, _, _, row = failed_retention(device_config)
    assert recover_one(runner, row)['status'] == 'completed'


def test_missing_index_rebuilt_from_canonical_receipts_without_models(device_config):
    from visioncortex.device_day_recovery import repair_missing_index
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    assert runner.process(record)['status'] == 'completed'
    original = read_json(layout.index)
    calls = (backend.vision_calls, backend.semantic_calls)
    layout.index.unlink()  # Owned synthetic fixture only.
    result = repair_missing_index(runner, layout.name)
    assert result['status'] == 'rebuilt_from_stage_receipts'
    assert result['model_invoked'] is False
    assert read_json(layout.index)['segments'] == original['segments']
    assert (backend.vision_calls, backend.semantic_calls) == calls
    assert repair_missing_index(runner, layout.name)['status'] == 'already_present'


def test_recovery_prioritizes_retention_blocking_already_queued_yolo(device_config, monkeypatch):
    runner, record, _, _ = failed_retention(device_config)
    old = record | {'recording_id': 'older_unprocessed', 'recording_start_us': record['recording_start_us'] - 1000000}
    queue = runner.queues['retention']
    queue.enqueue(old, 'old')
    queue.claim('test-older')
    queue.finish('test-older',old['recording_id'],{'status':'failed','error_type':'FileNotFoundError'},1)
    runner.queues['vision'].enqueue(record, 'waiting')
    calls = []
    monkeypatch.setattr('visioncortex.device_day_recovery.recover_one',
                        lambda _, row: calls.append(row['recording_id']) or {'status':'completed'})
    RetentionRecovery().tick(runner)
    assert calls == [record['recording_id']]


def test_restored_source_requeues_once_and_preserves_other_stages(device_config, monkeypatch):
    from visioncortex.device_day_recovery import resume_reappeared_input
    runner, record, _, row = failed_retention(device_config)
    Path(record['video_path']).write_bytes(b'restored owned fixture')
    monkeypatch.setattr('visioncortex.nas_recordings._inspect', lambda *a: record | {'processable': True})
    result = resume_reappeared_input(runner, row)
    assert result['status'] == 'restored_input_queued'
    atomic_json(runner.runtime_root / 'InputAvailability' / f"{record['recording_id']}.json", result)
    assert resume_reappeared_input(runner, row)['status'] == 'unchanged_input_retry_already_attempted'
    with runner.queues['retention'].connect() as db:
        assert db.execute('select status,attempts from recordings').fetchone()[:] == ('queued', 0)
    with runner.queues['understanding'].connect() as db:
        assert db.execute('select count(*) from recordings').fetchone()[0] == 0
