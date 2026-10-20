"""Durable primary receipts hand off before asynchronous day rendering."""
from copy import deepcopy
from threading import Event

import pytest

from test_device_day import FakeModels, capture, device_config as device_config_fixture, item_and_layout
from test_device_day_retention_worker import record, setup
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_retention_worker import DeferredPrimaryPublicationRunner, RetentionWorker
from visioncortex.publication_journal import PublicationJournal, reconcile

device_config = device_config_fixture


def legacy_day_with_oversized_sibling(config, monkeypatch):
    """A cold publication cache must not own legacy stage completion."""
    monkeypatch.setenv('VISIONCORTEX_RECEIPT_PROJECTION_SHARED_AUDITS', '1')
    config['device_day'].update(defer_primary_publication=True, defer_audio_day_publication=False,
                                inplace_preprocessing=False)
    capture(config)
    source, layout = item_and_layout(config)
    previous = DeviceDayRunner(config, FakeModels())
    assert previous.process(source, stage='all')['status'] == 'completed'
    before = layout.index.read_bytes()
    other = deepcopy(source)
    other['recording_id'] = 'f' * 24
    oversized = layout.receipts / other['recording_id'] / 'vision.json'
    atomic_json(oversized, {'stage': 'vision', 'recording_id': other['recording_id'],
                           'status': 'completed', 'dense_diagnostics': 'x' * (256 * 1024)})
    # Reduce the actual native stream bound instead of allocating a 2 GiB fixture.
    monkeypatch.setattr('visioncortex.receipt_projection._MAX_STREAM_BYTES', 256 * 1024)
    cold = deepcopy(config)
    cold['storage']['local_runtime_root'] += '-cold'
    runner = DeferredPrimaryPublicationRunner(cold, FakeModels())
    assert not (layout.receipts / source['recording_id'] / 'input-state.json').exists()
    return cold, source, layout, runner, oversized, before


@pytest.mark.parametrize('stage', ['retention', 'vision', 'stt'])
def test_legacy_primary_completion_is_independent_of_oversized_other_recording(
        device_config, monkeypatch, stage):
    config, source, layout, runner, oversized, canonical = legacy_day_with_oversized_sibling(
        device_config, monkeypatch)
    receipt_path = runner._receipt(layout, source, stage)
    receipt = receipt_path.read_bytes()
    sibling = oversized.read_bytes()
    monkeypatch.setattr(runner, '_backend', lambda: pytest.fail('Completed native stage must be reused'))
    queue = runner.queues[stage]
    queue.enqueue(source, 'existing-native-stage')
    claimed = queue.claim('primary-owner')
    result = RetentionWorker(stage=stage)._execute_claimed(runner, claimed, 'primary-owner', Event())
    assert result['result']['status'] == 'completed'
    assert result['result']['formal_publication'] == 'pending'
    with queue.connect() as db:
        row = db.execute('SELECT status,result FROM recordings').fetchone()
        assert row['status'] == 'completed'
    assert receipt_path.read_bytes() == receipt
    assert oversized.read_bytes() == sibling
    assert layout.index.read_bytes() == canonical
    assert {r['recording_id'] for _, r in PublicationJournal(runner.runtime_root).pending()} == {
        source['recording_id']}

    # The normal replay retains the same stream cap and preserves this day.
    # A different archive can still publish in the same bounded replay round.
    capture(config, camera='b_cam02')
    from visioncortex.nas_recordings import scan_recordings
    other = next(r for r in scan_recordings(config)['recordings'] if r['camera_key'] == 'b_cam02')
    warm = DeviceDayRunner(device_config, FakeModels())
    assert warm.process(other, stage='all')['status'] == 'completed'
    PublicationJournal(runner.runtime_root).begin(other)
    replay = DeviceDayRunner(config, FakeModels())
    monkeypatch.setattr(replay, '_backend', lambda: pytest.fail('Replay cannot start models'))
    assert reconcile(replay, limit=4) == 1
    assert {r['recording_id'] for _, r in PublicationJournal(runner.runtime_root).pending()} == {
        source['recording_id']}
    deferred = next((runner.runtime_root / 'ReceiptProjectionDeferred').glob('*.json'))
    details = read_json(deferred)
    assert (details['archive'], details['requested_recording_id'], details['recording_id']) == (
        layout.name, source['recording_id'], oversized.parent.name)
    assert details['metric'] == 'stream_bytes' and details['observed'] > details['limit'] == 256 * 1024
    assert layout.index.read_bytes() == canonical and oversized.read_bytes() == sibling


def test_legacy_primary_failure_is_not_hidden_by_day_projection(device_config, monkeypatch):
    _, source, layout, runner, oversized, canonical = legacy_day_with_oversized_sibling(
        device_config, monkeypatch)
    runner._receipt(layout, source, 'vision').unlink()
    backend = FakeModels()
    backend.fail = True
    monkeypatch.setattr(runner, '_backend', lambda: backend)
    result = runner.process(source, stage='vision', retry=True)
    assert result['status'] == 'failed' and result['error_type'] == 'ValueError'
    assert result['message'] == 'synthetic scan failure'
    assert backend.vision_calls == 1
    assert read_json(runner._receipt(layout, source, 'vision'))['status'] == 'failed'
    assert layout.index.read_bytes() == canonical and oversized.is_file()
    assert PublicationJournal(runner.runtime_root).pending()


@pytest.mark.parametrize('stage', ['all', 'understanding', 'report'])
def test_legacy_nonprimary_publication_keeps_its_original_gate(device_config, monkeypatch, stage):
    _, source, layout, runner, _, canonical = legacy_day_with_oversized_sibling(device_config, monkeypatch)
    monkeypatch.setattr(runner, '_backend', lambda: pytest.fail('Completed receipts must be reused'))
    result = runner.process(source, stage=stage)
    assert result['status'] == 'waiting_for_publication'
    assert result['reason'] == 'readable_projection_limit'
    assert result['metric'] == 'stream_bytes' and result['limit'] == 256 * 1024
    assert layout.index.read_bytes() == canonical


def test_legacy_primary_without_defer_flag_keeps_publication_gate(device_config, monkeypatch):
    config, source, layout, _, _, canonical = legacy_day_with_oversized_sibling(device_config, monkeypatch)
    config['device_day']['defer_primary_publication'] = False
    runner = DeviceDayRunner(config, FakeModels())
    monkeypatch.setattr(runner, '_backend', lambda: pytest.fail('Completed native stage must be reused'))
    result = runner.process(source, stage='vision')
    assert result['status'] == 'waiting_for_publication'
    assert result['metric'] == 'stream_bytes' and result['limit'] == 256 * 1024
    assert layout.index.read_bytes() == canonical


def test_native_primary_finishes_without_rendering_and_replay_checks_before_understanding(device_config, monkeypatch):
    device_config['device_day']['inplace_preprocessing'] = True
    capture(device_config)
    source, layout = item_and_layout(device_config)
    models = FakeModels()
    runner = DeferredPrimaryPublicationRunner(device_config, models)
    original = DeviceDayRunner._guarded_refresh_index
    monkeypatch.setattr(DeviceDayRunner, '_guarded_refresh_index', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('Primary must not render the whole day')))
    for stage in ('vision', 'stt', 'retention'):
        result = runner.process(source, stage=stage)
        assert result['status'] == 'completed'
        assert result['formal_publication'] == 'pending'
        assert read_json(runner._receipt(layout, source, stage))['status'] == 'completed'
    assert not layout.index.exists()
    assert PublicationJournal(runner.runtime_root).pending()
    assert runner.process(source, stage='understanding')['status'] == 'waiting_for_publication'
    assert models.semantic_calls == 0
    monkeypatch.setattr(DeviceDayRunner, '_guarded_refresh_index', original)
    replay = DeviceDayRunner(device_config, models)
    monkeypatch.setattr(replay, '_backend', lambda: (_ for _ in ()).throw(AssertionError('No model startup in replay')))
    assert reconcile(replay) == 1
    assert read_json(runner._receipt(layout, source, 'publication'))['status'] == 'completed'
    assert layout.index.is_file() and not PublicationJournal(runner.runtime_root).pending()
    assert models.vision_calls == 1
    assert runner.process(source, stage='understanding')['status'] == 'completed'


def test_latest_retention_does_not_demote_newer_admitted_work_behind_completed_cv(tmp_path, monkeypatch):
    rows = [record('new', 3_000_000), record('old', 2_000_000)]
    runner = setup(tmp_path, monkeypatch, rows)
    runner.settings['latest_first'] = True
    queue = runner.queues['vision']
    queue.enqueue(rows[1], 'old-vision')
    queue.claim('old-owner')
    queue.finish('old-owner', 'old', {'status': 'completed'}, 1)
    assert RetentionWorker(stage='retention').tick(runner, Event())['recording_id'] == 'new'


def test_publication_latest_first_spans_dates_beyond_recent_horizon(tmp_path):
    journal = PublicationJournal(tmp_path)
    old = journal.begin(record('old', 1_000_000))
    newer = journal.begin(record('newer', 2_000_000))
    assert [token for token, _ in journal.pending(prefer_since_us=10_000_000, latest_first=True)] == [newer, old]
    assert [token for token, _ in journal.pending()] == [old, newer]
