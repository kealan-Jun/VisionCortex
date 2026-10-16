"""Durable primary receipts hand off before asynchronous day rendering."""
from threading import Event

from test_device_day import FakeModels, capture, device_config as device_config_fixture, item_and_layout
from test_device_day_retention_worker import record, setup
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_retention_worker import DeferredPrimaryPublicationRunner, RetentionWorker
from visioncortex.publication_journal import PublicationJournal, reconcile

device_config = device_config_fixture


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
