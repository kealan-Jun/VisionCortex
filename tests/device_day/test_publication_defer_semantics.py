"""A bounded projection is a named defer, never a lock or completion claim."""
from contextlib import contextmanager
import json

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout
from visioncortex.device_day import DeviceDayRunner
from visioncortex.publication_journal import PublicationJournal, reconcile
from visioncortex.receipt_projection import ProjectionDeferred

__all__ = ['device_config']


def test_real_projection_defer_preserves_exception_metrics_and_real_lock_remains_none(device_config, monkeypatch):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    deferred = ProjectionDeferred('stream_events', 8000001, 8000000, record['recording_id'])
    @contextmanager
    def limited(*_args, **_kwargs):
        raise deferred
        yield
    monkeypatch.setattr('visioncortex.device_day.exclusive', limited)
    with pytest.raises(ProjectionDeferred) as caught:
        runner._guarded_refresh_index(layout, recording_id=record['recording_id'])
    assert caught.value is deferred
    assert (caught.value.metric, caught.value.observed, caught.value.limit) == ('stream_events', 8000001, 8000000)
    @contextmanager
    def busy(*_args, **_kwargs):
        raise BlockingIOError('owned lock is busy')
        yield
    monkeypatch.setattr('visioncortex.device_day.exclusive', busy)
    assert runner._guarded_refresh_index(layout, recording_id=record['recording_id']) is None


def test_existing_journal_keeps_pending_and_retains_projection_metrics(device_config, monkeypatch):
    from visioncortex.sqlite_store import connection
    capture(device_config)
    record, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    journal = PublicationJournal(runner.runtime_root)
    journal.begin(record)
    deferred = ProjectionDeferred('resident_bytes', 1025, 1024, record['recording_id'])
    def limited(*_args, **_kwargs):
        raise deferred
    monkeypatch.setattr(runner, 'refresh_index', limited)
    assert reconcile(runner) == 0
    with connection(journal.path, readonly=True) as db:
        assert db.execute('SELECT count(*) FROM dirty').fetchone()[0] == 1
        reason = json.loads(db.execute('SELECT reason FROM retries').fetchone()[0])
    assert reason == {'reason': 'readable_projection_limit', 'error_type': 'ProjectionDeferred',
                      'metric': 'resident_bytes', 'observed': 1025, 'limit': 1024}
