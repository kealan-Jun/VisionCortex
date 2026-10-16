"""A verified yesterday admission stays in latest-first primary processing."""
import time

import pytest

from test_device_day_retention_worker import record, setup
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.observed_inventory import observe


@pytest.mark.parametrize('stage', ['retention', 'vision', 'stt'])
def test_admitted_yesterday_precedes_older_history_without_expanding_permission(tmp_path, monkeypatch, stage):
    now = time.time()
    rows = [record('yesterday', int((now-86400)*1e6), camera='camera'),
            record('older', int((now-3*86400)*1e6), camera='camera'),
            record('not-admitted', int((now-5*3600)*1e6), camera='other')]
    runner = setup(tmp_path, monkeypatch, rows)
    runner.settings['backfill'] = {'enabled': True, 'stages': [stage], 'paid_history_authorized': False}
    queue = runner.queues[stage]
    for row in rows[:2]:
        queue.enqueue(row, 'same-execution')
        queue.grant_admission(row, cohort='historical_backfill')
    worker = RetentionWorker(stage=stage, recent_seconds=14400)
    assert [r['recording_id'] for r in worker.candidates(runner)] == ['yesterday']
    worker.save_admission(runner, rows[0])
    with queue.connect() as db:
        assert db.execute("SELECT processing_cohort FROM recordings WHERE recording_id='yesterday'").fetchone()[0] == 'historical_backfill'
    observe(runner.runtime_root, {'recordings': [rows[0] | {'source_signature': 'changed'}]})
    assert 'yesterday' not in [r['recording_id'] for r in worker.candidates(runner)]
    runner.settings['backfill']['enabled'] = False
    assert worker.candidates(runner) == []
