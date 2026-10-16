"""Ready history uses the existing pool while preprocessing holds its lock."""
import time
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from test_device_day_retention_worker import record
from visioncortex.device_day import exclusive
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.observed_inventory import observe


@pytest.mark.parametrize('stage', ['understanding', 'report'])
def test_ready_history_admits_without_global_history_lock(tmp_path, monkeypatch, stage):
    root = tmp_path / 'device-day'
    queues = {s: DeviceDayQueue(root / f'queue-{s}.sqlite3')
              for s in ('retention', 'vision', 'stt', 'understanding', 'report')}
    records = [record(name, int((time.time() - 86400) * 1e6), camera=name) for name in ('a', 'b')]
    settings = {'understanding_workers': 2, 'understanding_coverage_since_us': 1,
                'backfill': {'enabled': True, 'stages': [stage], 'paid_history_authorized': True}}
    config = {'storage': {'local_runtime_root': str(tmp_path)}, 'collection_ingest': {},
              'runtime': {'local_only': True, 'resource_limits': {'cloud': 2}}, 'device_day': settings}
    runner = SimpleNamespace(runtime_root=root, queues=queues, config=config, settings=settings)
    worker = RetentionWorker(stage=stage, recent_seconds=14400)
    observe(root, {'recordings': records})
    for parent in worker.parents(runner, records[0]):
        for row in records:
            queues[parent].enqueue(row, 'parent')
            queues[parent].claim('parent-owner', allowed={row['recording_id']})
            queues[parent].finish('parent-owner', row['recording_id'], {'status': 'completed'}, 1)
    # Only replace the read-only eligibility port; normal candidates, admission,
    # queue leases, context and the real bounded pool still execute.
    monkeypatch.setattr('visioncortex.device_day_backfill_policy.historical_candidates', lambda *a, **k: records)
    barrier = Barrier(2, timeout=3) if stage == 'understanding' else None
    seen = []

    def admit(owner, row):
        queues[stage].enqueue(row, 'current')
        worker.save_admission(owner, row)
        return True

    def process(row, **kwargs):
        if barrier:
            barrier.wait()
        seen.append(row['recording_id'])
        return {'status': 'completed'}

    monkeypatch.setattr(worker, 'admit', admit)
    runner.process = process
    with exclusive(root / 'locks' / 'backfill.lock'):
        assert worker.tick(runner, Event())['status'] == 'processed'
    assert len(seen) == (2 if stage == 'understanding' else 1)
    with queues[stage].connect() as db:
        assert {r[0] for r in db.execute('SELECT processing_cohort FROM recordings')} == {'historical_backfill'}
    # Revoking explicit authorization closes the independent history lane.
    settings['backfill']['paid_history_authorized'] = False
    assert worker.candidates(runner) == []
