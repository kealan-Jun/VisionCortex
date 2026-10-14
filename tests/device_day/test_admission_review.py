"""Independent admission review with local inventory/queue fixtures only."""
import time
from types import SimpleNamespace

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.observed_inventory import observe


def record(identifier, stamp, signature='stable-content', camera='one-camera'):
    return {'recording_id': identifier, 'camera_key': camera, 'configured_role': 'first_person',
            'recording_start_us': stamp, 'recording_end_us': stamp+1_000_000,
            'source_signature': signature, 'processable': True}


def test_live_readmission_never_promotes_an_existing_historical_paid_cohort(tmp_path):
    item = record('historical', int(time.time()*1e6))
    queue = DeviceDayQueue(tmp_path/'queue-vision.sqlite3')
    queue.enqueue(item, 'same-source-revision')
    queue.grant_admission(item, cohort='historical_backfill', reason='separate_history_authorization')
    runner = SimpleNamespace(settings={}, queues={'vision': queue})
    RetentionWorker(stage='vision', admission_cohort='live').save_admission(runner, item)
    with queue.connect() as db:
        cohort = db.execute('SELECT processing_cohort FROM recordings').fetchone()[0]
    assert cohort == 'historical_backfill'
    assert queue.admissions() == {}


def test_aged_checkpoint_remains_visible_during_same_camera_live_arrivals(tmp_path):
    current = time.time()
    root = tmp_path/'device-day'
    old = record('admitted-five-hours-ago', int((current-5*3600)*1e6))
    newest = record('new-arrival', int((current-1)*1e6))
    queue = DeviceDayQueue(root/'queue-understanding.sqlite3', latest_first=True,
                           scheduling={'enabled': True, 'max_ready_wait_seconds': 10})
    queue.enqueue(old, 'older-source-revision')
    queue.grant_admission(old, at=current-5*3600)
    parents = {stage: DeviceDayQueue(root/f'queue-{stage}.sqlite3') for stage in ('vision', 'retention')}
    for parent in parents.values():
        for item in (old, newest):
            parent.enqueue(item, item['recording_id'])
            parent.claim('fixture-owner')
            parent.finish('fixture-owner', item['recording_id'], {'status': 'completed'}, 0)
    observe(root, {'recordings': [old, newest]})
    runner = SimpleNamespace(runtime_root=root, queues={'understanding': queue, **parents},
                             settings={'understanding_coverage_since_us': 1},
                             config={'collection_ingest': {}})
    selected = RetentionWorker(stage='understanding', recent_seconds=14400).candidates(runner)
    assert 'admitted-five-hours-ago' in {item['recording_id'] for item in selected}


def test_content_change_revokes_the_exact_previous_admission(tmp_path):
    item = record('source', 1)
    queue = DeviceDayQueue(tmp_path/'queue-vision.sqlite3')
    queue.enqueue(item, 'v1')
    queue.grant_admission(item)
    queue.enqueue(item | {'source_signature': 'new-content'}, 'v2')
    assert queue.admissions() == {}


def test_checkpoint_accounting_does_not_reset_real_failure_attempts(tmp_path):
    queue = DeviceDayQueue(tmp_path/'queue-understanding.sqlite3', scheduling={'enabled': True})
    item = record('long-recording', 1)
    queue.enqueue(item, 'v1')
    queue.grant_admission(item)
    with queue.connect() as db:
        db.execute('UPDATE recordings SET attempts=2')
    queue.claim('owner', retry=True, max_attempts=3)
    queue.finish('owner', item['recording_id'], {'status': 'scheduled_checkpoint'}, 12)
    with queue.connect() as db:
        row = db.execute('SELECT status,attempts FROM recordings').fetchone()
    assert tuple(row) == ('queued', 2)
