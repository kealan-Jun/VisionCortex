from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from visioncortex.config import load_config
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_schedule import (
    in_processing_scope,
    refresh_queue_priorities,
    scheduling_record,
)


def test_production_drains_live_then_preserved_history_and_prioritizes_new_arrivals(tmp_path):
    config = load_config(Path('configs/rtx3090ti-ubuntu-production.yaml'))
    settings = config['device_day']
    current = datetime.now(ZoneInfo('Asia/Shanghai'))
    # Noon keeps the recent-live and older-same-day cases distinct at any test time.
    current = current.replace(hour=12, minute=0, second=0, microsecond=0)
    queue = DeviceDayQueue(tmp_path / 'queue.sqlite3')

    def enqueue(name, start):
        record = {
            'recording_id': name, 'camera_key': 'camera',
            'configured_role': 'first_person',
            'recording_start_us': round(start.timestamp() * 1e6),
            'recording_end_us': round((start + timedelta(minutes=1)).timestamp() * 1e6),
        }
        assert in_processing_scope(settings, record)
        queue.enqueue(scheduling_record(
            record, today=current.date().isoformat(),
            live_priority_seconds=settings['live_priority_seconds'],
            now_us=round(current.timestamp() * 1e6),
        ), name + '-original-revision')

    enqueue('history', current - timedelta(days=30))
    enqueue('earlier_today', current - timedelta(hours=5))
    enqueue('recent', current - timedelta(minutes=30))
    enqueue('newest', current - timedelta(minutes=15))

    def finish_next(expected):
        record = queue.claim('worker', camera_serial=True)
        assert record['recording_id'] == expected
        queue.finish('worker', expected, {'status': 'completed'}, 1)

    finish_next('newest')
    finish_next('recent')
    finish_next('earlier_today')
    enqueue('arrival', current - timedelta(minutes=1))
    finish_next('arrival')
    finish_next('history')
    assert queue.claim('worker', camera_serial=True) is None
    # Reclassification must preserve completed work, provenance and retry history.
    refresh_queue_priorities(queue, live_priority_seconds=settings['live_priority_seconds'])
    with queue.connect() as db:
        rows = list(db.execute('SELECT recording_id,revision,status,attempts FROM recordings'))
    assert len(rows) == 5
    assert all(row['status'] == 'completed' and row['attempts'] == 1
               and row['revision'] == row['recording_id'] + '-original-revision' for row in rows)
