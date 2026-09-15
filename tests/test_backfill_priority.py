import json
from datetime import datetime
from zoneinfo import ZoneInfo

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_schedule import priority_date, refresh_queue_priorities, scheduling_record


def test_focus_is_shared_by_sql_and_inventory_without_resetting_evidence(tmp_path):
    zone = ZoneInfo('Asia/Shanghai')
    today = datetime.now(zone).date().isoformat()
    q = DeviceDayQueue(tmp_path / 'Queue.sqlite3')
    records = []
    for identity, day in [('other', '2020-08-26'), ('focus', '2020-08-27'), ('live', today)]:
        record = {'recording_id': identity, 'configured_role': 'first_person',
                  'recording_start_us': int(datetime.fromisoformat(day).replace(tzinfo=zone).timestamp()*1e6)}
        q.enqueue(record, 'unchanged-model-and-input')
        records.append(record)
    with q.connect() as db:
        before = [dict(r) for r in db.execute('select * from recordings order by recording_id')]
    refresh_queue_priorities(q, '2020-08-27')
    with q.connect() as db:
        after = [dict(r) for r in db.execute('select * from recordings order by recording_id')]
    for left, right in zip(before, after, strict=True):
        assert left | {'payload': right['payload']} == right
        original = next(r for r in records if r['recording_id'] == right['recording_id'])
        assert json.loads(right['payload'])['processing_priority'] == scheduling_record(original, focus_date='2020-08-27')['processing_priority']
    for expected in ('live', 'focus', 'other'):
        row = q.claim(expected)
        assert row['recording_id'] == expected
        q.finish(expected, expected, {'status': 'completed'}, 1)


def test_priority_file_is_optional_and_validated(tmp_path):
    assert priority_date(tmp_path) is None
    path = tmp_path / 'BackfillPriority.json'
    path.write_text('{"date":"2026-08-27"}')
    assert priority_date(tmp_path) == '2026-08-27'
    path.write_text('{"date":"2026-02-31"}')
    assert priority_date(tmp_path) is None
