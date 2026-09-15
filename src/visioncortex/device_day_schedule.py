"""Live-day work precedes historical backfill without changing media identity."""
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo


def priority_date(runtime_root):
    """Read the operator's backfill preference without altering input identity."""
    try:
        value = json.loads((runtime_root / 'BackfillPriority.json').read_text())['date']
        return date.fromisoformat(value).isoformat()
    except (OSError, ValueError, KeyError, TypeError):
        return None


def scheduling_record(record, today=None, focus_date=None):
    zone = ZoneInfo('Asia/Shanghai')
    today = today or datetime.now(zone).date().isoformat()
    captured = datetime.fromtimestamp(record.get('recording_start_us', 0) / 1e6, zone).date().isoformat()
    priority = 0 if captured >= today else 1 if not focus_date or captured == focus_date else 2
    return record | {'processing_priority': priority}


def refresh_queue_priorities(queue, focus_date=None):
    """Reclassify old pending rows without resetting results, attempts or leases."""
    import time
    zone = ZoneInfo('Asia/Shanghai')
    midnight = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = round(midnight.timestamp()*1e6)
    focus_start = (round(datetime.fromisoformat(focus_date).replace(tzinfo=zone).timestamp()*1e6)
                   if focus_date else -1)
    expression = ("CASE WHEN COALESCE(json_extract(payload,'$.recording_start_us'),0)>=? THEN 0 "
                  "WHEN ?=-1 OR json_extract(payload,'$.recording_start_us') BETWEEN ? AND ? THEN 1 ELSE 2 END")
    values = (cutoff, focus_start, focus_start, focus_start + 86400000000 - 1)
    with queue.connect() as db:
        db.execute("UPDATE recordings SET payload=json_set(payload,'$.processing_priority'," + expression + ") "
                   "WHERE (status!='running' OR COALESCE(lease_until,0)<?) AND "
                   "COALESCE(json_extract(payload,'$.processing_priority'),-1) != " + expression,
                   (*values, time.time(), *values))
