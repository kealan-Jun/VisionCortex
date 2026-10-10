"""Live-day work precedes historical backfill without changing media identity."""
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo


def processing_cutoff(settings):
    value = settings.get('process_since_us')
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError('device_day.process_since_us must be a positive Unix microsecond timestamp')
    return value


def in_processing_scope(settings, record):
    """Include captures overlapping the fixed cutover, never late old uploads."""
    cutoff = processing_cutoff(settings)
    return cutoff is None or max(record.get('recording_start_us') or 0,
                                 record.get('recording_end_us') or 0) >= cutoff


def priority_date(runtime_root):
    """Read the operator's backfill preference without altering input identity."""
    try:
        value = json.loads((runtime_root / 'BackfillPriority.json').read_text())['date']
        return date.fromisoformat(value).isoformat()
    except (OSError, ValueError, KeyError, TypeError):
        return None


def scheduling_record(record, today=None, focus_date=None, live_priority_seconds=14400,
                      now_us=None, *, latest_first=False):
    if latest_first:
        # One time-descending lane across dates. A saved backfill focus cannot
        # demote yesterday's newest slice at midnight or after four hours.
        return record | {'processing_priority': -1}
    zone = ZoneInfo('Asia/Shanghai')
    today = today or datetime.now(zone).date().isoformat()
    start_us = int(record.get('recording_start_us') or 0)
    captured = datetime.fromtimestamp(start_us / 1e6, zone).date().isoformat()
    # Keep a short live lane ahead of the day's older backlog.  A newly closed
    # slice must not wait behind hours of historical work, while the normal
    # same-day lane remains available once the live lane drains.
    now_us = int(now_us if now_us is not None else datetime.now(zone).timestamp() * 1e6)
    live_cutoff = now_us - max(0, int(live_priority_seconds)) * 1_000_000
    if captured >= today:
        priority = -1 if start_us >= live_cutoff else 0
    else:
        priority = 1 if not focus_date or captured == focus_date else 2
    return record | {'processing_priority': priority}


def refresh_queue_priorities(queue, focus_date=None, live_priority_seconds=14400, *, latest_first=False):
    """Reclassify old pending rows without resetting results, attempts or leases."""
    import time
    if latest_first:
        with queue.connect() as db:
            db.execute("UPDATE recordings SET payload=json_set(payload,'$.processing_priority',-1) "
                       "WHERE status!='completed' AND (status!='running' OR COALESCE(lease_until,0)<?) "
                       "AND COALESCE(json_extract(payload,'$.processing_priority'),0)!=-1", (time.time(),))
        return
    zone = ZoneInfo('Asia/Shanghai')
    midnight = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = round(midnight.timestamp()*1e6)
    focus_start = (round(datetime.fromisoformat(focus_date).replace(tzinfo=zone).timestamp()*1e6)
                   if focus_date else -1)
    live_cutoff = round((time.time() - max(0, int(live_priority_seconds))) * 1e6)
    expression = ("CASE WHEN COALESCE(json_extract(payload,'$.recording_start_us'),0)>=? THEN -1 "
                  "WHEN COALESCE(json_extract(payload,'$.recording_start_us'),0)>=? THEN 0 "
                  "WHEN ?=-1 OR json_extract(payload,'$.recording_start_us') BETWEEN ? AND ? THEN 1 ELSE 2 END")
    values = (live_cutoff, cutoff, focus_start, focus_start, focus_start + 86400000000 - 1)
    with queue.connect() as db:
        db.execute("UPDATE recordings SET payload=json_set(payload,'$.processing_priority'," + expression + ") "
                   "WHERE (status!='running' OR COALESCE(lease_until,0)<?) AND "
                   "COALESCE(json_extract(payload,'$.processing_priority'),-1) != " + expression,
                   (*values, time.time(), *values))
