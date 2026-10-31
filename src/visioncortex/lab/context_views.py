"""Bounded projections for existing trace/report, never a second fact writer."""
from .domain import clean
from .timeline_policy import ZONE
from datetime import timedelta


def mark_context_dirty(conn, customer, *times):
    for day in {value.astimezone(ZONE).date() for value in times if value is not None}:
        conn.execute('INSERT INTO hj_timeline_dirty_day(customer_id,day) VALUES(%s,%s) ON CONFLICT DO NOTHING', (customer, day))


def mark_context_interval_dirty(conn, customer, start, end):
    if start is None:
        return
    if end is None or end < start or (end-start).total_seconds() > 7*86400:
        raise ValueError('Bounded event interval required for report refresh')
    day, last = start.astimezone(ZONE).date(), end.astimezone(ZONE).date()
    while day <= last:
        conn.execute('INSERT INTO hj_timeline_dirty_day(customer_id,day) VALUES(%s,%s) ON CONFLICT DO NOTHING', (customer, day))
        day += timedelta(days=1)


def context_evidence(conn, customer, start, end):
    from .weighing_view import projection
    from .spoken_usage_view import projection as spoken_projection
    dynamic = conn.execute('''WITH latest AS (
      SELECT DISTINCT ON(media_id,source_session_key)
        id,media_id,source_session_key,revision,processed_at,
        snapshot->'physical_clock_known' AS physical_clock_known,
        snapshot->'focus_intervals' AS focus_intervals,
        snapshot->'status' AS status
      FROM hj_dynamic_context_revision WHERE customer_id=%s
      ORDER BY media_id,source_session_key,revision DESC)
      SELECT l.* FROM latest l WHERE l.physical_clock_known='false'::jsonb
        OR l.physical_clock_known IS NULL OR EXISTS (
          SELECT 1 FROM hj_dynamic_frame_event e WHERE e.customer_id=%s
            AND e.media_id=l.media_id AND e.source_session_key=l.source_session_key
            AND e.event_at>=%s AND e.event_at<%s)
      ORDER BY l.processed_at DESC,l.id LIMIT 21''', (customer, customer, start, end)).fetchall()
    for row in dynamic:
        intervals = row.get('focus_intervals') or []
        row['focus_intervals'] = intervals[:128]
        row['focus_intervals_truncated'] = len(intervals) > 128
        row['range_assignment'] = 'source_relative_only_not_assigned_to_requested_day' if row['physical_clock_known'] is not True else 'contains_events_in_requested_range; intervals_remain_source_relative'
        row['experiment_completion_implied'] = False
    materials = conn.execute('''WITH latest AS (
      SELECT DISTINCT ON(occurrence_key) * FROM hj_material_observation_revision
       WHERE customer_id=%s ORDER BY occurrence_key,revision DESC)
      SELECT * FROM latest WHERE event_start IS NULL OR
       (event_start<%s AND event_end>=%s)
      ORDER BY processed_at DESC,id LIMIT 101''', (customer, end, start)).fetchall()
    for row in materials:
        row['range_assignment'] = 'unknown_time_not_assigned_to_requested_day' if row['event_start'] is None else 'event_time_in_requested_range'
    return clean(dict(dynamic_contexts=dynamic[:20], dynamic_contexts_truncated=len(dynamic)>20,
                      material_observations=materials[:100], material_observations_truncated=len(materials)>100,
                      material_quantities_authority='Core.operate', recognition_consumption=False,
                      weighings=projection(conn,customer,start,end),
                      spoken_material_usage=spoken_projection(conn,customer,start,end)))
