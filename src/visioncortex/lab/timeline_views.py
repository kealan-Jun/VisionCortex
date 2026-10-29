"""Evidence indexes, one-event photo bursts, immutable daily report revisions."""
from __future__ import annotations

from psycopg.types.json import Jsonb

from .domain import Missing, clean, fingerprint, ident, instant
from .timeline import Timeline, day_bounds, offset_time
from .ocr_fields import consensus
from .context_views import context_evidence
from .person_report import person_timelines, empty_person


class TimelineViews:
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.core, self.customer = timeline.core, timeline.customer

    def _activities(self, conn, start=None, end=None, run_id=None, activity_ids=None):
        rows = conn.execute("SELECT a.*,COALESCE((SELECT array_agg(r.run_id ORDER BY r.run_id) FROM hj_activity_run r WHERE r.customer_id=a.customer_id AND r.activity_id=a.id),'{}') AS run_ids FROM hj_activity a WHERE a.customer_id=%s AND (%s::timestamptz IS NULL OR a.end_at>%s OR (a.start_at=a.end_at AND a.start_at=%s)) AND (%s::timestamptz IS NULL OR a.start_at<%s) AND (%s::uuid IS NULL OR EXISTS(SELECT 1 FROM hj_activity_run r WHERE r.customer_id=a.customer_id AND r.activity_id=a.id AND r.run_id=%s)) AND (%s::uuid[] IS NULL OR a.id=ANY(%s)) ORDER BY a.start_at NULLS LAST,a.id", (self.customer, start, start, start, end, end, ident(run_id) if run_id else None, ident(run_id) if run_id else None, activity_ids, activity_ids)).fetchall()
        ids = [row['id'] for row in rows]
        revisions = conn.execute('SELECT * FROM hj_activity_association_revision WHERE customer_id=%s AND activity_id=ANY(%s) ORDER BY activity_id,revision', (self.customer, ids)).fetchall() if ids else []
        by_activity = {}
        for revision in revisions:
            by_activity.setdefault(revision['activity_id'], []).append(revision)
        evidence = self._evidence(conn, list({eid for row in rows for eid in row['evidence_ids']}))
        by_evidence = {item['id']: item for item in evidence}
        for row in rows:
            row['association_revisions'] = by_activity.get(row['id'], [])
            if row['association_revisions']:
                row['current_association_status'] = row['association_revisions'][-1]['association_status']
                row['current_label_resolution'] = row['association_revisions'][-1]['label_resolution']
            else:
                row['current_association_status'] = row['association_status']
                row['current_label_resolution'] = row['label_resolution']
            row['evidence'] = [by_evidence[eid] for eid in row['evidence_ids']]
            row['time_interval_semantics'] = 'evidence_extent_not_continuous_activity_proof'
        return rows

    def _untimed_temporal(self, conn):
        ids = [row['id'] for row in conn.execute("SELECT id FROM hj_activity WHERE customer_id=%s AND start_at IS NULL AND payload->>'temporal_world_clock_status'='uncalibrated' ORDER BY created_at DESC,id LIMIT 101", (self.customer,)).fetchall()]
        return dict(activities=self._activities(conn, activity_ids=ids[:100]) if ids else [],
                    truncated=len(ids)>100, attribution='customer_pending_time_not_assigned_to_requested_day_or_range')

    def _untimed_values(self, conn):
        from .value_candidates import project
        ids = [r['id'] for r in conn.execute("SELECT id FROM hj_activity WHERE customer_id=%s AND start_at IS NULL AND payload ? 'value_candidate' ORDER BY created_at DESC,id LIMIT 101", (self.customer,)).fetchall()]
        return dict(candidates=project(self._activities(conn, activity_ids=ids[:100])) if ids else [],
                    truncated=len(ids)>100, attribution='customer_pending_time_not_assigned_to_requested_day_or_range')

    def _evidence(self, conn, ids):
        return conn.execute('SELECT e.*,m.kind,m.stream_key,m.session_key,m.start_at AS media_start_at,m.duration_seconds,a.verified_sha256,a.evidence_status FROM evidence_ref e JOIN hj_timeline_media m USING(customer_id,media_id) JOIN media_asset a ON a.customer_id=e.customer_id AND a.id=e.media_id WHERE e.customer_id=%s AND e.id=ANY(%s) ORDER BY e.media_id,e.start_seconds,e.id', (self.customer, ids)).fetchall() if ids else []

    def clip_index(self, run_id, *, include_low_priority=False):
        with self.core.db.transaction(readonly=True) as conn:
            self.timeline._run(conn, run_id)
            activities = self._activities(conn, run_id=run_id)
            clips = []
            for activity in activities:
                if activity['priority'] == 'low' and not include_low_priority:
                    continue
                for evidence in activity['evidence']:
                    if evidence['kind'] != 'video':
                        continue
                    clips.append(dict(activity_id=activity['id'], kind=activity['kind'], priority=activity['priority'],
                                      media_id=evidence['media_id'], evidence_id=evidence['id'],
                                      start_seconds=evidence['start_seconds'], end_seconds=evidence['end_seconds'],
                                      start_at=offset_time(evidence['media_start_at'], evidence['start_seconds']),
                                      end_at=offset_time(evidence['media_start_at'], evidence['end_seconds']),
                                      stream_key=evidence['stream_key'], sha256=evidence['verified_sha256'],
                                      media_url='/internal/media/' + str(evidence['media_id'])))
            return clean(dict(run_id=run_id, clips=clips, raw_recordings_preserved=True, physical_concatenation_required=False))

    def trace(self, start, end):
        start, end = instant(start), instant(end)
        if end <= start or (end - start).total_seconds() > 7 * 86400:
            raise ValueError('Trace range must be positive and at most seven days')
        with self.core.db.transaction(readonly=True) as conn:
            media = conn.execute("SELECT m.*,EXISTS(SELECT 1 FROM media_derivation d WHERE d.customer_id=m.customer_id AND d.child_id=m.media_id AND d.transform->>'schema_version'='vc-lab-retained-clip/1') AS retained_derivative FROM hj_timeline_media m WHERE m.customer_id=%s AND m.start_at<%s AND m.end_at>=%s ORDER BY m.stream_key,m.start_at,m.media_id", (self.customer, end, start)).fetchall()
            streams = {}
            for item in media:
                streams.setdefault(item['stream_key'], []).append(item)
            coverage = []
            for stream, items in streams.items():
                cursor, gaps, overlaps = start, [], []
                for item in items:
                    lo, hi = max(start, item['start_at']), min(end, item['end_at'])
                    if item['kind'] == 'photo' or item['retained_derivative']:
                        continue  # A derived view does not add original recording coverage.
                    if lo > cursor:
                        gaps.append(dict(start_at=cursor, end_at=lo, status='no_indexed_recording'))
                    if lo < cursor:
                        overlaps.append(dict(start_at=lo, end_at=min(cursor, hi)))
                    cursor = max(cursor, hi)
                if cursor < end:
                    gaps.append(dict(start_at=cursor, end_at=end, status='no_indexed_recording'))
                coverage.append(dict(stream_key=stream, gaps=gaps, overlaps=overlaps))
            unknown = conn.execute('SELECT media_id,stream_key FROM hj_timeline_media WHERE customer_id=%s AND start_at IS NULL ORDER BY media_id LIMIT 100', (self.customer,)).fetchall()
            from .value_candidates import project
            activities = self._activities(conn, start, end)
            return clean(dict(start_at=start, end_at=end, media=media, activities=activities,
                              video_value_candidates=project(activities), unknown_time_video_value_candidates=self._untimed_values(conn),
                              indexed_coverage=coverage, unknown_time_media=unknown,
                              unknown_time_temporal_activities=self._untimed_temporal(conn),
                              context_evidence=context_evidence(conn,self.customer,start,end),
                              coverage='unknown' if not media else 'partial',
                              coverage_reason='indexed_files_do_not_prove_capture_or_upload_completeness'))

    def readout(self, session_id):
        with self.core.db.transaction(readonly=True) as conn:
            return clean(self._readout(conn, ident(session_id)))

    def _readout(self, conn, session_id):
        session = conn.execute('SELECT * FROM capture_session WHERE customer_id=%s AND id=%s', (self.customer, session_id)).fetchone()
        if not session:
            raise Missing('Photo burst outside customer scope')
        images = conn.execute("SELECT m.*,a.verified_sha256 FROM hj_timeline_media m JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id WHERE m.customer_id=%s AND c.session_id=%s AND m.kind='photo' ORDER BY m.start_at,m.media_id", (self.customer, session_id)).fetchall()
        if not images:
            raise Missing('Session has no timeline photos')
        image_ids = [x['media_id'] for x in images]
        readings = conn.execute("SELECT DISTINCT a.* FROM hj_activity a JOIN evidence_ref e ON e.customer_id=a.customer_id AND e.id=ANY(a.evidence_ids) WHERE a.customer_id=%s AND a.source_type='ocr' AND e.media_id=ANY(%s) ORDER BY a.created_at,a.id", (self.customer, image_ids)).fetchall()
        evidence = self._evidence(conn, list({eid for row in readings for eid in row['evidence_ids']}))
        evidence_media = {row['id']: row['media_id'] for row in evidence}
        latest, superseded = {}, []
        for reading in readings:
            reading['source_photo_ids'] = sorted({str(evidence_media[eid]) for eid in reading['evidence_ids'] if evidence_media[eid] in image_ids})
            if reading['payload'].get('task_id'):
                for photo in reading['source_photo_ids']:
                    rank = (reading['payload'].get('frame_registration_revision') or 0, reading['created_at'], str(reading['id']))
                    if photo not in latest or rank > latest[photo][0]:
                        latest[photo] = (rank, reading['id'])
        fields = []
        for reading in readings:
            if reading['payload'].get('task_id') and not any(latest[p][1] == reading['id'] for p in reading['source_photo_ids']):
                superseded.append(reading['id'])
                continue
            for value in reading['payload'].get('fields', []):
                fields.append(dict(value, activity_id=str(reading['id']), evidence_ids=clean(reading['evidence_ids']),
                                   instrument_id=clean(reading['instrument_id']), source_photo_ids=reading['source_photo_ids']))
        # Do not merge values from different instruments in a shared photo burst.
        instruments = sorted({f['instrument_id'] for f in fields}, key=str)
        grouped = [dict(group, instrument_id=instrument) for instrument in instruments
                   for group in consensus([f for f in fields if f['instrument_id'] == instrument])]
        completed = conn.execute("SELECT count(*) AS n FROM (SELECT DISTINCT ON(media_id) status FROM hj_timeline_task WHERE customer_id=%s AND media_id=ANY(%s) AND kind='ocr' ORDER BY media_id,created_at DESC,id DESC) latest WHERE status='completed'", (self.customer, image_ids)).fetchone()['n']
        expected = session['metadata'].get('expected_photos')
        for group in grouped:
            group['source_photo_count'] = len({p for c in group['candidates'] for p in c['source_photo_ids']})
            if group['status'] == 'consistent_candidate' and group['source_photo_count'] < (expected or len(images)):
                group['status'] = 'partial_candidate'
        return dict(readout_id=session_id, source_burst_id=session['source_session_id'], expected_photos=expected,
                    received_photos=len(images), ocr_completed_photos=completed, images=images, fields=grouped,
                    superseded_observation_ids=superseded, history_preserved=True,
                    burst_status='complete' if expected is not None and expected == len(images) and completed == len(images) else 'partial_or_unknown',
                    external_confirmation='not_provided', business_write_performed=False)

    def _report(self, conn, day):
        start, end = day_bounds(day)
        activities = self._activities(conn, start, end)
        bindings = conn.execute('SELECT * FROM hj_experiment_binding WHERE customer_id=%s AND occurred_at>=%s AND occurred_at<%s ORDER BY occurred_at,id', (self.customer, start, end)).fetchall()
        run_ids = sorted({rid for a in activities for rid in a['run_ids']} | {b['run_id'] for b in bindings}, key=str)
        experiments = []
        runs = conn.execute('SELECT r.*,e.code,e.metadata AS registration FROM hj_experiment_record r JOIN entity e ON e.customer_id=r.customer_id AND e.id=r.id WHERE r.customer_id=%s AND r.id=ANY(%s)', (self.customer, run_ids)).fetchall() if run_ids else []
        by_run = {r['id']: r for r in runs}
        all_relations = conn.execute('SELECT * FROM hj_run_relation WHERE customer_id=%s AND (parent_id=ANY(%s) OR child_id=ANY(%s)) ORDER BY parent_id,child_id', (self.customer, run_ids, run_ids)).fetchall() if run_ids else []
        for rid in run_ids:
            run = by_run[rid]
            relations = [r for r in all_relations if rid in {r['parent_id'], r['child_id']}]
            experiments.append(dict(record=run, activities=[a for a in activities if rid in a['run_ids']],
                                    bindings=[b for b in bindings if b['run_id'] == rid], relations=relations,
                                    instrument_completion_time=None, result_file_status='not_collected',
                                    closure_is_not_instrument_completion=True))
        sessions = conn.execute("SELECT DISTINCT c.session_id FROM hj_timeline_media m JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id WHERE m.customer_id=%s AND m.kind='photo' AND m.start_at>=%s AND m.start_at<%s AND c.session_id IS NOT NULL ORDER BY c.session_id", (self.customer, start, end)).fetchall()
        pending_ids=[r['id'] for r in conn.execute('SELECT id FROM hj_activity WHERE customer_id=%s AND start_at IS NULL ORDER BY created_at DESC,id LIMIT 101',(self.customer,)).fetchall()]
        pending=self._activities(conn,activity_ids=pending_ids) if pending_ids else []
        people=person_timelines(conn,self.core,activities,runs,pending)
        jobs = conn.execute('SELECT t.kind,t.status,count(*) AS count FROM hj_timeline_task t JOIN hj_timeline_media m USING(customer_id,media_id) WHERE t.customer_id=%s AND m.start_at<%s AND m.end_at>=%s GROUP BY t.kind,t.status ORDER BY t.kind,t.status', (self.customer, end, start)).fetchall()
        from .value_candidates import project
        return clean(dict(schema_version='vc-lab-evidence-daily-report/1', day=str(start.date()), timezone='Asia/Shanghai',
                          experiments=experiments, experiment_count=len(experiments), activity_count=len(activities),
                          activity_count_basis='stored_observations_not_distinct_physical_actions',
                          analysis_jobs=jobs, person_timelines=people,
                          unassigned_activities=[a for a in activities if not a['run_ids']],
                          readouts=[self._readout(conn, s['session_id']) for s in sessions],
                          unknown_time_temporal_activities=self._untimed_temporal(conn),
                          video_value_candidates=project(activities), unknown_time_video_value_candidates=self._untimed_values(conn),
                          context_evidence=context_evidence(conn,self.customer,start,end),
                          quality='NOT_PROVEN', coverage='partial',
                          scope='evidence_recording_and_traceability', raw_evidence_preserved=True))

    def snapshot(self, day, conn=None):
        if conn is None:
            with self.core.db.transaction() as conn:
                self.timeline.lock(conn)
                return self.snapshot(day, conn)
        document = self._report(conn, day)
        digest = fingerprint(document)
        old = conn.execute('SELECT * FROM hj_daily_report WHERE customer_id=%s AND day=%s ORDER BY revision DESC LIMIT 1', (self.customer, day)).fetchone()
        if old and old['fingerprint'] == digest:
            return clean(dict(old, day=str(old['day'])))
        revision = conn.execute('SELECT COALESCE(MAX(revision),0)+1 AS n FROM hj_daily_report WHERE customer_id=%s AND day=%s', (self.customer, day)).fetchone()['n']
        row = conn.execute('INSERT INTO hj_daily_report(customer_id,day,revision,fingerprint,document) VALUES(%s,%s,%s,%s,%s) RETURNING *', (self.customer, day, revision, digest, Jsonb(document))).fetchone()
        return clean(dict(row, day=str(row['day'])))

    def person_day(self, person_id, day):
        """Same versioned daily snapshot; membership never attributes actions/speech."""
        with self.core.db.transaction() as conn:
            self.timeline.lock(conn)
            person=self.core._entity(conn,person_id,'person')
            snapshot=self.snapshot(day,conn)
            projection=snapshot['document']['person_timelines']
            selected=next((p for p in projection['persons'] if p['person_id']==str(person['id'])),
                          empty_person(person['id'],person['code']))
            return clean(dict(schema_version='vc-lab-person-day/1',day=snapshot['day'],timezone='Asia/Shanghai',
                snapshot_revision=snapshot['revision'],snapshot_fingerprint=snapshot['fingerprint'],
                projection_schema=projection['schema_version'],timeline=selected,coverage=projection['coverage'],
                pending_scope=projection.get('pending_scope'),
                projection_status=projection.get('status','partial_evidence')))

    def refresh_dirty(self, limit=20):
        reports = []
        with self.core.db.transaction() as conn:
            self.timeline.lock(conn)
            days = conn.execute('SELECT day FROM hj_timeline_dirty_day WHERE customer_id=%s ORDER BY day LIMIT %s FOR UPDATE', (self.customer, limit)).fetchall()
            for row in days:
                report = self.snapshot(str(row['day']), conn)
                reports.append(dict(day=report['day'], revision=report['revision']))
                conn.execute('DELETE FROM hj_timeline_dirty_day WHERE customer_id=%s AND day=%s', (self.customer, row['day']))
        return dict(reports=reports)
