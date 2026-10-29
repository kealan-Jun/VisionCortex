"""Actual day throughput; incomplete coverage and unknown quality cannot pass."""
from collections import Counter
from decimal import Decimal

from .domain import clean
from .timeline import day_bounds


def measured(values):
    values=list(values)
    return dict(samples=len(values),total_seconds=sum(values,Decimal(0)) if values else None,
                maximum_seconds=max(values) if values else None)


class TimelineMetrics:
    def __init__(self,runtime):
        self.runtime=runtime
        self.core,self.customer=runtime.timeline.core,runtime.timeline.customer

    def day(self,day):
        start,end=day_bounds(day)
        selected=self.runtime.timeline.policies.video_input_unit=='complete_retained_clip'
        with self.core.db.transaction(readonly=True) as conn:
            tasks=conn.execute("SELECT t.*,m.duration_seconds,m.start_at,m.end_at,EXISTS(SELECT 1 FROM media_derivation d WHERE d.customer_id=t.customer_id AND d.child_id=t.media_id AND d.transform->>'schema_version'='vc-lab-retained-clip/1') AS retained_clip FROM hj_timeline_task t JOIN hj_timeline_media m USING(customer_id,media_id) WHERE t.customer_id=%s AND m.start_at<%s AND m.end_at>=%s ORDER BY t.created_at,t.id",(self.customer,end,start)).fetchall()
            ids=[t['id'] for t in tasks]
            attempts=conn.execute('SELECT * FROM hj_timeline_task_attempt WHERE customer_id=%s AND task_id=ANY(%s) ORDER BY started_at,token',(self.customer,ids)).fetchall() if ids else []
            timings=conn.execute('SELECT * FROM hj_timeline_timing WHERE customer_id=%s AND attempt_token=ANY(%s) ORDER BY started_at,stage,window_index',(self.customer,[a['token'] for a in attempts])).fetchall() if attempts else []
            checkpoints=conn.execute('SELECT task_id,window_index,execution_receipt FROM hj_timeline_window_result WHERE customer_id=%s AND task_id=ANY(%s) ORDER BY task_id,window_index',(self.customer,ids)).fetchall() if ids else []
            publication=conn.execute('SELECT * FROM hj_timeline_publication_timing WHERE customer_id=%s AND day=%s ORDER BY started_at,id',(self.customer,start.date())).fetchall()
            report=conn.execute('SELECT revision,created_at FROM hj_daily_report WHERE customer_id=%s AND day=%s ORDER BY revision DESC LIMIT 1',(self.customer,start.date())).fetchone()
        awaiting=[t for t in tasks if selected and t['kind']=='video_understanding' and not t['retained_clip']]
        awaiting_ids={t['id'] for t in awaiting}
        work=[t for t in tasks if t['id'] not in awaiting_ids]
        # Every analysis revision stays visible. Duration denominator is unique
        # selected media, never inflated by raw+clip copies or OCR/ASR tasks.
        videos={t['media_id']:t['duration_seconds'] for t in work if t['kind']=='video_understanding'}
        duration=sum(videos.values(),Decimal(0))
        ready=bool(work) and all(t['status']=='completed' for t in work)
        published=bool(report and any(p['outcome']=='completed' and p['revision']==report['revision'] for p in publication))
        first=min((a['started_at'] for a in attempts),default=None)
        finished=[a['finished_at'] for a in attempts if a['finished_at']]
        finished += [p['finished_at'] for p in publication]
        last=max(finished,default=None)
        observed=Decimal(str((last-first).total_seconds())) if first and last else None
        # No live-clock estimate: a partial batch has an observed span, not an ETA.
        complete_wall=observed if ready and published and not awaiting else None
        by_stage={stage:measured(r['elapsed_seconds'] for r in timings if r['stage']==stage)
                  for stage in ['queue_wait','provider_preflight','window_execution','checkpoint_commit','result_commit','attempt_wall']}
        receipts=[i for c in checkpoints for i in c['execution_receipt'].get('invocations',[])]
        source=[];prepare=[];post=[];http=[]
        for receipt in receipts:
            if 'execution' in receipt and 'elapsed_seconds' in receipt['execution']:
                http.append(Decimal(str(receipt['execution']['elapsed_seconds'])))
            derivations=receipt.get('derivations',[]) + ([receipt['derivation']] if 'derivation' in receipt else [])
            for derivation in derivations:
                d=derivation.get('timing',{})
                for key,out in [('source_locate_and_verify_seconds',source),('prepare_seconds',prepare),('post_source_verify_seconds',post)]:
                    if key in d:out.append(Decimal(str(d[key])))
        by_stage.update(source_locate_and_verify=measured(source),media_preparation=measured(prepare),
                        post_source_verify=measured(post),model_http=measured(http),
                        publication_wall=measured(p['elapsed_seconds'] for p in publication))
        counts=Counter((t['kind'],t['status']) for t in work)
        return clean(dict(schema_version='vc-lab-timeline-metrics/1',day=str(start.date()),timezone='Asia/Shanghai',
            tasks=[dict(kind=k,status=s,count=n) for (k,s),n in sorted(counts.items())],
            awaiting_clip_selection=len(awaiting),selected_video_count=len(videos),selected_video_seconds=duration,
            all_selected_analysis_tasks_completed=ready,all_known_tasks_completed=ready and not awaiting,
            latest_report_published=published,attempts=len(attempts),
            retries=sum(max(0,t['attempts']-1) for t in tasks),stage_times=by_stage,
            stages_overlap_do_not_sum=True,observed_batch_started_at=first,observed_batch_last_finished_at=last,
            observed_batch_span_seconds=observed,complete_batch_wall_seconds=complete_wall,remaining_eta_seconds=None,
            publication_attempts=publication,coverage='indexed_inputs_only_not_verified_whole_day',quality='NOT_PROVEN',
            nightly=[dict(hours=h,allowed_end_to_end_rtf=Decimal(h*3600)/duration if duration else None,
                          measured_processing_within_budget=(complete_wall<=h*3600) if complete_wall is not None else None,
                          acceptance='NOT_DEMONSTRATED') for h in [8,10,12]],
            timing_scope='attempts_through_report_publication_excludes_unmeasured_ingest_and_selection',
            next_shift_time=None,notes=['No shift time, whole-day intake completeness or quality acceptance supplied.',
                                       'Observed spans include gaps/retries; sums of overlapping phases are not batch wall time.']))
