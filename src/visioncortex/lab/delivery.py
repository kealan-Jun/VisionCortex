"""Bind native understanding/report jobs to exact completed parent executions."""
from psycopg.types.json import Jsonb

from .domain import clean, fingerprint, ident
from .processing import ModelRegistry, ModelUnavailable
from .rules import Rules


class NativeDelivery:
    def __init__(self,core,settings):
        self.core,self.settings=core,settings

    def tick(self,limit=20):
        if self.settings.processing_profile == 'timeline-v1':
            return dict(created=0, dependencies_ready=0, profile='timeline-v1')
        rules=Rules().identity()
        try:
            revision=ModelRegistry(self.settings).descriptor('understanding')['revision']
        except ModelUnavailable:
            revision='customer-model-unconfigured'
        created=0
        with self.core.db.transaction() as conn:
            # Exclude already scheduled exact parents before LIMIT, so a large backlog
            # cannot be starved by the same newest twenty completed vision jobs.
            videos=conn.execute("""
                WITH candidates AS (
                  SELECT v.id,v.capture_id,c.source_id,c.source_capture_id,i.payload AS receipt,
                    a.id AS audio_id,
                    (i.payload->'native_record'->'audio'->>'status'='provided'
                     OR COALESCE((i.payload->'native_record'->>'embedded_audio')::boolean,false)) AS needs_audio
                  FROM processing_job v
                  JOIN capture c ON c.customer_id=v.customer_id AND c.id=v.capture_id
                  JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                  LEFT JOIN LATERAL (
                    SELECT j.id FROM processing_job j JOIN capture ac ON ac.customer_id=j.customer_id AND ac.id=j.capture_id
                    WHERE j.customer_id=v.customer_id AND j.worker_kind='asr' AND j.rule_fingerprint=v.rule_fingerprint AND ac.source_id=c.source_id
                      AND (ac.source_capture_id=c.source_capture_id||':audio' OR (ac.id=c.id AND COALESCE((i.payload->'native_record'->>'embedded_audio')::boolean,false)))
                    ORDER BY j.created_at DESC,j.id DESC LIMIT 1
                  ) a ON true
                  WHERE v.customer_id=%s AND v.worker_kind='vision' AND v.status='completed' AND v.rule_fingerprint=%s AND i.payload ? 'native_record'
                ) SELECT v.* FROM candidates v
                WHERE (NOT COALESCE(v.needs_audio,false) OR v.audio_id IS NOT NULL)
                  AND NOT EXISTS (
                    SELECT 1 FROM processing_job u
                    WHERE u.customer_id=%s AND u.capture_id=v.capture_id AND u.worker_kind='understanding' AND u.model_revision=%s AND u.rule_fingerprint=%s
                      AND EXISTS(SELECT 1 FROM processing_dependency d WHERE d.customer_id=u.customer_id AND d.job_id=u.id AND d.parent_id=v.id)
                      AND (NOT COALESCE(v.needs_audio,false) OR EXISTS(SELECT 1 FROM processing_dependency d WHERE d.customer_id=u.customer_id AND d.job_id=u.id AND d.parent_id=v.audio_id))
                      AND (SELECT count(*) FROM processing_dependency d WHERE d.customer_id=u.customer_id AND d.job_id=u.id)=CASE WHEN COALESCE(v.needs_audio,false) THEN 2 ELSE 1 END
                  )
                ORDER BY v.id LIMIT %s
                """,(self.core.customer,rules['sha256'],self.core.customer,revision,rules['sha256'],limit)).fetchall()
            for video in videos:
                parents=[video['id']]
                if video['needs_audio']:
                    parents.append(video['audio_id'])
                input_id=fingerprint(clean(sorted(map(str,parents))))
                job=conn.execute("INSERT INTO processing_job(customer_id,id,capture_id,worker_kind,model_revision,status,rule_identity,rule_fingerprint,input_fingerprint) VALUES(%s,%s,%s,'understanding',%s,'waiting_dependencies',%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id",(self.core.customer,ident(),video['capture_id'],revision,Jsonb(rules),rules['sha256'],input_id)).fetchone()
                if job:
                    for parent in parents:
                        conn.execute('INSERT INTO processing_dependency(customer_id,job_id,parent_id) VALUES(%s,%s,%s)',(self.core.customer,job['id'],parent))
                    created+=1
            ready=conn.execute("UPDATE processing_job j SET status='queued',error_code=NULL WHERE j.customer_id=%s AND j.status='waiting_dependencies' AND NOT EXISTS(SELECT 1 FROM processing_dependency d JOIN processing_job p ON p.customer_id=d.customer_id AND p.id=d.parent_id WHERE d.customer_id=j.customer_id AND d.job_id=j.id AND p.status<>'completed')",(self.core.customer,)).rowcount
        return dict(created=created,dependencies_ready=ready)
