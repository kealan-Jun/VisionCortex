ALTER TABLE hj_activity_association_revision ADD COLUMN revision integer;
WITH ordered AS (
 SELECT customer_id,id,row_number() OVER(PARTITION BY customer_id,activity_id ORDER BY created_at,id) AS n
 FROM hj_activity_association_revision
) UPDATE hj_activity_association_revision r SET revision=o.n FROM ordered o WHERE r.customer_id=o.customer_id AND r.id=o.id;
ALTER TABLE hj_activity_association_revision ALTER COLUMN revision SET NOT NULL;
CREATE UNIQUE INDEX hj_association_revision ON hj_activity_association_revision(customer_id,activity_id,revision);
CREATE INDEX hj_task_waiting ON hj_timeline_task(customer_id,kind,status,created_at);
CREATE INDEX hj_activity_tag ON hj_activity(customer_id,tag_id);
CREATE INDEX hj_activity_by_run ON hj_activity_run(customer_id,run_id,activity_id);
CREATE TABLE hj_timeline_pending_association (
 customer_id uuid NOT NULL, activity_id uuid NOT NULL,
 PRIMARY KEY(customer_id,activity_id),
 FOREIGN KEY(customer_id,activity_id) REFERENCES hj_activity(customer_id,id)
);
ALTER TABLE hj_experiment_binding ADD COLUMN business_snapshot jsonb NOT NULL DEFAULT '{}';
