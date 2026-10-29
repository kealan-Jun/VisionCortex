ALTER TABLE processing_job ADD COLUMN input_fingerprint text NOT NULL DEFAULT '';
DO $$ DECLARE item record; BEGIN
 FOR item IN SELECT conname FROM pg_constraint WHERE conrelid='processing_job'::regclass AND contype='u' LOOP
  EXECUTE format('ALTER TABLE processing_job DROP CONSTRAINT %I',item.conname);
 END LOOP;
END $$;
ALTER TABLE processing_job ADD CONSTRAINT processing_job_execution_identity
 UNIQUE(customer_id,capture_id,worker_kind,model_revision,rule_fingerprint,input_fingerprint);
ALTER TABLE processing_job DROP CONSTRAINT processing_job_status_check;
ALTER TABLE processing_job ADD CONSTRAINT processing_job_status_check
 CHECK(status IN ('queued','waiting_dependencies','running','completed','failed','blocked_model','needs_review'));
CREATE TABLE processing_dependency (
 customer_id uuid NOT NULL, job_id uuid NOT NULL, parent_id uuid NOT NULL,
 PRIMARY KEY(customer_id,job_id,parent_id), CHECK(job_id<>parent_id),
 FOREIGN KEY(customer_id,job_id) REFERENCES processing_job(customer_id,id),
 FOREIGN KEY(customer_id,parent_id) REFERENCES processing_job(customer_id,id)
);
