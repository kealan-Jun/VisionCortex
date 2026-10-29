ALTER TABLE processing_job ADD COLUMN rule_identity jsonb;
ALTER TABLE processing_job ADD COLUMN rule_fingerprint text NOT NULL DEFAULT 'legacy-unrecorded';
ALTER TABLE processing_job DROP CONSTRAINT processing_job_customer_id_capture_id_worker_kind_model_rev_key;
ALTER TABLE processing_job ADD UNIQUE(customer_id,capture_id,worker_kind,model_revision,rule_fingerprint);
CREATE TABLE archive_checkpoint (
 customer_id uuid PRIMARY KEY REFERENCES customer_instance(id), cursor bigint NOT NULL DEFAULT 0
);
CREATE TABLE archive_publication (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), outbox_seq bigint NOT NULL REFERENCES outbox(seq),
 status text NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','readable','needs_attention','not_applicable')),
 attempts integer NOT NULL DEFAULT 0, next_attempt_at timestamptz NOT NULL DEFAULT now(),
 event_key text, directory text, receipt_path text, files jsonb NOT NULL DEFAULT '[]',
 error_code text, published_at timestamptz, readable_at timestamptz,
 PRIMARY KEY(customer_id,outbox_seq)
);
CREATE TABLE archive_event (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), event_key text NOT NULL,
 directory text NOT NULL, result_files jsonb NOT NULL DEFAULT '{}', last_seq bigint NOT NULL DEFAULT 0,
 PRIMARY KEY(customer_id,event_key), UNIQUE(customer_id,directory)
);
