-- Local runner execution and bounded recovery, without changing prior observations.
ALTER TABLE hj_timeline_task DROP CONSTRAINT hj_timeline_task_status_check;
ALTER TABLE hj_timeline_task ADD CONSTRAINT hj_timeline_task_status_check
 CHECK(status IN ('awaiting_provider','leased','completed','needs_attention'));
ALTER TABLE hj_timeline_task ADD COLUMN attempts integer NOT NULL DEFAULT 0;
ALTER TABLE hj_timeline_task ADD COLUMN next_attempt_at timestamptz NOT NULL DEFAULT '-infinity';
ALTER TABLE hj_timeline_task ADD COLUMN error_code text;
ALTER TABLE hj_timeline_task_attempt ADD COLUMN finished_at timestamptz;
CREATE TABLE hj_timeline_window_result (
 customer_id uuid NOT NULL, task_id uuid NOT NULL, input_fingerprint text NOT NULL,
 provider_fingerprint text NOT NULL, window_index integer NOT NULL CHECK(window_index>=0),
 normalized_result jsonb NOT NULL, execution_receipt jsonb NOT NULL,
 result_fingerprint text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,task_id,input_fingerprint,provider_fingerprint,window_index),
 FOREIGN KEY(customer_id,task_id) REFERENCES hj_timeline_task(customer_id,id)
);
CREATE TABLE hj_ocr_frame_registration (
 customer_id uuid NOT NULL, media_id uuid NOT NULL, revision integer NOT NULL,
 source_sha256 text NOT NULL, registration jsonb NOT NULL,
 actor text NOT NULL, reason text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,media_id,revision),
 FOREIGN KEY(customer_id,media_id) REFERENCES hj_timeline_media(customer_id,media_id)
);
CREATE INDEX hj_timeline_retry ON hj_timeline_task(customer_id,kind,next_attempt_at)
 WHERE status='awaiting_provider';
CREATE TABLE hj_timeline_publication (
 customer_id uuid NOT NULL, day date NOT NULL, revision integer NOT NULL,
 destination_fingerprint text NOT NULL, files jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,day,revision,destination_fingerprint),
 FOREIGN KEY(customer_id,day,revision) REFERENCES hj_daily_report(customer_id,day,revision)
);
