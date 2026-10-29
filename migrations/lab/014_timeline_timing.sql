-- Append attempt/stage measurements; NULL/unrecorded stages are never zero time.
CREATE TABLE hj_timeline_timing (
 customer_id uuid NOT NULL, attempt_token uuid NOT NULL,
 stage text NOT NULL, window_index integer NOT NULL DEFAULT -1,
 started_at timestamptz NOT NULL, finished_at timestamptz NOT NULL,
 elapsed_seconds numeric(20,6) NOT NULL CHECK(elapsed_seconds>=0),
 details jsonb NOT NULL DEFAULT '{}',
 PRIMARY KEY(customer_id,attempt_token,stage,window_index),
 FOREIGN KEY(customer_id,attempt_token) REFERENCES hj_timeline_task_attempt(customer_id,token)
);
CREATE INDEX hj_timing_customer_clock ON hj_timeline_timing(customer_id,started_at);
