-- Publication is a report/day operation, independent of a model task lease.
CREATE TABLE hj_timeline_publication_timing (
 customer_id uuid NOT NULL REFERENCES customer_instance(id),
 id uuid NOT NULL, day date NOT NULL, revision integer NOT NULL,
 started_at timestamptz NOT NULL, finished_at timestamptz NOT NULL,
 elapsed_seconds numeric(20,6) NOT NULL CHECK(elapsed_seconds>=0),
 outcome text NOT NULL CHECK(outcome IN ('completed','failed')),
 stages jsonb NOT NULL, error_code text,
 PRIMARY KEY(customer_id,id)
);
CREATE INDEX hj_publication_timing_day ON hj_timeline_publication_timing(customer_id,day,started_at);
