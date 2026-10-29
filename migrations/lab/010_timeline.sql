-- Additive, opt-in timeline path. Frozen native processing is unchanged.
CREATE TABLE hj_timeline_media (
 customer_id uuid NOT NULL, media_id uuid NOT NULL, capture_id uuid NOT NULL,
 kind text NOT NULL CHECK(kind IN ('video','audio','photo')),
 stream_key text NOT NULL, session_key text,
 start_at timestamptz, end_at timestamptz, duration_seconds numeric(20,6) NOT NULL,
 time_basis jsonb NOT NULL, PRIMARY KEY(customer_id,media_id),
 CHECK(duration_seconds>=0), CHECK(end_at IS NULL OR end_at>=start_at),
 FOREIGN KEY(customer_id,media_id) REFERENCES media_asset(customer_id,id),
 FOREIGN KEY(customer_id,capture_id) REFERENCES capture(customer_id,id)
);
CREATE INDEX hj_media_clock ON hj_timeline_media(customer_id,stream_key,start_at,end_at);

CREATE TABLE hj_label_event (
 customer_id uuid NOT NULL, id uuid NOT NULL, tag_id uuid NOT NULL,
 target_id uuid, occurred_at timestamptz NOT NULL,
 actor text NOT NULL, reason text NOT NULL, PRIMARY KEY(customer_id,id),
 UNIQUE(customer_id,tag_id,occurred_at),
 FOREIGN KEY(customer_id,tag_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,target_id) REFERENCES entity(customer_id,id)
);

CREATE TABLE hj_experiment_record (
 customer_id uuid NOT NULL, id uuid NOT NULL, experiment_type text NOT NULL,
 sample_ids uuid[] NOT NULL, person_id uuid, batch_id uuid,
 policy jsonb NOT NULL, policy_fingerprint text NOT NULL,
 first_observed_at timestamptz, last_observed_at timestamptz,
 record_status text NOT NULL DEFAULT 'open' CHECK(record_status IN ('open','archived')),
 closed_at timestamptz, closure_reason text, closure_evidence_ids uuid[] NOT NULL DEFAULT '{}',
 revision integer NOT NULL DEFAULT 1, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,person_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,batch_id) REFERENCES entity(customer_id,id)
);
CREATE TABLE hj_run_relation (
 customer_id uuid NOT NULL, parent_id uuid NOT NULL, child_id uuid NOT NULL,
 relation text NOT NULL CHECK(relation IN ('next_step','sample_derivation')),
 PRIMARY KEY(customer_id,parent_id,child_id), CHECK(parent_id<>child_id),
 FOREIGN KEY(customer_id,parent_id) REFERENCES hj_experiment_record(customer_id,id),
 FOREIGN KEY(customer_id,child_id) REFERENCES hj_experiment_record(customer_id,id)
);
CREATE TABLE hj_experiment_binding (
 customer_id uuid NOT NULL, id uuid NOT NULL, run_id uuid NOT NULL,
 instrument_id uuid NOT NULL, sample_id uuid NOT NULL, person_id uuid,
 position text NOT NULL DEFAULT '', occurred_at timestamptz NOT NULL,
 semantics text NOT NULL CHECK(semantics IN ('association','preparation','handover')),
 handover_effective boolean NOT NULL DEFAULT false,
 evidence_ids uuid[] NOT NULL DEFAULT '{}', label_assignment_id uuid,
 source text NOT NULL, PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,run_id) REFERENCES hj_experiment_record(customer_id,id),
 FOREIGN KEY(customer_id,instrument_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,sample_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,person_id) REFERENCES entity(customer_id,id)
);
CREATE INDEX hj_binding_clock ON hj_experiment_binding(customer_id,instrument_id,position,occurred_at);

CREATE TABLE hj_activity (
 customer_id uuid NOT NULL, id uuid NOT NULL, kind text NOT NULL,
 source_type text NOT NULL CHECK(source_type IN ('video','speech','ocr','reviewed_note')),
 text_content text NOT NULL, start_at timestamptz, end_at timestamptz,
 sample_id uuid, instrument_id uuid, person_id uuid,
 tag_id uuid, label_resolution jsonb NOT NULL,
 evidence_ids uuid[] NOT NULL, payload jsonb NOT NULL, model_revision text,
 association_status text NOT NULL, priority text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,sample_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,instrument_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,person_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,tag_id) REFERENCES entity(customer_id,id)
);
CREATE INDEX hj_activity_clock ON hj_activity(customer_id,start_at,end_at);
CREATE TABLE hj_activity_run (
 customer_id uuid NOT NULL, activity_id uuid NOT NULL, run_id uuid NOT NULL,
 basis text NOT NULL, PRIMARY KEY(customer_id,activity_id,run_id),
 FOREIGN KEY(customer_id,activity_id) REFERENCES hj_activity(customer_id,id),
 FOREIGN KEY(customer_id,run_id) REFERENCES hj_experiment_record(customer_id,id)
);
-- Association revisions do not rewrite the original model observation.
CREATE TABLE hj_activity_association_revision (
 customer_id uuid NOT NULL, id uuid NOT NULL, activity_id uuid NOT NULL,
 run_ids uuid[] NOT NULL, reason text NOT NULL, actor text NOT NULL,
 label_resolution jsonb NOT NULL DEFAULT '{}', association_status text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,activity_id) REFERENCES hj_activity(customer_id,id)
);

CREATE TABLE hj_timeline_task (
 customer_id uuid NOT NULL, id uuid NOT NULL, media_id uuid NOT NULL,
 kind text NOT NULL CHECK(kind IN ('video_understanding','transcription','ocr')),
 analysis_revision text NOT NULL DEFAULT 'initial',
 status text NOT NULL DEFAULT 'awaiting_provider'
 CHECK(status IN ('awaiting_provider','leased','completed')),
 lease_token uuid, lease_until timestamptz, model_revision text,
 request jsonb, result jsonb, created_at timestamptz NOT NULL DEFAULT now(),
 completed_at timestamptz, PRIMARY KEY(customer_id,id),
 UNIQUE(customer_id,media_id,kind,analysis_revision),
 FOREIGN KEY(customer_id,media_id) REFERENCES hj_timeline_media(customer_id,media_id)
);
CREATE TABLE hj_timeline_task_attempt (
 customer_id uuid NOT NULL, token uuid NOT NULL, task_id uuid NOT NULL,
 model_revision text NOT NULL, request jsonb NOT NULL,
 started_at timestamptz NOT NULL, lease_until timestamptz NOT NULL,
 outcome text NOT NULL DEFAULT 'leased', PRIMARY KEY(customer_id,token),
 FOREIGN KEY(customer_id,task_id) REFERENCES hj_timeline_task(customer_id,id)
);
CREATE TABLE hj_daily_report (
 customer_id uuid NOT NULL, day date NOT NULL, revision integer NOT NULL,
 fingerprint text NOT NULL, document jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(customer_id,day,revision),
 UNIQUE(customer_id,day,fingerprint),
 FOREIGN KEY(customer_id) REFERENCES customer_instance(id)
);
CREATE TABLE hj_timeline_dirty_day (
 customer_id uuid NOT NULL, day date NOT NULL, PRIMARY KEY(customer_id,day),
 FOREIGN KEY(customer_id) REFERENCES customer_instance(id)
);
