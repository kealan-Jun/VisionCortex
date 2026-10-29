CREATE TABLE customer_instance (
 id uuid PRIMARY KEY, code text NOT NULL UNIQUE,
 data_scope text NOT NULL CHECK(data_scope IN ('customer','development','fixture')),
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE source_instance (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), id uuid NOT NULL,
 name text NOT NULL, revision text NOT NULL, kind text NOT NULL,
 coverage text NOT NULL DEFAULT 'unknown' CHECK(coverage IN ('complete','partial','unknown')),
 checked_at timestamptz, PRIMARY KEY(customer_id,id), UNIQUE(customer_id,name)
);
CREATE TABLE entity (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), id uuid NOT NULL,
 kind text NOT NULL CHECK(kind IN ('instrument','capture_device','container','tag','material_type','source_lot','material_unit','process_batch','experiment_run','person')),
 code text NOT NULL, metadata jsonb NOT NULL DEFAULT '{}', revision integer NOT NULL DEFAULT 1,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(customer_id,id), UNIQUE(customer_id,kind,code)
);
CREATE TABLE source_id_mapping (
 customer_id uuid NOT NULL, source_id uuid NOT NULL, source_entity_id text NOT NULL,
 source_revision text NOT NULL, target_id uuid NOT NULL,
 PRIMARY KEY(customer_id,source_id,source_entity_id,source_revision),
 FOREIGN KEY(customer_id,source_id) REFERENCES source_instance(customer_id,id),
 FOREIGN KEY(customer_id,target_id) REFERENCES entity(customer_id,id)
);
CREATE TABLE tag_assignment (
 customer_id uuid NOT NULL, id uuid NOT NULL, tag_id uuid NOT NULL, target_id uuid NOT NULL,
 valid_from timestamptz NOT NULL, valid_to timestamptz, version integer NOT NULL,
 actor text NOT NULL, reason text NOT NULL, PRIMARY KEY(customer_id,id),
 CHECK(valid_to IS NULL OR valid_to>valid_from), UNIQUE(customer_id,tag_id,version),
 FOREIGN KEY(customer_id,tag_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,target_id) REFERENCES entity(customer_id,id)
);
CREATE UNIQUE INDEX tag_active ON tag_assignment(customer_id,tag_id) WHERE valid_to IS NULL;
CREATE TABLE container_content_binding (
 customer_id uuid NOT NULL, id uuid NOT NULL, container_id uuid NOT NULL, material_id uuid NOT NULL,
 valid_from timestamptz NOT NULL, valid_to timestamptz, version integer NOT NULL,
 PRIMARY KEY(customer_id,id), CHECK(valid_to IS NULL OR valid_to>valid_from),
 FOREIGN KEY(customer_id,container_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,material_id) REFERENCES entity(customer_id,id)
);
CREATE UNIQUE INDEX container_active ON container_content_binding(customer_id,container_id) WHERE valid_to IS NULL;
CREATE TABLE equipment_use (
 customer_id uuid NOT NULL, id uuid NOT NULL, instrument_id uuid NOT NULL,
 batch_id uuid NOT NULL, material_id uuid, position text, valid_from timestamptz NOT NULL,
 valid_to timestamptz, actor text NOT NULL, PRIMARY KEY(customer_id,id),
 CHECK(valid_to IS NULL OR valid_to>=valid_from),
 FOREIGN KEY(customer_id,instrument_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,batch_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,material_id) REFERENCES entity(customer_id,id)
);
CREATE TABLE capture_session (
 customer_id uuid NOT NULL, id uuid NOT NULL, source_id uuid NOT NULL,
 source_session_id text NOT NULL, camera_id uuid, metadata jsonb NOT NULL DEFAULT '{}',
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,source_id,source_session_id),
 FOREIGN KEY(customer_id,source_id) REFERENCES source_instance(customer_id,id),
 FOREIGN KEY(customer_id,camera_id) REFERENCES entity(customer_id,id)
);
CREATE TABLE capture (
 customer_id uuid NOT NULL, id uuid NOT NULL, source_id uuid NOT NULL, source_capture_id text NOT NULL,
 source_revision text NOT NULL, session_id uuid, kind text NOT NULL CHECK(kind IN ('photo','video','audio')),
 occurred_at timestamptz, captured_at timestamptz, received_at timestamptz NOT NULL DEFAULT now(),
 time_basis jsonb NOT NULL, source_kind text NOT NULL,
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,source_id,source_capture_id,source_revision),
 FOREIGN KEY(customer_id,source_id) REFERENCES source_instance(customer_id,id),
 FOREIGN KEY(customer_id,session_id) REFERENCES capture_session(customer_id,id)
);
CREATE TABLE media_asset (
 customer_id uuid NOT NULL, id uuid NOT NULL, capture_id uuid NOT NULL, version integer NOT NULL DEFAULT 1,
 kind text NOT NULL, declared_sha256 text, verified_sha256 text, byte_size bigint,
 evidence_status text NOT NULL DEFAULT 'unverified' CHECK(evidence_status IN ('unverified','available','unavailable','integrity_failed')),
 verified_at timestamptz, PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,capture_id) REFERENCES capture(customer_id,id)
);
CREATE TABLE media_location (
 customer_id uuid NOT NULL, id uuid NOT NULL, media_id uuid NOT NULL,
 storage text NOT NULL, relative_path text NOT NULL, availability text NOT NULL,
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,media_id,storage,relative_path),
 FOREIGN KEY(customer_id,media_id) REFERENCES media_asset(customer_id,id)
);
CREATE TABLE media_derivation (
 customer_id uuid NOT NULL, parent_id uuid NOT NULL, child_id uuid NOT NULL,
 transform jsonb NOT NULL, PRIMARY KEY(customer_id,parent_id,child_id), CHECK(parent_id<>child_id),
 FOREIGN KEY(customer_id,parent_id) REFERENCES media_asset(customer_id,id),
 FOREIGN KEY(customer_id,child_id) REFERENCES media_asset(customer_id,id)
);
CREATE TABLE evidence_ref (
 customer_id uuid NOT NULL, id uuid NOT NULL, media_id uuid NOT NULL, media_version integer NOT NULL,
 start_seconds numeric(20,6), end_seconds numeric(20,6), coordinates jsonb,
 PRIMARY KEY(customer_id,id), CHECK(start_seconds IS NULL OR (start_seconds>=0 AND end_seconds>=start_seconds)),
 FOREIGN KEY(customer_id,media_id) REFERENCES media_asset(customer_id,id)
);
CREATE TABLE processing_job (
 customer_id uuid NOT NULL, id uuid NOT NULL, capture_id uuid NOT NULL, worker_kind text NOT NULL,
 model_revision text NOT NULL, status text NOT NULL DEFAULT 'queued'
 CHECK(status IN ('queued','running','completed','failed','blocked_model','needs_review')),
 analysis_outcome text NOT NULL DEFAULT 'unknown', association_status text NOT NULL DEFAULT 'unknown',
 fact_status text NOT NULL DEFAULT 'candidate', index_status text NOT NULL DEFAULT 'pending',
 attempts integer NOT NULL DEFAULT 0, lease_token uuid, lease_until timestamptz,
 next_attempt_at timestamptz NOT NULL DEFAULT now(), error_code text, result jsonb,
 created_at timestamptz NOT NULL DEFAULT now(), processed_at timestamptz,
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,capture_id,worker_kind,model_revision),
 FOREIGN KEY(customer_id,capture_id) REFERENCES capture(customer_id,id)
);
CREATE TABLE processing_attempt (
 customer_id uuid NOT NULL, id uuid NOT NULL, job_id uuid NOT NULL,
 attempt integer NOT NULL, lease_token uuid NOT NULL, started_at timestamptz NOT NULL DEFAULT now(),
 finished_at timestamptz, outcome text, component_identity jsonb,
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,job_id,attempt),
 FOREIGN KEY(customer_id,job_id) REFERENCES processing_job(customer_id,id)
);
CREATE TABLE observation (
 customer_id uuid NOT NULL, id uuid NOT NULL, job_id uuid, source_event_id text NOT NULL,
 source_revision text NOT NULL, kind text NOT NULL, occurred_at timestamptz,
 payload jsonb NOT NULL, evidence_ids uuid[] NOT NULL DEFAULT '{}', source_kind text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(customer_id,id),
 UNIQUE(customer_id,source_event_id,source_revision),
 FOREIGN KEY(customer_id,job_id) REFERENCES processing_job(customer_id,id)
);
CREATE TABLE field_candidate (
 customer_id uuid NOT NULL, id uuid NOT NULL, observation_id uuid NOT NULL,
 field_key text NOT NULL, value jsonb, unit text, semantics text NOT NULL,
 association jsonb NOT NULL DEFAULT '{}', conflicts jsonb NOT NULL DEFAULT '[]',
 status text NOT NULL DEFAULT 'candidate', revision integer NOT NULL DEFAULT 1,
 PRIMARY KEY(customer_id,id), FOREIGN KEY(customer_id,observation_id) REFERENCES observation(customer_id,id)
);
CREATE TABLE fact_revision (
 customer_id uuid NOT NULL, id uuid NOT NULL, candidate_id uuid NOT NULL, revision integer NOT NULL,
 scope text NOT NULL, value jsonb, unit text, actor text NOT NULL, policy_version text,
 supersedes_id uuid, status text NOT NULL DEFAULT 'confirmed', confirmed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,candidate_id,revision),
 FOREIGN KEY(customer_id,candidate_id) REFERENCES field_candidate(customer_id,id),
 FOREIGN KEY(customer_id,supersedes_id) REFERENCES fact_revision(customer_id,id)
);
CREATE TABLE confirmation (
 customer_id uuid NOT NULL, id uuid NOT NULL, fact_id uuid NOT NULL,
 scope text NOT NULL, actor text NOT NULL, policy_version text, evidence_ids uuid[] NOT NULL,
 PRIMARY KEY(customer_id,id), FOREIGN KEY(customer_id,fact_id) REFERENCES fact_revision(customer_id,id)
);
CREATE TABLE material_state (
 customer_id uuid NOT NULL, material_id uuid NOT NULL, unit text NOT NULL,
 balance numeric(28,9), used numeric(28,9) NOT NULL DEFAULT 0, reserved numeric(28,9) NOT NULL DEFAULT 0,
 revision integer NOT NULL DEFAULT 1, state text NOT NULL DEFAULT 'active',
 PRIMARY KEY(customer_id,material_id), CHECK(balance IS NULL OR balance>=0), CHECK(reserved>=0),
 FOREIGN KEY(customer_id,material_id) REFERENCES entity(customer_id,id)
);
CREATE TABLE operation (
 customer_id uuid NOT NULL, id uuid NOT NULL, kind text NOT NULL, batch_id uuid, run_id uuid,
 occurred_at timestamptz NOT NULL, actor text NOT NULL, evidence_ids uuid[] NOT NULL DEFAULT '{}',
 revision integer NOT NULL DEFAULT 1, status text NOT NULL DEFAULT 'confirmed',
 supersedes_id uuid, metadata jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,batch_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,run_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,supersedes_id) REFERENCES operation(customer_id,id)
);
CREATE TABLE operation_line (
 customer_id uuid NOT NULL, id uuid NOT NULL, operation_id uuid NOT NULL, material_id uuid NOT NULL,
 direction text NOT NULL CHECK(direction IN ('input','output')), quantity numeric(28,9) NOT NULL CHECK(quantity>0),
 unit text NOT NULL, quantity_basis text NOT NULL, state_revision integer NOT NULL,
 PRIMARY KEY(customer_id,id), FOREIGN KEY(customer_id,operation_id) REFERENCES operation(customer_id,id),
 FOREIGN KEY(customer_id,material_id) REFERENCES entity(customer_id,id)
);
CREATE TABLE quantity_ledger_entry (
 customer_id uuid NOT NULL, id uuid NOT NULL, operation_id uuid NOT NULL, material_id uuid NOT NULL,
 delta numeric(28,9) NOT NULL, used_delta numeric(28,9) NOT NULL DEFAULT 0, unit text NOT NULL,
 reverses_id uuid, created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,operation_id) REFERENCES operation(customer_id,id),
 FOREIGN KEY(customer_id,material_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,reverses_id) REFERENCES quantity_ledger_entry(customer_id,id)
);
CREATE TABLE reservation (
 customer_id uuid NOT NULL, id uuid NOT NULL, material_id uuid NOT NULL,
 quantity numeric(28,9) NOT NULL CHECK(quantity>0), unit text NOT NULL, status text NOT NULL DEFAULT 'active',
 reason text NOT NULL, PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,material_id) REFERENCES entity(customer_id,id)
);
CREATE TABLE command_receipt (
 customer_id uuid NOT NULL, key text NOT NULL, fingerprint text NOT NULL, result jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(customer_id,key),
 FOREIGN KEY(customer_id) REFERENCES customer_instance(id)
);
CREATE TABLE inbox (
 customer_id uuid NOT NULL, source_id uuid NOT NULL, event_id text NOT NULL, revision text NOT NULL,
 fingerprint text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY(customer_id,source_id,event_id,revision),
 FOREIGN KEY(customer_id,source_id) REFERENCES source_instance(customer_id,id)
);
CREATE TABLE consumer_checkpoint (
 customer_id uuid NOT NULL, source_id uuid NOT NULL, partition text NOT NULL, cursor bigint NOT NULL,
 checked_at timestamptz NOT NULL DEFAULT now(), status text NOT NULL,
 PRIMARY KEY(customer_id,source_id,partition),
 FOREIGN KEY(customer_id,source_id) REFERENCES source_instance(customer_id,id)
);
CREATE TABLE outbox (
 seq bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, customer_id uuid NOT NULL REFERENCES customer_instance(id),
 event_id uuid NOT NULL, kind text NOT NULL, record_id uuid NOT NULL, revision integer NOT NULL,
 payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(customer_id,event_id)
);
CREATE TABLE index_document (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), record_id uuid NOT NULL, revision integer NOT NULL,
 kind text NOT NULL, occurred_at timestamptz, text_content text NOT NULL, payload jsonb NOT NULL,
 outbox_seq bigint NOT NULL, PRIMARY KEY(customer_id,record_id)
);
CREATE TABLE index_checkpoint (
 customer_id uuid PRIMARY KEY REFERENCES customer_instance(id), cursor bigint NOT NULL DEFAULT 0,
 checked_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE form_template_version (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), id uuid NOT NULL,
 name text NOT NULL, version integer NOT NULL, fields jsonb NOT NULL,
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,name,version)
);
CREATE TABLE field_snapshot (
 customer_id uuid NOT NULL, id uuid NOT NULL, template_id uuid NOT NULL,
 input_revisions jsonb NOT NULL, fields jsonb NOT NULL, stale boolean NOT NULL DEFAULT false,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,template_id) REFERENCES form_template_version(customer_id,id)
);
CREATE INDEX job_claim ON processing_job(customer_id,status,next_attempt_at,created_at DESC);
CREATE INDEX operation_material ON operation_line(customer_id,material_id,operation_id);
CREATE INDEX evidence_media ON evidence_ref(customer_id,media_id);
CREATE INDEX outbox_customer ON outbox(customer_id,seq);
